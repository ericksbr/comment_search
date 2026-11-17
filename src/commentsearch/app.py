"""Gradio UI over a Searcher: a query box that updates as you type, exact or fuzzy matching, a hits list with a
preview panel, and CSV download. Nothing leaves the runtime."""

import html
import os
import re
import tempfile

import pandas as pd

from .exact import highlight_exact
from .search import highlight

_TABLE_INTERNAL = {"comment_id", "text", "dup_group", "dup_group_size", "boilerplate"}
EXCERPT_CHARS = 160
# Shortest to longest, as shown on the right of the table.
SHOW_MODES = ["Excerpt", "Passage", "Full comment"]
SHOW_MODE_INFO = (
    "Excerpt: about 160 characters around the first matching word. "
    "Passage: the part of the comment that matched best. "
    "Full comment: the whole comment text."
)
ALWAYS_SHOWN = ["rank"]

# Exact = every word must appear (stems count), and "quoted phrases" must appear in order.
# Fuzzy = comments with a passage at least this similar (cosine) to the query.
MATCH_EXACT = "Exact words"
MATCH_FUZZY = "Fuzzy (by meaning)"
MATCH_MODES = [MATCH_EXACT, MATCH_FUZZY]
MATCH_INFO = (
    "Exact: every word you type must appear (word endings are ignored, so 'impact' finds 'impacts'). "
    "Put a phrase in quotes to require the words in that order. "
    "Fuzzy: comments whose sentences are similar in meaning, even without the same words."
)
HOW_TO_TEXT = (
    "**Exact words:** every word you type must appear in the comment. Word endings are ignored, so "
    "\"impact\" finds \"impacts\". Put words in quotes to require them in that order.\n\n"
    "**Fuzzy (by meaning):** finds comments whose sentences are similar in meaning, even without the same "
    "words. The cutoff sets how close a match must be. The blue sentences are the ones that matched."
)
HOW_TO_PLAIN = (
    "Exact words: every word must appear; word endings are ignored ('impact' finds 'impacts'); "
    "quotes require the words in order. Yellow = matched words. "
    "Fuzzy (by meaning): comments similar in meaning, even without the same words; the cutoff sets how close. "
    "Blue = sentences at or above the cutoff. Arrow keys or j / k move through the hits."
)
# Provisional guess. A data-based default is planned (see notes); the slider stays adjustable.
DEFAULT_MIN_SIMILARITY = 0.6
SIMILARITY_INFO = "Higher = stricter. Fuzzy results and blue sentences are at or above this value."

CARD_CSS = """
<style>
.cs-card{border:1px solid #d0d7de;border-radius:10px;padding:12px 14px;margin:6px 0}
.cs-head{font-size:15px;margin-bottom:4px}
.cs-rank{font-weight:700;margin-right:6px}
.cs-full{font-size:14.5px;line-height:1.5;margin:6px 0 2px 0;max-height:calc(100vh - 250px);min-height:200px;overflow:auto;padding:6px 10px;border-left:3px solid #d0d7de}
.cs-card mark, .cs-exact{background:#ffe08a;color:#1f2328;padding:0 2px;border-radius:3px}
.cs-sent{background:#cfe8ff;color:#1f2328;padding:0 2px;border-radius:3px}
.cs-legend{font-size:13px;color:#57606a;margin:4px 0}
.cs-muted{color:#57606a;font-size:13px}
.cs-title{margin:0 0 2px 0;font-size:15px}
.cs-info{cursor:help;color:#57606a;border:1px solid #d0d7de;border-radius:50%;padding:0 6px;font-size:12px;margin-left:4px}
.cs-table, .cs-table table, .cs-table td, .cs-table th {font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;}
.cs-table td, .cs-table th {font-size:12.5px !important;padding:2px 6px !important;line-height:1.25 !important;}
.cs-table tr.selected td, .cs-table td.selected, .cs-table .selected {background:#e8f1ff !important;box-shadow:none !important;outline:none !important;}
.cs-position{font-size:13px;color:#57606a;text-align:center;padding-top:8px}
</style>
"""

# Arrow keys and j/k move through the hits. Buttons are found by their labels.
KEY_JS = """() => {
  const press = (label) => {
    const b = [...document.querySelectorAll('button')].find(x => x.innerText.trim() === label);
    if (b) b.click();
  };
  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || (e.target && e.target.isContentEditable)) return;
    if (e.key === 'ArrowDown' || e.key === 'j') { press('Next ▶'); e.preventDefault(); }
    if (e.key === 'ArrowUp' || e.key === 'k') { press('◀ Previous'); e.preventDefault(); }
  });
}"""


def meta_columns(comments: pd.DataFrame) -> list[str]:
    """Metadata columns from the file (everything except the text and internal columns), in file order."""
    return [c for c in comments.columns if c not in _TABLE_INTERNAL]


def matched_terms(passage: str, terms: list[str]) -> list[str]:
    """Query terms that appear as whole words in the passage."""
    return [t for t in terms if re.search(rf"\b{re.escape(t)}\b", passage, re.IGNORECASE)]


def _message(text: str) -> str:
    return f"{CARD_CSS}<p class='cs-muted'>{html.escape(text)}</p>"


def render_cards(frame: pd.DataFrame, terms: list[str], text_by_id: dict, collapse: bool) -> str:
    """HTML cards for the results (kept for export and tests; the window shows the list and preview)."""
    if frame.empty:
        return _message("No matching comments.")
    parts = [CARD_CSS]
    for _, row in frame.iterrows():
        passage = str(row["passage"]) if "passage" in row else str(row["snippet"])
        parts.append(
            "<div class='cs-card'>"
            f"<div class='cs-head'><span class='cs-rank'>#{int(row['rank'])}</span>"
            f"<b>{html.escape(str(row['comment_id']))}</b></div>"
            f"<div class='cs-full'>{highlight(passage, terms)}</div>"
            "</div>"
        )
    return "\n".join(parts)


def _cell(value) -> str:
    """A table cell as text: blank for missing values, the date for dates, the text otherwise."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def hits_table(frame: pd.DataFrame, meta_cols: list[str]) -> pd.DataFrame:
    """One row per hit: rank, comment ID and the file's metadata columns. The text is shown in the preview."""
    cols = ["rank", "comment_id", *meta_cols]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"rank": frame["rank"].to_numpy(), "comment_id": frame["comment_id"].to_numpy()})
    for col in meta_cols:
        out[col] = [_cell(v) for v in frame[col]] if col in frame.columns else ""
    return out[cols]


def default_table_columns(hits: pd.DataFrame) -> list[str]:
    """Shown by default: rank, comment ID, and the first few metadata columns."""
    cols = list(hits.columns)
    meta = [c for c in cols if c not in ("rank", "comment_id")]
    return ["rank", "comment_id", *meta[:3]]


def select_table_columns(hits: pd.DataFrame, chosen) -> pd.DataFrame:
    """The table with only the chosen columns, in the file's order. Rank is always kept."""
    if hits.empty and len(hits.columns) == 0:
        return hits
    keep = [c for c in hits.columns if c in set(chosen) | set(ALWAYS_SHOWN)]
    return hits[keep] if keep else hits


def sentence_spans(body: str, sentences, min_similarity: float) -> str:
    """HTML: the body with every sentence at or above the cutoff shaded blue (hover shows its similarity)."""
    out, pos = [], 0
    for text, score in sentences or []:
        if score < min_similarity or not text:
            continue
        at = body.find(text, pos)
        if at < 0:
            continue
        out.append(html.escape(body[pos:at]))
        out.append(f'<span class="cs-sent" title="similarity {score:.2f}">{html.escape(text)}</span>')
        pos = at + len(text)
    out.append(html.escape(body[pos:]))
    return "".join(out)


def detail_html(frame, row, mode: str, text_by_id: dict, meta_cols: list[str], sentences=None) -> str:
    """The selected hit. Exact matches are shaded yellow; fuzzy matches shade their similar sentences blue.

    ``sentences`` is the list of (sentence, similarity) for this comment, from the searcher.
    """
    if frame is None or len(frame) == 0 or row is None:
        return _message("Click a row in the table to see the comment here.")
    r = frame.iloc[int(row)]
    plan = frame.attrs.get("plan") or {}
    method = frame.attrs.get("method")
    if mode == "Excerpt":
        body = str(r["snippet"])
    elif mode == "Full comment":
        body = text_by_id.get(r["comment_id"], "")
    else:
        body = str(r["passage"]) if "passage" in r.index else str(r["snippet"])
    if method == "exact" and plan:
        body_html = highlight_exact(body, plan)
        legend = "Yellow: words that matched (word endings count, so 'impact' matches 'impacts')."
    elif method == "semantic":
        cutoff = float(plan.get("min_similarity", DEFAULT_MIN_SIMILARITY))
        body_html = sentence_spans(body, sentences, cutoff)
        legend = f"Blue: sentences at or above similarity {cutoff:.2f} to your search."
    else:
        body_html = html.escape(body)
        legend = ""
    facts = " · ".join(
        f"{html.escape(col)}: {html.escape(str(r[col]))}"
        for col in meta_cols
        if col in r.index and str(r[col]) != ""
    )
    return (
        f"{CARD_CSS}<div class='cs-card'>"
        f"<div class='cs-head'><span class='cs-rank'>#{int(r['rank'])}</span><b>{html.escape(str(r['comment_id']))}</b></div>"
        f"<div class='cs-muted'>{facts}</div>"
        f"<div class='cs-legend'>{html.escape(legend)}</div>"
        f"<div class='cs-full'>{body_html}</div>"
        "</div>"
    )


def position_text(frame, row) -> str:
    """'Hit 3 of 45' under the list, or nothing when there are no hits."""
    if frame is None or len(frame) == 0 or row is None:
        return ""
    return f"Hit {int(row) + 1:,} of {len(frame):,}"


def search_comments(searcher, query: str, match: str, collapse: bool, min_similarity: float):
    """Exact: every comment with all the words (or phrases). Fuzzy: comments at or above the cutoff."""
    n = len(searcher.index.comments)
    if match == MATCH_EXACT:
        return searcher.search(query, top_k=n, method="exact", backend="fts5", collapse=collapse)
    return searcher.search(query, top_k=n, method="semantic", min_similarity=float(min_similarity), collapse=collapse)


def make_app(searcher, title: str = "Local comment search"):
    """Build the Gradio Blocks app. Nothing leaves the runtime; there is no external API."""
    import gradio as gr

    comments = searcher.index.comments
    n_comments = len(comments)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    table_meta = meta_columns(comments)
    empty_hits = hits_table(pd.DataFrame(), table_meta)

    def sentences_for(frame, row):
        """Sentence similarities for the selected comment (fuzzy results only)."""
        if frame is None or row is None or len(frame) == 0:
            return None
        plan = frame.attrs.get("plan") or {}
        query_text = plan.get("query")
        if not query_text:
            return None
        comment_row = int(frame.attrs["rows"][int(row)])
        return searcher.sentence_similarities(query_text, [comment_row]).get(comment_row, [])

    def render(frame, row, mode):
        return detail_html(frame, row, mode, text_by_id, table_meta, sentences_for(frame, row)), position_text(frame, row)

    def run(query, match, similarity, collapse, mode):
        no_columns = gr.update()
        if not (query or "").strip():
            return "Type a word or phrase to search.", None, empty_hits, None, no_columns, None, None, *render(None, None, mode)
        frame = search_comments(searcher, query, match, bool(collapse), similarity)
        if frame.empty:
            if match == MATCH_EXACT:
                status = "No comment contains all of these words (or phrases in quotes)."
            else:
                status = (
                    f"**No close matches** at similarity {float(similarity):.2f} or higher. "
                    "Move the cutoff left to see weaker matches."
                )
        else:
            n = len(frame)
            if match == MATCH_EXACT:
                status = f"**{n:,} comments** contain all of the words."
            else:
                status = f"**{n:,} comments** with a passage at similarity {float(similarity):.2f} or higher."
            if match != MATCH_EXACT and not searcher.has_sentences:
                status += " Rebuild the index in Act 1 to see sentence highlights."
        fd, csv_path = tempfile.mkstemp(suffix=".csv", prefix="commentsearch_results_")
        os.close(fd)
        frame.to_csv(csv_path, index=False)  # plain text in the CSV (no <mark> tags)
        hits = hits_table(frame, table_meta)
        defaults = default_table_columns(hits) if len(hits.columns) else []
        choices = list(hits.columns) if len(hits.columns) else list(empty_hits.columns)
        first = 0 if len(frame) else None
        detail, position = render(frame, first, mode)
        return (
            status,
            csv_path,
            select_table_columns(hits, defaults),
            frame,
            gr.update(choices=choices, value=defaults),
            hits,
            first,
            detail,
            position,
        )

    with gr.Blocks(title=title, fill_height=True) as demo:
        gr.HTML(CARD_CSS)
        gr.HTML(
            f"<div class='cs-title'><b>{html.escape(title)}</b> · {n_comments:,} comments · "
            "models run locally; no comment text leaves this runtime"
            f"<span class='cs-info' title='{html.escape(HOW_TO_PLAIN, quote=True)}'>ⓘ</span></div>"
        )
        with gr.Row(equal_height=True):
            query = gr.Textbox(
                lines=1, max_lines=1, show_label=False, container=False, scale=6, autofocus=True,
                placeholder='Search, e.g. economic impacts or "public hearing"',
            )
            go = gr.Button("Search", variant="primary", scale=1, min_width=90)
        with gr.Row():
            exact_btn = gr.Button("Exact words", size="sm", variant="primary", min_width=110)
            fuzzy_btn = gr.Button("Fuzzy (by meaning)", size="sm", variant="secondary", min_width=150)
            similarity = gr.Slider(
                0.0, 1.0, value=DEFAULT_MIN_SIMILARITY, step=0.01,
                label="Similarity cutoff", info="Higher = stricter", visible=False, scale=2,
            )
            collapse = gr.Checkbox(value=True, label="One per duplicate group", scale=2)
        status = gr.Markdown()

        match_state = gr.State(MATCH_EXACT)
        frame_state = gr.State(None)
        row_state = gr.State(None)
        hits_state = gr.State(None)  # every column of every hit; the list shows a chosen subset

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                with gr.Accordion("Columns and download", open=False):
                    table_columns = gr.CheckboxGroup(choices=list(empty_hits.columns), value=[], label="Columns")
                    download = gr.DownloadButton("Download results as CSV", value=None)
                hits = gr.Dataframe(
                    value=empty_hits,
                    interactive=False,
                    wrap=False,
                    show_label=False,
                    max_height=520,
                    elem_classes="cs-table",
                )
                with gr.Row():
                    prev_btn = gr.Button("◀ Previous", size="sm")
                    position = gr.Markdown("", elem_classes="cs-position")
                    next_btn = gr.Button("Next ▶", size="sm")
            with gr.Column(scale=3):
                with gr.Row():
                    show_mode = gr.Radio(SHOW_MODES, value="Excerpt", show_label=False, container=False, scale=5)
                    gr.HTML(
                        f"<span class='cs-info' title='{html.escape(SHOW_MODE_INFO, quote=True)}'>ⓘ</span>",
                        min_width=40,
                    )
                detail = gr.HTML(value=_message("Click a row in the table to see the comment here."))

        with gr.Accordion("Compare the two exact-match engines (developer check)", open=False):
            compare_btn = gr.Button("Compare for the current search")
            compare_out = gr.Markdown()

        inputs = [query, match_state, similarity, collapse, show_mode]
        outputs = [status, download, hits, frame_state, table_columns, hits_state, row_state, detail, position]
        go.click(run, inputs=inputs, outputs=outputs)
        query.submit(run, inputs=inputs, outputs=outputs)
        query.change(run, inputs=inputs, outputs=outputs, trigger_mode="always_last")
        similarity.release(run, inputs=inputs, outputs=outputs)
        collapse.change(run, inputs=inputs, outputs=outputs)

        def set_match(choice):
            is_fuzzy = choice == MATCH_FUZZY
            return (
                choice,
                gr.update(variant="secondary" if is_fuzzy else "primary"),
                gr.update(variant="primary" if is_fuzzy else "secondary"),
                gr.update(visible=is_fuzzy),
            )

        match_outputs = [match_state, exact_btn, fuzzy_btn, similarity]
        exact_btn.click(lambda: set_match(MATCH_EXACT), None, match_outputs).then(run, inputs, outputs)
        fuzzy_btn.click(lambda: set_match(MATCH_FUZZY), None, match_outputs).then(run, inputs, outputs)

        def on_columns(chosen, all_hits):
            if all_hits is None:
                return empty_hits
            return select_table_columns(all_hits, chosen)

        table_columns.change(on_columns, inputs=[table_columns, hits_state], outputs=hits)

        def on_select(evt: gr.SelectData, frame, mode):
            row = evt.index[0]
            detail_text, pos = render(frame, row, mode)
            return row, detail_text, pos

        def on_step(step, frame, row, mode):
            if frame is None or len(frame) == 0 or row is None:
                return row, *render(frame, row, mode)
            new_row = min(max(int(row) + step, 0), len(frame) - 1)
            return new_row, *render(frame, new_row, mode)

        def on_mode(mode, frame, row):
            return render(frame, row, mode)[0]

        hits.select(on_select, inputs=[frame_state, show_mode], outputs=[row_state, detail, position])
        prev_btn.click(lambda f, r, m: on_step(-1, f, r, m), inputs=[frame_state, row_state, show_mode],
                       outputs=[row_state, detail, position])
        next_btn.click(lambda f, r, m: on_step(1, f, r, m), inputs=[frame_state, row_state, show_mode],
                       outputs=[row_state, detail, position])
        show_mode.change(on_mode, inputs=[show_mode, frame_state, row_state], outputs=detail)

        def on_compare(q):
            if not (q or "").strip():
                return "Type a search first."
            report = searcher.compare_exact(q)
            lines = [
                f"**FTS5 (SQLite):** {report['fts5']:,} comments  ·  **bm25s (NLTK Porter, our index):** {report['bm25s']:,} comments  ·  "
                f"in both: {report['both']:,}",
            ]
            if report["only_fts5_count"] or report["only_bm25s_count"]:
                lines.append(
                    f"Only FTS5: {report['only_fts5_count']:,} (first IDs: {', '.join(report['only_fts5']) or 'none'})  \n"
                    f"Only bm25s: {report['only_bm25s_count']:,} (first IDs: {', '.join(report['only_bm25s']) or 'none'})"
                )
            else:
                lines.append("The two engines return the same comments for this search.")
            return "\n\n".join(lines)

        compare_btn.click(on_compare, inputs=query, outputs=compare_out)
        demo.load(None, None, None, js=KEY_JS)
    return demo


def launch(searcher, share: bool = False, **kwargs):
    """Launch the UI. ``share=True`` would route traffic through Gradio's servers, so it is off by default."""
    demo = make_app(searcher)
    return demo.launch(share=share, **kwargs)
