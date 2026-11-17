"""Gradio UI over a Searcher: a query box that updates as you type, exact or fuzzy matching, a hits table with a
detail panel, and CSV download. Nothing leaves the runtime."""

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
# Provisional guess. A data-based default is planned (see notes); the slider stays adjustable.
DEFAULT_MIN_SIMILARITY = 0.6
SIMILARITY_INFO = "Higher = stricter. Fuzzy results and blue sentences are at or above this value."

CARD_CSS = """
<style>
.cs-card{border:1px solid #d0d7de;border-radius:10px;padding:14px 16px;margin:8px 0}
.cs-head{font-size:15px;margin-bottom:6px}
.cs-rank{font-weight:700;margin-right:6px}
.cs-full{font-size:15px;line-height:1.6;margin:8px 0 4px 0;max-height:70vh;overflow:auto;padding:8px 10px;border-left:3px solid #d0d7de}
.cs-card mark, .cs-exact{background:#ffe08a;color:#1f2328;padding:0 2px;border-radius:3px}
.cs-sent{background:#cfe8ff;color:#1f2328;padding:0 2px;border-radius:3px}
.cs-legend{font-size:13px;color:#57606a;margin:4px 0}
.cs-muted{color:#57606a;font-size:13px}
.cs-table, .cs-table table, .cs-table td, .cs-table th {font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;}
</style>
"""


def meta_columns(comments: pd.DataFrame) -> list[str]:
    """Metadata columns from the file (everything except the text and internal columns), in file order."""
    return [c for c in comments.columns if c not in _TABLE_INTERNAL]


def matched_terms(passage: str, terms: list[str]) -> list[str]:
    """Query terms that appear as whole words in the passage."""
    return [t for t in terms if re.search(rf"\b{re.escape(t)}\b", passage, re.IGNORECASE)]


def _message(text: str) -> str:
    return f"{CARD_CSS}<p class='cs-muted'>{html.escape(text)}</p>"


def render_cards(frame: pd.DataFrame, terms: list[str], text_by_id: dict, collapse: bool) -> str:
    """HTML cards for the results (kept for export and tests; the window shows the table)."""
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
    """One row per hit: rank, comment ID, the file's metadata columns, and a short excerpt."""
    cols = ["rank", "comment_id", *meta_cols, "excerpt"]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"rank": frame["rank"].to_numpy(), "comment_id": frame["comment_id"].to_numpy()})
    for col in meta_cols:
        out[col] = [_cell(v) for v in frame[col]] if col in frame.columns else ""
    excerpts = [str(s) for s in frame["snippet"]]
    out["excerpt"] = [s[:EXCERPT_CHARS] + ("…" if len(s) > EXCERPT_CHARS else "") for s in excerpts]
    return out[cols]


def default_table_columns(hits: pd.DataFrame) -> list[str]:
    """Shown by default: rank, comment ID, and the first few metadata columns, then the excerpt."""
    cols = list(hits.columns)
    meta = [c for c in cols if c not in ("rank", "comment_id", "excerpt")]
    return ["rank", "comment_id", *meta[:3], "excerpt"] if "excerpt" in cols else cols


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

    def run(query, match, similarity, collapse):
        no_columns = gr.update()
        if not (query or "").strip():
            return "Type a word or phrase to search.", None, empty_hits, None, no_columns, None
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
        return (
            status,
            csv_path,
            select_table_columns(hits, defaults),
            frame,
            gr.update(choices=choices, value=defaults),
            hits,
        )

    with gr.Blocks(title=title, fill_height=True) as demo:
        gr.HTML(CARD_CSS)
        gr.Markdown(
            f"## {title}\nSearch {n_comments:,} comments. Models run locally; no comment text leaves this runtime."
        )
        gr.Markdown(HOW_TO_TEXT)
        with gr.Row():
            query = gr.Textbox(
                label="Search", placeholder='e.g. economic impacts  or  "public hearing"', scale=4, autofocus=True
            )
            go = gr.Button("Search", variant="primary", scale=1)
        with gr.Row():
            match = gr.Radio(MATCH_MODES, value=MATCH_EXACT, label="Match", info=MATCH_INFO, scale=3)
            similarity = gr.Slider(
                0.0, 1.0, value=DEFAULT_MIN_SIMILARITY, step=0.01,
                label="Similarity cutoff (cosine)", info=SIMILARITY_INFO, visible=False, scale=2,
            )
            collapse = gr.Checkbox(value=True, label="Show one comment per group of duplicates", scale=2)
        status = gr.Markdown()
        download = gr.DownloadButton("Download results as CSV", value=None)

        frame_state = gr.State(None)
        row_state = gr.State(None)
        hits_state = gr.State(None)  # every column of every hit; the table shows a chosen subset

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                table_columns = gr.CheckboxGroup(choices=list(empty_hits.columns), value=[], label="Table columns")
                hits = gr.Dataframe(
                    value=empty_hits,
                    interactive=False,
                    wrap=False,
                    label="Hits (click a row to preview it)",
                    max_height=650,
                    elem_classes="cs-table",
                )
            with gr.Column(scale=2):
                show_mode = gr.Radio(SHOW_MODES, value="Excerpt", label="Show", info=SHOW_MODE_INFO)
                detail = gr.HTML(value=_message("Click a row in the table to see the comment here."))

        with gr.Accordion("Compare the two exact-match engines (developer check)", open=False):
            compare_btn = gr.Button("Compare for the current search")
            compare_out = gr.Markdown()

        inputs = [query, match, similarity, collapse]
        outputs = [status, download, hits, frame_state, table_columns, hits_state]
        go.click(run, inputs=inputs, outputs=outputs)
        query.submit(run, inputs=inputs, outputs=outputs)
        query.change(run, inputs=inputs, outputs=outputs, trigger_mode="always_last")
        match.change(run, inputs=inputs, outputs=outputs)
        similarity.release(run, inputs=inputs, outputs=outputs)
        collapse.change(run, inputs=inputs, outputs=outputs)

        def on_match(choice):
            return gr.update(visible=choice == MATCH_FUZZY)

        match.change(on_match, inputs=match, outputs=similarity)

        def on_columns(chosen, all_hits):
            if all_hits is None:
                return empty_hits
            return select_table_columns(all_hits, chosen)

        table_columns.change(on_columns, inputs=[table_columns, hits_state], outputs=hits)

        def sentences_for(frame, row, query_text):
            if frame is None or row is None or len(frame) == 0 or not query_text:
                return None
            comment_row = int(frame.attrs["rows"][int(row)])
            return searcher.sentence_similarities(query_text, [comment_row]).get(comment_row, [])

        def show(frame, row, mode):
            query_text = (frame.attrs.get("plan") or {}).get("query") if frame is not None and len(frame) else None
            return detail_html(frame, row, mode, text_by_id, table_meta, sentences_for(frame, row, query_text))

        def on_select(evt: gr.SelectData, frame, mode):
            row = evt.index[0]
            return show(frame, row, mode), row

        def on_mode(mode, frame, row):
            return show(frame, row, mode)

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

        hits.select(on_select, inputs=[frame_state, show_mode], outputs=[detail, row_state])
        show_mode.change(on_mode, inputs=[show_mode, frame_state, row_state], outputs=detail)
        compare_btn.click(on_compare, inputs=query, outputs=compare_out)
    return demo


def launch(searcher, share: bool = False, **kwargs):
    """Launch the UI. ``share=True`` would route traffic through Gradio's servers, so it is off by default."""
    demo = make_app(searcher)
    return demo.launch(share=share, **kwargs)
