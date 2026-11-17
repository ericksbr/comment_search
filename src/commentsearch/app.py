"""Gradio UI over a Searcher: a query box that updates as you type, exact or fuzzy matching, a hits list with a
preview panel, and CSV download. Nothing leaves the runtime."""

import html
import os
import re
import tempfile

import pandas as pd

from .columns import detect_name_column
from .exact import MODE_ALL, MODE_ANY, highlight_exact
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

# Exact = every word must appear (stems count), and "quoted phrases" must appear in order.
# Fuzzy = comments similar in meaning to the query, at or above a cutoff, at the sentence or passage level.
MATCH_EXACT = "Exact words"
MATCH_FUZZY = "Fuzzy (by meaning)"
MATCH_MODES = [MATCH_EXACT, MATCH_FUZZY]
EXACT_ALL = "All words"
EXACT_ANY = "Any word"
EXACT_MODE_INFO = (
    "All words: every word and phrase must appear. Any word: a comment matches if it contains any one of them "
    "(useful for acronyms, e.g. PTW \"permissible total withdrawal\")."
)
LEVEL_SENTENCE = "Sentence match"
LEVEL_PASSAGE = "Passage match"
LEVEL_INFO = (
    "Sentence match: a comment is listed if one of its sentences is at or above the cutoff, and each such "
    "sentence is shaded blue. Passage match: a comment is listed if its best passage (about 250 words) is at "
    "or above the cutoff, and that passage is shaded blue."
)
# Sentence scores run lower than passage scores, so each level keeps its own cutoff.
# Provisional guesses; a data-based default is planned.
DEFAULT_CUTOFF = {LEVEL_SENTENCE: 0.5, LEVEL_PASSAGE: 0.6}
SIMILARITY_TIP = (
    "Similarity: how closely the best-matching sentence or passage matches your search. "
    "1 = same meaning, 0 = unrelated. Scores depend on the model, so compare within one search."
)
SCORED_METHODS = ("semantic", "sentence")

SORT_BEST = "Best match"
SORT_SIM = "Similarity"
SORT_ID = "Comment ID"
SORT_DATE = "Date"
SORT_ORDERS = ["Ascending", "Descending"]

HOW_TO_TEXT = (
    "**Exact words:** every word you type must appear in the comment. Word endings are ignored, so "
    "\"impact\" finds \"impacts\". Put words in quotes to require them in that order.\n\n"
    "**Fuzzy (by meaning):** finds comments similar in meaning, even without the same words. Choose "
    "**Sentence match** or **Passage match** to decide what is compared and shaded blue. The cutoff sets how "
    "close a match must be."
)
HOW_TO_PLAIN = (
    "Exact words: every word must appear; word endings are ignored ('impact' finds 'impacts'); "
    "quotes require the words in order. Yellow = matched words. "
    "Fuzzy (by meaning): comments similar in meaning, even without the same words. "
    "Sentence match lists comments with a sentence at or above the cutoff and shades those sentences blue. "
    "Passage match lists comments whose best passage is at or above the cutoff and shades that passage blue. "
    "Arrow keys or j / k move through the hits."
)

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


def has_score(frame) -> bool:
    return frame is not None and frame.attrs.get("method") in SCORED_METHODS


def hits_table(frame: pd.DataFrame, meta_cols: list[str]) -> pd.DataFrame:
    """One row per hit: similarity (fuzzy searches only), comment ID and the file's metadata columns.
    The text is shown in the preview, not the list."""
    scored = has_score(frame)
    cols = (["similarity"] if scored else []) + ["comment_id", *meta_cols]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    out = {}
    if scored:
        out["similarity"] = [f"{float(s):.2f}" for s in frame["score"]]
    out["comment_id"] = frame["comment_id"].to_numpy()
    for col in meta_cols:
        out[col] = [_cell(v) for v in frame[col]] if col in frame.columns else [""] * len(frame)
    return pd.DataFrame(out)[cols]


def default_table_columns(hits: pd.DataFrame) -> list[str]:
    """Shown by default: similarity (if any), comment ID, and the first few metadata columns."""
    cols = list(hits.columns)
    meta = [c for c in cols if c not in ("similarity", "comment_id")]
    return [c for c in ("similarity", "comment_id") if c in cols] + meta[:3]


def select_table_columns(hits: pd.DataFrame, chosen) -> pd.DataFrame:
    """The table with only the chosen columns, in the file's order."""
    keep = [c for c in hits.columns if c in set(chosen)]
    return hits[keep] if keep else hits


def sort_options(match: str, date_col: str | None) -> list[str]:
    options = [SORT_BEST] + ([SORT_SIM] if match == MATCH_FUZZY else []) + [SORT_ID]
    return options + ([SORT_DATE] if date_col else [])


def effective_sort(key: str, match: str, date_col: str | None) -> str:
    """The sort key if this search offers it (e.g. not 'Similarity' for exact words), otherwise best match."""
    return key if key in sort_options(match, date_col) else SORT_BEST


def sort_frame(frame, key: str, descending: bool, date_col: str | None = None):
    """Reorder the hits by a column. The comment rows (used by the sentence lookup) move with them."""
    if frame is None or len(frame) == 0:
        return frame
    if key == SORT_SIM and has_score(frame):
        values = frame["score"].to_numpy()
    elif key == SORT_ID:
        values = frame["comment_id"].astype(str).to_numpy()
    elif key == SORT_DATE and date_col:
        values = pd.to_datetime(frame[date_col], errors="coerce").to_numpy()
    else:  # best match first: the rank the search gave
        values = -frame["rank"].to_numpy()
    order = pd.Series(values).sort_values(kind="stable", na_position="last").index.to_numpy()
    if descending:
        order = order[::-1]
    out = frame.iloc[order].reset_index(drop=True)
    out.attrs = dict(frame.attrs)
    out.attrs["rows"] = frame.attrs["rows"][order]
    return out


def _pattern(text: str) -> re.Pattern:
    """Matches the words of text with any whitespace between them (passages and sentences differ in spacing)."""
    return re.compile(r"\s+".join(re.escape(w) for w in text.split()))


def sentence_ranges(body: str, sentences, min_similarity: float) -> list[tuple[int, int, str]]:
    """Where each sentence at or above the cutoff sits in the body text."""
    ranges, pos = [], 0
    for text, score in sentences or []:
        if score < min_similarity or not text.strip():
            continue
        m = _pattern(text.strip()).search(body, pos)
        if m is None:
            continue
        ranges.append((m.start(), m.end(), f"similarity {score:.2f}"))
        pos = m.end()
    return ranges


def shade(body: str, ranges: list[tuple[int, int, str]]) -> str:
    """HTML: the body with the given ranges shaded blue (hover shows the similarity)."""
    out, pos = [], 0
    for start, end, title in ranges:
        out.append(html.escape(body[pos:start]))
        out.append(f'<span class="cs-sent" title="{html.escape(title, quote=True)}">{html.escape(body[start:end])}</span>')
        pos = end
    out.append(html.escape(body[pos:]))
    return "".join(out)


def sentence_spans(body: str, sentences, min_similarity: float) -> str:
    """HTML: the body with every sentence at or above the cutoff shaded blue."""
    return shade(body, sentence_ranges(body, sentences, min_similarity))


def index_name_column(index) -> str:
    """The commenter-name column chosen when the index was built. Older indexes: detect it ("auto")."""
    return index.meta["name_column"] if "name_column" in index.meta else "auto"


def detail_html(frame, row, mode: str, text_by_id: dict, meta_cols: list[str], sentences=None,
                name_col: str | None = "auto") -> str:
    """The selected hit. Exact matches are shaded yellow. Fuzzy matches are shaded blue: the similar sentences
    (sentence match) or the matching passage (passage match).

    ``sentences`` is the list of (sentence, similarity) for this comment, from the searcher.
    """
    if frame is None or len(frame) == 0 or row is None:
        return _message("Click a row in the table to see the comment here.")
    r = frame.iloc[int(row)]
    plan = frame.attrs.get("plan") or {}
    method = frame.attrs.get("method")
    cutoff = float(plan.get("min_similarity", 0.0))
    score = float(r["score"]) if has_score(frame) else None
    passage_text = str(r["passage"]) if "passage" in r.index else str(r["snippet"])
    if mode == "Excerpt":
        body = str(r["snippet"])
    elif mode == "Full comment":
        body = text_by_id.get(r["comment_id"], "")
    else:
        body = passage_text

    if method == "exact" and plan:
        body_html = highlight_exact(body, plan)
        legend = "Yellow: words that matched (word endings count, so 'impact' matches 'impacts')."
    elif method == "sentence":
        body_html = sentence_spans(body, sentences, cutoff)
        legend = f"Blue: sentences at or above similarity {cutoff:.2f} to your search."
    elif method == "semantic":
        if mode == "Full comment":
            # Shade the matching passage inside the full comment.
            m = _pattern(passage_text).search(body) if passage_text.strip() else None
            ranges = [(m.start(), m.end(), f"passage similarity {score:.2f}")] if m else []
        else:
            # Excerpt and Passage are both inside the matching passage: shade all of it.
            ranges = [(0, len(body), f"passage similarity {score:.2f}")] if body else []
        body_html = shade(body, ranges)
        legend = f"Blue: the passage that matched (similarity {score:.2f}, cutoff {cutoff:.2f})."
    else:
        body_html = html.escape(body)
        legend = ""
    comment_id = html.escape(str(r["comment_id"]))
    if name_col == "auto":
        name_col = detect_name_column(frame)
    name = str(r[name_col]).strip() if name_col and name_col in r.index and not pd.isna(r[name_col]) else ""
    if name and name.lower() not in ("none", "nan"):
        heading = f"<b>{html.escape(name)}</b> ({comment_id})"
    else:
        heading = f"<b>{comment_id}</b>"
    return (
        f"{CARD_CSS}<div class='cs-card'>"
        f"<div class='cs-head'>{heading}</div>"
        f"<div class='cs-legend'>{html.escape(legend)}</div>"
        f"<div class='cs-full'>{body_html}</div>"
        "</div>"
    )


def position_text(frame, row) -> str:
    """'Hit 3 of 45' under the list, or nothing when there are no hits."""
    if frame is None or len(frame) == 0 or row is None:
        return ""
    return f"Hit {int(row) + 1:,} of {len(frame):,}"


def exact_mode_for(label: str) -> str:
    """The search's mode value for a button label: 'Any word' -> 'any', anything else -> 'all'."""
    return MODE_ANY if label == EXACT_ANY else MODE_ALL


def search_comments(searcher, query: str, match: str, collapse: bool, cutoff: float, level: str,
                    exact_mode: str = MODE_ALL):
    """Exact: every comment with all the words (or phrases). Fuzzy: at or above the cutoff, at the chosen level."""
    n = len(searcher.index.comments)
    if match == MATCH_EXACT:
        return searcher.search(query, top_k=n, method="exact", backend="fts5", collapse=collapse, exact_mode=exact_mode)
    method = "sentence" if level == LEVEL_SENTENCE else "semantic"
    return searcher.search(query, top_k=n, method=method, min_similarity=float(cutoff), collapse=collapse)


def make_app(searcher, title: str = "Local comment search"):
    """Build the Gradio Blocks app. Nothing leaves the runtime; there is no external API."""
    import gradio as gr

    comments = searcher.index.comments
    n_comments = len(comments)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    table_meta = meta_columns(comments)
    empty_hits = hits_table(pd.DataFrame(), table_meta)
    date_cols = [c for c in comments.columns if pd.api.types.is_datetime64_any_dtype(comments[c]) and comments[c].notna().any()]
    date_col = date_cols[0] if date_cols else None

    def sentences_for(frame, row):
        """Sentence similarities for the selected comment (sentence match only)."""
        if frame is None or row is None or len(frame) == 0 or frame.attrs.get("method") != "sentence":
            return None
        query_text = (frame.attrs.get("plan") or {}).get("query")
        if not query_text:
            return None
        comment_row = int(frame.attrs["rows"][int(row)])
        return searcher.sentence_similarities(query_text, [comment_row]).get(comment_row, [])

    name_col = index_name_column(searcher.index)

    def render(frame, row, mode):
        detail = detail_html(frame, row, mode, text_by_id, table_meta, sentences_for(frame, row), name_col)
        return detail, position_text(frame, row)

    def run(query, match, cutoffs, level, collapse, mode, sort_key, descending, exact_mode):
        sort_key = effective_sort(sort_key, match, date_col)
        keep_sort = gr.update(choices=sort_options(match, date_col), value=sort_key)
        if not (query or "").strip():
            return ("Type a word or phrase to search.", None, empty_hits, None, gr.update(), None, None,
                    *render(None, None, mode), keep_sort)
        cutoff = cutoffs.get(level, DEFAULT_CUTOFF[level])
        frame = search_comments(searcher, query, match, bool(collapse), cutoff, level, exact_mode)
        frame = sort_frame(frame, sort_key, bool(descending), date_col)
        if frame.empty:
            if match == MATCH_EXACT:
                status = ("No comment contains all of these words (or phrases in quotes)."
                          if exact_mode == MODE_ALL else "No comment contains any of these words or phrases.")
            elif level == LEVEL_SENTENCE and not searcher.has_sentences:
                status = "Rebuild the index in Act 1 to see sentence matches."
            else:
                status = (
                    f"**No close matches** at similarity {float(cutoff):.2f} or higher. "
                    "Move the cutoff left to see weaker matches."
                )
        else:
            n = len(frame)
            if match == MATCH_EXACT:
                status = f"**{n:,} comments** contain " + ("all of the words." if exact_mode == MODE_ALL else "any of the words.")
            else:
                unit = "a sentence" if level == LEVEL_SENTENCE else "a passage"
                status = f"**{n:,} comments** with {unit} at similarity {float(cutoff):.2f} or higher."
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
            keep_sort,
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
            collapse = gr.Checkbox(value=True, label="One per duplicate group", scale=2)
        with gr.Row(visible=True) as exact_row:
            all_btn = gr.Button(EXACT_ALL, size="sm", variant="primary", min_width=110)
            any_btn = gr.Button(EXACT_ANY, size="sm", variant="secondary", min_width=110)
            gr.HTML(
                f"<span class='cs-info' title='{html.escape(EXACT_MODE_INFO, quote=True)}'>ⓘ</span>",
                min_width=40,
            )
        with gr.Row(visible=False) as fuzzy_row:
            sentence_btn = gr.Button(LEVEL_SENTENCE, size="sm", variant="primary", min_width=120)
            passage_btn = gr.Button(LEVEL_PASSAGE, size="sm", variant="secondary", min_width=120)
            gr.HTML(
                f"<span class='cs-info' title='{html.escape(LEVEL_INFO, quote=True)}'>ⓘ</span>",
                min_width=40,
            )
            gr.Markdown("**Similarity cutoff** (higher = stricter)", min_width=180)
            similarity = gr.Slider(
                0.0, 1.0, value=DEFAULT_CUTOFF[LEVEL_SENTENCE], step=0.01,
                show_label=False, container=False, scale=3,
            )
        status = gr.Markdown()

        match_state = gr.State(MATCH_EXACT)
        exact_mode_state = gr.State(MODE_ALL)
        level_state = gr.State(LEVEL_SENTENCE)
        cutoffs_state = gr.State(dict(DEFAULT_CUTOFF))
        frame_state = gr.State(None)
        row_state = gr.State(None)
        hits_state = gr.State(None)  # every column of every hit; the list shows a chosen subset

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                with gr.Accordion("Columns and download", open=False):
                    table_columns = gr.CheckboxGroup(choices=list(empty_hits.columns), value=[], label="Columns")
                    download = gr.DownloadButton("Download results as CSV", value=None)
                with gr.Row():
                    sort_dd = gr.Dropdown(choices=sort_options(MATCH_EXACT, date_col), value=SORT_BEST,
                                          label="Sort by", scale=3)
                    sort_order = gr.Radio(SORT_ORDERS, value="Descending", label="Order", scale=2)
                gr.HTML(
                    f"<div class='cs-muted'>Hits <span class='cs-info' title='{html.escape(SIMILARITY_TIP, quote=True)}'>"
                    "ⓘ similarity</span></div>"
                )
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

        inputs = [query, match_state, cutoffs_state, level_state, collapse, show_mode, sort_dd, sort_order,
                  exact_mode_state]
        outputs = [status, download, hits, frame_state, table_columns, hits_state, row_state, detail, position, sort_dd]
        go.click(run, inputs=inputs, outputs=outputs)
        query.submit(run, inputs=inputs, outputs=outputs)
        query.change(run, inputs=inputs, outputs=outputs, trigger_mode="always_last")
        collapse.change(run, inputs=inputs, outputs=outputs)

        def set_match(choice):
            is_fuzzy = choice == MATCH_FUZZY
            return (
                choice,
                gr.update(variant="secondary" if is_fuzzy else "primary"),
                gr.update(variant="primary" if is_fuzzy else "secondary"),
                gr.update(visible=is_fuzzy),
                gr.update(visible=not is_fuzzy),
            )

        match_outputs = [match_state, exact_btn, fuzzy_btn, fuzzy_row, exact_row]

        def set_exact_mode(label):
            is_all = label == EXACT_ALL
            return (
                exact_mode_for(label),
                gr.update(variant="primary" if is_all else "secondary"),
                gr.update(variant="secondary" if is_all else "primary"),
            )

        exact_mode_outputs = [exact_mode_state, all_btn, any_btn]
        all_btn.click(lambda: set_exact_mode(EXACT_ALL), None, exact_mode_outputs).then(run, inputs, outputs)
        any_btn.click(lambda: set_exact_mode(EXACT_ANY), None, exact_mode_outputs).then(run, inputs, outputs)
        exact_btn.click(lambda: set_match(MATCH_EXACT), None, match_outputs).then(run, inputs, outputs)
        fuzzy_btn.click(lambda: set_match(MATCH_FUZZY), None, match_outputs).then(run, inputs, outputs)

        def set_level(choice, cutoffs):
            is_sentence = choice == LEVEL_SENTENCE
            return (
                choice,
                gr.update(variant="primary" if is_sentence else "secondary"),
                gr.update(variant="secondary" if is_sentence else "primary"),
                gr.update(value=cutoffs.get(choice, DEFAULT_CUTOFF[choice])),
            )

        level_outputs = [level_state, sentence_btn, passage_btn, similarity]
        sentence_btn.click(lambda c: set_level(LEVEL_SENTENCE, c), [cutoffs_state], level_outputs).then(run, inputs, outputs)
        passage_btn.click(lambda c: set_level(LEVEL_PASSAGE, c), [cutoffs_state], level_outputs).then(run, inputs, outputs)

        def on_cutoff(value, cutoffs, level):
            return {**cutoffs, level: float(value)}

        similarity.release(on_cutoff, [similarity, cutoffs_state, level_state], [cutoffs_state]).then(run, inputs, outputs)

        def on_columns(chosen, all_hits):
            if all_hits is None:
                return empty_hits
            return select_table_columns(all_hits, chosen)

        table_columns.change(on_columns, inputs=[table_columns, hits_state], outputs=hits)

        def on_sort(key, order, frame, chosen):
            if frame is None:
                return frame, empty_hits, None, None, *render(None, None, "Excerpt")
            ordered = sort_frame(frame, key, order == "Descending", date_col)
            all_hits = hits_table(ordered, table_meta)
            first = 0 if len(ordered) else None
            detail_text, pos = render(ordered, first, "Excerpt")
            return ordered, select_table_columns(all_hits, chosen or default_table_columns(all_hits)), all_hits, first, detail_text, pos

        sort_outputs = [frame_state, hits, hits_state, row_state, detail, position]
        sort_dd.change(on_sort, [sort_dd, sort_order, frame_state, table_columns], sort_outputs)
        sort_order.change(on_sort, [sort_dd, sort_order, frame_state, table_columns], sort_outputs)

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
