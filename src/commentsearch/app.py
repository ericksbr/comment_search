"""Gradio UI over a Searcher: a query box that updates as you type, exact or fuzzy matching, a hits list with a
preview panel, and selecting passages to view together or export to Excel. Nothing leaves the runtime."""

import html
import os
import re
import tempfile
from datetime import datetime

import pandas as pd

from .columns import detect_date_column, detect_name_column, detect_page_column
from .exact import MODE_ALL, MODE_ANY, highlight_exact

_TABLE_INTERNAL = {"comment_id", "text", "dup_group", "dup_group_size", "boilerplate"}
# Shown on the right of the list.
SHOW_MODES = ["Passage", "Full comment"]
SHOW_MODE_INFO = (
    "Passage: the part of the comment that matched best (about 250 words), with the matching words or "
    "sentences shaded. Full comment: the whole comment text."
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
# The direction each sort key starts with when chosen. Best match and similarity: strongest first.
# Comment ID: A to Z. Date: newest first.
DEFAULT_SORT_ORDER = {SORT_BEST: "Descending", SORT_SIM: "Descending", SORT_ID: "Ascending", SORT_DATE: "Descending"}

# The first column of the list: a tick box for picking passages to view together or export.
PICK = "Pick"
EXCEL_TEXT_LIMIT = 32000  # Excel allows 32,767 characters per cell; stay under it.

HOW_TO_PLAIN = (
    "Exact words: every word must appear; word endings are ignored ('impact' finds 'impacts'); "
    "quotes require the words in order. Yellow = matched words. "
    "Fuzzy (by meaning): comments similar in meaning, even without the same words. "
    "Sentence match lists comments with a sentence at or above the cutoff and shades those sentences blue. "
    "Passage match lists comments whose best passage is at or above the cutoff and shades that passage blue. "
    "Tick Pick in the list to choose passages. With any ticked, the right side shows them together. "
    "Columns and export (top left) saves the ticked passages to Excel. Arrow keys or j / k move through the hits."
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
.cs-strip{font-size:13.5px;padding-left:20px;margin:4px 0;line-height:1.5}
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


def _message(text: str) -> str:
    return f"{CARD_CSS}<p class='cs-muted'>{html.escape(text)}</p>"


def _cell(value) -> str:
    """A table cell as text: blank for missing values, the date for dates, the text otherwise."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def has_score(frame) -> bool:
    return frame is not None and frame.attrs.get("method") in SCORED_METHODS


def column_roles(meta_cols: list[str], name_col: str | None = "auto") -> tuple[str | None, str | None, str | None]:
    """The commenter name, page and date columns, if the file has them. The name column is the one chosen when
    the index was built ("auto" means detect it)."""
    probe = pd.DataFrame(columns=list(meta_cols))
    name = detect_name_column(probe) if name_col == "auto" else name_col
    name = name if name in meta_cols else None
    return name, detect_page_column(probe), detect_date_column(probe)


def lead_columns(roles: tuple[str | None, str | None, str | None]) -> list[str]:
    """The columns that come first in the list and in the export: name, comment ID, page, date."""
    name, page, date = roles
    lead = []
    for col in (name, "comment_id", page, date):
        if col and col not in lead:
            lead.append(col)
    return lead


def hits_table(frame: pd.DataFrame, meta_cols: list[str], roles=None) -> pd.DataFrame:
    """One row per hit: name, comment ID, page and date first, then similarity (fuzzy searches only), then the
    file's other metadata. The text is shown in the preview, not the list."""
    roles = column_roles(meta_cols) if roles is None else roles
    scored = has_score(frame)
    lead = lead_columns(roles)
    cols = lead + (["similarity"] if scored else []) + [c for c in meta_cols if c not in lead]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    out = {}
    if scored:
        out["similarity"] = [f"{float(s):.2f}" for s in frame["score"]]
    out["comment_id"] = frame["comment_id"].to_numpy()
    for col in meta_cols:
        out[col] = [_cell(v) for v in frame[col]] if col in frame.columns else [""] * len(frame)
    return pd.DataFrame(out)[cols]


def default_table_columns(hits: pd.DataFrame, roles=None) -> list[str]:
    """Shown by default: name, comment ID, page, date and similarity (if any). Other columns can be added."""
    cols = [c for c in hits.columns if c != PICK]
    roles = column_roles(cols) if roles is None else roles
    keep = {"similarity", "comment_id", *(r for r in roles if r)}
    return [c for c in cols if c in keep]


def select_table_columns(hits: pd.DataFrame, chosen) -> pd.DataFrame:
    """The table with only the chosen columns, in the file's order. The comment ID always stays, so each row
    can be matched to its passage."""
    wanted = set(chosen or [])
    keep = [c for c in hits.columns if c in wanted or c == "comment_id"]
    return hits[keep] if keep else hits


def with_pick(table: pd.DataFrame, selected) -> pd.DataFrame:
    """The table with a Pick column in front: ticked for each comment ID in ``selected``."""
    chosen = set(selected or [])
    out = table.copy()
    out.insert(0, PICK, [bool(cid in chosen) for cid in out["comment_id"]])
    return out


def table_datatypes(view: pd.DataFrame) -> list[str]:
    """Pick is a tick box; every other column is text."""
    return ["bool"] + ["str"] * (len(view.columns) - 1)


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


def display_name(row, name_col: str | None) -> str:
    """The commenter's name from the chosen column, or '' when there is none."""
    if not name_col or name_col not in row.index or pd.isna(row[name_col]):
        return ""
    name = str(row[name_col]).strip()
    return "" if name.lower() in ("none", "nan") else name


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
    if mode == "Full comment":
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
            # The passage view is the matching passage: shade all of it.
            ranges = [(0, len(body), f"passage similarity {score:.2f}")] if body else []
        body_html = shade(body, ranges)
        legend = f"Blue: the passage that matched (similarity {score:.2f}, cutoff {cutoff:.2f})."
    else:
        body_html = html.escape(body)
        legend = ""
    comment_id = html.escape(str(r["comment_id"]))
    if name_col == "auto":
        name_col = detect_name_column(frame)
    name = display_name(r, name_col)
    if name:
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


def selection_text(frame, ids, name_col: str | None) -> tuple[str, str]:
    """The selection count (Markdown) and the list of selected names for the collapsed table (HTML)."""
    if frame is None or not ids:
        return "**Nothing selected.** Tick Pick in the list to choose passages.", (
            "<p class='cs-muted'>Nothing selected. Show the table and tick Pick to choose passages.</p>"
        )
    chosen = set(ids)
    names = []
    for _, r in frame.iterrows():
        if r["comment_id"] in chosen:
            cid = html.escape(str(r["comment_id"]))
            names.append(f"<li>{html.escape(display_name(r, name_col)) or cid}</li>")
    n = len(names)
    return f"**{n:,} selected**", (
        f"<div class='cs-muted'>{n:,} selected</div><ol class='cs-strip'>{''.join(names)}</ol>"
    )


def selected_rows(frame, ids) -> list[int]:
    """Row numbers of the selected comments, in the order of the list."""
    chosen = set(ids or [])
    return [i for i, cid in enumerate(frame["comment_id"]) if cid in chosen]


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


def split_text(text: str, limit: int = EXCEL_TEXT_LIMIT) -> list[str]:
    """Split text into pieces of at most ``limit`` characters, breaking at spaces where possible.
    Joining the pieces gives back the original text exactly."""
    parts, rest = [], text
    while len(rest) > limit:
        cut = max(rest.rfind(" ", 1, limit), rest.rfind("\n", 1, limit))
        if cut <= 0:
            cut = limit
        parts.append(rest[:cut])
        rest = rest[cut:]
    parts.append(rest)
    return parts


_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def export_rows(frame, ids, text_by_id: dict, meta_cols: list[str], roles,
                chosen=None) -> tuple[list[str], list[list]]:
    """Headers and rows for the Excel export: the chosen list columns (all of them if ``chosen`` is None; the
    comment ID always), then the matched passage and the full text. The full text is split across several
    columns when it is longer than an Excel cell can hold."""
    keep = selected_rows(frame, ids)
    sub = frame.iloc[keep].reset_index(drop=True)
    sub.attrs = dict(frame.attrs)
    hits = hits_table(sub, meta_cols, roles)
    if chosen is not None:
        hits = select_table_columns(hits, chosen)
    passages = [str(p) for p in sub["passage"]] if "passage" in sub.columns else [str(s) for s in sub["snippet"]]
    pieces = [split_text(text_by_id.get(cid, "")) for cid in sub["comment_id"]]
    n_parts = max((len(p) for p in pieces), default=1)
    headers = [*hits.columns, "Matched passage", "Full text",
               *[f"Full text (part {k})" for k in range(2, n_parts + 1)]]
    rows = []
    for i, values in enumerate(hits.values.tolist()):
        text_cells = pieces[i] + [""] * (n_parts - len(pieces[i]))
        rows.append([*values, passages[i], *text_cells])
    return headers, rows


def _fill_sheet(ws, rows: list[list], bold_first_row: bool) -> None:
    """Write rows into a sheet. Text that starts with '=' is kept as text, not run as a formula."""
    from openpyxl.styles import Font

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            if isinstance(value, str):
                value = _ILLEGAL_XLSX.sub("", value)[:EXCEL_TEXT_LIMIT + 767]
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            if isinstance(value, str) and value.startswith("="):
                cell.data_type = "s"
            if bold_first_row and r_idx == 1:
                cell.font = Font(bold=True)


def write_xlsx(path: str, headers: list[str], rows: list[list], info_rows: list[list] | None = None) -> None:
    """Write the passages to one sheet, and the search settings to a second sheet if given."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Selected passages"
    _fill_sheet(ws, [headers, *rows], bold_first_row=True)
    ws.freeze_panes = "A2"
    if info_rows:
        info = wb.create_sheet("Search info")
        _fill_sheet(info, info_rows, bold_first_row=True)
        info.column_dimensions["A"].width = 24
        info.column_dimensions["B"].width = 90
    wb.save(path)


def export_filename(info) -> str:
    """commentsearch_export_<the search words>.xlsx, with anything but letters and digits replaced by '_'."""
    query = (info or {}).get("query", "")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_")[:60]
    return f"commentsearch_export_{slug}.xlsx" if slug else "commentsearch_export.xlsx"


def search_info_rows(info, n_rows: int, ticked: bool) -> list[list]:
    """The 'Search info' sheet: what was searched, how, when, and what the export holds."""
    info = info or {}
    if info.get("match") == MATCH_EXACT:
        method = "Exact words, " + ("all words" if info.get("exact_mode") == MODE_ALL else "any word")
    else:
        unit = "sentence" if info.get("level") == LEVEL_SENTENCE else "passage"
        method = f"Fuzzy (by meaning), {unit} match, cutoff {float(info.get('cutoff', 0.0)):.2f}"
    return [
        ["Setting", "Value"],
        ["Search", info.get("query", "")],
        ["Match", method],
        ["Duplicates", "One per duplicate group" if info.get("collapse") else "Every hit"],
        ["Search date and time", info.get("time", "")],
        ["Rows exported", n_rows],
        ["Rows", "Ticked passages only" if ticked else "All current hits"],
    ]


def make_app(searcher, title: str = "Local comment search"):
    """Build the Gradio Blocks app. Nothing leaves the runtime; there is no external API."""
    import gradio as gr

    comments = searcher.index.comments
    n_comments = len(comments)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    table_meta = meta_columns(comments)
    roles = column_roles(table_meta, index_name_column(searcher.index))
    name_col = roles[0]  # the commenter-name column itself ("auto" resolved), or None
    empty_hits = hits_table(pd.DataFrame(), table_meta, roles)
    empty_view = with_pick(empty_hits, [])
    list_choices = [c for c in empty_hits.columns if c != "comment_id"]
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

    def render(frame, row, mode, selected=()):
        """The right-hand panel: the ticked passages together if any are ticked, otherwise the one selected hit."""
        if frame is not None and len(frame) and selected:
            picked = selected_rows(frame, selected)
            if picked:
                cards = [detail_html(frame, i, mode, text_by_id, table_meta, sentences_for(frame, i), name_col)
                         for i in picked]
                body = "".join(card.replace(CARD_CSS, "", 1) for card in cards)
                return CARD_CSS + body, (
                    f"**{len(picked):,} selected.** Clear the selection to browse one hit at a time."
                )
        detail = detail_html(frame, row, mode, text_by_id, table_meta, sentences_for(frame, row), name_col)
        return detail, position_text(frame, row)

    def table_update(all_hits, chosen, selected):
        """The list with the chosen columns and the Pick tick boxes for the selected comments."""
        base = empty_hits if all_hits is None else all_hits
        view = with_pick(select_table_columns(base, chosen), selected)
        return gr.update(value=view, datatype=table_datatypes(view),
                         static_columns=list(range(1, len(view.columns))))

    def run(query, match, cutoffs, level, collapse, mode, sort_key, descending, exact_mode):
        sort_key = effective_sort(sort_key, match, date_col)
        keep_sort = gr.update(choices=sort_options(match, date_col), value=sort_key)
        if not (query or "").strip():
            detail, position = render(None, None, mode)
            count, strip = selection_text(None, [], name_col)
            return ("Type a word or phrase to search.", table_update(None, [], []), None, gr.update(), None, None,
                    detail, position, keep_sort, [], count, strip, None)
        cutoff = cutoffs.get(level, DEFAULT_CUTOFF[level])
        search_info = {
            "query": query.strip(), "match": match, "exact_mode": exact_mode, "level": level,
            "cutoff": float(cutoff), "collapse": bool(collapse), "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
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
        hits = hits_table(frame, table_meta, roles)
        defaults = default_table_columns(hits, roles) if len(hits.columns) else []
        choices = [c for c in (hits if len(hits.columns) else empty_hits).columns if c != "comment_id"]
        first = 0 if len(frame) else None
        detail, position = render(frame, first, mode)
        count, strip = selection_text(frame, [], name_col)
        return (
            status,
            table_update(hits, defaults, []),
            frame,
            gr.update(choices=choices, value=[c for c in defaults if c in choices]),
            hits,
            first,
            detail,
            position,
            keep_sort,
            [],
            count,
            strip,
            search_info,
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
            collapse = gr.Checkbox(value=False, label="One per duplicate group", scale=2)
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
        selected_state = gr.State([])  # comment IDs ticked in the list
        search_state = gr.State(None)  # the search settings, for the export's file name and 'Search info' sheet

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300) as left:
                with gr.Column() as table_panel:
                    with gr.Accordion("Columns and export", open=False):
                        table_columns = gr.CheckboxGroup(choices=list_choices, value=[], label="Columns")
                        export_btn = gr.Button("Export to Excel", size="sm")
                        export_file = gr.File(label="Excel file", interactive=False)
                        export_note = gr.Markdown()
                    with gr.Row():
                        sort_dd = gr.Dropdown(choices=sort_options(MATCH_EXACT, date_col), value=SORT_BEST,
                                              label="Sort by", scale=3)
                        sort_order = gr.Radio(SORT_ORDERS, value="Descending", label="Order", scale=2)
                    with gr.Row():
                        select_all_btn = gr.Button("Select all", size="sm")
                        select_none_btn = gr.Button("Select none", size="sm")
                        hide_btn = gr.Button("Hide table ◀", size="sm")
                    sel_count = gr.Markdown(selection_text(None, [], name_col)[0])
                    gr.HTML(
                        f"<div class='cs-muted'>Hits <span class='cs-info' title='{html.escape(SIMILARITY_TIP, quote=True)}'>"
                        "ⓘ similarity</span> · tick Pick to choose passages</div>"
                    )
                    hits = gr.Dataframe(
                        value=empty_view,
                        datatype=table_datatypes(empty_view),
                        static_columns=list(range(1, len(empty_view.columns))),
                        interactive=True,
                        wrap=False,
                        show_label=False,
                        max_height=520,
                        elem_classes="cs-table",
                    )
                    with gr.Row():
                        prev_btn = gr.Button("◀ Previous", size="sm")
                        position = gr.Markdown("", elem_classes="cs-position")
                        next_btn = gr.Button("Next ▶", size="sm")
                with gr.Column(visible=False) as strip_panel:
                    strip = gr.HTML(selection_text(None, [], name_col)[1])
                    show_btn = gr.Button("▶ Show table", size="sm")
            with gr.Column(scale=3):
                with gr.Row():
                    show_mode = gr.Radio(SHOW_MODES, value="Passage", show_label=False, container=False, scale=5)
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
        outputs = [status, hits, frame_state, table_columns, hits_state, row_state, detail, position, sort_dd,
                   selected_state, sel_count, strip, search_state]
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

        def on_columns(chosen, all_hits, selected):
            return table_update(all_hits, chosen, selected)

        table_columns.change(on_columns, inputs=[table_columns, hits_state, selected_state], outputs=hits)

        def on_sort(key, order, frame, chosen, selected, mode):
            if frame is None:
                detail_text, pos = render(None, None, mode)
                return frame, table_update(None, [], []), None, None, detail_text, pos
            ordered = sort_frame(frame, key, order == "Descending", date_col)
            all_hits = hits_table(ordered, table_meta, roles)
            first = 0 if len(ordered) else None
            detail_text, pos = render(ordered, first, mode, selected)
            shown = chosen or default_table_columns(all_hits, roles)
            return ordered, table_update(all_hits, shown, selected), all_hits, first, detail_text, pos

        def on_sort_key(key, frame, chosen, selected, mode):
            """Changing the sort key also sets the direction that key starts with."""
            order = DEFAULT_SORT_ORDER.get(key, "Descending")
            return (*on_sort(key, order, frame, chosen, selected, mode), gr.update(value=order))

        sort_outputs = [frame_state, hits, hits_state, row_state, detail, position]
        sort_inputs = [sort_dd, sort_order, frame_state, table_columns, selected_state, show_mode]
        sort_dd.change(on_sort_key, [sort_dd, frame_state, table_columns, selected_state, show_mode],
                       sort_outputs + [sort_order])
        sort_order.change(on_sort, sort_inputs, sort_outputs)

        def on_select(evt: gr.SelectData, frame, mode, selected):
            if evt.index[1] == 0:  # a tick in the Pick column: handled by the input event below
                return gr.update(), gr.update(), gr.update()
            row = evt.index[0]
            detail_text, pos = render(frame, row, mode, selected)
            return row, detail_text, pos

        def on_pick(table, frame, mode, row):
            """A tick box changed: the new selection is every ticked row in the list."""
            ids = [cid for cid, on in zip(table["comment_id"], table[PICK]) if on]
            detail_text, pos = render(frame, row, mode, ids)
            count, strip_html = selection_text(frame, ids, name_col)
            return ids, count, strip_html, detail_text, pos

        def apply_selection(ids, frame, all_hits, chosen, mode, row):
            detail_text, pos = render(frame, row, mode, ids)
            count, strip_html = selection_text(frame, ids, name_col)
            return ids, table_update(all_hits, chosen, ids), count, strip_html, detail_text, pos

        select_outputs = [selected_state, hits, sel_count, strip, detail, position]

        def on_select_all(frame, all_hits, chosen, mode, row):
            ids = [] if frame is None else list(frame["comment_id"])
            return apply_selection(ids, frame, all_hits, chosen, mode, row)

        def on_select_none(frame, all_hits, chosen, mode, row):
            return apply_selection([], frame, all_hits, chosen, mode, row)

        select_all_btn.click(on_select_all, [frame_state, hits_state, table_columns, show_mode, row_state], select_outputs)
        select_none_btn.click(on_select_none, [frame_state, hits_state, table_columns, show_mode, row_state], select_outputs)
        hits.select(on_select, [frame_state, show_mode, selected_state], [row_state, detail, position])
        hits.input(on_pick, [hits, frame_state, show_mode, row_state],
                   [selected_state, sel_count, strip, detail, position])

        def on_step(step, frame, row, mode, selected):
            if frame is None or len(frame) == 0 or row is None:
                return row, *render(frame, row, mode, selected)
            new_row = min(max(int(row) + step, 0), len(frame) - 1)
            return new_row, *render(frame, new_row, mode, selected)

        step_inputs = [frame_state, row_state, show_mode, selected_state]
        prev_btn.click(lambda f, r, m, s: on_step(-1, f, r, m, s), inputs=step_inputs, outputs=[row_state, detail, position])
        next_btn.click(lambda f, r, m, s: on_step(1, f, r, m, s), inputs=step_inputs, outputs=[row_state, detail, position])

        def on_mode(mode, frame, row, selected):
            return render(frame, row, mode, selected)[0]

        show_mode.change(on_mode, inputs=[show_mode, frame_state, row_state, selected_state], outputs=detail)

        def on_hide():
            return gr.update(visible=False), gr.update(visible=True), gr.update(scale=1)

        def on_show():
            return gr.update(visible=True), gr.update(visible=False), gr.update(scale=2)

        hide_btn.click(on_hide, None, [table_panel, strip_panel, left])
        show_btn.click(on_show, None, [table_panel, strip_panel, left])

        def on_export(frame, selected, chosen, info):
            if frame is None or len(frame) == 0:
                return None, "Nothing to export. Run a search first."
            ids = selected or list(frame["comment_id"])
            headers, rows = export_rows(frame, ids, text_by_id, table_meta, roles, chosen or [])
            folder = tempfile.mkdtemp(prefix="commentsearch_")
            path = os.path.join(folder, export_filename(info))
            write_xlsx(path, headers, rows, search_info_rows(info, len(rows), bool(selected)))
            which = "ticked" if selected else "all current"
            return path, f"Exported {len(rows):,} {which} hit(s) to Excel."

        export_btn.click(on_export, [frame_state, selected_state, table_columns, search_state],
                         [export_file, export_note])

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
