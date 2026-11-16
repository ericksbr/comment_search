"""Gradio UI over a Searcher: query, fusion, rerank, collapse, metadata filters, result cards, CSV download."""

from __future__ import annotations

import html
import os
import re
import tempfile

import pandas as pd

from .search import highlight

MAX_CATEGORICAL_VALUES = 100
_NOT_FILTERABLE = {"comment_id", "text", "dup_group", "dup_group_size"}

CARD_CSS = """
<style>
.cs-results{font-family:inherit}
.cs-card{border:1px solid #d0d7de;border-radius:10px;padding:14px 16px;margin:12px 0;max-width:960px}
.cs-head{font-size:15px;margin-bottom:6px}
.cs-rank{font-weight:700;margin-right:6px}
.cs-score{color:#57606a;margin-left:8px;font-size:13px}
.cs-badge{background:#ddf4ff;color:#0969da;border-radius:12px;padding:1px 8px;margin-left:6px;font-size:13px}
.cs-chips{margin:6px 0}
.cs-chip{background:#fff8c5;border:1px solid #d4a72c;border-radius:12px;padding:1px 8px;margin-right:4px;font-size:13px;color:#24292f}
.cs-snip{font-size:16px;line-height:1.55;margin:8px 0}
.cs-full{font-size:15px;line-height:1.6;margin:8px 0 4px 0;max-height:420px;overflow:auto;padding:8px 10px;border-left:3px solid #d0d7de}
.cs-card details summary{cursor:pointer;color:#0969da;margin-top:6px}
.cs-card mark{background:#ffe08a;color:#1f2328;padding:0 2px;border-radius:3px}
.cs-muted{color:#57606a;font-size:13px}
</style>
"""


def _filter_columns(comments: pd.DataFrame):
    """Categorical columns with a short list of values, and datetime columns with at least one date."""
    categorical, dates = [], []
    for col in comments.columns:
        if col in _NOT_FILTERABLE:
            continue
        if pd.api.types.is_datetime64_any_dtype(comments[col]):
            if comments[col].notna().any():
                dates.append(col)
            continue
        values = sorted({str(v) for v in comments[col].unique() if str(v) != ""})
        if 1 < len(values) <= MAX_CATEGORICAL_VALUES:
            categorical.append((col, values))
    return categorical, dates


def _parse_date(text):
    text = (text or "").strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not read date: {text!r} (use YYYY-MM-DD)")
    return parsed


def matched_terms(passage: str, terms: list[str]) -> list[str]:
    """Query terms that appear as whole words in the passage."""
    return [t for t in terms if re.search(rf"\b{re.escape(t)}\b", passage, re.IGNORECASE)]


def _message(text: str) -> str:
    return f"{CARD_CSS}<p class='cs-muted'>{html.escape(text)}</p>"


def _card(row, terms: list[str], text_by_id: dict, collapse: bool) -> str:
    passage = str(row["passage"]) if "passage" in row else str(row["snippet"])
    full_comment = text_by_id.get(row["comment_id"], "")
    chips = "".join(f"<span class='cs-chip'>{html.escape(t)}</span>" for t in matched_terms(passage, terms))
    if not chips:
        chips = "<span class='cs-muted'>none of the query words appear in this passage</span>"
    similar = int(row["group_size"]) - 1
    badge = f"<span class='cs-badge'>+{similar} similar</span>" if collapse and similar > 0 else ""
    name = str(row.get("commenter_name", "") or "")
    name_html = f" · {html.escape(name)}" if name else ""
    return (
        "<div class='cs-card'>"
        f"<div class='cs-head'><span class='cs-rank'>#{int(row['rank'])}</span>"
        f"<b>{html.escape(str(row['comment_id']))}</b>{name_html}"
        f"<span class='cs-score'>score {float(row['score']):.3f}</span>{badge}</div>"
        f"<div class='cs-chips'>{chips}</div>"
        f"<div class='cs-snip'>{highlight(str(row['snippet']), terms)}</div>"
        "<details><summary>Read the full passage</summary>"
        f"<div class='cs-full'>{highlight(passage, terms)}</div></details>"
        "<details><summary>Read the full comment</summary>"
        f"<div class='cs-full'>{highlight(full_comment, terms)}</div></details>"
        "</div>"
    )


def render_cards(frame: pd.DataFrame, terms: list[str], text_by_id: dict, collapse: bool) -> str:
    """HTML for the results: one card per comment, highlighted matches, expandable full text."""
    if frame.empty:
        return _message("No matching comments. Check the filters.")
    shown = ", ".join(terms) if terms else "none"
    parts = [CARD_CSS, f"<div class='cs-results'><p class='cs-muted'>{len(frame)} results · query words: {html.escape(shown)}</p>"]
    parts += [_card(row, terms, text_by_id, collapse) for _, row in frame.iterrows()]
    parts.append("</div>")
    return "\n".join(parts)


def make_app(searcher, title: str = "Local comment search"):
    """Build the Gradio Blocks app. Nothing leaves the runtime; there is no external API."""
    import gradio as gr

    comments = searcher.index.comments
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    cat_cols, date_cols = _filter_columns(comments)

    def run(query, top_k, method, alpha, rerank, collapse, date_from, date_to, *cat_values):
        filters: dict = {}
        for (col, _), values in zip(cat_cols, cat_values):
            if values:
                filters[col] = list(values)
        if date_cols:
            lo, hi = _parse_date(date_from), _parse_date(date_to)
            if lo is not None or hi is not None:
                filters[date_cols[0]] = (lo, hi)
        if not (query or "").strip():
            return _message("Enter a query."), "Enter a query.", None
        frame = searcher.search(
            query,
            top_k=int(top_k),
            method=method,
            alpha=float(alpha),
            rerank=bool(rerank),
            collapse=bool(collapse),
            filters=filters,
        )
        terms = frame.attrs.get("query_terms", [])
        timings = frame.attrs.get("timings", {})
        status = "No matching comments (check the filters)." if frame.empty else "  ·  ".join(
            f"{k} {v:.3f}s" for k, v in timings.items()
        )
        cards = render_cards(frame, terms, text_by_id, bool(collapse))
        fd, csv_path = tempfile.mkstemp(suffix=".csv", prefix="commentsearch_results_")
        os.close(fd)
        frame.to_csv(csv_path, index=False)  # plain text in the CSV (no <mark> tags)
        return cards, status, csv_path

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"## {title}\nHybrid search over {len(comments):,} comments "
            f"({searcher.index.meta['n_passages']:,} passages). Models run locally; no comment text leaves this runtime."
        )
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="e.g. economic impacts on ranchers", scale=4)
            go = gr.Button("Search", variant="primary", scale=1)
        with gr.Row():
            top_k = gr.Slider(1, 50, value=10, step=1, label="Top-k comments")
            method = gr.Radio(["rrf", "blend", "dense", "bm25"], value="rrf", label="Fusion (rrf = Reciprocal Rank Fusion)")
            alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="Blend alpha (dense weight, used only by blend)")
        with gr.Row():
            rerank = gr.Checkbox(value=False, label="Rerank top 50 passages with bge-reranker-base")
            collapse = gr.Checkbox(value=True, label="Collapse near-duplicates")
        with gr.Accordion("Metadata filters (applied before ranking)", open=False):
            cat_inputs = [
                gr.Dropdown(choices=values, value=[], multiselect=True, label=col) for col, values in cat_cols
            ]
            if date_cols:
                with gr.Row():
                    date_from = gr.Textbox(label=f"{date_cols[0]} from (YYYY-MM-DD)")
                    date_to = gr.Textbox(label=f"{date_cols[0]} to (YYYY-MM-DD)")
            else:
                date_from = gr.Textbox(visible=False)
                date_to = gr.Textbox(visible=False)
        status = gr.Markdown()
        cards = gr.HTML(value=_message("Type a query and press Search."))
        download = gr.DownloadButton("Download results as CSV", value=None)

        inputs = [query, top_k, method, alpha, rerank, collapse, date_from, date_to, *cat_inputs]
        outputs = [cards, status, download]
        go.click(run, inputs=inputs, outputs=outputs)
        query.submit(run, inputs=inputs, outputs=outputs)
    return demo


def launch(searcher, share: bool = False, **kwargs):
    """Launch the UI. ``share=True`` would route traffic through Gradio's servers, so it is off by default."""
    demo = make_app(searcher)
    return demo.launch(share=share, **kwargs)
