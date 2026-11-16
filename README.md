# comment_search

Local hybrid search over public comments: dense embeddings (meaning) plus BM25 (keywords), with duplicate collapsing and a Gradio UI for demos. Models are downloaded once from Hugging Face. Comment text is never sent to an outside service.

## Run it in Colab

Open `notebooks/quickstart.ipynb` from the Colab badge in that file. The first code cell installs the package straight from this public repo, so no token or Drive copy is needed. Then edit the paths and column names in the settings cell.

## Run it locally

```bash
pip install -e ".[test]"
python -m pytest -q        # no model downloads: the tests use a fake encoder
```

To run the notebook locally, set `COMMENTS_PATH` and `COMMENTSEARCH_INDEX_DIR` (the notebook reads them if set) and open `notebooks/quickstart.ipynb` with Jupyter.

## What it does

1. Loads CSV, XLSX, or Parquet. Strips HTML, collapses whitespace, drops empty comments, and flags short or "see attached" comments (flagged, not removed).
2. Chunks each comment into about 250-word passages on sentence boundaries, with about 50 words of overlap.
3. Embeds passages with `BAAI/bge-small-en-v1.5` (configurable) and searches with exact cosine similarity in NumPy. Embeddings are cached to the index folder, so an interrupted build resumes.
4. Keyword search with `bm25s`.
5. Fuses dense and BM25 results with Reciprocal Rank Fusion (k=60), or a min-max blend with `alpha`.
6. Optionally reranks the top 50 passages with `BAAI/bge-reranker-base`.
7. Scores each comment by its best passage and shows that passage as a snippet.
8. Collapses duplicates: identical normalized text, cosine ≥ 0.95 on mean comment embeddings (configurable), and existing Grouper `group_id` groups. Each group shows one representative with "+N similar".
9. Applies metadata filters before ranking. Categorical columns become dropdowns, and a date column gets a from/to range.

## Index folder

`index.save(folder)` writes `comments.parquet`, `passages.parquet`, `passage_emb.npy`, `comment_emb.npy`, `bm25/`, and `meta.json`. `CommentIndex.load(folder)` reads it back. Keep the folder on Google Drive (or any local disk) between sessions.

## Known limitations

- Exact search only. Embeddings and comparisons are held in memory (about 0.5 GB for 100k comments with 384-dimension vectors). Duplicate detection compares every pair in blocks, so 100k comments will be slow on CPU.
- The reranker runs on CPU in Colab unless a GPU is attached, and it only reorders the top 50 passages.
- Dates are only parsed if the input has them.
- Stop words are English. Matching is on whole words, with no stemming, so "well" and "wells" are different terms for BM25.
- Snippets are the best-scoring passage. A comment that matches only through a passage outside the top pool will not appear.
- The UI's `share=True` routes traffic through Gradio's servers, so the notebook keeps it off.
- Notebook outputs contain comment text. Do not commit an executed notebook.
- Your comment data is not part of this repo. `data/` is git-ignored.
