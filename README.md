# comment_search

Search over public comments in two ways: by exact words, or by meaning. The search runs in a browser window, with duplicate comments grouped so each group shows one representative.

## How the search works

The search works in two stages. First, the comments are prepared: the text is cleaned, each comment is split into sentences and into passages of about 250 words, and the result is saved as a searchable index. Second, when someone types a query, the search window checks that index in one of two ways. For **exact words**, it finds comments that contain the words typed, ranked by how distinctive the matches are. For **meaning**, it converts the query into a numerical "fingerprint" and finds sentences or passages with similar fingerprints, even when they use different words. Comments that are copies or near-copies of each other are grouped, so the results can show one representative per group.

### Step 1: Cleaning the comments

- Web formatting is removed, such as HTML tags and special characters, and extra spaces are collapsed.
- Comments with no text are dropped. The count is recorded.
- Very short comments (fewer than 5 words) and short "see attached" stubs (60 words or fewer) are flagged. They are kept in the data, not removed.

### Step 2: Splitting the text

- Each comment is split into **sentences**. The splitter knows common abbreviations, such as "Dr." and "U.S.", so it does not break a sentence there.
- Each comment is also split into **passages** of about 250 words. Passages break only at sentence boundaries. Each passage overlaps the next by up to about 50 words, so a point near a break is not lost.

### Step 3: Building the index

The index holds three things:

- **Meaning fingerprints.** Each sentence and each passage is converted into a list of 384 numbers by an open-source language model (`BAAI/bge-small-en-v1.5`). Texts with similar meaning get similar lists, whether or not they share words. Each comment's score is taken from its best-matching sentence or passage.
- **Passage keyword data.** Passages are stored with their word statistics, for the keyword ranking used by the comparison tools. The exact-word search in the window builds its own keyword table over whole comments the first time it is used in a session.
- **Duplicate groups.** Two comments are linked if their normalized text is identical (ignoring capitalization and punctuation), if their overall meaning fingerprints are at least 95% similar, or if they already share a group label in the source file. All linked comments form one group.

Building the index is the slow step, because the fingerprints are computed for every passage and sentence. A GPU speeds this up considerably. The index is then saved to a folder, so it does not have to be rebuilt for each search.

### Step 4: Searching for exact words

- **All words:** a comment must contain every word typed. For example, "tribe tribal" returns only comments containing both.
- **Any word:** a comment needs at least one of the words. Comments containing more of them rank higher.
- **Quotation marks:** words in quotes must appear together, in that order. `"public hearing"` will not match "hearing, public."
- Word endings are ignored, so "impact" also finds "impacts."
- Common English words such as "the" and "and" are ignored in the query.
- Results are ranked by a standard relevance score (BM25). It rewards words that are uncommon across all comments and that appear often in a given comment. The user can instead sort by comment ID, date, or similarity.

### Step 5: Searching by meaning

- The query is converted into the same kind of fingerprint, and each sentence or passage receives a **similarity score** from 0 to 1. A score near 1 means the same meaning, and a score near 0 means unrelated.
- **Sentence match** lists a comment if at least one of its sentences meets the cutoff. Each qualifying sentence is shaded blue.
- **Passage match** lists a comment if its best passage meets the cutoff. That passage is shaded blue.
- The default cutoffs are 0.5 for sentences and 0.6 for passages. These are starting values, not validated thresholds. The user can adjust them with the slider. Sentence scores run lower than passage scores, which is why they have separate cutoffs.
- If nothing reaches the cutoff, the window reports "No close matches."

### Step 6: Showing one result per duplicate group

By default, the list shows every hit. Tick **One per duplicate group** to show only the highest-ranked comment from each group of copies. Untick it to see every match again. The count in the status line depends on this setting.

### Limits to keep in mind

- Exact search matches words, not ideas. "Tribal nations" will not match a search for "tribe" unless the word forms share a root.
- Meaning search can miss an on-topic comment that is worded very differently from the query. It can also return a comment that is only loosely related.
- Both methods depend on the text cleaning. Unusual formatting, scanned text, or typos can reduce matches.
- The cutoffs are provisional. Test a few comments you already know before relying on a setting.
- **Where the data goes.** The notebook version runs in Google Colab, so comments are processed on Google's servers during that session. The notebook states that comment text stays in the Colab session and in the user's own Google Drive. The local version (`run_search.py`) keeps comment text on the user's computer. Language models are downloaded once from Hugging Face.

## Run it in Colab

Open `notebooks/quickstart.ipynb` from the Colab badge in that file. The first code cells install the package from this repository (branch `agency-feedback-ui`) and connect Google Drive, where the index is saved by default. Then edit the file path and column names in the settings cell.

## Run it locally

Search a saved index from the terminal: see [RUN_THE_SEARCH.md](RUN_THE_SEARCH.md).

```bash
pip install -e ".[test]"
python -m pytest -q        # no model downloads: the tests use a fake encoder
```

To run the notebook on your own computer, set `COMMENTS_PATH` (the comments file) and `COMMENTSEARCH_INDEX_DIR` (where the index is saved). The notebook reads them if they are set, and it uses `COMMENTS_PATH` for local testing. Open `notebooks/quickstart.ipynb` with Jupyter.

## Index folder

`index.save(folder)` writes `comments.parquet`, `passages.parquet`, `passage_emb.npy`, `comment_emb.npy`, `sentences.parquet`, `sentence_emb.npy`, `bm25/`, and `meta.json`. `CommentIndex.load(folder)` reads it back. Keep the folder on Google Drive (or any local disk) between sessions.

## Known limitations

- Exact-word matching uses stemming and English stop words. It does not correct spelling.
- Building the index is slow for large collections, especially without a GPU. Duplicate detection compares comments in blocks, so it also slows as the collection grows.
- Dates are only parsed if the input has them.
- Notebook outputs contain comment text. Do not commit an executed notebook.
- The search window does not create a public sharing link by default (`share=False`).
- Comment data is not part of this repository. `data/`, `comment_indexes/`, and `*_search-index.zip` files are git-ignored.
