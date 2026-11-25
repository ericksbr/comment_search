# How the comment search works

*Draft for review. Not yet part of the README.*

The comment search lets you look through a set of public comments in two ways:

- **Exact words** finds comments that contain the words you type.
- **Fuzzy (by meaning)** finds comments that say something similar, even when they use different words.

Everything runs on your own computer. Comment text is not sent to any outside service.

## Using the notebook

`notebooks/comment_search.ipynb` runs top to bottom in Colab:

1. **Act 1 (build):** loads your file, lets you choose the text, ID, and commenter-name columns, and builds the search index. This is the slow step, and it uses the GPU for the meaning part.
2. **Save:** the index is saved to a folder, and a zip is offered for download.
3. **Act 2 (search):** opens the search window. You can also run `python run_search.py` on your own computer to pick a saved index from a menu.

## What the index contains

Before you can search, the notebook prepares the comments in three ways.

**Passages.** Each comment is cut into passages of about 250 words, on sentence boundaries. Sentences are found with NLTK's Punkt splitter, the same one the Preparer and Grouper notebooks use. Passages overlap slightly, so a sentence at a boundary is not lost.

**Keyword table.** For exact-word search, each passage is reduced to its root words. *Tribe* and *tribes* share a root, so both are stored as the same word. *Tribal* has a different root, so it is stored separately. Common words such as "the" and "and" are dropped. Each word is also given a score for each passage, using the BM25 formula. The score is higher when the word is rare across all comments and appears often in a short passage. The scores are calculated once, when the index is built.

**Meaning numbers.** For fuzzy search, each sentence and each passage is turned into a list of 384 numbers that describe its meaning. The model is `BAAI/bge-small-en-v1.5`. Passages about similar things get similar lists of numbers, even when they share no words.

## Exact words

Example: you search for **tribe tribal**.

- With **All words**, a comment must contain both *tribe* (or *tribes*) and *tribal*. The status line shows how many.
- With **Any word**, a comment needs only one of them. The status line shows the total, and comments that contain both rise to the top, because BM25 gives more credit for more matching words.
- Quotes require the words in order. `"public hearing"` will not match a comment that says "hearing, public".

Hits are sorted by BM25 score (**Best match**) unless you choose another column in **Sort by**.

## Fuzzy (by meaning)

Fuzzy search compares your query with the meaning numbers of every sentence or passage. It gives each one a similarity score from 0 to 1. A score of 1 means the same meaning, and a score near 0 means unrelated. Two levels are available:

- **Sentence match:** a comment is listed if at least one of its sentences scores at or above the cutoff. Every sentence at or above the cutoff is shaded blue.
- **Passage match:** a comment is listed if its best passage scores at or above the cutoff. That passage is shaded blue.

Example: the query **economic harm** can match "The rule will cost ranchers their land and income" with no shared words. Sentence scores run lower than passage scores, so each level has its own cutoff. The defaults are 0.5 for sentences and 0.6 for passages. They are starting guesses, and you can move the slider to make matches stricter or looser.

If nothing reaches the cutoff, the window says **No close matches**.

## Duplicate groups

Many comments are copies of each other, either word for word or nearly so. The index marks these as duplicate groups. By default, the list shows every hit. With **One per duplicate group** ticked, it shows only the best-matching comment from each group, and the others are hidden. The count in the status line depends on this setting.

## What to keep in mind

- Exact search matches words, not ideas. A comment that says "tribal nations" will not match a search for "tribe" unless the forms share a root.
- Fuzzy search can miss a comment that is on topic but written very differently from your query.
- Both kinds of search depend on how the text was cleaned. Comments with unusual formatting or typos may match less well.
- The cutoffs are provisional. Check a few known comments before relying on a particular setting.
