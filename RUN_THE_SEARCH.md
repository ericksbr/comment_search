# Run the comment search

This guide covers the comment search on your own computer: where the saved index zips go, the one-time setup, and how to start the search. Comment data stays on this computer. Nothing is published, and the search only answers on this computer.

## 1. The saved index zip

Each dataset is built in the notebook (`notebooks/quickstart.ipynb`). Step 3b offers a download named `<file>_search-index.zip`. Keep these rules:

- **Store the zip in the index folder** (or anywhere you like) and unzip it there. The index folder is:
  `C:\Users\ericksbr\Box\Coding\comment_indexes`
- **Unzip into its own subfolder**, one per dataset. The subfolder name becomes the name in the menu. Example:
  `C:\Users\ericksbr\Box\Coding\comment_indexes\owrd_graywolf_2022\`
- **Check the subfolder.** After unzipping, `meta.json` and `comments.parquet` should sit directly inside it. If they sit one folder deeper, move them up one level.
- **Never put comment data or zips in the repo.** The repo is public. `.gitignore` blocks `comment_indexes/` and `*_search-index.zip`, but keep the files outside the repo anyway.

PowerShell example (replace the zip path and folder name):

```powershell
Expand-Archive -Path "C:\path\to\comments_clean_search-index.zip" -DestinationPath "C:\Users\ericksbr\Box\Coding\comment_indexes\owrd_graywolf_2022" -Force
```

An index built before 2026-11-17 is missing the stemming and sentence data. Rebuild it in the notebook first.

## 2. One-time setup

From the repo folder, install the required packages once:

```powershell
cd C:\Users\ericksbr\Documents\GitHub\comment_search
pip install -e .
```

Run this again only after the packages change in `pyproject.toml`.

## 3. Start the search

From the repo folder:

```powershell
python run_search.py
```

The menu lists every index in the index folder:

```
Found 2 comment index(es):
  1. Owrd Graywolf 2022
  2. Noaa Narw 2026
Pick a number (Enter = 1):
```

Type a number and press Enter. The search opens in your browser. Keep the terminal window open while you search. Press **Ctrl+C** in that window to stop.

The first start takes a few extra seconds while the fuzzy-search model loads.

## Using the window

- **Search:** type in the box at the top. Results update as you type. Choose **Exact words** or **Fuzzy (by meaning)** above the box.
- **Every hit or one per group:** the list shows every hit by default. Tick **One per duplicate group** to show only the best-matching comment from each group of copies.
- **The list (left):** one row per comment. Name, comment ID, page and date come first. Use **Columns and export** to show or hide other columns.
- **Picking passages:** tick **Pick** in a row, or use **Select all** / **Select none**. With one or more ticked, the right side shows all of them together, one after another. Clear the selection to go back to browsing one hit at a time with the arrow keys or **Previous** / **Next**.
- **Hide table:** collapses the list to a narrow panel of the selected commenters' names, so the passages have more room. **Show table** brings the list back.
- **Export:** open **Columns and export** and press **Export to Excel**. The file holds the ticked passages, or every current hit if none are ticked. Comment text longer than one Excel cell is split across columns named "Full text (part 2)" and so on, so nothing is cut off.
- **Show (right side):** **Passage** shows the matching part of each comment, and **Full comment** shows the whole comment.
 The model is downloaded from Hugging Face the first time only, so the first start needs internet access.

## Options

| What you want | Command |
|---|---|
| Skip the menu and open one index | `python run_search.py --index owrd_graywolf_2022` |
| Use a different index folder | `python run_search.py --folder "D:\some\other\folder"` |
| Don't open the browser automatically | `python run_search.py --no-browser` |
| Use a specific port | `python run_search.py --port 7865` |

To change the default index folder for good, set an environment variable once:

```powershell
setx COMMENTSEARCH_INDEX_DIR "C:\Users\ericksbr\Box\Coding\comment_indexes"
```

Open a new terminal after `setx`.

## Troubleshooting

- **"No saved indexes found"**: the index folder is empty, or the subfolders don't contain `meta.json`. See step 1.
- **"No index named ..."** with `--index`: use one of the names printed in the message.
- **Fuzzy search is unavailable**: the model could not load, usually because there's no internet on the first run. Keyword search still works.
- **Browser did not open**: copy the address printed in the terminal (`http://127.0.0.1:7860/`) into your browser.
- **Port already in use**: stop the other program, or add `--port` with a different number.
