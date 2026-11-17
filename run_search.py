"""Start the comment search. Pick a saved index from the menu; the search opens in your browser.

    python run_search.py

See RUN_THE_SEARCH.md for setup and options.
"""

import sys
from pathlib import Path

# Use the code in this folder (src/), even if an older copy of commentsearch is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from commentsearch.launcher import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
