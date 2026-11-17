"""Terminal launcher: pick a saved comment index from a numbered menu, then open its search in the browser.

Each index is a folder (one per dataset) inside the index folder. The index folder is set by --folder,
by the COMMENTSEARCH_INDEX_DIR environment variable, or defaults to Box\\Coding\\comment_indexes.
Nothing here sends comment text anywhere. The search runs on this computer only (127.0.0.1 by default).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

INDEX_DIR_ENV = "COMMENTSEARCH_INDEX_DIR"
DEFAULT_INDEX_DIR = Path.home() / "Box" / "Coding" / "comment_indexes"


def default_index_dir() -> Path:
    env = os.environ.get(INDEX_DIR_ENV)
    return Path(env).expanduser() if env else DEFAULT_INDEX_DIR


def display_name(folder_name: str) -> str:
    """'owrd_graywolf_2022' -> 'Owrd Graywolf 2022'."""
    words = folder_name.replace("-", "_").split("_")
    return " ".join(w.capitalize() for w in words if w)


def find_indexes(folder: Path) -> list[tuple[str, Path]]:
    """(display name, folder) for each subfolder that holds a saved index (a meta.json file), sorted by name."""
    if not folder.is_dir():
        return []
    return [
        (display_name(p.name), p)
        for p in sorted(folder.iterdir())
        if p.is_dir() and not p.name.startswith(".") and (p / "meta.json").is_file()
    ]


def choose(indexes: list[tuple[str, Path]], input_fn=input, print_fn=print) -> tuple[str, Path]:
    """Show the numbered menu and return the chosen index. Enter picks the first one."""
    print_fn(f"Found {len(indexes)} comment index(es):")
    for number, (name, _) in enumerate(indexes, start=1):
        print_fn(f"  {number}. {name}")
    while True:
        answer = input_fn("Pick a number (Enter = 1): ").strip()
        if not answer:
            return indexes[0]
        if answer.isdigit() and 1 <= int(answer) <= len(indexes):
            return indexes[int(answer) - 1]
        print_fn(f"Please type a number from 1 to {len(indexes)}.")


def open_index(name: str, path: Path, host: str, port: int | None, no_browser: bool) -> int:
    from .app import make_app
    from .index import CommentIndex
    from .search import Searcher

    print(f"Loading {name} ...")
    searcher = Searcher(CommentIndex.load(path))
    print(f"{len(searcher.index.comments):,} comments loaded.")
    print("Loading the fuzzy-search model (a few seconds). Keyword search works without it.")
    try:
        searcher.encoder  # load now, so the first fuzzy search is fast
    except Exception as exc:  # no model available: exact search still works
        print(f"Fuzzy search is unavailable: {exc}")
    demo = make_app(searcher, title=f"Comment search: {name}")
    print("Opening the search on this computer. Press Ctrl+C in this window to stop.")
    try:
        # share=False: nothing is published through Gradio's servers.
        demo.launch(server_name=host, server_port=port, share=False, inbrowser=not no_browser)
    except KeyboardInterrupt:
        pass
    print("Stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a saved comment index in your browser.")
    parser.add_argument("--folder", type=Path, default=None,
                        help=f"Folder with one subfolder per index (default: ${INDEX_DIR_ENV}, or {DEFAULT_INDEX_DIR}).")
    parser.add_argument("--index", default=None, help="Folder name of the index to open, without the menu.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: this computer only).")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 7860, or the next free one).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args(argv)

    folder = args.folder or default_index_dir()
    indexes = find_indexes(folder)
    if not indexes:
        print(f"No saved indexes found in: {folder}")
        print("Unzip a search-index zip into its own subfolder there. See RUN_THE_SEARCH.md.")
        return 1

    if args.index:
        matches = [(name, path) for name, path in indexes if path.name == args.index]
        if not matches:
            print(f"No index named '{args.index}' in {folder}.")
            print("Available: " + ", ".join(path.name for _, path in indexes))
            return 1
        name, path = matches[0]
    else:
        try:
            name, path = choose(indexes)
        except (EOFError, KeyboardInterrupt):
            print("\nNo index chosen.")
            return 1
    return open_index(name, path, args.host, args.port, args.no_browser)
