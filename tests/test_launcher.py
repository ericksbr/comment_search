from commentsearch.launcher import default_index_dir, display_name, find_indexes, main


def _make_index(folder):
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text("{}", encoding="utf-8")


def test_display_name_from_folder_name():
    assert display_name("owrd_graywolf_2022") == "Owrd Graywolf 2022"
    assert display_name("noaa-narw_2026") == "Noaa Narw 2026"


def test_find_indexes_lists_only_folders_with_meta_json(tmp_path):
    _make_index(tmp_path / "wdfw_wolf")
    _make_index(tmp_path / "owrd_graywolf_2022")
    (tmp_path / "notes").mkdir()                      # no index in it
    _make_index(tmp_path / ".hidden")                  # hidden folders are skipped
    (tmp_path / "loose.zip").write_text("x", encoding="utf-8")
    found = find_indexes(tmp_path)
    assert [name for name, _ in found] == ["Owrd Graywolf 2022", "Wdfw Wolf"]


def test_find_indexes_on_missing_folder_is_empty(tmp_path):
    assert find_indexes(tmp_path / "does_not_exist") == []


def test_default_folder_follows_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMENTSEARCH_INDEX_DIR", str(tmp_path))
    assert default_index_dir() == tmp_path


def test_main_reports_empty_folder_without_loading_models(tmp_path, capsys):
    assert main(["--folder", str(tmp_path)]) == 1
    assert "No saved indexes found" in capsys.readouterr().out


def test_main_unknown_index_name_lists_available(tmp_path, capsys):
    _make_index(tmp_path / "owrd_graywolf_2022")
    assert main(["--folder", str(tmp_path), "--index", "nope"]) == 1
    out = capsys.readouterr().out
    assert "owrd_graywolf_2022" in out


def test_choose_handles_enter_numbers_and_retries(monkeypatch):
    from commentsearch.launcher import choose

    indexes = [("A", "pa"), ("B", "pb")]
    printed = []
    answers = iter(["", "2", "9", "x", "1"])
    assert choose(indexes, input_fn=lambda _: next(answers), print_fn=printed.append) == ("A", "pa")
    answers = iter(["2"])
    assert choose(indexes, input_fn=lambda _: next(answers), print_fn=printed.append) == ("B", "pb")
    answers = iter(["9", "1"])
    assert choose(indexes, input_fn=lambda _: next(answers), print_fn=printed.append) == ("A", "pa")
    assert any("from 1 to 2" in line for line in printed)
