"""TASK-31250: saved user themes (~/.config/tldw_cli/themes/*.toml) load as Theme objects."""

from pathlib import Path

from tldw_chatbook.css.Themes.themes import load_user_themes


def _write(dir_: Path, name: str, body: str) -> None:
    (dir_ / f"{name}.toml").write_text(body, encoding="utf-8")


def test_load_user_themes_reads_good_files_and_skips_bad_ones(tmp_path):
    _write(
        tmp_path,
        "ocean",
        '[theme]\nname = "ocean"\ndark = true\n[colors]\nprimary = "#9966FF"\n',
    )
    _write(tmp_path, "broken", "this is not toml = = =\n")
    # Textual's Theme requires a primary colour; a file without one is
    # unusable and must be skipped like a parse error, not crash startup.
    _write(tmp_path, "noprimary", '[theme]\nname = "bare"\n')

    themes = load_user_themes(tmp_path)

    assert [t.name for t in themes] == ["ocean"]
    ocean = next(t for t in themes if t.name == "ocean")
    primary = ocean.primary
    assert (primary.hex if hasattr(primary, "hex") else str(primary)).upper() == "#9966FF"
    assert ocean.dark is True


def test_load_user_themes_missing_dir_returns_empty(tmp_path):
    assert load_user_themes(tmp_path / "nope") == []
