"""TASK-31250: saved user themes (~/.config/tldw_cli/themes/*.toml) load as Theme objects."""

from pathlib import Path

import pytest
import toml

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


_MALFORMED_VARIABLES_THEME = """[theme]
name = "hostile"
dark = true
[colors]
primary = "#9966FF"
[variables]
text-muted = "red; } Screen { display: none"
footer-key-foreground = 12
"Bad_Name" = "#FFFFFF"
input-selection-background = "#81A1C1 35%"
scrollbar-color = "auto 50%"
block-cursor-text-style = "bold underline"
footer-background = "#101010"
"""


def test_load_user_themes_drops_malformed_variables(tmp_path):
    """Review finding #1: [variables] reach Textual's CSS tokenizer verbatim,
    so a non-colour value must be dropped at load, keeping the good ones."""
    _write(tmp_path, "hostile", _MALFORMED_VARIABLES_THEME)

    [theme] = load_user_themes(tmp_path)

    assert "footer-key-foreground" not in theme.variables
    assert "Bad_Name" not in theme.variables
    assert theme.variables.get("text-muted") != "red; } Screen { display: none"
    assert theme.variables["input-selection-background"] == "#81A1C1 35%"
    assert theme.variables["scrollbar-color"] == "auto 50%"
    assert theme.variables["block-cursor-text-style"] == "bold underline"
    assert theme.variables["footer-background"] == "#101010"


async def test_malformed_user_theme_survives_css_refresh(tmp_path):
    """Selecting the loaded theme must not kill the app on the next refresh
    (the tokenizer error fires after ``app.theme = ...`` returns)."""
    from textual.app import App

    _write(tmp_path, "hostile", _MALFORMED_VARIABLES_THEME)
    [theme] = load_user_themes(tmp_path)

    app = App()
    async with app.run_test() as pilot:
        app.register_theme(theme)
        app.theme = "hostile"
        await pilot.pause()
        app.refresh_css()
        await pilot.pause()
        assert app.theme == "hostile"
    assert app.return_code in (None, 0)


def test_theme_from_file_data_refuses_non_colour_keys():
    """R32: `variables`/`dark` are Theme kwargs, not colours; accepting them
    built a theme whose generate() raised AttributeError."""
    import pytest

    from tldw_chatbook.css.Themes.themes import theme_from_file_data

    for key, value in (("variables", "#ffffff"), ("dark", "#000000"), ("bogus", "#fff")):
        data = {"colors": {"primary": "#112233", key: value}}
        with pytest.raises(ValueError, match=f"^{key} is not a theme colour$"):
            theme_from_file_data(data, "x", "x.toml")


def test_load_user_themes_skips_non_colour_keys(tmp_path):
    _write(tmp_path, "good", '[colors]\nprimary = "#9966FF"\n')
    _write(tmp_path, "vars", '[colors]\nprimary = "#9966FF"\nvariables = "#ffffff"\n')
    _write(tmp_path, "dark", '[colors]\nprimary = "#9966FF"\ndark = "#000000"\n')

    themes = load_user_themes(tmp_path)

    assert [t.name for t in themes] == ["good"]
    for theme in themes:
        theme.to_color_system().generate()  # would raise for a bad key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [('"false"', False), ('"OFF"', False), ('"0"', False), ('"no"', False),
     ('"true"', True), ('"Yes"', True), ("false", False), ("true", True), ('"maybe"', True)],
)
def test_theme_file_dark_flag_coerces_strings(raw, expected):
    """R31: ``dark = "false"`` is a light theme, not ``bool("false")``."""
    from tldw_chatbook.css.Themes.themes import theme_from_file_data

    data = toml.loads(f'[theme]\nname = "t"\ndark = {raw}\n[colors]\nprimary = "#FFAA00"\n')
    assert theme_from_file_data(data, "t", "t.toml").dark is expected


def test_theme_name_with_control_characters_is_refused_and_skipped(tmp_path):
    """R39: a name like ``x<ESC>c`` would emit a terminal reset when shown."""
    from tldw_chatbook.css.Themes.themes import printable, theme_from_file_data

    body = '[theme]\nname = "x\\u001bc"\n[colors]\nprimary = "#FFAA00"\n'
    with pytest.raises(ValueError, match="name has control characters"):
        theme_from_file_data(toml.loads(body), "t", "t.toml")
    _write(tmp_path, "t", body)
    assert load_user_themes(tmp_path) == []
    assert printable("a\x1b]52;c;eA==\x07b\u00e9 ") == "a?]52;c;eA==?b\u00e9 "


_OSC_SECONDARY = (
    '[theme]\nname = "osc"\n[colors]\nprimary = "#112233"\n'
    'secondary = "\\u001b]52;c;eA==\\u001b\\\\"\n'
)


def test_non_hex_colour_value_is_refused_and_skipped(tmp_path):
    """R41: a non-primary colour Textual can't parse used to be silently
    dropped, so an ESC-laden value was accepted and later echoed into Edit."""
    from loguru import logger

    from tldw_chatbook.css.Themes.themes import theme_from_file_data

    for value in ("\x1b]52;c;eA==\x1b\\", "red", "#12345", "#1122334", "#fff\n", "rgb(1,2,3)", 7):
        data = {"colors": {"primary": "#112233", "secondary": value}}
        with pytest.raises(ValueError, match=r"^invalid colour 'secondary'$"):
            theme_from_file_data(data, "x", "x.toml")
    # Follow-up: #RRGGBBAA is what the editor writes for a translucent
    # shipped colour (deep_dive_cyberspace.error = #FF33AACC).
    for value in ("#abc", "#ABCDEF", "#a1B2c3", "#FF33AACC"):
        theme_from_file_data({"colors": {"primary": value}}, "x", "x.toml")

    _write(tmp_path, "osc", _OSC_SECONDARY)
    records = []
    sink = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        assert load_user_themes(tmp_path) == []
    finally:
        logger.remove(sink)
    skipped = [r for r in records if "osc.toml" in r["message"]]
    assert skipped and skipped[0]["level"].name == "WARNING"
    assert "invalid colour 'secondary'" in skipped[0]["message"]


def test_create_theme_from_dict_does_not_print_unparseable_colours(capsys):
    from tldw_chatbook.css.Themes.themes import create_theme_from_dict

    create_theme_from_dict("x", {"primary": "#112233", "secondary": "not-a-colour"})
    assert capsys.readouterr().out == ""


def test_variable_warning_file_label_is_printable():
    """R41 item 4: the dropped-variable warning names the file; a file name
    with control characters must not carry them into the log."""
    from loguru import logger

    from tldw_chatbook.css.Themes.themes import sanitize_theme_variables

    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    try:
        assert sanitize_theme_variables({"bad": "red; }"}, "x\x1b]52;c;eA==\x07.toml") == {}
    finally:
        logger.remove(sink)
    assert messages and all(m.isprintable() for m in messages), [repr(m) for m in messages]


def test_rrggbbaa_colour_file_loads(tmp_path):
    _write(tmp_path, "alpha", '[colors]\nprimary = "#112233"\nerror = "#FF33AACC"\n')
    [theme] = load_user_themes(tmp_path)
    assert theme.name == "alpha"
    theme.to_color_system().generate()


def test_variables_with_trailing_newline_are_dropped():
    """Follow-up item 4: ``$`` matches before a trailing newline."""
    from tldw_chatbook.css.Themes.themes import sanitize_theme_variables

    assert sanitize_theme_variables(
        {"foo\n": "red", "bar": "#fff\n", "baz": "auto 50%\n", "ok": "#fff"}, "t.toml"
    ) == {"ok": "#fff"}
