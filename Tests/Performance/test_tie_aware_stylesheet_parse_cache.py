"""TieAwareStylesheet keeps one parse per theme (TASK-33120).

Textual's `refresh_css` (every theme switch) clears the parse cache and
re-parses the whole ~830 KB bundle into a fresh stylesheet (~430 ms,
TASK-33075). The app's stylesheet keeps a small LRU of per-variables parse
caches so returning to a theme skips that parse.

The root conftest installs `Tests/UI/css_cache.py`, a process-global parse
cache in front of Textual's; it would answer a revisit on its own and hide
whether THIS cache works, so every test here steps around it.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import textual.css.stylesheet as stylesheet_module
from textual.app import App, ComposeResult
from textual.css.stylesheet import Stylesheet
from textual.widgets import Button, Input, Static

from tldw_chatbook.css import tie_aware_stylesheet
from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet

_CSS = "Static { color: $accent; background: $surface; }"
_LOCATION = ("synthetic.tcss", "")
_BUNDLE = (
    Path(tie_aware_stylesheet.__file__).resolve().parent / "tldw_cli_modular.tcss"
)


def _vars(accent: str) -> dict[str, str]:
    return {"accent": accent, "surface": "#101010"}


@pytest.fixture
def parse_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count real Textual parses, with the test-suite global cache bypassed."""
    original = getattr(Stylesheet._parse_rules, "__wrapped__", None)
    if original is not None:
        monkeypatch.setattr(Stylesheet, "_parse_rules", original)
    calls: list[str] = []
    real_parse = stylesheet_module.parse

    def spy(scope, css, *args, **kwargs):
        calls.append(css)
        return real_parse(scope, css, *args, **kwargs)

    monkeypatch.setattr(stylesheet_module, "parse", spy)
    return calls


def _switch(sheet: Stylesheet, variables: dict[str, str]) -> None:
    """What `App.refresh_css` does to the stylesheet on a theme switch."""
    sheet.set_variables(variables)
    sheet.reparse()


def _accent(sheet: Stylesheet) -> str:
    (rule,) = sheet.rules
    return rule.styles.color.hex


@pytest.mark.unit
def test_returning_to_a_theme_does_not_reparse(parse_calls: list[str]) -> None:
    sheet = TieAwareStylesheet(variables=_vars("#ff0000"))
    sheet.add_source(_CSS, read_from=_LOCATION)
    _switch(sheet, _vars("#ff0000"))
    _switch(sheet, _vars("#00ff00"))
    assert len(parse_calls) == 2
    _switch(sheet, _vars("#ff0000"))
    assert len(parse_calls) == 2, "a revisited theme must not re-parse"
    assert _accent(sheet) == "#FF0000"


@pytest.mark.unit
def test_a_changed_variable_misses_the_cache(parse_calls: list[str]) -> None:
    sheet = TieAwareStylesheet(variables=_vars("#ff0000"))
    sheet.add_source(_CSS, read_from=_LOCATION)
    _switch(sheet, _vars("#ff0000"))
    # Same theme name in the app, edited colour: a different variables dict.
    _switch(sheet, _vars("#ff0001"))
    assert len(parse_calls) == 2
    assert _accent(sheet) == "#FF0001"


@pytest.mark.unit
def test_a_changed_css_source_misses_the_cache(parse_calls: list[str]) -> None:
    sheet = TieAwareStylesheet(variables=_vars("#ff0000"))
    sheet.add_source(_CSS, read_from=_LOCATION)
    _switch(sheet, _vars("#ff0000"))
    _switch(sheet, _vars("#00ff00"))
    # The same location re-read with new content (a rebuilt bundle, an edit).
    sheet.add_source(_CSS.replace("$accent", "$surface"), read_from=_LOCATION)
    _switch(sheet, _vars("#ff0000"))
    assert len(parse_calls) == 3, "new CSS content must be parsed, never served stale"
    assert _accent(sheet) == "#101010"


@pytest.mark.unit
def test_theme_caches_are_bounded_lru(
    parse_calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tie_aware_stylesheet, "THEME_PARSE_CACHE_SIZE", 2)
    sheet = TieAwareStylesheet(variables=_vars("#000001"))
    sheet.add_source(_CSS, read_from=_LOCATION)
    for accent in ("#000001", "#000002", "#000003"):
        _switch(sheet, _vars(accent))
    assert len(sheet._theme_parse_caches) == 2
    assert len(parse_calls) == 3
    _switch(sheet, _vars("#000003"))  # still cached
    assert len(parse_calls) == 3
    _switch(sheet, _vars("#000001"))  # evicted: least recently used
    assert len(parse_calls) == 4


@pytest.mark.unit
def test_reparse_mirrors_upstream() -> None:
    """`TieAwareStylesheet.reparse` copies textual 8.2.8's body plus one line.

    If this fails, Textual changed `Stylesheet.reparse`: re-mirror the new
    body into the subclass (keeping the shared-cache line) and update the hash.
    """
    digest = hashlib.sha256(inspect.getsource(Stylesheet.reparse).encode()).hexdigest()
    assert digest == "dd193eb05f8baeb5f73d962b99719f6b55391333c06064c1a10cf0de4d8e30a4"


class _BundleApp(App[None]):
    CSS_PATH = [_BUNDLE]

    def __init__(self, cached: bool) -> None:
        super().__init__()
        if cached:
            self.stylesheet = TieAwareStylesheet(variables=self.get_css_variables())

    def compose(self) -> ComposeResult:
        yield Static("text", id="static")
        yield Button("go", id="button", variant="primary")
        yield Input(placeholder="type", id="input")


def _resolved(app: App) -> dict[str, tuple]:
    return {
        f"#{widget_id}": (
            str(widget.styles.color),
            str(widget.styles.background),
            str(widget.styles.border),
            str(widget.styles.height),
        )
        for widget_id in ("static", "button", "input")
        for widget in [app.query_one(f"#{widget_id}")]
    }


async def _styles_after_round_trip(cached: bool, calls: list[str]) -> tuple[dict, int]:
    app = _BundleApp(cached)
    async with app.run_test(size=(100, 30)) as pilot:
        app.theme = "nord"
        await pilot.pause()
        app.theme = "gruvbox"
        await pilot.pause()
        parses_before_return = len(calls)
        app.theme = "nord"
        await pilot.pause()
        return _resolved(app), len(calls) - parses_before_return


@pytest.mark.asyncio
async def test_styles_after_a_cached_return_match_a_fresh_parse(
    parse_calls: list[str],
) -> None:
    fresh_styles, fresh_parses = await _styles_after_round_trip(False, parse_calls)
    cached_styles, cached_parses = await _styles_after_round_trip(True, parse_calls)

    assert fresh_parses > 0, "plain Textual re-parses on return (control)"
    assert cached_parses == 0, "the app's stylesheet must serve the revisit from cache"
    assert cached_styles == fresh_styles
