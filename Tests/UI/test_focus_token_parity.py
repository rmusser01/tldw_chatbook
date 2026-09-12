"""Focus/selected states must resolve the BUNDLE's $ds-focus-* tokens.

TASK-16811: a widget-local ``$var:`` "fallback" declared inside
``DEFAULT_CSS``/``BUNDLED_SCREEN_CSS`` shadows the app bundle's design
tokens for that whole CSS source (Textual resolves ``$variables``
per-source), so the affected focus/active states silently rendered
``$surface`` instead of the ``#51677e`` focus colour every other selected
row uses. Caught live on the Console turn file card (PR #1728) and then
audited repo-wide; the token-dependent rules now live in bundle modules.

These tests pin the RESOLVED colour on the real CSS stack -- a class-toggle
assertion cannot catch this failure mode (the class always toggled; the
colour was what silently diverged).
"""
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.css import build_css
from tldw_chatbook.Widgets.base_components import NavigationButton
from tldw_chatbook.Widgets.emoji_picker import EmojiButton

_CSS_DIR = Path(build_css.__file__).parent
_SCOPED, _SELF = build_css.screen_css_paths(_CSS_DIR)


class ParityHost(App):
    """Real-CSS-stack host: scoped sheet, app bundle, then self sheet."""

    # TASK-25812: the split screen sheets carry the console/library/settings
    # rules; the running app loads them after the bundle, so the harness does.
    CSS_PATH = [
        str(_SCOPED),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
        str(_CSS_DIR / "screen_agentic_console.tcss"),
        str(_CSS_DIR / "screen_agentic_library.tcss"),
        str(_CSS_DIR / "screen_agentic_settings.tcss"),
        str(_SELF),
    ]

    def compose(self) -> ComposeResult:
        # The reference for "what a selected/focused row looks like":
        # the transcript's selected-message class, straight from the bundle.
        yield Static(
            "peer",
            classes="console-transcript-message-selected",
            id="selected-peer",
        )
        yield NavigationButton("Nav", id="nav-active")
        yield NavigationButton("Nav plain", id="nav-plain")
        yield EmojiButton(
            {"char": "🙂", "name": "smile", "aliases": [], "group": "test"},
            id="emoji",
            classes="emoji_button",
        )


@pytest.mark.asyncio
async def test_active_navigation_button_matches_the_focus_token():
    """The UNFOCUSED active nav button must resolve the bundle's focus token.

    run_test auto-focuses the first focusable widget, and the bundle's
    generic ``Button:focus`` already paints ``$ds-focus-bg`` at app tier --
    measured focused, this test cannot see the ``.active`` rule at all
    (it passed against the pre-fix shadowed code). The bug only shows on
    the UNFOCUSED active state, so blur before measuring.
    """
    async with ParityHost().run_test(size=(80, 24)) as pilot:
        pilot.app.query_one("#nav-active").add_class("active")
        pilot.app.set_focus(None)
        await pilot.pause()
        assert "focus" not in pilot.app.query_one("#nav-active").pseudo_classes
        peer_bg = pilot.app.query_one("#selected-peer").styles.background
        active_bg = pilot.app.query_one("#nav-active").styles.background
        plain_bg = pilot.app.query_one("#nav-plain").styles.background
        assert active_bg == peer_bg
        assert active_bg != plain_bg


@pytest.mark.asyncio
async def test_focused_emoji_button_matches_the_focus_token():
    """Regression pin, not a divergence catch: Button subclasses were
    already rescued when focused by the bundle's generic ``Button:focus``
    (app tier beats any shadowed DEFAULT_CSS rule). This pins that the
    relocated explicit rule keeps that parity."""
    async with ParityHost().run_test(size=(80, 24)) as pilot:
        emoji = pilot.app.query_one("#emoji")
        before_bg = emoji.styles.background
        emoji.focus()
        await pilot.pause()
        peer_bg = pilot.app.query_one("#selected-peer").styles.background
        focused_bg = emoji.styles.background
        assert focused_bg == peer_bg
        assert focused_bg != before_bg
