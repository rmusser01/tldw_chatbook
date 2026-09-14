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
from textual.widgets import Button, Static

from tldw_chatbook.css import build_css
from tldw_chatbook.Widgets.emoji_picker import EmojiButton

_CSS_DIR = Path(build_css.__file__).parent
_SCOPED, _SELF = build_css.screen_css_paths(_CSS_DIR)


class ParityHost(App):
    """Real-CSS-stack host: scoped sheet, app bundle, then self sheet."""

    # TASK-25812 + ADR-161 task 10: the lazily-loaded split sheets carry the
    # library/settings rules; the console vocabulary rides the bundle itself
    # (its always-boot-parsed sheet was dissolved). The running app loads
    # the split sheets after the bundle, so the harness does too.
    CSS_PATH = [
        str(_SCOPED),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
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
        # Focus sink: run_test auto-focuses the first focusable widget; a
        # plain Button ahead of the emoji keeps the emoji UNFOCUSED at mount,
        # which test_focused_emoji_button_matches_the_focus_token's
        # before/after measurement depends on. (Previously the dead
        # base_components.NavigationButton sat here.)
        yield Button("focus sink", id="focus-sink")
        yield EmojiButton(
            {"char": "🙂", "name": "smile", "aliases": [], "group": "test"},
            id="emoji",
            classes="emoji_button",
        )


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
