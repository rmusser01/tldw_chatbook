"""TASK-1513 AC3: one regression pin per markup-parsing widget kind.

The three surfaces the task names -- toasts (``App.notify``), tooltips, and
``Button`` labels -- all parse Rich markup by default on installed Textual
(``_toast.py:115`` ``Content.from_markup(notification.message)``,
``_button.py`` ``Content.from_text(label)`` which parses markup, and the
``Tooltip`` Static rendering the tooltip string). An interpolated
user-derived name carrying a bare ``[/]`` raises ``MarkupError``.

Each pin below exercises the REAL parser on the REAL convention shape, plus
a negative control proving the parser really rejects the unescaped shape
(mirroring ``Tests/UI/test_evals_empty_states.py``'s precedent for the
Evals-package fixes; these are the non-Evals representatives fixed with the
task: ChatbookCreationWindow's toast, skills_screen's tooltip, and
library_skills_canvas' import-review Button label).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Button, Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (Tests/UI RecoveryRequired at setup)
from tldw_chatbook.Utils.input_validation import escape_markup

pytestmark = pytest.mark.unit

#: A user-derived name with markup metacharacters -- the filed hazard shape.
HOSTILE_NAME = "loaded-nouns[/]v1"


# --------------------------------------------------------------------------
# toast (App.notify)
# --------------------------------------------------------------------------


def test_toast_parser_rejects_the_unescaped_shape():
    """Negative control: Textual's toast render path (Content.from_markup)
    really raises on a bare `[/]` -- the crash the convention prevents."""
    from textual.markup import MarkupError

    with pytest.raises(MarkupError):
        Content.from_markup(f"Creating chatbook '{HOSTILE_NAME}'...")


def test_toast_markup_false_branch_renders_literally():
    """`markup=False` routes the toast through `Content(message)` (no parse),
    so the metacharacter name renders literally."""
    message = f"Creating chatbook '{HOSTILE_NAME}'..."
    assert Content(message).plain == message


def test_toast_escaped_message_round_trips_through_the_real_parser():
    """If a notify site keeps markup=True, escape_markup is the alternative:
    the parser accepts the escaped message and the rendered plain text still
    contains the raw, unmangled brackets."""
    escaped = f"Creating chatbook '{escape_markup(HOSTILE_NAME)}'..."
    content = Content.from_markup(escaped)  # must not raise
    assert content.plain == f"Creating chatbook '{HOSTILE_NAME}'..."


class _ToastApp(App):
    def compose(self) -> ComposeResult:
        yield Static("hello")


async def test_real_app_notify_markup_false_keeps_the_name_literal():
    """Live toast through a real running app: the stored notification keeps
    ``markup=False`` and the message text verbatim (the ChatbookCreationWindow
    convention shape)."""
    app = _ToastApp()
    async with app.run_test() as pilot:
        app.notify(f"Creating chatbook '{HOSTILE_NAME}'...", markup=False)
        await pilot.pause()
        notifications = list(app._notifications)
        assert len(notifications) == 1
        notification = notifications[0]
        assert notification.message == f"Creating chatbook '{HOSTILE_NAME}'..."
        assert notification.markup is False


# --------------------------------------------------------------------------
# tooltip
# --------------------------------------------------------------------------


def test_tooltip_parser_rejects_the_unescaped_shape():
    from textual.markup import MarkupError

    with pytest.raises(MarkupError):
        Content.from_markup(f"Use {HOSTILE_NAME} as the Console skill target.")


def test_tooltip_escaped_name_round_trips_through_the_real_parser():
    """The skills_screen `Use`-button convention shape: the escaped tooltip
    parses without raising and renders the raw brackets back."""
    tooltip = f"Use {escape_markup(HOSTILE_NAME)} as the Console skill target."
    content = Content.from_markup(tooltip)  # must not raise
    assert content.plain == f"Use {HOSTILE_NAME} as the Console skill target."


async def test_real_widget_tooltip_keeps_the_escaped_name():
    """A real mounted widget's tooltip attribute carries the escaped string
    (the Tooltip Static parses it at display time, off this code path)."""
    tooltip = f"Use {escape_markup(HOSTILE_NAME)} as the Console skill target."
    widget = Static("Use")
    widget.tooltip = tooltip

    class _TooltipApp(App):
        def compose(self) -> ComposeResult:
            yield widget

    app = _TooltipApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert widget.tooltip == tooltip
        assert "[/]" in Content.from_markup(widget.tooltip).plain


# --------------------------------------------------------------------------
# Button label
# --------------------------------------------------------------------------


def test_button_label_rejects_the_unescaped_shape():
    """Negative control: `Button(label=...)` parses markup at construction
    (Content.from_text) and raises MarkupError on a bare `[/]` -- the crash
    the library_skills_canvas import-review row would hit on compose."""
    from textual.markup import MarkupError

    with pytest.raises(MarkupError):
        Button(f'Review "{HOSTILE_NAME}"…')


async def test_real_button_label_with_escaped_name_renders_literally():
    """The library_skills_canvas convention shape, through a real mounted
    Button: no MarkupError, and the rendered label keeps the raw brackets."""
    label = f'Review "{escape_markup(HOSTILE_NAME)}"…'
    button = Button(label)

    class _ButtonApp(App):
        def compose(self) -> ComposeResult:
            yield button

    app = _ButtonApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        rendered = button.label
        assert isinstance(rendered, Content)
        assert rendered.plain == f'Review "{HOSTILE_NAME}"…'
