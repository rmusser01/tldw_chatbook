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

from types import SimpleNamespace
from typing import Any

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


# --------------------------------------------------------------------------
# TASK-34780: the two notify sites the census caught red on dev
# --------------------------------------------------------------------------
#
# Both show exception text in a toast. Exception text is the least
# controlled string there is, so each pin drives the real code path with a
# message carrying a stray closing tag and reads the toast back off a real
# running app: the stored notification must keep the text verbatim with
# ``markup=False``, and the mounted Toast must render it literally. Before
# the fix the toast parsed the message and ``Content.from_markup`` raised
# ``MarkupError`` (the negative control above pins that parser behaviour).

#: Exception text with a stray closing tag -- unparseable as markup.
HOSTILE_ERROR = "disk said [/b] no"


async def _wait_for_notifications(pilot: Any, app: App, count: int) -> list[Any]:
    """Return the app's notifications once ``count`` have arrived."""
    for _ in range(200):
        notifications = list(app._notifications)
        if len(notifications) >= count:
            return notifications
        await pilot.pause(0.02)
    raise AssertionError(f"expected {count} notification(s), got {list(app._notifications)}")


async def _wait_for_notification_message(pilot: Any, app: App, message: str) -> Any:
    """Return the app's notification whose message is ``message`` once it arrives."""
    for _ in range(200):
        for notification in list(app._notifications):
            if notification.message == message:
                return notification
        await pilot.pause(0.02)
    raise AssertionError(f"no {message!r} notification, got {list(app._notifications)}")


def _rendered_toasts(app: App) -> list[str]:
    """Plain text of every Toast mounted on the active screen."""
    return [toast.render().plain for toast in app.screen.query("Toast")]


async def _assert_toast_literal(pilot: Any, app: App, notification: Any, text: str) -> None:
    """The notification keeps ``text`` verbatim and its Toast renders it."""
    assert notification.message == text
    assert notification.markup is False
    for _ in range(100):
        if any(text in rendered for rendered in _rendered_toasts(app)):
            return
        await pilot.pause(0.02)
    raise AssertionError(f"no Toast rendered {text!r}: {_rendered_toasts(app)}")


@pytest.mark.parametrize("path", ["worker", "legacy_async"])
async def test_smart_content_tree_load_failure_toast_renders_the_error_literally(path):
    """SmartContentTree reports a failed content load in a toast on both its
    mount-time worker path (``_report_load_failure``) and the legacy
    ``load_all_content`` entry point; the exception text renders verbatim."""
    from tldw_chatbook.UI.Widgets.SmartContentTree import SmartContentTree

    def failing_load() -> dict:
        raise RuntimeError(HOSTILE_ERROR)

    tree = SmartContentTree(load_content=failing_load if path == "worker" else None)

    class _TreeApp(App):
        def compose(self) -> ComposeResult:
            yield tree

    app = _TreeApp()
    # notifications=True: run_test leaves the ToastRack out by default, and
    # the Toast's render is where the markup parse (and MarkupError) happens.
    async with app.run_test(notifications=True) as pilot:
        if path == "legacy_async":
            tree.load_content_callback = failing_load
            await tree.load_all_content()
        (notification,) = await _wait_for_notifications(pilot, app, 1)
        await _assert_toast_literal(
            pilot, app, notification, f"Error loading content: {HOSTILE_ERROR}"
        )


def _console_delete_host(app: App) -> SimpleNamespace:
    """The controller attributes ``message_delete._delete`` reaches."""

    async def refresh() -> None:
        return None

    return SimpleNamespace(
        app_instance=app,
        push_screen=app.push_screen,
        _pending_console_delete_message_id="m1",
        _console_delete_scope=None,
        _last_console_action=None,
        _pending_console_swipe_selection=None,
        _sync_native_console_chat_ui=refresh,
        _invalidate_console_fork_image_selections=lambda _ids: None,
        _invalidate_console_persisted_rows_cache=lambda: None,
    )


_DELETE_SCOPE = SimpleNamespace(message_id="m1", subtree_ids=("m1", "m2"), removed_count=2)


async def test_console_delete_refusal_toast_renders_the_error_literally(monkeypatch):
    """The Console Delete flow's refusal (``message_delete._delete.delete``):
    a storage ``ValueError`` -- a pending dispatch owns the branch -- is shown
    verbatim. Drives the real receipt modal; only the durable write is
    replaced."""
    from tldw_chatbook.UI.Console_Modules import message_delete

    async def refused_delete(_store: Any, _message_id: str) -> tuple[Any, tuple]:
        raise ValueError(HOSTILE_ERROR)

    monkeypatch.setattr(message_delete, "delete_subtree_off_loop", refused_delete)

    app = _ToastApp()
    async with app.run_test(size=(120, 40), notifications=True) as pilot:
        await message_delete._delete(_console_delete_host(app), object(), _DELETE_SCOPE)
        (refused,) = await _wait_for_notifications(pilot, app, 1)
        await _assert_toast_literal(pilot, app, refused, HOSTILE_ERROR)


async def test_console_delete_undo_toasts_render_runtime_text_literally(monkeypatch):
    """The Console Delete flow's Undo (``message_delete._delete.undo``):
    a refused Undo shows the refusal's text verbatim, and the success toast
    is literal too. Drives the real receipt modal; only the durable writes
    are replaced."""
    from tldw_chatbook.Chat.console_message_delete import ConsoleDeleteUndoError
    from tldw_chatbook.UI.Console_Modules import message_delete

    deleted = SimpleNamespace(count=2, root_id="m1", session_id="s1")
    restore_outcomes: list[Exception | None] = [
        ConsoleDeleteUndoError(HOSTILE_ERROR, retryable=True),
        None,
    ]

    async def fake_delete(_store: Any, _message_id: str) -> tuple[Any, tuple]:
        return deleted, ()

    async def fake_restore(_store: Any, _deleted: Any) -> None:
        outcome = restore_outcomes.pop(0)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(message_delete, "delete_subtree_off_loop", fake_delete)
    monkeypatch.setattr(message_delete, "restore_subtree_off_loop", fake_restore)

    app = _ToastApp()
    async with app.run_test(size=(120, 40), notifications=True) as pilot:
        await message_delete._delete(_console_delete_host(app), object(), _DELETE_SCOPE)

        async def press_undo() -> None:
            for _ in range(200):
                if app.screen.query("#console-delete-receipt-undo"):
                    break
                await pilot.pause(0.02)
            app.screen.query_one("#console-delete-receipt-undo", Button).press()

        # Refused: the refusal's own text, verbatim, and Undo stays on offer.
        await press_undo()
        refused, *_ = await _wait_for_notifications(pilot, app, 1)
        await _assert_toast_literal(pilot, app, refused, HOSTILE_ERROR)

        # Restored: the success toast is literal as well. Found by its message, not its
        # position: Textual expires a toast after 5 s, so on a slow runner the refusal
        # can be gone before the success toast arrives.
        await press_undo()
        restored = await _wait_for_notification_message(pilot, app, "Restored 2 messages.")
        assert restored.markup is False
        assert not restore_outcomes
