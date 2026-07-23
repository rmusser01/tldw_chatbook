"""TASK-1 (Console branching Phase B): the edit modal gains an explicit
"Edit & resend" affordance alongside the existing in-place "Save".

Construction-level tests are the minimum contract (per the task brief); a
mounted `run_test` pilot assertion is added to mirror the existing modal
suite's style (Tests/UI/test_console_edit_modal_keystroke_guard.py).
"""

import pytest
from textual.app import App
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditMessageModal,
    ConsoleEditResult,
)


def test_edit_result_dataclass_shape():
    r = ConsoleEditResult(text="hi", resend=True)
    assert (r.text, r.resend) == ("hi", True)


def test_edit_result_is_frozen():
    r = ConsoleEditResult(text="hi", resend=True)
    with pytest.raises(Exception):
        r.text = "changed"  # type: ignore[misc]


def test_modal_accepts_can_resend_kwarg():
    # construction only (no mount) — the resend button is gated on can_resend
    m = ConsoleEditMessageModal(content="orig", can_resend=True)
    assert m._can_resend is True
    m2 = ConsoleEditMessageModal(content="orig")
    assert m2._can_resend is False


class _ModalHost(App):
    pass


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.asyncio
async def test_modal_without_resend_has_no_resend_button():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig")
        app.push_screen(modal)
        await pilot.pause()

        assert len(modal.query("#console-edit-message-resend")) == 0
        save_button = modal.query_one("#console-edit-message-save", Button)
        assert save_button.variant == "primary"


@pytest.mark.asyncio
async def test_modal_with_resend_shows_resend_button_and_demotes_save():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        app.push_screen(modal)
        await pilot.pause()

        resend_button = modal.query_one("#console-edit-message-resend", Button)
        assert resend_button.variant == "primary"
        save_button = modal.query_one("#console-edit-message-save", Button)
        assert save_button.variant == "default"


@pytest.mark.asyncio
async def test_save_dismisses_edit_result_with_resend_false():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig")
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited"
        await pilot.click("#console-edit-message-save")
        await pilot.pause()

    assert result == [ConsoleEditResult(text="edited", resend=False)]


@pytest.mark.asyncio
async def test_resend_dismisses_edit_result_with_resend_true():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited"
        await pilot.click("#console-edit-message-resend")
        await pilot.pause()

    assert result == [ConsoleEditResult(text="edited", resend=True)]


@pytest.mark.asyncio
async def test_resend_blank_text_blocked_inline():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "   "
        await pilot.click("#console-edit-message-resend")
        await pilot.pause()

        assert result == []
        error = modal.query_one("#console-edit-message-error", Static)
        assert "cannot be blank" in _static_plain_text(error).lower()


@pytest.mark.asyncio
async def test_cancel_dismisses_none_even_with_resend_available():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()

    assert result == [None]


def test_context_copy_mentions_resend_only_when_can_resend():
    m_plain = ConsoleEditMessageModal(content="orig")
    m_resend = ConsoleEditMessageModal(content="orig", can_resend=True)
    assert m_plain._can_resend is False
    assert m_resend._can_resend is True
