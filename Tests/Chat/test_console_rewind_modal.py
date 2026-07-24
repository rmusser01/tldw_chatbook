"""Console `/rewind` menu modal (Task 1 of the SP2 `/rewind` program).

Mirrors the mounted-test style of ``Tests/Chat/test_console_edit_message_modal.py``
and ``Tests/UI/test_console_edit_modal_keystroke_guard.py``: construction-level
shape assertions plus a handful of ``run_test`` pilot flows covering the
two-level select-a-prompt -> choose-an-action flow.
"""

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_rewind_modal import (
    ConsoleRewindChoice,
    ConsoleRewindModal,
    RewindPromptRow,
)


def _row(message_id: str, index_label: str, preview: str) -> RewindPromptRow:
    return RewindPromptRow(
        message_id=message_id, index_label=index_label, preview=preview
    )


def test_rewind_choice_dataclass_shape():
    choice = ConsoleRewindChoice(
        kind="restore", message_id="m1", prompt_text="hello"
    )
    assert (choice.kind, choice.message_id, choice.prompt_text) == (
        "restore",
        "m1",
        "hello",
    )


def test_rewind_choice_is_frozen():
    choice = ConsoleRewindChoice(kind="restore", message_id="m1", prompt_text="hi")
    with pytest.raises(Exception):
        choice.kind = "summarize-up-to"  # type: ignore[misc]


def test_rewind_prompt_row_dataclass_shape():
    row = _row("m1", "#1", "hello there")
    assert (row.message_id, row.index_label, row.preview) == (
        "m1",
        "#1",
        "hello there",
    )


def test_rewind_prompt_row_is_frozen():
    row = _row("m1", "#1", "hello")
    with pytest.raises(Exception):
        row.preview = "changed"  # type: ignore[misc]


def test_modal_accepts_prompts_tuple_kwarg():
    rows = (_row("m1", "#1", "first"), _row("m2", "#2", "second"))
    modal = ConsoleRewindModal(prompts=rows)
    assert modal._prompts == rows


class _ModalHost(App):
    pass


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.asyncio
async def test_modal_renders_one_button_per_prompt_row_newest_first():
    rows = (
        _row("m2", "#2", "second prompt"),
        _row("m1", "#1", "first prompt"),
    )
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=rows)
        app.push_screen(modal)
        await pilot.pause()

        row_buttons = modal.query(".console-rewind-row")
        assert len(row_buttons) == 2
        assert row_buttons[0].id == "console-rewind-row-0"
        assert row_buttons[1].id == "console-rewind-row-1"
        # No action row until a prompt is selected.
        assert len(modal.query("#console-rewind-action-restore")) == 0


@pytest.mark.asyncio
async def test_selecting_a_row_reveals_the_action_row():
    rows = (_row("m1", "#1", "only prompt"),)
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=rows)
        app.push_screen(modal)
        await pilot.pause()

        await pilot.click("#console-rewind-row-0")
        await pilot.pause()

        assert modal.query_one("#console-rewind-action-restore", Button)
        assert modal.query_one("#console-rewind-action-summarize", Button)
        assert modal.query_one("#console-rewind-action-cancel", Button)


@pytest.mark.asyncio
async def test_restore_action_dismisses_choice_with_selected_row_id_and_text():
    rows = (
        _row("m2", "#2", "second prompt"),
        _row("m1", "#1", "first prompt"),
    )
    app = _ModalHost()
    result: list = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=rows)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        # Select the second row (native id "m1").
        await pilot.click("#console-rewind-row-1")
        await pilot.pause()
        await pilot.click("#console-rewind-action-restore")
        await pilot.pause()

    assert result == [
        ConsoleRewindChoice(kind="restore", message_id="m1", prompt_text="first prompt")
    ]


@pytest.mark.asyncio
async def test_summarize_action_dismisses_summarize_up_to_choice():
    rows = (_row("m1", "#1", "only prompt"),)
    app = _ModalHost()
    result: list = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=rows)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        await pilot.click("#console-rewind-row-0")
        await pilot.pause()
        await pilot.click("#console-rewind-action-summarize")
        await pilot.pause()

    assert result == [
        ConsoleRewindChoice(
            kind="summarize-up-to", message_id="m1", prompt_text="only prompt"
        )
    ]


@pytest.mark.asyncio
async def test_never_mind_dismisses_none():
    rows = (_row("m1", "#1", "only prompt"),)
    app = _ModalHost()
    result: list = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=rows)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        await pilot.click("#console-rewind-row-0")
        await pilot.pause()
        await pilot.click("#console-rewind-action-cancel")
        await pilot.pause()

    assert result == [None]


@pytest.mark.asyncio
async def test_escape_dismisses_none_before_any_selection():
    rows = (_row("m1", "#1", "only prompt"),)
    app = _ModalHost()
    result: list = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=rows)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

    assert result == [None]


@pytest.mark.asyncio
async def test_empty_prompts_shows_placeholder_and_no_rows():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(prompts=())
        app.push_screen(modal)
        await pilot.pause()

        assert len(modal.query(".console-rewind-row")) == 0
        empty = modal.query_one("#console-rewind-empty", Static)
        assert "No prior prompts" in _static_plain_text(empty)
