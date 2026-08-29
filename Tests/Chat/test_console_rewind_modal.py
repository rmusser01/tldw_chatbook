"""Console `/rewind` menu modal (Task 1 of the SP2 `/rewind` program).

Mirrors the mounted-test style of ``Tests/Chat/test_console_edit_message_modal.py``
and ``Tests/UI/test_console_edit_modal_keystroke_guard.py``: construction-level
shape assertions plus a handful of ``run_test`` pilot flows covering the
two-level select-a-prompt -> choose-an-action flow.
"""

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console import console_rewind_modal
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


def test_summarize_from_kind_is_stable():
    assert getattr(console_rewind_modal, "KIND_SUMMARIZE_FROM", None) == (
        "summarize-from"
    )


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


def test_modal_defines_no_forbidden_terminal_or_global_bindings():
    forbidden = {
        "ctrl+c",
        "ctrl+v",
        "ctrl+x",
        "ctrl+s",
        "ctrl+d",
        "ctrl+z",
        "ctrl+a",
        "ctrl+r",
        "ctrl+w",
        "ctrl+p",
        "ctrl+q",
        "f1",
        "f6",
    }
    keys = {binding[0] for binding in ConsoleRewindModal.BINDINGS}
    assert keys == {"escape"}
    assert keys.isdisjoint(forbidden)


class _ModalHost(App):
    pass


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _painted_widget_text(modal: ConsoleRewindModal, widget: Static) -> str:
    strips = modal._compositor.render_strips()
    visible_rows = strips[
        max(0, widget.region.y) : min(len(strips), widget.region.bottom)
    ]
    return "\n".join(
        row.text[max(0, widget.region.x) : widget.region.right]
        for row in visible_rows
    )


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
        assert modal.query_one("#console-rewind-action-summarize-from", Button)
        assert modal.query_one("#console-rewind-action-cancel", Button)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_action_order_copy_and_keyboard_reachability_at_supported_widths(size):
    """A missing/reordered action or clipped cost warning breaks this flow."""
    app = _ModalHost()
    result: list = []
    async with app.run_test(size=size) as pilot:
        modal = ConsoleRewindModal(
            prompts=(_row("m1", "#1", "only prompt"),),
            has_effective_memory=True,
        )
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        await pilot.click("#console-rewind-row-0")
        await pilot.pause()

        actions = list(modal.query("#console-rewind-actions Button"))
        assert [button.id for button in actions] == [
            "console-rewind-action-restore",
            "console-rewind-action-summarize",
            "console-rewind-action-summarize-from",
            "console-rewind-action-cancel",
        ]
        assert [str(button.label) for button in actions] == [
            "Restore to here",
            "Summarize up to here",
            "Summarize from here",
            "Never mind",
        ]
        panel = modal.query_one("#console-rewind-modal")
        summary_pairs = (
            (
                "#console-rewind-action-summarize",
                "#console-rewind-action-summarize-copy",
            ),
            (
                "#console-rewind-action-summarize-from",
                "#console-rewind-action-summarize-from-copy",
            ),
        )
        for action_id, copy_id in summary_pairs:
            copy = _static_plain_text(modal.query_one(copy_id, Static))
            assert copy == (
                "Uses the active model once\n"
                "Replaces current conversation memory"
            )
            action = modal.query_one(action_id, Button)
            warning = modal.query_one(copy_id, Static)
            action.focus()
            await pilot.pause()
            assert action.region.y >= panel.content_region.y
            assert action.region.bottom <= panel.content_region.bottom
            assert warning.region.y >= panel.content_region.y
            assert warning.region.bottom <= panel.content_region.bottom
            painted = _painted_widget_text(modal, warning)
            assert "Uses the active model once" in painted
            assert "Replaces current conversation memory" in painted

        modal.query_one("#console-rewind-row-0", Button).focus()
        await pilot.pause()
        for action_id in [
            "console-rewind-action-restore",
            "console-rewind-action-summarize",
            "console-rewind-action-summarize-from",
            "console-rewind-action-cancel",
        ]:
            await pilot.press("tab")
            await pilot.pause()
            assert app.focused is modal.query_one(f"#{action_id}", Button)
            assert app.focused.region.width > 0

        await pilot.press("escape")
        await pilot.pause()

    assert result == [None]


@pytest.mark.asyncio
async def test_summary_actions_show_cost_without_replacement_copy_for_raw_memory():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(
            prompts=(_row("m1", "#1", "only prompt"),),
            has_effective_memory=False,
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-rewind-row-0")
        await pilot.pause()

        for copy_id in (
            "#console-rewind-action-summarize-copy",
            "#console-rewind-action-summarize-from-copy",
        ):
            assert _static_plain_text(modal.query_one(copy_id, Static)) == (
                "Uses the active model once"
            )


@pytest.mark.asyncio
async def test_known_summary_refusal_disables_only_summary_actions_with_guidance():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleRewindModal(
            prompts=(_row("m1", "#1", "only prompt"),),
            summary_disabled_reason="Finish the current exchange before summarizing.",
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-rewind-row-0")
        await pilot.pause()

        assert not modal.query_one("#console-rewind-action-restore", Button).disabled
        assert modal.query_one("#console-rewind-action-summarize", Button).disabled
        assert modal.query_one(
            "#console-rewind-action-summarize-from", Button
        ).disabled
        assert not modal.query_one("#console-rewind-action-cancel", Button).disabled
        guidance = modal.query_one("#console-rewind-summary-disabled", Static)
        assert _static_plain_text(guidance) == (
            "Finish the current exchange before summarizing."
        )


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
async def test_summarize_from_action_preserves_selected_native_id_and_text():
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

        await pilot.click("#console-rewind-row-1")
        await pilot.pause()
        await pilot.click("#console-rewind-action-summarize-from")
        await pilot.pause()

    assert result == [
        ConsoleRewindChoice(
            kind="summarize-from",
            message_id="m1",
            prompt_text="first prompt",
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
