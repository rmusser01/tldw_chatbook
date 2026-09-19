"""Keyboard journeys through saved Prompt actions and local recovery."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _painted_text,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
)


def _saved_prompt_host(tmp_path, theme):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _, _ = db.add_prompt(
        name="Action review [bold] café",
        author="",
        details="",
        user_prompt="Keep this reusable message.",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    return db, service, prompt_id, host


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_more_actions_keyboard_reaches_every_painted_control(
    tmp_path, size, theme
):
    _, _, prompt_id, host = _saved_prompt_host(tmp_path, theme)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        opener = screen.query_one("#library-prompt-more-actions", Button)
        opener.focus()
        await pilot.press("enter")
        for action_id, label in (
            ("export", "Export"),
            ("copy", "Copy Markdown"),
            ("duplicate", "Duplicate"),
            ("more-collections", "Collections"),
            ("more-history", "History"),
            ("delete", "Delete"),
        ):
            await pilot.press("tab")
            action = screen.query_one(f"#library-prompt-{action_id}", Button)
            assert screen.focused is action
            assert label in _painted_text(host, action.region), (
                label,
                action.region,
                host.size,
            )
        await pilot.press("escape")
        assert not screen.query_one("#library-prompt-more-actions-region").display
        assert screen.focused is opener
        assert "More actions" in _painted_text(host, opener.region)


async def _activate_more_action(screen, pilot, action_id):
    opener = screen.query_one("#library-prompt-more-actions", Button)
    opener.focus()
    if not screen.query_one("#library-prompt-more-actions-region").display:
        await pilot.press("enter")
    for _ in range(6):
        await pilot.press("tab")
        if screen.focused.id == f"library-prompt-{action_id}":
            await pilot.press("enter")
            return
    raise AssertionError(f"Could not reach Prompt action {action_id}")


async def _answer_delete(host, pilot, *, confirm):
    await _wait_for_condition(
        pilot,
        lambda: isinstance(host.screen, PromptDeleteConfirmationModal),
        message="Delete confirmation did not open",
    )
    button = host.screen.query_one(
        "#prompt-delete-confirm" if confirm else "#prompt-delete-cancel", Button
    )
    button.focus()
    await pilot.pause()
    assert str(button.label) in _painted_text(host, button.region)
    await pilot.press("enter")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_duplicate_delete_cancel_failure_undo_and_dismiss_journey(
    tmp_path, monkeypatch, size, theme
):
    db, service, original_id, host = _saved_prompt_host(tmp_path, theme)
    original = db.fetch_prompt_details(original_id)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, original_id)
        await _activate_more_action(screen, pilot, "duplicate")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.query_one("#library-prompt-name", Input).value
                == "Action review [bold] café (copy)"
            ),
            message="Duplicate did not present its detached working copy",
        )
        assert screen._prompts_state.selected_prompt_id is None
        assert screen._prompts_state.dirty
        assert (
            screen.query_one("#library-prompt-user", TextArea).text
            == original["user_prompt"]
        )
        assert db.fetch_prompt_details(original_id) == original
        assert not screen.query_one("#library-prompt-more-actions", Button).display
        screen.query_one("#library-prompt-save", Button).focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._prompts_state.selected_prompt_id not in (None, original_id)
                and not screen._prompts_state.dirty
                and screen._library_prompt_browse_controller.result.status == "ready"
                and screen._library_prompt_browse_controller.result.total_items == 2
            ),
            message="Saved duplicate did not settle as a distinct Prompt",
        )
        copy_id = screen._prompts_state.selected_prompt_id
        field = screen.query_one("#library-prompt-user", TextArea)
        await _activate_more_action(screen, pilot, "delete")
        await _answer_delete(host, pilot, confirm=False)
        await pilot.pause()
        assert screen.query_one("#library-prompt-user", TextArea) is field
        assert db.fetch_prompt_details(copy_id)["version"] == 1

        actual_delete = service.delete_prompts
        attempts = 0

        async def fail_once(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("test storage unavailable")
            return await actual_delete(**kwargs)

        monkeypatch.setattr(service, "delete_prompts", fail_once)
        await _activate_more_action(screen, pilot, "delete")
        await _answer_delete(host, pilot, confirm=True)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._prompts_state.status
                == "Could not delete this prompt. Nothing was deleted."
                and not screen._prompts_state.mutation_in_flight
            ),
            message="Delete failure did not settle with a recovery message",
        )
        assert screen.query_one("#library-prompt-user", TextArea) is field
        assert field.text == original["user_prompt"]
        assert db.fetch_prompt_details(copy_id)["version"] == 1
        status = screen.query_one("#library-prompt-save-status", Static)
        assert "Nothing was deleted" in _painted_text(host, status.region)
        delete_action = screen.query_one("#library-prompt-delete", Button)
        assert not delete_action.disabled
        assert "Delete" in _painted_text(host, delete_action.region)

        for recovery in ("undo", "receipt-dismiss"):
            await _activate_more_action(screen, pilot, "delete")
            await _answer_delete(host, pilot, confirm=True)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._prompts_state.view == "list"
                    and not screen._prompts_state.mutation_in_flight
                    and screen._library_prompt_browse_controller.result.status
                    == "ready"
                    and screen._library_prompt_browse_controller.result.total_items == 1
                    and not screen.query(f"#library-prompt-row-{copy_id}")
                ),
                message="Delete did not settle its row and recovery receipt",
            )
            assert (
                db.fetch_prompt_details(copy_id, include_deleted=True)["deleted"] == 1
            )
            action = screen.query_one(f"#library-prompts-delete-{recovery}", Button)
            action.focus()
            await pilot.pause()
            assert screen.focused is action
            assert str(action.label) in _painted_text(host, action.region)
            receipt = screen.query_one("#library-prompts-delete-receipt-copy", Static)
            assert "(copy)" in str(receipt.renderable)
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    not screen.query("#library-prompts-delete-receipt-copy")
                    and not screen._prompts_state.mutation_in_flight
                ),
                message="Recovery action did not clear its receipt",
            )
            if recovery == "undo":
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        bool(screen.query(f"#library-prompt-row-{copy_id}"))
                        and screen._library_prompt_browse_controller.result.total_items
                        == 2
                    ),
                    message="Undo did not restore the copied Prompt",
                )
                restored = db.fetch_prompt_details(copy_id)
                assert restored["user_prompt"] == original["user_prompt"]
                assert restored["version"] == 3
                screen.query_one(f"#library-prompt-row-{copy_id}", Button).focus()
                await pilot.press("enter")
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._prompts_state.selected_prompt_id == copy_id
                        and bool(screen.query("#library-prompt-more-actions"))
                    ),
                    message="Restored Prompt did not reopen",
                )
            else:
                assert (
                    db.fetch_prompt_details(copy_id, include_deleted=True)["deleted"]
                    == 1
                )
                assert not screen.query(f"#library-prompt-row-{copy_id}")
        assert attempts == 3
        assert db.fetch_prompt_details(original_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_new_prompt_route_can_delete_and_restore_its_saved_item(tmp_path, size):
    db, _, original_id, host = _saved_prompt_host(tmp_path, "textual-dark")
    original = db.fetch_prompt_details(original_id)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("create-prompt")
        await pilot.pause()
        screen.query_one("#library-prompt-name", Input).value = "Created source"
        screen.query_one("#library-prompt-user", TextArea).text = "Created body"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._prompts_state.selected_prompt_id is not None
                and not screen._prompts_state.dirty
                and not screen._library_prompt_write_worker_is_active()
            ),
            message="Create-route source did not save",
        )
        created_id = screen._prompts_state.selected_prompt_id
        assert screen._library_selected_row_id == "create-prompt"
        await _activate_more_action(screen, pilot, "delete")
        await _answer_delete(host, pilot, confirm=True)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._prompts_state.view == "list"
                and not screen._prompts_state.mutation_in_flight
                and bool(screen.query("#library-prompts-delete-undo"))
            ),
            message="Confirmed Create-route deletion was not applied",
        )
        assert db.fetch_prompt_details(created_id, include_deleted=True)["deleted"] == 1
        undo = screen.query_one("#library-prompts-delete-undo", Button)
        undo.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                bool(screen.query(f"#library-prompt-row-{created_id}"))
                and not screen.query("#library-prompts-delete-receipt-copy")
                and screen._library_prompt_browse_controller.result.total_items == 2
            ),
            message="Create-route Undo did not restore its row and count",
        )
        assert db.fetch_prompt_details(created_id)["version"] == 3
        assert db.fetch_prompt_details(original_id) == original


@pytest.mark.asyncio
async def test_replaced_prompt_work_pane_ignores_pending_resize_callback(tmp_path):
    _, _, prompt_id, host = _saved_prompt_host(tmp_path, "textual-dark")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        work = screen.query_one("#library-prompt-work-pane")
        pending_reveal = work._reveal_editor_focus
        await work.remove()
        target = screen.query_one("#library-search-input", Input)
        target.focus()
        await pilot.pause()
        pending_reveal()
        assert screen.focused is target
