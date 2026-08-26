"""Mounted Prompt journeys through the shared Library adaptive reader."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _open_prompts_list,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library import (
    LibraryAdaptiveReaderShell,
    LibraryPromptWorkPane,
    LibraryPromptsListCanvas,
)
from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas


def _seed_prompt(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Release assistant",
        author="Ada",
        details="Prepares release notes",
        system_prompt="Be exact.",
        user_prompt="Summarize {changes}.",
        keywords=["release", "summary"],
    )
    return prompt_id, service


@pytest.mark.asyncio
async def test_basic_edit_reaches_screen_owned_prompt_draft(tmp_path) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_editor_armed,
            message="Prompt editor did not arm",
        )

        screen.query_one("#library-prompt-user", TextArea).load_text(
            "Summarize the verified changes."
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_dirty,
            message="Basic edit did not reach the screen-owned Prompt draft",
        )

        assert screen._library_prompt_block_state is not None
        assert (
            screen._library_prompt_block_state.definition.lanes[1].blocks[0].content
            == "Summarize the verified changes."
        )

        name = screen.query_one("#library-prompt-name", Input)
        name.value = ""
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is name,
            message="Prompt validation did not focus the owning Name control",
        )
        assert screen._library_prompt_status == (
            "Name is required; enter a Prompt name."
        )


@pytest.mark.asyncio
async def test_prompts_mount_three_retained_roles_once(tmp_path) -> None:
    _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-prompts-reader-shell")

        shell = screen.query_one(
            "#library-prompts-reader-shell", LibraryAdaptiveReaderShell
        )
        rail = shell.query_one("#library-rail")
        items = shell.query_one("#library-prompts-canvas", LibraryPromptsListCanvas)
        work = shell.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        identities = (id(shell), id(rail), id(items), id(work))

        shell.library_grip.press()
        await pilot.pause()
        shell.library_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()

        assert (id(shell), id(rail), id(items), id(work)) == identities
        assert work.is_mounted and work.display
        assert len(shell.query(".library-adaptive-reader-pane-grip")) == 2


@pytest.mark.asyncio
async def test_list_and_work_identity_survive_basic_advanced_and_info(tmp_path) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)
        prompts_list = screen.query_one(
            "#library-prompts-canvas", LibraryPromptsListCanvas
        )
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)

        row = await _wait_for_selector(
            screen,
            pilot,
            f"#library-prompt-row-{prompt_id}",
        )
        assert isinstance(row, Button)
        row.press()
        for _ in range(150):
            if screen._library_prompt_detail is not None:
                break
            await pilot.pause(0.02)
        await _wait_for_selector(screen, pilot, "#library-prompt-mode-info")
        name = screen.query_one("#library-prompt-name", Input)

        assert screen.query_one("#library-prompt-basic-region").display is True
        screen.query_one("#library-prompt-mode-advanced", Button).press()
        await pilot.pause()
        screen.query_one("#library-prompt-mode-info", Button).press()
        await pilot.pause()

        assert screen.query_one("#library-prompts-canvas") is prompts_list
        assert screen.query_one("#library-prompt-work-pane") is work
        assert screen.query_one("#library-prompt-name") is name
        assert screen.query_one("#library-prompt-info-region").display is True


@pytest.mark.asyncio
async def test_import_replaces_only_work_content_and_keeps_list_mounted(
    tmp_path,
) -> None:
    _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)
        prompts_list = screen.query_one("#library-prompts-canvas")
        work = screen.query_one("#library-prompt-work-pane")

        screen.query_one("#library-prompts-import", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-import-path")

        assert screen.query_one("#library-prompts-canvas") is prompts_list
        assert screen.query_one("#library-prompt-work-pane") is work
        assert not prompts_list.query("#library-prompts-import-path")
        assert work.query_one("#library-prompts-import-path").is_mounted


@pytest.mark.asyncio
async def test_unchanged_list_sync_does_not_recompose_prompt_work_pane(
    tmp_path,
    monkeypatch,
) -> None:
    _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        refresh = Mock()
        monkeypatch.setattr(work, "refresh", refresh)

        work.sync_state(**screen._library_prompt_work_pane_kwargs())

        refresh.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_items_sync_is_not_blocked_by_work_projection_failure(
    tmp_path,
    monkeypatch,
) -> None:
    _prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)
        prompts_list = screen.query_one(
            "#library-prompts-canvas", LibraryPromptsListCanvas
        )
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        monkeypatch.setattr(
            work,
            "sync_state",
            Mock(side_effect=RuntimeError("work projection failed")),
        )
        screen._library_prompts_sort_choices_visible = True

        synced = _sync_library_canvas(
            screen,
            "prompts",
            allow_screen_fallback=False,
        )

        assert synced is False
        assert prompts_list.sort_choices_visible is True


@pytest.mark.asyncio
async def test_prompt_work_sync_is_not_blocked_by_items_projection_failure(
    tmp_path,
    monkeypatch,
) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        prompts_list = screen.query_one(
            "#library-prompts-canvas", LibraryPromptsListCanvas
        )
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        monkeypatch.setattr(
            prompts_list,
            "sync_state",
            Mock(side_effect=RuntimeError("items projection failed")),
        )
        screen._library_prompt_status = "Restore failed safely."

        synced = _sync_library_canvas(
            screen,
            "prompts",
            allow_screen_fallback=False,
        )

        assert synced is False
        assert work.status == "Restore failed safely."


@pytest.mark.asyncio
async def test_prompt_projection_fallback_runs_follow_up_once(
    tmp_path,
    monkeypatch,
) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        prompts_list = screen.query_one(
            "#library-prompts-canvas", LibraryPromptsListCanvas
        )
        monkeypatch.setattr(
            prompts_list,
            "sync_state",
            Mock(side_effect=RuntimeError("items projection failed")),
        )
        follow_up = Mock()

        synced = _sync_library_canvas(screen, "prompts", then=follow_up)
        for _ in range(10):
            await pilot.pause()

        assert synced is False
        follow_up.assert_called_once_with()


@pytest.mark.asyncio
async def test_bulk_mode_keeps_loaded_prompt_as_labelled_read_only_preview(
    tmp_path,
) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        prompts_list = screen.query_one(
            "#library-prompts-canvas", LibraryPromptsListCanvas
        )
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)

        assert screen._library_prompt_dirty is False
        prompts_list.query_one("#library-prompts-select", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_select_mode,
            message=lambda: (
                "Prompt Select action did not enter bulk mode: "
                f"disabled={prompts_list.query_one('#library-prompts-select', Button).disabled!r}, "
                f"freshness={screen._library_prompt_browse_controller.freshness!r}, "
                f"status={screen._library_prompt_browse_controller.result.status!r}, "
                f"rows={len(screen._build_library_prompts_state().rows)!r}, "
                f"dirty={screen._library_prompt_dirty!r}"
            ),
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                work.query_one("#library-prompt-bulk-status", Static).display
                and work.query_one("#library-prompt-name", Input).disabled
            ),
            message="Prompt bulk preview did not become read-only",
        )

        bulk_status = work.query_one("#library-prompt-bulk-status", Static)
        assert "Read-only preview" in str(bulk_status.renderable)
        assert "Not included" in str(bulk_status.renderable)
        assert work.query_one("#library-prompt-name", Input).disabled is True
        assert work.query_one("#library-prompt-system", TextArea).read_only is True
        for selector in (
            "#library-prompt-back",
            "#library-prompt-save",
            "#library-prompt-insert-console",
            "#library-prompt-more-actions",
        ):
            assert work.query_one(selector, Button).disabled is True
        assert screen.check_action("library_prompt_editor_back", ()) is False

        prompts_list.query_one(f"#library-prompt-row-{prompt_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: str(
                work.query_one("#library-prompt-bulk-status", Static).renderable
            ).endswith("Included in bulk selection"),
            message="Prompt bulk preview did not reflect list membership",
        )

        bulk_status = work.query_one("#library-prompt-bulk-status", Static)
        assert str(bulk_status.renderable).endswith("Included in bulk selection")
        assert screen._library_prompt_detail is not None
        assert screen._selected_prompt_id == prompt_id

        prompts_list.query_one("#library-prompts-selection-done", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._library_prompt_select_mode
                and not work.query_one("#library-prompt-bulk-status", Static).display
                and not work.query_one("#library-prompt-name", Input).disabled
            ),
            message="Leaving Prompt bulk mode did not restore the loaded editor",
        )
        assert screen.query_one("#library-prompt-work-pane") is work
        assert screen._selected_prompt_id == prompt_id


@pytest.mark.asyncio
async def test_eighty_columns_protect_basic_editor_and_keep_restore_grips(
    tmp_path,
) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        shell = screen.query_one(
            "#library-prompts-reader-shell", LibraryAdaptiveReaderShell
        )
        await pilot.pause()

        assert shell.work.region.width >= 48
        assert shell.library_grip.region.width == 5
        assert shell.items_grip.region.width == 5
        assert shell.library_grip.region.right <= 80
        assert shell.items_grip.region.right <= 80
