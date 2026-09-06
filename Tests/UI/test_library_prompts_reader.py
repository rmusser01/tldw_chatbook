"""Mounted Prompt journeys through the shared Library adaptive reader."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from unittest.mock import AsyncMock, Mock

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
    LibraryGlobalKeyProductionCSSHarness,
    LibraryHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_PROMPT,
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
async def test_prompts_global_f6_reaches_permanent_work_region(tmp_path) -> None:
    prompt_id, service = _seed_prompt(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryGlobalKeyProductionCSSHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        work_name = screen.query_one("#library-prompt-name", Input)
        rail = screen.query_one("#library-search-input", Input)
        items = screen.query_one("#library-prompts-filter", Input)
        rail.focus()
        await pilot.pause()

        for expected in (items, work_name, rail):
            await pilot.press("f6")
            await pilot.pause()
            assert screen.focused is expected


def _structured_prompt_definition(*, user_title: str = "Delivery contract"):
    return {
        "schema_version": 2,
        "kind": "block_prompt",
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Specialized role",
                        "syntax": "markdown",
                        "content": "Be exact.",
                        "mapping_hint": "Advanced-only system mapping hint.",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "delivery",
                        "title": user_title,
                        "syntax": "freeform",
                        "content": "Ship it.",
                        "mapping_hint": "Advanced-only user mapping hint.",
                    }
                ],
            },
        ],
    }


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
            lambda: screen._prompts_state.editor_armed,
            message="Prompt editor did not arm",
        )
        screen.query_one("#library-prompt-user", TextArea).load_text(
            "Summarize the verified changes."
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Basic edit did not reach the screen-owned Prompt draft",
        )

        assert screen._prompts_state.block_state is not None
        assert (
            screen._prompts_state.block_state.definition.lanes[1].blocks[0].content
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
        assert screen._prompts_state.status == (
            "Name is required; enter a Prompt name."
        )


@pytest.mark.asyncio
async def test_basic_save_preserves_advanced_only_prompt_fields(tmp_path) -> None:
    definition = _structured_prompt_definition()
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Structured release assistant",
        author="Advanced Author",
        details="Keep this Advanced description.",
        system_prompt="# Specialized role\n\nBe exact.",
        user_prompt="Ship it.",
        keywords=["release", "advanced-only"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="prompt",
    )
    app = _build_test_app()
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = "basic"
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.editor_armed,
            message="Prompt editor did not arm",
        )
        screen._prompts_state.editor_mode = "basic"
        work = screen.query_one(
            "#library-prompt-work-pane", LibraryPromptWorkPane
        )
        assert not work.basic_unavailable_reason, work.basic_unavailable_reason
        await work.set_editor_mode("basic")
        assert screen.query_one("#library-prompt-basic-region").display is True

        user_prompt = screen.query_one("#library-prompt-user", TextArea)
        user_prompt.focus()
        await pilot.press("end")
        await pilot.press("!")
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Basic edit did not dirty the structured Prompt",
        )
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._prompts_state.dirty
            and screen._prompts_state.version == 2,
            message="Basic structured Prompt save did not settle",
        )

    expected = json.loads(json.dumps(definition))
    expected["lanes"][1]["blocks"][0]["content"] = "Ship it.!"
    persisted = db.fetch_prompt_details(prompt_id)
    assert json.loads(persisted["prompt_definition"]) == expected
    assert persisted["author"] == "Advanced Author"
    assert persisted["details"] == "Keep this Advanced description."
    assert db.fetch_keywords_for_prompt(prompt_id) == ["advanced-only", "release"]


@pytest.mark.asyncio
async def test_invalid_advanced_block_routes_save_focus_to_its_owner(tmp_path) -> None:
    definition = _structured_prompt_definition(user_title="")
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Invalid structured assistant",
        author="Author",
        details="Needs a block title.",
        system_prompt="# Specialized role\n\nBe exact.",
        user_prompt="Ship it.",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="prompt",
    )
    app = _build_test_app()
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = "basic"
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen._prompts_state.editor_mode = "info"
        await screen.query_one(
            "#library-prompt-work-pane", LibraryPromptWorkPane
        ).set_editor_mode("info")
        title = screen.query_one("#prompt-block-title-delivery", Input)

        assert screen.query_one("#library-prompt-info-region").display is True
        screen.query_one("#library-prompt-details", Input).value = (
            "Still needs a block title."
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Info-view edit did not dirty the invalid Prompt",
        )
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-prompt-advanced-region").display
            and screen.focused is title,
            message="Invalid block save did not route to its Advanced owner",
        )

        assert screen.query_one("#library-prompt-info-region").display is False
        assert screen._prompts_state.status == (
            "Fix block validation errors before saving."
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
            if screen._prompts_state.detail is not None:
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
async def test_dirty_prompt_draft_survives_reader_route_and_revisit(tmp_path) -> None:
    """Reader routing parks one explicit-save Prompt draft without saving it."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Release assistant",
        author="Ada",
        details="Prepares release notes",
        system_prompt="Be exact.",
        user_prompt="Summarize {changes}.",
        keywords=["release", "summary"],
    )
    persisted = db.fetch_prompt_details(prompt_id)
    save_prompt = AsyncMock(wraps=service.save_prompt)
    service.save_prompt = save_prompt
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-mode-info", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.editor_mode == "info",
            message="Prompt Info mode did not settle",
        )
        screen.query_one(
            "#library-prompt-name", Input
        ).value = "Unsaved release assistant"
        screen.query_one(
            "#library-prompt-details", Input
        ).value = "Unsaved route details"
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Prompt route draft did not become dirty",
        )
        assert screen._prompts_state.detail is not None

        screen.query_one("#library-row-browse-skills", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skills-reader-shell")
        assert screen._library_selected_row_id == "browse-skills"

        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-reader-shell")
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        assert screen.query_one("#library-prompt-name", Input).value == (
            "Unsaved release assistant"
        )
        assert screen.query_one("#library-prompt-details", Input).value == (
            "Unsaved route details"
        )
        assert screen._prompts_state.editor_mode == "info"
        assert screen._prompts_state.dirty is True
        assert screen._prompts_state.selected_prompt_id == prompt_id
        assert screen._prompts_state.loaded_id == prompt_id
        assert len(screen.query("#library-prompts-reader-shell")) == 1

    save_prompt.assert_not_awaited()
    after = db.fetch_prompt_details(prompt_id)
    assert after["name"] == persisted["name"]
    assert after["details"] == persisted["details"]
    assert after["version"] == persisted["version"]


@pytest.mark.asyncio
async def test_dirty_create_prompt_draft_vetoes_reader_route(tmp_path) -> None:
    """A not-yet-saved Create draft stays put until saved or abandoned."""
    _db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.editor_armed,
            message="Create Prompt editor did not arm",
        )
        name = screen.query_one("#library-prompt-name", Input)
        name.value = "Unsaved Create draft"
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Create Prompt draft did not become dirty",
        )

        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_SKILLS}", Button).press()
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_PROMPT
        assert screen._prompts_state.view == "editor"
        assert screen._prompts_state.dirty is True
        assert screen.query_one("#library-prompt-name", Input).value == (
            "Unsaved Create draft"
        )
        assert not screen.query("#library-skills-reader-shell")


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
async def test_import_from_clean_editor_replaces_work_and_cancel_restores_editor(
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
        work = screen.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)

        screen.query_one("#library-prompts-import", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-import-path")
        await _wait_for_condition(
            pilot,
            lambda: len(work.query("#library-prompt-name")) == 0,
            message="Prompt editor content remained mounted behind Import",
        )

        assert screen.query_one("#library-prompt-work-pane") is work
        assert len(work.query("#library-prompt-name")) == 0
        screen.query_one("#library-prompts-import-cancel", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")
        assert screen.query_one("#library-prompt-name", Input).value == (
            "Release assistant"
        )
        assert screen._prompts_state.selected_prompt_id == prompt_id


@pytest.mark.asyncio
async def test_import_from_dirty_editor_is_vetoed_without_hiding_draft(tmp_path) -> None:
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
            lambda: screen._prompts_state.editor_armed,
            message="Prompt editor did not arm",
        )
        name = screen.query_one("#library-prompt-name", Input)
        name.value = "Unsaved release assistant"
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Prompt draft did not become dirty",
        )

        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()

        assert not screen.query("#library-prompts-import-path")
        assert screen.query_one("#library-prompt-name", Input).value == (
            "Unsaved release assistant"
        )
        assert screen._prompts_state.selected_prompt_id == prompt_id


@pytest.mark.asyncio
async def test_items_browse_settles_while_prompt_editor_remains_open(
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
        controller = screen._library_prompt_browse_controller
        prior_token = controller.result.request_token

        screen.query_one("#library-prompts-sort", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-sort-name")
        screen.query_one("#library-prompts-sort-name", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.result.request_token > prior_token
                and controller.result.status == "ready"
            ),
            message="Retained Prompt Items browse did not settle in editor mode",
        )

        assert screen._prompts_state.view == "editor"
        assert screen._prompts_state.selected_prompt_id == prompt_id
        assert screen.query_one("#library-prompt-name", Input).value == (
            "Release assistant"
        )

        async def request_and_wait(**changes) -> None:
            prior_request = controller.result.request_token
            screen._request_library_prompts_browse(
                dataclasses.replace(controller.mutation_refresh_scope, **changes),
                focus_identity=None,
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    controller.result.request_token > prior_request
                    and controller.result.status != "loading"
                ),
                message=f"Retained Prompt Items request did not settle: {changes}",
            )
            assert controller.result.status != "error"
            assert screen._prompts_state.view == "editor"
            assert screen._prompts_state.selected_prompt_id == prompt_id

        await request_and_wait(query="release", page=1)
        await request_and_wait(query="", collection_id=999_999, page=1)
        await request_and_wait(collection_id=None, page=2)

        original_browse = service.browse_prompts

        async def fail_browse(**_kwargs):
            raise RuntimeError("temporary browse failure")

        monkeypatch.setattr(service, "browse_prompts", fail_browse)
        failed_token = controller.result.request_token
        screen._request_library_prompts_browse(
            dataclasses.replace(
                controller.mutation_refresh_scope,
                collection_id=None,
                page=1,
            ),
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.result.request_token > failed_token
                and controller.result.status == "error"
            ),
            message="Retained Prompt Items request did not expose retry",
        )
        monkeypatch.setattr(service, "browse_prompts", original_browse)
        retry_token = controller.result.request_token
        screen.query_one("#library-prompts-retry", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.result.request_token > retry_token
                and controller.result.status == "ready"
            ),
            message="Retained Prompt Items retry did not settle",
        )

        membership_token = controller.result.request_token
        screen._refresh_library_prompt_after_membership_apply()
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.result.request_token > membership_token
                and controller.result.status == "ready"
            ),
            message="Membership Apply did not refresh retained Prompt Items",
        )
        assert screen.query_one("#library-prompt-name", Input).value == (
            "Release assistant"
        )


@pytest.mark.asyncio
async def test_same_prompt_older_detail_load_cannot_overwrite_newer_generation(
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
        older = asyncio.Future()
        newer = asyncio.Future()
        replies = iter((older, newer))

        async def delayed_detail(*_args, **_kwargs):
            return await next(replies)

        monkeypatch.setattr(screen, "_run_library_service_call", delayed_detail)
        first = asyncio.create_task(screen._refresh_library_prompt_detail(prompt_id))
        await pilot.pause()
        second = asyncio.create_task(screen._refresh_library_prompt_detail(prompt_id))
        await pilot.pause()
        newer.set_result(
            {
                "local_id": prompt_id,
                "name": "Newer detail",
                "author": "Ada",
                "details": "new",
                "system_prompt": "Be exact.",
                "user_prompt": "Summarize {changes}.",
                "keywords": ["release"],
                "version": 3,
            }
        )
        await second
        older.set_result(
            {
                "local_id": prompt_id,
                "name": "Older detail",
                "author": "Ada",
                "details": "old",
                "system_prompt": "Be exact.",
                "user_prompt": "Summarize {changes}.",
                "keywords": ["release"],
                "version": 2,
            }
        )
        await first
        await pilot.pause()

        assert screen._prompts_state.detail is not None
        assert screen._prompts_state.detail["name"] == "Newer detail"
        assert screen._prompts_state.version == 3


@pytest.mark.asyncio
async def test_detail_failure_keeps_prior_prompt_locked_and_retry_loads_selection(
    tmp_path,
    monkeypatch,
) -> None:
    db, service = _real_prompt_scope_service(tmp_path)
    first_id, _uuid, _message = db.add_prompt(
        name="First prompt",
        author="Ada",
        details="first",
        system_prompt="First system",
        user_prompt="First user",
        keywords=["first"],
    )
    second_id, _uuid, _message = db.add_prompt(
        name="Second prompt",
        author="Grace",
        details="second",
        system_prompt="Second system",
        user_prompt="Second user",
        keywords=["second"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, first_id)
        original_get_prompt = service.get_prompt

        def fail_second(*, prompt_identifier, **kwargs):
            if prompt_identifier == second_id:
                raise RuntimeError("simulated detail failure")
            return original_get_prompt(prompt_identifier=prompt_identifier, **kwargs)

        monkeypatch.setattr(service, "get_prompt", fail_second)
        screen.query_one(f"#library-prompt-row-{second_id}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-detail-retry")

        assert screen._prompts_state.selected_prompt_id == second_id
        assert screen._prompts_state.loaded_id == first_id
        assert screen.query_one("#library-prompt-name", Input).value == "First prompt"
        assert screen.query_one("#library-prompt-name", Input).disabled is True
        assert "showing “first prompt”" in str(
            screen.query_one("#library-prompt-detail-status", Static).renderable
        ).lower()

        monkeypatch.setattr(service, "get_prompt", original_get_prompt)
        screen.query_one("#library-prompt-detail-retry", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._prompts_state.loaded_id == second_id
                and screen.query_one("#library-prompt-name", Input).value
                == "Second prompt"
            ),
            message="Prompt detail retry did not load the selected identity",
        )
        assert screen.query_one("#library-prompt-name", Input).disabled is False


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
        screen._prompts_state.sort_choices_visible = True

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
        screen._prompts_state.status = "Restore failed safely."

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

        assert screen._prompts_state.dirty is False
        prompts_list.query_one("#library-prompts-select", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.select_mode,
            message=lambda: (
                "Prompt Select action did not enter bulk mode: "
                f"disabled={prompts_list.query_one('#library-prompts-select', Button).disabled!r}, "
                f"freshness={screen._library_prompt_browse_controller.freshness!r}, "
                f"status={screen._library_prompt_browse_controller.result.status!r}, "
                f"rows={len(screen._build_library_prompts_state().rows)!r}, "
                f"dirty={screen._prompts_state.dirty!r}"
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
        assert screen._prompts_state.detail is not None
        assert screen._prompts_state.selected_prompt_id == prompt_id

        prompts_list.query_one("#library-prompts-selection-done", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._prompts_state.select_mode
                and not work.query_one("#library-prompt-bulk-status", Static).display
                and not work.query_one("#library-prompt-name", Input).disabled
            ),
            message="Leaving Prompt bulk mode did not restore the loaded editor",
        )
        assert screen.query_one("#library-prompt-work-pane") is work
        assert screen._prompts_state.selected_prompt_id == prompt_id


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
