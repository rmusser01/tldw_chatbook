"""Mounted Skills journeys through the shared Library adaptive reader."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skills_canvas import _build_test_app
from tldw_chatbook.Library.library_skills_state import (
    SkillEditorSupportingFile,
    build_skill_editor_state,
)
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.Widgets.Library import LibraryAdaptiveReaderShell
from tldw_chatbook.Widgets.Library.library_skills_canvas import (
    LibrarySkillsListCanvas,
)


def _wire_skills(app, tmp_path) -> None:
    local = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=None,
        allow_untrusted_without_trust_service=True,
        policy_enforcer=None,
    )
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.prompt_scope_service = object()
    app.study_scope_service = object()
    app.study_quiz_scope_service = object()
    app.skills_scope_service = SkillsScopeService(
        local_service=local,
        server_service=None,
        policy_enforcer=None,
    )
    app.local_skill_trust_service = object()
    return local


async def _open_skills_reader(screen, pilot) -> None:
    screen.query_one("#library-row-browse-skills", Button).press()
    await _wait_for_selector(screen, pilot, "#library-skills-reader-shell")


@pytest.mark.asyncio
async def test_skills_mount_three_retained_roles_and_default_to_overview(
    tmp_path,
) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)

        shell = screen.query_one(
            "#library-skills-reader-shell", LibraryAdaptiveReaderShell
        )
        items = shell.query_one("#library-skills-canvas", LibrarySkillsListCanvas)
        work = shell.query_one("#library-skill-work-pane")
        identities = (id(shell), id(items), id(work))

        screen.query_one("#library-skills-import", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skills-import-path")
        assert screen.query_one("#library-skills-canvas") is items
        assert screen.query_one("#library-skill-work-pane") is work
        screen.query_one("#library-skills-import-cancel", Button).press()
        await pilot.pause()

        row = await _wait_for_selector(
            screen, pilot, "#library-skill-row-release-notes"
        )
        assert isinstance(row, Button)
        row.press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-overview")

        assert screen.query_one("#library-skills-reader-shell") is shell
        assert screen.query_one("#library-skills-canvas") is items
        assert screen.query_one("#library-skill-work-pane") is work
        assert (id(shell), id(items), id(work)) == identities
        assert screen.query_one("#library-skill-overview-region").display is True
        assert work.is_mounted and work.display

        selected_row = screen.query_one(
            "#library-skill-row-release-notes", Button
        )
        assert selected_row.has_class("is-selected")
        assert str(selected_row.label).startswith("› ")


@pytest.mark.asyncio
async def test_skill_modes_preserve_list_work_and_one_live_draft(
    tmp_path,
) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        items = screen.query_one("#library-skills-canvas")
        work = screen.query_one("#library-skill-work-pane")
        screen.query_one("#library-skill-row-release-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-edit")

        screen.query_one("#library-skill-mode-edit", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-name")
        description = screen.query_one("#library-skill-description", Input)
        description.value = "Unsaved reader draft"
        await pilot.pause()
        screen.query_one("#library-skill-mode-trust", Button).press()
        await pilot.pause()
        screen.query_one("#library-skill-mode-files", Button).press()
        await pilot.pause()
        screen.query_one("#library-skill-mode-edit", Button).press()
        await pilot.pause()

        assert screen.query_one("#library-skills-canvas") is items
        assert screen.query_one("#library-skill-work-pane") is work
        assert screen.query_one("#library-skill-description", Input).value == (
            "Unsaved reader draft"
        )
        assert screen.query_one("#library-skill-edit-region").display is True


@pytest.mark.asyncio
async def test_same_skill_older_detail_result_cannot_replace_newer_generation(
    tmp_path,
) -> None:
    """A cancelled thread may settle late even when the selected name is unchanged."""
    app = _build_test_app()
    _wire_skills(app, tmp_path)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._selected_skill_name = "release-notes"
        screen._library_skills_view = "editor"
        screen._library_skill_reader_mode = "overview"
        app.skills_scope_service = SimpleNamespace(get_skill=lambda *_: None)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def delayed_call(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
                return {
                    "name": "release-notes",
                    "description": "stale",
                    "body": "old",
                    "version": 1,
                }
            return {
                "name": "release-notes",
                "description": "current",
                "body": "new",
                "version": 2,
            }

        screen._run_library_service_call = delayed_call
        older = asyncio.create_task(
            screen._refresh_library_skill_detail("release-notes")
        )
        await first_started.wait()
        newer = asyncio.create_task(
            screen._refresh_library_skill_detail("release-notes")
        )
        await newer
        release_first.set()
        await older

        assert screen._library_skill_editor_state is not None
        assert screen._library_skill_editor_state.version == 2


@pytest.mark.asyncio
async def test_same_skill_older_delete_cannot_reset_a_newer_work_generation(
    tmp_path,
) -> None:
    app = _build_test_app()
    _wire_skills(app, tmp_path)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._selected_skill_name = "release-notes"
        screen._library_skills_view = "editor"
        screen._library_skill_detail_generation = 3
        screen._library_skill_editor_state = build_skill_editor_state(
            {
                "name": "release-notes",
                "description": "Release notes",
                "body": "Be exact.",
                "version": 1,
            }
        )
        delete_started = asyncio.Event()
        release_delete = asyncio.Event()

        async def delayed_delete(*_args, **_kwargs):
            delete_started.set()
            await release_delete.wait()
            return {"deleted": True}

        screen._run_library_service_call = delayed_delete
        older = asyncio.create_task(
            screen._delete_library_skill("release-notes", request_generation=3)
        )
        await delete_started.wait()
        screen._library_skill_detail_generation = 4
        release_delete.set()
        await older

        assert screen._selected_skill_name == "release-notes"
        assert screen._library_skills_view == "editor"


@pytest.mark.asyncio
async def test_same_skill_older_trust_review_cannot_patch_newer_generation(
    tmp_path,
) -> None:
    """A same-name reopen must reject the preceding Work session's review."""
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        screen.query_one("#library-skill-row-release-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-overview")
        review_started = asyncio.Event()
        release_review = asyncio.Event()

        async def delayed_review(*_args, **_kwargs):
            review_started.set()
            await release_review.wait()
            return {
                "review_id": "stale-review",
                "manifest_generation": 1,
                "current_digest": "a" * 64,
            }, True

        screen._call_library_skill_trust_service = delayed_review
        older = asyncio.create_task(screen._review_library_skill_trust())
        await review_started.wait()
        screen._library_skill_detail_generation += 1
        release_review.set()
        await older

        assert screen._library_skill_active_review is None


@pytest.mark.asyncio
async def test_skill_detail_failure_stays_scoped_and_retries_in_work_pane(
    tmp_path,
) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        row = await _wait_for_selector(
            screen, pilot, "#library-skill-row-release-notes"
        )
        assert isinstance(row, Button)
        attempts = 0

        async def run_service_call(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary read failure")
            return {
                "name": "release-notes",
                "description": "Recovered",
                "body": "Be exact.",
                "version": 2,
            }

        screen.app_instance.skills_scope_service = SimpleNamespace(
            get_skill=lambda *_: None
        )
        screen._run_library_service_call = run_service_call
        row.press()
        retry = await _wait_for_selector(screen, pilot, "#library-skill-detail-retry")

        assert isinstance(retry, Button)
        assert (
            "Couldn’t load"
            in screen.query_one("#library-skill-loading", Static).content
        )
        assert screen.query_one("#library-skills-canvas").is_mounted

        retry.press()
        await pilot.pause()
        assert attempts == 2
        await _wait_for_selector(screen, pilot, "#library-skill-mode-overview")

        assert attempts == 2
        assert screen._library_skill_editor_state is not None
        assert screen._library_skill_editor_state.version == 2


@pytest.mark.asyncio
async def test_skills_reader_f6_reaches_items_and_work_regions(tmp_path) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        row = await _wait_for_selector(
            screen, pilot, "#library-skill-row-release-notes"
        )
        assert isinstance(row, Button)
        row.press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-overview")
        items_filter = screen.query_one("#library-skills-filter", Input)
        items_filter.focus()
        await pilot.pause()

        screen.action_focus_next_workbench_pane()
        await pilot.pause()

        assert screen.focused is screen.query_one("#library-skill-mode-overview")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_size",
    ((160, 50), (120, 35), (100, 30), (80, 24)),
)
async def test_skills_reader_live_matrix_preserves_modes_work_floor_and_grips(
    tmp_path,
    terminal_size,
) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=terminal_size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        shell = screen.query_one(
            "#library-skills-reader-shell", LibraryAdaptiveReaderShell
        )
        await pilot.pause()

        screen.query_one("#library-skill-row-release-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-overview")
        for mode in ("edit", "trust", "files", "overview"):
            screen.query_one(f"#library-skill-mode-{mode}", Button).press()
            await pilot.pause()

        assert shell.work.region.width >= 48
        assert shell.library_grip.region.width == 5
        assert shell.items_grip.region.width == 5
        assert shell.library_grip.display and shell.items_grip.display
        assert screen.query_one("#library-skill-overview-region").display is True


@pytest.mark.asyncio
async def test_skills_files_mode_is_read_only_and_labels_binary_files(
    tmp_path,
) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
        supporting_files={"references/guide.md": "Read this guide."},
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        screen.query_one("#library-skill-row-release-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-files")
        screen._library_skill_editor_state = dataclasses.replace(
            screen._library_skill_editor_state,
            supporting_files=(
                SkillEditorSupportingFile(
                    name="references/guide.md", size=16, is_text=True
                ),
                SkillEditorSupportingFile(
                    name="assets/logo.png", size=8, is_text=False
                ),
            ),
        )
        screen.query_one("#library-skill-work-pane").sync_state(
            **screen._library_skill_work_pane_kwargs()
        )
        await pilot.pause()
        screen.query_one("#library-skill-mode-files", Button).press()
        region = await _wait_for_selector(screen, pilot, "#library-skill-files-region")

        copy = str(screen.query_one("#library-skill-supporting", Static).renderable)
        assert "references/guide.md" in copy
        assert "assets/logo.png" in copy
        assert "binary" in copy
        assert "Read-only in Library" in str(
            screen.query_one("#library-skill-files-read-only", Static).renderable
        )
        assert not region.query(Input)
        assert not region.query(TextArea)


@pytest.mark.asyncio
async def test_skills_trust_mode_identifies_exact_review_snapshot(tmp_path) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        screen.query_one("#library-skill-row-release-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-trust")
        screen.query_one("#library-skill-mode-trust", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-trust-region")

        digest = "a" * 64
        screen._library_skill_active_review = {
            "review_id": "review-7",
            "manifest_generation": 7,
            "current_digest": digest,
        }
        screen._render_library_skill_trust_panel()
        await pilot.pause()

        assert (
            str(
                screen.query_one(
                    "#library-skill-trust-review-identity", Static
                ).renderable
            )
            == f"Reviewed files · trust generation 7 · sha256:{digest}"
        )


@pytest.mark.asyncio
async def test_successful_skill_save_discards_the_reviewed_snapshot(tmp_path) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        screen.query_one("#library-skill-row-release-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skill-mode-trust")
        screen._library_skill_active_review = {
            "review_id": "review-7",
            "manifest_generation": 7,
            "current_digest": "a" * 64,
        }

        screen._apply_library_skill_save_success(
            {
                "name": "release-notes",
                "description": "Updated",
                "body": "Be exact.",
                "version": 2,
            },
            is_create=False,
        )
        await pilot.pause()

        assert screen._library_skill_active_review is None


@pytest.mark.asyncio
async def test_items_projection_cannot_consume_work_scroll_receipt(tmp_path) -> None:
    app = _build_test_app()
    _wire_skills(app, tmp_path)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        screen._enter_library_skill_create_editor()
        screen._library_skill_scroll_pending = True

        items_kwargs = screen._library_skills_list_canvas_kwargs()
        assert items_kwargs["scroll_to_actions"] is False
        assert screen._library_skill_scroll_pending is True

        work_kwargs = screen._library_skill_work_pane_kwargs()
        assert work_kwargs["scroll_to_actions"] is True
        assert screen._library_skill_scroll_pending is False


@pytest.mark.asyncio
async def test_skills_manual_pane_collapses_are_independent_and_expand_work(
    tmp_path,
) -> None:
    app = _build_test_app()
    _wire_skills(app, tmp_path)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        shell = screen.query_one(
            "#library-skills-reader-shell", LibraryAdaptiveReaderShell
        )
        initial_work_width = shell.work.region.width

        shell.library_grip.press()
        await pilot.pause()
        assert shell.library.display is False
        assert shell.items.display is True
        assert shell.work.region.width > initial_work_width
        library_collapsed_width = shell.work.region.width

        shell.items_grip.press()
        await pilot.pause()
        assert shell.library.display is False
        assert shell.items.display is False
        assert shell.work.region.width > library_collapsed_width


@pytest.mark.asyncio
async def test_skills_trust_posture_sync_keeps_interactive_rows_mounted(
    tmp_path,
) -> None:
    app = _build_test_app()
    local = _wire_skills(app, tmp_path)
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_reader(screen, pilot)
        canvas = screen.query_one("#library-skills-canvas", LibrarySkillsListCanvas)
        row = screen.query_one("#library-skill-row-release-notes", Button)

        screen._library_skills_trust_posture = "needs_setup"
        canvas.sync_state(**screen._library_skills_list_canvas_kwargs())
        await pilot.pause()

        assert screen.query_one("#library-skill-row-release-notes") is row
        assert screen.query_one("#library-skills-trust-header", Static)
