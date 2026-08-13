from __future__ import annotations

import asyncio
import dataclasses
import statistics
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widget import Widget

from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.UI.Library_Modules.library_prompt_browse_controller import (
    LibraryPromptBrowseController,
)
from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS,
    LibraryEntryReconcileResult,
    LibraryScreen,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_COLLECTIONS,
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_STUDY,
    LIBRARY_ROW_INGEST_EXPORT,
)
from tldw_chatbook.Widgets.Library import (
    LibraryCollectionsPanel,
    LibraryConversationsCanvas,
    LibraryExportCanvas,
    LibraryLandingCanvas,
    LibraryLandingCanvasState,
    LibraryLandingRecentItem,
    LibraryMediaCanvas,
    LibraryMediaTrashCanvas,
    LibraryMediaViewer,
    LibraryNotesCanvas,
    LibraryPromptsListCanvas,
    LibrarySkillsListCanvas,
    LibraryStudyHandoffCanvas,
)
from Tests.UI.test_library_content_hub import StaticLibraryCollectionsService
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _FakeSkillsScopeService,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


@dataclasses.dataclass(frozen=True)
class _EntryWorkerCase:
    name: str
    terminal_selector: str
    owner_type: type[Widget]
    owner_replaced: bool = False


_ENTRY_WORKER_CASES = (
    _EntryWorkerCase("prompts", "#library-prompt-row-1", LibraryPromptsListCanvas),
    _EntryWorkerCase(
        "collections", "#library-collection-select-0", LibraryCollectionsPanel
    ),
    _EntryWorkerCase("skills", "#library-skill-row-code-review", LibrarySkillsListCanvas),
    _EntryWorkerCase(
        "notes", "#library-note-body", LibraryNotesCanvas, owner_replaced=True
    ),
    _EntryWorkerCase(
        "media", "#library-media-viewer", LibraryMediaViewer, owner_replaced=True
    ),
    _EntryWorkerCase("export", "#library-export-header", LibraryExportCanvas),
    _EntryWorkerCase(
        "pending-media",
        "#library-media-viewer",
        LibraryMediaViewer,
        owner_replaced=True,
    ),
    _EntryWorkerCase(
        "pending-notes",
        "#library-note-body",
        LibraryNotesCanvas,
        owner_replaced=True,
    ),
    _EntryWorkerCase(
        "pending-conversations",
        "#library-conversations-canvas",
        LibraryConversationsCanvas,
        owner_replaced=True,
    ),
    _EntryWorkerCase(
        "pending-prompt",
        "#library-prompts-canvas",
        LibraryPromptsListCanvas,
        owner_replaced=True,
    ),
)


def _wire_entry_prompt_service(app: Any, db_path: Path) -> int:
    """Wire one real Prompt record through the production scope service."""
    prompts_db = PromptsDatabase(db_path, client_id=f"entry-{db_path.stem}")
    prompt_id, _prompt_uuid, _message = prompts_db.add_prompt(
        name="Entry prompt",
        author="Codex",
        details="Entry worker fixture",
        system_prompt="Be exact.",
        user_prompt="Summarize {text}.",
    )
    app.prompts_db = prompts_db
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts_db),
        server_service=None,
    )
    return prompt_id


def _wire_entry_export_databases(app: Any, root: Path) -> None:
    """Use file-backed databases so Export counts run on its real worker."""
    media_db = MediaDatabase(root / "entry-export-media.db", client_id="entry-export")
    media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.media_db = media_db
    app.media_reading_scope_service = MediaReadingScopeService(
        LocalMediaReadingService(media_db),
        None,
    )
    chachanotes_db = CharactersRAGDB(
        root / "entry-export-notes.db", client_id="entry-export"
    )
    chachanotes_db.add_conversation({"title": "Conversation"})
    chachanotes_db.add_note("Note", "Body")
    app.chachanotes_db = chachanotes_db


def _arrange_entry_worker_case(
    case: _EntryWorkerCase,
    *,
    app: Any,
    screen: LibraryScreen,
    prompt_id: int,
) -> None:
    """Arrange the initial route before mount, matching production ordering."""
    if case.name == "prompts":
        screen.restore_state(
            {
                "library_selected_row_id": LIBRARY_ROW_BROWSE_PROMPTS,
                "library_prompts_view": "list",
            }
        )
    elif case.name == "collections":
        screen.apply_navigation_context({"mode": "collections"})
    elif case.name == "skills":
        screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_SKILLS})
    elif case.name == "notes":
        screen.restore_state(
            {
                "library_selected_row_id": LIBRARY_ROW_BROWSE_NOTES,
                "selected_note_id": "n-1",
                "library_notes_view": "editor",
            }
        )
    elif case.name == "media":
        screen.restore_state(
            {
                "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
                "selected_media_id": "media-1",
                "library_media_view": "viewer",
            }
        )
    elif case.name == "export":
        screen.restore_state({"library_selected_row_id": LIBRARY_ROW_INGEST_EXPORT})
    elif case.name.startswith("pending-"):
        source_type = case.name.removeprefix("pending-")
        source_id = {
            "media": "media-1",
            "notes": "n-1",
            "conversations": "chat-2",
            "prompt": str(prompt_id),
        }[source_type]
        screen.apply_navigation_context(
            {
                LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: source_type,
                LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: source_id,
            }
        )


def _entry_worker_terminal(case: _EntryWorkerCase, screen: LibraryScreen) -> bool:
    """Return the real worker terminal state paired with its mounted selector."""
    selector_ready = bool(screen.query(case.terminal_selector))
    if case.name == "prompts":
        return (
            screen._library_prompt_browse_controller.result.status == "ready"
            and selector_ready
        )
    if case.name == "collections":
        return screen._library_collections_loaded and selector_ready
    if case.name == "skills":
        return screen._library_skills_trust_posture == "ready" and selector_ready
    if case.name in {"notes", "pending-notes"}:
        return screen._library_note_load_state == "loaded" and selector_ready
    if case.name in {"media", "pending-media"}:
        return screen._library_media_detail is not None and selector_ready
    if case.name == "export":
        return screen._library_export_counts is not None and selector_ready
    if case.name == "pending-conversations":
        return screen._selected_conversation_id == "chat-2" and selector_ready
    if case.name == "pending-prompt":
        return screen._library_prompt_detail is not None and selector_ready
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _ENTRY_WORKER_CASES, ids=lambda case: case.name)
async def test_automatic_entry_worker_composes_screen_once_and_routes_in_place(
    case: _EntryWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restoring a former broad completion call must fail its exact API spy."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=_two_notes(),
        media=_two_media_items(),
    )
    prompt_id = _wire_entry_prompt_service(app, tmp_path / f"{case.name}-prompts.db")
    app.library_collections_service = StaticLibraryCollectionsService(
        [
            {
                "collection_id": "collection-1",
                "name": "Launch Evidence",
                "description": "Sources for release review.",
                "item_count": 3,
                "source_authority": "local",
                "sync_status": "local-only",
                "updated_at": "2026-08-13T10:00:00Z",
            }
        ]
    )
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}],
    )
    app.local_skill_trust_service = SimpleNamespace(trust_posture=lambda: "ready")
    if case.name == "export":
        _wire_entry_export_databases(app, tmp_path)

    screen = LibraryScreen(app)
    _arrange_entry_worker_case(
        case,
        app=app,
        screen=screen,
        prompt_id=prompt_id,
    )

    started = asyncio.Event()
    release = asyncio.Event()
    thread_started = threading.Event()
    thread_release = threading.Event()
    if case.name == "prompts":
        original_load = LibraryPromptBrowseController._load

        async def gated_prompt_load(controller, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_load(controller, *args, **kwargs)

        monkeypatch.setattr(LibraryPromptBrowseController, "_load", gated_prompt_load)
    elif case.name == "collections":
        original_load = LibraryScreen._refresh_library_collections_snapshot

        async def gated_collections_load(active_screen, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_load(active_screen, *args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen,
            "_refresh_library_collections_snapshot",
            gated_collections_load,
        )
    elif case.name == "skills":
        original_load = LibraryScreen._load_library_skills_trust_posture

        async def gated_skills_load(active_screen, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_load(active_screen, *args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen,
            "_load_library_skills_trust_posture",
            gated_skills_load,
        )
    elif case.name == "notes":
        original_load = LibraryScreen._refresh_library_note_detail

        async def gated_note_load(active_screen, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_load(active_screen, *args, **kwargs)

        monkeypatch.setattr(LibraryScreen, "_refresh_library_note_detail", gated_note_load)
    elif case.name == "media":
        original_load = LibraryScreen._refresh_library_media_detail

        async def gated_media_load(active_screen, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_load(active_screen, *args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen,
            "_refresh_library_media_detail",
            gated_media_load,
        )
    elif case.name == "export":
        original_compute = LibraryScreen._compute_library_export_counts

        def gated_export_counts(*args, **kwargs):
            thread_started.set()
            assert thread_release.wait(timeout=10), "Export counts gate was not released."
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen,
            "_compute_library_export_counts",
            staticmethod(gated_export_counts),
        )
    else:
        original_load = LibraryScreen._open_pending_library_source

        async def gated_pending_open(active_screen, *args, **kwargs):
            started.set()
            await release.wait()
            return await original_load(active_screen, *args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen,
            "_open_pending_library_source",
            gated_pending_open,
        )

    compose_calls: list[LibraryScreen] = []
    refresh_recompose_calls: list[LibraryScreen] = []
    recompose_calls: list[LibraryScreen] = []
    original_compose = LibraryScreen.compose_content
    original_refresh = LibraryScreen.refresh
    original_recompose = LibraryScreen.recompose

    def counted_compose(active_screen):
        compose_calls.append(active_screen)
        yield from original_compose(active_screen)

    def recorded_refresh(active_screen, *regions, **kwargs):
        if kwargs.get("recompose"):
            refresh_recompose_calls.append(active_screen)
        return original_refresh(active_screen, *regions, **kwargs)

    async def recorded_recompose(active_screen):
        recompose_calls.append(active_screen)
        return await original_recompose(active_screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counted_compose)
    monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
    monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)

    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_library_shell(active_screen, pilot)
        if case.name == "export":
            await _wait_for_condition(
                pilot,
                thread_started.is_set,
                message="Export entry worker did not reach its counts gate.",
            )
        else:
            await _wait_for_condition(
                pilot,
                started.is_set,
                message=f"{case.name} entry worker did not reach its gate.",
            )
        await _wait_for_condition(
            pilot,
            lambda: active_screen._library_entry_canvas_owner() is not None,
            message=lambda: (
                f"{case.name} initial route owner did not mount: "
                f"{[(type(child).__name__, child.id, child.display, child.is_mounted) for child in active_screen.query_one('#library-canvas').children]}"
            ),
        )

        first_screen = active_screen
        first_rail = active_screen.query_one("#library-rail")
        first_host = active_screen.query_one("#library-canvas")
        first_owner = active_screen._library_entry_canvas_owner()
        assert first_owner is not None

        if case.name == "export":
            thread_release.set()
        else:
            release.set()
        await _wait_for_condition(
            pilot,
            lambda: _entry_worker_terminal(case, active_screen),
            message=lambda: (
                f"{case.name} did not reach its terminal state; "
                f"route={active_screen._library_entry_route_key()!r}."
            ),
        )
        await pilot.pause()

        final_owner = active_screen._library_entry_canvas_owner()
        assert _active_library_screen(host) is first_screen
        assert active_screen.query_one("#library-rail") is first_rail
        assert active_screen.query_one("#library-canvas") is first_host
        assert isinstance(final_owner, case.owner_type)
        if case.owner_replaced:
            assert final_owner is not first_owner
        else:
            assert final_owner is first_owner
        assert compose_calls.count(active_screen) == 1
        assert refresh_recompose_calls == []
        assert recompose_calls == []


async def _capture_new_conversations_route(screen: LibraryScreen, pilot):
    """Switch through the live interaction path and capture its owner/focus."""
    await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
    focus = await _wait_for_selector(screen, pilot, "#library-conversations-filter")
    owner = screen._library_entry_canvas_owner()
    focus.focus()
    await pilot.pause()
    assert owner is not None
    assert screen.focused is focus
    return owner, focus


def _assert_new_route_unchanged(
    screen: LibraryScreen, *, owner: Widget, focus: Widget
) -> None:
    """Assert stale entry completion did not mutate the successor route."""
    assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
    assert screen._library_entry_canvas_owner() is owner
    assert screen.focused is focus


@pytest.mark.asyncio
async def test_stale_prompt_token_cannot_project_after_route_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    _wire_entry_prompt_service(app, tmp_path / "stale-prompt.db")
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_PROMPTS})
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    original_load = LibraryPromptBrowseController._load

    async def gated_load(controller, *args, **kwargs):
        started.set()
        await release.wait()
        try:
            return await original_load(controller, *args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(LibraryPromptBrowseController, "_load", gated_load)
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Prompt browse did not reach its service gate.",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                active_screen._library_loaded
                and active_screen._library_snapshot_rendered_generation
                == active_screen._library_snapshot_state_generation
            ),
            message="Prompt stale-race setup did not settle its source snapshot.",
        )
        stale_result = active_screen._library_prompt_browse_controller.result
        owner, focus = await _capture_new_conversations_route(active_screen, pilot)

        release.set()
        await _wait_for_condition(
            pilot,
            finished.is_set,
            message="Stale Prompt browse did not settle after release.",
        )

        result = active_screen._sync_library_prompts_browse_result(
            stale_result, "library-prompts-sort"
        )
        assert result is LibraryEntryReconcileResult.SUPERSEDED
        _assert_new_route_unchanged(active_screen, owner=owner, focus=focus)


@pytest.mark.asyncio
async def test_stale_prompt_token_is_rejected_on_the_same_route(
    tmp_path: Path,
) -> None:
    """The controller token, not an incidental route change, owns Prompt results."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    _wire_entry_prompt_service(app, tmp_path / "same-route-stale-prompt.db")
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_PROMPTS})
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        row = await _wait_for_selector(active_screen, pilot, "#library-prompt-row-1")
        stale_result = active_screen._library_prompt_browse_controller.result
        active_screen._library_prompt_browse_controller.begin(stale_result.scope)
        owner = active_screen._library_entry_canvas_owner()
        row.focus()
        await pilot.pause()

        result = active_screen._sync_library_prompts_browse_result(
            stale_result, row.id
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert active_screen._library_entry_canvas_owner() is owner
        assert active_screen.focused is row


@pytest.mark.asyncio
async def test_stale_skills_posture_cannot_project_after_route_switch() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}]
    )
    app.local_skill_trust_service = SimpleNamespace(trust_posture=lambda: "ready")
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_SKILLS)
        await _wait_for_selector(screen, pilot, "#library-skills-canvas")
        started = threading.Event()
        release = threading.Event()

        def gated_posture() -> str:
            started.set()
            assert release.wait(timeout=10), "Skills posture gate was not released."
            return "ready"

        task = asyncio.create_task(screen._load_library_skills_trust_posture(gated_posture))
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Skills posture did not reach its service gate.",
        )
        owner, focus = await _capture_new_conversations_route(screen, pilot)

        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        _assert_new_route_unchanged(screen, owner=owner, focus=focus)


@pytest.mark.asyncio
async def test_stale_skills_generation_cannot_project_on_the_same_route() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}]
    )
    app.local_skill_trust_service = SimpleNamespace(trust_posture=lambda: "ready")
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_SKILLS})
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        focus = await _wait_for_selector(
            active_screen, pilot, "#library-skill-row-code-review"
        )
        await active_screen.workers.wait_for_complete()
        await pilot.pause()
        focus = active_screen.query_one("#library-skill-row-code-review")
        owner = active_screen._library_entry_canvas_owner()
        focus.focus()
        await pilot.pause()
        started = threading.Event()
        release = threading.Event()

        def gated_posture() -> str:
            started.set()
            assert release.wait(timeout=10), "Skills generation gate was not released."
            return "updated"

        task = asyncio.create_task(
            active_screen._load_library_skills_trust_posture(gated_posture)
        )
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Skills generation test did not reach its gate.",
        )
        active_screen._library_snapshot_state_generation += 1
        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert active_screen._library_entry_canvas_owner() is owner
        assert active_screen.focused is focus


@pytest.mark.asyncio
async def test_stale_collections_snapshot_cannot_project_after_route_switch() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService([])
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_COLLECTIONS)
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        started = threading.Event()
        release = threading.Event()

        def gated_collections():
            started.set()
            assert release.wait(timeout=10), "Collections snapshot gate was not released."
            return []

        app.library_collections_service = SimpleNamespace(
            list_collections=gated_collections
        )
        task = asyncio.create_task(screen._sync_collections_panel(refresh_snapshot=True))
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Collections snapshot did not reach its service gate.",
        )
        owner, focus = await _capture_new_conversations_route(screen, pilot)

        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        _assert_new_route_unchanged(screen, owner=owner, focus=focus)


@pytest.mark.asyncio
async def test_stale_collections_generation_cannot_project_on_the_same_route() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService([])
    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "collections"})
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        focus = await _wait_for_selector(
            active_screen, pilot, "#library-collection-name-input"
        )
        await active_screen.workers.wait_for_complete()
        await pilot.pause()
        focus = active_screen.query_one("#library-collection-name-input")
        owner = active_screen._library_entry_canvas_owner()
        focus.focus()
        await pilot.pause()
        started = threading.Event()
        release = threading.Event()

        def gated_collections():
            started.set()
            assert release.wait(timeout=10), "Collections generation gate not released."
            return []

        app.library_collections_service = SimpleNamespace(
            list_collections=gated_collections
        )
        task = asyncio.create_task(
            active_screen._sync_collections_panel(refresh_snapshot=True)
        )
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Collections generation test did not reach its gate.",
        )
        active_screen._library_snapshot_state_generation += 1
        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert active_screen._library_entry_canvas_owner() is owner
        assert active_screen.focused is focus


@pytest.mark.asyncio
async def test_replace_canvas_child_repairs_current_route_after_remove_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route switch inside remove_children must not strand an empty host."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_library_shell(active_screen, pilot)
        await active_screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(active_screen, pilot, "#library-media-canvas")
        active_screen._selected_media_id = "media-1"
        active_screen._library_media_view = "viewer"
        active_screen._library_media_detail = _two_media_items()[0]
        generation = active_screen._library_snapshot_state_generation
        route_key = active_screen._library_entry_route_key()
        canvas_host = active_screen.query_one("#library-canvas")
        original_remove = canvas_host.remove_children

        def route_switching_remove(*children):
            removal = original_remove(*children)
            active_screen._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
            active_screen._library_media_view = "list"
            return removal

        monkeypatch.setattr(canvas_host, "remove_children", route_switching_remove)
        result = await active_screen._replace_library_canvas_child(
            active_screen._build_library_media_active_child(),
            generation=generation,
            route_key=route_key,
        )
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert isinstance(
            active_screen._library_entry_canvas_owner(),
            LibraryConversationsCanvas,
        )


@pytest.mark.asyncio
async def test_replace_canvas_child_repairs_owner_after_mount_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed mount must reconstruct the current route below the shell."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_library_shell(active_screen, pilot)
        await active_screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(active_screen, pilot, "#library-media-canvas")
        active_screen._selected_media_id = "media-1"
        active_screen._library_media_view = "viewer"
        active_screen._library_media_detail = _two_media_items()[0]
        generation = active_screen._library_snapshot_state_generation
        route_key = active_screen._library_entry_route_key()
        canvas_host = active_screen.query_one("#library-canvas")
        original_mount = canvas_host.mount
        attempts = 0

        def fail_once_mount(*widgets, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected mount failure")
            return original_mount(*widgets, **kwargs)

        monkeypatch.setattr(canvas_host, "mount", fail_once_mount)
        result = await active_screen._replace_library_canvas_child(
            active_screen._build_library_media_active_child(),
            generation=generation,
            route_key=route_key,
        )
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.FAILED
        assert isinstance(active_screen._library_entry_canvas_owner(), LibraryMediaViewer)


@pytest.mark.asyncio
async def test_media_replacement_rereads_state_after_mount_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_library_shell(active_screen, pilot)
        await active_screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(active_screen, pilot, "#library-media-canvas")
        active_screen._selected_media_id = "media-1"
        active_screen._library_media_view = "viewer"
        active_screen._library_media_detail = _two_media_items()[0]
        generation = active_screen._library_snapshot_state_generation
        route_key = active_screen._library_entry_route_key()
        canvas_host = active_screen.query_one("#library-canvas")
        original_mount = canvas_host.mount

        def state_changing_mount(*widgets, **kwargs):
            mounted = original_mount(*widgets, **kwargs)
            active_screen._library_media_editing = True
            return mounted

        monkeypatch.setattr(canvas_host, "mount", state_changing_mount)
        result = await active_screen._replace_library_canvas_child(
            active_screen._build_library_media_active_child(),
            generation=generation,
            route_key=route_key,
        )
        await pilot.pause()
        viewer = active_screen.query_one("#library-media-viewer", LibraryMediaViewer)

        assert result is LibraryEntryReconcileResult.APPLIED
        assert viewer.editing is True
        assert active_screen.query("#library-media-edit-title")


@pytest.mark.asyncio
async def test_stale_media_detail_cannot_project_after_route_switch() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        screen._selected_media_id = "media-1"
        screen._library_media_view = "viewer"
        route_key = screen._library_entry_route_key()
        replaced = await screen._replace_library_canvas_child(
            screen._build_library_media_active_child(),
            generation=screen._library_snapshot_state_generation,
            route_key=route_key,
        )
        assert replaced is LibraryEntryReconcileResult.APPLIED
        started = threading.Event()
        release = threading.Event()

        def gated_media(**_kwargs):
            started.set()
            assert release.wait(timeout=10), "Media detail gate was not released."
            return _two_media_items()[0]

        app.media_reading_scope_service = SimpleNamespace(get_media_item=gated_media)
        task = asyncio.create_task(
            screen._refresh_library_media_detail("media-1", entry_origin=True)
        )
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Media detail did not reach its service gate.",
        )
        owner, focus = await _capture_new_conversations_route(screen, pilot)

        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        _assert_new_route_unchanged(screen, owner=owner, focus=focus)


@pytest.mark.asyncio
async def test_stale_media_generation_cannot_project_on_the_same_route() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    screen = LibraryScreen(app)
    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
            "selected_media_id": "media-1",
            "library_media_view": "viewer",
        }
    )
    started = threading.Event()
    release = threading.Event()

    def gated_media(**_kwargs):
        started.set()
        assert release.wait(timeout=10), "Media generation gate was not released."
        return _two_media_items()[0]

    app.media_reading_scope_service = SimpleNamespace(get_media_item=gated_media)
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Media generation test did not reach its gate.",
        )
        owner = active_screen._library_entry_canvas_owner()
        assert owner is not None
        active_screen._library_snapshot_state_generation += 1
        release.set()
        await _wait_for_condition(
            pilot,
            lambda: active_screen._library_media_detail is not None,
            message="Stale Media worker did not settle its owned state.",
        )
        await pilot.pause()

        assert active_screen._library_entry_canvas_owner() is owner
        assert not active_screen.query("#library-media-viewer")


@pytest.mark.asyncio
async def test_stale_pending_conversation_open_cannot_project_after_route_switch() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        started = threading.Event()
        release = threading.Event()

        def gated_conversation(_conversation_id, *, include_deleted=False):
            assert include_deleted is False
            started.set()
            assert release.wait(timeout=10), "Pending-open gate was not released."
            return {
                "conversation_id": "chat-pending",
                "title": "Late pending conversation",
                "message_count": 1,
            }

        app.chachanotes_db = SimpleNamespace(
            is_memory_db=False,
            get_conversation_by_id=gated_conversation,
        )
        screen._pending_library_source_open = ("conversations", "chat-pending")
        task = asyncio.create_task(screen._open_pending_library_source())
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Pending conversation open did not reach its service gate.",
        )
        owner, focus = await _capture_new_conversations_route(screen, pilot)

        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        _assert_new_route_unchanged(screen, owner=owner, focus=focus)


@pytest.mark.asyncio
async def test_pending_conversation_open_cannot_overwrite_same_route_user_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected conversation is part of an entry worker's ownership."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        started = asyncio.Event()
        release = asyncio.Event()

        async def gated_fetch(_conversation_id: str):
            started.set()
            await release.wait()
            return {
                "conversation_id": "chat-pending",
                "title": "Late pending conversation",
                "message_count": 1,
            }

        monkeypatch.setattr(
            screen,
            "_fetch_library_conversation_by_id",
            gated_fetch,
        )
        screen._selected_conversation_id = "chat-pending"
        screen._pending_library_source_open = ("conversations", "chat-pending")
        task = asyncio.create_task(screen._open_pending_library_source())
        await started.wait()

        screen._selected_conversation_id = "chat-1"
        owner = screen._library_entry_canvas_owner()
        assert isinstance(owner, LibraryConversationsCanvas)
        owner.sync_state(screen._build_library_conversations_state())
        await pilot.pause()
        focus = next(
            row
            for row in screen.query(".library-conversation-row")
            if getattr(row, "conversation_id", "") == "chat-1"
        )
        focus.focus()
        await pilot.pause()

        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._selected_conversation_id == "chat-1"
        assert screen._library_entry_canvas_owner() is owner
        assert screen.focused is focus


@pytest.mark.asyncio
async def test_pending_conversation_open_rejects_same_route_stale_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation alone supersedes a pending fetch on the retained route."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        focus = await _wait_for_selector(screen, pilot, "#library-conversations-filter")
        started = asyncio.Event()
        release = asyncio.Event()

        async def gated_fetch(_conversation_id: str):
            started.set()
            await release.wait()
            return {
                "conversation_id": "chat-pending",
                "title": "Late pending conversation",
                "message_count": 1,
            }

        monkeypatch.setattr(
            screen,
            "_fetch_library_conversation_by_id",
            gated_fetch,
        )
        screen._selected_conversation_id = "chat-pending"
        screen._pending_library_source_open = ("conversations", "chat-pending")
        owner = screen._library_entry_canvas_owner()
        assert isinstance(owner, LibraryConversationsCanvas)
        focus.focus()
        await pilot.pause()
        task = asyncio.create_task(screen._open_pending_library_source())
        await started.wait()

        screen._library_snapshot_state_generation += 1
        release.set()
        result = await task

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._selected_conversation_id == "chat-pending"
        assert screen._library_entry_canvas_owner() is owner
        assert screen.focused is focus


@pytest.mark.asyncio
async def test_replace_canvas_child_repairs_media_trash_successor_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swap superseded by Media Trash must reconstruct the Trash owner."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        screen._selected_media_id = "media-1"
        screen._library_media_view = "viewer"
        screen._library_media_detail = _two_media_items()[0]
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        canvas_host = screen.query_one("#library-canvas")
        original_remove = canvas_host.remove_children

        def trash_switching_remove(*children):
            removal = original_remove(*children)
            screen._library_media_view = "trash"
            return removal

        monkeypatch.setattr(canvas_host, "remove_children", trash_switching_remove)
        result = await screen._replace_library_canvas_child(
            screen._build_library_media_active_child(),
            generation=generation,
            route_key=route_key,
        )
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert isinstance(screen._library_entry_canvas_owner(), LibraryMediaTrashCanvas)


@pytest.mark.asyncio
async def test_canvas_owner_repair_converges_after_repeated_supersession(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two route changes during mounts must not exhaust owner repair."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        screen._selected_media_id = "media-1"
        screen._library_media_view = "viewer"
        screen._library_media_detail = _two_media_items()[0]
        canvas_host = screen.query_one("#library-canvas")
        original_mount = canvas_host.mount
        mounts = 0

        def repeatedly_superseded_mount(*widgets, **kwargs):
            nonlocal mounts
            mounted = original_mount(*widgets, **kwargs)
            mounts += 1
            if mounts == 1:
                screen._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
                screen._library_media_view = "list"
            elif mounts == 2:
                screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
            return mounted

        monkeypatch.setattr(canvas_host, "mount", repeatedly_superseded_mount)
        repaired = await screen._repair_library_entry_canvas_owner()
        await pilot.pause()

        assert repaired is True
        assert mounts == 3
        assert isinstance(screen._library_entry_canvas_owner(), LibraryMediaCanvas)


@pytest.mark.asyncio
async def test_export_counts_reject_same_route_stale_generation(tmp_path: Path) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    _wire_entry_export_databases(app, tmp_path)
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_INGEST_EXPORT})
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        scope_line = await _wait_for_selector(
            active_screen, pilot, "#library-export-scope-line"
        )
        await _wait_for_condition(
            pilot,
            lambda: active_screen._library_export_counts is not None,
            message="Initial Export counts did not settle.",
        )
        rendered_before = str(scope_line.renderable)
        generation = active_screen._library_snapshot_state_generation
        route_key = active_screen._library_entry_route_key()
        active_screen._library_snapshot_state_generation += 1

        result = active_screen._apply_library_export_counts(
            active_screen._library_export_scope,
            {"media": 99, "conversations": 99, "notes": 99, "prompts": 99},
            generation=generation,
            route_key=route_key,
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert str(scope_line.renderable) == rendered_before


def _compositor_text(screen: LibraryScreen) -> str:
    """Return only text actually painted in the current terminal frame."""
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


def _assert_widget_text_is_painted(
    screen: LibraryScreen, selector: str, expected: str
) -> None:
    """Assert a widget and its literal label are inside the rendered viewport."""
    widget = screen.query_one(selector)
    viewport = screen.region
    assert viewport.contains_region(widget.region)
    lines = [
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    ]
    painted = "\n".join(
        line[widget.region.x : widget.region.right]
        for line in lines[widget.region.y : widget.region.bottom]
    )
    assert expected in painted, (
        f"{selector} region={widget.region!r} display={widget.display!r} "
        f"painted={painted!r} frame={_compositor_text(screen)!r}"
    )


def _apply_changed_snapshot(
    screen: LibraryScreen,
    *,
    conversations: tuple[dict[str, object], ...] | None = None,
    notes: tuple[dict[str, object], ...] | None = None,
    study_decks: int | None = None,
) -> bool:
    """Apply one literal changed snapshot through the production boundary."""
    records = dict(screen._local_source_records)
    counts = dict(screen._local_source_counts)
    if conversations is not None:
        records["conversations"] = conversations
        counts["conversations"] = len(conversations)
    if notes is not None:
        records["notes"] = notes
        counts["notes"] = len(notes)
    study_counts = dict(screen._library_study_counts)
    if study_decks is not None:
        study_counts["study_decks"] = study_decks
    return screen._apply_local_source_snapshot(
        records,
        counts,
        dict(screen._local_source_total_known),
        screen._library_lookup_error,
        screen._library_lookup_recovery_state,
        study_counts,
    )


@pytest.mark.asyncio
async def test_landing_snapshot_sync_retains_actions_focus_and_updates_recents():
    """Recomposing the inline landing branch would replace all three actions."""
    app = _build_test_app()
    conversations = _two_conversations()
    _seed_conversations(app, conversations[:1])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        import_button = screen.query_one("#library-hub-action-import")
        search_button = screen.query_one("#library-hub-action-search")
        new_note_button = screen.query_one("#library-hub-action-new-note")
        search_button.focus()
        await pilot.pause()

        changed = _apply_changed_snapshot(
            screen,
            conversations=(conversations[1], conversations[0]),
        )
        await pilot.pause()
        await pilot.pause()

        assert changed is True
        assert screen.query_one("#library-landing-canvas") is landing
        assert screen.query_one("#library-hub-action-import") is import_button
        assert screen.query_one("#library-hub-action-search") is search_button
        assert screen.query_one("#library-hub-action-new-note") is new_note_button
        assert screen.focused is search_button
        assert "Conversations (2)" in str(
            screen.query_one("#library-hub-counts").renderable
        )
        recent = screen.query_one("#library-hub-recent-conversations")
        assert getattr(recent, "record_id", "") == "chat-2"


@pytest.mark.asyncio
async def test_landing_deferred_recents_converge_on_latest_state(monkeypatch):
    """Capturing recents before the deferred await would mount stale rows."""
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        first = LibraryLandingCanvasState(
            purpose=landing.state.purpose,
            counts_line="Conversations (1)",
            recent_items=(
                LibraryLandingRecentItem(
                    "conversations", "stale", "Stale row", "Conversation"
                ),
            ),
        )
        latest = LibraryLandingCanvasState(
            purpose=landing.state.purpose,
            counts_line="Conversations (1)",
            recent_items=(
                LibraryLandingRecentItem(
                    "conversations", "latest", "Latest row", "Conversation"
                ),
            ),
        )

        recents_owner = landing.query_one("#library-hub-recents")
        original_remove = recents_owner.remove_children
        removal_started = asyncio.Event()
        release_removal = asyncio.Event()

        async def delayed_remove():
            removal_started.set()
            await release_removal.wait()
            await original_remove()

        monkeypatch.setattr(recents_owner, "remove_children", delayed_remove)
        landing.state = first
        replacement = asyncio.create_task(landing._replace_recent_rows())
        await removal_started.wait()
        landing.state = latest
        release_removal.set()
        await replacement

        recents = list(landing.query(".library-hub-recent"))
        assert [getattr(recent, "record_id", "") for recent in recents] == ["latest"]


@pytest.mark.asyncio
async def test_stale_landing_deferred_sync_performs_zero_dom_mutation():
    """A route-stale deferred replacement must leave mounted rows untouched."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations()[:1])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        recents_owner = landing.query_one("#library-hub-recents")
        children_before = tuple(recents_owner.children)
        assert len(children_before) == 1

        records = dict(screen._local_source_records)
        records["conversations"] = (
            {
                "title": "Newer conversation",
                "conversation_id": "chat-new",
                "message_count": 1,
                "updated_at": "2026-08-13T10:00:00Z",
            },
        )
        screen._local_source_records = records
        generation = screen._library_snapshot_state_generation + 1
        route_key = screen._library_entry_route_key()
        screen._library_snapshot_state_generation = generation
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)

        await screen._reconcile_library_entry_state(generation, route_key)
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        await pilot.pause()
        await pilot.pause()

        assert tuple(recents_owner.children) == children_before
        assert children_before[0].parent is recents_owner


@pytest.mark.asyncio
async def test_study_handoff_snapshot_sync_retains_open_action_and_paints_readiness():
    """A source/readiness change must patch the mounted handoff owner in place."""
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_CREATE_STUDY)
        await _wait_for_selector(screen, pilot, "#library-study-handoff-canvas")
        handoff = screen.query_one(
            "#library-study-handoff-canvas", LibraryStudyHandoffCanvas
        )
        open_button = screen.query_one("#library-open-study")
        assert "Import sources or create notes first" in _compositor_text(screen)

        changed = _apply_changed_snapshot(
            screen,
            notes=(
                {
                    "id": "note-1",
                    "title": "Retained source",
                    "content": "Body",
                    "last_modified": "2026-08-13T10:00:00Z",
                },
            ),
            study_decks=2,
        )
        await pilot.pause()
        await pilot.pause()

        painted = _compositor_text(screen)
        assert changed is True
        assert screen.query_one("#library-study-handoff-canvas") is handoff
        assert screen.query_one("#library-open-study") is open_button
        assert "Source snapshot is ready." in painted
        assert "Study decks (2)" in painted
        assert "Retained source" in painted


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (170, 48)])
@pytest.mark.parametrize("surface", ["landing", "handoff"])
async def test_retained_entry_actions_paint_before_and_after_sync(size, surface):
    """Existence is insufficient when compact geometry clips an entry action."""
    app = _build_test_app()
    _seed_conversations(app, [])
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        if size[0] == 60 and surface == "landing":
            # Compact Library is a one-pane presentation. Expose its mounted
            # content stage so the render oracle measures the owner rather than
            # correctly-hidden children behind the entry rail.
            screen._library_notes_stage = "notes"
            screen._set_library_rail_collapsed(True)
            await pilot.pause()
        if surface == "handoff":
            await screen._select_library_rail_row(LIBRARY_ROW_CREATE_STUDY)
            await _wait_for_selector(screen, pilot, "#library-open-study")
            if size[0] == 60:
                screen._set_library_rail_collapsed(True)
                await pilot.pause()
            checks = (("#library-open-study", "Continue in Study"),)
        else:
            checks = (
                ("#library-hub-action-import", "Import…"),
                ("#library-hub-action-search", "Search"),
                ("#library-hub-action-new-note", "New note"),
            )
        for selector, expected in checks:
            _assert_widget_text_is_painted(screen, selector, expected)

        _apply_changed_snapshot(
            screen,
            notes=(
                {
                    "id": "note-geometry",
                    "title": "Geometry source",
                    "content": "Body",
                    "last_modified": "2026-08-13T10:00:00Z",
                },
            ),
            study_decks=2,
        )
        await pilot.pause()
        await pilot.pause()

        for selector, expected in checks:
            _assert_widget_text_is_painted(screen, selector, expected)


@pytest.mark.asyncio
async def test_warm_repeat_visit_composes_once_before_fresh_reconcile(monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    calls: list[LibraryScreen] = []
    original = LibraryScreen.compose_content

    def counted_compose(screen):
        calls.append(screen)
        yield from original(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counted_compose)
    samples: list[float] = []
    revisits: list[LibraryScreen] = []
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        first = _active_library_screen(host)
        await _wait_for_library_shell(first, pilot)
        await host.pop_screen()
        await pilot.pause()
        for _ in range(5):
            revisit = LibraryScreen(app)
            revisits.append(revisit)
            started = time.perf_counter()
            await host.push_screen(revisit)
            await _wait_for_library_shell(revisit, pilot)
            samples.append((time.perf_counter() - started) * 1000)
            await host.pop_screen()
            await pilot.pause()
    print(
        f"warm_visit_median_ms={statistics.median(samples):.3f} "
        f"min_ms={min(samples):.3f} max_ms={max(samples):.3f} n={len(samples)}"
    )
    print(f"warm_visit_compose_counts={[calls.count(revisit) for revisit in revisits]}")
    assert all(calls.count(revisit) == 1 for revisit in revisits)


@pytest.mark.asyncio
async def test_library_source_snapshot_changed_reconciles_conversations_below_screen(
    monkeypatch,
):
    """Restoring either whole-screen refresh would replace captured owners."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        rail = screen.query_one("#library-rail")
        canvas_host = screen.query_one("#library-canvas")
        canvas = screen.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        refresh_calls: list[bool] = []
        recompose_calls: list[LibraryScreen] = []
        original_refresh = LibraryScreen.refresh
        original_recompose = LibraryScreen.recompose

        def recorded_refresh(active_screen, *regions, **kwargs):
            refresh_calls.append(bool(kwargs.get("recompose")))
            return original_refresh(active_screen, *regions, **kwargs)

        async def recorded_recompose(active_screen):
            recompose_calls.append(active_screen)
            return await original_recompose(active_screen)

        monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
        monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
        records = dict(screen._local_source_records)
        records["conversations"] = (
            *records["conversations"],
            {
                "title": "Incident review",
                "conversation_id": "chat-3",
                "message_count": 5,
                "updated_at": "2026-06-03T12:00:00Z",
            },
        )
        counts = dict(screen._local_source_counts)
        counts["conversations"] = 3

        changed = screen._apply_local_source_snapshot(
            records,
            counts,
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert changed is True
        assert _active_library_screen(host) is screen
        assert screen.query_one("#library-rail") is rail
        assert screen.query_one("#library-canvas") is canvas_host
        assert (
            screen.query_one(
                "#library-conversations-canvas", LibraryConversationsCanvas
            )
            is canvas
        )
        assert "Conversations (3)" in str(
            screen.query_one("#library-conversations-title").renderable
        )
        assert "(3)" in str(screen.query_one("#library-row-browse-conversations").label)
        assert True not in refresh_calls
        assert recompose_calls == []


@pytest.mark.asyncio
async def test_library_source_snapshot_changed_retains_conversation_row_focus():
    """Dropping the Conversations follow-up moves focus outside its canvas."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await screen.workers.wait_for_complete()
        row = screen.query_one("#library-conversation-row-0")
        row.focus()
        await pilot.pause()
        assert getattr(screen.focused, "conversation_id", None) == "chat-2"

        records = dict(screen._local_source_records)
        records["conversations"] = (
            *records["conversations"],
            {
                "title": "Incident review",
                "conversation_id": "chat-3",
                "message_count": 5,
                "updated_at": "2026-06-03T12:00:00Z",
            },
        )
        counts = dict(screen._local_source_counts)
        counts["conversations"] = 3

        screen._apply_local_source_snapshot(
            records,
            counts,
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        focused = screen.focused
        assert getattr(focused, "conversation_id", None) == "chat-2"
        assert focused is not None and focused.disabled is False
        assert screen.query_one("#library-conversations-canvas") in (
            focused.ancestors_with_self
        )


@pytest.mark.asyncio
async def test_library_source_snapshot_equal_clean_refreshes_cache_without_dom_work(
    monkeypatch,
):
    """Removing the equality gate would call canvas or screen recomposition."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        canvas = screen.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        sync_calls: list[None] = []
        refresh_calls: list[bool] = []
        recompose_calls: list[LibraryScreen] = []
        original_sync_state = canvas.sync_state
        original_refresh = LibraryScreen.refresh
        original_recompose = LibraryScreen.recompose

        def recorded_sync_state(*args, **kwargs):
            sync_calls.append(None)
            return original_sync_state(*args, **kwargs)

        def recorded_refresh(active_screen, *regions, **kwargs):
            refresh_calls.append(bool(kwargs.get("recompose")))
            return original_refresh(active_screen, *regions, **kwargs)

        async def recorded_recompose(active_screen):
            recompose_calls.append(active_screen)
            return await original_recompose(active_screen)

        async def equal_snapshot():
            return (
                dict(screen._local_source_records),
                dict(screen._local_source_counts),
                dict(screen._local_source_total_known),
                screen._library_lookup_error,
                screen._library_lookup_recovery_state,
                dict(screen._library_study_counts),
            )

        monkeypatch.setattr(canvas, "sync_state", recorded_sync_state)
        monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
        monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
        monkeypatch.setattr(screen, "_list_local_source_snapshot", equal_snapshot)
        previous_stamp = time.monotonic() - 1.0
        app._library_source_snapshot_cache_stamp = previous_stamp

        screen._refresh_local_source_snapshot()
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert app._library_source_snapshot_cache_stamp > previous_stamp
        assert screen._library_snapshot_rendered_generation == (
            screen._library_snapshot_state_generation
        )
        assert screen._library_entry_reconcile_dirty is False
        assert sync_calls == []
        assert True not in refresh_calls
        assert recompose_calls == []


@pytest.mark.asyncio
async def test_library_source_snapshot_equal_dirty_repairs_with_targeted_sync(
    monkeypatch,
):
    """Skipping dirty equal state would leave the mounted canvas stale."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        canvas = screen.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        sync_calls: list[None] = []
        refresh_calls: list[bool] = []
        recompose_calls: list[LibraryScreen] = []
        original_sync_state = canvas.sync_state
        original_refresh = LibraryScreen.refresh
        original_recompose = LibraryScreen.recompose

        def recorded_sync_state(*args, **kwargs):
            sync_calls.append(None)
            return original_sync_state(*args, **kwargs)

        def recorded_refresh(active_screen, *regions, **kwargs):
            refresh_calls.append(bool(kwargs.get("recompose")))
            return original_refresh(active_screen, *regions, **kwargs)

        async def recorded_recompose(active_screen):
            recompose_calls.append(active_screen)
            return await original_recompose(active_screen)

        monkeypatch.setattr(canvas, "sync_state", recorded_sync_state)
        monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
        monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
        generation = screen._library_snapshot_state_generation
        screen._library_entry_reconcile_dirty = True

        changed = screen._apply_local_source_snapshot(
            dict(screen._local_source_records),
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert changed is False
        assert sync_calls == [None]
        assert screen._library_snapshot_state_generation == generation
        assert screen._library_snapshot_rendered_generation == generation
        assert screen._library_entry_reconcile_dirty is False
        assert True not in refresh_calls
        assert recompose_calls == []


@pytest.mark.asyncio
async def test_library_source_snapshot_stale_route_clears_retry_markers():
    """Leaving the retry generation armed would skip the new route's retry."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        generation = screen._library_snapshot_state_generation
        stale_route = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, stale_route)
        screen._library_entry_reconcile_retry_generation = generation
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA

        result = await screen._reconcile_library_entry_state(
            generation, stale_route
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None


@pytest.mark.asyncio
async def test_library_source_snapshot_missing_skills_retries_then_equal_can_retry(
    monkeypatch,
):
    """Falling through a missing Skills selector would falsely mark it rendered."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
        screen._library_skills_view = "list"
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        queued: list[tuple[object, tuple[object, ...]]] = []

        def capture_call_later(callback, *args):
            queued.append((callback, args))

        monkeypatch.setattr(screen, "call_later", capture_call_later)

        first = await screen._reconcile_library_entry_state(generation, route_key)

        assert first is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert screen._library_entry_reconcile_retry_generation == generation
        assert len(queued) == 1

        callback, args = queued.pop()
        second = await callback(*args)

        assert second is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None

        changed = screen._apply_local_source_snapshot(
            dict(screen._local_source_records),
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )

        assert changed is False
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert len(queued) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_site", ["rail", "header"])
async def test_library_source_snapshot_shell_exception_releases_retry_markers(
    monkeypatch, failure_site
):
    """An owned shell failure must not deduplicate the next equal repair."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        screen._library_entry_reconcile_retry_generation = generation
        queued: list[tuple[object, tuple[object, ...]]] = []

        def capture_call_later(callback, *args):
            queued.append((callback, args))

        def fail_shell_sync(*args, **kwargs):
            raise RuntimeError(f"forced {failure_site} sync failure")

        monkeypatch.setattr(screen, "call_later", capture_call_later)
        if failure_site == "rail":
            rail = screen.query_one("#library-rail")
            monkeypatch.setattr(rail, "sync_state", fail_shell_sync)
        else:
            header = screen.query_one("#library-header-line")
            header.update("stale header")
            monkeypatch.setattr(header, "update", fail_shell_sync)

        result = await screen._reconcile_library_entry_state(
            generation, route_key
        )

        assert result is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None

        changed = screen._apply_local_source_snapshot(
            dict(screen._local_source_records),
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )

        assert changed is False
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert len(queued) == 1


def test_constructor_seeds_cached_snapshot_before_restore_state_wins_selection():
    """Changing pre-compose cache seeding to mount-time seeding would leave
    this fresh screen unloaded before its first composition.
    """
    app = _build_test_app()
    app._library_source_snapshot_cache = (
        {
            "notes": ({"id": "n1"},),
            "media": ({"id": "m1"},),
            "conversations": ({"id": "c1"},),
            "prompts": (None, ()),
            "skills": (None, {"available_skills": [], "blocked_skills": []}),
        },
        {"notes": 1, "media": 1, "conversations": 1},
        {"notes": True, "media": True, "conversations": True},
        None,
        None,
        {"study_decks": None, "flashcards_due": None, "quizzes": None},
    )
    app._library_source_snapshot_cache_stamp = time.monotonic()

    screen = LibraryScreen(app)

    assert screen._library_loaded is True
    assert screen._local_source_counts == {
        "notes": 1,
        "media": 1,
        "conversations": 1,
    }

    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
            "selected_media_id": "m1",
            "library_media_view": "viewer",
        }
    )

    assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
    assert screen._selected_media_id == "m1"
    assert screen._library_media_view == "viewer"


def test_cache_seed_rejects_future_and_ttl_boundary_stamps():
    """Changing either cache-age guard would accept a future or expired seed."""
    app = _build_test_app()
    app._library_source_snapshot_cache = (
        {
            "notes": (),
            "media": (),
            "conversations": (),
            "prompts": (None, ()),
            "skills": (None, {"available_skills": [], "blocked_skills": []}),
        },
        {"notes": 0, "media": 0, "conversations": 0},
        {"notes": True, "media": True, "conversations": True},
        None,
        None,
        {"study_decks": None, "flashcards_due": None, "quizzes": None},
    )
    stamp = 100.0
    app._library_source_snapshot_cache_stamp = stamp
    screen = LibraryScreen(app)

    assert screen._seed_local_source_snapshot_from_cache(now=stamp - 0.1) is False
    assert (
        screen._seed_local_source_snapshot_from_cache(
            now=stamp + LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS
        )
        is False
    )
    assert screen._library_loaded is False

    assert (
        screen._seed_local_source_snapshot_from_cache(
            now=stamp + LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS - 0.1
        )
        is True
    )
    assert screen._library_loaded is True
