from __future__ import annotations

import asyncio
import dataclasses
import math
import statistics
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Input, Static
from textual.widgets._input import Selection

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
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
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
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.Widgets.Library import (
    LibraryCollectionsPanel,
    LibraryConversationsCanvas,
    LibraryExportCanvas,
    LibraryLandingCanvas,
    LibraryLandingCanvasState,
    LibraryLandingAttentionAction,
    LibraryLandingContinueAction,
    LibraryLandingRecentItem,
    LibraryMediaCanvas,
    LibraryMediaTrashCanvas,
    LibraryMediaViewer,
    LibraryNotesCanvas,
    LibraryPromptsListCanvas,
    LibrarySkillsListCanvas,
    LibraryStudyHandoffCanvas,
)
from Tests.UI.background_signals import (
    await_background_task,
    wait_for_background_signal,
    wait_for_signal,
)
from Tests.UI.test_library_content_hub import StaticLibraryCollectionsService
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _FakeSkillsScopeService,
    _active_library_screen,
    _build_test_app as _build_library_test_app,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _build_test_app():
    """Build the legacy-profile app assumed by retained-entry owner tests."""
    app = _build_library_test_app()
    app.library_new_profile_admission = False
    return app


@dataclasses.dataclass(frozen=True)
class _EntryWorkerCase:
    name: str
    terminal_selector: str
    owner_type: type[Widget]
    owner_replaced: bool | None = False


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
        # The initial paged snapshot may either patch the pre-mounted route
        # owner or replace it before the pending point lookup settles.
        owner_replaced=None,
    ),
    _EntryWorkerCase(
        "pending-prompt",
        "#library-prompts-canvas",
        LibraryPromptsListCanvas,
        owner_replaced=True,
    ),
)


class _LandingCanvasHarness(App):
    """Mount one retained landing owner without the surrounding screen."""

    def __init__(self, state: LibraryLandingCanvasState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield LibraryLandingCanvas(self.state, id="library-landing-canvas")


def _landing_state(
    lifecycle: LibraryLifecycle,
    *,
    status: str = "",
    show_retry: bool = False,
    show_explore: bool = False,
    continue_action: LibraryLandingContinueAction | None = None,
    attention_action: LibraryLandingAttentionAction | None = None,
    recent_items: tuple[LibraryLandingRecentItem, ...] = (),
) -> LibraryLandingCanvasState:
    """Build one explicit lifecycle presentation state for retained-owner tests."""
    return LibraryLandingCanvasState(
        purpose="Add something useful, then use it in Console or Study.",
        counts_line="Notes (3) · Media (2) · Conversations (1)",
        continue_action=continue_action,
        attention_action=attention_action,
        recent_items=recent_items,
        lifecycle=lifecycle,
        lifecycle_status=status,
        show_retry=show_retry,
        show_explore=show_explore,
    )


@pytest.mark.asyncio
async def test_library_returning_landing_orders_continue_attention_from_library_and_quick_actions():
    continue_action = LibraryLandingContinueAction(
        label="Media · <audio> · page 2 · 音声 🧪 e\u0301 \u2067مرحبا\u2069",
        row_id=LIBRARY_ROW_BROWSE_MEDIA,
        adjustment="Item views resume at the source list.",
    )
    attention_action = LibraryLandingAttentionAction(
        message="Media list may be out of date.",
        action_label="Retry",
        action_kind="media-retry",
    )
    recent = LibraryLandingRecentItem(
        "notes",
        "note-1",
        "<Reading> list · 読書 🗂️ e\u0301 \u2067مرحبا\u2069",
        "Note",
    )
    app = _LandingCanvasHarness(
        _landing_state(
            LibraryLifecycle.GRADUATED,
            continue_action=continue_action,
            attention_action=attention_action,
            recent_items=(recent,),
        )
    )

    async with app.run_test():
        landing = app.query_one("#library-landing-canvas", LibraryLandingCanvas)
        ordered_ids = [
            widget.id for widget in landing.walk_children() if widget.id is not None
        ]
        assert ordered_ids.index("library-hub-continue-heading") < ordered_ids.index(
            "library-hub-continue"
        )
        assert ordered_ids.index("library-hub-continue") < ordered_ids.index(
            "library-hub-attention-heading"
        )
        assert ordered_ids.index("library-hub-attention-action") < ordered_ids.index(
            "library-hub-from-library-heading"
        )
        assert ordered_ids.index("library-hub-recent-notes") < ordered_ids.index(
            "library-hub-quick-actions-heading"
        )
        quick_ids = [
            widget.id
            for widget in app.query(".library-hub-action")
        ]
        assert quick_ids == [
            "library-hub-action-import",
            "library-hub-action-new-note",
            "library-hub-action-search",
        ]
        continue_button = app.query_one("#library-hub-continue", Button)
        attention_button = app.query_one("#library-hub-attention-action", Button)
        assert str(continue_button.label) == continue_action.label
        assert getattr(continue_button, "row_id", "") == LIBRARY_ROW_BROWSE_MEDIA
        assert getattr(attention_button, "action_kind", "") == "media-retry"
        assert str(
            app.query_one("#library-hub-continue-adjustment", Static).renderable
        ) == "Item views resume at the source list."
        assert str(app.query_one("#library-hub-attention-copy", Static).renderable) == (
            "Media list may be out of date."
        )
        assert "<Reading> list" in str(
            app.query_one("#library-hub-recent-notes", Button).label
        )


@pytest.mark.asyncio
async def test_library_returning_landing_omits_absent_optional_sections():
    app = _LandingCanvasHarness(_landing_state(LibraryLifecycle.EXPANDED))

    async with app.run_test():
        assert not app.query("#library-hub-continue-heading")
        assert not app.query("#library-hub-continue")
        assert not app.query("#library-hub-attention-heading")
        assert not app.query("#library-hub-attention-action")
        assert not app.query("#library-hub-from-library-heading")
        assert not app.query("#library-hub-recents")
        assert len(app.query("#library-hub-quick-actions-heading")) == 1
        assert len(app.query("#library-hub-action-import")) == 1
        assert len(app.query("#library-hub-action-new-note")) == 1
        assert len(app.query("#library-hub-action-search")) == 1


@pytest.mark.asyncio
async def test_library_returning_landing_sync_retains_actions_focus_and_updates_copy():
    initial_continue = LibraryLandingContinueAction(
        label="Media · type: audio · page 2",
        row_id=LIBRARY_ROW_BROWSE_MEDIA,
        adjustment="Item views resume at the source list.",
    )
    initial_attention = LibraryLandingAttentionAction(
        message="Media list may be out of date.",
        action_label="Retry",
        action_kind="media-retry",
    )
    app = _LandingCanvasHarness(
        _landing_state(
            LibraryLifecycle.EXPANDED,
            continue_action=initial_continue,
            attention_action=initial_attention,
            recent_items=(
                LibraryLandingRecentItem("notes", "note-1", "Reading list", "Note"),
            ),
        )
    )

    async with app.run_test() as pilot:
        landing = app.query_one("#library-landing-canvas", LibraryLandingCanvas)
        continue_button = app.query_one("#library-hub-continue", Button)
        attention_button = app.query_one("#library-hub-attention-action", Button)
        import_button = app.query_one("#library-hub-action-import", Button)
        continue_button.focus()
        await pilot.pause()

        landing.sync_state(
            _landing_state(
                LibraryLifecycle.EXPANDED,
                continue_action=dataclasses.replace(
                    initial_continue,
                    label="Media · type: document · page 3",
                    adjustment="",
                ),
                attention_action=dataclasses.replace(
                    initial_attention,
                    message="Prompt results may be out of date.",
                    action_kind="prompts-retry",
                ),
                recent_items=(
                    LibraryLandingRecentItem(
                        "notes", "note-2", "Updated reading list", "Note"
                    ),
                ),
            )
        )
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#library-hub-continue", Button) is continue_button
        assert app.query_one("#library-hub-attention-action", Button) is attention_button
        assert app.query_one("#library-hub-action-import", Button) is import_button
        assert app.focused is continue_button
        assert str(continue_button.label) == "Media · type: document · page 3"
        assert not app.query_one("#library-hub-continue-adjustment", Static).display
        assert str(app.query_one("#library-hub-attention-copy", Static).renderable) == (
            "Prompt results may be out of date."
        )
        assert getattr(attention_button, "action_kind", "") == "prompts-retry"
        assert getattr(
            app.query_one("#library-hub-recent-notes", Button), "record_id", ""
        ) == "note-2"


@pytest.mark.asyncio
async def test_library_landing_syncs_unknown_to_starter_without_duplicate_actions():
    app = _LandingCanvasHarness(
        _landing_state(
            LibraryLifecycle.UNKNOWN,
            status="Checking existing Library content…",
        )
    )

    async with app.run_test() as pilot:
        landing = app.query_one("#library-landing-canvas", LibraryLandingCanvas)
        import_action = app.query_one("#library-hub-action-import", Button)
        note_action = app.query_one("#library-hub-action-new-note", Button)
        assert not app.query("#library-hub-counts")
        assert not app.query("#library-hub-action-search")
        assert not app.query(".library-hub-recent")
        assert str(
            app.query_one("#library-hub-lifecycle-status", Static).renderable
        ) == "Checking existing Library content…"
        assert str(
            app.query_one("#library-canvas-landing", Static).renderable
        ) == "Add something useful, then use it in Console or Study."

        landing.sync_state(_landing_state(LibraryLifecycle.STARTER))
        await pilot.pause()

        assert app.query_one("#library-hub-action-import", Button) is import_action
        assert app.query_one("#library-hub-action-new-note", Button) is note_action
        assert len(app.query("#library-hub-action-import")) == 1
        assert len(app.query("#library-hub-action-new-note")) == 1
        assert "1 Add · 2 Find · 3 Use" in str(
            app.query_one("#library-hub-orientation", Static).renderable
        )


@pytest.mark.asyncio
async def test_library_landing_syncs_starter_to_expanded_without_stale_recents():
    stale = LibraryLandingRecentItem("notes", "stale", "Stale note", "Note")
    app = _LandingCanvasHarness(
        _landing_state(LibraryLifecycle.STARTER, recent_items=(stale,))
    )

    async with app.run_test() as pilot:
        landing = app.query_one("#library-landing-canvas", LibraryLandingCanvas)
        assert not app.query(".library-hub-recent")

        landing.sync_state(_landing_state(LibraryLifecycle.EXPANDED))
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#library-hub-counts", Static)
        assert app.query_one("#library-hub-action-search", Button)
        assert not app.query("#library-hub-orientation")
        assert not app.query(".library-hub-recent")


@pytest.mark.asyncio
async def test_library_landing_partial_failure_shows_one_retry():
    app = _LandingCanvasHarness(
        _landing_state(
            LibraryLifecycle.UNKNOWN,
            status="Some Library sources are unavailable.",
            show_retry=True,
        )
    )

    async with app.run_test():
        retry = app.query("#library-hub-retry-evidence")
        assert len(retry) == 1
        assert str(retry.first().label) == "Retry source check"
        assert str(app.query_one("#library-hub-lifecycle-status", Static).renderable) == (
            "Some Library sources are unavailable."
        )
        assert not app.query("#library-hub-counts")
        assert not app.query("#library-hub-action-search")


@pytest.mark.asyncio
async def test_library_landing_does_not_duplicate_screen_persistence_warning():
    app = _LandingCanvasHarness(_landing_state(LibraryLifecycle.STARTER))

    async with app.run_test():
        assert not app.query("#library-hub-persistence-warning")
        assert app.query_one("#library-hub-action-import", Button).disabled is False
        assert (
            app.query_one("#library-hub-action-new-note", Button).disabled is False
        )


@pytest.mark.asyncio
async def test_library_landing_composes_explore_only_when_rail_action_is_absent():
    app = _LandingCanvasHarness(
        _landing_state(LibraryLifecycle.STARTER, show_explore=True)
    )

    async with app.run_test():
        assert len(app.query("#library-hub-explore-all")) == 1


@pytest.mark.asyncio
async def test_library_landing_late_sync_cannot_replace_a_new_route_owner(
):
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        replacement = screen.query_one("#library-media-canvas", LibraryMediaCanvas)

        landing.sync_state(_landing_state(LibraryLifecycle.STARTER))
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one("#library-media-canvas") is replacement
        assert not screen.query("#library-landing-canvas")


@pytest.mark.asyncio
async def test_library_graduation_announcement_survives_reconcile_and_same_route_replace():
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_for_selector(screen, pilot, "#library-notes-canvas")

        screen._set_library_lifecycle(LibraryLifecycle.GRADUATED)
        screen._sync_library_rail_lifecycle_presentation()
        await pilot.pause()
        focus = await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        focus.focus()
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.id == focus.id

        assert "Library tools are now available." in str(
            screen.query_one("#library-lifecycle-status", Static).renderable
        )

        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        reconciled = await screen._reconcile_library_entry_state(
            generation, route_key
        )
        await pilot.pause()

        assert reconciled is LibraryEntryReconcileResult.APPLIED
        assert "Library tools are now available." in str(
            screen.query_one("#library-lifecycle-status", Static).renderable
        )
        assert screen.focused is not None
        assert screen.focused.id == focus.id

        replacement = screen._build_library_entry_active_child()
        assert replacement is not None
        child_replaced = await screen._replace_library_canvas_child(
            replacement,
            generation=generation,
            route_key=route_key,
        )
        await pilot.pause()

        assert child_replaced is LibraryEntryReconcileResult.APPLIED
        assert "Library tools are now available." in str(
            screen.query_one("#library-lifecycle-status", Static).renderable
        )
        assert screen.focused is not None
        assert screen.focused.id == focus.id

        shell = library_screen_module.build_library_shell_state(
            screen._build_library_shell_input(),
            selected_row_id=screen._library_selected_row_id,
        )
        replaced = await screen._replace_library_browse_canvas(shell)
        await pilot.pause()

        assert replaced is True
        assert "Library tools are now available." in str(
            screen.query_one("#library-lifecycle-status", Static).renderable
        )
        assert screen.focused is not None
        assert screen.focused.id == focus.id


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("reconcile", "replace"))
async def test_library_notes_recompose_does_not_steal_newer_focus(
    monkeypatch, operation
):
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_for_selector(screen, pilot, "#library-notes-canvas")

        screen._set_library_lifecycle(LibraryLifecycle.GRADUATED)
        screen._sync_library_rail_lifecycle_presentation()
        await pilot.pause()
        row = await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        row.focus()
        await pilot.pause()

        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        original_recompose = LibraryNotesCanvas.recompose
        recompose_started = asyncio.Event()
        release_recompose = asyncio.Event()

        async def gated_recompose(canvas):
            recompose_started.set()
            await release_recompose.wait()
            await original_recompose(canvas)

        try:
            monkeypatch.setattr(LibraryNotesCanvas, "recompose", gated_recompose)
            if operation == "reconcile":
                screen._library_entry_reconcile_dirty = True
                screen._library_entry_reconcile_pending = (generation, route_key)
                operation_task = asyncio.create_task(
                    screen._reconcile_library_entry_state(generation, route_key)
                )
            else:
                replacement = screen._build_library_entry_active_child()
                assert isinstance(replacement, LibraryNotesCanvas)
                operation_task = asyncio.create_task(
                    screen._replace_library_canvas_child(
                        replacement,
                        generation=generation,
                        route_key=route_key,
                    )
                )
            await _wait_for_condition(
                pilot,
                recompose_started.is_set,
                message="Notes canvas did not start its recompose",
            )
            newer_target = screen.query_one("#library-rail-collapse", Button)
            newer_target.focus()
            release_recompose.set()

            assert await operation_task is LibraryEntryReconcileResult.APPLIED
            await pilot.pause()
            await pilot.pause()

            assert newer_target.has_focus
            assert "Library tools are now available." in str(
                screen.query_one("#library-lifecycle-status", Static).renderable
            )
        finally:
            release_recompose.set()


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
@pytest.mark.parametrize("size", [(60, 20), (170, 48)])
@pytest.mark.parametrize("case", _ENTRY_WORKER_CASES, ids=lambda case: case.name)
async def test_automatic_entry_worker_composes_screen_once_and_routes_in_place(
    case: _EntryWorkerCase,
    size: tuple[int, int],
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
    async with host.run_test(size=size) as pilot:
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
        if size == (60, 20):
            active_screen._library_notes_stage = "notes"
            active_screen._set_library_rail_collapsed(True)
            await pilot.pause()
            await pilot.pause()

        if case.name == "export":
            thread_release.set()
        else:
            release.set()
        painted_copy = {
            "prompts": "Entry prompt",
            "collections": "Launch Evidence",
            "skills": "code-review",
            "notes": "Q3 retro",
            "media": "Interview Recording",
            "export": "Export bundle (.zip)",
            "pending-media": "Interview Recording",
            "pending-notes": "Q3 retro",
            "pending-conversations": "Design review notes",
            "pending-prompt": "Entry prompt",
        }[case.name]
        await _wait_for_condition(
            pilot,
            lambda: _entry_worker_terminal(case, active_screen),
            message=lambda: (
                f"{case.name} did not reach its terminal state; "
                f"route={active_screen._library_entry_route_key()!r}."
            ),
        )
        await _wait_for_condition(
            pilot,
            lambda: painted_copy in _compositor_text(active_screen),
            message=lambda: (
                f"{case.name} terminal copy was not painted: "
                f"{_compositor_text(active_screen)!r}"
            ),
        )

        final_owner = active_screen._library_entry_canvas_owner()
        assert _active_library_screen(host) is first_screen
        assert active_screen.query_one("#library-rail") is first_rail
        assert active_screen.query_one("#library-canvas") is first_host
        assert isinstance(final_owner, case.owner_type)
        if case.owner_replaced is True:
            assert final_owner is not first_owner
        elif case.owner_replaced is False:
            assert final_owner is first_owner
        assert compose_calls.count(active_screen) == 1
        assert refresh_recompose_calls == []
        assert recompose_calls == []
        compositor = _compositor_text(active_screen)
        exported_svg = _exported_svg_text(host)
        assert painted_copy in compositor
        assert painted_copy in exported_svg
        print(
            "task5_uat_entry "
            f"size={size} route={case.name} copy={painted_copy!r} "
            f"screen={id(active_screen)} rail={id(first_rail)} host={id(first_host)} "
            f"owner_before={id(first_owner)} owner_after={id(final_owner)} "
            f"compose={compose_calls.count(active_screen)} "
            f"refresh_recompose={len(refresh_recompose_calls)} "
            f"recompose={len(recompose_calls)}"
        )


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
async def test_unmounted_prompt_screen_cannot_apply_into_a_fresh_visit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    _wire_entry_prompt_service(app, tmp_path / "unmounted-prompt.db")
    old_screen = LibraryScreen(app)
    old_screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_PROMPTS})
    old_controller = old_screen._library_prompt_browse_controller
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    original_load = LibraryPromptBrowseController._load

    async def gated_old_load(controller, *args, **kwargs):
        if controller is not old_controller:
            return await original_load(controller, *args, **kwargs)
        started.set()
        try:
            await release.wait()
            return await original_load(controller, *args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(LibraryPromptBrowseController, "_load", gated_old_load)
    host = LibraryHarness(app, screen=old_screen)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _wait_for_condition(
            pilot,
            started.is_set,
            message="Old Prompt screen did not reach its request gate.",
        )
        await host.pop_screen()
        fresh_screen = LibraryScreen(app)
        fresh_screen.restore_state(
            {"library_selected_row_id": LIBRARY_ROW_BROWSE_PROMPTS}
        )
        await host.push_screen(fresh_screen)
        await _wait_for_selector(fresh_screen, pilot, "#library-prompt-row-1")
        fresh_result = fresh_screen._library_prompt_browse_controller.applied_result

        release.set()
        await _wait_for_condition(
            pilot,
            finished.is_set,
            message="Old Prompt request did not finish after release.",
        )
        await pilot.pause()

        assert old_controller.applied_result is None
        assert fresh_screen._library_prompt_browse_controller.applied_result is fresh_result
        assert fresh_result is not None
        assert fresh_result.items[0]["name"] == "Entry prompt"


@pytest.mark.asyncio
async def test_late_broad_snapshot_cannot_replace_the_dedicated_prompt_page(
    tmp_path: Path,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    prompts_db = PromptsDatabase(
        tmp_path / "prompt-snapshot-isolation.db",
        client_id="prompt-snapshot-isolation",
    )
    for index in range(1, 26):
        prompts_db.add_prompt(
            name=f"Prompt {index:02d}",
            author="Codex",
            details="Dedicated page",
            system_prompt="Be exact.",
            user_prompt=str(index),
        )
    app.prompts_db = prompts_db
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts_db),
        server_service=None,
    )
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_PROMPTS})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_selector(active, pilot, "#library-prompt-row-25")
        page_two = dataclasses.replace(
            active._library_prompt_browse_controller.applied_result.scope,
            page=2,
        )
        active._request_library_prompts_browse(page_two)
        await _wait_for_selector(active, pilot, "#library-prompt-row-5")
        applied = active._library_prompt_browse_controller.applied_result

        records = dict(active._local_source_records)
        records["prompts"] = (
            {
                "id": 999,
                "name": "Broad snapshot row",
                "version": 1,
            },
        )
        counts = dict(active._local_source_counts)
        counts["prompts"] = 999
        active._apply_local_source_snapshot(
            records,
            counts,
            dict(active._local_source_total_known),
            active._library_lookup_error,
            active._library_lookup_recovery_state,
            dict(active._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert active._library_prompt_browse_controller.applied_result is applied
        assert [row.prompt_id for row in active._build_library_prompts_state().rows] == [
            5,
            4,
            3,
            2,
            1,
        ]
        assert not active.query("#library-prompt-row-999")


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
async def test_skills_posture_sync_composes_focus_with_render_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A posture callback must not replace the pending generation completion."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}]
    )
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_SKILLS})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        focus = await _wait_for_selector(
            active_screen, pilot, "#library-skills-filter"
        )
        await active_screen.workers.wait_for_complete()
        canvas = active_screen.query_one(
            "#library-skills-canvas", LibrarySkillsListCanvas
        )
        focus = active_screen.query_one("#library-skills-filter", Input)
        focus.focus()
        await pilot.pause()
        restore_calls: list[str | None] = []
        original_restore = active_screen._restore_library_entry_focus

        def record_restore(identity, **kwargs) -> None:
            restore_calls.append(identity.widget_id)
            original_restore(identity, **kwargs)

        monkeypatch.setattr(
            active_screen,
            "_restore_library_entry_focus",
            record_restore,
        )

        # Hold both canvas refreshes before their shared replace-latest callback
        # slot fires: source reconciliation queues completion first, then the
        # automatic posture result queues its focus intent.
        monkeypatch.setattr(canvas, "refresh", lambda *args, **kwargs: canvas)
        generation = active_screen._library_snapshot_state_generation + 1
        route_key = active_screen._library_entry_route_key()
        active_screen._library_snapshot_state_generation = generation
        active_screen._library_entry_reconcile_dirty = True
        active_screen._library_entry_reconcile_pending = (generation, route_key)

        result = await active_screen._reconcile_library_entry_state(
            generation, route_key
        )
        await pilot.pause()
        posture_result = await active_screen._load_library_skills_trust_posture(
            lambda: "ready"
        )
        callback = canvas._post_recompose_callback
        assert callback is not None
        canvas._post_recompose_callback = None
        callback()

        assert (result, posture_result) == (
            LibraryEntryReconcileResult.APPLIED,
            LibraryEntryReconcileResult.APPLIED,
        )
        assert "library-skills-filter" in restore_calls
        assert active_screen._library_snapshot_rendered_generation == generation
        assert active_screen._library_entry_reconcile_dirty is False
        assert active_screen._library_entry_reconcile_pending is None


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
@pytest.mark.parametrize("intervening_focus", [False, True], ids=["restore", "veto"])
async def test_automatic_collections_result_preserves_input_semantics_unless_focus_moves(
    monkeypatch: pytest.MonkeyPatch,
    intervening_focus: bool,
) -> None:
    """Collections refresh restores one live Input but never overrides a later focus."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService([])
    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "collections"})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_selector(
            active_screen, pilot, "#library-collection-name-input"
        )
        await active_screen.workers.wait_for_complete()
        await pilot.pause()
        panel = active_screen.query_one(
            "#library-collections-panel", LibraryCollectionsPanel
        )
        name_input = active_screen.query_one(
            "#library-collection-name-input", Input
        )
        name_input.value = "Launch Evidence"
        await pilot.pause()
        name_input = active_screen.query_one(
            "#library-collection-name-input", Input
        )
        name_input.focus()
        await pilot.pause()
        expected_selection = Selection(2, 9)
        name_input.selection = expected_selection
        await pilot.pause()

        recompose_started = asyncio.Event()
        release_recompose = asyncio.Event()
        original_recompose = panel.recompose

        async def gated_recompose() -> None:
            recompose_started.set()
            await release_recompose.wait()
            await original_recompose()

        monkeypatch.setattr(panel, "recompose", gated_recompose)
        app.library_collections_service = StaticLibraryCollectionsService(
            [
                {
                    "collection_id": "collection-1",
                    "name": "Release Sources",
                    "description": "Fresh automatic result.",
                    "item_count": 1,
                    "source_authority": "local",
                    "sync_status": "local-only",
                    "updated_at": "2026-08-13T10:00:00Z",
                }
            ]
        )

        if intervening_focus:
            monkeypatch.setattr(panel, "refresh", lambda *args, **kwargs: panel)
            result = await active_screen._sync_collections_panel(
                refresh_snapshot=True
            )
            callback = panel._post_recompose_callback
            assert callback is not None
            intervening_target = active_screen.query_one(
                "#console-rail-section-toggle-library-details"
            )
            active_screen.set_focus(intervening_target)
            panel._post_recompose_callback = None
            callback()

            assert result is LibraryEntryReconcileResult.APPLIED
            assert active_screen.focused is intervening_target
            return

        sync_task = asyncio.create_task(
            active_screen._sync_collections_panel(refresh_snapshot=True)
        )
        await asyncio.wait_for(
            recompose_started.wait(),
            timeout=10,
        )
        release_recompose.set()
        result = await sync_task
        await _wait_for_condition(
            pilot,
            lambda: bool(active_screen.query("#library-collection-select-0")),
            message="Collections automatic result never rendered its row.",
        )
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.APPLIED
        restored = active_screen.query_one(
            "#library-collection-name-input", Input
        )
        assert restored.value == "Launch Evidence"
        assert restored.disabled is False
        assert active_screen.focused is restored
        assert restored.selection == expected_selection
        assert restored.cursor_position == expected_selection.end


@pytest.mark.asyncio
@pytest.mark.parametrize("intervening_focus", [False, True], ids=["restore", "veto"])
async def test_coalesced_collections_sync_preserves_pending_input_semantics(
    monkeypatch: pytest.MonkeyPatch,
    intervening_focus: bool,
) -> None:
    """A capture-less second sync must retain the first sync's focus intent."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService([])
    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "collections"})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_selector(
            active_screen, pilot, "#library-collection-name-input"
        )
        await active_screen.workers.wait_for_complete()
        await pilot.pause()
        panel = active_screen.query_one(
            "#library-collections-panel", LibraryCollectionsPanel
        )
        name_input = active_screen.query_one(
            "#library-collection-name-input", Input
        )
        name_input.value = "Coalesced Evidence"
        name_input.focus()
        await pilot.pause()
        expected_selection = Selection(3, 11)
        name_input.selection = expected_selection
        await pilot.pause()

        monkeypatch.setattr(panel, "refresh", lambda *args, **kwargs: panel)
        sync_kwargs = {
            "name_value": "stale worker value",
            "description_value": panel.description_value,
            "delete_pending": panel.delete_pending,
            "deferred_guard": lambda: True,
        }
        panel.sync_state(panel.state, **sync_kwargs)
        assert active_screen.focused is None
        panel.sync_state(panel.state, **sync_kwargs)

        callback = panel._post_recompose_callback
        assert callback is not None
        if intervening_focus:
            intervening_target = active_screen.query_one(
                "#console-rail-section-toggle-library-details"
            )
            active_screen.set_focus(intervening_target)
        panel._post_recompose_callback = None
        callback()

        if intervening_focus:
            assert active_screen.focused is intervening_target
        else:
            assert active_screen.focused is name_input
            assert name_input.value == "Coalesced Evidence"
            assert name_input.selection == expected_selection
            assert name_input.cursor_position == expected_selection.end


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["media", "export", "collections"])
async def test_superseded_entry_result_converges_on_current_dirty_generation(
    surface: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A same-route dirty generation must admit automatic-result repair."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
    )
    started = threading.Event()
    release = threading.Event()

    if surface == "media":

        def gated_media(**kwargs):
            started.set()
            assert release.wait(timeout=10), "Media detail gate was not released."
            return _two_media_items()[0]

        app.media_reading_scope_service = SimpleNamespace(
            get_media_item=gated_media,
        )
        screen = LibraryScreen(app)
        screen.restore_state(
            {
                "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
                "selected_media_id": "media-1",
                "library_media_view": "viewer",
            }
        )
    elif surface == "collections":

        def gated_collections():
            started.set()
            assert release.wait(timeout=10), "Collections gate was not released."
            return (
                {
                    "collection_id": "collection-1",
                    "name": "Dirty-generation sources",
                    "description": "Must replace Loading.",
                    "item_count": 1,
                    "source_authority": "local",
                    "sync_status": "local-only",
                    "updated_at": "2026-08-13T10:00:00Z",
                },
            )

        app.library_collections_service = SimpleNamespace(
            list_collections=gated_collections
        )
        screen = LibraryScreen(app)
        screen.apply_navigation_context({"mode": "collections"})
    else:
        _wire_entry_export_databases(app, tmp_path)
        original_compute = LibraryScreen._compute_library_export_counts
        compute_calls = 0

        def gated_export(*args, **kwargs):
            nonlocal compute_calls
            compute_calls += 1
            if compute_calls == 1:
                started.set()
                assert release.wait(timeout=10), "Export counts gate was not released."
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(
            LibraryScreen,
            "_compute_library_export_counts",
            staticmethod(gated_export),
        )
        screen = LibraryScreen(app)
        screen.restore_state({"library_selected_row_id": LIBRARY_ROW_INGEST_EXPORT})

    host = LibraryHarness(app, screen=screen)
    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            active_screen = _active_library_screen(host)
            await _wait_for_condition(
                pilot,
                started.is_set,
                message=f"{surface} automatic worker did not reach its gate.",
            )
            await _wait_for_library_shell(active_screen, pilot)
            current_generation = (
                active_screen._library_snapshot_state_generation + 1
            )
            active_screen._library_snapshot_state_generation = current_generation
            active_screen._library_snapshot_rendered_generation = (
                current_generation - 1
            )
            active_screen._library_entry_reconcile_dirty = True
            active_screen._library_entry_reconcile_pending = None
            active_screen._library_entry_reconcile_retry_generation = None
            release.set()

            def rendered_result() -> bool:
                if surface == "media":
                    titles = list(active_screen.query("#library-media-viewer-title"))
                    return bool(
                        titles
                        and "Interview Recording"
                        in str(titles[0].renderable)
                    )
                if surface == "collections":
                    titles = list(active_screen.query("#library-collections-title"))
                    return bool(
                        titles and "Collections (1)" in str(titles[0].renderable)
                    )
                scope_lines = list(active_screen.query("#library-export-scope-line"))
                return bool(
                    scope_lines
                    and not str(scope_lines[0].renderable).startswith("Counting")
                )

            await _wait_for_condition(
                pilot,
                rendered_result,
                timeout=4.0,
                message=(
                    f"{surface} automatic result did not converge on the "
                    "current dirty generation."
                ),
            )
            await active_screen.workers.wait_for_complete()
    finally:
        release.set()


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
async def test_entry_reconcile_repairs_current_route_after_remove_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entry reconciler itself must not strand an empty canvas host."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        canvas_host = screen.query_one("#library-canvas")
        original_remove = canvas_host.remove_children

        def route_switching_remove(*children):
            removal = original_remove(*children)
            screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
            screen._library_snapshot_state_generation += 1
            return removal

        monkeypatch.setattr(canvas_host, "remove_children", route_switching_remove)
        result = await screen._reconcile_library_entry_state(generation, route_key)
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert isinstance(screen._library_entry_canvas_owner(), LibraryMediaCanvas)


@pytest.mark.asyncio
async def test_entry_reconcile_repairs_current_route_after_mount_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale child mounted by entry reconcile must yield to the latest route."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        canvas_host = screen.query_one("#library-canvas")
        original_mount = canvas_host.mount
        mounts = 0

        def route_switching_mount(*widgets, **kwargs):
            nonlocal mounts
            mounted = original_mount(*widgets, **kwargs)
            mounts += 1
            if mounts == 1:
                screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
                screen._library_snapshot_state_generation += 1
            return mounted

        monkeypatch.setattr(canvas_host, "mount", route_switching_mount)
        result = await screen._reconcile_library_entry_state(generation, route_key)
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert isinstance(screen._library_entry_canvas_owner(), LibraryMediaCanvas)


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
        await wait_for_background_signal(
            started,
            task,
            what="pending conversation point fetch",
        )

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
        result = await await_background_task(
            task,
            what="pending conversation point fetch",
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._selected_conversation_id == "chat-1"
        assert screen._library_entry_canvas_owner() is owner
        assert screen.focused is focus


@pytest.mark.asyncio
async def test_pending_conversation_open_retries_initial_snapshot_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An initial snapshot race must still open an out-of-page deep link."""
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
        fetch_calls = 0

        async def gated_fetch(_conversation_id: str):
            nonlocal fetch_calls
            fetch_calls += 1
            if fetch_calls == 1:
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
        await wait_for_background_signal(
            started,
            task,
            what="initial pending conversation point fetch",
        )

        snapshot_records = dict(screen._local_source_records)
        snapshot_records["conversations"] = (
            {
                **_two_conversations()[0],
                "title": "Initial snapshot landed while point fetch waited",
            },
        )
        snapshot_counts = dict(screen._local_source_counts)
        snapshot_counts["conversations"] = 1
        changed = screen._apply_local_source_snapshot(
            snapshot_records,
            snapshot_counts,
            dict(screen._local_source_total_known),
        )
        generation = screen._library_snapshot_state_generation
        assert changed is True
        await _wait_for_condition(
            pilot,
            lambda: screen._library_snapshot_rendered_generation == generation,
            message="Initial source snapshot did not finish reconciling.",
        )
        release.set()
        result = await await_background_task(
            task,
            what="pending conversation retry",
        )
        await pilot.pause()

        assert result is LibraryEntryReconcileResult.APPLIED
        assert fetch_calls == 2
        assert screen._selected_conversation_id == "chat-pending"
        assert screen._conversation_record_id(
            screen._local_source_records["conversations"][0], 0
        ) == "chat-pending"
        assert isinstance(
            screen._library_entry_canvas_owner(), LibraryConversationsCanvas
        )


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
        counts_before = dict(active_screen._library_export_counts or {})
        generation = active_screen._library_snapshot_state_generation
        route_key = active_screen._library_entry_route_key()
        request_id = active_screen._library_export_counts_request_id
        active_screen._library_snapshot_state_generation += 1

        result = active_screen._apply_library_export_counts(
            active_screen._library_export_scope,
            {"media": 99, "conversations": 99, "notes": 99, "prompts": 99},
            generation=generation,
            route_key=route_key,
            request_id=request_id,
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert active_screen._library_export_counts == counts_before
        assert str(scope_line.renderable) == rendered_before


@pytest.mark.asyncio
async def test_export_counts_leave_return_same_scope_rejects_older_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A same-route/same-scope ABA visit keeps the newest landed counts."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    _wire_entry_export_databases(app, tmp_path)
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_snapshot_rendered_generation
            == screen._library_snapshot_state_generation,
            message="Export ABA setup did not settle its source snapshot.",
        )
        requests: list[tuple[Any, ...]] = []

        def capture_counts_request(*args: Any) -> None:
            requests.append(args)

        monkeypatch.setattr(
            screen, "_run_library_export_counts_worker", capture_counts_request
        )

        await screen._select_library_rail_row(LIBRARY_ROW_INGEST_EXPORT)
        await _wait_for_selector(screen, pilot, "#library-export-canvas")
        assert len(requests) == 1
        old_request = requests[-1]

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await screen._select_library_rail_row(LIBRARY_ROW_INGEST_EXPORT)
        await _wait_for_selector(screen, pilot, "#library-export-canvas")
        assert len(requests) == 2
        new_request = requests[-1]
        assert len(old_request) == len(new_request) == 7
        old_scope, *_, old_generation, old_route_key, old_request_id = old_request
        new_scope, *_, new_generation, new_route_key, new_request_id = new_request
        assert old_scope == new_scope
        assert old_route_key == new_route_key
        assert old_request_id < new_request_id

        newest_counts = {
            "media": 2,
            "conversations": 3,
            "notes": 5,
            "prompts": 7,
        }
        newest_result = screen._apply_library_export_counts(
            new_scope,
            newest_counts,
            generation=new_generation,
            route_key=new_route_key,
            request_id=new_request_id,
        )
        scope_line = screen.query_one("#library-export-scope-line")
        rendered_newest = str(scope_line.renderable)
        stale_result = screen._apply_library_export_counts(
            old_scope,
            {"media": 99, "conversations": 99, "notes": 99, "prompts": 99},
            generation=old_generation,
            route_key=old_route_key,
            request_id=old_request_id,
        )

        assert newest_result is LibraryEntryReconcileResult.APPLIED
        assert stale_result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._library_export_counts == newest_counts
        assert (
            screen.query_one("#library-export-canvas").state.scope_line
            == rendered_newest
        )
        assert str(scope_line.renderable) == rendered_newest


def _compositor_text(screen: LibraryScreen) -> str:
    """Return only text actually painted in the current terminal frame."""
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


def _exported_svg_text(host: LibraryHarness) -> str:
    """Return text nodes from an exported compositor frame as plain text."""
    import re
    from html import unescape

    svg = host.export_screenshot(simplify=True)
    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


def _install_screen_lifecycle_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Record compose plus both whole-screen recompose APIs for UAT."""
    evidence = SimpleNamespace(compose=[], refresh_recompose=[], recompose=[])
    original_compose = LibraryScreen.compose_content
    original_refresh = LibraryScreen.refresh
    original_recompose = LibraryScreen.recompose

    def counted_compose(screen):
        evidence.compose.append(screen)
        yield from original_compose(screen)

    def recorded_refresh(screen, *regions, **kwargs):
        if kwargs.get("recompose"):
            evidence.refresh_recompose.append(screen)
        return original_refresh(screen, *regions, **kwargs)

    async def recorded_recompose(screen):
        evidence.recompose.append(screen)
        return await original_recompose(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counted_compose)
    monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
    monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
    return evidence


def _screen_identity_tuple(screen: LibraryScreen) -> tuple[int, int, int, int]:
    """Return screen/rail/host/active-owner identities for evidence output."""
    owner = screen._library_entry_canvas_owner()
    assert owner is not None
    return (
        id(screen),
        id(screen.query_one("#library-rail")),
        id(screen.query_one("#library-canvas")),
        id(owner),
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
@pytest.mark.parametrize("size", [(60, 20), (170, 48)])
async def test_uat_warm_landing_fresh_reconcile_retains_frame_and_focus(
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm Landing frame must paint cached data before fresh data lands."""
    app = _build_test_app()
    conversations = _two_conversations()
    _seed_conversations(app, conversations[:1])
    evidence = _install_screen_lifecycle_spies(monkeypatch)
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        first = _active_library_screen(host)
        await _wait_for_library_shell(first, pilot)
        await first.workers.wait_for_complete()
        await host.pop_screen()
        await pilot.pause()

        _seed_conversations(app, conversations)
        fresh_started = asyncio.Event()
        release_fresh = asyncio.Event()
        original_list = LibraryScreen._list_local_source_snapshot

        async def gated_fresh(screen: LibraryScreen):
            fresh_started.set()
            await release_fresh.wait()
            return await original_list(screen)

        monkeypatch.setattr(
            LibraryScreen,
            "_list_local_source_snapshot",
            gated_fresh,
        )
        revisit = LibraryScreen(app)
        await host.push_screen(revisit)
        await _wait_for_library_shell(revisit, pilot)
        await wait_for_signal(
            fresh_started,
            what="warm landing fresh reconciliation",
        )
        if size == (60, 20):
            revisit._library_notes_stage = "notes"
            revisit._set_library_rail_collapsed(True)
            await pilot.pause()
            await pilot.pause()

        identity_before = _screen_identity_tuple(revisit)
        search = revisit.query_one("#library-hub-action-search")
        search.focus()
        await pilot.pause()
        assert "Conversations (1)" in _compositor_text(revisit)
        assert "Conversations (1)" in _exported_svg_text(host)

        release_fresh.set()
        await revisit.workers.wait_for_complete()
        await _wait_for_condition(
            pilot,
            lambda: (
                revisit._local_source_counts["conversations"] == 2
                and revisit._library_snapshot_rendered_generation
                == revisit._library_snapshot_state_generation
            ),
            message="Warm Landing fresh reconciliation did not settle.",
        )
        await pilot.pause()
        identity_after = _screen_identity_tuple(revisit)
        compositor = _compositor_text(revisit)
        exported_svg = _exported_svg_text(host)

        assert identity_after == identity_before
        assert revisit.focused is search
        assert "Conversations (2)" in compositor
        assert "Conversations (2)" in exported_svg
        assert "Search" in compositor
        assert "Search" in exported_svg
        assert evidence.compose.count(revisit) == 1
        assert revisit not in evidence.refresh_recompose
        assert revisit not in evidence.recompose
        print(
            "task5_uat_warm_landing "
            f"size={size} identity_before={identity_before} "
            f"identity_after={identity_after} compose={evidence.compose.count(revisit)} "
            f"refresh_recompose={evidence.refresh_recompose.count(revisit)} "
            f"recompose={evidence.recompose.count(revisit)} focus={search.id}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (170, 48)])
async def test_uat_cold_conversations_loading_to_rows_is_compositor_visible(
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold Conversations must paint loading and then rows without a screen recompose."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    evidence = _install_screen_lifecycle_spies(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()
    original_list = LibraryScreen._list_local_source_snapshot

    async def gated_list(screen: LibraryScreen):
        started.set()
        await release.wait()
        return await original_list(screen)

    monkeypatch.setattr(LibraryScreen, "_list_local_source_snapshot", gated_list)
    screen = LibraryScreen(app)
    screen.restore_state(
        {"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS}
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=size) as pilot:
        active = _active_library_screen(host)
        await wait_for_signal(
            started,
            what="cold conversations snapshot load",
        )
        await _wait_for_selector(active, pilot, "#library-canvas-loading")
        if size == (60, 20):
            active._library_notes_stage = "notes"
            active._set_library_rail_collapsed(True)
            await pilot.pause()
            await pilot.pause()
        loading_identity = _screen_identity_tuple(active)
        assert "Loading local Library sources" in _compositor_text(active)
        assert "Loading local Library sources" in _exported_svg_text(host)

        release.set()
        await _wait_for_selector(active, pilot, "#library-conversation-row-0")
        await active.workers.wait_for_complete()
        await pilot.pause()
        rows_identity = _screen_identity_tuple(active)
        compositor = _compositor_text(active)
        exported_svg = _exported_svg_text(host)

        assert rows_identity[:3] == loading_identity[:3]
        assert rows_identity[3] != loading_identity[3]
        assert "Conversations (2)" in compositor
        assert "Conversations (2)" in exported_svg
        assert "Design review notes" in compositor
        assert "Design review notes" in exported_svg
        assert evidence.compose.count(active) == 1
        assert active not in evidence.refresh_recompose
        assert active not in evidence.recompose
        print(
            "task5_uat_cold_conversations "
            f"size={size} loading_identity={loading_identity} "
            f"rows_identity={rows_identity} compose={evidence.compose.count(active)} "
            f"refresh_recompose={evidence.refresh_recompose.count(active)} "
            f"recompose={evidence.recompose.count(active)}"
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
        await wait_for_background_signal(
            removal_started,
            replacement,
            what="landing recent-row replacement",
        )
        landing.state = latest
        release_removal.set()
        await await_background_task(
            replacement,
            what="landing recent-row replacement",
        )

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
async def test_source_worker_completion_during_mount_dispatch_reconciles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing mount-safe scheduling would lose a fetch that completes in Mount."""
    app = _build_test_app()
    _seed_conversations(app, [])
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()
    fresh_applied = asyncio.Event()
    mount_dispatch_active = False
    apply_during_mount: list[bool] = []
    mounted_at_apply: list[bool] = []
    attached_at_apply: list[bool] = []
    target_sync_calls: list[int] = []
    original_on_mount = LibraryScreen.on_mount
    original_apply = LibraryScreen._apply_local_source_snapshot
    original_sync = library_screen_module._sync_library_canvas

    async def gated_snapshot(_screen: LibraryScreen):
        fetch_started.set()
        await release_fetch.wait()
        return (
            {
                "notes": (),
                "media": (),
                "conversations": tuple(_two_conversations()),
                "prompts": (0, ()),
                "skills": (
                    0,
                    {"available_skills": [], "blocked_skills": []},
                ),
            },
            {"notes": 0, "media": 0, "conversations": 2},
            {"notes": True, "media": True, "conversations": True},
            None,
            None,
            {"study_decks": 0, "flashcards_due": 0, "quizzes": 0},
        )

    def recorded_apply(screen: LibraryScreen, records, *args, **kwargs):
        result = original_apply(screen, records, *args, **kwargs)
        if len(records.get("conversations", ())) == 2:
            apply_during_mount.append(mount_dispatch_active)
            mounted_at_apply.append(screen.is_mounted)
            attached_at_apply.append(screen.is_attached)
            fresh_applied.set()
        return result

    def recorded_sync(screen: LibraryScreen, kind: str, **kwargs):
        target_sync_calls.append(screen._library_snapshot_state_generation)
        return original_sync(screen, kind, **kwargs)

    async def gated_on_mount(screen: LibraryScreen) -> None:
        nonlocal mount_dispatch_active
        mount_dispatch_active = True
        try:
            original_on_mount(screen)
            await fetch_started.wait()
            release_fetch.set()
            async with asyncio.timeout(10):
                await fresh_applied.wait()
        finally:
            mount_dispatch_active = False

    monkeypatch.setattr(LibraryScreen, "_list_local_source_snapshot", gated_snapshot)
    monkeypatch.setattr(LibraryScreen, "_apply_local_source_snapshot", recorded_apply)
    monkeypatch.setattr(library_screen_module, "_sync_library_canvas", recorded_sync)
    monkeypatch.setattr(LibraryScreen, "on_mount", gated_on_mount)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert apply_during_mount == [True]
        assert mounted_at_apply == [False]
        assert attached_at_apply == [True]
        assert screen._local_source_counts["conversations"] == 2
        assert screen._library_snapshot_rendered_generation == (
            screen._library_snapshot_state_generation
        )
        assert target_sync_calls == [screen._library_snapshot_state_generation]
        assert "Conversations (2)" in _compositor_text(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (170, 48)])
async def test_snapshot_timeout_is_repaired_by_blocked_fresh_success(
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout generation must not prevent the in-flight fetch from repairing it."""
    app = _build_test_app()
    _seed_conversations(app, [])
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()
    success_applied = asyncio.Event()
    target_sync_calls: list[int] = []
    original_apply = LibraryScreen._apply_local_source_snapshot
    original_sync = library_screen_module._sync_library_canvas
    evidence = _install_screen_lifecycle_spies(monkeypatch)

    async def gated_snapshot(_screen: LibraryScreen):
        fetch_started.set()
        await release_fetch.wait()
        return (
            {
                "notes": (),
                "media": (),
                "conversations": tuple(_two_conversations()),
                "prompts": (0, ()),
                "skills": (
                    0,
                    {"available_skills": [], "blocked_skills": []},
                ),
            },
            {"notes": 0, "media": 0, "conversations": 2},
            {"notes": True, "media": True, "conversations": True},
            None,
            None,
            {"study_decks": 0, "flashcards_due": 0, "quizzes": 0},
        )

    def recorded_apply(screen: LibraryScreen, records, *args, **kwargs):
        result = original_apply(screen, records, *args, **kwargs)
        if len(records.get("conversations", ())) == 2:
            success_applied.set()
        return result

    def recorded_sync(screen: LibraryScreen, kind: str, **kwargs):
        target_sync_calls.append(screen._library_snapshot_state_generation)
        return original_sync(screen, kind, **kwargs)

    monkeypatch.setattr(LibraryScreen, "_list_local_source_snapshot", gated_snapshot)
    monkeypatch.setattr(LibraryScreen, "_apply_local_source_snapshot", recorded_apply)
    monkeypatch.setattr(library_screen_module, "_sync_library_canvas", recorded_sync)

    host = LibraryHarness(app)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await wait_for_signal(
            fetch_started,
            what="fresh Library snapshot fetch",
        )
        screen._apply_source_snapshot_timeout()
        await pilot.pause()
        if size == (60, 20):
            screen._library_notes_stage = "notes"
            screen._set_library_rail_collapsed(True)
            await pilot.pause()
            await pilot.pause()
        identity_before = _screen_identity_tuple(screen)
        assert screen._library_lookup_error == library_screen_module.LIBRARY_SERVICE_ERROR_COPY
        assert library_screen_module.LIBRARY_SERVICE_ERROR_COPY in _compositor_text(screen)
        assert library_screen_module.LIBRARY_SERVICE_ERROR_COPY in _exported_svg_text(
            host
        )

        release_fetch.set()
        async with asyncio.timeout(10):
            await success_applied.wait()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_lookup_error is None
        assert screen._local_source_counts["conversations"] == 2
        assert screen._library_entry_reconcile_dirty is False
        assert target_sync_calls[-1] == screen._library_snapshot_state_generation
        identity_after = _screen_identity_tuple(screen)
        assert identity_after == identity_before
        assert "Conversations (2)" in _compositor_text(screen)
        assert "Conversations (2)" in _exported_svg_text(host)
        assert library_screen_module.LIBRARY_SERVICE_ERROR_COPY not in _compositor_text(
            screen
        )
        assert evidence.compose.count(screen) == 1
        assert screen not in evidence.refresh_recompose
        assert screen not in evidence.recompose
        print(
            "task5_uat_timeout_success "
            f"size={size} identity_before={identity_before} "
            f"identity_after={identity_after} compose={evidence.compose.count(screen)} "
            f"refresh_recompose={evidence.refresh_recompose.count(screen)} "
            f"recompose={evidence.recompose.count(screen)}"
        )


@pytest.mark.asyncio
async def test_two_changed_generations_render_only_the_newer_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the generation guard would project both queued generations."""
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
        sync_generations: list[int] = []
        original_sync_state = canvas.sync_state

        def recorded_sync_state(*args, **kwargs):
            sync_generations.append(screen._library_snapshot_state_generation)
            return original_sync_state(*args, **kwargs)

        monkeypatch.setattr(canvas, "sync_state", recorded_sync_state)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        original_reconcile = screen._reconcile_library_entry_state
        route_key = screen._library_entry_route_key()

        first_records = dict(screen._local_source_records)
        first_records["conversations"] = (
            *first_records["conversations"],
            {
                "title": "Superseded generation",
                "conversation_id": "chat-old",
                "message_count": 1,
                "updated_at": "2026-08-13T10:00:00Z",
            },
        )
        assert screen._apply_local_source_snapshot(
            first_records,
            {**screen._local_source_counts, "conversations": 3},
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
            schedule_reconcile=False,
        )
        first_generation = screen._library_snapshot_state_generation

        async def gated_reconcile(generation, queued_route):
            if generation == first_generation:
                first_started.set()
                await release_first.wait()
            return await original_reconcile(generation, queued_route)

        monkeypatch.setattr(screen, "_reconcile_library_entry_state", gated_reconcile)
        first_task = asyncio.create_task(
            screen._reconcile_library_entry_state(first_generation, route_key)
        )
        await wait_for_background_signal(
            first_started,
            first_task,
            what="first Library entry reconciliation",
        )

        newer_records = dict(first_records)
        newer_records["conversations"] = (
            *tuple(
                record
                for record in first_records["conversations"]
                if record.get("conversation_id") != "chat-old"
            ),
            {
                "title": "Newest generation",
                "conversation_id": "chat-new",
                "message_count": 2,
                "updated_at": "2026-08-13T10:01:00Z",
            },
        )
        assert screen._apply_local_source_snapshot(
            newer_records,
            {**screen._local_source_counts, "conversations": 3},
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
            schedule_reconcile=False,
        )
        newer_generation = screen._library_snapshot_state_generation
        newer_result = await original_reconcile(newer_generation, route_key)
        await pilot.pause()
        await pilot.pause()

        release_first.set()
        first_result = await await_background_task(
            first_task,
            what="first Library entry reconciliation",
        )

        assert (first_result, newer_result) == (
            LibraryEntryReconcileResult.SUPERSEDED,
            LibraryEntryReconcileResult.APPLIED,
        )
        assert sync_generations == [newer_generation]
        assert screen._library_snapshot_rendered_generation == newer_generation
        assert "Newest generation" in _compositor_text(screen)
        assert "Superseded generation" not in _compositor_text(screen)


@pytest.mark.asyncio
async def test_queued_reconcile_supersedes_after_route_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the queued route guard would mutate the successor route."""
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
        queued: list[tuple[object, tuple[object, ...]]] = []
        results: list[LibraryEntryReconcileResult] = []
        target_sync_calls: list[str] = []
        original_sync = library_screen_module._sync_library_canvas

        def capture_call_later(callback, *args):
            queued.append((callback, args))

        def recorded_sync(active_screen: LibraryScreen, kind: str, **kwargs):
            target_sync_calls.append(kind)
            return original_sync(active_screen, kind, **kwargs)

        monkeypatch.setattr(screen, "call_later", capture_call_later)
        monkeypatch.setattr(library_screen_module, "_sync_library_canvas", recorded_sync)
        screen._schedule_library_entry_reconcile(generation, stale_route)
        assert len(queued) == 1

        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        callback, args = queued.pop()
        results.append(await callback(*args))

        assert results == [LibraryEntryReconcileResult.SUPERSEDED]
        assert target_sync_calls == []
        assert screen._library_entry_reconcile_pending is None


@pytest.mark.asyncio
async def test_detached_queued_reconcile_completion_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the attachment guard would let detached completion touch the DOM."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    completion_started = asyncio.Event()
    release_completion = asyncio.Event()
    target_sync_calls: list[str] = []
    result: LibraryEntryReconcileResult | None = None
    task: asyncio.Task[LibraryEntryReconcileResult] | None = None

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        original_reconcile = screen._reconcile_library_entry_state
        original_sync = library_screen_module._sync_library_canvas

        def recorded_sync(active_screen: LibraryScreen, kind: str, **kwargs):
            target_sync_calls.append(kind)
            return original_sync(active_screen, kind, **kwargs)

        async def delayed_completion() -> LibraryEntryReconcileResult:
            completion_started.set()
            await release_completion.wait()
            return await original_reconcile(generation, route_key)

        monkeypatch.setattr(library_screen_module, "_sync_library_canvas", recorded_sync)
        task = asyncio.create_task(delayed_completion())
        await wait_for_background_signal(
            completion_started,
            task,
            what="detached queued reconciliation",
        )

    assert task is not None
    assert screen.is_attached is False
    release_completion.set()
    result = await await_background_task(
        task,
        what="detached queued reconciliation",
    )

    assert result is LibraryEntryReconcileResult.SUPERSEDED
    assert target_sync_calls == []
    assert screen._library_entry_reconcile_pending is None


@pytest.mark.asyncio
async def test_warm_repeat_visit_composes_once_before_fresh_reconcile(monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    calls: list[LibraryScreen] = []
    refresh_recompose_calls: list[LibraryScreen] = []
    recompose_calls: list[LibraryScreen] = []
    original = LibraryScreen.compose_content
    original_refresh = LibraryScreen.refresh
    original_recompose = LibraryScreen.recompose

    def counted_compose(screen):
        calls.append(screen)
        yield from original(screen)

    def recorded_refresh(screen, *regions, **kwargs):
        if kwargs.get("recompose"):
            refresh_recompose_calls.append(screen)
        return original_refresh(screen, *regions, **kwargs)

    async def recorded_recompose(screen):
        recompose_calls.append(screen)
        return await original_recompose(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counted_compose)
    monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
    monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
    samples: list[float] = []
    revisits: list[LibraryScreen] = []
    identity_samples: list[tuple[int, int, int, int]] = []
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
            await revisit.workers.wait_for_complete()
            await pilot.pause()
            identity_samples.append(
                (
                    id(revisit),
                    id(revisit.query_one("#library-rail")),
                    id(revisit.query_one("#library-canvas")),
                    id(revisit._library_entry_canvas_owner()),
                )
            )
            await host.pop_screen()
            await pilot.pause()
    print(
        f"warm_visit_median_ms={statistics.median(samples):.3f} "
        f"min_ms={min(samples):.3f} max_ms={max(samples):.3f} n={len(samples)}"
    )
    print(f"warm_visit_compose_counts={[calls.count(revisit) for revisit in revisits]}")
    print(f"warm_visit_identity_samples={identity_samples}")
    print(
        "warm_visit_screen_recompose_counts="
        f"refresh={len(refresh_recompose_calls)} recompose={len(recompose_calls)}"
    )
    assert all(calls.count(revisit) == 1 for revisit in revisits)
    assert len(identity_samples) == 5
    assert all(all(identity > 0 for identity in sample) for sample in identity_samples)
    assert refresh_recompose_calls == []
    assert recompose_calls == []


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
        screen._library_entry_reconcile_retry_generation = (generation, stale_route)
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA

        result = await screen._reconcile_library_entry_state(
            generation, stale_route
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None


@pytest.mark.asyncio
async def test_same_generation_new_route_receives_its_own_first_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route-A retry marker must not consume route B's first retry."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        generation = screen._library_snapshot_state_generation
        route_a = screen._library_entry_route_key()
        queued: list[tuple[object, tuple[object, ...]]] = []

        def capture_call_later(callback, *args):
            queued.append((callback, args))

        monkeypatch.setattr(screen, "call_later", capture_call_later)
        first = screen._retry_or_fail_library_entry_reconcile(generation, route_a)
        assert first is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_retry_generation == (
            generation,
            route_a,
        )
        assert len(queued) == 1

        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        route_b = screen._library_entry_route_key()
        second = screen._retry_or_fail_library_entry_reconcile(generation, route_b)

        assert second is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_pending == (generation, route_b)
        assert screen._library_entry_reconcile_retry_generation == (
            generation,
            route_b,
        )
        assert len(queued) == 2


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
        scheduled_attempts: list[tuple[object, ...]] = []
        reconcile_results: list[LibraryEntryReconcileResult] = []

        def capture_call_later(callback, *args):
            queued.append((callback, args))
            scheduled_attempts.append(args)

        monkeypatch.setattr(screen, "call_later", capture_call_later)

        reconcile_results.append(
            await screen._reconcile_library_entry_state(generation, route_key)
        )

        assert reconcile_results == [LibraryEntryReconcileResult.FAILED]
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert screen._library_entry_reconcile_retry_generation == (
            generation,
            route_key,
        )
        assert len(queued) == 1

        callback, args = queued.pop()
        reconcile_results.append(await callback(*args))

        assert reconcile_results == [
            LibraryEntryReconcileResult.FAILED,
            LibraryEntryReconcileResult.FAILED,
        ]
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None
        assert queued == []
        assert len(scheduled_attempts) == 1

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
        assert len(scheduled_attempts) == 2


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
        screen._library_entry_reconcile_retry_generation = (generation, route_key)
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


def test_cache_seed_rejects_non_finite_timestamp():
    """A NaN stamp must not bypass both cache-age comparisons."""
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
    app._library_source_snapshot_cache_stamp = math.nan

    screen = LibraryScreen(app)

    assert screen._library_loaded is False
    assert screen._library_snapshot_state_generation == 0
