"""Ordered assembly of the existing Library subsystem controllers.

Keep named dependencies explicit and resolve sibling/state lookups at call time.
This assembly runs at the original construction position in LibraryScreen.__init__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .library_collections_controller import LibraryCollectionsController
from .library_conversation_reader_controller import LibraryConversationReaderController
from .library_conversations_controller import LibraryConversationsController
from .library_export_controller import LibraryExportController
from .library_rag_search_controller import LibraryRagSearchController
from .library_skills_controller import LibrarySkillsController

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


def build_library_controllers(screen: LibraryScreen) -> None:
    """Wire existing owners without evaluating any of their lazy dependencies."""
    screen._conversation_reader_controller = LibraryConversationReaderController(
        screen,
        conversations_state_accessor=lambda: screen._conversations_state,
        build_conversations_state=lambda: screen._build_library_conversations_state(),
        adaptive_reader_allocation_is_current=(
            lambda reader: screen._library_adaptive_reader_allocation_is_current(reader)
        ),
        run_library_service_call=(
            lambda *args, **kwargs: screen._run_library_service_call(*args, **kwargs)
        ),
        conversation_records=lambda: screen._conversation_records(),
        conversation_record_id=(
            lambda record, index: screen._conversation_record_id(record, index)
        ),
        library_loaded_accessor=lambda: screen._library_loaded,
        library_lookup_error_accessor=lambda: screen._library_lookup_error,
        notes_focus_intent_generation_accessor=(
            lambda: screen._library_notes_focus_intent_generation
        ),
        selected_row_id_accessor=lambda: screen._library_selected_row_id,
        selected_conversation_id_accessor=lambda: screen._selected_conversation_id,
    )
    screen._conversations_controller = LibraryConversationsController(
        screen,
        conversations_state_accessor=lambda: screen._conversations_state,
        ensure_reader_selection=(
            lambda: (
                screen._conversation_reader_controller._ensure_library_conversation_reader_selection()
            )
        ),
        start_reader_selection=(
            lambda *a, **k: (
                screen._conversation_reader_controller._start_library_conversation_reader_selection(
                    *a, **k
                )
            )
        ),
        sync_reader=(
            lambda: (
                screen._conversation_reader_controller._sync_library_conversation_reader()
            )
        ),
        selected_conversation_id_accessor=lambda: screen._selected_conversation_id,
        set_selected_conversation_id=(
            lambda value: setattr(screen, "_selected_conversation_id", value)
        ),
        pending_library_source_open_accessor=(
            lambda: screen._pending_library_source_open
        ),
        set_pending_library_source_open=(
            lambda value: setattr(screen, "_pending_library_source_open", value)
        ),
        selected_row_id_accessor=lambda: screen._library_selected_row_id,
        set_selected_row_id=(
            lambda value: setattr(screen, "_library_selected_row_id", value)
        ),
        local_source_records_accessor=(
            lambda: getattr(screen, "_local_source_records", {})
        ),
        library_canvas_projection_depth_accessor=(
            lambda: screen._library_canvas_projection_depth
        ),
        library_canvas_resync_pending_accessor=(
            lambda: screen._library_canvas_resync_pending
        ),
        set_library_canvas_resync_pending=(
            lambda value: setattr(screen, "_library_canvas_resync_pending", value)
        ),
        acknowledge_destination_change=(
            lambda: screen._acknowledge_library_destination_change()
        ),
        library_workspace_depth_state=(
            lambda *a, **k: screen._library_workspace_depth_state(*a, **k)
        ),
        open_library_export_canvas=(
            lambda *a, **k: screen._open_library_export_canvas(*a, **k)
        ),
        open_library_item_by_id=(
            lambda *a, **k: screen._open_library_item_by_id(*a, **k)
        ),
        run_library_service_call=(
            lambda *args, **kwargs: screen._run_library_service_call(*args, **kwargs)
        ),
        source_record_id=lambda record: screen._source_record_id(record),
        source_title=(
            lambda source_type, record: screen._source_title(source_type, record)
        ),
        library_conversation_loaded_preview_selected=(
            lambda: screen._library_conversation_loaded_preview_selected()
        ),
        selected_conversation_handoff_payload=(
            lambda: screen._selected_conversation_handoff_payload()
        ),
    )
    # Sentinel: `self._export_state` does NOT exist yet at this point in
    # `__init__` -- it is constructed later, at ~:3288, specifically to
    # preserve the computed `form` default's original `__init__`
    # evaluation position (see `LibraryExportState`'s module docstring).
    # Every dependency below is a lazy accessor (a `lambda`, not a bound
    # value), and no controller method may run during `__init__` -- an
    # eager `export_state_accessor()` call made from here would raise
    # `AttributeError: 'LibraryScreen' object has no attribute
    # '_export_state'`.
    screen._export_controller = LibraryExportController(
        screen,
        export_state_accessor=lambda: screen._export_state,
        apply_open_item_surface=(
            lambda *a, **k: screen._apply_library_open_item_surface(*a, **k)
        ),
        flush_note_save=lambda: screen._flush_library_note_save(),
        set_library_destination_with_conversation_fence=(
            lambda value: screen._set_library_destination_with_conversation_fence(value)
        ),
        sync_library_emergency_guard_presentation=(
            lambda: screen._sync_library_emergency_guard_presentation()
        ),
        close_open_library_choice_strip=(
            lambda: screen._close_open_library_choice_strip()
        ),
        focus_library_hub_entry=lambda: screen._focus_library_hub_entry(),
        select_library_rail_row=(
            lambda *a, **k: screen._select_library_rail_row(*a, **k)
        ),
        focus_library_choice_strip_active=(
            lambda *a, **k: screen._focus_library_choice_strip_active(*a, **k)
        ),
        focus_library_control=(lambda *a, **k: screen._focus_library_control(*a, **k)),
        library_selected_row_id_accessor=lambda: screen._library_selected_row_id,
        library_prompts_mutation_in_flight_accessor=(
            lambda: screen._prompts_state.mutation_in_flight
        ),
        build_library_export_state=lambda: screen._build_library_export_state(),
        start_library_export_counts_worker=(
            lambda: screen._start_library_export_counts_worker()
        ),
        start_library_export_worker=(
            lambda **k: screen._start_library_export_worker(**k)
        ),
        apply_library_export_success=(
            lambda *a, **k: screen._apply_library_export_success(*a, **k)
        ),
        apply_library_export_cancelled=(
            lambda run_id: screen._apply_library_export_cancelled(run_id)
        ),
        update_library_export_canvas_after_run=(
            lambda: screen._update_library_export_canvas_after_run()
        ),
        handle_library_export_cancel=(
            lambda event: screen.handle_library_export_cancel(event)
        ),
    )
    screen._collections_controller = LibraryCollectionsController(
        screen,
        collections_state_accessor=lambda: screen._collections_state,
        library_adaptive_reader_allocation_is_current=(
            lambda reader: screen._library_adaptive_reader_allocation_is_current(reader)
        ),
        library_selected_row_id_accessor=lambda: screen._library_selected_row_id,
        library_collections_capture_controller_accessor=(
            lambda: screen._library_collections_capture_controller
        ),
        set_library_collections_capture_controller=(
            lambda value: setattr(
                screen, "_library_collections_capture_controller", value
            )
        ),
    )
    screen._rag_search_controller = LibraryRagSearchController(
        screen,
        rag_search_state_accessor=lambda: screen._rag_search_state,
        active_library_rail=lambda: screen._active_library_rail(),
        console_setup_would_block=lambda: screen._console_setup_would_block(),
        open_library_item_by_id=(
            lambda *a, **k: screen._open_library_item_by_id(*a, **k)
        ),
        safe_text=lambda *a, **k: screen._safe_text(*a, **k),
        select_library_rail_row=(
            lambda row_id: screen._select_library_rail_row(row_id)
        ),
        trailing_index=lambda button_id: screen._trailing_index(button_id),
        library_selected_row_id_accessor=lambda: screen._library_selected_row_id,
        library_canvas_projection_depth_accessor=(
            lambda: screen._library_canvas_projection_depth
        ),
        library_canvas_resync_pending_accessor=(
            lambda: screen._library_canvas_resync_pending
        ),
        set_library_canvas_resync_pending=(
            lambda value: setattr(screen, "_library_canvas_resync_pending", value)
        ),
        execute_library_rag_answer=(
            lambda request, **kwargs: screen._execute_library_rag_answer(
                request, **kwargs
            )
        ),
        execute_library_rag_search=(
            lambda request: screen._execute_library_rag_search(request)
        ),
        save_library_search_history=(
            lambda history_list: screen._save_library_search_history(history_list)
        ),
        library_rag_panel_state=lambda: screen._library_rag_panel_state(),
        mirror_library_rag_scope_recovery=(
            lambda: screen._mirror_library_rag_scope_recovery()
        ),
        patch_sibling_library_search_input=(
            lambda selector, value: screen._patch_sibling_library_search_input(
                selector, value
            )
        ),
        refresh_search_rag_panel_state_widgets=(
            lambda **k: screen._refresh_search_rag_panel_state_widgets(**k)
        ),
    )
    screen._skills_controller = LibrarySkillsController(
        screen,
        skills_state_accessor=lambda: screen._skills_state,
        library_skill_import_coordinator_accessor=(
            lambda: screen._library_skill_import_coordinator
        ),
        set_library_skill_import_coordinator=(
            lambda value: setattr(screen, "_library_skill_import_coordinator", value)
        ),
        library_skills_browse_controller_accessor=(
            lambda: screen._library_skills_browse_controller
        ),
        set_library_skills_browse_controller=(
            lambda value: setattr(screen, "_library_skills_browse_controller", value)
        ),
        run_library_service_call=(
            lambda *args, **kwargs: screen._run_library_service_call(*args, **kwargs)
        ),
        sanitize_media_field=lambda *a, **k: screen._sanitize_media_field(*a, **k),
        sanitize_note_content=(lambda *a, **k: screen._sanitize_note_content(*a, **k)),
        refresh_local_source_snapshot=(
            lambda *a, **k: screen._refresh_local_source_snapshot(*a, **k)
        ),
        library_entry_route_key=lambda: screen._library_entry_route_key(),
        library_entry_reconcile_is_current=(
            lambda *a, **k: screen._library_entry_reconcile_is_current(*a, **k)
        ),
        capture_library_entry_focus=lambda: screen._capture_library_entry_focus(),
        restore_library_entry_focus=(
            lambda *a, **k: screen._restore_library_entry_focus(*a, **k)
        ),
        library_selected_row_id_accessor=lambda: screen._library_selected_row_id,
        set_library_selected_row_id=(
            lambda value: setattr(screen, "_library_selected_row_id", value)
        ),
        library_snapshot_state_generation_accessor=(
            lambda: screen._library_snapshot_state_generation
        ),
        library_entry_reconcile_dirty_accessor=(
            lambda: screen._library_entry_reconcile_dirty
        ),
        library_entry_reconcile_pending_accessor=(
            lambda: screen._library_entry_reconcile_pending
        ),
        library_canvas_projection_depth_accessor=(
            lambda: screen._library_canvas_projection_depth
        ),
        library_canvas_resync_pending_accessor=(
            lambda: screen._library_canvas_resync_pending
        ),
        set_library_canvas_resync_pending=(
            lambda value: setattr(screen, "_library_canvas_resync_pending", value)
        ),
        library_skills_import_open_accessor=(
            lambda: screen._library_skills_import_open
        ),
        library_skills_import_path_accessor=(
            lambda: screen._library_skills_import_path
        ),
        library_skills_import_status_accessor=(
            lambda: screen._library_skills_import_status
        ),
        library_skills_import_review_name_accessor=(
            lambda: screen._library_skills_import_review_name
        ),
        library_skills_import_in_flight_accessor=(
            lambda: screen._library_skills_import_in_flight
        ),
        library_skills_import_generation_accessor=(
            lambda: screen._library_skills_import_generation
        ),
        set_library_skills_import_generation=(
            lambda value: setattr(screen, "_library_skills_import_generation", value)
        ),
        approve_library_skill_trust=(
            lambda *a, **k: screen._approve_library_skill_trust(*a, **k)
        ),
        begin_library_skill_save=(
            lambda *a, **k: screen._begin_library_skill_save(*a, **k)
        ),
        build_library_skills_state=lambda: screen._build_library_skills_state(),
        call_library_skill_trust_service=(
            lambda *a, **k: screen._call_library_skill_trust_service(*a, **k)
        ),
        exit_library_skill_editor_guarded=(
            lambda *a, **k: screen._exit_library_skill_editor_guarded(*a, **k)
        ),
        persist_library_skill_editor_mode=(
            lambda *a, **k: screen._persist_library_skill_editor_mode(*a, **k)
        ),
        refresh_library_skills_trust_posture=(
            lambda: screen._refresh_library_skills_trust_posture()
        ),
        request_library_skills_browse=(
            lambda *a, **k: screen._request_library_skills_browse(*a, **k)
        ),
        reset_library_skill_editor_state=(
            lambda *a, **k: screen._reset_library_skill_editor_state(*a, **k)
        ),
        start_library_skills_import=(lambda: screen._start_library_skills_import()),
    )
