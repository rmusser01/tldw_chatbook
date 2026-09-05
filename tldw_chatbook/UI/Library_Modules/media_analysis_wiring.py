"""Explicit late-binding wiring for the Library media analysis controller."""

from typing import Any

from .library_media_analysis_controller import LibraryMediaAnalysisController


def build_library_media_analysis_controller(
    screen: Any,
) -> LibraryMediaAnalysisController:
    """Bind each sibling port at call time, preserving screen monkeypatch seams."""
    return LibraryMediaAnalysisController(
        screen=screen,
        app_instance=screen.app_instance,
        build_library_media_state=lambda *args, **kwargs: (
            screen._build_library_media_state(*args, **kwargs)
        ),
        dispatch_library_media_analysis=lambda: screen._dispatch_library_media_analysis,
        exit_library_media_select_mode=lambda *args, **kwargs: (
            screen._exit_library_media_select_mode(*args, **kwargs)
        ),
        library_canvas_projection_depth=lambda: screen._library_canvas_projection_depth,
        library_canvas_resync_pending=lambda: screen._library_canvas_resync_pending,
        set_library_canvas_resync_pending=lambda value: setattr(
            screen, "_library_canvas_resync_pending", value
        ),
        library_media_analysis_provider_reason=lambda *args, **kwargs: (
            screen._library_media_analysis_provider_reason(*args, **kwargs)
        ),
        library_media_analyze_running=lambda: screen._library_media_analyze_running,
        set_library_media_analyze_running=lambda value: setattr(
            screen, "_library_media_analyze_running", value
        ),
        library_media_backing_id=lambda *args, **kwargs: (
            screen._library_media_backing_id(*args, **kwargs)
        ),
        library_media_bulk_delete_in_flight=lambda: (
            screen._library_media_bulk_delete_in_flight
        ),
        library_media_canvas_presentation=lambda *args, **kwargs: (
            screen._library_media_canvas_presentation(*args, **kwargs)
        ),
        library_media_detail=lambda: screen._library_media_detail,
        library_media_select_mode=lambda: screen._library_media_select_mode,
        refresh_library_media_detail=lambda *args, **kwargs: (
            screen._refresh_library_media_detail(*args, **kwargs)
        ),
        run_library_service_call=lambda *args, **kwargs: (
            screen._run_library_service_call(*args, **kwargs)
        ),
        sanitize_media_field=lambda *args, **kwargs: screen._sanitize_media_field(
            *args, **kwargs
        ),
        selected_media_id=lambda: screen._selected_media_id,
        set_selected_media_id=lambda value: setattr(
            screen, "_selected_media_id", value
        ),
        sync_library_media_viewer_or_recompose=lambda *args, **kwargs: (
            screen._sync_library_media_viewer_or_recompose(*args, **kwargs)
        ),
        update_library_ingest_dynamic_regions=lambda *args, **kwargs: (
            screen._update_library_ingest_dynamic_regions(*args, **kwargs)
        ),
    )
