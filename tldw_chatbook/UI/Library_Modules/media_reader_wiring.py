"""Explicit call-time bindings for the Library media Reader controller."""

from typing import Any

from .library_media_reader_controller import LibraryMediaReaderController


def build_library_media_reader_controller(screen: Any) -> LibraryMediaReaderController:
    """Bind current shell state and sibling operations when each port is called."""
    return LibraryMediaReaderController(
        screen=screen,
        app_instance=screen.app_instance,
        library_media_backing_id=lambda *args, **kwargs: (
            screen._library_media_backing_id(*args, **kwargs)
        ),
        library_media_detail=lambda: screen._library_media_detail,
        library_media_editing_analysis=lambda: screen._library_media_editing_analysis,
        library_media_generating_analysis=lambda: (
            screen._library_media_generating_analysis
        ),
        library_media_reader_session=lambda: screen._library_media_reader_session,
        library_media_view=lambda: screen._library_media_view,
        mounted_library_media_viewer=lambda *args, **kwargs: (
            screen._mounted_library_media_viewer(*args, **kwargs)
        ),
        refresh_library_media_detail=lambda *args, **kwargs: (
            screen._refresh_library_media_detail(*args, **kwargs)
        ),
        run_library_service_call=lambda *args, **kwargs: (
            screen._run_library_service_call(*args, **kwargs)
        ),
        selected_media_id=lambda: screen._selected_media_id,
        sync_library_media_viewer_or_recompose=lambda *args, **kwargs: (
            screen._sync_library_media_viewer_or_recompose(*args, **kwargs)
        ),
    )
