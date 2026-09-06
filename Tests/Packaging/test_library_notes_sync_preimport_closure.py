"""ADR-097: Library discovery must not load Notes synchronization execution."""

from Tests.Packaging.test_chat_persistence_import_closure import _run_isolated_python


def test_library_defers_notes_sync_until_screen_construction(tmp_path):
    result = _run_isolated_python(
        tmp_path,
        """
import sys
import tldw_chatbook.app
import tldw_chatbook.UI.Screens.chat_screen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

deferred = (
    'tldw_chatbook.UI.Library_Modules.library_notes_sync_controller',
    'tldw_chatbook.Notes.notes_sync_runtime',
    'tldw_chatbook.Notes.notes_sync_executor',
    'tldw_chatbook.Notes.notes_sync_coordinator',
)
resident = [name for name in deferred if name in sys.modules]
assert not resident, 'Unvisited Notes sync payload: ' + repr(resident)
assert LibraryScreen.__module__ in sys.modules

from Tests.UI.app_factory import (
    _build_test_app, drain_active_service_patches, drain_created_dirs,
)
try:
    app = _build_test_app()
    screen = LibraryScreen(app)
    from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
        LibraryNotesSyncController,
    )
    assert type(screen._library_notes_sync_controller) is LibraryNotesSyncController
    assert screen._library_notes_sync_controller._runtime is app.notes_sync_runtime_owner
    assert all(name in sys.modules for name in deferred)
    print('LIBRARY_NOTES_SYNC_FIRST_USE_OK')
finally:
    drain_active_service_patches()
    drain_created_dirs()
""",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "LIBRARY_NOTES_SYNC_FIRST_USE_OK" in result.stdout
