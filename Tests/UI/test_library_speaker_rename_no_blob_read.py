"""Speaker rename patches detail from the message, without a blob re-read."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer

pytestmark = pytest.mark.bootstrap_profile


class _RenameScreen(LibraryScreen):
    """Bare shell: only the rename handler's collaborators are stubbed."""

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen(db_reads):
    screen = _RenameScreen()
    screen._media_state = SimpleNamespace(
        detail={"content": "old text", "title": "kept"}
    )

    def backing_id():
        return 7

    screen._library_media_selected_backing_id = backing_id

    class _DB:
        def get_media_by_id(self, media_id):
            db_reads.append(media_id)
            return {"content": "db text"}

    screen.app_instance = SimpleNamespace(media_db=_DB())
    return screen


def test_rename_with_content_on_the_message_skips_the_db():
    db_reads: list[int] = []
    screen = _screen(db_reads)
    event = LibraryMediaViewer.SpeakerRenamed(7, content="renamed text")
    event.stop = lambda: None

    screen._handle_library_media_speaker_renamed(event)

    assert not db_reads, "no full-content SELECT on the rename path"
    assert screen._media_state.detail["content"] == "renamed text"
    assert screen._media_state.detail["title"] == "kept"


def test_rename_without_content_falls_back_to_the_db_read():
    db_reads: list[int] = []
    screen = _screen(db_reads)
    event = LibraryMediaViewer.SpeakerRenamed(7)  # legacy shape: no content
    event.stop = lambda: None

    screen._handle_library_media_speaker_renamed(event)

    assert db_reads == [7]
    assert screen._media_state.detail["content"] == "db text"
