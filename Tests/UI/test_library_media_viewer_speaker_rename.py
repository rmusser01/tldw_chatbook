"""TASK-31745: rename meeting speakers from the LIVE media reader.

The rename legend shipped inside ``LibraryMediaCanvas``' preview sub-pane,
which is ``display: none`` app-wide -- so the feature existed and could not
be reached. These pin it on ``LibraryMediaViewer``, the surface a user
actually reads a media item in.
"""

import dataclasses

import pytest
from textual.widgets import Input, Static

from tldw_chatbook.Library.library_media_viewer_state import (
    build_library_media_viewer_state,
)

def _library_screen(tmp_media_db):
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    app.media_db = tmp_media_db
    return LibraryScreen(app)


def _viewer_app(tmp_media_db, media_id, *, rows=(("S1", "Speaker 1"),)):
    """A one-widget harness holding the real viewer over a real meeting item.

    The app records every ``SpeakerRenamed`` that bubbles out of the viewer.
    """
    from textual import on

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer

    detail = dict(tmp_media_db.get_media_by_id(media_id))
    state = dataclasses.replace(
        build_library_media_viewer_state(detail),
        can_rename_speakers=True,
        speaker_legend_rows=tuple(rows),
    )

    class _App(ConsolidatedCSSApp):
        def __init__(self):
            super().__init__()
            self.renamed_ids: list[int] = []

        def compose(self):
            yield LibraryMediaViewer(
                state,
                media_db=tmp_media_db,
                speaker_rename_media_id=media_id,
                id="library-media-viewer",
            )

        @on(LibraryMediaViewer.SpeakerRenamed)
        def _record(self, event) -> None:
            self.renamed_ids.append(event.media_id)

    return _App()


def test_viewer_state_carries_the_rename_legend_for_a_meeting(
    tmp_media_db, meeting_folder_media_item
):
    """The screen's viewer state, not just the canvas presentation, knows the
    selected item is a renameable meeting and what its speakers are."""
    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    screen = _library_screen(tmp_media_db)
    screen._selected_media_id = f"local:media:{media_id}"

    state = screen._build_library_media_viewer_display_state(
        dict(tmp_media_db.get_media_by_id(media_id))
    )

    assert state.can_rename_speakers is True
    assert state.speaker_legend_rows == (("S1", "Speaker 1"),)


def test_viewer_state_has_no_legend_for_a_plain_document(tmp_media_db):
    media_id, _uuid, _msg = tmp_media_db.add_media_with_keywords(
        title="A plain document", media_type="article", content="not a meeting",
    )
    screen = _library_screen(tmp_media_db)
    screen._selected_media_id = f"local:media:{media_id}"

    state = screen._build_library_media_viewer_display_state(
        dict(tmp_media_db.get_media_by_id(media_id))
    )

    assert state.can_rename_speakers is False
    assert state.speaker_legend_rows == ()


@pytest.mark.asyncio
async def test_reader_renders_one_legend_row_per_speaker(
    tmp_media_db, meeting_folder_media_item
):
    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])

    app = _viewer_app(tmp_media_db, media_id)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        label = app.query_one("#library-media-speaker-label-S1", Static)
        assert str(label.renderable) == "Speaker 1"
        assert app.query_one("#library-media-speaker-input-S1", Input) is not None


@pytest.mark.asyncio
async def test_legend_container_is_height_auto_not_a_half_pane_vertical(
    tmp_media_db, meeting_folder_media_item
):
    """A bare ``Vertical`` here is ``height: 1fr`` with no rule to say
    otherwise, so as a sibling of the ``1fr`` content body the legend would
    claim HALF the reading pane (the task-31222/31276 trap)."""
    from textual.containers import VerticalGroup

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])

    app = _viewer_app(tmp_media_db, media_id)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        legend = app.query_one("#library-media-speaker-legend")
        assert isinstance(legend, VerticalGroup)
        assert legend.styles.height.is_auto
        body = app.query_one("#library-media-viewer-content")
        assert legend.size.height < body.size.height


@pytest.mark.asyncio
async def test_reader_rename_rewrites_the_item_and_refreshes_the_shown_content(
    tmp_media_db, meeting_folder_media_item
):
    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])

    app = _viewer_app(tmp_media_db, media_id)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#library-media-speaker-input-S1", Input)
        input_widget.post_message(Input.Submitted(input_widget, "Alice"))
        await pilot.pause()
        # The write runs on a thread worker; wait for it rather than racing it.
        await app.workers.wait_for_complete()
        await pilot.pause()

        viewer = app.query_one("#library-media-viewer")
        assert "Alice:" in viewer.viewer.content
        label = app.query_one("#library-media-speaker-label-S1", Static)
        assert str(label.renderable) == "Alice"
        # The screen needs to know WHICH item to re-read, not just that
        # something was renamed.
        assert app.renamed_ids == [media_id]

    assert "Alice:" in tmp_media_db.get_media_by_id(media_id)["content"]


def test_a_later_viewer_sync_keeps_the_new_name(
    tmp_media_db, meeting_folder_media_item
):
    """The regression the ``SpeakerRenamed`` seam exists for.

    The screen memoizes viewer state by detail IDENTITY, so after a rename
    the next sync rebuilds from the PRE-rename detail and repaints the old
    speaker name over the new one. Negative control first: without the
    handler the stale text is exactly what comes back.
    """
    from tldw_chatbook.Library.meeting_speaker_rename import rename_meeting_speaker
    from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    screen = _library_screen(tmp_media_db)
    screen._selected_media_id = f"local:media:{media_id}"
    screen._library_media_detail = dict(tmp_media_db.get_media_by_id(media_id))
    assert "Speaker 1:" in screen._build_library_media_viewer_display_state(
        screen._library_media_detail
    ).content

    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")

    # Negative control: the memo still holds the pre-rename render.
    assert "Speaker 1:" in screen._build_library_media_viewer_display_state(
        screen._library_media_detail
    ).content

    screen._handle_library_media_speaker_renamed(
        LibraryMediaViewer.SpeakerRenamed(media_id)
    )

    refreshed = screen._build_library_media_viewer_display_state(
        screen._library_media_detail
    )
    assert "Alice:" in refreshed.content
    assert refreshed.speaker_legend_rows == (("S1", "Alice"),)


def test_a_rename_for_another_item_leaves_the_loaded_detail_alone(
    tmp_media_db, meeting_folder_media_item
):
    """The message names an id; a stale one (selection moved on) is ignored."""
    from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    screen = _library_screen(tmp_media_db)
    screen._selected_media_id = f"local:media:{media_id}"
    detail = dict(tmp_media_db.get_media_by_id(media_id))
    screen._library_media_detail = detail

    screen._handle_library_media_speaker_renamed(
        LibraryMediaViewer.SpeakerRenamed(media_id + 999)
    )

    assert screen._library_media_detail is detail


@pytest.mark.asyncio
async def test_reader_rename_refusal_explains_and_touches_nothing(
    tmp_media_db, meeting_folder_media_item
):
    """Ingest-produced content is never overwritten (fix C2); the reader says
    so in static copy instead of silently doing nothing."""
    media_id, _folder = meeting_folder_media_item(
        names={},
        segments=[("S1", "hello")],
        content="A much better offline transcription from the ingest pipeline.",
    )
    before = tmp_media_db.get_media_by_id(media_id)["content"]

    app = _viewer_app(tmp_media_db, media_id)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#library-media-speaker-input-S1", Input)
        input_widget.post_message(Input.Submitted(input_widget, "Alice"))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        viewer = app.query_one("#library-media-viewer")
        assert viewer.viewer.content == before

    assert tmp_media_db.get_media_by_id(media_id)["content"] == before
    assert notices, "a refused rename must say why"
    # Static copy only -- never the item's own transcript/name/path text.
    assert (
        "This transcript came from ingest; rename the live transcript in Meetings."
        in notices[0]
    )
