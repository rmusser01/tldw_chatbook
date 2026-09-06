"""Task 8: rename meeting speakers after the meeting, on the Library media item."""

import pytest
from textual.widgets import Input, Static


def test_rename_after_rewrites_content_and_reindexes(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in row["content"]
    hits, _total = tmp_media_db.search_media_db(search_query="Alice")
    assert any(h["id"] == media_id for h in hits)


def test_rename_survives_a_soft_deleted_prior_transcript_row(tmp_media_db, meeting_folder_media_item):
    """M2: a soft-deleted prior Transcripts row must not collide the rename's
    INSERT -- UNIQUE(media_id, whisper_model) counts deleted rows, so the old
    `WHERE deleted = 0` lookup missed it and the INSERT hit a caught
    IntegrityError that silently dropped the rename."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")   # creates the Transcripts row

    # Soft-delete that row (as a prior delete would), keeping the sync trigger
    # happy: version increments by exactly 1, client_id stays non-empty.
    with tmp_media_db.transaction() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, version, client_id FROM Transcripts WHERE media_id=?", (media_id,))
        prior = cur.fetchone()
        cur.execute(
            "UPDATE Transcripts SET deleted=1, version=?, client_id=? WHERE id=?",
            (prior["version"] + 1, prior["client_id"], prior["id"]),
        )

    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Bob")     # must not silently fail
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Bob:" in row["content"]
    hits, _total = tmp_media_db.search_media_db(search_query="Bob")
    assert any(h["id"] == media_id for h in hits)


def test_rename_after_disabled_when_folder_gone(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    import shutil; shutil.rmtree(folder)
    from tldw_chatbook.Widgets.Library.library_media_canvas import can_rename_meeting_speakers
    assert can_rename_meeting_speakers(tmp_media_db, media_id) is False


def test_rename_to_empty_name_removes_map_entry_and_drops_the_label(
    tmp_media_db, meeting_folder_media_item
):
    """Empty-name removal via the public path (review item 4)."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker

    media_id, folder = meeting_folder_media_item(names={"S1": "Alice"}, segments=[("S1", "hello")])
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "")

    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice" not in row["content"]
    assert "Speaker 1:" in row["content"]

    import json
    meeting = json.loads((folder / "meeting.json").read_text())
    assert "S1" not in meeting["speaker_names"]


def test_presentation_reachability_reflects_meeting_folder(
    tmp_media_db, meeting_folder_media_item
):
    """Review item 1: ``_library_media_canvas_presentation`` actually drives
    ``can_rename_speakers`` from the real DB/selection, not a stub default."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from Tests.UI.app_factory import _build_test_app

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    other_id, _uuid, _msg = tmp_media_db.add_media_with_keywords(
        title="A plain document", media_type="article", content="not a meeting",
    )

    app = _build_test_app()
    app.media_db = tmp_media_db
    screen = LibraryScreen(app)

    screen._selected_media_id = f"local:media:{media_id}"
    presentation = screen._library_media_canvas_presentation()
    assert presentation["can_rename_speakers"] is True
    assert presentation["speaker_rename_media_id"] == media_id

    screen._selected_media_id = f"local:media:{other_id}"
    assert screen._library_media_canvas_presentation()["can_rename_speakers"] is False


@pytest.mark.asyncio
async def test_speaker_legend_submit_renames_and_refreshes_preview(
    tmp_media_db, meeting_folder_media_item
):
    """Review item 2: submitting a legend rename input calls
    ``rename_meeting_speaker`` and the canvas's own shown preview text
    picks up the new name (no page reload needed)."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState, LibraryMediaRow
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    row = tmp_media_db.get_media_by_id(media_id)
    state = LibraryMediaCanvasState(
        rows=(
            LibraryMediaRow(
                media_id=str(media_id), title="Meeting", media_type="audio",
                secondary="audio · today", selected=True,
            ),
        ),
        type_options=(None, "audio"), active_type=None, status_copy="",
        empty_copy="", selected_id=str(media_id),
        preview_lines=tuple(row["content"].splitlines()), count=1,
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(
                canvas=state, can_rename_speakers=True, media_db=tmp_media_db,
                speaker_rename_media_id=media_id, id="canvas",
            )

    app = _App()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#library-media-speaker-input-S1", Input)
        input_widget.post_message(Input.Submitted(input_widget, "Alice"))
        await pilot.pause()

        preview = app.query_one("#library-media-preview-lines", Static)
        assert "Alice" in str(preview.renderable)
        label = app.query_one("#library-media-speaker-label-S1", Static)
        assert str(label.renderable) == "Alice"

    updated = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in updated["content"]
