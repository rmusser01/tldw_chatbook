"""Task 8: rename meeting speakers after the meeting, on the Library media item."""


def test_rename_after_rewrites_content_and_reindexes(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in row["content"]
    hits, _total = tmp_media_db.search_media_db(search_query="Alice")
    assert any(h["id"] == media_id for h in hits)


def test_rename_after_disabled_when_folder_gone(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    import shutil; shutil.rmtree(folder)
    from tldw_chatbook.Widgets.Library.library_media_canvas import can_rename_meeting_speakers
    assert can_rename_meeting_speakers(tmp_media_db, media_id) is False
