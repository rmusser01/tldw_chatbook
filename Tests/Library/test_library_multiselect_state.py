from tldw_chatbook.Library.library_media_state import build_library_media_state
from tldw_chatbook.Library.library_conversations_state import build_library_conversations_state
from tldw_chatbook.Library.library_notes_state import build_library_notes_list_state


def test_media_checked_and_count():
    records = [
        {"id": "1", "title": "A", "type": "video", "updated_at": None},
        {"id": "2", "title": "B", "type": "audio", "updated_at": None},
    ]
    state = build_library_media_state(records, select_mode=True, selected_ids=frozenset({"1"}))
    by_id = {r.media_id: r for r in state.rows}
    assert by_id["1"].checked is True and by_id["2"].checked is False
    assert state.select_mode is True and state.selected_count == 1


def test_media_default_no_select_mode():
    records = [{"id": "1", "title": "A", "type": "video", "updated_at": None}]
    state = build_library_media_state(records)
    assert state.select_mode is False and state.selected_count == 0
    assert state.rows[0].checked is False


def test_conversations_checked_and_count():
    records = [
        {"id": "c1", "title": "One", "message_count": 1, "updated_at": None},
        {"id": "c2", "title": "Two", "message_count": 2, "updated_at": None},
    ]
    state = build_library_conversations_state(records, select_mode=True, selected_ids=frozenset({"c2"}))
    by_id = {r.conversation_id: r for r in state.rows}
    assert by_id["c2"].checked is True and by_id["c1"].checked is False
    assert state.selected_count == 1


def test_notes_checked_and_count():
    records = [{"id": "n1", "title": "N1"}, {"id": "n2", "title": "N2"}]
    state = build_library_notes_list_state(records, select_mode=True, selected_ids=frozenset({"n1"}))
    by_id = {r.note_id: r for r in state.rows}
    assert by_id["n1"].checked is True and by_id["n2"].checked is False
    assert state.select_mode is True and state.selected_count == 1
