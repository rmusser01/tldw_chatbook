from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState


def test_media_runtime_state_defaults_to_local_backend():
    state = MediaRuntimeState()

    assert state.runtime_backend == "local"
    assert state.selected_record_id is None
    assert state.search_term == ""


def test_media_runtime_state_backend_reset_clears_selection_and_caches():
    state = MediaRuntimeState(
        runtime_backend="local",
        selected_record_id="local:media:7",
        browse_items=[{"id": "local:media:7"}],
        reading_progress_by_record_id={"local:media:7": {"current_page": 3}},
        ingestion_source_items_by_id={"server:ingestion_source:2": [{"id": 1}]},
    )

    state.reset_for_backend("server")

    assert state.runtime_backend == "server"
    assert state.selected_record_id is None
    assert state.browse_items == []
    assert state.reading_progress_by_record_id == {}
    assert state.ingestion_source_items_by_id == {}
