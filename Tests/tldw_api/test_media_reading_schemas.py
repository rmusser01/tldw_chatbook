import pytest

from tldw_chatbook.tldw_api import (
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ReadingProgressUpdate,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchUpdateRequest,
    ReadingUpdateRequest,
)


def test_reading_update_request_strips_tag_whitespace():
    payload = ReadingUpdateRequest(status="read", favorite=True, tags=[" ai ", "priority "])
    assert payload.status == "read"
    assert payload.favorite is True
    assert payload.tags == ["ai", "priority"]


def test_file_create_request_requires_persist_true():
    request = FileCreateRequest(
        file_type="markdown_table",
        payload={"headers": ["A"], "rows": [["1"]]},
        options=FileCreateOptions(persist=True),
    )
    assert request.options.persist is True


def test_ingestion_source_patch_rejects_extra_fields():
    with pytest.raises(Exception):
        IngestionSourcePatchRequest(unsupported=True)


def test_ingestion_source_create_defaults():
    request = IngestionSourceCreateRequest(source_type="local_directory", sink_type="media")
    assert request.enabled is True
    assert request.schedule_enabled is False


def test_reading_progress_update_serializes_mode_default():
    payload = ReadingProgressUpdate(current_page=3, total_pages=10)
    assert payload.view_mode == "single"


def test_reading_saved_search_normalizes_query_and_sort():
    request = ReadingSavedSearchCreateRequest(
        name=" Morning ",
        query={"status": [" saved "], "tags": [" ai "], "favorite": True, "sort": "UPDATED_DESC"},
        sort="created_desc",
    )

    assert request.name == "Morning"
    assert request.query == {
        "status": ["saved"],
        "tags": ["ai"],
        "favorite": True,
        "sort": "updated_desc",
    }
    assert request.sort == "created_desc"


def test_reading_saved_search_rejects_unsupported_query_keys():
    with pytest.raises(Exception, match="unsupported_query_key"):
        ReadingSavedSearchCreateRequest(name="Bad", query={"workspace_id": "ws-1"})


def test_reading_saved_search_update_allows_sparse_patch():
    request = ReadingSavedSearchUpdateRequest(query={"status": "read"})

    assert request.name is None
    assert request.query == {"status": "read"}
    assert request.sort is None
