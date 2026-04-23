import pytest

from tldw_chatbook.tldw_api import (
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ReadingProgressUpdate,
    ReadingArchiveCreateRequest,
    ReadingExportRequest,
    ReadingImportJobStatus,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchUpdateRequest,
    ReadingSummarizeRequest,
    ReadingSummaryResponse,
    ReadingTTSRequest,
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


def test_reading_archive_create_defaults_to_html_auto_source():
    request = ReadingArchiveCreateRequest()

    assert request.format == "html"
    assert request.source == "auto"


def test_reading_import_job_status_accepts_server_payload():
    status = ReadingImportJobStatus(
        job_id=42,
        job_uuid="job-uuid-42",
        status="completed",
        progress_percent=100,
        result={"source": "pocket", "imported": 3, "updated": 1, "skipped": 0, "errors": []},
    )

    assert status.job_id == 42
    assert status.result["imported"] == 3


def test_reading_export_request_matches_server_filters_and_defaults():
    request = ReadingExportRequest(
        status=["saved"],
        tags=["ai"],
        favorite=True,
        q="rag",
        include_text=True,
        format="zip",
    )

    assert request.model_dump(exclude_none=True, mode="json") == {
        "status": ["saved"],
        "tags": ["ai"],
        "favorite": True,
        "q": "rag",
        "page": 1,
        "size": 1000,
        "include_metadata": True,
        "include_clean_html": False,
        "include_text": True,
        "include_highlights": False,
        "include_notes": True,
        "format": "zip",
    }


def test_reading_summarize_request_and_response_match_server_contract():
    request = ReadingSummarizeRequest(
        provider="openai",
        model="gpt-4o-mini",
        prompt="Summarize for a product brief.",
        temperature=0.4,
        recursive=True,
    )
    response = ReadingSummaryResponse(
        item_id=31,
        summary="Short summary",
        provider="openai",
        model="gpt-4o-mini",
        citations=[
            {
                "item_id": 31,
                "url": "https://example.com",
                "canonical_url": "https://example.com",
                "title": "Example",
                "source": "reading",
            }
        ],
        generated_at="2026-04-23T12:00:00Z",
    )

    assert request.model_dump(exclude_none=True, mode="json") == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt": "Summarize for a product brief.",
        "temperature": 0.4,
        "recursive": True,
        "chunked": False,
    }
    assert response.citations[0].source == "reading"


def test_reading_tts_request_matches_server_contract():
    request = ReadingTTSRequest(
        model="kokoro",
        voice="af_heart",
        response_format="mp3",
        stream=False,
        speed=1.1,
        max_chars=12000,
        text_source="text",
    )

    assert request.model_dump(exclude_none=True, mode="json") == {
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "mp3",
        "stream": False,
        "speed": 1.1,
        "max_chars": 12000,
        "text_source": "text",
    }
