import pytest

from tldw_chatbook.tldw_api import (
    DocumentAnnotationCreateRequest,
    DocumentAnnotationListResponse,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationSyncResponse,
    DocumentAnnotationUpdateRequest,
    DocumentFiguresResponse,
    DocumentOutlineResponse,
    DocumentVersionCreateRequest,
    DocumentVersionDetailResponse,
    FileCreateOptions,
    FileCreateRequest,
    ItemsBulkRequest,
    ItemsBulkResponse,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ReadingDigestOutput,
    ReadingDigestOutputsListResponse,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleFilters,
    ReadingDigestScheduleResponse,
    ReadingDigestScheduleUpdateRequest,
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


def test_document_workspace_read_models_match_server_contracts():
    outline = DocumentOutlineResponse(
        media_id=99,
        has_outline=True,
        entries=[{"level": 1, "title": "Intro", "page": 1}],
        total_pages=10,
    )
    figures = DocumentFiguresResponse(
        media_id=99,
        has_figures=True,
        figures=[
            {
                "id": "fig_1_0",
                "page": 1,
                "width": 640,
                "height": 480,
                "format": "png",
                "data_url": "data:image/png;base64,abc",
            }
        ],
        total_count=1,
    )

    assert outline.entries[0].title == "Intro"
    assert figures.figures[0].data_url == "data:image/png;base64,abc"


def test_document_annotation_requests_and_responses_match_server_contracts():
    create = DocumentAnnotationCreateRequest(
        location="page:1",
        text="Highlighted text",
        color="blue",
        note="Important",
        annotation_type="highlight",
        percentage=42.5,
    )
    update = DocumentAnnotationUpdateRequest(text="Updated", color="green", note="Changed")
    listed = DocumentAnnotationListResponse(
        media_id=99,
        annotations=[
            {
                "id": "ann_1",
                "media_id": 99,
                "location": "page:1",
                "text": "Highlighted text",
                "color": "blue",
                "annotation_type": "highlight",
                "created_at": "2026-04-23T12:00:00Z",
                "updated_at": "2026-04-23T12:00:00Z",
            }
        ],
        total_count=1,
    )
    sync = DocumentAnnotationSyncRequest(annotations=[create], client_ids=["client-1"])
    sync_response = DocumentAnnotationSyncResponse(
        media_id=99,
        synced_count=1,
        annotations=listed.annotations,
        id_mapping={"client-1": "ann_1"},
    )

    assert create.model_dump(exclude_none=True, mode="json") == {
        "location": "page:1",
        "text": "Highlighted text",
        "color": "blue",
        "note": "Important",
        "annotation_type": "highlight",
        "percentage": 42.5,
    }
    assert update.model_dump(exclude_none=True, mode="json") == {
        "text": "Updated",
        "color": "green",
        "note": "Changed",
    }
    assert listed.annotations[0].id == "ann_1"
    assert sync.model_dump(exclude_none=True, mode="json")["client_ids"] == ["client-1"]
    assert sync_response.id_mapping == {"client-1": "ann_1"}


def test_document_version_request_and_response_match_server_contract():
    request = DocumentVersionCreateRequest(
        content="Source text",
        prompt="Analyze this",
        analysis_content="Analysis",
        safe_metadata={"kind": "analysis"},
    )
    response = DocumentVersionDetailResponse(
        uuid="version-uuid",
        media_id=99,
        version_number=2,
        created_at="2026-04-23T12:00:00Z",
        prompt="Analyze this",
        analysis_content="Analysis",
        safe_metadata={"kind": "analysis"},
        content="Source text",
    )

    assert request.model_dump(exclude_none=True, mode="json") == {
        "content": "Source text",
        "prompt": "Analyze this",
        "analysis_content": "Analysis",
        "safe_metadata": {"kind": "analysis"},
    }
    assert response.uuid == "version-uuid"
    assert response.media_id == 99
    assert response.version_number == 2


def test_reading_update_request_strips_tag_whitespace():
    payload = ReadingUpdateRequest(status="read", favorite=True, tags=[" ai ", "priority "])
    assert payload.status == "read"
    assert payload.favorite is True
    assert payload.tags == ["ai", "priority"]


def test_items_bulk_request_matches_server_contract():
    request = ItemsBulkRequest(
        item_ids=[31, 31, 32],
        action="add_tags",
        tags=[" ai ", "research"],
    )
    response = ItemsBulkResponse(
        total=2,
        succeeded=1,
        failed=1,
        results=[
            {"item_id": 31, "success": True},
            {"item_id": 32, "success": False, "error": "item_not_found"},
        ],
    )

    assert request.model_dump(exclude_none=True, mode="json") == {
        "item_ids": [31, 31, 32],
        "action": "add_tags",
        "tags": ["ai", "research"],
        "hard": False,
    }
    assert response.results[1].error == "item_not_found"


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


def test_reading_digest_schedule_contract_matches_server_shapes():
    request = ReadingDigestScheduleCreateRequest(
        name="Morning Digest",
        cron="0 8 * * *",
        timezone="UTC",
        filters={
            "status": "saved",
            "tags": "ai",
            "suggestions": {"enabled": True, "limit": 5, "status": "reading"},
        },
    )
    update = ReadingDigestScheduleUpdateRequest(enabled=False)
    response = ReadingDigestScheduleResponse(
        id="sched-1",
        name="Morning Digest",
        cron="0 8 * * *",
        timezone="UTC",
        enabled=True,
        require_online=False,
        format="md",
        filters=ReadingDigestScheduleFilters(status=["saved"], tags=["ai"]),
    )
    outputs = ReadingDigestOutputsListResponse(
        items=[
            ReadingDigestOutput(
                output_id=77,
                title="Morning Digest",
                format="md",
                created_at="2026-04-23T12:00:00Z",
                download_url="/api/v1/outputs/77/download",
                schedule_id="sched-1",
                schedule_name="Morning Digest",
                item_count=3,
            )
        ],
        total=1,
        limit=25,
        offset=5,
    )

    assert request.model_dump(exclude_none=True, mode="json") == {
        "name": "Morning Digest",
        "cron": "0 8 * * *",
        "timezone": "UTC",
        "enabled": True,
        "require_online": False,
        "format": "md",
        "filters": {
            "status": ["saved"],
            "tags": ["ai"],
            "suggestions": {
                "enabled": True,
                "limit": 5,
                "status": ["reading"],
                "include_read": False,
                "include_archived": False,
            },
        },
    }
    assert update.model_dump(exclude_none=True, mode="json") == {"enabled": False}
    assert response.filters.status == ["saved"]
    assert outputs.items[0].download_url == "/api/v1/outputs/77/download"
