import pytest

from tldw_chatbook.tldw_api import (
    DocumentAnnotationCreateRequest,
    DocumentAnnotationListResponse,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationSyncResponse,
    DocumentAnnotationUpdateRequest,
    DocumentFiguresResponse,
    DocumentInsightsRequest,
    DocumentInsightsResponse,
    DocumentOutlineResponse,
    DocumentReferencesResponse,
    DocumentVersionCreateRequest,
    DocumentVersionDetailResponse,
    FileCreateOptions,
    FileCreateRequest,
    ItemsBulkRequest,
    ItemsBulkResponse,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    MediaDetailResponse,
    MediaIdentifierLookupResponse,
    MediaKeywordsResponse,
    MediaKeywordListResponse,
    MediaKeywordsUpdateRequest,
    MediaMetadataSearchResponse,
    MediaNavigationContentResponse,
    MediaNavigationResponse,
    MediaSearchRequest,
    MediaTrashEmptyResponse,
    MediaUpdateRequest,
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
    ServerMediaListResponse,
)


def test_server_media_listing_and_search_adjunct_models_match_server_contracts():
    listing = ServerMediaListResponse(
        items=[
            {
                "id": 99,
                "title": "Paper",
                "url": "/api/v1/media/99",
                "type": "pdf",
                "keywords": [" ai ", "testing"],
            }
        ],
        pagination={
            "page": 1,
            "results_per_page": 20,
            "total_pages": 1,
            "total_items": 1,
        },
        keywords_available=True,
    )
    keywords = MediaKeywordListResponse(keywords=[" ai ", "testing"])
    metadata_search = MediaMetadataSearchResponse(
        results=[
            {
                "media_id": 99,
                "version_number": 1,
                "safe_metadata": {"doi": "10/example"},
            }
        ],
        pagination={"page": 1, "per_page": 20, "total": 1, "total_pages": 1},
    )
    identifier_lookup = MediaIdentifierLookupResponse(
        results=[{"media_id": 99, "safe_metadata": {"doi": "10/example"}}],
        total=1,
    )
    empty_trash = MediaTrashEmptyResponse(
        deleted_count=2,
        failed_count=0,
        failed_ids=[],
        remaining_count=0,
    )

    assert listing.items[0].keywords == ["ai", "testing"]
    assert listing.pagination.total_items == 1
    assert keywords.keywords == ["ai", "testing"]
    assert metadata_search.results[0]["safe_metadata"]["doi"] == "10/example"
    assert identifier_lookup.total == 1
    assert empty_trash.deleted_count == 2


def test_media_navigation_models_match_server_contracts():
    navigation = MediaNavigationResponse(
        media_id=99,
        available=True,
        navigation_version="nav-v1",
        source_order_used=["pdf_outline"],
        nodes=[
            {
                "id": "node-1",
                "parent_id": None,
                "level": 1,
                "title": "Chapter 1",
                "order": 0,
                "path_label": "1",
                "target_type": "page",
                "target_start": 1,
                "source": "pdf_outline",
                "confidence": 0.9,
            }
        ],
        stats={
            "returned_node_count": 1,
            "node_count": 1,
            "max_depth": 1,
            "truncated": False,
        },
    )
    content = MediaNavigationContentResponse(
        media_id=99,
        node_id="node-1",
        title="Chapter 1",
        content_format="markdown",
        available_formats=["plain", "markdown"],
        content="# Chapter 1\n\nBody",
        alternate_content={"plain": "Chapter 1\n\nBody"},
        target={"target_type": "page", "target_start": 1},
    )

    assert navigation.nodes[0].target_type == "page"
    assert navigation.stats.returned_node_count == 1
    assert content.content_format == "markdown"
    assert content.available_formats == ["plain", "markdown"]


def test_media_item_models_match_server_contracts():
    detail = MediaDetailResponse(
        media_id=99,
        source={
            "url": "https://example.com/paper.pdf",
            "title": "Paper",
            "duration": None,
            "type": "pdf",
        },
        processing={
            "prompt": "Summarize",
            "analysis": "Analysis",
            "safe_metadata": {"doi": "10/example"},
            "model": "local",
            "timestamp_option": False,
            "chunking_status": "completed",
            "vector_processing_status": 1,
        },
        content={
            "metadata": {"pages": 3},
            "text": "Body",
            "word_count": 1,
        },
        keywords=[" ai ", "testing"],
        timestamps=[],
        versions=[
            {
                "uuid": "version-1",
                "media_id": 99,
                "version_number": 1,
                "created_at": "2026-04-23T12:00:00Z",
                "prompt": "Summarize",
                "analysis_content": "Analysis",
                "content": None,
            }
        ],
        has_original_file=True,
        original_file_url="/api/v1/media/99/file",
    )
    update = MediaUpdateRequest(
        title=" New title ",
        content="Body 2",
        author="Ada",
        analysis="Analysis 2",
        prompt="Prompt 2",
    )
    keywords_update = MediaKeywordsUpdateRequest(keywords=[" ai ", "ml"], mode="set")
    keywords_response = MediaKeywordsResponse(media_id=99, keywords=["ai", "ml"])

    assert detail.media_id == 99
    assert detail.source.title == "Paper"
    assert detail.keywords == ["ai", "testing"]
    assert detail.versions[0].version_number == 1
    assert update.model_dump(exclude_none=True, mode="json") == {
        "title": "New title",
        "content": "Body 2",
        "author": "Ada",
        "analysis": "Analysis 2",
        "prompt": "Prompt 2",
    }
    assert keywords_update.model_dump(mode="json") == {"keywords": ["ai", "ml"], "mode": "set"}
    assert keywords_response.keywords == ["ai", "ml"]


def test_document_insights_and_references_models_match_server_contracts():
    insights_request = DocumentInsightsRequest(
        categories=["summary", "key_findings"],
        model="gpt-4o-mini",
        max_content_length=2000,
        force=True,
    )
    insights_response = DocumentInsightsResponse(
        media_id=99,
        insights=[
            {
                "category": "summary",
                "title": "Short summary",
                "content": "The document discusses testing.",
                "confidence": 0.9,
            }
        ],
        model_used="gpt-4o-mini",
        cached=False,
    )
    references_response = DocumentReferencesResponse(
        media_id=99,
        has_references=True,
        references=[
            {
                "raw_text": "Doe, J. 2024. Testing systems.",
                "title": "Testing systems",
                "authors": "Doe, J.",
                "year": 2024,
            }
        ],
        enrichment_source="semantic_scholar",
        enriched_count=1,
        total_detected=1,
        limit=20,
        returned_count=1,
        total_available=1,
    )

    assert insights_request.model_dump(exclude_none=True, mode="json") == {
        "categories": ["summary", "key_findings"],
        "model": "gpt-4o-mini",
        "max_content_length": 2000,
        "force": True,
    }
    assert insights_response.insights[0].category == "summary"
    assert references_response.references[0].title == "Testing systems"
    assert references_response.has_more is False


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
