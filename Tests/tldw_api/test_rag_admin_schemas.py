from tldw_chatbook.tldw_api import (
    ChunkingTemplateApplyRequest,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    EmbeddingCollectionStatsResponse,
)


def test_chunking_template_create_request_round_trips():
    payload = ChunkingTemplateCreateRequest(
        name="demo",
        description="Demo template",
        tags=["rag"],
        template={
            "preprocessing": [],
            "chunking": {"method": "words", "config": {"max_size": 400}},
            "postprocessing": [],
        },
        user_id="u1",
    )

    dumped = payload.model_dump()
    assert dumped["name"] == "demo"
    assert dumped["template"]["chunking"]["method"] == "words"
    assert dumped["user_id"] == "u1"


def test_chunking_template_update_request_is_partial():
    payload = ChunkingTemplateUpdateRequest(description="Updated")
    assert payload.model_dump(exclude_none=True) == {"description": "Updated"}


def test_chunking_template_response_parses_uuid_and_timestamps():
    payload = ChunkingTemplateResponse.model_validate(
        {
            "id": 5,
            "uuid": "b9012f62-fd66-4b43-bdb7-ab2bdb36fb37",
            "name": "demo",
            "description": "Demo template",
            "template_json": {"chunking": {"method": "words", "config": {"max_size": 200}}},
            "is_builtin": False,
            "tags": ["demo"],
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:10:00Z",
            "version": 3,
            "user_id": "u1",
        }
    )

    assert str(payload.uuid) == "b9012f62-fd66-4b43-bdb7-ab2bdb36fb37"
    assert payload.created_at.year == 2026
    assert payload.template_json.startswith("{")


def test_chunking_template_list_response_wraps_template_records():
    payload = ChunkingTemplateListResponse.model_validate(
        {
            "templates": [
                {
                    "id": 5,
                    "uuid": "b9012f62-fd66-4b43-bdb7-ab2bdb36fb37",
                    "name": "demo",
                    "description": "Demo template",
                    "template_json": '{"chunking": {"method": "words", "config": {"max_size": 200}}}',
                    "is_builtin": False,
                    "tags": ["demo"],
                    "created_at": "2026-04-20T00:00:00Z",
                    "updated_at": "2026-04-20T00:10:00Z",
                    "version": 3,
                    "user_id": "u1",
                }
            ],
            "total": 1,
        }
    )

    assert payload.total == 1
    assert payload.templates[0].name == "demo"


def test_chunking_template_apply_request_uses_override_options():
    payload = ChunkingTemplateApplyRequest(
        template_name="demo",
        text="alpha beta gamma",
        override_options={"max_size": 32},
    )

    assert payload.model_dump(exclude_none=True)["override_options"] == {"max_size": 32}


def test_chunking_template_diagnostics_defaults_missing_methods_to_empty_list():
    payload = ChunkingTemplateDiagnosticsResponse(
        db_class="sqlite.MediaDatabase",
        capability="native",
        fallback_enabled=True,
        hint="hint",
    )

    assert payload.missing_methods == []


def test_embedding_collection_stats_defaults_metadata_to_empty_dict():
    payload = EmbeddingCollectionStatsResponse(name="demo", count=3)
    assert payload.metadata == {}
