from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    GenerateFromChatRequest,
    GenerateFromMediaRequest,
    GenerateFromNotesRequest,
    GenerateFromPromptRequest,
    GenerateFromRagRequest,
    PresentationCreateRequest,
    PresentationListResponse,
    PresentationPatchRequest,
    PresentationRenderRequest,
    PresentationResponse,
    PresentationSearchResponse,
    PresentationUpdateRequest,
    ReadingExportResponse,
    Slide,
    SlidesHealthResponse,
    SlidesTemplateListResponse,
    TLDWAPIClient,
    VisualStyleCreateRequest,
    VisualStyleListResponse,
    VisualStylePatchRequest,
)


def _slide() -> dict:
    return {"order": 0, "layout": "title", "title": "Intro", "content": "Hello", "metadata": {}}


def _presentation(**overrides) -> dict:
    payload = {
        "id": "deck-1",
        "title": "Deck",
        "description": "A deck",
        "theme": "black",
        "marp_theme": None,
        "template_id": None,
        "visual_style_id": None,
        "visual_style_scope": None,
        "visual_style_name": None,
        "visual_style_version": None,
        "visual_style_snapshot": None,
        "settings": None,
        "studio_data": None,
        "slides": [_slide()],
        "custom_css": None,
        "source_type": "manual",
        "source_ref": None,
        "source_query": None,
        "created_at": "2026-04-25T12:00:00Z",
        "last_modified": "2026-04-25T12:05:00Z",
        "deleted": False,
        "client_id": "user-1",
        "version": 3,
    }
    payload.update(overrides)
    return payload


def _summary(**overrides) -> dict:
    payload = {
        "id": "deck-1",
        "title": "Deck",
        "description": "A deck",
        "theme": "black",
        "created_at": "2026-04-25T12:00:00Z",
        "last_modified": "2026-04-25T12:05:00Z",
        "deleted": False,
        "version": 3,
    }
    payload.update(overrides)
    return payload


def _template() -> dict:
    return {
        "id": "template-1",
        "name": "Default",
        "theme": "black",
        "marp_theme": None,
        "settings": {"paginate": True},
        "default_slides": [_slide()],
        "custom_css": None,
    }


def _style(**overrides) -> dict:
    payload = {
        "id": "style-1",
        "scope": "user",
        "name": "Minimal",
        "description": "Clean style",
        "generation_rules": {},
        "artifact_preferences": [],
        "appearance_defaults": {},
        "fallback_policy": {},
        "version": 1,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_slides_routes_wire_crud_generation_render_and_export(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"service": "slides", "status": "ok"},
            _presentation(),
            {"presentations": [_summary()], "total": 1, "limit": 10, "offset": 2},
            {"presentations": [_summary(title="Search hit")], "total": 1, "limit": 5, "offset": 0},
            _presentation(),
            _presentation(title="Updated"),
            _presentation(title="Patched"),
            _presentation(),
            _presentation(deleted=True),
            _presentation(deleted=False),
            {"templates": [_template()]},
            _template(),
            {"styles": [_style()], "total_count": 1, "limit": 20, "offset": 0},
            _style(scope="builtin"),
            _style(id="style-2"),
            _style(name="Updated style"),
            {},
            {
                "versions": [
                    {
                        "presentation_id": "deck-1",
                        "version": 2,
                        "created_at": "2026-04-25T12:04:00Z",
                        "title": "Deck v2",
                        "deleted": False,
                    }
                ],
                "total": 1,
                "limit": 10,
                "offset": 0,
            },
            _presentation(version=2),
            _presentation(version=4),
            {
                "job_id": 12,
                "status": "queued",
                "job_type": "presentation_render",
                "presentation_id": "deck-1",
                "presentation_version": 3,
                "format": "mp4",
            },
            {
                "job_id": 12,
                "status": "completed",
                "job_type": "presentation_render",
                "presentation_id": "deck-1",
                "presentation_version": 3,
                "format": "mp4",
                "output_id": 44,
                "download_url": "/api/v1/outputs/44/download",
                "error": None,
            },
            {
                "presentation_id": "deck-1",
                "artifacts": [
                    {
                        "output_id": 44,
                        "format": "mp4",
                        "title": "Deck render",
                        "download_url": "/api/v1/outputs/44/download",
                        "presentation_version": 3,
                    }
                ],
            },
            _presentation(source_type="prompt", source_query="Make a deck"),
            _presentation(source_type="chat", source_ref="conversation-1"),
            _presentation(source_type="notes", source_ref=["note-1"]),
            _presentation(source_type="media", source_ref="7"),
            _presentation(source_type="rag", source_query="topic"),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)
    exported = ReadingExportResponse(
        content=b"# Deck",
        content_type="text/markdown",
        content_disposition='attachment; filename="presentation_deck-1.md"',
        filename="presentation_deck-1.md",
    )
    binary = AsyncMock(return_value=exported)
    monkeypatch.setattr(client, "_binary_request", binary)

    health = await client.get_slides_health()
    created = await client.create_presentation(
        PresentationCreateRequest(
            title="Deck",
            description="A deck",
            slides=[Slide(order=0, layout="title", title="Intro", content="Hello")],
        )
    )
    listed = await client.list_presentations(limit=10, offset=2, sort="created_at desc", include_deleted=True)
    searched = await client.search_presentations("deck", limit=5, include_deleted=True)
    loaded = await client.get_presentation("deck-1", include_deleted=True)
    updated = await client.update_presentation(
        "deck-1",
        PresentationUpdateRequest(title="Updated", slides=[]),
        if_match='"3"',
    )
    patched = await client.patch_presentation(
        "deck-1",
        PresentationPatchRequest(title="Patched"),
        if_match='"3"',
    )
    reordered = await client.reorder_presentation("deck-1", [0], if_match='"3"')
    deleted = await client.delete_presentation("deck-1", if_match='"3"')
    restored = await client.restore_presentation("deck-1", if_match='"4"')
    templates = await client.list_slide_templates()
    template = await client.get_slide_template("template-1")
    styles = await client.list_visual_styles(limit=20)
    style = await client.get_visual_style("style-1")
    created_style = await client.create_visual_style(VisualStyleCreateRequest(name="Minimal"))
    patched_style = await client.patch_visual_style("style-2", VisualStylePatchRequest(name="Updated style"))
    delete_style_response = await client.delete_visual_style("style-2")
    versions = await client.list_presentation_versions("deck-1", limit=10)
    version = await client.get_presentation_version("deck-1", 2)
    restored_version = await client.restore_presentation_version("deck-1", 2, if_match='"3"')
    render = await client.submit_presentation_render_job(
        "deck-1",
        PresentationRenderRequest(format="mp4"),
        if_match='"3"',
    )
    render_status = await client.get_presentation_render_job_status(12)
    artifacts = await client.list_presentation_render_artifacts("deck-1")
    prompt_deck = await client.generate_presentation_from_prompt(GenerateFromPromptRequest(prompt="Make a deck"))
    chat_deck = await client.generate_presentation_from_chat(GenerateFromChatRequest(conversation_id="conversation-1"))
    notes_deck = await client.generate_presentation_from_notes(GenerateFromNotesRequest(note_ids=["note-1"]))
    media_deck = await client.generate_presentation_from_media(GenerateFromMediaRequest(media_id=7))
    rag_deck = await client.generate_presentation_from_rag(GenerateFromRagRequest(query="topic", top_k=3))
    export = await client.export_presentation(
        "deck-1",
        format="markdown",
        pdf_landscape=True,
        pdf_margin_top="1cm",
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/slides/health")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/slides/presentations")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "title": "Deck",
        "description": "A deck",
        "theme": "black",
        "slides": [{"order": 0, "layout": "title", "title": "Intro", "content": "Hello", "metadata": {}}],
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/slides/presentations")
    assert mocked.await_args_list[2].kwargs["params"] == {
        "limit": 10,
        "offset": 2,
        "sort": "created_at desc",
        "include_deleted": "true",
    }
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/slides/presentations/search")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "q": "deck",
        "limit": 5,
        "offset": 0,
        "include_deleted": "true",
    }
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/slides/presentations/deck-1")
    assert mocked.await_args_list[4].kwargs["params"] == {"include_deleted": "true"}
    assert mocked.await_args_list[5].args[:2] == ("PUT", "/api/v1/slides/presentations/deck-1")
    assert mocked.await_args_list[5].kwargs["headers"] == {"If-Match": '"3"'}
    assert mocked.await_args_list[6].args[:2] == ("PATCH", "/api/v1/slides/presentations/deck-1")
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/slides/presentations/deck-1/reorder")
    assert mocked.await_args_list[7].kwargs["json_data"] == {"order": [0]}
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/slides/presentations/deck-1")
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/slides/presentations/deck-1/restore")
    assert mocked.await_args_list[10].args[:2] == ("GET", "/api/v1/slides/templates")
    assert mocked.await_args_list[11].args[:2] == ("GET", "/api/v1/slides/templates/template-1")
    assert mocked.await_args_list[12].args[:2] == ("GET", "/api/v1/slides/styles")
    assert mocked.await_args_list[12].kwargs["params"] == {"limit": 20, "offset": 0}
    assert mocked.await_args_list[13].args[:2] == ("GET", "/api/v1/slides/styles/style-1")
    assert mocked.await_args_list[14].args[:2] == ("POST", "/api/v1/slides/styles")
    assert mocked.await_args_list[15].args[:2] == ("PATCH", "/api/v1/slides/styles/style-2")
    assert mocked.await_args_list[16].args[:2] == ("DELETE", "/api/v1/slides/styles/style-2")
    assert mocked.await_args_list[17].args[:2] == ("GET", "/api/v1/slides/presentations/deck-1/versions")
    assert mocked.await_args_list[18].args[:2] == ("GET", "/api/v1/slides/presentations/deck-1/versions/2")
    assert mocked.await_args_list[19].args[:2] == (
        "POST",
        "/api/v1/slides/presentations/deck-1/versions/2/restore",
    )
    assert mocked.await_args_list[20].args[:2] == ("POST", "/api/v1/slides/presentations/deck-1/render-jobs")
    assert mocked.await_args_list[20].kwargs["headers"] == {"If-Match": '"3"'}
    assert mocked.await_args_list[21].args[:2] == ("GET", "/api/v1/slides/render-jobs/12")
    assert mocked.await_args_list[22].args[:2] == ("GET", "/api/v1/slides/presentations/deck-1/render-artifacts")
    assert mocked.await_args_list[23].args[:2] == ("POST", "/api/v1/slides/generate")
    assert mocked.await_args_list[24].args[:2] == ("POST", "/api/v1/slides/generate/from-chat")
    assert mocked.await_args_list[25].args[:2] == ("POST", "/api/v1/slides/generate/from-notes")
    assert mocked.await_args_list[26].args[:2] == ("POST", "/api/v1/slides/generate/from-media")
    assert mocked.await_args_list[27].args[:2] == ("POST", "/api/v1/slides/generate/from-rag")
    assert binary.await_args.args[:2] == ("GET", "/api/v1/slides/presentations/deck-1/export")
    assert binary.await_args.kwargs["params"] == {
        "format": "markdown",
        "pdf_landscape": "true",
        "pdf_margin_top": "1cm",
    }
    assert isinstance(health, SlidesHealthResponse)
    assert isinstance(created, PresentationResponse)
    assert isinstance(listed, PresentationListResponse)
    assert isinstance(searched, PresentationSearchResponse)
    assert loaded.id == "deck-1"
    assert updated.title == "Updated"
    assert patched.title == "Patched"
    assert reordered.id == "deck-1"
    assert deleted.deleted is True
    assert restored.deleted is False
    assert isinstance(templates, SlidesTemplateListResponse)
    assert template.id == "template-1"
    assert isinstance(styles, VisualStyleListResponse)
    assert style.scope == "builtin"
    assert created_style.id == "style-2"
    assert patched_style.name == "Updated style"
    assert delete_style_response == {}
    assert versions.versions[0].version == 2
    assert version.version == 2
    assert restored_version.version == 4
    assert render.job_id == 12
    assert render_status.output_id == 44
    assert artifacts.artifacts[0].output_id == 44
    assert prompt_deck.source_type == "prompt"
    assert chat_deck.source_ref == "conversation-1"
    assert notes_deck.source_ref == ["note-1"]
    assert media_deck.source_ref == "7"
    assert rag_deck.source_query == "topic"
    assert export.content == b"# Deck"
