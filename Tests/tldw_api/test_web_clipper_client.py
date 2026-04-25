from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    WebClipperEnrichmentPayload,
    WebClipperSaveRequest,
    TLDWAPIClient,
)


def _save_response() -> dict:
    return {
        "clip_id": "clip-1",
        "status": "saved",
        "note": {"id": "note-1", "title": "Example", "version": 1},
        "workspace_placement": None,
        "attachments": [],
        "warnings": [],
        "note_id": "note-1",
        "workspace_placement_saved": False,
        "workspace_placement_count": 0,
    }


@pytest.mark.asyncio
async def test_web_clipper_client_routes_save_status_and_enrichment(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _save_response(),
            {
                "clip_id": "clip-1",
                "status": "saved",
                "note": {"id": "note-1", "title": "Example", "version": 1},
                "workspace_placements": [],
                "attachments": [],
                "analysis": {},
                "content_budget": {},
            },
            {
                "clip_id": "clip-1",
                "enrichment_type": "ocr",
                "status": "complete",
                "source_note_version": 1,
                "inline_applied": True,
                "inline_summary": "Detected text",
                "warnings": [],
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    saved = await client.save_web_clip(
        WebClipperSaveRequest(
            clip_id="clip-1",
            clip_type="article",
            source_url="https://example.com",
            source_title="Example",
            content={"visible_body": "Body"},
        )
    )
    status = await client.get_web_clip_status("clip-1")
    enrichment = await client.persist_web_clip_enrichment(
        "clip-1",
        WebClipperEnrichmentPayload(
            clip_id="clip-1",
            enrichment_type="ocr",
            status="complete",
            source_note_version=1,
            inline_summary="Detected text",
        ),
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/web-clipper/save")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "clip_id": "clip-1",
        "clip_type": "article",
        "source_url": "https://example.com",
        "source_title": "Example",
        "destination_mode": "note",
        "note": {"keywords": []},
        "content": {"visible_body": "Body"},
        "attachments": [],
        "enhancements": {"run_ocr": False, "run_vlm": False},
        "capture_metadata": {},
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/web-clipper/clip-1")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/web-clipper/clip-1/enrichments")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "clip_id": "clip-1",
        "enrichment_type": "ocr",
        "status": "complete",
        "inline_summary": "Detected text",
        "structured_payload": {},
        "source_note_version": 1,
    }

    assert saved.note_id == "note-1"
    assert status.note.id == "note-1"
    assert enrichment.inline_applied is True


def test_web_clipper_save_requires_workspace_for_workspace_destinations():
    with pytest.raises(ValueError):
        WebClipperSaveRequest(
            clip_id="clip-1",
            clip_type="article",
            source_url="https://example.com",
            source_title="Example",
            destination_mode="workspace",
        )
