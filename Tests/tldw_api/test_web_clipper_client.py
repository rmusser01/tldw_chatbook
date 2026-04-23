from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    TLDWAPIClient,
    WebClipperEnrichmentPayload,
    WebClipperEnrichmentResponse,
    WebClipperSaveRequest,
    WebClipperSaveResponse,
    WebClipperSaveResult,
    WebClipperStatusResponse,
)


def _attachment_payload() -> dict:
    return {
        "slot": "screenshot",
        "file_name": "clip.png",
        "original_file_name": "clip.png",
        "content_type": "image/png",
        "size_bytes": 128,
        "uploaded_at": "2026-04-23T20:00:00Z",
        "url": "server://attachments/clip.png",
    }


def _save_response_payload() -> dict:
    return {
        "clip_id": "clip-1",
        "status": "saved",
        "note": {
            "id": "note-1",
            "title": "Saved Article",
            "version": 1,
        },
        "workspace_placement": None,
        "attachments": [_attachment_payload()],
        "warnings": [],
        "note_id": "note-1",
        "workspace_placement_saved": False,
        "workspace_placement_count": 0,
    }


def _save_result_payload() -> dict:
    return {
        "clip_id": "clip-1",
        "note_id": "note-1",
        "status": "saved",
        "workspace_placement_saved": False,
        "workspace_placement_count": 0,
        "warnings": [],
    }


def _status_response_payload() -> dict:
    return {
        "clip_id": "clip-1",
        "status": "saved",
        "note": {
            "id": "note-1",
            "title": "Saved Article",
            "version": 1,
        },
        "workspace_placements": [],
        "attachments": [_attachment_payload()],
        "analysis": {"ocr": {"status": "complete"}},
        "content_budget": {"visible_body_chars": 120},
    }


@pytest.mark.asyncio
async def test_client_routes_web_clipper_save_status_and_enrichment_calls():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            _save_response_payload(),
            _status_response_payload(),
            {
                "clip_id": "clip-1",
                "enrichment_type": "ocr",
                "status": "complete",
                "source_note_version": 1,
                "inline_applied": True,
                "inline_summary": "Text from screenshot",
                "conflict_reason": None,
                "warnings": [],
            },
        ]
    )

    saved = await client.save_web_clip(
        WebClipperSaveRequest(
            clip_id="clip-1",
            clip_type="article",
            source_url="https://example.com/article",
            source_title="Saved Article",
            destination_mode="note",
            note={
                "title": "Saved Article",
                "comment": "Read later",
                "keywords": ["clipper", "article"],
            },
            content={
                "visible_body": "Visible body",
                "selected_text": "Selected quote",
            },
            capture_metadata={"browser": "firefox"},
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
            inline_summary="Text from screenshot",
            structured_payload={"text": "Text from screenshot"},
        ),
    )

    assert isinstance(saved, WebClipperSaveResponse)
    assert WebClipperSaveResult.model_validate(_save_result_payload()).note_id == "note-1"
    assert isinstance(status, WebClipperStatusResponse)
    assert isinstance(enrichment, WebClipperEnrichmentResponse)
    assert saved.note_id == "note-1"
    assert status.analysis["ocr"]["status"] == "complete"
    assert enrichment.inline_applied is True
    assert [call.args for call in client._request.await_args_list] == [
        ("POST", "/api/v1/web-clipper/save"),
        ("GET", "/api/v1/web-clipper/clip-1"),
        ("POST", "/api/v1/web-clipper/clip-1/enrichments"),
    ]
    assert client._request.await_args_list[0].kwargs["json_data"] == {
        "clip_id": "clip-1",
        "clip_type": "article",
        "source_url": "https://example.com/article",
        "source_title": "Saved Article",
        "destination_mode": "note",
        "note": {
            "title": "Saved Article",
            "comment": "Read later",
            "keywords": ["clipper", "article"],
        },
        "content": {
            "visible_body": "Visible body",
            "selected_text": "Selected quote",
        },
        "attachments": [],
        "enhancements": {
            "run_ocr": False,
            "run_vlm": False,
        },
        "capture_metadata": {"browser": "firefox"},
    }
    assert client._request.await_args_list[2].kwargs["json_data"] == {
        "clip_id": "clip-1",
        "enrichment_type": "ocr",
        "status": "complete",
        "inline_summary": "Text from screenshot",
        "structured_payload": {"text": "Text from screenshot"},
        "source_note_version": 1,
    }
