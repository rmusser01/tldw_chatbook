from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.WebClipper import ServerWebClipperScopeService, ServerWebClipperService


class FakePolicyEnforcer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def save_web_clip(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("save_web_clip", payload))
        return {
            "clip_id": payload["clip_id"],
            "status": "saved",
            "note": {"id": "note-1", "title": payload["source_title"], "version": 1},
            "workspace_placement": None,
            "attachments": [],
            "warnings": [],
            "note_id": "note-1",
            "workspace_placement_saved": False,
            "workspace_placement_count": 0,
        }

    async def get_web_clip_status(self, clip_id):
        self.calls.append(("get_web_clip_status", clip_id))
        return {
            "clip_id": clip_id,
            "status": "saved",
            "note": {"id": "note-1", "title": "Saved Article", "version": 1},
            "workspace_placements": [],
            "attachments": [],
            "analysis": {},
            "content_budget": {},
        }

    async def persist_web_clip_enrichment(self, clip_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("persist_web_clip_enrichment", clip_id, payload))
        return {
            "clip_id": clip_id,
            "enrichment_type": payload["enrichment_type"],
            "status": payload["status"],
            "source_note_version": payload["source_note_version"],
            "inline_applied": True,
            "inline_summary": payload.get("inline_summary"),
            "conflict_reason": None,
            "warnings": [],
        }


@pytest.mark.asyncio
async def test_server_web_clipper_service_routes_typed_client_calls():
    client = FakeClient()
    service = ServerWebClipperService(client=client)

    saved = await service.save_clip(
        clip_id="clip-1",
        clip_type="article",
        source_url="https://example.com/article",
        source_title="Saved Article",
        destination_mode="note",
        note={"title": "Saved Article", "keywords": ["clipper"]},
        content={"visible_body": "Visible body"},
    )
    status = await service.get_clip_status("clip-1")
    enriched = await service.persist_enrichment(
        "clip-1",
        clip_id="clip-1",
        enrichment_type="ocr",
        status="complete",
        source_note_version=1,
        inline_summary="OCR text",
        structured_payload={"text": "OCR text"},
    )

    assert saved["clip_id"] == "clip-1"
    assert status["note"]["id"] == "note-1"
    assert enriched["inline_applied"] is True
    assert client.calls == [
        (
            "save_web_clip",
            {
                "clip_id": "clip-1",
                "clip_type": "article",
                "source_url": "https://example.com/article",
                "source_title": "Saved Article",
                "destination_mode": "note",
                "note": {"title": "Saved Article", "keywords": ["clipper"]},
                "content": {"visible_body": "Visible body"},
                "attachments": [],
                "enhancements": {"run_ocr": False, "run_vlm": False},
                "capture_metadata": {},
            },
        ),
        ("get_web_clip_status", "clip-1"),
        (
            "persist_web_clip_enrichment",
            "clip-1",
            {
                "clip_id": "clip-1",
                "enrichment_type": "ocr",
                "status": "complete",
                "inline_summary": "OCR text",
                "structured_payload": {"text": "OCR text"},
                "source_note_version": 1,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_web_clipper_scope_service_enforces_remote_only_policy_and_normalizes_ids():
    policy = FakePolicyEnforcer()
    service = ServerWebClipperService(client=FakeClient())
    scope = ServerWebClipperScopeService(server_service=service, policy_enforcer=policy)

    saved = await scope.save_clip(
        mode="server",
        clip_id="clip-1",
        clip_type="article",
        source_url="https://example.com/article",
        source_title="Saved Article",
        destination_mode="note",
    )
    status = await scope.get_clip_status(mode="server", clip_id="clip-1")
    enriched = await scope.persist_enrichment(
        mode="server",
        clip_id="clip-1",
        enrichment_type="ocr",
        status="complete",
        source_note_version=1,
        inline_summary="OCR text",
        structured_payload={"text": "OCR text"},
    )

    assert saved["id"] == "server:web_clip:clip-1"
    assert saved["entity_kind"] == "web_clip"
    assert saved["backend"] == "server"
    assert saved["note"]["id"] == "server:note:note-1"
    assert status["id"] == "server:web_clip:clip-1"
    assert enriched["id"] == "server:web_clip:clip-1:enrichment:ocr"
    assert policy.calls == [
        "web_clipper.capture.server",
        "web_clipper.status.server",
        "web_clipper.capture.server",
    ]


@pytest.mark.asyncio
async def test_web_clipper_scope_service_rejects_local_mode_before_policy_dispatch():
    policy = FakePolicyEnforcer()
    scope = ServerWebClipperScopeService(
        server_service=ServerWebClipperService(client=FakeClient()),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server web clipper requires server mode"):
        await scope.save_clip(
            mode="local",
            clip_id="clip-1",
            clip_type="article",
            source_url="https://example.com/article",
            source_title="Saved Article",
        )

    assert policy.calls == []
