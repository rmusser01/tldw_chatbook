from unittest.mock import Mock

import pytest

from tldw_chatbook.Web_Clipper_Interop import ServerWebClipperService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeWebClipperClient:
    def __init__(self):
        self.calls = []

    async def save_web_clip(self, request_data):
        self.calls.append(("save_web_clip", request_data.model_dump(exclude_none=True, mode="json")))
        return {"clip_id": "clip-1", "note_id": "note-1", "status": "saved"}

    async def get_web_clip_status(self, clip_id):
        self.calls.append(("get_web_clip_status", clip_id))
        return {"clip_id": clip_id, "status": "saved"}

    async def persist_web_clip_enrichment(self, clip_id, request_data):
        self.calls.append(("persist_web_clip_enrichment", clip_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"clip_id": clip_id, "status": "complete"}


@pytest.mark.asyncio
async def test_server_web_clipper_service_routes_with_policy_actions():
    client = FakeWebClipperClient()
    policy = Mock()
    service = ServerWebClipperService(client=client, policy_enforcer=policy)

    saved = await service.save_clip(
        clip_id="clip-1",
        clip_type="article",
        source_url="https://example.com",
        source_title="Example",
        content={"visible_body": "Body"},
    )
    status = await service.get_status("clip-1")
    enrichment = await service.persist_enrichment(
        clip_id="clip-1",
        enrichment_type="ocr",
        status="complete",
        source_note_version=1,
    )

    assert saved["note_id"] == "note-1"
    assert status["clip_id"] == "clip-1"
    assert enrichment["status"] == "complete"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "web_clipper.capture.server",
        "web_clipper.status.server",
        "web_clipper.capture.server",
    ]


@pytest.mark.asyncio
async def test_server_web_clipper_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeWebClipperClient()
    service = ServerWebClipperService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_status("clip-1")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
