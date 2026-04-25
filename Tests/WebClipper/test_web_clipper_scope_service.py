import pytest

from tldw_chatbook.Web_Clipper_Interop.web_clipper_scope_service import WebClipperScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeWebClipperService:
    def __init__(self):
        self.calls = []

    async def save_clip(self, **kwargs):
        self.calls.append(("save_clip", kwargs))
        return {"clip_id": kwargs["clip_id"], "note_id": "note-1", "status": "saved"}

    async def get_status(self, clip_id):
        self.calls.append(("get_status", clip_id))
        return {"clip_id": clip_id, "status": "saved"}

    async def persist_enrichment(self, **kwargs):
        self.calls.append(("persist_enrichment", kwargs))
        return {"clip_id": kwargs["clip_id"], "enrichment_type": kwargs["enrichment_type"], "status": "complete"}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_web_clipper_scope_service_routes_server_capture_status_and_enrichment():
    server = FakeWebClipperService()
    policy = FakePolicyEnforcer()
    scope = WebClipperScopeService(server_service=server, policy_enforcer=policy)

    saved = await scope.save_clip(
        mode="server",
        clip_id="clip-1",
        clip_type="article",
        source_url="https://example.com",
        source_title="Example",
        content={"visible_body": "Body"},
    )
    status = await scope.get_status(mode="server", clip_id="clip-1")
    enrichment = await scope.persist_enrichment(
        mode="server",
        clip_id="clip-1",
        enrichment_type="ocr",
        source_note_version=1,
        status="complete",
    )

    assert saved["record_id"] == "server:web_clip:clip-1"
    assert saved["backend"] == "server"
    assert status["record_id"] == "server:web_clip:clip-1"
    assert enrichment["record_id"] == "server:web_clip_enrichment:clip-1:ocr"
    assert server.calls == [
        (
            "save_clip",
            {
                "clip_id": "clip-1",
                "clip_type": "article",
                "source_url": "https://example.com",
                "source_title": "Example",
                "content": {"visible_body": "Body"},
            },
        ),
        ("get_status", "clip-1"),
        (
            "persist_enrichment",
            {
                "clip_id": "clip-1",
                "enrichment_type": "ocr",
                "source_note_version": 1,
                "status": "complete",
            },
        ),
    ]
    assert policy.calls == [
        "web_clipper.capture.server",
        "web_clipper.status.server",
        "web_clipper.capture.server",
    ]


@pytest.mark.asyncio
async def test_web_clipper_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeWebClipperService()
    scope = WebClipperScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Web clipper is a server-only capability"):
        await scope.get_status(mode="local", clip_id="clip-1")

    assert server.calls == []


@pytest.mark.asyncio
async def test_web_clipper_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeWebClipperService()
    scope = WebClipperScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_unreachable"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.save_clip(
            mode="server",
            clip_id="clip-1",
            clip_type="article",
            source_url="https://example.com",
            source_title="Example",
        )

    assert exc.value.reason_code == "server_unreachable"
    assert server.calls == []
