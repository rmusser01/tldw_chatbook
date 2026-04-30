from __future__ import annotations

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload, HANDOFF_BODY_CHAR_LIMIT
from tldw_chatbook.UX_Interop import (
    build_server_parity_fixture_payloads,
    build_server_parity_handoff_packet,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def test_chat_handoff_ui_smoke_consumes_server_parity_fixture_payloads():
    fixtures = build_server_parity_fixture_payloads()

    assert fixtures["local"]["active_source"] == "local"
    assert fixtures["server"]["active_server_profile_id"] == "srv-primary"
    assert fixtures["unavailable_server"]["server_reachability"] == "unreachable"
    assert fixtures["auth_failure"]["reason_code"] == "auth_required"
    assert fixtures["unsupported_action"]["unsupported_reason_code"] == "server_contract_missing"
    assert fixtures["workspace_isolation"]["workspace_scope_id"] == "workspace-a"
    assert fixtures["sync_dry_run_report"]["write_enabled"] is False


def test_chat_handoff_packet_exposes_sections_ui_needs_without_screen_inference():
    packet = build_server_parity_handoff_packet(
        RuntimeSourceState(
            active_source="server",
            active_server_id="srv-primary",
            server_configured=True,
            server_reachability="reachable",
            server_auth_state="authenticated",
            last_known_server_label="Primary Server",
        ),
        workspace_scope_ids=("workspace-a",),
    )

    sections = packet["sections"]
    assert sections["active_server"]["active_server_profile_id"] == "srv-primary"
    assert sections["source_selector"]["source_options"]
    assert sections["sync"]["dry_run_only"] is True
    assert sections["workspace_isolation"][0]["workspace_scope_id"] == "workspace-a"
    assert "server_unavailable" in sections["error_contracts"]


def test_chat_handoff_payload_round_trip_preserves_runtime_scope_and_metadata():
    payload = ChatHandoffPayload(
        source="workspace",
        item_type="workspace-source",
        title="Transcript",
        body="source body",
        body_truncated=False,
        content_ref="workspace:workspace-1:source:source-1",
        source_id="source-1",
        display_summary="A workspace source",
        suggested_prompt="Use this source.",
        runtime_backend="server",
        source_owner="workspace",
        source_selector_state="workspace",
        active_server_profile_id="srv-primary",
        discovery_owner="workspace",
        discovery_entity_id="source-1",
        scope_type="workspace",
        workspace_id="workspace-1",
        backend_contracts={
            "workspace_isolation": {"workspace_scope_id": "workspace-1"},
            "active_server": {
                "active_server_profile_id": "srv-primary",
                "credential_source": "keyring:chatbook:server:srv-primary:access",
            },
        },
        unsupported_reports=[
            {
                "operation_id": "notes.graph.unsupported.workspace",
                "source": "workspace",
                "supported": False,
                "reason_code": "scope_not_supported",
                "user_message": "Workspace graph is not available.",
                "affected_action_ids": ["notes.graph.list.server"],
            }
        ],
        sync_dry_run_report={"dry_run": True, "write_enabled": False},
        metadata={"score": 0.87, "url": "https://example.com"},
    )

    restored = ChatHandoffPayload.from_dict(payload.to_dict())

    assert restored is not None
    assert restored.source == "workspace"
    assert restored.item_type == "workspace-source"
    assert restored.body_truncated is False
    assert restored.content_ref == "workspace:workspace-1:source:source-1"
    assert restored.runtime_backend == "server"
    assert restored.source_owner == "workspace"
    assert restored.source_selector_state == "workspace"
    assert restored.active_server_profile_id == "srv-primary"
    assert restored.scope_type == "workspace"
    assert restored.workspace_id == "workspace-1"
    assert restored.backend_contracts["workspace_isolation"]["workspace_scope_id"] == "workspace-1"
    assert "credential_source" not in restored.backend_contracts["active_server"]
    assert restored.unsupported_reports[0]["reason_code"] == "scope_not_supported"
    assert restored.sync_dry_run_report["write_enabled"] is False
    assert restored.metadata["score"] == 0.87


def test_chat_handoff_payload_persistence_redacts_secrets_and_normalizes_tuples():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Secret-adjacent",
        body="Body",
        backend_contracts={
            "active_server": {
                "active_server_profile_id": "srv-primary",
                "credential_source": "keyring:chatbook:server:srv-primary:access",
            },
            "source_selector": {"source_options": ({"source": "local"}, {"source": "server"})},
        },
        sync_dry_run_report={"conflict_ids": ("conflict-1",)},
    )

    data = payload.to_dict()

    assert "credential_source" not in data["backend_contracts"]["active_server"]
    assert data["backend_contracts"]["source_selector"]["source_options"] == [
        {"source": "local"},
        {"source": "server"},
    ]
    assert data["sync_dry_run_report"]["conflict_ids"] == ["conflict-1"]


def test_chat_handoff_payload_persistence_omits_nulls_and_redacts_model_metadata():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Metadata redaction",
        body="Body",
        metadata={"api_key": "should-not-persist", "safe": "ok"},
    )

    data = payload.to_dict()
    context = payload.model_context_block()

    assert "content_ref" not in data
    assert "api_key" not in data["metadata"]
    assert data["metadata"] == {"safe": "ok"}
    assert "api_key" not in context
    assert "should-not-persist" not in context
    assert "- safe: ok" in context


def test_chat_handoff_payload_from_source_content_caps_persisted_body():
    payload = ChatHandoffPayload.from_source_content(
        source="media",
        item_type="media",
        title="Long transcript",
        body="x" * (HANDOFF_BODY_CHAR_LIMIT + 1),
        content_ref="media:record-1",
    )

    assert len(payload.body) == HANDOFF_BODY_CHAR_LIMIT
    assert payload.body_truncated is True
    assert payload.content_ref == "media:record-1"


def test_chat_handoff_payload_from_source_content_preserves_upstream_truncation_flag():
    payload = ChatHandoffPayload.from_source_content(
        source="media",
        item_type="media",
        title="Already summarized transcript",
        body="short summary",
        body_truncated=True,
    )

    assert payload.body == "short summary"
    assert payload.body_truncated is True
