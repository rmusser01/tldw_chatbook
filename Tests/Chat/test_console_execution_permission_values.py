"""Original permission values cross the wire without their local authority."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from tldw_chatbook.Chat import console_control_permissions as operations
from tldw_chatbook.Chat.console_execution_permission_values import (
    encode_permission_result,
)
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError,
    WireIdentity,
    encode_message,
)
from tldw_chatbook.Chat.console_permission_operations import PermissionRenderCapture
from tldw_chatbook.MCP.permission_store import (
    SCHEMA_VERSION,
    EffectiveToolState,
    PermissionStoreSnapshot,
    _freeze_snapshot,
)
from tldw_chatbook.Tool_Packs.activation import (
    InstalledToolProfile,
    ToolPackActivationResult,
)
from tldw_chatbook.Tool_Packs.binding import (
    ToolProfileBindingReview,
    ToolProfileBindingSummary,
)
from tldw_chatbook.Tool_Packs.contracts import PortableFallback, PortableToolRule
from tldw_chatbook.Tool_Packs.importer import (
    MappedToolRule,
    ServerMapping,
    ToolPackImportReview,
)
from tldw_chatbook.Tool_Packs.removal import (
    RemovedToolProfile,
    ToolProfileRemovalResult,
)
from tldw_chatbook.Tool_Packs.service import ToolProfileListing, ToolProfilePresentation
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


IDENTITY = WireIdentity("profile", "service", 1, "permission-reply", "", "", 0)
EXPIRY = datetime(2030, 1, 2, 3, 4, tzinfo=timezone.utc)


def _reply(operation, result, identity=IDENTITY):
    return encode_permission_result(identity, operation, result)


def _value(operation, result):
    return encode_message(_reply(operation, result))["payload"]["result"]


def _snapshot(payload):
    return PermissionStoreSnapshot(_freeze_snapshot(payload), "sha256:generation", True)


def _review():
    rule = PortableToolRule("mcp", "local:docs", "read", "allow", "a" * 64)
    return ToolPackImportReview(
        archive_path=Path("private-import-location.tldwtoolpack"),
        archive_sha256="b" * 64,
        manifest_sha256="c" * 64,
        payload_sha256="d" * 64,
        destination_id="imported-profile",
        store_generation="local-authority-generation",
        inventory_digest="local-inventory-authority",
        mappings=(ServerMapping("local:docs", "local:mapped"),),
        expires_at=EXPIRY,
        matched=(
            MappedToolRule(rule, ("mcp", "local:mapped", "read"), "e" * 64, True),
        ),
        changed=(),
        missing=(),
        pending_denies=(),
        omitted_allow_ask=(),
        content_digest="local-content-authority",
        display_name="Documentation tools",
        producer=("Producer", "1"),
        fallbacks=(PortableFallback("mcp", "*", "ask"),),
    )


def _binding_review():
    summary = ToolProfileBindingSummary(
        global_fallback="ask",
        builtin_fallback="deny",
        allow_server_fallbacks=("local:docs",),
        stored_exact_allows=(("local:docs", "read"),),
        effective_allows=(("local:docs", "read"),),
        unavailable_allows=(),
        downgraded_allows=(),
        high_risk_allows=(),
        allow_count=1,
        ask_count=1,
        deny_count=0,
        inventory_digest="display-inventory",
        effective_asks=(("local:docs", "write"),),
    )
    return ToolProfileBindingReview(
        workspace_id="workspace",
        action="set",
        intended_defaults_digest="defaults",
        profile_id="imported-profile",
        policy_digest="policy",
        revision=2,
        expires_at=EXPIRY,
        summary=summary,
    )


def test_snapshot_domains_remain_distinct_and_detached():
    compatibility = {"legacy": {"unknown": [None, {"retained": True}]}}
    message = _reply(operations.ReadPermissionSnapshot(strict=False), compatibility)
    compatibility["legacy"]["unknown"].clear()
    encoded = encode_message(message)
    assert encoded["payload"]["result"] == {
        "kind": "compatibility",
        "payload": {"legacy": {"unknown": [None, {"retained": True}]}},
    }
    encoded["payload"]["result"]["payload"].clear()
    assert encode_message(message)["payload"]["result"]["payload"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": False,
        "profiles": {"default": {"servers": {}}},
    }
    snapshot = _snapshot(payload)
    strict = _value(operations.ReadPermissionSnapshot(), snapshot)
    assert strict == {
        "kind": "snapshot",
        "payload": payload,
        "generation": snapshot.generation,
        "file_exists": True,
    }
    malformed = {"profiles": {"broken": {"tool_pack_lifecycle": [None, {"odd": True}]}}}
    inventory = _value(operations.ReadPermissionInventory(), _snapshot(malformed))
    assert inventory["payload"] == malformed and "kind" not in inventory

    with pytest.raises(ConsoleProtocolError, match="PermissionStoreSnapshot"):
        _reply(operations.ReadPermissionSnapshot(), payload)
    with pytest.raises(ConsoleProtocolError, match="expected dict"):
        _reply(operations.ReadPermissionSnapshot(strict=False), snapshot)
    with pytest.raises(ConsoleProtocolError):
        _reply(operations.ReadPermissionSnapshot(strict=False), {"opaque": object()})


def test_actual_effective_states_and_render_identity_keep_their_named_rows():
    states = {
        ("local:docs", "read"): EffectiveToolState("ask", "tool_override", True, True),
        ("local:docs", "write"): EffectiveToolState("deny", "global_default"),
    }
    expected = [
        {
            "server_key": "local:docs",
            "tool_name": "read",
            "state": "ask",
            "origin": "tool_override",
            "config_changed": True,
            "risk_floored": True,
        },
        {
            "server_key": "local:docs",
            "tool_name": "write",
            "state": "deny",
            "origin": "global_default",
            "config_changed": False,
            "risk_floored": False,
        },
    ]
    assert _value(operations.ReadEffectiveStates(tools=[]), states) == expected
    capture = PermissionRenderCapture(
        _snapshot({"profiles": {}}), states, ("policy", None, "chain")
    )
    value = _value(operations.CapturePermissionRender(tools=[]), capture)
    assert value["effective_states"] == expected
    assert value["profile_identity"] == ["policy", None, "chain"]
    states.clear()
    assert value["effective_states"] == expected


@pytest.mark.parametrize(
    "operation,result,invalid",
    [
        (operations.SetKillSwitch(value=False), None, False),
        (operations.ClearSessionApprovals(), None, True),
        (operations.ReadKillSwitch(), False, None),
        (
            operations.RemoveArgRule(
                server_key="local:docs", tool_name="read", rule_id="{}"
            ),
            False,
            0,
        ),
    ],
)
def test_none_and_boolean_operation_results_are_not_interchangeable(
    operation, result, invalid
):
    assert _value(operation, result) is result
    with pytest.raises(ConsoleProtocolError):
        _reply(operation, invalid)


def test_original_rules_and_session_pairs_preserve_order_and_legacy_values():
    legacy = {"old": [1, None]}
    rows = [
        {
            "rule_id": "{}",
            "args_json": "{}",
            "created_at": legacy,
            "profile_id": "parent",
        }
    ]
    operation = operations.ReadArgRules(server_key="local:docs", tool_name="read")
    message = _reply(operation, rows)
    legacy["old"].clear()
    assert encode_message(message)["payload"]["result"][0]["created_at"] == {
        "old": [1, None]
    }
    pairs = [("local:docs", "read"), ("local:docs", "write")]
    assert _value(operations.ListSessionApprovals(), pairs) == [
        list(pair) for pair in pairs
    ]


def test_import_review_exports_presentation_without_paths_or_evidence():
    review = _review()
    operation = operations.InspectImport(
        archive_path=review.archive_path, destination_id=review.destination_id
    )
    value = _value(operation, review)
    assert value["display_name"] == review.display_name
    assert value["producer"] == ["Producer", "1"]
    assert value["expires_at"] == EXPIRY.isoformat()
    assert value["matched"][0]["source_rule"]["contract_sha256"] == "a" * 64
    assert value["matched"][0]["destination_identity"] == [
        "mcp",
        "local:mapped",
        "read",
    ]
    assert value["fallbacks"] == [
        {"authority": "mcp", "server_key": "*", "state": "ask"}
    ]
    assert set(value).isdisjoint(
        {
            "archive_path",
            "archive_sha256",
            "manifest_sha256",
            "payload_sha256",
            "store_generation",
            "inventory_digest",
            "content_digest",
        }
    )
    assert str(review.archive_path) not in json.dumps(value)
    assert _value(operation, replace(review, producer=()))["producer"] == []
    with pytest.raises(ConsoleProtocolError, match="timezone-aware"):
        _reply(operation, replace(review, expires_at=EXPIRY.replace(tzinfo=None)))


def test_first_bind_confirmation_exports_request_reference_and_keeps_original_review():
    review = _binding_review()
    operation = operations.ReviewFirstBind(
        workspace_id=review.workspace_id,
        action=review.action,
        intended_defaults=WorkspaceAssistantDefaults(
            assistant_id="persona-for-review", tool_policy_profile_id=review.profile_id
        ),
    )
    value = _value(operation, review)
    assert value["summary"]["effective_allows"] == [["local:docs", "read"]]
    assert value["summary"]["effective_asks"] == [["local:docs", "write"]]
    assert value["intended_defaults_digest"] == review.intended_defaults_digest
    confirm = operations.ConfirmFirstBind(review=review)
    token = "opaque-original-confirmation-token"
    message = _reply(confirm, token)
    assert confirm.review is review
    assert encode_message(message)["payload"]["result"] == {
        "confirmation_request_id": IDENTITY.request_id
    }
    assert token not in json.dumps(encode_message(message))
    for invalid in ("", None, object()):
        with pytest.raises(ConsoleProtocolError):
            _reply(confirm, invalid)


def test_original_activation_removal_and_unavailable_listing_keep_detached_outcomes():
    activation = ToolPackActivationResult(
        InstalledToolProfile("profile", "digest", 1, "receipt"), "generation", True
    )
    assert _value(operations.ImportUnboundProfile(review=_review()), activation) == {
        "installed": {
            "profile_id": "profile",
            "policy_digest": "digest",
            "revision": 1,
            "receipt_id": "receipt",
        },
        "store_generation": "generation",
        "reconciled": True,
    }
    removal = ToolProfileRemovalResult(
        RemovedToolProfile(
            "profile", "tool_pack_tombstone", 2, "digest", "receipt", "receipt-digest"
        ),
        "next-generation",
    )
    removed = _value(
        operations.RemoveImportedProfile(profile_id="profile", expected_revision=1),
        removal,
    )
    assert removed["tombstone"]["profile_kind"] == "tool_pack_tombstone"
    assert removed["tombstone"]["receipt_digest"] == "receipt-digest"
    assert removed["reconciled"] is False
    row = ToolProfilePresentation(
        profile_id="broken",
        origin="imported",
        lifecycle_valid=False,
        binding_state="unbound",
        first_bind_confirmation_required=False,
        reference_counts=(0, 0),
        posture_counts=(0, 1, 0),
        receipt_health="unavailable",
        removal_eligible=False,
        removal_blocker="lifecycle_invalid",
        revision=None,
        policy_digest=None,
    )
    listing = _value(operations.ListToolProfiles(), ToolProfileListing((row,)))
    assert listing["profiles"][0]["lifecycle_valid"] is False
    assert listing["profiles"][0]["posture_counts"] == [0, 1, 0]
    assert _value(
        operations.ListToolProfiles(),
        ToolProfileListing(unavailable_category="inventory_unavailable"),
    ) == {
        "profiles": [],
        "unavailable_category": "inventory_unavailable",
    }


def test_custom_operation_and_nested_result_classes_are_refused():
    class CustomRead(operations.ReadKillSwitch):
        pass

    with pytest.raises(ConsoleProtocolError, match="unsupported permission operation"):
        _reply(CustomRead(), False)
    with pytest.raises(ConsoleProtocolError, match="EffectiveToolState"):
        _reply(
            operations.ReadEffectiveStates(tools=[]),
            {("local:docs", "read"): {"state": "ask"}},
        )
    with pytest.raises(ConsoleProtocolError, match="InstalledToolProfile"):
        _reply(
            operations.ImportUnboundProfile(review=_review()),
            ToolPackActivationResult({"profile_id": "fake"}, "generation"),
        )
