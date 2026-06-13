from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from tldw_chatbook.runtime_policy.server_parity_models import (
    EventCursor,
    EventDedupeKey,
    NormalizedEventRecord,
    NotificationPresentationRecord,
    ProviderMigrationStatus,
    SyncIdentityMapEntry,
    SyncReadinessReport,
)


def test_event_cursor_key_is_scoped_by_source_server_principal_stream_and_instance() -> None:
    server_cursor = EventCursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="default",
        cursor="abc",
    )
    other_principal_cursor = EventCursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-b",
        stream_name="notifications",
        stream_instance_id="default",
        cursor="abc",
    )
    local_cursor = EventCursor(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        cursor="abc",
    )

    assert server_cursor.storage_key() == "server:server-a:user-a:notifications:default"
    assert other_principal_cursor.storage_key() == "server:server-a:user-b:notifications:default"
    assert local_cursor.storage_key() == "local:none:none:notifications:default"
    assert server_cursor.storage_key() != other_principal_cursor.storage_key()
    assert server_cursor.storage_key() != local_cursor.storage_key()


def test_server_normalized_event_requires_server_profile_id() -> None:
    with pytest.raises(ValueError, match="server_profile_id"):
        NormalizedEventRecord(
            source_authority="server",
            server_profile_id=None,
            stream_name="notifications",
            stream_instance_id="default",
            event_kind="notification.created",
            entity_ref={"type": "notification", "id": "n-1"},
            payload_hash="hash",
        )


def test_local_normalized_event_does_not_require_server_profile_id() -> None:
    record = NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="hash",
    )

    assert record.source_authority == "local"
    assert record.server_profile_id is None


def test_event_dedupe_key_falls_back_to_normalized_event_identity() -> None:
    record = NormalizedEventRecord(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="payload-sha",
        emitted_at="2026-04-29T01:02:03Z",
    )

    assert EventDedupeKey.from_event(record) == EventDedupeKey(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_id="n-1",
        timestamp="2026-04-29T01:02:03Z",
        payload_hash="payload-sha",
    )


def test_sync_readiness_defaults_to_not_registered_and_not_write_enabled() -> None:
    report = SyncReadinessReport(domain="chat")

    assert report.sync_eligible is False
    assert report.write_enabled is False
    assert report.reason_codes == ("not_registered",)


def test_sync_identity_map_entry_preserves_source_scope_server_and_workspace_ids() -> None:
    entry = SyncIdentityMapEntry(
        domain="notes",
        source_authority="server",
        source_scope="workspace",
        local_entity_id="local-1",
        remote_entity_id="remote-1",
        server_profile_id="server-a",
        workspace_id="workspace-a",
    )

    assert entry.source_authority == "server"
    assert entry.source_scope == "workspace"
    assert entry.server_profile_id == "server-a"
    assert entry.workspace_id == "workspace-a"


def test_provider_migration_status_represents_migrated_and_compatibility_mode() -> None:
    migrated = ProviderMigrationStatus(service_name="chat", provider_backed=True)
    compatibility = ProviderMigrationStatus(
        service_name="notes",
        provider_backed=False,
        compatibility_mode=True,
        reason_code="legacy_config_factory",
    )

    assert migrated.provider_backed is True
    assert migrated.compatibility_mode is False
    assert compatibility.provider_backed is False
    assert compatibility.compatibility_mode is True
    assert compatibility.reason_code == "legacy_config_factory"


def test_notification_presentation_keeps_local_delivery_state_separate_from_server_state() -> None:
    presentation = NotificationPresentationRecord(
        event_key="server:server-a:notifications:default:n-1",
        local_delivery_state="delivered",
        server_read_state="unread",
        server_dismiss_state="active",
    )

    assert presentation.local_delivery_state == "delivered"
    assert presentation.server_read_state == "unread"
    assert presentation.server_dismiss_state == "active"


def test_importing_server_parity_models_does_not_load_heavy_runtime_policy_modules() -> None:
    script = """
import json
import sys
import tldw_chatbook.runtime_policy.server_parity_models

heavy_modules = [
    "tldw_chatbook.runtime_policy.engine",
    "tldw_chatbook.runtime_policy.enforcement",
    "tldw_chatbook.runtime_policy.server_context",
    "tldw_chatbook.runtime_policy.server_capabilities",
]
print(json.dumps({name: name in sys.modules for name in heavy_modules}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "tldw_chatbook.runtime_policy.engine": False,
        "tldw_chatbook.runtime_policy.enforcement": False,
        "tldw_chatbook.runtime_policy.server_capabilities": False,
        "tldw_chatbook.runtime_policy.server_context": False,
    }


def test_normalized_event_record_dedupe_identity_is_stable_after_input_mutation() -> None:
    entity_ref = {"type": "notification", "id": "n-1"}
    record = NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref=entity_ref,
        payload_hash="hash",
    )

    entity_ref["id"] = "n-2"

    assert record.fallback_dedupe_key().entity_id == "n-1"


def test_normalized_event_record_payload_is_stable_after_input_mutation() -> None:
    payload = {"outer": {"inner": "initial"}, "items": ["a"]}
    record = NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="hash",
        payload=payload,
    )

    payload["outer"]["inner"] = "mutated"
    payload["items"].append("b")

    assert record.payload == {"outer": {"inner": "initial"}, "items": ("a",)}


def test_sync_readiness_report_details_are_stable_after_input_mutation() -> None:
    details = {"nested": {"state": "initial"}, "reasons": ["missing_identity"]}
    report = SyncReadinessReport(domain="notes", details=details)

    details["nested"]["state"] = "mutated"
    details["reasons"].append("mutated")

    assert report.details == {"nested": {"state": "initial"}, "reasons": ("missing_identity",)}


def test_normalized_event_record_json_fields_support_direct_json_dumps() -> None:
    record = NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="hash",
        payload={"nested": {"ok": True}, "items": ["a", 1]},
    )
    report = SyncReadinessReport(
        domain="notes",
        details={"nested": {"state": "initial"}, "reasons": ["missing_identity"]},
    )

    assert json.loads(json.dumps(record.entity_ref)) == {"type": "notification", "id": "n-1"}
    assert json.loads(json.dumps(record.payload)) == {"nested": {"ok": True}, "items": ["a", 1]}
    assert json.loads(json.dumps(report.details)) == {
        "nested": {"state": "initial"},
        "reasons": ["missing_identity"],
    }


def test_frozen_json_mappings_reject_mutation() -> None:
    record = NormalizedEventRecord(
        source_authority="local",
        server_profile_id=None,
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="hash",
        payload={"nested": {"ok": True}},
    )
    report = SyncReadinessReport(domain="notes", details={"state": "initial"})

    with pytest.raises(TypeError):
        record.entity_ref["id"] = "n-2"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.payload["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.payload["nested"]["ok"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        report.details.update({"state": "mutated"})  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("entity_ref", {"entity_ref": {"type": object(), "id": "n-1"}}),
        ("payload", {"payload": {"unsupported": object()}}),
        ("details", {"details": {"unsupported": object()}}),
    ],
)
def test_unsupported_json_values_raise_type_error(field_name: str, kwargs: dict) -> None:
    if field_name == "details":
        with pytest.raises(TypeError):
            SyncReadinessReport(domain="notes", **kwargs)
        return

    event_kwargs = {
        "source_authority": "local",
        "server_profile_id": None,
        "stream_name": "notifications",
        "stream_instance_id": "default",
        "event_kind": "notification.created",
        "entity_ref": {"type": "notification", "id": "n-1"},
        "payload_hash": "hash",
    }
    event_kwargs.update(kwargs)

    with pytest.raises(TypeError):
        NormalizedEventRecord(**event_kwargs)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("entity_ref", {"entity_ref": {"type": "notification", "score": math.inf}}),
        ("payload", {"payload": {"score": math.nan}}),
        ("details", {"details": {"score": -math.inf}}),
    ],
)
def test_non_finite_json_float_values_raise_type_error(field_name: str, kwargs: dict) -> None:
    if field_name == "details":
        with pytest.raises(TypeError):
            SyncReadinessReport(domain="notes", **kwargs)
        return

    event_kwargs = {
        "source_authority": "local",
        "server_profile_id": None,
        "stream_name": "notifications",
        "stream_instance_id": "default",
        "event_kind": "notification.created",
        "entity_ref": {"type": "notification", "id": "n-1"},
        "payload_hash": "hash",
    }
    event_kwargs.update(kwargs)

    with pytest.raises(TypeError):
        NormalizedEventRecord(**event_kwargs)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("entity_ref", {"entity_ref": {"type": "notification", 1: "n-1"}}),
        ("payload", {"payload": {1: "numeric-key"}}),
        ("payload", {"payload": {"nested": {1: "numeric-key"}}}),
        ("details", {"details": {1: "numeric-key"}}),
    ],
)
def test_non_string_json_mapping_keys_raise_type_error(field_name: str, kwargs: dict) -> None:
    if field_name == "details":
        with pytest.raises(TypeError):
            SyncReadinessReport(domain="notes", **kwargs)
        return

    event_kwargs = {
        "source_authority": "local",
        "server_profile_id": None,
        "stream_name": "notifications",
        "stream_instance_id": "default",
        "event_kind": "notification.created",
        "entity_ref": {"type": "notification", "id": "n-1"},
        "payload_hash": "hash",
    }
    event_kwargs.update(kwargs)

    with pytest.raises(TypeError):
        NormalizedEventRecord(**event_kwargs)


def test_tuple_like_fields_are_normalized_and_mutation_safe() -> None:
    reason_codes = ["not_registered"]
    notes = ["provider-backed"]

    report = SyncReadinessReport(domain="chat", reason_codes=reason_codes)
    status = ProviderMigrationStatus(service_name="chat", notes=notes)

    reason_codes.append("mutated")
    notes.append("mutated")

    assert report.reason_codes == ("not_registered",)
    assert status.notes == ("provider-backed",)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SyncReadinessReport(domain="chat", reason_codes="not_registered"),
        lambda: ProviderMigrationStatus(service_name="chat", notes="provider-backed"),
    ],
)
def test_string_tuple_fields_reject_plain_string_input(factory) -> None:
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SyncReadinessReport(domain="chat", reason_codes=["not_registered", 1]),
        lambda: ProviderMigrationStatus(service_name="chat", notes=["provider-backed", 1]),
    ],
)
def test_string_tuple_fields_reject_non_string_items(factory) -> None:
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SyncReadinessReport(domain="chat", reason_codes={"not_registered": "detail"}),
        lambda: ProviderMigrationStatus(service_name="chat", notes={"provider-backed": "detail"}),
    ],
)
def test_string_tuple_fields_reject_mapping_input(factory) -> None:
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EventCursor(
            source_authority="remote",
            server_profile_id=None,
            stream_name="notifications",
            stream_instance_id="default",
        ),
        lambda: NormalizedEventRecord(
            source_authority="remote",
            server_profile_id=None,
            stream_name="notifications",
            stream_instance_id="default",
            event_kind="notification.created",
            entity_ref={"type": "notification", "id": "n-1"},
            payload_hash="hash",
        ),
        lambda: EventDedupeKey(
            source_authority="remote",
            server_profile_id=None,
            stream_name="notifications",
            stream_instance_id="default",
            event_kind="notification.created",
            entity_id="n-1",
            timestamp=None,
            payload_hash="hash",
        ),
        lambda: SyncIdentityMapEntry(
            domain="notes",
            source_authority="remote",
            source_scope="workspace",
            local_entity_id="local-1",
        ),
    ],
)
def test_invalid_source_authority_values_raise_value_error(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_invalid_event_transport_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        NormalizedEventRecord(
            source_authority="local",
            server_profile_id=None,
            stream_name="notifications",
            stream_instance_id="default",
            event_kind="notification.created",
            entity_ref={"type": "notification", "id": "n-1"},
            payload_hash="hash",
            transport_type="carrier_pigeon",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_delivery_state": "queued"},
        {"server_read_state": "seen"},
        {"server_dismiss_state": "archived"},
    ],
)
def test_invalid_notification_state_values_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        NotificationPresentationRecord(event_key="event-1", **kwargs)
