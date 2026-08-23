"""Pure Trace v2 collaboration bundle contract (TASK-19913)."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Chat import trajectory_export, trajectory_import
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)

FIXED_EXPORTED_AT = "2026-08-22T19:00:00+00:00"
SECRET_VALUES = (
    "sk-adversarial-1234567890",
    "Bearer nested-auth-token",
    "correct-horse-battery-staple",
)
LOCAL_PATH = "/Users/alice/private/trace.txt"
LONG_CONTENT = "diagnostic-body-" * 30
PRIVATE_IDENTIFIER = "customer-999-secret-id"


def _record(**overrides: object) -> TrajectoryRecord:
    values: dict[str, object] = {
        "seq": 1,
        "kind": "tool_result",
        "turn_id": "turn-1",
        "message_id": "message-1",
        "content_preview": "A bounded diagnostic summary",
        "usage": ProviderUsage(
            uncached_input=11,
            cache_read=3,
            output=7,
            provider="provider-a",
            model="model-a",
        ),
        "step_started_at": 10.0,
        "first_token_at": 11.5,
        "completed_at": 13.0,
        "model": "model-a",
        "provider": "provider-a",
        "payload": {"result": "ordinary diagnostic payload"},
        "variants": ("superseded answer",),
        "depth": 1,
        "event_id": "event-1",
        "conversation_id": "conversation-1",
        "source_seq": 41,
        "label": "Tool result",
        "status": "succeeded",
        "actor_kind": "tool",
        "actor_id": "fs_read",
        "run_id": "run-1",
        "parent_event_id": None,
        "source_event_id": None,
        "replacement_event_id": None,
        "observed_at": 12.75,
        "field_states": {
            "content_preview": "observed",
            "payload": "observed",
        },
        "sensitivity": "diagnostic",
    }
    values.update(overrides)
    return TrajectoryRecord(**values)  # type: ignore[arg-type]


def _snapshot(*records: TrajectoryRecord) -> TrajectorySnapshot:
    return TrajectorySnapshot(
        turns=(TrajectoryTurn("turn-1", records or (_record(),)),)
    )


def _build(
    snapshot: TrajectorySnapshot,
    profile: object | None = None,
    *,
    confirm_full: bool = False,
) -> dict:
    selected = profile or trajectory_export.TraceExportProfile.REDACTED_DIAGNOSTIC
    preflight = trajectory_export.preflight_trace_export(snapshot, profile=selected)
    return trajectory_export.build_trace_export(
        snapshot,
        preflight=preflight,
        confirm_full=confirm_full,
        exported_at=FIXED_EXPORTED_AT,
    )


def _canonical_digest(payload: dict) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned["integrity"].pop("digest", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resign(payload: dict) -> dict:
    payload["integrity"]["digest"] = _canonical_digest(payload)
    return payload


def test_profile_enum_and_redacted_diagnostic_default_manifest() -> None:
    profile = trajectory_export.TraceExportProfile
    assert {member.value for member in profile} == {
        "safe_summary",
        "redacted_diagnostic",
        "full_trace",
    }

    preflight = trajectory_export.preflight_trace_export(_snapshot())
    payload = trajectory_export.build_trace_export(
        _snapshot(), preflight=preflight, exported_at=FIXED_EXPORTED_AT
    )

    assert preflight.profile is profile.REDACTED_DIAGNOSTIC
    assert payload["format"] == "tldw-trace"
    assert payload["version"] == 2
    assert payload["manifest"]["profile"] == "redacted_diagnostic"
    assert payload["manifest"]["schema_version"] == 2
    assert payload["manifest"]["event_count"] == 2
    assert payload["manifest"]["privacy_inventory"] == preflight.privacy_inventory
    assert payload["events"][-1]["kind"] == "trace_export"
    assert payload["events"][-1]["payload"]["profile"] == "redacted_diagnostic"


def test_safe_summary_omits_payload_bodies_and_records_provenance() -> None:
    payload = _build(
        _snapshot(_record(payload={"result": LONG_CONTENT})),
        trajectory_export.TraceExportProfile.SAFE_SUMMARY,
    )
    event = payload["events"][0]

    assert event["payload"] is None
    assert event["field_states"]["payload"] == "omitted"
    assert event["field_provenance"]["payload"]["reason"] == "safe_summary"
    assert LONG_CONTENT not in json.dumps(payload)


def test_redacted_diagnostic_previews_content_and_redacts_paths_and_ids() -> None:
    payload = _build(
        _snapshot(
            _record(
                payload={
                    "result": LONG_CONTENT,
                    "path": LOCAL_PATH,
                    "customer_id": PRIVATE_IDENTIFIER,
                }
            )
        )
    )
    event = payload["events"][0]
    serialized = json.dumps(payload)

    assert LONG_CONTENT not in serialized
    assert LOCAL_PATH not in serialized
    assert PRIVATE_IDENTIFIER not in serialized
    assert event["payload"]["result"].endswith("…")
    assert event["field_states"]["payload"] in {"redacted", "truncated"}
    assert event["redaction_provenance"]
    assert payload["manifest"]["redaction_provenance"]


def test_redacted_profiles_alias_envelope_identifiers_without_breaking_lineage() -> (
    None
):
    first = _record(
        event_id="event:customer@example.com:1",
        conversation_id="conversation:customer@example.com",
        turn_id="turn:account-4481",
        message_id="message:private-4481",
        run_id="run:private-4481",
        actor_id="customer@example.com",
    )
    second = _record(
        seq=2,
        source_seq=42,
        event_id="event:customer@example.com:2",
        conversation_id=first.conversation_id,
        turn_id=first.turn_id,
        message_id="message:private-4482",
        run_id=first.run_id,
        actor_id=first.actor_id,
        parent_event_id=first.event_id,
    )

    payload = _build(_snapshot(first, second))
    serialized = json.dumps(payload)
    for private in (
        first.event_id,
        first.conversation_id,
        first.turn_id,
        first.message_id,
        first.run_id,
        first.actor_id,
        second.event_id,
        second.message_id,
    ):
        assert private not in serialized
    assert payload["events"][1]["parent_event_id"] == payload["events"][0]["event_id"]
    assert payload["lineage"][0]["target"] == payload["events"][0]["event_id"]

    full = _build(
        _snapshot(first, second),
        trajectory_export.TraceExportProfile.FULL_TRACE,
        confirm_full=True,
    )
    assert full["events"][0]["event_id"] == first.event_id
    assert full["events"][1]["parent_event_id"] == first.event_id


def test_full_trace_requires_confirmation_and_still_forbids_credentials() -> None:
    profile = trajectory_export.TraceExportProfile.FULL_TRACE
    snapshot = _snapshot(
        _record(
            payload={
                "api_key": SECRET_VALUES[0],
                "nested": {
                    "authorization": SECRET_VALUES[1],
                    "password": SECRET_VALUES[2],
                    "body": LONG_CONTENT,
                    "path": LOCAL_PATH,
                    "customer_id": PRIVATE_IDENTIFIER,
                },
            }
        )
    )
    preflight = trajectory_export.preflight_trace_export(snapshot, profile=profile)

    with pytest.raises(trajectory_export.TrajectoryExportError, match="confirm"):
        trajectory_export.build_trace_export(
            snapshot, preflight=preflight, exported_at=FIXED_EXPORTED_AT
        )

    payload = trajectory_export.build_trace_export(
        snapshot,
        preflight=preflight,
        confirm_full=True,
        exported_at=FIXED_EXPORTED_AT,
    )
    serialized = json.dumps(payload)
    for secret in SECRET_VALUES:
        assert secret not in serialized
    assert LONG_CONTENT in serialized
    assert LOCAL_PATH in serialized
    assert PRIVATE_IDENTIFIER in serialized


def test_full_trace_aliases_credentials_embedded_in_envelope_identifiers() -> None:
    secret_event_id = f"event:{SECRET_VALUES[0]}"
    snapshot = _snapshot(
        _record(
            event_id=secret_event_id,
            actor_id=SECRET_VALUES[1],
            conversation_id=f"conversation:password={SECRET_VALUES[2]}",
        )
    )

    payload = _build(
        snapshot,
        trajectory_export.TraceExportProfile.FULL_TRACE,
        confirm_full=True,
    )
    serialized = json.dumps(payload)

    for secret in SECRET_VALUES:
        assert secret not in serialized
    assert payload["events"][0]["event_id"] != secret_event_id


@pytest.mark.parametrize(
    "profile",
    ["safe_summary", "redacted_diagnostic", "full_trace"],
)
def test_credentials_are_scrubbed_from_usage_envelope_and_pem_fields(
    profile: str,
) -> None:
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "TOP-SECRET-PRIVATE-MATERIAL\n"
        "-----END PRIVATE KEY-----"
    )
    snapshot = _snapshot(
        _record(
            label=pem,
            status=f"password={SECRET_VALUES[2]}",
            usage=ProviderUsage(
                uncached_input=1,
                output=1,
                provider=SECRET_VALUES[1],
                model="model-a",
            ),
            payload={"private_key": pem},
        )
    )

    payload = _build(snapshot, profile, confirm_full=profile == "full_trace")
    serialized = json.dumps(payload)

    assert "TOP-SECRET-PRIVATE-MATERIAL" not in serialized
    for secret in SECRET_VALUES:
        assert secret not in serialized
    assert any(
        item["reason"] == "credential"
        for item in payload["manifest"]["redaction_provenance"]
    )


@pytest.mark.parametrize(
    "profile",
    ["safe_summary", "redacted_diagnostic", "full_trace"],
)
def test_credentials_are_absent_from_serialized_bytes_in_every_profile(
    profile: str,
) -> None:
    snapshot = _snapshot(
        _record(
            content_preview=f"password={SECRET_VALUES[2]}",
            payload={
                "api_key": SECRET_VALUES[0],
                "innocent": {"authorization": SECRET_VALUES[1]},
                SECRET_VALUES[0]: "credential used as a nested key",
            },
        )
    )
    payload = _build(snapshot, profile, confirm_full=profile == "full_trace")
    serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for secret in SECRET_VALUES:
        assert secret.encode() not in serialized


def test_preflight_has_one_decision_per_material_field_without_count_drift() -> None:
    snapshot = _snapshot(
        _record(
            field_states={
                "content_preview": "redacted",
                "payload": "capture_failed",
                "variants": "truncated",
                "model": "omitted",
                "provider": "not_available",
            }
        )
    )
    preflight = trajectory_export.preflight_trace_export(snapshot)
    decisions = preflight.field_decisions

    keys = [(decision.event_id, decision.field) for decision in decisions]
    assert len(keys) == len(set(keys))
    assert {(decision.field, decision.state) for decision in decisions} >= {
        ("content_preview", "redacted"),
        ("payload", "capture_failed"),
        ("variants", "truncated"),
        ("model", "omitted"),
        ("provider", "not_available"),
    }
    states = [decision.state for decision in decisions]
    inventory = preflight.privacy_inventory
    assert inventory["redacted"] == states.count("redacted")
    assert inventory["omitted"] == states.count("omitted")
    assert inventory["truncated"] == states.count("truncated")
    assert inventory["capture_failed"] == states.count("capture_failed")
    assert inventory["missing"] == sum(
        state in {"not_available", "capture_failed"} for state in states
    )
    assert inventory["sensitive"] == sum(d.sensitive for d in decisions)


def test_preflight_counts_overlapping_redaction_and_truncation_once_each() -> None:
    preflight = trajectory_export.preflight_trace_export(
        _snapshot(
            _record(
                payload={
                    "api_key": SECRET_VALUES[0],
                    "result": LONG_CONTENT,
                }
            )
        )
    )

    redactions = {
        (item["event_id"], item["field"], item["state"])
        for item in preflight.redaction_provenance
        if item["state"] == "redacted"
    }
    assert preflight.privacy_inventory["redacted"] == len(redactions)
    assert any(
        item["reason"] == "credential" for item in preflight.redaction_provenance
    )
    assert preflight.privacy_inventory["truncated"] == 1
    assert preflight.privacy_inventory["observed"] == sum(
        decision.source_state == "observed" for decision in preflight.field_decisions
    )


def test_safe_summary_coarsens_timing_and_records_provenance() -> None:
    payload = _build(
        _snapshot(
            _record(
                observed_at=12.875,
                step_started_at=10.625,
                first_token_at=11.875,
                completed_at=13.625,
            )
        ),
        trajectory_export.TraceExportProfile.SAFE_SUMMARY,
    )
    event = payload["events"][0]

    assert event["observed_at"] == 12.0
    assert event["step_started_at"] == 10.0
    assert event["first_token_at"] == 11.0
    assert event["completed_at"] == 13.0
    assert all(
        event["field_states"][field] == "truncated"
        for field in (
            "observed_at",
            "step_started_at",
            "first_token_at",
            "completed_at",
        )
    )
    assert {
        item["field"]
        for item in payload["manifest"]["redaction_provenance"]
        if item["reason"] == "coarse_timing_1s"
    } == {
        "observed_at",
        "step_started_at",
        "first_token_at",
        "completed_at",
    }


def test_builder_rejects_a_preflight_from_a_different_snapshot() -> None:
    first = _snapshot(_record(event_id="event-1"))
    second = _snapshot(_record(event_id="event-2"))
    preflight = trajectory_export.preflight_trace_export(first)

    with pytest.raises(trajectory_export.TrajectoryExportError, match="snapshot"):
        trajectory_export.build_trace_export(
            second, preflight=preflight, exported_at=FIXED_EXPORTED_AT
        )


def test_round_trip_preserves_order_identity_lineage_timing_usage_and_missing() -> None:
    first = _record(
        event_id="event-1",
        seq=7,
        source_seq=41,
        field_states={"payload": "not_available"},
        payload=None,
    )
    second = _record(
        event_id="event-2",
        seq=8,
        source_seq=42,
        parent_event_id="event-1",
        source_event_id="event-1",
        replacement_event_id="event-2",
        field_states={"payload": "observed"},
    )
    payload = _build(
        _snapshot(first, second),
        trajectory_export.TraceExportProfile.FULL_TRACE,
        confirm_full=True,
    )

    imported = trajectory_import.load_imported_trace(payload)
    records = [record for turn in imported.snapshot.turns for record in turn.records]

    assert [record.event_id for record in records] == [
        "event-1",
        "event-2",
        payload["events"][-1]["event_id"],
    ]
    restored = records[:2]
    assert [record.seq for record in restored] == [7, 8]
    assert [record.source_seq for record in restored] == [41, 42]
    assert restored[1].parent_event_id == "event-1"
    assert restored[1].source_event_id == "event-1"
    assert restored[1].replacement_event_id == "event-2"
    assert restored[1].observed_at == 12.75
    assert restored[1].usage == second.usage
    assert restored[0].field_states["payload"] == "not_available"
    assert records[-1].kind == "trace_export"
    assert imported.operation_event.kind == "trace_import"
    assert imported.operation_event.event_id not in {
        event["event_id"] for event in payload["events"]
    }
    assert imported.integrity["verified"] is True
    assert imported.privacy_inventory == payload["manifest"]["privacy_inventory"]
    with pytest.raises(FrozenInstanceError):
        imported.snapshot = _snapshot()  # type: ignore[misc]


def test_canonical_digest_is_deterministic_and_detects_tampering() -> None:
    snapshot = _snapshot()
    one = _build(snapshot)
    two = _build(snapshot)

    assert one == two
    assert one["integrity"] == {
        "algorithm": "sha256",
        "digest": _canonical_digest(one),
        "authenticity": False,
    }

    tampered = copy.deepcopy(one)
    tampered["events"][0]["status"] = "tampered"
    with pytest.raises(
        trajectory_import.TrajectoryImportError, match="digest mismatch"
    ):
        trajectory_import.load_imported_trace(tampered)


def test_malformed_digest_is_rejected_actionably() -> None:
    payload = _build(_snapshot())
    payload["integrity"]["digest"] = "not-a-sha256"

    with pytest.raises(trajectory_import.TrajectoryImportError, match="digest"):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_cannot_claim_digest_authenticity() -> None:
    payload = _build(_snapshot())
    payload["integrity"]["authenticity"] = True
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError, match="authenticity.*false"
    ):
        trajectory_import.load_imported_trace(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(version=3), "version 3"),
        (lambda p: p["manifest"].update(profile="unsafe"), "profile"),
        (lambda p: p["manifest"].update(profile=[]), "profile"),
        (
            lambda p: p["manifest"]["privacy_inventory"].update(redacted=-1),
            "privacy_inventory",
        ),
        (
            lambda p: p["manifest"]["privacy_inventory"].pop("sensitive"),
            "privacy_inventory.*sensitive",
        ),
        (lambda p: p["manifest"].pop("missing_metadata"), "missing_metadata"),
        (lambda p: p["manifest"].pop("redaction_provenance"), "redaction_provenance"),
        (lambda p: p["events"].__setitem__(0, "bad-event"), r"events\[0\]"),
        (lambda p: p["events"][0].pop("observed_at"), r"events\[0\].observed_at"),
        (
            lambda p: p["events"][0].pop("field_provenance"),
            r"events\[0\].field_provenance",
        ),
        (
            lambda p: p["events"][0].pop("redaction_provenance"),
            r"events\[0\].redaction_provenance",
        ),
        (lambda p: p["events"][0].update(seq="not-an-integer"), r"events\[0\].seq"),
        (lambda p: p["events"][0].update(kind=[]), r"events\[0\].kind"),
        (lambda p: p["events"][0].update(turn_id={}), r"events\[0\].turn_id"),
        (
            lambda p: p["events"][0].update(step_started_at="yesterday"),
            r"events\[0\].step_started_at",
        ),
        (lambda p: p["events"][0].update(variants=[{}]), r"events\[0\].variants"),
        (
            lambda p: p["events"].append(copy.deepcopy(p["events"][0])),
            "duplicate.*event_id",
        ),
        (
            lambda p: p["events"][0].update(parent_event_id="missing-event"),
            "dangling.*parent_event_id",
        ),
        (
            lambda p: p["events"][0].update(source_event_id="missing-event"),
            "dangling.*source_event_id",
        ),
        (
            lambda p: p["events"][0].update(replacement_event_id="missing-event"),
            "dangling.*replacement_event_id",
        ),
        (
            lambda p: p["lineage"].append(
                {
                    "source": "event-1",
                    "target": "missing-event",
                    "relationship": "parent",
                }
            ),
            "lineage",
        ),
    ],
)
def test_malformed_v2_structures_fail_closed(mutation, message: str) -> None:
    payload = _build(_snapshot())
    mutation(payload)
    if isinstance(payload.get("integrity"), dict):
        _resign(payload)

    with pytest.raises(trajectory_import.TrajectoryImportError, match=message):
        trajectory_import.load_imported_trace(payload)


def test_resigned_lineage_must_exactly_match_event_reference_fields() -> None:
    first = _record(event_id="event-1")
    second = _record(event_id="event-2", seq=2, parent_event_id="event-1")
    payload = _build(_snapshot(first, second))
    payload["lineage"][0]["target"] = payload["events"][1]["event_id"]
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError, match="lineage does not match events"
    ):
        trajectory_import.load_imported_trace(payload)


def test_v2_import_module_has_no_database_or_textual_imports() -> None:
    source = Path(trajectory_import.__file__).read_text(encoding="utf-8")
    imported_names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_names.append(node.module or "")

    assert not any(name.startswith("tldw_chatbook.DB") for name in imported_names)
    assert not any(name.startswith("textual") for name in imported_names)
    assert "ChaChaNotes_DB" not in source
    assert "sqlite3" not in imported_names
