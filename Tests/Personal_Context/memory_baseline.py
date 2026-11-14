"""Offline, test-owned measurements of real Personal Context read paths.

Run through pytest so its configuration, keyring and network isolation apply.
This module never opens the user's profile or evaluates generated answers.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import TypeAdapter
from tldw_profile_core import (
    ProfileControls,
    ProfilePayload,
    ProfileProvenance,
    ProfileRecord,
    SemanticKey,
)

from tldw_chatbook.Agents.profile_tool_provider import (
    ProfileToolProvider,
    ProfileToolRunScope,
)
from tldw_chatbook.Personal_Context.context_service import (
    ProfileContextRequest,
    ProfileContextService,
)
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    RecordMutation,
)
from tldw_chatbook.Utils import token_counter

MAX_MANIFEST_BYTES = 256 * 1024
_SCOPES = {"global", "work", "other"}
_PAYLOAD = TypeAdapter(ProfilePayload)
_CASE_KEYS = {
    "id",
    "split",
    "category",
    "record_labels",
    "query",
    "active_scope",
    "available_input_tokens",
    "actions",
    "eligible_labels",
    "relevant_labels",
    "expected_context_labels",
    "forbidden_labels",
    "expected_search_status",
    "metadata_expectations",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: Any, keys: set[str]) -> None:
    _require(type(value) is dict and set(value) == keys, "invalid object fields")


def _labels(value: Any, allowed: set[str]) -> None:
    _require(type(value) is list, "labels must be a list")
    _require(
        all(type(item) is str and item in allowed for item in value), "unknown label"
    )
    _require(len(value) == len(set(value)), "duplicate labels")


def _text(value: Any, maximum: int = 16_384) -> None:
    _require(
        type(value) is str and bool(value.strip()) and len(value) <= maximum,
        "invalid text",
    )


def _integer(value: Any, minimum: int, maximum: int) -> None:
    _require(type(value) is int and minimum <= value <= maximum, "invalid integer")


def _validate_manifest(data: Any) -> dict[str, Any]:
    _keys(
        data,
        {"version", "synthetic", "now", "k", "records", "cases", "source_messages"},
    )
    _require(type(data["version"]) is int and data["version"] == 1, "invalid version")
    _require(data["synthetic"] is True, "synthetic fixtures required")
    _integer(data["k"], 1, 20)
    _text(data["now"], 64)
    _require(
        datetime.fromisoformat(data["now"]).utcoffset() is not None,
        "aware clock required",
    )
    messages = data["source_messages"]
    _require(type(messages) is dict and len(messages) <= 32, "invalid source messages")
    for label, message in messages.items():
        _text(label, 128)
        _text(message)
    records = data["records"]
    _require(type(records) is dict and 1 <= len(records) <= 32, "invalid records")
    for label, record in records.items():
        _text(label, 64)
        _keys(
            record,
            {
                "scope",
                "payload",
                "semantic_key",
                "controls",
                "provenance",
                "expires_after_seconds",
            },
        )
        _require(record["scope"] in _SCOPES, "invalid record scope")
        payload = _PAYLOAD.validate_python(record["payload"])
        _require(
            payload.kind in {"preference", "constraint", "working_context"},
            "invalid record kind",
        )
        ProfileControls.model_validate(record["controls"])
        provenance = ProfileProvenance.model_validate(record["provenance"])
        _require(
            set(provenance.source_references) <= set(messages),
            "unknown source reference",
        )
        if record["semantic_key"] is not None:
            SemanticKey.model_validate(record["semantic_key"])
        if record["expires_after_seconds"] is not None:
            _require(
                payload.kind == "working_context", "expiry requires working context"
            )
            _integer(record["expires_after_seconds"], 1, 86_400)
    _require(
        type(data["cases"]) is list and 1 <= len(data["cases"]) <= 32, "invalid cases"
    )
    seen = set()
    for case in data["cases"]:
        _keys(case, _CASE_KEYS)
        _text(case["id"], 3)
        _require(
            re.fullmatch(r"[dh](0[1-9]|1[0-2])", case["id"]) is not None,
            "invalid case id",
        )
        _require(case["id"] not in seen, "duplicate case id")
        seen.add(case["id"])
        expected_split = "development" if case["id"].startswith("d") else "held_out"
        _require(case["split"] == expected_split, "invalid partition")
        _text(case["category"], 64)
        _text(case["query"])
        _require(case["active_scope"] in _SCOPES, "invalid active scope")
        _integer(case["available_input_tokens"], 0, 1_000_000)
        _labels(case["record_labels"], set(records))
        _require(1 <= len(case["record_labels"]) <= 8, "invalid record count")
        seeded = set(case["record_labels"])
        for key in (
            "eligible_labels",
            "relevant_labels",
            "expected_context_labels",
            "forbidden_labels",
        ):
            _labels(case[key], seeded)
        eligible = set(case["eligible_labels"])
        _require(
            set(case["relevant_labels"]) <= eligible, "relevance outside eligibility"
        )
        _require(
            set(case["expected_context_labels"]) <= eligible,
            "context outside eligibility",
        )
        _require(
            set(case["forbidden_labels"]) == seeded - eligible,
            "incomplete eligibility labels",
        )
        _require(
            case["expected_search_status"] in {"applied", "permission_denied"},
            "invalid expected status",
        )
        _require(
            type(case["actions"]) is list and len(case["actions"]) <= 8,
            "invalid actions",
        )
        for action in case["actions"]:
            _require(
                type(action) is dict and type(action.get("op")) is str, "invalid action"
            )
            operation = action["op"]
            action_keys = {
                "update": {"op", "label", "value"},
                "archive": {"op", "label"},
                "delete": {"op", "label"},
                "quarantine": {"op", "label"},
                "advance_clock": {"op", "seconds"},
                "disable_after_capture": {"op"},
            }
            _require(operation in action_keys, "unsupported action")
            _keys(action, action_keys[operation])
            if "label" in action:
                _require(action["label"] in seeded, "unknown action target")
            if operation == "update":
                _require(
                    records[action["label"]]["payload"]["kind"] == "preference",
                    "unsupported update kind",
                )
                _text(action["value"])
            if operation == "advance_clock":
                _integer(action["seconds"], 1, 86_400)
        revoked = any(
            action["op"] == "disable_after_capture" for action in case["actions"]
        )
        _require(
            revoked == (case["expected_search_status"] == "permission_denied"),
            "invalid denial scenario",
        )
        if revoked:
            _require(not eligible, "revoked scope must have no eligible records")
        metadata = case["metadata_expectations"]
        _require(
            type(metadata) is dict and set(metadata) <= eligible,
            "invalid metadata targets",
        )
        for expected in metadata.values():
            _keys(expected, {"source_references", "source_hashes"})
            _labels(expected["source_references"], set(messages))
            _require(type(expected["source_hashes"]) is list, "invalid hashes")
            _require(
                all(
                    type(item) is str and re.fullmatch(r"[a-fA-F0-9]{64}", item)
                    for item in expected["source_hashes"]
                ),
                "invalid hashes",
            )
    return data


def _load_manifest_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    _require(len(raw) <= MAX_MANIFEST_BYTES, "manifest exceeds byte limit")
    try:
        manifest = _validate_manifest(json.loads(raw.decode("utf-8")))
    except (TypeError, KeyError, UnicodeError) as exc:
        raise ValueError("invalid manifest structure") from exc
    return manifest, hashlib.sha256(raw).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load bounded synthetic inputs without reading any production state."""
    return _load_manifest_with_hash(path)[0]


def score_search(
    returned: tuple[str, ...], relevant: frozenset[str], *, k: int
) -> dict[str, float | int | bool | None]:
    """Score actual ordered results; empty gold sets have undefined recall."""
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")
    if len(returned) > k:
        raise ValueError("result count exceeds k")
    if len(set(returned)) != len(returned):
        raise ValueError("duplicate result labels")
    hits = sum(label in relevant for label in returned)
    first = next((i for i, label in enumerate(returned, 1) if label in relevant), None)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant) if relevant else None,
        "reciprocal_rank": (1 / first if first else 0.0) if relevant else None,
        "false_positive_count": len(returned) - hits,
        "empty_result": not returned,
    }


@dataclass
class BaselineClock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now


@dataclass
class BaselineIds:
    counter: int = 0

    def __call__(self, label: str) -> str:
        self.counter += 1
        return f"{label}-{self.counter:04d}"


class BaselineContractError(ValueError):
    """An enumerated, body-free caller-contract failure in a synthetic run."""


def _check(condition: bool, code: str) -> None:
    if not condition:
        raise BaselineContractError(code)


@contextmanager
def _offline_estimator():
    # Both full suites and individually probed cases must remain offline.
    with (
        patch.object(token_counter, "TIKTOKEN_AVAILABLE", False),
        patch.object(token_counter, "CUSTOM_TOKENIZERS_AVAILABLE", False),
    ):
        token_counter.clear_estimate_cache()
        try:
            yield
        finally:
            token_counter.clear_estimate_cache()


def _response(result: Any, operation: str, expected_status: str) -> dict[str, Any]:
    if expected_status != "applied":
        _check(result.ok is False, "inconsistent_tool_ok")
        _check(result.error == expected_status, "unexpected_tool_status")
        _check(result.content == "", "denial_contains_body")
        return {"operation": operation, "status": result.error}
    try:
        value = json.loads(result.content)
    except (TypeError, ValueError) as exc:
        raise BaselineContractError("malformed_tool_json") from exc
    _check(type(value) is dict, "invalid_tool_object")
    _check(value.get("operation") == operation, "unexpected_tool_operation")
    _check(value.get("status") == expected_status, "unexpected_tool_status")
    _check(result.ok is (expected_status == "applied"), "inconsistent_tool_ok")
    _check(result.error == "", "success_contains_error")
    return value


def _snapshot_body(snapshot: Any) -> dict[str, Any]:
    if not snapshot.serialized_block:
        _check(not snapshot.source_version_ids, "empty_snapshot_has_versions")
        return {"records": []}
    try:
        body = json.loads(snapshot.serialized_block.split("\n", 2)[2])
    except (IndexError, TypeError, ValueError) as exc:
        raise BaselineContractError("malformed_snapshot_json") from exc
    _check(
        type(body) is dict and type(body.get("records")) is list,
        "invalid_snapshot_records",
    )
    return body


def _expected_context_record(
    record: ProfileRecord, active_scope_id: str | None
) -> dict[str, Any]:
    value = {
        "kind": record.kind.value,
        "scope": "workspace" if record.scope_id == active_scope_id else "global",
        "payload": record.payload.model_dump(mode="json"),
    }
    if record.semantic_key is not None:
        value["semantic_key"] = record.semantic_key.model_dump(mode="json")
    return value


def _seed(case, manifest, root, repository, clock):
    ids = BaselineIds()
    service = PersonalContextService(repository, clock=clock, id_factory=ids)
    profile = service.create_profile()
    scopes = {"global": service.list_scopes()[0]}
    needed = {manifest["records"][label]["scope"] for label in case["record_labels"]}
    needed.add(case["active_scope"])
    for name in sorted(needed - {"global"}):
        scopes[name] = service.create_workspace_scope(f"baseline-{name}", name)
    service.set_runtime_enabled(True)
    for scope in scopes.values():
        service.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    records = {}
    for label in case["record_labels"]:
        spec = manifest["records"][label]
        expiry = spec["expires_after_seconds"]
        records[label] = service.create_record(
            ProfileRecord(
                profile_id=profile.profile_id,
                record_id=ids("record"),
                version_id=ids("record-version"),
                parent_version_id=None,
                scope_id=scopes[spec["scope"]].scope_id,
                kind=spec["payload"]["kind"],
                payload=spec["payload"],
                semantic_key=spec["semantic_key"],
                controls=spec["controls"],
                provenance=spec["provenance"],
                state="active",
                created_at=clock(),
                updated_at=clock(),
                expires_at=clock() + timedelta(seconds=expiry) if expiry else None,
                no_expiry=spec["payload"]["kind"] == "working_context"
                and expiry is None,
            )
        )
    for action in case["actions"]:
        op = action["op"]
        if op in {"disable_after_capture"}:
            continue
        if op == "advance_clock":
            clock.now += timedelta(seconds=action["seconds"])
            continue
        label = action["label"]
        current = records[label]
        if op == "quarantine":
            repository.quarantine_object(
                "record", current.record_id, current.version_id, "unsupported_kind"
            )
        elif op == "update":
            payload = {
                **current.payload.model_dump(mode="json"),
                "value": action["value"],
            }
            records[label] = service.update_record(
                current.record_id,
                RecordMutation(payload=_PAYLOAD.validate_python(payload)),
                expected_version_id=current.version_id,
            )
        elif op == "archive":
            records[label] = service.archive_record(
                current.record_id, expected_version_id=current.version_id
            )
        elif op == "delete":
            records[label] = service.delete_record(
                current.record_id, expected_version_id=current.version_id
            )
    return service, scopes, records


def _observe(case, manifest, service, scopes, records, clock, outcome):
    scope = scopes[case["active_scope"]]
    active_scope_id = scope.scope_id if scope.kind.value == "workspace" else None
    view = service.authorized_context_view(active_workspace_scope_id=active_scope_id)
    provider = ProfileToolProvider(
        service,
        run_scope=ProfileToolRunScope(
            run_id=f"baseline:{case['id']}",
            session_id="baseline-session",
            profile_id=service.get_manifest().profile_id,
            scope_id=scope.scope_id,
            authority=AgentAuthority.READ_ONLY,
            generation=view.generation,
            authority_revision=view.authority_revision,
        ),
    )
    builder = ProfileContextService(service, clock=clock)

    def snapshot_for(query, budget):
        return builder.build_snapshot(
            ProfileContextRequest(
                current_user_text=query,
                available_input_tokens=budget,
                model="memory-baseline-v1",
                provider="",
                active_workspace_scope_id=active_scope_id,
            )
        )

    candidates = (
        case["expected_context_labels"] or case["eligible_labels"] or ["control"]
    )
    # Positive probes need a permitted control after the device-only ceiling.
    # Frozen selection and relevance labels still score the measured outputs below.
    candidates = [
        label
        for label in candidates
        if manifest["records"][label]["controls"]["sync_mode"] == "syncable"
    ]
    _check(bool(candidates), "missing_syncable_positive_control")
    control = records[
        min(
            candidates,
            key=lambda label: len(manifest["records"][label]["payload"]["value"]),
        )
    ]
    search_control = provider.invoke(
        "profile_search", {"query": control.payload.subject, "limit": 20}
    )
    positive_search = _response(search_control, "search", "applied")
    positive_records = positive_search.get("data", {}).get("records", [])
    _check(
        any(
            record.get("record_id") == control.record_id for record in positive_records
        ),
        "positive_search_failed",
    )
    get_control = provider.invoke("profile_get", {"record_id": control.record_id})
    positive_get = _response(get_control, "get", "applied")
    _check(
        positive_get.get("data", {}).get("record") == control.model_dump(mode="json"),
        "positive_get_failed",
    )
    positive_snapshot = snapshot_for(control.payload.subject, 100_000)
    _check(
        control.version_id in positive_snapshot.source_version_ids,
        "positive_context_failed",
    )
    _snapshot_body(positive_snapshot)
    outcome["positive_control_passed"] = True
    if case["expected_search_status"] == "permission_denied":
        service.set_runtime_enabled(False)
    result = provider.invoke(
        "profile_search", {"query": case["query"], "limit": manifest["k"]}
    )
    response = _response(result, "search", case["expected_search_status"])
    outcome["search_status"] = response["status"]
    id_labels = {record.record_id: label for label, record in records.items()}
    returned = []
    observed = {}
    if response["status"] == "applied":
        data = response.get("data")
        _check(
            type(data) is dict and type(data.get("records")) is list,
            "invalid_search_records",
        )
        for value in data["records"]:
            _check(
                type(value) is dict and type(value.get("record_id")) is str,
                "invalid_search_record",
            )
            label = id_labels.get(value["record_id"])
            _check(label is not None, "unknown_search_record")
            _check(label not in returned, "duplicate_search_record")
            _check(
                value == records[label].model_dump(mode="json"), "search_record_changed"
            )
            returned.append(label)
            observed[label] = value
        _check(len(returned) <= manifest["k"], "search_limit_exceeded")
    outcome["returned_labels"] = returned
    snapshot = snapshot_for(case["query"], case["available_input_tokens"])
    body = _snapshot_body(snapshot)
    versions = {record.version_id: label for label, record in records.items()}
    selected = []
    for version in snapshot.source_version_ids:
        _check(version in versions, "unknown_source_version")
        _check(versions[version] not in selected, "duplicate_source_version")
        selected.append(versions[version])
    _check(
        body["records"]
        == [
            _expected_context_record(records[label], active_scope_id)
            for label in selected
        ],
        "snapshot_payload_mismatch",
    )
    outcome["selected_labels"] = selected
    outcome["context_bytes"] = len(snapshot.serialized_block.encode("utf-8"))
    outcome["context_tokens"] = snapshot.estimated_tokens
    if selected != case["expected_context_labels"]:
        outcome["context_checks"].append("selection_mismatch")
    if outcome["context_bytes"] > 12_288:
        outcome["context_checks"].append("byte_budget_exceeded")
    if snapshot.estimated_tokens > case["available_input_tokens"] // 10:
        outcome["context_checks"].append("token_budget_exceeded")
    if snapshot.estimated_tokens != token_counter.estimate_tokens(
        snapshot.serialized_block, model="memory-baseline-v1", provider=""
    ):
        outcome["context_checks"].append("token_estimate_mismatch")
    raw_outputs = [result.content, result.error or "", snapshot.serialized_block]
    if case["expected_search_status"] == "applied":
        # Revocation's control was authorized before runtime was disabled.
        raw_outputs.extend(
            (
                search_control.content,
                get_control.content,
                positive_snapshot.serialized_block,
            )
        )
    for label in case["forbidden_labels"]:
        denied = provider.invoke("profile_get", {"record_id": records[label].record_id})
        try:
            _response(denied, "get", "permission_denied")
        except BaselineContractError:
            outcome["authority_failures"].append(f"get_disclosed:{label}")
        raw_outputs.extend((denied.content, denied.error or ""))
        canaries = (
            records[label].record_id,
            records[label].version_id,
            manifest["records"][label]["payload"]["value"],
        )
        if (
            label in returned
            or label in selected
            or any(canary in output for canary in canaries for output in raw_outputs)
        ):
            outcome["authority_failures"].append(f"forbidden_record:{label}")
    for action in case["actions"]:
        if action["op"] == "update":
            old = manifest["records"][action["label"]]["payload"]["value"]
            if old in result.content or old in snapshot.serialized_block:
                outcome["context_checks"].append("old_wording_retained")
    for label in selected:
        if manifest["records"][label]["controls"]["sync_mode"] == "device_only":
            outcome["policy_gaps"].append("device_only_in_provider_context")
    if body.get("unsupported_records_present"):
        outcome["policy_gaps"].append("unscoped_quarantine_signal")
    for label, expected in case["metadata_expectations"].items():
        if label not in observed:
            outcome["metadata"][label] = {"status": "not_retrieved"}
            continue
        actual = {key: observed[label]["provenance"][key] for key in expected}
        outcome["metadata"][label] = actual
        if actual != expected:
            outcome["context_checks"].append(f"metadata_mismatch:{label}")
    if response["status"] == "applied":
        outcome["search_metrics"] = score_search(
            tuple(returned), frozenset(case["relevant_labels"]), k=manifest["k"]
        )


def run_case(
    case: dict[str, Any], manifest: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Measure one synthetic case through real service/tool/snapshot entry points."""
    outcome = {
        "id": case["id"],
        "split": case["split"],
        "category": case["category"],
        "search_status": "unmeasured",
        "returned_labels": [],
        "selected_labels": [],
        "search_metrics": None,
        "positive_control_passed": False,
        "context_bytes": None,
        "context_tokens": None,
        "metadata": {},
        "context_checks": [],
        "authority_failures": [],
        "policy_gaps": [],
        "harness_errors": [],
    }
    root.mkdir(parents=True, exist_ok=False)
    repository = PersonalContextRepository(
        root / "synthetic-profile.db", key_protector=InMemoryProfileKeyProtector()
    )
    clock = BaselineClock(datetime.fromisoformat(manifest["now"]))
    try:
        with _offline_estimator():
            service, scopes, records = _seed(case, manifest, root, repository, clock)
            _observe(case, manifest, service, scopes, records, clock, outcome)
    except BaselineContractError as exc:
        outcome["harness_errors"].append(str(exc))
        outcome["search_metrics"] = None
    finally:
        repository.close()
    return outcome


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [
        case["search_metrics"] for case in cases if case["search_metrics"] is not None
    ]
    summary = {"case_count": len(cases), "ranked_case_count": len(ranked)}
    for metric in ("precision_at_k", "recall_at_k", "reciprocal_rank"):
        values = [result[metric] for result in ranked if result[metric] is not None]
        summary[metric] = {
            "mean": sum(values) / len(values) if values else None,
            "denominator": len(values),
        }
    summary["false_positive_count"] = sum(
        result["false_positive_count"] for result in ranked
    )
    return summary


def run_suite(manifest_path: Path, root: Path) -> dict[str, Any]:
    """Produce deterministic measurements, preserving failed policy checks."""
    manifest, fixture_sha256 = _load_manifest_with_hash(manifest_path)
    with _offline_estimator():
        cases = [
            run_case(case, manifest, root / case["id"]) for case in manifest["cases"]
        ]
    cases.sort(key=lambda case: case["id"])
    report = {
        "version": 1,
        "fixture_sha256": fixture_sha256,
        "synthetic": True,
        "now": manifest["now"],
        "k": manifest["k"],
        "tokenizer_mode": "production_chars_fallback_v1",
        "cases": cases,
        "unmeasured": [
            "provenance_ui",
            "semantic_support",
            "edit_history",
            "answer_quality",
        ],
        "summaries": {},
    }
    for split in ("development", "held_out"):
        group = [case for case in cases if case["split"] == split]
        report["summaries"][split] = {
            "overall": _summarize(group),
            "categories": {
                category: _summarize(
                    [case for case in group if case["category"] == category]
                )
                for category in sorted({case["category"] for case in group})
            },
        }
    for field in (
        "harness_errors",
        "authority_failures",
        "policy_gaps",
        "context_checks",
    ):
        report[field] = [
            {"case": case["id"], "code": code} for case in cases for code in case[field]
        ]
    report["disclosure_checks_passed"] = not any(
        report[field]
        for field in ("harness_errors", "authority_failures", "policy_gaps")
    )
    report["context_checks_passed"] = (
        not report["harness_errors"] and not report["context_checks"]
    )
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    """Write new UTF-8 evidence without replacing existing files."""
    serialized = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    created = False
    try:
        with output.open("x", encoding="utf-8") as stream:
            created = True
            stream.write(serialized)
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
