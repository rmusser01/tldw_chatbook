"""Disposable fixed native terminal shapes, priced before target effects."""

import hashlib
import json
from pathlib import Path

from . import bootstrap
from .journal import (
    MAX_EVENTS,
    _encoded,
    _Event,
    _MobileCommitted,
)
from .limits import RECOVERY_RECORD_BYTES
from .mobile_rollback_io import observation
from .mobile_runs_goal_budget import _serialization
from .storage_admission import _mobile_coordinator_budget

_UINT64 = 2**64 - 1
_COUNT = 2**63 - 1
_HASH = "f" * 64


def _check_terminal_dimensions(value, budget):
    """Check the same fixed native scalar/count domain on actual completion."""
    budget.container(8)
    for target in value["targets"]:
        budget.container(2)
        for group in (target["witnesses"], target["sidecars"]):
            for witness in group:
                for key in ("device", "inode", "mode"):
                    number = witness[key]
                    if number is not None and (
                        type(number) is not int or not 0 <= number <= _UINT64
                    ):
                        raise ValueError("mobile_terminal_dimension_invalid")
    for key in ("pending", "bootstrap"):
        for field in ("device", "inode"):
            number = value[key][field]
            if type(number) is not int or not 0 <= number <= _UINT64:
                raise ValueError("mobile_terminal_dimension_invalid")
    for owner in value["owners"]:
        for key in ("entry_count", "command_history_count"):
            number = owner[key]
            if number is not None and (
                type(number) is not int or not 0 <= number <= _COUNT
            ):
                raise ValueError("mobile_terminal_dimension_invalid")
    if not 0 <= value["pending"]["size"] <= bootstrap.MAX_RECORD:
        raise ValueError("mobile_terminal_dimension_invalid")


def _preprice_terminal(captured, *, native_inputs, receipts, history_location):
    """Allocate/validate/discard a maximum-width terminal; return no authority."""
    budget = _mobile_coordinator_budget(captured.session)
    budget.reserve(512)
    # Sizing-only values from this call's ORIGINAL native preopen. Every native
    # owner/final check stays fresh; these values admit no read, effect or scope.
    rows, prepared = native_inputs
    custody = captured.claim.custody

    def reference():
        budget.container(2)
        budget.reserve(64 + 64 + 4 * len(_HASH))
        return {"sequence": MAX_EVENTS - 1, "sha256": _HASH}

    def witness(path):
        budget.container(4)
        budget.value(path)
        budget.reserve(3 * 64)
        return {"path": path, "device": _UINT64, "inode": _UINT64, "mode": _UINT64}

    # Cover every future native Ref dump/encoding/hash before effects. The
    # sizing view borrows validated values; it does not compute a discarded hash.
    # Actual current-record hashes and all checks still run at completion.
    budget.container(len(rows) + 3)
    captures = []
    for row in rows:
        if row.event in {
            "mobile_prepared",
            "mobile_claim_observed",
            "mobile_capture_verified",
        }:
            budget.container(6)
            record_data = {
                "version": row.version,
                "operation_id": row.operation_id,
                "sequence": row.sequence,
                "previous": row.previous,
                "event": row.event,
                "evidence": row.evidence,
            }
            budget.value(record_data)  # Future native model_dump, not a clone now.
            _serialization(record_data, budget, encoded=True)
            budget.container(2)  # Future Ref dictionary.
            budget.reserve(128 + 4 * len(_HASH))  # Native digest/hex representation.
        if row.event == "mobile_capture_verified":
            budget.container(2)
            captures.append(
                {"purpose": row.evidence["purpose"], "verified": reference()}
            )
    budget.container(3)
    targets = []
    for target, version in zip(prepared.targets, (82, 9, 26), strict=True):
        budget.container(4)
        budget.container(len(target.witnesses))
        budget.container(len(target.sidecars))
        targets.append(
            {
                "owner_id": target.owner_id,
                "witnesses": [witness(row.path) for row in target.witnesses],
                "sidecars": [witness(row.path) for row in target.sidecars],
                "schema_version": version,
            }
        )
    budget.container(3)
    owners = []
    if receipts is not None:
        for family, owner_id in (
            ("chat", "db.chachanotes.primary"),
            ("workspaces", "db.workspaces"),
            ("runs", "db.agent_runs"),
        ):
            raw = next(raw for key, raw in receipts[family] if key == family)
            budget.json(raw)
            budget.value(raw, text_copies=8)
            master = json.loads(raw)
            # Future history publication changes counts/digests, not this fixed
            # native master grammar. Exercise its maximum-width decode/hash too.
            master["entry_count"] = _COUNT
            master["command_history_count"] = _COUNT if family == "runs" else None
            master["entries_sha256"] = _HASH
            _serialization(master, budget, encoded=True)
            prospective_raw = json.dumps(master, sort_keys=True, separators=(",", ":"))
            budget.json(prospective_raw)
            budget.value(prospective_raw, text_copies=8)
            json.loads(prospective_raw)
            hashlib.sha256(prospective_raw.encode("utf-8")).hexdigest()
            budget.container(6)
            budget.reserve(6 * 64 + 8 * len(_HASH))
            owners.append(
                {
                    "owner_id": owner_id,
                    "master_sha256": _HASH,
                    "entry_count": _COUNT,
                    "entries_sha256": _HASH,
                    "command_history_count": _COUNT if family == "runs" else None,
                }
            )
    budget.value(prepared.bootstrap_root, text_copies=8)
    budget.value(captured.journal.operation_id, text_copies=8)
    budget.container(10)
    root = Path(prepared.bootstrap_root)
    pending_path = root / (
        "pending-" + bootstrap._key(captured.journal.operation_id) + ".json"
    )
    pending = observation(pending_path, budget, custody, captured.cancel)
    if pending["size"] > bootstrap.MAX_RECORD:
        raise ValueError("mobile_pending_scope_changed")
    # The pending file is fixed already; reserve native scalar maximum width as
    # well, while all real pointer observations/checks remain at completion.
    pending["device"], pending["inode"], pending["size"] = (
        _UINT64,
        _UINT64,
        bootstrap.MAX_RECORD,
    )
    pending["sha256"] = _HASH
    if history_location is not None:
        budget.value(str(history_location), text_copies=8)
        budget.container(9)
        history_evidence = {
            "intent_digest": _HASH,
            "artifact": {
                "path": str(history_location),
                "device": _UINT64,
                "inode": _UINT64,
                "kind": "directory",
                "size": 1024**4,
                "sha256": _HASH,
            },
        }
        budget.value(history_evidence, text_copies=2)
        budget.container(8)
        history_event = _Event(
            operation_id=captured.journal.operation_id,
            sequence=MAX_EVENTS - 1,
            previous=_HASH,
            event="mobile_history_verified",
            evidence=history_evidence,
        )
        budget.value(history_evidence, text_copies=2)
        budget.container(6)
        _serialization(history_event.model_dump(), budget, encoded=True)
        _encoded(history_event)
    budget.container(13)
    budget.container(3)
    prospective = {
        "version": 1,
        "kind": "imported" if receipts is not None else "no_data",
        "source_fingerprint": captured.preview.source_fingerprint,
        "destination_identity": captured.preview.destination.identity,
        "prepared": reference(),
        "claim": reference(),
        "captures": captures,
        "targets": targets,
        "owners": owners,
        "history": reference() if history_location is not None else None,
        "pending": pending,
        "bootstrap": {"path": str(root), "device": _UINT64, "inode": _UINT64},
    }
    _check_terminal_dimensions(prospective, budget)
    # Model/dump copies and ASCII JSON/UTF-8 buffers are separate allocations.
    # The installed codec sizing includes twelve-character non-BMP escapes.
    budget.value(prospective, text_copies=8)
    model = _MobileCommitted.model_validate(prospective)
    # One native model dump and one conservative full evidence-validation tree.
    budget.value(prospective)
    budget.value(prospective)
    budget.container(8)  # Event model and field tracking.
    budget.container(6)  # Constructor keyword dictionary.
    budget.container(6)  # Event field dictionary, including default version.
    budget.value(captured.journal.operation_id)
    budget.value(_HASH)
    budget.value("mobile_committed")
    budget.reserve(2 * 64)  # Sequence and default version scalars.
    event = _Event(
        operation_id=captured.journal.operation_id,
        sequence=MAX_EVENTS - 1,
        previous=_HASH,
        event="mobile_committed",
        evidence=model.model_dump(),
    )
    budget.container(6)
    event_data = {
        "version": event.version,
        "operation_id": event.operation_id,
        "sequence": event.sequence,
        "previous": event.previous,
        "event": event.event,
        "evidence": event.evidence,
    }
    # Independent dumps for sizing and _encoded each allocate a complete tree.
    budget.value(event_data)
    budget.value(event_data)
    _serialization(event.model_dump(), budget, encoded=True)
    encoded = _encoded(event)
    if len(encoded) > RECOVERY_RECORD_BYTES:
        raise ValueError("journal_record_limit")
    budget.json(encoded)
    budget.value(prospective, text_copies=4)
    _Event.model_validate_json(encoded)
    # No shape, encoded record, receipt, or pre-admitted flag escapes this call.
