"""Bounded, non-executing recovery transfer and active-state admission.

Call serialization/validation off the UI loop. Limits use canonical UTF-8 JSON:
samples count exact text bytes, results count the complete RunResult, and the
8 MiB checkpoint omits sample/result payloads and captured batch requests.
The envelope includes those payloads, captured requests, and integrity digests.
"""

from __future__ import annotations

import hashlib
import json
import math

from .lab_models import LabSession, RunRequest, RunResult, canonical_json, validate_view

MAX_ENVELOPE_BYTES = 256 * 1024 * 1024
MAX_DRAFT_BYTES = 2 * 1024 * 1024
MAX_SAMPLE_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024
MAX_BLOBS = 16
MAX_DEPTH = 64
MAX_CHUNKS = 10_000


class RecoveryImportError(ValueError):
    """A transfer cannot be interpreted safely; leave the active state intact."""


def _depth(value, level=0):
    if level + int(isinstance(value, (dict, tuple, list))) > MAX_DEPTH:
        raise ValueError("Recovery JSON exceeds depth 64")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Recovery keys must be strings")
        return max([level + 1, *(_depth(child, level + 1) for child in value.values())])
    if isinstance(value, (tuple, list)):
        return max([level + 1, *(_depth(child, level + 1) for child in value)])
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Recovery numbers must be finite")
    return level


def _json(text):
    # Bound nesting before the JSON parser can recurse; raw draft strings are opaque.
    level, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            level += 1
            if level > MAX_DEPTH:
                raise ValueError("Recovery JSON exceeds depth 64")
        elif char in "]}":
            level -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Recovery numbers must be finite")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def prune_session(session: LabSession, *, include_undo: bool = True) -> LabSession:
    """Keep shallow immutable references reachable from inspectable state."""
    candidates = list(session.candidates.values())
    undo = session.undo if include_undo else ()
    sample_ids = {session.view["sample_hash"]}
    for entry in undo:
        if entry.get("kind") == "baseline" and entry.get("candidate"):
            candidates.append(entry["candidate"])
        elif entry.get("kind") == "sample":
            sample_ids.add(entry["sample_hash"])
    run_ids = {
        candidate[field]
        for candidate in candidates
        for field in ("current_run_id", "previous_run_id")
        if candidate.get(field) is not None
    }
    if session.batch:
        run_ids.update(session.batch.get("outcomes", {}))
        sample_ids.update(
            request["sample"]["sample_hash"]
            for request in session.batch["requests"].values()
        )
    results = {key: value for key, value in session.results.items() if key in run_ids}
    sample_ids.update(
        result["request"]["sample"]["sample_hash"] for result in results.values()
    )
    samples = {
        key: value for key, value in session.samples.items() if key in sample_ids
    }
    return session.model_copy(
        update={
            "samples": session.samples
            if len(samples) == len(session.samples)
            else samples,
            "results": session.results
            if len(results) == len(session.results)
            else results,
            "undo": undo,
        }
    )


def _document(session):
    return {
        name: getattr(session, name)
        for name in LabSession.model_fields
        if name != "undo"
    }


def _recipe(recipe):
    identity = {
        "authored": _json(recipe["authored_json"]),
        "effective": _json(recipe["effective_json"]),
        "runtime": recipe["runtime"],
    }
    _depth(identity)
    if (
        hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        != recipe["recipe_hash"]
    ):
        raise ValueError("Captured recipe digest mismatch")


def _measure(value, kind):
    depth = _depth(value)
    encoded = canonical_json(value).encode("utf-8")
    samples = []
    if kind == "sample":
        samples.append(value)
    else:
        request = value["request"] if kind == "result" else value
        if type(request["revision"]) is not int or request["revision"] < 0:
            raise ValueError("Captured revision must be a nonnegative integer")
        if any(
            not isinstance(request[field], str) or not request[field]
            for field in ("run_id", "batch_id", "candidate_id", "epoch")
        ):
            raise ValueError("Captured request identities must be nonempty strings")
        samples.append(request["sample"])
        _recipe(request["recipe"])
    for sample in samples:
        if len(sample["text"].encode("utf-8")) > MAX_SAMPLE_BYTES:
            raise ValueError("Sample exceeds the 2 MiB limit")
        if (
            hashlib.sha256(sample["text"].encode("utf-8")).hexdigest()
            != sample["sample_hash"]
        ):
            raise ValueError("Sample digest mismatch")
    if kind == "result":
        if type(value["elapsed_ms"]) not in (int, float):
            raise ValueError("Elapsed time must be numeric")
        if len(encoded) > MAX_RESULT_BYTES:
            raise ValueError("Result exceeds the 32 MiB limit")
        if value["report"] and len(value["report"]["chunks"]) > MAX_CHUNKS:
            raise ValueError("Result exceeds the 10,000 chunk limit")
    return (
        value,
        len(encoded),
        depth,
        hashlib.sha256(encoded).hexdigest(),
        tuple(sample["sample_hash"] for sample in samples),
    )


def validate_active(session: LabSession, *, reuse: bool = False) -> dict:
    """Check matching export limits, reusing only local immutable identity measures.

    The non-field cache is neither serialized nor involved in model equality.
    It contains only this session's reachable blob objects, never older sessions.
    Untrusted ingress/export always rebuilds it rather than trusting cached sizes.
    """
    validate_view(session.view)
    active = prune_session(session, include_undo=False)
    document = _document(active)
    small = {
        **document,
        "samples": {key: None for key in active.samples},
        "results": {key: None for key in active.results},
    }
    if active.batch:
        small["batch"] = {
            **active.batch,
            "requests": {key: None for key in active.batch["requests"]},
        }
    for candidate in active.candidates.values():
        if (
            candidate.get("draft")
            and len(candidate["draft"]["raw_json"].encode("utf-8")) > MAX_DRAFT_BYTES
        ):
            raise ValueError("Raw draft exceeds the 2 MiB limit")
        if candidate.get("pinned_recipe"):
            _recipe(candidate["pinned_recipe"])
    _depth({"session": small})
    if len(canonical_json(small).encode("utf-8")) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Recovery checkpoint exceeds the 8 MiB limit")
    prior = getattr(session, "_recovery_measurements", {}) if reuse else {}
    measured, sample_ids, digests = (
        {},
        set(),
        {"samples": {}, "results": {}, "requests": {}},
    )
    additional_bytes = 0
    groups = [
        ("samples", "sample", active.samples),
        ("results", "result", active.results),
        ("requests", "request", active.batch["requests"] if active.batch else {}),
    ]
    for group, kind, values in groups:
        for key, value in values.items():
            cached = prior.get((kind, key))
            item = (
                cached
                if cached is not None and cached[0] is value
                else _measure(value, kind)
            )
            measured[kind, key] = item
            if item[2] + (4 if kind == "request" else 3) > MAX_DEPTH:
                raise ValueError("Recovery JSON exceeds depth 64")
            additional_bytes += item[1] - 4  # Replace the small graph's null.
            sample_ids.update(item[4])
            digests[group][key] = item[3]
    if len(sample_ids) + len(active.results) > MAX_BLOBS:
        raise ValueError("Recovery exceeds 16 referenced sample/result blobs")
    envelope = {
        "format": "chunking-lab-recovery",
        "version": 1,
        "session": small,
        "digests": digests,
    }
    if (
        len(canonical_json(envelope).encode("utf-8")) + additional_bytes
        > MAX_ENVELOPE_BYTES
    ):
        raise ValueError("Recovery envelope exceeds the 256 MiB limit")
    object.__setattr__(session, "_recovery_measurements", measured)
    return digests


def _membership(session):
    for candidate in session.candidates.values():
        for field in ("current_run_id", "previous_run_id"):
            run_id = candidate.get(field)
            if run_id is None:
                continue
            request = session.results[run_id]["request"]
            if request["candidate_id"] != candidate["candidate_id"]:
                origin = session.candidates.get(request["candidate_id"])
                if not (
                    candidate["role"] == "A"
                    and origin
                    and origin["role"] == "B"
                    and candidate["pinned_recipe"] == request["recipe"]
                ):
                    raise ValueError("Illegal result candidate membership")
    if session.batch:
        requests = session.batch["requests"]
        if len(requests) > 2 or len(
            {request["candidate_id"] for request in requests.values()}
        ) != len(requests):
            raise ValueError("Illegal batch candidate membership")
        for run_id, outcome in session.batch.get("outcomes", {}).items():
            result = session.results.get(run_id)
            if (
                run_id not in requests
                or result is None
                or result["request"] != requests[run_id]
                or result["status"] != outcome
            ):
                raise ValueError("Invalid batch outcome reference")


def export_recovery(session: LabSession) -> bytes:
    """Export current referenced content without database access or undo history."""
    active = prune_session(session, include_undo=False)
    digests = validate_active(active)
    active = LabSession.model_validate({**_document(active), "undo": []})
    _membership(active)
    return canonical_json(
        {
            "format": "chunking-lab-recovery",
            "version": 1,
            "session": _document(active),
            "digests": digests,
        }
    ).encode("utf-8")


def interrupt_unfinished(session: LabSession) -> LabSession:
    """Terminalize unfinished recovery members, retaining manifest and authority."""
    if session.batch:
        from .lab_state import accept_result

        for run_id, request in session.batch["requests"].items():
            if run_id not in session.batch.get("outcomes", {}):
                session = accept_result(
                    session,
                    RunResult(
                        request=RunRequest.model_validate(request),
                        status="interrupted",
                        report=None,
                        started_at="",
                        finished_at="",
                        elapsed_ms=0,
                        error={"message": "Preview interrupted before recovery"},
                    ),
                )
    return session


def rebase_recovery(session: LabSession, profile_key: str, epoch: str) -> LabSession:
    """Retire work before replacing authority; never rewrite captured run provenance."""
    session = interrupt_unfinished(session)
    return prune_session(
        session.model_copy(
            update={"profile_key": profile_key, "epoch": epoch, "batch": None}
        )
    )


def parse_recovery(payload: bytes) -> LabSession:
    """Inspect a bounded envelope without executing recipes or reading source paths."""
    try:
        if not isinstance(payload, bytes) or len(payload) > MAX_ENVELOPE_BYTES:
            raise ValueError("Recovery envelope exceeds the 256 MiB limit")
        envelope = _json(payload.decode("utf-8"))
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"format", "version", "session", "digests"}
            or envelope["format"] != "chunking-lab-recovery"
            or type(envelope["version"]) is not int
            or envelope["version"] != 1
        ):
            raise ValueError("Unsupported recovery envelope")
        document = envelope["session"]
        if not isinstance(document, dict) or set(document) - {
            "content_revision"
        } != set(LabSession.model_fields) - {"undo", "content_revision"}:
            raise ValueError("Invalid recovery session")
        if (
            type(document["revision"]) is not int
            or type(document.get("content_revision", 0)) is not int
            or not isinstance(document["profile_key"], str)
            or not isinstance(document["epoch"], str)
        ):
            raise ValueError("Invalid session identity")
        # Check byte/count limits before Pydantic reconstructs nested blobs.
        raw = LabSession.model_construct(**document, undo=())
        if not isinstance(raw.candidates, dict) or not 1 <= len(raw.candidates) <= 2:
            raise ValueError("Lab v1 supports at most two candidates")
        active = prune_session(raw, include_undo=False)
        if (
            active.samples.keys() != raw.samples.keys()
            or active.results.keys() != raw.results.keys()
        ):
            raise ValueError("Recovery contains unreferenced blobs")
        if validate_active(raw) != envelope["digests"]:
            raise ValueError("Recovery reference digest mismatch")
        session = LabSession.model_validate({**document, "undo": []})
        _membership(session)
        session = interrupt_unfinished(session)
        validate_active(session)
        return session
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        OverflowError,
    ) as exc:
        raise RecoveryImportError("Invalid or unsupported recovery snapshot") from exc
