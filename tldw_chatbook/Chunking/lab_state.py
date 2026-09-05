"""Pure authoring transitions for the local Chunking Lab (ADR-118)."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from typing import Any

from .lab_models import (
    DraftState,
    LabSession,
    PreparedRecipe,
    RunRequest,
    RunResult,
    SampleSnapshot,
    canonical_json,
)
from .lab_preflight import current_local_runtime, prepare_recipe

_DEFAULT_BODY = {"chunking": {"method": "words"}}
_MAX_DRAFT_BYTES = 2 * 1024 * 1024
_MAX_SAMPLE_BYTES = 2 * 1024 * 1024


def _require_b(session: LabSession, candidate_id: str) -> dict:
    try:
        candidate = session.candidates[candidate_id]
    except KeyError as exc:
        raise ValueError("Unknown candidate") from exc
    if candidate.get("role") != "B" or candidate.get("editable") is not True:
        raise ValueError("Only the editable B candidate can be changed")
    return candidate


def _draft(candidate: dict) -> DraftState:
    return DraftState.model_validate(candidate["draft"])


def _candidate_undo(candidate_id: str, draft: dict) -> dict:
    return {"kind": "candidate_draft", "candidate_id": candidate_id, "draft": draft}


def _with_candidate_draft(
    session: LabSession,
    candidate_id: str,
    draft: DraftState,
    *,
    undo_draft: dict,
) -> LabSession:
    candidate = session.candidates[candidate_id]
    candidates = dict(session.candidates)
    candidates[candidate_id] = {**candidate, "draft": draft.model_dump(mode="json")}
    return session.model_copy(
        update={
            "revision": session.revision + 1,
            "candidates": candidates,
            "undo": session.undo + (_candidate_undo(candidate_id, undo_draft),),
        }
    )


def _sample(text: str, source: dict) -> SampleSnapshot:
    return SampleSnapshot(
        sample_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text=text,
        source=source,
    )


def new_session(profile_key: str) -> LabSession:
    """Create a profile-bound session with one stable editable B candidate.

    Args:
        profile_key: Stable local profile identity.

    Returns:
        A detached empty-sample session.

    Raises:
        ValueError: If the profile identity is empty.
    """
    if not isinstance(profile_key, str) or not profile_key:
        raise ValueError("Profile key must be a nonempty string")
    candidate_id = str(uuid.uuid4())
    sample = _sample("", {"kind": "paste"})
    raw = json.dumps(_DEFAULT_BODY, ensure_ascii=False)
    draft = DraftState(
        raw_json=raw,
        parsed_json=raw,
        parse_error=None,
        pending_controls={},
        authority="synced",
        record_fields={"name": "", "description": "", "tags": []},
        expected_record=None,
    )
    return LabSession(
        profile_key=profile_key,
        epoch=str(uuid.uuid4()),
        revision=0,
        candidates={
            candidate_id: {
                "candidate_id": candidate_id,
                "role": "B",
                "editable": True,
                "draft": draft.model_dump(mode="json"),
                "current_run_id": None,
                "previous_run_id": None,
            }
        },
        samples={sample.sample_hash: sample.model_dump(mode="json")},
        results={},
        batch=None,
        view={"sample_hash": sample.sample_hash},
        undo=(),
    )


def edit_json(session: LabSession, candidate_id: str, raw: str) -> LabSession:
    """Replace the JSON-owned raw draft without rolling invalid text back.

    Args:
        session: Current session snapshot.
        candidate_id: Stable editable B identity.
        raw: Exact editor text.

    Returns:
        A new session retaining the prior last-valid document on parse failure.

    Raises:
        ValueError: If controls own pending input or the draft exceeds its limit.
    """
    if not isinstance(raw, str):
        raise TypeError("Raw JSON must be text")
    if len(raw.encode("utf-8")) > _MAX_DRAFT_BYTES:
        raise ValueError("Raw draft exceeds the 2 MiB limit")
    candidate = _require_b(session, candidate_id)
    previous = _draft(candidate)
    if previous.pending_controls:
        raise ValueError("Discard or correct pending control edits before editing JSON")
    parsed_json = previous.parsed_json
    parse_error = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parse_error = {
            "message": exc.msg,
            "line": exc.lineno,
            "column": exc.colno,
        }
    else:
        parsed_json = json.dumps(parsed, ensure_ascii=False)
    draft = DraftState(
        raw_json=raw,
        parsed_json=parsed_json,
        parse_error=parse_error,
        pending_controls={},
        authority="json",
        record_fields=previous.record_fields,
        expected_record=previous.expected_record,
    )
    return _with_candidate_draft(
        session,
        candidate_id,
        draft,
        undo_draft=candidate["draft"],
    )


def _parse_string(raw: str) -> str:
    if not raw:
        raise ValueError("Value must not be empty")
    return raw


def _parse_integer(raw: str) -> int:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Expected an integer") from exc
    if type(value) is not int:
        raise ValueError("Expected an integer")
    return value


def _parse_boolean(raw: str) -> bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError("Expected true or false")


_CONTROL_PARSERS: dict[str, Callable[[str], Any]] = {
    "chunking.method": _parse_string,
    "chunking.config.max_size": _parse_integer,
    "chunking.config.overlap": _parse_integer,
    "chunking.config.language": _parse_string,
    "chunking.config.preserve_sentences": _parse_boolean,
    "chunking.config.min_chunk_size": _parse_integer,
}


def _patch_path(document: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    current = document
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise TypeError(f"{'.'.join(parts[:-1])} must be an object")
        current = child
    current[parts[-1]] = value


def edit_control(
    session: LabSession, candidate_id: str, path: str, raw: str
) -> LabSession:
    """Patch one documented flat-body path or retain its incomplete raw text.

    Args:
        session: Current session snapshot.
        candidate_id: Stable editable B identity.
        path: Documented flat-body control path.
        raw: Exact control text.

    Returns:
        A new session with either a patched base or pending raw control value.

    Raises:
        ValueError: If JSON owns an invalid draft or the path is unsupported.
    """
    if path not in _CONTROL_PARSERS:
        raise ValueError(f"Unsupported Lab control path: {path}")
    if not isinstance(raw, str):
        raise TypeError("Control input must be text")
    candidate = _require_b(session, candidate_id)
    previous = _draft(candidate)
    if previous.parse_error is not None:
        raise ValueError("Discard or correct invalid JSON before editing controls")
    if previous.parsed_json is None:
        raise ValueError("Controls require a last-valid JSON document")

    pending = dict(previous.pending_controls)
    try:
        value = _CONTROL_PARSERS[path](raw)
    except ValueError:
        pending[path] = raw
        draft = DraftState(
            raw_json=previous.raw_json,
            parsed_json=previous.parsed_json,
            parse_error=None,
            pending_controls=pending,
            authority="controls",
            record_fields=previous.record_fields,
            expected_record=previous.expected_record,
        )
    else:
        document = json.loads(previous.parsed_json)
        if not isinstance(document, dict):
            raise TypeError("Controls require a JSON object")
        _patch_path(document, path, value)
        pending.pop(path, None)
        parsed_json = json.dumps(document, ensure_ascii=False)
        draft = DraftState(
            raw_json=parsed_json,
            parsed_json=parsed_json,
            parse_error=None,
            pending_controls=pending,
            authority="controls" if pending else "synced",
            record_fields=previous.record_fields,
            expected_record=previous.expected_record,
        )
    return _with_candidate_draft(
        session,
        candidate_id,
        draft,
        undo_draft=candidate["draft"],
    )


def discard_pending_edit(session: LabSession, candidate_id: str) -> LabSession:
    """Explicitly restore the last-valid base for the owning invalid editor.

    Args:
        session: Current session snapshot.
        candidate_id: Stable editable B identity.

    Returns:
        A new synchronized session with pending invalid input removed.

    Raises:
        ValueError: If no invalid input or last-valid base exists.
    """
    candidate = _require_b(session, candidate_id)
    previous = _draft(candidate)
    if previous.parse_error is None and not previous.pending_controls:
        raise ValueError("Candidate has no invalid draft edit to discard")
    if previous.parsed_json is None:
        raise ValueError("Invalid draft has no last-valid document; use Undo")
    draft = DraftState(
        raw_json=previous.parsed_json,
        parsed_json=previous.parsed_json,
        parse_error=None,
        pending_controls={},
        authority="synced",
        record_fields=previous.record_fields,
        expected_record=previous.expected_record,
    )
    return _with_candidate_draft(
        session,
        candidate_id,
        draft,
        undo_draft=candidate["draft"],
    )


def replace_template(
    session: LabSession,
    candidate_id: str,
    body: dict | str,
    *,
    record_fields: dict,
    expected_record: dict | None,
) -> LabSession:
    """Install a detached loaded/imported body in B as one undoable transition.

    Args:
        session: Current session snapshot.
        candidate_id: Stable editable B identity.
        body: Complete flat body or exact imported raw JSON.
        record_fields: Authored name, description, and tags.
        expected_record: Loaded record ID, UUID, and version, if any.

    Returns:
        A new session detached from the caller's catalog values.
    """
    candidate = _require_b(session, candidate_id)
    if isinstance(body, dict):
        raw = json.dumps(body, ensure_ascii=False)
    elif isinstance(body, str):
        raw = body
    else:
        raise TypeError("Template body must be a JSON object or raw JSON text")
    if len(raw.encode("utf-8")) > _MAX_DRAFT_BYTES:
        raise ValueError("Raw draft exceeds the 2 MiB limit")
    parsed_json = None
    parse_error = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parse_error = {
            "message": exc.msg,
            "line": exc.lineno,
            "column": exc.colno,
        }
    else:
        parsed_json = json.dumps(parsed, ensure_ascii=False)
    draft = DraftState(
        raw_json=raw,
        parsed_json=parsed_json,
        parse_error=parse_error,
        pending_controls={},
        authority="json" if parse_error is not None else "synced",
        record_fields=record_fields,
        expected_record=expected_record,
    )
    return _with_candidate_draft(
        session,
        candidate_id,
        draft,
        undo_draft=candidate["draft"],
    )


def can_execute(session: LabSession, candidate_id: str) -> bool:
    """Return whether a candidate's current inputs can be prepared faithfully.

    Args:
        session: Current session snapshot.
        candidate_id: Candidate to validate.

    Returns:
        True only for current, fully qualified local inputs.
    """
    try:
        candidate = session.candidates[candidate_id]
        if candidate.get("role") == "A":
            recipe = PreparedRecipe.model_validate(candidate["pinned_recipe"])
            refreshed = prepare_recipe(
                json.loads(recipe.authored_json), runtime=current_local_runtime()
            )
            return refreshed == recipe
        draft = _draft(candidate)
        if draft.parse_error is not None or draft.pending_controls:
            return False
        if draft.parsed_json is None:
            return False
        prepare_recipe(json.loads(draft.parsed_json), runtime=current_local_runtime())
    except (KeyError, TypeError, ValueError):
        return False
    return True


def replace_sample(session: LabSession, text: str, source: dict) -> LabSession:
    """Copy exact sample text/source into a new immutable sample identity.

    Args:
        session: Current session snapshot.
        text: Exact sampled text.
        source: Detached descriptive source record.

    Returns:
        A new session selecting the copied sample.
    """
    if not isinstance(text, str):
        raise TypeError("Sample must be text")
    if len(text.encode("utf-8")) > _MAX_SAMPLE_BYTES:
        raise ValueError("Sample exceeds the 2 MiB limit")
    snapshot = _sample(text, source)
    samples = dict(session.samples)
    samples[snapshot.sample_hash] = snapshot.model_dump(mode="json")
    view = dict(session.view)
    previous_hash = view["sample_hash"]
    previous_sample = session.samples[previous_hash]
    view["sample_hash"] = snapshot.sample_hash
    return session.model_copy(
        update={
            "revision": session.revision + 1,
            "samples": samples,
            "view": view,
            "undo": session.undo
            + (
                {
                    "kind": "sample",
                    "sample_hash": previous_hash,
                    "sample": previous_sample,
                },
            ),
        }
    )


def update_view(session: LabSession, changes: dict) -> LabSession:
    """Publish recovery-relevant view state without consuming content undo.

    Args:
        session: Current session snapshot.
        changes: JSON-safe view fields to merge.

    Returns:
        A new session retaining the identical undo tuple.
    """
    if not isinstance(changes, dict):
        raise TypeError("View changes must be an object")
    view = dict(session.view)
    view.update(json.loads(canonical_json(changes)))
    if view.get("sample_hash") not in session.samples:
        raise ValueError("View must reference a retained sample")
    return session.model_copy(update={"revision": session.revision + 1, "view": view})


def _current_sample(session: LabSession) -> SampleSnapshot:
    return SampleSnapshot.model_validate(session.samples[session.view["sample_hash"]])


def _candidate_recipe(session: LabSession, candidate_id: str) -> PreparedRecipe:
    candidate = session.candidates[candidate_id]
    if candidate.get("role") == "A":
        return PreparedRecipe.model_validate(candidate["pinned_recipe"])
    draft = _draft(candidate)
    if (
        draft.parse_error is not None
        or draft.pending_controls
        or draft.parsed_json is None
    ):
        raise ValueError("Candidate does not have executable current inputs")
    return prepare_recipe(
        json.loads(draft.parsed_json), runtime=current_local_runtime()
    )


def _template_record(candidate: dict) -> dict | None:
    if candidate.get("role") == "A":
        record = candidate.get("template_record")
        return None if record is None else json.loads(json.dumps(record))
    draft = _draft(candidate)
    if draft.expected_record is None:
        return None
    record = {
        key: draft.expected_record[key]
        for key in ("id", "uuid", "version")
        if key in draft.expected_record
    }
    record.update(
        {
            key: draft.record_fields[key]
            for key in ("name", "description", "tags")
            if key in draft.record_fields
        }
    )
    return record


def capture_batch(
    session: LabSession, candidate_ids: tuple[str, ...]
) -> tuple[RunRequest, ...]:
    """Purely capture one immutable sample/configuration manifest.

    Args:
        session: Current session snapshot.
        candidate_ids: One or two distinct candidates to capture.

    Returns:
        Detached requests ordered A then B.

    Raises:
        ValueError: If a candidate is unknown, duplicated, or not executable.
    """
    if (
        not candidate_ids
        or len(candidate_ids) > 2
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("Capture requires one or two distinct candidates")
    try:
        candidate_ids = tuple(
            sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    0 if session.candidates[candidate_id]["role"] == "A" else 1
                ),
            )
        )
    except KeyError as exc:
        raise ValueError("Capture references an unknown candidate") from exc
    sample = _current_sample(session)
    batch_id = str(uuid.uuid4())
    requests = []
    for candidate_id in candidate_ids:
        if not can_execute(session, candidate_id):
            raise ValueError(f"Candidate {candidate_id} is not executable")
        candidate = session.candidates[candidate_id]
        requests.append(
            RunRequest(
                run_id=str(uuid.uuid4()),
                batch_id=batch_id,
                candidate_id=candidate_id,
                epoch=session.epoch,
                revision=session.revision,
                sample=sample,
                recipe=_candidate_recipe(session, candidate_id),
                template_record=_template_record(candidate),
            )
        )
    return tuple(requests)


def install_batch(session: LabSession, requests: tuple[RunRequest, ...]) -> LabSession:
    """Publish a previously pure capture as the active result manifest.

    This separate transition preserves :func:`capture_batch`'s pure return contract
    while giving :func:`accept_result` durable membership to fence against.

    Args:
        session: The unchanged session used for capture.
        requests: Requests returned by one capture call.

    Returns:
        A new session with the exact manifest installed.

    Raises:
        ValueError: If the epoch, revision, membership, or inputs changed.
    """
    if not requests:
        raise ValueError("Batch manifest cannot be empty")
    batch_id = requests[0].batch_id
    if any(
        request.batch_id != batch_id
        or request.epoch != session.epoch
        or request.revision != session.revision
        for request in requests
    ):
        raise ValueError("Batch does not match the session epoch or captured inputs")
    if len({request.run_id for request in requests}) != len(requests) or len(
        {request.candidate_id for request in requests}
    ) != len(requests):
        raise ValueError("Batch request membership must be unique")
    sample = _current_sample(session)
    for request in requests:
        if request.candidate_id not in session.candidates:
            raise ValueError("Batch references an unknown candidate")
        if (
            request.sample != sample
            or request.recipe != _candidate_recipe(session, request.candidate_id)
            or request.template_record
            != _template_record(session.candidates[request.candidate_id])
        ):
            raise ValueError("Batch no longer matches its captured inputs")
    manifest = {
        "batch_id": batch_id,
        "epoch": session.epoch,
        "captured_revision": session.revision,
        "requests": {
            request.run_id: request.model_dump(mode="json") for request in requests
        },
        "outcomes": {},
    }
    return session.model_copy(
        update={"revision": session.revision + 1, "batch": manifest}
    )


def accept_result(session: LabSession, result: RunResult) -> LabSession:
    """Accept only an exact member of the active epoch's installed batch.

    Args:
        session: Session containing the active manifest.
        result: Terminal result carrying its captured request.

    Returns:
        A new session retaining the immutable terminal result.

    Raises:
        ValueError: If the request is late across an epoch/batch fence or duplicated.
    """
    request = result.request
    if request.epoch != session.epoch:
        raise ValueError("Result belongs to a different session epoch")
    if session.batch is None or request.batch_id != session.batch.get("batch_id"):
        raise ValueError("Result batch is not active")
    member = session.batch["requests"].get(request.run_id)
    if member is None:
        raise ValueError("Result request is not in active batch membership")
    if RunRequest.model_validate(member) != request:
        raise ValueError("Result request differs from captured batch membership")
    if request.run_id in session.results:
        raise ValueError("Result was already accepted")

    results = dict(session.results)
    results[request.run_id] = result.model_dump(mode="json")
    candidate = session.candidates[request.candidate_id]
    candidates = dict(session.candidates)
    candidates[request.candidate_id] = {
        **candidate,
        "previous_run_id": candidate.get("current_run_id"),
        "current_run_id": request.run_id,
    }
    batch = dict(session.batch)
    outcomes = dict(batch.get("outcomes", {}))
    outcomes[request.run_id] = result.status
    batch["outcomes"] = outcomes
    return session.model_copy(
        update={
            "revision": session.revision + 1,
            "candidates": candidates,
            "results": results,
            "batch": batch,
        }
    )


def is_result_stale(
    session: LabSession, candidate_id: str, run_id: str | None = None
) -> bool:
    """Compare retained result inputs with this candidate's current inputs.

    Args:
        session: Current session snapshot.
        candidate_id: Candidate whose live inputs are authoritative.
        run_id: Retained result identity, or the candidate's current result.

    Returns:
        True when no result exists or sample/recipe inputs differ.
    """
    candidate = session.candidates[candidate_id]
    selected = run_id if run_id is not None else candidate.get("current_run_id")
    if selected is None or selected not in session.results:
        return True
    request = RunRequest.model_validate(session.results[selected]["request"])
    try:
        recipe = _candidate_recipe(session, candidate_id)
        sample = _current_sample(session)
    except (KeyError, TypeError, ValueError):
        return True
    return request.sample.sample_hash != sample.sample_hash or request.recipe != recipe


def pin_baseline(session: LabSession, *, replace: bool = False) -> LabSession:
    """Freeze A from B's current completed, non-stale result.

    Args:
        session: Current session snapshot.
        replace: Deliberate permission to replace an existing A.

    Returns:
        A new session with a stable frozen baseline candidate.

    Raises:
        ValueError: If B has no completed current result or replacement is implicit.
    """
    candidate_id = next(
        key for key, candidate in session.candidates.items() if candidate["role"] == "B"
    )
    candidate = session.candidates[candidate_id]
    run_id = candidate.get("current_run_id")
    if run_id is None or run_id not in session.results:
        raise ValueError("Pin requires a completed current B result")
    stored_result = session.results[run_id]
    if stored_result.get("status") != "completed" or is_result_stale(
        session, candidate_id, run_id
    ):
        raise ValueError("Pin requires a completed current B result")
    request = RunRequest.model_validate(stored_result["request"])
    existing = next(
        (
            (key, value)
            for key, value in session.candidates.items()
            if value["role"] == "A"
        ),
        None,
    )
    if existing is not None and not replace:
        raise ValueError("Baseline already exists; choose replace explicitly")
    baseline_id = existing[0] if existing is not None else str(uuid.uuid4())
    baseline = {
        "candidate_id": baseline_id,
        "role": "A",
        "editable": False,
        "pinned_recipe": request.recipe.model_dump(mode="json"),
        "template_record": request.template_record,
        "current_run_id": run_id,
        "previous_run_id": None,
    }
    candidates = dict(session.candidates)
    candidates[baseline_id] = baseline
    undo = {
        "kind": "baseline",
        "candidate_id": baseline_id,
        "candidate": None if existing is None else existing[1],
    }
    return session.model_copy(
        update={
            "revision": session.revision + 1,
            "candidates": candidates,
            "undo": session.undo + (undo,),
        }
    )


def undo_edit(session: LabSession) -> LabSession:
    """Undo the most recent content transition without rewinding revision.

    Args:
        session: Current session snapshot.

    Returns:
        A new session with the referenced draft, sample, or baseline restored.

    Raises:
        ValueError: If no content undo is available.
    """
    if not session.undo:
        raise ValueError("There is no Lab edit to undo")
    entry = session.undo[-1]
    update: dict[str, Any] = {
        "revision": session.revision + 1,
        "undo": session.undo[:-1],
    }
    if entry["kind"] == "candidate_draft":
        candidate_id = entry["candidate_id"]
        candidates = dict(session.candidates)
        candidates[candidate_id] = {
            **candidates[candidate_id],
            "draft": entry["draft"],
        }
        update["candidates"] = candidates
    elif entry["kind"] == "sample":
        samples = dict(session.samples)
        samples[entry["sample_hash"]] = entry["sample"]
        view = dict(session.view)
        view["sample_hash"] = entry["sample_hash"]
        update["samples"] = samples
        update["view"] = view
    elif entry["kind"] == "baseline":
        candidates = dict(session.candidates)
        if entry["candidate"] is None:
            candidates.pop(entry["candidate_id"], None)
            if session.batch is not None and any(
                request.get("candidate_id") == entry["candidate_id"]
                for request in session.batch.get("requests", {}).values()
            ):
                # The later process coordinator must treat this transition as a
                # cancellation request. Clearing membership here immediately
                # fences every late worker result in the pure state boundary.
                update["batch"] = None
        else:
            candidates[entry["candidate_id"]] = entry["candidate"]
        update["candidates"] = candidates
    else:
        raise ValueError("Unknown Lab undo transition")
    return session.model_copy(update=update)
