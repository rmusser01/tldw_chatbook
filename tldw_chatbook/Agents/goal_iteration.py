"""Bounded goal reports and runtime-owned, versioned verification observations.

Manifests detect observed changes, not isolation from concurrent writers. Unreadable,
unstable or over-budget input scopes fail closed. Model references are lookup keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import unicodedata
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tldw_chatbook.Agents.goal_models import (
    GoalCheck,
    GoalCheckpoint,
    GoalCriterion,
    GoalDecision,
    GoalEvidence,
    GoalSnapshot,
    IterationReport,
    VerificationSpec,
)

REPORT_BYTES = 65536
RECORD_BYTES = 131072
GOAL_BYTES = 4 * 1024 * 1024


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_iteration_report(text: str) -> IterationReport:
    """Parse one exact JSON object without coercion, repair or authority fields."""
    if type(text) is not str or len(text.encode("utf-8")) > REPORT_BYTES:
        raise ValueError("report_size")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_report_field")
            result[key] = value
        return result

    try:
        body = json.loads(text, object_pairs_hook=unique)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("malformed_report") from exc
    if type(body) is not dict or set(body) != set(IterationReport.model_fields):
        raise ValueError("report_fields")
    return IterationReport.model_validate_json(text)


def _version(info):
    # Access time may change merely because the verifier read the file.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@contextmanager
def _open_bound(root: str, relative: str):
    """Walk components from / with no symlink following, then yield a pinned fd."""
    path = Path(root) / relative
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(Path(root))
    ):
        raise ValueError("manifest_path")
    with ExitStack() as stack:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, fd)
        parts = path.parts[1:]
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(parts) - 1:
                flags |= os.O_DIRECTORY
            fd = os.open(part, flags, dir_fd=fd)
            stack.callback(os.close, fd)
        yield fd


def capture_manifest(
    root: str, spec: VerificationSpec, *, seconds: float = 2.0
) -> str | None:
    """Hash declared inputs within 256 entries/16 MiB/2 seconds and exact root."""
    deadline = time.monotonic() + min(2.0, max(0.0, seconds))
    entries = {}
    total = 0
    try:
        pending = list(spec.input_paths)
        visited = set()
        while pending:
            if time.monotonic() >= deadline or len(visited) >= 256:
                return None
            relative = str(Path(pending.pop()))
            if relative in visited:
                continue
            visited.add(relative)
            with _open_bound(root, relative) as fd:
                before = os.fstat(fd)
                if stat.S_ISDIR(before.st_mode):
                    children = []
                    with os.scandir(fd) as listing:
                        for child in listing:
                            if len(children) + len(visited) + len(pending) >= 256:
                                return None
                            children.append(str(Path(relative) / child.name))
                    pending.extend(children)
                    entries[relative] = ["directory", *_version(before)]
                elif stat.S_ISREG(before.st_mode):
                    total += before.st_size
                    if total > 16 * 1024 * 1024:
                        return None
                    hasher = hashlib.sha256()
                    read = 0
                    while data := os.read(fd, 65536):
                        read += len(data)
                        if read > before.st_size or time.monotonic() >= deadline:
                            return None
                        hasher.update(data)
                    if read != before.st_size:
                        return None
                    entries[relative] = [hasher.hexdigest(), *_version(before)]
                else:
                    return None
                if _version(before) != _version(os.fstat(fd)):
                    return None
            # Detect replacement/unlink while the descriptor was pinned.
            with _open_bound(root, relative) as current:
                if _version(before) != _version(os.fstat(current)):
                    return None
        return digest(json.dumps(entries, sort_keys=True))
    except (OSError, ValueError):
        return None


def verifier_digest(spec: VerificationSpec) -> str | None:
    try:
        with _open_bound("/", spec.verifier_path.lstrip("/")) as fd:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
                return None
            chunks = []
            size = 0
            while data := os.read(fd, 65536):
                size += len(data)
                if size > 1024 * 1024:
                    return None
                chunks.append(data)
            if _version(before) != _version(os.fstat(fd)):
                return None
            return hashlib.sha256(b"".join(chunks)).hexdigest()
    except (OSError, ValueError):
        return None


def goal_criteria(goal: GoalSnapshot) -> tuple[GoalCriterion, ...]:
    criteria = tuple(
        GoalCriterion(id=v.id, verifier_id=v.id) for v in goal.request.verifiers
    )
    if goal.request.human_review_required:
        criteria += (GoalCriterion(id="human_result_review", human=True),)
    return criteria


def refresh_evidence(goal: GoalSnapshot, item: GoalEvidence) -> GoalEvidence:
    spec = next((v for v in goal.request.verifiers if v.id == item.verifier_id), None)
    fresh = bool(
        spec
        and item.checked_manifest
        and verifier_digest(spec) == item.verifier_sha256 == spec.verifier_sha256
        and capture_manifest(goal.request.binding.locator, spec)
        == item.checked_manifest
    )
    return item.model_copy(
        update={
            "fresh": fresh,
            "reason": item.reason if fresh else "stale_or_unavailable",
        }
    )


def resolve_runtime_evidence(goal: GoalSnapshot, result) -> tuple[GoalEvidence, ...]:
    """Copy exact typed observations after the store checks native run ownership."""
    records = []
    for observation in result.tool_records[:32]:
        inv, raw = observation.invocation, observation.result
        if (inv.goal_id, inv.attempt_id, inv.ordinal, inv.run_id) != (
            goal.id,
            result.attempt_id,
            result.ordinal,
            result.native_run_id,
        ):
            continue
        spec = next(
            (
                v
                for v in goal.request.verifiers
                if (v.verifier_path, v.verifier_sha256, v.arguments, v.skill_trust_ref)
                == (
                    inv.verifier_path,
                    inv.verifier_sha256,
                    inv.arguments,
                    inv.skill_trust_ref,
                )
                and v.executor_tool_id
                in ("run_skill_script", "runtime:run_skill_script")
            ),
            None,
        )
        if spec is None:
            continue
        stable = bool(
            inv.before_manifest and inv.before_manifest == observation.after_manifest
        )
        complete = not (
            raw.output_capped or raw.truncated_stdout or raw.truncated_stderr
        )
        passed = bool(
            stable
            and raw.exit_code == spec.expected_exit_code
            and not raw.timed_out
            and (complete or not spec.require_complete_output)
        )
        # Copy bounded output, never follow output_dir or model paths.
        stdout, stderr = raw.stdout, raw.stderr
        too_large = len((stdout + stderr).encode("utf-8")) > 96 * 1024
        if too_large:
            stdout = stdout.encode("utf-8")[: 48 * 1024].decode("utf-8", "ignore")
            stderr = stderr.encode("utf-8")[: 48 * 1024].decode("utf-8", "ignore")
            passed = False
        source = digest(
            json.dumps(
                [
                    spec.id,
                    inv.verifier_sha256,
                    inv.arguments,
                    inv.before_manifest,
                    raw.exit_code,
                    raw.timed_out,
                    complete,
                ]
            )
        )
        item = GoalEvidence(
            id=inv.id,
            goal_id=goal.id,
            attempt_id=inv.attempt_id,
            run_id=inv.run_id,
            verifier_id=spec.id,
            source_digest=source,
            checked_manifest=inv.before_manifest if stable else None,
            verifier_sha256=inv.verifier_sha256,
            passed=passed,
            fresh=False,
            reason="passed" if passed else "verification_unavailable_or_failed",
            stdout=stdout,
            stderr=stderr,
        )
        if len(item.model_dump_json().encode()) > RECORD_BYTES:
            item = item.model_copy(
                update={
                    "stdout": "",
                    "stderr": "",
                    "passed": False,
                    "reason": "evidence_size",
                }
            )
        records.append(refresh_evidence(goal, item))
    for observation in result.observations[: max(0, 32 - len(records))]:
        if (observation.goal_id, observation.attempt_id, observation.run_id) != (
            goal.id,
            result.attempt_id,
            result.native_run_id,
        ):
            continue
        content = observation.content.encode("utf-8")[: 48 * 1024].decode(
            "utf-8", "ignore"
        )
        item = GoalEvidence(
            id=observation.id,
            goal_id=goal.id,
            attempt_id=result.attempt_id,
            run_id=result.native_run_id,
            verifier_id="",
            source_digest=digest(
                json.dumps([observation.tool, observation.arguments_digest, content])
            ),
            checked_manifest=None,
            verifier_sha256="",
            passed=False,
            fresh=False,
            reason="observation_only"
            if observation.complete
            else "incomplete_observation",
            stdout=content,
        )
        if len(item.model_dump_json().encode()) > RECORD_BYTES:
            item = item.model_copy(
                update={"stdout": "", "reason": "incomplete_observation"}
            )
        records.append(item)
    return tuple(records)


def evaluate_iteration(
    report: IterationReport,
    evidence: Sequence[GoalEvidence],
    previous: GoalCheckpoint | None,
    criteria: Sequence[GoalCriterion],
) -> GoalDecision:
    owned = {e.id: e for e in evidence}
    errors = tuple(dict.fromkeys(i for i in report.evidence_ids if i not in owned))
    checks = []
    for criterion in criteria:
        # The native goal loop executes tools sequentially. Runtime observation
        # order, then durable insertion order, is authoritative: a report cannot
        # cherry-pick an older pass over a later failed or unavailable check.
        latest = next(
            (e for e in reversed(evidence) if e.verifier_id == criterion.verifier_id),
            None,
        )
        matched = (
            latest
            if latest
            and latest.id in report.evidence_ids
            and latest.passed
            and latest.fresh
            else None
        )
        checks.append(
            GoalCheck(
                criterion_id=criterion.id,
                satisfied=bool(matched) and not criterion.human,
                evidence_id=matched.id if matched else None,
                reason="human_review_required"
                if criterion.human
                else "passed"
                if matched
                else "unavailable",
            )
        )
    prior = previous.decision if previous else None
    # Store one digest of the complete resolved source set, keeping checkpoint
    # metadata bounded even when a goal has many small retained observations.
    sources = (
        (digest(json.dumps(sorted({e.source_digest for e in evidence}))),)
        if evidence
        else (prior.source_digests if prior else ())
    )
    normalized_draft = unicodedata.normalize(
        "NFC", report.candidate_draft.replace("\r\n", "\n")
    ).strip()
    draft = digest(normalized_draft)
    prior_checks = (
        {c.criterion_id for c in prior.checks if c.satisfied} if prior else set()
    )
    progress = bool(
        set(sources) - set(prior.source_digests if prior else ())
        or (report.candidate_draft and draft != (prior.draft_digest if prior else ""))
        or {c.criterion_id for c in checks if c.satisfied} - prior_checks
    )
    no_progress = 0 if progress else (prior.no_progress_count if prior else 0) + 1
    action, reason = "continue", "observed_progress" if progress else "no_progress"
    objective_checks = [
        c for c, criterion in zip(checks, criteria) if not criterion.human
    ]
    if not errors and objective_checks and all(c.satisfied for c in objective_checks):
        action = (
            "awaiting_result_review" if any(c.human for c in criteria) else "completed"
        )
        reason = (
            "human_review_required"
            if action == "awaiting_result_review"
            else "verified_checks"
        )
    elif (
        not errors
        and not objective_checks
        and any(c.human for c in criteria)
        and report.completion_recommended
        and any(i in owned for i in report.evidence_ids)
    ):
        action, reason = "awaiting_result_review", "human_review_required"
    elif no_progress >= 2:
        action, reason = "pause", "no_progress"
    return GoalDecision(
        action=action,
        reason=reason,
        checks=tuple(checks),
        evidence_errors=errors,
        no_progress_count=no_progress,
        source_digests=sources,
        draft_digest=draft,
    )


def build_goal_handoff(
    goal: GoalSnapshot, checkpoints: Sequence[GoalCheckpoint]
) -> str:
    """Keep immutable mandatory objective/criteria plus at most 16 KiB of memory."""
    if goal.request is None:
        raise ValueError("goal_payload_removed")
    memory = {}
    if checkpoints:
        latest = checkpoints[-1]
        memory = {
            "ordinal": latest.ordinal,
            "summary": latest.report.summary,
            "next_action": latest.report.next_action,
            "candidate_draft": latest.report.candidate_draft,
            "checks": [c.model_dump() for c in latest.decision.checks],
            "evidence_ids": list(latest.report.evidence_ids),
        }
        # Deterministic newest-first learnings, omitted wholesale if too large.
        memory["learnings"] = list(
            dict.fromkeys(
                l for c in reversed(checkpoints[-3:]) for l in c.report.learnings
            )
        )[:8]
        for field in ("learnings", "candidate_draft", "summary", "next_action"):
            if len(json.dumps(memory, ensure_ascii=False).encode()) <= 16384:
                break
            memory.pop(field, None)
    payload = json.dumps(memory, ensure_ascii=False)
    if len(payload.encode()) > 16384:
        raise ValueError("goal_memory_capacity")
    protocol = "\n\nReturn exactly one JSON object with summary (string), learnings (array of up to 8 strings), next_action (string), candidate_draft (string), evidence_ids (array of up to 32 runtime-issued goal_evidence_id lookup keys), completion_recommended (boolean). No extra fields or fences. Report <=64 KiB UTF-8; draft <=32 KiB. Report recommendations do not establish completion.\n"
    return (
        protocol
        + "Goal objective:\n"
        + goal.request.objective
        + "\n\nCompletion criteria:\n"
        + goal.request.criteria
        + "\n\nPrivate checkpoint memory (advisory):\n"
        + payload
    )
