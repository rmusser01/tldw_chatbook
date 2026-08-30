"""Deterministic structured-key review diffs for interview output."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

from tldw_profile_core import (
    AgentVisibility,
    InterviewProposalBatch,
    InterviewProposedChange,
    ProfileRecord,
    ProposalOperation,
    RecordState,
    canonical_bytes,
)


@dataclass(frozen=True, slots=True)
class InterviewDiffChange:
    change_id: str
    change: InterviewProposedChange = field(repr=False)
    possible_private_duplicate: bool = False


@dataclass(frozen=True, slots=True)
class InterviewDiff:
    pack_id: str
    audience: str
    changes: tuple[InterviewDiffChange, ...] = field(default=(), repr=False)

    @property
    def additions(self) -> tuple[InterviewDiffChange, ...]:
        return tuple(
            item
            for item in self.changes
            if item.change.operation is ProposalOperation.CREATE
        )

    @property
    def updates(self) -> tuple[InterviewDiffChange, ...]:
        return tuple(
            item
            for item in self.changes
            if item.change.operation is ProposalOperation.UPDATE
        )


def _identity(change: InterviewProposedChange) -> tuple[str, str, str] | None:
    if change.proposed_payload is None or change.semantic_key is None:
        return None
    return (
        change.proposed_payload.kind,
        change.semantic_key.namespace,
        change.semantic_key.subject,
    )


def _record_identity(record: ProfileRecord) -> tuple[str, str, str] | None:
    if record.semantic_key is None:
        return None
    return (
        record.kind.value,
        record.semantic_key.namespace,
        record.semantic_key.subject,
    )


def build_interview_diff(
    batch: InterviewProposalBatch,
    existing_records: tuple[ProfileRecord, ...],
    *,
    now: datetime,
) -> InterviewDiff:
    """Return a content-safe deterministic diff without semantic text merging."""

    visible = {
        identity: record
        for record in existing_records
        if record.state is RecordState.ACTIVE
        if record.expires_at is None or record.expires_at > now
        if record.controls.agent_visibility is AgentVisibility.AGENT_VISIBLE
        if (identity := _record_identity(record)) is not None
    }
    private_keys = {
        identity
        for record in existing_records
        if record.state is RecordState.ACTIVE
        if record.expires_at is None or record.expires_at > now
        if record.controls.agent_visibility is AgentVisibility.USER_ONLY
        if (identity := _record_identity(record)) is not None
    }
    last_by_identity: dict[tuple[str, ...], tuple[int, InterviewProposedChange]] = {}
    for index, change in enumerate(batch.changes):
        identity = _identity(change)
        selection_identity = (
            ("semantic", *identity)
            if identity is not None
            else (
                "target",
                change.operation.value,
                change.target_record_id or "",
                change.base_version_id or "",
            )
        )
        last_by_identity[selection_identity] = (index, change)

    normalized: list[tuple[bytes, InterviewProposedChange, bool]] = []
    for _index, change in sorted(last_by_identity.values()):
        identity = _identity(change)
        normalized_change = change
        if change.operation is ProposalOperation.CREATE and identity in visible:
            existing = visible[identity]
            normalized_change = InterviewProposedChange(
                operation=ProposalOperation.UPDATE,
                target_record_id=existing.record_id,
                base_version_id=existing.version_id,
                proposed_payload=change.proposed_payload,
                controls=change.controls,
                semantic_key=change.semantic_key,
            )
        encoded = canonical_bytes(normalized_change)
        normalized.append((encoded, normalized_change, identity in private_keys))
    normalized.sort(key=lambda item: item[0])
    return InterviewDiff(
        pack_id=batch.pack_id,
        audience=batch.audience.value,
        changes=tuple(
            InterviewDiffChange(
                change_id="change-" + hashlib.sha256(encoded).hexdigest()[:20],
                change=change,
                possible_private_duplicate=private_duplicate,
            )
            for encoded, change, private_duplicate in normalized
        ),
    )
