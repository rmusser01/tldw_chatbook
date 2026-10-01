"""Disposable Settings metadata, never an evidence authority or export format."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Literal

from tldw_profile_core import ProfileProposal, ProfileRecord, canonical_bytes


@dataclass(frozen=True, slots=True)
class SettingsProfileIdentity:
    profile_id: str = field(repr=False)
    purge_generation: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class SettingsProvenanceSubject:
    profile: SettingsProfileIdentity = field(repr=False)
    object_type: Literal["record", "proposal"]
    object_id: str = field(repr=False)
    version_token: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SettingsDeletedRecord:
    subject: SettingsProvenanceSubject = field(repr=False)
    scope_id: str = field(repr=False)
    updated_at: datetime = field(repr=False)


@dataclass(frozen=True, slots=True)
class SettingsProvenanceField:
    label: str
    value: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SettingsProvenanceProjection:
    subject: SettingsProvenanceSubject = field(repr=False)
    fields: tuple[SettingsProvenanceField, ...] = field(repr=False)
    reference_status: str
    source_references: tuple[str, ...] = field(repr=False)
    source_hashes: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SettingsProvenanceResult:
    state: Literal["available", "changed", "unavailable"]
    projection: SettingsProvenanceProjection | None = field(default=None, repr=False)


def provenance_subject(
    profile: SettingsProfileIdentity, value: ProfileRecord | ProfileProposal
) -> SettingsProvenanceSubject:
    """Capture the current object identity without inventing a proposal revision."""
    if isinstance(value, ProfileRecord):
        return SettingsProvenanceSubject(
            profile, "record", value.record_id, value.version_id
        )
    return SettingsProvenanceSubject(
        profile,
        "proposal",
        value.proposal_id,
        sha256(canonical_bytes(value)).hexdigest(),
    )


def project_provenance(
    subject: SettingsProvenanceSubject, value: ProfileRecord | ProfileProposal
) -> SettingsProvenanceProjection:
    """Describe retained fields only; callers own availability and version checks."""
    provenance = value.provenance
    pairs = [
        ("Recorded source", provenance.source.value),
        ("Recorded actor", provenance.actor.value),
        ("Recorded reason", provenance.reason_code),
        ("Created", value.created_at.isoformat()),
    ]
    if isinstance(value, ProfileRecord):
        pairs.extend(
            [
                ("Updated", value.updated_at.isoformat()),
                ("State", value.state.value),
                ("Record ID", value.record_id),
                ("Current version ID", value.version_id),
                ("Parent version ID", value.parent_version_id or "Not recorded"),
            ]
        )
    else:
        pairs.extend(
            [
                ("Expires", value.expires_at.isoformat()),
                ("State", value.state.value),
                ("Proposal ID", value.proposal_id),
                ("Operation", value.operation.value),
                ("Target record ID", value.target_record_id or "Not recorded"),
                ("Target base version ID", value.base_version_id or "Not recorded"),
                (
                    "Proposal revision",
                    "Proposal revision not retained as a canonical field",
                ),
            ]
        )
    pairs.extend(
        [
            (
                "Promotion origin record ID",
                provenance.derived_from_record_id or "Not recorded",
            ),
            ("Edit history", "Edit history not recorded"),
            ("Inference classification", "Inference classification not recorded"),
        ]
    )
    return SettingsProvenanceProjection(
        subject=subject,
        fields=tuple(SettingsProvenanceField(label, text) for label, text in pairs),
        reference_status=(
            "Legacy source reference — quotation not verified"
            if provenance.source_references
            else "No source reference retained"
        ),
        source_references=provenance.source_references,
        source_hashes=provenance.source_hashes,
    )
