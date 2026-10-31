"""Blinded substantive claims and attempt-local verification state (ADR-281)."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .code_review_models import FindingRecord, ReviewRefused, canonical_record_bytes
from .code_review_source import SourceManifest


@dataclass(frozen=True, repr=False)
class VerificationUnitSet:
    original_to_unit: dict[str, str]
    public_units: tuple[dict, ...]


@dataclass(repr=False)
class VerificationLedger:
    units: VerificationUnitSet
    dispositions: dict[str, dict] = field(default_factory=dict)
    delivered: set[str] = field(default_factory=set)
    pages: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    cursor: str | None = None
    acknowledged_cursors: list[str] = field(default_factory=list)
    request_gaps: list[str] = field(default_factory=list)


def project_claim(finding: FindingRecord, manifest: SourceManifest) -> dict:
    """Select substantive source claims; omit votes, titles and provenance."""
    evidence = [item.model_dump(mode="json") for item in finding.evidence]
    if any(item["kind"] != "source" for item in evidence):
        raise ReviewRefused("reviewer_check_evidence_unavailable")
    return {
        "target_digest": manifest.target_digest,
        "locations": [item.model_dump(mode="json") for item in finding.locations],
        "trigger": finding.trigger,
        "impact": finding.impact,
        "evidence": evidence,
        "uncertainty": finding.uncertainty,
    }


def build_verification_units(
    originals: tuple[dict, ...], manifest: SourceManifest
) -> VerificationUnitSet:
    """Coalesce byte-identical projections and assign unrelated opaque IDs."""
    claims, mapping = {}, {}
    for original in originals:
        claim = project_claim(
            FindingRecord.model_validate(original["finding"]), manifest
        )
        key = canonical_record_bytes(claim)
        if key not in claims:
            claims[key] = {"verification_unit_id": uuid4().hex, "claim": claim}
        original_id = original["original_id"]
        if original_id in mapping:
            raise ReviewRefused("duplicate_original_id")
        mapping[original_id] = claims[key]["verification_unit_id"]
    return VerificationUnitSet(
        mapping,
        tuple(sorted(claims.values(), key=lambda row: row["verification_unit_id"])),
    )


def unresolved(unit_id: str, reason: str) -> dict:
    return {
        "schema_version": 1,
        "verification_unit_id": unit_id,
        "verdict": "unresolved",
        "reason": reason,
        "evidence": [],
        "duplicate_unit_id": None,
        "next_check": "Start an explicitly authorized verification attempt with sufficient evidence and capacity.",
    }
