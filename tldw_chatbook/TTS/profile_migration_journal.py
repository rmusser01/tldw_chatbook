"""Bounded exact codec for profile-migration publication journals."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Final

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


class ProfileMigrationPublicationSlot(StrEnum):
    """Recognized authority slots in one migration publication."""

    ACTIVE = "active"
    PRE_V3 = "pre_v3"
    PRE_V4 = "pre_v4"


class ProfileMigrationPublicationStage(StrEnum):
    """Bounded deterministic checkpoints for cancellation and fault injection."""

    PREFLIGHT = "preflight"
    JOURNAL_DURABLE = "journal_durable"
    PONR = "ponr"
    ACTIVE_RETAINED = "active_retained"
    ACTIVE_REPLACED = "active_replaced"
    ACTIVE_FSYNCED = "active_fsynced"
    ACTIVE_REOPENED = "active_reopened"
    BACKUP_RETAINED = "backup_retained"
    BACKUP_REPLACED = "backup_replaced"
    BACKUP_FSYNCED = "backup_fsynced"
    BACKUP_REOPENED = "backup_reopened"
    FINAL_JOURNAL_DURABLE = "final_journal_durable"


@dataclass(frozen=True, slots=True, repr=False)
class ProfileMigrationJournalSlot:
    """One recognized relative namespace row in a journal."""

    slot: ProfileMigrationPublicationSlot
    candidate: str
    target: str
    rollback: str
    had_prior: bool
    _candidate_evidence: _ArtifactEvidence | None = field(
        default=None,
        compare=False,
    )
    _prior_evidence: _ArtifactEvidence | None = field(
        default=None,
        compare=False,
    )

    def __repr__(self) -> str:
        return "ProfileMigrationJournalSlot(<private>)"

    def matches_candidate(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> bool:
        """Classify one verified object against the prepared authority."""

        return (
            self._candidate_evidence is not None
            and self._candidate_evidence.matches(
                identity,
                byte_length=byte_length,
                sha256_digest=sha256_digest,
                schema_version=schema_version,
            )
        )

    def matches_prior(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> bool:
        """Classify one verified object against retained prior authority."""

        return self._prior_evidence is not None and self._prior_evidence.matches(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
        )

    def classify_artifact(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> str | None:
        """Return a bounded authority label for one verified namespace object."""

        if self.matches_candidate(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
        ):
            return "candidate"
        if self.matches_prior(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
        ):
            return "prior"
        return None


@dataclass(frozen=True, slots=True, repr=False)
class _ArtifactEvidence:
    dev: int
    ino: int
    byte_length: int
    sha256_digest: bytes
    schema_version: int

    def __repr__(self) -> str:
        return "_ArtifactEvidence(<private>)"

    def matches(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> bool:
        return (
            identity.st_dev == self.dev
            and identity.st_ino == self.ino
            and byte_length == self.byte_length
            and sha256_digest == self.sha256_digest
            and schema_version == self.schema_version
        )


@dataclass(frozen=True, slots=True, repr=False)
class _JournalAuthority:
    parent_dev: int
    parent_ino: int
    rows: tuple[ProfileMigrationJournalSlot, ...]

    def __repr__(self) -> str:
        return "_JournalAuthority(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class ParsedProfileMigrationJournal:
    """Path-free facts from one exact recognized publication journal."""

    version: int
    phase: str
    recovery_rows: tuple[ProfileMigrationJournalSlot, ...]
    _parent_dev: int
    _parent_ino: int
    _authority_checksum: bytes

    @property
    def slots(self) -> tuple[str, ...]:
        """Return only recognized slot labels for compatibility checks."""

        return tuple(row.slot.value for row in self.recovery_rows)

    def __repr__(self) -> str:
        return (
            "ParsedProfileMigrationJournal("
            f"version={self.version!r}, phase={self.phase!r}, recovery_rows=<private>)"
        )

    def matches_parent(self, identity: os.stat_result) -> bool:
        """Classify an already verified directory against journal authority."""

        return (
            identity.st_dev == self._parent_dev and identity.st_ino == self._parent_ino
        )


MAX_PROFILE_MIGRATION_JOURNAL_BYTES: Final = 4096
_PHASES: Final = frozenset(
    {"prepared", "publishing", "restoring", "complete", "unavailable"}
)
PROFILE_MIGRATION_SLOT_SEQUENCES: Final = frozenset(
    {
        (ProfileMigrationPublicationSlot.ACTIVE,),
        (
            ProfileMigrationPublicationSlot.ACTIVE,
            ProfileMigrationPublicationSlot.PRE_V4,
        ),
        (
            ProfileMigrationPublicationSlot.ACTIVE,
            ProfileMigrationPublicationSlot.PRE_V3,
            ProfileMigrationPublicationSlot.PRE_V4,
        ),
    }
)
_CANDIDATE_SCHEMA_BY_SLOT: Final = {
    ProfileMigrationPublicationSlot.ACTIVE: 4,
    ProfileMigrationPublicationSlot.PRE_V3: 2,
    ProfileMigrationPublicationSlot.PRE_V4: 3,
}


def _validate_recovery_rows(
    rows: tuple[ProfileMigrationJournalSlot, ...],
) -> None:
    if (
        type(rows) is not tuple
        or tuple(row.slot for row in rows) not in PROFILE_MIGRATION_SLOT_SEQUENCES
    ):
        raise ValueError
    leaves: set[str] = set()
    for row in rows:
        if (
            type(row) is not ProfileMigrationJournalSlot
            or type(row.had_prior) is not bool
        ):
            raise ValueError
        for value in (row.candidate, row.target, row.rollback):
            if (
                type(value) is not str
                or not value
                or len(value.encode("utf-8")) > 255
                or Path(value).name != value
                or value in {".", ".."}
                or "\x00" in value
            ):
                raise ValueError
            if value in leaves:
                raise ValueError
            leaves.add(value)
        if row.rollback != f".{row.target}.{row.slot.value}.rollback":
            raise ValueError


def _validate_authority_evidence(
    rows: tuple[ProfileMigrationJournalSlot, ...],
) -> None:
    for row in rows:
        expected_schema = _CANDIDATE_SCHEMA_BY_SLOT[row.slot]
        if (
            row._candidate_evidence is None
            or row._candidate_evidence.schema_version != expected_schema
            or (row._prior_evidence is None) != (not row.had_prior)
            or (
                row._prior_evidence is not None
                and row.slot is not ProfileMigrationPublicationSlot.ACTIVE
                and row._prior_evidence.schema_version != expected_schema
            )
        ):
            raise ValueError


def _evidence_payload(evidence: _ArtifactEvidence) -> dict[str, object]:
    return {
        "byte_length": evidence.byte_length,
        "dev": evidence.dev,
        "ino": evidence.ino,
        "schema_version": evidence.schema_version,
        "sha256": evidence.sha256_digest.hex(),
    }


def _frame(recovery: dict[str, object]) -> tuple[bytes, bytes]:
    recovery_payload = json.dumps(
        recovery,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    checksum = sha256(recovery_payload).digest()
    payload = (
        json.dumps(
            {"checksum": checksum.hex(), "recovery": recovery},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if len(payload) > MAX_PROFILE_MIGRATION_JOURNAL_BYTES:
        raise ValueError
    return payload, checksum


def _new_profile_migration_journal_authority(
    *,
    parent_identity: os.stat_result,
    rows: tuple[
        tuple[
            ProfileMigrationJournalSlot,
            os.stat_result,
            tuple[int, bytes],
            int,
            os.stat_result | None,
            tuple[int, bytes] | None,
            int | None,
        ],
        ...,
    ],
) -> _JournalAuthority:
    """Build private authority only from publisher-retained exact identities."""

    authority_rows: list[ProfileMigrationJournalSlot] = []
    for (
        row,
        candidate,
        candidate_content,
        candidate_schema,
        prior,
        prior_content,
        prior_schema,
    ) in rows:
        candidate_length, candidate_sha256 = candidate_content
        candidate_evidence = _ArtifactEvidence(
            candidate.st_dev,
            candidate.st_ino,
            candidate_length,
            candidate_sha256,
            candidate_schema,
        )
        prior_evidence = None
        if row.had_prior:
            if prior is None or prior_content is None or prior_schema is None:
                raise ValueError
            prior_length, prior_sha256 = prior_content
            prior_evidence = _ArtifactEvidence(
                prior.st_dev,
                prior.st_ino,
                prior_length,
                prior_sha256,
                prior_schema,
            )
        elif prior is not None or prior_content is not None or prior_schema is not None:
            raise ValueError
        authority_rows.append(
            ProfileMigrationJournalSlot(
                slot=row.slot,
                candidate=row.candidate,
                target=row.target,
                rollback=row.rollback,
                had_prior=row.had_prior,
                _candidate_evidence=candidate_evidence,
                _prior_evidence=prior_evidence,
            )
        )
    result = _JournalAuthority(
        parent_identity.st_dev,
        parent_identity.st_ino,
        tuple(authority_rows),
    )
    _validate_recovery_rows(result.rows)
    _validate_authority_evidence(result.rows)
    return result


def _encode_initial_journal(authority: _JournalAuthority) -> tuple[bytes, bytes]:
    rows = []
    for row in authority.rows:
        if row._candidate_evidence is None:
            raise ValueError
        rows.append(
            {
                "candidate": row.candidate,
                "candidate_evidence": _evidence_payload(row._candidate_evidence),
                "had_prior": row.had_prior,
                "prior_evidence": (
                    None
                    if row._prior_evidence is None
                    else _evidence_payload(row._prior_evidence)
                ),
                "rollback": row.rollback,
                "slot": row.slot.value,
                "target": row.target,
            }
        )
    return _frame(
        {
            "authority": {
                "parent": {"dev": authority.parent_dev, "ino": authority.parent_ino},
                "slots": rows,
            },
            "phase": "prepared",
            "version": 2,
        }
    )


def _encode_later_journal(authority_checksum: bytes, *, phase: str) -> bytes:
    if phase not in _PHASES - {"prepared"} or len(authority_checksum) != 32:
        raise ValueError
    return _frame(
        {
            "authority_checksum": authority_checksum.hex(),
            "phase": phase,
            "version": 2,
        }
    )[0]


def _parse_evidence(value: object) -> _ArtifactEvidence:
    if type(value) is not dict or set(value) != {
        "byte_length",
        "dev",
        "ino",
        "schema_version",
        "sha256",
    }:
        raise ValueError
    if (
        type(value["dev"]) is not int
        or type(value["ino"]) is not int
        or type(value["byte_length"]) is not int
        or type(value["schema_version"]) is not int
        or value["schema_version"] not in (1, 2, 3, 4)
        or type(value["sha256"]) is not str
        or value["dev"] < 0
        or value["ino"] <= 0
        or value["byte_length"] < 0
    ):
        raise ValueError
    digest = bytes.fromhex(value["sha256"])
    if len(digest) != 32:
        raise ValueError
    return _ArtifactEvidence(
        value["dev"],
        value["ino"],
        value["byte_length"],
        digest,
        value["schema_version"],
    )


def _decode_journal_frame(
    frame: bytes,
    *,
    previous_phase: str | None,
    authority_checksum: bytes | None,
) -> tuple[
    str, tuple[ProfileMigrationJournalSlot, ...] | None, tuple[int, int] | None, bytes
]:
    transitions = {
        None: {"prepared"},
        "prepared": {"publishing"},
        "publishing": {"complete", "restoring"},
        "restoring": {"unavailable"},
        "complete": set(),
        "unavailable": set(),
    }
    decoded = json.loads(frame)
    if type(decoded) is not dict or set(decoded) != {"checksum", "recovery"}:
        raise ValueError
    recovery = decoded["recovery"]
    if type(recovery) is not dict:
        raise ValueError
    canonical_recovery = json.dumps(
        recovery,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    phase = recovery.get("phase")
    checksum = sha256(canonical_recovery).digest()
    if (
        type(decoded["checksum"]) is not str
        or decoded["checksum"] != checksum.hex()
        or recovery.get("version") != 2
        or phase not in transitions[previous_phase]
    ):
        raise ValueError
    if previous_phase is not None:
        if set(recovery) != {"authority_checksum", "phase", "version"}:
            raise ValueError
        if (
            authority_checksum is None
            or recovery["authority_checksum"] != authority_checksum.hex()
        ):
            raise ValueError
        if _encode_later_journal(authority_checksum, phase=phase) != frame:
            raise ValueError
        return phase, None, None, authority_checksum
    if set(recovery) != {"authority", "phase", "version"}:
        raise ValueError
    authority = recovery["authority"]
    if type(authority) is not dict or set(authority) != {"parent", "slots"}:
        raise ValueError
    parent = authority["parent"]
    if (
        type(parent) is not dict
        or set(parent) != {"dev", "ino"}
        or type(parent["dev"]) is not int
        or type(parent["ino"]) is not int
        or parent["dev"] < 0
        or parent["ino"] <= 0
    ):
        raise ValueError
    rows = authority["slots"]
    if type(rows) is not list:
        raise ValueError
    recovery_rows: list[ProfileMigrationJournalSlot] = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "candidate",
            "candidate_evidence",
            "had_prior",
            "prior_evidence",
            "rollback",
            "slot",
            "target",
        }:
            raise ValueError
        had_prior = row["had_prior"]
        if type(had_prior) is not bool:
            raise ValueError
        prior_evidence = _parse_evidence(row["prior_evidence"]) if had_prior else None
        if (row["prior_evidence"] is None) != (not had_prior):
            raise ValueError
        recovery_rows.append(
            ProfileMigrationJournalSlot(
                slot=ProfileMigrationPublicationSlot(row["slot"]),
                candidate=row["candidate"],
                target=row["target"],
                rollback=row["rollback"],
                had_prior=had_prior,
                _candidate_evidence=_parse_evidence(row["candidate_evidence"]),
                _prior_evidence=prior_evidence,
            )
        )
    result = tuple(recovery_rows)
    _validate_recovery_rows(result)
    _validate_authority_evidence(result)
    return phase, result, (parent["dev"], parent["ino"]), checksum


def parse_profile_migration_journal(raw: bytes) -> ParsedProfileMigrationJournal:
    """Parse only the bounded exact journal grammar used by startup recovery."""

    result: ParsedProfileMigrationJournal | None = None
    parse_error: BaseException | None = None
    try:
        if (
            type(raw) is not bytes
            or not raw
            or len(raw) > MAX_PROFILE_MIGRATION_JOURNAL_BYTES
        ):
            raise ValueError
        complete_frames = raw.splitlines(keepends=True)
        if complete_frames and not complete_frames[-1].endswith(b"\n"):
            complete_frames.pop()
        if not complete_frames:
            raise ValueError
        expected_rows: tuple[ProfileMigrationJournalSlot, ...] | None = None
        parent_identity: tuple[int, int] | None = None
        authority_checksum: bytes | None = None
        previous_phase: str | None = None
        for frame in complete_frames:
            previous_phase, recovery_rows, parsed_parent, parsed_checksum = (
                _decode_journal_frame(
                    frame,
                    previous_phase=previous_phase,
                    authority_checksum=authority_checksum,
                )
            )
            if recovery_rows is not None:
                expected_rows = recovery_rows
                parent_identity = parsed_parent
                authority_checksum = parsed_checksum
        assert (
            expected_rows is not None
            and previous_phase is not None
            and parent_identity is not None
            and authority_checksum is not None
        )
        result = ParsedProfileMigrationJournal(
            2,
            previous_phase,
            expected_rows,
            parent_identity[0],
            parent_identity[1],
            authority_checksum,
        )
    except BaseException as error:
        parse_error = error
    if parse_error is not None:
        if not isinstance(parse_error, Exception):
            raise parse_error
        raise ProfileRepositoryError("migration_failed") from None
    assert result is not None
    return result


__all__ = [
    "PROFILE_MIGRATION_SLOT_SEQUENCES",
    "ParsedProfileMigrationJournal",
    "ProfileMigrationJournalSlot",
    "ProfileMigrationPublicationSlot",
    "ProfileMigrationPublicationStage",
    "parse_profile_migration_journal",
]
