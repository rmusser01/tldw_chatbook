"""Bounded exact codec for profile-migration publication journals."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Final, NoReturn, SupportsIndex

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_reference_types import MAX_REFERENCE_TOTAL_BYTES


# Total operational cap for one migration artifact: the existing 512 MiB
# aggregate reference quota plus 64 MiB of SQLite/storage headroom. The cap,
# rather than any assumption about intrinsically bounded free-list growth,
# supplies the bound enforced by publication, journaling, and recovery.
MAX_PROFILE_MIGRATION_ARTIFACT_BYTES: Final = (
    MAX_REFERENCE_TOTAL_BYTES + 64 * 1024 * 1024
)


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


PROFILE_MIGRATION_JOURNAL_LEAF: Final = ".profiles.migration-publication.json"
PROFILE_MIGRATION_CANDIDATE_LEAVES: Final = {
    ProfileMigrationPublicationSlot.ACTIVE: ".profile-migration-active.candidate.sqlite3",
    ProfileMigrationPublicationSlot.PRE_V3: ".profile-migration-pre-v3.candidate.sqlite3",
    ProfileMigrationPublicationSlot.PRE_V4: ".profile-migration-pre-v4.candidate.sqlite3",
}
PROFILE_MIGRATION_ROLLBACK_LEAVES: Final = {
    ProfileMigrationPublicationSlot.ACTIVE: ".profile-migration-active.rollback.sqlite3",
    ProfileMigrationPublicationSlot.PRE_V3: ".profile-migration-pre-v3.rollback.sqlite3",
    ProfileMigrationPublicationSlot.PRE_V4: ".profile-migration-pre-v4.rollback.sqlite3",
}


def _canonical_migration_leaves(
    slot: ProfileMigrationPublicationSlot,
    target: str,
) -> tuple[str, str]:
    """Return the only candidate and rollback leaves authorized for a slot."""

    if type(slot) is not ProfileMigrationPublicationSlot or type(target) is not str:
        raise ValueError
    return (
        PROFILE_MIGRATION_CANDIDATE_LEAVES[slot],
        PROFILE_MIGRATION_ROLLBACK_LEAVES[slot],
    )


class _OpaqueAuthority:
    """Refuse generic serialization and copying of private authority."""

    __slots__ = ()

    @staticmethod
    def _refuse() -> NoReturn:
        raise TypeError("private_authority")

    def __reduce__(self) -> NoReturn:
        self._refuse()

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        self._refuse()

    def __getstate__(self) -> NoReturn:
        self._refuse()

    def __copy__(self) -> NoReturn:
        self._refuse()

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        self._refuse()


class _ArtifactAuthorityCapsule(_OpaqueAuthority):
    """Non-serializable exact artifact authority."""

    __slots__ = ("__values",)
    __values: tuple[int, int, int, bytes, int]

    def __init__(
        self,
        dev: int,
        ino: int,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> None:
        if (
            type(byte_length) is not int
            or not 0 <= byte_length <= MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
        ):
            raise ValueError
        object.__setattr__(
            self,
            "_ArtifactAuthorityCapsule__values",
            (dev, ino, byte_length, sha256_digest, schema_version),
        )

    def __repr__(self) -> str:
        return "_ArtifactAuthorityCapsule(<private>)"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)

    def identity(self) -> tuple[int, int]:
        dev, ino, _, _, _ = self.__values
        return dev, ino

    def schema_is(self, schema_version: int) -> bool:
        return self.__values[4] == schema_version

    def fits(self, maximum_bytes: int) -> bool:
        return self.__values[2] <= maximum_bytes

    def payload(self) -> dict[str, object]:
        dev, ino, byte_length, digest, schema_version = self.__values
        return {
            "byte_length": byte_length,
            "dev": dev,
            "ino": ino,
            "schema_version": schema_version,
            "sha256": digest.hex(),
        }

    def matches(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> bool:
        dev, ino, expected_length, expected_digest, expected_schema = self.__values
        return (
            identity.st_dev == dev
            and identity.st_ino == ino
            and byte_length == expected_length
            and sha256_digest == expected_digest
            and schema_version == expected_schema
        )


class _ArtifactEvidence(_OpaqueAuthority):
    """Opaque exact artifact evidence used only through bounded methods."""

    __slots__ = ("__authority",)
    __authority: _ArtifactAuthorityCapsule

    def __init__(
        self,
        dev: int,
        ino: int,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> None:
        object.__setattr__(
            self,
            "_ArtifactEvidence__authority",
            _ArtifactAuthorityCapsule(
                dev,
                ino,
                byte_length,
                sha256_digest,
                schema_version,
            ),
        )

    def __repr__(self) -> str:
        return "_ArtifactEvidence(<private>)"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)

    def _identity(self) -> tuple[int, int]:
        return self.__authority.identity()

    def _schema_is(self, schema_version: int) -> bool:
        return self.__authority.schema_is(schema_version)

    def _fits(self, maximum_bytes: int) -> bool:
        return self.__authority.fits(maximum_bytes)

    def _payload(self) -> dict[str, object]:
        return self.__authority.payload()

    def matches(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> bool:
        return self.__authority.matches(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
        )


class ProfileMigrationJournalSlot(_OpaqueAuthority):
    """One recognized relative namespace row in a journal."""

    __slots__ = (
        "__authority",
        "__candidate",
        "__had_prior",
        "__rollback",
        "__slot",
        "__target",
    )
    __authority: tuple[_ArtifactEvidence | None, _ArtifactEvidence | None]
    __candidate: str
    __had_prior: bool
    __rollback: str
    __slot: ProfileMigrationPublicationSlot
    __target: str

    def __init__(
        self,
        *,
        slot: ProfileMigrationPublicationSlot,
        candidate: str,
        target: str,
        rollback: str,
        had_prior: bool,
    ) -> None:
        self._initialize(slot, candidate, target, rollback, had_prior, None, None)

    @classmethod
    def _with_authority(
        cls,
        *,
        slot: ProfileMigrationPublicationSlot,
        candidate: str,
        target: str,
        rollback: str,
        had_prior: bool,
        candidate_evidence: _ArtifactEvidence,
        prior_evidence: _ArtifactEvidence | None,
    ) -> ProfileMigrationJournalSlot:
        result = object.__new__(cls)
        result._initialize(
            slot,
            candidate,
            target,
            rollback,
            had_prior,
            candidate_evidence,
            prior_evidence,
        )
        return result

    def _initialize(
        self,
        slot: ProfileMigrationPublicationSlot,
        candidate: str,
        target: str,
        rollback: str,
        had_prior: bool,
        candidate_evidence: _ArtifactEvidence | None,
        prior_evidence: _ArtifactEvidence | None,
    ) -> None:
        object.__setattr__(self, "_ProfileMigrationJournalSlot__slot", slot)
        object.__setattr__(self, "_ProfileMigrationJournalSlot__candidate", candidate)
        object.__setattr__(self, "_ProfileMigrationJournalSlot__target", target)
        object.__setattr__(self, "_ProfileMigrationJournalSlot__rollback", rollback)
        object.__setattr__(self, "_ProfileMigrationJournalSlot__had_prior", had_prior)
        object.__setattr__(
            self,
            "_ProfileMigrationJournalSlot__authority",
            (candidate_evidence, prior_evidence),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)

    def __repr__(self) -> str:
        return "ProfileMigrationJournalSlot(<private>)"

    def __eq__(self, other: object) -> bool:
        return type(other) is ProfileMigrationJournalSlot and (
            self.slot,
            self.candidate,
            self.target,
            self.rollback,
            self.had_prior,
        ) == (
            other.slot,
            other.candidate,
            other.target,
            other.rollback,
            other.had_prior,
        )

    def __hash__(self) -> int:
        return hash(
            (self.slot, self.candidate, self.target, self.rollback, self.had_prior)
        )

    @property
    def slot(self) -> ProfileMigrationPublicationSlot:
        return self.__slot

    @property
    def candidate(self) -> str:
        return self.__candidate

    @property
    def target(self) -> str:
        return self.__target

    @property
    def rollback(self) -> str:
        return self.__rollback

    @property
    def had_prior(self) -> bool:
        return self.__had_prior

    def _evidence(self) -> tuple[_ArtifactEvidence | None, _ArtifactEvidence | None]:
        return self.__authority

    def matches_candidate(
        self,
        identity: os.stat_result,
        *,
        byte_length: int,
        sha256_digest: bytes,
        schema_version: int,
    ) -> bool:
        """Classify one verified object against the prepared authority."""

        candidate, _ = self.__authority
        return candidate is not None and candidate.matches(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
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

        _, prior = self.__authority
        return prior is not None and prior.matches(
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

        candidate_matches = self.matches_candidate(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
        )
        prior_matches = self.matches_prior(
            identity,
            byte_length=byte_length,
            sha256_digest=sha256_digest,
            schema_version=schema_version,
        )
        if candidate_matches and prior_matches:
            return None
        if candidate_matches:
            return "candidate"
        if prior_matches:
            return "prior"
        return None

    def evidence_fits(self, maximum_bytes: int) -> bool:
        """Return whether every recorded artifact fits a recovery byte bound."""

        if type(maximum_bytes) is not int or maximum_bytes < 0:
            return False
        candidate, prior = self.__authority
        return (
            candidate is not None
            and candidate._fits(maximum_bytes)
            and (prior is None or prior._fits(maximum_bytes))
        )


class _ParentAuthorityCapsule(_OpaqueAuthority):
    __slots__ = ("__identity",)
    __identity: tuple[int, int]

    def __init__(self, dev: int, ino: int) -> None:
        object.__setattr__(self, "_ParentAuthorityCapsule__identity", (dev, ino))

    def __repr__(self) -> str:
        return "_ParentAuthorityCapsule(<private>)"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)

    def identity(self) -> tuple[int, int]:
        return self.__identity

    def matches(self, identity: os.stat_result) -> bool:
        return (identity.st_dev, identity.st_ino) == self.__identity


class _JournalAuthority(_OpaqueAuthority):
    __slots__ = ("__parent", "__rows")
    __parent: _ParentAuthorityCapsule
    __rows: tuple[ProfileMigrationJournalSlot, ...]

    def __init__(
        self,
        parent_dev: int,
        parent_ino: int,
        rows: tuple[ProfileMigrationJournalSlot, ...],
    ) -> None:
        object.__setattr__(
            self,
            "_JournalAuthority__parent",
            _ParentAuthorityCapsule(parent_dev, parent_ino),
        )
        object.__setattr__(self, "_JournalAuthority__rows", rows)

    def __repr__(self) -> str:
        return "_JournalAuthority(<private>)"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)

    def _parent_identity(self) -> tuple[int, int]:
        return self.__parent.identity()

    def _recovery_rows(self) -> tuple[ProfileMigrationJournalSlot, ...]:
        return self.__rows


class ParsedProfileMigrationJournal(_OpaqueAuthority):
    """Path-free facts from one exact recognized publication journal."""

    __slots__ = ("__parent", "__phase", "__recovery_rows", "__version")
    __parent: _ParentAuthorityCapsule
    __phase: str
    __recovery_rows: tuple[ProfileMigrationJournalSlot, ...]
    __version: int

    def __init__(
        self,
        version: int,
        phase: str,
        recovery_rows: tuple[ProfileMigrationJournalSlot, ...],
        parent_dev: int,
        parent_ino: int,
    ) -> None:
        object.__setattr__(self, "_ParsedProfileMigrationJournal__version", version)
        object.__setattr__(self, "_ParsedProfileMigrationJournal__phase", phase)
        object.__setattr__(
            self, "_ParsedProfileMigrationJournal__recovery_rows", recovery_rows
        )
        object.__setattr__(
            self,
            "_ParsedProfileMigrationJournal__parent",
            _ParentAuthorityCapsule(parent_dev, parent_ino),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(name)

    @property
    def version(self) -> int:
        return self.__version

    @property
    def phase(self) -> str:
        return self.__phase

    @property
    def recovery_rows(self) -> tuple[ProfileMigrationJournalSlot, ...]:
        return self.__recovery_rows

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

        return self.__parent.matches(identity)


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
        if (row.candidate, row.rollback) != _canonical_migration_leaves(
            row.slot, row.target
        ):
            raise ValueError


def _validate_authority_evidence(
    rows: tuple[ProfileMigrationJournalSlot, ...],
    *,
    parent_dev: int,
) -> None:
    identities: set[tuple[int, int]] = set()
    for row in rows:
        expected_schema = _CANDIDATE_SCHEMA_BY_SLOT[row.slot]
        candidate, prior = row._evidence()
        if (
            candidate is None
            or not candidate._schema_is(expected_schema)
            or (prior is None) != (not row.had_prior)
            or (
                prior is not None
                and row.slot is not ProfileMigrationPublicationSlot.ACTIVE
                and not prior._schema_is(expected_schema)
            )
        ):
            raise ValueError
        for item in (candidate, prior):
            if item is None:
                continue
            identity = item._identity()
            if identity[0] != parent_dev or identity in identities:
                raise ValueError
            identities.add(identity)


def _evidence_payload(evidence: _ArtifactEvidence) -> dict[str, object]:
    return evidence._payload()


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
            ProfileMigrationJournalSlot._with_authority(
                slot=row.slot,
                candidate=row.candidate,
                target=row.target,
                rollback=row.rollback,
                had_prior=row.had_prior,
                candidate_evidence=candidate_evidence,
                prior_evidence=prior_evidence,
            )
        )
    result = _JournalAuthority(
        parent_identity.st_dev,
        parent_identity.st_ino,
        tuple(authority_rows),
    )
    result_rows = result._recovery_rows()
    _validate_recovery_rows(result_rows)
    _validate_authority_evidence(
        result_rows,
        parent_dev=result._parent_identity()[0],
    )
    return result


def _encode_initial_journal(authority: _JournalAuthority) -> tuple[bytes, bytes]:
    rows = []
    for row in authority._recovery_rows():
        if not row.evidence_fits(MAX_PROFILE_MIGRATION_ARTIFACT_BYTES):
            raise ValueError
        candidate_evidence, prior_evidence = row._evidence()
        if candidate_evidence is None:
            raise ValueError
        rows.append(
            {
                "candidate": row.candidate,
                "candidate_evidence": _evidence_payload(candidate_evidence),
                "had_prior": row.had_prior,
                "prior_evidence": (
                    None
                    if prior_evidence is None
                    else _evidence_payload(prior_evidence)
                ),
                "rollback": row.rollback,
                "slot": row.slot.value,
                "target": row.target,
            }
        )
    return _frame(
        {
            "authority": {
                "parent": dict(
                    zip(("dev", "ino"), authority._parent_identity(), strict=True)
                ),
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
        or value["byte_length"] > MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
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
            ProfileMigrationJournalSlot._with_authority(
                slot=ProfileMigrationPublicationSlot(row["slot"]),
                candidate=row["candidate"],
                target=row["target"],
                rollback=row["rollback"],
                had_prior=had_prior,
                candidate_evidence=_parse_evidence(row["candidate_evidence"]),
                prior_evidence=prior_evidence,
            )
        )
    result = tuple(recovery_rows)
    _validate_recovery_rows(result)
    _validate_authority_evidence(result, parent_dev=parent["dev"])
    reconstructed = _JournalAuthority(parent["dev"], parent["ino"], result)
    if _encode_initial_journal(reconstructed)[0] != frame:
        raise ValueError
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
    "MAX_PROFILE_MIGRATION_ARTIFACT_BYTES",
    "PROFILE_MIGRATION_SLOT_SEQUENCES",
    "ParsedProfileMigrationJournal",
    "ProfileMigrationJournalSlot",
    "ProfileMigrationPublicationSlot",
    "ProfileMigrationPublicationStage",
    "parse_profile_migration_journal",
]
