"""Bounded exact codec for profile-migration publication journals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
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

    def __repr__(self) -> str:
        return "ProfileMigrationJournalSlot(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class ParsedProfileMigrationJournal:
    """Path-free facts from one exact recognized publication journal."""

    version: int
    phase: str
    recovery_rows: tuple[ProfileMigrationJournalSlot, ...]

    @property
    def slots(self) -> tuple[str, ...]:
        """Return only recognized slot labels for compatibility checks."""

        return tuple(row.slot.value for row in self.recovery_rows)

    def __repr__(self) -> str:
        return (
            "ParsedProfileMigrationJournal("
            f"version={self.version!r}, phase={self.phase!r}, recovery_rows=<private>)"
        )


MAX_PROFILE_MIGRATION_JOURNAL_BYTES: Final = 4096
_PHASES: Final = frozenset(
    {"prepared", "publishing", "restoring", "complete", "unavailable"}
)
_CANONICAL_SLOT_ORDERS: Final = frozenset(
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


def _validate_recovery_rows(
    rows: tuple[ProfileMigrationJournalSlot, ...],
) -> None:
    if (
        type(rows) is not tuple
        or tuple(row.slot for row in rows) not in _CANONICAL_SLOT_ORDERS
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


def encode_profile_migration_journal(
    slots: tuple[ProfileMigrationJournalSlot, ...],
    *,
    phase: str,
) -> bytes:
    """Encode the single canonical bounded journal representation."""

    if phase not in _PHASES:
        raise ValueError
    _validate_recovery_rows(slots)
    rows = [
        {
            "candidate": slot.candidate,
            "had_prior": slot.had_prior,
            "rollback": slot.rollback,
            "slot": slot.slot.value,
            "target": slot.target,
        }
        for slot in slots
    ]
    payload = (
        json.dumps(
            {"phase": phase, "slots": rows, "version": 1},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if len(payload) > MAX_PROFILE_MIGRATION_JOURNAL_BYTES:
        raise ValueError
    return payload


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
        decoded = json.loads(raw)
        if set(decoded) != {"phase", "slots", "version"}:
            raise ValueError
        if decoded["version"] != 1 or decoded["phase"] not in _PHASES:
            raise ValueError
        rows = decoded["slots"]
        if type(rows) is not list or not 1 <= len(rows) <= 3:
            raise ValueError
        slot_names: list[str] = []
        recovery_rows: list[ProfileMigrationJournalSlot] = []
        for row in rows:
            if type(row) is not dict or set(row) != {
                "candidate",
                "had_prior",
                "rollback",
                "slot",
                "target",
            }:
                raise ValueError
            slot = ProfileMigrationPublicationSlot(row["slot"])
            if slot.value in slot_names or type(row["had_prior"]) is not bool:
                raise ValueError
            for key in ("candidate", "rollback", "target"):
                value = row[key]
                if (
                    type(value) is not str
                    or not value
                    or len(value.encode("utf-8")) > 255
                    or Path(value).name != value
                    or value in {".", ".."}
                    or "\x00" in value
                ):
                    raise ValueError
            slot_names.append(slot.value)
            recovery_rows.append(
                ProfileMigrationJournalSlot(
                    slot=slot,
                    candidate=row["candidate"],
                    target=row["target"],
                    rollback=row["rollback"],
                    had_prior=row["had_prior"],
                )
            )
        if (
            encode_profile_migration_journal(
                tuple(recovery_rows),
                phase=decoded["phase"],
            )
            != raw
        ):
            raise ValueError
        result = ParsedProfileMigrationJournal(
            1,
            decoded["phase"],
            tuple(recovery_rows),
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
    "ParsedProfileMigrationJournal",
    "ProfileMigrationJournalSlot",
    "ProfileMigrationPublicationSlot",
    "ProfileMigrationPublicationStage",
    "encode_profile_migration_journal",
    "parse_profile_migration_journal",
]
