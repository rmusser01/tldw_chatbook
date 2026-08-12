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


@dataclass(frozen=True, slots=True)
class ProfileMigrationJournalSlot:
    """One recognized relative namespace row in a journal."""

    slot: ProfileMigrationPublicationSlot
    candidate: str
    target: str
    rollback: str
    had_prior: bool


@dataclass(frozen=True, slots=True)
class ParsedProfileMigrationJournal:
    """Path-free facts from one exact recognized publication journal."""

    version: int
    phase: str
    slots: tuple[str, ...]


MAX_PROFILE_MIGRATION_JOURNAL_BYTES: Final = 4096
_PHASES: Final = frozenset(
    {"prepared", "publishing", "restoring", "complete", "unavailable"}
)


def encode_profile_migration_journal(
    slots: tuple[ProfileMigrationJournalSlot, ...],
    *,
    phase: str,
) -> bytes:
    """Encode the single canonical bounded journal representation."""

    if phase not in _PHASES or not 1 <= len(slots) <= 3:
        raise ValueError
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
        slots: list[str] = []
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
            if slot.value in slots or type(row["had_prior"]) is not bool:
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
            slots.append(slot.value)
        if slots[0] != ProfileMigrationPublicationSlot.ACTIVE.value:
            raise ValueError
        if (
            encode_profile_migration_journal(
                tuple(
                    ProfileMigrationJournalSlot(
                        slot=ProfileMigrationPublicationSlot(row["slot"]),
                        candidate=row["candidate"],
                        target=row["target"],
                        rollback=row["rollback"],
                        had_prior=row["had_prior"],
                    )
                    for row in rows
                ),
                phase=decoded["phase"],
            )
            != raw
        ):
            raise ValueError
        result = ParsedProfileMigrationJournal(1, decoded["phase"], tuple(slots))
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
