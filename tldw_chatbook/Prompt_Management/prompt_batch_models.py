"""Strict immutable contracts for atomic local Prompt batch mutations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

_SQLITE_MAX_INTEGER = 2**63 - 1
_ARTIFACT_TYPES = frozenset({"prompt", "recipe"})


def _require_positive_sqlite_integer(value: object, *, field_name: str) -> None:
    if type(value) is not int or not 1 <= value <= _SQLITE_MAX_INTEGER:
        raise ValueError(f"{field_name} must be a positive integer in SQLite range.")


def _require_title(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("title must be non-empty exact text.")


def _require_artifact_type(value: object) -> None:
    if type(value) is not str or value not in _ARTIFACT_TYPES:
        raise ValueError("artifact_type must be exactly 'prompt' or 'recipe'.")


def _require_canonical_entries(entries: tuple[object, ...], entry_type: type) -> None:
    if type(entries) is not tuple:
        raise TypeError("entries must be an exact tuple.")
    if not entries:
        raise ValueError("entries must be non-empty.")
    if any(type(entry) is not entry_type for entry in entries):
        raise TypeError(
            f"entries must contain only exact {entry_type.__name__} values."
        )
    validated_entries = cast(tuple[Any, ...], entries)
    local_ids = tuple(entry.local_id for entry in validated_entries)
    if local_ids != tuple(sorted(set(local_ids))):
        raise ValueError("entries must use unique canonical ascending local IDs.")


@dataclass(frozen=True, slots=True)
class PromptBatchTarget:
    """One exact active or tombstoned Prompt row version to mutate.

    Attributes:
        local_id: Positive SQLite-range local Prompt row ID.
        expected_version: Positive SQLite-range version captured by the caller.
    """

    local_id: int = field(repr=False)
    expected_version: int = field(repr=False)

    def __post_init__(self) -> None:
        _require_positive_sqlite_integer(self.local_id, field_name="local_id")
        _require_positive_sqlite_integer(
            self.expected_version, field_name="expected_version"
        )


def validate_prompt_batch_targets(
    targets: tuple[PromptBatchTarget, ...],
) -> tuple[PromptBatchTarget, ...]:
    """Validate and canonically order one strict Prompt mutation batch.

    Args:
        targets: Exact non-empty tuple of exact batch targets.

    Returns:
        Targets in ascending local-ID order. An already canonical tuple is
        returned unchanged.

    Raises:
        TypeError: If the container or any target has a non-exact type.
        ValueError: If the tuple is empty or repeats a local ID.
    """
    if type(targets) is not tuple:
        raise TypeError("targets must be an exact tuple.")
    if not targets:
        raise ValueError("targets must be non-empty.")
    if any(type(target) is not PromptBatchTarget for target in targets):
        raise TypeError("targets must contain only exact PromptBatchTarget values.")
    for target in targets:
        local_id = getattr(target, "local_id", None)
        expected_version = getattr(target, "expected_version", None)
        _require_positive_sqlite_integer(local_id, field_name="local_id")
        _require_positive_sqlite_integer(
            expected_version, field_name="expected_version"
        )
    if len({target.local_id for target in targets}) != len(targets):
        raise ValueError("targets must use unique local IDs.")
    if all(
        previous.local_id < current.local_id
        for previous, current in zip(targets, targets[1:])
    ):
        return targets
    return tuple(sorted(targets, key=lambda target: target.local_id))


@dataclass(frozen=True, slots=True)
class PromptDeleteReceiptEntry:
    """One committed Prompt tombstone needed by the UI and atomic Undo.

    Attributes:
        local_id: Positive SQLite-range local Prompt row ID.
        title: Non-empty literal Prompt or Recipe title.
        artifact_type: Exact supported artifact type.
        tombstone_version: Positive version produced by the delete.
    """

    local_id: int = field(repr=False)
    title: str = field(repr=False)
    artifact_type: Literal["prompt", "recipe"]
    tombstone_version: int = field(repr=False)

    def __post_init__(self) -> None:
        _require_positive_sqlite_integer(self.local_id, field_name="local_id")
        _require_title(self.title)
        _require_artifact_type(self.artifact_type)
        _require_positive_sqlite_integer(
            self.tombstone_version, field_name="tombstone_version"
        )


@dataclass(frozen=True, slots=True)
class PromptBatchDeleteResult:
    """Canonical non-empty receipt for one committed batch delete.

    Attributes:
        entries: Receipt entries in unique ascending local-ID order.
    """

    entries: tuple[PromptDeleteReceiptEntry, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_canonical_entries(self.entries, PromptDeleteReceiptEntry)

    @property
    def targets(self) -> tuple[PromptBatchTarget, ...]:
        """Return canonical targets for restoring every receipt entry."""
        return tuple(
            PromptBatchTarget(entry.local_id, entry.tombstone_version)
            for entry in self.entries
        )


@dataclass(frozen=True, slots=True)
class PromptRestoreResultEntry:
    """One Prompt row version produced by a committed restore.

    Attributes:
        local_id: Positive SQLite-range restored local Prompt row ID.
        restored_version: Positive version produced by the restore.
    """

    local_id: int = field(repr=False)
    restored_version: int = field(repr=False)

    def __post_init__(self) -> None:
        _require_positive_sqlite_integer(self.local_id, field_name="local_id")
        _require_positive_sqlite_integer(
            self.restored_version, field_name="restored_version"
        )


@dataclass(frozen=True, slots=True)
class PromptBatchRestoreResult:
    """Canonical non-empty result for one committed batch restore.

    Attributes:
        entries: Restored entries in unique ascending local-ID order.
    """

    entries: tuple[PromptRestoreResultEntry, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_canonical_entries(self.entries, PromptRestoreResultEntry)
