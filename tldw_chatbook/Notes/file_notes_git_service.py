"""Pure session grouping and Git status policy for File Notes."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Notes.file_notes_session_owner import (
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionChangeAction,
    SessionChangeGroup,
    SessionGitRow,
    StagingOwnership,
)

PorcelainKind = Literal[
    "ordinary",
    "rename",
    "unmerged",
    "untracked",
    "ignored",
    "nested_repository",
    "unavailable",
    "error",
]


class PorcelainV2ParseError(ValueError):
    """Raised when porcelain-v2 bytes are incomplete or malformed."""


class PorcelainPathOutsideSessionError(PorcelainV2ParseError):
    """Raised when Git reports a path outside the complete session whitelist."""


@dataclass(frozen=True, slots=True)
class PorcelainRecord:
    """One byte-safely decoded porcelain-v2 status record."""

    kind: PorcelainKind
    path: str | None
    index_status: str = "."
    worktree_status: str = "."
    submodule: str | None = None
    modes: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    original_path: str | None = None
    score: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "object_ids", tuple(self.object_ids))


@dataclass(slots=True)
class _GroupBuilder:
    group_id: int
    endpoints: list[str]
    source_path: str
    destination_path: str | None
    current_path: str
    latest_action: SessionChangeAction
    latest_sequence: int

    def add_endpoint(self, path: str) -> None:
        if path not in self.endpoints:
            self.endpoints.append(path)

    def freeze(self) -> SessionChangeGroup:
        return SessionChangeGroup(
            group_id=self.group_id,
            endpoints=tuple(self.endpoints),
            source_path=self.source_path,
            destination_path=self.destination_path,
            current_path=self.current_path,
            latest_action=self.latest_action,
            latest_sequence=self.latest_sequence,
        )


def coalesce_session_changes(
    changes: Sequence[SequencedSessionChange],
) -> tuple[SessionChangeGroup, ...]:
    """Coalesce session events using only each lineage's active path."""
    active_paths: dict[str, _GroupBuilder] = {}
    builders: list[_GroupBuilder] = []

    for sequenced in sorted(changes, key=lambda item: item.sequence):
        change = sequenced.change
        path = change.relative_path
        builder = active_paths.get(path)
        if builder is None:
            builder = _GroupBuilder(
                group_id=sequenced.sequence,
                endpoints=[path],
                source_path=path,
                destination_path=None,
                current_path=path,
                latest_action=change.action,
                latest_sequence=sequenced.sequence,
            )
            builders.append(builder)

        if change.action == "moved":
            destination = change.destination_path
            if destination is None:
                raise ValueError("A moved session change requires a destination")
            if active_paths.get(path) is builder:
                del active_paths[path]
            builder.add_endpoint(path)
            builder.add_endpoint(destination)
            builder.destination_path = destination
            builder.current_path = destination
            active_paths[destination] = builder
        else:
            builder.add_endpoint(path)
            builder.current_path = path
            active_paths[path] = builder

        builder.latest_action = change.action
        builder.latest_sequence = sequenced.sequence

    return tuple(builder.freeze() for builder in builders)


def parse_porcelain_v2_z(
    payload: bytes,
    *,
    allowed_paths: frozenset[str],
) -> tuple[PorcelainRecord, ...]:
    """Parse NUL-delimited porcelain-v2 bytes and fail closed on every path."""
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise PorcelainV2ParseError("Porcelain-v2 payload is not NUL terminated")

    fields = payload[:-1].split(b"\0")
    records: list[PorcelainRecord] = []
    position = 0
    while position < len(fields):
        raw_record = fields[position]
        position += 1
        if not raw_record:
            raise PorcelainV2ParseError("Porcelain-v2 payload contains an empty record")
        if raw_record.startswith(b"# "):
            continue
        marker = raw_record[:1]
        if marker == b"1":
            records.append(_parse_ordinary(raw_record, allowed_paths))
            continue
        if marker == b"2":
            if position >= len(fields):
                raise PorcelainV2ParseError(
                    "Porcelain-v2 rename record lacks its original path"
                )
            original_path = _decode_allowed_path(
                fields[position],
                allowed_paths,
            )
            position += 1
            records.append(
                _parse_rename(
                    raw_record,
                    original_path,
                    allowed_paths,
                )
            )
            continue
        if marker == b"u":
            records.append(_parse_unmerged(raw_record, allowed_paths))
            continue
        if marker == b"?":
            records.append(
                _parse_simple_record(
                    raw_record,
                    kind="untracked",
                    allowed_paths=allowed_paths,
                )
            )
            continue
        if marker == b"!":
            records.append(
                _parse_simple_record(
                    raw_record,
                    kind="ignored",
                    allowed_paths=allowed_paths,
                )
            )
            continue
        raise PorcelainV2ParseError("Unsupported porcelain-v2 record type")
    return tuple(records)


def compute_stage_closure(
    endpoints: Collection[str],
    index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]:
    """Return literal endpoints plus tracked index ancestors/descendants."""
    closure = set(endpoints)
    for endpoint in endpoints:
        closure.update(
            path
            for path in index_entries
            if _paths_overlap(endpoint, path)
        )
    return frozenset(closure)


def compute_unstage_closure(
    baselines: Mapping[str, IndexBaseline],
    current_index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]:
    """Return paths an exact baseline restoration may replace in the index."""
    closure = set(baselines)
    for path, baseline in baselines.items():
        if baseline.entry is None:
            continue
        closure.update(
            current_path
            for current_path in current_index_entries
            if _paths_overlap(path, current_path)
        )
    return frozenset(closure)


def stage_group_is_closed(
    group: SessionChangeGroup,
    index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Return whether the Stage closure remains inside one session lineage."""
    return compute_stage_closure(
        group.endpoints,
        index_entries,
    ).issubset(group.endpoints)


def unstage_group_is_closed(
    group: SessionChangeGroup,
    baselines: Mapping[str, IndexBaseline],
    current_index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Return whether baseline restoration stays inside one session lineage."""
    return compute_unstage_closure(
        baselines,
        current_index_entries,
    ).issubset(group.endpoints)


def stage_pathspecs(
    group: SessionChangeGroup,
    status_records: Sequence[PorcelainRecord],
    index_entries: Mapping[str, IndexEntry],
) -> tuple[bytes, ...]:
    """Encode only effective endpoints, omitting absent transient lineage."""
    changed_paths: set[str] = set()
    for record in status_records:
        if record.kind != "untracked" and record.worktree_status == ".":
            continue
        if record.path is not None:
            changed_paths.add(record.path)
        if record.original_path is not None:
            changed_paths.add(record.original_path)
    return tuple(
        os.fsencode(path)
        for path in group.endpoints
        if path in changed_paths
    )


def index_entry_has_unsupported_semantics(entry: IndexEntry) -> bool:
    """Return whether an entry carries semantics this slice will not alter."""
    return bool(entry.semantic_flags) or (
        bool(entry.object_id)
        and set(entry.object_id) == {"0"}
    )


def index_entry_signature(
    entry: IndexEntry,
) -> tuple[str, str, int, tuple[str, ...]]:
    """Return the exact mode/object/stage/semantic signature."""
    return (
        entry.mode,
        entry.object_id,
        entry.stage,
        entry.semantic_flags,
    )


def ownership_signature_matches(
    ownership: StagingOwnership,
    *,
    repository: RepositoryIdentity,
    head: HeadIdentity,
    topology: tuple[str, ...],
    current_index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Compare exact repository, HEAD, topology, entry, and flag evidence."""
    if (
        ownership.repository != repository
        or ownership.head != head
        or ownership.approved_endpoint_topology != topology
    ):
        return False
    return all(
        current_index_entries.get(path) == expected
        for path, expected in ownership.post_stage_entries.items()
    )


def classify_session_rows(
    groups: Sequence[SessionChangeGroup],
    status_records: Sequence[PorcelainRecord],
    index_entries: Mapping[str, IndexEntry],
    ownership: Mapping[int, StagingOwnership],
) -> tuple[SessionGitRow, ...]:
    """Apply the frozen row/action policy to every coalesced session group."""
    global_records = tuple(
        record for record in status_records if record.path is None
    )
    rows: list[SessionGitRow] = []
    for group in groups:
        records = global_records + tuple(
            record
            for record in status_records
            if _record_touches_group(record, group)
        )
        entries = tuple(
            entry
            for path, entry in index_entries.items()
            if path in group.endpoints
        )
        rows.append(
            _classify_group(
                group,
                records,
                entries,
                index_entries,
                ownership.get(group.group_id),
            )
        )
    return tuple(rows)


def _parse_ordinary(
    raw_record: bytes,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 8)
    if len(fields) != 9 or fields[0] != b"1":
        raise PorcelainV2ParseError("Malformed ordinary porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="ordinary",
        path=_decode_allowed_path(fields[8], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:6]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[6:8]
        ),
    )


def _parse_rename(
    raw_record: bytes,
    original_path: str,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 9)
    if len(fields) != 10 or fields[0] != b"2":
        raise PorcelainV2ParseError("Malformed rename porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="rename",
        path=_decode_allowed_path(fields[9], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:6]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[6:8]
        ),
        original_path=original_path,
        score=_decode_ascii(fields[8], "rename score"),
    )


def _parse_unmerged(
    raw_record: bytes,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 10)
    if len(fields) != 11 or fields[0] != b"u":
        raise PorcelainV2ParseError("Malformed unmerged porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="unmerged",
        path=_decode_allowed_path(fields[10], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:7]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[7:10]
        ),
    )


def _parse_simple_record(
    raw_record: bytes,
    *,
    kind: Literal["untracked", "ignored"],
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    if len(raw_record) < 3 or raw_record[1:2] != b" ":
        raise PorcelainV2ParseError(f"Malformed {kind} porcelain-v2 record")
    return PorcelainRecord(
        kind=kind,
        path=_decode_allowed_path(raw_record[2:], allowed_paths),
    )


def _decode_allowed_path(
    raw_path: bytes,
    allowed_paths: frozenset[str],
) -> str:
    if not raw_path:
        raise PorcelainV2ParseError("Porcelain-v2 record contains an empty path")
    path = os.fsdecode(raw_path)
    if path not in allowed_paths:
        raise PorcelainPathOutsideSessionError(
            f"Git reported a path outside the session whitelist: {path!r}"
        )
    return path


def _decode_xy(raw_xy: bytes) -> tuple[str, str]:
    if len(raw_xy) != 2:
        raise PorcelainV2ParseError("Porcelain-v2 XY status must contain two bytes")
    xy = _decode_ascii(raw_xy, "XY status")
    return xy[0], xy[1]


def _decode_ascii(value: bytes, label: str) -> str:
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as error:
        raise PorcelainV2ParseError(
            f"Porcelain-v2 {label} is not ASCII"
        ) from error


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _record_touches_group(
    record: PorcelainRecord,
    group: SessionChangeGroup,
) -> bool:
    return (
        record.path in group.endpoints
        or record.original_path in group.endpoints
    )


def _classify_group(
    group: SessionChangeGroup,
    records: Sequence[PorcelainRecord],
    entries: Sequence[IndexEntry],
    all_index_entries: Mapping[str, IndexEntry],
    owned: StagingOwnership | None,
) -> SessionGitRow:
    error = next((record for record in records if record.kind == "error"), None)
    if error is not None:
        return SessionGitRow(
            group,
            "error",
            disabled_reason=error.message or "Git status failed",
        )
    unavailable = next(
        (record for record in records if record.kind == "unavailable"),
        None,
    )
    if unavailable is not None:
        return SessionGitRow(
            group,
            "unavailable",
            disabled_reason=unavailable.message or "Git is unavailable",
        )
    if any(record.kind == "nested_repository" for record in records):
        return SessionGitRow(
            group,
            "nested_repository",
            disabled_reason="Nested repository unsupported",
        )
    if (
        any(record.kind == "unmerged" for record in records)
        or any(entry.stage != 0 for entry in entries)
    ):
        return SessionGitRow(
            group,
            "conflict",
            disabled_reason="Git conflict",
        )
    if any(record.kind == "ignored" for record in records):
        return SessionGitRow(
            group,
            "ignored",
            disabled_reason="Ignored by Git",
        )
    if any(index_entry_has_unsupported_semantics(entry) for entry in entries):
        return SessionGitRow(
            group,
            "unsupported",
            disabled_reason="Unsupported Git index state",
        )
    if not stage_group_is_closed(group, all_index_entries):
        return SessionGitRow(
            group,
            "unsafe_closure",
            disabled_reason="Git mutation closure leaves this session lineage",
        )

    unstaged = any(
        record.kind == "untracked" or record.worktree_status != "."
        for record in records
    )
    staged = any(
        record.kind in {"ordinary", "rename"}
        and record.index_status != "."
        for record in records
    )

    if owned is not None:
        owned_entries_match = all(
            all_index_entries.get(path) == expected
            for path, expected in owned.post_stage_entries.items()
        )
        if (
            owned_entries_match
            and owned.approved_endpoint_topology != group.endpoints
        ):
            return SessionGitRow(
                group,
                "owned_topology_changed",
                stage_action="stage_update",
                disabled_reason="Path lineage changed; Stage update required",
            )
        if owned_entries_match:
            if unstaged:
                return SessionGitRow(
                    group,
                    "owned_newer_edits",
                    stage_action="stage_update",
                    unstage_eligible=True,
                )
            return SessionGitRow(
                group,
                "owned",
                unstage_eligible=True,
            )

    if staged and unstaged:
        return SessionGitRow(
            group,
            "external_partial",
            disabled_reason="External index state",
        )
    if staged:
        return SessionGitRow(
            group,
            "external_staged",
            disabled_reason="External index state",
        )
    if unstaged:
        return SessionGitRow(group, "unstaged", stage_action="stage")
    return SessionGitRow(group, "clean")
