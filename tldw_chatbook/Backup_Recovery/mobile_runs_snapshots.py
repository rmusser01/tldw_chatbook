"""Retained snapshot locators, never synthetic snapshots or allocator writes."""

from dataclasses import dataclass, field, replace

from .mobile_content_graph import _ContentBlocker
from .mobile_identity_plan import _Reservation

_MAX_INTEGER = 2**63 - 1
_MIN_INTEGER = -(2**63)


@dataclass(frozen=True)
class _NoteSnapshot:
    original_note: int
    original_snapshot: int | None
    mapped_snapshot: int | None
    original_run: str
    original_root: str = field(repr=False)
    missing: bool


@dataclass(frozen=True)
class _SnapshotAllocator:
    source_sequence: int | None
    canonical_sequence: int | None
    required_frontier: int


def _sequence(reads, purpose, budget):
    """One literal native metadata row on the already retained runs member."""
    reads.check()
    member = reads._connections.get((purpose, "runs"))
    if member is None:
        return None  # The fixed captured family has authenticated optional absence.
    budget.container(4)
    cursor = member[0].execute(
        "SELECT seq FROM sqlite_sequence WHERE name='change_snapshots'"
    )
    try:
        rows = cursor.fetchmany(2)
    finally:
        cursor.close()
    if not rows:
        return None
    if (
        len(rows) != 1
        or type(rows[0][0]) is not int
        or not 0 <= rows[0][0] <= _MAX_INTEGER
    ):
        raise ValueError("mobile_snapshot_allocator_provenance_invalid")
    return rows[0][0]


def _extend_snapshots(reads, identities, source, canonical, *, _read_budget):
    budget = _read_budget
    budget.container(
        sum(
            len(source[t]) + len(canonical[t])
            for t in ("change_notes", "change_snapshots")
        )
        * 16
    )
    blockers = []
    owners, live = {}, {}
    for purpose, tables in (("legacy", source), ("canonical", canonical)):
        live[purpose] = {row["id"]: row for row in tables["change_snapshots"]}
        owners[purpose] = {}
        for row in tables["change_snapshots"]:
            identity = row["id"]
            if (
                type(identity) is not int
                or not _MIN_INTEGER <= identity <= _MAX_INTEGER
            ):
                raise ValueError("mobile_snapshot_identity_invalid")
            owners[purpose].setdefault(identity, set()).add(
                (row["run_id"], row["root"])
            )
        for row in tables["change_notes"]:
            identity = row["snapshot_id"]
            if identity is None:
                continue
            if (
                type(identity) is not int
                or not _MIN_INTEGER <= identity <= _MAX_INTEGER
            ):
                raise ValueError("mobile_snapshot_identity_invalid")
            budget.value((row["run_id"], row["root"]))
            owners[purpose].setdefault(identity, set()).add(
                (row["run_id"], row["root"])
            )
        for row in tables["change_notes"]:
            identity = row["snapshot_id"]
            if identity is not None and len(owners[purpose][identity]) != 1:
                blockers.append(
                    _ContentBlocker(
                        "change_notes"
                        if purpose == "legacy"
                        else "canonical.change_notes",
                        (row["id"],),
                        "native_snapshot_owner_unproved",
                    )
                )

    budget.container(len(identities.reservations) * 3)
    base = {
        r.original: r.mapped
        for r in identities.reservations
        if r.domain == "change_snapshot"
    }
    source_ids, canonical_ids = set(owners["legacy"]), set(owners["canonical"])
    occupied = source_ids | canonical_ids | set(base.values())
    for original, target in base.items():
        if target in canonical_ids:
            blockers.append(
                _ContentBlocker(
                    "change_snapshots",
                    (original,),
                    "native_snapshot_historical_collision",
                )
            )
            for note in source["change_notes"]:
                if note["snapshot_id"] == original:
                    blockers.append(
                        _ContentBlocker(
                            "change_notes",
                            (note["id"],),
                            "native_snapshot_historical_collision",
                        )
                    )
    additions = []
    next_id = max(0, max(occupied, default=0)) + 1
    for original in sorted(source_ids - set(base)):
        if len(owners["legacy"][original]) != 1:
            continue
        target = original
        if target in canonical_ids or target in base.values():
            target = next_id
            next_id += 1
        if target > _MAX_INTEGER:
            blockers.append(
                _ContentBlocker(
                    "change_snapshots",
                    (original,),
                    "native_snapshot_allocator_exhausted",
                )
            )
            continue
        occupied.add(target)
        additions.append(_Reservation("change_snapshot", original, target))
    budget.container(len(identities.reservations) + len(additions))
    extended = replace(
        identities, reservations=identities.reservations + tuple(additions)
    )
    source_sequence = _sequence(reads, "legacy", budget)
    canonical_sequence = _sequence(reads, "canonical", budget)
    frontier = max(
        0,
        canonical_sequence or 0,
        *canonical_ids,
        *base.values(),
        *(r.mapped for r in additions),
    )
    if frontier == _MAX_INTEGER:
        blockers.append(
            _ContentBlocker(
                "change_snapshots", (), "native_snapshot_allocator_exhausted"
            )
        )
    if any(
        row["id"] > (source_sequence or 0) for row in source["change_snapshots"]
    ) or any(
        row["id"] > (canonical_sequence or 0) for row in canonical["change_snapshots"]
    ):
        blockers.append(
            _ContentBlocker(
                "change_snapshots", (), "native_snapshot_allocator_frontier_unproved"
            )
        )
    provenance = []
    for note in source["change_notes"]:
        identity = note["snapshot_id"]
        mapped = None
        if identity is not None:
            try:
                mapped = extended.mapped("change_snapshot", identity)
            except ValueError:
                blockers.append(
                    _ContentBlocker(
                        "change_notes",
                        (note["id"],),
                        "native_snapshot_locator_unproved",
                    )
                )
        budget.value((note["id"], identity, mapped, note["run_id"], note["root"]))
        provenance.append(
            _NoteSnapshot(
                note["id"],
                identity,
                mapped,
                note["run_id"],
                note["root"],
                identity is not None and identity not in live["legacy"],
            )
        )
    budget.container(len(provenance) + len(blockers) + 4)
    return (
        extended,
        tuple(provenance),
        _SnapshotAllocator(source_sequence, canonical_sequence, frontier),
        tuple(blockers),
    )
