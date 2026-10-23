"""Binding-wide native admission and conservative project SQLite page allowance."""

from __future__ import annotations

from contextlib import contextmanager
import asyncio
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading

from .lifetime import BindingLifetime, LifetimeBusy
from .models import MAX_CACHE_BYTES, ProjectIndexRetirementUnproven

_SECTOR_UPPER_BOUND = 65536
_CONTROL_GROWTH = 4096
_SOURCE_ID = (
    "2025-02-18 13:38:58 "
    "873d4e274b4988d260ba8354a9718324a1c26187a4ab4c1cc0227c03d0f10e70"
)
_GROUP_ENTRY = re.compile(r"[0-9a-f]{64}")
_LIFETIME_ENTRY = re.compile(r"[0-9a-f]{64}\.lifetime\.lock")
_ACTIVE = threading.local()
_QUARANTINED: dict[Path, BindingLifetime] = {}
_UNRETIRED_OLDER: list[BindingLifetime] = []
_SYNC_EXECUTION = object()


def _execution_identity():
    try:
        return asyncio.current_task() or _SYNC_EXECUTION
    except RuntimeError:
        return _SYNC_EXECUTION


def _live_gate(group: Path) -> BindingLifetime:
    active = getattr(_ACTIVE, "gate", None)
    if (
        active is None
        or active[0] != group
        or active[1] != os.getpid()
        or active[2] != threading.get_ident()
        or active[3] is not _execution_identity()
        or active[4].closed
    ):
        raise ValueError("index_allocation_borrow_refused")
    return active[4]


def _regular_length(path: Path) -> int:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("index_allocation_alias_refused")
    if getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("index_allocation_alias_refused")
    return info.st_size


def _regular_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("index_allocation_alias_refused")


def _inventory(group: Path) -> dict[Path, int]:
    """Inspect the closed binding inventory; no symlink or unknown sidecar."""
    _regular_directory(group)
    lengths: dict[Path, int] = {}
    seen = 0
    for entry in group.iterdir():
        seen += 1
        if seen > 4096:
            raise ValueError("index_allocation_inventory_limit")
        if entry.name in {
            "allocation-v1.lock",
            "cleanup-owner-v1.lock",
        } or _LIFETIME_ENTRY.fullmatch(entry.name):
            lengths[entry] = _regular_length(entry)
            continue
        if not _GROUP_ENTRY.fullmatch(entry.name):
            raise ValueError("index_allocation_unknown_sidecar")
        _regular_directory(entry)
        main = entry / "workspace_project_index.db"
        for child in entry.iterdir():
            seen += 1
            if seen > 4096:
                raise ValueError("index_allocation_inventory_limit")
            if child.name not in {
                main.name,
                main.name + "-journal",
                main.name + "-wal",
                main.name + "-shm",
            }:
                raise ValueError("index_allocation_unknown_sidecar")
            lengths[child] = _regular_length(child)
    return lengths


@contextmanager
def allocation_gate(group: Path, *, existing_only: bool = False):
    """Try one binding-stable native capability; retain it on uncertain close."""
    group = Path(group).absolute()
    if group in _QUARANTINED:
        raise ValueError("index_allocation_retirement_unproven")
    active = getattr(_ACTIVE, "gate", None)
    if active is not None:
        yield _live_gate(group)
        return
    lease = BindingLifetime.acquire(
        group / "allocation-v1.lock", existing_only=existing_only
    )
    _ACTIVE.gate = (
        group,
        os.getpid(),
        threading.get_ident(),
        _execution_identity(),
        lease,
    )
    _ACTIVE.owned_lifetimes = []
    try:
        if _regular_length(lease.path) != 0:
            raise ValueError("index_allocation_control_not_empty")
        yield lease
    except ProjectIndexRetirementUnproven:
        _QUARANTINED[group] = lease
        raise
    finally:
        try:
            if _QUARANTINED.get(group) is not lease:
                lease.close()
        except BaseException:
            _QUARANTINED[group] = lease
            raise
        finally:
            _ACTIVE.gate = None
            _ACTIVE.owned_lifetimes = []


@contextmanager
def allocation_store(group: Path, store: object):
    """Allow only one independently owned native SQL store within a gate."""
    group = Path(group).absolute()
    _live_gate(group)
    previous = getattr(_ACTIVE, "store", None)
    if previous is not None and previous is not store:
        raise ValueError("index_allocation_overlapping_store")
    _ACTIVE.store = store
    try:
        yield
    finally:
        _ACTIVE.store = previous


def assert_store_unoccupied(group: Path) -> None:
    """Refuse peer preflight while another native store is still active."""
    group = Path(group).absolute()
    _live_gate(group)
    if getattr(_ACTIVE, "store", None) is not None:
        raise ValueError("index_allocation_overlapping_store")


def retain_uncertain(group: Path, lease: BindingLifetime) -> None:
    """Quarantine a native gate after unproven rollback/close/inventory."""
    group = Path(group).absolute()
    if _live_gate(group) is not lease:
        raise ValueError("index_allocation_gate_required")
    _QUARANTINED[group] = lease


def register_owned_lifetime(group: Path, lease: BindingLifetime) -> None:
    """Present an exact live native owner inside a cleanup control unit."""
    group = Path(group).absolute()
    _live_gate(group)
    if (
        not isinstance(lease, BindingLifetime)
        or lease.path.parent.absolute() != group
        or not _LIFETIME_ENTRY.fullmatch(lease.path.name)
    ):
        raise ValueError("index_allocation_owner_refused")
    lease.assert_live()
    _ACTIVE.owned_lifetimes.append(lease)


def legacy_namespace_paths(group: Path) -> tuple[Path, ...]:
    """List validated existing databases without creating a missing group gate."""
    group = Path(group).absolute()
    return tuple(
        sorted(
            path
            for path in _inventory(group)
            if path.name == "workspace_project_index.db"
        )
    )


def existing_namespace_paths(group: Path) -> tuple[Path, ...]:
    """List validated existing namespace databases under the held group gate."""
    group = Path(group).absolute()
    _live_gate(group)
    return legacy_namespace_paths(group)


def verify_pre_open(path: Path, group: Path) -> None:
    """Reject recovery sidecars before the target is opened by SQLite."""
    path, group = Path(path).absolute(), Path(group).absolute()
    _live_gate(group)
    verify_legacy_pre_open(path, group)


def verify_legacy_pre_open(path: Path, group: Path) -> None:
    """Validate an existing pre-gate keyword namespace without creating control."""
    path, group = Path(path).absolute(), Path(group).absolute()
    lengths = _inventory(group)
    allowed = {
        path,
        *(Path(str(path) + suffix) for suffix in ("-journal", "-wal", "-shm")),
    }
    if any(child.parent == path.parent and child not in allowed for child in lengths):
        raise ValueError("index_allocation_unknown_sidecar")
    if any(
        lengths.get(Path(str(path) + suffix), 0)
        for suffix in ("-journal", "-wal", "-shm")
    ):
        raise ValueError("index_allocation_recovery_sidecar")


def verify_legacy_settlement(path: Path, group: Path) -> None:
    verify_legacy_pre_open(path, group)


def admit_connection(
    conn: sqlite3.Connection,
    path: Path,
    group: Path,
    *,
    budget: int = MAX_CACHE_BYTES,
    owned_lifetime: BindingLifetime | None = None,
) -> int:
    """Set and read back the held main-file cap after source/geometry checks."""
    path, group = Path(path).absolute(), Path(group).absolute()
    _live_gate(group)
    if type(budget) is not int or budget <= 0:  # noqa: E721 - reject bool
        raise ValueError("invalid_index_allocation_budget")
    lengths = _inventory(group)
    owned = (*getattr(_ACTIVE, "owned_lifetimes", ()), owned_lifetime)
    for item in lengths:
        if not _LIFETIME_ENTRY.fullmatch(item.name):
            continue
        granted = False
        for capability in owned:
            if (
                capability is None
                or capability.closed
                or capability.path.absolute() != item
            ):
                continue
            try:
                capability.assert_live()
            except BaseException as error:
                _UNRETIRED_OLDER.append(capability)
                raise ProjectIndexRetirementUnproven(
                    "index_current_lifetime_retirement_unproven"
                ) from error
            granted = True
            break
        if granted:
            continue
        try:
            older = BindingLifetime.acquire(item, existing_only=True)
        except LifetimeBusy as error:
            raise ValueError("legacy_retirement_unproven") from error
        try:
            older.close()
        except BaseException as error:
            _UNRETIRED_OLDER.append(older)
            raise ProjectIndexRetirementUnproven(
                "index_older_lifetime_retirement_unproven"
            ) from error
    if any(
        lengths.get(Path(str(path) + suffix), 0)
        for suffix in ("-journal", "-wal", "-shm")
    ):
        raise ValueError("index_allocation_recovery_sidecar")
    dbs = conn.execute("PRAGMA database_list").fetchall()
    if len(dbs) != 1 or dbs[0][1] != "main" or Path(dbs[0][2]).absolute() != path:
        raise ValueError("index_allocation_attached_database")
    if (
        sqlite3.sqlite_version != "3.49.1"
        or conn.execute("SELECT sqlite_source_id()").fetchone()[0] != _SOURCE_ID
    ):
        raise ValueError("index_allocation_sqlite_unqualified")
    compile_options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    if not {"TEMP_STORE=1", "MAX_PAGE_SIZE=65536", "THREADSAFE=1"} <= compile_options:
        raise ValueError("index_allocation_sqlite_unqualified")
    # These connection-local controls do not write a separate durable file.
    conn.execute("PRAGMA cache_spill=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA secure_delete=ON")
    settings = {
        name: conn.execute(f"PRAGMA {name}").fetchone()[0]
        for name in (
            "journal_mode",
            "locking_mode",
            "synchronous",
            "auto_vacuum",
            "temp_store",
            "foreign_keys",
            "secure_delete",
            "cache_spill",
        )
    }
    if settings != {
        "journal_mode": "delete",
        "locking_mode": "normal",
        "synchronous": 2,
        "auto_vacuum": 2,
        "temp_store": 2,
        "foreign_keys": 1,
        "secure_delete": 1,
        "cache_spill": 0,
    } and not (
        settings["synchronous"] == 3
        and all(
            settings[name] == want
            for name, want in {
                "journal_mode": "delete",
                "locking_mode": "normal",
                "auto_vacuum": 2,
                "temp_store": 2,
                "foreign_keys": 1,
                "secure_delete": 1,
                "cache_spill": 0,
            }.items()
        )
    ):
        raise ValueError("index_allocation_sqlite_mode")
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    old_pages = conn.execute("PRAGMA page_count").fetchone()[0]
    if (
        page_size < 512
        or page_size > 65536
        or page_size & (page_size - 1)
        or old_pages < 0
        or lengths.get(path, 0) != old_pages * page_size
    ):
        raise ValueError("index_allocation_geometry")
    other = sum(
        length
        for item, length in lengths.items()
        if item not in {path, Path(str(path) + "-journal")}
    )
    observed_journal = lengths.get(Path(str(path) + "-journal"), 0)
    journal_bound = 2 * _SECTOR_UPPER_BOUND + old_pages * (page_size + 8)
    residual = budget - other - _CONTROL_GROWTH - max(observed_journal, journal_bound)
    cap = residual // page_size
    if cap < old_pages or cap < 0:
        raise ValueError("index_cache_limit")
    returned = conn.execute(f"PRAGMA main.max_page_count={cap}").fetchone()[0]
    read_back = conn.execute("PRAGMA main.max_page_count").fetchone()[0]
    if not old_pages <= returned <= cap or not old_pages <= read_back <= cap:
        raise ValueError("index_allocation_cap_refused")
    return cap


def admit_keyword_connection(
    conn: sqlite3.Connection,
    path: Path,
    group: Path,
    *,
    budget: int = MAX_CACHE_BYTES,
) -> int:
    """Preserve the established aggregate two-copy keyword page allowance."""
    path, group = Path(path).absolute(), Path(group).absolute()
    _live_gate(group)
    lengths = _inventory(group)
    page_size = conn.execute("PRAGMA main.page_size").fetchone()[0]
    old_pages = conn.execute("PRAGMA main.page_count").fetchone()[0]
    if page_size <= 0 or lengths.get(path, 0) != old_pages * page_size:
        raise ValueError("index_allocation_geometry")
    other = sum(length for item, length in lengths.items() if item != path)
    cap = (budget - other - 1024**2) // (2 * page_size)
    if cap < old_pages or cap < 0:
        raise ValueError("index_cache_limit")
    returned = conn.execute(f"PRAGMA main.max_page_count={cap}").fetchone()[0]
    read_back = conn.execute("PRAGMA main.max_page_count").fetchone()[0]
    if not old_pages <= returned <= cap or not old_pages <= read_back <= cap:
        raise ValueError("index_allocation_cap_refused")
    return cap


def verify_settlement(path: Path, group: Path, *, budget: int = MAX_CACHE_BYTES) -> int:
    """Inventory after native close, while the gate still excludes peers."""
    path, group = Path(path).absolute(), Path(group).absolute()
    _live_gate(group)
    lengths = _inventory(group)
    total = sum(lengths.values())
    if total > budget:
        raise ValueError("index_allocation_settlement_over_budget")
    if any(
        lengths.get(Path(str(path) + suffix), 0)
        for suffix in ("-journal", "-wal", "-shm")
    ):
        raise ValueError("index_allocation_settlement_sidecar")
    return total
