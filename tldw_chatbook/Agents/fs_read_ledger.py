"""TASK-28238 phase 1: per-run read-ledger for the fs stale-write guard.

Concurrent fleet children share ONE LocalToolProvider (that is why
RunToolPolicy keys its caps by (run_id, tool)); a per-provider ledger would
let one child's write mask a sibling's staleness. So entries key on
(run_id, canonical_path). The canonical path uses the SAME normalization as
the fs_write CAS lock (`os.path.normcase(str(p.absolute()))`,
Tools/local_tool_impls.py `_write_lock_for`) so the two mechanisms agree.

Bounded: at most ``max_paths_per_run`` entries per run_id, oldest evicted —
necessary because the MCP server provider (MCP/local_server_tools.py
build_server_local_provider) lives for the process with run_id always "".
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_PATHS_PER_RUN = 512


def canonical_ledger_key(resolved: Path) -> str:
    """The ledger's path identity; byte-identical to the fs_write CAS lock key.

    Args:
        resolved: An already-resolved path (resolve_workspace_path output).

    Returns:
        ``os.path.normcase(str(resolved.absolute()))``.
    """
    return os.path.normcase(str(resolved.absolute()))


@dataclass(frozen=True)
class ReadStamp:
    """What this run last saw at a path: a whole-file hash, or absence."""

    sha256: str | None
    size: int

    @classmethod
    def absent(cls) -> "ReadStamp":
        """A stamp recording that the path did not exist when read."""
        return cls(sha256=None, size=0)

    @property
    def is_absent(self) -> bool:
        """True when this stamp recorded a missing file."""
        return self.sha256 is None


class ReadLedger:
    """Thread-safe (run_id, canonical_path) -> ReadStamp map with a per-run cap."""

    def __init__(self, max_paths_per_run: int = DEFAULT_MAX_PATHS_PER_RUN) -> None:
        """Create a ledger.

        Args:
            max_paths_per_run: Entry cap per run_id; oldest evicted on overflow.
        """
        self._max = max(1, int(max_paths_per_run))
        self._lock = threading.Lock()
        # run_id -> OrderedDict[canonical_path, ReadStamp] (insertion-ordered
        # for oldest-first eviction; move_to_end on re-record).
        self._by_run: dict[str, OrderedDict[str, ReadStamp]] = {}

    def _put(self, run_id: str, canonical_path: str, stamp: ReadStamp) -> None:
        with self._lock:
            entries = self._by_run.setdefault(str(run_id), OrderedDict())
            if canonical_path in entries:
                entries.move_to_end(canonical_path)
            entries[canonical_path] = stamp
            while len(entries) > self._max:
                entries.popitem(last=False)

    def record_present(
        self, run_id: str, canonical_path: str, sha256: str, size: int
    ) -> None:
        """Record that ``run_id`` read a present file with this content hash."""
        self._put(run_id, canonical_path, ReadStamp(sha256=sha256, size=int(size)))

    def record_absent(self, run_id: str, canonical_path: str) -> None:
        """Record that ``run_id`` observed the path missing."""
        self._put(run_id, canonical_path, ReadStamp.absent())

    def update_written(
        self, run_id: str, canonical_path: str, sha256: str, size: int
    ) -> None:
        """Record the content ``run_id`` itself just wrote (same as a fresh read)."""
        self.record_present(run_id, canonical_path, sha256, size)

    def stamp_for(self, run_id: str, canonical_path: str) -> ReadStamp | None:
        """Return what ``run_id`` last saw at the path, or None if never read."""
        with self._lock:
            entries = self._by_run.get(str(run_id))
            if entries is None:
                return None
            return entries.get(canonical_path)
