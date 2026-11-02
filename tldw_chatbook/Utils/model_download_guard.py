"""Speech-model downloads inside the running app: tqdm's lock and visible progress.

TASK-33621.21. With no speech model cached, Console dictation sat on
"Preparing speech model…" forever. The first Hugging Face fetch built tqdm's
process-wide write lock, and building it creates a ``multiprocessing.RLock``.
On macOS that starts CPython's resource tracker, which passes
``sys.stderr.fileno()`` to the child. Inside Textual, ``sys.stderr`` is a
capture whose ``fileno()`` returns ``-1``, so the spawn raised
``ValueError: bad value(s) in fds_to_keep``. tqdm catches only
``ImportError``/``OSError`` there, so the class lock it had just taken
(``TqdmDefaultWriteLock.th_lock``) was never released. The next download on
any other thread blocked on that lock for good.

Hugging Face progress bars are already disabled at package import
(``HF_HUB_DISABLE_PROGRESS_BARS``), but a disabled bar still builds the lock
in ``tqdm.__new__``, so that setting never helped.

This module holds the two pieces the Console needs around a model download:

* `allow_tqdm_without_stderr_descriptor` gives tqdm the thread-only lock it
  already falls back to when a multiprocessing lock cannot be built, before
  anything downloads. Nothing in a TUI shares progress bars across processes.
* `HubDownloadProgress` counts only explicitly attributed HTTP artifacts.
  Current legacy speech adapters cannot supply that attribution and stay opaque.

Stdlib-only at import; tqdm and huggingface_hub are imported inside the calls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

__all__ = [
    "HubDownloadProgress",
    "allow_tqdm_without_stderr_descriptor",
    "format_download_size",
]

#: How long the guard waits for tqdm's class lock before treating it as held
#: by a construction that died. The guard waits only while tqdm's
#: multiprocessing lock is still undecided, and until then tqdm holds the
#: class lock just while it builds its write lock (microseconds). Once a write
#: lock exists, the class lock also guards every bar refresh and write, so a
#: wait that long proves nothing there: do not reuse this heuristic elsewhere.
WEDGED_LOCK_WAIT_SECONDS = 1.0

_GUARD_LOCK = threading.Lock()


def allow_tqdm_without_stderr_descriptor() -> bool:
    """Make tqdm's process-wide write lock buildable while Textual owns stderr.

    Sets ``TqdmDefaultWriteLock.mp_lock = None`` before tqdm's first lock is
    built, which is exactly the state tqdm itself falls back to when a
    multiprocessing lock cannot be created; tqdm then uses its thread lock
    only. Idempotent and cheap once tqdm's lock exists.

    It also repairs the wedge an earlier unguarded construction may have left:
    when the multiprocessing lock was never set and the class lock cannot be
    taken (or is already "owned" by this thread, which never holds it here,
    so the owner is a dead thread whose id was reused), the class lock is
    replaced with a fresh one.

    Returns:
        True when this call changed tqdm's state, False when there was nothing
        to do (tqdm is not installed, or its lock was already decided).
    """
    try:
        from tqdm import std as tqdm_std
    except ImportError:
        return False
    lock_type = getattr(tqdm_std, "TqdmDefaultWriteLock", None)
    if lock_type is None:
        return False
    with _GUARD_LOCK:
        if hasattr(lock_type, "mp_lock"):
            return False
        class_lock = getattr(lock_type, "th_lock", None)
        if class_lock is None:
            lock_type.mp_lock = None
            return True
        is_owned = getattr(class_lock, "_is_owned", None)
        stale_owner = bool(is_owned()) if callable(is_owned) else False
        if not stale_owner and class_lock.acquire(timeout=WEDGED_LOCK_WAIT_SECONDS):
            try:
                if not hasattr(lock_type, "mp_lock"):
                    lock_type.mp_lock = None
            finally:
                class_lock.release()
            return True
        logger.warning(
            "tqdm's class lock is held by a lock construction that never "
            "finished; replacing it so model downloads can proceed"
        )
        lock_factory = getattr(tqdm_std, "TRLock", threading.RLock)
        lock_type.th_lock = lock_factory()
        lock_type.mp_lock = None
        return True


def format_download_size(num_bytes: int) -> str:
    """Render a byte count the way the voice chip shows it.

    Args:
        num_bytes: Bytes downloaded so far.

    Returns:
        "412 KB", "38 MB" or "1.2 GB" (decimal units, like download sites).
    """
    if round(num_bytes / 1_000_000) >= 1000:
        return f"{num_bytes / 1_000_000_000:.1f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.0f} MB"
    return f"{max(num_bytes, 0) // 1000} KB"


@dataclass
class HubDownloadProgress:
    """Observe explicitly owned plain-HTTP Hub artifacts, never a cache scan.

    The caller must supply the exact ``.incomplete`` paths belonging to its
    download and establish plain-HTTP transport. A repo id, mtime, Xet cache,
    or start-only provider callback does not establish that contract. Current
    Console legacy adapters cannot supply it and therefore use no observer.
    The loopback transport control exercises this seam with known ETags.

    Positive deltas are accumulated per artifact. Completion renames retain
    the same identity; truncation/deletion of any other file cannot erase
    activity. ``active`` is true only while an owned incomplete file exists.
    Absence or an unreadable artifact is unknown/loading, never a stall.
    """

    artifacts: tuple[Path, ...] = ()
    active: bool = field(default=False, init=False)
    _sizes: dict[Path, int] = field(default_factory=dict, init=False)
    _total: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.artifacts = tuple(dict.fromkeys(self.artifacts))
        if any(not p.name.endswith(".incomplete") for p in self.artifacts):
            raise ValueError("exact incomplete artifact paths required")
        for path in self.artifacts:
            size, _ = self._artifact_size(path)
            self._sizes[path] = size or 0

    def downloaded_bytes(self) -> int:
        """Return cumulative positive byte deltas and refresh active status."""
        self.active = False
        for path in self.artifacts:
            size, active = self._artifact_size(path)
            self.active |= active
            if size is None:
                continue  # Unknown is not truncation; retain rename credit.
            self._total += max(0, size - self._sizes[path])
            self._sizes[path] = size
        return self._total

    @staticmethod
    def _artifact_size(path: Path) -> tuple[int | None, bool]:
        for candidate, active in ((path, True), (path.with_suffix(""), False)):
            try:
                if not candidate.is_symlink():
                    return candidate.stat().st_size, active
            except OSError:
                pass
        return None, False
