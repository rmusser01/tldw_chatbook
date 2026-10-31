"""tqdm's write lock and download progress for in-app model downloads.

TASK-33621.21. The first Hugging Face download inside the app built tqdm's
lock; on macOS that spawned the resource tracker with Textual's
``fileno() == -1`` stderr, raised ``fds_to_keep``, and left tqdm's class lock
held so every later download blocked. Each guard test first shows the
fixture reproduces that (an unguarded construction raises), then that the
guard prevents it.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from Tests.Chat import model_download_support as _support
from tldw_chatbook.Utils import model_download_guard as guard

tqdm_std = _support.tqdm_std
#: Shared fixture (a first launch: tqdm's lock unbuilt, no resource tracker).
first_download_in_process = _support.first_download_in_process


class _FilenoLessStderr:
    """What Textual installs as ``sys.stderr`` while an app runs."""

    def fileno(self) -> int:
        return -1

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def _build_tqdm_on_a_thread() -> BaseException | None:
    """Construct a disabled bar (as huggingface_hub does) on another thread.

    Returns:
        The exception it raised, None when it built; a construction still
        blocked after 5 s is reported as a TimeoutError.
    """
    box: dict[str, BaseException] = {}

    def _build() -> None:
        try:
            with contextlib.redirect_stderr(_FilenoLessStderr()):
                tqdm_std.tqdm(total=1, disable=True).close()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            box["error"] = exc

    worker = threading.Thread(target=_build, daemon=True)
    worker.start()
    worker.join(timeout=5)
    if worker.is_alive():
        return TimeoutError("tqdm construction blocked")
    return box.get("error")


def _class_lock_is_free() -> bool:
    lock = tqdm_std.TqdmDefaultWriteLock.th_lock
    if not lock.acquire(timeout=1):
        return False
    lock.release()
    return True


def _reset_to_first_launch() -> None:
    """Undo what one construction did, inside the fixture's first launch."""
    lock_type = tqdm_std.TqdmDefaultWriteLock
    lock_type.th_lock = tqdm_std.TRLock()
    if "mp_lock" in lock_type.__dict__:
        del lock_type.mp_lock


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX spawn path only")
def test_without_the_guard_the_first_lock_build_dies_and_wedges(
    first_download_in_process,
):
    """Negative control: the fixture really reproduces the live defect."""
    error = _build_tqdm_on_a_thread()

    assert isinstance(error, ValueError), error
    assert "fds_to_keep" in str(error)
    assert not _class_lock_is_free(), "the failed build should leave the lock held"


def test_the_guard_lets_tqdm_build_its_lock_under_a_fileno_less_stderr(
    first_download_in_process,
):
    assert guard.allow_tqdm_without_stderr_descriptor() is True

    assert _build_tqdm_on_a_thread() is None
    assert _build_tqdm_on_a_thread() is None, "a second download must not block"
    assert _class_lock_is_free()
    assert tqdm_std.TqdmDefaultWriteLock.mp_lock is None


def test_the_guard_is_a_no_op_once_tqdm_has_decided_its_lock(
    first_download_in_process,
):
    lock_type = tqdm_std.TqdmDefaultWriteLock
    sentinel = object()
    lock_type.mp_lock = sentinel
    class_lock = lock_type.th_lock

    assert guard.allow_tqdm_without_stderr_descriptor() is False
    assert lock_type.mp_lock is sentinel
    assert lock_type.th_lock is class_lock


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX spawn path only")
def test_the_guard_repairs_a_lock_a_dead_construction_left_held(
    first_download_in_process, monkeypatch
):
    """An earlier unguarded download (any feature) wedged the class lock."""
    assert isinstance(_build_tqdm_on_a_thread(), ValueError)
    assert not _class_lock_is_free()
    monkeypatch.setattr(guard, "WEDGED_LOCK_WAIT_SECONDS", 0.1)

    assert guard.allow_tqdm_without_stderr_descriptor() is True

    assert _class_lock_is_free()
    assert _build_tqdm_on_a_thread() is None


def test_the_guard_repairs_a_lock_owned_by_a_reused_thread_id(
    first_download_in_process,
):
    """A dead owner whose thread id this thread now carries looks like ours.

    Seen in the TASK-33621.21 reproduction: the next download ran on a thread
    with the dead one's id, so the reentrant lock let *it* through and wedged
    everyone else. The guard never holds the lock itself, so any ownership it
    sees on entry is stale.
    """
    lock_type = tqdm_std.TqdmDefaultWriteLock
    stale = lock_type.th_lock
    stale.acquire()  # stands in for the dead thread with this thread's id
    try:
        assert guard.allow_tqdm_without_stderr_descriptor() is True
        assert lock_type.th_lock is not stale
        assert _build_tqdm_on_a_thread() is None
    finally:
        stale.release()
        _reset_to_first_launch()


def _write(path: Path, size: int, *, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_progress_counts_in_flight_and_new_blobs_but_not_the_existing_cache(
    tmp_path: Path,
):
    started = time.time()
    blobs = tmp_path / "models--org--speech" / "blobs"
    _write(blobs / "cached-before", 9_000_000, mtime=started - 3600)
    _write(blobs / "abc.incomplete", 0)
    progress = guard.HubDownloadProgress(root=tmp_path, started_at=started)

    assert progress.downloaded_bytes() == 0

    _write(blobs / "abc.incomplete", 2_500_000)
    assert progress.downloaded_bytes() == 2_500_000

    # The finished file is renamed to its blob; it keeps counting.
    (blobs / "abc.incomplete").rename(blobs / "abc")
    _write(tmp_path / "models--org--other" / "blobs" / "def.incomplete", 500_000)
    assert progress.downloaded_bytes() == 3_000_000


def test_progress_without_a_cache_reports_nothing(tmp_path: Path):
    assert guard.HubDownloadProgress(root=None).downloaded_bytes() == 0
    missing = guard.HubDownloadProgress(root=tmp_path / "not-created-yet")
    assert missing.downloaded_bytes() == 0


@pytest.mark.parametrize(
    ("num_bytes", "shown"),
    [
        (0, "0 KB"),
        (412_000, "412 KB"),
        (38_400_000, "38 MB"),
        (999_999_999, "1.0 GB"),
        (1_240_000_000, "1.2 GB"),
    ],
)
def test_download_size_reads_like_a_download(num_bytes: int, shown: str):
    assert guard.format_download_size(num_bytes) == shown


_XET_PROBE = (
    "import tldw_chatbook, huggingface_hub.constants as c; "
    "print(c.HF_HUB_DISABLE_XET)"
)


@pytest.mark.parametrize(
    ("configured", "disabled"),
    [(None, "True"), ("0", "False")],
    ids=["default", "user-opted-in"],
)
def test_hub_downloads_stream_to_disk_unless_the_user_opts_into_xet(
    configured, disabled
):
    """In-app model downloads write progressively, so progress can be shown.

    Live, TASK-33621.21: with hf_xet the 2.4 GB parakeet download kept its
    cache file at 0 bytes while the app's memory grew from 1.1 to 1.7 GB in
    70 s, so no progress could be measured and a cancel lost everything. The
    hub's plain HTTP path streams into ``.incomplete`` and resumes. A user who
    sets ``HF_HUB_DISABLE_XET`` keeps their choice.
    """
    import subprocess

    pytest.importorskip("huggingface_hub")
    env = dict(os.environ)
    env.pop("HF_HUB_DISABLE_XET", None)
    if configured is not None:
        env["HF_HUB_DISABLE_XET"] = configured
    project_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [sys.executable, "-c", _XET_PROBE],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().splitlines()[-1] == disabled
