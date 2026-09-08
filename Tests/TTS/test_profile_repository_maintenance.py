"""Actual serialized TTS repository maintenance lifetime regressions."""

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


@pytest.mark.asyncio
async def test_paused_first_open_allocates_no_executor_or_profile_files(tmp_path):
    selected = tmp_path / "profiles.sqlite"
    repo = TTSProfileRepository(selected)
    original_entries = set(tmp_path.iterdir())
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(ProfileRepositoryError):
            await repo.open()
        assert repo._executor is None
        assert set(tmp_path.iterdir()) == original_entries
    finally:
        pause.resume()
        await repo.close()


@pytest.mark.asyncio
async def test_paused_existing_repository_refuses_new_mutation(tmp_path):
    from Tests.TTS.test_profile_repository_lifecycle import _draft

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(ProfileRepositoryError):
            await repo.create_profile(_draft("must not commit"))
    finally:
        pause.resume()
        await repo.close()


@pytest.mark.asyncio
async def test_reversible_pause_retires_native_on_worker_and_reopens_same_source(
    tmp_path,
):
    import asyncio
    import sqlite3
    import time
    from Tests.TTS.test_profile_repository_lifecycle import _draft
    from tldw_chatbook.TTS.profile_types import ProfileRepositoryState

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    created = await repo.create_profile(_draft("kept"))
    connection = repo._connection
    executor = repo._executor
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    try:
        assert await repo._maintenance_drain(time.monotonic() + 3)
        assert repo.state is ProfileRepositoryState.CLOSED
        assert not repo.terminal
        assert repo.generation > created.generation
        assert repo._executor is executor
        assert repo._connection is None and repo._lease is None

        def prove_native_closed():
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
            assert connection.parent_fd == -1 and connection.file_fd == -1
            assert not connection.sidecar_fds

        await asyncio.wrap_future(executor.submit(prove_native_closed))
        with pytest.raises(ProfileRepositoryError):
            await repo._maintenance_resume()
    finally:
        pause.resume()
    try:
        await repo._maintenance_resume()
        assert (
            await repo.get_profile(created.value.profile_id)
        ).value.display_name == "kept"
        assert repo._executor is executor
    finally:
        await repo.close()
    with pytest.raises(ProfileRepositoryError) as caught:
        await repo.open()
    assert caught.value.code == "terminal"


@pytest.mark.asyncio
async def test_running_cancelled_write_and_queued_write_finish_before_retirement(
    tmp_path,
):
    import asyncio
    import threading
    import time
    from datetime import UTC, datetime
    from Tests.TTS.test_profile_repository_lifecycle import _draft

    entered, finish = threading.Event(), threading.Event()

    def clock():
        entered.set()
        assert finish.wait(5)
        return datetime.now(UTC)

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite", _clock=clock)
    await repo.open()
    running = asyncio.create_task(repo.create_profile(_draft("running")))
    assert await asyncio.to_thread(entered.wait, 3)
    queued = asyncio.create_task(repo.create_profile(_draft("queued")))
    await asyncio.sleep(0)
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    try:
        running.cancel()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not await repo._maintenance_drain(time.monotonic() + 0.02)
        assert repo._connection is not None and repo._lease.acquired
        with pytest.raises(ProfileRepositoryError):
            await repo.create_profile(_draft("late"))
        finish.set()
        assert (await queued).value.display_name == "queued"
        assert await repo._maintenance_drain(time.monotonic() + 3)
        assert not repo._pending_futures and not repo._publication_completions
    finally:
        finish.set()
        pause.resume()
    try:
        await repo._maintenance_resume()
        assert {
            p.display_name for p in (await repo.list_profiles()).value.profiles
        } == {"running", "queued"}
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_owner_result_bookkeeping_blocks_drain_without_waiting_caller_task(
    tmp_path,
):
    import asyncio
    import time
    from Tests.TTS.test_profile_repository_lifecycle import _draft

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    # Split the existing private admission/publication pair on this same Task.
    # Waiting for the caller Task here would deadlock its own maintenance call.
    admission = repo._admit_operation(
        lambda connection: connection.execute("SELECT 1").fetchone()[0]
    )
    assert await asyncio.wrap_future(admission.future) == 1
    repo._maintenance_close_admission()
    try:
        assert not await repo._maintenance_drain(time.monotonic() + 0.02)
        assert repo._connection is not None
        assert (await repo._publish_operation(admission)).value == 1
        assert await repo._maintenance_drain(time.monotonic() + 3)
        await repo._maintenance_resume()
        await repo.create_profile(_draft("after"))
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_cancelled_maintenance_waiter_keeps_actual_cleanup_owned(
    tmp_path, monkeypatch
):
    import asyncio
    import threading
    import time
    import tldw_chatbook.TTS.profile_schema as schema

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    connection = repo._connection
    entered, finish = threading.Event(), threading.Event()
    original = schema._ExactCurrentProfileConnection.close

    def close(native):
        if native is connection:
            entered.set()
            assert finish.wait(5)
        return original(native)

    monkeypatch.setattr(schema._ExactCurrentProfileConnection, "close", close)
    repo._maintenance_close_admission()
    waiter = asyncio.create_task(repo._maintenance_drain(time.monotonic() + 3))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        waiter.cancel()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not repo._maintenance_completion.done()
        assert repo._connection is connection and repo._lease.acquired
        finish.set()
        assert await repo._maintenance_drain(time.monotonic() + 3)
    finally:
        finish.set()
        await repo.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["configured", "inode", "terminal", "initially_closed"]
)
async def test_resume_never_opens_changed_terminal_or_previously_closed_source(
    tmp_path, change
):
    import os
    import time

    if (
        change in {"configured", "inode"}
        and os.environ.get("TASK10_TTS_RETARGET_CHILD") != change
    ):
        _run_private_child(
            tmp_path,
            """
import asyncio, os, sys
from pathlib import Path
from Tests.TTS.test_profile_repository_maintenance import test_resume_never_opens_changed_terminal_or_previously_closed_source
os.environ['TASK10_TTS_RETARGET_CHILD'] = sys.argv[2]
asyncio.run(test_resume_never_opens_changed_terminal_or_previously_closed_source(Path(sys.argv[1]), sys.argv[2]))
""",
            change,
        )
        return
    from tldw_chatbook.TTS.profile_types import ProfileRepositoryState

    selected = tmp_path / "profiles.sqlite"
    repo = TTSProfileRepository(selected)
    if change != "initially_closed":
        await repo.open()
    repo._maintenance_close_admission()
    assert await repo._maintenance_drain(time.monotonic() + 3)
    try:
        if change == "initially_closed":
            await repo._maintenance_resume()
            assert repo.state is ProfileRepositoryState.CLOSED
            assert repo._executor is None
            assert not selected.exists()
            return
        if change == "configured":
            repo._database_path = tmp_path / "other.sqlite"
        elif change == "inode":
            selected.rename(tmp_path / "original.sqlite")
            selected.write_bytes(b"foreign sentinel")
        else:
            await repo.close()
        with pytest.raises(ProfileRepositoryError):
            await repo._maintenance_resume()
        assert repo._connection is None
        if change == "inode":
            assert selected.read_bytes() == b"foreign sentinel"
    finally:
        await repo.close()


async def _failed_native_cleanup_child(root, after):
    import time
    import tldw_chatbook.TTS.profile_schema as schema

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    original = schema._ExactCurrentProfileConnection.close
    connection = repo._connection

    def fail_close(native):
        if native is connection:
            if after:
                original(native)
            raise OSError("injected uncertain native close")
        return original(native)

    schema._ExactCurrentProfileConnection.close = fail_close
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    assert not await repo._maintenance_drain(time.monotonic() + 3)
    assert repo._connection is connection and repo._lease.acquired
    assert repo._executor is not None
    assert not pause.drain(time.monotonic() + 0.03)
    # Unrelated public retries cannot erase this maintenance uncertainty.
    assert not await repo._maintenance_drain(time.monotonic() + 0.03)
    pause.resume()
    with pytest.raises(ProfileRepositoryError):
        await repo._maintenance_resume()
    # Retained unsafe ownership intentionally lives until this private child exits.


def _run_private_child(tmp_path, code, *arguments):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.toml"
    config.write_text('[general]\nuser_folder = "default_user"\n')
    config.chmod(0o600)
    env.update(
        HOME=str(home), TLDW_CONFIG_PATH=str(config), PYTHONDONTWRITEBYTECODE="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("after", [False, True])
def test_uncertain_native_close_remains_global_drain_blocker(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_repository_maintenance import _failed_native_cleanup_child
asyncio.run(_failed_native_cleanup_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _native_observer_child(root, failure):
    import asyncio
    import json
    import select
    import subprocess
    import sys
    import time
    import tldw_chatbook.TTS.profile_schema as schema

    # This diagnostic child deliberately owns a known minimal startup: only this
    # repository and observer. It is not evidence of actual app startup drainage.
    storage.admit_startup()
    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    hold = next(iter(storage._holds.values()))
    script = """
import json, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print('waiting', flush=True)
with Admission(Path(sys.argv[1])).maintenance(tuple(json.loads(sys.argv[2])), 8):
    print('entered', flush=True)
    input()
"""
    observer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(hold.authority.control_root),
            json.dumps(hold.names),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert observer.stdout.readline().strip() == "waiting"
        deadline = time.monotonic() + 3
        while not hold.authority.pause_requested(hold.names):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        assert not select.select([observer.stdout], [], [], 0.03)[0]
        if failure:
            original = schema._ExactCurrentProfileConnection.close
            connection = repo._connection

            def uncertain_close(native):
                original(native)
                if native is connection:
                    raise OSError("synthetic post-close uncertainty")

            schema._ExactCurrentProfileConnection.close = uncertain_close
        repo._maintenance_close_admission()
        pause = storage._begin_local_pause()
        assert await repo._maintenance_drain(time.monotonic() + 3) is (not failure)
        assert not select.select([observer.stdout], [], [], 0.03)[0]
        if failure:
            assert not pause.drain(time.monotonic() + 0.03)
            assert storage._startups
        else:
            assert pause.drain(time.monotonic() + 1)
            # All diagnostic-child source work has positively retired. Only here
            # release its known startup to demonstrate independent native entry.
            storage._shutdown()
            assert select.select([observer.stdout], [], [], 3)[0]
            assert observer.stdout.readline().strip() == "entered"
        pause.resume()
    finally:
        if observer.poll() is None:
            if failure:
                observer.terminate()
            else:
                observer.stdin.write("\n")
                observer.stdin.flush()
        observer.wait(timeout=3)
        observer.stdin.close()
        observer.stdout.close()
        observer.stderr.close()
        if not failure:
            await repo.close()


@pytest.mark.parametrize("failure", [False, True])
def test_private_startup_and_native_maintainer_observe_actual_source_retirement(
    tmp_path, failure
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_repository_maintenance import _native_observer_child
asyncio.run(_native_observer_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(failure),
    )


@pytest.mark.asyncio
async def test_second_real_repository_keeps_profile_native_lock_until_its_own_drain(
    tmp_path,
):
    import asyncio
    import subprocess
    import sys
    import time

    selected = tmp_path / "profiles.sqlite"
    first, second = TTSProfileRepository(selected), TTSProfileRepository(selected)
    await first.open()
    await second.open()
    script = """
import sys
from pathlib import Path
from tldw_chatbook.TTS.profile_store_lock import ProfileStoreLease, ProfileStoreLockMode
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
try:
    lease = ProfileStoreLease(Path(sys.argv[1]), ProfileStoreLockMode.EXCLUSIVE, timeout_seconds=.1).acquire()
except ProfileRepositoryError:
    print('blocked')
else:
    print('entered')
    lease.release()
"""

    def observe():
        result = subprocess.run(
            [sys.executable, "-c", script, str(selected)],
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    try:
        first._maintenance_close_admission()
        assert await first._maintenance_drain(time.monotonic() + 3)
        assert await asyncio.to_thread(observe) == "blocked"
        second._maintenance_close_admission()
        assert await second._maintenance_drain(time.monotonic() + 3)
        assert await asyncio.to_thread(observe) == "entered"
        await first._maintenance_resume()
        assert await asyncio.to_thread(observe) == "blocked"
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_pause_while_first_open_waits_for_lifecycle_lock_stays_typed_and_pure(
    tmp_path,
):
    import asyncio

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    lock = repo._bind_or_check_loop()
    await lock.acquire()
    opening = asyncio.create_task(repo.open())
    await asyncio.sleep(0)
    pause = storage._begin_local_pause()
    lock.release()
    try:
        with pytest.raises(ProfileRepositoryError):
            await opening
        assert repo._executor is None
        assert not (tmp_path / "profiles.sqlite.lock").exists()
    finally:
        pause.resume()
        await repo.close()


@pytest.mark.asyncio
async def test_resume_revalidates_source_on_worker_after_owner_loop_dispatch(
    tmp_path, monkeypatch
):
    import os
    import time

    if os.environ.get("TASK10_TTS_SWAP_CHILD") != "1":
        _run_private_child(
            tmp_path,
            """
import asyncio, os, sys, pytest
from pathlib import Path
from Tests.TTS.test_profile_repository_maintenance import test_resume_revalidates_source_on_worker_after_owner_loop_dispatch
os.environ['TASK10_TTS_SWAP_CHILD'] = '1'
with pytest.MonkeyPatch.context() as monkeypatch:
    asyncio.run(test_resume_revalidates_source_on_worker_after_owner_loop_dispatch(Path(sys.argv[1]), monkeypatch))
""",
        )
        return

    selected = tmp_path / "profiles.sqlite"
    repo = TTSProfileRepository(selected)
    await repo.open()
    repo._maintenance_close_admission()
    assert await repo._maintenance_drain(time.monotonic() + 3)
    other = TTSProfileRepository(tmp_path / "other.sqlite")
    await other.open()
    await other.close()
    original = repo._worker_open

    def replace_before_worker():
        selected.rename(tmp_path / "prior.sqlite")
        (tmp_path / "other.sqlite").rename(selected)
        original()

    monkeypatch.setattr(repo, "_worker_open", replace_before_worker)
    try:
        with pytest.raises(ProfileRepositoryError):
            await repo._maintenance_resume()
        assert repo._connection is None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_cancelled_resume_settles_owned_open_and_reopens_admission(
    tmp_path, monkeypatch
):
    import asyncio
    import threading
    import time
    from Tests.TTS.test_profile_repository_lifecycle import _draft

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    repo._maintenance_close_admission()
    assert await repo._maintenance_drain(time.monotonic() + 3)
    entered, finish = threading.Event(), threading.Event()
    original = repo._worker_open

    def opening():
        entered.set()
        assert finish.wait(5)
        original()

    monkeypatch.setattr(repo, "_worker_open", opening)
    resume = asyncio.create_task(repo._maintenance_resume())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not await repo._maintenance_drain(time.monotonic() + 0.01)
        resume.cancel()
        resume.cancel()
        await asyncio.sleep(0)
        assert not resume.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await resume
        assert (
            await repo.create_profile(_draft("resumed"))
        ).value.display_name == "resumed"
    finally:
        finish.set()
        await asyncio.gather(resume, return_exceptions=True)
        await repo.close()


async def _failed_resume_child(root):
    import time
    import tldw_chatbook.TTS.profile_schema as schema

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    repo._maintenance_close_admission()
    assert await repo._maintenance_drain(time.monotonic() + 3)
    original_check = repo._worker_check_maintenance_resume_source
    checks = []

    def fail_after_native_allocation():
        original_check()
        checks.append(True)
        if len(checks) == 2:
            raise ProfileRepositoryError("operation_failed")

    original_close = schema._ExactCurrentProfileConnection.close

    def fail_after_native_close(native):
        original_close(native)
        raise OSError("synthetic resume cleanup uncertainty")

    repo._worker_check_maintenance_resume_source = fail_after_native_allocation
    schema._ExactCurrentProfileConnection.close = fail_after_native_close
    with pytest.raises(ProfileRepositoryError):
        await repo._maintenance_resume()
    assert repo._connection is not None and repo._lease.acquired
    pause = storage._begin_local_pause()
    assert not pause.drain(time.monotonic() + 0.03)
    assert not await repo._maintenance_drain(time.monotonic() + 0.03)
    pause.resume()


def test_failed_resume_native_retirement_remains_a_local_drain_blocker(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_repository_maintenance import _failed_resume_child
asyncio.run(_failed_resume_child(Path(sys.argv[1])))
""",
    )


@pytest.mark.asyncio
async def test_open_already_running_settles_before_drain_without_auto_reopening(
    tmp_path, monkeypatch
):
    import asyncio
    import threading
    import time
    from tldw_chatbook.TTS.profile_types import ProfileRepositoryState

    repo = TTSProfileRepository(tmp_path / "profiles.sqlite")
    entered, finish = threading.Event(), threading.Event()
    original = repo._worker_open

    def opening():
        original()
        entered.set()
        assert finish.wait(5)

    monkeypatch.setattr(repo, "_worker_open", opening)
    opening_task = asyncio.create_task(repo.open())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        repo._maintenance_close_admission()
        assert not await repo._maintenance_drain(time.monotonic() + 0.02)
        assert repo._connection is not None
        finish.set()
        await opening_task
        assert await repo._maintenance_drain(time.monotonic() + 3)
        await repo._maintenance_resume()
        assert repo.state is ProfileRepositoryState.CLOSED
        assert repo._connection is None and not repo.terminal
    finally:
        finish.set()
        await asyncio.gather(opening_task, return_exceptions=True)
        await repo.close()
