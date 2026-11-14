"""Contended initialization waits for changes without polling native authority."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, control_records
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import repository_participant
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository


@contextmanager
def held_initializer(root):
    entered, release = threading.Event(), threading.Event()
    errors = []

    def run():
        attempt = storage._Acquisition()
        try:
            with attempt.initializing(root, None):
                entered.set()
                assert release.wait(10)
        except BaseException as error:
            errors.append(error)
        finally:
            attempt.close()

    leader = threading.Thread(target=run)
    leader.start()
    try:
        assert entered.wait(5)
        yield release
    finally:
        release.set()
        leader.join(10)
        assert not leader.is_alive()
        assert not errors


@pytest.fixture
def repository_case(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    directory = tmp_path / "store"
    directory.mkdir(mode=0o700)
    repository = EventStateRepository(directory / "state.sqlite")
    try:
        yield root, repository, repository_participant(repository)
    finally:
        repository.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows uses fresh admission")
def test_established_windows_hold_does_not_wait_for_first_use(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"
    selected = tmp_path / "data.sqlite"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))

    with storage.acquire_storage(selected) as original, ThreadPoolExecutor(1) as pool:
        expected = original.execution_scope()
        with held_initializer(root):
            result = pool.submit(storage.acquire_storage, selected)
            try:
                lease = result.result(timeout=5)
            except TimeoutError:
                lease = None
        # Take ownership even when the test fails so the cold behavior drains.
        with result.result(timeout=10) as acquired:
            assert acquired.execution_scope() == expected
            assert lease is acquired, "established Windows admission waited for first use"
        assert original.execution_scope() == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows uses fresh admission")
@pytest.mark.parametrize("publish_pending", [False, True])
def test_established_windows_hold_observes_fresh_records_on_each_side_of_count(
    tmp_path, monkeypatch, publish_pending
):
    root = tmp_path / "bootstrap"
    selected = tmp_path / "data.sqlite"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))

    with storage.acquire_storage(selected) as original:
        hold = storage._holds[original._key]
        original_records = bootstrap._records
        original_registry = bootstrap._registry
        lease_counts = []
        registry_counts = []

        def read_records(path):
            lease_counts.append(hold.count)
            result = original_records(path)
            if publish_pending and hold.count == 2:
                pending, profiles = result
                return [*pending, {"selectors": [str(tmp_path / "config.toml")]}], profiles
            return result

        def read_registry(path, **kwargs):
            registry_counts.append(hold.count)
            return original_registry(path, **kwargs)

        monkeypatch.setattr(bootstrap, "_records", read_records)
        monkeypatch.setattr(bootstrap, "_registry", read_registry)
        monkeypatch.setattr(control_records, "_registry", read_registry)
        if publish_pending:
            with pytest.raises(bootstrap.RecoveryRequired, match="recovery_pending"):
                with storage.acquire_storage(selected):
                    pytest.fail("acquired after recovery became pending")
        else:
            with storage.acquire_storage(selected) as borrowed:
                assert borrowed.execution_scope() == original.execution_scope()
        assert lease_counts == [1, 2]
        assert registry_counts == [1, 2]
        assert hold.count == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows uses fresh admission")
def test_established_windows_hold_refuses_missing_settled_ancestry(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    selected = tmp_path / "data.sqlite"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))

    with storage.acquire_storage(selected):
        ancestry = root / "ancestry-settled.json"
        ancestry.unlink()
        with pytest.raises(bootstrap.RecoveryRequired):
            with storage.acquire_storage(selected):
                pytest.fail("acquired without the settled ancestry receipt")
        assert not ancestry.exists()


@pytest.mark.parametrize("change", ["unchanged", "replaced-parent", "pause"])
def test_nested_waiter_checks_current_scope_on_wake_without_native_polling(
    repository_case, monkeypatch, change
):
    root, repository, participant = repository_case
    ready, begin, waiting, polled = (threading.Event() for _ in range(4))
    finished, admitted, errors, after_wait = [], [], [], []
    release = None
    timed_wake = False
    original_check = storage._Acquisition.check
    original_wait = storage._changed.wait

    def observed_wait(timeout=None):
        nonlocal timed_wake
        if threading.current_thread() is follower:
            waiting.set()
        notified = original_wait(timeout)
        if threading.current_thread() is follower:
            timed_wake = not notified
        return notified

    def observed_check(attempt, path=None):
        if threading.current_thread() is follower and waiting.is_set():
            after_wait.append(path)
            if timed_wake and not release.is_set():
                polled.set()
        return original_check(attempt, path)

    def run():
        try:
            with participant.operation():
                ready.set()
                assert begin.wait(10)
                with storage.acquire_storage(repository.db_path):
                    admitted.append(True)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.append(True)

    follower = threading.Thread(target=run)
    monkeypatch.setattr(storage._changed, "wait", observed_wait)
    monkeypatch.setattr(storage._Acquisition, "check", observed_check)
    if sys.platform == "win32":
        # Exercise the initializer fallback even though the nested operation
        # supplies a live lease. The established-hold route has its own control.
        original_serving = storage._hold_serving
        monkeypatch.setattr(
            storage,
            "_hold_serving",
            lambda hold: not (
                threading.current_thread() is follower and ready.is_set()
            )
            and original_serving(hold),
        )
    follower.start()
    pause = None
    try:
        assert ready.wait(10)
        with held_initializer(root) as release:
            begin.set()
            assert waiting.wait(5)
            assert not polled.wait(
                0.15
            ), "waiting rechecks native paths without a change"
            assert not admitted and not finished
            if change == "replaced-parent":
                directory = repository.db_path.parent
                directory.rename(directory.with_name("original-store"))
                directory.mkdir(mode=0o700)
            elif change == "pause":
                pause = storage._begin_local_pause()
                assert not pause.drain(0)
            release.set()
        follower.join(10)
        assert not follower.is_alive()
        assert after_wait, "waking must revalidate before acquiring authority"
        if change == "replaced-parent":
            assert not admitted
            assert len(errors) == 1
            assert isinstance(errors[0], bootstrap.RecoveryRequired)
            assert str(errors[0]) == "operation_path_outside_scope"
        else:
            assert admitted == [True]
            assert not errors
    finally:
        begin.set()
        if release is not None:
            release.set()
        follower.join(10)
        if pause is not None:
            pause.resume()


def test_pause_wakes_and_retires_waiter_before_initializer_finishes(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"
    waiting, done = threading.Event(), threading.Event()
    original_wait = storage._changed.wait
    errors, admitted, attempts = [], [], []

    def observed_wait(timeout=None):
        if threading.current_thread() is follower:
            waiting.set()
        return original_wait(timeout)

    def run():
        attempt = storage._Acquisition()
        attempts.append(attempt)
        try:
            with attempt.initializing(root, None):
                admitted.append(True)
        except BaseException as error:
            errors.append(error)
        finally:
            attempt.close()
            done.set()

    follower = threading.Thread(target=run)
    monkeypatch.setattr(storage._changed, "wait", observed_wait)
    pause = None
    try:
        with held_initializer(root) as release:
            follower.start()
            assert waiting.wait(5)
            pause = storage._begin_local_pause()
            assert done.wait(5), "pause must notify a blocked initializer waiter"
            assert not release.is_set()
            assert not admitted
            assert attempts[0] not in storage._pending_acquisitions
            assert len(errors) == 1
            assert isinstance(errors[0], bootstrap.RecoveryRequired)
            assert str(errors[0]) == "storage_locally_paused"
    finally:
        if follower.ident is not None:
            follower.join(10)
        if pause is not None:
            pause.resume()


def test_first_acquisition_waits_before_creating_authority(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    selected = tmp_path / "data.sqlite"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    waiting = threading.Event()
    original_wait = storage._changed.wait

    def observed_wait(timeout=None):
        if threading.current_thread().name.startswith("first-acquisition"):
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(storage._changed, "wait", observed_wait)
    with ThreadPoolExecutor(1, thread_name_prefix="first-acquisition") as pool:
        with held_initializer(root):
            result = pool.submit(storage.acquire_storage, selected)
            parked = waiting.wait(5)
            still_absent = not root.exists()
        with result.result(timeout=10) as lease:
            assert lease.execution_scope()
        assert parked and still_absent
