"""Contended initialization waits for changes without polling native authority."""

from contextlib import contextmanager
import threading

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import _repository_participant
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
        yield root, repository, _repository_participant(repository)
    finally:
        repository.close()


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
