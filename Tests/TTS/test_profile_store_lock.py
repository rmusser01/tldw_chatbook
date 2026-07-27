from __future__ import annotations

import multiprocessing
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import BinaryIO

import portalocker
import pytest

import tldw_chatbook.TTS.profile_store_lock as lock_module
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_store_lock import ProfileStoreLease, ProfileStoreLockMode


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


def _spawned_lease_holder(database_path: str, connection: Connection) -> None:
    lease = ProfileStoreLease(
        Path(database_path),
        ProfileStoreLockMode.SHARED,
        timeout_seconds=1.0,
        check_interval_seconds=0.01,
    )
    try:
        lease.acquire()
        connection.send(("ready", None))
        command = connection.recv()
        if command != "release":
            raise RuntimeError("unexpected command")
        lease.release()
        connection.send(("released", None))
    except BaseException as error:
        connection.send(("error", type(error).__name__))
        raise
    finally:
        lease.release()
        connection.close()


def _assert_safe_repository_error(
    error: ProfileRepositoryError,
    code: str,
    *secrets: str,
) -> None:
    assert error.code == code
    visible = " ".join(
        (
            str(error),
            repr(error),
            *(str(note) for note in getattr(error, "__notes__", ())),
        )
    )
    assert error.__cause__ is None
    assert error.__context__ is None
    for secret in secrets:
        assert secret not in visible


def test_constructor_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    entries_before = set(tmp_path.iterdir())

    lease = ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED)

    assert lease.acquired is False
    assert lease.lock_path == database_path.resolve().with_name(
        f"{database_path.name}.lock"
    )
    assert set(tmp_path.iterdir()) == entries_before


def test_lock_identity_is_stable_adjacent_and_path_specific(tmp_path: Path) -> None:
    first_path = tmp_path / "one.sqlite3"
    second_path = tmp_path / "two.sqlite3"

    first = ProfileStoreLease(first_path, ProfileStoreLockMode.SHARED)
    alias = ProfileStoreLease(
        tmp_path / "nested" / ".." / first_path.name,
        ProfileStoreLockMode.EXCLUSIVE,
    )
    second = ProfileStoreLease(second_path, ProfileStoreLockMode.SHARED)

    assert first.lock_path == alias.lock_path
    assert first.lock_path == tmp_path.resolve() / "one.sqlite3.lock"
    assert second.lock_path == tmp_path.resolve() / "two.sqlite3.lock"
    assert first.lock_path != second.lock_path


def test_symlink_alias_uses_the_same_lock_identity(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")

    real = ProfileStoreLease(
        real_parent / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )
    alias = ProfileStoreLease(
        alias_parent / "profiles.sqlite3",
        ProfileStoreLockMode.EXCLUSIVE,
    )

    assert real.lock_path == alias.lock_path


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        pytest.param("database_path", "profiles.sqlite3", id="string-path"),
        pytest.param("mode", "shared", id="string-mode"),
        pytest.param("mode", 1, id="integer-mode"),
        pytest.param("timeout_seconds", True, id="boolean-timeout"),
        pytest.param(
            "timeout_seconds",
            _IntSubclass(1),
            id="integer-subclass-timeout",
        ),
        pytest.param("timeout_seconds", 0, id="zero-timeout"),
        pytest.param("timeout_seconds", -1.0, id="negative-timeout"),
        pytest.param("timeout_seconds", float("nan"), id="nan-timeout"),
        pytest.param("timeout_seconds", float("inf"), id="infinite-timeout"),
        pytest.param("check_interval_seconds", False, id="boolean-interval"),
        pytest.param(
            "check_interval_seconds",
            _FloatSubclass(1.0),
            id="float-subclass-interval",
        ),
        pytest.param("check_interval_seconds", 0.0, id="zero-interval"),
        pytest.param("check_interval_seconds", -1, id="negative-interval"),
        pytest.param("check_interval_seconds", float("nan"), id="nan-interval"),
        pytest.param(
            "check_interval_seconds",
            float("-inf"),
            id="infinite-interval",
        ),
    ],
)
def test_invalid_constructor_input_fails_safely_without_path_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
) -> None:
    database_path: object = tmp_path / "private-profile-name.sqlite3"
    mode: object = ProfileStoreLockMode.SHARED
    kwargs: dict[str, object] = {}
    if argument == "database_path":
        database_path = value
    elif argument == "mode":
        mode = value
    else:
        kwargs[argument] = value
    resolve_called = False

    def unexpected_resolve(path: Path, *, strict: bool = False) -> Path:
        nonlocal resolve_called
        resolve_called = True
        raise AssertionError("path resolution should not run")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)

    with pytest.raises(ProfileRepositoryError) as exc_info:
        ProfileStoreLease(database_path, mode, **kwargs)  # type: ignore[arg-type]

    _assert_safe_repository_error(
        exc_info.value,
        "operation_failed",
        "private-profile-name",
        repr(value),
    )
    assert resolve_called is False


def test_timing_values_are_normalized_to_float(tmp_path: Path) -> None:
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
        timeout_seconds=2,
        check_interval_seconds=1,
    )

    assert lease.timeout_seconds == 2.0
    assert lease.check_interval_seconds == 1.0
    assert type(lease.timeout_seconds) is float
    assert type(lease.check_interval_seconds) is float


@pytest.mark.parametrize(
    "field_name",
    ["timeout_seconds", "check_interval_seconds"],
)
def test_oversized_integer_timing_fails_safely_before_path_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    oversized = 10**400
    database_path = tmp_path / "private-profile-name.sqlite3"
    resolve_called = False

    def unexpected_resolve(path: Path, *, strict: bool = False) -> Path:
        nonlocal resolve_called
        resolve_called = True
        raise AssertionError("path resolution should not run")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)

    with pytest.raises(ProfileRepositoryError) as exc_info:
        ProfileStoreLease(
            database_path,
            ProfileStoreLockMode.SHARED,
            **{field_name: oversized},
        )

    _assert_safe_repository_error(
        exc_info.value,
        "operation_failed",
        str(oversized),
        str(database_path),
    )
    assert resolve_called is False


def test_path_resolution_failure_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-path-resolution-secret"

    def fail_resolve(path: Path, *, strict: bool = False) -> Path:
        raise OSError(secret)

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(ProfileRepositoryError) as exc_info:
        ProfileStoreLease(
            tmp_path / secret,
            ProfileStoreLockMode.SHARED,
        )

    _assert_safe_repository_error(exc_info.value, "operation_failed", secret)


def test_two_shared_leases_coexist(tmp_path: Path) -> None:
    first = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )
    second = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )

    with first, second:
        assert first.acquired is True
        assert second.acquired is True


@pytest.mark.parametrize(
    ("held_mode", "waiting_mode"),
    [
        pytest.param(
            ProfileStoreLockMode.SHARED,
            ProfileStoreLockMode.EXCLUSIVE,
            id="exclusive-waits-for-shared",
        ),
        pytest.param(
            ProfileStoreLockMode.EXCLUSIVE,
            ProfileStoreLockMode.SHARED,
            id="shared-waits-for-exclusive",
        ),
    ],
)
def test_incompatible_lease_times_out(
    tmp_path: Path,
    held_mode: ProfileStoreLockMode,
    waiting_mode: ProfileStoreLockMode,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    held = ProfileStoreLease(database_path, held_mode)
    waiting = ProfileStoreLease(
        database_path,
        waiting_mode,
        timeout_seconds=0.05,
        check_interval_seconds=0.005,
    )

    with held:
        with pytest.raises(ProfileRepositoryError) as exc_info:
            waiting.acquire()

    _assert_safe_repository_error(exc_info.value, "lock_timeout", str(database_path))
    assert waiting.acquired is False


def test_different_database_paths_do_not_contend(tmp_path: Path) -> None:
    first = ProfileStoreLease(
        tmp_path / "first.sqlite3",
        ProfileStoreLockMode.EXCLUSIVE,
    )
    second = ProfileStoreLease(
        tmp_path / "second.sqlite3",
        ProfileStoreLockMode.EXCLUSIVE,
        timeout_seconds=0.05,
    )

    with first, second:
        assert first.acquired is True
        assert second.acquired is True


def test_release_allows_opposite_mode_and_keeps_empty_lock_file(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    shared = ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED)
    shared.acquire()
    shared.release()

    assert shared.acquired is False
    assert shared.lock_path.is_file()
    assert shared.lock_path.read_bytes() == b""
    with ProfileStoreLease(
        database_path,
        ProfileStoreLockMode.EXCLUSIVE,
        timeout_seconds=0.2,
    ) as exclusive:
        assert exclusive.acquired is True
    assert database_path.exists() is False
    assert shared.lock_path.is_file()


def test_first_lock_attempt_occurs_before_deadline_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self.values = iter((0.0, 2.0))

        def monotonic(self) -> float:
            return next(self.values)

        def sleep(self, seconds: float) -> None:
            raise AssertionError("successful initial attempt must not sleep")

    attempted = False
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)

    def record_lock(current_handle: object, flags: object) -> None:
        nonlocal attempted
        attempted = True

    monkeypatch.setattr(lock_module, "time", Clock())
    monkeypatch.setattr(portalocker, "lock", record_lock)
    monkeypatch.setattr(portalocker, "unlock", lambda current_handle: None)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
        timeout_seconds=1.0,
    )

    lease.acquire()
    lease.release()

    assert attempted is True
    assert handle.closed is True


def test_timeout_is_bounded_and_attempts_immediately(tmp_path: Path) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    waiting = ProfileStoreLease(
        database_path,
        ProfileStoreLockMode.EXCLUSIVE,
        timeout_seconds=0.06,
        check_interval_seconds=0.01,
    )

    with ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED):
        started = time.monotonic()
        with pytest.raises(ProfileRepositoryError) as exc_info:
            waiting.acquire()
        elapsed = time.monotonic() - started

    _assert_safe_repository_error(exc_info.value, "lock_timeout")
    assert 0.04 <= elapsed < 0.5
    assert waiting.lock_path.exists()


class _RecordingHandle:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.closed = False
        self.close_error = close_error

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _patch_open(
    monkeypatch: pytest.MonkeyPatch,
    handle: _RecordingHandle,
) -> None:
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)


def test_timeout_closes_unowned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(
        portalocker,
        "lock",
        lambda current_handle, flags: (_ for _ in ()).throw(
            portalocker.exceptions.AlreadyLocked(
                "contention-secret",
                fh=current_handle,
            )
        ),
    )
    monkeypatch.setattr(lock_module.time, "sleep", lambda seconds: None)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
        timeout_seconds=0.000001,
    )

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.acquire()

    _assert_safe_repository_error(
        exc_info.value,
        "lock_timeout",
        "contention-secret",
    )
    assert handle.closed is True
    assert lease.acquired is False


def test_timeout_remains_safe_when_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _RecordingHandle(OSError("close-private-secret"))
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(
        portalocker,
        "lock",
        lambda current_handle, flags: (_ for _ in ()).throw(
            portalocker.exceptions.AlreadyLocked(
                "contention-private-secret",
                fh=current_handle,
            )
        ),
    )
    monkeypatch.setattr(lock_module.time, "sleep", lambda seconds: None)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
        timeout_seconds=0.000001,
    )

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.acquire()

    _assert_safe_repository_error(
        exc_info.value,
        "lock_timeout",
        "close-private-secret",
        "contention-private-secret",
    )
    assert handle.closed is True
    assert lease.acquired is False


def test_backend_lock_failure_is_not_retried_and_closes_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)

    def fail_lock(current_handle: object, flags: object) -> None:
        raise portalocker.exceptions.LockException("backend-secret")

    monkeypatch.setattr(portalocker, "lock", fail_lock)
    monkeypatch.setattr(
        lock_module.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError("must not retry")),
    )
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.acquire()

    _assert_safe_repository_error(
        exc_info.value,
        "operation_failed",
        "backend-secret",
    )
    assert handle.closed is True
    assert lease.acquired is False


def test_open_failure_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "open-private-path-secret"
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(secret)),
    )
    lease = ProfileStoreLease(
        tmp_path / secret,
        ProfileStoreLockMode.SHARED,
    )

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.acquire()

    _assert_safe_repository_error(exc_info.value, "operation_failed", secret)
    assert lease.acquired is False


def test_missing_parent_open_failure_is_safe(tmp_path: Path) -> None:
    database_path = tmp_path / "missing-parent" / "private.sqlite3"
    lease = ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED)

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.acquire()

    _assert_safe_repository_error(
        exc_info.value,
        "operation_failed",
        str(database_path),
    )
    assert database_path.parent.exists() is False
    assert lease.acquired is False


def test_lock_flags_match_exact_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[portalocker.LockFlags] = []
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(
        portalocker,
        "lock",
        lambda current_handle, flags: seen.append(flags),
    )
    monkeypatch.setattr(portalocker, "unlock", lambda current_handle: None)

    for mode in (ProfileStoreLockMode.SHARED, ProfileStoreLockMode.EXCLUSIVE):
        ProfileStoreLease(tmp_path / f"{mode.value}.sqlite3", mode).acquire().release()

    assert seen == [
        portalocker.LockFlags.SHARED | portalocker.LockFlags.NON_BLOCKING,
        portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
    ]


def test_double_acquire_fails_safely_and_release_is_idempotent(
    tmp_path: Path,
) -> None:
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()
    try:
        with pytest.raises(ProfileRepositoryError) as exc_info:
            lease.acquire()
        _assert_safe_repository_error(exc_info.value, "invalid_state")
        assert lease.acquired is True
    finally:
        lease.release()

    lease.release()
    assert lease.acquired is False


def test_unlock_failure_still_closes_and_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)
    monkeypatch.setattr(
        portalocker,
        "unlock",
        lambda current_handle: (_ for _ in ()).throw(OSError("unlock-private-secret")),
    )
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.release()

    _assert_safe_repository_error(
        exc_info.value,
        "operation_failed",
        "unlock-private-secret",
    )
    assert handle.closed is True
    assert lease.acquired is False


def test_close_failure_after_unlock_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _RecordingHandle(OSError("close-private-secret"))
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)
    monkeypatch.setattr(portalocker, "unlock", lambda current_handle: None)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()

    with pytest.raises(ProfileRepositoryError) as exc_info:
        lease.release()

    _assert_safe_repository_error(
        exc_info.value,
        "operation_failed",
        "close-private-secret",
    )
    assert handle.closed is True
    assert lease.acquired is False


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(12)])
def test_close_control_exception_outranks_ordinary_unlock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    handle = _RecordingHandle(interrupt)
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)
    monkeypatch.setattr(
        portalocker,
        "unlock",
        lambda current_handle: (_ for _ in ()).throw(OSError("unlock-private-secret")),
    )
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()

    with pytest.raises(type(interrupt)) as exc_info:
        lease.release()

    assert exc_info.value is interrupt
    assert handle.closed is True
    assert lease.acquired is False


def test_first_control_exception_wins_when_unlock_and_close_both_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unlock_interrupt = KeyboardInterrupt()
    close_interrupt = SystemExit(13)
    handle = _RecordingHandle(close_interrupt)
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)

    def interrupt_unlock(current_handle: object) -> None:
        raise unlock_interrupt

    monkeypatch.setattr(portalocker, "unlock", interrupt_unlock)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()

    with pytest.raises(KeyboardInterrupt) as exc_info:
        lease.release()

    assert exc_info.value is unlock_interrupt
    assert handle.closed is True
    assert lease.acquired is False


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(7)])
def test_unlock_base_exception_is_preserved_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)

    def interrupt_unlock(current_handle: object) -> None:
        raise interrupt

    monkeypatch.setattr(portalocker, "unlock", interrupt_unlock)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()

    with pytest.raises(type(interrupt)) as exc_info:
        lease.release()

    assert exc_info.value is interrupt
    assert handle.closed is True
    assert lease.acquired is False


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(8)])
def test_close_base_exception_is_preserved_after_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    handle = _RecordingHandle(interrupt)
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)
    monkeypatch.setattr(portalocker, "unlock", lambda current_handle: None)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    ).acquire()

    with pytest.raises(type(interrupt)) as exc_info:
        lease.release()

    assert exc_info.value is interrupt
    assert handle.closed is True
    assert lease.acquired is False


def test_acquisition_base_exception_closes_handle_and_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupt = KeyboardInterrupt()
    handle = _RecordingHandle()
    _patch_open(monkeypatch, handle)

    def interrupt_lock(current_handle: object, flags: object) -> None:
        raise interrupt

    monkeypatch.setattr(portalocker, "lock", interrupt_lock)
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        lease.acquire()

    assert exc_info.value is interrupt
    assert handle.closed is True
    assert lease.acquired is False


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(10)])
def test_ownership_transfer_interrupt_releases_real_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    class InterruptingTransferLease(ProfileStoreLease):
        def __init__(self, database_path: Path) -> None:
            self.interrupt_transfer = False
            self.observed_handle: BinaryIO | None = None
            super().__init__(database_path, ProfileStoreLockMode.EXCLUSIVE)
            self.interrupt_transfer = True

        def __setattr__(self, name: str, value: object) -> None:
            if (
                name == "_handle"
                and value is not None
                and getattr(self, "interrupt_transfer", False)
            ):
                object.__setattr__(self, "observed_handle", value)
                raise interrupt
            super().__setattr__(name, value)

    database_path = tmp_path / "profiles.sqlite3"
    lease = InterruptingTransferLease(database_path)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                portalocker,
                "unlock",
                lambda handle: (_ for _ in ()).throw(OSError("cleanup-private-secret")),
            )
            with pytest.raises(type(interrupt)) as exc_info:
                lease.acquire()

        assert exc_info.value is interrupt
        assert lease.acquired is False
        with ProfileStoreLease(
            database_path,
            ProfileStoreLockMode.EXCLUSIVE,
            timeout_seconds=0.05,
            check_interval_seconds=0.005,
        ) as recovered:
            assert recovered.acquired is True
    finally:
        handle = lease.observed_handle
        if handle is not None and not handle.closed:
            try:
                portalocker.unlock(handle)
            finally:
                handle.close()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(9)])
def test_post_open_clock_base_exception_closes_handle_and_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    class InterruptingClock:
        def __init__(self) -> None:
            self.calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            if self.calls == 1:
                return 0.0
            raise interrupt

        def sleep(self, seconds: float) -> None:
            raise AssertionError("clock interrupt must happen before sleep")

    handle = _RecordingHandle(OSError("cleanup-private-secret"))
    _patch_open(monkeypatch, handle)
    monkeypatch.setattr(lock_module, "time", InterruptingClock())
    monkeypatch.setattr(
        portalocker,
        "lock",
        lambda current_handle, flags: (_ for _ in ()).throw(
            portalocker.exceptions.AlreadyLocked(
                "contention-private-secret",
                fh=current_handle,
            )
        ),
    )
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )

    with pytest.raises(type(interrupt)) as exc_info:
        lease.acquire()

    assert exc_info.value is interrupt
    assert handle.closed is True
    assert lease.acquired is False


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.EXCLUSIVE,
    )

    with lease as entered:
        assert entered is lease
        assert lease.acquired is True

    assert lease.acquired is False


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(11)])
def test_release_state_transition_interrupt_releases_real_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    class InterruptingReleaseLease(ProfileStoreLease):
        def __init__(self, database_path: Path) -> None:
            self.interrupt_clear = False
            super().__init__(database_path, ProfileStoreLockMode.EXCLUSIVE)

        def __setattr__(self, name: str, value: object) -> None:
            if (
                name == "_handle"
                and value is None
                and getattr(self, "interrupt_clear", False)
            ):
                object.__setattr__(self, "interrupt_clear", False)
                raise interrupt
            super().__setattr__(name, value)

    database_path = tmp_path / "profiles.sqlite3"
    lease = InterruptingReleaseLease(database_path)
    lease.acquire()
    lease.interrupt_clear = True

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                portalocker,
                "unlock",
                lambda handle: (_ for _ in ()).throw(OSError("cleanup-private-secret")),
            )
            with pytest.raises(type(interrupt)) as exc_info:
                lease.release()

        assert exc_info.value is interrupt
        assert lease.acquired is False
        with ProfileStoreLease(
            database_path,
            ProfileStoreLockMode.EXCLUSIVE,
            timeout_seconds=0.05,
            check_interval_seconds=0.005,
        ) as recovered:
            assert recovered.acquired is True
    finally:
        lease.interrupt_clear = False
        lease.release()


def test_context_preserves_body_error_and_adds_only_safe_cleanup_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )
    body_error = ValueError("body-error")
    original_release = lease.release

    def fail_after_release() -> None:
        original_release()
        raise RuntimeError("cleanup-private-secret")

    monkeypatch.setattr(lease, "release", fail_after_release)

    with pytest.raises(ValueError) as exc_info:
        with lease:
            raise body_error

    assert exc_info.value is body_error
    assert getattr(body_error, "__notes__", ()) == [
        "TTS profile store lease cleanup failed"
    ]
    assert "cleanup-private-secret" not in repr(body_error)
    assert lease.acquired is False


def test_context_preserves_body_when_add_note_override_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileBodyError(Exception):
        def add_note(self, note: str) -> None:
            raise RuntimeError("hostile-add-note-secret")

    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )
    body_error = HostileBodyError("body-error")
    original_release = lease.release

    def fail_after_release() -> None:
        original_release()
        raise RuntimeError("cleanup-private-secret")

    monkeypatch.setattr(lease, "release", fail_after_release)

    with pytest.raises(HostileBodyError) as exc_info:
        with lease:
            raise body_error

    assert exc_info.value is body_error
    assert getattr(body_error, "__notes__", ()) == [
        "TTS profile store lease cleanup failed"
    ]
    assert "cleanup-private-secret" not in repr(body_error)
    assert "hostile-add-note-secret" not in repr(body_error)
    assert lease.acquired is False


def test_context_propagates_cleanup_failure_without_body_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = ProfileStoreLease(
        tmp_path / "profiles.sqlite3",
        ProfileStoreLockMode.SHARED,
    )
    cleanup_error = RuntimeError("cleanup failure")
    original_release = lease.release

    def fail_after_release() -> None:
        original_release()
        raise cleanup_error

    monkeypatch.setattr(lease, "release", fail_after_release)

    with pytest.raises(RuntimeError) as exc_info:
        with lease:
            pass

    assert exc_info.value is cleanup_error
    assert lease.acquired is False


def test_spawned_shared_lease_blocks_parent_exclusive(tmp_path: Path) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_spawned_lease_holder,
        args=(str(database_path), child_connection),
    )
    started = False
    release_sent = False
    try:
        process.start()
        started = True
        child_connection.close()
        assert parent_connection.poll(5.0), "child did not report ready"
        assert parent_connection.recv() == ("ready", None)

        waiting = ProfileStoreLease(
            database_path,
            ProfileStoreLockMode.EXCLUSIVE,
            timeout_seconds=0.1,
            check_interval_seconds=0.01,
        )
        with pytest.raises(ProfileRepositoryError) as exc_info:
            waiting.acquire()
        _assert_safe_repository_error(exc_info.value, "lock_timeout")

        parent_connection.send("release")
        release_sent = True
        assert parent_connection.poll(5.0), "child did not report release"
        assert parent_connection.recv() == ("released", None)
    finally:
        if started and process.is_alive() and not release_sent:
            try:
                parent_connection.send("release")
            except (BrokenPipeError, EOFError, OSError):
                pass
        if started:
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
        child_connection.close()
        parent_connection.close()

    assert process.exitcode == 0
