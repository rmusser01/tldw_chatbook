"""Actual native capture authority reuses bytes, never file verification results."""

import errno
import hashlib
from contextlib import contextmanager
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


@contextmanager
def _sources(tmp_path, payloads):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    paths = []
    for index, payload in enumerate(payloads):
        path = root / f"source-{index}.yaml"
        path.write_bytes(payload)
        path.chmod(0o600)
        paths.append(path)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (root,))
    limits = ArchiveLimits()
    with authority.maintenance(("source",), 2) as session:
        selected = tuple(
            (path, path.stat().st_dev, path.stat().st_ino) for path in paths
        )
        with session._capture_bound_sources(
            selected, stage, limits, limits.expanded_bytes
        ):
            scope = session._scopes[-1]
            assert type(scope) is storage._CaptureScope
            assert not hasattr(scope, "read_budget")
            yield paths, scope
        assert not scope.active and not scope.resources
    with authority.normal(("source",)):
        assert all(path.is_file() for path in paths)


def _closed(native, descriptors):
    for descriptor in descriptors:
        with pytest.raises(OSError) as closed:
            native.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_two_files_two_passes_share_actual_workspace_and_original_allowance(
    tmp_path, monkeypatch
):
    payloads = [
        bytes(range(251)) * 600 + b"first tail",
        bytes(range(255, -1, -1)) * 700 + b"second different tail",
    ]
    budget = _TraceReadBudget()
    budget.reserve(budget._remaining - 128 * 1024)
    native, native_reserve = storage.os, _TraceReadBudget.reserve
    buffers, charges, lengths = [], [], []
    descriptors = set()

    class Calls:
        def __getattr__(self, name):
            return getattr(native, name)

        def readv(self, descriptor, views):
            descriptors.add(descriptor)
            buffers.append(views[0].obj)
            length = native.readv(descriptor, views)
            lengths.append(length)
            return length

    def reserve(self, amount):
        if self is budget and amount == 66560:
            charges.append(amount)
        return native_reserve(self, amount)

    with _sources(tmp_path, payloads) as (paths, scope):
        before = [path.stat() for path in paths]
        with monkeypatch.context() as observation:
            observation.setattr(storage, "os", Calls())
            observation.setattr(_TraceReadBudget, "reserve", reserve)
            for _ in range(2):
                for path, payload in zip(paths, payloads, strict=True):
                    actual = storage._digest_recovery_file(
                        "eval.definitions",
                        path,
                        max_bytes=len(payload),
                        cancel=Event(),
                        _read_budget=budget,
                    )
                    assert actual == (len(payload), hashlib.sha256(payload).hexdigest())
                    _closed(native, descriptors)
                    assert not scope.resources
        assert charges == [66560]
        assert buffers and all(item is buffers[0] for item in buffers)
        assert scope._digest_buffer.obj is buffers[0]
        assert len(buffers[0]) == 65536 and lengths.count(0) == 4
        assert sum(lengths) == 2 * sum(map(len, payloads))
        assert 0 < budget._remaining < 128 * 1024
        for path, payload, prior in zip(paths, payloads, before, strict=True):
            now = path.stat()
            assert (
                now.st_dev,
                now.st_ino,
                now.st_mode,
                now.st_size,
                now.st_mtime_ns,
                now.st_ctime_ns,
            ) == (
                prior.st_dev,
                prior.st_ino,
                prior.st_mode,
                prior.st_size,
                prior.st_mtime_ns,
                prior.st_ctime_ns,
            )
            assert path.read_bytes() == payload
    assert scope._digest_buffer is None


def test_different_native_scopes_have_distinct_workspaces(tmp_path):
    retained = []
    budget = _TraceReadBudget()
    for index in range(2):
        selected = tmp_path / str(index)
        selected.mkdir(mode=0o700)
        with _sources(selected, [b"a" * 100000]) as (paths, scope):
            storage._digest_recovery_file(
                "eval.definitions", paths[0], max_bytes=100000, _read_budget=budget
            )
            retained.append(scope._digest_buffer.obj)
        assert scope._digest_buffer is None
        with pytest.raises(storage.bootstrap.RecoveryRequired):
            scope.check()
    assert retained[0] is not retained[1]


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "tiny_limit",
        "short",
        "changed_between_passes",
        "default",
        "collect",
        "non_digest",
        "fallback",
        "before_buffer_limit",
        "later_loop_limit",
        "cancel_before",
        "cancel_after",
    ],
)
def test_capture_workspace_faults_defaults_and_actual_fd_cleanup(
    tmp_path, monkeypatch, case
):
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    payload = (
        b"" if case in {"empty", "tiny_limit"} else bytes(range(251)) * 600 + b"tail"
    )
    budget, cancel = _TraceReadBudget(), Event()
    if case == "tiny_limit":
        budget.reserve(budget._remaining - 32000)
    native, native_reserve = storage.os, _TraceReadBudget.reserve
    native_init = storage._CaptureFileDescriptors.__init__
    descriptors, custodies, calls, charges = set(), [], [], []

    def retain_custody(self, scope):
        native_init(self, scope)
        custodies.append(self)

    def reserve(self, amount):
        if self is budget and amount == 66560:
            charges.append(amount)
            assert scope._digest_buffer is None
            if case == "before_buffer_limit":
                native_reserve(self, self._remaining)
        return native_reserve(self, amount)

    with _sources(tmp_path, [payload]) as (paths, scope):
        source = paths[0]
        if case == "cancel_before":
            cancel.set()

        class Calls:
            def __getattr__(self, name):
                if name == "readv":
                    if case == "fallback":
                        raise AttributeError(name)
                    return self.vector_read
                return getattr(native, name)

            def open(self, path, flags, *args, **kwargs):
                descriptor = native.open(path, flags, *args, **kwargs)
                if path == source.name:
                    descriptors.add(descriptor)
                return descriptor

            def read(self, descriptor, size):
                calls.append(("read", size))
                return native.read(descriptor, size)

            def vector_read(self, descriptor, views):
                calls.append(("readv", len(views[0])))
                length = native.readv(
                    descriptor, [views[0][:4093]] if case == "short" else views
                )
                if len(calls) == 1:
                    if case == "later_loop_limit":
                        budget.reserve(budget._remaining)
                    elif case == "cancel_after":
                        cancel.set()
                return length

        options = {} if case == "default" else {"_read_budget": budget}
        with monkeypatch.context() as fault:
            fault.setattr(storage, "os", Calls())
            fault.setattr(storage._CaptureFileDescriptors, "__init__", retain_custody)
            fault.setattr(_TraceReadBudget, "reserve", reserve)

            def consume():
                return storage._consume_recovery_file(
                    "eval.definitions",
                    source,
                    max_bytes=max(1, len(payload)),
                    collect=case == "collect",
                    digest=case not in {"collect", "non_digest"},
                    cancel=cancel,
                    **options,
                )

            if case in {"tiny_limit", "before_buffer_limit", "later_loop_limit"}:
                with pytest.raises(_TraceReadLimit):
                    consume()
            elif case in {"cancel_before", "cancel_after"}:
                with pytest.raises(InterruptedError, match="cancelled"):
                    consume()
            else:
                actual = consume()
                if case == "collect":
                    assert actual == payload
                elif case == "non_digest":
                    assert actual is None
                else:
                    assert actual == (len(payload), hashlib.sha256(payload).hexdigest())
                if case == "changed_between_passes":
                    previous_buffer = scope._digest_buffer
                    replacement = b"Z" * len(payload)
                    source.write_bytes(replacement)
                    assert consume() == (
                        len(replacement),
                        hashlib.sha256(replacement).hexdigest(),
                    )
                    assert scope._digest_buffer is previous_buffer

        assert descriptors and custodies
        _closed(native, descriptors)
        assert all(
            not item.fds and not item.directory_iterators and not item.close_failed
            for item in custodies
        )
        assert not scope.resources
        if case in {"tiny_limit", "before_buffer_limit", "cancel_before"}:
            assert calls == [] and scope._digest_buffer is None
            assert charges == ([] if case == "cancel_before" else [66560])
        elif case in {"cancel_after", "later_loop_limit"}:
            assert len(calls) == 1 and scope._digest_buffer is not None
        elif case in {"default", "collect", "non_digest", "fallback"}:
            assert calls and all(row[0] == "read" for row in calls)
            assert scope._digest_buffer is None and charges == []
        else:
            assert charges == [66560]
            assert scope._digest_buffer is not None
            if case == "empty":
                assert calls == [("readv", 1)]
            if case == "short":
                assert len(calls) > len(payload) // 4093
        if case == "default":
            assert budget._remaining == 16 * 1024**2
        if case != "changed_between_passes":
            assert source.read_bytes() == payload
    assert scope._digest_buffer is None


def test_workspace_clears_on_native_sqlite_verification_failure(tmp_path, monkeypatch):
    import sqlite3

    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE test(value TEXT)")
        payload = connection.serialize()
    connection.close()
    with (
        pytest.raises(ValueError, match="injected_retirement_verification"),
        _sources(tmp_path, [payload]) as (paths, scope),
    ):
        storage._digest_recovery_file(
            "eval.definitions",
            paths[0],
            max_bytes=len(payload),
            _read_budget=_TraceReadBudget(),
        )
        copied = scope.sqlite_target("recovery.mobile.preview", paths[0])
        directory = scope.sqlite_directory
        assert (
            copied.is_file()
            and scope.sqlite_snapshots
            and scope._digest_buffer is not None
        )
        native_verify = scope.verify_sqlite_sources

        def verify():
            native_verify()
            raise ValueError("injected_retirement_verification")

        monkeypatch.setattr(scope, "verify_sqlite_sources", verify)
    assert not scope.active and scope._digest_buffer is None and not scope.resources
    assert not directory.exists() and scope.sqlite_directory is None
    assert getattr(storage._local, "capture_scope", None) is None


def test_workspace_clears_but_native_ambiguous_close_stays_quarantined(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import admission

    held_before = len(storage._failed_capture_holds)
    native = storage.os
    descriptors, attempted = set(), []

    class Calls:
        def __getattr__(self, name):
            return getattr(native, name)

        def readv(self, descriptor, views):
            descriptors.add(descriptor)
            return native.readv(descriptor, views)

        def close(self, descriptor):
            native.close(descriptor)
            if descriptor in descriptors:
                attempted.append(descriptor)
                raise OSError("injected ambiguous close after actual close")

    with (
        pytest.raises(admission.AdmissionError, match="capture_resources_not_retired"),
        _sources(tmp_path, [b"x" * 100000]) as (paths, scope),
    ):
        monkeypatch.setattr(storage, "os", Calls())
        storage._digest_recovery_file(
            "eval.definitions",
            paths[0],
            max_bytes=100000,
            _read_budget=_TraceReadBudget(),
        )
    assert scope is not None and not scope.active and scope._digest_buffer is None
    assert len(attempted) == 1 and descriptors
    _closed(native, descriptors)
    assert any(item.close_failed and item.fds for item in scope.resources)
    assert getattr(storage._local, "capture_scope", None) is None
    assert len(storage._failed_capture_holds) == held_before + 1
    assert storage._failed_capture_holds[-1][2] is scope.session
