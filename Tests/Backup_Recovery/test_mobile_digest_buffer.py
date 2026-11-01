"""Native digest reads retain admission while reusing local byte storage."""

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
def _native_source(tmp_path, payload, held):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    source = root / "source.yaml"
    source.write_bytes(payload)
    source.chmod(0o600)
    if not held:
        yield source, None
        return
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (root,))
    limits = ArchiveLimits()
    with authority.maintenance(("source",), 2) as session:
        info = source.stat()
        with session._capture_bound_sources(
            ((source, info.st_dev, info.st_ino),), stage, limits, limits.expanded_bytes
        ):
            scope = session._scopes[-1]
            assert not hasattr(scope, "read_budget")
            yield source, scope
            assert not scope.resources
        assert not scope.active and not scope.resources
    with authority.normal(("source",)):
        assert source.is_file()


@pytest.mark.parametrize("held", [False, True], ids=["ordinary", "native_capture"])
def test_explicit_digest_fits_remaining_allowance_with_one_local_buffer(
    tmp_path, monkeypatch, held
):
    # Nonuniform final tail catches hashing stale bytes from a reused full view.
    payload = bytes(range(256)) * 1024 + b"different final tail"
    budget = _TraceReadBudget()
    budget.reserve(budget._remaining - 128 * 1024)
    native = storage.os
    buffers, descriptors, lengths, closed_leases = [], set(), [], []
    native_close = storage.StorageLease.close

    def close_lease(self):
        if descriptors:
            for descriptor in descriptors:
                with pytest.raises(OSError) as closed:
                    native.fstat(descriptor)
                assert closed.value.errno == errno.EBADF
            closed_leases.append(self)
        return native_close(self)

    class FileCalls:
        def __getattr__(self, name):
            return getattr(native, name)

        def readv(self, fd, views):
            descriptors.add(fd)
            buffers.append(views[0].obj)
            length = native.readv(fd, views)
            lengths.append(length)
            return length

    with _native_source(tmp_path, payload, held) as (source, scope):
        before = source.stat()
        with monkeypatch.context() as calls:
            calls.setattr(storage, "os", FileCalls())
            calls.setattr(storage.StorageLease, "close", close_lease)
            actual = storage._digest_recovery_file(
                "eval.definitions",
                source,
                max_bytes=len(payload),
                cancel=Event(),
                _read_budget=budget,
            )
        assert actual == (len(payload), hashlib.sha256(payload).hexdigest())
        assert len(buffers) >= 3 and all(item is buffers[0] for item in buffers)
        assert len(buffers[0]) <= 65536 and lengths[-1] == 0
        assert sum(lengths) == len(payload) and 0 < lengths[-2] < 65536
        assert 0 < budget._remaining < 128 * 1024
        if not held:
            assert closed_leases
        for descriptor in descriptors:
            with pytest.raises(OSError) as closed:
                native.fstat(descriptor)
            assert closed.value.errno == errno.EBADF
        if scope is not None:
            assert not scope.resources
        after = source.stat()
        assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        assert source.read_bytes() == payload


@pytest.mark.parametrize(
    "case",
    [
        "short",
        "empty",
        "before_buffer_limit",
        "next_loop_limit",
        "cancel_before",
        "cancel_after",
        "growth",
        "fallback",
        "default",
        "collect",
        "non_digest",
        "changed_bytes",
    ],
)
def test_local_digest_faults_defaults_and_physical_retirement(
    tmp_path, monkeypatch, case
):
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    payload = b"" if case == "empty" else bytes(range(251)) * 600 + b"tail"
    budget = _TraceReadBudget()
    cancel = Event()
    native = storage.os
    calls, descriptors, custodies, buffer_charges = [], set(), [], []
    native_init = storage._CaptureFileDescriptors.__init__
    native_reserve = _TraceReadBudget.reserve

    def retain_custody(self, scope):
        native_init(self, scope)
        custodies.append(self)

    width = min(65536, max(1, len(payload) + 1))

    def reserve(self, amount):
        if self is budget and amount == width + 1024:
            buffer_charges.append(amount)
            if case == "before_buffer_limit":
                native_reserve(self, self._remaining)
        native_reserve(self, amount)

    with _native_source(tmp_path, payload, False) as (source, _):
        if case == "changed_bytes":
            source.write_bytes(b"Z" * len(payload))
        expected = source.read_bytes()
        if case == "cancel_before":
            cancel.set()

        class FileCalls:
            def __getattr__(self, name):
                if name == "readv":
                    if case == "fallback":
                        raise AttributeError(name)
                    return self.vector_read
                return getattr(native, name)

            def open(self, name, flags, *args, **kwargs):
                fd = native.open(name, flags, *args, **kwargs)
                if name == source.name:
                    descriptors.add(fd)
                return fd

            def read(self, fd, size):
                calls.append(("read", size))
                return native.read(fd, size)

            def vector_read(self, fd, views):
                calls.append(("readv", len(views[0])))
                length = native.readv(
                    fd, [views[0][:4093]] if case == "short" else views
                )
                if len(calls) == 1:
                    if case == "next_loop_limit":
                        budget.reserve(budget._remaining)
                    elif case == "cancel_after":
                        cancel.set()
                    elif case == "growth":
                        with source.open("ab") as destination:
                            destination.write(b"unexpected growth")
                return length

        options = {} if case == "default" else {"_read_budget": budget}
        with monkeypatch.context() as fault:
            fault.setattr(storage, "os", FileCalls())
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

            if case in {"before_buffer_limit", "next_loop_limit"}:
                with pytest.raises(_TraceReadLimit):
                    consume()
            elif case in {"cancel_before", "cancel_after"}:
                with pytest.raises(InterruptedError, match="cancelled"):
                    consume()
            elif case == "growth":
                with pytest.raises(ValueError, match="definition_byte_limit"):
                    consume()
            else:
                actual = consume()
                if case == "collect":
                    assert actual == expected
                elif case == "non_digest":
                    assert actual is None
                else:
                    assert actual == (
                        len(expected),
                        hashlib.sha256(expected).hexdigest(),
                    )
                    if case == "changed_bytes":
                        assert actual[1] != hashlib.sha256(payload).hexdigest()

        assert descriptors and custodies
        for descriptor in descriptors:
            with pytest.raises(OSError) as closed:
                native.fstat(descriptor)
            assert closed.value.errno == errno.EBADF
        assert all(
            not row.fds and not row.directory_iterators and not row.close_failed
            for row in custodies
        )
        if case in {"before_buffer_limit", "cancel_before"}:
            assert calls == []
        elif case in {"next_loop_limit", "cancel_after"}:
            assert len(calls) == 1
        elif case in {"fallback", "default", "collect", "non_digest"}:
            assert calls and all(row[0] == "read" for row in calls)
            assert buffer_charges == []
        else:
            assert calls and all(row[0] == "readv" for row in calls)
            assert buffer_charges == [width + 1024]
        if case == "empty":
            assert calls == [("readv", 1)]
        if case == "short":
            assert len(calls) > len(payload) // 4093
        if case == "default":
            assert budget._remaining == 16 * 1024**2
        elif case in {"before_buffer_limit", "next_loop_limit"}:
            assert budget._remaining == 0
        else:
            assert budget._remaining < 16 * 1024**2
        if case != "growth":
            assert source.read_bytes() == expected
        # A fresh real lease can enter after all physical reader resources retire.
        with storage.acquire_storage(source):
            assert source.is_file()


def test_native_capture_rejects_replaced_source_before_read(tmp_path, monkeypatch):
    payload = b"original source"
    native = storage.os
    reads = []

    class Calls:
        def __getattr__(self, name):
            return getattr(native, name)

        def readv(self, fd, views):
            reads.append(fd)
            return native.readv(fd, views)

    with _native_source(tmp_path, payload, True) as (source, scope):
        retained = source.with_suffix(".retained")
        source.rename(retained)
        source.write_bytes(b"different inode")
        source.chmod(0o600)
        with monkeypatch.context() as fault:
            fault.setattr(storage, "os", Calls())
            with pytest.raises(
                storage.bootstrap.RecoveryRequired, match="capture_source_outside_scope"
            ):
                storage._digest_recovery_file(
                    "eval.definitions",
                    source,
                    max_bytes=100,
                    _read_budget=_TraceReadBudget(),
                )
        assert reads == [] and not scope.resources
        assert (
            retained.read_bytes() == payload
            and source.read_bytes() == b"different inode"
        )


def test_current_native_budget_uses_its_own_buffer(tmp_path, monkeypatch):
    source = tmp_path / "source.yaml"
    payload = bytes(range(251)) * 600 + b"native tail"
    source.write_bytes(payload)
    source.chmod(0o600)
    budget = _TraceReadBudget()
    native = storage.os
    buffers = []

    class Calls:
        def __getattr__(self, name):
            return getattr(native, name)

        def readv(self, fd, views):
            buffers.append(views[0].obj)
            return native.readv(fd, views)

    with storage._preview_reads(_read_budget=budget):
        scope = storage._local.preview_scope
        with monkeypatch.context() as observation:
            observation.setattr(storage, "os", Calls())
            actual = storage._digest_recovery_file(
                "eval.definitions", source, max_bytes=len(payload), _read_budget=budget
            )
        assert actual == (len(payload), hashlib.sha256(payload).hexdigest())
        assert buffers and all(row is scope.copy_buffer.obj for row in buffers)
        assert scope.resources == [scope.custody]
        assert not scope.custody.fds and not scope.custody.directory_iterators
        assert not scope.custody.close_failed
    assert not scope.active and not scope.resources
