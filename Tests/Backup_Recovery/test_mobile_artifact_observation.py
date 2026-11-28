"""Bounded native observation preserves actual object evidence and custody."""

import pytest

from tldw_chatbook.Backup_Recovery import journal
from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors


def _limits(*, count, size, characters, length, depth):
    return journal._ArtifactObservationLimits(
        entry_count=count,
        total_file_bytes=size,
        total_relative_characters=characters,
        maximum_relative_length=length,
        maximum_depth=depth,
        maximum_component_length=255,
    )


def _observe(path, limits, *, reusable=False):
    custody = _CaptureFileDescriptors(None)
    buffer = bytearray(65536)
    view = memoryview(buffer)
    try:
        result = journal.observe_artifact(
            path,
            _limits=limits,
            _custody=custody,
            _read_buffer=(buffer, view, [view]) if reusable else None,
        )
        assert not custody.fds and not custody.directory_iterators
        return result
    finally:
        custody.retire()


@pytest.mark.parametrize("reusable", [False, True])
def test_bounded_native_observation_preserves_dfs_digest_and_nonbmp(tmp_path, reusable):
    root = tmp_path / "native"
    root.mkdir(mode=0o700)
    (root / "a").mkdir(mode=0o700)
    (root / "a" / "😀.txt").write_bytes(b"exact native bytes")
    (root / "a.txt").write_bytes(b"sibling")
    paths = ("", "/a", "/a/😀.txt", "/a.txt")
    limits = _limits(
        count=4,
        size=25,
        characters=sum(map(len, paths)),
        length=max(map(len, paths)),
        depth=2,
    )
    assert _observe(root, limits, reusable=reusable) == journal.observe_artifact(root)


def test_reusable_native_hash_covers_full_partial_empty_and_multiple_files(tmp_path):
    root = tmp_path / "native"
    root.mkdir(mode=0o700)
    values = {"a": b"a" * 131072, "b": b"b" * 65537, "empty": b""}
    for name, value in values.items():
        (root / name).write_bytes(value)
    paths = ("", "/a", "/b", "/empty")
    limits = _limits(
        count=4,
        size=sum(map(len, values.values())),
        characters=sum(map(len, paths)),
        length=6,
        depth=1,
    )
    assert _observe(root, limits, reusable=True) == journal.observe_artifact(root)


def test_reusable_native_hash_refuses_short_read_and_retires_descriptors(
    tmp_path, monkeypatch
):
    path = tmp_path / "value"
    path.write_bytes(b"complete original" * 8192)
    limits = _limits(count=1, size=path.stat().st_size, characters=0, length=0, depth=0)
    native = journal.os.readv

    def short(fd, buffers):
        return native(fd, [buffers[0][:128]])

    monkeypatch.setattr(journal.os, "readv", short)
    with pytest.raises(ValueError, match="artifact_changed"):
        _observe(path, limits, reusable=True)
    assert path.read_bytes() == b"complete original" * 8192


@pytest.mark.parametrize("excess", ["entry", "bytes", "depth"])
def test_native_limits_refuse_before_excess_name_retention_or_file_read(
    tmp_path, monkeypatch, excess
):
    root = tmp_path / "native"
    root.mkdir(mode=0o700)
    if excess == "depth":
        (root / "a").mkdir(mode=0o700)
        (root / "a" / "b").mkdir(mode=0o700)
    else:
        (root / "file").write_bytes(b"retained bytes")

    # Body reads must never happen when fstat/name/depth already exceed the
    # admitted fixed census. The actual native tree is retained on refusal.
    def unexpected_read(*args, **kwargs):
        pytest.fail("native payload read preceded its resource admission")

    monkeypatch.setattr(journal.os, "read", unexpected_read)
    limits = _limits(
        count=1 if excess == "entry" else 3,
        size=0,
        characters=100,
        length=100,
        depth=1 if excess == "depth" else 2,
    )
    with pytest.raises(ValueError, match="artifact_limit"):
        _observe(root, limits)
    assert root.is_dir()
    assert (root / ("a/b" if excess == "depth" else "file")).exists()


@pytest.mark.parametrize(
    "field,value",
    [("entry_count", True), ("total_file_bytes", -1), ("maximum_depth", 1.0)],
)
def test_native_limit_record_rejects_noninteger_or_negative_bounds(field, value):
    from dataclasses import replace

    limits = _limits(count=1, size=0, characters=0, length=0, depth=0)
    with pytest.raises(ValueError, match="artifact_limits_invalid"):
        replace(limits, **{field: value})


@pytest.mark.parametrize("field", ["device", "inode"])
def test_native_artifact_physical_fields_admit_exact_unsigned64_only(field):
    from tldw_chatbook.DB.mobile_consolidation_receipts import _artifact

    value = {
        "path": "/native/artifact",
        "device": 1,
        "inode": 1,
        "kind": "file",
        "size": 0,
        "sha256": "a" * 64,
    }
    for admitted in (0, 2**63, 2**64 - 1):
        assert _artifact(value | {field: admitted})
    for refused in (-1, 2**64, True, 1.0):
        with pytest.raises(ValueError, match="mobile_owner_receipt_invalid"):
            _artifact(value | {field: refused})
    with pytest.raises(ValueError, match="mobile_owner_receipt_invalid"):
        _artifact(value | {"size": 2**63})


def test_actual_unexhausted_iterator_close_failure_retains_native_custody_until_exit(
    tmp_path,
):
    import os
    import subprocess
    import sys

    root = tmp_path / "close-fault"
    root.mkdir(mode=0o700)
    (root / "a").write_bytes(b"a")
    (root / "b").write_bytes(b"b")
    script = r"""
import os, sys
from pathlib import Path
from Tests import real_profile_guard, network_guard
real_profile_guard.install()
network_guard.install()
from tldw_chatbook.Backup_Recovery import journal, bootstrap
from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors
native_scandir = journal.os.scandir
class CloseFault:
    def __init__(self, native):
        self.native = native
        self.close_calls = 0
    def __iter__(self): return self
    def __next__(self): return next(self.native)
    def close(self):
        self.close_calls += 1
        raise OSError("injected iterator close refusal before native retirement")
def open_iterator(fd):
    return CloseFault(native_scandir(fd))
journal.os.scandir = open_iterator
custody = _CaptureFileDescriptors(None)
limits = journal._ArtifactObservationLimits(1, 0, 0, 0, 0, 255)
try:
    journal.observe_artifact(Path(sys.argv[1]), _limits=limits, _custody=custody)
except bootstrap.RecoveryRequired:
    pass
else:
    raise AssertionError("actual iterator close uncertainty was swallowed")
assert custody.close_failed and len(custody.directory_iterators) == 1
iterator = custody.directory_iterators[0]
assert iterator.close_calls == 1
assert custody.fds and all(os.fstat(fd) for fd in custody.fds)
# The iterator was interrupted after its first yielded entry, not exhausted.
# A real native second entry remains readable; no underlying close was called.
assert next(iterator.native).name in {"a", "b"}
try:
    custody.retire()
except bootstrap.RecoveryRequired:
    pass
else:
    raise AssertionError("uncertain custody was released")
assert iterator.close_calls == 1 and custody.directory_iterators[0] is iterator
assert not real_profile_guard.take_violations()
assert not network_guard.blocked_attempts()
print("native_iterator_retained_guards_empty", flush=True)
os._exit(0)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "native_iterator_retained_guards_empty" in child.stdout


@pytest.mark.parametrize("mode", ["implicit", "explicit", "cancelled", "changed"])
def test_budgeted_artifact_match_uses_native_cancel_and_retires(tmp_path, mode):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import current
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    path = tmp_path / "native-cipher"
    path.write_bytes(b"exact private artifact")
    path.chmod(0o600)
    expected = journal._Object.model_validate(journal.observe_artifact(path))
    if mode == "changed":
        path.write_bytes(b"wrong private artifact")
    budget = _TraceReadBudget()
    cancel = Event()
    if mode == "cancelled":
        cancel.set()
    with _preview_reads(_read_budget=budget):
        scope = current()
        options = {} if mode == "implicit" else {"cancel": cancel}
        assert journal._matches(expected, str(path), **options) is (
            mode not in {"cancelled", "changed"}
        )
        assert not scope.custody.fds and not scope.custody.directory_iterators
    assert not scope.active and budget._remaining < 16 * 1024**2
    assert path.read_bytes() == (
        b"wrong private artifact" if mode == "changed" else b"exact private artifact"
    )
