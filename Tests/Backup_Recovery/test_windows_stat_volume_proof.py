"""Real single-path volume proof and opt-in complete-operation measurements."""

import contextlib
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys
import time

import pytest

from Tests.Backup_Recovery.test_windows_metadata_prerequisites import native_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows stat traversal")
_SETTLE_NS = storage._EVIDENCE_SETTLE_NS


@pytest.fixture(autouse=True)
def select_stat_prototype(monkeypatch):
    monkeypatch.setattr(windows_files, "_ORDINARY_STAT_VOLUME_PROOF", True)


def test_stock_facade_cached_bound_methods_qualify_and_overrides_fall_back(monkeypatch):
    native = windows_files._native()
    facade, other = windows_files.WindowsOS(), windows_files.WindowsOS()
    current = windows_files._stat_sources_current
    assert all(
        name in vars(facade) for name in ("stat", "_stat_absolute", "_stat_handle")
    )
    assert current(native, facade)
    with monkeypatch.context() as patch:
        patch.setattr(facade, "_stat_handle", other._stat_handle)
        assert not current(native, facade)
    with monkeypatch.context() as patch:
        patch.setattr(
            windows_files.WindowsOS, "_stat_handle", lambda self, handle: None
        )
        assert not current(native, facade)
    with monkeypatch.context() as patch:
        patch.setattr(native.kernel, "GetVolumeInformationByHandleW", lambda *args: 1)
        assert not current(native, facade)
    assert current(native, facade)


@contextlib.contextmanager
def _observed_stat():
    """Observe original Python call boundaries without replacing native sources."""
    original = sys.getprofile()
    codes = {
        getattr(windows_files._Native, name).__code__: name
        for name in ("open_handle", "ntfs", "security", "info")
    }
    rows = {
        "opens": [],
        "volume_calls": 0,
        "security_calls": 0,
        "unicode_audits": 0,
        "selected_births": 0,
        "info_calls": 0,
    }
    active = True

    def profile(frame, event, result):
        name = codes.get(frame.f_code)
        if event == "call" and name in {"ntfs", "security"}:
            rows["volume_calls" if name == "ntfs" else "security_calls"] += 1
        elif event == "call" and name == "info":
            rows["info_calls"] += 1
        elif event == "return" and name == "open_handle" and type(result) is int:
            rows["opens"].append(result)
        elif event == "call" and name == "open_handle":
            record = frame.f_locals.get("_metadata_record")
            if record is not None and record.stat_audit:
                rows["selected_births"] += 1
        if original is not None:
            original(frame, event, result)

    def audit(event, args):
        if active and event == "ctypes.create_unicode_buffer" and args == (None, 32):
            rows["unicode_audits"] += 1

    sys.addaudithook(audit)
    sys.setprofile(profile)
    try:
        yield rows
    finally:
        active = False
        sys.setprofile(original)


def test_absolute_stat_keeps_metadata_audits_and_births_with_one_volume_query(tmp_path):
    native, win = windows_files._native(), windows_files.WindowsOS()
    directory = tmp_path / "first" / "second"
    directory.mkdir(parents=True)
    leaf = directory / "leaf.bin"
    leaf.write_bytes(b"real stat")
    expected = win.stat_for_admission(leaf)[0]
    with _observed_stat() as rows:
        actual = win.stat(leaf, follow_symlinks=False)
    assert tuple(actual) == tuple(expected)
    assert actual.st_mtime_ns == expected.st_mtime_ns
    assert actual.st_ctime_ns == expected.st_ctime_ns
    assert actual.st_file_attributes == expected.st_file_attributes
    assert len(rows["opens"]) == len(leaf.parts)
    assert rows["unicode_audits"] == len(rows["opens"])
    assert rows["security_calls"] == 1
    assert rows["info_calls"] == len(rows["opens"]) + 1
    for handle in set(rows["opens"]):
        with pytest.raises(OSError) as invalid:
            native.info(handle)
        assert invalid.value.winerror == 6
    counts = {
        key: len(value) if isinstance(value, list) else value
        for key, value in rows.items()
    }
    print("STAT_VOLUME_COUNTS " + json.dumps(counts, sort_keys=True))
    assert rows["volume_calls"] == 1, counts
    assert rows["selected_births"] == len(rows["opens"])


@pytest.mark.parametrize("component", ["NUL", "bad:stream", "trailing."])
def test_selected_stat_validates_every_component_before_open(tmp_path, component):
    windows_files._native()
    with _observed_stat() as rows:
        with pytest.raises(ValueError):
            windows_files.WindowsOS().stat(tmp_path / component / "leaf")
    assert rows["opens"] == [] and rows["unicode_audits"] == 0


@pytest.mark.parametrize("missing_parent", [False, True])
def test_selected_stat_preserves_missing_path_and_retires_parents(
    tmp_path, missing_parent
):
    native = windows_files._native()
    leaf = (tmp_path / "missing-parent" if missing_parent else tmp_path) / "absent"
    with _observed_stat() as rows:
        with pytest.raises(FileNotFoundError):
            windows_files.WindowsOS().stat(leaf)
    assert rows["opens"] and rows["selected_births"] > 0
    for handle in set(rows["opens"]):
        with pytest.raises(OSError) as invalid:
            native.info(handle)
        assert invalid.value.winerror == 6


@pytest.mark.parametrize(
    "route",
    [
        "relative",
        "borrowed",
        "root",
        "normalized_root",
        "subclass",
        "ntfs_override",
        "volume_override",
        "open",
    ],
)
def test_unsupported_stat_routes_keep_original_volume_calls(
    tmp_path, monkeypatch, route
):
    native, win = windows_files._native(), windows_files.WindowsOS()
    leaf = tmp_path / "original-route"
    leaf.write_bytes(b"original")
    parent = win.open(tmp_path, win.O_RDONLY | win.O_DIRECTORY)
    if route == "relative":
        monkeypatch.chdir(tmp_path)
    elif route == "subclass":

        class Custom(windows_files.WindowsOS):
            pass

        win = Custom()
    elif route == "ntfs_override":
        original = native.ntfs
        monkeypatch.setattr(native, "ntfs", lambda handle: original(handle))
    elif route == "volume_override":
        original = native.kernel.GetVolumeInformationByHandleW
        monkeypatch.setattr(
            native.kernel,
            "GetVolumeInformationByHandleW",
            lambda *args: original(*args),
        )
    try:
        with _observed_stat() as rows:
            if route == "borrowed":
                actual = win.stat(leaf.name, dir_fd=parent)
            elif route in {"root", "normalized_root"}:
                path = leaf.anchor if route == "root" else leaf.anchor + "unused\\.."
                actual = win.stat(path)
            elif route == "open":
                fd = win.open(leaf, win.O_RDWR)
                try:
                    actual = win.fstat(fd)
                finally:
                    win.close(fd)
            else:
                actual = win.stat(leaf.name if route == "relative" else leaf)
        assert actual.st_size == 8 or route in {"root", "normalized_root"}
        assert rows["opens"] and rows["volume_calls"] == len(rows["opens"])
        assert rows["selected_births"] == 0
    finally:
        win.close(parent)


@contextlib.contextmanager
def _audit_at_child(action):
    """Inject at the real allocation boundary, without replacing any source."""
    active, calls, handles = True, 0, []

    def hook(event, args):
        nonlocal calls
        if not active or event != "ctypes.create_unicode_buffer" or args != (None, 32):
            return
        frame = sys._getframe(1)
        while (
            frame is not None
            and frame.f_code is not windows_files._Native.open_handle.__code__
        ):
            frame = frame.f_back
        if frame is None:
            return
        calls += 1
        handle = frame.f_locals["handle"].value
        handles.append(handle)
        if calls == 2:
            action(frame.f_locals)

    sys.addaudithook(hook)
    try:
        yield handles
    finally:
        active = False


@pytest.mark.parametrize("fault", ["audit_deny", "source_change", "volume_mismatch"])
def test_selected_stat_child_refusals_retire_every_native_birth(tmp_path, fault):
    native, win = windows_files._native(), windows_files.WindowsOS()
    leaf = tmp_path / "child-refusal"
    leaf.write_bytes(b"unchanged")
    denied = PermissionError("controlled audit refusal")

    def action(values):
        if fault == "audit_deny":
            raise denied
        if fault == "source_change":
            native.ntfs = native.ntfs  # Instance shadow is not an original source.
        else:
            values["info"].volume ^= 1

    try:
        with _audit_at_child(action) as handles:
            with pytest.raises(OSError) as caught:
                win.stat(leaf)
        if fault == "audit_deny":
            assert caught.value is denied
        else:
            assert (
                "source_changed" in str(caught.value)
                if fault == "source_change"
                else "volume_proof_invalid" in str(caught.value)
            )
        assert len(handles) == 2
        for handle in set(handles):
            with pytest.raises(OSError) as invalid:
                native.info(handle)
            assert invalid.value.winerror == 6
        assert leaf.read_bytes() == b"unchanged"
    finally:
        if fault == "source_change" and "ntfs" in vars(native):
            del native.ntfs


def test_selected_stat_retains_failed_parent_close_without_retry(tmp_path):
    native, win = windows_files._native(), windows_files.WindowsOS()
    leaf = tmp_path / "protected-close"
    leaf.write_bytes(b"unchanged")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    protect = kernel.SetHandleInformation
    protect.argtypes = [windows_files._HANDLE, windows_files._U32, windows_files._U32]
    protect.restype = windows_files._I32
    protected = []

    def action(values):
        parent = values["parent"]
        info = native.info(parent)
        assert protect(parent, 2, 2)
        protected.append(
            (parent, (info.volume, (info.index_high << 32) | info.index_low))
        )

    try:
        with _audit_at_child(action) as handles:
            with pytest.raises(windows_files._AdmissionMetadataCloseError) as caught:
                win.stat(leaf)
        failed = caught.value.failed_handles
        assert len(failed) == 1 and len(handles) == 2
        assert failed[0].handle == protected[0][0] and failed[0].close_error != 0
        assert id(failed[0]) not in failed[0].owner
        assert native.info(protected[0][0]).attributes & windows_files._DIRECTORY
        with pytest.raises(OSError) as closed:
            native.info(handles[1])
        assert closed.value.winerror == 6
    finally:
        for handle, identity in protected:
            info = native.info(handle)
            assert (info.volume, (info.index_high << 32) | info.index_low) == identity
            assert protect(handle, 2, 0)
            assert native.kernel.CloseHandle(handle)


@pytest.mark.skipif(
    os.environ.get("TLDW_STAT_BENCHMARK") != "1",
    reason="explicit native measurement only",
)
def test_measure_single_stat_and_complete_warm_operation(native_scope, monkeypatch):  # noqa: F811
    """Fixed finite samples, no profiler, original live hold and positive retirement."""
    _, _, data, startup = native_scope
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", _SETTLE_NS)
    win = windows_files.WindowsOS()
    leaf = data / "measurement.bin"
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
    win.write(fd, b"measurement")
    win.close(fd)
    time.sleep(_SETTLE_NS / 1_000_000_000 + 0.05)
    for _ in range(8):
        with storage.acquire_storage(leaf) as lease:
            if lease._warm_execution is not None:
                break
    else:
        pytest.fail("The ordinary operation did not establish confirmed warm evidence")
    hold = storage._holds[startup._key]
    initial_count = hold.count

    def measure(selected):
        monkeypatch.setattr(windows_files, "_ORDINARY_STAT_VOLUME_PROOF", selected)
        started, cpu = time.perf_counter_ns(), time.thread_time_ns()
        for _ in range(32):
            assert win.stat(leaf).st_size == 11
        stat_wall = (time.perf_counter_ns() - started) / 32
        stat_cpu = (time.thread_time_ns() - cpu) / 32
        started, cpu = time.perf_counter_ns(), time.thread_time_ns()
        for _ in range(4):
            with storage.acquire_storage(leaf) as lease:
                assert lease._warm_execution is not None
                assert storage._holds[lease._key] is hold
                assert hold.count == initial_count + 1
            assert lease._key is None and hold.count == initial_count
        return {
            "stat_wall_ns": stat_wall,
            "stat_cpu_ns": stat_cpu,
            "warm_lease_wall_ns": (time.perf_counter_ns() - started) / 4,
            "warm_lease_cpu_ns": (time.thread_time_ns() - cpu) / 4,
        }

    samples = {"baseline": [], "candidate": []}
    for order in ((False, True), (True, False)) * 3:
        for selected in order:
            samples["candidate" if selected else "baseline"].append(measure(selected))
    result = {
        "samples": samples,
        "medians_ns": {
            side: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
            for side, rows in samples.items()
        },
        "path_components": len(Path(leaf).parts),
        "positive_lease_retirement": True,
        "evidence_settle_ns": _SETTLE_NS,
    }
    print("STAT_VOLUME_MEASUREMENT " + json.dumps(result, sort_keys=True))
