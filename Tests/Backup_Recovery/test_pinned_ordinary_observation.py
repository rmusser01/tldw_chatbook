"""Real Windows authority reads share ancestry without sharing stale evidence."""

import ctypes
import json
import sys
import threading
from collections import Counter
from contextlib import contextmanager

import pytest

from tldw_chatbook.Backup_Recovery import activation, bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows authority observations"
)


@pytest.fixture(params=[False, True], ids=["unbound", "bound"])
def authority_input(tmp_path, monkeypatch, request):
    win = windows_files.WindowsOS()
    parent = tmp_path / "private-observation"
    win.mkdir(parent, 0o700)
    root, selector = parent / "bootstrap", parent / "config.toml"
    fd = win.open(selector, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
    try:
        win.write(fd, b"[general]\nusers_name='ordinary'\n")
    finally:
        win.close(fd)
    authority = admission_authority(root)
    if request.param:
        authority.register("profile", (selector,))
        with authority.maintenance(
            (storage.UNBOUND_NAMESPACE, "profile"), 5
        ) as session:
            bind_profile(
                root, selector, ("profile",), authority.control_root, session=session
            )
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    with storage.acquire_storage() as lease:
        assert lease._key is not None
        yield root, selector, authority, request.param


def _scalar_observation(root, selector):
    controls = bootstrap._control_records(root, ordinary_only=True)
    assert controls is not None
    _, profiles, _ = controls
    lock = root / "admission" / "registry.lock"
    before = bootstrap.os.stat(lock, follow_symlinks=False)
    registry, allowed, _reason = bootstrap._ordinary_registry(root, profiles)
    assert allowed
    after = bootstrap.os.stat(lock, follow_symlinks=False)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    fingerprint = (
        bootstrap._fingerprint(selector)
        if any(profile["selector"] == str(selector) for profile in profiles)
        else None
    )
    return (
        sorted(profiles, key=lambda profile: profile["selector"]),
        registry,
        (after.st_dev, after.st_ino),
        fingerprint,
    )


@contextmanager
def _profile(observer):
    previous = sys.getprofile()

    def record(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        observer(frame, event, result)

    sys.setprofile(record)
    try:
        yield
    finally:
        sys.setprofile(previous)


def _native_observation(read, root, selector):
    native = windows_files._native()
    codes = {
        native.open_handle.__code__: "opens",
        native.security.__code__: "security",
        native.ntfs.__code__: "volume_device",
    }
    counts = Counter()
    live, retired, closing = {}, set(), {}
    native_close = native.kernel.CloseHandle
    descriptor_close = bootstrap.os.close
    tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    monitor = sys.monitoring
    events = monitor.events
    close_codes = set()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetHandleInformation.argtypes = (
        windows_files._HANDLE,
        ctypes.POINTER(windows_files._U32),
    )
    kernel.GetHandleInformation.restype = windows_files._I32
    flags = windows_files._U32()

    def begin(code, offset, function, argument):
        if function is descriptor_close:
            handle = native.handle(argument)
        elif function is native_close:
            handle = getattr(argument, "value", argument)
        else:
            return
        if handle in live:
            closing[code, offset] = handle

    def end(code, offset, function, _argument):
        if function is not descriptor_close and function is not native_close:
            return
        handle = closing.pop((code, offset), None)
        if handle is not None:
            previous_error = ctypes.get_last_error()
            try:
                assert not kernel.GetHandleInformation(handle, ctypes.byref(flags))
                assert ctypes.get_last_error() == 6  # ERROR_INVALID_HANDLE
            finally:
                ctypes.set_last_error(previous_error)
            retired.add(live.pop(handle))

    def observe(frame, event, result):
        name = codes.get(frame.f_code)
        if event == "call":
            if name is not None:
                counts[name] += 1
            if frame.f_globals.get("__name__", "").startswith("tldw_chatbook."):
                code = frame.f_code
                if code not in close_codes:
                    close_codes.add(code)
                    monitor.set_local_events(tool, code, events.CALL)
        if event == "return" and name == "opens" and result is not None:
            assert result not in live
            counts["births"] += 1
            live[result] = counts["births"]

    monitor.use_tool_id(tool, "ordinary-observation-retirement")
    monitor.register_callback(tool, events.CALL, begin)
    monitor.register_callback(tool, events.C_RETURN, end)
    try:
        with _profile(observe):
            result = read(root, selector)
    finally:
        for code in close_codes:
            monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, events.CALL, None)
        monitor.register_callback(tool, events.C_RETURN, None)
        monitor.free_tool_id(tool)
    assert not live and not closing
    counts["retired_births"] = len(retired)
    assert counts["opens"] == counts["births"] == counts["retired_births"]
    assert counts["volume_device"] >= counts["opens"]
    return result, counts


def test_pinned_ordinary_observation_has_fewer_native_walks(authority_input, request):
    root, selector, _authority, bound = authority_input
    leases = set(storage._live_leases)
    original, before = _native_observation(_scalar_observation, root, selector)
    observed, after = _native_observation(
        activation._ordinary_activation_observation, root, selector
    )
    assert observed == original
    assert bool(observed[0]) is bool(observed[3]) is bound
    assert set(storage._live_leases) == leases
    request.node.user_properties.append(
        ("ordinary_open_counts", json.dumps({"scalar": before, "selected": after}))
    )
    # The final proof may walk the whole named tree once more. It may not
    # repeat the independent absolute walks of the scalar subreaders.
    assert after["opens"] <= before["opens"] - 2 * len(root.parts), (before, after)
    assert after["security"] and after["retired_births"]


def test_control_pending_during_real_registry_wait_refuses(authority_input):
    root, selector, authority, _bound = authority_input
    locked, waiting, release = (threading.Event() for _ in range(3))
    writer_errors, reader_errors, results = [], [], []
    pending = root / "pending-late.json"
    lock_code = windows_files.WindowsLocks.flock.__code__
    registry_code = bootstrap._registry_read_lock.__wrapped__.__code__

    def hold_registry():
        try:
            with bootstrap.pinned_directory(authority.control_root) as parent:
                with authority._lock(parent, "registry.lock", bootstrap.fcntl.LOCK_EX):
                    locked.set()
                    assert release.wait(15)
        except BaseException as error:
            writer_errors.append(error)
            locked.set()

    def read():
        def observe(frame, event, _result):
            if (
                event == "call"
                and frame.f_code is lock_code
                and frame.f_back.f_code is registry_code
                and frame.f_locals["operation"] == bootstrap.fcntl.LOCK_SH
            ):
                waiting.set()

        try:
            with _profile(observe):
                results.append(
                    activation._ordinary_activation_observation(root, selector)
                )
        except BaseException as error:
            reader_errors.append(error)

    writer = threading.Thread(target=hold_registry)
    reader = threading.Thread(target=read)
    writer.start()
    try:
        assert locked.wait(15) and not writer_errors
        reader.start()
        assert waiting.wait(15) and reader.is_alive()
        assert not results and not reader_errors
        # Independent control publication is not serialized by registry.lock.
        pending.write_bytes(b"{}")
    finally:
        release.set()
        writer.join(16)
        if reader.ident is not None:
            reader.join(16)
        pending.unlink(missing_ok=True)
    assert not writer.is_alive() and not reader.is_alive() and not writer_errors
    assert not results and len(reader_errors) == 1
    assert isinstance(reader_errors[0], bootstrap.RecoveryRequired)


@pytest.mark.parametrize("change", ["control", "parent", "selector", "registry_helper"])
def test_authority_drift_before_return_refuses(authority_input, change, monkeypatch):
    root, selector, _authority, bound = authority_input
    if change == "selector" and not bound:
        pytest.skip("unbound selectors have no fingerprint")
    win = windows_files.WindowsOS()
    record = root / "ancestry-settled.json"
    original_record, original_selector = record.read_bytes(), selector.read_bytes()
    admission = root / "admission"
    displaced = root / "displaced-admission"
    read_bytes = bootstrap._read_record_bytes.__code__
    changed = []
    registry_reads = 0

    def observe(frame, event, result):
        nonlocal registry_reads
        if event != "return" or changed:
            return
        if (
            frame.f_code is not read_bytes
            or frame.f_locals["name"] != "registry.json"
            or not isinstance(result, bytes)
        ):
            return
        registry_reads += 1
        if change in {"selector", "registry_helper"} and registry_reads != 2:
            return
        changed.append(True)
        if change == "control":
            record.write_bytes(original_record + b"\n")
        elif change == "parent":
            win.rename(admission, displaced)
            win.mkdir(admission, 0o700)
        elif change == "selector":
            selector.write_bytes(original_selector + b"\n")
        else:

            def unexpected(_parent):
                pytest.fail(
                    "selected read called a substituted final registry predicate"
                )

            monkeypatch.setattr(bootstrap, "_require_finished_registry", unexpected)

    leases = set(storage._live_leases)
    try:
        with _profile(observe):
            with pytest.raises(bootstrap.RecoveryRequired):
                activation._ordinary_activation_observation(root, selector)
    finally:
        if changed and change == "control":
            record.write_bytes(original_record)
        elif changed and change == "selector":
            selector.write_bytes(original_selector)
        elif displaced.exists():
            admission.rmdir()
            win.rename(displaced, admission)
    assert changed == [True]
    assert set(storage._live_leases) == leases


@pytest.mark.parametrize("binding", ["facade_body", "lock_shadow"])
def test_late_native_reader_substitution_refuses(authority_input, binding, monkeypatch):
    root, selector, _authority, _bound = authority_input
    assert activation._ordinary_native_readers_current()
    parser = bootstrap._parse_control_records.__code__
    substitutions = []

    def unexpected(*_args, **_kwargs):
        raise AssertionError("selected observation called a substituted native reader")

    def observe(frame, event, result):
        if event != "return" or frame.f_code is not parser or result is None:
            return
        substitutions.append(binding)
        if binding == "facade_body":
            patch.setattr(
                windows_files.WindowsOS.listdir, "__code__", unexpected.__code__
            )
        else:
            patch.setitem(vars(bootstrap.fcntl), "flock", unexpected)

    with monkeypatch.context() as patch, _profile(observe):
        with pytest.raises(
            bootstrap.RecoveryRequired, match="ordinary_activation_sources_changed"
        ):
            activation._ordinary_activation_observation(root, selector)
    assert substitutions == [binding]
