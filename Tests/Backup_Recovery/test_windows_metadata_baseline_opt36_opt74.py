"""Native baseline observations before selecting OPT36/74.

No optimization or Windows admission reuse is enabled by these controls.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import CodeType

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Utils import windows_files as files

pytestmark = pytest.mark.skipif(os.name != "nt", reason="actual Windows baseline")


class _ObserverSourceChanged(AssertionError):
    pass


def _defaults(value):
    if isinstance(value, dict):
        return tuple((key, _defaults(item)) for key, item in sorted(value.items()))
    if isinstance(value, tuple):
        return tuple(_defaults(item) for item in value)
    return (type(value).__name__, value)


@contextmanager
def _native_returns(win):
    """Observe actual native allocation/close and original Python returns."""
    native = files._native()
    kernel, nt = native.kernel, native.nt
    token_user_sid = native.user_sid
    source = Path(files.__file__).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    lf_hash = "6b665fa28c2b3916a7db03d25877d098219edc3de57d53f0c5d649454a4eafad"
    crlf_hash = "af62e79f163c4f43bebcac238dd315f177719b22e12c87437039b520363a29e8"
    assert source_hash in (lf_hash, crlf_hash)
    # Only these two exact, separately bound checkout representations are accepted.
    canonical = source if source_hash == lf_hash else source.replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == lf_hash
    compiled = compile(source, files.__file__, "exec", dont_inherit=True)
    original_classes = {
        code.co_name: code for code in compiled.co_consts if isinstance(code, CodeType)
    }
    specifications = (
        ("open", type(native), native, "open_handle"),
        ("info", type(native), native, "info"),
        ("ntfs", type(native), native, "ntfs"),
        ("security", type(native), native, "security"),
        ("stat", type(win), win, "_stat_handle"),
        ("batch", type(win), win, "stat_many_for_admission"),
        ("component", files, None, "_component"),
        (None, type(native), native, "_decoded_security"),
        (None, type(native), native, "_decode_security"),
        (None, type(native), native, "sid_string"),
        (None, type(native), native, "check"),
        (None, type(native), native, "ntcheck"),
    )
    slots, names, actors = [], {}, {}
    for label, owner, actor, attribute in specifications:
        fn = getattr(owner, attribute)
        container = original_classes[owner.__name__] if isinstance(owner, type) else compiled
        expected = next(code for code in container.co_consts if isinstance(code, CodeType) and code.co_name == attribute)
        original = inspect.unwrap(fn)
        assert original.__code__ == expected
        assert (original.__code__.co_filename, original.__code__.co_qualname, original.__code__.co_firstlineno) == (expected.co_filename, expected.co_qualname, expected.co_firstlineno)
        live = getattr(actor, attribute) if actor is not None else fn
        assert getattr(live, "__func__", live) is fn
        slots.append((owner, actor, attribute, fn, original, original.__code__, _defaults(original.__defaults__), _defaults(original.__kwdefaults__)))
        if label is not None:
            names[original.__code__], actors[original.__code__] = label, actor
    counts, ancestry, errors, births, closes, excluded = Counter(), Counter(), [], [], [], []
    current = {}
    security_observations = []
    caller = threading.get_ident()
    real_open, real_close = nt.NtCreateFile, kernel.CloseHandle
    api_shapes = (tuple(real_open.argtypes), real_open.restype, tuple(real_close.argtypes), real_close.restype)

    def source_current():
        for owner, actor, attribute, fn, original, code, defaults, kwdefaults in slots:
            live = getattr(actor, attribute) if actor is not None else getattr(owner, attribute)
            if (getattr(owner, attribute) is not fn or getattr(live, "__func__", live) is not fn
                or original.__code__ is not code or _defaults(original.__defaults__) != defaults
                or _defaults(original.__kwdefaults__) != kwdefaults):
                return False
        return (native.kernel is kernel and native.nt is nt and native.user_sid == token_user_sid and kernel.CloseHandle is closed
                and nt.NtCreateFile is opened
                and (tuple(real_open.argtypes), real_open.restype, tuple(real_close.argtypes), real_close.restype) == api_shapes)

    def opened(*args):
        result = real_open(*args)
        try:
            if result >= 0:
                value = C.cast(args[0], C.POINTER(files._HANDLE))[0]
                record = {"handle": value, "status": result, "thread": threading.get_ident(), "closes": []}
                births.append(record)
                current[value] = record
        except BaseException as error:  # noqa: BLE001 - observer cannot prevent producer return/retirement.
            errors.append(type(error).__name__)
        return result

    def closed(handle):
        result = real_close(handle)
        last_error = C.get_last_error() if not result else 0
        try:
            value = handle.value if isinstance(handle, C.c_void_p) else int(handle)
            record = current.get(value)
            if record is None:
                errors.append("close-without-observed-birth")
            else:
                record["closes"].append((bool(result), last_error))
                closes.append(record)
                if result:
                    current.pop(value, None)
        except BaseException as error:  # noqa: BLE001 - always preserve the original close result.
            errors.append(type(error).__name__)
        return result

    monitor = sys.monitoring
    slot = next(number for number in range(5, 0, -1)
                if number != monitor.DEBUGGER_ID and monitor.get_tool(number) is None)
    monitor.use_tool_id(slot, "native-baseline-returns")

    def frame_for(code):
        frame = sys._getframe()
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        return frame

    def qualified_frame(code):
        frame = frame_for(code)
        if frame is None:
            errors.append("missing-frame")
            return None
        if threading.get_ident() != caller or (actors[code] is not None and frame.f_locals.get("self") is not actors[code]):
            excluded.append((names[code], threading.get_ident()))
            return None
        if not source_current():
            errors.append("source-changed")
            return None
        return frame

    def started(code, offset):
        try:
            if qualified_frame(code) is not None:
                counts[names[code]] += 1
        except BaseException as error:  # noqa: BLE001 - passive diagnostics settle after original drain.
            errors.append(type(error).__name__)

    def returned(code, offset, value):
        try:
            frame = qualified_frame(code)
            if frame is not None and names[code] == "info":
                ancestry[names.get(frame.f_back.f_code, "other")] += 1
            elif frame is not None and names[code] == "security":
                security_observations.append((frame.f_locals["self"] is native, frame.f_locals["self"].user_sid == token_user_sid, value[0], value[1]))
        except BaseException as error:  # noqa: BLE001 - passive diagnostics settle after original drain.
            errors.append(type(error).__name__)

    try:
        nt.NtCreateFile, kernel.CloseHandle = opened, closed
        assert source_current()
        monitor.register_callback(slot, monitor.events.PY_START, started)
        monitor.register_callback(slot, monitor.events.PY_RETURN, returned)
        for code in names:
            monitor.set_local_events(slot, code, monitor.events.PY_START | monitor.events.PY_RETURN)
        yield counts, ancestry, births, closes, errors, excluded, security_observations
    finally:
        primary = sys.exc_info()[1]
        unchanged = source_current()
        for code in names:
            monitor.set_local_events(slot, code, 0)
        monitor.register_callback(slot, monitor.events.PY_START, None)
        monitor.register_callback(slot, monitor.events.PY_RETURN, None)
        monitor.free_tool_id(slot)
        nt.NtCreateFile, kernel.CloseHandle = real_open, real_close
        if monitor.get_tool(slot) is not None:
            errors.append("monitor-not-retired")
        if not all(record["closes"] == [(True, 0)] for record in births) or current:
            errors.append("native-birth-not-retired-once")
        # No close/retry is issued here; query only after all original closes.
        valid = kernel.GetHandleInformation
        valid.argtypes, valid.restype = [files._HANDLE, C.POINTER(files._U32)], files._I32
        for handle in {record["handle"] for record in births}:
            flags = files._U32()
            if valid(handle, C.byref(flags)) or C.get_last_error() != 6:
                errors.append("post-drain-handle-still-valid")
        if not unchanged or errors:
            if primary is not None:
                primary.add_note("Native baseline observer mismatch after original drain.")
            else:
                raise _ObserverSourceChanged(str(errors))


def _posture(info):
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), stat.S_IMODE(info.st_mode), info.st_uid)


def test_current_full_ancestor_batch_schedule_and_retirement(tmp_path, record_property):
    win = files.WindowsOS()
    root, branch = tmp_path / "metadata-root", tmp_path / "metadata-root" / "branch"
    win.mkdir(root, 0o700)
    win.mkdir(branch, 0o700)
    leaves = (branch / "first.bin", branch / "second.bin")
    for leaf in leaves:
        fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
        win.close(fd)
    paths = (root, branch, *leaves)
    expected = {path: _posture(win.stat(path)) for path in paths}
    nodes = {node for path in paths for node in (*path.parents, path)}
    with _native_returns(win) as (counts, ancestry, births, closes, errors, excluded, security_observations):
        observed = win.stat_many_for_admission(paths)
    n = len(nodes)
    assert dict(counts) == {
        "batch": 1, "component": sum(len(node.parts) - 1 for node in nodes),
        "open": 2 * n, "info": 3 * n, "ntfs": 2 * n,
        "stat": n, "security": n,
    }
    assert dict(ancestry) == {"open": 2 * n, "stat": n}
    assert len(births) == len(closes) == 2 * n
    assert len({id(record) for record in births}) == 2 * n
    assert Counter(id(record) for record in births) == Counter(id(record) for record in closes)
    assert all(record["status"] >= 0 and record["thread"] == threading.get_ident() for record in births)
    assert all(record["closes"] == [(True, 0)] for record in births)
    assert not excluded
    assert list(observed) == list(paths)
    assert {path: _posture(value[0]) for path, value in observed.items()} == expected
    assert all(type(value[1]) is bytes and value[1] for value in observed.values())
    assert len(security_observations) == n
    assert all(current_owner and current_user for current_owner, current_user, uid, mode in security_observations)
    record_property("current_native_baseline", json.dumps({"nodes": n, "counts": dict(counts), "info_ancestry": dict(ancestry), "births_retired_once": len(births), "observer_errors": errors}))


def test_current_config_stamps_preserve_five_fields_order_duplicates_missing_empty(tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission

    win = files.WindowsOS()
    leaf = tmp_path / "stamped.bin"
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    win.close(fd)
    missing = tmp_path / "missing.bin"
    paths = (leaf, tmp_path, leaf, missing)
    expected = tuple(storage_admission._posture(path) for path in paths)
    assert config._user_data_dir_stamps(paths) == expected
    assert len(expected[0]) == 5 and expected[0] == expected[2]
    assert expected[3] is None
    assert config._user_data_dir_stamps(()) == ()


def _warm(monkeypatch):
    from tldw_chatbook import config

    original = config._resolve_user_data_dir
    resolutions = []

    def resolve():
        resolutions.append(True)
        return original()

    monkeypatch.setattr(config, "_resolve_user_data_dir", resolve)
    first = config.get_user_data_dir()
    for _ in range(3):
        assert config.get_user_data_dir() == first
    resolutions.clear()
    assert config.get_user_data_dir() == first
    assert not resolutions
    return first, resolutions


@pytest.mark.asyncio
@private_profile_test
def test_windows_memo_warm_reload_replacement_and_native_permissions(request, monkeypatch):
    from tldw_chatbook import config

    win = files.WindowsOS()
    user_dir, resolutions = _warm(monkeypatch)
    for _ in range(5):
        assert config.get_user_data_dir() == user_dir
    assert not resolutions
    # A real public ACL observation must force the original resolution/hardening.
    subprocess.run(["icacls", str(user_dir), "/grant", "*S-1-1-0:(R)"], check=True, capture_output=True)  # nosec B603 B607 -- fixed native tool and disposable private-profile directory.
    assert win.stat(user_dir).st_mode & 0o044
    assert config.get_user_data_dir() == user_dir
    assert resolutions
    assert stat.S_IMODE(win.stat(user_dir).st_mode) == 0o700
    before = win.stat(user_dir).st_ino
    win.rename(user_dir, user_dir.with_name(user_dir.name + ".old"))
    win.mkdir(user_dir, 0o700)
    resolutions.clear()
    assert config.get_user_data_dir() == user_dir
    assert resolutions and win.stat(user_dir).st_ino != before
    resolutions.clear()
    config.load_settings(force_reload=True)
    assert config.get_user_data_dir() == user_dir
    assert resolutions


@pytest.mark.asyncio
@private_profile_test
def test_windows_memo_refuses_public_ancestor_and_preserves_original_key_timing(request, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus

    win = files.WindowsOS()
    user_dir, resolutions = _warm(monkeypatch)
    ancestor = user_dir.parents[2]
    try:
        subprocess.run(["icacls", str(ancestor), "/grant", "*S-1-1-0:(W)"], check=True, capture_output=True)  # nosec B603 B607 -- fixed native tool and disposable private-profile directory.
        public_stat = win.stat(ancestor)
        assert public_stat.st_mode & 0o022
        with pytest.raises(PrivatePathError) as refused:
            config.get_user_data_dir()
        result = refused.value.result
        # The unchanged walk reports diagnostics from root plus traversed
        # components, while its actual selected lexical path retains the drive.
        assert result.lexical_path.is_absolute()
        assert result.lexical_path.parts[:len(ancestor.parts)] == ancestor.parts
        assert result.lexical_path.drive == ancestor.drive == user_dir.drive
        assert ancestor.drive and ancestor.is_absolute()
        assert result.status is PrivatePathStatus.UNSAFE_PARENT
        assert result.reason == "shared_writable_parent"
        expected_diagnostic = Path(ancestor.root, *ancestor.parts[1:])
        assert result.offender_path == expected_diagnostic
        assert result.offender_path.drive == ""
        assert result.offender_path.parts == expected_diagnostic.parts
        assert result.offender_mode == stat.S_IMODE(public_stat.st_mode)
        assert result.offender_mode & 0o022
        assert resolutions
    finally:
        win.chmod(ancestor, 0o700)
    # Change the generation after original key capture to distinguish publication timing.
    monkeypatch.setattr(config, "_USER_DATA_DIR_MEMO", None)
    original = config._resolve_user_data_dir
    captured = []

    def resolve():
        captured.append(config._user_data_dir_inputs()[0])
        result = original()
        monkeypatch.setattr(config, "_CONFIG_GENERATION", config._CONFIG_GENERATION + 1)
        return result

    monkeypatch.setattr(config, "_resolve_user_data_dir", resolve)
    config.get_user_data_dir.__wrapped__()
    assert captured
    assert config._USER_DATA_DIR_MEMO[0] == captured[0]
    assert config._user_data_dir_inputs()[0][1] != captured[0][1]


def test_current_config_custom_posture_affinity_arguments_and_oserror(monkeypatch, tmp_path):
    import threading

    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission

    paths = (tmp_path / "first", tmp_path / "second", tmp_path / "first")
    caller = threading.get_ident()
    calls = []
    value = (1, 2, stat.S_IFDIR, 0o700, 1000)

    def posture(path):
        calls.append((path, threading.get_ident()))
        return value

    monkeypatch.setattr(storage_admission, "_posture", posture)
    assert config._user_data_dir_stamps(paths) == (value, value, value)
    assert calls == [(path, caller) for path in paths]
    failure = OSError("original scalar failure")

    def refused(path):
        raise failure

    monkeypatch.setattr(storage_admission, "_posture", refused)
    assert config._user_data_dir_stamps(paths) is None
    assert config._user_data_dir_stamps(()) == ()


def test_current_config_scalar_native_schedule(tmp_path, record_property):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    win = storage.os
    leaf = tmp_path / 'scalar-baseline.bin'
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    win.close(fd)
    paths = (leaf, tmp_path, leaf)
    expected = tuple(storage._posture(path) for path in paths)
    with _native_returns(win) as (counts, ancestry, births, closes, errors, excluded, security_observations):
        actual = config._user_data_dir_stamps(paths)
    depth = sum(len(path.parts) for path in paths)
    assert actual == expected
    assert not errors
    assert dict(counts) == {'component': depth - len(paths), 'open': depth,
                            'info': depth + len(paths), 'ntfs': depth,
                            'stat': len(paths), 'security': len(paths)}
    assert dict(ancestry) == {'open': depth, 'stat': len(paths)}
    assert len(births) == len(closes) == depth and not excluded
    assert all(current_owner and current_user for current_owner, current_user, uid, mode in security_observations)
    record_property('current_scalar_config_baseline', json.dumps({'counts': dict(counts), 'info_ancestry': dict(ancestry), 'paths': len(paths), 'births': len(births)}))


@pytest.mark.parametrize('mutation', ['slot', 'body', 'kwdefault'])
def test_observer_detects_live_source_drift_after_original_drain(tmp_path, monkeypatch, mutation):
    win = files.WindowsOS()
    leaf = tmp_path / 'source-control.bin'
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    win.close(fd)
    with pytest.raises(_ObserverSourceChanged), _native_returns(win), monkeypatch.context() as patch:
        if mutation == 'slot':
            original = files._component
            patch.setattr(files, '_component', lambda name: original(name))
        elif mutation == 'body':
            original = files._component
            patch.setattr(original, '__code__', original.__code__.replace(co_name='changed-component-body'))
        else:
            original = type(files._native()).open_handle
            patch.setitem(original.__kwdefaults__, 'extra_access', False)
        win.stat_many_for_admission((leaf,))

def test_observer_excludes_real_foreign_thread_and_facade(tmp_path):
    win, other = files.WindowsOS(), files.WindowsOS()
    leaf = tmp_path / 'foreign-source.bin'
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    win.close(fd)
    outcomes = []
    failures = []

    def read():
        try:
            outcomes.append(other.stat_many_for_admission((leaf,)))
        except BaseException as error:  # noqa: BLE001 - retain exact worker error for the owning test.
            failures.append(error)

    with _native_returns(win) as (counts, ancestry, births, closes, errors, excluded, security_observations):
        thread = threading.Thread(target=read)
        try:
            thread.start()
            assert other.stat_many_for_admission(()) == {}
        finally:
            # Retain this exact worker through every body/error path; outer pytest bounds apply.
            if thread.ident is not None:
                thread.join()
        assert not thread.is_alive()
    assert not failures and outcomes[0][leaf] is not None
    assert counts['batch'] == 0 and counts['stat'] == 0
    assert excluded and len(births) == len(closes) > 0
    assert not ancestry and not errors and not security_observations


@pytest.mark.asyncio
@private_profile_test
def test_windows_bracketed_posture_drift(request, monkeypatch):
    from tldw_chatbook import config

    win = files.WindowsOS()
    user_dir, _ = _warm(monkeypatch)
    monkeypatch.setattr(config, '_USER_DATA_DIR_MEMO', None)
    original = config._resolve_user_data_dir
    before = win.stat(user_dir)

    def changed():
        result = original()
        subprocess.run(['icacls', str(user_dir), '/grant', '*S-1-1-0:(R)'], check=True, capture_output=True)  # nosec B603 B607 -- fixed native tool against the disposable private profile.
        return result

    monkeypatch.setattr(config, '_resolve_user_data_dir', changed)
    try:
        assert config.get_user_data_dir.__wrapped__() == user_dir
        assert win.stat(user_dir).st_mode & 0o044
        assert config._USER_DATA_DIR_MEMO is None
        assert (before.st_dev, before.st_ino) == (win.stat(user_dir).st_dev, win.stat(user_dir).st_ino)
    finally:
        win.chmod(user_dir, 0o700)


def test_current_scalar_and_batch_use_identical_requested_workload(tmp_path, record_property):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    win = storage.os
    root = tmp_path / "matched-root"
    branch = root / "branch"
    win.mkdir(root, 0o700)
    win.mkdir(branch, 0o700)
    first, second = branch / "first.bin", branch / "second.bin"
    for leaf in (first, second):
        fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
        win.close(fd)
    paths = (root, branch, first, second, first)
    with _native_returns(win) as scalar:
        scalar_value = config._user_data_dir_stamps(paths)
    with _native_returns(win) as batch:
        metadata = win.stat_many_for_admission(paths)
    batch_value = tuple(None if metadata[path] is None else _posture(metadata[path][0]) for path in paths)
    assert scalar_value == batch_value
    assert scalar_value[2] == scalar_value[4]
    assert all(len(value) == 5 for value in scalar_value)
    assert list(metadata) == list(dict.fromkeys(paths))
    depth = sum(len(path.parts) for path in paths)
    nodes = {node for path in paths for node in (*path.parents, path)}
    n = len(nodes)
    scalar_counts, scalar_ancestry, scalar_births, scalar_closes, scalar_errors, scalar_excluded, _ = scalar
    batch_counts, batch_ancestry, batch_births, batch_closes, batch_errors, batch_excluded, _ = batch
    assert dict(scalar_counts) == {"component": depth - len(paths), "open": depth, "info": depth + len(paths), "ntfs": depth, "stat": len(paths), "security": len(paths)}
    assert dict(batch_counts) == {"batch": 1, "component": sum(len(node.parts) - 1 for node in nodes), "open": 2 * n, "info": 3 * n, "ntfs": 2 * n, "stat": n, "security": n}
    assert dict(scalar_ancestry) == {"open": depth, "stat": len(paths)}
    assert dict(batch_ancestry) == {"open": 2 * n, "stat": n}
    assert len(scalar_births) == len(scalar_closes) == depth
    assert len(batch_births) == len(batch_closes) == 2 * n
    assert not scalar_errors and not batch_errors
    assert not scalar_excluded and not batch_excluded
    record_property("matched_native_workload", json.dumps({"paths": [str(path) for path in paths], "unique_nodes": n, "scalar_counts": dict(scalar_counts), "batch_counts": dict(batch_counts), "scalar_info_ancestry": dict(scalar_ancestry), "batch_info_ancestry": dict(batch_ancestry), "scalar_births": len(scalar_births), "batch_births": len(batch_births), "postures_equal": scalar_value == batch_value, "scope": "same requested workload baseline only; no optimization selection"}))



def test_real_large_private_descriptor_falls_back_only_after_physical_drain(tmp_path, record_property):
    """Test-only proposed representability fallback; shipping config unchanged."""
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    win, native = storage.os, files._native()
    leaf = tmp_path / "large-private-dacl.bin"
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    descriptor = files._P()
    observed_descriptor = files._P()
    owner, dacl = files._P(), files._P()
    sddl = "O:" + native.user_sid + "D:P" + ("(A;;FA;;;" + native.user_sid + ")") * 192
    scalar_calls, fallback_facts = [], []
    paths = (leaf, leaf)
    try:
        native.check(native.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, C.byref(descriptor), None))
        present, defaulted = files._I32(), files._I32()
        native.check(native.advapi.GetSecurityDescriptorDacl(
            descriptor, C.byref(present), C.byref(dacl), C.byref(defaulted)))
        assert present.value and dacl.value
        with native.reopen(native.handle(fd), files._READ_CONTROL | files._WRITE_DAC) as writable:
            result = native.advapi.SetSecurityInfo(writable, 1, 4 | 0x80000000, None, None, dacl, None)
            if result:
                raise C.WinError(result)
        actual_dacl = files._P()
        result = native.advapi.GetSecurityInfo(native.handle(fd), 1, 5, C.byref(owner), None,
                                             C.byref(actual_dacl), None, C.byref(observed_descriptor))
        if result:
            raise C.WinError(result)
        size = native.advapi.GetSecurityDescriptorLength(observed_descriptor)
        assert 4096 < size < 16384
        expected = config._user_data_dir_stamps(paths)
        assert expected is not None and expected[0] == expected[1]
        assert expected[0][3:] == (0o600, 1000)

        # The candidate adapter owns no product state and catches only the
        # concrete stock descriptor representability outcome. The context
        # manager finishes original physical probes before this except body.
        def proposed():
            ledger = None
            try:
                with _native_returns(win) as ledger:
                    values = win.stat_many_for_admission(paths)
                return tuple(None if values[path] is None else _posture(values[path][0]) for path in paths)
            except OSError as error:
                if type(error) is not OSError or error.errno != 13 or error.strerror != "windows_admission_descriptor_unavailable":
                    raise
                assert ledger is not None
                counts, _ancestry, births, closes, errors, excluded, _security = ledger
                assert not errors and not excluded and births
                assert len(births) == len(closes)
                assert all(birth['closes'] == [(True, 0)] for birth in births)
                fallback_facts.append({'size': size, 'native_births': len(births), 'counts': dict(counts),
                                       'original_error': [type(error).__name__, error.errno, error.strerror],
                                       'physical_drain_before_scalar': True})
                scalar_calls.append((paths, threading.get_ident()))
                return config._user_data_dir_stamps(paths)
        assert proposed() == expected
        assert scalar_calls == [(paths, threading.get_ident())]
        assert len(fallback_facts) == 1
        record_property('descriptor_fallback_after_original_drain', json.dumps(fallback_facts[0]))
    finally:
        if observed_descriptor.value:
            native.kernel.LocalFree(observed_descriptor)
        if descriptor.value:
            native.kernel.LocalFree(descriptor)
        # This continuously owned CRT descriptor was created by the fixture,
        # never handed to metadata retirement or retried as a stale HANDLE.
        try:
            win.fchmod(fd, 0o600)
        finally:
            win.close(fd)



@pytest.mark.parametrize("api", ["scalar", "batch"])
@pytest.mark.parametrize("outcome", ["protected_false", "raising"])
def test_proposed_fallback_propagates_exact_retained_native_close(tmp_path, record_property, api, outcome):
    """Retain actual continuous failed owner; only fixture can unprotect it."""
    from tldw_chatbook import config

    win, native = files.WindowsOS(), files._native()
    leaf = tmp_path / "retained-close.bin"
    fd = win.open(leaf, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    win.close(fd)
    paths = (leaf,)
    kernel = native.kernel
    real_close = kernel.CloseHandle
    get_flags, set_flags = kernel.GetHandleInformation, kernel.SetHandleInformation
    get_flags.argtypes, get_flags.restype = [files._HANDLE, C.POINTER(files._U32)], files._I32
    set_flags.argtypes, set_flags.restype = [files._HANDLE, files._U32, files._U32], files._I32
    selected, errors, calls = [], [], []
    protected = set()
    scalar_calls = []
    failure = RuntimeError("controlled defining close raised")
    producer_error = None
    producer_chain = None
    ledger = None
    caller = threading.get_ident()
    scalar = config._user_data_dir_stamps
    opener = type(native).open_handle
    monitor, slot = sys.monitoring, None

    def observed_close(handle):
        if selected and selected[0] in protected and handle == selected[0].handle:
            calls.append(handle)
            if outcome == "raising":
                raise failure
        return real_close(handle)
    observed_close.argtypes = real_close.argtypes
    observed_close.restype = real_close.restype

    def record_open(code, offset, value):
        frame = None
        try:
            frame = sys._getframe(1)
            assert code is opener.__code__ and frame.f_code is code
            assert frame.f_globals is opener.__globals__ and frame.f_locals['self'] is native
            record = frame.f_locals.get('_metadata_record')
            if record is None or record.path != leaf or selected:
                return
            assert type(record) is files._AdmissionMetadataHandle
            assert record.native is native and record.handle == value
            assert record.identity is not None and record.close_error is None
            assert threading.get_ident() == caller
            flags = files._U32()
            native.check(get_flags(record.handle, C.byref(flags)))
            assert not flags.value & 2
            selected.append(record)  # retain exact known birth before fixture mutation
            native.check(set_flags(record.handle, 2, 2))
            protected.add(record)
            native.check(get_flags(record.handle, C.byref(flags)))
            assert flags.value & 2
        except BaseException as error:  # noqa: BLE001 - passive test control records failure without replacing producer outcome.
            errors.append(type(error).__name__ + ':' + str(error))
        finally:
            frame = None

    def record_scalar(code, offset):
        frame = None
        try:
            frame = sys._getframe(1)
            assert code is scalar.__code__ and frame.f_code is code
            assert frame.f_globals is scalar.__globals__ and threading.get_ident() == caller
            scalar_calls.append(frame.f_locals['paths'])
        except BaseException as error:  # noqa: BLE001 - passive observer reports only after exact native cleanup.
            errors.append(type(error).__name__)
        finally:
            frame = None

    # Same proposed representability-only fallback as the descriptor fixture;
    # no CloseError/unknown OSError/source error is caught or converted to None.
    def proposed():
        nonlocal ledger, producer_error, producer_chain
        try:
            with _native_returns(win) as ledger:
                if api == 'scalar':
                    result = {leaf: win.stat_for_admission(leaf)}
                else:
                    result = win.stat_many_for_admission(paths)
            return tuple(None if result[path] is None else _posture(result[path][0]) for path in paths)
        except OSError as error:
            producer_error = error
            tail = error.__traceback__
            nodes = []
            cursor = tail
            while cursor is not None:
                nodes.append((cursor, cursor.tb_frame, cursor.tb_frame.f_code,
                              cursor.tb_lineno, cursor.tb_next))
                cursor = cursor.tb_next
            producer_chain = (error.__cause__, error.__context__,
                              error.__suppress_context__, tail, tuple(nodes))
            if type(error) is OSError and error.errno == 13 and error.strerror == 'windows_admission_descriptor_unavailable':
                return config._user_data_dir_stamps(paths)
            raise
    try:
        slot = next(n for n in range(5, 0, -1) if n != monitor.DEBUGGER_ID and monitor.get_tool(n) is None)
        monitor.use_tool_id(slot, 'actual-retained-close-control')
        assert monitor.register_callback(slot, monitor.events.PY_RETURN, record_open) is None
        assert monitor.register_callback(slot, monitor.events.PY_START, record_scalar) is None
        monitor.set_local_events(slot, opener.__code__, monitor.events.PY_RETURN)
        monitor.set_local_events(slot, scalar.__code__, monitor.events.PY_START)
        # This explicit defining API control either calls the actual native
        # CloseHandle (real protected FALSE) or raises before touching exactly
        # that still-protected continuous incarnation. Other births use native.
        kernel.CloseHandle = observed_close
        with pytest.raises(files._AdmissionMetadataCloseError) as caught:
            proposed()
        assert caught.value is producer_error and producer_chain is not None
        cause, context, suppression, original_tail, original_nodes = producer_chain
        assert caught.value.__cause__ is cause
        assert caught.value.__context__ is context
        assert caught.value.__suppress_context__ is suppression
        assert original_tail is not None and original_nodes
        cursor = caught.value.__traceback__
        caller_code = sys._getframe().f_code
        outer_frames = 0
        while cursor is not original_tail:
            assert cursor is not None and cursor.tb_frame.f_code is caller_code
            outer_frames += 1
            cursor = cursor.tb_next
        assert outer_frames == 1
        for node, frame, code, line, successor in original_nodes:
            assert cursor is node and cursor.tb_frame is frame
            assert cursor.tb_frame.f_code is code and cursor.tb_lineno == line
            assert cursor.tb_next is successor
            cursor = cursor.tb_next
        assert cursor is None
        assert len(selected) == len(calls) == 1 and not errors and not scalar_calls
        record = selected[0]
        assert caught.value.failed_handles == (record,)
        assert record.native is native and record.path == leaf and record.handle == calls[0]
        assert caught.value.errno == 5 and caught.value.strerror == 'windows_admission_metadata_not_retired'
        if outcome == 'raising':
            assert record.close_error is failure
        else:
            assert type(record.close_error) is int and record.close_error != 0
        flags = files._U32()
        native.check(get_flags(record.handle, C.byref(flags)))
        assert flags.value & 2
        info = native.info(record.handle)
        assert (info.volume, info.index_high << 32 | info.index_low) == record.identity
        _counts, _ancestry, births, _closes, observer_errors, excluded, _security = ledger
        retained = [birth for birth in births if birth['handle'] == record.handle]
        assert len(retained) == 1 and not excluded
        if outcome == 'protected_false':
            assert retained[0]['closes'] == [(False, record.close_error)]
        else:
            assert retained[0]['closes'] == []
        assert all(birth is retained[0] or birth['closes'] == [(True, 0)] for birth in births)
        assert observer_errors == ['native-birth-not-retired-once', 'post-drain-handle-still-valid']
        record_property('retained_close_without_fallback_or_retry', json.dumps({'api': api, 'outcome': outcome,
            'births': len(births), 'native_close_attempts_for_exact_owner': len(calls), 'scalar_fallbacks': len(scalar_calls),
            'same_failed_owner': True, 'protected_physical_owner_still_valid': True,
            'same_actual_cause_context_suppression': True, 'original_traceback_tail_preserved': True,
            'expected_outer_caller_frames': outer_frames,
            'observer_retention_expected_not_retirement_claim': observer_errors}))
    finally:
        # Restore observer/defining API after original metadata call fully
        # returned; it has no asynchronous worker or outstanding native call.
        kernel.CloseHandle = real_close
        if slot is not None:
            monitor.set_local_events(slot, opener.__code__, 0)
            monitor.set_local_events(slot, scalar.__code__, 0)
            monitor.register_callback(slot, monitor.events.PY_RETURN, None)
            monitor.register_callback(slot, monitor.events.PY_START, None)
            monitor.free_tool_id(slot)
        if selected and selected[0] in protected:
            record = selected[0]
            flags = files._U32()
            native.check(get_flags(record.handle, C.byref(flags)))
            assert flags.value & 2 and record.native is native
            info = native.info(record.handle)
            assert (info.volume, info.index_high << 32 | info.index_low) == record.identity
            native.check(set_flags(record.handle, 2, 0))
            native.check(real_close(record.handle))
            assert not get_flags(record.handle, C.byref(flags)) and C.get_last_error() == 6
