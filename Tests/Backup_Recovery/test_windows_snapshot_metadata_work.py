"""Measure the unchanged Windows tree's native call schedule, no App required."""

from collections import Counter
from contextlib import contextmanager
import inspect
import os
import sys
import threading

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from tldw_chatbook.Utils import windows_files

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(
        os.name != "nt", reason="Actual Windows native metadata control"
    ),
]


@contextmanager
def _metadata_counts(native, facade):
    members = [
        (windows_files._Native, name)
        for name in ("open_handle", "info", "ntfs", "security", "_token_sid")
    ]
    members += [
        (windows_files.WindowsOS, name)
        for name in ("stat_many_for_admission", "_stat_handle")
    ]
    members.append((windows_files, "_component"))
    functions = {
        (owner, name): inspect.getattr_static(owner, name) for owner, name in members
    }
    counts, caller_counts, handles = Counter(), Counter(), []
    actor = threading.current_thread()
    witness = OriginalStorageUnitObserver({}, lambda: True, lambda unit: None)
    for (owner, name), function in functions.items():
        witness._pin(function)
        witness.slots.append((owner, name, function))
    witness.codes = {
        function.__code__: name for (_owner, name), function in functions.items()
    }
    open_code = functions[windows_files._Native, "open_handle"].__code__
    info_code = functions[windows_files._Native, "info"].__code__
    token_code = functions[windows_files._Native, "_token_sid"].__code__
    stat_code = functions[windows_files.WindowsOS, "_stat_handle"].__code__
    snapshot_code = functions[
        windows_files.WindowsOS, "stat_many_for_admission"
    ].__code__
    component_code = windows_files._component.__code__
    named_code = next(
        code
        for code in snapshot_code.co_consts
        if inspect.iscode(code) and code.co_name == "named_handle"
    )
    monitor = sys.monitoring
    tool = next(
        slot
        for slot in range(5, 0, -1)
        if slot != monitor.DEBUGGER_ID and monitor.get_tool(slot) is None
    )
    monitor.use_tool_id(tool, "original-windows-metadata-schedule")
    witness.tool, witness.active, witness.installed = tool, True, True

    def started(code, offset):
        if threading.current_thread() is not actor:
            return
        witness._start(code, offset)
        frame = witness._frame(code)
        assert threading.current_thread() is actor
        if code is component_code:
            if frame.f_back.f_code is snapshot_code:
                counts["preflight_component"] += 1
            return
        expected = facade if code in {stat_code, snapshot_code} else native
        assert frame.f_locals.get("self") is expected
        counts[witness.codes[code]] += 1
        if code is info_code:
            parent = frame.f_back
            label = {
                open_code: "open_validation",
                named_code: "identity_record",
                stat_code: "fresh_stat",
            }.get(parent.f_code)
            assert label is not None
            caller_counts[label] += 1
        if code is token_code:
            assert frame.f_locals.get("information_class") == 4

    def returned(code, offset, value):
        if threading.current_thread() is not actor:
            return
        witness._return(code, offset, value)
        if code is open_code:
            assert type(value) is int  # noqa: E721 - returned native HANDLE value.
            handles.append(value)

    witness.registered = {
        monitor.events.PY_START: started,
        monitor.events.PY_RETURN: returned,
    }
    receipt = None
    try:
        for event, callback in witness.registered.items():
            assert monitor.register_callback(tool, event, callback) is None
        for code in witness.codes:
            monitor.set_local_events(
                tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
            )
        assert monitor.get_events(tool) == 0
        yield counts, caller_counts, handles
    finally:
        receipt = witness.close()
        assert receipt["complete"] and receipt["original_source_current"], receipt
        assert (
            receipt["global_events"] == 0 and receipt["hooks_retired_before_inactive"]
        ), receipt
        assert monitor.get_tool(tool) is None


def _independent_owner_sid(path):
    """Owner SID text via GetFileSecurityW, independent of the facade under test."""
    import ctypes as C
    from ctypes import wintypes as W

    advapi = C.WinDLL("advapi32", use_last_error=True)
    kernel = C.WinDLL("kernel32", use_last_error=True)
    advapi.GetFileSecurityW.argtypes = [
        W.LPCWSTR,
        W.DWORD,
        C.c_void_p,
        W.DWORD,
        C.POINTER(W.DWORD),
    ]
    advapi.GetSecurityDescriptorOwner.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_void_p),
        C.POINTER(W.BOOL),
    ]
    advapi.ConvertSidToStringSidW.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
    kernel.LocalFree.argtypes = [C.c_void_p]
    needed = W.DWORD()
    advapi.GetFileSecurityW(str(path), 1, None, 0, C.byref(needed))
    buffer = C.create_string_buffer(needed.value)
    assert advapi.GetFileSecurityW(
        str(path), 1, buffer, needed, C.byref(needed)
    ), C.get_last_error()
    owner, defaulted = C.c_void_p(), W.BOOL()
    assert advapi.GetSecurityDescriptorOwner(buffer, C.byref(owner), C.byref(defaulted))
    text = C.c_void_p()
    assert advapi.ConvertSidToStringSidW(owner, C.byref(text))
    try:
        return C.wstring_at(text)
    finally:
        kernel.LocalFree(text)


def test_original_tree_snapshot_counts_and_retires_actual_handles(tmp_path, request):
    directory = tmp_path / "metadata-tree"
    directory.mkdir()
    selected = tuple(directory / name for name in ("left.bin", "right.bin"))
    for path in selected:
        path.write_bytes(b"metadata-control")
    native, facade = windows_files._native(), windows_files.WindowsOS()
    nodes = {node for path in selected for node in (*path.parents, path)}
    with _metadata_counts(native, facade) as (counts, callers, handles):
        observed = facade.stat_many_for_admission(selected)
    # Original code retired every snapshot handle before the callback mask and
    # tool were retired. Verify physical absence without trying another close.
    for handle in handles:
        with pytest.raises(OSError) as error:
            native.info(handle)
        assert error.value.winerror == 6
    assert set(observed) == set(selected)
    for path in selected:
        assert observed[path][0].st_ino == os.stat(path).st_ino
        assert type(observed[path][1]) is bytes  # noqa: E721 - exact copied descriptor bytes.
    n = len(nodes)
    request.node.user_properties.append(
        (
            "original_windows_metadata_schedule",
            {
                "nodes": n,
                "starts": dict(counts),
                "info_callers": dict(callers),
                "opened_incarnations": len(handles),
                "physical_handles_retired": True,
                "actor": "actual_snapshot_caller",
                "observer_complete": True,
                "global_events": 0,
                "hooks_retired": True,
            },
        )
    )
    # TokenOwner is read fresh exactly for nodes whose projection depends on it:
    # an administrative owner that is not the token user (independent oracle).
    token_owner_reads = sum(
        1
        for node in nodes
        if (owner := _independent_owner_sid(node)) != native.user_sid
        and owner in windows_files._SYSTEM_SIDS
    )
    assert counts == Counter(
        preflight_component=sum(node.parent != node for node in nodes),
        open_handle=2 * n,
        info=5 * n,
        ntfs=2 * n,
        security=n,
        _token_sid=token_owner_reads,
        stat_many_for_admission=1,
        _stat_handle=n,
    )
    assert callers == Counter(
        open_validation=2 * n, identity_record=2 * n, fresh_stat=n
    )


@pytest.mark.parametrize("ancestor", ["NUL", "bad:stream", "trailing.", "bad\x01name"])
def test_snapshot_refuses_invalid_ancestor_before_native_open(
    tmp_path, monkeypatch, ancestor
):
    def unexpected_native():
        pytest.fail("invalid ancestor reached the native filesystem")

    monkeypatch.setattr(windows_files, "_native", unexpected_native)
    selected = tmp_path / ancestor / "valid-leaf.bin"
    with pytest.raises(ValueError, match="(?:invalid|reserved)_windows_component"):
        windows_files.WindowsOS().stat_many_for_admission((selected,))
