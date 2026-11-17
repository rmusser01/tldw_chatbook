"""Native descriptor retrieval preserves the current live-handle projection."""

import ctypes as C
import json
import os
import struct
from collections import Counter

import pytest

from tldw_chatbook.Utils import windows_files as files


pytestmark = pytest.mark.skipif(os.name != "nt", reason="actual Windows security queries")


def _gsi_projection(native, handle, directory):
    """Previous retrieval plus the unchanged current decoder, on this handle."""
    owner, dacl, descriptor = files._P(), files._P(), files._P()
    result = native.advapi.GetSecurityInfo(
        handle, 1, 5, C.byref(owner), None, C.byref(dacl), None, C.byref(descriptor)
    )
    if result:
        raise C.WinError(result)
    try:
        return native._decode_security(owner, dacl, directory, native.user_sid)
    finally:
        native.kernel.LocalFree(descriptor)


def _fresh_nt_bytes(handle):
    """Independent documented ABI oracle; no product helper is required for RED."""
    query = C.WinDLL("ntdll", use_last_error=True).NtQuerySecurityObject
    query.argtypes = [files._HANDLE, files._U32, files._P, files._U32, C.POINTER(files._U32)]
    query.restype = files._I32
    buffer, needed = C.create_string_buffer(65536), files._U32()
    status = query(handle, 5, buffer, len(buffer), C.byref(needed))
    assert status == 0 and 20 <= needed.value <= len(buffer)
    return buffer.raw[:needed.value]


def test_each_security_observation_queries_current_handle_natively(
    tmp_path, monkeypatch, record_property
):
    win = files.WindowsOS()
    fd = win.open(tmp_path / "native-query", win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    native = files._native()
    handle = native.handle(fd)
    security = native.security
    code = security.__func__.__code__
    counts, allocated = Counter(), set()
    gsi, nt, free = native.advapi.GetSecurityInfo, native.nt.NtQuerySecurityObject, native.kernel.LocalFree
    outputs = []

    def observed_gsi(*args):
        counts["GetSecurityInfo"] += 1
        result = gsi(*args)
        if result == 0:
            allocated.add(C.cast(args[-1], C.POINTER(files._P))[0])
        return result

    def observed_nt(*args):
        counts["NtQuerySecurityObject"] += 1
        assert args[0] == handle and args[1] == 5
        return nt(*args)

    def observed_free(pointer):
        address = C.cast(pointer, files._P).value
        result = free(pointer)
        if address in allocated and not result:
            allocated.remove(address)
            counts["descriptor_free"] += 1
        return result

    try:
        expected = _gsi_projection(native, handle, False)
        # The native private descriptor fits one initial query, so a count is
        # meaningful without depending on resize behavior or an absent helper.
        assert len(_fresh_nt_bytes(handle)) <= 512
        assert expected[1] == 0o600
        with monkeypatch.context() as patch:
            patch.setattr(native.advapi, "GetSecurityInfo", observed_gsi)
            patch.setattr(native.nt, "NtQuerySecurityObject", observed_nt)
            patch.setattr(native.kernel, "LocalFree", observed_free)
            for _ in range(3):
                outputs.append(security(handle, False))
        assert outputs == [expected] * 3
        assert not allocated and counts["descriptor_free"] == counts["GetSecurityInfo"]
        assert native.security.__func__.__code__ is code
    finally:
        win.close(fd)
        with pytest.raises(OSError):
            os.fstat(fd)
        assert not native.kernel.GetFileInformationByHandle(handle, C.byref(files._FileInfo()))
        assert C.get_last_error() == 6  # ERROR_INVALID_HANDLE, after actual close.
    record_property("native_security_query", json.dumps({
        "counts": dict(counts), "observations": len(outputs),
        "same_handle_projection": True, "descriptor_allocations_retired": not allocated,
        "native_handle_closed": True, "original_security_code": code.co_qualname,
    }, sort_keys=True))
    # Deliberately last: output, allocation and actual handle retirement above
    # must pass before the old retrieval primitive supplies the intended RED.
    assert (counts["NtQuerySecurityObject"], counts["GetSecurityInfo"]) == (3, 0), dict(counts)


def _set_dacl(native, handle, sddl):
    """Change only the disposable fixture's DACL through its retained handle."""
    descriptor, dacl = files._P(), files._P()
    try:
        if sddl is not None:
            native.check(native.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, C.byref(descriptor), None
            ))
            present, defaulted = files._I32(), files._I32()
            native.check(native.advapi.GetSecurityDescriptorDacl(
                descriptor, C.byref(present), C.byref(dacl), C.byref(defaulted)
            ))
            assert present.value
        result = native.advapi.SetSecurityInfo(
            handle, 1, 4 | 0x80000000, None, None, dacl, None
        )
        if result:
            raise C.WinError(result)
    finally:
        if descriptor.value:
            native.kernel.LocalFree(descriptor)


def _acl_shape(data):
    """Fixture census only: prove the requested ACE kind/flags/SID reached NTFS."""
    offset = struct.unpack_from("<I", data, 16)[0]
    if not offset:
        return ()
    size, count = struct.unpack_from("<HH", data, offset + 2)
    cursor, rows = offset + 8, []
    for _ in range(count):
        kind, flags, length = struct.unpack_from("<BBH", data, cursor)
        assert 8 <= length and cursor + length <= offset + size <= len(data)
        rows.append((kind, flags, data[cursor + 8:cursor + length]))
        cursor += length
    return tuple(rows)


def _assert_fresh_projection(native, handle, directory):
    reference = _gsi_projection(native, handle, directory)
    actual = native.security(handle, directory, with_descriptor=True)
    queried = _fresh_nt_bytes(handle)
    length = native.advapi.GetSecurityDescriptorLength(C.create_string_buffer(queried))
    assert 20 <= length <= len(queried) <= 4096
    assert actual[:2] == reference
    # GSI may add inheritance bookkeeping. Compare projection against GSI,
    # and the actual returned descriptor only against the fresh NT bytes.
    assert actual[2] == queried[:length]
    return actual


@pytest.mark.parametrize("directory", [False, True], ids=["file", "directory"])
@pytest.mark.parametrize("acl", ["private", "inherited", "deny", "null", "owner-rights"])
def test_native_query_matches_current_projection_through_live_acl_changes(
    tmp_path, directory, acl
):
    from Tests.Backup_Recovery.windows_acl_fixture import _preserve_dacl

    win, native = files.WindowsOS(), files._native()
    parent, selected = tmp_path / "private-parent", tmp_path / "private-parent" / "selected"
    win.mkdir(parent, 0o700)
    parent_fd = win.open(parent, win.O_RDONLY | win.O_DIRECTORY)
    rights = files._READ_CONTROL | files._WRITE_DAC
    user = native.user_sid
    private = f"D:P(A;;FA;;;{user})(A;;FA;;;SY)(A;;FA;;;BA)"
    try:
        with (
            native.reopen(native.handle(parent_fd), rights) as writable_parent,
            _preserve_dacl(native, writable_parent),
        ):
            if acl == "inherited":
                _set_dacl(native, writable_parent,
                    f"D:P(A;OICI;FA;;;{user})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FR;;;WD)")
                if directory:
                    selected.mkdir()
                else:
                    selected.write_bytes(b"parity")
            elif directory:
                win.mkdir(selected, 0o700)
            flags = win.O_RDONLY | win.O_DIRECTORY if directory else win.O_RDWR
            if not directory and acl != "inherited":
                flags |= win.O_CREAT | win.O_EXCL
            fd = win.open(selected, flags, 0o700 if directory else 0o600)
            handle = native.handle(fd)
            try:
                identity = native.info(handle)
                if not directory and acl != "inherited":
                    # The creation ACL has OI|CI flags, which SetSecurityInfo
                    # strips on files. Establish this explicit file fixture
                    # before the shared helper requires byte-exact restoration.
                    created = _assert_fresh_projection(native, handle, False)
                    with native.reopen(handle, rights) as writable:
                        _set_dacl(native, writable, private)
                    assert _assert_fresh_projection(native, handle, False)[:2] == created[:2]
                with native.reopen(handle, rights) as writable, _preserve_dacl(native, writable):
                    before = _assert_fresh_projection(native, handle, directory)
                    if acl == "inherited":
                        assert any(flag & 0x10 for _, flag, _ in _acl_shape(before[2]))
                        _set_dacl(native, writable, private)
                    elif acl == "null":
                        _set_dacl(native, writable, None)
                    elif acl == "deny":
                        _set_dacl(native, writable,
                            f"D:P(D;;FW;;;WD)(A;;FA;;;{user})(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;WD)")
                    elif acl == "owner-rights":
                        _set_dacl(native, writable, "D:P(A;;FA;;;OW)(A;;FA;;;SY)(A;;FA;;;BA)")
                    else:
                        _set_dacl(native, writable, private + "(A;;FR;;;WD)")
                    changed = _assert_fresh_projection(native, handle, directory)
                    assert changed[2] != before[2]
                    rows = _acl_shape(changed[2])
                    if acl == "null":
                        assert not rows and changed[1] == 0o777
                    elif acl == "deny":
                        assert any(kind == 1 for kind, _, _ in rows)
                    elif acl == "owner-rights":
                        owner_rights = b"\x01\x01\x00\x00\x00\x00\x00\x03\x04\x00\x00\x00"
                        assert any(sid == owner_rights for _, _, sid in rows)
                    elif acl == "private":
                        assert changed[1] & 0o044
                restored = _assert_fresh_projection(native, handle, directory)
                assert restored == before
                current = native.info(handle)
                assert (current.volume, current.index_high, current.index_low) == (
                    identity.volume, identity.index_high, identity.index_low
                )
            finally:
                win.close(fd)
                with pytest.raises(OSError):
                    os.fstat(fd)
                assert not native.kernel.GetFileInformationByHandle(handle, C.byref(files._FileInfo()))
                assert C.get_last_error() == 6
    finally:
        win.close(parent_fd)


@pytest.mark.parametrize("ace_count", [24, 128], ids=["resized-metadata", "oversized-ordinary"])
def test_native_query_grows_for_real_acl_and_retires_every_buffer(
    tmp_path, monkeypatch, ace_count
):
    import weakref

    from Tests.Backup_Recovery.windows_acl_fixture import _preserve_dacl

    win, native = files.WindowsOS(), files._native()
    fd = win.open(tmp_path / "large-acl", win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    handle = native.handle(fd)
    original_query = native.nt.NtQuerySecurityObject
    observations, buffers = [], []

    def observe_query(actual_handle, flags, buffer, size, needed):
        assert actual_handle == handle and flags == 5
        buffers.append(weakref.ref(buffer))
        result = original_query(actual_handle, flags, buffer, size, needed)
        length = C.cast(needed, C.POINTER(files._U32)).contents.value
        observations.append((size, length, result))
        return result

    try:
        before = _assert_fresh_projection(native, handle, False)
        identity = native.info(handle)
        private = f"D:P(A;;FA;;;{native.user_sid})(A;;FA;;;SY)(A;;FA;;;BA)"
        with native.reopen(handle, files._READ_CONTROL | files._WRITE_DAC) as writable:
            # GSI normalizes inherited file flags; start from an exact stable DACL.
            _set_dacl(native, writable, private)
            normalized = _assert_fresh_projection(native, handle, False)
            assert normalized[:2] == before[:2]
            with _preserve_dacl(native, writable):
                sddl = private
                # Distinct valid trustees prevent duplicate ACE coalescing.
                sddl += "".join(f"(A;;FR;;;S-1-5-21-1-2-3-{10000 + i})" for i in range(ace_count))
                _set_dacl(native, writable, sddl)
                queried = _fresh_nt_bytes(handle)
                size = native.advapi.GetSecurityDescriptorLength(C.create_string_buffer(queried))
                assert 20 <= size <= len(queried)
                assert len(_acl_shape(queried)) >= ace_count
                oversized = ace_count == 128
                if oversized:
                    assert 4096 < len(queried) < 8192
                else:
                    assert 512 < len(queried) < 4096
                expected = _gsi_projection(native, handle, False)
                cache_before = native._decoded_security.cache_info()
                with monkeypatch.context() as patch:
                    patch.setattr(native.nt, "NtQuerySecurityObject", observe_query)
                    for _ in range(2):
                        actual = native.security(handle, False, with_descriptor=not oversized)
                        assert actual == (expected if oversized else (*expected, queried[:size]))
                    if oversized:
                        with pytest.raises(OSError, match="windows_admission_descriptor_unavailable"):
                            native.security(handle, False, with_descriptor=True)
                assert native.nt.NtQuerySecurityObject is original_query
                assert _fresh_nt_bytes(handle) == queried
                assert all(buffer() is None for buffer in buffers)
                assert len(observations) == (6 if oversized else 4)
                for first, second in zip(observations[::2], observations[1::2]):
                    assert first[0:2] == (512, len(queried))
                    assert first[2] in {-1073741789, -2147483643}
                    assert second == (len(queried), len(queried), 0)
                if oversized:
                    assert native._decoded_security.cache_info() == cache_before
            assert _assert_fresh_projection(native, handle, False) == normalized
        current = native.info(handle)
        assert (current.volume, current.index_high, current.index_low) == (
            identity.volume, identity.index_high, identity.index_low
        )
    finally:
        win.close(fd)
        with pytest.raises(OSError):
            os.fstat(fd)
        assert not native.kernel.GetFileInformationByHandle(handle, C.byref(files._FileInfo()))
        assert C.get_last_error() == 6
