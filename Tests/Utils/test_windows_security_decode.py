"""Fresh native security observations may reuse only identical byte decoding."""

import ctypes as C
import os
import struct
import weakref
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_chatbook.Utils import windows_files as files


def _sid(identifier):
    return b"\x01\x01\x00\x00\x00\x00\x00\x05" + struct.pack("<I", identifier)


def _descriptor(*, owner=1000, trustee=1000, mask=1, flags=0, padding=0, relative=True):
    owner_bytes = _sid(owner)
    ace = struct.pack("<BBHI", 0, flags, 20, mask) + _sid(trustee)
    acl = struct.pack("<BBHHH", 2, 0, 8 + len(ace), 1, 0) + ace
    return (struct.pack("<BBHIIII", 1, 0, 0x8004 if relative else 4, 20, 0, 0, 32)
            + owner_bytes + acl + b"\0" * padding)


@pytest.fixture
def native(monkeypatch):
    """Controlled valid buffers exercise the ABI seam on every Python platform."""
    api = object.__new__(files._Native)
    state = SimpleNamespace(
        data=_descriptor(), buffers=[], counts=Counter(), error=None,
        responses=[], sizes=[], length=None,
    )
    api.user_sid = "S-1-5-1000"

    def address(pointer):
        return C.cast(pointer, C.c_void_p).value

    def assign(pointer, value):
        C.cast(pointer, C.POINTER(C.c_void_p))[0] = value

    def owner(descriptor, output, defaulted):
        if state.error == "owner":
            return 0
        state.counts["owner"] += 1
        offset = struct.unpack_from("<I", C.string_at(descriptor, 20), 4)[0]
        assign(output, address(descriptor) + offset)
        return 1

    def dacl(descriptor, present, output, defaulted):
        if state.error == "dacl" and present is not None:
            return 0
        offset = struct.unpack_from("<I", C.string_at(descriptor, 20), 16)[0]
        assign(output, address(descriptor) + offset if offset else None)
        return 1

    def query(handle, flags, buffer, size, needed):
        assert handle == 7 and flags == 5
        state.counts["query"] += 1
        state.sizes.append(size)
        state.buffers.append(weakref.ref(buffer))
        required = len(state.data)
        if state.responses:
            status, required = state.responses.pop(0)
        elif state.error == "acquire":
            status = -1073741790  # STATUS_ACCESS_DENIED
        else:
            status = -1073741789 if size < required else 0  # STATUS_BUFFER_TOO_SMALL
        C.cast(needed, C.POINTER(files._U32))[0] = required
        if status == 0:
            assert len(state.data) <= size
            C.memmove(buffer, state.data, len(state.data))
            state.counts["acquire"] += 1
        return status

    def free(pointer):
        state.counts["free"] += 1
        raise AssertionError("call-local query buffers must not be passed to LocalFree")

    def length(pointer):
        state.counts["length"] += 1
        return len(state.data) if state.length is None else state.length

    def sid_string(pointer):
        state.counts["sid"] += 1
        if state.error == "decode":
            raise ValueError("controlled SID conversion failure")
        return "S-1-5-" + str(struct.unpack_from("<I", C.string_at(pointer, 12), 8)[0])

    def ace(dacl_pointer, index, output):
        if state.error == "ace":
            return 0
        assert index == 0
        state.counts["ace"] += 1
        assign(output, address(dacl_pointer) + 8)
        return 1

    def check(result):
        if not result:
            raise OSError("controlled native failure")
        return result

    api.advapi = SimpleNamespace(
        GetSecurityDescriptorLength=length,
        GetSecurityDescriptorOwner=owner, GetSecurityDescriptorDacl=dacl, GetAce=ace,
    )
    api.nt = SimpleNamespace(NtQuerySecurityObject=query, RtlNtStatusToDosError=lambda status: 5)
    api.kernel = SimpleNamespace(LocalFree=free)
    api.sid_string = sid_string
    api.check = check
    monkeypatch.setattr(files.C, "WinError", lambda code: PermissionError(code, "native refusal"), raising=False)
    yield api, state
    assert all(buffer() is None for buffer in state.buffers)
    assert state.counts["free"] == 0


def test_repeated_bytes_still_query_each_native_descriptor(native):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    first = state.counts.copy()
    assert api.security(7, False) == (1000, 0o600)
    assert state.counts["query"] == state.counts["acquire"] == 2
    assert state.counts["sid"] == first["sid"]
    assert state.counts["ace"] == first["ace"]


def test_changed_permissions_are_observed_on_the_same_handle(native):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    state.data = _descriptor(trustee=2000)
    assert api.security(7, False) == (1000, 0o644)
    state.data = _descriptor(owner=2000, trustee=2000)
    assert api.security(7, False) == (-1, 0o644)
    state.data = _descriptor()
    assert api.security(7, False) == (1000, 0o600)
    assert state.counts["query"] == state.counts["acquire"] == 4


def test_directory_flag_and_current_user_are_part_of_decode_identity(native):
    api, state = native
    state.data = _descriptor(trustee=2000, flags=8)
    assert api.security(7, False) == (1000, 0o600)
    assert api.security(7, True) == (1000, 0o744)
    api.user_sid = "S-1-5-2000"
    assert api.security(7, True) == (-1, 0o700)


def test_warm_decode_never_hides_a_fresh_native_acquisition_failure(native):
    api, state = native
    api.security(7, False)
    state.error = "acquire"
    with pytest.raises(PermissionError):
        api.security(7, False)
    assert state.counts["query"] == 2 and state.counts["acquire"] == 1


def test_failed_decode_is_not_cached_and_releases_the_acquired_descriptor(native):
    api, state = native
    state.error = "decode"
    with pytest.raises(ValueError, match="controlled SID"):
        api.security(7, False)
    state.error = None
    assert api.security(7, False) == (1000, 0o600)
    assert state.counts["query"] == state.counts["acquire"] == 2
    assert state.counts["sid"] == 3


@pytest.mark.parametrize("phase", ["owner", "dacl", "ace"])
def test_native_decoder_refusal_is_not_cached_or_replaced_by_an_older_result(native, phase):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    state.data = _descriptor(trustee=2000)
    state.error = phase
    with pytest.raises(OSError, match="controlled native"):
        api.security(7, False)
    state.error = None
    assert api.security(7, False) == (1000, 0o644)
    assert state.counts["query"] == state.counts["acquire"] == 3


def test_large_descriptors_keep_the_original_uncached_decoder(native):
    api, state = native
    state.data = _descriptor(padding=4096)
    assert api.security(7, False) == api.security(7, False) == (1000, 0o600)
    assert state.counts["sid"] == 4
    assert state.counts["acquire"] == 2 and state.counts["query"] == 4


def test_successful_decode_retention_has_a_fixed_entry_bound(native):
    api, state = native
    for index in range(160):
        state.data = _descriptor(owner=3000 + index)
        assert api.security(7, False) == (-1, 0o600)
    assert api._decoded_security.cache_info().currsize <= 128


@pytest.mark.parametrize("change,expected", [("null_dacl", 0o777), ("unknown_ace", 0o677)])
def test_changed_null_or_unknown_acl_remains_public(native, change, expected):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    changed = bytearray(state.data)
    if change == "null_dacl":
        struct.pack_into("<I", changed, 16, 0)
    else:
        changed[40] = 9
    state.data = bytes(changed)
    assert api.security(7, False) == (1000, expected)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows security acquisition")
def test_real_warm_decode_cannot_hide_invalid_handle(tmp_path):
    windows = files.WindowsOS()
    descriptor = windows.open(tmp_path / "invalid-handle", windows.O_CREAT | windows.O_EXCL | windows.O_RDWR, 0o600)
    try:
        native = files._native()
        assert native.security(native.handle(descriptor), False) == (1000, 0o600)
        with pytest.raises(OSError):
            # -1 is a process pseudo-handle; NULL is not an acquired file handle.
            native.security(0, False)
    finally:
        windows.close(descriptor)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows identities")
def test_equal_acl_decode_does_not_hide_replaced_path_identity(tmp_path):
    windows = files.WindowsOS()
    path = tmp_path / "original"
    descriptor = windows.open(path, windows.O_CREAT | windows.O_EXCL | windows.O_RDWR, 0o600)
    replacement = None
    try:
        original = windows.fstat(descriptor)
        windows.rename(path, tmp_path / "retained")
        replacement = windows.open(path, windows.O_CREAT | windows.O_EXCL | windows.O_RDWR, 0o600)
        current = windows.stat(path)
        assert current.st_mode == original.st_mode and current.st_uid == original.st_uid
        assert (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)
        assert windows.fstat(descriptor).st_ino == original.st_ino
    finally:
        if replacement is not None:
            windows.close(replacement)
        windows.close(descriptor)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows decode cost and allocation counts")
def test_native_decode_reuse_preserves_fresh_acquisition_and_records_cost(tmp_path, monkeypatch, record_property):
    import json
    import statistics
    import time

    windows = files.WindowsOS()
    descriptor = windows.open(tmp_path / "measured-security", windows.O_CREAT | windows.O_EXCL | windows.O_RDWR, 0o600)
    native = files._native()
    real = native.advapi
    nt = native.nt
    counts = Counter()

    class Observed:
        def __getattr__(self, name):
            function = getattr(real, name)
            if name not in {"GetSecurityInfo", "GetAce", "ConvertSidToStringSidW"}:
                return function
            def call(*args):
                counts[name] += 1
                return function(*args)
            return call

    class ObservedNt:
        def __getattr__(self, name):
            function = getattr(nt, name)
            if name != "NtQuerySecurityObject":
                return function
            def query(*args):
                counts[name] += 1
                return function(*args)
            return query

    def uncached(handle):
        return native._decode_descriptor(
            native.security_descriptor(handle), False, native.user_sid
        )

    try:
        handle = native.handle(descriptor)
        native._decoded_security.cache_clear()
        with monkeypatch.context() as patch:
            patch.setattr(native, "advapi", Observed())
            patch.setattr(native, "nt", ObservedNt())
            assert native.security(handle, False) == (1000, 0o600)
            first = counts.copy()
            for _ in range(10):
                assert native.security(handle, False) == (1000, 0o600)
            assert counts["NtQuerySecurityObject"] == 11
            assert counts["GetSecurityInfo"] == 0
            assert counts["GetAce"] == first["GetAce"]
            assert counts["ConvertSidToStringSidW"] == first["ConvertSidToStringSidW"]
        samples = {"cached": [], "uncached": []}
        for _ in range(3):
            for label, read in (("uncached", uncached), ("cached", lambda handle: native.security(handle, False))):
                started = time.perf_counter()
                for _ in range(200):
                    assert read(handle) == (1000, 0o600)
                samples[label].append(time.perf_counter() - started)
        record_property("fresh_security_decode", json.dumps({
            "calls": dict(counts), "iterations_per_sample": 200, "samples": samples,
            "median_seconds": {name: statistics.median(values) for name, values in samples.items()},
        }, sort_keys=True))
    finally:
        windows.close(descriptor)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows handles and ACL mutation")
def test_real_warm_acl_observes_public_grant_and_hardening(tmp_path):
    import stat
    import subprocess  # nosec B404 -- fixed icacls command against a disposable owned file below.

    windows = files.WindowsOS()
    path = tmp_path / "fresh-permissions"
    descriptor = windows.open(path, windows.O_CREAT | windows.O_EXCL | windows.O_RDWR, 0o600)
    try:
        before = windows.fstat(descriptor)
        assert stat.S_IMODE(windows.fstat(descriptor).st_mode) == 0o600
        subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:(R)"], check=True, capture_output=True)  # nosec B603 B607 -- fixed native tool, disposable owned file
        assert windows.fstat(descriptor).st_mode & 0o044
        windows.fchmod(descriptor, 0o600)
        after = windows.fstat(descriptor)
        assert stat.S_IMODE(after.st_mode) == 0o600
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    finally:
        windows.close(descriptor)


def test_admission_descriptor_is_fresh_and_copied_from_call_local_buffer(native):
    api, state = native
    first = state.data
    assert api.security(7, False, with_descriptor=True) == (1000, 0o600, first)
    assert all(buffer() is None for buffer in state.buffers)
    state.data = _descriptor(trustee=999)
    assert api.security(7, False, with_descriptor=True) == (1000, 0o644, state.data)
    assert state.counts["query"] == state.counts["acquire"] == 2
    assert first != state.data
    # Ordinary two-positional-argument return shape remains unchanged.
    assert api.security(7, False) == (1000, 0o644)


def test_admission_oversized_descriptor_refuses_but_ordinary_stays_compatible(native):
    api, state = native
    state.data = _descriptor(padding=5000)
    with pytest.raises(OSError, match="windows_admission_descriptor_unavailable"):
        api.security(7, False, with_descriptor=True)
    assert all(buffer() is None for buffer in state.buffers)
    assert api.security(7, False) == (1000, 0o600)


def test_exact_metadata_descriptor_bound_keeps_byte_only_decode_cache(native):
    api, state = native
    state.data = _descriptor(padding=4096 - len(_descriptor()))
    assert len(state.data) == 4096
    assert api.security(7, False, with_descriptor=True) == (1000, 0o600, state.data)
    first = state.counts.copy()
    assert api.security(7, False, with_descriptor=True) == (1000, 0o600, state.data)
    assert state.counts["query"] == 4 and state.counts["acquire"] == 2
    assert state.counts["sid"] == first["sid"]


@pytest.mark.parametrize("status", [-1073741789, -2147483643])
def test_native_buffer_growth_retries_only_with_a_larger_bounded_length(native, status):
    api, state = native
    state.responses = [(status, 1024)]
    assert api.security(7, False) == (1000, 0o600)
    assert state.sizes == [512, 1024]
    assert state.counts["query"] == 2 and state.counts["acquire"] == 1


@pytest.mark.parametrize("required", [0, 512, 65537])
def test_native_short_buffer_refuses_invalid_growth(native, required):
    api, state = native
    state.responses = [(-1073741789, required)]
    with pytest.raises(PermissionError):
        api.security(7, False)
    assert state.sizes == [512] and state.counts["length"] == 0


def test_native_descriptor_growth_has_a_finite_retry_bound(native):
    api, state = native
    state.responses = [(-1073741789, size) for size in (1024, 2048, 4096, 8192)]
    with pytest.raises(OSError, match="windows_security_descriptor_unstable"):
        api.security(7, False)
    assert state.sizes == [512, 1024, 2048, 4096]
    assert state.counts["length"] == state.counts["acquire"] == 0


def test_largest_native_descriptor_keeps_ordinary_uncached_projection(native):
    api, state = native
    state.data = _descriptor(padding=65536 - len(_descriptor()))
    assert api.security(7, False) == api.security(7, False) == (1000, 0o600)
    assert state.sizes == [512, 65536, 512, 65536]
    assert state.counts["sid"] == 4


@pytest.mark.parametrize("required", [0, 19, 513])
def test_success_with_invalid_returned_length_fails_before_native_decode(native, required):
    api, state = native
    state.responses = [(0, required)]
    with pytest.raises(OSError, match="windows_security_descriptor_invalid"):
        api.security(7, False)
    assert state.counts["length"] == state.counts["owner"] == 0


@pytest.mark.parametrize("length", [0, 19, 61])
def test_descriptor_length_cannot_escape_fresh_returned_bytes(native, length):
    api, state = native
    assert len(state.data) == 60
    state.length = length
    with pytest.raises(OSError, match="windows_security_descriptor_invalid"):
        api.security(7, False)
    assert state.counts["length"] == 1 and state.counts["owner"] == 0


@pytest.mark.parametrize("metadata", [False, True])
def test_nonrelative_nt_result_is_invalid_on_both_routes(native, metadata):
    api, state = native
    state.data = _descriptor(relative=False)
    with pytest.raises(OSError, match="windows_security_descriptor_invalid"):
        api.security(7, False, with_descriptor=metadata)
    assert state.counts["length"] == state.counts["owner"] == 0


def test_malformed_fresh_ace_does_not_reuse_an_older_decode(native):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    changed = bytearray(state.data)
    struct.pack_into("<H", changed, 42, 4)
    state.data = bytes(changed)
    with pytest.raises(OSError, match="malformed_windows_acl"):
        api.security(7, False)
    state.data = _descriptor(trustee=2000)
    assert api.security(7, False) == (1000, 0o644)
