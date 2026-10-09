"""Fresh native security observations may reuse only identical byte decoding."""

import ctypes as C
import os
import struct
from collections import Counter
from pathlib import Path
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
        data=_descriptor(), counts=Counter(), error=None, token_owner="S-1-5-18"
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
        if state.error == "dacl":
            return 0
        offset = struct.unpack_from("<I", C.string_at(descriptor, 20), 16)[0]
        assign(output, address(descriptor) + offset if offset else None)
        return 1

    def query(handle, information, buffer, size, needed):
        # One NtQuerySecurityObject per observation into the caller's buffer.
        state.counts["query"] += 1
        assert information == 5  # OWNER | DACL
        if state.error == "query":
            return -1073741790  # STATUS_ACCESS_DENIED
        needed._obj.value = len(state.data)
        if len(state.data) > size:
            return -1073741789  # STATUS_BUFFER_TOO_SMALL
        C.memmove(buffer, state.data, len(state.data))
        return 0

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

    def token_sid(information_class):
        assert information_class == 4
        state.counts["token_owner"] += 1
        return state.token_owner

    api.nt = SimpleNamespace(
        NtQuerySecurityObject=query, RtlNtStatusToDosError=lambda status: 5
    )
    api.advapi = SimpleNamespace(
        GetSecurityDescriptorLength=lambda pointer: len(state.data),
        GetSecurityDescriptorOwner=owner,
        GetSecurityDescriptorDacl=dacl,
        GetAce=ace,
    )
    api.sid_string = sid_string
    api.check = check
    api._token_sid = token_sid
    monkeypatch.setattr(
        files.C,
        "WinError",
        lambda code: PermissionError(code, "native refusal"),
        raising=False,
    )
    yield api, state


def test_repeated_bytes_still_query_each_native_descriptor(native):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    first = state.counts.copy()
    assert api.security(7, False) == (1000, 0o600)
    assert state.counts["query"] == 2
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
    assert state.counts["query"] == 4


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
    state.error = "query"
    with pytest.raises(PermissionError):
        api.security(7, False)
    assert state.counts["query"] == 2


def test_failed_decode_is_not_cached(native):
    api, state = native
    state.error = "decode"
    with pytest.raises(ValueError, match="controlled SID"):
        api.security(7, False)
    state.error = None
    assert api.security(7, False) == (1000, 0o600)
    assert state.counts["query"] == 2
    assert state.counts["sid"] == 3


@pytest.mark.parametrize("phase", ["owner", "dacl", "ace"])
def test_native_decoder_refusal_is_not_cached_or_replaced_by_an_older_result(
    native, phase
):
    api, state = native
    assert api.security(7, False) == (1000, 0o600)
    state.data = _descriptor(trustee=2000)
    state.error = phase
    with pytest.raises(OSError, match="controlled native"):
        api.security(7, False)
    state.error = None
    assert api.security(7, False) == (1000, 0o644)
    assert state.counts["query"] == 3


def test_large_descriptors_are_requeried_and_keep_the_uncached_decoder(native):
    api, state = native
    state.data = _descriptor(padding=4096)
    assert api.security(7, False) == api.security(7, False) == (1000, 0o600)
    assert state.counts["sid"] == 4  # never retained in the bounded decode cache
    assert state.counts["query"] == 4  # each observation: short buffer, exact size
    with pytest.raises(OSError, match="stamp_unavailable"):
        api.security(7, False, with_descriptor=True)


def test_absolute_descriptor_bytes_fail_closed(native):
    api, state = native
    state.data = _descriptor(relative=False)
    with pytest.raises(OSError, match="descriptor_invalid"):
        api.security(7, False)
    assert state.counts["sid"] == 0


def test_successful_decode_retention_has_a_fixed_entry_bound(native):
    api, state = native
    for index in range(160):
        state.data = _descriptor(owner=3000 + index)
        assert api.security(7, False) == (-1, 0o600)
    assert api._decoded_security.cache_info().currsize <= 128


@pytest.mark.parametrize(
    "change,expected", [("null_dacl", 0o777), ("unknown_ace", 0o677)]
)
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


def test_token_owner_is_read_fresh_only_when_the_projection_depends_on_it(native):
    api, state = native
    # The token user's own objects never consult TokenOwner.
    for _ in range(3):
        assert api.security(7, False) == (1000, 0o600)
    assert state.counts["token_owner"] == 0
    # Non-administrative foreign owners project identically for any TokenOwner.
    state.data = _descriptor(owner=2000)
    assert api.security(7, False) == (-1, 0o600)
    assert state.counts["token_owner"] == 0
    # SYSTEM ownership with private user full control is custody only while the
    # current TokenOwner is SYSTEM: read afresh on every such observation.
    state.data = _descriptor(owner=18, trustee=1000, mask=0x1F01FF)
    assert api.security(7, False) == (1000, 0o600)
    state.token_owner = "S-1-5-32-544"
    assert api.security(7, False) == (0, 0o600)
    state.token_owner = "S-1-5-18"
    assert api.security(7, False) == (1000, 0o600)
    assert state.counts["token_owner"] == 3


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


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows decode cost")
def test_native_decode_reuse_preserves_fresh_acquisition_and_records_cost(
    tmp_path, monkeypatch, record_property
):
    import json
    import statistics
    import time

    windows = files.WindowsOS()
    descriptor = windows.open(
        tmp_path / "measured-security",
        windows.O_CREAT | windows.O_EXCL | windows.O_RDWR,
        0o600,
    )
    native = files._native()
    real_nt, real_advapi = native.nt, native.advapi
    counts = Counter()

    class Observed:
        def __init__(self, real, names):
            self.real, self.names = real, names

        def __getattr__(self, name):
            function = getattr(self.real, name)
            if name not in self.names:
                return function

            def call(*args):
                counts[name] += 1
                return function(*args)

            return call

    def uncached(handle):
        data = native.security_descriptor(handle)
        sid, aces, mode = native._decode_descriptor(data, False, native.user_sid)
        return files._owner_uid(sid, native.user_sid, None, aces, mode), mode

    try:
        handle = native.handle(descriptor)
        native._decoded_security.cache_clear()
        with monkeypatch.context() as patch:
            patch.setattr(native, "nt", Observed(real_nt, {"NtQuerySecurityObject"}))
            patch.setattr(
                native,
                "advapi",
                Observed(
                    real_advapi,
                    {"GetAce", "ConvertSidToStringSidW", "GetSecurityInfo"},
                ),
            )
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
            for label, read in (
                ("uncached", uncached),
                ("cached", lambda handle: native.security(handle, False)),
            ):
                started = time.perf_counter()
                for _ in range(200):
                    assert read(handle) == (1000, 0o600)
                samples[label].append(time.perf_counter() - started)
        record_property(
            "fresh_security_decode",
            json.dumps(
                {
                    "calls": dict(counts),
                    "iterations_per_sample": 200,
                    "samples": samples,
                    "median_seconds": {
                        name: statistics.median(values)
                        for name, values in samples.items()
                    },
                },
                sort_keys=True,
            ),
        )
    finally:
        windows.close(descriptor)


@pytest.mark.skipif(
    os.name != "nt", reason="requires actual Windows security descriptors"
)
def test_object_descriptor_projects_like_the_former_getsecurityinfo_route(tmp_path):
    """The stored descriptor and GetSecurityInfo's reconstruction project alike.

    GetSecurityInfo may add INHERITED_ACE bits and a group SID derived from the
    parent; owner, ACE order/types/masks/trustees and every projection match.
    """
    native = files._native()
    windows = files.WindowsOS()
    private = tmp_path / "private-directory"
    private.mkdir()
    leaf = private / "leaf.bin"
    leaf.write_bytes(b"x")
    targets = [Path.home().parent, Path.home(), tmp_path]  # roots are not facade-openable
    targets += [private, leaf]

    def former(handle):
        owner, dacl, acquired = files._P(), files._P(), files._P()
        result = native.advapi.GetSecurityInfo(
            handle, 1, 5, C.byref(owner), None, C.byref(dacl), None, C.byref(acquired)
        )
        if result:
            raise C.WinError(result)
        try:
            length = native.advapi.GetSecurityDescriptorLength(acquired)
            return C.string_at(acquired, length)
        finally:
            native.kernel.LocalFree(acquired)

    def strip_inherited(aces):
        return [(kind, flags & ~0x10, mask, sid) for kind, flags, mask, sid in aces]

    for target in targets:
        descriptor = windows.open(target, windows.O_RDONLY)
        try:
            handle = native.handle(descriptor)
            directory = target.is_dir()
            new = native._decode_descriptor(
                native.security_descriptor(handle), directory, native.user_sid
            )
            old = native._decode_descriptor(former(handle), directory, native.user_sid)
            assert new[0] == old[0] and new[2] == old[2], target
            assert strip_inherited(new[1]) == strip_inherited(old[1]), target
            token_owner = native._token_sid(4)
            assert native.security(handle, directory) == (
                files._owner_uid(old[0], native.user_sid, token_owner, old[1], old[2]),
                old[2],
            ), target
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
