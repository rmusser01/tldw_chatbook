"""Native token and SQLite-created owner evidence for TASK-34404."""

from __future__ import annotations

import ctypes as C
import json
import os
import stat

import pytest

from Tests.windows_custody import elevated_custody_required, unavailable_custody
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.windows_files import WindowsOS

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows ownership")


def _token_sid(native, information_class):
    token = windows_files._HANDLE()
    native.check(
        native.advapi.OpenProcessToken(
            native.kernel.GetCurrentProcess(), 8, C.byref(token)
        )
    )
    try:
        size = windows_files._U32()
        native.advapi.GetTokenInformation(
            token, information_class, None, 0, C.byref(size)
        )
        buffer = C.create_string_buffer(size.value)
        native.check(
            native.advapi.GetTokenInformation(
                token, information_class, buffer, size, C.byref(size)
            )
        )
        return native.sid_string(C.cast(buffer, C.POINTER(windows_files._P))[0])
    finally:
        native.kernel.CloseHandle(token)


def _security_receipt(path):
    native, win = windows_files._native(), WindowsOS()
    with windows_files._parent(path) as (parent, leaf):
        handle = native.open_handle(leaf, parent=parent, metadata=True)
    owner, dacl, descriptor = (windows_files._P() for _ in range(3))
    try:
        result = native.advapi.GetSecurityInfo(
            handle,
            1,
            5,
            C.byref(owner),
            None,
            C.byref(dacl),
            None,
            C.byref(descriptor),
        )
        if result:
            raise C.WinError(result)
        info = win._stat_handle(handle)
        return {
            "owner_sid": native.sid_string(owner),
            "uid": info.st_uid,
            "mode": oct(stat.S_IMODE(info.st_mode)),
            "descriptor_hex": C.string_at(
                descriptor, native.advapi.GetSecurityDescriptorLength(descriptor)
            ).hex(),
        }
    finally:
        if descriptor:
            native.kernel.LocalFree(descriptor)
        native.kernel.CloseHandle(handle)


def test_native_sqlite_sidecars_reopen_with_exact_private_owner(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    native, win = windows_files._native(), WindowsOS()
    user, token_owner = _token_sid(native, 1), _token_sid(native, 4)
    if elevated_custody_required():
        if token_owner == user:
            unavailable_custody("requires genuine elevated default-owner mismatch")
        assert token_owner == "S-1-5-32-544"
        _enable_restore_privilege()()  # Verify capability, then restore token state.
    # Print native token and inherited fixture custody before any hardening.
    print(
        "WINDOWS_TOKEN_RECEIPT="
        + json.dumps(
            {
                "token_user_sid": _token_sid(native, 1),
                "token_owner_sid": _token_sid(native, 4),
                "fixture_parent": _security_receipt(tmp_path),
            },
            sort_keys=True,
        )
    )
    # Explicit-owner creation avoids pretending an elevated pytest directory
    # belongs to TokenUser merely because TokenOwner is Administrators.
    parent = tmp_path / "explicit-user-private"
    win.mkdir(parent, 0o700)
    root, config = parent / "bootstrap", parent / "config.toml"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    database = parent / "native.db"
    first = connect_private_sqlite("db.base", database)
    try:
        assert first.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        first.execute("CREATE TABLE sample(value)")
        first.execute("INSERT INTO sample VALUES (1)")
        first.commit()
        receipt = {
            "token_user_sid": _token_sid(native, 1),
            "token_owner_sid": _token_sid(native, 4),
            "parent": _security_receipt(parent),
            "database": _security_receipt(database),
            "wal": _security_receipt(str(database) + "-wal"),
            "shm": _security_receipt(str(database) + "-shm"),
        }
        print("WINDOWS_PRIVATE_SQLITE_RECEIPT=" + json.dumps(receipt, sort_keys=True))
        if elevated_custody_required():
            assert receipt["parent"]["owner_sid"] == user
            assert receipt["database"]["owner_sid"] == user
            assert receipt["wal"]["owner_sid"] == token_owner
            assert receipt["shm"]["owner_sid"] == token_owner
        second = connect_private_sqlite("db.base", database, must_exist=True)
        try:
            assert second.execute("SELECT value FROM sample").fetchall() == [(1,)]
        finally:
            second.close()
    finally:
        first.close()


def _replace_security(path, sddl, *, replace_owner=False):
    native = windows_files._native()
    win = WindowsOS()
    fd = win.open(path, win.O_RDONLY)
    descriptor = windows_files._P()
    native.check(
        native.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl,
            1,
            C.byref(descriptor),
            None,
        )
    )
    try:
        dacl, owner = windows_files._P(), windows_files._P()
        present, defaulted = windows_files._I32(), windows_files._I32()
        native.check(
            native.advapi.GetSecurityDescriptorDacl(
                descriptor,
                C.byref(present),
                C.byref(dacl),
                C.byref(defaulted),
            )
        )
        if replace_owner:
            native.check(
                native.advapi.GetSecurityDescriptorOwner(
                    descriptor,
                    C.byref(owner),
                    C.byref(defaulted),
                )
            )
        with native.reopen(
            native.handle(fd),
            windows_files._READ_CONTROL
            | windows_files._WRITE_DAC
            | (0x80000 if replace_owner else 0),
        ) as writable:
            result = native.advapi.SetSecurityInfo(
                writable,
                1,
                4 | 0x80000000 | (1 if replace_owner else 0),
                owner if replace_owner else None,
                None,
                dacl,
                None,
            )
            if result:
                raise C.WinError(result)
    finally:
        native.kernel.LocalFree(descriptor)
        win.close(fd)


def _enable_restore_privilege():
    """Enable the test runner's existing privilege, returning exact old state."""

    class Luid(C.Structure):
        _fields_ = [("low", C.c_uint32), ("high", C.c_int32)]

    class Privileges(C.Structure):
        _fields_ = [("count", C.c_uint32), ("luid", Luid), ("attributes", C.c_uint32)]

    native = windows_files._native()
    lookup = native.advapi.LookupPrivilegeValueW
    lookup.argtypes, lookup.restype = (
        [C.c_wchar_p, C.c_wchar_p, C.POINTER(Luid)],
        C.c_int32,
    )
    adjust = native.advapi.AdjustTokenPrivileges
    adjust.argtypes = [
        windows_files._HANDLE,
        C.c_int32,
        windows_files._P,
        C.c_uint32,
        windows_files._P,
        windows_files._P,
    ]
    adjust.restype = C.c_int32
    token = windows_files._HANDLE()
    native.check(
        native.advapi.OpenProcessToken(
            native.kernel.GetCurrentProcess(),
            8 | 32,
            C.byref(token),
        )
    )
    requested, previous, returned = Privileges(), Privileges(), windows_files._U32()
    requested.count, requested.attributes = 1, 2
    native.check(lookup(None, "SeRestorePrivilege", C.byref(requested.luid)))
    native.check(
        adjust(
            token,
            False,
            C.byref(requested),
            C.sizeof(previous),
            C.byref(previous),
            C.byref(returned),
        )
    )
    if C.get_last_error() == 1300:
        native.kernel.CloseHandle(token)
        unavailable_custody(
            "elevated runner has no SeRestorePrivilege for foreign-owner control"
        )

    def restore():
        try:
            native.check(adjust(token, False, C.byref(previous), 0, None, None))
        finally:
            native.kernel.CloseHandle(token)

    return restore


@pytest.mark.parametrize(
    "damage",
    [
        "no-user-ace",
        "owner-rights-only",
        "shared",
        "deny-user",
        "foreign-owner",
        "token-owner-changed",
    ],
)
def test_native_default_owner_does_not_adopt_unsafe_sidecars(
    tmp_path, monkeypatch, damage
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus

    native, win = windows_files._native(), WindowsOS()
    user, token_owner = _token_sid(native, 1), _token_sid(native, 4)
    if token_owner == user:
        unavailable_custody("requires genuine elevated default-owner mismatch")
    assert token_owner == "S-1-5-32-544"
    parent = tmp_path / "explicit-user-private"
    win.mkdir(parent, 0o700)
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: parent / "bootstrap"
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(parent / "config"))
    database = parent / "negative.db"
    first = connect_private_sqlite("db.base", database)
    restore = None
    try:
        first.execute("PRAGMA journal_mode=WAL")
        first.execute("CREATE TABLE sample(value)")
        first.commit()
        wal = str(database) + "-wal"
        assert _security_receipt(wal)["owner_sid"] == token_owner
        assert win.stat(wal).st_uid == 1000  # positive proof before mutation
        trusted = "(A;;FA;;;SY)(A;;FA;;;BA)"
        user_allow = f"(A;;FA;;;{user})"
        if damage == "no-user-ace":
            sddl = "D:P" + trusted
        elif damage == "owner-rights-only":
            sddl = "D:P(A;;FA;;;OW)" + trusted
        elif damage == "shared":
            sddl = "D:P" + user_allow + trusted + "(A;;FR;;;WD)"
        elif damage == "deny-user":
            sddl = f"D:P(D;;0x1;;;{user})" + user_allow + trusted
        elif damage == "foreign-owner":
            restore = _enable_restore_privilege()
            sddl = "O:SYD:P" + user_allow + trusted
        else:
            original_query = native._token_sid
            monkeypatch.setattr(
                native,
                "_token_sid",
                lambda kind: user if kind == 4 else original_query(kind),
            )
            sddl = "D:P" + user_allow + trusted
        if damage != "token-owner-changed":
            _replace_security(wal, sddl, replace_owner=damage == "foreign-owner")
        print(
            "WINDOWS_UNSAFE_SIDECAR_RECEIPT="
            + json.dumps(
                {
                    "damage": damage,
                    "token_user_sid": user,
                    "token_owner_sid": token_owner,
                    "wal": _security_receipt(wal),
                },
                sort_keys=True,
            )
        )
        assert win.stat(wal).st_uid != 1000
        with pytest.raises(PrivatePathError) as error:
            connect_private_sqlite("db.base", database, must_exist=True)
        assert error.value.result.status is PrivatePathStatus.WRONG_OWNER
    finally:
        first.close()
        if restore is not None:
            restore()
