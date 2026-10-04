"""Native token and SQLite-created owner evidence for TASK-34404."""

from __future__ import annotations

import ctypes as C
import json
import os
import sqlite3
import stat

import pytest

from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.windows_files import WindowsOS

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows ownership")


def _token_sid(native, information_class):
    token = windows_files._HANDLE()
    native.check(native.advapi.OpenProcessToken(
        native.kernel.GetCurrentProcess(), 8, C.byref(token)
    ))
    try:
        size = windows_files._U32()
        native.advapi.GetTokenInformation(token, information_class, None, 0, C.byref(size))
        buffer = C.create_string_buffer(size.value)
        native.check(native.advapi.GetTokenInformation(
            token, information_class, buffer, size, C.byref(size)
        ))
        return native.sid_string(C.cast(buffer, C.POINTER(windows_files._P))[0])
    finally:
        native.kernel.CloseHandle(token)


def _security_receipt(path):
    native, win = windows_files._native(), WindowsOS()
    fd = win.open(path, win.O_RDONLY)
    owner, dacl, descriptor = (windows_files._P() for _ in range(3))
    try:
        result = native.advapi.GetSecurityInfo(
            native.handle(fd), 1, 5, C.byref(owner), None, C.byref(dacl),
            None, C.byref(descriptor)
        )
        if result:
            raise C.WinError(result)
        info = win.fstat(fd)
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
        win.close(fd)


def test_native_sqlite_sidecars_reopen_with_exact_private_owner(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    native, win = windows_files._native(), WindowsOS()
    # Never infer custody from synthetic Windows chmod/stat values.
    win.chmod(tmp_path, 0o700)
    root, config = tmp_path / "bootstrap", tmp_path / "config.toml"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    database = tmp_path / "native.db"
    first = connect_private_sqlite("db.base", database)
    try:
        assert first.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        first.execute("CREATE TABLE sample(value)")
        first.execute("INSERT INTO sample VALUES (1)")
        first.commit()
        receipt = {
            "token_user_sid": _token_sid(native, 1),
            "token_owner_sid": _token_sid(native, 4),
            "parent": _security_receipt(tmp_path),
            "database": _security_receipt(database),
            "wal": _security_receipt(str(database) + "-wal"),
            "shm": _security_receipt(str(database) + "-shm"),
        }
        print("WINDOWS_PRIVATE_SQLITE_RECEIPT=" + json.dumps(receipt, sort_keys=True))
        second = connect_private_sqlite("db.base", database, must_exist=True)
        try:
            assert second.execute("SELECT value FROM sample").fetchall() == [(1,)]
        finally:
            second.close()
    finally:
        first.close()
