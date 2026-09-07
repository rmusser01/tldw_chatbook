from __future__ import annotations

import base64
import datetime
import json
import os
from pathlib import Path
import sqlite3
import stat

import pytest

from tldw_chatbook.DB.private_sqlite import (
    connect_private_sqlite as real_connect_private_sqlite,
)
from tldw_chatbook.Web_Scraping.cookie_scraping import cookie_cloner


def _future_chromium_expiry() -> int:
    future = datetime.datetime.now() + datetime.timedelta(days=1)
    return int((future - datetime.datetime(1601, 1, 1)).total_seconds() * 1_000_000)


def _create_chromium_cookie_store(cookie_path: Path, local_state_path: Path) -> None:
    cookie_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(cookie_path)
    try:
        connection.execute(
            """
            CREATE TABLE cookies (
                host_key TEXT,
                name TEXT,
                path TEXT,
                encrypted_value BLOB,
                expires_utc INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO cookies VALUES (?, ?, ?, ?, ?)",
            (
                ".example.com",
                "session",
                "/",
                b"encrypted-value",
                _future_chromium_expiry(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    local_state_path.parent.mkdir(parents=True, exist_ok=True)
    local_state_path.write_text(
        json.dumps(
            {
                "os_crypt": {
                    "encrypted_key": base64.b64encode(b"DPAPI-test-key").decode("ascii")
                }
            }
        ),
        encoding="utf-8",
    )


def _create_firefox_cookie_store(cookie_path: Path) -> None:
    cookie_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(cookie_path)
    try:
        connection.execute(
            """
            CREATE TABLE moz_cookies (
                host TEXT,
                name TEXT,
                value TEXT,
                expiry INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO moz_cookies VALUES (?, ?, ?, ?)",
            (
                ".example.com",
                "session",
                "session-value",
                int(datetime.datetime.now().timestamp()) + 3600,
            ),
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("browser", "owner_id"),
    [
        ("chrome", "cookies.chrome"),
        ("firefox", "cookies.firefox"),
        ("edge", "cookies.edge"),
    ],
)
def test_browser_cookie_clone_is_private_read_only_and_cleaned(
    browser: str,
    owner_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_root = tmp_path / browser
    if browser == "chrome":
        cookie_path = browser_root / "Default" / "Cookies"
        local_state_path = browser_root / "Local State"
        _create_chromium_cookie_store(cookie_path, local_state_path)
        expanded_paths = {
            "~/.config/google-chrome/Default/Cookies": os.fspath(cookie_path),
            "~/.config/google-chrome/Local State": os.fspath(local_state_path),
        }
        reader = cookie_cloner.get_chrome_cookies
    elif browser == "edge":
        cookie_path = browser_root / "Default" / "Cookies"
        local_state_path = browser_root / "Local State"
        _create_chromium_cookie_store(cookie_path, local_state_path)
        expanded_paths = {
            "~/.config/microsoft-edge/Default/Cookies": os.fspath(cookie_path),
            "~/.config/microsoft-edge/Local State": os.fspath(local_state_path),
        }
        reader = cookie_cloner.get_edge_cookies
    else:
        profile_root = browser_root / "Profiles"
        cookie_path = profile_root / "first.default-release" / "cookies.sqlite"
        _create_firefox_cookie_store(cookie_path)
        expanded_paths = {"~/.mozilla/firefox": os.fspath(profile_root)}
        reader = cookie_cloner.get_firefox_cookies

    monkeypatch.setattr(cookie_cloner.sys, "platform", "linux")
    real_expanduser = os.path.expanduser
    monkeypatch.setattr(
        cookie_cloner.os.path,
        "expanduser",
        lambda selected: expanded_paths.get(selected, real_expanduser(selected)),
    )
    monkeypatch.setattr(
        cookie_cloner,
        "decrypt_edge_cookie",
        lambda encrypted_value, key: b"session-value",
    )

    clone_paths: list[Path] = []

    def observe_private_connect(
        selected_owner: str,
        database: str | os.PathLike[str],
        *,
        read_only: bool = False,
        **kwargs: object,
    ) -> sqlite3.Connection:
        clone_path = Path(database)
        clone_paths.append(clone_path)
        assert selected_owner == owner_id
        assert read_only is True
        if os.name == "posix":
            assert stat.S_IMODE(clone_path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(clone_path.stat().st_mode) == 0o600

        connection = real_connect_private_sqlite(
            selected_owner,
            database,
            read_only=read_only,
            **kwargs,
        )
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE must_be_read_only (id INTEGER)")
        return connection

    monkeypatch.setattr(
        cookie_cloner,
        "connect_private_sqlite",
        observe_private_connect,
        raising=False,
    )

    assert reader("example.com") == {"session": "session-value"}
    assert len(clone_paths) == 1
    assert not clone_paths[0].exists()
    assert not clone_paths[0].parent.exists()


def test_chrome_clone_directory_is_cleaned_when_read_only_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_root = tmp_path / "chrome"
    cookie_path = browser_root / "Default" / "Cookies"
    local_state_path = browser_root / "Local State"
    _create_chromium_cookie_store(cookie_path, local_state_path)

    expanded_paths = {
        "~/.config/google-chrome/Default/Cookies": os.fspath(cookie_path),
        "~/.config/google-chrome/Local State": os.fspath(local_state_path),
    }
    monkeypatch.setattr(cookie_cloner.sys, "platform", "linux")
    real_expanduser = os.path.expanduser
    monkeypatch.setattr(
        cookie_cloner.os.path,
        "expanduser",
        lambda selected: expanded_paths.get(selected, real_expanduser(selected)),
    )

    clone_paths: list[Path] = []

    def fail_private_connect(
        owner_id: str,
        database: str | os.PathLike[str],
        *,
        read_only: bool = False,
        **kwargs: object,
    ) -> sqlite3.Connection:
        del kwargs
        assert owner_id == "cookies.chrome"
        assert read_only is True
        clone_paths.append(Path(database))
        raise RuntimeError("read-only open failed")

    monkeypatch.setattr(
        cookie_cloner,
        "connect_private_sqlite",
        fail_private_connect,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="read-only open failed"):
        cookie_cloner.get_chrome_cookies("example.com")

    assert len(clone_paths) == 1
    assert not clone_paths[0].exists()
    assert not clone_paths[0].parent.exists()


def test_firefox_preserves_first_profile_database_error_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "firefox" / "Profiles"
    cookie_path = profile_root / "first.default-release" / "cookies.sqlite"
    cookie_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(cookie_path)
    connection.close()

    monkeypatch.setattr(cookie_cloner.sys, "platform", "linux")
    real_expanduser = os.path.expanduser
    monkeypatch.setattr(
        cookie_cloner.os.path,
        "expanduser",
        lambda selected: (
            os.fspath(profile_root)
            if selected == "~/.mozilla/firefox"
            else real_expanduser(selected)
        ),
    )

    assert cookie_cloner.get_firefox_cookies("example.com") == {}
