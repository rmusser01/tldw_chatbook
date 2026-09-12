"""Fresh restored profiles retain native home selection without ambient providers."""

import ntpath
import os

import pytest

from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_environment


@pytest.mark.parametrize(
    "home",
    [
        {"USERPROFILE": r"C:\Users\fixture"},
        {"HOMEDRIVE": "C:", "HOMEPATH": r"\Users\fixture"},
    ],
)
def test_fresh_windows_profile_can_resolve_home_without_provider_overrides(
    monkeypatch, home
):
    for name in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in home.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HOME", "/posix-home")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-secret")
    monkeypatch.setenv("TLDW_MEDIA_DB_PATH", "/ambient/database")
    environment = _launch_environment()
    with monkeypatch.context() as selected:
        selected.setattr(os, "environ", environment)
        assert ntpath.expanduser("~") == r"C:\Users\fixture"
    assert "OPENAI_API_KEY" not in environment
    assert "TLDW_MEDIA_DB_PATH" not in environment
