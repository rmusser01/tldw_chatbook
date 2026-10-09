"""Native default owner selection is test-only, real, and reversible."""

import os
import subprocess
import sys

import pytest

from Tests.windows_private_fixture_runner import user_fixture_default_owner

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows token owner")


def test_user_fixture_owner_is_native_and_restored(tmp_path):
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
    )
    try:
        original = win32security.GetTokenInformation(token, win32security.TokenOwner)
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        with user_fixture_default_owner():
            target = tmp_path / "stdlib-owner.txt"
            target.write_text("fixture", encoding="utf-8")
            descriptor = win32security.GetFileSecurity(
                str(target), win32security.OWNER_SECURITY_INFORMATION
            )
            assert descriptor.GetSecurityDescriptorOwner() == user
            child = tmp_path / "child-owner.txt"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('child')",
                    str(child),
                ],
                check=True,
                timeout=30,
            )
            descriptor = win32security.GetFileSecurity(
                str(child), win32security.OWNER_SECURITY_INFORMATION
            )
            assert descriptor.GetSecurityDescriptorOwner() == user
        assert (
            win32security.GetTokenInformation(token, win32security.TokenOwner)
            == original
        )
        with pytest.raises(ValueError, match="real failure"):
            with user_fixture_default_owner():
                raise ValueError("real failure")
        assert (
            win32security.GetTokenInformation(token, win32security.TokenOwner)
            == original
        )
    finally:
        token.Close()


def test_user_fixture_launcher_refuses_required_elevated_custody(monkeypatch):
    monkeypatch.setenv("TLDW_REQUIRE_ELEVATED_CUSTODY", "1")
    with pytest.raises(RuntimeError, match="retain its original token"):
        with user_fixture_default_owner():
            pytest.fail("required custody was modified")
