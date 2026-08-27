"""Tests for ``Utils/atomic_file_ops.py``, in particular the
``preserve_existing_mode`` parameter added by the task-851 review (finding
2): ``atomic_write_text`` used to always ``chmod`` the replacement file to
its ``mode`` parameter's default (0o644), which widened permissions on any
target file that had been deliberately tightened (e.g. a config file
holding secrets, chmod'd to 0o600). Enabling then disabling config
encryption measured 0600 -> 0644 -> (still) 0644 with plaintext keys.
"""

import stat

import pytest
from loguru import logger

from tldw_chatbook.Utils.atomic_file_ops import atomic_write_text


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_preserve_existing_mode_keeps_restrictive_permissions(tmp_path):
    """A pre-existing 0600 file must stay 0600 after an atomic rewrite."""
    target = tmp_path / "secrets.toml"
    target.write_text("a = 1\n")
    target.chmod(0o600)

    atomic_write_text(
        target, "a = 2\n", mode=0o644, preserve_existing_mode=True
    )

    assert _mode(target) == 0o600
    assert target.read_text() == "a = 2\n"


def test_preserve_existing_mode_keeps_permissive_permissions(tmp_path):
    """The preserve path must not *tighten* an existing file either --
    it carries forward whatever mode the file already had, in either
    direction."""
    target = tmp_path / "public.toml"
    target.write_text("a = 1\n")
    target.chmod(0o644)

    atomic_write_text(
        target, "a = 2\n", mode=0o600, preserve_existing_mode=True
    )

    assert _mode(target) == 0o644


def test_preserve_existing_mode_uses_fallback_mode_for_new_file(tmp_path):
    """When the target does not exist yet, there is nothing to preserve --
    the caller-supplied ``mode`` (e.g. a restrictive default for a secrets
    file) is applied instead."""
    target = tmp_path / "new_secrets.toml"
    assert not target.exists()

    atomic_write_text(
        target, "a = 1\n", mode=0o600, preserve_existing_mode=True
    )

    assert _mode(target) == 0o600


def test_preserve_existing_mode_false_keeps_legacy_behavior(tmp_path):
    """Default (``preserve_existing_mode=False``) behavior for existing
    callers must be unchanged: the file is always chmod'd to ``mode``,
    even if it previously had a different mode."""
    target = tmp_path / "notes.md"
    target.write_text("hello\n")
    target.chmod(0o600)

    atomic_write_text(target, "hello again\n")

    assert _mode(target) == 0o644


def test_privacy_safe_failure_log_contains_only_category_and_exception_class(
    tmp_path, monkeypatch
):
    path_canary = "PATH-CANARY-22507"
    body_canary = "BODY-CANARY-22507"
    exception_canary = "EXCEPTION-CANARY-22507"
    target = tmp_path / path_canary
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    class PrivateWriteError(OSError):
        pass

    def fail_replace(*_args):
        raise PrivateWriteError(exception_canary)

    monkeypatch.setattr("os.replace", fail_replace)
    try:
        with pytest.raises(PrivateWriteError):
            atomic_write_text(
                target,
                body_canary,
                privacy_safe_log=True,
            )
    finally:
        logger.remove(sink)

    log_text = "".join(messages)
    assert "atomic_write_failed" in log_text
    assert "PrivateWriteError" in log_text
    assert path_canary not in log_text
    assert body_canary not in log_text
    assert exception_canary not in log_text
    assert "Traceback" not in log_text


def test_no_overwrite_refuses_an_existing_destination(tmp_path):
    target = tmp_path / "appeared.json"
    target.write_text("other writer", encoding="utf-8")

    with pytest.raises(FileExistsError):
        atomic_write_text(target, "private export", overwrite=False)

    assert target.read_text(encoding="utf-8") == "other writer"
