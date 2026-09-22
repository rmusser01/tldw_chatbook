# Tests/Utils/test_startup_errors.py
"""
Regression tests for task-32900: plain-language diagnostics for private-path
startup refusals.

A fresh install on a machine whose `~/.config` (or another ancestor of the
config/data directories) is group/world-writable used to die at import with a
bare `PrivatePathError: unsafe_parent: shared_writable_parent` traceback.
These tests pin the diagnostics contract: the formatter names the blocked
directory and the exact repair command per refusal reason, the emitter prints
once per process, and config's import-time bootstrap prints the diagnostic
before re-raising (so every entry point explains itself).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Utils import startup_errors
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_emit_guard():
    startup_errors.reset_emit_guard_for_tests()
    yield
    startup_errors.reset_emit_guard_for_tests()


def _refusal(
    reason: str,
    *,
    status: PrivatePathStatus = PrivatePathStatus.UNSAFE_PARENT,
    offender_path: Path | None = Path("/home/ubuntu/.config"),
    offender_detail: str | None = (
        "mode drwxrwxr-x (0o775), owner uid=1000, group gid=1000"
    ),
    offender_mode: int | None = 0o775,
) -> PrivatePathError:
    return PrivatePathError(
        PrivatePathResult(
            Path("/home/ubuntu/.config/tldw_cli"),
            status,
            reason=reason,
            offender_path=offender_path,
            offender_detail=offender_detail,
            offender_mode=offender_mode,
        )
    )


# --- Formatter: what the user reads ------------------------------------------


def test_shared_writable_guidance_names_chmod_command():
    text = startup_errors.format_private_path_error(_refusal("shared_writable_parent"))

    # The exact repair command for the named directory.
    assert "chmod g-w,o-w -- /home/ubuntu/.config" in text
    # The blocked directory and the location the app wanted both appear.
    assert "/home/ubuntu/.config" in text
    assert "/home/ubuntu/.config/tldw_cli" in text
    assert "drwxrwxr-x" in text
    # Why the app refuses at all.
    assert "secrets" in text or "API keys" in text
    # The escape hatch and the technical detail for bug reports.
    assert "TLDW_CONFIG_PATH" in text
    assert "shared_writable_parent" in text


def test_sticky_home_guidance_recommends_real_home_not_chmod():
    text = startup_errors.format_private_path_error(
        _refusal(
            "missing_component_in_shared_sticky_parent",
            offender_path=Path("/tmp"),
            offender_detail="mode drwxrwxrwt (0o1777), owner uid=0, group gid=0",
        )
    )

    assert "/tmp" in text
    assert "sticky" in text or "shared" in text
    # chmod cannot fix a sticky shared parent; the guidance must not claim it can.
    assert "chmod g-w,o-w" not in text
    # Relocating the config under the same shared HOME would refuse again.
    assert 'TLDW_CONFIG_PATH="$HOME"' not in text
    assert "HOME=/home/you" in text


def test_untrusted_owner_guidance_points_at_ownership():
    text = startup_errors.format_private_path_error(
        _refusal(
            "untrusted_directory_owner",
            offender_path=Path("/home/ubuntu"),
            offender_detail="mode drwxr-x--- (0o750), owner uid=1000, group gid=1000",
            offender_mode=0o750,
        )
    )

    assert "owner" in text.lower()
    # Ownership alone (mode 0750) is not repaired with the shared-writable
    # command; the shared-mode variant is covered by its own test below.
    assert "chmod g-w,o-w" not in text


def test_ambiguous_data_roots_guidance_mentions_explicit_choice():
    text = startup_errors.format_private_path_error(
        _refusal("ambiguous_default_data_roots", offender_path=None, offender_detail=None)
    )

    assert "data" in text.lower()
    assert "data_dir" in text
    # A config-path alternative cannot resolve a data-root refusal.
    assert "TLDW_CONFIG_PATH" not in text


def test_unknown_reason_still_formats_with_generic_guidance():
    text = startup_errors.format_private_path_error(
        _refusal("some_future_reason", offender_path=None, offender_detail=None)
    )

    assert "some_future_reason" in text
    assert "Chatbook" in text


def test_message_claims_no_permissions_changed():
    """Fail-closed means fail-without-touching; the text must promise that."""

    text = startup_errors.format_private_path_error(_refusal("shared_writable_parent"))
    flat = " ".join(text.split())  # the diagnostic wraps at 72 columns
    assert (
        "Nothing outside Chatbook's own config/data directories was created"
        " or changed" in flat
    )


# --- Emitter: once per process ------------------------------------------------


def test_emit_prints_formatted_error_to_stderr_once(capsys):
    error = _refusal("shared_writable_parent")

    startup_errors.emit_private_path_startup_error(error)
    first = capsys.readouterr().err
    assert "chmod g-w,o-w -- /home/ubuntu/.config" in first

    # A second call (e.g. cli.py catching what config.py already reported)
    # must not duplicate the diagnostic on screen.
    startup_errors.emit_private_path_startup_error(error)
    assert capsys.readouterr().err == ""


def test_emit_without_offender_still_prints_reason(capsys):
    startup_errors.emit_private_path_startup_error(
        _refusal("shared_writable_parent", offender_path=None, offender_detail=None)
    )
    captured = capsys.readouterr().err
    assert "shared_writable_parent" in captured
    assert captured.strip() != ""


# --- Startup integration: config's import-time bootstrap ----------------------


def test_config_import_prints_diagnostic_before_raising(tmp_path: Path) -> None:
    """`import tldw_chatbook.config` against a group-writable ~/.config must
    print the plain-language diagnostic and still fail closed.

    Isolated subprocess (HOME-based, no TLDW_CONFIG_PATH) because the refusal
    happens at module import: sys.modules is process-global, so an in-process
    check would false-pass once config is already imported.
    """
    pytest.importorskip("portalocker", reason="config import needs hard deps")

    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    (home / ".config").chmod(0o775)

    env = {
        **os.environ,
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
        "TLDW_TEST_MODE": "1",
    }
    env.pop("TLDW_CONFIG_PATH", None)
    env.pop("PYTEST_CURRENT_TEST", None)

    result = subprocess.run(
        [sys.executable, "-c", "import tldw_chatbook.config"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode != 0, "the refusal must still fail closed"
    assert "chmod g-w,o-w" in result.stderr
    assert str(home / ".config") in result.stderr
    # The traceback stays available for bug reports on non-cli entry points.
    assert "Traceback" in result.stderr


# --- Qodo remediation: safe copy-paste commands and honest alternatives -------


def test_repair_commands_shell_quote_hostile_paths():
    """Paths with spaces/metacharacters must survive copy-paste as one operand."""

    hostile = Path("/home/u/my configs/x;touch pwned")
    text = startup_errors.format_private_path_error(
        _refusal("shared_writable_parent", offender_path=hostile)
    )

    quoted = shlex.quote(str(hostile))
    assert f"chmod g-w,o-w -- {quoted}" in text
    # The unquoted path must not appear as a bare command operand.
    assert f"chmod g-w,o-w -- {hostile}\n" not in text
    # Round-trip: the embedded command parses back to the exact path.
    command_line = next(
        line.strip()[4:] for line in text.splitlines() if line.startswith("    chmod ")
    )
    assert shlex.split(command_line)[-1] == str(hostile)


def test_ownership_guidance_includes_chmod_when_mode_is_shared_writable():
    """A foreign-owned 0775 directory needs both repairs to restart cleanly."""

    text = startup_errors.format_private_path_error(
        _refusal(
            "untrusted_directory_owner",
            offender_path=Path("/home/ubuntu"),
            offender_detail="mode drwxrwxr-x (0o775), owner uid=1000, group gid=1000",
            offender_mode=0o775,
        )
    )
    assert 'sudo chown "$USER" -- /home/ubuntu' in text
    assert "chmod g-w,o-w -- /home/ubuntu" in text


def test_ownership_guidance_omits_chmod_when_mode_is_private():
    text = startup_errors.format_private_path_error(
        _refusal(
            "untrusted_directory_owner",
            offender_path=Path("/home/ubuntu"),
            offender_detail="mode drwxr-x--- (0o750), owner uid=1000, group gid=1000",
            offender_mode=0o750,
        )
    )
    assert "chmod g-w,o-w" not in text


def test_default_alternative_mentions_data_dir_escape_hatch():
    """The config alternative must not claim to fix data-side refusals."""

    text = startup_errors.format_private_path_error(
        _refusal("shared_writable_parent")
    )
    assert "data_dir" in text
