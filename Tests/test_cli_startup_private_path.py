# Tests/test_cli_startup_private_path.py
"""
Regression tests for task-32900: the packaged `tldw-cli` entry point must not
show a first-time user a bare private-path traceback.

`cli.main_cli_runner` runs the Backup_Recovery startup preflight and then
imports the heavy `tldw_chatbook.app` chain, whose `import
tldw_chatbook.config` can refuse the machine's filesystem posture (fail-closed
by design, ADR-029/ADR-127). The CLI entry catches that refusal, makes sure
the plain-language diagnostic was emitted, and exits 1 without a traceback.

The recovery preflight modules and the app module are stubbed in sys.modules
so the test does not need the full dependency chain or a recovery profile:
the preflight stub admits startup (`startup_preflight() -> (None, None)`),
`from ... import` resolves the stub, and its module `__getattr__` (or stubbed
function) raises the refusal.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from tldw_chatbook.Utils import startup_errors
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
)


@pytest.fixture(autouse=True)
def _reset_emit_guard():
    startup_errors.reset_emit_guard_for_tests()
    yield
    startup_errors.reset_emit_guard_for_tests()


@pytest.fixture(autouse=True)
def _clean_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry point parses and then rewrites sys.argv; keep it inert."""

    monkeypatch.setattr(sys, "argv", ["tldw-cli"])


def _stub_startup_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the recovery preflight admit startup without touching the fs."""

    launcher = types.ModuleType("tldw_chatbook.Backup_Recovery.launcher")
    launcher.startup_preflight = lambda: (None, None)
    launcher.minimal_recovery = lambda reason: 0
    launcher.recovery_main = lambda argv: 0
    monkeypatch.setitem(
        sys.modules, "tldw_chatbook.Backup_Recovery.launcher", launcher
    )

    admission = types.ModuleType("tldw_chatbook.Backup_Recovery.storage_admission")
    admission.admit_startup = lambda: None
    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.Backup_Recovery.storage_admission",
        admission,
    )


def _refusal() -> PrivatePathError:
    return PrivatePathError(
        PrivatePathResult(
            Path("/home/ubuntu/.config/tldw_cli"),
            PrivatePathStatus.UNSAFE_PARENT,
            reason="shared_writable_parent",
            offender_path=Path("/home/ubuntu/.config"),
            offender_detail="mode drwxrwxr-x (0o775), owner uid=1000, group gid=1000",
        )
    )


def _stub_app_module(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    stub = types.ModuleType("tldw_chatbook.app")

    def _raise(name: str):
        raise exc

    stub.__getattr__ = _raise  # PEP 562: module attribute access
    monkeypatch.setitem(sys.modules, "tldw_chatbook.app", stub)


def test_tldw_cli_startup_refusal_exits_1_with_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub_startup_preflight(monkeypatch)
    _stub_app_module(monkeypatch, _refusal())

    from tldw_chatbook import cli

    with pytest.raises(SystemExit) as exit_info:
        cli.main_cli_runner()

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "chmod g-w,o-w -- /home/ubuntu/.config" in err
    assert "Traceback" not in err


def test_tldw_cli_runtime_refusal_also_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A refusal raised by the runner call (not the import) gets the same UX."""

    _stub_startup_preflight(monkeypatch)

    stub = types.ModuleType("tldw_chatbook.app")

    def _run() -> int:
        raise _refusal()

    stub.main_cli_runner = _run
    monkeypatch.setitem(sys.modules, "tldw_chatbook.app", stub)

    from tldw_chatbook import cli

    with pytest.raises(SystemExit) as exit_info:
        cli.main_cli_runner()

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "chmod g-w,o-w -- /home/ubuntu/.config" in err
    assert "Traceback" not in err


def test_tldw_cli_does_not_emit_twice_when_config_already_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """config.py emits before re-raising; the CLI's own emit must not repeat it."""

    startup_errors.emit_private_path_startup_error(_refusal())
    first = capsys.readouterr().err
    assert "chmod g-w,o-w" in first

    _stub_startup_preflight(monkeypatch)
    _stub_app_module(monkeypatch, _refusal())
    from tldw_chatbook import cli

    with pytest.raises(SystemExit):
        cli.main_cli_runner()

    assert capsys.readouterr().err == ""
