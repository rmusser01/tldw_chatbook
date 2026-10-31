"""TASK-18925: OS notification dispatcher.

Pins the three AC surfaces of the pure dispatcher module:
- argv construction per platform (fixed argv, no shell, no message content),
- default-off + per-type gating of the normalized settings,
- failure tolerance (absent binary, timeout, unsupported platform) and the
  scrubbed subprocess environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tldw_chatbook.Utils.os_notifications import (
    OS_NOTIFICATION_APP_NAME,
    OS_NOTIFICATION_TIMEOUT_S,
    ConsoleOSNotificationSettings,
    build_notification_argv,
    normalize_console_os_notifications,
    send_os_notification,
)

SECRET_CANARY = "sk-canary-DO-NOT-LEAK"


class TestNormalizeConsoleOSNotifications:
    def test_absent_config_defaults_to_off_master_with_all_types_enabled(self) -> None:
        """AC#1: the feature is opt-in -- missing config means OFF, and the
        per-type toggles default on so flipping the master on enables all
        three (the user then narrows, never widens, per type)."""
        settings = normalize_console_os_notifications(None)
        assert settings.enabled is False
        assert settings.run_finished is True
        assert settings.run_failed is True
        assert settings.needs_approval is True

    def test_empty_mapping_defaults_to_off(self) -> None:
        assert normalize_console_os_notifications({}).enabled is False

    def test_non_mapping_input_defaults_to_off(self) -> None:
        assert normalize_console_os_notifications("on").enabled is False

    def test_explicit_values_round_trip(self) -> None:
        settings = normalize_console_os_notifications(
            {"enabled": True, "run_finished": False}
        )
        assert settings == ConsoleOSNotificationSettings(
            enabled=True, run_finished=False, run_failed=True, needs_approval=True
        )

    def test_garbage_values_coerce_to_defaults(self) -> None:
        settings = normalize_console_os_notifications(
            {"enabled": "definitely", "run_finished": None, "run_failed": []}
        )
        assert settings.enabled is False
        assert settings.run_finished is True
        assert settings.run_failed is True
        assert settings.needs_approval is True

    def test_allows_requires_master_and_per_type(self) -> None:
        on = normalize_console_os_notifications({"enabled": True})
        assert on.allows("run_finished") is True
        assert on.allows("run_failed") is True
        assert on.allows("needs_approval") is True

        narrow = normalize_console_os_notifications(
            {"enabled": True, "run_failed": False}
        )
        assert narrow.allows("run_failed") is False
        assert narrow.allows("run_finished") is True

        off = normalize_console_os_notifications({"enabled": False})
        assert off.allows("run_finished") is False

    def test_to_config_round_trips_through_normalize(self) -> None:
        settings = normalize_console_os_notifications(
            {"enabled": True, "needs_approval": False}
        )
        assert normalize_console_os_notifications(settings.to_config()) == settings


class TestBuildNotificationArgv:
    def test_darwin_argv_is_exact_fixed_form(self) -> None:
        argv = build_notification_argv("darwin", "run_finished", "Ops (Lab)")
        assert argv == [
            "osascript",
            "-e",
            (
                'display notification "Agent in Ops (Lab) run finished." '
                f'with title "{OS_NOTIFICATION_APP_NAME}"'
            ),
        ]

    def test_linux_argv_is_exact_fixed_form(self) -> None:
        argv = build_notification_argv("linux", "needs_approval", "Ops (Lab)")
        assert argv == [
            "notify-send",
            "--app-name",
            OS_NOTIFICATION_APP_NAME,
            "--",
            OS_NOTIFICATION_APP_NAME,
            "Agent in Ops (Lab) needs approval.",
        ]

    def test_windows_returns_none(self) -> None:
        assert build_notification_argv("win32", "run_finished", "Ops") is None

    def test_unknown_kind_returns_none(self) -> None:
        assert build_notification_argv("darwin", "message_received", "Ops") is None

    def test_applescript_quotes_and_backslashes_are_escaped(self) -> None:
        argv = build_notification_argv("darwin", "run_failed", 'Ev"il\\Name')
        assert argv is not None
        script = argv[2]
        assert "Ev\\" + '"il\\\\Name' in script
        # Every quote that is not one of the four string delimiters (body
        # open/close + title open/close) must be escaped, so no embedded
        # quote can terminate the AppleScript string early.
        assert script.count('\\"') == script.count('"') - 4

    def test_control_characters_never_reach_any_argv_element(self) -> None:
        for platform in ("darwin", "linux"):
            argv = build_notification_argv(platform, "run_finished", "Ops\n\tLab")
            assert argv is not None
            for element in argv:
                assert "\n" not in element
                assert "\t" not in element

    def test_argv_carries_only_session_display_and_category(self) -> None:
        """AC#3: nothing beyond the session display name and the outcome
        category may enter the command line -- construct with a message
        content canary and prove it absent."""
        argv = build_notification_argv("linux", "run_finished", "Ops (Lab)")
        assert argv is not None
        joined = " ".join(argv)
        assert SECRET_CANARY not in joined
        assert "tool output" not in joined


class _RecordingRunner:
    def __init__(self, exc: BaseException | None = None) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.exc = exc

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(argv, 0, b"", b"")


class TestSendOSNotification:
    def test_allowed_kind_invokes_runner_with_bounded_subprocess_contract(self) -> None:
        runner = _RecordingRunner()
        settings = ConsoleOSNotificationSettings(enabled=True)
        result = send_os_notification(
            "run_finished",
            "Ops (Lab)",
            platform="linux",
            settings=settings,
            runner=runner,
        )
        assert result is True
        assert len(runner.calls) == 1
        argv, kwargs = runner.calls[0]
        assert argv[0] == "notify-send"
        assert kwargs["timeout"] == OS_NOTIFICATION_TIMEOUT_S
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["start_new_session"] is True

    def test_master_off_never_spawns_a_process(self) -> None:
        runner = _RecordingRunner()
        result = send_os_notification(
            "run_finished",
            "Ops",
            platform="linux",
            settings=ConsoleOSNotificationSettings(enabled=False),
            runner=runner,
        )
        assert result is False
        assert runner.calls == []

    def test_per_type_off_never_spawns_a_process(self) -> None:
        runner = _RecordingRunner()
        result = send_os_notification(
            "run_failed",
            "Ops",
            platform="linux",
            settings=ConsoleOSNotificationSettings(enabled=True, run_failed=False),
            runner=runner,
        )
        assert result is False
        assert runner.calls == []

    def test_unsupported_platform_never_spawns_and_reports_false(self) -> None:
        runner = _RecordingRunner()
        result = send_os_notification(
            "run_finished",
            "Ops",
            platform="win32",
            settings=ConsoleOSNotificationSettings(enabled=True),
            runner=runner,
        )
        assert result is False
        assert runner.calls == []

    def test_missing_binary_returns_false_without_raising(self) -> None:
        runner = _RecordingRunner(exc=FileNotFoundError("notify-send"))
        result = send_os_notification(
            "run_finished",
            "Ops",
            platform="linux",
            settings=ConsoleOSNotificationSettings(enabled=True),
            runner=runner,
        )
        assert result is False

    def test_timeout_returns_false_without_raising(self) -> None:
        runner = _RecordingRunner(
            exc=subprocess.TimeoutExpired(
                cmd="osascript", timeout=OS_NOTIFICATION_TIMEOUT_S
            )
        )
        result = send_os_notification(
            "run_finished",
            "Ops",
            platform="darwin",
            settings=ConsoleOSNotificationSettings(enabled=True),
            runner=runner,
        )
        assert result is False

    def test_subprocess_env_is_scrubbed_to_the_minimum(self) -> None:
        import os

        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = SECRET_CANARY
        try:
            runner = _RecordingRunner()
            send_os_notification(
                "run_finished",
                "Ops",
                platform="linux",
                settings=ConsoleOSNotificationSettings(enabled=True),
                runner=runner,
            )
        finally:
            if old is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old
        assert len(runner.calls) == 1
        env = runner.calls[0][1]["env"]
        assert "OPENAI_API_KEY" not in env
        assert SECRET_CANARY not in "".join(str(v) for v in env.values())
        assert "PATH" in env

    def test_linux_env_keeps_dbus_address_when_present(self) -> None:
        import os

        old = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/tmp/canary-bus"
        try:
            runner = _RecordingRunner()
            send_os_notification(
                "run_finished",
                "Ops",
                platform="linux",
                settings=ConsoleOSNotificationSettings(enabled=True),
                runner=runner,
            )
        finally:
            if old is None:
                os.environ.pop("DBUS_SESSION_BUS_ADDRESS", None)
            else:
                os.environ["DBUS_SESSION_BUS_ADDRESS"] = old
        env = runner.calls[0][1]["env"]
        assert env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/tmp/canary-bus"

    def test_darwin_env_is_exactly_path(self) -> None:
        """macOS needs no D-Bus plumbing: the scrubbed env is PATH alone."""
        import os

        old_dbus = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        old_xdg = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/tmp/canary-bus"
        os.environ["XDG_RUNTIME_DIR"] = "/tmp/canary-xdg"
        try:
            runner = _RecordingRunner()
            send_os_notification(
                "run_finished",
                "Ops",
                platform="darwin",
                settings=ConsoleOSNotificationSettings(enabled=True),
                runner=runner,
            )
        finally:
            for key, old in (
                ("DBUS_SESSION_BUS_ADDRESS", old_dbus),
                ("XDG_RUNTIME_DIR", old_xdg),
            ):
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
        env = runner.calls[0][1]["env"]
        assert set(env) == {"PATH"}


def test_linux_notification_body_escapes_markup_but_preserves_literal_text() -> None:
    display = """<b>Ops</b> & <img src="file:///private.png"> 'quoted'"""
    argv = build_notification_argv("linux", "run_finished", display)
    assert argv is not None
    assert argv[-1] == (
        "Agent in &lt;b&gt;Ops&lt;/b&gt; &amp; "
        """&lt;img src="file:///private.png"&gt; 'quoted' run finished."""
    )
    darwin = build_notification_argv("darwin", "run_finished", "<Ops> & Lab")
    assert darwin is not None
    assert "Agent in <Ops> & Lab run finished." in darwin[-1]


def test_nonzero_notifier_exit_returns_false_and_logs_only_status(monkeypatch) -> None:
    from tldw_chatbook.Utils import os_notifications

    logs = []
    monkeypatch.setattr(os_notifications.logger, "debug", logs.append)
    result = send_os_notification(
        "run_failed",
        SECRET_CANARY,
        platform="linux",
        runner=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 7, SECRET_CANARY.encode(), SECRET_CANARY.encode()
        ),
    )
    assert result is False
    assert len(logs) == 1
    assert "run_failed" in logs[0]
    assert "7" in logs[0]
    assert SECRET_CANARY not in logs[0]
    assert "notify-send" not in logs[0]


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(SECRET_CANARY),
        subprocess.TimeoutExpired(cmd=["osascript", SECRET_CANARY], timeout=3),
    ],
)
def test_notifier_exception_logs_exclude_command_and_exception_text(
    monkeypatch, failure
) -> None:
    from tldw_chatbook.Utils import os_notifications

    logs = []
    monkeypatch.setattr(os_notifications.logger, "debug", logs.append)
    result = send_os_notification(
        "run_finished",
        SECRET_CANARY,
        platform="darwin",
        runner=_RecordingRunner(exc=failure),
    )
    assert result is False
    assert len(logs) == 1
    assert "run_finished" in logs[0]
    assert type(failure).__name__ in logs[0]
    assert SECRET_CANARY not in logs[0]
    assert "osascript" not in logs[0]


def test_notification_timeout_survives_orderly_interpreter_shutdown(tmp_path) -> None:
    """A daemon regression loses the timeout owner and never writes the result.

    The real screen helper and sender run in a fresh parent interpreter; only
    the external notifier binary is replaced with a deliberately slow Python
    child. Its natural exit also bounds cleanup when this regression fails.
    """
    script = textwrap.dedent(
        """
        import json
        import pathlib
        import subprocess
        import sys
        import threading
        import time
        import Tests.conftest  # Private profile/egress guard before app imports.
        from tldw_chatbook.UI.Screens.chat_screen import _spawn_os_notification_thread
        from tldw_chatbook.Utils.os_notifications import send_os_notification

        root = pathlib.Path(sys.argv[1])
        child = (
            "import pathlib,sys,time; root=pathlib.Path(sys.argv[1]); "
            "(root / 'started').write_text('yes'); time.sleep(6); "
            "(root / 'natural-exit').write_text('yes')"
        )
        main_thread = threading.get_ident()

        def runner(argv, **kwargs):
            return subprocess.run([sys.executable, "-c", child, str(root)], **kwargs)

        def notify():
            result = send_os_notification(
                "run_finished", "Ops", platform="linux", runner=runner
            )
            (root / "result.json").write_text(json.dumps({
                "result": result,
                "off_thread": threading.get_ident() != main_thread,
                "returned_before_completion": (root / "returned").exists(),
            }))

        _spawn_os_notification_thread(notify)
        (root / "returned").write_text("yes")
        deadline = time.monotonic() + 10
        while not (root / "started").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("slow notifier never started")
            time.sleep(0.01)
        # Do not join: orderly interpreter shutdown must retain the owner.
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = tmp_path / "result.json"
    if not result.exists():
        # In the broken daemon case the orphan exits naturally within six
        # seconds; wait before pytest removes its private directory.
        deadline = time.monotonic() + 10
        while not (tmp_path / "natural-exit").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    assert result.exists(), "interpreter exit abandoned the notifier timeout owner"
    assert json.loads(result.read_text()) == {
        "result": False,
        "off_thread": True,
        "returned_before_completion": True,
    }
    assert not (tmp_path / "natural-exit").exists(), "slow notifier escaped its timeout"
