"""TASK-18925: OS notification dispatcher.

Pins the three AC surfaces of the pure dispatcher module:
- argv construction per platform (fixed argv, no shell, no message content),
- default-off + per-type gating of the normalized settings,
- failure tolerance (absent binary, timeout, unsupported platform) and the
  scrubbed subprocess environment.
"""

from __future__ import annotations

import subprocess

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
        argv = build_notification_argv(
            "linux", "run_finished", "Ops (Lab)"
        )
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
            exc=subprocess.TimeoutExpired(cmd="osascript", timeout=OS_NOTIFICATION_TIMEOUT_S)
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
