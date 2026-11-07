"""TASK-18925: OS notification hook sites in the Console screen.

Pins the three bridge methods' OS-notification behavior: one notification
per once-guarded toast event (the dedupe the toasts already carry), the
content budget (conversation/tab display name + outcome category only --
never the run's error copy), and the setting gate (master off / per-type
off fire nothing).
"""

from __future__ import annotations

import pytest

import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app as _build_base_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus

ERROR_COPY_CANARY = "sk-secret-provider-error-DO-NOT-LEAK"


def _build_test_app(*args, **kwargs):
    """Build a mounted-UI app without the first-run setup overlay."""
    app = _build_base_test_app(*args, **kwargs)
    _configure_native_ready_console(app)
    return app


class _RecordingSender:
    """Stand-in for ``send_os_notification`` capturing the gated calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, kind: str, session_display: str, **kwargs) -> bool:
        self.calls.append((kind, session_display))
        return True


def _wire_os_notifications(
    monkeypatch, config: dict | None
) -> _RecordingSender:
    """Patch the chat_screen seam: config table, synchronous spawn, recorder."""
    monkeypatch.setattr(
        chat_screen_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            config if key == "os_notifications" else default
        ),
    )
    recorder = _RecordingSender()
    monkeypatch.setattr(chat_screen_module, "send_os_notification", recorder)
    monkeypatch.setattr(
        chat_screen_module, "_spawn_os_notification_thread", lambda fn: fn()
    )
    return recorder


def _project_raw_identity(console, title: str, workspace: str) -> None:
    """Force the identity resolver like the escape tests do (raw strings)."""
    console._workspace._console_session_title_and_workspace_name = (
        lambda _controller, _session_id: (title, workspace)
    )


@pytest.mark.asyncio
@private_profile_test
async def test_background_run_outcome_fires_one_os_notification(
    request,
    monkeypatch,
) -> None:
    recorder = _wire_os_notifications(monkeypatch, {"enabled": True})
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.switch_session(viewed)
        _project_raw_identity(console, "Alpha Ops", "Lab North")

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=background
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )
        # Re-stamping the same terminal state must not double-notify.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )

        assert recorder.calls == [("run_finished", "Alpha Ops (Lab North)")]


@pytest.mark.asyncio
@private_profile_test
async def test_background_outcome_os_notification_excludes_message_copy(
    request,
    monkeypatch,
) -> None:
    """AC#3: the notification body carries the session display name and the
    outcome category only -- the run's own copy must never enter it."""
    recorder = _wire_os_notifications(monkeypatch, {"enabled": True})
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.switch_session(viewed)
        _project_raw_identity(console, "Alpha Ops", "Lab North")

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, ERROR_COPY_CANARY),
            session_id=background,
        )

        assert len(recorder.calls) == 1
        kind, display = recorder.calls[0]
        assert kind == "run_failed"
        assert ERROR_COPY_CANARY not in display
        assert "Alpha Ops" in display


@pytest.mark.asyncio
@private_profile_test
async def test_viewed_failure_os_notification_uses_title_not_error_copy(
    request,
    monkeypatch,
) -> None:
    """The viewed session's failure toast carries the run's visible copy;
    the OS notification must NOT -- it resolves the session title instead."""
    recorder = _wire_os_notifications(monkeypatch, {"enabled": True})
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        _project_raw_identity(console, "Front Ops", "Lab South")

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=viewed
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, ERROR_COPY_CANARY),
            session_id=viewed,
        )

        assert len(recorder.calls) == 1
        kind, display = recorder.calls[0]
        assert kind == "run_failed"
        assert ERROR_COPY_CANARY not in display
        assert "Front Ops" in display


@pytest.mark.asyncio
@private_profile_test
async def test_park_approval_os_notification_dedupes_with_the_toast(
    request,
    monkeypatch,
) -> None:
    """AC#4: the OS channel rides the park toast's own round-id once-guard
    -- a re-announcement of an unchanged parked round fires neither."""
    recorder = _wire_os_notifications(monkeypatch, {"enabled": True})
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.switch_session(viewed)
        _project_raw_identity(console, "Alpha Ops", "Lab North")

        controller._parked_approval_payloads["round-1"] = {
            "session_id": background,
            "round_id": "round-1",
        }
        console._park_console_approval(background)
        console._park_console_approval(background)

        assert recorder.calls == [("needs_approval", "Alpha Ops (Lab North)")]


@pytest.mark.asyncio
@private_profile_test
async def test_no_os_notification_when_master_setting_off(
    request,
    monkeypatch,
) -> None:
    recorder = _wire_os_notifications(monkeypatch, None)
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.switch_session(viewed)

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=background
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )
        console._park_console_approval(background)

        assert recorder.calls == []


@pytest.mark.asyncio
@private_profile_test
async def test_per_type_toggle_blocks_its_kind(request, monkeypatch) -> None:
    recorder = _wire_os_notifications(
        monkeypatch, {"enabled": True, "run_failed": False}
    )
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        failing = controller.new_session().id
        finishing = controller.new_session().id
        controller.store.switch_session(viewed)

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=failing
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, "bang"), session_id=failing
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=finishing
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=finishing
        )

        assert [kind for kind, _display in recorder.calls] == ["run_finished"]
