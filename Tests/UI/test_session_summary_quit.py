"""Quit-flow session summary integration tests (issue #365).

Harness idiom from Tests/UI/test_app_quit_guard.py: bind the REAL unbound
methods, stub only the collaborators they touch. CancelledError semantics
on the quit path must be preserved (lessons-textual).

The methods under test live on LifecycleMixin (tldw_chatbook/app_lifecycle.py),
reached through the TldwCli class proxy; get_cli_setting is read from the
app_lifecycle namespace, so that is what gets monkeypatched.

Importing tldw_chatbook.app pulls the guarded config bootstrap, which under
the per-test sandbox fails closed with RecoveryRequired("raw_source_selection_changed")
-- the signature Tests/conftest.py's keep-list documents. Keep the bootstrap profile.
"""

import asyncio
import time

import pytest

from tldw_chatbook import app_lifecycle
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.session_usage import reset_for_tests, session_usage

pytestmark = pytest.mark.bootstrap_profile


class _CleanupHarness:
    _run_approved_quit_cleanup = TldwCli._run_approved_quit_cleanup
    _show_session_summary_before_exit = TldwCli._show_session_summary_before_exit
    _session_summary_duration_seconds = TldwCli._session_summary_duration_seconds

    def __init__(self, *, never_dismiss: bool = False) -> None:
        self._startup_start_time = time.perf_counter()
        self.never_dismiss = never_dismiss
        self.exited = False
        self.pushed = []
        self.screen_stack = []  # await_quit_prompt polls this on the no-answer path

    async def _cleanup_audio_for_quit(self) -> None:
        return None

    def _run_blocking_quit_persistence(self) -> None:
        return None

    def push_screen(self, screen, **kwargs):
        # await_quit_prompt (the quit-flow choke point) uses
        # push_screen(wait_for_dismiss=True) and awaits the returned future.
        self.pushed.append(screen)
        future = asyncio.get_running_loop().create_future()
        if not self.never_dismiss:
            future.set_result(None)
        return future

    def notify(self, message, *, severity=None) -> None:
        return None

    def exit(self) -> None:
        self.exited = True


def _settings(enabled: bool, duration: int = 1):
    def setting(section, key, default=None):
        if section == "session_summary":
            if key == "enabled":
                return enabled
            if key == "duration_seconds":
                return duration
        return default

    return setting


async def test_disabled_summary_never_pushes(monkeypatch):
    monkeypatch.setattr(app_lifecycle, "get_cli_setting", _settings(enabled=False))
    harness = _CleanupHarness()
    await harness._run_approved_quit_cleanup()
    assert harness.pushed == []
    assert harness.exited is True


async def test_enabled_summary_pushes_then_exits(monkeypatch):
    monkeypatch.setattr(app_lifecycle, "get_cli_setting", _settings(enabled=True))
    reset_for_tests()
    session_usage().record_estimate("some prompt", "some reply")
    harness = _CleanupHarness()
    await harness._run_approved_quit_cleanup()
    assert len(harness.pushed) == 1
    assert harness.exited is True


async def test_stuck_dialog_hard_cap_still_exits(monkeypatch):
    monkeypatch.setattr(
        app_lifecycle, "get_cli_setting", _settings(enabled=True, duration=1)
    )
    harness = _CleanupHarness(never_dismiss=True)
    started = time.perf_counter()
    await harness._run_approved_quit_cleanup()
    elapsed = time.perf_counter() - started
    assert harness.exited is True
    # An unanswered prompt resolves via await_quit_prompt's vanish path
    # (~0.2s) and the wait_for remains a belt-and-braces cap -- either way
    # exit proceeds promptly, never blocking on the unresolved future.
    assert elapsed < 10.0
