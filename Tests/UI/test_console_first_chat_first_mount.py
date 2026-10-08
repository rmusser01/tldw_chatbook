"""TASK-34100.5 AC#1: the first-run handoff through Console's REAL first mount.

Every older handoff test opens the wizard over a Console that is already
mounted, so Console's "seed what is missing" rail-scope write never lands
between stage and consume (lessons-live-verification, TASK-33001.5 entry).
Adopting a saved workspace layout advances the global config generation, and the
old generation fence released a perfectly good handoff with "Provider settings
changed before Console opened. Review setup and try again." (8 of 8 live runs,
entry-exit-handoff-04). These tests start on Home so Console has never been
mounted, stage the handoff exactly as Start chatting does, and then let the
app's own wizard-result route open Console for the first time.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from Tests.app_module_patches import patch_app_global
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_product_maturity_phase1_first_run import _test_cli_setting
from tldw_chatbook.Chat.console_rail_state import build_console_rail_preference_key
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.config import (
    get_runtime_config_snapshot,
    save_settings_to_cli_config,
)
from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    SETUP_COMPLETED_KEY,
    SETUP_STARTED_KEY,
    WIZARD_STATE_SECTION,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

_SETTLE_TIMEOUT_SECONDS = 30.0


async def _wait_until(pilot, condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _SETTLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(0.05)
    assert condition(), "condition was not met in time"


def _persist_setup(model: str = "model-a") -> None:
    assert save_settings_to_cli_config(
        {
            "api_settings.custom": {
                "api_url": "http://127.0.0.1:8080/v1",
                "model": model,
            },
            "chat_defaults": {"provider": "custom", "model": model},
            WIZARD_STATE_SECTION: {
                SETUP_STARTED_KEY: True,
                SETUP_COMPLETED_KEY: True,
            },
        }
    )


def _staged_revision(app) -> int:
    """The staged handoff's exact revision (a claim+release requeues it)."""
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert app.pending_handoffs.release(claim)
    return claim.revision


def _handoff_status(app, revision: int) -> str:
    """Review round 1 (F12): ask for the exact revision's state. A claim that
    leaked in flight reads 'in_flight' here; ``claim() is None`` could not
    tell it from a settled one."""
    return app.pending_handoffs.exact_revision_status(
        HandoffChannel.CONSOLE_FIRST_CHAT, revision
    )


def _first_mount_app(
    monkeypatch, notices: list[tuple[str, str]], *, saved_layout: bool = False
):
    _persist_setup()
    if saved_layout:
        # Keep the real stage-to-consume generation race through durable
        # adoption; untouched product defaults no longer cause a write.
        key = build_console_rail_preference_key(
            workspace_id=DEFAULT_WORKSPACE_ID, layout_scope="workspace"
        )
        assert save_settings_to_cli_config(
            {"console.rail_state": {key.value: {"left_open": True}}}
        )
    app = _build_test_app(first_run_setup_completed=True)
    if saved_layout:
        assert app.app_config["console"]["rail_state"][key.value] == {"left_open": True}
        shared = build_console_rail_preference_key(layout_scope="global")
        assert shared.value not in app.app_config["console"]["rail_state"]
    app._initial_tab_value = "home"
    real_notify = app.notify

    def record(message, *args, **kwargs):
        notices.append((str(message), str(kwargs.get("severity", "information"))))
        return real_notify(message, *args, **kwargs)

    monkeypatch.setattr(app, "notify", record)
    return app


@pytest.mark.asyncio
@private_profile_test
async def test_start_chatting_through_consoles_first_mount_warns_nothing(
    monkeypatch: pytest.MonkeyPatch, request
) -> None:
    notices: list[tuple[str, str]] = []
    app = _first_mount_app(monkeypatch, notices, saved_layout=True)

    with patch_app_global("get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(pilot, lambda: type(app.screen).__name__ == "HomeScreen")
            assert not any(isinstance(s, ChatScreen) for s in app.screen_stack)
            staged_at = get_runtime_config_snapshot().generation
            assert SetupWizardContainer(app)._stage_console_first_chat_handoff()
            revision = _staged_revision(app)

            app.handle_first_run_wizard_result(
                {"completed": True, "exit_route": TAB_CHAT}
            )
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, ChatScreen) and app.screen.is_mounted,
            )
            console = app.screen
            await pilot.pause(1.0)

            # The race this test exists for really happened: Console's first
            # mount published a config write after the handoff was staged.
            assert get_runtime_config_snapshot().generation > staged_at
            assert [text for text, severity in notices if severity == "warning"] == []
            assert not any("Provider settings changed" in text for text, _ in notices)
            await _wait_until(
                pilot, lambda: _handoff_status(app, revision) == "settled"
            )
            # Review round 1 (C-F8): re-check after the settle wait (a late
            # warning would have been missed above), and count: the first-run
            # handoff raises at most one notice (AC#11).
            assert [text for text, severity in notices if severity == "warning"] == []
            assert len(notices) <= 1, notices
            shown = console._session._ensure_active_console_session_settings()
            assert (provider_config_key(shown.provider), shown.model) == (
                "custom",
                "model-a",
            )
            # Live 2026-10-03: once the handoff applied, the tab strip read
            # 'Chat 1 ✕  Chat 1 ✕' -- the reserved first chat opened next to
            # the untouched chat Console's own first mount had just made.
            store = console._console_chat_store
            tabs = [s.title for s in store.sessions() if not s.ephemeral]
            assert len(tabs) == 1, tabs


@pytest.mark.asyncio
@private_profile_test
async def test_a_default_changed_after_start_chatting_names_the_model_in_use(
    monkeypatch: pytest.MonkeyPatch, request
) -> None:
    notices: list[tuple[str, str]] = []
    app = _first_mount_app(monkeypatch, notices)

    with patch_app_global("get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(pilot, lambda: type(app.screen).__name__ == "HomeScreen")
            container = SetupWizardContainer(app)
            assert container._stage_console_first_chat_handoff()
            claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
            assert claim is not None
            intent = claim.value
            assert isinstance(intent, ConsoleFirstChatIntent)
            assert app.pending_handoffs.release(claim)
            # A genuine change of the saved default between Start chatting and
            # Console's consume: the stale handoff must never apply.
            _persist_setup(model="model-b")

            app.handle_first_run_wizard_result(
                {"completed": True, "exit_route": TAB_CHAT}
            )
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, ChatScreen) and app.screen.is_mounted,
            )
            console = app.screen
            await _wait_until(
                pilot,
                lambda: _handoff_status(app, claim.revision)
                not in {"pending", "in_flight"},
            )
            await pilot.pause(0.3)

            store = console._session._ensure_console_chat_store()
            assert all(session.id != intent.session_id for session in store.sessions())
            assert not any("Review setup" in text for text, _ in notices)
            changed = [text for text, _ in notices if "model-b" in text]
            assert len(changed) == 1, notices
            assert "model-a" not in changed[0]
            shown = console._session._ensure_active_console_session_settings()
            assert shown.model == "model-b"
