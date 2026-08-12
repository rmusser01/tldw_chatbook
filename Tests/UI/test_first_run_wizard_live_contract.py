"""App-level Pilot contract tests standing in for Task 12's live-verification
checklist (see backlog/docs/lessons-live-verification.md: no interactive
terminal is available in this environment, so every MECHANICALLY-CHECKABLE
item of the checklist is pinned here against a real ``TldwCli`` app via
``app.run_test()`` instead of a fabricated manual walkthrough).

Two checklist items genuinely need a human's eyes (splash-ON boot, and
overall visual look-and-feel) and are recorded as "needs human spot-check"
in the backlog task notes instead of being faked here.

Every test uses ``_build_test_app`` (see Tests/UI/test_screen_navigation.py),
the same real-app harness ``test_product_maturity_phase1_first_run.py``
uses, and the same ``TLDW_CONFIG_PATH``-isolated config the root conftest's
autouse ``isolate_test_environment`` fixture already provides -- no test
here ever touches a real user config file.

Two empirically-found traps drove the choices below:

- ``pilot.app.workers.wait_for_complete()`` waits for EVERY worker,
  including ``ProviderStep``'s real (unmocked) local-server discovery
  worker, which can block for a very long time with no local server
  reachable. Tests poll for the specific condition they need instead.
- A pixel-coordinate ``pilot.click(selector)`` resolves its target from the
  widget's own cached ``region``, and that can go stale (observed directly:
  ``app.get_widget_at()`` at a button's own reported region center resolved
  to its *parent* step, not the button, after this wizard's Summary step
  filled in async content) without ``pilot.click`` raising -- it just
  returns ``False`` and the test silently proceeds as if the click landed.
  Every state-changing interaction here therefore drives the widget
  directly (``Button.press()`` / setting ``RadioButton.value``), which is
  exactly what a click ultimately posts, without depending on compositor
  timing that is irrelevant to what these tests check. The one test that
  legitimately needs pixel/render truth (80x24 clipping) checks
  ``region`` and the compositor directly instead of clicking anything.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Input, OptionList, RadioButton, Static, Switch

from Tests.UI.test_product_maturity_phase1_first_run import (
    _prepare_clean_environment,
    _test_cli_setting,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    ModelStep,
    SpeechSetupStep,
    SetupWizardContainer,
    _SettlingGuardedConfirmationDialog,
)
from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    SETUP_COMPLETED_KEY,
    SETUP_DRAFT_KEYS,
    STEP_APPEARANCE,
    STEP_MODEL,
    STEP_NOTES,
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_SUMMARY,
    STEP_WELCOME,
    TRACK_QUICK,
    WIZARD_STATE_SECTION,
    SETUP_STARTED_KEY,
)


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _press(screen, selector: str) -> None:
    """Press a Button by selector -- posts Button.Pressed exactly like a
    real click, without depending on the widget's cached screen region.
    """
    screen.query_one(selector, Button).press()


def _select_radio(screen, selector: str) -> None:
    """Select a RadioButton by selector (mirrors what a click toggles)."""
    screen.query_one(selector, RadioButton).value = True


def _build_fresh_wizard_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A truly fresh config: no provider configured, no first_run state at
    all -- the exact condition ``should_offer_wizard`` auto-offers under.
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


async def _open_settings_diagnostics(pilot) -> None:
    """Navigate the real shell to Settings, then its Diagnostics category."""
    app = pilot.app
    # The nav strip mounts a tick after the initial screen swap (same race
    # noted in test_product_maturity_phase1_first_run.py); wait for the
    # button to actually exist before pressing it.
    await _wait_until(pilot, lambda: len(app.screen.query("#nav-settings")) == 1)
    _press(app.screen, "#nav-settings")
    await _wait_until(
        pilot,
        lambda: app.current_tab == "settings"
        and app.screen.__class__.__name__ == "SettingsScreen",
    )
    await pilot.pause(0.2)
    _press(app.screen, "#settings-category-diagnostics")
    await pilot.pause(0.2)


# ---------------------------------------------------------------------------
# 1. Fresh config, splash OFF -> wizard auto-offers (checklist item 3).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_config_splash_disabled_wizard_auto_offers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checklist item 3: with splash pre-seeded OFF, the wizard must still
    auto-offer on a truly fresh config -- the auto-offer path does not run
    through the splash screen's own post-mount hook, so this pins that the
    no-splash boot path (``_run_no_splash_post_mount_setup``) wires the
    same ``_maybe_offer_first_run_wizard`` call as the splash path.
    """
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            assert type(app.screen).__name__ == "FirstRunSetupWizard"
            assert app.current_tab == "home"  # wizard is pushed ON TOP, not swapped in
            # Basic navigation sanity: the chrome the rest of this file relies
            # on is actually present, not just the screen class name.
            for widget_id in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                assert len(app.screen.query(widget_id)) == 1


# ---------------------------------------------------------------------------
# 2. Esc -> confirm -> Finish later -> dismissed; next boot shows the resume
#    toast instead of re-pushing (checklist item 4).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escape_finish_later_dismisses_and_next_boot_offers_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            await pilot.press("escape")
            await pilot.pause(0.2)
            # The confirm dialog is on top; the wizard must still be mounted.
            # TASK-2314: this is now `_SettlingGuardedConfirmationDialog`, a
            # `ConfirmationDialog` subclass that absorbs a reflexive
            # double-tap of Escape while the wizard is still settling --
            # `isinstance` captures the real contract (a confirm dialog is
            # up) without pinning to the exact subclass name.
            assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
            _press(app.screen, "#confirm-button")  # confirm_label="Finish later"
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # Dismissed back onto whatever was underneath (Home), not
            # navigated anywhere -- Escape/Finish-later carries no exit route.
            assert app.current_tab == "home"

            # The started flag is persisted by a `@work(thread=True)` worker
            # fired from FirstRunSetupWizard.on_mount(). Poll for the flag
            # directly with a bound, rather than `workers.wait_for_complete()`
            # -- that call waits for EVERY worker including ProviderStep's
            # real (unmocked) local-server discovery, which can block far
            # longer than any reasonable per-assertion timeout in a sandboxed
            # test environment with no reachable local servers.
            await _wait_until(
                pilot,
                lambda: app.app_config.get(WIZARD_STATE_SECTION, {}).get(
                    SETUP_STARTED_KEY
                )
                is True,
            )

    # Prove it is a REAL write, not just the in-memory mirror: read the same
    # (test-isolated) config file back independently.
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    persisted_config = load_cli_config_and_ensure_existence(force_reload=True)
    assert (
        persisted_config.get(WIZARD_STATE_SECTION, {}).get(SETUP_STARTED_KEY) is True
    )

    # Next boot: a fresh TldwCli instance reading that SAME real persisted
    # checkpoint must offer bounded recovery and must NOT directly re-push
    # the wizard.
    app2 = _build_test_app(first_run_setup_completed=False)
    app2.app_config = persisted_config
    app2._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app2.run_test(size=(140, 40)) as pilot2:
            await _wait_until(
                pilot2,
                lambda: type(app2.screen).__name__ == "SetupRecoveryDialog",
            )
            assert type(app2.screen).__name__ != "FirstRunSetupWizard"
            assert app2.current_tab == "home"


@pytest.mark.asyncio
async def test_recovery_dialog_has_exact_labels_initial_focus_and_credential_reentry():
    from textual.app import App
    from tldw_chatbook.UI.Wizards.first_run_recovery_dialog import SetupRecoveryDialog

    class RecoveryHost(App):
        def on_mount(self):
            self.push_screen(SetupRecoveryDialog())

    app = RecoveryHost()
    async with app.run_test(size=(60, 20)):
        dialog = app.screen
        buttons = {
            button.id.removeprefix("setup-recovery-"): button
            for button in dialog.query(Button)
        }

        assert {action: str(button.label) for action, button in buttons.items()} == {
            "resume": "Resume",
            "start_over": "Start over",
            "later": "Later",
        }
        assert app.focused is buttons["resume"]
        assert "credentials" in dialog.message.lower()
        assert "not retained" in dialog.message.lower()
        assert "re-enter" in dialog.message.lower()
        assert dialog.query_one("Container").region.width <= app.size.width


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interaction", "expected"),
    [("enter", "resume"), ("escape", "later"), ("start_over", "start_over")],
)
async def test_recovery_dialog_keyboard_and_button_actions(interaction, expected):
    from textual.app import App
    from tldw_chatbook.UI.Wizards.first_run_recovery_dialog import SetupRecoveryDialog

    results = []

    class RecoveryHost(App):
        def on_mount(self):
            self.push_screen(SetupRecoveryDialog(), results.append)

    app = RecoveryHost()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        if interaction == "start_over":
            app.screen.query_one("#setup-recovery-start_over", Button).press()
        else:
            await pilot.press(interaction)
        await _wait_until(pilot, lambda: bool(results))

    assert results == [expected]


@pytest.mark.asyncio
async def test_successful_step_checkpoints_once_after_commit_before_navigation():
    from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER

    events: list[str] = []
    container = SetupWizardContainer(SimpleNamespace(app_config={}))
    step = container.steps[container._step_index_for_id(STEP_PROVIDER)]

    async def commit():
        events.append("commit")
        return True, ""

    step.commit = commit
    step.get_step_data = lambda: {"provider_value": "openai"}
    container.current_step = container._step_index_for_id(STEP_PROVIDER)
    container.wizard_data = {}
    container.track = TRACK_QUICK
    container.active_ids = (STEP_PROVIDER, STEP_MODEL)
    container._set_advancing = lambda active: None
    container._next_active_index = lambda current: container._step_index_for_id(STEP_MODEL)
    container.show_step = lambda index: events.append("navigate")

    async def checkpoint(next_step_id):
        events.append(f"checkpoint:{next_step_id}")
        return True

    container.persist_setup_checkpoint = checkpoint

    await container._advance()

    assert events == ["commit", f"checkpoint:{STEP_MODEL}", "navigate"]


@pytest.mark.asyncio
async def test_failed_step_commit_does_not_checkpoint_or_navigate():
    events: list[str] = []
    container = SetupWizardContainer(SimpleNamespace(app_config={}))
    step = container.steps[container._step_index_for_id(STEP_PROVIDER)]

    async def commit():
        events.append("commit")
        return False, "save failed"

    step.commit = commit
    step.get_step_data = lambda: {"provider_value": "openai"}
    step.show_step_error = lambda message: events.append("error")
    container.current_step = container._step_index_for_id(STEP_PROVIDER)
    container.wizard_data = {}
    container.track = TRACK_QUICK
    container._set_advancing = lambda active: None

    async def checkpoint(next_step_id):
        events.append("checkpoint")
        return True

    container.persist_setup_checkpoint = checkpoint
    container.show_step = lambda index: events.append("navigate")

    await container._advance()

    assert events == ["commit", "error"]


@pytest.mark.asyncio
async def test_completion_marks_complete_and_clears_draft_atomically():
    calls = []
    container = object.__new__(SetupWizardContainer)
    container._finalized = False
    container._dismiss_screen = lambda result: calls.append(("dismiss", result))

    async def commit_config(settings, *, delete_keys=None, after_write=None):
        calls.append(("commit", settings, delete_keys))
        return True

    container.commit_config = commit_config

    await container._finalize(None)

    assert calls[0] == (
        "commit",
        {"first_run": {SETUP_COMPLETED_KEY: True}},
        {"first_run": SETUP_DRAFT_KEYS},
    )
    assert calls[1][0] == "dismiss"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["finish", "skip"])
async def test_completion_write_failure_keeps_wizard_open_and_retains_draft(operation):
    events = []
    retained_draft = object()
    container = SetupWizardContainer(SimpleNamespace(app_config={}))
    container._finalized = False
    container.resume_draft = retained_draft
    container.current_step = 0
    container.steps[0].show_step_error = lambda message: events.append(
        ("error", message)
    )
    container._dismiss_screen = lambda result: events.append(("dismiss", result))

    async def commit_config(settings, *, delete_keys=None, after_write=None):
        events.append(("commit", settings, delete_keys))
        return False

    container.commit_config = commit_config

    if operation == "finish":
        await container._finalize("chat")
    else:
        await container._skip_entirely()

    assert [event[0] for event in events] == ["commit", "error"]
    assert "could not be saved" in events[1][1].lower()
    assert container._finalized is False
    assert container.resume_draft is retained_draft


@pytest.mark.parametrize("save_result", [False, RuntimeError("private-value")])
def test_started_flag_mirrors_only_after_success_and_logs_bounded_failure(
    monkeypatch, save_result
):
    from tldw_chatbook.UI.Wizards import FirstRunSetupWizard as wizard_module

    app_instance = SimpleNamespace(app_config={"first_run": {"unrelated": "keep"}})
    screen = SimpleNamespace(app_instance=app_instance)
    warning = MagicMock()

    def save(*args, **kwargs):
        if isinstance(save_result, Exception):
            raise save_result
        return save_result

    monkeypatch.setattr("tldw_chatbook.config.save_settings_to_cli_config", save)
    monkeypatch.setattr(wizard_module.logger, "warning", warning)

    FirstRunSetupWizard._persist_started_flag.__wrapped__(screen)

    assert app_instance.app_config["first_run"] == {"unrelated": "keep"}
    warning.assert_called_once()
    rendered_log = repr(warning.call_args)
    assert "category=persistence" in rendered_log
    assert "private-value" not in rendered_log


@pytest.mark.asyncio
async def test_resume_marks_attempt_before_pushing_restored_wizard(monkeypatch):
    from tldw_chatbook.app import TldwCli

    app_config = {
        "first_run": {
            "setup_started": True,
            "draft_version": 1,
            "draft_track": "quick",
            "active_step_id": "model",
            "draft_values": {
                "welcome": {"track": "quick"},
                "provider": {"provider_value": "openai"},
            },
            "resume_attempted": False,
        }
    }
    events = []
    fake = SimpleNamespace(app_config=app_config)
    fake._mirror_first_run_setup_mutation = lambda settings, deletes: events.append(
        ("mirror", settings, deletes)
    )
    fake._handle_first_run_wizard_result = lambda result: None
    fake.push_screen = lambda screen, callback: events.append(("push", screen, callback))

    def save(settings, *, delete_keys=None):
        events.append(("save", settings, delete_keys))
        return True

    monkeypatch.setattr("tldw_chatbook.config.save_settings_to_cli_config", save)

    await TldwCli._apply_first_run_recovery_result(fake, "resume")

    assert [event[0] for event in events] == ["save", "mirror", "push"]
    assert events[0][1]["first_run"]["resume_attempted"] is True
    pushed_wizard = events[2][1]
    assert pushed_wizard.resume_draft.active_step_id == STEP_MODEL
    assert pushed_wizard.resume_draft.resume_attempted is True
    assert pushed_wizard.resume_draft.values["provider"] == {
        "provider_value": "openai"
    }


@pytest.mark.asyncio
async def test_start_over_deletes_only_draft_keys_and_pushes_clean_wizard(monkeypatch):
    from tldw_chatbook.app import TldwCli

    fake = SimpleNamespace(
        app_config={
            "first_run": {
                "setup_started": True,
                "setup_completed": False,
                "unrelated": "keep",
                **{key: "stale" for key in SETUP_DRAFT_KEYS},
            }
        }
    )
    events = []
    fake._mirror_first_run_setup_mutation = lambda settings, deletes: (
        TldwCli._mirror_first_run_setup_mutation(fake, settings, deletes)
    )
    fake._handle_first_run_wizard_result = lambda result: None
    fake.push_screen = lambda screen, callback: events.append((screen, callback))

    mutations = []

    def save(settings, *, delete_keys=None):
        mutations.append((settings, delete_keys))
        return True

    monkeypatch.setattr("tldw_chatbook.config.save_settings_to_cli_config", save)

    await TldwCli._apply_first_run_recovery_result(fake, "start_over")

    assert mutations == [({}, {"first_run": SETUP_DRAFT_KEYS})]
    assert fake.app_config["first_run"] == {
        "setup_started": True,
        "setup_completed": False,
        "unrelated": "keep",
    }
    assert len(events) == 1
    assert events[0][0].resume_draft is None


@pytest.mark.asyncio
async def test_recovery_later_performs_no_mutation_or_push(monkeypatch):
    from tldw_chatbook.app import TldwCli

    fake = SimpleNamespace(app_config={})
    fake.push_screen = MagicMock()
    save = MagicMock()
    monkeypatch.setattr("tldw_chatbook.config.save_settings_to_cli_config", save)

    await TldwCli._apply_first_run_recovery_result(fake, "later")

    save.assert_not_called()
    fake.push_screen.assert_not_called()


def _attempted_model_resume_config():
    return {
        "first_run": {
            "setup_started": True,
            "setup_completed": False,
            "draft_version": 1,
            "draft_track": "quick",
            "active_step_id": "model",
            "draft_values": {
                "welcome": {"track": "quick"},
                "provider": {
                    "provider_key": "openai",
                    "provider_value": "openai",
                },
            },
            "resume_attempted": True,
        }
    }


def _attempted_full_appearance_resume_config():
    return {
        "first_run": {
            "setup_started": True,
            "setup_completed": False,
            "draft_version": 1,
            "draft_track": "full",
            "active_step_id": "appearance",
            "draft_values": {
                "welcome": {"track": "full"},
                "provider": {
                    "provider_key": "openai",
                    "provider_value": "openai",
                },
                "model": {"model_id": "gpt-resume-test"},
                "notes": {"auto_sync_enabled": True},
                "appearance": {
                    "theme": "textual-light",
                    "splash_card": "",
                },
                "protect-keys": {"encryption_enabled": True},
            },
            "resume_attempted": True,
        }
    }


@pytest.mark.asyncio
async def test_resumed_target_restores_then_clears_attempt_after_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config = _attempted_full_appearance_resume_config()
    app._initial_tab_value = "chat"
    draft = read_setup_draft(app.app_config)
    assert draft is not None

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: getattr(app, "_initial_screen_pushed", False) is True
            )
            await app.push_screen(FirstRunSetupWizard(app, resume_draft=draft))
            await _wait_until(
                pilot,
                lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
                and app.screen.query_one(SetupWizardContainer)
                .steps[app.screen.query_one(SetupWizardContainer).current_step]
                .config.id
                == "appearance",
            )
            container = app.screen.query_one(SetupWizardContainer)
            assert container.track == "full"
            assert container.wizard_data[STEP_PROVIDER]["provider_value"] == "openai"
            assert app.screen.query_one("#setup-track-full", RadioButton).value is True
            assert app.screen.query_one("#setup-track-quick", RadioButton).value is False

            provider_step = container.steps[container._step_index_for_id(STEP_PROVIDER)]
            assert provider_step.selected_provider_key == "openai"
            provider_choice = provider_step.query_one(
                "#setup-provider-choice", OptionList
            )
            assert provider_choice.highlighted_option.provider_key == "openai"

            model_step = container.steps[container._step_index_for_id(STEP_MODEL)]
            assert model_step.selected_model_id == "gpt-resume-test"
            assert (
                model_step.query_one("#setup-model-custom", Input).value
                == "gpt-resume-test"
            )
            notes_step = container.steps[container._step_index_for_id(STEP_NOTES)]
            assert notes_step.get_step_data() == {"auto_sync_enabled": True}
            assert notes_step.query_one("#setup-notes-enable", Switch).value is True

            appearance_step = container.steps[
                container._step_index_for_id(STEP_APPEARANCE)
            ]
            assert appearance_step.get_step_data() == {
                "theme": "textual-light",
                "splash_card": "",
            }
            selected_themes = [
                getattr(button, "_theme_name", "")
                for button in appearance_step.query("#setup-theme-choice RadioButton")
                if button.value
            ]
            assert selected_themes == ["textual-light"]
            selected_splash = [
                str(button.label)
                for button in appearance_step.query("#setup-splash-choice RadioButton")
                if button.value
            ]
            assert selected_splash == ["Surprise me (random)"]

            protect_step = container.steps[
                container._step_index_for_id(STEP_PROTECT)
            ]
            assert protect_step.get_step_data() == {"encryption_enabled": True}
            assert "Encryption enabled" in str(
                protect_step.query_one("#setup-protect-status", Static).renderable
            )
            await _wait_until(
                pilot,
                lambda: app.app_config["first_run"]["resume_attempted"] is False,
            )


@pytest.mark.asyncio
async def test_resume_control_restore_failure_keeps_attempt_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config = _attempted_model_resume_config()
    draft = read_setup_draft(app.app_config)
    assert draft is not None

    def fail_restore(*args, **kwargs):
        raise RuntimeError("control-value-must-not-log")

    monkeypatch.setattr(SetupWizardContainer, "_restore_radio_selection", fail_restore)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: getattr(app, "_initial_screen_pushed", False) is True
            )
            await app.push_screen(FirstRunSetupWizard(app, resume_draft=draft))
            await pilot.pause(0.3)

            assert app.app_config["first_run"]["resume_attempted"] is True


@pytest.mark.asyncio
async def test_resume_target_change_before_after_refresh_keeps_attempt_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config = _attempted_model_resume_config()
    draft = read_setup_draft(app.app_config)
    assert draft is not None
    callbacks = []
    wizard = FirstRunSetupWizard(app, resume_draft=draft)
    original_call_after_refresh = wizard.call_after_refresh

    def capture_resume_callback(callback, *args):
        if callback.__name__ == "_clear_resume_attempt_after_target_mount":
            callbacks.append((callback, args))
            return True
        return original_call_after_refresh(callback, *args)

    wizard.call_after_refresh = capture_resume_callback

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: getattr(app, "_initial_screen_pushed", False) is True
            )
            await app.push_screen(wizard)
            await _wait_until(pilot, lambda: bool(callbacks))
            container = wizard.query_one(SetupWizardContainer)
            container.show_step(container._step_index_for_id(STEP_WELCOME))
            scheduled = []
            wizard.run_worker = lambda awaitable, **kwargs: scheduled.append(awaitable)

            callback, args = callbacks.pop()
            assert callback.__name__ == "_clear_resume_attempt_after_target_mount"
            callback(*args)
            if scheduled:
                await scheduled.pop()
            await pilot.pause(0.2)

            assert app.app_config["first_run"]["resume_attempted"] is True


@pytest.mark.asyncio
async def test_failed_resumed_mount_leaves_attempt_and_next_launch_on_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        read_setup_draft,
        setup_recovery_action,
    )

    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config = _attempted_model_resume_config()
    app._initial_tab_value = "chat"
    draft = read_setup_draft(app.app_config)
    assert draft is not None

    def fail_model_compose(self):
        raise RuntimeError("bounded test mount failure")

    monkeypatch.setattr(ModelStep, "compose_step", fail_model_compose)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: getattr(app, "_initial_screen_pushed", False) is True
            )
            await app.push_screen(FirstRunSetupWizard(app, resume_draft=draft))
            await _wait_until(
                pilot,
                lambda: type(app.screen).__name__ == "FirstRunSetupWizard",
            )
            container = app.screen.query_one(SetupWizardContainer)
            model = container.steps[container._step_index_for_id(STEP_MODEL)]
            await _wait_until(pilot, lambda: model.compose_failed)
            await pilot.pause(0.2)
            assert app.app_config["first_run"]["resume_attempted"] is True

    assert setup_recovery_action(app.app_config, {}) == "home"

    app2 = _build_test_app(first_run_setup_completed=False)
    app2.app_config = app.app_config
    app2._initial_tab_value = "chat"
    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app2.run_test(size=(120, 40)) as pilot2:
            await _wait_until(
                pilot2, lambda: getattr(app2, "_initial_screen_pushed", False) is True
            )
            await pilot2.pause(0.2)
            assert app2.current_tab == TAB_HOME
            assert type(app2.screen).__name__ not in {
                "FirstRunSetupWizard",
                "SetupRecoveryDialog",
            }


# ---------------------------------------------------------------------------
# 3. Full track, skip every step -> app fully usable afterwards
#    (checklist item 5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_track_skip_everything_leaves_app_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)

            _select_radio(app.screen, "#setup-track-full")
            await pilot.pause(0.1)
            _press(app.screen, "#wizard-next")  # Welcome -> Provider, track=full
            await pilot.pause(0.2)

            seen_step_ids: list[str] = []
            for _ in range(12):
                step = container.steps[container.current_step]
                step_id = step.config.id if step.config else None
                if step_id == STEP_SUMMARY:
                    break
                seen_step_ids.append(step_id)
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)
            else:
                raise AssertionError("never reached the summary step")

            # TASK-1301: Speech transcription joins the FULL track right
            # after RAG; every step here is skip-safe with nothing selected.
            assert seen_step_ids == [
                "provider",
                "model",
                "rag",
                "speech",
                "tools",
                "notes",
                "appearance",
            ]

            # Exit via "Explore on my own" (TAB_HOME) to prove the app is
            # usable afterwards, not just that the wizard closed.
            _press(app.screen, "#setup-exit-home")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            await _wait_until(
                pilot,
                lambda: app.current_tab == TAB_HOME
                and app.screen.__class__.__name__ == "HomeScreen",
            )

            # "Fully usable": the shell nav still works after the wizard.
            _press(app.screen, "#nav-console")
            await _wait_until(pilot, lambda: app.current_tab == TAB_CHAT)
            assert app.current_tab == TAB_CHAT


# ---------------------------------------------------------------------------
# 4. Re-run entry over Settings -> finishing via "Done" returns to Settings
#    (exit_route None path; checklist item 6, partial -- prefill/"configured"
#    copy is covered at the unit level in Tests/Wizards/).
# ---------------------------------------------------------------------------


async def _open_rerun_wizard_from_settings(pilot):
    """Drive the real Settings ▸ Diagnostics ▸ "Run setup wizard" button.

    Returns the pushed FirstRunSetupWizard screen. Shared by the "Done" and
    "Go to Console" re-entry tests below.
    """
    app = pilot.app
    await _wait_until(
        pilot, lambda: app.screen.__class__.__name__ in ("HomeScreen", "ChatScreen")
    )
    await _open_settings_diagnostics(pilot)
    run_wizard_button = app.screen.query_one("#settings-run-setup-wizard", Button)
    assert "Run Setup Wizard" in str(run_wizard_button.label)

    run_wizard_button.press()
    await _wait_until(
        pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
    )
    wizard_screen = app.screen
    assert wizard_screen.rerun is True
    await pilot.pause(0.2)
    return wizard_screen


async def _walk_rerun_quick_track_to_summary(pilot, wizard_screen) -> "SetupWizardContainer":
    # Quick track is pre-selected; walk welcome -> provider -> model ->
    # summary without picking anything (every step is skip-safe).
    for _ in range(3):
        _press(wizard_screen, "#wizard-next")
        await pilot.pause(0.2)
    container = wizard_screen.query_one(SetupWizardContainer)
    step = container.steps[container.current_step]
    assert step.config.id == STEP_SUMMARY
    return container


@pytest.mark.asyncio
async def test_rerun_over_settings_done_returns_to_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_clean_environment(monkeypatch, tmp_path)
    # Already-completed config: no auto-offer noise for this test.
    app = _build_test_app(first_run_setup_completed=True)
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 55)) as pilot:
            wizard_screen = await _open_rerun_wizard_from_settings(pilot)
            await _walk_rerun_quick_track_to_summary(pilot, wizard_screen)
            # Rerun summary exposes "Done"/"Go to Console", not the first-run
            # exit pair -- "Done" is the exit_route=None path.
            assert len(app.screen.query("#setup-exit-done")) == 1

            _press(app.screen, "#setup-exit-done")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # the final-review fix wave wired a result callback onto this push
            # (settings_screen.py's handle_run_setup_wizard now passes
            # app_instance.handle_first_run_wizard_result), but "Done"'s
            # exit_route is None -- _handle_first_run_wizard_result() is a
            # no-op for a falsy exit_route, so this must still simply pop
            # back to Settings with no navigation side effect.
            assert type(app.screen).__name__ == "SettingsScreen"
            assert app.current_tab == "settings"


@pytest.mark.asyncio
async def test_rerun_over_settings_go_to_chat_navigates_to_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Final-review finding 2: before the fix, both re-entry pushes
    (Settings' button and the command palette) omitted the result
    callback, so a truthy exit_route off the Summary step's "Go to Console"
    button was silently dropped -- the button looked live but did nothing.
    Now that settings_screen.py's push wires
    app_instance.handle_first_run_wizard_result, this must actually
    navigate to Chat, exactly like the auto-offer path already does in
    test_full_track_skip_everything_leaves_app_usable above.
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=True)
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 55)) as pilot:
            wizard_screen = await _open_rerun_wizard_from_settings(pilot)
            await _walk_rerun_quick_track_to_summary(pilot, wizard_screen)
            assert len(app.screen.query("#setup-exit-chat")) == 1

            _press(app.screen, "#setup-exit-chat")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # Dismiss pops back to Settings first; the exit_route is applied
            # via a separately-queued NavigateToScreen message (same race
            # noted in test_back_next_mashing_... above) -- wait for the
            # final tab rather than racing the first screen-stack pop.
            await _wait_until(pilot, lambda: app.current_tab == TAB_CHAT)
            assert app.current_tab == TAB_CHAT


# ---------------------------------------------------------------------------
# 4b. Command-palette re-entry wires the same result callback (finding 2,
#     palette path) -- cheap coverage over the actual production code in
#     SetupWizardProvider.handle_setup_wizard_action, without driving the
#     full command-palette search UI.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_palette_setup_wizard_action_wires_result_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tldw_chatbook.app import SetupWizardProvider

    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=True)
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.screen.__class__.__name__ in ("HomeScreen", "ChatScreen"),
            )
            captured: dict = {}
            real_push_screen = app.push_screen

            def _spy_push_screen(screen, callback=None, **kwargs):
                captured["screen"] = screen
                captured["callback"] = callback
                return real_push_screen(screen, callback, **kwargs)

            monkeypatch.setattr(app, "push_screen", _spy_push_screen)

            provider = SetupWizardProvider(app.screen)
            provider.handle_setup_wizard_action("run_setup_wizard")
            await pilot.pause(0.2)

            assert captured.get("callback") == app.handle_first_run_wizard_result, (
                "palette re-entry must wire the app-level result callback, "
                "same as the Settings button and the auto-offer path"
            )
            assert type(app.screen).__name__ == "FirstRunSetupWizard"


# ---------------------------------------------------------------------------
# 5. 80x24 terminal: wizard renders without clipped navigation
#    (checklist item 7).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wizard_navigation_visible_at_80x24(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            nav_buttons = {
                widget_id: app.screen.query_one(widget_id, Button)
                for widget_id in ("#wizard-back", "#wizard-next", "#wizard-cancel")
            }
            for widget_id, button in nav_buttons.items():
                assert button.visible, f"{widget_id} is not visible at 80x24"
                region = button.region
                assert region.width > 0 and region.height > 0, (
                    f"{widget_id} has an empty region at 80x24: {region}"
                )
                assert region.right <= 80, f"{widget_id} clipped past column 80: {region}"
                assert region.bottom <= 24, f"{widget_id} clipped past row 24: {region}"

            # Cross-check against the actual compositor output rather than
            # trusting pre-paint widget state alone (a clipped overlay can
            # report a plausible region and never actually reach the
            # screen -- see backlog/docs/lessons-live-verification.md).
            strips = app.screen._compositor.render_strips()
            rendered_text = "\n".join(
                "".join(segment.text for segment in strip) for strip in strips
            )
            # TASK-2154.9 (FR-01): the cancel button now carries its
            # consequence as the label -- "Finish later" (same dialog as
            # Esc) -- so the rendered frame is checked for that wording.
            for expected in ("Back", "Next", "Finish later"):
                assert expected in rendered_text, (
                    f"{expected!r} button text missing from the rendered frame"
                )


# ---------------------------------------------------------------------------
# 5b. TASK-1495: Provider's own step viewport clipped its content with no
#     scrollbar -- the API-key Input (and everything after the RadioSet)
#     rendered below a hard, non-scrolling fold at 120x40. Root cause:
#     BaseWizard.py's shared ".wizard-step" is "height: 100%" with no
#     overflow (never modified -- see FirstRunSetupWizard.py's own
#     docstring), and each step's own content wrapper inherited Textual's
#     Vertical default of "height: 1fr; overflow: hidden hidden", so
#     anything taller than the step's fixed viewport was clipped by that
#     INNER wrapper before ".wizard-steps-container"'s own
#     "overflow-y: auto" ever got a chance to scroll anything. The fix scopes
#     new CSS to ".setup-step" (added by SetupStep.__init__, this module
#     only) and caps each step's internal RadioSet ("setup-choice-list") --
#     both are scoped to the setup wizard's own classes, so the Chatbook
#     wizards (whose steps carry neither class) are unaffected; see
#     Tests/Chatbooks/ in the full suite gate for that invariant.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_key_input_visible_at_120x40_without_scrolling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-1495 AC #1: the Provider step's API-key field must be visible
    at 120x40 with NO scrolling -- not merely reachable by scrolling (that
    weaker guarantee is covered by the ".setup-step" scroll region itself,
    exercised by the 80x24/100x30 tests below). Capping the provider
    RadioSet ("setup-choice-list", max-height: 5) and trimming this step's
    own padding/margins (see _wizards.tcss's "First-run setup wizard"
    section) reclaims just enough of the step's fixed ~15-row viewport for
    the Input to fit un-scrolled alongside it.
    """
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            _press(app.screen, "#wizard-next")  # Welcome -> Provider
            await pilot.pause(0.3)
            container = app.screen.query_one(SetupWizardContainer)
            assert container.steps[container.current_step].config.id == STEP_PROVIDER

            key_input = app.screen.query_one("#setup-provider-key-input", Input)
            region = key_input.region
            assert region.width > 0 and region.height > 0, (
                f"key Input has an empty region at 120x40: {region}"
            )
            assert region.y >= 0, f"key Input clipped above row 0: {region}"
            assert region.bottom <= 40, f"key Input clipped past row 40: {region}"
            assert region.right <= 120, f"key Input clipped past column 120: {region}"

            # Cross-check against the actual compositor output: a widget
            # nested inside a scrollable ancestor can report a perfectly
            # plausible on-screen `region` while still being scrolled out of
            # that ancestor's visible clip window -- `region` alone does not
            # prove the compositor actually painted it (see this suite's own
            # docstring, and backlog/docs/lessons-live-verification.md).
            assert key_input in app.screen._compositor.visible_widgets, (
                "key Input's region looked on-screen, but the compositor "
                "never actually painted it -- it is scrolled out of view"
            )
            strips = app.screen._compositor.render_strips()
            rendered_text = "\n".join(
                "".join(segment.text for segment in strip) for strip in strips
            )
            assert "Paste your API key" in rendered_text, (
                "key Input's placeholder never reached the rendered frame"
            )


@pytest.mark.asyncio
async def test_summary_exit_buttons_visible_at_120x40_full_track(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-1495 AC #3: the Full-track Summary step's exit buttons ("Start
    chatting" / "Explore on my own") must be on screen at 120x40 after
    walking the entire 8-step full track -- these could previously render
    below the same non-scrolling fold as the Provider step's key Input,
    just reached via a longer path (more accumulated summary rows).
    SetupWizardContainer.show_step() (unrelated to this fix) auto-focuses
    the incoming step's first focusable descendant -- Summary's first exit
    Button -- and Textual's own Screen.set_focus(scroll_visible=True) then
    scrolls it into view, but only because ".setup-step" is now an actual
    scroll region at all (TASK-1495's CSS change); before the fix nothing
    under the step was scrollable, so focus-follows-into-view had nothing
    to work with.
    """
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)

            _select_radio(app.screen, "#setup-track-full")
            await pilot.pause(0.1)
            _press(app.screen, "#wizard-next")  # Welcome -> Provider, track=full
            await pilot.pause(0.2)

            for _ in range(10):
                step = container.steps[container.current_step]
                if step.config.id == STEP_SUMMARY:
                    break
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)
            else:
                raise AssertionError("never reached the summary step")

            # SummaryStep._render_rows() is an async worker that fills in
            # "#setup-summary-rows" after the step is shown; wait for it to
            # actually finish (rather than a fixed sleep) so the layout has
            # settled to its FINAL height before measuring anything below it.
            await _wait_until(
                pilot,
                lambda: bool(
                    str(
                        app.screen.query_one("#setup-summary-rows", Static).render()
                    ).strip()
                ),
            )
            await pilot.pause(0.2)

            exit_chat = app.screen.query_one("#setup-exit-chat", Button)
            exit_home = app.screen.query_one("#setup-exit-home", Button)
            strips = app.screen._compositor.render_strips()
            rendered_text = "\n".join(
                "".join(segment.text for segment in strip) for strip in strips
            )
            for button, label in (
                (exit_chat, "Open Console"),
                (exit_home, "Explore on my own"),
            ):
                region = button.region
                assert region.width > 0 and region.height > 0, (
                    f"{label!r} exit button has an empty region: {region}"
                )
                assert region.y >= 0 and region.bottom <= 40 and region.right <= 120, (
                    f"{label!r} exit button clipped at 120x40: {region}"
                )
                assert button in app.screen._compositor.visible_widgets, (
                    f"{label!r} exit button's region looked on-screen but the "
                    "compositor never painted it"
                )
            assert "Open Console" in rendered_text
            assert "Explore on my own" in rendered_text


@pytest.mark.asyncio
async def test_speech_step_install_button_visible_at_120x40_without_scrolling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-1301 review Important 2: the Speech step's primary action
    ("Review and install…") must be painted at the wizard's own tested
    120x40 budget, not merely reachable by scrolling -- same standard as
    test_provider_key_input_visible_at_120x40_without_scrolling. Before the
    fix, a live probe found the rendered frame ending after an empty
    Precision box with the button at y=35, painted=False; the fix moves the
    status/action row ahead of the informational language/precision
    RadioSets so it renders within the step's first few rows regardless of
    how tall that catalog grows below it.
    """
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            _select_radio(app.screen, "#setup-track-full")
            await pilot.pause(0.1)
            _press(app.screen, "#wizard-next")  # Welcome -> Provider, track=full
            await pilot.pause(0.2)

            for _ in range(4):
                step = app.screen.query_one(SetupWizardContainer).steps[
                    app.screen.query_one(SetupWizardContainer).current_step
                ]
                if step.config and step.config.id == "speech":
                    break
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)
            else:
                raise AssertionError("never reached the speech step")

            # The install-state load is a real background worker (real
            # ModelArtifactService over the isolated test data dir) -- wait
            # for it to settle so the button has actually rendered rather
            # than the transient "Checking installed models…" placeholder.
            # Existence in the DOM alone is not enough: a widget can be
            # mounted (query finds it) a beat before Textual's own layout
            # pass gives it a non-empty region, so wait for the region too
            # -- otherwise this flakes with Region(0, 0, 0, 0).
            def _speech_actions_laid_out() -> bool:
                install = app.screen.query("#setup-speech-install")
                disk = app.screen.query("#setup-speech-use-from-disk")
                return (
                    bool(install)
                    and install[0].region.height > 0
                    and bool(disk)
                    and disk[0].region.height > 0
                )

            await _wait_until(pilot, _speech_actions_laid_out, timeout_seconds=10.0)

            install_button = app.screen.query_one("#setup-speech-install", Button)
            disk_button = app.screen.query_one("#setup-speech-use-from-disk", Button)
            for button, label in (
                (disk_button, "Use model from disk"),
                (install_button, "Review and install"),
            ):
                region = button.region
                assert region.width > 0 and region.height > 0, (
                    f"{label} button has an empty region at 120x40: {region}"
                )
                assert region.y >= 0, f"{label} button clipped above row 0: {region}"
                assert region.bottom <= 40, (
                    f"{label} button clipped past row 40: {region}"
                )
                assert region.right <= 120, (
                    f"{label} button clipped past column 120: {region}"
                )
                assert button in app.screen._compositor.visible_widgets, (
                    f"{label} button's region looked on-screen, but the compositor "
                    "never painted it"
                )

            # Cross-check against the actual compositor output -- region
            # alone does not prove the compositor painted it (see this
            # suite's own docstring / lessons-live-verification.md).
            assert install_button in app.screen._compositor.visible_widgets, (
                "install button's region looked on-screen, but the "
                "compositor never actually painted it -- it is scrolled "
                "out of view"
            )
            strips = app.screen._compositor.render_strips()
            rendered_text = "\n".join(
                "".join(segment.text for segment in strip) for strip in strips
            )
            assert "Use model from disk" in rendered_text, (
                "external-directory button's label never reached the rendered frame"
            )
            assert "Review and install" in rendered_text, (
                "install button's label never reached the rendered frame"
            )

            # Mounted production flow: the real affordance must push the real
            # directory picker, not a test-only stand-in.
            disk_button.press()
            await _wait_until(pilot, lambda: isinstance(app.screen, SelectDirectory))
            assert isinstance(app.screen, SelectDirectory)


@pytest.mark.asyncio
async def test_external_cancel_is_keyboard_reachable_and_in_bounds_at_80_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(80, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            _select_radio(app.screen, "#setup-track-full")
            _press(app.screen, "#wizard-next")
            await pilot.pause(0.2)
            container = app.screen.query_one(SetupWizardContainer)
            while container.steps[container.current_step].config.id != "speech":
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)

            step = container.steps[container.current_step]
            assert isinstance(step, SpeechSetupStep)
            worker = MagicMock(is_finished=False)
            step._external_selection_worker = worker
            step._external_selection_token = (1, id(step))
            step._external_busy = True
            step._external_status = "Verifying model files…"
            step.refresh(recompose=True)
            await _wait_until(
                pilot,
                lambda: len(app.screen.query("#setup-speech-cancel-external")) == 1,
            )
            cancel = app.screen.query_one("#setup-speech-cancel-external", Button)
            await _wait_until(pilot, lambda: cancel.region.height > 0)
            assert cancel.region.right <= 80 and cancel.region.bottom <= 40
            assert cancel in app.screen._compositor.visible_widgets

            cancel.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert step._external_busy is False
            assert "prior source is unchanged" in step._external_status.lower()
            worker.cancel.assert_called_once_with()
            disk = step.query_one("#setup-speech-use-from-disk", Button)
            await _wait_until(pilot, lambda: app.focused is disk)

            await pilot.press("enter")
            await _wait_until(pilot, lambda: isinstance(app.screen, SelectDirectory))


@pytest.mark.parametrize("attempt", ["back", "finish-later"])
@pytest.mark.asyncio
async def test_external_commit_fences_back_and_finish_later_until_handoff_settles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attempt: str,
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            _select_radio(app.screen, "#setup-track-full")
            _press(app.screen, "#wizard-next")
            await pilot.pause(0.2)
            container = app.screen.query_one(SetupWizardContainer)
            while container.steps[container.current_step].config.id != "speech":
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)

            step = container.steps[container.current_step]
            assert isinstance(step, SpeechSetupStep)
            prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
            source_commit = SimpleNamespace(
                section_values={"transcription": {"parakeet_external_sources": {}}}
            )
            source_service = MagicMock()
            source_service.prepare_config_commit.return_value = source_commit
            owners = {"setup-speech-live-handoff"}
            source_service.release_scope.side_effect = owners.discard
            source_service.accept_committed.side_effect = lambda commit: None
            app._parakeet_source_service = source_service
            step._pending_external_selection = prepared
            token = (1, id(step))
            step._external_selection_token = token
            step._external_scope_ids[token] = "setup-speech-live-handoff"
            write_started = asyncio.Event()
            allow_write = asyncio.Event()

            async def delayed_commit(values, *, after_write=None):
                write_started.set()
                await allow_write.wait()
                if after_write is not None:
                    after_write()
                return True

            container.commit_config = delayed_commit
            _press(app.screen, "#wizard-next")
            await asyncio.wait_for(write_started.wait(), timeout=2)
            if attempt == "back":
                _press(app.screen, "#wizard-back")
            else:
                _press(app.screen, "#wizard-cancel")
            await pilot.pause()
            stayed_on_speech = (
                type(app.screen).__name__ == "FirstRunSetupWizard"
                and container.steps[container.current_step].config.id == "speech"
            )
            nav_fenced = (
                all(
                    app.screen.query_one(selector, Button).disabled
                    for selector in ("#wizard-back", "#wizard-cancel")
                )
                if type(app.screen).__name__ == "FirstRunSetupWizard"
                else False
            )
            owner_retained = owners == {"setup-speech-live-handoff"}
            allow_write.set()
            await _wait_until(
                pilot,
                lambda: source_service.accept_committed.call_count == 1,
            )

            assert stayed_on_speech
            assert nav_fenced
            assert owner_retained
            assert owners == set()


# ---------------------------------------------------------------------------
# 5c. TASK-1495/1496 at small terminals. Investigating this fix surfaced a
#     SEPARATE, pre-existing constraint (confirmed present before this fix
#     too, by stashing it and re-measuring): at 80x24 the wizard's own
#     fixed-height chrome -- the title (3 rows), WizardProgress (~8 rows),
#     and the navigation bar (5 rows), none of it touched by this fix, all
#     shared with the Chatbook wizards via BaseWizard.py/its DEFAULT_CSS and
#     _wizards.tcss's ".wizard-title"/".wizard-progress"/".wizard-navigation"
#     classes -- leaves ".wizard-steps-container" only ~3 rows, less than
#     ".wizard-step"'s own shared "padding: 2" alone consumes. Every step's
#     content box therefore measures ZERO height at exactly 80x24,
#     independent of which step is showing and independent of this fix
#     (filed as TASK-1509 -- compressing that shared chrome needs its own
#     design decision and cannot be done by scoping CSS to the setup
#     wizard's own classes alone). The first test below pins what THIS fix
#     must not regress at that floor: the nav bar and keyboard focus stay
#     stable. The second test proves the actual scroll-into-view mechanism
#     TASK-1496 adds, at a size just past that chrome floor (100x30) where
#     Provider's own content genuinely has a non-zero viewport to overflow.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigation_and_focus_stay_stable_at_80x24(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            _press(app.screen, "#wizard-next")  # Welcome -> Provider
            await pilot.pause(0.3)
            container = app.screen.query_one(SetupWizardContainer)
            assert container.steps[container.current_step].config.id == STEP_PROVIDER

            for widget_id in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                button = app.screen.query_one(widget_id, Button)
                assert button.visible, f"{widget_id} is not visible at 80x24"
                region = button.region
                assert region.width > 0 and region.height > 0
                assert region.right <= 80 and region.bottom <= 24

            # Focusing a widget deep inside the step (whose own content box
            # measures zero height at this exact size -- see this section's
            # docstring) must not crash the wizard or break the focus chain,
            # even though there is currently no room to actually show it.
            key_input = app.screen.query_one("#setup-provider-key-input", Input)
            key_input.focus()
            await pilot.pause(0.2)
            assert container.is_running, "focusing an off-screen widget crashed the wizard"
            assert app.focused is key_input


@pytest.mark.asyncio
async def test_focus_scrolls_offscreen_widget_into_view_when_step_overflows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-1496 AC #1: "focusing any wizard widget scrolls it into view."

    100x30 keeps the wizard's fixed chrome overhead (see this section's
    docstring) from swallowing the ENTIRE step viewport the way it does at
    80x24, while staying small enough that Provider's own content (even
    capped per TASK-1495) still overflows the step's ~5-row box -- exactly
    the condition this fix's ".setup-step { overflow-y: auto }" targets.
    Textual's own Screen.set_focus (invoked by Widget.focus(), the default
    for both a real Tab press and this test's explicit call) already
    scrolls a newly-focused widget into view once some ancestor is
    genuinely scrollable; before this fix ".setup-step" was not one, so
    nothing could.
    """
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            _select_radio(app.screen, "#setup-track-full")
            await pilot.pause(0.1)
            _press(app.screen, "#wizard-next")  # Welcome -> Provider, track=full
            await pilot.pause(0.3)
            container = app.screen.query_one(SetupWizardContainer)
            assert container.steps[container.current_step].config.id == STEP_PROVIDER

            key_input = app.screen.query_one("#setup-provider-key-input", Input)
            region_before = key_input.region
            fits_before = (
                region_before.y >= 0
                and region_before.bottom <= 30
                and region_before.right <= 100
            )
            assert not fits_before, (
                "test assumption broken: key Input already fits at 100x30 "
                f"without any scroll ({region_before}) -- this test needs "
                "genuine overflow to prove the scroll-into-view fix"
            )

            key_input.focus()
            await pilot.pause(0.3)

            region_after = key_input.region
            assert region_after.width > 0 and region_after.height > 0
            assert region_after.y >= 0 and region_after.bottom <= 30, (
                f"key Input still clipped after focusing it: {region_after}"
            )
            assert region_after.right <= 100
            assert key_input in app.screen._compositor.visible_widgets, (
                "focusing the key Input did not actually scroll it into the "
                "compositor's visible set"
            )

            for widget_id in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                button = app.screen.query_one(widget_id, Button)
                region = button.region
                assert region.width > 0 and region.height > 0
                assert region.right <= 100 and region.bottom <= 30


# ---------------------------------------------------------------------------
# 6. Back/Next mashing across provider -> model must not crash or
#    double-advance (carried over from an earlier ledger item).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_back_next_mashing_across_provider_model_does_not_double_advance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)

            # Quick track is pre-selected on Welcome; advance to Provider,
            # then to Model, each with a real settle so the starting point
            # for the mash is deterministic.
            _press(app.screen, "#wizard-next")  # welcome -> provider
            await pilot.pause(0.2)
            assert container.steps[container.current_step].config.id == STEP_PROVIDER
            _press(app.screen, "#wizard-next")  # provider -> model
            await pilot.pause(0.2)
            assert container.steps[container.current_step].config.id == STEP_MODEL

            # Rapid, unsettled Back/Next mashing at the provider<->model
            # boundary -- no pilot.pause() between presses in the burst, so
            # a Next worker can still be in flight when Back fires.
            for _ in range(8):
                _press(app.screen, "#wizard-back")
                await pilot.pause(0)
                _press(app.screen, "#wizard-next")
                await pilot.pause(0)

            # Let everything drain, then the app must still be alive and on
            # a real, valid step -- not crashed, not stuck. (Not
            # `workers.wait_for_complete()`: that waits for EVERY worker,
            # including Provider/Model's real discovery workers, which can
            # block far longer than this settle needs.)
            await pilot.pause(0.5)
            assert container.is_running
            current = container.steps[container.current_step]
            current_id = current.config.id if current.config else None
            assert current_id in container.active_ids, (
                f"landed on {current_id!r}, outside the active quick-track "
                f"subset {container.active_ids!r} -- a mash-induced derail"
            )

            # Finish the walk from wherever the mash left it; the quick
            # track must complete exactly once through provider/model/
            # summary, with no repeats and no skipped/extra steps -- proof
            # the mash did not double-advance or corrupt navigation.
            seen_step_ids: list[str] = []
            for _ in range(10):
                step = container.steps[container.current_step]
                step_id = step.config.id if step.config else None
                if step_id != (seen_step_ids[-1] if seen_step_ids else None):
                    seen_step_ids.append(step_id)
                if step_id == STEP_SUMMARY:
                    break
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)
            else:
                raise AssertionError("mashing left the wizard unable to complete")

            _press(app.screen, "#setup-exit-chat")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # Dismiss pops back to Home first; the exit_route is applied via
            # a separately-queued NavigateToScreen message, so wait for the
            # final tab rather than racing the first screen-stack pop.
            await _wait_until(pilot, lambda: app.current_tab == TAB_CHAT)
            assert app.current_tab == TAB_CHAT


# ---------------------------------------------------------------------------
# 7. ctrl+n / ctrl+b keyboard shortcuts must not crash (final-review finding 1).
#
# BaseWizard.BINDINGS (never modified -- see that class's own docstring)
# maps ctrl+n/ctrl+b to action_next()/action_back(), which call
# self.handle_next()/self.handle_back() with NO arguments. SetupWizardContainer
# overrides handle_next(self, event)/handle_back(self, event) to require a
# Button.Pressed event (so they can call event.prevent_default() -- see
# those methods' own docstrings), so before this fix pressing ctrl+n or
# ctrl+b on a mounted wizard raised a TypeError out of the binding's action
# dispatch. The fix overrides action_next()/action_back() in
# SetupWizardContainer to route through the same event-free
# advance_programmatically() / _previous_active_index() path the mouse
# handlers use.
#
# Textual's key-binding resolution (Screen._binding_chain) only walks the
# ancestors of the currently FOCUSED widget; when nothing is focused it
# falls back to just Screen + App bindings and never reaches a plain
# Container like SetupWizardContainer at all. Welcome's RadioSet keeps a
# button focused from mount (needed for the very first ctrl+n/ctrl+b to
# reach this container's BINDINGS at all), but Provider's own RadioSet has
# no default-pressed button, so nothing is auto-focused once the wizard
# lands there. This is a pre-existing focus-management gap orthogonal to
# this crash fix (and would require touching BaseWizard.py -- out of scope,
# never to be modified -- or every step's on_show()), so this test focuses
# a real field after each step transition, exactly as Tab or a click would,
# rather than asserting on implicit default focus this wizard does not
# provide.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_n_ctrl_b_do_not_crash_and_move_one_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from textual.widgets import Input

    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)
            assert container.steps[container.current_step].config.id == STEP_WELCOME

            # ctrl+b on the very first step (Welcome's RadioSet is focused
            # from mount) must not crash and must not move anywhere -- this
            # is the exact scenario that raised TypeError before the fix.
            await pilot.press("ctrl+b")
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+b crashed the wizard"
            assert container.steps[container.current_step].config.id == STEP_WELCOME

            # Quick setup is the default pre-selected RadioButton on Welcome
            # (untouched here) -- ctrl+n must apply that choice exactly like
            # clicking Next does (advance_programmatically() -> _advance()
            # -> select_track(step.chosen_track())), not just move a step.
            await pilot.press("ctrl+n")  # welcome -> provider
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+n crashed the wizard"
            assert container.track == TRACK_QUICK
            assert container.steps[container.current_step].config.id == STEP_PROVIDER

            app.screen.query_one("#setup-provider-key-input", Input).focus()
            await pilot.pause(0.1)

            await pilot.press("ctrl+n")  # provider -> model
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+n crashed the wizard"
            assert container.steps[container.current_step].config.id == STEP_MODEL

            app.screen.query_one("#setup-model-custom", Input).focus()
            await pilot.pause(0.1)

            # ctrl+b must move exactly one active step back, not crash and
            # not flat-decrement past the active-id subset.
            await pilot.press("ctrl+b")  # model -> provider
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+b crashed the wizard"
            assert container.steps[container.current_step].config.id == STEP_PROVIDER


def test_setup_wizard_constructs_before_base_init_sets_app_instance():
    """(task-2040) Step constructors read ``wizard.app_instance`` at
    ``__init__`` time (SpeechSetupStep pulls ``app_config`` through it),
    but the container built its steps BEFORE the base ``__init__`` that
    assigns ``app_instance`` -- so every fresh-profile first boot crashed
    with ``AttributeError`` inside the wizard and the whole app died."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        SetupWizardContainer,
    )

    app_instance = SimpleNamespace(app_config={})
    wizard = SetupWizardContainer(app_instance)
    assert wizard.app_instance is app_instance
