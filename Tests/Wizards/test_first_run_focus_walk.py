"""TASK-34100.1 AC#4: keyboard focus survives the whole setup walk.

Review finding cross-cutting-17 (2026-10-02): the wizard kept losing
keystrokes. Focus jumped after async work, a modal opened without focus, and
Ctrl+N/Ctrl+B went dead once focus was None. These walks drive Quick and
Full setup with keys only (arrows, Enter, Tab, typing, Ctrl+N, Ctrl+B,
Esc). After every step change, every async completion and every modal
dismissal they check one invariant: something visible and attached has
focus. Only network and disk are faked.

The two walks are standing guards, not RED proof: the headless walk lost no
focus on the base code either (the step-change focus fix predates this
task; removing it turns both red). The RED proof for AC#4 is
``test_a_refused_next_gives_the_keyboard_back_to_next``, which fails on the
pre-fix code (review round 1). Voice "Test and Hear" moving focus to the
Service radio is review finding voice-speech-05, owned by the Voice group,
so these walks check only that focus is alive there.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import RadioButton

from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Wizards.first_run_provider_step import ProviderChoiceList
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
)

pytestmark = pytest.mark.bootstrap_profile

_BUNDLE = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
)


class _Host(App):
    """The app stylesheet loaded, so ``.hidden`` really hides, as in production."""

    CSS_PATH = str(_BUNDLE)

    def __init__(self, wizard: FirstRunSetupWizard) -> None:
        super().__init__()
        self._wizard = wizard

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard)

    @on(STTSSettingsSaveEvent)
    def _answer_voice_save(self, event: STTSSettingsSaveEvent) -> None:
        """Stand in for the app's TTS save, which Voice's Next waits on."""
        event.reply_to.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=event.request_id,
                persisted=True,
                provider_statuses={"openai": "applied"},
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
                defaults_activated=True if event.commit_defaults_after_handoff else None,
                defaults_activation_status=(
                    "committed" if event.commit_defaults_after_handoff else None
                ),
            )
        )


def _discovery(*, failed: bool) -> ModelDiscoveryResult:
    if failed:
        return ModelDiscoveryResult(
            provider="custom",
            provider_list_key="custom",
            endpoint_fingerprint="safe-fingerprint",
            status="error",
            error=ModelDiscoveryError(
                kind="missing_credentials",
                message="401 unauthorized",
                recovery_hint="fix the key",
            ),
        )
    return ModelDiscoveryResult(
        provider="custom",
        provider_list_key="custom",
        endpoint_fingerprint="https://walk.example.test/v1",
        status="success",
        models=(
            DiscoveredModel(
                provider="custom",
                provider_list_key="custom",
                model_id="walk-model",
                display_name="walk-model",
                source="runtime_discovered",
                endpoint_fingerprint="https://walk.example.test/v1",
                discovered_at="2026-10-03T00:00:00Z",
            ),
        ),
    )


def _wizard(monkeypatch, *, failed_discovery: bool) -> FirstRunSetupWizard:
    async def no_local_servers(*_args, **_kwargs):
        return ()

    monkeypatch.setattr(
        "tldw_chatbook.Chat.local_server_discovery.discover_local_servers",
        no_local_servers,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        MagicMock(return_value=ConfigMutationResult(True, True, None)),
    )

    async def probe(*_args, **_kwargs):
        from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
            SettingsEndpointProbeOutcome,
        )

        await asyncio.sleep(0.2)
        if failed_discovery:
            return SettingsEndpointProbeOutcome(
                state="unreachable", summary="401 unauthorized", category="unauthorized"
            )
        return SettingsEndpointProbeOutcome(
            state="reachable", summary="reachable (1 models)", model_ids=("walk-model",)
        )

    monkeypatch.setattr(
        "tldw_chatbook.UI.Wizards.first_run_provider_step."
        "_probe_first_run_provider_connection",
        probe,
    )

    async def voice_sample(*_args, **_kwargs):
        from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice

        await asyncio.sleep(0.2)
        return voice.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Wizards.first_run_voice_step_state.run_voice_sample",
        voice_sample,
    )
    app_instance = MagicMock()
    app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://walk.example.test/v1"}}
    }
    app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(return_value=_discovery(failed=failed_discovery))
    )
    return FirstRunSetupWizard(app_instance)


def _assert_focus_alive(app: App, where: str) -> None:
    focused = app.focused
    assert focused is not None, f"keyboard focus lost {where}"
    assert focused.is_attached, f"focus on a removed widget {where}"
    hidden = [
        node
        for node in focused.ancestors_with_self
        if not getattr(node, "display", True)
    ]
    assert not hidden, f"focus inside a hidden widget {where}: {hidden[0]!r}"
    assert focused.screen is app.screen, f"focus off the active screen {where}"


class _Walk:
    def __init__(self, app: App, pilot, wizard: FirstRunSetupWizard) -> None:
        self.app = app
        self.pilot = pilot
        self.wizard = wizard
        self.container = wizard.query_one(SetupWizardContainer)
        container = self.container

        async def commit_config(
            settings, *, delete_keys=None, after_write=None, provider_setup_mutation=None
        ):
            # The suite's disk seam: mirror the write, touch no file.
            del after_write, provider_setup_mutation
            container._mirror_into_app_config(settings, delete_keys)
            return True

        container.commit_config = commit_config

    @property
    def step_id(self) -> str:
        step = self.container.steps[self.container.current_step]
        return step.config.id if step.config else ""

    async def settle(self, seconds: float = 0.1) -> None:
        for _ in range(100):
            await self.pilot.pause(0.05)
            if not self.container._advancing and not any(
                worker.is_running
                and worker.group in {"setup-wizard-advance", "setup-model-load"}
                for worker in self.app.workers
            ):
                break
        await self.pilot.pause(seconds)

    async def until(self, predicate, what: str) -> None:
        for _ in range(100):
            if predicate():
                return
            await self.pilot.pause(0.05)
        raise AssertionError(f"timed out waiting for {what}")

    async def next(self, expected: str) -> None:
        await self.pilot.press("ctrl+n")
        await self.settle()
        assert self.step_id == expected, f"Next landed on {self.step_id}"
        _assert_focus_alive(self.app, f"after Next to {expected}")

    async def back(self, expected: str) -> None:
        await self.pilot.press("ctrl+b")
        await self.settle()
        assert self.step_id == expected, f"Back landed on {self.step_id}"
        _assert_focus_alive(self.app, f"after Back to {expected}")

    async def pick_custom_provider(self) -> None:
        """Arrow down the provider list to Custom, then Enter to select it."""
        choice = self.wizard.query_one(ProviderChoiceList)
        assert self.app.focused is choice, "Provider must open on its list"
        for _ in range(120):
            highlighted = choice.highlighted
            option = (
                choice.get_option_at_index(highlighted)
                if highlighted is not None
                else None
            )
            if getattr(option, "provider_key", None) == "custom":
                break
            await self.pilot.press("down")
        await self.pilot.press("enter")
        provider = self.container.steps[self.container.current_step]
        await self.until(
            lambda: provider._selected_discovery_state in {"complete", "failed"},
            "provider discovery",
        )
        await self.pilot.pause(0.1)
        _assert_focus_alive(self.app, "after provider discovery completed")

    async def type_key_and_probe(self) -> None:
        """Tab to the key field, type a key, Enter runs the connection check.

        A typed key also puts the Protect step on the track.
        """
        provider = self.container.steps[self.container.current_step]
        # An endpoint provider keeps its key under "Authentication
        # (optional)": Tab to that title, Enter opens it, Tab into the field.
        for _ in range(30):
            if type(self.app.focused).__name__ == "CollapsibleTitle":
                break
            await self.pilot.press("tab")
        await self.pilot.press("enter")
        await self.pilot.pause(0.1)
        _assert_focus_alive(self.app, "after opening the key section")
        await self.tab_to("setup-provider-api-key")
        await self.pilot.press(*"walk-secret-key")
        await self.pilot.press("enter")
        await self.until(
            lambda: provider._last_tested_provider_identity is not None,
            "the connection check",
        )
        await self.until(
            lambda: provider._selected_discovery_state in {"complete", "failed"},
            "rediscovery for the typed key",
        )
        await self.pilot.pause(0.1)
        _assert_focus_alive(self.app, "after the connection check completed")
        # Not just alive: still on the field whose Enter started the check.
        assert getattr(self.app.focused, "id", None) == "setup-provider-api-key"

    async def wait_model_list(self, *, failed: bool) -> None:
        model = self.container.steps[self.container.current_step]
        selector = (
            "#setup-model-connection-failed" if failed else "#setup-model-choice RadioButton"
        )

        def rendered() -> bool:
            if failed:
                return bool(model.query(selector))
            return any(
                getattr(button, "_model_id", "") == "walk-model"
                for button in model.query(selector)
            )

        await self.until(rendered, "the model list")
        await self.pilot.pause(0.1)
        _assert_focus_alive(self.app, "after model discovery completed")

    async def tab_to(self, widget_id: str) -> None:
        for _ in range(30):
            if getattr(self.app.focused, "id", None) == widget_id:
                return
            await self.pilot.press("tab")
        raise AssertionError(f"Tab never reached #{widget_id}")

    async def dismiss_modal_with_escape(self, modal_type: str) -> None:
        await self.until(
            lambda: type(self.app.screen).__name__ == modal_type, modal_type
        )
        await self.pilot.pause(0.1)
        _assert_focus_alive(self.app, f"when {modal_type} opened")
        # The finish-later dialog swallows an Escape within its 0.5 s grace.
        await self.pilot.pause(0.6)
        await self.pilot.press("escape")
        await self.until(lambda: self.app.screen is self.wizard, "the dismissal")
        await self.pilot.pause(0.1)
        _assert_focus_alive(self.app, f"after {modal_type} was dismissed")


@pytest.mark.asyncio
async def test_quick_track_keyboard_walk_never_loses_focus(monkeypatch):
    wizard = _wizard(monkeypatch, failed_discovery=False)
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        walk = _Walk(app, pilot, wizard)
        await walk.settle(0.3)
        _assert_focus_alive(app, "on Welcome")

        # Welcome: Quick is preselected; Enter on the radio group continues.
        await pilot.press("enter")
        await walk.settle()
        assert walk.step_id == "provider"
        _assert_focus_alive(app, "after Next to provider")
        await walk.pick_custom_provider()
        await walk.type_key_and_probe()

        # The exit dialog opens and closes without stranding the keyboard.
        await pilot.press("escape")
        await walk.dismiss_modal_with_escape("_SettlingGuardedConfirmationDialog")

        await walk.next("model")
        await walk.wait_model_list(failed=False)
        await pilot.press("down")  # selection follows the highlight
        await walk.next("voice")

        # Voice: "Test and Hear" from the keyboard; the sample is async work.
        # TASK-34100.8: the step starts on "No voice for now"; the arrow key
        # picks PocketTTS (selection follows the highlight), which shows it.
        voice = walk.container.steps[walk.container.current_step]
        await walk.tab_to("setup-voice-preset")
        await pilot.press("right")
        await walk.until(lambda: voice._preset == "pocket_tts", "PocketTTS picked")
        await walk.tab_to("setup-voice-test")
        await pilot.press("enter")
        await walk.until(
            lambda: voice._test_in_progress_generation is None
            and voice._verified_draft is not None,
            "the voice sample",
        )
        await pilot.pause(0.1)
        _assert_focus_alive(app, "after the voice sample completed")
        # voice-speech-05: focus came back to the button, not the top.
        assert app.focused is voice.query_one("#setup-voice-test")

        await walk.next("protect-keys")

        # Protect: open the password dialog from the keyboard, then cancel.
        await walk.tab_to("setup-protect-set-password")
        await pilot.press("enter")
        await walk.dismiss_modal_with_escape("PasswordDialog")

        await walk.next("summary")
        await walk.until(
            lambda: not any(
                worker.is_running and worker.group == "setup-summary-load"
                for worker in app.workers
            ),
            "the summary read-back",
        )
        await pilot.pause(0.1)
        _assert_focus_alive(app, "after the summary read-back completed")

        await walk.back("protect-keys")
        await walk.back("voice")
        await walk.back("model")
        await walk.wait_model_list(failed=False)
        await walk.back("provider")


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["commit", "checkpoint"])
@pytest.mark.parametrize("press", ["enter", "click"])
async def test_a_refused_next_gives_the_keyboard_back_to_next(
    monkeypatch, refusal: str, press: str
):
    """A Next that stays on its step leaves focus on Next, so Retry works.

    TASK-34100.1 review: the advance fence disables Next, and Textual blurs a
    focused widget the moment it is disabled. A Next that moved on re-anchored
    focus in ``show_step``, but one the step refused (``commit`` said no, or
    the checkpoint failed) left focus at None: the copy said "Retry with
    Next" while Enter, Ctrl+N and Ctrl+B all did nothing until Tab.
    """
    wizard = _wizard(monkeypatch, failed_discovery=False)
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        walk = _Walk(app, pilot, wizard)
        await walk.settle(0.3)
        container = walk.container
        refusing = [True]
        if refusal == "commit":
            welcome = container.steps[container.current_step]
            real_commit = welcome.commit

            async def commit():
                await asyncio.sleep(0.05)
                return (False, "Not yet.") if refusing[0] else await real_commit()

            welcome.commit = commit
        else:
            real_checkpoint = container.persist_setup_checkpoint

            async def checkpoint(step_id):
                await asyncio.sleep(0.05)
                return False if refusing[0] else await real_checkpoint(step_id)

            container.persist_setup_checkpoint = checkpoint

        if press == "enter":
            await walk.tab_to("wizard-next")
            await pilot.press("enter")
        else:
            await pilot.click("#wizard-next")
        await walk.settle()
        assert walk.step_id == "welcome"
        _assert_focus_alive(app, f"after a Next refused by its {refusal}")
        assert getattr(app.focused, "id", None) == "wizard-next"

        # "Retry with Next": the key that started it retries it.
        refusing[0] = False
        await pilot.press("ctrl+n" if press == "click" else "enter")
        await walk.settle()
        assert walk.step_id == "provider"
        _assert_focus_alive(app, "after the retried Next")


@pytest.mark.asyncio
async def test_full_track_keyboard_walk_never_loses_focus(monkeypatch):
    wizard = _wizard(monkeypatch, failed_discovery=True)
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        walk = _Walk(app, pilot, wizard)
        await walk.settle(0.3)
        _assert_focus_alive(app, "on Welcome")

        await pilot.press("down")  # Full setup
        assert wizard.query_one("#setup-track-full", RadioButton).value
        await walk.next("provider")
        await walk.pick_custom_provider()
        await walk.type_key_and_probe()

        await walk.next("model")
        await walk.wait_model_list(failed=True)

        # A failed check gates Next behind "Continue anyway?": cancel it...
        await pilot.press("ctrl+n")
        await walk.dismiss_modal_with_escape("_SettlingGuardedConfirmationDialog")
        assert walk.step_id == "model"

        # ...then enter a model by hand and continue anyway, keys only.
        await walk.tab_to("setup-model-custom")
        await pilot.press(*"manual-model")
        _assert_focus_alive(app, "after typing a model id")
        await pilot.press("ctrl+n")
        await walk.until(
            lambda: type(app.screen).__name__ == "_SettlingGuardedConfirmationDialog",
            "the continue-anyway dialog",
        )
        await pilot.pause(0.1)
        _assert_focus_alive(app, "when continue-anyway opened")
        await walk.tab_to("confirm-button")
        await pilot.press("enter")
        await walk.settle()
        assert walk.step_id == "voice"
        _assert_focus_alive(app, "after continuing past the failed check")

        await walk.next("rag")
        await walk.next("speech")
        await walk.until(
            lambda: not any(
                worker.is_running and worker.group == "setup-speech-load"
                for worker in app.workers
            ),
            "the speech model check",
        )
        await pilot.pause(0.2)
        _assert_focus_alive(app, "after the speech model check completed")
        await walk.next("tools")
        await walk.next("notes")
        await walk.next("appearance")
        await walk.next("protect-keys")
        await walk.next("summary")
        await walk.until(
            lambda: not any(
                worker.is_running and worker.group == "setup-summary-load"
                for worker in app.workers
            ),
            "the summary read-back",
        )
        await pilot.pause(0.1)
        _assert_focus_alive(app, "after the summary read-back completed")

        await walk.back("protect-keys")
        await walk.back("appearance")
        await walk.back("notes")
        await walk.back("tools")
        await walk.back("speech")
