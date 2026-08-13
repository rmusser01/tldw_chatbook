"""Pilot tests for the first-run setup wizard skeleton."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
    Switch,
)

from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.UI.Wizards.BaseWizard import (
    WizardNavigation,
    WizardProgress,
    WizardStepConfig,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    SETUP_DRAFT_VERSION,
    STEP_APPEARANCE,
    STEP_MODEL,
    STEP_NOTES,
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SPEECH,
    STEP_SUMMARY,
    STEP_TOOLS,
    STEP_VOICE,
    STEP_WELCOME,
    TRACK_FULL,
    TRACK_QUICK,
    FirstRunModelDiscoveryKey,
    SetupDraft,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    CLOUD_PROBE_TIMEOUT_SECONDS,
    AppearanceStep,
    FirstRunSetupWizard,
    ModelStep,
    NotesSyncStep,
    ProtectKeysStep,
    ProviderChoiceOption,
    ProviderStep,
    RagStep,
    SetupStep,
    SetupStepFailure,
    SetupWizardContainer,
    SummaryStep,
    ToolsStep,
    VoiceSetupStep,
    _probe_first_run_provider_connection,
    _provider_group_option_id,
    _provider_options,
)


class _HostApp(App):
    def __init__(self, wizard: FirstRunSetupWizard):
        super().__init__()
        self._wizard = wizard
        self.wizard_result = "UNSET"
        self.wizard_results = []

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard, self._capture)

    def _capture(self, result) -> None:
        self.wizard_result = result
        self.wizard_results.append(result)

    @on(STTSSettingsSaveEvent)
    def _complete_voice_settings_save(self, event: STTSSettingsSaveEvent) -> None:
        assert event.request_id is not None
        event.reply_to.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=event.request_id,
                persisted=True,
                provider_statuses={"openai": "applied"},
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
                defaults_activated=(
                    True if event.commit_defaults_after_handoff else None
                ),
                defaults_activation_status=(
                    "committed" if event.commit_defaults_after_handoff else None
                ),
            )
        )


class _StyledHostApp(_HostApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
    )


def _make_wizard(**kwargs) -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = {}
    wizard = FirstRunSetupWizard(app_instance, **kwargs)
    return wizard


def _raising_compose_step(self):
    """Generator-shaped compose helper that fails before yielding widgets."""
    raise RuntimeError("sensitive compose detail")
    yield  # pragma: no cover


@pytest.mark.asyncio
async def test_welcome_track_choice_activates_quick_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        assert STEP_PROVIDER in container.active_ids
        assert STEP_RAG not in container.active_ids
        assert container.active_ids.index(STEP_VOICE) == (
            container.active_ids.index(STEP_MODEL) + 1
        )
        assert container.active_ids[-1] == STEP_SUMMARY


@pytest.mark.asyncio
async def test_voice_step_compact_controls_are_ordered_and_default_is_opt_in():
    from types import SimpleNamespace

    wizard = SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={})
    step = VoiceSetupStep(
        wizard=wizard,
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        ordered_ids = [
            widget.id
            for widget in step.walk_children()
            if widget.id
            in {
                "setup-voice-preset",
                "setup-voice-endpoint",
                "setup-voice-auth",
                "setup-voice-model",
                "setup-voice-voice",
                "setup-voice-sample",
                "setup-voice-test",
                "setup-voice-status",
                "setup-voice-default",
            }
        ]

        assert ordered_ids == [
            "setup-voice-preset",
            "setup-voice-endpoint",
            "setup-voice-auth",
            "setup-voice-model",
            "setup-voice-voice",
            "setup-voice-sample",
            "setup-voice-test",
            "setup-voice-status",
            "setup-voice-default",
        ]
        assert step.query_one("#setup-voice-default", Checkbox).value is False
        assert step.query_one("#setup-voice-test", Button).disabled is False


@pytest.mark.asyncio
async def test_invalid_voice_sample_disables_only_test_and_preserves_configuration():
    from types import SimpleNamespace

    wizard = SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={})
    step = VoiceSetupStep(
        wizard=wizard,
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        endpoint = step.query_one("#setup-voice-endpoint", Input)
        model = step.query_one("#setup-voice-model", Input)
        voice = step.query_one("#setup-voice-voice", Input)
        sample = step.query_one("#setup-voice-sample", Input)
        endpoint.value = "http://127.0.0.1:8765/v1/audio/speech"
        model.value = "pocket-tts"
        voice.value = "alba"
        sample.value = " "
        await pilot.pause()

        assert step.query_one("#setup-voice-test", Button).disabled is True
        assert endpoint.value == "http://127.0.0.1:8765/v1/audio/speech"
        assert model.value == "pocket-tts"
        assert voice.value == "alba"
        assert "0 / 500" in str(
            step.query_one("#setup-voice-sample-count", Static).renderable
        )
        ok, error = await step.commit()
        assert ok is False
        assert "sample" in error.casefold()


@pytest.mark.asyncio
async def test_invalid_voice_speed_returns_inline_validation_instead_of_raising():
    from types import SimpleNamespace

    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-speed", Input).value = "not-a-number"

        ok, error = await step.commit()

        assert ok is False
        assert "speed" in error.casefold()


@pytest.mark.asyncio
async def test_default_choice_does_not_invalidate_verified_sample() -> None:
    from types import SimpleNamespace

    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        verified = step._draft_from_controls()
        step._verified_draft = verified
        step.query_one("#setup-voice-status", Static).update("Verified.")

        step.query_one("#setup-voice-default", Checkbox).value = True
        await pilot.pause()

        assert step._verified_draft == verified
        assert "Verified" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_voice_save_result_uses_submitted_default_choice() -> None:
    from types import SimpleNamespace

    class DelayedSaveHost(_StepHost):
        saved_event: STTSSettingsSaveEvent | None = None

        @on(STTSSettingsSaveEvent)
        def capture_save(self, event: STTSSettingsSaveEvent) -> None:
            self.saved_event = event

    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = DelayedSaveHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-default", Checkbox).value = True
        await pilot.pause()
        commit = __import__("asyncio").create_task(step.commit())
        await pilot.pause()
        assert app.saved_event is not None

        step.query_one("#setup-voice-default", Checkbox).value = False
        step.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.saved_event.request_id,
                persisted=True,
                provider_statuses={"openai": "applied"},
                provider_configuration_revisions={"openai": 1},
                provider_runtime_revisions={"openai": 1},
                defaults_activated=False,
                defaults_activation_status="activation_not_ready",
            )
        )

        ok, error = await commit
        assert ok is False
        assert "default" in error.casefold()


@pytest.mark.asyncio
async def test_voice_save_waits_for_applied_runtime_when_default_is_opted_out() -> None:
    from types import SimpleNamespace

    class DelayedSaveHost(_StepHost):
        saved_event: STTSSettingsSaveEvent | None = None

        @on(STTSSettingsSaveEvent)
        def capture_save(self, event: STTSSettingsSaveEvent) -> None:
            self.saved_event = event

    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = DelayedSaveHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        commit = __import__("asyncio").create_task(step.commit())
        await pilot.pause()
        assert app.saved_event is not None

        step.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=app.saved_event.request_id,
                persisted=True,
                provider_statuses={"openai": "pending"},
                provider_configuration_revisions={"openai": 7},
                staged_provider_ids=frozenset({"openai"}),
            )
        )
        await pilot.pause()
        assert commit.done() is False

        step.receive_stts_settings_runtime_result(
            STTSSettingsSaveResult(
                request_id=app.saved_event.request_id,
                persisted=True,
                provider_statuses={"openai": "applied"},
                provider_configuration_revisions={"openai": 7},
                provider_runtime_revisions={"openai": 41},
            )
        )

        assert await commit == (True, "")


@pytest.mark.asyncio
async def test_voice_resume_restores_all_non_secret_controls():
    resume = SetupDraft(
        version=SETUP_DRAFT_VERSION,
        track=TRACK_QUICK,
        active_step_id=STEP_VOICE,
        values={
            STEP_WELCOME: {"track": TRACK_QUICK},
            STEP_VOICE: {
                "endpoint": "http://127.0.0.1:9876/v1/audio/speech",
                "authentication_mode": "none",
                "model_id": "resume-model",
                "voice_id": "resume-voice",
                "response_format": "wav",
                "speed": 1.25,
                "sample_text": "Resume my voice sample.",
                "use_as_default": True,
            },
        },
    )
    wizard = _make_wizard(resume_draft=resume)
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container = wizard.query_one(SetupWizardContainer)
        step = container.steps[container._step_index_for_id(STEP_VOICE)]
        assert isinstance(step, VoiceSetupStep)
        assert step.query_one("#setup-voice-endpoint", Input).value.endswith(
            ":9876/v1/audio/speech"
        )
        assert step.query_one("#setup-voice-model", Input).value == "resume-model"
        assert step.query_one("#setup-voice-voice", Input).value == "resume-voice"
        assert step.query_one("#setup-voice-speed", Input).value == "1.25"
        assert step.query_one("#setup-voice-default", Checkbox).value is True


@pytest.mark.asyncio
async def test_voice_sample_failure_stays_locally_valid_and_needs_test(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    async def fail_sample(*_args, **_kwargs):
        raise ValueError("server-owned detail")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Wizards.first_run_voice_step_state.run_voice_sample",
        fail_sample,
    )
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-test", Button).press()
        await pilot.pause(0.1)

        status = str(step.query_one("#setup-voice-status", Static).renderable)
        assert "Needs test" in status
        assert "server-owned" not in status
        assert step.query_one("#setup-voice-test", Button).disabled is False
        assert voice_state.validate_voice_setup_draft(
            step._draft_from_controls()
        ).configuration_valid


@pytest.mark.asyncio
async def test_voice_late_sample_success_cannot_verify_changed_endpoint(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_sample(draft, **_kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", delayed_sample)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-test", Button).press()
        await asyncio.wait_for(started.wait(), timeout=1)
        step.query_one("#setup-voice-endpoint", Input).value = (
            "http://127.0.0.1:9999/v1/audio/speech"
        )
        await pilot.pause()
        release.set()
        await pilot.pause(0.1)

        assert "Needs test" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )
        assert step._verified_draft is None
        assert step.query_one("#setup-voice-test", Button).disabled is False


@pytest.mark.parametrize(
    ("control_id", "invalid_value"),
    [
        ("setup-voice-endpoint", "not-an-endpoint"),
        ("setup-voice-model", ""),
        ("setup-voice-voice", ""),
        ("setup-voice-format", "unsupported"),
        ("setup-voice-speed", "not-a-number"),
    ],
)
@pytest.mark.asyncio
async def test_voice_test_enablement_tracks_the_entire_current_configuration(
    control_id: str,
    invalid_value: str,
) -> None:
    from types import SimpleNamespace

    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-voice-test", Button)
        assert button.disabled is False

        step.query_one(f"#{control_id}", Input).value = invalid_value
        await pilot.pause()

        assert button.disabled is True


@pytest.mark.asyncio
async def test_voice_edit_cancels_inflight_sample_and_reenables_valid_test(
    monkeypatch,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def delayed_sample(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(voice_state, "run_voice_sample", delayed_sample)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-voice-test", Button)
        button.press()
        await asyncio.wait_for(started.wait(), timeout=1)
        assert button.disabled is True

        step.query_one("#setup-voice-model", Input).value = "edited-pocket-model"
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        await pilot.pause()

        assert button.disabled is False
        assert "Needs test" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )


@pytest.mark.parametrize("change", ["authentication", "preset"])
@pytest.mark.asyncio
async def test_voice_auth_and_preset_changes_cancel_inflight_sample(
    monkeypatch,
    change: str,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def delayed_sample(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(voice_state, "run_voice_sample", delayed_sample)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-voice-test", Button)
        button.press()
        await asyncio.wait_for(started.wait(), timeout=1)

        target = (
            "#setup-voice-auth-key"
            if change == "authentication"
            else "#setup-voice-preset-official"
        )
        step.query_one(target, RadioButton).value = True
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        await pilot.pause()

        assert button.disabled is True
        assert "API key required" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_voice_external_worker_cancel_restores_retry_state(monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    started = asyncio.Event()

    async def delayed_sample(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(voice_state, "run_voice_sample", delayed_sample)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-voice-test", Button)
        button.press()
        await asyncio.wait_for(started.wait(), timeout=1)

        app.workers.cancel_group(step, "setup-voice-sample")
        await pilot.pause(0.1)

        assert button.disabled is False
        assert "Needs test" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )


@pytest.mark.parametrize("lifecycle_method", ["on_hide", "on_unmount"])
@pytest.mark.asyncio
async def test_voice_lifecycle_cancels_sample_and_restores_retry_state(
    monkeypatch,
    lifecycle_method: str,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def delayed_sample(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(voice_state, "run_voice_sample", delayed_sample)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-voice-test", Button)
        button.press()
        await asyncio.wait_for(started.wait(), timeout=1)

        getattr(step, lifecycle_method)()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        await pilot.pause()

        assert button.disabled is False
        assert "Needs test" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_stale_voice_completion_cannot_overwrite_newer_testing_state(
    monkeypatch,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    releases = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]
    call = 0

    async def delayed_sample(*_args, **_kwargs):
        nonlocal call
        index = call
        call += 1
        started[index].set()
        try:
            await releases[index].wait()
        except asyncio.CancelledError:
            await releases[index].wait()
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", delayed_sample)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-voice-test", Button)
        button.press()
        await asyncio.wait_for(started[0].wait(), timeout=1)

        step.query_one("#setup-voice-model", Input).value = "new-model"
        await pilot.pause()
        button.press()
        await asyncio.wait_for(started[1].wait(), timeout=1)
        releases[0].set()
        await pilot.pause(0.1)

        assert button.disabled is True
        assert str(step.query_one("#setup-voice-status", Static).renderable) == (
            "Testing voice…"
        )

        releases[1].set()
        await pilot.pause(0.1)
        assert button.disabled is False
        assert "Verified" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_reselecting_current_voice_preset_preserves_user_edits() -> None:
    from types import SimpleNamespace

    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        model = step.query_one("#setup-voice-model", Input)
        model.value = "user-edited-pocket-model"
        await pilot.pause()

        pressed = step.query_one("#setup-voice-preset-pocket", RadioButton)
        step._on_preset(SimpleNamespace(pressed=pressed))
        await pilot.pause()

        assert model.value == "user-edited-pocket-model"


@pytest.mark.asyncio
async def test_official_voice_without_key_is_actionable_and_cannot_test_or_save(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    sample_calls = 0

    async def unexpected_sample(*_args, **_kwargs):
        nonlocal sample_calls
        sample_calls += 1
        raise AssertionError("sample request must remain blocked")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.UI.Wizards.first_run_voice_step_state.run_voice_sample",
        unexpected_sample,
    )
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-preset-official", RadioButton).value = True
        await pilot.pause()

        status = str(step.query_one("#setup-voice-status", Static).renderable)
        assert "API key required" in status
        assert len(status) <= 120
        assert step.query_one("#setup-voice-add-key", Button).display is True
        assert step.query_one("#setup-voice-test", Button).disabled is True

        step.query_one("#setup-voice-test", Button).press()
        await pilot.pause()
        assert sample_calls == 0

        ok, error = await step.commit()
        assert ok is False
        assert "Add an API key in Settings" in error
        assert sample_calls == 0


@pytest.mark.asyncio
async def test_official_voice_refreshes_after_configured_environment_key_added(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.delenv("VOICE_OPENAI_KEY", raising=False)
    app_config = {"api_settings": {"openai": {"api_key_env_var": "VOICE_OPENAI_KEY"}}}
    step = VoiceSetupStep(
        wizard=SimpleNamespace(
            app_instance=MagicMock(app_config=app_config), wizard_data={}
        ),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-preset-official", RadioButton).value = True
        await pilot.pause()
        assert step.query_one("#setup-voice-test", Button).disabled is True

        monkeypatch.setenv("VOICE_OPENAI_KEY", "sk-added-outside-draft")
        step.on_show()
        await pilot.pause()

        assert step.query_one("#setup-voice-test", Button).disabled is False
        assert step.query_one("#setup-voice-add-key", Button).display is False
        assert "Needs test" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )
        assert "sk-added-outside-draft" not in repr(step.get_step_data())


@pytest.mark.parametrize(
    "app_config",
    [
        {"openai_api": {"api_key": "sk-saved"}},
        {"API": {"openai_api_key": "sk-saved"}},
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "API": {"openai_api_key": "sk-saved"}
            }
        },
    ],
)
@pytest.mark.asyncio
async def test_official_voice_recognizes_existing_settings_credential_locations(
    monkeypatch,
    app_config,
) -> None:
    from types import SimpleNamespace

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(
            app_instance=MagicMock(app_config=app_config), wizard_data={}
        ),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-preset-official", RadioButton).value = True
        await pilot.pause()

        assert step.query_one("#setup-voice-test", Button).disabled is False
        assert step.query_one("#setup-voice-add-key", Button).display is False
        assert "sk-saved" not in repr(step.get_step_data())


@pytest.mark.asyncio
async def test_missing_key_action_checkpoints_voice_and_routes_to_tts_settings(
    monkeypatch,
) -> None:
    from unittest.mock import AsyncMock

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = _make_wizard()
    wizard.app_instance.app_config = {"first_run": {"setup_completed": False}}
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert voice_index is not None
        container.show_step(voice_index)
        step = container.steps[voice_index]
        assert isinstance(step, VoiceSetupStep)
        step.query_one("#setup-voice-preset-official", RadioButton).value = True
        await pilot.pause()
        persist = AsyncMock(return_value=True)
        container.persist_current_checkpoint = persist

        step.query_one("#setup-voice-add-key", Button).press()
        await pilot.pause(0.2)

        persist.assert_awaited_once()
        assert container.wizard_data[STEP_VOICE]["endpoint"] == (
            "https://api.openai.com/v1/audio/speech"
        )
        assert "api_key" not in container.wizard_data[STEP_VOICE]
        assert app.wizard_result == {
            "completed": False,
            "exit_route": "settings",
            "exit_context": {"category": "speech-tts"},
        }


@pytest.mark.asyncio
async def test_missing_key_action_without_callback_stays_bounded_and_does_not_crash(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-preset-official", RadioButton).value = True
        await pilot.pause()
        step.query_one("#setup-voice-add-key", Button).press()
        await pilot.pause()

        status = str(step.query_one("#setup-voice-status", Static).renderable)
        assert "Settings" in status
        assert len(status) <= 120


@pytest.mark.asyncio
async def test_voice_playback_failure_cleans_new_file_and_keeps_verification(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    async def sample(*_args, **_kwargs):
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    class FailingPlayer:
        def play(self, _path):
            raise RuntimeError("player detail")

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)
    app.audio_player = FailingPlayer()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-test", Button).press()
        await pilot.pause(0.1)

        status = str(step.query_one("#setup-voice-status", Static).renderable)
        assert step._verified_draft is not None
        assert status == "Verified, playback failed. Retry playback/test."
        assert len(status) <= 80
        assert list(tmp_path.glob("chatbook-voice-sample-*")) == []
        assert step._sample_audio_path is None


@pytest.mark.asyncio
async def test_voice_playback_cancellation_cleans_new_file(
    monkeypatch,
    tmp_path,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state

    playback_started = asyncio.Event()

    async def sample(*_args, **_kwargs):
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    class BlockingPlayer:
        async def play(self, _path):
            playback_started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    step = VoiceSetupStep(
        wizard=SimpleNamespace(app_instance=MagicMock(app_config={}), wizard_data={}),
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )
    app = _StepHost(step)
    app.audio_player = BlockingPlayer()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-voice-test", Button).press()
        await asyncio.wait_for(playback_started.wait(), timeout=1)
        app.workers.cancel_group(step, "setup-voice-sample")
        await pilot.pause(0.1)

        assert list(tmp_path.glob("chatbook-voice-sample-*")) == []
        assert step._sample_audio_path is None
        assert step._verified_draft is not None
        status = str(step.query_one("#setup-voice-status", Static).renderable)
        assert status == "Verified, playback failed. Retry playback/test."


@pytest.mark.asyncio
async def test_show_step_runs_exactly_once_on_wizard_mount(monkeypatch):
    """TASK-2710 regression guard: WizardContainer.on_mount calls
    show_step(0) exactly once, at the end of its own initialization.

    Before TASK-2710, SetupWizardContainer.on_mount called
    super().on_mount() explicitly AND Textual's dispatcher separately
    invoked WizardContainer.on_mount again for the same Mount event (it
    walks the whole MRO -- see WizardContainer.on_mount's docstring), so
    show_step(0) -- and the on_hide/on_show pair on the current step, plus
    the validation timer -- ran twice per wizard mount. Harmless in
    practice (idempotent state), but exactly the fragile pattern this task
    exists to guard against; pin it here so it can't silently come back.
    """
    from tldw_chatbook.UI.Wizards import BaseWizard as base_wizard_module

    calls: list[int] = []
    original_show_step = base_wizard_module.WizardContainer.show_step

    def _counting_show_step(self, step_index):
        calls.append(step_index)
        return original_show_step(self, step_index)

    monkeypatch.setattr(
        base_wizard_module.WizardContainer, "show_step", _counting_show_step
    )

    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)

    assert calls == [0]


@pytest.mark.asyncio
async def test_select_track_rebuilds_progress_in_original_slot():
    """F-C regression (live-verified via tmux screenshot): _rebuild_progress
    replaces the WizardProgress widget wholesale on every track change, but
    ``parent.mount(fresh)`` with no ``before=``/``after=`` appends at the
    container's END -- after WizardNavigation -- so the whole progress bar
    rendered BELOW the Back/Next buttons instead of staying in its original
    slot right after the title."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.2)
        children = list(container.children)
        progress = container.query_one(".wizard-progress", WizardProgress)
        nav = container.query_one(".wizard-navigation", WizardNavigation)
        steps_container = container.query_one(".wizard-steps-container")
        assert children.index(progress) < children.index(steps_container)
        assert children.index(progress) < children.index(nav)


@pytest.mark.asyncio
async def test_welcome_full_track_activates_all_non_conditional_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_FULL)
        assert STEP_RAG in container.active_ids


@pytest.mark.asyncio
async def test_escape_asks_for_confirmation_instead_of_dismissing():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        # The wizard must still be open (confirm dialog on top), not dismissed.
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_escape_pressed_with_no_settle_pause_still_opens_confirmation():
    """TASK-2314: a single Escape pressed the instant the wizard appears --
    with NO settling pause first, unlike every other test in this file --
    must still reach the finish-later confirmation. Live UAT reproduction
    showed this single-press path always worked; this test pins it so a
    future fix for the double-press race (below) cannot regress it by
    swallowing early Escapes wholesale."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            _SettlingGuardedConfirmationDialog,
        )

        assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_rapid_double_escape_during_first_render_reaches_finish_later_flow():
    """TASK-2314 regression, from a live reproduction: the wizard is pushed
    while several heavy steps are still settling (10 composed steps, the
    full provider catalog). A user who presses Escape once, perceives no
    immediate feedback because of that render lag, and reflexively presses
    it again lands the SECOND press on the confirmation dialog's own
    Escape-cancels binding -- which silently snapped the wizard back open
    with no visible sign anything happened (live-confirmed via tmux: two
    Escape presses sent within 5ms of the wizard's first paint left the
    dialog closed and the wizard showing, before this fix). The
    confirmation flow must stay up through a reflexive double-press, not
    revert."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("escape", "escape")
        await pilot.pause()
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            _SettlingGuardedConfirmationDialog,
        )

        assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_escape_still_cancels_the_dialog_once_it_has_actually_settled():
    """The grace window must not make Escape inert forever -- once the
    dialog has genuinely been up longer than the grace period, a second
    Escape (a deliberate one, this time) must still dismiss it back to
    "Keep going", exactly like before this fix."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            _SettlingGuardedConfirmationDialog,
        )

        dialog = app.screen
        assert isinstance(dialog, _SettlingGuardedConfirmationDialog)
        dialog._escape_grace_seconds = 0.0  # simulate the window having elapsed
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        # Dismissed via "Keep going" (False) -- the wizard itself stays open.
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_clicking_keep_going_immediately_is_never_swallowed():
    """The grace window is scoped to the Escape BINDING only -- a deliberate
    mouse click on "Keep going" must stay instant regardless of timing,
    even in the same instant the dialog appeared."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
        await pilot.click("#cancel-button")
        await pilot.pause()
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            _SettlingGuardedConfirmationDialog,
        )

        assert not isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        assert app.wizard_result == "UNSET"


@pytest.mark.asyncio
async def test_next_button_click_drives_quick_track_to_completion():
    """Regression test for a real Textual double-dispatch trap.

    Textual's @on-decorated handlers are collected across the WHOLE MRO
    (textual.message_pump.MessagePump._get_dispatch_methods), so both
    WizardContainer.handle_next (base) and SetupWizardContainer.handle_next
    (override) fire on a single Button.Pressed("#wizard-next"). Without
    event.prevent_default() in the override, the base handler flat-advances
    current_step by one BEFORE the override's own worker runs — silently
    breaking track selection (select_track() on the Welcome step never
    actually applies) and skipping/duplicating steps. This test drives the
    real click path (not container.select_track() directly) so a regression
    of that suppression would fail it.
    """
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        await pilot.click("#setup-track-quick")
        await pilot.pause(0.1)

        seen_step_ids = []
        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            await pilot.click("#wizard-next")
            await pilot.pause(0.2)
            step = container.steps[container.current_step]
            seen_step_ids.append(step.config.id if step.config else None)

        assert app.wizard_result == {"completed": True, "exit_route": None}
        # Exactly the quick-track subset, each step visited once, in order.
        assert seen_step_ids == [
            "provider",
            "model",
            "voice",
            "summary",
            "summary",
        ]
        assert set(container.wizard_data.keys()) == {
            "welcome",
            "provider",
            "model",
            "voice",
            "summary",
        }


@pytest.mark.asyncio
async def test_mounted_provider_and_model_advance_checkpoint_then_commit_atomically():
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {"api_url": "https://mounted.test/v1/chat/completions"}
        }
    }
    wizard.app_instance.llm_provider_catalog_scope_service = None
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        checkpoints = []
        atomic_mutations = []

        async def commit_config(
            settings,
            *,
            delete_keys=None,
            after_write=None,
            provider_setup_mutation=None,
        ):
            if provider_setup_mutation is not None:
                atomic_mutations.append(provider_setup_mutation)
                container._mirror_into_app_config(settings, delete_keys)
            elif "first_run" in settings:
                checkpoints.append(settings["first_run"])
            return True

        container.commit_config = commit_config
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        assert provider_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        provider_step.query_one(
            "#setup-provider-api-key", Input
        ).value = "mounted-private-secret"

        await container._advance()

        assert container.steps[container.current_step].config.id == STEP_MODEL
        assert container.staged_provider_draft is not None
        assert container.provider_setup_committed is False
        assert checkpoints[-1]["active_step_id"] == STEP_PROVIDER
        assert "mounted-private-secret" not in json.dumps(checkpoints[-1])

        model_step = container.steps[container.current_step]
        assert isinstance(model_step, ModelStep)
        model_step.query_one("#setup-model-custom", Input).value = "mounted-model"
        await pilot.pause()
        await container._advance()

        assert len(atomic_mutations) == 1
        assert container.provider_setup_committed is True
        assert container.committed_provider_model == "mounted-model"
        assert container.steps[container.current_step].config.id == STEP_VOICE
        assert checkpoints[-1]["active_step_id"] == STEP_VOICE
        assert checkpoints[-1]["draft_values"][STEP_MODEL] == {
            "model_id": "mounted-model"
        }
        assert wizard.app_instance.app_config["chat_defaults"] == {
            "provider": "custom",
            "model": "mounted-model",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["endpoint", "credential"])
async def test_mounted_back_provider_identity_change_cannot_commit_old_model(change):
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = None
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        atomic_models = []

        async def commit_config(
            settings,
            *,
            delete_keys=None,
            after_write=None,
            provider_setup_mutation=None,
        ):
            if provider_setup_mutation is not None:
                atomic_models.append(settings["chat_defaults"]["model"])
                container._mirror_into_app_config(settings, delete_keys)
            return True

        container.commit_config = commit_config
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        provider_step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://first.example.test/v1"
        await pilot.pause()

        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        model_step.query_one("#setup-model-custom", Input).value = "old-model"
        await pilot.pause()
        old_key = model_step._shown_for_discovery_key
        assert old_key is not None

        container.show_step(provider_index)
        if change == "endpoint":
            provider_step.query_one(
                "#setup-provider-endpoint", Input
            ).value = "https://second.example.test/v1"
            await pilot.pause()
        else:
            provider_step.query_one(
                "#setup-provider-api-key", Input
            ).value = "replacement-secret"
        await container._advance()
        await pilot.pause()

        assert container.current_step == model_index
        assert model_step._shown_for_discovery_key != old_key
        assert model_step.selected_model_id == ""
        assert model_step.query_one("#setup-model-custom", Input).value == ""
        await container._advance()
        assert atomic_models == []
        assert container.wizard_data[STEP_MODEL] == {"model_id": ""}

        container.show_step(model_index)
        model_step.query_one("#setup-model-custom", Input).value = "new-model"
        await pilot.pause()
        await container._advance()
        assert atomic_models == ["new-model"]


@pytest.mark.asyncio
async def test_mounted_unchanged_backtrack_preserves_model_and_discovery_request():
    from unittest.mock import AsyncMock

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://stable.example.test/v1"}}
    }
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "stable-model")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        await pilot.pause(0.1)
        await container._advance()
        await pilot.pause(0.1)

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        model_step.query_one("#setup-model-custom", Input).value = "stable-model"
        await pilot.pause()
        original_key = model_step._shown_for_discovery_key
        request_count = scope_service.discover_models.await_count

        container.show_step(provider_index)
        await container._advance()
        await pilot.pause(0.1)

        assert model_step._shown_for_discovery_key == original_key
        assert model_step._effective_model_id() == "stable-model"
        assert scope_service.discover_models.await_count == request_count


@pytest.mark.asyncio
async def test_first_provider_transition_reuses_completed_exact_typed_discovery():
    from unittest.mock import AsyncMock

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://first.example.test/v1"}}
    }
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "first-live-model")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")

        for _ in range(20):
            if provider_step._selected_discovery_state == "complete":
                break
            await pilot.pause(0.05)
        assert provider_step._selected_discovery_state == "complete"
        selected_key = provider_step._selected_discovery_key

        await container._advance()
        for _ in range(20):
            radios = list(
                container.steps[model_index]
                .query_one("#setup-model-choice", RadioSet)
                .query(RadioButton)
            )
            if any(
                getattr(button, "_model_id", "") == "first-live-model"
                for button in radios
            ):
                break
            await pilot.pause(0.05)

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        assert model_step._shown_for_discovery_key == selected_key
        assert any(
            getattr(button, "_model_id", "") == "first-live-model"
            for button in model_step.query_one("#setup-model-choice", RadioSet).query(
                RadioButton
            )
        ), tuple(
            getattr(button, "_model_id", str(button.label))
            for button in model_step.query_one("#setup-model-choice", RadioSet).query(
                RadioButton
            )
        )
        assert scope_service.discover_models.await_count == 1


@pytest.mark.asyncio
async def test_slow_exact_discovery_crosses_provider_to_model_without_restart():
    import asyncio
    from unittest.mock import AsyncMock

    started = asyncio.Event()
    release = asyncio.Event()

    async def discover_models(**_kwargs):
        started.set()
        await release.wait()
        return _typed_model_discovery_result("custom", "slow-live-model")

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://slow.example.test/v1"}}
    }
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(side_effect=discover_models)
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        await asyncio.wait_for(started.wait(), timeout=2)

        await container._advance()
        assert container.current_step == model_index
        release.set()
        for _ in range(30):
            model_step = container.steps[model_index]
            radios = list(
                model_step.query_one("#setup-model-choice", RadioSet).query(RadioButton)
            )
            if any(
                getattr(button, "_model_id", "") == "slow-live-model"
                for button in radios
            ):
                break
            await pilot.pause(0.05)

        assert any(
            getattr(button, "_model_id", "") == "slow-live-model"
            for button in container.steps[model_index]
            .query_one("#setup-model-choice", RadioSet)
            .query(RadioButton)
        )
        assert scope_service.discover_models.await_count == 1


@pytest.mark.asyncio
async def test_unchanged_provider_model_backtrack_keeps_live_radio_and_one_writer():
    from unittest.mock import AsyncMock

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://stable.example.test/v1"}}
    }
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "stable-radio-model")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        writes = []

        async def commit_config(
            settings,
            *,
            delete_keys=None,
            after_write=None,
            provider_setup_mutation=None,
        ):
            if provider_setup_mutation is not None:
                writes.append(settings["chat_defaults"]["model"])
                container._mirror_into_app_config(settings, delete_keys)
            return True

        container.commit_config = commit_config
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert provider_index is not None and model_index is not None
        assert voice_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(20):
            radios = list(
                model_step.query_one("#setup-model-choice", RadioSet).query(RadioButton)
            )
            target = next(
                (
                    button
                    for button in radios
                    if getattr(button, "_model_id", "") == "stable-radio-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()
        await container._advance()
        assert container.current_step == voice_index
        assert writes == ["stable-radio-model"]

        container.show_step(provider_index)
        await container._advance()
        await pilot.pause(0.1)

        radio_set = model_step.query_one("#setup-model-choice", RadioSet)
        pressed = radio_set.pressed_button
        assert pressed is not None
        assert pressed in radio_set.query(RadioButton)
        assert pressed.value is True
        assert getattr(pressed, "_model_id", "") == "stable-radio-model"
        assert model_step._effective_model_id() == "stable-radio-model"

        await container._advance()
        assert container.current_step == voice_index
        assert writes == ["stable-radio-model"]
        assert scope_service.discover_models.await_count == 1


@pytest.mark.asyncio
async def test_model_step_late_render_is_fenced_by_exact_discovery_key():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
        build_first_run_model_discovery_key,
    )

    old_draft = FirstRunProviderDraft(
        "custom",
        "https://old.example.test/v1/chat/completions",
        ProviderCredentialDraft("none", "", 1),
    )
    new_draft = FirstRunProviderDraft(
        "custom",
        "https://new.example.test/v1/chat/completions",
        ProviderCredentialDraft("draft", "replacement-secret", 2),
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "custom", "provider_value": "custom"}
        },
        staged_provider_draft=old_draft,
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        old_key = build_first_run_model_discovery_key(old_draft)
        wizard.staged_provider_draft = new_draft
        step.on_show()
        new_key = build_first_run_model_discovery_key(new_draft)
        await pilot.pause(0.2)

        await step._render_models(["late-old-model"], discovery_key=old_key)
        await step._render_models(["current-model"], discovery_key=new_key)

        buttons = list(step.query("#setup-model-choice RadioButton"))
        assert [getattr(button, "_model_id", "") for button in buttons] == [
            "current-model"
        ]


@pytest.mark.asyncio
async def test_model_step_cannot_recover_old_radio_before_new_key_renders():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    old_draft = FirstRunProviderDraft(
        "custom",
        "https://old.example.test/v1/chat/completions",
        ProviderCredentialDraft("none", "", 1),
    )
    new_draft = FirstRunProviderDraft(
        "custom",
        "https://new.example.test/v1/chat/completions",
        ProviderCredentialDraft("none", "", 1),
    )
    commit = AsyncMock(return_value=True)
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "custom", "provider_value": "custom"}
        },
        staged_provider_draft=old_draft,
        commit_staged_provider_setup=commit,
        rerun=False,
    )
    step = _model_step(
        wizard, discover_models=AsyncMock(return_value=["old-radio-model"])
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio = step.query_one("#setup-model-choice RadioButton", RadioButton)
        radio.value = True
        await pilot.pause()
        assert step.selected_model_id == "old-radio-model"

        wizard.staged_provider_draft = new_draft
        step.on_show()
        ok, error = await step.commit()

        assert ok, error
        commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_mounted_noop_provider_write_does_not_advance_or_checkpoint(monkeypatch):
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = None
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence."
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: ConfigMutationResult(False, False, None),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        provider_step.detected_base_url = "https://noop.example.test/v1"
        provider_step._detected_endpoint_provider_key = "custom"
        await container._advance()
        draft = container.staged_provider_draft
        before_model = json.loads(json.dumps(wizard.app_instance.app_config))

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        model_step.query_one("#setup-model-custom", Input).value = "noop-model"
        await pilot.pause()
        await container._advance()

        assert container.current_step == model_index
        assert STEP_MODEL not in container.wizard_data
        assert wizard.app_instance.app_config == before_model
        assert container.provider_setup_committed is False
        assert container.committed_provider_model == ""
        assert container.staged_provider_draft is draft


def _provider_step(
    wizard=None,
    environ=None,
    discover=None,
    probe=None,
    local_discover=None,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = wizard or SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    if not callable(getattr(wizard, "stage_provider_setup", None)):
        wizard.stage_provider_setup = MagicMock(return_value=True)
    return ProviderStep(
        wizard=wizard,
        config=WizardStepConfig(id="provider", title="Provider", step_number=2),
        discover=discover,
        probe=probe or AsyncMock(),
        local_discover=local_discover or AsyncMock(return_value=()),
        environ=environ or {},
    )


def _staged_provider_draft(wizard):
    wizard.commit_config.assert_not_called()
    wizard.stage_provider_setup.assert_called_once()
    return wizard.stage_provider_setup.call_args.args[0]


def _provider_endpoint_config(*provider_keys):
    endpoints = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "ollama": "http://127.0.0.1:11434/v1/chat/completions",
    }
    return {
        "api_settings": {
            provider_key: {"api_url": endpoints[provider_key]}
            for provider_key in provider_keys
        }
    }


def _typed_model_discovery_result(provider: str, *model_ids: str):
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
        ModelDiscoveryResult,
    )

    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint=f"https://{provider}.example.test/v1",
        status="success",
        models=tuple(
            DiscoveredModel(
                provider=provider,
                provider_list_key=provider,
                model_id=model_id,
                display_name=model_id,
                source="runtime_discovered",
                endpoint_fingerprint=f"https://{provider}.example.test/v1",
                discovered_at="2026-08-12T00:00:00Z",
            )
            for model_id in model_ids
        ),
    )


class _StepHost(App):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _reachable_endpoint_outcome(*model_ids: str):
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    return SettingsEndpointProbeOutcome(
        state="reachable",
        summary=f"reachable ({len(model_ids)} models)",
        model_ids=tuple(model_ids),
    )


@pytest.mark.asyncio
async def test_llama_provider_shows_manual_endpoint_and_optional_auth():
    step = _provider_step()
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("llama_cpp")
        await pilot.pause()

        endpoint = step.query_one("#setup-provider-endpoint", Input)
        auth = step.query_one("#setup-provider-auth-toggle", Collapsible)
        api_key = step.query_one("#setup-provider-api-key", Input)
        assert endpoint.display
        assert auth.display
        assert str(auth.title) == "Authentication (optional)"
        assert api_key.password is True
        assert step.query_one("#setup-provider-test", Button).display


@pytest.mark.asyncio
async def test_custom_provider_credentials_are_optional_and_secret_safe():
    secret = "wizard-custom-secret-value"
    step = _provider_step()
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        auth = step.query_one("#setup-provider-auth-toggle", Collapsible)
        auth.collapsed = False
        api_key = step.query_one("#setup-provider-api-key", Input)
        api_key.value = secret
        await pilot.pause()

        assert str(auth.title) == "Authentication (optional)"
        assert secret not in app.export_screenshot()
        assert secret not in str(step.render())


@pytest.mark.asyncio
async def test_cloud_provider_marks_authentication_required_and_expanded():
    step = _provider_step()
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()

        auth = step.query_one("#setup-provider-auth-toggle", Collapsible)
        assert str(auth.title) == "Authentication"
        assert not auth.collapsed


@pytest.mark.parametrize(
    ("provider", "expected_title", "expected_collapsed"),
    [
        ("openai", "Authentication", False),
        ("anthropic", "Authentication", False),
        ("custom", "Authentication (optional)", True),
        ("custom-openai-api", "Authentication (optional)", True),
        ("llama_cpp", "Authentication (optional)", True),
        ("local_llamacpp", "Authentication (optional)", True),
        ("ollama", "Authentication (optional)", True),
    ],
)
@pytest.mark.asyncio
async def test_auth_requiredness_follows_shared_provider_catalog(
    provider: str,
    expected_title: str,
    expected_collapsed: bool,
):
    step = _provider_step()
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider(provider)
        await pilot.pause()

        auth = step.query_one("#setup-provider-auth-toggle", Collapsible)
        assert str(auth.title) == expected_title
        assert auth.collapsed is expected_collapsed


@pytest.mark.asyncio
async def test_missing_required_cloud_key_blocks_test_and_continue_with_recovery():
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("model-a"))
    step = _provider_step(probe=probe, environ={})
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()

        test_button = step.query_one("#setup-provider-test", Button)
        key_status = str(
            step.query_one("#setup-provider-key-status", Static).renderable
        )
        assert test_button.disabled
        assert "API key" in key_status
        assert "OPENAI_API_KEY" in key_status

        step._on_test_pressed(Button.Pressed(test_button))
        await pilot.pause()
        probe.assert_not_awaited()
        ok, error = await step.commit()
        assert not ok
        assert "API key" in error


@pytest.mark.parametrize(
    (
        "provider",
        "app_config",
        "environ",
        "replacement",
        "expected_source",
        "expected_value",
        "probe_expected",
    ),
    [
        (
            "openai",
            {"api_settings": {"openai": {"api_key": "stored-secret"}}},
            {},
            None,
            "stored",
            "stored-secret",
            True,
        ),
        (
            "openai",
            {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
            {"OPENAI_API_KEY": "environment-secret"},
            None,
            "environment",
            "environment-secret",
            True,
        ),
        ("custom", {"api_settings": {"custom": {}}}, {}, None, "none", None, True),
        (
            "openai",
            {"api_settings": {"openai": {"api_key": "YOUR_KEY"}}},
            {},
            None,
            "none",
            None,
            False,
        ),
        (
            "openai",
            {"api_settings": {"openai": {"api_key": "old-secret"}}},
            {},
            "rotated-secret",
            "draft",
            "rotated-secret",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_probe_resolves_exact_credential_only_at_request_boundary(
    provider,
    app_config,
    environ,
    replacement,
    expected_source,
    expected_value,
    probe_expected,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("model-a"))
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        note_key_entered=MagicMock(),
        stage_provider_setup=MagicMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, probe=probe, environ=environ)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider(provider)
        if provider == "custom":
            step.query_one(
                "#setup-provider-endpoint", Input
            ).value = "https://custom.example.test/v1"
        if replacement is not None:
            step._on_replace()
            step.query_one("#setup-provider-api-key", Input).value = replacement
        await pilot.pause()

        test_button = step.query_one("#setup-provider-test", Button)
        step._on_test_pressed(Button.Pressed(test_button))
        await pilot.pause(0.1)

        if not probe_expected:
            probe.assert_not_awaited()
            return
        assert probe.await_args.kwargs["credential_source"] == expected_source
        assert probe.await_args.kwargs["credential_value"] == expected_value
        identity = step._provider_current_draft_identity()
        assert identity is not None
        assert identity.credential_source == expected_source
        for rendered in (
            repr(identity),
            repr(step._provider_evidence_store()),
            app.export_screenshot(),
        ):
            assert expected_value is None or expected_value not in rendered


@pytest.mark.parametrize(
    ("provider", "configured_endpoint", "endpoint_edit", "test_supported"),
    [
        ("anthropic", None, None, False),
        ("anthropic", "https://api.anthropic.com/v1", None, False),
        ("cohere", None, None, False),
        ("deepseek", None, None, True),
        ("google", None, None, False),
        ("groq", None, None, True),
        ("huggingface", None, None, False),
        ("mistral", None, None, True),
        ("MistralAI", None, None, True),
        ("moonshot", None, None, False),
        ("openai", None, None, True),
        ("OpenAI", "https://gateway.example.test/openai/v1", None, True),
        ("openrouter", None, None, True),
        ("qwencloud", None, None, True),
        ("qwencloud", None, "", False),
        ("qwencloud", None, "https://bad host/v1", False),
        (
            "QwenCloud",
            "https://gateway.example.test/qwen/compatible-mode/v1",
            None,
            True,
        ),
        ("zai", None, None, False),
    ],
)
@pytest.mark.asyncio
async def test_cloud_provider_test_is_enabled_only_with_compatible_probe_target(
    provider: str,
    configured_endpoint: str | None,
    endpoint_edit: str | None,
    test_supported: bool,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook.Chat.provider_endpoint_contract import (
        resolve_provider_endpoint,
    )
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    canonical_provider = ProviderStep._canonical_provider_key(provider)
    env_var = default_api_key_env_var(canonical_provider)
    assert env_var is not None
    probe = AsyncMock(return_value=_reachable_endpoint_outcome("matrix-model"))
    app_config = {}
    if configured_endpoint is not None:
        app_config = {
            "api_settings": {canonical_provider: {"api_base_url": configured_endpoint}}
        }
    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config=app_config,
        llm_provider_catalog_scope_service=None,
    )
    wizard.note_key_entered = MagicMock()
    wizard.stage_provider_setup = MagicMock(return_value=True)
    wizard.rerun = False
    step = _provider_step(
        wizard=wizard,
        environ={env_var: f"{provider}-matrix-secret"},
        discover=AsyncMock(return_value=()),
        probe=probe,
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider(provider)
        await pilot.pause()
        if endpoint_edit is not None:
            step.query_one("#setup-provider-endpoint", Input).value = endpoint_edit
            await pilot.pause()

        assert step.query_one("#setup-provider-auth-toggle", Collapsible).title == (
            "Authentication"
        )
        test_button = step.query_one("#setup-provider-test", Button)
        target = step._probe_target()
        if test_supported:
            assert not test_button.disabled
            assert target
            resolution = resolve_provider_endpoint(canonical_provider, target)
            assert resolution.models_url is not None
            assert not resolution.errors
            test_button.press()
            await pilot.pause(0.1)
            probe.assert_awaited_once()
            assert probe.await_args.args[0] == target
            assert probe.await_args.kwargs["provider"] == canonical_provider
        else:
            assert test_button.disabled
            assert not target
            test_button.press()
            await pilot.pause()
            probe.assert_not_awaited()
            assert (
                "connection testing is unavailable"
                in str(
                    step.query_one("#setup-provider-key-status", Static).renderable
                ).lower()
            )


@pytest.mark.parametrize("edit_action", ["type", "keep", "replace", "clear", "env"])
@pytest.mark.asyncio
async def test_every_credential_semantic_edit_cancels_mounted_probe(
    edit_action: str,
):
    import asyncio
    from types import SimpleNamespace

    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def delayed_probe(*_args, **_kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return _reachable_endpoint_outcome("stale-model")

    app_config = {"api_settings": {"custom": {"api_key": "stored-secret"}}}
    environ = {"CUSTOM_API_KEY": "environment-secret"}
    if edit_action == "env":
        app_config = {"api_settings": {"custom": {"api_key_env_var": "CUSTOM_API_KEY"}}}
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        note_key_entered=MagicMock(),
        stage_provider_setup=MagicMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, probe=delayed_probe, environ=environ)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://custom.example.test/v1"
        if edit_action == "type":
            step._on_replace()
            step.query_one("#setup-provider-api-key", Input).value = "key-a"
        step._on_test_pressed(
            Button.Pressed(step.query_one("#setup-provider-test", Button))
        )
        await asyncio.wait_for(started.wait(), timeout=2)

        if edit_action == "type":
            step.query_one("#setup-provider-api-key", Input).value = "key-b"
        elif edit_action == "keep":
            step._on_keep()
        elif edit_action == "replace":
            step._on_replace()
        elif edit_action == "clear":
            step._on_clear()
        else:
            step._on_replace()
            step.query_one("#setup-provider-api-key", Input).value = "draft-secret"

        await asyncio.wait_for(cancelled.wait(), timeout=2)
        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert "Testing" not in status
        release.set()
        await pilot.pause(0.1)
        assert "stale-model" not in str(
            step.query_one("#setup-provider-probe-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_late_key_a_cancellation_cannot_clear_newer_key_b_evidence():
    import asyncio

    old_started = asyncio.Event()
    old_cancelled = asyncio.Event()
    release_old = asyncio.Event()
    calls = 0

    async def probe(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            old_started.set()
            try:
                await release_old.wait()
            except asyncio.CancelledError:
                old_cancelled.set()
                await release_old.wait()
            return _reachable_endpoint_outcome("old-only")
        return _reachable_endpoint_outcome("new-a", "new-b")

    step = _provider_step(probe=probe)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://custom.example.test/v1"
        key_input = step.query_one("#setup-provider-api-key", Input)
        key_input.value = "key-a"
        step._on_test_pressed(
            Button.Pressed(step.query_one("#setup-provider-test", Button))
        )
        await asyncio.wait_for(old_started.wait(), timeout=2)

        key_input.value = "key-b"
        await asyncio.wait_for(old_cancelled.wait(), timeout=2)
        await pilot.pause()
        step._on_test_pressed(
            Button.Pressed(step.query_one("#setup-provider-test", Button))
        )
        await pilot.pause(0.1)
        identity = step._provider_current_draft_identity()
        assert identity is not None
        evidence = step._provider_evidence_store().evidence_for(identity)
        assert evidence is not None
        assert evidence.model_ids == ("new-a", "new-b")

        release_old.set()
        await pilot.pause(0.1)
        evidence = step._provider_evidence_store().evidence_for(identity)
        assert evidence is not None
        assert evidence.model_ids == ("new-a", "new-b")


@pytest.mark.parametrize(
    ("state", "category", "model_ids"),
    [
        ("reachable", None, ("model-a",)),
        pytest.param(
            "model_listing_unavailable",
            "http_status",
            (),
            id="model-listing-unavailable",
        ),
        pytest.param("unreachable", "connection_refused", (), id="connection-refused"),
    ],
)
def test_settings_probe_outcomes_convert_to_exact_provider_probe_results(
    state,
    category,
    model_ids,
):
    from tldw_chatbook.Chat.provider_test_evidence import ProviderProbeResult
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    outcome = SettingsEndpointProbeOutcome(
        state=state,
        summary="ignored arbitrary detail",
        category=category,
        model_ids=model_ids,
    )
    result = ProviderStep._provider_probe_result_from_outcome(outcome)
    assert type(result) is ProviderProbeResult
    assert result.endpoint == state
    assert result.category == category
    assert result.model_ids == model_ids


def test_probe_conversion_rejects_duck_type_without_property_access():
    class HostileDuck:
        touched = False

        @property
        def state(self):
            self.touched = True
            raise AssertionError("duck property accessed")

    outcome = HostileDuck()
    with pytest.raises(ValueError, match="outcome"):
        ProviderStep._provider_probe_result_from_outcome(outcome)
    assert not outcome.touched


@pytest.mark.asyncio
async def test_probe_settles_exact_evidence_and_endpoint_edit_invalidates_it():
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("model-a", "model-b"))
    step = _provider_step(probe=probe)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "https://gateway.example.test/v1"
        await pilot.pause()
        step._on_test_pressed(
            Button.Pressed(step.query_one("#setup-provider-test", Button))
        )
        await pilot.pause(0.1)

        identity = step._provider_current_draft_identity()
        assert identity is not None
        evidence = step._provider_evidence_store().evidence_for(identity)
        assert evidence is not None
        assert evidence.endpoint == "reachable"
        assert evidence.model_ids == ("model-a", "model-b")

        endpoint.value = "https://changed.example.test/v1"
        await pilot.pause()
        assert step._provider_evidence_store().evidence_for(identity) is None
        assert (
            "test again"
            in str(
                step.query_one("#setup-provider-probe-status", Static).renderable
            ).lower()
        )


@pytest.mark.asyncio
async def test_mounted_test_continue_returned_model_save_rebases_exact_evidence(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    secret = "mounted-save-boundary-secret"
    probe = AsyncMock(return_value=_reachable_endpoint_outcome("returned-model"))
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "returned-model")
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        lambda _mutation: ConfigMutationResult(True, True, None),
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._probe = probe
        provider_step.select_provider("custom")
        provider_step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://save.example.test/v1"
        provider_step.query_one("#setup-provider-api-key", Input).value = secret
        await pilot.pause()
        provider_step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        tested = provider_step._provider_current_draft_identity()
        assert tested is not None
        assert tested.credential_source == "draft"
        tested_evidence = provider_step._provider_evidence_store().evidence_for(tested)
        assert tested_evidence is not None
        assert tested_evidence.model_ids == ("returned-model",)

        await container._advance()
        assert container.current_step == model_index
        assert container.staged_provider_draft is not None
        assert container.staged_provider_draft.credential.revision == (
            tested.credential_revision
        )
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        target = None
        for _ in range(20):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "returned-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()
        assert provider_step._provider_evidence_store().evidence_for(tested) is not None

        await container._advance()
        assert container.provider_setup_committed
        saved = provider_step._last_tested_provider_identity
        assert saved is not None
        assert saved.credential_source == "stored"
        saved_evidence = provider_step._provider_evidence_store().evidence_for(saved)
        assert saved_evidence is not None
        assert saved_evidence.model_ids == ("returned-model",)
        verdict = get_provider_readiness(
            "custom", wizard.app_instance.app_config, environ={}
        ).verdict(
            selected_model="returned-model",
            evidence=saved_evidence,
            current_identity=saved,
        )
        assert verdict.code == "verified"
        assert verdict.verified
        for rendered in (repr(saved), repr(saved_evidence), app.export_screenshot()):
            assert secret not in rendered


@pytest.mark.asyncio
async def test_mounted_stored_key_test_continue_returned_model_save_preserves_evidence(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    secret = "preexisting-inline-boundary-secret"
    probe = AsyncMock(return_value=_reachable_endpoint_outcome("stored-key-model"))
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "stored-key-model")
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {
                "api_url": "https://stored.example.test/v1/chat/completions",
                "api_key": secret,
            }
        }
    }
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        lambda _mutation: ConfigMutationResult(True, True, None),
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._probe = probe
        provider_step.select_provider("custom")
        await pilot.pause()

        provider_step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        tested = provider_step._provider_current_draft_identity()
        assert tested is not None
        assert tested.credential_source == "stored"
        assert provider_step._provider_evidence_store().evidence_for(tested) is not None
        assert probe.await_args.kwargs["credential_value"] == secret

        await container._advance()
        assert container.current_step == model_index
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        target = None
        for _ in range(20):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "stored-key-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()
        assert provider_step._provider_evidence_store().evidence_for(tested) is not None

        await container._advance()
        assert container.provider_setup_committed
        saved = provider_step._last_tested_provider_identity
        assert saved is not None
        assert saved.credential_source == "stored"
        saved_evidence = provider_step._provider_evidence_store().evidence_for(saved)
        assert saved_evidence is not None
        verdict = get_provider_readiness(
            "custom", wizard.app_instance.app_config, environ={}
        ).verdict(
            selected_model="stored-key-model",
            evidence=saved_evidence,
            current_identity=saved,
        )
        assert verdict.verified
        for rendered in (
            repr(container.staged_provider_draft),
            repr(tested),
            repr(saved),
            repr(saved_evidence),
            app.export_screenshot(),
        ):
            assert secret not in rendered


@pytest.mark.parametrize("credential_kind", ["environment", "stored"])
@pytest.mark.parametrize("rotation_timing", ["before_selection", "before_save"])
@pytest.mark.asyncio
async def test_credential_rotation_after_provider_handoff_invalidates_before_save(
    monkeypatch,
    credential_kind: str,
    rotation_timing: str,
):
    from unittest.mock import AsyncMock

    first_secret = f"{credential_kind}-handoff-secret-a"
    rotated_secret = f"{credential_kind}-handoff-secret-b"
    provider_settings = {"api_url": "https://rotation.example.test/v1/chat/completions"}
    if credential_kind == "environment":
        provider_settings["api_key_env_var"] = "CUSTOM_API_KEY"
        monkeypatch.setenv("CUSTOM_API_KEY", first_secret)
    else:
        provider_settings["api_key"] = first_secret

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("rotation-model"))
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "rotation-model")
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {"api_settings": {"custom": provider_settings}}
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        lambda _mutation: ConfigMutationResult(True, True, None),
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._probe = probe
        provider_step.select_provider("custom")
        await pilot.pause()
        provider_step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        tested = provider_step._last_tested_provider_identity
        assert tested is not None
        assert provider_step._provider_evidence_store().evidence_for(tested) is not None
        observation = provider_step._credential_observations["custom"]
        assert "b'" not in repr(observation)
        assert first_secret not in repr(observation)

        await container._advance()
        assert container.current_step == model_index
        if rotation_timing == "before_selection":
            if credential_kind == "environment":
                monkeypatch.setenv("CUSTOM_API_KEY", rotated_secret)
            else:
                provider_settings["api_key"] = rotated_secret

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        target = None
        for _ in range(20):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "rotation-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()

        if rotation_timing == "before_save":
            assert (
                provider_step._provider_evidence_store().evidence_for(tested)
                is not None
            )
            if credential_kind == "environment":
                monkeypatch.setenv("CUSTOM_API_KEY", rotated_secret)
            else:
                provider_settings["api_key"] = rotated_secret

        await container._advance()
        assert container.provider_setup_committed
        assert provider_step._credential_revision > tested.credential_revision
        assert provider_step._provider_evidence_store().evidence_for(tested) is None
        assert provider_step._last_tested_provider_identity == tested
        for rendered in (
            repr(container.staged_provider_draft),
            repr(provider_step._provider_evidence_store()),
            repr(provider_step._selected_provider_models),
            app.export_screenshot(),
        ):
            assert first_secret not in rendered
            assert rotated_secret not in rendered


@pytest.mark.parametrize("save_outcome", ["noop", "partial", "conflict", "cancelled"])
@pytest.mark.asyncio
async def test_mounted_incomplete_save_never_rebases_tested_draft_evidence(
    monkeypatch,
    save_outcome: str,
):
    import asyncio
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("returned-model"))
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "returned-model")
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )

    def persist(_mutation):
        if save_outcome == "cancelled":
            raise asyncio.CancelledError
        return {
            "noop": ConfigMutationResult(False, False, None),
            "partial": ConfigMutationResult(True, False, "cache_reload"),
            "conflict": ConfigMutationResult(False, False, None, conflict=True),
        }[save_outcome]

    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        persist,
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._probe = probe
        provider_step.select_provider("custom")
        provider_step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://incomplete-save.example.test/v1"
        provider_step.query_one(
            "#setup-provider-api-key", Input
        ).value = "incomplete-save-secret"
        await pilot.pause()
        provider_step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        tested = provider_step._provider_current_draft_identity()
        assert tested is not None
        assert tested.credential_source == "draft"

        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        target = None
        for _ in range(20):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "returned-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()

        if save_outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await container._advance()
        else:
            await container._advance()

        assert not container.provider_setup_committed
        assert container.current_step == model_index
        assert provider_step._last_tested_provider_identity == tested
        assert provider_step._provider_evidence_store().evidence_for(tested) is not None
        assert "incomplete-save-secret" not in app.export_screenshot()


@pytest.mark.parametrize("dismissal", ["container", "finish_later"])
@pytest.mark.asyncio
async def test_mounted_dismissal_clears_all_staged_provider_secrets_and_tasks(
    monkeypatch,
    dismissal: str,
):
    import asyncio
    from unittest.mock import AsyncMock

    secret = f"{dismissal}-staged-provider-secret"
    environment_secret = f"{dismissal}-environment-sentinel"
    endpoint_secret = f"{dismissal}-private-endpoint-path"
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {"api_key_env_var": "CUSTOM_API_KEY"},
        }
    }
    wizard.app_instance.llm_provider_catalog_scope_service = None
    monkeypatch.setenv("CUSTOM_API_KEY", environment_secret)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        assert provider_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        endpoint_input = provider_step.query_one("#setup-provider-endpoint", Input)
        endpoint_input.value = f"https://dismiss.example.test/{endpoint_secret}/v1"
        key_input = provider_step.query_one("#setup-provider-api-key", Input)
        key_input.value = secret
        await pilot.pause()
        await container._advance()
        assert container.staged_provider_draft is not None

        blocker = asyncio.Event()
        commit_task = asyncio.create_task(blocker.wait())
        container._provider_commit_task = commit_task
        container._provider_commit_identity = (container._provider_stage_generation, "")
        if dismissal == "finish_later":
            container.persist_current_checkpoint = AsyncMock(return_value=True)
            await wizard._finish_later()
        else:
            container._dismiss_screen(None)
        await pilot.pause()

        assert app.wizard_result is None
        assert container.staged_provider_draft is None
        assert container._provider_commit_task is None
        assert container._provider_commit_identity is None
        assert commit_task.cancelled()
        assert provider_step._provider_drafts == {}
        assert key_input.value == ""
        assert endpoint_input.value == ""
        assert getattr(container, "_first_run_selected_provider_models", {}) == {}
        for rendered in (
            repr(container.__dict__),
            repr(provider_step.__dict__),
            app.export_screenshot(),
        ):
            assert secret not in rendered
            assert environment_secret not in rendered
            assert endpoint_secret not in rendered


@pytest.mark.parametrize("writer_outcome", ["success", "error"])
@pytest.mark.asyncio
async def test_dismissal_waits_for_irreversible_provider_executor_write(
    monkeypatch,
    writer_outcome: str,
):
    import asyncio
    import threading

    secret = f"executor-{writer_outcome}-credential"
    endpoint_secret = f"executor-{writer_outcome}-private-path"
    writer_started = threading.Event()
    release_writer = threading.Event()
    writer_settled = threading.Event()
    events: list[str] = []

    def blocked_persist(mutation):
        values = mutation.section_values["api_settings.custom"]
        assert values["api_key"] == secret
        writer_started.set()
        assert release_writer.wait(timeout=5)
        events.append("writer-settled")
        writer_settled.set()
        if writer_outcome == "error":
            raise RuntimeError("bounded writer failure")
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        blocked_persist,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = None
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        assert provider_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        provider_step.query_one(
            "#setup-provider-endpoint", Input
        ).value = f"https://executor.example.test/{endpoint_secret}/v1"
        provider_step.query_one("#setup-provider-api-key", Input).value = secret
        await pilot.pause()
        await container._advance()

        commit_waiter = asyncio.create_task(
            container.commit_staged_provider_setup("executor-model")
        )
        for _ in range(40):
            if writer_started.is_set():
                break
            await pilot.pause(0.025)
        assert writer_started.is_set()
        assert container._provider_commit_write_started

        try:
            container._dismiss_screen(None)
            await pilot.pause(0.1)
            assert app.wizard_result == "UNSET"
            assert not writer_settled.is_set()
            assert not commit_waiter.done()
        finally:
            release_writer.set()

        assert await commit_waiter is (writer_outcome == "success")
        for _ in range(40):
            if app.wizard_result is None:
                break
            await pilot.pause(0.025)
        assert app.wizard_result is None
        events.append("dismissed")
        assert events == ["writer-settled", "dismissed"]
        assert container._provider_commit_task is None
        assert container._provider_commit_identity is None
        assert not container._provider_commit_write_started
        assert container.staged_provider_draft is None
        for rendered in (
            repr(container.__dict__),
            repr(provider_step.__dict__),
            app.export_screenshot(),
        ):
            assert secret not in rendered
            assert endpoint_secret not in rendered


@pytest.mark.asyncio
async def test_credential_revision_changes_only_for_new_semantic_decisions():
    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config={"api_settings": {"custom": {"api_key": "stored-secret"}}},
        llm_provider_catalog_scope_service=None,
    )
    wizard.note_key_entered = MagicMock()
    wizard.stage_provider_setup = MagicMock(return_value=True)
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        keep = step.query_one("#setup-provider-key-keep", Button)
        replace = step.query_one("#setup-provider-key-replace", Button)
        clear = step.query_one("#setup-provider-key-clear", Button)

        initial = step._credential_revision
        keep.press()
        await pilot.pause()
        assert step._credential_revision == initial

        replace.press()
        await pilot.pause()
        after_replace = step._credential_revision
        assert after_replace == initial + 1
        replace.press()
        await pilot.pause()
        assert step._credential_revision == after_replace

        clear.press()
        await pilot.pause()
        after_clear = step._credential_revision
        assert after_clear == after_replace + 1
        clear.press()
        await pilot.pause()
        assert step._credential_revision == after_clear

        keep.press()
        await pilot.pause()
        after_keep = step._credential_revision
        assert after_keep == after_clear + 1
        keep.press()
        await pilot.pause()
        assert step._credential_revision == after_keep


@pytest.mark.parametrize("change", ["endpoint", "credential", "manual-model"])
@pytest.mark.asyncio
async def test_mounted_changed_semantics_do_not_rebase_test_evidence(
    monkeypatch,
    change: str,
):
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("returned-model"))
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "returned-model")
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        lambda _mutation: ConfigMutationResult(True, True, None),
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._probe = probe
        provider_step.select_provider("custom")
        endpoint = provider_step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "https://before-change.example.test/v1"
        key = provider_step.query_one("#setup-provider-api-key", Input)
        key.value = "credential-a"
        await pilot.pause()
        provider_step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        tested = provider_step._provider_current_draft_identity()
        assert tested is not None

        if change == "endpoint":
            endpoint.value = "https://after-change.example.test/v1"
            await pilot.pause()
        elif change == "credential":
            key.value = "credential-b"
            await pilot.pause()
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        if change == "manual-model":
            model_step.query_one("#setup-model-custom", Input).value = "manual-model"
            await pilot.pause()
        else:
            target = None
            for _ in range(20):
                target = next(
                    (
                        button
                        for button in model_step.query(RadioButton)
                        if getattr(button, "_model_id", "") == "returned-model"
                    ),
                    None,
                )
                if target is not None:
                    break
                await pilot.pause(0.05)
            assert target is not None
            target.value = True
            await pilot.pause()
        await container._advance()

        assert provider_step._provider_evidence_store().evidence_for(tested) is None
        saved = provider_step._last_tested_provider_identity
        if saved is not None:
            assert saved.credential_source != "stored" or (
                provider_step._provider_evidence_store().evidence_for(saved) is None
            )


@pytest.mark.asyncio
async def test_mounted_models_404_then_manual_model_never_reports_verified(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    probe = AsyncMock(
        return_value=SettingsEndpointProbeOutcome(
            state="model_listing_unavailable",
            summary="models unavailable",
            category="http_status",
        )
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {}
    wizard.app_instance.llm_provider_catalog_scope_service = None
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        lambda _mutation: ConfigMutationResult(True, True, None),
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._probe = probe
        provider_step.select_provider("custom")
        provider_step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://listing-unavailable.example.test/v1"
        await pilot.pause()
        provider_step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        tested = provider_step._provider_current_draft_identity()
        assert tested is not None
        evidence = provider_step._provider_evidence_store().evidence_for(tested)
        assert evidence is not None
        assert evidence.endpoint == "model_listing_unavailable"

        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        model_step.query_one("#setup-model-custom", Input).value = "manual-model"
        await pilot.pause()
        await container._advance()

        current_evidence = provider_step._provider_evidence_store().evidence_for(tested)
        verdict = get_provider_readiness(
            "custom", wizard.app_instance.app_config, environ={}
        ).verdict(
            selected_model="manual-model",
            evidence=current_evidence,
            current_identity=tested,
        )
        assert verdict.code != "verified"
        assert not verdict.verified


@pytest.mark.asyncio
async def test_provider_switch_restores_each_provider_owned_draft_without_leakage():
    from copy import copy
    from unittest.mock import AsyncMock

    custom_server = DiscoveredLocalServer("custom", "https://custom-detect.test/v1")
    llama_server = DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:8181")
    step = _provider_step(local_discover=AsyncMock(return_value=()))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://custom-draft.test/v1"
        step.query_one("#setup-provider-api-key", Input).value = "custom-secret"
        step._detected_servers = (custom_server,)
        step._render_detection_results((custom_server,))

        step.select_provider("llama_cpp")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "http://127.0.0.1:8282"
        step.query_one("#setup-provider-api-key", Input).value = "llama-secret"
        step._detected_servers = (llama_server,)
        step._render_detection_results((llama_server,))

        step.select_provider("custom")
        await pilot.pause()
        assert step.query_one("#setup-provider-endpoint", Input).value == (
            "https://custom-draft.test/v1"
        )
        assert step.query_one("#setup-provider-api-key", Input).value == (
            "custom-secret"
        )
        custom_results = step.query_one("#setup-provider-detection-results", OptionList)
        assert "custom-detect.test" in " ".join(
            str(custom_results.get_option_at_index(index).prompt)
            for index in range(custom_results.option_count)
        )
        assert "127.0.0.1:8181" not in app.export_screenshot()

        step.select_provider("llama_cpp")
        await pilot.pause()
        assert step.query_one("#setup-provider-endpoint", Input).value == (
            "http://127.0.0.1:8282"
        )
        assert step.query_one("#setup-provider-api-key", Input).value == (
            "llama-secret"
        )
        assert "custom-secret" not in repr(step._provider_drafts)
        assert "llama-secret" not in repr(step._provider_drafts)
        assert "custom-draft.test" not in repr(step._provider_drafts)
        assert "127.0.0.1:8282" not in repr(step._provider_drafts)
        with pytest.raises(TypeError):
            copy(next(iter(step._provider_drafts.values())))

    assert not step._provider_drafts


@pytest.mark.asyncio
async def test_selected_duplicate_detection_candidate_restores_exactly_per_provider():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    duplicate_url = "https://duplicate-detect.test/v1"
    first = DiscoveredLocalServer("custom", duplicate_url, ("first-model",))
    selected = DiscoveredLocalServer("custom", duplicate_url, ("selected-model",))
    step = _provider_step(local_discover=AsyncMock(return_value=()))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step._detected_servers = (first, selected)
        step._render_detection_results(step._detected_servers)
        results = step.query_one("#setup-provider-detection-results", OptionList)
        selected_index = next(
            index
            for index in range(results.option_count)
            if getattr(results.get_option_at_index(index), "server", None) is selected
        )
        results.highlighted = selected_index
        step._on_detected_endpoint_selected(
            SimpleNamespace(option=results.get_option_at_index(selected_index))
        )
        await pilot.pause()

        endpoint = step.query_one("#setup-provider-endpoint", Input)
        banner = step.query_one("#setup-provider-detected", Static)
        use_button = step.query_one("#setup-provider-use-detected", Button)
        assert endpoint.value == selected.base_url
        assert step.detected_server is selected
        assert "duplicate-detect.test" in str(banner.renderable)
        assert not banner.has_class("hidden")
        assert not use_button.has_class("hidden")

        step.select_provider("llama_cpp")
        await pilot.pause()
        assert "duplicate-detect.test" not in app.export_screenshot()
        assert not hasattr(step, "detected_server")

        step.select_provider("custom")
        await pilot.pause()
        restored_results = step.query_one(
            "#setup-provider-detection-results", OptionList
        )
        assert step.detected_server is selected
        assert endpoint.value == selected.base_url
        assert "duplicate-detect.test" in str(banner.renderable)
        assert not banner.has_class("hidden")
        assert not use_button.has_class("hidden")
        assert getattr(restored_results.highlighted_option, "server", None) is selected


@pytest.mark.asyncio
async def test_effective_chat_url_is_safe_and_rejects_unsafe_endpoint_parts():
    step = _provider_step()
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "https://gateway.example.test/proxy/v1"
        await pilot.pause()

        effective = str(
            step.query_one("#setup-provider-effective-chat", Static).renderable
        )
        assert "https://gateway.example.test/proxy/v1/chat/completions" in effective

        for hostile, expected_error in (
            ("https://user:private@example.test/v1", "user information"),
            ("https://example.test/v1?token=private", "query"),
            ("https://example.test/v1#private-fragment", "fragment"),
        ):
            endpoint.value = hostile
            await pilot.pause()
            rendered = "\n".join(
                str(widget.renderable) for widget in step.query(Static)
            )
            assert "user:private" not in rendered
            assert "token=private" not in rendered
            assert "private-fragment" not in rendered
            assert expected_error in rendered


@pytest.mark.asyncio
async def test_detection_does_not_replace_typed_endpoint_and_lists_every_candidate():
    from unittest.mock import AsyncMock

    typed = "http://127.0.0.1:9999/v1/chat/completions"
    servers = (
        DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:8080", ("a",)),
        DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:8080", ("b",)),
        DiscoveredLocalServer("ollama", "http://127.0.0.1:11434", ("c",)),
    )
    discover = AsyncMock(return_value=servers)
    step = _provider_step(local_discover=discover)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        step.select_provider("llama_cpp")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = typed
        step.query_one("#setup-provider-detect", Button).press()
        await pilot.pause(0.1)

        choices = step.query_one("#setup-provider-detection-results", OptionList)
        candidates = [
            choices.get_option_at_index(index)
            for index in range(choices.option_count)
            if getattr(choices.get_option_at_index(index), "server", None) is not None
        ]
        non_candidates = [
            choices.get_option_at_index(index)
            for index in range(choices.option_count)
            if getattr(choices.get_option_at_index(index), "server", None) is None
        ]
        assert endpoint.value == typed
        assert len(candidates) == 3
        assert all(option.disabled for option in non_candidates)
        assert len({option.id for option in candidates}) == 3


@pytest.mark.asyncio
async def test_keyboard_candidate_selection_updates_exact_draft_identity():
    from unittest.mock import AsyncMock

    server = DiscoveredLocalServer(
        "llama_cpp", "http://127.0.0.1:8080/v1/chat/completions", ("m1",)
    )
    step = _provider_step(local_discover=AsyncMock(return_value=(server,)))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        results = step.query_one("#setup-provider-detection-results", OptionList)
        candidate_index = next(
            index
            for index in range(results.option_count)
            if getattr(results.get_option_at_index(index), "server", None) is not None
        )
        results.highlighted = candidate_index
        results.focus()
        await pilot.press("enter")
        await pilot.pause()

        endpoint = step.query_one("#setup-provider-endpoint", Input)
        effective = str(
            step.query_one("#setup-provider-effective-chat", Static).renderable
        )
        draft = step._effective_provider_draft()
        identity = step._model_discovery_key(draft)
        assert endpoint.value == "http://127.0.0.1:8080/v1/chat/completions"
        assert "http://127.0.0.1:8080/v1/chat/completions" in effective
        assert identity is not None
        assert identity.connection_identity == (
            "llama_cpp",
            "http://127.0.0.1:8080",
        )


@pytest.mark.asyncio
async def test_provider_switch_clears_owned_endpoint_auth_and_detection_state():
    from unittest.mock import AsyncMock

    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config={
            "api_settings": {
                "custom": {"api_url": "https://custom.example.test/v1/chat/completions"}
            }
        },
        llm_provider_catalog_scope_service=None,
    )
    wizard.note_key_entered = MagicMock()
    wizard.stage_provider_setup = MagicMock(return_value=True)
    server = DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:8080", ())
    step = _provider_step(
        wizard=wizard,
        local_discover=AsyncMock(return_value=(server,)),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        step.select_provider("llama_cpp")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "http://127.0.0.1:9999"
        step.query_one("#setup-provider-api-key", Input).value = "llama-secret"
        await pilot.pause()

        step.select_provider("custom")
        await pilot.pause()

        assert step.query_one("#setup-provider-endpoint", Input).value == (
            "https://custom.example.test/v1/chat/completions"
        )
        assert step.query_one("#setup-provider-api-key", Input).value == ""
        results = step.query_one("#setup-provider-detection-results", OptionList)
        assert all(
            getattr(results.get_option_at_index(index), "server", None) is None
            for index in range(results.option_count)
        )
        assert "llama-secret" not in app.export_screenshot()


@pytest.mark.asyncio
async def test_connection_test_receives_exact_current_provider_draft_only_at_boundary():
    from unittest.mock import AsyncMock

    secret = "exact-request-boundary-secret"
    probe = AsyncMock(return_value=_reachable_endpoint_outcome("exact-model"))
    step = _provider_step(probe=probe)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        typed_endpoint = "  https://gateway.example.test/proxy/v1/chat/completions  "
        step.query_one("#setup-provider-endpoint", Input).value = typed_endpoint
        step.query_one("#setup-provider-api-key", Input).value = secret
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        probe.assert_awaited_once_with(
            typed_endpoint,
            provider="custom",
            credential_source="draft",
            credential_value=secret,
        )
        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert status.startswith("✓ ")
        assert secret not in status
        assert secret not in repr(
            step._model_discovery_key(step._effective_provider_draft())
        )


@pytest.mark.asyncio
async def test_connection_test_exception_is_bounded_and_secret_safe():
    from unittest.mock import AsyncMock

    secret = "never-render-this-probe-secret"
    probe = AsyncMock(side_effect=RuntimeError(secret))
    step = _provider_step(probe=probe)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://gateway.example.test/v1"
        step.query_one("#setup-provider-api-key", Input).value = secret
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        rendered = "\n".join(str(widget.renderable) for widget in step.query(Static))
        assert "connection error" in rendered.lower()
        assert secret not in rendered


@pytest.mark.asyncio
async def test_late_detection_result_is_discarded_after_provider_switch():
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()

    async def discover(_config):
        started.set()
        await release.wait()
        return (DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:8080"),)

    step = _provider_step(local_discover=discover)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        step.select_provider("custom")
        release.set()
        await pilot.pause(0.1)

        results = step.query_one("#setup-provider-detection-results", OptionList)
        assert all(
            getattr(results.get_option_at_index(index), "server", None) is None
            for index in range(results.option_count)
        )


@pytest.mark.asyncio
async def test_zero_detection_results_render_disabled_bounded_status():
    from unittest.mock import AsyncMock

    step = _provider_step(local_discover=AsyncMock(return_value=()))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        step.query_one("#setup-provider-detect", Button).press()
        await pilot.pause(0.1)

        results = step.query_one("#setup-provider-detection-results", OptionList)
        assert results.option_count == 2
        assert all(
            results.get_option_at_index(index).disabled
            for index in range(results.option_count)
        )
        rendered = " ".join(
            str(results.get_option_at_index(index).prompt)
            for index in range(results.option_count)
        )
        assert "No local endpoints found" in rendered


@pytest.mark.asyncio
async def test_detection_never_renders_hostile_provider_or_endpoint_strings():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    provider_secret = "hostile-provider-secret"
    endpoint_secret = "hostile-endpoint-secret"
    server = DiscoveredLocalServer(
        f"llama_cpp\n{provider_secret}",
        f"http://user:{endpoint_secret}@127.0.0.1:8080/v1",
    )
    step = _provider_step(local_discover=AsyncMock(return_value=(server,)))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        screenshot = app.export_screenshot()
        assert provider_secret not in screenshot
        assert endpoint_secret not in screenshot
        assert step.query_one("#setup-provider-use-detected", Button).has_class(
            "hidden"
        )
        results = step.query_one("#setup-provider-detection-results", OptionList)
        candidate = next(
            results.get_option_at_index(index)
            for index in range(results.option_count)
            if "invalid endpoint" in str(results.get_option_at_index(index).prompt)
        )
        assert candidate.disabled
        assert getattr(candidate, "server", None) is None
        assert "invalid endpoint" in str(candidate.prompt)
        before = step.query_one("#setup-provider-endpoint", Input).value
        step._on_detected_endpoint_selected(SimpleNamespace(option=candidate))
        assert step.query_one("#setup-provider-endpoint", Input).value == before
        assert not hasattr(step, "detected_server")

        step.select_provider("custom")
        step.select_provider("llama_cpp")
        await pilot.pause()
        assert not hasattr(step, "detected_server")
        assert step.query_one("#setup-provider-use-detected", Button).has_class(
            "hidden"
        )


@pytest.mark.parametrize(
    ("typed_endpoint", "expected_chat"),
    [
        (
            "https://gateway.example.test/proxy/v1",
            "https://gateway.example.test/proxy/v1/chat/completions",
        ),
        (
            "https://gateway.example.test/proxy/v1/chat/completions",
            "https://gateway.example.test/proxy/v1/chat/completions",
        ),
    ],
)
@pytest.mark.asyncio
async def test_connection_test_accepts_base_or_full_chat_without_route_duplication(
    typed_endpoint: str,
    expected_chat: str,
):
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome())
    step = _provider_step(probe=probe)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one("#setup-provider-endpoint", Input).value = typed_endpoint
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        assert probe.await_args.args == (typed_endpoint,)
        effective = str(
            step.query_one("#setup-provider-effective-chat", Static).renderable
        )
        assert expected_chat in effective
        assert effective.count("/v1/chat/completions") == 1


@pytest.mark.asyncio
async def test_default_probe_client_construction_failure_is_bounded(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx

    secret = "client-construction-secret"

    def fail_client(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(httpx, "AsyncClient", fail_client)
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        stage_provider_setup=MagicMock(return_value=True),
        rerun=False,
    )
    step = ProviderStep(
        wizard=wizard,
        config=WizardStepConfig(id="provider", title="Provider", step_number=2),
        local_discover=AsyncMock(return_value=()),
        environ={},
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://gateway.example.test/v1"
        step.query_one("#setup-provider-api-key", Input).value = "draft-key"
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert "connection error" in status.lower()
        assert secret not in status


@pytest.mark.asyncio
async def test_default_probe_client_close_failure_preserves_structured_outcome(
    monkeypatch,
):
    import httpx

    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    secret = "client-close-secret"

    class CloseFailureClient:
        def __init__(self, **_kwargs):
            pass

        async def aclose(self):
            raise RuntimeError(secret)

    async def reachable_probe(*_args, **_kwargs):
        return SettingsEndpointProbeOutcome(
            state="reachable",
            summary="reachable (1 model)",
            model_ids=("model-a",),
        )

    monkeypatch.setattr(httpx, "AsyncClient", CloseFailureClient)
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        reachable_probe,
    )

    outcome = await _probe_first_run_provider_connection(
        "https://gateway.example.test/v1",
        provider="custom",
        credential_source="draft",
        credential_value="draft-key",
    )

    assert type(outcome) is SettingsEndpointProbeOutcome
    assert outcome.state == "reachable"
    assert outcome.category is None
    assert outcome.summary == "reachable (1 model)"
    assert secret not in repr(outcome)


@pytest.mark.asyncio
async def test_first_run_contacts_only_selected_provider():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    selected_discovery = AsyncMock(return_value=())
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("ollama", "ollama-model")
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config=_provider_endpoint_config("ollama"),
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=selected_discovery)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        notify_spy = MagicMock()
        app.notify = notify_spy
        await pilot.pause()

        step.select_provider("Ollama")
        await pilot.pause(0.1)

        assert [call.args[0] for call in selected_discovery.await_args_list] == [
            "ollama"
        ]
        assert [
            call.kwargs["provider"]
            for call in scope_service.discover_models.await_args_list
        ] == ["ollama"]
        notify_spy.assert_not_called()


@pytest.mark.asyncio
async def test_provider_discovery_generation_discards_late_prior_provider():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    openai_started = asyncio.Event()
    ollama_started = asyncio.Event()
    release_ollama = asyncio.Event()

    async def discover(provider_key):
        if provider_key == "openai":
            openai_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return ("openai-late",)
        ollama_started.set()
        await release_ollama.wait()
        return ("ollama-current",)

    scope_service = MagicMock()

    async def discover_models(*, provider, **_kwargs):
        return _typed_model_discovery_result(provider, f"{provider}-scope")

    scope_service.discover_models = AsyncMock(side_effect=discover_models)
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config=_provider_endpoint_config("openai", "ollama"),
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=discover)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await asyncio.wait_for(openai_started.wait(), timeout=2)
        step.select_provider("ollama")
        await asyncio.wait_for(ollama_started.wait(), timeout=2)
        await pilot.pause()

        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert "openai" not in status.casefold()
        assert step.selected_provider_key == "ollama"

        release_ollama.set()
        await pilot.pause(0.1)
        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert "ollama" in status.casefold()
        assert "openai" not in status.casefold()
        assert list(step._selected_provider_models.values()) == [("ollama-scope",)]
        [identity] = step._selected_provider_models
        assert type(identity) is FirstRunModelDiscoveryKey
        assert identity.provider_key == "ollama"
        wizard.staged_provider_draft = step._effective_provider_draft()

    wizard.wizard_data = {
        "provider": {"provider_key": "ollama", "provider_value": "ollama"}
    }
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        build_first_run_model_discovery_key,
    )

    assert wizard._first_run_selected_provider_models == {identity: ("ollama-scope",)}
    assert build_first_run_model_discovery_key(wizard.staged_provider_draft) == identity
    model_step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=None,
    )
    model_host = _StepHost(model_step)
    async with model_host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        model_step.on_show()
        await pilot.pause(0.1)
        model_ids = [
            str(getattr(button, "_model_id", button.label))
            for button in model_step.query_one("#setup-model-choice", RadioSet).query(
                RadioButton
            )
        ]
        assert model_ids == ["ollama-scope"]

    assert [
        call.kwargs["provider"]
        for call in scope_service.discover_models.await_args_list
    ] == ["ollama"]


@pytest.mark.asyncio
async def test_provider_discovery_same_selection_idempotent_and_retry_adds_one():
    from unittest.mock import AsyncMock

    selected_discovery = AsyncMock(return_value=())
    step = _provider_step(discover=selected_discovery)
    step.wizard.app_instance.app_config = _provider_endpoint_config("ollama")
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("ollama")
        await pilot.pause()
        step.select_provider("Ollama")
        await pilot.pause()
        assert selected_discovery.await_count == 1

        step._begin_selected_provider_discovery("ollama")
        await pilot.pause()
        assert selected_discovery.await_count == 2


@pytest.mark.asyncio
async def test_provider_discovery_reentry_restarts_cancelled_request_and_discards_late():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    attempt = 0

    async def discover(_provider_key):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                return ("late-private-value",)
        second_started.set()
        await release_second.wait()
        return ("current-provider-result",)

    selected_discovery = AsyncMock(side_effect=discover)
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("ollama", "current-model")
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config=_provider_endpoint_config("ollama"),
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=selected_discovery)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("ollama")
        await asyncio.wait_for(first_started.wait(), timeout=2)
        status_widget = step.query_one("#setup-provider-probe-status", Static)
        assert str(status_widget.renderable).startswith("Checking")

        step.on_hide()
        await asyncio.wait_for(first_cancelled.wait(), timeout=2)
        assert step._selected_discovery_state == "cancelled"
        assert "Checking" not in str(status_widget.renderable)

        step.on_show()
        await asyncio.wait_for(second_started.wait(), timeout=2)
        assert step._selected_discovery_state == "in_progress"
        assert selected_discovery.await_count == 2

        release_second.set()
        await pilot.pause(0.1)

        assert step._selected_discovery_state == "complete"
        assert list(step._selected_provider_models.values()) == [("current-model",)]
        [identity] = step._selected_provider_models
        assert type(identity) is FirstRunModelDiscoveryKey
        assert identity.provider_key == "ollama"
        assert "late-private-value" not in str(status_widget.renderable)
        scope_service.discover_models.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_discovery_complete_then_reentry_does_not_request_again():
    from unittest.mock import AsyncMock

    selected_discovery = AsyncMock(return_value=())
    step = _provider_step(discover=selected_discovery)
    step.wizard.app_instance.app_config = _provider_endpoint_config("ollama")
    step.wizard.app_instance.llm_provider_catalog_scope_service = None
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("ollama")
        await pilot.pause(0.1)
        assert step._selected_discovery_state == "complete"

        step.on_hide()
        step.on_show()
        await pilot.pause()

        assert selected_discovery.await_count == 1
        assert step._selected_discovery_state == "complete"


@pytest.mark.asyncio
async def test_provider_discovery_uses_exact_draft_settings_and_secret_free_key():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "exact-model")
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "api_settings": {
                    "custom": {
                        "api_url": "https://exact.test/proxy/v1/chat/completions",
                        "api_key_env_var": "CUSTOM_API_KEY",
                    }
                }
            },
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=AsyncMock(return_value=()))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        await pilot.pause(0.1)

        call = scope_service.discover_models.await_args
        assert call.kwargs["provider"] == "custom"
        assert call.kwargs["staged_settings"] == {
            "api_settings": {
                "custom": {
                    "api_url": "https://exact.test/proxy/v1/chat/completions",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        }
        [(identity, models)] = step._selected_provider_models.items()
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            FirstRunModelDiscoveryKey,
        )

        assert type(identity) is FirstRunModelDiscoveryKey
        assert identity.connection_identity == (
            "custom",
            "https://exact.test/proxy/v1/chat/completions",
        )
        assert identity.credential_source == "environment"
        assert models == ("exact-model",)
        assert "CUSTOM_API_KEY" not in repr(identity)


@pytest.mark.asyncio
async def test_open_wizard_uses_rotated_environment_key_for_next_test(monkeypatch):
    import os
    from unittest.mock import AsyncMock

    first_secret = "open-wizard-env-secret-a"
    rotated_secret = "open-wizard-env-secret-b"
    monkeypatch.setenv("OPENAI_API_KEY", first_secret)
    probe = AsyncMock(return_value=_reachable_endpoint_outcome("env-model"))
    step = _provider_step(
        probe=probe,
        environ=os.environ,
        discover=AsyncMock(return_value=()),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        first_identity = step._last_tested_provider_identity
        assert first_identity is not None
        assert probe.await_args.kwargs["credential_value"] == first_secret

        monkeypatch.setenv("OPENAI_API_KEY", rotated_secret)
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        assert probe.await_count == 2
        assert probe.await_args.kwargs["credential_value"] == rotated_secret
        rotated_identity = step._last_tested_provider_identity
        assert rotated_identity is not None
        assert rotated_identity.credential_source == "environment"
        assert rotated_identity.credential_revision > first_identity.credential_revision
        assert step._provider_evidence_store().evidence_for(first_identity) is None
        for rendered in (
            repr(first_identity),
            repr(rotated_identity),
            repr(step._provider_evidence_store()),
            app.export_screenshot(),
        ):
            assert first_secret not in rendered
            assert rotated_secret not in rendered


@pytest.mark.asyncio
async def test_open_wizard_restarts_discovery_for_rotated_environment_key(monkeypatch):
    import os
    from unittest.mock import AsyncMock

    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    first_secret = "discovery-env-secret-a"
    rotated_secret = "discovery-env-secret-b"
    monkeypatch.setenv("OPENAI_API_KEY", first_secret)
    seen_keys: list[str | None] = []

    async def discover_models(*, provider, staged_settings, **_kwargs):
        seen_keys.append(
            get_provider_readiness(
                provider, staged_settings, environ=os.environ
            ).api_key
        )
        return _typed_model_discovery_result("openai", f"model-{len(seen_keys)}")

    scope_service = MagicMock(discover_models=AsyncMock(side_effect=discover_models))
    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config={}, llm_provider_catalog_scope_service=scope_service
    )
    wizard.stage_provider_setup = MagicMock(return_value=True)
    wizard.rerun = False
    step = _provider_step(
        wizard=wizard,
        environ=os.environ,
        discover=AsyncMock(return_value=()),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause(0.1)
        first_key = step._selected_discovery_key
        assert first_key is not None
        assert seen_keys == [first_secret]

        step.on_hide()
        monkeypatch.setenv("OPENAI_API_KEY", rotated_secret)
        step.on_show()
        await pilot.pause(0.1)

        rotated_key = step._selected_discovery_key
        assert rotated_key is not None
        assert rotated_key.credential_revision > first_key.credential_revision
        assert seen_keys == [first_secret, rotated_secret]
        assert first_key not in step._selected_provider_models
        assert step._selected_provider_models == {rotated_key: ("model-2",)}
        assert first_secret not in repr(step._selected_provider_models)
        assert rotated_secret not in repr(step._selected_provider_models)


@pytest.mark.asyncio
async def test_provider_credential_revision_change_discards_late_discovery_result():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def discover_models(*, staged_settings, **_kwargs):
        provider_settings = staged_settings["api_settings"]["custom"]
        if provider_settings.get("api_key") == "replacement-secret":
            return _typed_model_discovery_result("custom", "current-model")
        first_started.set()
        try:
            await release_first.wait()
        except asyncio.CancelledError:
            return _typed_model_discovery_result("custom", "late-model")
        return _typed_model_discovery_result("custom", "late-model")

    scope_service = MagicMock(discover_models=AsyncMock(side_effect=discover_models))
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "api_settings": {
                    "custom": {"api_url": "https://exact.test/v1/chat/completions"}
                }
            },
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=AsyncMock(return_value=()))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        await asyncio.wait_for(first_started.wait(), timeout=2)

        key_input = step.query_one("#setup-provider-api-key", Input)
        key_input.value = "replacement-secret"
        await pilot.pause()
        ok, error = await step.commit()
        assert ok, error
        release_first.set()
        await pilot.pause(0.1)

        assert list(step._selected_provider_models.values()) == [("current-model",)]
        [identity] = step._selected_provider_models
        assert identity.credential_source == "draft"
        assert identity.credential_revision > 0
        assert "replacement-secret" not in repr(identity)


@pytest.mark.asyncio
async def test_provider_endpoint_change_discards_late_discovery_result():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    started = asyncio.Event()
    release = asyncio.Event()

    async def discover_models(**_kwargs):
        started.set()
        await release.wait()
        return _typed_model_discovery_result("custom", "stale-model")

    app_config = {
        "api_settings": {
            "custom": {"api_url": "https://first.test/v1/chat/completions"}
        }
    }
    scope_service = MagicMock(discover_models=AsyncMock(side_effect=discover_models))
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config=app_config,
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, discover=AsyncMock(return_value=()))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        await asyncio.wait_for(started.wait(), timeout=2)
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://replacement.test/v1/chat/completions"
        await pilot.pause()
        release.set()
        await pilot.pause(0.1)

        assert step._selected_provider_models == {}


@pytest.mark.asyncio
async def test_provider_switch_clears_adopted_llama_endpoint_before_custom_commit():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    server = DiscoveredLocalServer(
        provider_key="llama_cpp",
        base_url="http://127.0.0.1:8080",
        model_ids=("llama-model",),
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "api_settings": {
                    "custom": {"api_url": "https://custom.test/v1/chat/completions"}
                }
            },
            llm_provider_catalog_scope_service=None,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(
        wizard=wizard,
        local_discover=AsyncMock(return_value=(server,)),
        discover=AsyncMock(return_value=()),
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#setup-provider-use-detected")
        await pilot.pause()
        assert step.detected_base_url == "http://127.0.0.1:8080"

        step.select_provider("custom")
        await pilot.pause()
        assert not hasattr(step, "detected_base_url")
        ok, error = await step.commit()

        assert ok, error
        draft = _staged_provider_draft(wizard)
        assert draft.provider == "custom"
        assert draft.endpoint == "https://custom.test/v1/chat/completions"
        assert "127.0.0.1:8080" not in repr(draft)


@pytest.mark.asyncio
async def test_provider_step_blocks_unset_declared_environment_source():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "api_settings": {"openai": {"api_key_env_var": "PRIVATE_OPENAI_KEY"}}
            }
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, environ={})
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        ok, error = await step.commit()

        assert not ok
        assert "PRIVATE_OPENAI_KEY" in error
        wizard.stage_provider_setup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inline_key", "environment", "expected_source", "expected_ready"),
    [
        ("inline-secret", {}, "none", True),
        ("inline-secret", {"PRIVATE_OPENAI_KEY": "environment-secret"}, "none", True),
        (None, {}, None, False),
        (None, {"PRIVATE_OPENAI_KEY": "environment-secret"}, "environment", True),
    ],
)
async def test_provider_credential_precedence_matches_first_chat_readiness(
    inline_key, environment, expected_source, expected_ready
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    provider_settings = {"api_key_env_var": "PRIVATE_OPENAI_KEY"}
    if inline_key is not None:
        provider_settings["api_key"] = inline_key
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"api_settings": {"openai": provider_settings}},
            llm_provider_catalog_scope_service=None,
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, environ=environment)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        ok, error = await step.commit()

        assert ok is expected_ready, error
        if not expected_ready:
            assert "PRIVATE_OPENAI_KEY" in error
            wizard.stage_provider_setup.assert_not_called()
            return
        draft = _staged_provider_draft(wizard)
        assert draft.credential.source == expected_source
        assert "inline-secret" not in repr(draft)
        assert "environment-secret" not in repr(draft)


@pytest.mark.asyncio
async def test_unchanged_provider_backtrack_preserves_discovery_key_and_request_count():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "stable-model")
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "api_settings": {
                    "custom": {"api_url": "https://stable.example.test/v1"}
                }
            },
            llm_provider_catalog_scope_service=scope_service,
        ),
        note_key_entered=MagicMock(),
        rerun=False,
        staged_provider_draft=None,
    )
    staged = []

    def stage_provider_setup(draft):
        staged.append(draft)
        wizard.staged_provider_draft = draft
        return True

    wizard.stage_provider_setup = MagicMock(side_effect=stage_provider_setup)
    step = _provider_step(wizard=wizard, discover=AsyncMock(return_value=()))
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        await pilot.pause(0.1)
        ok, error = await step.commit()
        assert ok, error
        await pilot.pause(0.1)
        first_key = step._model_discovery_key(staged[-1])
        first_request_count = scope_service.discover_models.await_count

        ok, error = await step.commit()
        assert ok, error
        await pilot.pause(0.1)
        second_key = step._model_discovery_key(staged[-1])

        assert second_key == first_key
        assert staged[-1].credential.revision == staged[-2].credential.revision
        assert scope_service.discover_models.await_count == first_request_count


@pytest.mark.asyncio
async def test_provider_neutral_discovery_reentry_restarts_cancelled_scan_once():
    import asyncio
    from unittest.mock import AsyncMock

    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    attempt = 0

    async def discover_local(_app_config):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                return ()
        second_started.set()
        await release_second.wait()
        return ()

    local_discover = AsyncMock(side_effect=discover_local)
    step = _provider_step(local_discover=local_discover)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await asyncio.wait_for(first_started.wait(), timeout=2)
        step.on_hide()
        await asyncio.wait_for(first_cancelled.wait(), timeout=2)
        assert step._local_discovery_state == "cancelled"

        step.on_show()
        await asyncio.wait_for(second_started.wait(), timeout=2)
        step.on_show()
        await pilot.pause()

        assert local_discover.await_count == 2
        assert step._local_discovery_state == "in_progress"

        release_second.set()
        await pilot.pause(0.1)
        assert step._local_discovery_state == "complete"


@pytest.mark.asyncio
async def test_provider_repeated_on_show_while_visible_does_not_duplicate_scan():
    import asyncio
    from unittest.mock import AsyncMock

    started = asyncio.Event()
    release = asyncio.Event()

    async def discover_local(_app_config):
        started.set()
        await release.wait()
        return ()

    local_discover = AsyncMock(side_effect=discover_local)
    step = _provider_step(local_discover=local_discover)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        step.on_show()
        step.on_show()
        await pilot.pause()
        assert local_discover.await_count == 1

        release.set()
        await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_provider_explicit_test_cancels_selected_discovery_before_probe():
    import asyncio
    from unittest.mock import AsyncMock

    discovery_started = asyncio.Event()
    discovery_cancelled = asyncio.Event()

    async def discover(_provider_key):
        discovery_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            discovery_cancelled.set()
            return ("late-private-value",)

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("model-a"))
    step = _provider_step(
        discover=AsyncMock(side_effect=discover),
        probe=probe,
        environ={"OPENAI_API_KEY": "test-secret"},
    )
    step.wizard.app_instance.app_config = _provider_endpoint_config("openai")
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await asyncio.wait_for(discovery_started.wait(), timeout=2)

        step.query_one("#setup-provider-test", Button).press()
        await asyncio.wait_for(discovery_cancelled.wait(), timeout=2)
        await pilot.pause(0.1)

        assert step._selected_discovery_state == "cancelled"
        assert step._selected_provider_models == {}
        probe.assert_awaited_once()
        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert status.startswith("✓ ")
        assert "late-private-value" not in status


@pytest.mark.asyncio
async def test_provider_initial_mount_only_runs_provider_neutral_local_discovery():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    local_discover = AsyncMock(return_value=())
    selected_discovery = AsyncMock(return_value=())
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("openai")
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={}, llm_provider_catalog_scope_service=scope_service
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(
        wizard=wizard,
        discover=selected_discovery,
        local_discover=local_discover,
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)

    local_discover.assert_awaited_once_with({})
    selected_discovery.assert_not_awaited()
    scope_service.discover_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_explicit_test_feedback_is_preserved():
    from unittest.mock import AsyncMock

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("model-a"))
    step = _provider_step(probe=probe, environ={"OPENAI_API_KEY": "test-secret"})
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)

        probe.assert_awaited_once()
        assert probe.await_args.kwargs["provider"] == "openai"
        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert status.startswith("✓ ")
        assert "reached" in status.casefold()


@pytest.mark.asyncio
async def test_provider_down_and_space_never_selects_group_heading():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        assert all(
            option.disabled
            for index in range(choices.option_count)
            if (option := choices.get_option_at_index(index)).id.startswith("group-")
        )

        choices.focus()
        await pilot.press("down", "space", "down", "space")
        await pilot.pause()

        assert step.selected_provider_key
        assert not step.selected_provider_key.startswith("group-")


@pytest.mark.asyncio
async def test_provider_keyboard_walk_visits_only_provider_rows():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        choices.focus()

        for _ in range(choices.option_count + 3):
            await pilot.press("down")
            assert choices.highlighted is not None
            highlighted = choices.get_option_at_index(choices.highlighted)
            assert not highlighted.disabled


@pytest.mark.asyncio
async def test_provider_step_env_key_shows_found_in_environment():
    step = _provider_step(environ={"OPENAI_API_KEY": "sk-x"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        assert "environment" in str(status.render()).lower()


@pytest.mark.asyncio
async def test_provider_step_stale_probe_result_is_discarded():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        generation_before = step.probe_generation
        step.select_provider("anthropic")
        # A result stamped with the old generation must not render.
        step.apply_probe_result(generation_before, reachable=True, summary="stale ok")
        status = step.query_one("#setup-provider-probe-status", Static)
        assert "stale ok" not in str(status.render())


@pytest.mark.asyncio
async def test_provider_switch_cancels_in_flight_probe_without_late_status():
    import asyncio

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def probe(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    step = _provider_step(probe=probe)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        step.query_one(
            "#setup-provider-endpoint", Input
        ).value = "https://first.example.test/v1"
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await asyncio.wait_for(started.wait(), timeout=2)

        step.select_provider("llama_cpp")
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        await pilot.pause()

        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
        assert "Testing" not in status
        assert "first.example.test" not in status


@pytest.mark.asyncio
async def test_provider_step_commit_writes_key_and_notes_key_entered():
    from unittest.mock import AsyncMock

    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-api-key", Input).value = "sk-new"
        ok, error = await step.commit()
        assert ok, error
        draft = _staged_provider_draft(wizard)
        assert draft.provider == "openai"
        assert draft.credential.source == "draft"
        assert not hasattr(draft.credential, "value")
        wizard.note_key_entered.assert_called_once()


@pytest.mark.asyncio
async def test_provider_step_commit_recovers_user_driven_highlight():
    """Commit fallback requires prior user navigation of the provider list."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard, environ={"ANTHROPIC_API_KEY": "test-secret"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        choices.focus()
        await pilot.press("down")
        await pilot.pause()
        assert step.selected_provider_key == "anthropic"

        # Exercise fallback independently after confirmed user navigation:
        # the live highlight remains user-driven, so commit may recover it.
        step.selected_provider_key = ""

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_provider_key == "anthropic"
        assert _staged_provider_draft(wizard).provider == "anthropic"


@pytest.mark.asyncio
async def test_provider_step_untouched_mount_does_not_select_or_commit():
    """The framework's initial highlight is not a provider choice."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        assert choices.highlighted is not None
        assert step.selected_provider_key == ""

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_provider_key == ""
        wizard.commit_config.assert_not_called()
        wizard.stage_provider_setup.assert_not_called()


@pytest.mark.asyncio
async def test_provider_space_only_selects_initial_provider_and_stages():
    step = _provider_step(environ={"OPENAI_API_KEY": "test-secret"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        assert step.selected_provider_key == ""

        choices.focus()
        await pilot.press("space")
        await pilot.pause()

        assert step.selected_provider_key == "openai"
        ok, error = await step.commit()
        assert ok, error
        assert _staged_provider_draft(step.wizard).provider == "openai"


@pytest.mark.asyncio
async def test_provider_home_on_initial_row_selects_and_stages_openai():
    step = _provider_step(environ={"OPENAI_API_KEY": "sk-x"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        assert step.selected_provider_key == ""

        choices.focus()
        await pilot.press("home")
        await pilot.pause()

        assert step.selected_provider_key == "openai"
        status = step.query_one("#setup-provider-key-status", Static)
        assert "environment" in str(status.render()).lower()
        ok, error = await step.commit()
        assert ok, error
        draft = _staged_provider_draft(step.wizard)
        assert draft.provider == "openai"
        assert draft.credential.source == "environment"
        assert not hasattr(draft.credential, "value")


@pytest.mark.asyncio
async def test_provider_page_up_initial_row_preserves_openai_and_stages():
    step = _provider_step(environ={"OPENAI_API_KEY": "test-secret"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        assert step.selected_provider_key == ""

        choices.focus()
        await pilot.press("pageup")
        await pilot.pause()

        assert choices.highlighted is not None
        highlighted = choices.get_option_at_index(choices.highlighted)
        assert not highlighted.disabled
        assert getattr(highlighted, "provider_key", None) == "openai"
        assert step.selected_provider_key == "openai"

        await pilot.press("space")
        await pilot.pause()
        assert step.selected_provider_key == "openai"

        ok, error = await step.commit()
        assert ok, error
        assert _staged_provider_draft(step.wizard).provider == "openai"


def test_provider_grouping_orders_popular_then_other_nonempty_sections():
    """Mirror settings_screen.py:6423's grouping rule (task-6 brief interface)."""
    from tldw_chatbook.Chat.console_provider_support import ConsoleProviderCatalogEntry

    entries = (
        ConsoleProviderCatalogEntry(
            readiness_key="ollama",
            execution_key="ollama",
            display_name="Ollama",
            requires_api_key=False,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="local_llamacpp",
            execution_key="custom-openai-api",
            display_name="local llama.cpp",
            requires_api_key=False,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="openai",
            execution_key="openai",
            display_name="OpenAI",
            requires_api_key=True,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="anthropic",
            execution_key="anthropic",
            display_name="Anthropic",
            requires_api_key=True,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="groq",
            execution_key="groq",
            display_name="Groq",
            requires_api_key=True,
        ),
        ConsoleProviderCatalogEntry(
            readiness_key="vllm",
            execution_key="vllm",
            display_name="vLLM",
            requires_api_key=False,
        ),
    )
    # TASK-1498: flat _grouped was replaced by sectioned _grouped_sections —
    # Popular (fixed order) first, then Cloud/Local alphabetical, custom/
    # legacy alias keys under Other, empty sections dropped.
    sections = ProviderStep._grouped_sections(entries)
    titles = [title for title, _ in sections]
    assert titles == ["Popular", "Cloud", "Local", "Other"]
    popular_keys = [e.readiness_key for e in dict(sections)["Popular"]]
    assert popular_keys == ["openai", "anthropic", "ollama"]
    assert [e.readiness_key for e in dict(sections)["Cloud"]] == ["groq"]
    assert [e.readiness_key for e in dict(sections)["Local"]] == ["vllm"]
    assert [e.readiness_key for e in dict(sections)["Other"]] == ["local_llamacpp"]

    options = _provider_options(entries)
    headings = [
        option
        for option in options
        if isinstance(option, ProviderChoiceOption) and option.provider_key is None
    ]
    assert [option.id for option in headings] == [
        "group-popular",
        "group-cloud",
        "group-local",
        "group-other",
    ]
    assert all(option.disabled for option in headings)
    assert _provider_group_option_id("Custom Provider Group") == (
        "group-custom-provider-group"
    )


@pytest.mark.asyncio
async def test_provider_step_one_click_connect_adopts_discovered_server():
    """One click stages the discovered endpoint without writing config."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    server = DiscoveredLocalServer(
        provider_key="llama_cpp", base_url="http://127.0.0.1:8080", model_ids=("m1",)
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(
        wizard=wizard,
        local_discover=AsyncMock(return_value=(server,)),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        detected = step.query_one("#setup-provider-detected", Static)
        assert "127.0.0.1:8080" in str(detected.render())
        use_button = step.query_one("#setup-provider-use-detected", Button)
        assert "hidden" not in use_button.classes

        await pilot.click("#setup-provider-use-detected")
        await pilot.pause()
        assert step.selected_provider_key == "llama_cpp"
        assert step.detected_base_url == "http://127.0.0.1:8080"

        ok, error = await step.commit()
        assert ok, error
        draft = _staged_provider_draft(wizard)
        assert draft.provider == "llama_cpp"
        assert draft.endpoint == "http://127.0.0.1:8080"
        assert draft.credential.source == "none"
        wizard.note_key_entered.assert_not_called()


@pytest.mark.asyncio
async def test_provider_step_tab_from_list_reaches_endpoint_before_detection():
    """The first connection field follows the provider list in keyboard order."""
    from unittest.mock import AsyncMock

    server = DiscoveredLocalServer(
        provider_key="llama_cpp", base_url="http://127.0.0.1:8080", model_ids=("m1",)
    )
    step = _provider_step(local_discover=AsyncMock(return_value=(server,)))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        use_button = step.query_one("#setup-provider-use-detected", Button)
        assert "hidden" not in use_button.classes  # sanity: discovery landed

        choices = step.query_one("#setup-provider-choice", OptionList)
        choices.focus()
        await pilot.pause(0.1)
        assert app.focused is choices  # sanity: focus starts on the provider list

        await pilot.press("tab")
        await pilot.pause(0.1)
        endpoint_input = step.query_one("#setup-provider-endpoint", Input)
        assert app.focused is endpoint_input, (
            f"Tab from the provider list landed on {app.focused!r}, not the "
            "endpoint Input -- detection must not steal focus ahead of it"
        )

        endpoint_input.value = ""
        await pilot.press(*list("localhost9080"))
        assert endpoint_input.value == "localhost9080", (
            "typed characters after Tab must land in the endpoint Input, not be "
            "silently swallowed by a focused Button"
        )


@pytest.mark.asyncio
async def test_provider_step_masked_key_never_round_trips_configured_secret():
    """A configured (non-env) secret renders as presence only -- never a value."""
    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config={"api_settings": {"openai": {"api_key": "sk-existing-secret"}}}
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        key_input = step.query_one("#setup-provider-api-key", Input)
        assert key_input.password is True
        assert key_input.value == ""
        status = step.query_one("#setup-provider-key-status", Static)
        assert "sk-existing-secret" not in str(status.render())
        actions = step.query_one("#setup-provider-key-actions")
        assert "hidden" not in actions.classes


@pytest.mark.asyncio
async def test_provider_step_keep_preserves_existing_key_without_note():
    """Keep stages no replacement and does not trigger the protect-keys gate."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"api_settings": {"openai": {"api_key": "sk-existing"}}}
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        # Direct handler call: the full provider catalog can push these
        # buttons below the visible test-terminal region, and this test is
        # about the handler's effect, not the click hit-region.
        step._on_keep()
        await pilot.pause()
        ok, error = await step.commit()
        assert ok, error
        draft = _staged_provider_draft(wizard)
        assert draft.provider == "openai"
        assert draft.credential.source == "none"
        assert not hasattr(draft.credential, "value")
        wizard.note_key_entered.assert_not_called()


@pytest.mark.asyncio
async def test_provider_step_clear_blocks_required_provider_without_replacement():
    """Clear remains distinct from Keep and cannot stage an unusable cloud draft."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"api_settings": {"openai": {"api_key": "sk-existing"}}}
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step._on_clear()  # see comment in the Keep test above
        await pilot.pause()
        key_input = step.query_one("#setup-provider-api-key", Input)
        assert key_input.value == ""
        ok, error = await step.commit()
        assert not ok
        assert "API key" in error
        wizard.stage_provider_setup.assert_not_called()
        wizard.note_key_entered.assert_not_called()


@pytest.mark.asyncio
async def test_provider_step_switching_provider_isolates_and_restores_key_input():
    """Provider-owned credentials never leak and restore only for their owner."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-api-key", Input).value = "sk-under-openai"
        step.select_provider("anthropic")
        key_input = step.query_one("#setup-provider-api-key", Input)
        assert key_input.value == ""

        ok, error = await step.commit()
        assert not ok
        assert "API key" in error
        wizard.stage_provider_setup.assert_not_called()

        step.select_provider("openai")
        assert step.query_one("#setup-provider-api-key", Input).value == (
            "sk-under-openai"
        )


@pytest.mark.asyncio
async def test_provider_step_reselecting_same_provider_keeps_typed_key():
    """Guards the boundary of the Bug-1 fix above: the key Input must only
    be cleared on an actual provider CHANGE, not on every select_provider()
    call (e.g. a redundant re-selection of the currently-active provider)."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-api-key", Input).value = "sk-under-openai"
        step.select_provider("openai")
        key_input = step.query_one("#setup-provider-api-key", Input)
        assert key_input.value == "sk-under-openai"


@pytest.mark.asyncio
async def test_provider_step_first_selection_stages_without_writing_defaults():
    """A first selection remains memory-only until Model commits the pair."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-api-key", Input).value = "sk-new"
        ok, error = await step.commit()
        assert ok, error
        draft = _staged_provider_draft(wizard)
        assert draft.provider == "openai"
        assert draft.credential.source == "draft"
        assert not hasattr(draft.credential, "value")


@pytest.mark.asyncio
async def test_provider_step_rerun_same_provider_stages_without_changing_defaults():
    """A rerun never changes persisted defaults before Model Continue."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
                "api_settings": {"openai": {"api_key": "sk-existing"}},
            }
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step._on_keep()
        ok, error = await step.commit()
        assert ok, error
        assert _staged_provider_draft(wizard).provider == "openai"
        assert wizard.app_instance.app_config["chat_defaults"] == {
            "provider": "openai",
            "model": "gpt-4o",
        }


@pytest.mark.asyncio
async def test_provider_step_rerun_different_provider_leaves_old_pair_until_model():
    """A provider switch does not create a cross-provider partial default."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"chat_defaults": {"provider": "openai", "model": "gpt-4o"}}
        ),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("anthropic")
        step.query_one("#setup-provider-api-key", Input).value = "sk-new-anthropic"
        ok, error = await step.commit()
        assert ok, error
        draft = _staged_provider_draft(wizard)
        assert draft.provider == "anthropic"
        assert wizard.app_instance.app_config["chat_defaults"] == {
            "provider": "openai",
            "model": "gpt-4o",
        }


@pytest.mark.asyncio
async def test_provider_step_probe_budgets_cloud_vs_local(monkeypatch):
    """8.0s for a cloud key probe; 2.5s for a bare local-endpoint probe."""
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Screens import settings_endpoint_probe
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _probe_first_run_provider_connection,
    )

    probe = AsyncMock(return_value=_reachable_endpoint_outcome())
    monkeypatch.setattr(settings_endpoint_probe, "probe_settings_endpoint", probe)

    await _probe_first_run_provider_connection(
        "https://api.openai.com/v1",
        provider="openai",
        credential_source="draft",
        credential_value="sk-cloud-key",
    )
    assert probe.call_args.kwargs["timeout"] == CLOUD_PROBE_TIMEOUT_SECONDS
    assert probe.call_args.kwargs["http_client"] is not None

    await _probe_first_run_provider_connection(
        "http://127.0.0.1:8080",
        provider="llama_cpp",
        credential_source="none",
        credential_value=None,
    )
    assert probe.call_args.kwargs["timeout"] == 2.5
    assert probe.call_args.kwargs["http_client"] is None


def _model_step(wizard, discover_models=None):
    from unittest.mock import AsyncMock

    if not callable(getattr(wizard, "commit_staged_provider_setup", None)):
        wizard.commit_staged_provider_setup = AsyncMock(return_value=True)
    return ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover_models or AsyncMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_model_step_provider_change_resets_selection():
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        step.set_selected_model("gpt-5.6-terra")
        assert step.selected_model_id == "gpt-5.6-terra"
        wizard.wizard_data["provider"] = {
            "provider_key": "anthropic",
            "provider_value": "Anthropic",
        }
        step.on_show()
        assert step.selected_model_id == ""


@pytest.mark.asyncio
async def test_model_step_commit_hands_model_to_staged_provider_commit():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.set_selected_model("gpt-5.6-terra")
        ok, error = await step.commit()
        assert ok, error
        wizard.commit_staged_provider_setup.assert_awaited_once_with("gpt-5.6-terra")
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_model_step_empty_selection_commits_nothing():
    """Skip-safe: leaving the model step untouched must not touch config."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_called()
        wizard.commit_staged_provider_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_step_clearing_custom_input_clears_stale_selection():
    """Bug-5: typing then clearing the custom-model Input previously left
    selected_model_id stuck at the last typed value (Input.Changed only
    assigned when the value was non-empty) -- clearing it must reset the
    selection so a skip-safe commit doesn't silently keep committing it."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        custom_input = step.query_one("#setup-model-custom", Input)
        custom_input.value = "my-custom-model"
        await pilot.pause()
        assert step.selected_model_id == "my-custom-model"

        custom_input.value = ""
        await pilot.pause()
        assert step.selected_model_id == ""

        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_called()
        wizard.commit_staged_provider_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_step_clearing_custom_input_does_not_restore_hidden_radio():
    """A manual edit owns selection after it visibly clears the radio choice."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(
        wizard, discover_models=AsyncMock(return_value=["radio-model-a"])
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        radio_set.query_one(RadioButton).value = True
        await pilot.pause()
        assert step.selected_model_id == "radio-model-a"

        custom_input = step.query_one("#setup-model-custom", Input)
        custom_input.value = "my-custom-model"
        await pilot.pause()
        assert step.selected_model_id == "my-custom-model"
        assert radio_set.pressed_button is None

        custom_input.value = ""
        await pilot.pause()
        assert step.selected_model_id == ""


@pytest.mark.asyncio
async def test_model_step_commit_reads_pressed_radio_without_changed_event():
    """F-A regression, same pattern as ProviderStep: a RadioButton pressed
    without ever firing Changed must still be recovered at commit time."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(
        wizard, discover_models=AsyncMock(return_value=["radio-model-a"])
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        target = radio_set.query_one(RadioButton)
        radio_set._pressed_button = target
        assert step.selected_model_id == ""  # sanity: Changed truly never fired

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_model_id == "radio-model-a"
        wizard.commit_staged_provider_setup.assert_awaited_once_with("radio-model-a")
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_model_step_provider_switch_does_not_resurrect_stale_pressed_radio():
    """F1 regression: Textual's ``RadioSet._pressed_button`` is a plain
    instance attribute that ``remove_children()`` never touches (confirmed
    by reading ``textual/widgets/_radio_set.py`` -- pruning children is
    purely a DOM operation with no watcher on ``_pressed_button``).
    ``_render_models`` calls ``remove_children()``/``mount_all()`` on every
    provider switch, but the OLD, now-detached RadioButton object stays
    referenced by ``_pressed_button`` until a NEW button is pressed in the
    fresh set. Sequence: press a real radio for provider A (via ``.value =
    True``, a genuine toggle -- not manipulating ``_pressed_button``
    directly), switch to provider B via wizard_data + on_show, let the
    re-render happen, then commit with nothing pressed in B's list yet --
    the commit must NOT resurrect provider A's model."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    async def discover(provider_key):
        return {"openai": ["model-a"], "anthropic": ["model-b"]}[provider_key]

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=discover)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        radio_set.query_one(RadioButton).value = True  # real press, fires Changed
        await pilot.pause()
        assert step.selected_model_id == "model-a"  # sanity: the press registered

        wizard.wizard_data["provider"] = {
            "provider_key": "anthropic",
            "provider_value": "Anthropic",
        }
        step.on_show()
        await pilot.pause(0.1)
        ids = [
            str(getattr(b, "_model_id", b.label)) for b in radio_set.query(RadioButton)
        ]
        assert ids == ["model-b"]  # the re-render itself landed correctly

        ok, error = await step.commit()
        assert ok, error
        assert step._effective_model_id() != "model-a"
        wizard.commit_config.assert_not_called()  # skip-safe: nothing pressed in B's list
        wizard.commit_staged_provider_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_step_no_provider_shows_pick_a_provider_copy():
    """F-F regression: with no provider chosen yet, on_show must not leave
    the initial "(loading models...)" placeholder forever -- there is
    nothing to discover against, so the old code's ``if provider_key:``
    guard just skipped the load entirely and the placeholder never got
    replaced."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={},  # no "provider" entry at all -- provider_key is ""
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=AsyncMock(return_value=[]))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        labels = [str(b.label) for b in radio_set.query(RadioButton)]
        assert "(loading models…)" not in labels
        assert any("pick a provider" in label.lower() for label in labels)


@pytest.mark.asyncio
async def test_model_step_curated_fallback_bridges_raw_provider_key(monkeypatch):
    """Task-6/7 finding: ProviderStep persists chat_defaults.provider as the
    RAW provider_key (e.g. "openai"), but config.toml's curated [providers]
    table is keyed by display name (e.g. "OpenAI"). A naive
    ``catalog.get(provider_value)`` would silently return [] for the raw-key
    form even though a matching curated entry exists -- the fallback must
    bridge key forms regardless of which form ProviderStep handed it."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {"OpenAI": ["gpt-curated-1", "gpt-curated-2"]},
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        # provider_value in the RAW form ProviderStep actually persists.
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "openai"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=AsyncMock(return_value=[]))
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        ids = [
            str(getattr(button, "_model_id", button.label))
            for button in radio_set.query(RadioButton)
        ]
        assert ids == ["gpt-curated-1", "gpt-curated-2"]


@pytest.mark.asyncio
async def test_model_step_uses_scope_service_when_available():
    """The scope-service path (no injected discover_models) renders whatever
    the service reports on a "success" result -- mirrors
    settings_screen.py:7079's call shape."""
    from unittest.mock import AsyncMock, MagicMock as Mock
    from types import SimpleNamespace

    scope_result = _typed_model_discovery_result("openai", "svc-model-a", "svc-model-b")
    scope_service = Mock()
    scope_service.discover_models = AsyncMock(return_value=scope_result)
    app_instance = MagicMock(app_config={})
    app_instance.llm_provider_catalog_scope_service = scope_service
    wizard = SimpleNamespace(
        app_instance=app_instance,
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=None,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.1)
        # _StepHost mounts the step directly (no hidden/visible toggling like
        # the real wizard), so Textual's own Show event fires on top of this
        # test's explicit on_show() call -- exclusive=True on the worker
        # group (like ProviderStep._start_discovery) means only the shape of
        # the *last* call matters here, not the exact invocation count.
        assert scope_service.discover_models.await_args.kwargs == {
            "mode": "local",
            "provider": "openai",
            "staged_settings": None,
        }
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        ids = [
            str(getattr(button, "_model_id", button.label))
            for button in radio_set.query(RadioButton)
        ]
        assert ids == ["svc-model-a", "svc-model-b"]


def test_real_discovery_result_extracts_exact_safe_unique_model_ids():
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _model_ids_from_discovery_result,
    )

    result = _typed_model_discovery_result("openai", "model-a", "model-a", "model-b")

    assert _model_ids_from_discovery_result(result) == ("model-a", "model-b")


@pytest.mark.parametrize("malformed", ["object", "subclass", "unsafe", "oversized"])
def test_real_discovery_result_rejects_malformed_or_unsafe_models(malformed):
    from dataclasses import replace

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
    )
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _model_ids_from_discovery_result,
    )

    result = _typed_model_discovery_result("openai", "safe-model")
    model = result.models[0]
    if malformed == "object":
        models = (object(),)
    elif malformed == "subclass":

        class DiscoveredModelSubclass(DiscoveredModel):
            pass

        models = (
            DiscoveredModelSubclass(
                **{field: getattr(model, field) for field in model.__dataclass_fields__}
            ),
        )
    elif malformed == "unsafe":
        models = (replace(model, model_id="unsafe\nmodel"),)
    else:
        models = (replace(model, model_id="x" * 121),)

    with pytest.raises(ValueError, match="discovery"):
        _model_ids_from_discovery_result(replace(result, models=models))


@pytest.mark.asyncio
async def test_typing_manual_model_clears_keyboard_selected_radio():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "openai"}
        },
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(
        wizard, discover_models=AsyncMock(return_value=["radio-a", "radio-b"])
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        radio_set.focus()
        await pilot.press("down", "space")
        await pilot.pause()
        assert radio_set.pressed_button is not None

        manual = step.query_one("#setup-model-custom", Input)
        manual.value = "manual-model"
        await pilot.pause()

        assert radio_set.pressed_button is None
        assert step._effective_model_id() == "manual-model"


@pytest.mark.asyncio
async def test_clicking_discovered_model_clears_visible_manual_input():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "openai"}
        },
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(
        wizard, discover_models=AsyncMock(return_value=["radio-a", "radio-b"])
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        manual = step.query_one("#setup-model-custom", Input)
        manual.value = "manual-model"
        await pilot.pause()

        await pilot.click("#setup-model-option-1")
        await pilot.pause()

        assert manual.value == ""
        assert step._effective_model_id() == "radio-b"


@pytest.mark.asyncio
async def test_model_step_discovery_timeout_falls_back_to_curated(monkeypatch):
    """Behavior spec: an 8s guard on model discovery -- a slow/hanging
    discover() must not block the step forever; it degrades to the curated
    fallback instead of hanging Next indefinitely."""
    import asyncio as asyncio_module
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    import tldw_chatbook.config as config_module
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(wizard_module, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {"OpenAI": ["fallback-model"]},
    )

    async def _hangs(_provider_key):
        await asyncio_module.sleep(1.0)
        return ["too-slow-model"]

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard, discover_models=_hangs)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.3)
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        ids = [
            str(getattr(button, "_model_id", button.label))
            for button in radio_set.query(RadioButton)
        ]
        assert ids == ["fallback-model"]


def test_model_step_worker_group_is_not_wizard_advance():
    """Parked Task-5 finding: "setup-wizard-advance" is reserved for the
    container's own commit-on-Next worker; a step reusing it would race or
    duplicate with that worker."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    calls = []

    def _fake_run_worker(work, **kwargs):
        close = getattr(work, "close", None)
        if callable(close):
            close()
        calls.append(kwargs)

    step.run_worker = _fake_run_worker
    step.query_one = MagicMock(side_effect=Exception("not mounted"))
    step.on_show()
    assert calls, "expected on_show to schedule a model-load worker"
    assert calls[0]["group"] == "setup-model-load"


@pytest.mark.asyncio
async def test_rag_step_missing_deps_shows_install_copy_and_commits_nothing():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="RAG", step_number=4),
        deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        body = str(step.query_one("#setup-rag-status", Static).render())
        assert "tldw_chatbook[embeddings_rag]" in body
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_rag_step_commit_reads_pressed_radio_without_changed_event():
    """F-A regression, same pattern as ProviderStep/ModelStep, applied to
    RagStep's embedding-model RadioSet."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"embedding_config": {"models": {"embed-a": {}, "embed-b": {}}}}
        ),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="RAG", step_number=4),
        deps_installed=lambda: True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-rag-model-choice", RadioSet)
        target = step.query_one(RadioButton)
        radio_set._pressed_button = target
        assert step.selected_embedding_model == ""  # sanity: Changed never fired

        ok, error = await step.commit()
        assert ok, error
        assert step.selected_embedding_model == str(target.label)
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "embedding_config": {"default_model_id": str(target.label)}
        }


@pytest.mark.asyncio
async def test_tools_step_commits_only_changed_gates():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switches = list(step.query(Switch))
        assert switches, "tools step must render one switch per gateable tool"
        assert all(sw.value is False for sw in switches)  # default OFF
        switches[0].value = True
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed["tools"][step.gate_key_for(switches[0])] is True


@pytest.mark.asyncio
async def test_tools_step_fresh_config_no_changes_commits_nothing():
    """Pin the no-op: on a fresh config every switch starts and stays False,
    so the delta-aware commit added by the final-review fix wave must not regress the
    original "commits nothing when nothing changed" behavior."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_tools_step_on_to_off_transition_writes_false():
    """Re-run prefills a previously-enabled gate ON; turning it back off in
    the UI must persist False, not silently no-op (final-review finding 3)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"tools": {"read_file_enabled": True}}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switch = step.query_one("#setup-tool-read_file", Switch)
        assert switch.value is True  # prefilled ON from config
        switch.value = False  # user turns it back off
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"tools": {"read_file_enabled": False}}


@pytest.mark.asyncio
async def test_notes_step_commit_writes_directory_and_toggle():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-notes-enable", Switch).value = True
        step.query_one("#setup-notes-directory", Input).value = "~/MyNotes"
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "notes": {"sync_directory": "~/MyNotes", "auto_sync_enabled": True}
        }


@pytest.mark.asyncio
async def test_notes_step_disabled_commits_nothing():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_notes_step_enabled_to_disabled_writes_auto_sync_false():
    """Re-run prefills the toggle ON from a previously-enabled sync;
    turning it off must persist auto_sync_enabled=False while leaving
    sync_directory untouched (final-review finding 3)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "notes": {"sync_directory": "~/Notes", "auto_sync_enabled": True}
            }
        ),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switch = step.query_one("#setup-notes-enable", Switch)
        assert switch.value is True  # prefilled ON from config
        switch.value = False  # user disables sync
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"notes": {"auto_sync_enabled": False}}


@pytest.mark.asyncio
async def test_protect_keys_enables_encryption_via_injected_callable():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    calls = []
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: calls.append(pw) or True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok = await step.apply_password("hunter2-long-password")
        assert ok is True
        assert calls == ["hunter2-long-password"]


@pytest.mark.asyncio
async def test_protect_keys_failure_leaves_step_skippable_with_inline_error():
    """Failure must not raise nor block Next -- keys stay plaintext, skippable."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok = await step.apply_password("hunter2-long-password")
        assert ok is False
        ok2, error = await step.commit()
        assert ok2, error  # the step itself never blocks Next


def test_protect_keys_password_worker_uses_dedicated_group_not_wizard_advance():
    """Parked Task-5 finding (deviation from the task-10 brief's pseudocode):
    "setup-wizard-advance" is the CONTAINER's own advance/finalize worker
    group. Reusing it here for the password-apply worker would let a slow
    password-hash operation race the container's own commit-on-Next worker
    (both exclusive=True on the same group cancels/blocks the other). Use a
    dedicated group instead; the config RLock inside enable_config_encryption
    is what actually serializes writes, not this group name."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: True,
    )
    calls = []

    def _fake_run_worker(coro, **kwargs):
        coro.close()
        calls.append(kwargs)

    step.run_worker = _fake_run_worker
    step._on_password_result("hunter2-long-password")
    assert calls, "expected a worker to be scheduled for the password result"
    assert calls[0]["group"] == "setup-protect-encrypt"
    assert calls[0]["group"] != "setup-wizard-advance"


@pytest.mark.asyncio
async def test_summary_step_renders_rows_from_read_back():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
        },
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        rendered = str(step.query_one("#setup-summary-rows", Static).render())
        assert "Provider" in rendered
        assert "✓" in rendered and "✗" in rendered


@pytest.mark.asyncio
async def test_summary_default_speech_check_skips_service_construction_when_store_root_absent(
    monkeypatch, tmp_path
):
    """TASK-1301 review Minor 12: a Quick-track user (who never saw the
    Speech step) reaching Summary must not cause the managed artifact
    store's directories to be created on disk just to render a row that
    will read "not set up (optional)" either way. The default
    speech_installed check must do a read-only existence check on the
    store root FIRST and skip constructing a real ModelArtifactService
    (which mkdirs) when nothing has ever been installed by anyone."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    absent_root = tmp_path / "never-created"
    monkeypatch.setattr(
        wizard_module, "managed_model_artifact_root", lambda: absent_root
    )
    probe = MagicMock()
    monkeypatch.setattr(wizard_module, "active_managed_parakeet_dir", probe)

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
        # speech_installed deliberately NOT overridden -- exercising the
        # real default this test is pinning.
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)

    probe.assert_not_called()
    assert not absent_root.exists(), "the managed store root must not be created"


@pytest.mark.asyncio
async def test_summary_default_speech_check_still_checks_when_store_root_exists(
    monkeypatch, tmp_path
):
    """The other half: once the store root legitimately exists (Library
    install, a Full-track visit to the Speech step, ...), the default
    check must still do the real check -- the mkdir-avoidance guard must
    not turn into a silent "always False"."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    existing_root = tmp_path / "already-there"
    existing_root.mkdir()
    monkeypatch.setattr(
        wizard_module, "managed_model_artifact_root", lambda: existing_root
    )
    probe = MagicMock(return_value=None)
    monkeypatch.setattr(wizard_module, "active_managed_parakeet_dir", probe)

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)

    # on_show can legitimately fire more than once in this harness (natural
    # Show event plus the explicit call above); the point being pinned is
    # "the real check still runs at all", not an exact call count.
    assert probe.call_count >= 1
    probe.assert_called_with("nemo-parakeet-tdt-0.6b-v2", "int8")


@pytest.mark.asyncio
async def test_summary_speech_runtime_check_defaults_to_the_real_onnx_asr_probe(
    monkeypatch,
):
    """Important 4 residual (re-review): SummaryStep must resolve a real
    onnx-asr runtime probe by default (same shape as rag_deps_installed/
    speech_installed) and hand it to build_summary_rows, so a config
    persisted while the extra was present but later removed reads as
    ATTENTION, not a stale CONFIGURED."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import tldw_chatbook.UI.Wizards.first_run_setup_state as setup_state_module
    import tldw_chatbook.Utils.optional_deps as optional_deps_module

    # Both patched at their SOURCE modules: SummaryStep._render_rows() does
    # function-local imports of both parakeet_onnx_deps_installed (matching
    # embeddings_rag_deps_installed's own pattern) and build_summary_rows,
    # so patching FirstRunSetupWizard's own namespace would not be observed
    # -- each local import re-resolves the current attribute on its source
    # module at call time.
    monkeypatch.setattr(
        optional_deps_module, "parakeet_onnx_deps_installed", lambda: False
    )
    captured: dict[str, object] = {}
    original = setup_state_module.build_summary_rows

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(setup_state_module, "build_summary_rows", _capture)

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "transcription": {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        },
        rag_deps_installed=lambda: False,
        speech_installed=lambda: True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        rendered = str(step.query_one("#setup-summary-rows", Static).render())

    assert captured.get("speech_runtime_installed") is False
    assert "onnx-asr" in rendered.lower() or "runtime" in rendered.lower()


@pytest.mark.asyncio
async def test_summary_speech_runtime_check_is_injectable(monkeypatch):
    """The other half: an explicitly-injected check overrides the default,
    same pattern as rag_deps_installed/speech_installed."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import tldw_chatbook.UI.Wizards.first_run_setup_state as setup_state_module

    captured: dict[str, object] = {}
    original = setup_state_module.build_summary_rows

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(setup_state_module, "build_summary_rows", _capture)

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
        speech_runtime_installed=lambda: True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)

    assert captured.get("speech_runtime_installed") is True


@pytest.mark.asyncio
async def test_summary_footer_shows_the_effective_config_path(monkeypatch, tmp_path):
    """F-D regression (UAT): the footer's "Config file:" line must show the
    REAL effective path -- resolved fresh via get_cli_config_path(), which
    honors a TLDW_CONFIG_PATH override -- not an empty value."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    scratch_config = tmp_path / "scratch-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch_config))

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        footer = str(step.query_one("#setup-summary-footer", Static).render())
        assert str(scratch_config) in footer
        assert "Config file:" in footer


@pytest.mark.asyncio
async def test_summary_quick_track_shows_defaults_note():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        note = str(step.query_one("#setup-summary-defaults-note", Static).render())
        assert "recommended defaults" in note.lower()


@pytest.mark.asyncio
async def test_summary_first_run_exit_buttons_set_expected_routes():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.Constants import TAB_HOME

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
        advance_programmatically=MagicMock(),
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert {b.id for b in step.query(Button)} == {
            "setup-exit-chat",
            "setup-exit-home",
        }
        # Direct handler call, not pilot.click(): the actions row sits below
        # what fits in this fixed 120x40 test viewport (same clipping the
        # provider-catalog tests above hit -- see _on_keep's comment), so a
        # click here actually lands on the docked WizardNavigation bar
        # instead of this button. The test is about the handler's effect.
        step._exit_home()
        await pilot.pause()
        assert step.get_step_data() == {"exit_route": TAB_HOME}
        wizard.advance_programmatically.assert_called_once()


@pytest.mark.asyncio
async def test_summary_rerun_exit_buttons_are_done_and_go_to_chat():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.Constants import TAB_CHAT

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
        wizard_data={"welcome": {"track": "quick"}},
        advance_programmatically=MagicMock(),
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert {b.id for b in step.query(Button)} == {
            "setup-exit-done",
            "setup-exit-chat",
        }
        # See the comment in the first-run-exit test above: direct handler
        # call, not pilot.click() -- the actions row is clipped below this
        # fixed test viewport.
        step._exit_chat()
        await pilot.pause()
        assert step.get_step_data() == {"exit_route": TAB_CHAT}
        wizard.advance_programmatically.assert_called_once()


@pytest.mark.asyncio
async def test_summary_exit_button_advances_the_wizard_without_an_event():
    """SummaryStep's own exit buttons must drive the SAME advance/finalize
    path as the wizard-level Next button (Summary is the last active step),
    but they have no Button.Pressed event targeting "#wizard-next" to hand
    to SetupWizardContainer.handle_next(event) -- which requires one to call
    event.prevent_default(). Exercises the real container end to end (not a
    stub wizard) so a regression back to calling handle_next() with no/None
    event would fail loudly instead of being masked by a mock.

    Reaches Summary via real "#wizard-next" clicks (that button is clear of
    the viewport), then calls the exit handler directly rather than
    pilot.click("#setup-exit-chat") -- the actions row sits below what fits
    in this fixed 120x40 viewport, same as the provider-catalog tests above
    (see _on_keep's comment): a click there actually lands on the docked
    WizardNavigation bar. This test is about advance_programmatically()'s
    wiring, not click hit-regions.
    """
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        await pilot.click("#setup-track-quick")
        await pilot.pause(0.1)
        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            step = container.steps[container.current_step]
            if isinstance(step, SummaryStep):
                step._exit_chat()
            else:
                await pilot.click("#wizard-next")
            await pilot.pause(0.2)
        from tldw_chatbook.Constants import TAB_CHAT

        assert app.wizard_result == {"completed": True, "exit_route": TAB_CHAT}


@pytest.mark.asyncio
async def test_ctrl_n_on_summary_dismisses_and_completes():
    """F-B regression (UAT): pressing ctrl+n while ON the Summary step (the
    last active step) must finish the wizard exactly like clicking its own
    exit buttons or the WizardNavigation "Finish" button does -- dismiss the
    screen and persist first_run.setup_completed.

    Reaches Summary directly via select_track + show_step (not by clicking
    through every prior step) so this test isolates ctrl+n's own dispatch
    and _advance()/complete_wizard()/_handle_complete()'s worker wiring from
    anything upstream."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        summary_index = container._step_index_for_id(STEP_SUMMARY)
        container.show_step(summary_index)
        await pilot.pause(0.2)
        assert isinstance(container.steps[container.current_step], SummaryStep)

        await pilot.press("ctrl+n")
        await pilot.pause(0.3)

        assert app.wizard_result == {"completed": True, "exit_route": None}


@pytest.mark.asyncio
async def test_ctrl_n_still_works_after_focus_was_on_a_now_hidden_widget():
    """F-B ROOT CAUSE (found via live tmux repro + diagnostic instrumentation,
    not the worker-group theory below): Textual's own focus-recovery when
    the currently-focused widget becomes hidden (Screen._reset_focus, run
    when a step's container gets `display: none` on every step change --
    BaseWizard.show_step()'s `current.add_class("hidden")`) is unreliable:
    depending on what else happens to sit in the global focus chain at that
    moment, it can land back on None, OR on some OTHER incidentally-hidden
    widget from the very step that just got hidden (observed live and
    reproduced here: with nothing else to fall back to it goes fully None;
    with an unrelated hidden sibling button present as a candidate, Textual
    quietly refocuses THAT non-interactive widget instead -- neither is a
    real focus target). Either way, a user whose last interaction was with a
    control INSIDE a step's own content (a RadioButton, an Input -- as
    opposed to the persistent WizardNavigation bar, which is never hidden)
    ends up with no RELIABLE focus target; ctrl+n/ctrl+b (bound several
    ancestors up from wherever the user last interacted) then have no
    guaranteed focus chain to resolve bindings through and can go silently
    inert -- confirmed live: a diagnostic log line inside
    advance_programmatically() fired for three consecutive successful
    ctrl+n presses and produced NOTHING on the fourth (Summary -> Finish),
    while clicking the same "Finish" button worked immediately after and
    also proved app.focused had indeed become None by then.

    Round-2 regression + fix: the FIRST cut of this fix always re-focused
    the persistent nav bar's own Next/Cancel button after every step change.
    That broke direct keyboard interaction with the new step's own content
    -- landing on Provider with focus already parked on "Next" meant
    Down/Space (which only act on a FOCUSED RadioSet) silently did nothing,
    reproducing the exact "selection doesn't commit" symptom one level up
    in the UI. The corrected fix prefers the incoming step's own first
    focusable descendant (DOM order) and falls back to the nav bar only
    when the step truly has none. Pin that exact invariant -- "not None" is
    too weak a check, since Textual's own incidental fallback can
    accidentally satisfy it without the wizard being reliably
    keyboard-navigable, and "always the nav bar" is now the wrong
    behavior."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.1)

        def _first_focusable(step):
            # Mirrors production: hidden (display:none / .hidden) widgets must
            # never be focus targets (TASK-1496/1498).
            return next(
                (
                    w
                    for w in step.walk_children(Widget)
                    if w.focusable and w.display and not w.has_class("hidden")
                ),
                None,
            )

        def _assert_focus_on_current_step_content() -> None:
            current = container.steps[container.current_step]
            expected = _first_focusable(current)
            assert expected is not None, f"{current!r} has no focusable widget"
            assert app.focused is expected, (
                f"expected focus on {current!r}'s first focusable widget "
                f"{expected!r}, got {app.focused!r}"
            )

        await pilot.press("ctrl+n")  # Welcome -> Provider
        await pilot.pause(0.2)
        _assert_focus_on_current_step_content()
        provider_step = container.steps[container.current_step]
        assert isinstance(provider_step, ProviderStep)
        choices = provider_step.query_one("#setup-provider-choice", OptionList)
        assert app.focused is choices  # the auto-focus landed here, no Tab needed

        await pilot.press("ctrl+n")  # Provider -> Model
        await pilot.pause(0.2)
        _assert_focus_on_current_step_content()

        model_step = container.steps[container.current_step]
        assert isinstance(model_step, ModelStep)
        # Simulate the live UAT sequence: the user clicks into the custom-
        # model Input specifically (overriding the RadioSet the auto-focus
        # landed on) rather than accepting a curated radio option.
        custom_input = model_step.query_one("#setup-model-custom", Input)
        custom_input.focus()
        await pilot.pause(0.1)
        assert app.focused is custom_input  # sanity: focus is inside Model's own Input

        for _ in range(10):
            if app.wizard_result != "UNSET":
                break
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            # Once the wizard has actually completed, the screen is
            # dismissed and app.focused legitimately going None reflects
            # that there is no more wizard to hold it -- only check the
            # focus invariant while the wizard is still open.
            if app.wizard_result == "UNSET":
                _assert_focus_on_current_step_content()

        assert app.wizard_result == {"completed": True, "exit_route": None}


@pytest.mark.asyncio
async def test_down_space_selects_provider_with_no_tab_presses():
    """Round-2 regression pin (live-confirmed by the controller): Down then
    Space on the Provider step, immediately after ctrl+n from Welcome, with
    NO Tab press in between, must select a provider. The first cut of the
    F-B focus fix parked focus on the nav bar's Next button after every step
    change, so Down/Space (RadioSet-only bindings) landed on the wrong
    widget and silently selected nothing -- reproducing F-A's "no provider
    commit" symptom purely through keyboard navigation, no click/OptionList
    stub involved."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.1)

        await pilot.press("ctrl+n")  # Welcome -> Provider
        await pilot.pause(0.2)
        provider_step = container.steps[container.current_step]
        assert isinstance(provider_step, ProviderStep)
        choices = provider_step.query_one("#setup-provider-choice", OptionList)
        assert app.focused is choices  # sanity: no Tab needed to reach it

        await pilot.press("down")
        await pilot.press("space")
        await pilot.pause(0.2)

        assert provider_step.selected_provider_key != ""


def test_finalize_worker_uses_a_dedicated_group_not_wizard_advance():
    """F-B fix pin: _handle_complete() runs synchronously from inside
    complete_wizard(), itself called synchronously from _advance() -- the
    body of the CURRENTLY-RUNNING "setup-wizard-advance" worker whenever the
    step being advanced past has no real await in its own commit() (true for
    SummaryStep, which never overrides SetupStep's trivial default commit).
    Scheduling _finalize into that same exclusive group asks Textual to
    cancel_group() the group it is currently executing from inside itself --
    confirmed harmless only by scheduling luck (a separately-created task
    survives regardless), not by design. Pin the dedicated group so this
    does not regress back to relying on that accident."""
    app_instance = MagicMock()
    app_instance.app_config = {}
    real_container = SetupWizardContainer(app_instance)
    calls = []

    def _fake_run_worker(coro, **kwargs):
        coro.close()  # never actually scheduled; avoid a "never awaited" warning
        calls.append(kwargs)

    real_container.run_worker = _fake_run_worker
    real_container._handle_complete({"summary": {"exit_route": None}})
    assert calls, "expected _handle_complete to schedule the finalize worker"
    assert calls[0]["group"] == "setup-wizard-finalize"
    assert calls[0]["group"] != "setup-wizard-advance"


@pytest.mark.asyncio
async def test_finalize_and_dismiss_screen_never_double_dismiss():
    """F3 hardening: a duplicate entry into _finalize/_dismiss_screen (e.g.
    a stray extra Finish click/ctrl+n racing the "setup-wizard-finalize"
    worker, or Skip-entirely arriving after Finish already completed) must
    be a clean no-op, not a second Screen.dismiss() call."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        summary_index = container._step_index_for_id(STEP_SUMMARY)
        container.show_step(summary_index)
        await pilot.pause(0.2)

        dismiss_calls = []
        wizard.dismiss = lambda result=None: dismiss_calls.append(result)

        await pilot.press("ctrl+n")
        await pilot.pause(0.3)
        assert len(dismiss_calls) == 1
        assert container._finalized is True

        # Duplicate entries via BOTH public entry points must be no-ops.
        await container._finalize(None)
        container._dismiss_screen({"completed": True, "exit_route": "duplicate"})
        assert len(dismiss_calls) == 1


@pytest.mark.asyncio
async def test_appearance_step_commits_theme_and_card():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.selected_theme = "textual-light"
        step.selected_splash_card = "matrix"
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed["general"] == {"default_theme": "textual-light"}
        assert committed["splash_screen"] == {"card_selection": "matrix"}


@pytest.mark.asyncio
async def test_appearance_step_rerun_preselects_configured_theme():
    """Added scope (Task-11 controller decision): re-run must prefill every
    step from current config. AppearanceStep previously always rendered its
    theme RadioSet with nothing pressed, even when general.default_theme was
    already set -- pre-select the RadioButton matching it, when the theme is
    in the rendered list."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"general": {"default_theme": "nord"}}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-theme-choice", RadioSet)
        pressed = radio_set.pressed_button
        assert pressed is not None
        # TASK-1500: the label carries "(current)" decoration; the clean
        # theme name rides on the button as _theme_name.
        assert getattr(pressed, "_theme_name", str(pressed.label)) == "nord"
        assert "(current)" in str(pressed.label)


@pytest.mark.asyncio
async def test_appearance_step_no_config_theme_preselects_nothing():
    """First-run behavior must stay unchanged: with no general.default_theme,
    no RadioButton is pre-pressed."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-theme-choice", RadioSet)
        assert radio_set.pressed_button is None


@pytest.mark.asyncio
async def test_appearance_step_rerun_change_only_splash_card_leaves_theme_untouched():
    """Bug-2a/b: AppearanceStep.commit() used to fall back to a hardcoded
    "textual-dark" default whenever selected_theme was empty, clobbering a
    persisted theme on a rerun that only touches the splash card. compose()
    must initialize selected_theme from the persisted default (a), and the
    delta-aware commit must omit general.default_theme when the chosen
    theme matches what's already persisted (b)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"general": {"default_theme": "nord"}}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # compose() must have initialized selected_theme from the persisted
        # default -- pin that directly, since it's the crux of fix (a).
        assert step.selected_theme == "nord"

        # Only the splash card changes this run; the theme RadioSet is left
        # untouched at its pre-selected ("nord") position.
        step.selected_splash_card = "matrix"
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert "general" not in committed
        assert committed["splash_screen"] == {"card_selection": "matrix"}


@pytest.mark.asyncio
async def test_appearance_step_surprise_me_over_persisted_card_writes_random():
    """Bug-2c: "Surprise me (random)" maps to splash_card=None, which the
    old commit() unconditionally treated as "nothing to write" -- so a
    previously persisted specific card could never be reset back to random.
    Explicitly re-picking "Surprise me" over a persisted specific card must
    write card_selection="random"."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"splash_screen": {"card_selection": "matrix"}}
        ),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one("#setup-splash-choice", RadioSet)
        buttons = list(radio_set.query(RadioButton))
        surprise_button = next(
            b for b in buttons if str(b.label).startswith("Surprise me")
        )
        other_button = next(
            b for b in buttons if not str(b.label).startswith("Surprise me")
        )
        # "Surprise me" is already the default mount-time pre-selection, and
        # RadioSet does not fire Changed for its own initial state -- press
        # a different card first, then explicitly re-press "Surprise me",
        # to mirror a real user re-picking it.
        other_button.value = True
        await pilot.pause()
        surprise_button.value = True
        await pilot.pause()

        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed["splash_screen"] == {"card_selection": "random"}


@pytest.mark.asyncio
async def test_appearance_step_fresh_run_untouched_commits_nothing():
    """Bug-2 regression guard: a truly fresh run where the user never
    touches either RadioSet must still commit nothing at all (unchanged
    skip-safe behavior), even now that selected_theme is initialized from
    prefill in compose()."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_tools_step_rerun_prefills_switches_from_config():
    """Added scope: ToolsStep previously always initialized every Switch to
    False, even on re-run with gates already enabled in config -- initialize
    each Switch from prefill.tool_gates instead. First-run behavior (no
    "tools" section, or a section with everything off) is unchanged since
    tool_gates comes back empty/False in that case."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"tools": {"read_file_enabled": True}}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        enabled_switch = step.query_one("#setup-tool-read_file", Switch)
        assert enabled_switch.value is True
        other_switches = [
            sw for sw in step.query(Switch) if sw.id != "setup-tool-read_file"
        ]
        assert other_switches, "expect more than one gateable tool"
        assert all(sw.value is False for sw in other_switches)


@pytest.mark.asyncio
async def test_model_step_rerun_prefills_when_session_provider_matches_persisted():
    """TASK-1374: the prefill fires on the REACHABLE path — the normal
    sequential walk where the session provider equals the persisted
    chat_defaults.provider (a re-run keeping the same provider). The old
    no-provider-entry guard was dead code: _advance() always records a
    provider entry before Model can be shown."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={"chat_defaults": {"provider": "openai", "model": "gpt-4o"}}
        ),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "openai"}
        },  # the entry _advance always writes — this state is reachable
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        assert step.selected_model_id == "gpt-4o"
        assert step.query_one("#setup-model-custom", Input).value == "gpt-4o"


@pytest.mark.asyncio
async def test_model_step_with_provider_entry_present_does_not_prefill_stale_model():
    """Guards the boundary of the added scope above: once a "provider" entry
    exists in wizard_data (the normal sequential path, and a real
    Back-and-switch), the existing reset-to-blank behavior must still apply
    -- the prefill path is only for the "no entry yet" case."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={"chat_defaults": {"model": "gpt-4o"}}),
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        assert step.selected_model_id == ""
        assert step.query_one("#setup-model-custom", Input).value == ""


@pytest.mark.asyncio
async def test_rerun_with_stored_plaintext_key_activates_protect_step_without_typing():
    """Bug-4: active_step_ids previously dropped STEP_PROTECT unless a
    secret was typed THIS run, so a rerun over a config that already has a
    plaintext key on disk could never reach Protect Keys without retyping a
    credential. The gate must also fire from config alone."""
    app_instance = MagicMock()
    app_instance.app_config = {"api_settings": {"openai": {"api_key": "sk-existing"}}}
    wizard = FirstRunSetupWizard(app_instance, rerun=True)
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        assert STEP_PROTECT in container.active_ids


@pytest.mark.asyncio
async def test_fresh_config_without_stored_key_omits_protect_step():
    """Regression guard for the Bug-4 fix above: a fresh config with no
    stored key and nothing typed this run must still omit STEP_PROTECT."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        assert STEP_PROTECT not in container.active_ids


class TestAppOfferGating:
    """The app hook is thin; assert the state functions drive it correctly."""

    def test_fresh_config_offers_and_upgrader_does_not(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import should_offer_wizard

        assert should_offer_wizard({}, {}) is True
        upgrader = {"api_settings": {"openai": {"api_key": "sk-x"}}}
        assert should_offer_wizard(upgrader, {}) is False

    def test_rerun_flag_reaches_container(self):
        wizard = _make_wizard(rerun=True)
        assert wizard.rerun is True


class TestCommandPaletteReentry:
    """AC #4 (task-1264): "re-runnable from Settings and the command
    palette". The Settings re-entry button is covered app-level in
    Tests/UI/test_first_run_wizard_live_contract.py; a Task 12 audit found
    NOTHING anywhere exercised SetupWizardProvider (app.py), the command
    palette's entire bridge to the wizard -- this closes that gap.
    """

    def test_run_setup_wizard_action_pushes_rerun_wizard(self):
        from tldw_chatbook.app import SetupWizardProvider
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

        screen = MagicMock()
        provider = SetupWizardProvider(screen)

        provider.handle_setup_wizard_action("run_setup_wizard")

        screen.app.push_screen.assert_called_once()
        (pushed_wizard, callback), _kwargs = screen.app.push_screen.call_args
        assert isinstance(pushed_wizard, FirstRunSetupWizard)
        assert pushed_wizard.rerun is True
        # Final-review finding 2: this push must wire the app-level result
        # callback, exactly like the Settings button and the auto-offer
        # path (app.py's _push_first_run_wizard) already do -- without it,
        # a truthy exit_route off the Summary step's "Go to Console" button is
        # silently dropped instead of navigating anywhere.
        assert callback == screen.app.handle_first_run_wizard_result

    def test_unknown_action_id_is_a_no_op(self):
        from tldw_chatbook.app import SetupWizardProvider

        screen = MagicMock()
        provider = SetupWizardProvider(screen)

        provider.handle_setup_wizard_action("something_else")

        screen.app.push_screen.assert_not_called()


class TestSetupRadioButtonStructuralState:
    """TASK-1497: selection must be distinguishable without color."""

    @pytest.mark.asyncio
    async def test_selected_and_unselected_glyphs_differ_structurally(self):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupRadioButton

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield SetupRadioButton("On option", value=True, id="on-btn")
                yield SetupRadioButton("Off option", id="off-btn")

        app = _Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            on_btn = app.query_one("#on-btn", SetupRadioButton)
            off_btn = app.query_one("#off-btn", SetupRadioButton)
            on_glyph = str(on_btn._button)
            off_glyph = str(off_btn._button)
            assert on_glyph != off_glyph, (
                "selected and unselected radios render identical button text "
                f"({on_glyph!r}) — state is color-only"
            )
            assert "●" in on_glyph and "○" in off_glyph

    @pytest.mark.asyncio
    async def test_wizard_choice_lists_use_structural_radio(self):
        """Every wizard RadioSet renders SetupRadioButton, incl. dynamic lists."""
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            SetupRadioButton,
            SetupWizardContainer,
        )

        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            plain = [
                rb
                for rb in container.query(RadioButton)
                if not isinstance(rb, SetupRadioButton)
            ]
            assert not plain, (
                f"plain RadioButtons in wizard: {[rb.id or str(rb.label) for rb in plain]}"
            )


@pytest.mark.asyncio
async def test_rag_step_missing_deps_hides_model_list_and_copy_has_no_backticks():
    """TASK-1502: no disabled model wall under the not-installed message."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import RagStep
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="RAG", step_number=4),
        deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        radio_set = step.query_one(RadioSet)
        assert not radio_set.display, "disabled model list must be hidden entirely"
        copy = str(step.query_one("#setup-rag-status", Static).render())
        assert "`" not in copy


@pytest.mark.asyncio
async def test_model_step_subtitle_display_cases_provider_and_marks_recommended():
    """TASK-1503: no raw provider keys in copy; first curated model marked."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ModelStep, SetupRadioButton
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "anthropic", "provider_value": "anthropic"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=AsyncMock(return_value=["model-alpha", "model-beta"]),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.3)
        subtitle = str(step.query_one("#setup-model-provider-line", Static).render())
        assert "anthropic" not in subtitle and "Anthropic" in subtitle
        buttons = list(step.query(SetupRadioButton))
        assert "recommended" in str(buttons[0].label)
        assert "recommended" not in str(buttons[1].label)
        # selecting the recommended row must commit the CLEAN model id
        step.set_selected_model_from_button(buttons[0])
        assert step.selected_model_id == "model-alpha"


@pytest.mark.asyncio
async def test_tools_step_rows_are_described_and_do_not_overlap():
    """TASK-1501: aligned rows with plain-language descriptions."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ToolsStep
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switches = list(step.query(Switch))
        assert switches, "tools step must render switches"
        # Every row carries a human description (not just the raw tool name).
        descs = [str(s.render()) for s in step.query(".setup-tool-desc")]
        assert len(descs) == len(switches)
        assert all(d.strip() for d in descs)
        # The original defect: switch borders collided into following rows.
        regions = sorted((sw.region.y, sw.region.bottom) for sw in switches)
        for (y1, b1), (y2, _b2) in zip(regions, regions[1:]):
            assert b1 <= y2, f"tool rows overlap: row ending {b1} vs row starting {y2}"
        # Mutating tools carry a visible warning in their description.
        write_desc = str(step.query_one("#setup-tool-desc-write_file", Static).render())
        assert "⚠" in write_desc


@pytest.mark.asyncio
async def test_progress_defaults_to_quick_track_and_titles_fit():
    """TASK-1499: Welcome anchors at the recommended 5-step count, and no
    step title exceeds the ~8-char budget the progress row can render."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer
    from tldw_chatbook.UI.Wizards.first_run_setup_state import TRACK_QUICK

    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        assert container.track == TRACK_QUICK
        assert len(container.active_ids) == 5
        for step in container.steps:
            title = step.config.title
            assert len(title) <= 8, f"step title too long for progress row: {title!r}"


@pytest.mark.asyncio
async def test_setup_progress_renders_projection_state_classes_and_dynamic_total():
    """The setup tracker mirrors the resolved active track, including
    conditional steps, without relying on the generic widget's own count."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)

        rows = list(wizard.query(".setup-progress-item"))
        assert len(rows) == len(container.active_ids) == 5
        assert [row.id for row in rows] == [
            f"setup-progress-{step_id}" for step_id in container.active_ids
        ]
        assert rows[0].has_class("-active")
        assert all(row.has_class("-upcoming") for row in rows[1:])

        container.show_step(container._step_index_for_id(STEP_PROVIDER))
        container.update_progress()
        await pilot.pause(0.1)
        rows = list(wizard.query(".setup-progress-item"))
        assert rows[0].has_class("-complete")
        assert rows[1].has_class("-active")

        container.note_key_entered()
        await pilot.pause(0.1)
        rows = list(wizard.query(".setup-progress-item"))
        assert len(rows) == len(container.active_ids) == 6
        progress_text = str(wizard.query_one("#wizard-progress", Static).render())
        assert "Step 2 of 6" in progress_text


@pytest.mark.parametrize("theme", ("textual-dark", "textual-light"))
@pytest.mark.parametrize("size", ((80, 24), (120, 40), (177, 45)))
@pytest.mark.parametrize("include_protect", (False, True))
@pytest.mark.asyncio
async def test_full_track_progress_content_stays_inside_non_overlapping_items(
    theme: str,
    size: tuple[int, int],
    include_protect: bool,
):
    wizard = _make_wizard()
    app = _StyledHostApp(wizard)
    app.theme = theme

    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_FULL)
        if include_protect:
            container.note_key_entered()
        await pilot.pause(0.1)

        rows = list(wizard.query(".setup-progress-item"))
        assert len(rows) == (11 if include_protect else 10)
        assert sum(row.has_class("-active") for row in rows) == 1
        assert all(
            row.has_class("-active")
            or row.has_class("-complete")
            or row.has_class("-upcoming")
            for row in rows
        )

        for row in rows:
            number = row.query_one(".step-number")
            title = row.query_one(".step-title")
            assert number.region.x >= row.region.x
            assert number.region.right <= row.region.right
            assert number.region.y >= row.region.y
            assert number.region.bottom <= row.region.bottom
            if title.display:
                assert title.region.x >= row.region.x
                assert title.region.right <= row.region.right
                assert title.region.y >= row.region.y
                assert title.region.bottom <= row.region.bottom
            else:
                assert wizard.query_one(".wizard-progress").has_class("-compact")

        for current, following in zip(rows, rows[1:]):
            assert current.region.right <= following.region.x


@pytest.mark.parametrize("theme", ("textual-dark", "textual-light"))
@pytest.mark.asyncio
async def test_quick_track_progress_recovers_titles_after_live_resize(theme: str):
    wizard = _make_wizard()
    app = _StyledHostApp(wizard)
    app.theme = theme

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        progress = wizard.query_one(".wizard-progress")

        def assert_compact() -> None:
            assert progress.has_class("-compact")
            assert all(not title.display for title in progress.query(".step-title"))
            assert all(
                not connector.display for connector in progress.query(".step-connector")
            )

        assert_compact()

        await pilot.resize_terminal(120, 40)
        await pilot.pause(0.2)
        assert not progress.has_class("-compact")
        rows = list(progress.query(".setup-progress-item"))
        assert len(rows) == 5
        for row in rows:
            title = row.query_one(".step-title")
            assert title.display
            assert title.region.x >= row.region.x
            assert title.region.right <= row.region.right
            assert title.region.y >= row.region.y
            assert title.region.bottom <= row.region.bottom
            connectors = list(row.query(".step-connector"))
            for connector in connectors:
                assert connector.display
                assert connector.region.x >= row.region.x
                assert connector.region.right <= row.region.right
                assert connector.region.y >= row.region.y
                assert connector.region.bottom <= row.region.bottom
        for current, following in zip(rows, rows[1:]):
            assert current.region.right <= following.region.x

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.2)
        assert_compact()


@pytest.mark.parametrize("theme", ("textual-dark", "textual-light"))
@pytest.mark.parametrize("size", ((80, 24), (120, 40), (177, 45)))
@pytest.mark.asyncio
async def test_first_run_provider_layout_stays_ordered_across_sizes_and_themes(
    theme: str, size: tuple[int, int]
):
    wizard = _make_wizard()
    app = _StyledHostApp(wizard)
    app.theme = theme

    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        assert provider_index is not None
        container.show_step(provider_index)
        container.update_progress()
        await pilot.pause(0.1)

        progress = wizard.query_one(".wizard-progress")
        steps = wizard.query_one(".wizard-steps-container")
        choices = wizard.query_one("#setup-provider-choice", OptionList)
        navigation = wizard.query_one(".wizard-navigation")
        hints = wizard.query_one("#setup-key-hints", Static)
        nav_buttons = tuple(navigation.query(Button))

        assert progress.region.bottom <= steps.region.y
        assert steps.region.bottom <= navigation.region.y
        assert navigation.region.bottom <= hints.region.y
        fixed_widgets = (progress, navigation, hints, *nav_buttons)
        for widget in fixed_widgets:
            assert widget.region.width > 0 and widget.region.height > 0
            assert widget.region.x >= 0 and widget.region.y >= 0
            assert widget.region.right <= size[0]
            assert widget.region.bottom <= size[1]
            assert widget in app.screen._compositor.visible_widgets, (
                f"{widget!r} was clipped at {size}/{theme}; region={widget.region}, "
                f"parent={widget.parent.region if widget.parent else None}, "
                f"steps={steps.region}"
            )

        assert 5 <= choices.region.height <= 7
        assert choices.region.right <= size[0]
        if size[0] >= 120:
            assert choices.region.bottom <= size[1]
            assert choices in app.screen._compositor.visible_widgets
        else:
            provider_step = container.steps[provider_index]
            assert provider_step.styles.overflow_y == "auto"


class TestThemePickerShortlist:
    """TASK-1500: curated shortlist, current marker, preview + revert."""

    def _appearance_step(self, app_config=None):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import AppearanceStep
        from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig

        wizard = SimpleNamespace(
            app_instance=MagicMock(app_config=app_config or {}),
            commit_config=AsyncMock(return_value=True),
            rerun=False,
        )
        return AppearanceStep(
            wizard=wizard,
            config=WizardStepConfig(id="appearance", title="Style", step_number=7),
        )

    @pytest.mark.asyncio
    async def test_shortlist_then_show_all_expands(self):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupRadioButton

        step = self._appearance_step()
        app = _StepHost(step)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            radio_set = step.query_one("#setup-theme-choice", RadioSet)
            short = list(radio_set.query(SetupRadioButton))
            assert len(short) <= 6, "shortlist must be curated, not the full wall"
            names = [getattr(b, "_theme_name", str(b.label)) for b in short]
            assert "textual-dark" in names and "textual-light" in names
            await pilot.pause()
            step.query_one("#setup-theme-show-all", Button).press()
            await pilot.pause(0.2)
            full = list(radio_set.query(SetupRadioButton))
            assert len(full) >= len(step._theme_names())
            assert not step.query_one("#setup-theme-show-all", Button).display

    @pytest.mark.asyncio
    async def test_current_theme_marked_and_clean_value_selected(self):
        step = self._appearance_step(
            app_config={"general": {"default_theme": "textual-light"}}
        )
        app = _StepHost(step)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            radio_set = step.query_one("#setup-theme-choice", RadioSet)
            current = [
                b for b in radio_set.query(RadioButton) if "(current)" in str(b.label)
            ]
            assert len(current) == 1
            assert getattr(current[0], "_theme_name") == "textual-light"
            assert step.selected_theme == "textual-light"

    @pytest.mark.asyncio
    async def test_selection_previews_and_revert_restores(self):
        step = self._appearance_step()
        app = _StepHost(step)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            original = str(app.theme)
            target = next(
                b
                for b in step.query_one("#setup-theme-choice", RadioSet).query(
                    RadioButton
                )
                if getattr(b, "_theme_name", "") not in ("", original)
            )
            target.value = True
            await pilot.pause()
            assert str(app.theme) == getattr(target, "_theme_name")
            step.revert_preview()
            assert str(app.theme) == original


@pytest.mark.asyncio
async def test_provider_list_grouped_popular_first_with_pinned_discovery():
    """TASK-1498: section headers, popular-first order, banner above list."""
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        choices = step.query_one("#setup-provider-choice", OptionList)
        options = [
            choices.get_option_at_index(index) for index in range(choices.option_count)
        ]
        headers = [str(option.prompt) for option in options if option.disabled]
        assert headers[0] == "Popular"
        assert "Cloud" in headers and "Local" in headers
        first_keys = [
            option.provider_key
            for option in options
            if getattr(option, "provider_key", None) is not None
        ][:4]
        assert first_keys[0] == "openai"
        assert "anthropic" in first_keys
        # Provider rows still function with disabled headers interleaved.
        choices.focus()
        await pilot.press("down")
        await pilot.pause()
        assert step.selected_provider_key == "anthropic"
        # The discovery banner sits ABOVE the list in DOM order.
        banner = step.query_one("#setup-provider-detected")
        siblings = list(banner.parent.children)
        assert siblings.index(banner) < siblings.index(choices)


@pytest.mark.asyncio
async def test_key_hints_footer_and_test_button_probe():
    """TASK-1505/1506: hints line renders; Test fires the injected probe."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProviderStep
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig

    # Footer: rendered by the real wizard screen.
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        hints = wizard.query_one("#setup-key-hints", Static)
        text = str(hints.render())
        assert "Ctrl+N" in text and "Esc" in text
        assert hints in app.screen._compositor.visible_widgets
        # The docked hints line must not push the nav bar's buttons off
        # screen (container yields a row via height:1fr).
        next_button = wizard.query_one("#wizard-next", Button)
        assert next_button in app.screen._compositor.visible_widgets
        # TASK-1499: the INITIAL progress render honors the quick default.
        from tldw_chatbook.UI.Wizards.BaseWizard import WizardProgress

        progress = wizard.query_one(WizardProgress)
        assert progress.total_steps == 5

    # Test button: fires the probe with the typed key.
    probe = AsyncMock()
    step_wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ProviderStep(
        wizard=step_wizard,
        config=WizardStepConfig(id="provider", title="Provider", step_number=2),
        discover=AsyncMock(return_value=()),
        probe=probe,
        environ={},
    )
    host = _StepHost(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-api-key", Input).value = "wizard-test-key-x"
        await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.3)
        assert probe.await_count >= 1


@pytest.mark.asyncio
async def test_provider_reentry_with_visible_discovery_button_focuses_list():
    """Review finding: after discovery unhides the pinned button, re-entering
    Provider must still focus the OptionList, not the earlier-in-DOM button."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        SetupWizardContainer,
    )
    from tldw_chatbook.UI.Wizards.first_run_setup_state import TRACK_QUICK

    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        await pilot.pause(0.1)
        await pilot.press("ctrl+n")  # Welcome -> Provider
        await pilot.pause(0.2)
        provider_step = container.steps[container.current_step]
        # Simulate discovery having found a server: banner + button visible.
        provider_step.query_one("#setup-provider-detected").remove_class("hidden")
        provider_step.query_one("#setup-provider-use-detected", Button).remove_class(
            "hidden"
        )
        await pilot.pause(0.1)
        await pilot.press("ctrl+n")  # Provider -> Model
        await pilot.pause(0.2)
        await pilot.press("ctrl+b")  # back to Provider (re-entry)
        await pilot.pause(0.2)
        choices = provider_step.query_one("#setup-provider-choice", OptionList)
        assert app.focused is choices, f"focus stole by {app.focused!r}"


class TestComposeCrashPolicy:
    """Required compose failures recover in place; optional failures skip."""

    def test_failure_contract_is_bounded_and_skip_state_exists_before_mount(self):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            SetupStepFailure,
            SetupWizardContainer,
        )

        app_instance = MagicMock()
        app_instance.app_config = {}
        container = SetupWizardContainer(app_instance)
        assert container.skipped_step_reasons == {}
        with pytest.raises(ValueError):
            SetupStepFailure(
                step_id=STEP_PROVIDER,
                required=True,
                reason_code="sensitive raw exception",
            )

    @pytest.mark.parametrize(
        ("step_id", "expected_category"),
        (
            (STEP_WELCOME, "diagnostics"),
            (STEP_PROVIDER, "providers-models"),
            (STEP_MODEL, "providers-models"),
            (STEP_VOICE, "speech-tts"),
            (STEP_SPEECH, "speech-tts"),
            (STEP_TOOLS, "advanced-config"),
            (STEP_NOTES, "advanced-config"),
            (STEP_APPEARANCE, "appearance"),
            (STEP_PROTECT, "privacy-security"),
            (STEP_SUMMARY, "diagnostics"),
        ),
    )
    def test_required_manual_route_mapping_is_exhaustive_and_actionable(
        self, step_id, expected_category
    ):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES,
            manual_settings_context_for_required_step,
        )

        app_instance = MagicMock()
        app_instance.app_config = {}
        container = SetupWizardContainer(app_instance)
        required_step_ids = {
            step.config.id
            for step in container.steps
            if isinstance(step, SetupStep) and step.required and step.config
        }

        assert set(REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES) == required_step_ids
        assert manual_settings_context_for_required_step(step_id) == {
            "category": expected_category
        }

    def test_unknown_required_failure_disables_manual_action(self):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            manual_settings_context_for_required_step,
        )

        step = SetupStep(
            wizard=MagicMock(),
            config=WizardStepConfig(
                id="forged-required-step",
                title="Forged",
                step_number=99,
                can_skip=False,
            ),
        )
        failure = SetupStepFailure(
            step_id="forged-required-step",
            required=True,
            reason_code="compose_failed",
        )
        actions = step._failure_widgets(failure)[1]
        manual_button = next(
            child
            for child in actions._pending_children
            if child.id == "setup-step-manual"
        )

        assert manual_settings_context_for_required_step(failure.step_id) is None
        assert manual_button.disabled is True

    @pytest.mark.asyncio
    async def test_required_provider_compose_failure_stays_active_and_blocks_next(
        self, monkeypatch
    ):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            STEP_PROVIDER,
        )

        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            failed_step = next(
                step
                for step in container.steps
                if step.config and step.config.id == STEP_PROVIDER
            )
            assert failed_step.required is True
            assert failed_step.compose_failure.required is True
            assert failed_step.compose_failure.reason_code == "compose_failed"
            assert STEP_PROVIDER in container.active_ids

            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            assert container.steps[container.current_step] is failed_step
            assert len(failed_step.query("#setup-step-retry")) == 1
            assert len(failed_step.query("#setup-step-manual")) == 1
            assert len(failed_step.query("#setup-step-later")) == 1

            wizard.query_one("#wizard-next", Button).press()
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            assert container.steps[container.current_step] is failed_step

    @pytest.mark.asyncio
    async def test_optional_step_failure_is_removed_and_reported_in_summary(
        self, monkeypatch
    ):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            RagStep,
            SetupWizardContainer,
            SummaryStep,
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import TRACK_FULL

        monkeypatch.setattr(RagStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            container.select_track(TRACK_FULL)
            await pilot.pause(0.1)
            failed_step = next(
                step
                for step in container.steps
                if step.config and step.config.id == STEP_RAG
            )
            assert failed_step.required is False
            assert STEP_RAG not in container.active_ids
            assert container.skipped_step_reasons == {STEP_RAG: "compose_failed"}

            summary = next(
                step for step in container.steps if isinstance(step, SummaryStep)
            )
            container.show_step(container.steps.index(summary))
            await pilot.pause(0.4)
            rendered = str(summary.query_one("#setup-summary-rows", Static).render())
            assert "RAG" in rendered
            assert "couldn't be shown" in rendered

    @pytest.mark.asyncio
    async def test_retry_reconstructs_only_failed_step_and_restores_focus(
        self, monkeypatch
    ):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )

        original_compose = ProviderStep.compose_step
        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            failed_step = container.steps[container.current_step]
            unchanged_steps = {
                step.config.id: step
                for step in container.steps
                if step.config and step is not failed_step
            }

            monkeypatch.setattr(ProviderStep, "compose_step", original_compose)
            failed_step.query_one("#setup-step-retry", Button).press()
            await pilot.pause(0.6)

            replacement = container.steps[container.current_step]
            assert replacement is not failed_step
            assert failed_step.parent is None
            assert failed_step not in list(container.walk_children(Widget))
            assert replacement.config.id == STEP_PROVIDER
            assert replacement.compose_failure is None
            assert len(container.query(ProviderStep)) == 1
            assert all(
                next(step for step in container.steps if step.config.id == step_id)
                is original
                for step_id, original in unchanged_steps.items()
            )
            assert app.focused is replacement.query_one(
                "#setup-provider-choice", OptionList
            )

    @pytest.mark.asyncio
    async def test_repeated_retry_failure_has_one_recovery_surface(self, monkeypatch):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )

        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.3)

            retired = []
            for _ in range(3):
                failed_step = container.steps[container.current_step]
                retired.append(failed_step)
                failed_step.query_one("#setup-step-retry", Button).press()
                await pilot.pause(0.3)
                assert len(container.query(ProviderStep)) == 1
                assert len(container.query("#setup-step-retry")) == 1
                assert len(container.query("#setup-step-manual")) == 1
                assert len(container.query("#setup-step-later")) == 1

            assert all(step.parent is None for step in retired)
            assert container._failure_action_running is False

    @pytest.mark.parametrize("failure_stage", ("mount", "refresh", "show"))
    @pytest.mark.asyncio
    async def test_partial_retry_failure_rolls_back_and_next_retry_succeeds(
        self, monkeypatch, failure_stage
    ):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProviderStep

        original_compose = ProviderStep.compose_step
        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            failed_index = container.current_step
            failed_step = container.steps[failed_index]
            parent = failed_step.parent
            assert parent is not None
            injected = False

            if failure_stage == "mount":
                original_mount = parent.mount

                def fail_replacement_mount(*widgets, **kwargs):
                    nonlocal injected
                    if not injected and any(
                        isinstance(widget, ProviderStep) and widget is not failed_step
                        for widget in widgets
                    ):
                        injected = True
                        raise RuntimeError("bounded replacement mount failure")
                    return original_mount(*widgets, **kwargs)

                monkeypatch.setattr(parent, "mount", fail_replacement_mount)
            elif failure_stage == "refresh":
                original_refresh = container._refresh_active_ids

                def fail_refresh_once():
                    nonlocal injected
                    if not injected:
                        injected = True
                        raise RuntimeError("bounded replacement refresh failure")
                    return original_refresh()

                monkeypatch.setattr(container, "_refresh_active_ids", fail_refresh_once)
            else:
                original_show = container.show_step

                def fail_show_once(step_index):
                    nonlocal injected
                    if not injected:
                        injected = True
                        raise RuntimeError("bounded replacement show failure")
                    return original_show(step_index)

                monkeypatch.setattr(container, "show_step", fail_show_once)

            monkeypatch.setattr(ProviderStep, "compose_step", original_compose)
            failed_step.query_one("#setup-step-retry", Button).press()
            await pilot.pause(0.8)

            assert injected is True
            assert container._failure_action_running is False
            assert container._failure_action is None
            recovery_step = container.steps[failed_index]
            assert recovery_step is not failed_step
            assert recovery_step.compose_failure is failed_step.compose_failure
            assert recovery_step.parent is parent
            assert failed_step.parent is None
            assert len(container.query(ProviderStep)) == 1
            assert len(container.query("#setup-step-retry")) == 1
            assert len(container.query("#setup-step-manual")) == 1
            assert len(container.query("#setup-step-later")) == 1
            assert wizard.query_one("#setup-step-retry", Button).disabled is False
            assert wizard.query_one("#setup-step-manual", Button).disabled is False
            assert wizard.query_one("#setup-step-later", Button).disabled is False
            assert not any(
                worker.group == "setup-step-recovery" and not worker.is_finished
                for worker in app.workers
            )

            recovery_step.query_one("#setup-step-retry", Button).press()
            await pilot.pause(0.8)

            replacement = container.steps[failed_index]
            assert replacement is not recovery_step
            assert replacement.compose_failure is None
            assert recovery_step.parent is None
            assert len(container.query(ProviderStep)) == 1
            assert container._failure_action_running is False
            assert container._failure_action is None

    @pytest.mark.parametrize(
        ("action_selector", "outcome"),
        (
            ("#setup-step-retry", "retry"),
            ("#setup-step-manual", "manual"),
            ("#setup-step-later", "later"),
        ),
    )
    @pytest.mark.asyncio
    async def test_failure_action_fences_navigation_and_duplicate_actions(
        self, monkeypatch, action_selector, outcome
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProviderStep

        original_compose = ProviderStep.compose_step
        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            failed_step = container.steps[container.current_step]
            failed_index = container.current_step
            started = asyncio.Event()
            release = asyncio.Event()

            if outcome == "retry":
                original_remove = failed_step.remove
                remove_calls = 0

                async def blocked_remove():
                    nonlocal remove_calls
                    remove_calls += 1
                    started.set()
                    await release.wait()
                    await original_remove()

                monkeypatch.setattr(failed_step, "remove", blocked_remove)
                monkeypatch.setattr(ProviderStep, "compose_step", original_compose)
            else:

                async def blocked_checkpoint():
                    started.set()
                    await release.wait()
                    return True

                checkpoint = AsyncMock(side_effect=blocked_checkpoint)
                container.persist_current_checkpoint = checkpoint

            wizard.query_one(action_selector, Button).press()
            await asyncio.wait_for(started.wait(), timeout=2)
            await pilot.pause()

            assert container._failure_action_running is True
            container.validate_step()
            await pilot.pause()
            for selector in (
                "#wizard-back",
                "#wizard-next",
                "#wizard-cancel",
                "#setup-step-retry",
                "#setup-step-manual",
                "#setup-step-later",
            ):
                assert wizard.query_one(selector, Button).disabled is True

            wizard.query_one("#wizard-back", Button).press()
            await pilot.press("ctrl+n")
            await pilot.press("escape")
            second_action = (
                "#setup-step-manual"
                if action_selector != "#setup-step-manual"
                else "#setup-step-later"
            )
            wizard.query_one(second_action, Button).press()
            await pilot.pause(0.1)

            assert app.screen is wizard
            assert container.current_step == failed_index
            assert container.steps[failed_index] is failed_step
            if outcome == "retry":
                assert remove_calls == 1
            else:
                assert checkpoint.await_count == 1

            release.set()
            await pilot.pause(0.8)

            if outcome == "retry":
                replacement = container.steps[failed_index]
                assert replacement is not failed_step
                assert replacement.config.id == STEP_PROVIDER
                assert app.wizard_results == []
                assert container._failure_action_running is False
            elif outcome == "manual":
                assert app.wizard_results == [
                    {
                        "completed": False,
                        "exit_route": "settings",
                        "exit_context": {"category": "providers-models"},
                    }
                ]
            else:
                assert app.wizard_results == [None]

    @pytest.mark.parametrize(
        ("action_selector", "outcome"),
        (
            ("#setup-step-retry", "retry"),
            ("#setup-step-manual", "manual"),
            ("#setup-step-later", "later"),
        ),
    )
    @pytest.mark.asyncio
    async def test_failure_action_completion_is_inert_after_external_dismiss(
        self, monkeypatch, action_selector, outcome
    ):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProviderStep

        original_compose = ProviderStep.compose_step
        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            failed_step = container.steps[container.current_step]
            failed_index = container.current_step
            started = asyncio.Event()
            release = asyncio.Event()

            if outcome == "retry":
                original_remove = failed_step.remove

                async def blocked_remove():
                    started.set()
                    await release.wait()
                    await original_remove()

                monkeypatch.setattr(failed_step, "remove", blocked_remove)
                monkeypatch.setattr(ProviderStep, "compose_step", original_compose)
            else:

                async def blocked_checkpoint():
                    started.set()
                    await release.wait()
                    return True

                container.persist_current_checkpoint = AsyncMock(
                    side_effect=blocked_checkpoint
                )

            dismiss_spy = MagicMock(wraps=container._dismiss_screen)
            monkeypatch.setattr(container, "_dismiss_screen", dismiss_spy)
            notify_spy = MagicMock()
            monkeypatch.setattr(app, "notify", notify_spy)
            navigation_messages = []
            original_post_message = app.post_message

            def capture_posted_message(message):
                if isinstance(message, NavigateToScreen):
                    navigation_messages.append(message)
                return original_post_message(message)

            monkeypatch.setattr(app, "post_message", capture_posted_message)
            recovery_tasks = []

            def independently_run_worker(coroutine, **_kwargs):
                task = asyncio.create_task(coroutine)
                recovery_tasks.append(task)
                return task

            monkeypatch.setattr(container, "run_worker", independently_run_worker)

            wizard.query_one(action_selector, Button).press()
            await asyncio.wait_for(started.wait(), timeout=2)
            wizard.dismiss(None)
            await pilot.pause(0.2)
            assert app.wizard_results == [None]

            release.set()
            task_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            await pilot.pause(0.2)

            assert task_results == [None]
            dismiss_spy.assert_not_called()
            notify_spy.assert_not_called()
            assert navigation_messages == []
            assert app.wizard_results == [None]
            if outcome == "retry":
                assert container.steps[failed_index] is failed_step

    @pytest.mark.asyncio
    async def test_manual_setup_preserves_checkpoint_and_routes_to_provider_settings(
        self, monkeypatch
    ):
        from unittest.mock import AsyncMock

        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )

        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        wizard.app_instance.app_config = {"first_run": {"setup_completed": False}}
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.3)
            saved_data = dict(container.wizard_data)
            persist = AsyncMock(return_value=True)
            container.persist_current_checkpoint = persist

            wizard.query_one("#setup-step-manual", Button).press()
            await pilot.pause(0.3)

            persist.assert_awaited_once()
            assert app.wizard_result == {
                "completed": False,
                "exit_route": "settings",
                "exit_context": {"category": "providers-models"},
            }
            assert container.wizard_data == saved_data
            assert (
                wizard.app_instance.app_config["first_run"]["setup_completed"] is False
            )

    @pytest.mark.asyncio
    async def test_finish_later_persists_checkpoint_and_dismisses(self, monkeypatch):
        from unittest.mock import AsyncMock

        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )

        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.3)
            persist = AsyncMock(return_value=True)
            container.persist_current_checkpoint = persist

            wizard.query_one("#setup-step-later", Button).press()
            await pilot.pause(0.3)

            persist.assert_awaited_once()
            assert app.wizard_result is None

    @pytest.mark.asyncio
    async def test_finish_later_write_failure_keeps_recovery_visible(self, monkeypatch):
        from unittest.mock import AsyncMock

        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )

        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.3)
            container.persist_current_checkpoint = AsyncMock(return_value=False)

            wizard.query_one("#setup-step-later", Button).press()
            await pilot.pause(0.3)

            assert app.wizard_result == "UNSET"
            assert len(wizard.query("#setup-step-retry")) == 1
            assert len(wizard.query("#setup-step-manual")) == 1
            assert len(wizard.query("#setup-step-later")) == 1

    @pytest.mark.asyncio
    async def test_required_failure_rapid_navigation_mashing_cannot_bypass(
        self, monkeypatch
    ):
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ProviderStep,
            SetupWizardContainer,
        )

        monkeypatch.setattr(ProviderStep, "compose_step", _raising_compose_step)
        wizard = _make_wizard()
        app = _HostApp(wizard)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            await pilot.press("ctrl+n")
            await pilot.pause(0.3)

            for _ in range(5):
                wizard.query_one("#wizard-next", Button).press()
                await pilot.press("ctrl+n")
            await pilot.pause(0.4)

            assert container.steps[container.current_step].config.id == STEP_PROVIDER
            assert len(wizard.query("#setup-step-retry")) == 1

    @pytest.mark.asyncio
    async def test_required_failure_remains_in_initial_progress_and_nav(self):
        """Required failures are representable before any track interaction."""
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            ModelStep,
            SetupWizardContainer,
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_MODEL

        original = ModelStep.compose_step
        ModelStep.compose_step = _raising_compose_step
        try:
            wizard = _make_wizard()
            app = _HostApp(wizard)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.2)
                container = wizard.query_one(SetupWizardContainer)
                # No select_track()/Next/note_key_entered() call yet -- this
                # is the wizard's INITIAL state, straight off mount.
                failed_step = next(
                    s for s in container.steps if s.config and s.config.id == STEP_MODEL
                )
                assert failed_step.compose_failed is True
                assert failed_step.required is True
                assert STEP_MODEL in container.active_ids
                nav = wizard.query_one(WizardNavigation)
                assert nav.total_steps == len(container.active_ids)
        finally:
            ModelStep.compose_step = original

    @pytest.mark.asyncio
    async def test_partial_yield_before_raise_is_not_mounted(self):
        """FINDING A (P2): compose() used to stream compose_step()'s yields
        straight through via ``yield from`` -- a step that yielded some
        widgets and THEN raised left those already-yielded widgets mounted,
        rendering a half-built form ABOVE the "couldn't be shown" notice
        (which then lied about the step having been skipped). compose_step()
        must be fully drained before anything is yielded to Textual: either
        ALL of its widgets show up, or NONE do (notice only)."""
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            RagStep,
            SetupWizardContainer,
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_RAG, TRACK_FULL

        original = RagStep.compose_step

        def _partial_then_boom(self):
            yield Static("partial-marker")
            raise RuntimeError("boom")

        RagStep.compose_step = _partial_then_boom
        try:
            wizard = _make_wizard()
            app = _HostApp(wizard)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.2)
                container = wizard.query_one(SetupWizardContainer)
                container.select_track(TRACK_FULL)
                await pilot.pause(0.1)
                failed_step = next(
                    s for s in container.steps if s.config and s.config.id == STEP_RAG
                )
                assert failed_step.compose_failed is True
                notice = str(
                    failed_step.query_one(".setup-step-error", Static).render()
                )
                assert "skipped" in notice.lower()
                markers = [
                    w
                    for w in failed_step.walk_children(Widget)
                    if isinstance(w, Static) and "partial-marker" in str(w.render())
                ]
                assert not markers, (
                    "widgets yielded before compose_step() raised must never "
                    "be mounted alongside the skip notice"
                )
        finally:
            RagStep.compose_step = original

    @pytest.mark.asyncio
    async def test_required_welcome_failure_stays_on_first_page(self):
        """A required first-step failure cannot vanish during initial mount."""
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            SetupWizardContainer,
            WelcomeStep,
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_WELCOME

        original = WelcomeStep.compose_step

        WelcomeStep.compose_step = _raising_compose_step
        try:
            wizard = _make_wizard()
            app = _HostApp(wizard)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.2)
                container = wizard.query_one(SetupWizardContainer)
                welcome_step = next(
                    s
                    for s in container.steps
                    if s.config and s.config.id == STEP_WELCOME
                )
                provider_step = next(
                    s
                    for s in container.steps
                    if s.config and s.config.id == STEP_PROVIDER
                )
                assert welcome_step.compose_failed is True
                assert STEP_WELCOME in container.active_ids
                assert welcome_step.has_class("active")
                assert not provider_step.has_class("active")
                assert len(welcome_step.query("#setup-step-retry")) == 1
                nav = wizard.query_one(WizardNavigation)
                assert nav.total_steps == len(container.active_ids)
                assert nav.current_step == 1
                assert container.active_ids[0] == STEP_WELCOME
        finally:
            WelcomeStep.compose_step = original


@pytest.mark.asyncio
async def test_welcome_exit_paths_state_their_consequences():
    """TASK-2154.9 (FR-01): the three Welcome exits are no longer
    indistinguishable -- the nav cancel button is relabeled to its real
    effect ("Finish later", same dialog as Esc, destructive error styling
    dropped), the whole-wizard skip link says it is permanent ("don't ask
    again") with a tooltip naming the way back, and the Esc hint copy stays
    accurate."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        cancel = wizard.query_one("#wizard-cancel", Button)
        assert str(cancel.label) == "Finish later"
        assert cancel.variant == "default"

        skip = wizard.query_one("#setup-skip-entirely", Button)
        assert str(skip.label) == "Skip setup — don't ask again"
        tooltip = str(skip.tooltip or "")
        assert "won't be offered again" in tooltip
        assert "Settings ▸ Diagnostics" in tooltip

        hints = wizard.query_one("#setup-key-hints", Static)
        assert "Esc finish later" in str(hints.render())


@pytest.mark.asyncio
async def test_quick_track_label_names_the_steps_the_tracker_shows():
    """TASK-2154.9 (FR-02): picking "provider & model" and then seeing four
    tracker entries was the surprise -- the quick-track label now names
    provider, model, voice & summary (Welcome being the step the choice is made
    on), matching the progress row and the "Step 1 of 5" count."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        quick = wizard.query_one("#setup-track-quick", RadioButton)
        label = str(quick.label)
        assert "provider, model, voice & summary" in label
        assert "recommended" in label
        container = wizard.query_one(SetupWizardContainer)
        assert len(container.active_ids) == 5
        nav = wizard.query_one(WizardNavigation)
        assert nav.total_steps == 5


@pytest.mark.asyncio
async def test_nav_text_total_syncs_when_protect_keys_joins_on_key_entry():
    """TASK-2154.9 (FR-02): the conditional protect-keys step joins the
    quick track once a secret exists -- the "Step X of Y" text must update
    in the same refresh as the progress dots; before this fix the text
    total lagged one navigation behind."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        nav = wizard.query_one(WizardNavigation)
        assert STEP_PROTECT not in container.active_ids
        assert nav.total_steps == 5

        container.note_key_entered()
        await pilot.pause(0.1)

        assert STEP_PROTECT in container.active_ids
        assert nav.total_steps == 6
        progress_text = str(wizard.query_one("#wizard-progress", Static).render())
        assert "Step 1 of 6" in progress_text
