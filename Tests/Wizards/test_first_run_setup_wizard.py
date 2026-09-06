"""Pilot tests for the first-run setup wizard skeleton."""

import json
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.config import ConfigMutationResult, RuntimeConfigSnapshot
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
    PendingHandoffStore,
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
    is_untouched_default_session,
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


def _first_chat_store_snapshot(store: ConsoleChatStore) -> list[dict[str, object]]:
    snapshots = []
    for session in store.sessions():
        snapshot = {
            item.name: deepcopy(getattr(session, item.name))
            for item in fields(session)
            if item.name not in {"rag_scope_holder", "todo_store"}
        }
        snapshot["rag_scope_holder"] = deepcopy(session.rag_scope_holder.scope)
        snapshot["todo_store"] = deepcopy(session.todo_store.export_snapshot())
        snapshots.append(snapshot)
    return snapshots


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


def test_untouched_default_session_requires_no_user_owned_state() -> None:
    defaults = ConsoleSessionSettings(
        provider="openai",
        model="model-a",
        source="derived",
    )
    store = ConsoleChatStore()
    session = store.create_session(
        settings=defaults,
        canonical_settings_baseline=defaults,
    )

    assert is_untouched_default_session(session, (), "", ()) is True
    assert is_untouched_default_session(session, (), "draft", ()) is False
    assert is_untouched_default_session(session, (object(),), "", ()) is False
    assert is_untouched_default_session(session, (), "", (object(),)) is False

    session.settings = replace(defaults, temperature=0.19, source="user")
    assert is_untouched_default_session(session, (), "", ()) is False


def test_untouched_default_session_rejects_custom_workspace_provenance() -> None:
    defaults = ConsoleSessionSettings(
        provider="openai",
        model="model-a",
        source="derived",
    )
    store = ConsoleChatStore()
    session = store.create_session(
        workspace_id="workspace-user",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )

    assert is_untouched_default_session(session, (), "", ()) is False


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("has_user_work", True),
        ("assistant_kind", "character"),
        ("character_name", "Private character"),
        ("user_display_name_override", "Private user"),
        ("character_system_template", "Private roleplay prompt"),
        ("identity_revision", 1),
        ("todos", [{"content": "private tool task"}]),
    ],
)
def test_untouched_default_session_rejects_roleplay_tool_and_custom_state(
    field_name: str,
    value: object,
) -> None:
    defaults = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        source="derived",
    )
    store = ConsoleChatStore()
    session = store.create_session(
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    if field_name == "todos":
        session.todo_store.create(content="private tool task")
    else:
        setattr(session, field_name, value)

    assert is_untouched_default_session(session, (), "", ()) is False


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

        # TASK-21148 (UAT V-1/V-2): outcome first — try-it controls lead,
        # plumbing (endpoint/auth/model/voice ids) follows under the
        # Advanced disclosure.
        assert ordered_ids == [
            "setup-voice-preset",
            "setup-voice-sample",
            "setup-voice-test",
            "setup-voice-status",
            "setup-voice-default",
            "setup-voice-endpoint",
            "setup-voice-auth",
            "setup-voice-model",
            "setup-voice-voice",
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
        assert "Not tested yet" in status
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
        step.query_one(
            "#setup-voice-endpoint", Input
        ).value = "http://127.0.0.1:9999/v1/audio/speech"
        await pilot.pause()
        release.set()
        await pilot.pause(0.1)

        assert "Not tested yet" in str(
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
        assert "Not tested yet" in str(
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
        assert "Not tested yet" in str(
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
        assert "Not tested yet" in str(
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
        assert "Not tested yet" in str(
            step.query_one("#setup-voice-status", Static).renderable
        )
        assert "sk-added-outside-draft" not in repr(step.get_step_data())


@pytest.mark.parametrize(
    "app_config",
    [
        {"openai_api": {"api_key": "synthetic-test-credential"}},
        {"API": {"openai_api_key": "synthetic-test-credential"}},
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "API": {"openai_api_key": "synthetic-test-credential"}
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
        assert "synthetic-test-credential" not in repr(step.get_step_data())


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
    must still reach the Welcome-step Skip setup confirmation. Live UAT reproduction
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
async def test_rapid_double_escape_during_first_render_keeps_skip_setup_open():
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
async def test_next_button_click_drives_quick_track_to_summary():
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
    # _StyledHostApp: this test drives real screen coordinates via
    # pilot.click, which needs the app stylesheet loaded — the wizard
    # tracker's layout rides the bundle as BUNDLED_CSS (class-level
    # DEFAULT_CSS is barred by the parse-cache rule).
    app = _StyledHostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        await pilot.click("#setup-track-quick")
        await pilot.pause(0.1)

        seen_step_ids = []
        for _ in range(10):
            await pilot.click("#wizard-next")
            await pilot.pause(0.2)
            step = container.steps[container.current_step]
            seen_step_ids.append(step.config.id if step.config else None)
            if isinstance(step, SummaryStep):
                break

        assert app.wizard_result == "UNSET"
        # Exactly the quick-track subset, each step visited once, in order.
        assert seen_step_ids == [
            "provider",
            "model",
            "voice",
            "protect-keys",
            "summary",
        ]
        assert set(container.wizard_data.keys()) == {
            "welcome",
            "provider",
            "model",
            "voice",
            "protect-keys",
        }
        assert container.query_one("#wizard-next", Button).display is False
        assert container.query_one("#wizard-cancel", Button).display is False


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
async def test_unchanged_provider_model_backtrack_keeps_live_radio_and_one_writer(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

    endpoint = "https://stable.example.test/v1"
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.custom": {"api_url": endpoint}}
    ).fully_applied
    writes = []
    real_persist = persistence_module.persist_provider_setup

    def counted_persist(mutation):
        writes.append(mutation.section_values["chat_defaults"]["model"])
        return real_persist(mutation)

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        counted_persist,
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {"api_settings": {"custom": {"api_url": endpoint}}}
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "stable-radio-model")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert provider_index is not None and model_index is not None
        assert voice_index is not None
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step._environment_provider = dict
        container.show_step(provider_index)
        provider_step.select_provider("custom")
        provider_step._on_clear()
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
        assert model_step.selected_model_id == "stable-radio-model"
        assert persistence_module.provider_setup_expected_state_matches_snapshot(
            container._committed_provider_expected_state,
            config_module.get_atomic_config_snapshot(),
        )

        container.show_step(provider_index)
        await container._advance()
        await pilot.pause(0.1)

        radio_set = model_step.query_one("#setup-model-choice", RadioSet)
        pressed = radio_set.pressed_button
        assert pressed is not None
        assert pressed in list(radio_set.query(RadioButton))
        assert pressed.value is True
        assert getattr(pressed, "_model_id", "") == "stable-radio-model"
        assert model_step._effective_model_id() == "stable-radio-model"
        assert persistence_module.provider_setup_expected_state_matches_snapshot(
            container._committed_provider_expected_state,
            config_module.get_atomic_config_snapshot(),
        )

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
    ("provider", "configured_endpoint", "endpoint_edit", "test_state"),
    [
        ("anthropic", None, None, "unsupported"),
        ("anthropic", "https://api.anthropic.com/v1", None, "unsupported"),
        ("cohere", None, None, "unsupported"),
        ("deepseek", None, None, "ready"),
        ("google", None, None, "unsupported"),
        ("groq", None, None, "ready"),
        ("huggingface", None, None, "unsupported"),
        ("mistral", None, None, "ready"),
        ("MistralAI", None, None, "ready"),
        ("moonshot", None, None, "unsupported"),
        ("openai", None, None, "ready"),
        ("OpenAI", "https://gateway.example.test/openai/v1", None, "ready"),
        ("openrouter", None, None, "ready"),
        ("qwencloud", None, None, "ready"),
        ("qwencloud", None, "", "invalid-endpoint"),
        ("qwencloud", None, "https://bad host/v1", "invalid-endpoint"),
        (
            "QwenCloud",
            "https://gateway.example.test/qwen/compatible-mode/v1",
            None,
            "ready",
        ),
        ("zai", None, None, "unsupported"),
    ],
)
@pytest.mark.asyncio
async def test_cloud_provider_test_is_enabled_only_with_compatible_probe_target(
    provider: str,
    configured_endpoint: str | None,
    endpoint_edit: str | None,
    test_state: str,
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
        if test_state == "ready":
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
            copy = str(
                step.query_one("#setup-provider-key-status", Static).renderable
            ).lower()
            if test_state == "unsupported":
                assert "connection testing is unavailable" in copy
                assert "valid endpoint" not in copy
            else:
                assert "valid endpoint" in copy
                assert "connection testing is unavailable" not in copy


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

    from tldw_chatbook import config as config_module
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
    assert config_module.apply_settings_mutation_to_cli_config(
        {
            "api_settings.custom": {
                "api_url": "https://stored.example.test/v1/chat/completions",
                "api_key": secret,
            }
        }
    ).fully_applied
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
@pytest.mark.parametrize("reconfirmation", ["discovery", "manual"])
@pytest.mark.asyncio
async def test_credential_rotation_after_provider_handoff_invalidates_before_save(
    monkeypatch,
    credential_kind: str,
    rotation_timing: str,
    reconfirmation: str,
):
    import asyncio
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module

    first_secret = f"{credential_kind}-handoff-secret-a"
    rotated_secret = f"{credential_kind}-handoff-secret-b"
    provider_settings = {"api_url": "https://rotation.example.test/v1/chat/completions"}
    if credential_kind == "environment":
        provider_settings["api_key_env_var"] = "CUSTOM_API_KEY"
        monkeypatch.setenv("CUSTOM_API_KEY", first_secret)
    else:
        provider_settings["api_key"] = first_secret

    probe = AsyncMock(return_value=_reachable_endpoint_outcome("rotation-model-a"))
    rotated_discovery_started = asyncio.Event()
    release_rotated_discovery = asyncio.Event()
    discovery_calls = 0

    async def discover_models(**_kwargs):
        nonlocal discovery_calls
        discovery_calls += 1
        if discovery_calls == 1:
            return _typed_model_discovery_result("custom", "rotation-model-a")
        rotated_discovery_started.set()
        await release_rotated_discovery.wait()
        return _typed_model_discovery_result("custom", "rotation-model-b")

    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(side_effect=discover_models)
    wizard = _make_wizard()
    wizard.app_instance.app_config = {"api_settings": {"custom": provider_settings}}
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.custom": provider_settings}
    ).fully_applied
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    persisted = MagicMock(return_value=ConfigMutationResult(True, True, None))
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        persisted,
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
        first_discovery_key = provider_step._selected_discovery_key
        assert first_discovery_key is not None
        assert provider_step._selected_provider_models == {
            first_discovery_key: ("rotation-model-a",)
        }
        assert container._first_run_selected_provider_models == {
            first_discovery_key: ("rotation-model-a",)
        }
        if rotation_timing == "before_selection":
            if credential_kind == "environment":
                monkeypatch.setenv("CUSTOM_API_KEY", rotated_secret)
            else:
                provider_settings["api_key"] = rotated_secret
                assert config_module.apply_settings_mutation_to_cli_config(
                    {"api_settings.custom": {"api_key": rotated_secret}}
                ).fully_applied

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        target = None
        for _ in range(20):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "rotation-model-a"
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
                assert config_module.apply_settings_mutation_to_cli_config(
                    {"api_settings.custom": {"api_key": rotated_secret}}
                ).fully_applied

            await container._advance()
            assert container.current_step == model_index

        await asyncio.wait_for(rotated_discovery_started.wait(), timeout=2)
        await pilot.pause()
        assert provider_step._credential_revision > tested.credential_revision
        assert provider_step._provider_evidence_store().evidence_for(tested) is None
        assert provider_step._last_tested_provider_identity == tested
        assert first_discovery_key not in provider_step._selected_provider_models
        assert provider_step._selected_provider_models == {}
        assert container._first_run_selected_provider_models == {}
        assert model_step.selected_model_id == ""
        assert model_step._selection_discovery_key is None
        assert all(
            getattr(button, "_model_id", "") != "rotation-model-a"
            for button in model_step.query(RadioButton)
        )
        persisted.assert_not_called()

        if reconfirmation == "manual":
            model_step.query_one(
                "#setup-model-custom", Input
            ).value = "manual-model-under-b"
            await pilot.pause()
            await container._advance()
            expected_model = "manual-model-under-b"
            release_rotated_discovery.set()
        else:
            release_rotated_discovery.set()
            target = None
            for _ in range(30):
                target = next(
                    (
                        button
                        for button in model_step.query(RadioButton)
                        if getattr(button, "_model_id", "") == "rotation-model-b"
                    ),
                    None,
                )
                if target is not None:
                    break
                await pilot.pause(0.05)
            assert target is not None
            target.value = True
            await pilot.pause()
            assert model_step._selection_discovery_key == (
                provider_step._selected_discovery_key
            )
            assert model_step._selection_discovery_key == (
                provider_step._model_discovery_key(
                    provider_step._effective_provider_draft()
                )
            )
            assert (
                model_step._selection_config_precondition
                is (
                    container._first_run_provider_config_preconditions[
                        model_step._selection_discovery_key
                    ]
                )
            )
            assert provider_step._sync_live_credential_revision() is False
            await container._advance()
            expected_model = "rotation-model-b"

        assert container.provider_setup_committed
        persisted.assert_called_once()
        assert (
            persisted.call_args.args[0].section_values["chat_defaults"]["model"]
            == expected_model
        )
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

    writer_calls = 0

    def blocked_persist(mutation):
        nonlocal writer_calls
        writer_calls += 1
        values = mutation.section_values["api_settings.custom"]
        if writer_calls == 1:
            assert values["api_key"] == secret
        writer_started.set()
        assert release_writer.wait(timeout=5)
        events.append("writer-settled")
        writer_settled.set()
        if writer_outcome == "error" and writer_calls == 1:
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
    wizard = _make_wizard(provider_dismiss_warning_seconds=0)
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
            pending = container.query_one("#setup-provider-save-status", Static)
            assert "finishing save" in str(pending.renderable).lower()
            settling_stack = tuple(app.screen_stack)
            await pilot.press("escape", "escape")
            await pilot.pause()
            assert tuple(app.screen_stack) == settling_stack
            assert app.screen is wizard
            pending_copy = str(pending.renderable)
            container.action_cancel()
            wizard.action_cancel()
            await pilot.pause()
            assert tuple(app.screen_stack) == settling_stack
            assert app.screen is wizard
            assert str(pending.renderable) == pending_copy
            container._dismiss_screen(None)
            await pilot.pause(0.1)
            assert app.wizard_result == "UNSET"
            assert not writer_settled.is_set()
            assert not commit_waiter.done()
            assert writer_calls == 1
            assert "longer than expected" in str(pending.renderable).lower()
            assert pending.display
            assert app.focused is pending
            for selector in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                assert container.query_one(selector, Button).disabled
        finally:
            release_writer.set()

        assert await commit_waiter is (writer_outcome == "success")
        await pilot.pause(0.2)
        assert container._provider_commit_task is None
        assert container._provider_commit_identity is None
        assert not container._provider_commit_write_started
        if writer_outcome == "success":
            assert app.wizard_result is None
            assert app.wizard_results == [None]
            assert wizard not in app.screen_stack
            events.append("dismissed")
            assert events == ["writer-settled", "dismissed"]
            assert container.staged_provider_draft is None
        else:
            assert app.wizard_result == "UNSET"
            assert container.current_step == provider_index
            assert container.staged_provider_draft is None
            status = container.query_one("#setup-provider-save-status", Static)
            assert "couldn't finish saving" in str(status.renderable).lower()
            assert not container.query_one("#wizard-next", Button).disabled
            assert app.focused is provider_step.query_one(
                "#setup-provider-endpoint", Input
            )
            for selector in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                assert not container.query_one(selector, Button).disabled
            assert secret not in repr(container.__dict__)
            assert endpoint_secret not in repr(provider_step.__dict__)

            await pilot.press("escape")
            await pilot.pause()
            from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                _SettlingGuardedConfirmationDialog,
            )

            assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
            await pilot.click("#cancel-button")
            await pilot.pause()
            assert app.screen is wizard

            provider_step.query_one(
                "#setup-provider-endpoint", Input
            ).value = "https://retry.example.test/v1"
            provider_step.query_one(
                "#setup-provider-api-key", Input
            ).value = "retry-credential"
            await pilot.pause()
            await container._advance()
            model_index = container._step_index_for_id(STEP_MODEL)
            assert container.current_step == model_index
            model_step = container.steps[model_index]
            assert isinstance(model_step, ModelStep)
            model_step.query_one("#setup-model-custom", Input).value = "retry-model"
            await pilot.pause()
            await container._advance()
            assert container.provider_setup_committed
            assert writer_calls == 2
        for rendered in (
            repr(container.__dict__),
            repr(provider_step.__dict__),
            app.export_screenshot(),
        ):
            assert secret not in rendered
            assert endpoint_secret not in rendered


@pytest.mark.asyncio
async def test_unmount_during_irreversible_provider_write_never_publishes_to_ui(
    monkeypatch,
):
    import asyncio
    import threading

    writer_started = threading.Event()
    release_writer = threading.Event()

    def blocked_persist(_mutation):
        writer_started.set()
        assert release_writer.wait(timeout=5)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup",
        blocked_persist,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    environment_secret = "unmount-environment-secret"
    monkeypatch.setenv("CUSTOM_API_KEY", environment_secret)
    wizard = _make_wizard(provider_dismiss_warning_seconds=0)
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {"api_key_env_var": "CUSTOM_API_KEY"},
        }
    }
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
        endpoint_input = provider_step.query_one("#setup-provider-endpoint", Input)
        key_input = provider_step.query_one("#setup-provider-api-key", Input)
        endpoint_input.value = "https://unmount.example.test/private/v1"
        key_input.value = "unmount-secret"
        await pilot.pause()
        await container._advance()
        commit_waiter = asyncio.create_task(
            container.commit_staged_provider_setup("unmount-model")
        )
        for _ in range(40):
            if writer_started.is_set():
                break
            await pilot.pause(0.025)
        assert writer_started.is_set()

        pop_results_before = list(app.wizard_results)
        try:
            await app.pop_screen()
            await pilot.pause(0.1)
            assert wizard not in app.screen_stack
            assert len(app.screen_stack) == 1

            provider_query = MagicMock(wraps=provider_step.query_one)
            cancel_workers = MagicMock(wraps=provider_step._cancel_discovery_workers)
            container_query = MagicMock(wraps=container.query_one)
            notify = MagicMock(wraps=container.notify)
            dismiss = MagicMock(wraps=wizard.dismiss)
            monkeypatch.setattr(provider_step, "query_one", provider_query)
            monkeypatch.setattr(
                provider_step, "_cancel_discovery_workers", cancel_workers
            )
            monkeypatch.setattr(container, "query_one", container_query)
            monkeypatch.setattr(container, "notify", notify)
            monkeypatch.setattr(wizard, "dismiss", dismiss)

            release_writer.set()
            assert await commit_waiter
            await pilot.pause(0.1)

            provider_query.assert_not_called()
            cancel_workers.assert_not_called()
            container_query.assert_not_called()
            notify.assert_not_called()
            dismiss.assert_not_called()
        finally:
            release_writer.set()

        assert container._provider_commit_task is None
        assert container._provider_commit_identity is None
        assert not container._provider_commit_write_started
        assert container.staged_provider_draft is None
        assert container._first_run_selected_provider_models == {}
        assert provider_step._environment() == {}
        assert provider_step._sensitive_key_input is None
        assert provider_step._sensitive_endpoint_input is None
        assert endpoint_input.value == ""
        assert key_input.value == ""
        assert app.wizard_results == pop_results_before
        assert "unmount-secret" not in repr(container.__dict__)
        assert environment_secret not in repr(provider_step.__dict__)
        assert "private/v1" not in repr(provider_step.__dict__)


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
        await pilot.pause(0.1)
        status_rows = list(
            model_step.query_one("#setup-model-choice", RadioSet).query(RadioButton)
        )
        assert len(status_rows) == 1
        assert str(status_rows[0].label) == (
            "Model listing unavailable; enter the model ID used by this endpoint."
        )
        assert status_rows[0].disabled
        assert getattr(status_rows[0], "_model_id", None) is None
        assert not model_step.query_one("#setup-model-custom", Input).disabled
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

    environment_canary = "exact-draft-environment-canary"
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
                        "credential_source": "environment",
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
    step = _provider_step(
        wizard=wizard,
        discover=AsyncMock(return_value=()),
        environ={"CUSTOM_API_KEY": environment_canary},
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        await pilot.pause(0.1)

        call = scope_service.discover_models.await_args
        assert call.kwargs["provider"] == "custom"
        assert call.kwargs["use_shared_cache"] is False
        assert call.kwargs["staged_settings"] == {
            "api_settings": {
                "custom": {
                    "api_url": "https://exact.test/proxy/v1/chat/completions",
                    "credential_source": "environment",
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
        assert environment_canary not in repr(identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_key", "saved_endpoint", "replacement_endpoint"),
    [
        (
            "custom",
            "https://saved.example.test/v1/chat/completions",
            "https://replacement.example.test/v1/chat/completions",
        ),
        (
            "custom_2",
            "https://saved-two.example.test/v1/chat/completions",
            "https://replacement-two.example.test/v1/chat/completions",
        ),
        (
            "llama_cpp",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:9090",
        ),
    ],
)
async def test_mounted_keep_and_clear_are_distinct_exact_auth_decisions(
    provider_key: str,
    saved_endpoint: str,
    replacement_endpoint: str,
):
    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
        ModelDiscoveryCache,
    )
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        build_first_run_model_discovery_key,
    )

    saved_canary = "wizard-saved-key-canary-never-send"
    environment_canary = "wizard-environment-key-canary-never-send"
    requests: list[dict[str, object]] = []

    async def record_discovery(**kwargs):
        requests.append(kwargs)
        return _typed_model_discovery_result(provider_key, "keyless-model")

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "providers": {provider_key: []},
        "api_settings": {
            provider_key: {
                "api_url": saved_endpoint,
                "api_key": saved_canary,
                "api_key_env_var": "WIZARD_CUSTOM_API_KEY",
            }
        },
    }
    shared_cache = ModelDiscoveryCache()
    local_service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {provider_key: []},
        settings_loader=lambda: wizard.app_instance.app_config,
        discovery_cache=shared_cache,
        discovery_client=record_discovery,
        environ={"WIZARD_CUSTOM_API_KEY": environment_canary},
    )
    wizard.app_instance.llm_provider_catalog_scope_service = (
        LLMProviderCatalogScopeService(
            local_service=local_service,
            server_service=None,
        )
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
        provider_step.select_provider(provider_key)
        provider_step._on_keep()
        for _ in range(20):
            if requests and provider_step._selected_provider_models:
                break
            await pilot.pause(0.05)
        assert requests
        assert requests[-1]["api_key"] == saved_canary
        prior_keys = set(provider_step._selected_provider_models)
        assert prior_keys
        [kept_key] = prior_keys
        assert kept_key.credential_source == "stored"
        assert saved_canary not in repr(kept_key)
        assert environment_canary not in repr(kept_key)

        provider_step.query_one("#setup-provider-key-clear", Button).press()
        provider_step.query_one(
            "#setup-provider-endpoint", Input
        ).value = replacement_endpoint
        requests.clear()
        await pilot.pause(0.2)
        await container._advance()
        await pilot.pause(0.1)

        assert container.current_step == model_index
        assert requests
        assert all(request["api_key"] is None for request in requests)
        assert saved_canary not in repr(requests)
        assert environment_canary not in repr(requests)
        assert shared_cache.snapshot_count == 0
        assert shared_cache.model_count == 0
        draft = container.staged_provider_draft
        assert draft is not None
        key = build_first_run_model_discovery_key(draft)
        assert key.credential_revision > 0
        assert key.credential_source == "draft"
        assert key not in prior_keys
        assert key in container._first_run_selected_provider_models
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        rendered_model_ids = [
            str(getattr(button, "_model_id", button.label))
            for button in model_step.query_one("#setup-model-choice", RadioSet).query(
                RadioButton
            )
        ]
        assert rendered_model_ids == ["keyless-model"]


@pytest.mark.asyncio
async def test_mounted_reloaded_explicit_keyless_stays_keyless_through_discovery_save(
    monkeypatch,
):
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )

    endpoint = "https://reloaded-keyless.example.test/v1/chat/completions"
    ambient_canary = "reloaded-ambient-custom-key-canary"
    monkeypatch.setenv("CUSTOM_API_KEY", ambient_canary)
    assert config_module.apply_settings_mutation_to_cli_config(
        {
            "api_settings.custom": {
                "api_url": endpoint,
                "credential_source": "none",
            }
        }
    ).fully_applied
    reloaded = config_module.load_settings(force_reload=True)
    assert reloaded["api_settings"]["custom"]["credential_source"] == "none"

    requests: list[dict[str, object]] = []

    async def record_discovery(**kwargs):
        requests.append(kwargs)
        return _typed_model_discovery_result("custom", "reloaded-keyless-model")

    wizard = _make_wizard()
    wizard.app_instance.app_config = reloaded
    local_service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"custom": []},
        settings_loader=lambda: wizard.app_instance.app_config,
        discovery_client=record_discovery,
        environ={"CUSTOM_API_KEY": ambient_canary},
    )
    wizard.app_instance.llm_provider_catalog_scope_service = (
        LLMProviderCatalogScopeService(local_service=local_service, server_service=None)
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert provider_index is not None
        assert model_index is not None
        assert voice_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        for _ in range(20):
            if provider_step._selected_discovery_state == "complete":
                break
            await pilot.pause(0.05)

        assert provider_step._selected_discovery_state == "complete"
        provider_key = provider_step._selected_discovery_key
        assert provider_key is not None
        assert provider_key.credential_source == "none"
        assert requests and all(request["api_key"] is None for request in requests)
        assert ambient_canary not in repr(requests)

        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(20):
            rows = [
                row
                for row in model_step.query(RadioButton)
                if getattr(row, "_model_id", "") == "reloaded-keyless-model"
            ]
            if rows:
                break
            await pilot.pause(0.05)
        assert rows
        rows[0].value = True
        await pilot.pause()
        await container._advance()

        assert container.current_step == voice_index
        persisted = config_module.load_settings(force_reload=True)
        custom = persisted["api_settings"]["custom"]
        assert custom["credential_source"] == "none"
        readiness = get_provider_readiness(
            "custom", persisted, environ={"CUSTOM_API_KEY": ambient_canary}
        )
        assert readiness.api_key is None
        assert readiness.api_key_source is None


@pytest.mark.asyncio
async def test_mounted_openai_builtin_endpoint_handoff_uses_one_exact_env_request(
    monkeypatch,
):
    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
        ModelDiscoveryCache,
    )

    environment_canary = "wizard-openai-env-canary-never-store"
    monkeypatch.setenv("OPENAI_API_KEY", environment_canary)
    requests: list[dict[str, object]] = []

    async def record_discovery(**kwargs):
        requests.append(kwargs)
        return _typed_model_discovery_result("openai", "gpt-live-exact")

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "providers": {"OpenAI": ["curated-must-not-replace-live"]},
        "api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}},
    }
    shared_cache = ModelDiscoveryCache()
    local_service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"OpenAI": []},
        settings_loader=lambda: wizard.app_instance.app_config,
        discovery_cache=shared_cache,
        discovery_client=record_discovery,
        environ={"OPENAI_API_KEY": environment_canary},
    )
    wizard.app_instance.llm_provider_catalog_scope_service = (
        LLMProviderCatalogScopeService(local_service=local_service, server_service=None)
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
        provider_step.select_provider("openai")
        for _ in range(20):
            if provider_step._selected_discovery_state == "complete":
                break
            await pilot.pause(0.05)
        assert provider_step._selected_discovery_state == "complete"
        provider_discovery_key = provider_step._selected_discovery_key
        assert provider_discovery_key is not None
        assert provider_discovery_key.connection_identity[1] == (
            "https://api.openai.com/v1/chat/completions"
        )

        await container._advance()
        for _ in range(20):
            if list(container.steps[model_index].query("#setup-model-option-0")):
                break
            await pilot.pause(0.05)

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        row = model_step.query_one("#setup-model-option-0", RadioButton)
        assert getattr(row, "_model_id", None) == "gpt-live-exact"
        assert model_step._shown_for_discovery_key == provider_discovery_key
        assert len(requests) == 1
        assert requests[0]["endpoint"] == ("https://api.openai.com/v1/chat/completions")
        assert requests[0]["api_key"] == environment_canary
        assert shared_cache.snapshot_count == 0
        assert environment_canary not in repr(provider_discovery_key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_key",
        "provider_list_key",
        "endpoint_settings",
        "expected_discovery_endpoint",
        "expected_models_url",
        "expected_identity_url",
    ),
    [
        (
            "moonshot",
            "Moonshot",
            {"api_region": "china"},
            "https://api.moonshot.cn/v1/chat/completions",
            "https://api.moonshot.cn/v1/models",
            "https://api.moonshot.cn/v1/chat/completions",
        ),
        (
            "moonshot",
            "Moonshot",
            {"api_region": "international"},
            "https://api.moonshot.ai/v1/chat/completions",
            "https://api.moonshot.ai/v1/models",
            "https://api.moonshot.ai/v1/chat/completions",
        ),
        (
            "huggingface",
            "HuggingFace",
            {"use_router_url_format": "true"},
            "https://router.huggingface.co/v1/chat/completions",
            "https://router.huggingface.co/v1/models",
            "https://router.huggingface.co/v1/chat/completions",
        ),
        (
            "huggingface",
            "HuggingFace",
            {"use_router_url_format": "false"},
            "https://api-inference.huggingface.co/v1/chat/completions",
            "https://api-inference.huggingface.co/v1/models",
            "https://api-inference.huggingface.co/v1/chat/completions",
        ),
    ],
)
async def test_mounted_settings_aware_builtin_discovery_uses_exact_runtime_host_once(
    provider_key: str,
    provider_list_key: str,
    endpoint_settings: dict[str, str],
    expected_discovery_endpoint: str,
    expected_models_url: str,
    expected_identity_url: str,
):
    from copy import deepcopy

    import httpx

    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
        ModelDiscoveryCache,
    )
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )

    credential_canary = f"{provider_key}-settings-aware-credential-canary"
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "runtime-model"}]})

    wizard = _make_wizard()
    provider_settings = {
        **endpoint_settings,
        "api_key": credential_canary,
    }
    wizard.app_instance.app_config = {
        "providers": {provider_list_key: ["curated-must-not-appear"]},
        "api_settings": {provider_key: provider_settings},
    }
    before = deepcopy(wizard.app_instance.app_config)
    shared_cache = ModelDiscoveryCache()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        async def real_discovery(**kwargs):
            return await discover_openai_compatible_models(**kwargs, client=client)

        local_service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: {provider_list_key: []},
            settings_loader=lambda: wizard.app_instance.app_config,
            discovery_cache=shared_cache,
            discovery_client=real_discovery,
            environ={},
        )
        wizard.app_instance.llm_provider_catalog_scope_service = (
            LLMProviderCatalogScopeService(
                local_service=local_service,
                server_service=None,
            )
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
            provider_step.select_provider(provider_key)
            for _ in range(30):
                if provider_step._selected_discovery_state == "complete":
                    break
                await pilot.pause(0.05)

            assert provider_step._selected_discovery_state == "complete"
            key = provider_step._selected_discovery_key
            assert key is not None
            assert key.connection_identity == (provider_key, expected_identity_url)
            assert key.credential_source == "stored"
            assert credential_canary not in repr(key)

            await container._advance()
            for _ in range(20):
                if list(container.steps[model_index].query("#setup-model-option-0")):
                    break
                await pilot.pause(0.05)

            draft = container.staged_provider_draft
            assert draft is not None
            assert draft.endpoint == ""
            assert draft.discovery_endpoint == expected_discovery_endpoint
            model_step = container.steps[model_index]
            row = model_step.query_one("#setup-model-option-0", RadioButton)
            assert getattr(row, "_model_id", None) == "runtime-model"
            assert len(requests) == 1
            assert str(requests[0].url) == expected_models_url
            assert requests[0].headers["Authorization"] == (
                f"Bearer {credential_canary}"
            )
            assert requests[0].url.host == httpx.URL(expected_models_url).host
            assert shared_cache.snapshot_count == 0
            assert (
                wizard.app_instance.app_config["api_settings"] == before["api_settings"]
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_key",
        "provider_list_key",
        "setting_name",
        "initial_setting",
        "changed_setting",
        "initial_host",
        "changed_host",
    ),
    [
        (
            "moonshot",
            "Moonshot",
            "api_region",
            "china",
            "international",
            "api.moonshot.cn",
            "api.moonshot.ai",
        ),
        (
            "huggingface",
            "HuggingFace",
            "use_router_url_format",
            "true",
            "false",
            "router.huggingface.co",
            "api-inference.huggingface.co",
        ),
    ],
)
async def test_mounted_model_save_rejects_settings_changed_discovery_identity(
    provider_key: str,
    provider_list_key: str,
    setting_name: str,
    initial_setting: str,
    changed_setting: str,
    initial_host: str,
    changed_host: str,
):
    """A model discovered for endpoint A cannot be atomically saved under B."""
    import httpx

    from tldw_chatbook import config as config_module
    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        model = (
            "identity-a-model"
            if request.url.host == initial_host
            else "identity-b-model"
        )
        return httpx.Response(200, json={"data": [{"id": model}]})

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "providers": {provider_list_key: []},
        "api_settings": {
            provider_key: {
                setting_name: initial_setting,
                "api_key": "save-boundary-credential-canary",
            }
        },
    }
    assert config_module.apply_settings_mutation_to_cli_config(
        {
            f"api_settings.{provider_key}": {
                setting_name: initial_setting,
                "api_key": "save-boundary-credential-canary",
            }
        }
    ).fully_applied
    writes: list[object] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        async def real_discovery(**kwargs):
            return await discover_openai_compatible_models(**kwargs, client=client)

        local_service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: {provider_list_key: []},
            settings_loader=lambda: wizard.app_instance.app_config,
            discovery_client=real_discovery,
            environ={},
        )
        wizard.app_instance.llm_provider_catalog_scope_service = (
            LLMProviderCatalogScopeService(
                local_service=local_service,
                server_service=None,
            )
        )
        app = _HostApp(wizard)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)

            async def commit_config(
                settings,
                *,
                delete_keys=None,
                after_write=None,
                provider_setup_mutation=None,
            ):
                if provider_setup_mutation is not None:
                    writes.append(provider_setup_mutation)
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
            provider_step.select_provider(provider_key)
            await container._advance()

            model_step = container.steps[model_index]
            assert isinstance(model_step, ModelStep)
            target = None
            for _ in range(30):
                target = next(
                    (
                        button
                        for button in model_step.query(RadioButton)
                        if getattr(button, "_model_id", "") == "identity-a-model"
                    ),
                    None,
                )
                if target is not None:
                    break
                await pilot.pause(0.05)
            assert target is not None
            target.value = True
            await pilot.pause()
            selected_key = model_step._selection_discovery_key
            assert selected_key is not None
            assert selected_key.connection_identity[1].startswith(
                f"https://{initial_host}/"
            )

            wizard.app_instance.app_config["api_settings"][provider_key][
                setting_name
            ] = changed_setting
            assert config_module.apply_settings_mutation_to_cli_config(
                {
                    f"api_settings.{provider_key}": {
                        setting_name: changed_setting,
                    }
                }
            ).fully_applied
            await container._advance()

            assert container.current_step == model_index
            assert writes == []
            assert not container.provider_setup_committed
            assert model_step.selected_model_id == ""
            assert model_step._selection_discovery_key is None
            assert model_step._effective_model_id() == ""
            error = str(container.query_one("#setup-step-error-pinned", Static).renderable)
            assert "connection settings changed" in error.lower()

            for _ in range(30):
                if any(
                    getattr(button, "_model_id", "") == "identity-b-model"
                    for button in model_step.query(RadioButton)
                ):
                    break
                await pilot.pause(0.05)
            current_key = model_step._current_discovery_key()
            assert current_key is not None and current_key != selected_key
            assert current_key.connection_identity[1].startswith(
                f"https://{changed_host}/"
            )
            assert [request.url.host for request in requests] == [
                initial_host,
                changed_host,
            ]

            manual = model_step.query_one("#setup-model-custom", Input)
            manual.value = "identity-a-model"
            await pilot.pause()
            await container._advance()

            assert len(writes) == 1
            assert container.provider_setup_committed
            assert container.committed_provider_model == "identity-a-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_key", "initial_settings", "changed_values"),
    [
        (
            "moonshot",
            {
                "api_region": "china",
                "api_base_url": "https://api.moonshot.cn/v1",
                "api_key": "selection-time-key-a",
            },
            {
                "api_region": "international",
                "api_base_url": "https://api.moonshot.ai/v1",
            },
        ),
        (
            "huggingface",
            {
                "use_router_url_format": "true",
                "api_base_url": "https://router.huggingface.co/v1",
                "api_key": "selection-time-key-a",
            },
            {
                "use_router_url_format": "false",
                "api_base_url": "https://api-inference.huggingface.co/v1",
            },
        ),
        (
            "custom",
            {
                "api_url": "https://selection-a.example/v1/chat/completions",
                "api_key": "selection-time-key-a",
            },
            {"api_url": "https://selection-b.example/v1/chat/completions"},
        ),
        (
            "custom",
            {
                "api_url": "https://selection-key.example/v1/chat/completions",
                "api_key": "selection-time-key-a",
            },
            {"api_key": "selection-time-key-b"},
        ),
    ],
    ids=["moonshot-region", "huggingface-router", "custom-endpoint", "stored-key"],
)
async def test_mounted_selection_precondition_rejects_completed_config_write_before_save(
    provider_key: str,
    initial_settings: dict[str, str],
    changed_values: dict[str, str],
):
    """A completed config write cannot rebase an already-selected model."""
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module

    assert config_module.apply_settings_mutation_to_cli_config(
        {f"api_settings.{provider_key}": initial_settings}
    ).fully_applied
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {provider_key: dict(initial_settings)}
    }
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                provider_key, "selection-a-model"
            )
        )
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider(provider_key)
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        target = None
        for _ in range(30):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "selection-a-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()
        selected_key = model_step._selection_discovery_key
        assert selected_key is not None

        assert config_module.apply_settings_mutation_to_cli_config(
            {f"api_settings.{provider_key}": changed_values}
        ).fully_applied
        assert model_step._current_discovery_key() == selected_key
        await container._advance()

        authoritative = config_module.get_atomic_config_snapshot().values
        current_settings = authoritative["api_settings"][provider_key]
        assert all(
            current_settings[key] == value for key, value in changed_values.items()
        )
        assert authoritative["chat_defaults"]["model"] != "selection-a-model"
        assert container.current_step == model_index
        assert not container.provider_setup_committed
        assert model_step.selected_model_id == ""
        assert model_step._selection_discovery_key is None
        error = str(container.query_one("#setup-step-error-pinned", Static).renderable)
        assert "connection settings changed" in error.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("change_kind", ["unchanged", "unrelated"])
async def test_mounted_selection_precondition_allows_current_relevant_config(
    change_kind: str,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module

    initial_settings = {
        "api_url": "https://selection-current.example/v1/chat/completions",
        "api_key": "selection-current-key",
    }
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.custom": initial_settings}
    ).fully_applied
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": dict(initial_settings)}
    }
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                "custom", "selection-current-model"
            )
        )
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
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
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(30):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "selection-current-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        target.value = True
        await pilot.pause()

        if change_kind == "unrelated":
            assert config_module.apply_settings_mutation_to_cli_config(
                {"general": {"users_name": "selection-unrelated-change"}}
            ).fully_applied
        await container._advance()

        authoritative = config_module.get_atomic_config_snapshot().values
        assert container.provider_setup_committed
        assert authoritative["chat_defaults"]["model"] == "selection-current-model"
        assert model_step._model_id_from_custom_input is False
        assert model_step._selection_config_precondition is not None
        if change_kind == "unrelated":
            assert authoritative["general"]["users_name"] == (
                "selection-unrelated-change"
            )


@pytest.mark.asyncio
async def test_mounted_manual_typing_captures_config_once_per_decision(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module

    endpoint = "https://manual-session.example/v1/chat/completions"
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.custom": {"api_url": endpoint}}
    ).fully_applied
    capture_calls = []
    original_capture = SetupWizardContainer.capture_provider_config_precondition

    def counted_capture(discovery_key):
        capture_calls.append(discovery_key)
        return original_capture(discovery_key)

    monkeypatch.setattr(
        SetupWizardContainer,
        "capture_provider_config_precondition",
        staticmethod(counted_capture),
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {"api_settings": {"custom": {"api_url": endpoint}}}
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                "custom", "discovered-session-model"
            )
        )
    )
    model_step = None

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
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
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(30):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "discovered-session-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        manual = model_step.query_one("#setup-model-custom", Input)
        baseline = len(capture_calls)
        for value in ("m", "manual", "manual-session", "manual-session-model"):
            manual.value = value
            await pilot.pause()

        assert len(capture_calls) == baseline + 1
        first_precondition = model_step._selection_config_precondition
        assert first_precondition is not None

        manual.value = ""
        await pilot.pause()
        assert model_step._selection_config_precondition is None
        baseline = len(capture_calls)
        for value in ("n", "new", "new-manual-model"):
            manual.value = value
            await pilot.pause()
        assert len(capture_calls) == baseline + 1
        assert model_step._selection_config_precondition is not first_precondition

        target.value = True
        await pilot.pause()
        assert model_step._model_id_from_custom_input is False
        assert (
            model_step._selection_config_precondition
            is (
                container._first_run_provider_config_preconditions[
                    model_step._selection_discovery_key
                ]
            )
        )

        manual.value = "retry-manual-model"
        await pilot.pause()
        retry = model_step.query_one("#setup-model-retry", Button)
        retry.press()
        await pilot.pause()
        assert model_step._selection_config_precondition is None

        manual.value = "unmount-manual-model"
        await pilot.pause()
        assert model_step._selection_config_precondition is not None

    assert model_step is not None
    assert model_step._selection_config_precondition is None


@pytest.mark.asyncio
async def test_mounted_successful_manual_save_ends_decision_before_back_edit(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

    endpoint = "https://manual-resave.example/v1/chat/completions"
    assert config_module.apply_settings_mutation_to_cli_config(
        {
            "api_settings.custom": {
                "api_url": endpoint,
                "api_key": "manual-resave-stored-canary",
            }
        }
    ).fully_applied
    capture_calls = []
    original_capture = SetupWizardContainer.capture_provider_config_precondition

    def counted_capture(discovery_key):
        capture_calls.append(discovery_key)
        return original_capture(discovery_key)

    monkeypatch.setattr(
        SetupWizardContainer,
        "capture_provider_config_precondition",
        staticmethod(counted_capture),
    )
    setup_writes = []
    real_persist = persistence_module.persist_provider_setup

    def counted_persist(mutation):
        setup_writes.append(mutation)
        return real_persist(mutation)

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        counted_persist,
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {
                "api_url": endpoint,
                "api_key": "manual-resave-stored-canary",
            }
        }
    }
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                "custom", "manual-resave-discovered-model"
            )
        )
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert provider_index is not None
        assert model_index is not None
        assert voice_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        manual = model_step.query_one("#setup-model-custom", Input)
        baseline = len(capture_calls)
        for value in ("m", "manual", "manual-one"):
            manual.value = value
            await pilot.pause()
        assert len(capture_calls) == baseline + 1

        await container._advance()
        assert container.current_step == voice_index
        assert container.provider_setup_committed
        assert model_step._manual_decision_active is False
        assert model_step._selection_config_precondition is None
        first_save_captures = len(capture_calls)
        assert len(setup_writes) == 1

        container.action_back()
        await pilot.pause()
        assert container.current_step == model_index
        assert manual.value == "manual-one"
        assert len(capture_calls) == first_save_captures

        assert config_module.apply_settings_mutation_to_cli_config(
            {"general": {"users_name": "manual-resave-unrelated"}}
        ).fully_applied
        await container._advance()
        assert container.current_step == voice_index
        assert len(setup_writes) == 1
        assert len(capture_calls) == first_save_captures
        assert (
            config_module.get_atomic_config_snapshot().values["general"]["users_name"]
            == "manual-resave-unrelated"
        )

        container.action_back()
        await pilot.pause()
        assert container.current_step == model_index
        assert manual.value == "manual-one"
        assert len(capture_calls) == first_save_captures

        for value in ("manual-t", "manual-two"):
            manual.value = value
            await pilot.pause()
        assert len(capture_calls) == first_save_captures + 1
        assert model_step._selection_discovery_key == (
            provider_step._model_discovery_key(
                provider_step._effective_provider_draft()
            )
        )
        assert provider_step._sync_live_credential_revision() is False
        assert model_step._selection_discovery_key == (
            provider_step._model_discovery_key(
                provider_step._effective_provider_draft()
            )
        )
        await container._advance()

        authoritative = config_module.get_atomic_config_snapshot().values
        assert container.current_step == voice_index
        assert container.provider_setup_committed
        assert container.committed_provider_model == "manual-two"
        assert authoritative["chat_defaults"]["model"] == "manual-two"
        assert len(setup_writes) == 2


@pytest.mark.asyncio
async def test_mounted_sparse_keyless_save_back_next_is_idempotent(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import provider_setup_persistence as persistence_module
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    endpoint = "https://sparse-keyless.example/v1/chat/completions"
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.custom": {"api_url": endpoint}}
    ).fully_applied
    setup_writes = []
    real_persist = persistence_module.persist_provider_setup

    def counted_persist(mutation):
        setup_writes.append(mutation)
        return real_persist(mutation)

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        counted_persist,
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "providers": {"custom": []},
        "api_settings": {"custom": {"api_url": endpoint}},
    }
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                "custom", "sparse-keyless-discovered"
            )
        )
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert provider_index is not None
        assert model_index is not None
        assert voice_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        manual = model_step.query_one("#setup-model-custom", Input)
        manual.value = "sparse-keyless-manual"
        await pilot.pause()
        await container._advance()

        assert container.current_step == voice_index
        assert len(setup_writes) == 1
        authoritative = config_module.get_atomic_config_snapshot().values
        custom = authoritative["api_settings"]["custom"]
        assert custom["credential_source"] == "none"
        readiness = get_provider_readiness(
            "custom",
            authoritative,
            environ={"CUSTOM_API_KEY": "late-environment-canary"},
        )
        assert readiness.api_key is None
        assert readiness.api_key_source is None

        container.action_back()
        await pilot.pause()
        await container._advance()

        assert container.current_step == voice_index
        assert len(setup_writes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change_kind",
    ["provider", "endpoint", "runtime", "credential", "model"],
)
async def test_mounted_unchanged_manual_next_rejects_relevant_external_change(
    monkeypatch,
    change_kind,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

    provider_key = "moonshot" if change_kind == "runtime" else "custom"
    if provider_key == "moonshot":
        provider_settings = {
            "api_region": "china",
            "api_base_url": "https://api.moonshot.cn/v1",
            "api_key": "manual-idempotent-key-a",
        }
    else:
        provider_settings = {
            "api_url": "https://manual-idempotent-a.example/v1/chat/completions",
            "api_key": "manual-idempotent-key-a",
        }
    assert config_module.apply_settings_mutation_to_cli_config(
        {f"api_settings.{provider_key}": provider_settings}
    ).fully_applied

    capture_calls = []
    original_capture = SetupWizardContainer.capture_provider_config_precondition

    def counted_capture(discovery_key):
        capture_calls.append(discovery_key)
        return original_capture(discovery_key)

    monkeypatch.setattr(
        SetupWizardContainer,
        "capture_provider_config_precondition",
        staticmethod(counted_capture),
    )
    setup_writes = []
    real_persist = persistence_module.persist_provider_setup

    def counted_persist(mutation):
        setup_writes.append(mutation)
        return real_persist(mutation)

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        counted_persist,
    )

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {provider_key: dict(provider_settings)}
    }
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                provider_key, "manual-idempotent-discovered"
            )
        )
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert provider_index is not None
        assert model_index is not None
        assert voice_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider(provider_key)
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        manual = model_step.query_one("#setup-model-custom", Input)
        manual.value = "manual-idempotent-model"
        await pilot.pause()
        await container._advance()
        assert container.current_step == voice_index
        assert len(setup_writes) == 1
        first_save_captures = len(capture_calls)

        container.action_back()
        await pilot.pause()
        assert container.current_step == model_index
        if change_kind == "provider":
            external_values = {"chat_defaults": {"provider": "openai"}}
        elif change_kind == "endpoint":
            external_values = {
                "api_settings.custom": {
                    "api_url": (
                        "https://manual-idempotent-b.example/v1/chat/completions"
                    )
                }
            }
        elif change_kind == "runtime":
            external_values = {
                "api_settings.moonshot": {
                    "api_region": "international",
                    "api_base_url": "https://api.moonshot.ai/v1",
                }
            }
        elif change_kind == "credential":
            external_values = {
                "api_settings.custom": {"api_key": "manual-idempotent-key-b"}
            }
        else:
            external_values = {"chat_defaults": {"model": "external-model-change"}}
        assert config_module.apply_settings_mutation_to_cli_config(
            external_values
        ).fully_applied

        await container._advance()

        assert container.current_step == model_index
        assert len(setup_writes) == 1
        assert len(capture_calls) == first_save_captures
        result = container._provider_last_config_result
        assert getattr(result, "conflict_reason", None) == "identity_changed"
        assert model_step.selected_model_id == ""
        assert model_step._selection_discovery_key is None
        error = str(container.query_one("#setup-step-error-pinned", Static).renderable)
        assert "connection settings changed" in error.lower()
        rendered = pilot.app.export_screenshot()
        assert "manual-idempotent-key-a" not in rendered
        assert "manual-idempotent-key-b" not in rendered

        authoritative = config_module.get_atomic_config_snapshot().values
        if change_kind == "provider":
            assert authoritative["chat_defaults"]["provider"] == "openai"
        elif change_kind == "endpoint":
            assert authoritative["api_settings"]["custom"]["api_url"].startswith(
                "https://manual-idempotent-b.example"
            )
        elif change_kind == "runtime":
            assert authoritative["api_settings"]["moonshot"]["api_region"] == (
                "international"
            )
        elif change_kind == "credential":
            assert authoritative["api_settings"]["custom"]["api_key"] == (
                "manual-idempotent-key-b"
            )
        else:
            assert authoritative["chat_defaults"]["model"] == ("external-model-change")


@pytest.mark.asyncio
async def test_mounted_manual_reconfirmation_binds_current_authoritative_precondition(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module

    endpoint_a = "https://manual-selection-a.example/v1/chat/completions"
    endpoint_b = "https://manual-selection-b.example/v1/chat/completions"
    assert config_module.apply_settings_mutation_to_cli_config(
        {
            "api_settings.custom": {
                "api_url": endpoint_a,
                "api_key": "manual-selection-key-a",
            }
        }
    ).fully_applied
    capture_calls = []
    original_capture = SetupWizardContainer.capture_provider_config_precondition

    def counted_capture(discovery_key):
        capture_calls.append(discovery_key)
        return original_capture(discovery_key)

    monkeypatch.setattr(
        SetupWizardContainer,
        "capture_provider_config_precondition",
        staticmethod(counted_capture),
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {
                "api_url": endpoint_a,
                "api_key": "manual-selection-key-a",
            }
        }
    }
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(
                "custom", "manual-selection-a-model"
            )
        )
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
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
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(30):
            target = next(
                (
                    button
                    for button in model_step.query(RadioButton)
                    if getattr(button, "_model_id", "") == "manual-selection-a-model"
                ),
                None,
            )
            if target is not None:
                break
            await pilot.pause(0.05)
        assert target is not None
        manual = model_step.query_one("#setup-model-custom", Input)
        baseline = len(capture_calls)
        for value in ("m", "manual-a", "manual-selection-a-model"):
            manual.value = value
            await pilot.pause()
        assert len(capture_calls) == baseline + 1

        assert config_module.apply_settings_mutation_to_cli_config(
            {
                "api_settings.custom": {
                    "api_url": endpoint_b,
                    "api_key": "manual-selection-key-b",
                }
            }
        ).fully_applied
        await container._advance()
        assert container.current_step == model_index
        assert not container.provider_setup_committed
        assert model_step._selection_config_precondition is None

        wizard.app_instance.app_config["api_settings"]["custom"].update(
            {
                "api_url": endpoint_b,
                "api_key": "manual-selection-key-b",
            }
        )
        container.show_step(provider_index)
        endpoint_input = provider_step.query_one("#setup-provider-endpoint", Input)
        endpoint_input.value = endpoint_b
        await pilot.pause()
        await container._advance()
        assert container.current_step == model_index
        baseline = len(capture_calls)
        for value in ("m", "manual-b", "manual-selection-b-model"):
            manual.value = value
            await pilot.pause()
        assert len(capture_calls) == baseline + 1
        await container._advance()

        authoritative = config_module.get_atomic_config_snapshot().values
        assert container.provider_setup_committed
        assert authoritative["api_settings"]["custom"]["api_url"] == endpoint_b
        assert authoritative["api_settings"]["custom"]["api_key"] == (
            "manual-selection-key-b"
        )
        assert authoritative["chat_defaults"]["model"] == ("manual-selection-b-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["endpoint", "credential"])
async def test_mounted_model_save_rejects_hidden_provider_identity_change(
    changed_field: str,
):
    """Hidden Provider controls cannot silently retarget a selected model."""
    from unittest.mock import AsyncMock

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {
                "api_url": "https://identity-a.example/v1/chat/completions",
                "api_key": "stored-save-boundary-canary",
            }
        }
    }
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(
        return_value=_typed_model_discovery_result("custom", "identity-a-model")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    writes: list[object] = []

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)

        async def commit_config(
            settings,
            *,
            delete_keys=None,
            after_write=None,
            provider_setup_mutation=None,
        ):
            if provider_setup_mutation is not None:
                writes.append(provider_setup_mutation)
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
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(20):
            rows = [
                button
                for button in model_step.query(RadioButton)
                if getattr(button, "_model_id", "") == "identity-a-model"
            ]
            if rows:
                break
            await pilot.pause(0.05)
        assert rows
        rows[0].value = True
        await pilot.pause()
        old_key = model_step._selection_discovery_key
        assert old_key is not None

        if changed_field == "endpoint":
            provider_step.query_one(
                "#setup-provider-endpoint", Input
            ).value = "https://identity-b.example/v1/chat/completions"
        else:
            provider_step.query_one(
                "#setup-provider-api-key", Input
            ).value = "replacement-save-boundary-canary"
        await pilot.pause()
        await container._advance()

        assert container.current_step == model_index
        assert writes == []
        assert model_step.selected_model_id == ""
        assert model_step._selection_discovery_key is None
        assert model_step._current_discovery_key() != old_key
        assert not container.provider_setup_committed
        for rendered in (repr(old_key), pilot.app.export_screenshot()):
            assert "stored-save-boundary-canary" not in rendered
            assert "replacement-save-boundary-canary" not in rendered


@pytest.mark.asyncio
async def test_mounted_save_identity_change_fences_cancellation_resistant_old_result():
    """A late endpoint-A retry cannot repopulate handoff state after save rejects."""
    import asyncio
    from unittest.mock import AsyncMock

    retry_started = asyncio.Event()
    release_retry = asyncio.Event()
    calls = 0

    async def discover_models(**kwargs):
        nonlocal calls
        calls += 1
        provider_settings = kwargs["staged_settings"]["api_settings"]["moonshot"]
        endpoint = next(
            value
            for key, value in provider_settings.items()
            if key in {"api_url", "api_base_url"}
        )
        if calls == 2:
            retry_started.set()
            while not release_retry.is_set():
                try:
                    await release_retry.wait()
                except asyncio.CancelledError:
                    continue
            return _typed_model_discovery_result("moonshot", "late-china-model")
        model = "china-model" if ".cn/" in endpoint else "global-model"
        return _typed_model_discovery_result("moonshot", model)

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "moonshot": {
                "api_region": "china",
                "api_key": "late-result-save-boundary-canary",
            }
        }
    }
    scope_service = MagicMock(discover_models=AsyncMock(side_effect=discover_models))
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.commit_config = AsyncMock(return_value=True)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("moonshot")
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(20):
            rows = [
                button
                for button in model_step.query(RadioButton)
                if getattr(button, "_model_id", "") == "china-model"
            ]
            if rows:
                break
            await pilot.pause(0.05)
        assert rows
        rows[0].value = True
        await pilot.pause()
        stale_key = model_step._selection_discovery_key
        assert stale_key is not None

        draft = container.staged_provider_draft
        assert draft is not None
        provider_step._begin_selected_provider_discovery(
            draft,
            sync_live_credential=False,
        )
        await asyncio.wait_for(retry_started.wait(), timeout=2)
        wizard.app_instance.app_config["api_settings"]["moonshot"]["api_region"] = (
            "international"
        )
        await container._advance()
        release_retry.set()
        await pilot.pause(0.2)

        current_key = model_step._current_discovery_key()
        assert current_key is not None and current_key != stale_key
        assert stale_key not in provider_step._selected_provider_models
        assert stale_key not in provider_step._selected_provider_outcomes
        assert stale_key not in container._first_run_selected_provider_models
        assert stale_key not in container._first_run_selected_provider_outcomes
        assert model_step.selected_model_id == ""
        assert all(
            getattr(button, "_model_id", "") != "late-china-model"
            for button in model_step.query(RadioButton)
        )


@pytest.mark.asyncio
async def test_mounted_save_lease_rechecks_identity_immediately_before_write(
    monkeypatch,
):
    """A settings change after task creation is still fenced before persistence."""
    from unittest.mock import AsyncMock

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "moonshot": {
                "api_region": "china",
                "api_key": "lease-recheck-credential-canary",
            }
        }
    }
    scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result("moonshot", "china-only-model")
        )
    )
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        writes: list[object] = []

        async def commit_config(
            settings,
            *,
            delete_keys=None,
            after_write=None,
            provider_setup_mutation=None,
        ):
            if provider_setup_mutation is not None:
                writes.append(provider_setup_mutation)
            return True

        container.commit_config = commit_config
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("moonshot")
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(20):
            rows = [
                button
                for button in model_step.query(RadioButton)
                if getattr(button, "_model_id", "") == "china-only-model"
            ]
            if rows:
                break
            await pilot.pause(0.05)
        assert rows
        rows[0].value = True
        await pilot.pause()
        china_key = model_step._selection_discovery_key
        assert china_key is not None

        original_effective_draft = provider_step._effective_provider_draft
        calls = 0

        def change_settings_after_first_boundary_check(*args, **kwargs):
            nonlocal calls
            draft = original_effective_draft(*args, **kwargs)
            calls += 1
            if calls == 1:
                wizard.app_instance.app_config["api_settings"]["moonshot"][
                    "api_region"
                ] = "international"
            return draft

        monkeypatch.setattr(
            provider_step,
            "_effective_provider_draft",
            change_settings_after_first_boundary_check,
        )
        await container._advance()

        assert calls >= 2
        assert writes == []
        assert container.current_step == model_index
        assert not container.provider_setup_committed
        assert model_step.selected_model_id == ""
        current_key = model_step._current_discovery_key()
        assert current_key is not None and current_key != china_key
        assert current_key.connection_identity[1].startswith("https://api.moonshot.ai/")


@pytest.mark.asyncio
async def test_mounted_builtin_settings_change_fences_prior_discovery_identity():
    import asyncio

    import httpx

    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )

    china_started = asyncio.Event()
    release_china = asyncio.Event()
    requests: list[httpx.Request] = []
    asyncio.get_running_loop().call_later(3, release_china.set)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.moonshot.cn":
            china_started.set()
            while not release_china.is_set():
                try:
                    await release_china.wait()
                except asyncio.CancelledError:
                    continue
            return httpx.Response(200, json={"data": [{"id": "stale-china"}]})
        return httpx.Response(200, json={"data": [{"id": "current-global"}]})

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "providers": {"Moonshot": []},
        "api_settings": {
            "moonshot": {
                "api_region": "china",
                "api_key": "moonshot-settings-change-canary",
            }
        },
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        async def real_discovery(**kwargs):
            return await discover_openai_compatible_models(**kwargs, client=client)

        local_service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: {"Moonshot": []},
            settings_loader=lambda: wizard.app_instance.app_config,
            discovery_client=real_discovery,
            environ={},
        )
        wizard.app_instance.llm_provider_catalog_scope_service = (
            LLMProviderCatalogScopeService(
                local_service=local_service,
                server_service=None,
            )
        )
        app = _HostApp(wizard)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            container = wizard.query_one(SetupWizardContainer)
            provider_index = container._step_index_for_id(STEP_PROVIDER)
            welcome_index = container._step_index_for_id(STEP_WELCOME)
            assert provider_index is not None and welcome_index is not None
            container.show_step(provider_index)
            provider_step = container.steps[provider_index]
            assert isinstance(provider_step, ProviderStep)
            provider_step.select_provider("moonshot")
            await asyncio.wait_for(china_started.wait(), timeout=2)
            stale_key = provider_step._selected_discovery_key
            assert stale_key is not None

            container.show_step(welcome_index)
            wizard.app_instance.app_config["api_settings"]["moonshot"]["api_region"] = (
                "international"
            )
            container.show_step(provider_index)
            for _ in range(30):
                if provider_step._selected_provider_models:
                    break
                await pilot.pause(0.05)

            current_key = provider_step._selected_discovery_key
            assert current_key is not None and current_key != stale_key
            assert current_key.connection_identity[1].startswith(
                "https://api.moonshot.ai/"
            )
            assert provider_step._selected_provider_models == {
                current_key: ("current-global",)
            }

            release_china.set()
            await pilot.pause(0.2)
            assert stale_key not in provider_step._selected_provider_models
            assert stale_key not in provider_step._selected_provider_outcomes
            assert provider_step._selected_provider_models == {
                current_key: ("current-global",)
            }
            assert [request.url.host for request in requests].count(
                "api.moonshot.cn"
            ) == 1
            assert [request.url.host for request in requests].count(
                "api.moonshot.ai"
            ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_change",
    ["moonshot_region", "huggingface_router", "custom_endpoint", "credential"],
)
async def test_mounted_executor_entry_rejects_stale_provider_identity(
    monkeypatch,
    identity_change: str,
):
    """The atomic writer rejects identity changes made after UI prechecks."""
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

    if identity_change == "moonshot_region":
        provider_key = "moonshot"
        provider_settings = {
            "api_region": "china",
            "api_base_url": "https://api.moonshot.cn/v1",
            "api_key": "writer-entry-saved-canary",
        }
    elif identity_change == "huggingface_router":
        provider_key = "huggingface"
        provider_settings = {
            "use_router_url_format": "true",
            "api_base_url": "https://router.huggingface.co/v1",
            "api_key": "writer-entry-saved-canary",
        }
    else:
        provider_key = "custom"
        provider_settings = {
            "api_url": "https://writer-a.example/v1/chat/completions",
            "api_key": "writer-entry-saved-canary",
        }

    assert config_module.apply_settings_mutation_to_cli_config(
        {f"api_settings.{provider_key}": provider_settings}
    ).fully_applied

    wizard = _make_wizard()
    wizard.app_instance.app_config = {"api_settings": {provider_key: provider_settings}}
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result(provider_key, "writer-model")
        )
    )
    real_persist = persistence_module.persist_provider_setup
    writer_entered = threading.Event()
    release_writer = threading.Event()

    def paused_persist(mutation):
        writer_entered.set()
        assert release_writer.wait(timeout=3)
        return real_persist(mutation)

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        paused_persist,
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider(provider_key)
        await container._advance()

        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        rows = []
        for _ in range(20):
            rows = [
                button
                for button in model_step.query(RadioButton)
                if getattr(button, "_model_id", "") == "writer-model"
            ]
            if rows:
                break
            await pilot.pause(0.05)
        assert rows
        rows[0].value = True
        await pilot.pause()

        save = asyncio.create_task(container._advance())
        assert await asyncio.to_thread(writer_entered.wait, 2)
        if identity_change == "moonshot_region":
            changed_values = {
                "api_region": "international",
                "api_base_url": "https://api.moonshot.ai/v1",
            }
        elif identity_change == "huggingface_router":
            changed_values = {
                "use_router_url_format": "false",
                "api_base_url": "https://api-inference.huggingface.co/v1",
            }
        elif identity_change == "custom_endpoint":
            changed_values = {"api_url": "https://writer-b.example/v1/chat/completions"}
        else:
            changed_values = {"api_key": "writer-entry-replacement-canary"}
        provider_settings.update(changed_values)
        shared_write = await asyncio.to_thread(
            config_module.apply_settings_mutation_to_cli_config,
            {f"api_settings.{provider_key}": changed_values},
        )
        assert shared_write.fully_applied
        release_writer.set()
        await asyncio.wait_for(save, timeout=3)

        authoritative = config_module.get_atomic_config_snapshot().values
        current_settings = authoritative["api_settings"][provider_key]
        assert all(
            current_settings[key] == value for key, value in changed_values.items()
        )
        assert authoritative["chat_defaults"]["model"] != "writer-model"
        assert container.current_step == model_index
        assert not container.provider_setup_committed
        assert model_step.selected_model_id == ""
        assert model_step._selection_discovery_key is None
        error = str(container.query_one("#setup-step-error-pinned", Static).renderable)
        assert "connection settings changed" in error.lower()
        rendered = pilot.app.export_screenshot()
        assert "writer-entry-saved-canary" not in rendered
        assert "writer-entry-replacement-canary" not in rendered


@pytest.mark.asyncio
async def test_mounted_executor_entry_unchanged_identity_writes_once(monkeypatch):
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

    provider_settings = {
        "api_region": "china",
        "api_base_url": "https://api.moonshot.cn/v1",
        "api_key": "writer-entry-unchanged-canary",
    }
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.moonshot": provider_settings}
    ).fully_applied
    wizard = _make_wizard()
    wizard.app_instance.app_config = {"api_settings": {"moonshot": provider_settings}}
    wizard.app_instance.llm_provider_catalog_scope_service = MagicMock(
        discover_models=AsyncMock(
            return_value=_typed_model_discovery_result("moonshot", "writer-model")
        )
    )
    real_persist = persistence_module.persist_provider_setup
    writer_entered = threading.Event()
    release_writer = threading.Event()

    def paused_persist(mutation):
        writer_entered.set()
        assert release_writer.wait(timeout=3)
        return real_persist(mutation)

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        paused_persist,
    )

    async with _HostApp(wizard).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        assert provider_index is not None and model_index is not None
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("moonshot")
        await container._advance()
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        rows = []
        for _ in range(20):
            rows = [
                button
                for button in model_step.query(RadioButton)
                if getattr(button, "_model_id", "") == "writer-model"
            ]
            if rows:
                break
            await pilot.pause(0.05)
        assert rows
        rows[0].value = True
        await pilot.pause()

        save = asyncio.create_task(container._advance())
        assert await asyncio.to_thread(writer_entered.wait, 2)
        real_apply = persistence_module.apply_settings_mutation_to_cli_config
        provider_writes = []

        def counting_write(section_values, **kwargs):
            chat_defaults = section_values.get("chat_defaults", {})
            if chat_defaults.get("model") == "writer-model":
                provider_writes.append(True)
            return real_apply(section_values, **kwargs)

        monkeypatch.setattr(
            persistence_module,
            "apply_settings_mutation_to_cli_config",
            counting_write,
        )
        release_writer.set()
        await asyncio.wait_for(save, timeout=3)

        assert provider_writes == [True]
        assert container.provider_setup_committed
        assert container.committed_provider_model == "writer-model"


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
async def test_environment_appearance_revisions_and_fences_keyless_discovery(
    monkeypatch,
):
    import asyncio
    import os
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    seen_keys: list[str | None] = []
    appeared_canary = "appeared-custom-environment-canary"
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    async def discover_models(*, provider, staged_settings, **_kwargs):
        api_key = get_provider_readiness(
            provider, staged_settings, environ=os.environ
        ).api_key
        seen_keys.append(api_key)
        if len(seen_keys) == 1:
            first_started.set()
            try:
                await release_first.wait()
            except asyncio.CancelledError:
                await release_first.wait()
            return _typed_model_discovery_result("custom", "late-keyless-model")
        return _typed_model_discovery_result("custom", "current-environment-model")

    scope_service = MagicMock(discover_models=AsyncMock(side_effect=discover_models))
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={
                "api_settings": {
                    "custom": {
                        "api_url": "https://appearing-env.example/v1/chat/completions",
                        "api_key_env_var": "CUSTOM_API_KEY",
                    }
                }
            },
            llm_provider_catalog_scope_service=scope_service,
        ),
        stage_provider_setup=MagicMock(return_value=True),
        invalidate_provider_model_handoff=MagicMock(),
        invalidate_provider_write_expectation=MagicMock(),
        rerun=False,
    )
    step = _provider_step(
        wizard=wizard,
        environ=os.environ,
        discover=AsyncMock(return_value=()),
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        await asyncio.wait_for(first_started.wait(), timeout=2)
        first_key = step._selected_discovery_key
        assert first_key is not None
        assert first_key.credential_source == "none"
        first_revision = first_key.credential_revision

        monkeypatch.setenv("CUSTOM_API_KEY", appeared_canary)
        assert step._sync_live_credential_revision() is True
        for _ in range(20):
            if len(seen_keys) == 2 and step._selected_discovery_state == "complete":
                break
            await pilot.pause(0.05)
        release_first.set()
        await pilot.pause(0.1)

        current_key = step._selected_discovery_key
        assert current_key is not None
        assert current_key.credential_source == "environment"
        assert current_key.credential_revision > first_revision
        assert seen_keys == [None, appeared_canary]
        assert first_key not in step._selected_provider_models
        assert step._selected_provider_models == {
            current_key: ("current-environment-model",)
        }
        assert appeared_canary not in repr(step._credential_observations)
        assert appeared_canary not in repr(step._selected_provider_models)


@pytest.mark.asyncio
async def test_explicit_keyless_suppresses_later_environment_appearance(monkeypatch):
    import os
    from unittest.mock import AsyncMock

    appeared_canary = "suppressed-custom-environment-canary"
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    requests: list[dict[str, object]] = []

    async def discover_models(**kwargs):
        requests.append(kwargs)
        return _typed_model_discovery_result("custom", "explicit-keyless-model")

    wizard = MagicMock()
    wizard.app_instance = MagicMock(
        app_config={
            "api_settings": {
                "custom": {
                    "api_url": "https://explicit-keyless.example/v1/chat/completions",
                    "credential_source": "none",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        },
        llm_provider_catalog_scope_service=MagicMock(
            discover_models=AsyncMock(side_effect=discover_models)
        ),
    )
    wizard.stage_provider_setup = MagicMock(return_value=True)
    wizard.rerun = False
    step = _provider_step(
        wizard=wizard,
        environ=os.environ,
        discover=AsyncMock(return_value=()),
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("custom")
        for _ in range(20):
            if step._selected_discovery_state == "complete":
                break
            await pilot.pause(0.05)
        first_key = step._selected_discovery_key
        assert first_key is not None
        assert first_key.credential_source == "none"
        first_revision = first_key.credential_revision
        assert (
            requests
            and requests[0]["staged_settings"]["api_settings"]["custom"][
                "credential_source"
            ]
            == "none"
        )

        monkeypatch.setenv("CUSTOM_API_KEY", appeared_canary)
        assert step._sync_live_credential_revision() is False
        await pilot.pause(0.1)

        assert step._selected_discovery_key == first_key
        assert step._credential_revision == first_revision
        assert len(requests) == 1
        assert requests[0]["staged_settings"]["api_settings"]["custom"].get(
            "api_key"
        ) in {None, ""}
        assert appeared_canary not in repr(requests)


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
        ("inline-secret", {}, "stored", True),
        ("inline-secret", {"PRIVATE_OPENAI_KEY": "environment-secret"}, "stored", True),
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
        assert draft.credential.source == "stored"
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

    from tldw_chatbook.Chat.console_provider_endpoints import builtin_provider_endpoint
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    if not callable(getattr(wizard, "commit_staged_provider_setup", None)):
        wizard.commit_staged_provider_setup = AsyncMock(return_value=True)
    provider_values = (getattr(wizard, "wizard_data", {}) or {}).get("provider", {})
    provider_key = str(provider_values.get("provider_key", ""))
    if provider_key and not isinstance(
        getattr(wizard, "staged_provider_draft", None), FirstRunProviderDraft
    ):
        endpoint = builtin_provider_endpoint(provider_key, {}) or (
            "https://custom.example/v1/chat/completions"
        )
        wizard.staged_provider_draft = FirstRunProviderDraft(
            provider_key,
            endpoint,
            ProviderCredentialDraft("none", "", 0),
        )
    return ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover_models or AsyncMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_model_step_provider_change_resets_selection():
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

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
        wizard.staged_provider_draft = FirstRunProviderDraft(
            "anthropic",
            "https://api.anthropic.com/v1",
            ProviderCredentialDraft("none", "", 0),
        )
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
        wizard.commit_staged_provider_setup.assert_awaited_once_with(
            "gpt-5.6-terra",
            discovery_key=step._current_discovery_key(),
            model_provenance="discovered",
        )
        wizard.commit_config.assert_not_called()


@pytest.mark.asyncio
async def test_model_step_commit_marks_typed_model_as_manual_provenance():
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
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-model-custom", Input).value = "manual-model"
        await pilot.pause()

        ok, error = await step.commit()

        assert ok, error
        wizard.commit_staged_provider_setup.assert_awaited_once_with(
            "manual-model",
            discovery_key=step._current_discovery_key(),
            model_provenance="manual",
        )
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
        wizard.commit_staged_provider_setup.assert_awaited_once_with(
            "radio-model-a",
            discovery_key=step._current_discovery_key(),
            model_provenance="discovered",
        )
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

    async def discover(*, provider, **_identity):
        return {"openai": ["model-a"], "anthropic": ["model-b"]}[provider]

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
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            FirstRunProviderDraft,
            ProviderCredentialDraft,
        )

        wizard.staged_provider_draft = FirstRunProviderDraft(
            "anthropic",
            "https://api.anthropic.com/v1",
            ProviderCredentialDraft("none", "", 0),
        )
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
async def test_model_step_uses_exact_provider_draft_with_scope_service():
    """The mounted scope-service path receives the staged connection, not config."""
    from unittest.mock import AsyncMock, MagicMock as Mock
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    scope_result = _typed_model_discovery_result("custom", "svc-model-a", "svc-model-b")
    scope_service = Mock()
    scope_service.discover_models = AsyncMock(return_value=scope_result)
    provider_draft = FirstRunProviderDraft(
        provider="custom",
        endpoint="https://draft.example/proxy/v1/chat/completions",
        credential=ProviderCredentialDraft("draft", "draft-secret", 7),
    )
    app_instance = MagicMock(
        app_config={
            "api_settings": {
                "custom": {
                    "api_url": "https://ambient.example/v1/chat/completions",
                    "api_key": "ambient-secret",
                }
            }
        }
    )
    app_instance.llm_provider_catalog_scope_service = scope_service
    wizard = SimpleNamespace(
        app_instance=app_instance,
        wizard_data={
            "provider": {"provider_key": "openai", "provider_value": "OpenAI"}
        },
        staged_provider_draft=provider_draft,
        commit_config=AsyncMock(return_value=True),
        commit_staged_provider_setup=AsyncMock(return_value=True),
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
            "provider": "custom",
            "use_shared_cache": False,
            "staged_settings": {
                "api_settings": {
                    "custom": {
                        "api_url": "https://draft.example/proxy/v1/chat/completions",
                        "credential_source": "draft",
                        "api_key": "draft-secret",
                    }
                }
            },
        }
        radio_set = step.query_one("#setup-model-choice", RadioSet)
        ids = [
            str(getattr(button, "_model_id", button.label))
            for button in radio_set.query(RadioButton)
        ]
        assert ids == ["svc-model-a", "svc-model-b"]
        identity = step._current_discovery_key()
        assert identity is not None
        assert identity.connection_identity == (
            "custom",
            "https://draft.example/proxy/v1/chat/completions",
        )
        for rendered in (repr(identity), repr(step.__dict__), app.export_screenshot()):
            assert "draft-secret" not in rendered
            assert "ambient-secret" not in rendered
        wizard.commit_config.assert_not_awaited()
        wizard.commit_staged_provider_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_step_injected_discovery_receives_secret_free_exact_identity():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    seen: list[dict[str, object]] = []

    async def discover(**identity):
        seen.append(identity)
        return ["draft-model"]

    draft = FirstRunProviderDraft(
        "llama_cpp",
        "http://127.0.0.1:8222/v1/chat/completions",
        ProviderCredentialDraft("draft", "never-pass-this-secret", 9),
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "llama_cpp", "provider_value": "llama_cpp"}
        },
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover,
        provider_draft=draft,
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        assert seen == [
            {
                "provider": "llama_cpp",
                "endpoint": "http://127.0.0.1:8222",
                "credential_source": "draft",
                "credential_revision": 9,
            }
        ]
        row = step.query_one("#setup-model-option-0", RadioButton)
        assert getattr(row, "_model_id", None) == "draft-model"
        assert "never-pass-this-secret" not in repr(seen)
    assert step._explicit_provider_draft is None


@pytest.mark.asyncio
async def test_model_cache_separates_exact_identity_and_reconfirms_manual_entry():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
        build_first_run_model_discovery_key,
    )

    draft_a = FirstRunProviderDraft(
        "custom",
        "https://a.example/v1/chat/completions",
        ProviderCredentialDraft("none", "", 1),
    )
    draft_b = FirstRunProviderDraft(
        "custom",
        "https://b.example/v1/chat/completions",
        ProviderCredentialDraft("none", "", 2),
    )
    key_a = build_first_run_model_discovery_key(draft_a)
    key_b = build_first_run_model_discovery_key(draft_b)
    discover = AsyncMock(side_effect=AssertionError("cache miss"))
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "custom", "provider_value": "custom"}
        },
        staged_provider_draft=draft_a,
        _first_run_selected_provider_models={
            key_a: ("model-a",),
            key_b: ("model-b",),
        },
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover,
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        assert (
            getattr(
                step.query_one("#setup-model-option-0", RadioButton), "_model_id", None
            )
            == "model-a"
        )
        manual = step.query_one("#setup-model-custom", Input)
        manual.value = "manual-a"
        await pilot.pause()

        wizard.staged_provider_draft = draft_b
        step.on_show()
        await pilot.pause(0.1)
        assert manual.value == ""
        assert step.selected_model_id == ""
        assert (
            getattr(
                step.query_one("#setup-model-option-0", RadioButton), "_model_id", None
            )
            == "model-b"
        )

        wizard.staged_provider_draft = draft_a
        step.on_show()
        await pilot.pause(0.1)
        assert (
            getattr(
                step.query_one("#setup-model-option-0", RadioButton), "_model_id", None
            )
            == "model-a"
        )
        discover.assert_not_awaited()
        assert "manual-a" not in app.export_screenshot()


@pytest.mark.asyncio
async def test_model_listing_unavailable_is_disabled_and_manual_entry_remains_enabled():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryError,
        ModelDiscoveryResult,
    )
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    draft = FirstRunProviderDraft(
        "llama_cpp",
        "http://127.0.0.1:8080/v1/chat/completions",
        ProviderCredentialDraft("none", "", 0),
    )
    unavailable = ModelDiscoveryResult(
        provider="llama_cpp",
        provider_list_key="Llama.cpp",
        endpoint_fingerprint="safe-fingerprint",
        status="unsupported",
        error=ModelDiscoveryError(
            kind="unsupported_endpoint",
            message="Models route returned 404.",
            recovery_hint="Enter the model manually.",
        ),
    )
    scope_service = MagicMock(discover_models=AsyncMock(return_value=unavailable))
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={}, llm_provider_catalog_scope_service=scope_service
        ),
        wizard_data={
            "provider": {"provider_key": "llama_cpp", "provider_value": "llama_cpp"}
        },
        staged_provider_draft=draft,
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=None,
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        rows = list(step.query_one("#setup-model-choice", RadioSet).query(RadioButton))
        assert len(rows) == 1
        assert str(rows[0].label) == (
            "Model listing unavailable; enter the model ID used by this endpoint."
        )
        assert rows[0].disabled
        assert getattr(rows[0], "_model_id", None) is None
        manual = step.query_one("#setup-model-custom", Input)
        assert not manual.disabled and manual.focusable
        assert step.selected_model_id == ""

        step.set_selected_model_from_button(rows[0])
        assert step.selected_model_id == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_result", "status_selector", "expected_copy"),
    [
        (
            ModelDiscoveryResult(
                provider="custom",
                provider_list_key="custom",
                endpoint_fingerprint="safe-fingerprint",
                status="success",
                models=(),
            ),
            "#setup-model-empty",
            "(no models found — enter one below)",
        ),
        (
            ModelDiscoveryResult(
                provider="custom",
                provider_list_key="custom",
                endpoint_fingerprint="safe-fingerprint",
                status="unsupported",
                error=ModelDiscoveryError(
                    kind="unsupported_endpoint",
                    message="Models route returned 404.",
                    recovery_hint="Enter the model manually.",
                ),
            ),
            "#setup-model-listing-unavailable",
            "Model listing unavailable; enter the model ID used by this endpoint.",
        ),
        (
            ModelDiscoveryResult(
                provider="custom",
                provider_list_key="custom",
                endpoint_fingerprint="safe-fingerprint",
                status="error",
                error=ModelDiscoveryError(
                    kind="request_failed",
                    message="hostile detail " * 100,
                    recovery_hint="hostile recovery " * 100,
                ),
            ),
            "#setup-model-connection-failed",
            "Couldn't reach the server (request failed). Check it's running, then Retry — or enter a model ID below.",
        ),
        (
            RuntimeError("transport detail must not reach the UI"),
            "#setup-model-connection-failed",
            "Couldn't reach the server (request failed). Check it's running, then Retry — or enter a model ID below.",
        ),
    ],
)
async def test_mounted_provider_handoff_preserves_typed_discovery_outcome(
    monkeypatch,
    initial_result,
    status_selector: str,
    expected_copy: str,
) -> None:
    from unittest.mock import AsyncMock

    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {"custom": ["curated-model-must-not-appear"]},
    )
    succeeded = _typed_model_discovery_result("custom", "retry-exact-model")
    retry_succeeds = False

    async def discover_models(**_kwargs):
        if retry_succeeds:
            return succeeded
        if isinstance(initial_result, BaseException):
            raise initial_result
        return initial_result

    scope_service = MagicMock(discover_models=AsyncMock(side_effect=discover_models))
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {"api_url": "https://outcome.example.test/v1/chat/completions"}
        }
    }
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
        status_matches = []
        for _ in range(20):
            status_matches = list(model_step.query(status_selector))
            if status_matches:
                break
            await pilot.pause(0.05)
        assert status_matches, {
            "rows": [
                str(button.label)
                for button in model_step.query_one(
                    "#setup-model-choice", RadioSet
                ).query(RadioButton)
            ],
            "owner_state": provider_step._selected_discovery_state,
            "owner_outcomes": provider_step._selected_provider_outcomes,
            "handoff_outcomes": getattr(
                container, "_first_run_selected_provider_outcomes", None
            ),
            "calls": scope_service.discover_models.await_count,
        }
        status = status_matches[0]
        assert isinstance(status, RadioButton)
        assert status.disabled
        assert str(status.label) == expected_copy
        assert getattr(status, "_model_id", None) is None
        assert "curated-model-must-not-appear" not in app.export_screenshot()
        manual = model_step.query_one("#setup-model-custom", Input)
        assert not manual.disabled and manual.focusable

        if status_selector == "#setup-model-connection-failed":
            initial_call_count = scope_service.discover_models.await_count
            first_settings = scope_service.discover_models.await_args_list[0].kwargs[
                "staged_settings"
            ]
            retry_succeeds = True
            model_step.query_one("#setup-model-retry", Button).press()
            await pilot.pause(0.1)

            assert scope_service.discover_models.await_count == initial_call_count + 1
            assert (
                scope_service.discover_models.await_args.kwargs["staged_settings"]
                == first_settings
            )
            row = model_step.query_one("#setup-model-option-0", RadioButton)
            assert getattr(row, "_model_id", None) == "retry-exact-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_kind", "expected_error_kind", "status_selector", "expected_copy"),
    [
        (
            "404",
            "unsupported_endpoint",
            "#setup-model-listing-unavailable",
            "Model listing unavailable; enter the model ID used by this endpoint.",
        ),
        (
            "500",
            "request_failed",
            "#setup-model-connection-failed",
            "Couldn't reach the server (request failed). Check it's running, then Retry — or enter a model ID below.",
        ),
        (
            "malformed",
            "invalid_response",
            "#setup-model-connection-failed",
            "Couldn't reach the server (invalid response). Check it's running, then Retry — or enter a model ID below.",
        ),
    ],
)
async def test_mounted_real_transport_preserves_404_server_and_payload_categories(
    monkeypatch,
    response_kind: str,
    expected_error_kind: str,
    status_selector: str,
    expected_copy: str,
):
    import httpx

    import tldw_chatbook.config as config_module
    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
        ModelDiscoveryCache,
    )
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )

    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {"custom": ["curated-must-not-appear"]},
    )
    transport_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        transport_requests.append(request)
        if response_kind == "404":
            return httpx.Response(404)
        if response_kind == "500":
            return httpx.Response(500)
        return httpx.Response(200, json={"data": "not-a-list"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        async def real_discovery(**kwargs):
            return await discover_openai_compatible_models(**kwargs, client=client)

        wizard = _make_wizard()
        wizard.app_instance.app_config = {
            "providers": {"custom": []},
            "api_settings": {
                "custom": {
                    "api_url": "https://transport.example.test/v1/chat/completions"
                }
            },
        }
        shared_cache = ModelDiscoveryCache()
        local_service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: {"custom": []},
            settings_loader=lambda: wizard.app_instance.app_config,
            discovery_cache=shared_cache,
            discovery_client=real_discovery,
            environ={},
        )
        wizard.app_instance.llm_provider_catalog_scope_service = (
            LLMProviderCatalogScopeService(
                local_service=local_service,
                server_service=None,
            )
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
            for _ in range(20):
                if provider_step._selected_provider_outcomes:
                    break
                await pilot.pause(0.05)
            assert provider_step._selected_provider_outcomes

            await container._advance()
            model_step = container.steps[model_index]
            assert isinstance(model_step, ModelStep)
            for _ in range(20):
                if list(model_step.query(status_selector)):
                    break
                await pilot.pause(0.05)

            row = model_step.query_one(status_selector, RadioButton)
            assert row.disabled
            assert str(row.label) == expected_copy
            assert getattr(row, "_model_id", None) is None
            assert "curated-must-not-appear" not in app.export_screenshot()
            manual = model_step.query_one("#setup-model-custom", Input)
            assert not manual.disabled and manual.focusable
            [outcome] = provider_step._selected_provider_outcomes.values()
            assert outcome.error is not None
            assert outcome.error.kind == expected_error_kind
            assert transport_requests
            assert all(
                str(request.url) == "https://transport.example.test/v1/models"
                for request in transport_requests
            )
            assert all(
                "Authorization" not in request.headers for request in transport_requests
            )
            assert shared_cache.snapshot_count == 0


@pytest.mark.asyncio
async def test_model_connection_failure_shows_bounded_category_and_retry():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryError,
        ModelDiscoveryResult,
    )
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    draft = FirstRunProviderDraft(
        "custom",
        "https://retry.example/v1/chat/completions",
        ProviderCredentialDraft("none", "", 0),
    )
    failed = ModelDiscoveryResult(
        provider="custom",
        provider_list_key="Custom",
        endpoint_fingerprint="safe-fingerprint",
        status="error",
        error=ModelDiscoveryError(
            kind="request_failed",
            message="x" * 500,
            recovery_hint="y" * 500,
        ),
    )
    succeeded = _typed_model_discovery_result("custom", "retry-model")
    discover_models = AsyncMock(side_effect=[failed, succeeded])
    wizard = SimpleNamespace(
        app_instance=MagicMock(
            app_config={},
            llm_provider_catalog_scope_service=MagicMock(
                discover_models=discover_models
            ),
        ),
        wizard_data={
            "provider": {"provider_key": "custom", "provider_value": "custom"}
        },
        staged_provider_draft=draft,
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=None,
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        failure = step.query_one("#setup-model-connection-failed", RadioButton)
        failure_copy = str(failure.label)
        assert failure.disabled
        assert "request failed" in failure_copy
        assert len(failure_copy) < 120
        assert "x" * 20 not in failure_copy and "y" * 20 not in failure_copy

        manual = step.query_one("#setup-model-custom", Input)
        manual.value = "manual-model"
        retry = step.query_one("#setup-model-retry", Button)
        assert retry.display and retry.focusable
        retry.press()
        await pilot.pause(0.1)

        row = step.query_one("#setup-model-option-0", RadioButton)
        assert getattr(row, "_model_id", None) == "retry-model"
        assert manual.value == "manual-model"
        assert step.selected_model_id == "manual-model"
        assert retry.has_class("hidden")


@pytest.mark.asyncio
async def test_model_navigation_discards_cancellation_resistant_late_discovery():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def discover(**_identity):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            return ["late-model"]
        return ["late-model"]

    draft = FirstRunProviderDraft(
        "custom",
        "https://late.example/v1/chat/completions",
        ProviderCredentialDraft("none", "", 2),
    )
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "custom", "provider_value": "custom"}
        },
        staged_provider_draft=draft,
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover,
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        generation = step._model_load_generation
        step.on_hide()
        release.set()
        await pilot.pause(0.1)

        assert cancelled.is_set()
        assert step._model_load_generation > generation
        assert not [
            row
            for row in step.query_one("#setup-model-choice", RadioSet).query(
                RadioButton
            )
            if (row.id or "").startswith("setup-model-option-")
        ]
        assert step.selected_model_id == ""


@pytest.mark.asyncio
async def test_model_external_unmount_fences_late_discovery_without_widget_access():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        FirstRunProviderDraft,
        ProviderCredentialDraft,
    )

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def discover(**_identity):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            return ["detached-late-model"]

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "custom", "provider_value": "custom"}
        },
        staged_provider_draft=FirstRunProviderDraft(
            "custom",
            "https://unmount.example/v1/chat/completions",
            ProviderCredentialDraft("none", "", 3),
        ),
        commit_staged_provider_setup=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover,
    )

    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        generation = step._model_load_generation
        await step.remove()
        await pilot.pause(0.1)

        assert cancelled.is_set()
        assert not step.is_attached
        assert step._model_load_generation > generation
        assert step.selected_model_id == ""


@pytest.mark.asyncio
async def test_mounted_model_owner_timeout_fences_late_result_and_keeps_manual_retry(
    monkeypatch,
):
    import asyncio
    from unittest.mock import AsyncMock

    import tldw_chatbook.config as config_module
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(wizard_module, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {"custom": ["curated-timeout-must-not-appear"]},
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release_late_result = asyncio.Event()
    late_result_returned = asyncio.Event()
    asyncio.get_running_loop().call_later(3, release_late_result.set)

    async def cancellation_resistant_discovery(**_kwargs):
        started.set()
        while not release_late_result.is_set():
            try:
                await release_late_result.wait()
            except asyncio.CancelledError:
                cancelled.set()
                continue
        late_result_returned.set()
        return _typed_model_discovery_result("custom", "late-timeout-model")

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {"api_url": "https://timeout.example.test/v1/chat/completions"}
        }
    }
    scope_service = MagicMock(
        discover_models=AsyncMock(side_effect=cancellation_resistant_discovery)
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
        await asyncio.wait_for(started.wait(), timeout=2)

        await container._advance()
        assert container.current_step == model_index
        model_step = container.steps[model_index]
        assert isinstance(model_step, ModelStep)
        for _ in range(30):
            if list(model_step.query("#setup-model-connection-failed")):
                break
            await pilot.pause(0.05)

        if not list(model_step.query("#setup-model-connection-failed")):
            release_late_result.set()
            await asyncio.wait_for(late_result_returned.wait(), timeout=2)
            pytest.fail("Model owner timeout did not render bounded failure")
        status = model_step.query_one("#setup-model-connection-failed", RadioButton)
        assert str(status.label) == (
            "Couldn't reach the server (timeout). Check it's running, then Retry — or enter a model ID below."
        )
        assert status.disabled
        assert cancelled.is_set()
        assert provider_step._selected_discovery_state == "cancelled"
        assert "curated-timeout-must-not-appear" not in app.export_screenshot()
        manual = model_step.query_one("#setup-model-custom", Input)
        retry = model_step.query_one("#setup-model-retry", Button)
        assert not manual.disabled and manual.focusable
        assert "hidden" not in retry.classes

        release_late_result.set()
        await asyncio.wait_for(late_result_returned.wait(), timeout=2)
        await pilot.pause(0.1)

        assert provider_step._selected_provider_models == {}
        assert provider_step._selected_provider_outcomes == {}
        assert container._first_run_selected_provider_models == {}
        assert container._first_run_selected_provider_outcomes == {}
        status = model_step.query_one("#setup-model-connection-failed", RadioButton)
        assert "timeout" in str(status.label)
        assert "late-timeout-model" not in app.export_screenshot()


@pytest.mark.asyncio
async def test_mounted_provider_handoff_is_fenced_after_model_navigation_and_unmount(
    monkeypatch,
):
    import asyncio

    from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
        ModelDiscoveryCache,
    )

    navigation_phase = False
    started = asyncio.Event()
    navigation_cancelled = asyncio.Event()
    release_late_result = asyncio.Event()
    late_result_returned = asyncio.Event()

    discovery_requests: list[dict[str, object]] = []

    async def discover_models(**kwargs):
        discovery_requests.append(kwargs)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not navigation_phase:
                raise
            navigation_cancelled.set()
            while not release_late_result.is_set():
                try:
                    await release_late_result.wait()
                except asyncio.CancelledError:
                    continue
            late_result_returned.set()
            return _typed_model_discovery_result("custom", "detached-late-model")

    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "providers": {"custom": []},
        "api_settings": {
            "custom": {"api_url": "https://slow.example.test/v1/chat/completions"}
        },
    }
    shared_cache = ModelDiscoveryCache()
    local_service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"custom": []},
        settings_loader=lambda: wizard.app_instance.app_config,
        discovery_cache=shared_cache,
        discovery_client=discover_models,
        environ={},
    )
    wizard.app_instance.llm_provider_catalog_scope_service = (
        LLMProviderCatalogScopeService(
            local_service=local_service,
            server_service=None,
        )
    )
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        voice_index = container._step_index_for_id(STEP_VOICE)
        assert None not in {provider_index, model_index, voice_index}
        container.show_step(provider_index)
        provider_step = container.steps[provider_index]
        assert isinstance(provider_step, ProviderStep)
        provider_step.select_provider("custom")
        await asyncio.wait_for(started.wait(), timeout=2)

        await container._advance()
        await pilot.pause(0.05)
        assert container.current_step == model_index
        generation = provider_step.probe_generation
        navigation_phase = True

        await container._advance()
        assert container.current_step == voice_index
        try:
            await asyncio.wait_for(navigation_cancelled.wait(), timeout=0.5)
        except TimeoutError:
            release_late_result.set()
            pytest.fail("Provider handoff discovery was not cancelled after Model")
        assert provider_step.probe_generation > generation
        assert provider_step._selected_discovery_state == "cancelled"

        wizard.dismiss(None)
        await pilot.pause(0.1)
        detached_widget_accesses: list[str] = []

        def detached_query(*args, **kwargs):
            detached_widget_accesses.append(str(args[0]) if args else "query")
            raise AssertionError("late discovery accessed detached provider widgets")

        monkeypatch.setattr(provider_step, "query_one", detached_query)
        release_late_result.set()
        await asyncio.wait_for(late_result_returned.wait(), timeout=2)
        await pilot.pause(0.1)

        assert not provider_step.is_attached
        assert detached_widget_accesses == []
        assert discovery_requests
        assert shared_cache.snapshot_count == 0
        assert shared_cache.model_count == 0
        assert shared_cache.list() == ()
        assert provider_step._selected_provider_models == {}
        assert provider_step._selected_provider_outcomes == {}
        assert container._first_run_selected_provider_models == {}
        assert container._first_run_selected_provider_outcomes == {}
        assert not [
            worker
            for worker in app.workers
            if worker.node is provider_step
            and worker.group == "setup-provider-discovery"
        ]


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
async def test_model_step_discovery_timeout_keeps_manual_entry_and_retry(monkeypatch):
    """A timed-out exact request stays honest and leaves recovery controls usable."""
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

    async def _hangs(**_identity):
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
        [status] = list(radio_set.query(RadioButton))
        assert status.disabled
        assert "timeout" in str(status.label)
        assert getattr(status, "_model_id", None) is None
        assert not step.query_one("#setup-model-custom", Input).disabled
        retry = step.query_one("#setup-model-retry", Button)
        assert retry.display and not retry.has_class("hidden")


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
async def test_notes_step_is_informational_and_never_writes_legacy_sync_config():
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
        assert step.get_step_data() == {}
        assert not step.query("#setup-notes-enable")
        assert not step.query("#setup-notes-directory")


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
        await app.workers.wait_for_complete()
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
        # TASK-21148 (UAT S-4): long paths middle-truncate on one line
        # instead of hard-wrapping mid-character — the budget derives from
        # the step's width (review follow-up), so assert the structure:
        # the rendered path is the head and tail of the real path joined by
        # a single ellipsis (or the full path when it fits).
        assert "Config file:" in footer
        line = next(
            ln for ln in footer.splitlines() if ln.startswith("Config file:")
        )
        rendered_path = line[len("Config file: "):]
        full_path = str(scratch_config)
        if rendered_path != full_path:
            assert "…" in rendered_path
            head, _, tail = rendered_path.partition("…")
            assert full_path.startswith(head) and full_path.endswith(tail)
        assert footer.count(scratch_config.name[-12:]) >= 1


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
async def test_summary_primary_first_run_exit_buttons_set_expected_routes():
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
            "setup-exit-settings",
        }
        assert [str(button.label) for button in step.query(Button)] == [
            "Review provider setup",
            "Explore Home",
            "Review settings",
        ]
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
async def test_summary_primary_rerun_complete_actions_start_chatting():
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
        load_config=lambda: {
            "api_settings": {"openai": {"api_key": "test-key"}},
            "chat_defaults": {"provider": "openai", "model": "model-a"},
        },
        rag_deps_installed=lambda: False,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        step.on_show()
        for _ in range(20):
            if str(step.query_one("#setup-exit-chat", Button).label) == (
                "Start chatting"
            ):
                break
            await pilot.pause(0.05)
        assert {b.id for b in step.query(Button)} == {
            "setup-exit-chat",
            "setup-exit-home",
            "setup-exit-settings",
        }
        assert [str(button.label) for button in step.query(Button)] == [
            "Start chatting",
            "Explore Home",
            "Review settings",
        ]
        # See the comment in the first-run-exit test above: direct handler
        # call, not pilot.click() -- the actions row is clipped below this
        # fixed test viewport.
        step._exit_chat()
        await pilot.pause()
        assert step.get_step_data() == {"exit_route": TAB_CHAT}
        wizard.advance_programmatically.assert_called_once()


@pytest.mark.asyncio
async def test_summary_destination_button_advances_without_an_event():
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
    # _StyledHostApp: this test drives real screen coordinates via
    # pilot.click, which needs the app stylesheet loaded — the wizard
    # tracker's layout rides the bundle as BUNDLED_CSS (class-level
    # DEFAULT_CSS is barred by the parse-cache rule).
    app = _StyledHostApp(wizard)
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
                step._exit_home()
            else:
                await pilot.click("#wizard-next")
            await pilot.pause(0.2)
        from tldw_chatbook.Constants import TAB_HOME

        assert app.wizard_result == {"completed": True, "exit_route": TAB_HOME}


@pytest.mark.asyncio
async def test_ctrl_n_on_summary_does_not_bypass_explicit_destination():
    """Summary has no global Next/Finish action, so its shortcut must not
    silently complete setup without one of the three visible destinations.

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

        assert app.wizard_result == "UNSET"
        assert container.steps[container.current_step].config.id == STEP_SUMMARY
        assert app.focused is container.query_one("#setup-exit-chat", Button)


@pytest.mark.asyncio
async def test_ctrl_n_recovers_hidden_widget_focus_and_stops_at_summary():
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
    ctrl+n presses and produced NOTHING on the fourth transition. Summary
    now deliberately has no global Next/Finish action; its explicit primary
    destination must receive focus instead.

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
            # Mirrors production: preferred_focus() wins when displayed
            # (TASK-21146: Summary prefers its primary exit button), then
            # hidden (display:none / .hidden) widgets must never be focus
            # targets (TASK-1496/1498).
            preferred = (
                step.preferred_focus() if isinstance(step, SetupStep) else None
            )
            if (
                preferred is not None
                and preferred.focusable
                and preferred.display
                and not preferred.has_class("hidden")
            ):
                return preferred
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
            if expected is None:
                # Mirrors production's final fallback: a step with no
                # focusable content (keyless Protect hides its only button,
                # TASK-21148) parks focus on the nav bar so the container
                # stays in the focus chain.
                nav_next = container.query_one("#wizard-next", Button)
                assert app.focused is nav_next, (
                    f"{current!r} has no focusable widget; expected nav "
                    f"fallback focus, got {app.focused!r}"
                )
                return
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
            await pilot.press("ctrl+n")
            await pilot.pause(0.2)
            _assert_focus_on_current_step_content()
            if container.steps[container.current_step].config.id == STEP_SUMMARY:
                break
        else:
            raise AssertionError("ctrl+n never reached Summary")

        assert app.wizard_result == "UNSET"
        assert app.focused is container.query_one("#setup-exit-chat", Button)
        await pilot.press("ctrl+n")
        await pilot.pause(0.2)
        assert container.steps[container.current_step].config.id == STEP_SUMMARY
        assert app.focused is container.query_one("#setup-exit-chat", Button)


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
async def test_finalize_stages_exact_first_chat_after_successful_setup_mutation(
    monkeypatch,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.Constants import TAB_CHAT

    pending = PendingHandoffStore()
    session_owner = MagicMock()
    session_owner.eligible_console_first_chat_session_id.return_value = "session-exact"
    session_owner.prepare_console_first_chat_target.side_effect = AssertionError(
        "the producer must not prepare or mutate Console"
    )
    console = SimpleNamespace(_session=session_owner)
    app_instance = MagicMock(
        app_config={},
        pending_handoffs=pending,
        screen_stack=[console],
    )
    container = SetupWizardContainer(app_instance)
    container._complete_setup_locked = AsyncMock(return_value=True)
    container._dismiss_screen = MagicMock()
    snapshot = RuntimeConfigSnapshot(
        41,
        {
            "chat_defaults": {"provider": "llama_cpp", "model": "local-a"},
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:8080/v1",
                    "model": "local-a",
                }
            },
        },
    )
    monkeypatch.setattr(
        wizard_module,
        "get_runtime_config_snapshot",
        lambda: snapshot,
        raising=False,
    )

    await container._finalize(TAB_CHAT)

    container._complete_setup_locked.assert_awaited_once()
    session_owner.prepare_console_first_chat_target.assert_not_called()
    session_owner.eligible_console_first_chat_session_id.assert_called_once_with()
    claim = pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert claim.value == ConsoleFirstChatIntent(
        "session-exact", "llama_cpp", "local-a", 41
    )
    assert "api_url" not in repr(claim.value)
    pending.release(claim)
    container._dismiss_screen.assert_called_once_with(
        {"completed": True, "exit_route": TAB_CHAT}
    )


def test_first_chat_stage_failure_leaves_mounted_console_byte_exact(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    pending = PendingHandoffStore()
    app_instance = MagicMock(
        app_config={},
        pending_handoffs=pending,
        screen_stack=[],
    )
    console = ChatScreen(app_instance)
    store = ConsoleChatStore()
    console._console_chat_store = store
    old_defaults = ConsoleSessionSettings(
        provider="openai",
        model="old-model",
        source="derived",
    )
    session = store.create_session(
        workspace_id="global",
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    app_instance.screen_stack = [console]
    sessions_before = _first_chat_store_snapshot(store)
    active_before = store.active_session_id
    controls_before = (
        console._console_control_provider,
        console._console_control_model,
        console._console_chat_controller,
    )
    monkeypatch.setattr(
        wizard_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(
            51,
            {
                "chat_defaults": {"provider": "openai", "model": "new-model"},
                "api_settings": {"openai": {"model": "new-model"}},
            },
        ),
    )

    def fail_stage(*_args, **_kwargs):
        raise RuntimeError("stage unavailable")

    monkeypatch.setattr(pending, "stage", fail_stage)
    monkeypatch.setattr(pending, "stage_reserved_console_first_chat", fail_stage)
    container = SetupWizardContainer(app_instance)

    assert container._stage_console_first_chat_handoff() is False
    assert store.active_session_id == active_before == session.id
    assert _first_chat_store_snapshot(store) == sessions_before
    assert (
        console._console_control_provider,
        console._console_control_model,
        console._console_chat_controller,
    ) == controls_before
    assert pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT) is None


def test_generation_advance_after_stage_before_consume_never_mutates_console(
    monkeypatch,
):
    import tldw_chatbook.UI.Console_Modules.session as session_module
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    pending = PendingHandoffStore()
    app_instance = MagicMock(
        app_config={},
        pending_handoffs=pending,
        screen_stack=[],
    )
    console = ChatScreen(app_instance)
    store = ConsoleChatStore()
    console._console_chat_store = store
    old_defaults = ConsoleSessionSettings(
        provider="openai",
        model="old-model",
        source="derived",
    )
    session = store.create_session(
        workspace_id="global",
        settings=old_defaults,
        canonical_settings_baseline=old_defaults,
    )
    app_instance.screen_stack = [console]
    sessions_before = _first_chat_store_snapshot(store)
    current = [
        RuntimeConfigSnapshot(
            53,
            {
                "chat_defaults": {"provider": "openai", "model": "new-model"},
                "api_settings": {"openai": {"model": "new-model"}},
            },
        )
    ]
    monkeypatch.setattr(
        wizard_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
    )
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: current[0],
    )
    original_stage = pending.stage

    def stage_then_publish(channel, intent):
        revision = original_stage(channel, intent)
        current[0] = RuntimeConfigSnapshot(54, current[0].values)
        return revision

    monkeypatch.setattr(pending, "stage", stage_then_publish)
    container = SetupWizardContainer(app_instance)

    assert container._stage_console_first_chat_handoff() is True
    assert store.active_session_id == session.id
    assert _first_chat_store_snapshot(store) == sessions_before
    assert console._session.consume_pending_console_first_chat_intent() is False
    assert store.active_session_id == session.id
    assert _first_chat_store_snapshot(store) == sessions_before
    claim = pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert claim.value.config_revision == 53
    assert pending.release(claim) is True


@pytest.mark.asyncio
async def test_finalize_does_not_stage_first_chat_when_setup_mutation_fails():
    from tldw_chatbook.Constants import TAB_CHAT

    pending = PendingHandoffStore()
    app_instance = MagicMock(
        app_config={},
        pending_handoffs=pending,
        screen_stack=[],
    )
    container = SetupWizardContainer(app_instance)
    container._complete_setup_locked = AsyncMock(return_value=False)
    container._show_completion_save_error = MagicMock()
    container._dismiss_screen = MagicMock()

    await container._finalize(TAB_CHAT)

    assert pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT) is None
    container._show_completion_save_error.assert_called_once()
    container._dismiss_screen.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_reserves_future_target_only_without_console_owner(
    monkeypatch,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.Constants import TAB_CHAT

    pending = PendingHandoffStore()
    app_instance = MagicMock(
        app_config={},
        pending_handoffs=pending,
        screen_stack=[
            SimpleNamespace(
                _session=SimpleNamespace(
                    eligible_console_first_chat_session_id=None,
                )
            )
        ],
    )
    container = SetupWizardContainer(app_instance)
    container._complete_setup_locked = AsyncMock(return_value=True)
    container._dismiss_screen = MagicMock()
    monkeypatch.setattr(
        wizard_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(
            43,
            {
                "chat_defaults": {"provider": "openai", "model": "model-a"},
                "api_settings": {"openai": {"api_key": "test-key", "model": "model-a"}},
            },
        ),
    )

    await container._finalize(TAB_CHAT)

    claim = pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert pending.claim_reserves_new_console_session(claim) is True
    assert claim.value.session_id
    assert pending.release(claim) is True


@pytest.mark.asyncio
async def test_finalize_reserves_future_target_when_mounted_console_is_ineligible(
    monkeypatch,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.Constants import TAB_CHAT

    pending = PendingHandoffStore()
    session_owner = MagicMock()
    session_owner.eligible_console_first_chat_session_id.return_value = None
    console = SimpleNamespace(_session=session_owner)
    app_instance = MagicMock(
        app_config={},
        pending_handoffs=pending,
        screen_stack=[console],
    )
    container = SetupWizardContainer(app_instance)
    container._complete_setup_locked = AsyncMock(return_value=True)
    container._show_first_chat_handoff_error = MagicMock()
    container._dismiss_screen = MagicMock()
    monkeypatch.setattr(
        wizard_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(
            47,
            {
                "chat_defaults": {"provider": "openai", "model": "model-a"},
                "api_settings": {"openai": {"api_key": "test-key", "model": "model-a"}},
            },
        ),
    )

    await container._finalize(TAB_CHAT)

    claim = pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    assert pending.claim_reserves_new_console_session(claim) is True
    assert claim.value.provider == "openai"
    assert claim.value.model == "model-a"
    assert pending.release(claim) is True
    session_owner.eligible_console_first_chat_session_id.assert_called_once_with()
    container._show_first_chat_handoff_error.assert_not_called()
    container._dismiss_screen.assert_called_once_with(
        {"completed": True, "exit_route": TAB_CHAT}
    )


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

        summary = container.steps[container.current_step]
        assert isinstance(summary, SummaryStep)
        summary._exit_home()
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
async def test_fresh_config_without_stored_key_still_includes_protect_step():
    """TASK-21148 (UAT N-6): Protect is always on the track — a stable step
    total beats a shorter one. The keyless run renders the step's
    nothing-to-do state instead of omitting the step (which used to move
    the goalposts mid-flight the moment a key was typed)."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        assert STEP_PROTECT in container.active_ids


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
        # a truthy exit_route off the Summary step's "Start chatting" button is
        # silently dropped instead of navigating anywhere.
        # TASK-31226: the re-run wires a cancel-stays-put adapter around
        # that same handler, so cancelling a re-run returns to Settings
        # instead of routing to the Console (the boot wizard's cancel now
        # lands there).
        assert callable(callback)
        probe = MagicMock()
        screen.app._handle_first_run_wizard_result.reset_mock()
        callback(None)
        screen.app._handle_first_run_wizard_result.assert_called_once_with(
            None, cancel_to_console=False
        )
        # Dict results flow through the shared handler unchanged.
        screen.app._handle_first_run_wizard_result.reset_mock()
        callback({"completed": True, "exit_route": "chat"})
        screen.app._handle_first_run_wizard_result.assert_called_once_with(
            {"completed": True, "exit_route": "chat"}, cancel_to_console=False
        )

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

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupRadioButton

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={
            "provider": {"provider_key": "anthropic", "provider_value": "anthropic"}
        },
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(
        wizard,
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
        assert len(container.active_ids) == 6
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
        assert len(rows) == len(container.active_ids) == 6
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
        # TASK-21148 (UAT N-6): Protect is always on the track.
        assert len(rows) == 11
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
        assert len(rows) == 6
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

    # Footer: rendered by the real wizard screen. _StyledHostApp (not the
    # bare _HostApp): this test asserts real geometry — that the nav button
    # is actually painted — which only holds with the app stylesheet
    # loaded. The tracker's stacked layout rides the bundle as BUNDLED_CSS
    # (class-level DEFAULT_CSS is barred by the parse-cache rule; see
    # Tests/UI/test_widget_css_consolidation.py).
    wizard = _make_wizard()
    app = _StyledHostApp(wizard)
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
        assert progress.total_steps == 6

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
    """Welcome has one Skip action; later steps expose one Exit action."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        cancel = wizard.query_one("#wizard-cancel", Button)
        assert str(cancel.label) == "Skip setup"
        assert cancel.variant == "default"
        assert len(wizard.query("#setup-skip-entirely")) == 0

        hints = wizard.query_one("#setup-key-hints", Static)
        assert "Esc skip setup" in str(hints.render())

        container = wizard.query_one(SetupWizardContainer)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        assert provider_index is not None
        container.show_step(provider_index)
        await pilot.pause()

        assert str(cancel.label) == "Exit setup"
        assert "Esc exit setup" in str(hints.render())


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
        assert "provider, model, voice, protection" in label
        assert "recommended" in label
        container = wizard.query_one(SetupWizardContainer)
        assert len(container.active_ids) == 6
        nav = wizard.query_one(WizardNavigation)
        assert nav.total_steps == 6


@pytest.mark.asyncio
async def test_nav_text_total_is_stable_when_a_key_is_entered():
    """TASK-21148 (UAT N-6): entering a key must NOT change the step total —
    Protect is always on the track, so "Step X of Y" never moves its
    goalposts mid-flight (the old behavior this test's predecessor pinned)."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        nav = wizard.query_one(WizardNavigation)
        assert STEP_PROTECT in container.active_ids
        assert nav.total_steps == 6

        container.note_key_entered()
        await pilot.pause(0.1)

        assert STEP_PROTECT in container.active_ids
        assert nav.total_steps == 6
        progress_text = str(wizard.query_one("#wizard-progress", Static).render())
        assert "Step 1 of 6" in progress_text


# ---------------------------------------------------------------------------
# TASK-21143 (UAT S-1/M-2/M-1/N-7/P-5): the provider trust chain. A failed
# discovery probe must reach every surface that previously said "✓":
# the model step's failure row (auth points Back, Retry hidden), the
# Next gate (explicit "Continue anyway"), the tracker (attention state),
# the Provider step on return (pinned notice), and the Summary
# (row overlay + review_provider primary).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_failed_probe_drives_row_gate_tracker_and_provider_notice(
    monkeypatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {"custom": ["curated-model-must-not-appear"]},
    )
    auth_failed = ModelDiscoveryResult(
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
    scope_service = MagicMock(
        discover_models=AsyncMock(return_value=auth_failed)
    )
    wizard = _make_wizard()
    wizard.app_instance.app_config = {
        "api_settings": {
            "custom": {"api_url": "https://outcome.example.test/v1/chat/completions"}
        }
    }
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _HostApp(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        provider_step = container.steps[provider_index]
        model_step = container.steps[model_index]
        container.show_step(provider_index)
        provider_step.select_provider("custom")
        await pilot.pause(0.1)
        await container._advance()
        for _ in range(40):
            await pilot.pause(0.1)
            if model_step.query("#setup-model-connection-failed"):
                break

        # M-1/M-4: auth copy points Back; Retry is hidden (it cannot fix a
        # rejected key).
        row = model_step.query_one("#setup-model-connection-failed")
        row_text = str(row.label)
        assert "Authentication failed" in row_text and "Back" in row_text
        assert model_step.query_one("#setup-model-retry", Button).has_class(
            "hidden"
        )
        assert container.provider_probe_failure() == "authentication"

        # M-2: Next gates behind an explicit confirmation; cancel keeps
        # editing, confirm advances exactly once.
        advanced = []

        async def fake_advance():
            advanced.append(True)

        monkeypatch.setattr(container, "_advance", fake_advance)
        container.can_proceed = True
        container.advance_programmatically()
        await pilot.pause(0.2)
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            _SettlingGuardedConfirmationDialog,
        )

        assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        assert not advanced
        app.screen.query_one("#cancel-button", Button).press()
        await pilot.pause(0.2)
        assert not advanced, "cancel must not advance"
        container.advance_programmatically()
        await pilot.pause(0.2)
        assert isinstance(app.screen, _SettlingGuardedConfirmationDialog)
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause(0.3)
        assert advanced == [True], "confirm advances exactly once"

        # N-7: the tracker downgrades the visited provider step to
        # "attention" while the failure stands.
        container._rebuild_progress()
        await pilot.pause(0.1)
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
            SetupWizardProgress,
        )

        items = container.query_one(
            ".wizard-progress", SetupWizardProgress
        ).items
        states = {item.step_id: item.state for item in items}
        assert states[STEP_PROVIDER] == "attention"

        # P-5: returning to Provider surfaces the failure where the fix is.
        container.show_step(provider_index)
        await pilot.pause(0.1)
        strip = wizard.query_one("#setup-step-error-pinned", Static)
        assert "rejected" in str(strip.renderable)
        assert not strip.has_class("hidden")


@pytest.mark.asyncio
async def test_summary_overlays_probe_failure_and_flips_primary():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
        provider_probe_failure=lambda: "authentication",
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
        await pilot.pause(0.2)
        rendered = str(step.query_one("#setup-summary-rows", Static).render())
        assert "key failed an authentication check" in rendered
        assert "✓ Provider" not in rendered
        primary = step.query_one("#setup-exit-chat", Button)
        assert str(primary.label) == "Review provider setup"


# ---------------------------------------------------------------------------
# TASK-21146 (UAT H-1): the online model-list consent lives in the wizard
# Summary (default OFF, shown only while unanswered) and persists through
# the exact [model_catalog] contract the Console modal writes — so a
# completed wizard never hands the user a surprise consent modal, while
# skipping the wizard keeps the existing Console flow.
# ---------------------------------------------------------------------------


def _summary_wizard_mock():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    return SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
        provider_probe_failure=lambda: "",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [False, True])
async def test_summary_consent_checkbox_persists_answer_on_commit(allowed):
    from textual.widgets import Checkbox

    wizard = _summary_wizard_mock()
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
        await app.workers.wait_for_complete()
        box = step.query_one("#setup-summary-model-catalog-consent", Checkbox)
        assert not box.has_class("hidden"), "unanswered consent must be offered"
        assert box.value is False, "consent defaults to OFF (deny-by-default)"
        box.value = allowed
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        expected = {"refresh_consent_recorded": True}
        if not allowed:
            expected["auto_refresh_enabled"] = False
        assert committed == {"model_catalog": expected}


@pytest.mark.asyncio
async def test_summary_consent_not_reoffered_once_recorded():
    from textual.widgets import Checkbox

    wizard = _summary_wizard_mock()
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
            "model_catalog": {"refresh_consent_recorded": True},
        },
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        box = step.query_one("#setup-summary-model-catalog-consent", Checkbox)
        assert box.has_class("hidden"), "an answered consent must never re-ask"
        ok, error = await step.commit()
        assert ok, error
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_checkbox_glyphs_differ_structurally():
    """Mirror of the SetupRadioButton TASK-1497 pin: checked state must
    survive a monochrome capture (live UAT read the unchecked consent box
    as checked because stock Checkbox renders a constant X)."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupCheckbox

    class Host(App):
        def compose(self):
            yield SetupCheckbox("x", id="box")

    app = Host()
    async with app.run_test(size=(40, 10)):
        box = app.query_one("#box", SetupCheckbox)
        box._button
        unchecked = box.BUTTON_INNER
        box.value = True
        box._button
        checked = box.BUTTON_INNER
        assert unchecked != checked
        assert checked == "✓"


# ---------------------------------------------------------------------------
# TASK-21150 item (a): saying yes on the Summary must behave like saying yes
# to the Console modal — which refreshes the catalogs immediately. Recording
# consent alone left the first session running on stale lists with no modal
# left to trigger the fetch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_summary_consent_allow_kicks_the_catalog_refresh(allowed):
    from textual.widgets import Checkbox

    wizard = _summary_wizard_mock()
    refreshed: list[str] = []
    wizard.request_model_catalog_refresh = lambda: refreshed.append("kick")
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
        await pilot.pause(0.2)
        step.query_one("#setup-summary-model-catalog-consent", Checkbox).value = allowed
        ok, error = await step.commit()
        assert ok, error

    if allowed:
        assert refreshed == ["kick"], "allow must refresh the catalogs this session"
    else:
        assert refreshed == [], "deny must never reach the network"


# ---------------------------------------------------------------------------
# TASK-21150 item (b): expanding "show all" rebuilds the radio list, and the
# row the user already picked must come back pressed — otherwise the screen
# says "nothing selected" while the step still holds the selection, and a
# resumed draft cannot match the row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_show_all_rebuilds_keep_the_selected_theme_and_card_pressed():
    from textual.widgets import Button, RadioButton, RadioSet

    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        appearance = wizard.query_one(AppearanceStep)

        # Pick a theme and a card from the curated (short) lists.
        theme_rows = [
            b
            for b in appearance.query("#setup-theme-choice RadioButton")
            if getattr(b, "_theme_name", "")
        ]
        assert theme_rows
        chosen_theme = theme_rows[-1]
        chosen_theme.value = True
        await pilot.pause(0.1)

        card_rows = [
            b
            for b in appearance.query("#setup-splash-choice RadioButton")
            if getattr(b, "_card_name", "")
        ]
        assert card_rows
        chosen_card_name = getattr(card_rows[-1], "_card_name")
        card_rows[-1].value = True
        await pilot.pause(0.1)

        chosen_theme_name = appearance.selected_theme
        assert chosen_theme_name and appearance.selected_splash_card == chosen_card_name

        # Expand both full lists.
        appearance.query_one("#setup-theme-show-all", Button).press()
        await pilot.pause(0.3)
        appearance.query_one("#setup-splash-show-all", Button).press()
        await pilot.pause(0.3)

        # The selection survives in state AND is pressed in the rebuilt lists.
        assert appearance.selected_theme == chosen_theme_name
        assert appearance.selected_splash_card == chosen_card_name

        theme_pressed = appearance.query_one("#setup-theme-choice", RadioSet).pressed_button
        assert theme_pressed is not None, "no theme row pressed after show-all"
        assert getattr(theme_pressed, "_theme_name", "") == chosen_theme_name

        card_pressed = appearance.query_one("#setup-splash-choice", RadioSet).pressed_button
        assert card_pressed is not None, "no card row pressed after show-all"
        assert getattr(card_pressed, "_card_name", "") == chosen_card_name


# ---------------------------------------------------------------------------
# Qodo review of PR #2131 (testability): the consent test above stubs
# request_model_catalog_refresh, so it only proves SummaryStep.commit calls
# *a* callback. These exercise the real boundary — the container's forwarder
# and the app's public seam — including the dispatch configuration that
# makes the wizard path and the Console modal mutually exclusive.
# ---------------------------------------------------------------------------


def test_refresh_model_catalogs_now_dispatches_the_shared_exclusive_worker():
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Constants import MODEL_CATALOG_REFRESH_WORKER_GROUP

    calls: list[dict] = []

    class _App:
        _startup_model_catalog_refresh_scheduled = False

        def run_worker(self, work, **kwargs):
            calls.append({"work": work, **kwargs})

        async def _refresh_model_catalogs(self):  # pragma: no cover - identity only
            return None

        refresh_model_catalogs_now = (
            TldwCli.refresh_model_catalogs_now  # the real implementation
        )

    app = _App()
    app.refresh_model_catalogs_now()

    assert len(calls) == 1, "consent must dispatch exactly one refresh"
    dispatched = calls[0]
    assert dispatched["work"] == app._refresh_model_catalogs
    assert dispatched["exclusive"] is True
    # Same group as the Console modal's allow path, so the two can never
    # run concurrently — and it comes from the shared constant, not a
    # hand-typed string.
    assert dispatched["group"] == MODEL_CATALOG_REFRESH_WORKER_GROUP
    # The startup path must not then queue a second refresh this launch.
    assert app._startup_model_catalog_refresh_scheduled is True


@pytest.mark.asyncio
async def test_summary_consent_reaches_the_app_seam_through_the_real_chain():
    """No stubbed forwarder: Summary -> container -> app_instance."""
    from textual.widgets import Checkbox

    reached: list[str] = []

    class _AppInstance:
        app_config: dict = {}

        def refresh_model_catalogs_now(self) -> None:
            reached.append("app")

    wizard = _summary_wizard_mock()
    wizard.app_instance = _AppInstance()
    # The real forwarder, bound to our fake container.
    wizard.request_model_catalog_refresh = (
        SetupWizardContainer.request_model_catalog_refresh.__get__(wizard)
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
        await pilot.pause(0.2)
        step.query_one("#setup-summary-model-catalog-consent", Checkbox).value = True
        ok, error = await step.commit()
        assert ok, error

    assert reached == ["app"], "consent never reached the app's refresh seam"


def test_real_sized_provider_catalog_reaches_the_picker_intact():
    """A production-sized catalog must arrive whole, not rejected or trimmed.

    Two live incidents in one line. api.openai.com returns 128 models for
    an ordinary account; this extractor bounded itself by the *probe's*
    MODEL_IDS_MAX_COUNT (100) rather than the discovery limit.

    First it raised ValueError on the over-bound result, and the caller
    folds any raise into a failed discovery -- so a successful 128-model
    discovery surfaced on the Model step as "Couldn't reach the server
    (request failed)" with a valid API key.

    Truncating to 100 instead was still wrong: OpenAI returns models in
    roughly chronological order, so the 28 dropped were the newest --
    gpt-5.4, gpt-5.4-pro and gpt-5.3-chat-latest were all lost, i.e.
    exactly the models a user opens the picker to find.

    Every fixture in this file is far under the bound, so nothing caught
    either one. Malformed-shape rejection is unchanged and still covered
    by test_model_discovery_result_rejects_malformed_payloads.
    """
    from tldw_chatbook.Chat.local_server_discovery import MODEL_IDS_MAX_COUNT
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        DISCOVERED_MODEL_MAX_COUNT,
    )
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _model_ids_from_discovery_result,
    )

    observed_openai_catalog_size = 128
    assert observed_openai_catalog_size > MODEL_IDS_MAX_COUNT, (
        "this regression only bites when the real catalog exceeds the probe bound"
    )
    assert observed_openai_catalog_size <= DISCOVERED_MODEL_MAX_COUNT, (
        "the discovery bound must stay above real provider catalogs"
    )
    newest_model = "gpt-5.4-pro"
    ids = [f"model-{index}" for index in range(observed_openai_catalog_size - 1)]
    ids.append(newest_model)  # newest last, as the real API orders them
    result = _typed_model_discovery_result("openai", *ids)

    model_ids = _model_ids_from_discovery_result(result)

    assert len(model_ids) == observed_openai_catalog_size
    assert model_ids[0] == "model-0"
    assert newest_model in model_ids, "a trim would drop the newest models first"
    assert len(set(model_ids)) == len(model_ids)


def test_typed_catalog_over_the_discovery_ceiling_is_rejected():
    """The relaxed bound is a ceiling, and it fails closed above it.

    Rejecting rather than truncating is deliberate (Qodo review, PR #2158):
    a truncating loop would stop validating once the ceiling was reached,
    so a malformed DiscoveredModel in the tail would slip past this
    helper's reject-malformed contract. Discovery itself fails closed above
    DISCOVERED_MODEL_MAX_COUNT, so an over-ceiling typed result never came
    from that path.
    """
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        DISCOVERED_MODEL_MAX_COUNT,
    )
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _model_ids_from_discovery_result,
    )

    oversized = _typed_model_discovery_result(
        "openai",
        *[f"model-{index}" for index in range(DISCOVERED_MODEL_MAX_COUNT + 25)],
    )

    with pytest.raises(ValueError, match="discovery"):
        _model_ids_from_discovery_result(oversized)


def test_malformed_entry_in_the_tail_is_still_rejected():
    """Every entry is validated, not just those before a truncation point."""
    from dataclasses import replace

    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _model_ids_from_discovery_result,
    )

    result = _typed_model_discovery_result(
        "openai", *[f"model-{index}" for index in range(128)]
    )
    poisoned = result.models[:-1] + (
        replace(result.models[-1], model_id="unsafe\nmodel"),
    )

    with pytest.raises(ValueError, match="discovery"):
        _model_ids_from_discovery_result(replace(result, models=poisoned))


# ---------------------------------------------------------------------------
# TASK-23091: ModelStep's handoff paths flattened every failure to
# "request failed", so an authentication rejection told the user to check
# whether their server was running. That wording masked the real cause for
# most of the TASK-23089 investigation.
# ---------------------------------------------------------------------------


def _errored_discovery_result(kind: str):
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryError,
        ModelDiscoveryResult,
    )

    return ModelDiscoveryResult(
        provider="openai",
        provider_list_key="openai",
        endpoint_fingerprint="https://api.openai.com/v1",
        status="error",
        error=ModelDiscoveryError(
            kind=kind,
            message="The models endpoint rejected the configured credentials.",
            recovery_hint="Check the API key configured for this provider.",
        ),
    )


def test_handed_off_auth_failure_keeps_its_authentication_category():
    """A rejected key must not be reported as an unreachable server.

    ProviderStep records the typed outcome even on handoff paths where
    ModelStep never receives one directly. Flattening those to
    "request failed" rendered "Couldn't reach the server ... Check it's
    running" for a 401 -- unactionable, and the opposite of the fix the
    user needs (the key lives one step Back).
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _handed_off_failure_category,
    )

    key = object()
    owner = SimpleNamespace(
        _selected_provider_outcomes={key: _errored_discovery_result(
            "missing_credentials"
        )}
    )

    category = _handed_off_failure_category(owner, key)

    assert category == "authentication"
    # The category is only useful if it still reaches the auth copy branch.
    assert wizard_state.classify_discovery_failure("connection_failed", category) == (
        wizard_state.PROVIDER_PROBE_AUTH
    )


def test_handed_off_failure_without_a_recorded_outcome_stays_generic():
    """Without a typed outcome there is nothing more specific to say."""
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _handed_off_failure_category,
    )

    for owner in (None, SimpleNamespace(_selected_provider_outcomes={})):
        category = _handed_off_failure_category(owner, object())
        assert category == "request failed"
        assert wizard_state.classify_discovery_failure(
            "connection_failed", category
        ) == wizard_state.PROVIDER_PROBE_CONNECTION


def test_handed_off_failure_category_survives_a_malformed_outcome():
    """A junk recorded outcome degrades to the generic wording, never raises."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        _handed_off_failure_category,
    )

    key = object()
    owner = SimpleNamespace(_selected_provider_outcomes={key: object()})

    assert _handed_off_failure_category(owner, key) == "request failed"
