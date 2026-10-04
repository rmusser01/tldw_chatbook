"""TASK-34100.8: the Voice step under SF3 ("untouched means unwritten").

Mounted behaviour for every acceptance criterion: an untouched step writes
nothing (byte-identical [app_tts] / [tts_settings] through the real config
writer and STTS handler), 'No voice for now', the service status line,
classified test failures, the inline OpenAI key, focus and Enter after a test,
Advanced edits making the service Custom, the pinned error clearing, and the
copy. The pure decisions are in ``test_first_run_voice_prefill.py``.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import toml
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, RadioButton, Select, Static

import tldw_chatbook.UI.Wizards.first_run_voice_step as voice_step_module
from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from Tests.Wizards.test_first_run_setup_wizard import _HostApp, _make_wizard, _StepHost
from tldw_chatbook import config as config_module
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_voice_status as voice_status
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_VOICE, TRACK_QUICK
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    SetupWizardContainer,
    VoiceSetupStep,
)

pytestmark = pytest.mark.bootstrap_profile

_OFFICIAL = "https://api.openai.com/v1/audio/speech"
_WORKING_OPENAI_APP_TTS = {
    "OPENAI_BASE_URL": _OFFICIAL,
    "OPENAI_AUTH_MODE": "api_key",
    "default_provider": "openai",
    "default_model_mode": "exact",
    "default_model": "tts-1-hd",
    "default_voice_mode": "exact",
    "default_voice": "shimmer",
    "default_format": "mp3",
    "default_speed": 1.0,
}
_WORKING_OPENAI_TTS_SETTINGS = {
    "default_tts_provider": "openai",
    "default_tts_voice": "shimmer",
    "default_openai_tts_model": "tts-1-hd",
    "default_openai_tts_output_format": "mp3",
    "default_openai_tts_speed": 1.0,
}


def _step(app_config: dict | None = None, **wizard_fields) -> VoiceSetupStep:
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config or {}),
        wizard_data={},
        **wizard_fields,
    )
    return VoiceSetupStep(
        wizard=wizard,
        config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4),
    )


class _CapturingHost(_StepHost):
    saved: STTSSettingsSaveEvent | None = None

    @on(STTSSettingsSaveEvent)
    def capture(self, event: STTSSettingsSaveEvent) -> None:
        self.saved = event


async def _settle(pilot, done, *, tries: int = 40) -> None:
    """Pause until ``done()`` holds (bounded): a loaded runner is slow."""
    for _ in range(tries):
        await pilot.pause(0.05)
        if done():
            return


def _raw(app_tts: dict) -> dict:
    return {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": app_tts}}


# -- AC#1: an untouched step writes nothing ------------------------------------


def _config_path() -> Path:
    return Path(os.environ["TLDW_CONFIG_PATH"])


def _replace_voice_tables(app_tts: dict | None, tts_settings: dict | None) -> None:
    """Rewrite the real (bootstrap) config with exactly these voice tables."""
    path = _config_path()
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name, table in (("app_tts", app_tts), ("tts_settings", tts_settings)):
        if table is None:
            data.pop(name, None)
        else:
            data[name] = dict(table)
    config_module.replace_cli_config_serialized(toml.dumps(data), create_backup=False)


class _RealVoiceSaveApp(App):
    """Routes the step's save through the real STTS handler and config writer."""

    def __init__(self, step: VoiceSetupStep, service: TTSService) -> None:
        super().__init__()
        self._step = step
        self.saves: list[STTSSettingsSaveEvent] = []
        self.stts_handler = STTSEventHandler(self)
        self.stts_handler._stts_service = service

    def compose(self) -> ComposeResult:
        yield self._step

    @on(STTSSettingsSaveEvent)
    async def handle_voice_save(self, event: STTSSettingsSaveEvent) -> None:
        self.saves.append(event)
        await self.stts_handler.handle_settings_save(event)


def _service() -> TTSService:
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", FakeAdapterFactory("openai"), {}),),
        aliases={},
    )
    return TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot.from_settings(
            config_module.settings
        ),
    )


async def _commit_through_real_writer(app_tts, tts_settings, *, act=None):
    """Mount the step on the real config, optionally act, then press Next."""
    _replace_voice_tables(app_tts, tts_settings)
    before = _config_path().read_bytes()
    service = _service()
    step = _step(dict(config_module.settings))
    app = _RealVoiceSaveApp(step, service)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            if act is not None:
                await act(step, pilot)
            outcome = await step.commit()
            await pilot.pause()
            after = _config_path().read_bytes()
            return step, app, outcome, before, after
    finally:
        await service.close()
        await service.wait_closed()


def _voice_tables(raw: bytes) -> tuple[object, object]:
    data = tomllib.loads(raw.decode("utf-8"))
    return data.get("app_tts"), data.get("tts_settings")


@pytest.mark.asyncio
async def test_rerun_over_a_working_openai_voice_writes_nothing(monkeypatch) -> None:
    """The power-user re-run: Next used to swap the official endpoint for
    the PocketTTS port with auth 'none' and keep tts-1-hd / shimmer."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    step, app, outcome, before, after = await _commit_through_real_writer(
        _WORKING_OPENAI_APP_TTS, _WORKING_OPENAI_TTS_SETTINGS
    )

    assert outcome == (True, "")
    assert app.saves == []
    assert after == before
    assert step._preset == voice_state.VOICE_PRESET_OFFICIAL_OPENAI


@pytest.mark.asyncio
async def test_first_run_with_an_openai_key_and_no_voice_table_writes_nothing(
    monkeypatch,
) -> None:
    """A cloud user's working (fallback) OpenAI speech survives the first run."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    step, app, outcome, before, after = await _commit_through_real_writer(None, None)

    assert outcome == (True, "")
    assert app.saves == []
    assert after == before
    assert _voice_tables(after) == (None, None)
    assert step._preset == voice_state.VOICE_PRESET_NONE


@pytest.mark.asyncio
async def test_rerun_over_a_custom_endpoint_writes_nothing() -> None:
    custom = {
        "OPENAI_BASE_URL": "http://127.0.0.1:8880/v1/audio/speech",
        "OPENAI_AUTH_MODE": "none",
        "default_provider": "openai",
        "default_model": "kokoro",
        "default_voice": "af_bella",
        "default_format": "flac",
        "default_speed": 1.25,
    }
    shown: list[str] = []

    async def read_prefill(step, pilot):
        shown.append(step.query_one("#setup-voice-endpoint", Input).value)

    step, app, outcome, before, after = await _commit_through_real_writer(
        custom, None, act=read_prefill
    )

    assert outcome == (True, "")
    assert app.saves == []
    assert after == before
    assert step._preset == voice_state.VOICE_PRESET_CUSTOM
    assert shown == [custom["OPENAI_BASE_URL"]]


@pytest.mark.asyncio
async def test_rerun_prefill_names_the_saved_voice(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    step = _step(_raw(_WORKING_OPENAI_APP_TTS))
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        assert step.query_one("#setup-voice-preset-official", RadioButton).value
        assert str(step.query_one("#setup-voice-status", Static).render()) == (
            "Current voice: OpenAI · tts-1-hd · shimmer — unchanged unless you edit it."
        )
        assert step.query_one("#setup-voice-default", Checkbox).value is True
        assert step.query_one("#setup-voice-voice", Input).value == "shimmer"


@pytest.mark.asyncio
async def test_unticked_pick_writes_the_presets_axes_but_no_default_selection(
    monkeypatch,
) -> None:
    """A PocketTTS URL is never paired with tts-1-hd / shimmer / mp3, and no
    default_provider is written that nobody chose."""

    async def pick_pocket(step, pilot):
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        assert step.query_one("#setup-voice-default", Checkbox).value is False

    step, app, outcome, before, after = await _commit_through_real_writer(
        None, None, act=pick_pocket
    )

    assert outcome == (True, "")
    app_tts, tts_settings = _voice_tables(after)
    assert app_tts["OPENAI_BASE_URL"] == voice_state.POCKET_TTS_ENDPOINT
    assert app_tts["OPENAI_AUTH_MODE"] == "none"
    assert (app_tts["default_model"], app_tts["default_voice"]) == (
        "pocket-tts",
        "alba",
    )
    assert app_tts["default_format"] == "wav"
    assert "default_provider" not in app_tts
    assert "default_tts_provider" not in (tts_settings or {})
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            tomllib.loads(after.decode()), {}, rag_deps_installed=False
        )
    }
    assert rows["Voice"].detail == (
        "PocketTTS · pocket-tts · alba (saved, not the default voice)"
    )


@pytest.mark.asyncio
async def test_untouched_fresh_step_leaves_the_summary_without_a_voice_tick() -> None:
    step, app, outcome, before, after = await _commit_through_real_writer(None, None)

    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            tomllib.loads(after.decode()), {}, rag_deps_installed=False
        )
    }
    assert rows["Voice"].state == wizard_state.ROW_DEFAULT
    assert rows["Voice"].detail == "not set up (optional)"


# -- AC#2: 'No voice for now', the service line, classified failures ---------


@pytest.mark.asyncio
async def test_no_voice_for_now_leads_and_hides_the_try_it_controls() -> None:
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        buttons = list(step.query_one("#setup-voice-preset").query(RadioButton))
        assert str(buttons[0].label) == "No voice for now"
        assert buttons[0].value is True
        assert step.query_one("#setup-voice-body").display is False
        assert (
            str(step.query_one("#setup-voice-service-status", Static).render())
            == voice_status.NO_VOICE_COPY
        )


@pytest.mark.asyncio
async def test_service_line_reports_the_probe_and_the_probe_is_one_worker_connect(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def probe(url: str) -> bool:
        calls.append(url)
        return False

    monkeypatch.setattr(voice_step_module, "probe_endpoint_reachable", probe)
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause(0.2)

        line = str(step.query_one("#setup-voice-service-status", Static).render())
        assert line.startswith("PocketTTS — not running at 127.0.0.1:8000.")
        assert calls == [voice_state.POCKET_TTS_ENDPOINT]

        monkeypatch.setattr(
            voice_step_module, "probe_endpoint_reachable", lambda _u: True
        )
        step._start_probe()
        await pilot.pause(0.2)
        assert "PocketTTS — a server is listening at 127.0.0.1:8000." in str(
            step.query_one("#setup-voice-service-status", Static).render()
        )


@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_a_failed_test_names_the_cause() -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step.query_one(
            "#setup-voice-endpoint", Input
        ).value = f"http://127.0.0.1:{port}/tts"
        await pilot.pause()
        step.query_one("#setup-voice-test", Button).press()
        for _ in range(50):
            await pilot.pause(0.05)
            status = str(step.query_one("#setup-voice-status", Static).render())
            if status.startswith("Test failed"):
                break

        assert status.startswith("Test failed — ")
        assert f"isn't running at 127.0.0.1:{port}" in status
        # The edit made the service Custom; a /tts address is still
        # pocket-tts's own API, so the advice names it.
        assert step._preset == voice_state.VOICE_PRESET_CUSTOM
        assert status.startswith(f"Test failed — PocketTTS isn't running at 127.0.0.1:{port}.")
        assert "Not tested yet" not in status


# -- AC#3: OpenAI without a key never strands the user -------------------------


@pytest.mark.asyncio
async def test_a_pasted_key_is_staged_tested_and_saved_where_settings_saves_it(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    noted: list[bool] = []
    seen_credentials: list[object] = []

    async def sample(draft, *, credential=None, **_kwargs):
        seen_credentials.append(credential)
        return voice_state.VoiceSampleResult(b"valid", "audio/mpeg", "mp3", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    step = _step(note_key_entered=lambda: noted.append(True))
    app = _CapturingHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-official")
        await pilot.pause()
        assert step.query_one("#setup-voice-key-row").display is True
        assert step.query_one("#setup-voice-test", Button).disabled is True

        step.query_one("#setup-voice-api-key", Input).value = "sk-test-pasted-key"
        await pilot.pause()
        assert step.query_one("#setup-voice-test", Button).disabled is False
        assert "sk-test-pasted-key" not in repr(step.get_step_data())

        step.query_one("#setup-voice-test", Button).press()
        await pilot.pause(0.2)
        assert seen_credentials == ["sk-test-pasted-key"]

        commit = asyncio.create_task(step.commit())
        await pilot.pause()
        assert app.saved is not None
        assert app.saved.settings["openai_api_key"] == "sk-test-pasted-key"
        step.receive_stts_settings_save_result(
            _applied_result(app.saved, defaults_activated=True)
        )
        assert await commit == (True, "")
        assert noted == [True]  # Protect now offers to encrypt it


@pytest.mark.asyncio
async def test_the_provider_steps_staged_openai_key_is_recognised(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    staged = wizard_state.FirstRunProviderDraft(
        provider="openai",
        endpoint="",
        credential=wizard_state.ProviderCredentialDraft("draft", "sk-test-provider"),
    )
    step = _step(staged_provider_draft=staged, provider_setup_committed=False)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-official")
        await pilot.pause()

        assert step.query_one("#setup-voice-key-row").display is False
        assert step.query_one("#setup-voice-test", Button).disabled is False
        assert "key found" in str(
            step.query_one("#setup-voice-service-status", Static).render()
        )


def _applied_result(event: STTSSettingsSaveEvent, *, defaults_activated=None):
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
        STTSSettingsSaveResult,
    )

    return STTSSettingsSaveResult(
        request_id=event.request_id,
        persisted=True,
        provider_statuses={"openai": "applied"},
        provider_configuration_revisions={"openai": 1},
        provider_runtime_revisions={"openai": 1},
        defaults_activated=defaults_activated,
    )


# -- AC#4: focus, sample selection, Enter ------------------------------------------


@pytest.mark.asyncio
async def test_focus_returns_to_test_and_hear_after_a_test(monkeypatch) -> None:
    release = asyncio.Event()

    async def sample(*_args, **_kwargs):
        await release.wait()
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        test = step.query_one("#setup-voice-test", Button)
        test.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert test.disabled is True
        assert step.app.focused is not test  # disabling dropped focus

        release.set()
        await pilot.pause(0.2)
        assert test.disabled is False
        assert step.app.focused is test


@pytest.mark.asyncio
async def test_sample_text_does_not_select_all_on_focus() -> None:
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert step.query_one("#setup-voice-sample", Input).select_on_focus is False


@pytest.mark.asyncio
async def test_enter_in_sample_text_tests_instead_of_advancing(monkeypatch) -> None:
    calls: list[object] = []

    async def sample(draft, **_kwargs):
        calls.append(draft.sample_text)
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        index = container._step_index_for_id(STEP_VOICE)
        container.show_step(index)
        await pilot.pause()
        step = container.steps[index]
        hints = str(wizard.query_one("#setup-key-hints", Static).render())
        assert hints.startswith("Enter in Sample text tests it")
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step.query_one("#setup-voice-sample", Input).focus()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert calls == [voice_state.DEFAULT_SAMPLE_TEXT]
        assert container.current_step == index


# -- AC#5: an Advanced edit makes the service Custom and is kept -------------------


@pytest.mark.asyncio
async def test_an_advanced_edit_switches_to_custom_and_survives_a_round_trip() -> None:
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        endpoint = step.query_one("#setup-voice-endpoint", Input)
        endpoint.value = "http://127.0.0.1:8766/tts"
        await pilot.pause()

        assert step._preset == voice_state.VOICE_PRESET_CUSTOM
        assert step.query_one("#setup-voice-preset-custom", RadioButton).value is True
        assert step.query_one("#setup-voice-preset-pocket", RadioButton).value is False

        step._select_preset_button("setup-voice-preset-official")
        await pilot.pause()
        assert endpoint.value == _OFFICIAL
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        assert endpoint.value == "http://127.0.0.1:8766/tts"


@pytest.mark.asyncio
async def test_typing_an_endpoint_keeps_focus_through_the_switch_to_custom() -> None:
    """Typing an endpoint from the keyboard: the keystroke that makes the
    service Custom re-points the Voice picker at "Other…", and that must not
    pull focus out of Endpoint, so the rest of the URL still lands there."""
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step.query_one("#setup-voice-advanced").collapsed = False
        await pilot.pause()
        endpoint = step.query_one("#setup-voice-endpoint", Input)
        endpoint.focus()
        await pilot.pause()

        await pilot.press("end", "ctrl+u")
        await pilot.pause()
        await pilot.press(*"http://127.0.0.1:8766/tts")
        await pilot.pause()

        assert step._preset == voice_state.VOICE_PRESET_CUSTOM
        assert step.app.focused is endpoint
        assert endpoint.value == "http://127.0.0.1:8766/tts"
        assert step.query_one("#setup-voice-voice", Input).value == (
            voice_state.POCKET_TTS_VOICE
        )


@pytest.mark.asyncio
async def test_picking_one_of_the_services_own_voices_keeps_the_service() -> None:
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step.query_one("#setup-voice-voice-select", Select).value = "marius"
        await pilot.pause()

        assert step.query_one("#setup-voice-voice", Input).value == "marius"
        assert step._preset == voice_state.VOICE_PRESET_POCKET_TTS


# -- AC#6: the pinned error clears on any change -----------------------------------


@pytest.mark.asyncio
async def test_a_stale_step_error_clears_on_input_and_service_change() -> None:
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        index = container._step_index_for_id(STEP_VOICE)
        container.show_step(index)
        await pilot.pause()
        step = container.steps[index]
        strip = wizard.query_one("#setup-step-error-pinned", Static)

        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step.show_step_error("Speed must be between 0.25 and 4.0.")
        step.query_one("#setup-voice-speed", Input).value = "1.5"
        await _settle(pilot, lambda: str(strip.render()) == "")
        assert str(strip.render()) == ""
        assert strip.has_class("hidden")

        step.show_step_error("Something else.")
        step._select_preset_button("setup-voice-preset-official")
        await _settle(pilot, lambda: str(strip.render()) == "")
        assert str(strip.render()) == ""


# -- AC#7: copy, the default box, one primary, pickers -----------------------------


@pytest.mark.asyncio
async def test_voice_copy_controls_and_auto_tick(monkeypatch) -> None:
    async def sample(*_args, **_kwargs):
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()

        status = str(step.query_one("#setup-voice-status", Static).render())
        assert status == voice_status.DEFAULT_STATUS_COPY
        assert "save" not in status.casefold()
        test = step.query_one("#setup-voice-test", Button)
        assert test.variant == "default"
        default = step.query_one("#setup-voice-default", Checkbox)
        assert str(default.label) == "Use this voice when Chatbook reads replies aloud"
        assert "Speak replies" in str(
            step.query_one("#setup-voice-default-help", Static).render()
        )
        # Live finding (g8): the long label was cut to "…from the Provider
        # st…" even at 160 columns. The option names the key; a help line
        # says where it comes from.
        assert str(step.query_one("#setup-voice-auth-key", RadioButton).label) == (
            "API key (your OpenAI key)"
        )
        auth_help = str(step.query_one("#setup-voice-auth-help", Static).render())
        assert "Provider step" in auth_help
        assert "OPENAI_API_KEY" in auth_help
        voice_select = step.query_one("#setup-voice-voice-select", Select)
        format_select = step.query_one("#setup-voice-format-select", Select)
        assert ("Other…", "__other__") in [
            (str(prompt), value) for prompt, value in voice_select._options
        ]
        assert format_select.value == "wav"

        assert default.value is False
        test.press()
        await pilot.pause(0.2)
        assert str(step.query_one("#setup-voice-status", Static).render()) == (
            voice_status.PLAYED_COPY
        )
        assert default.value is True  # ticked itself; counts as acting
        assert step._tested_this_run is True
