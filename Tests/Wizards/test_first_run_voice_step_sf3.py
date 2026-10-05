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
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import toml
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

import tldw_chatbook.UI.Wizards.first_run_voice_step as voice_step_module
from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from Tests.Wizards.test_first_run_setup_wizard import (
    _HostApp,
    _make_wizard,
    _StepHost,
    _StyledHostApp,
)
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
    # kokoro too: a re-run over another provider's default must validate it.
    registry = TTSAdapterRegistry(
        specs=tuple(
            provider_spec(provider, FakeAdapterFactory(provider), {})
            for provider in ("openai", "kokoro")
        ),
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


#: Settings ▸ Speech & TTS used to save this: a pocket-tts /tts address with
#: a format the server never returns (review round 2, G8-R2-F1 / G8-R2-F3).
_POCKET_TTS_MP3_APP_TTS = {
    "OPENAI_BASE_URL": "http://127.0.0.1:8766/tts",
    "OPENAI_AUTH_MODE": "none",
    "default_provider": "openai",
    "default_model": "pocket-tts",
    "default_voice": "alba",
    "default_format": "mp3",
}


@pytest.mark.asyncio
async def test_an_untouched_rerun_over_an_unspeakable_saved_voice_writes_nothing() -> (
    None
):
    """Review round 2 (G8-R2-F1): Next re-validated the saved draft before
    the delta gate, so an untouched step was refused ("PocketTTS returns WAV
    audio only …") although it writes nothing, and retrying Next could never
    get past it."""
    step, app, outcome, before, after = await _commit_through_real_writer(
        _POCKET_TTS_MP3_APP_TTS, None
    )

    assert step._preset == voice_state.VOICE_PRESET_CUSTOM
    assert outcome == (True, "")
    assert app.saves == []
    assert after == before


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


def _summary_voice(raw: bytes):
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            tomllib.loads(raw.decode()), {}, rag_deps_installed=False
        )
    }
    return rows["Voice"]


def _default_box(step) -> tuple[bool, bool, str]:
    box = step.query_one("#setup-voice-default", Checkbox)
    help_line = str(step.query_one("#setup-voice-default-help", Static).render())
    return box.value, box.disabled, help_line


@pytest.mark.asyncio
async def test_a_fresh_pick_is_the_reply_voice_and_the_box_says_so() -> None:
    """Review round 1 (F1 / G8-V1-F1): with no default_provider saved, the
    runtime reads replies with the OpenAI slot. An unticked box used to save
    the voice anyway while the box, the Summary and the User Guide all said
    it was "not the default voice"."""
    shown: list[tuple[bool, bool, str]] = []

    async def pick_pocket(step, pilot):
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        shown.append(_default_box(step))

    step, app, outcome, before, after = await _commit_through_real_writer(
        None, None, act=pick_pocket
    )

    [(ticked, locked, help_line)] = shown
    assert (ticked, locked) == (True, True)
    assert help_line.startswith(
        "Replies will use this voice — no other voice is set up."
    )
    assert outcome == (True, "")
    app_tts, _tts_settings = _voice_tables(after)
    assert app_tts["OPENAI_BASE_URL"] == voice_state.POCKET_TTS_ENDPOINT
    assert app_tts["default_provider"] == "openai"
    assert (
        app_tts["default_model"],
        app_tts["default_voice"],
        app_tts["default_format"],
    ) == ("pocket-tts", "alba", "wav")
    voice_row = _summary_voice(after)
    assert (voice_row.state, voice_row.detail) == (
        wizard_state.ROW_CONFIGURED,
        "PocketTTS · pocket-tts · alba",
    )


@pytest.mark.asyncio
async def test_a_rerun_pick_over_the_openai_reply_voice_says_it_replaces_it(
    monkeypatch,
) -> None:
    """G8-V1-F1 (b): an unticked Custom save replaced a working OpenAI reply
    voice while the Summary called it the default anyway."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    shown: list[tuple[bool, bool, str]] = []

    async def pick_custom_pocket(step, pilot):
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step.query_one(
            "#setup-voice-endpoint", Input
        ).value = "http://127.0.0.1:8766/tts"
        await pilot.pause()
        shown.append(_default_box(step))

    step, app, outcome, before, after = await _commit_through_real_writer(
        _WORKING_OPENAI_APP_TTS, _WORKING_OPENAI_TTS_SETTINGS, act=pick_custom_pocket
    )

    [(ticked, locked, help_line)] = shown
    assert (ticked, locked) == (True, True)
    assert help_line.startswith(
        "This becomes the voice replies use — it replaces OpenAI · tts-1-hd · shimmer."
    )
    assert outcome == (True, "")
    app_tts, _tts_settings = _voice_tables(after)
    assert app_tts["OPENAI_BASE_URL"] == "http://127.0.0.1:8766/tts"
    assert _summary_voice(after).detail == (
        "Custom endpoint 127.0.0.1:8766 · pocket-tts · alba"
    )


@pytest.mark.asyncio
async def test_custom_with_the_saved_values_claims_no_replacement(monkeypatch) -> None:
    """Review round 2 (R2-F5): over a saved OpenAI voice, Custom with the
    same values said "it replaces OpenAI · tts-1-hd · shimmer" although Next
    writes nothing; and the subtitle said "nothing is saved unless you choose
    one" over a voice that is saved and kept."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    shown: list[tuple[bool, bool, str]] = []
    subtitles: list[str] = []

    async def pick_custom(step, pilot):
        subtitles.append(str(step.query_one(".setup-subtitle", Static).render()))
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        shown.append(_default_box(step))

    step, app, outcome, before, after = await _commit_through_real_writer(
        _WORKING_OPENAI_APP_TTS, _WORKING_OPENAI_TTS_SETTINGS, act=pick_custom
    )

    assert "nothing is saved unless" not in subtitles[0]
    assert "current voice" in subtitles[0]
    [(ticked, locked, help_line)] = shown
    assert (ticked, locked) == (True, True)
    assert "replaces" not in help_line
    assert outcome == (True, "")
    assert app.saves == []
    assert after == before


@pytest.mark.asyncio
async def test_another_default_provider_keeps_the_box_free_and_its_defaults() -> None:
    """Review round 2 (F11): with kokoro reading replies, an unticked
    PocketTTS save writes the endpoint only; kokoro's defaults stay."""
    kokoro = {
        "default_provider": "kokoro",
        "default_model": "kokoro",
        "default_voice": "af_bella",
        "default_format": "wav",
        "default_speed": 1.0,
    }
    shown: list[tuple[bool, bool, str]] = []

    async def pick_pocket(step, pilot):
        assert step._preset == voice_state.VOICE_PRESET_NONE
        assert str(
            step.query_one("#setup-voice-service-status", Static).render()
        ).startswith("Current voice: kokoro")
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        shown.append(_default_box(step))

    step, app, outcome, before, after = await _commit_through_real_writer(
        kokoro, None, act=pick_pocket
    )

    [(ticked, locked, help_line)] = shown
    assert (ticked, locked) == (False, False)
    assert help_line.startswith("Saved for later; replies keep using kokoro.")
    assert outcome == (True, "")
    app_tts, _tts_settings = _voice_tables(after)
    assert app_tts["OPENAI_BASE_URL"] == voice_state.POCKET_TTS_ENDPOINT
    assert {key: app_tts[key] for key in kokoro} == kokoro
    assert _summary_voice(after).detail == (
        "kokoro (default voice); PocketTTS also saved"
    )


@pytest.mark.asyncio
async def test_custom_over_a_saved_pocket_tts_address_is_testable_and_untouched() -> (
    None
):
    """Review round 2 (G8-R2-F2): kokoro reads replies and the slot holds a
    pocket-tts /tts address on 8766. Choosing Custom filled tts-1-hd /
    shimmer / mp3, which silently disabled Test and Hear, and Next (nothing
    edited) refused with "PocketTTS returns WAV audio only"."""
    table = {
        "default_provider": "kokoro",
        "default_model": "kokoro",
        "default_voice": "af_bella",
        "default_format": "wav",
        "default_speed": 1.0,
        "OPENAI_BASE_URL": "http://127.0.0.1:8766/tts",
        "OPENAI_AUTH_MODE": "none",
    }
    seen: list[tuple[str, str, str, bool]] = []

    async def pick_custom(step, pilot):
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        seen.append(
            (
                *(
                    step.query_one(f"#setup-voice-{field}", Input).value
                    for field in ("model", "voice", "format")
                ),
                step.query_one("#setup-voice-test", Button).disabled,
            )
        )

    step, app, outcome, before, after = await _commit_through_real_writer(
        table, None, act=pick_custom
    )

    assert seen == [("pocket-tts", "alba", "wav", False)]
    assert outcome == (True, "")
    assert app.saves == []
    assert after == before


@pytest.mark.asyncio
async def test_the_old_wizards_pocket_tts_write_is_not_preselected_or_rewritten() -> (
    None
):
    """Review round 1 (F2): every profile that passed Voice untouched under
    the old wizard holds this table. It must not prefill as a working Custom
    voice, and an untouched Next still writes nothing."""
    legacy = {
        "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
        "OPENAI_AUTH_MODE": "none",
        "default_provider": "openai",
        "default_model": "tts-1-hd",
        "default_voice": "shimmer",
        "default_format": "mp3",
    }
    lines: list[str] = []

    async def read_line(step, pilot):
        lines.append(
            str(step.query_one("#setup-voice-service-status", Static).render())
        )

    step, app, outcome, before, after = await _commit_through_real_writer(
        legacy, None, act=read_line
    )

    assert step._preset == voice_state.VOICE_PRESET_NONE
    assert "127.0.0.1:8765" in lines[0] and "can't speak" in lines[0]
    assert outcome == (True, "")
    assert app.saves == []
    assert after == before
    assert _summary_voice(after).state == wizard_state.ROW_ATTENTION


@pytest.mark.asyncio
async def test_resuming_an_old_wizards_run_does_not_bring_back_the_8765_voice() -> None:
    """Review round 2 (R2-F4): an interrupted old-wizard run checkpointed the
    unspeakable 8765 PocketTTS address. Resumed, it came back as a
    working-looking Custom voice, and Next saved it with nothing edited."""
    legacy = {
        "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
        "OPENAI_AUTH_MODE": "none",
        "default_provider": "openai",
        "default_model": "tts-1-hd",
        "default_voice": "shimmer",
        "default_format": "mp3",
    }
    lines: list[str] = []

    async def resume(step, pilot):
        step.restore_checkpoint(
            {
                "preset": "pocket_tts",
                "endpoint": "http://127.0.0.1:8765/v1/audio/speech",
                "authentication_mode": "none",
                "model_id": "pocket-tts",
                "voice_id": "alba",
                "response_format": "wav",
                "speed": 1.0,
                "sample_text": voice_state.DEFAULT_SAMPLE_TEXT,
                "use_as_default": False,
            }
        )
        await pilot.pause()
        lines.append(
            str(step.query_one("#setup-voice-service-status", Static).render())
        )

    step, app, outcome, before, after = await _commit_through_real_writer(
        legacy, None, act=resume
    )

    assert step._preset == voice_state.VOICE_PRESET_NONE
    assert "127.0.0.1:8765" in lines[0] and "can't speak" in lines[0]
    assert outcome == (True, "")
    assert app.saves == []
    assert after == before


@pytest.mark.asyncio
async def test_no_voice_for_now_over_a_saved_voice_says_it_is_kept(
    monkeypatch,
) -> None:
    """Review round 1 (F6 / G8-V1-F2): the line said "Nothing is saved" while
    Next kept the saved voice and the Summary showed it with a tick."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")
    lines: list[str] = []

    async def choose_no_voice(step, pilot):
        step._select_preset_button("setup-voice-preset-none")
        await pilot.pause()
        lines.append(
            str(step.query_one("#setup-voice-service-status", Static).render())
        )

    step, app, outcome, before, after = await _commit_through_real_writer(
        _WORKING_OPENAI_APP_TTS, _WORKING_OPENAI_TTS_SETTINGS, act=choose_no_voice
    )

    assert lines[0].startswith("Keeps your current voice (OpenAI · tts-1-hd · shimmer)")
    assert "Nothing is saved" not in lines[0]
    assert app.saves == []
    assert after == before


@pytest.mark.asyncio
async def test_a_voice_saved_earlier_this_run_is_the_one_no_voice_keeps() -> None:
    """G8-V1-F2 (same run): save a voice, go Back, choose "No voice for now":
    the line names the voice just saved, and Next with it unchanged writes
    nothing more."""
    _replace_voice_tables(None, None)
    service = _service()
    step = _step(dict(config_module.settings))
    app = _RealVoiceSaveApp(step, service)
    app.app_config = dict(config_module.settings)
    step.wizard.app_instance = app
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            step._select_preset_button("setup-voice-preset-pocket")
            await pilot.pause()
            assert await step.commit() == (True, "")
            await pilot.pause()
            assert len(app.saves) == 1

            step._select_preset_button("setup-voice-preset-none")
            await pilot.pause()
            line = str(step.query_one("#setup-voice-service-status", Static).render())
            assert line.startswith(
                "Keeps your current voice (PocketTTS · pocket-tts · alba)"
            )

            step._select_preset_button("setup-voice-preset-pocket")
            await pilot.pause()
            assert await step.commit() == (True, "")
            await pilot.pause()
            assert len(app.saves) == 1
    finally:
        await service.close()
        await service.wait_closed()


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
    on_main_thread: list[bool] = []

    def probe(url: str) -> bool:
        # Review round 2 (F1) / round 1 (F12): a probe run on the event loop
        # would block the step; pin that it runs in a worker thread.
        calls.append(url)
        on_main_thread.append(threading.current_thread() is threading.main_thread())
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
        assert on_main_thread == [False]

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
        assert status.startswith(
            f"Test failed — PocketTTS isn't running at 127.0.0.1:{port}."
        )
        assert "Not tested yet" not in status


def _clipped_service_labels(step, app) -> list[str]:
    """Service radio labels narrower than their own content (they end in "…")."""
    return [
        str(button.label)
        for button in step.query_one("#setup-voice-preset").query(RadioButton)
        if button.content_size.width < button.get_content_width(button.size, app.size)
    ]


@pytest.mark.parametrize("size", [(120, 40), (100, 30)])
@pytest.mark.asyncio
async def test_every_service_label_fits_in_the_wizard(size) -> None:
    """Review round 1 (F8 / G8-V1-F3): with a service chosen the step
    scrolls, the scrollbar takes its columns, and five equal segments clipped
    the lead option to "No voice for no…" at 120 columns. Real stylesheet:
    without it the radio is not even a row."""
    wizard = _make_wizard()
    app = _StyledHostApp(wizard)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        index = container._step_index_for_id(STEP_VOICE)
        container.show_step(index)
        await pilot.pause()
        step = container.steps[index]
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause(0.3)

        buttons = list(step.query_one("#setup-voice-preset").query(RadioButton))
        assert len({button.region.y for button in buttons}) == 1  # one row
        assert step.show_vertical_scrollbar  # the case that clipped
        assert _clipped_service_labels(step, app) == []


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
async def test_focus_stays_where_the_user_moved_it_during_a_test(monkeypatch) -> None:
    """Review round 1 (F5): a test can take 20 s. Focus the user moved in
    the meantime must not be pulled back to Test and Hear when it ends, or
    the next Enter re-runs the test instead of doing what they chose."""
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
        dropped = step.app.focused  # where disabling the button sent focus
        assert dropped is not test
        moved_to = step.query_one("#setup-voice-sample", Input)
        if dropped is moved_to:
            moved_to = step.query_one("#setup-voice-preset", RadioSet)
        moved_to.focus()
        await pilot.pause()

        release.set()
        await pilot.pause(0.2)
        assert test.disabled is False
        assert step.app.focused is moved_to


@pytest.mark.asyncio
async def test_a_click_during_a_test_keeps_focus_where_it_landed(monkeypatch) -> None:
    """Review round 1 (F5), found live: disabling Test and Hear drops focus
    into Sample text, so a user who clicks Sample text during a test leaves
    focus exactly where the drop put it. Comparing focus alone took that for
    "untouched" and pulled focus back to Test and Hear, so the next keys went
    nowhere. A click during a test is the user choosing where focus goes."""
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
        sample_input = step.query_one("#setup-voice-sample", Input)
        await pilot.click("#setup-voice-test")
        await pilot.pause()
        assert test.disabled is True
        assert step.app.focused is sample_input  # where the drop put it

        await pilot.click("#setup-voice-sample")
        await pilot.pause()
        release.set()
        await pilot.pause(0.2)

        assert test.disabled is False
        assert step.app.focused is sample_input

        # The click that starts a test is not a click "during" it: with no
        # other click, focus still comes back to Test and Hear.
        release.clear()
        await pilot.click("#setup-voice-test")
        await pilot.pause()
        assert test.disabled is True
        release.set()
        await pilot.pause(0.2)
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
        hints = wizard.query_one("#setup-key-hints", Static)
        # Review round 1 (G8-V1-F6): "No voice for now" hides Sample text,
        # so the hint line is the wizard's own until a service shows it.
        assert str(hints.render()) == (
            "Enter / Ctrl+N next · Ctrl+B back · Esc exit setup"
        )
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        # Review round 1 (F13): Enter still advances from every other field.
        assert str(hints.render()) == (
            "Enter / Ctrl+N next · Enter in Sample text tests · Ctrl+B back "
            "· Esc exit setup"
        )
        step.query_one("#setup-voice-sample", Input).focus()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert calls == [voice_state.DEFAULT_SAMPLE_TEXT]
        assert container.current_step == index

        step._select_preset_button("setup-voice-preset-none")
        await pilot.pause()
        assert str(hints.render()) == (
            "Enter / Ctrl+N next · Ctrl+B back · Esc exit setup"
        )


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

        # Review round 2 (F4): the Authentication radio is an input too.
        step.show_step_error("A third thing.")
        step.query_one("#setup-voice-auth-none", RadioButton).value = True
        await _settle(pilot, lambda: str(strip.render()) == "")
        assert str(strip.render()) == ""
        assert strip.has_class("hidden")


# -- AC#7: copy, the default box, one primary, pickers -----------------------------


@pytest.mark.asyncio
async def test_voice_copy_controls_and_auto_tick(monkeypatch) -> None:
    async def sample(*_args, **_kwargs):
        return voice_state.VoiceSampleResult(b"valid", "audio/wav", "wav", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    # Another provider reads replies, so the box is the user's to tick (with
    # nothing else saved it is locked on: test_a_fresh_pick_is_the_reply_...).
    step = _step(_raw({"default_provider": "kokoro"}))
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


# -- review round 2: guards the mutation run found missing ----------------------


@pytest.mark.asyncio
async def test_a_disabled_test_button_says_why(monkeypatch) -> None:
    """Review round 2 (G8-R2-F5): with Test and Hear disabled by a blank
    sample or an Advanced value it cannot send, the status still read
    "Optional — press Test and Hear to play a short sample.", inviting a
    press that does nothing and never saying what to fix."""
    monkeypatch.setattr(voice_step_module, "probe_endpoint_reachable", lambda _u: False)
    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        status = step.query_one("#setup-voice-status", Static)
        test = step.query_one("#setup-voice-test", Button)

        step.query_one("#setup-voice-sample", Input).value = "   "
        await pilot.pause()
        assert test.disabled
        assert str(status.render()) == voice_status.BLANK_SAMPLE_COPY

        step.query_one("#setup-voice-sample", Input).value = "Hello"
        await pilot.pause()
        assert not test.disabled
        assert str(status.render()) == voice_status.DEFAULT_STATUS_COPY

        step.query_one("#setup-voice-format", Input).value = "mp3"
        await pilot.pause()
        assert test.disabled
        assert str(status.render()) == (
            "To test, fix this under Advanced: PocketTTS returns WAV audio only. "
            "Set the output format to wav."
        )

        step.query_one("#setup-voice-format", Input).value = "wav"
        await pilot.pause()
        assert not test.disabled
        assert str(status.render()) == voice_status.DEFAULT_STATUS_COPY


@pytest.mark.asyncio
async def test_a_successful_test_on_an_untouched_rerun_still_saves(monkeypatch) -> None:
    """Review round 2 (F2): a successful test counts as acting this run, so
    Next posts a save even when nothing was edited (the user verified it)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-sent")

    async def sample(*_args, **_kwargs):
        return voice_state.VoiceSampleResult(b"valid", "audio/mpeg", "mp3", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    step = _step(_raw(_WORKING_OPENAI_APP_TTS))
    app = _CapturingHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert step._preset == voice_state.VOICE_PRESET_OFFICIAL_OPENAI
        step.query_one("#setup-voice-test", Button).press()
        await pilot.pause(0.2)
        assert str(step.query_one("#setup-voice-status", Static).render()) == (
            voice_status.PLAYED_COPY
        )

        commit = asyncio.create_task(step.commit())
        await pilot.pause()
        assert app.saved is not None
        step.receive_stts_settings_save_result(
            _applied_result(app.saved, defaults_activated=True)
        )
        assert await commit == (True, "")


@pytest.mark.parametrize(
    ("select_id", "field_id", "typed", "attribute"),
    (
        ("#setup-voice-voice-select", "#setup-voice-voice", "verse", "voice_id"),
        (
            "#setup-voice-format-select",
            "#setup-voice-format",
            "flac",
            "response_format",
        ),
    ),
)
@pytest.mark.asyncio
async def test_other_reveals_a_focused_text_field_that_holds_the_value(
    select_id, field_id, typed, attribute
) -> None:
    """Review round 2 (F5): "Other…" was only checked for being listed."""
    from tldw_chatbook.UI.Wizards.first_run_voice_pickers import OTHER_VALUE

    step = _step()
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-official")
        await pilot.pause()
        step.query_one("#setup-voice-advanced").collapsed = False
        await pilot.pause()
        field = step.query_one(field_id, Input)
        assert field.display is False  # the service's own value is picked

        step.query_one(select_id, Select).value = OTHER_VALUE
        await pilot.pause()
        assert field.display is True
        assert step.app.focused is field

        field.value = ""
        await pilot.press(*typed)
        await pilot.pause()
        assert getattr(step._draft_from_controls(), attribute) == typed


# -- review round 1, F9: a placeholder key is not a key --------------------------


@pytest.mark.parametrize(
    "stored",
    ("<API_KEY_HERE>", "   ", "enc:v1:not-decrypted"),
    ids=("placeholder", "blank", "ciphertext"),
)
@pytest.mark.asyncio
async def test_an_unusable_saved_openai_key_is_not_found(monkeypatch, stored) -> None:
    """The step read saved keys without CLAUDE.md's validity check, so the
    shipped placeholder read as "key found" and went out as a Bearer token."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    raw = {"api_settings": {"openai": {"api_key": stored}}}
    step = _step({"COMPREHENSIVE_CONFIG_RAW": raw, "OPENAI_API_KEY": stored})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-official")
        await pilot.pause()

        assert step.query_one("#setup-voice-key-row").display is True
        assert step.query_one("#setup-voice-test", Button).disabled is True
        assert "no OpenAI API key found" in str(
            step.query_one("#setup-voice-service-status", Static).render()
        )


@pytest.mark.asyncio
async def test_a_padded_saved_openai_key_is_sent_trimmed(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list[object] = []

    async def sample(draft, *, credential=None, **_kwargs):
        seen.append(credential)
        return voice_state.VoiceSampleResult(b"valid", "audio/mpeg", "mp3", True)

    monkeypatch.setattr(voice_state, "run_voice_sample", sample)
    raw = {"api_settings": {"openai": {"api_key": "  sk-test-padded  "}}}
    step = _step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-official")
        await pilot.pause()
        step.query_one("#setup-voice-test", Button).press()
        await pilot.pause(0.2)

    assert seen == ["sk-test-padded"]
