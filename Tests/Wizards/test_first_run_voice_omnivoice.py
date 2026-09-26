"""VoiceSetupStep: OmniVoice as a fourth service (spec 2026-09-25)."""

from __future__ import annotations

import asyncio
import io
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual import on
from textual.widgets import Button, Checkbox, Collapsible, Input, Static

import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
from Tests.Wizards.test_first_run_setup_wizard import _StepHost
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_VOICE
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import VoiceSetupStep


def _step(app_config: dict | None = None) -> VoiceSetupStep:
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config or {}), wizard_data={}
    )
    return VoiceSetupStep(
        wizard=wizard, config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4)
    )


class _Host(_StepHost):
    saved: STTSSettingsSaveEvent | None = None

    @on(STTSSettingsSaveEvent)
    def capture(self, event: STTSSettingsSaveEvent) -> None:
        self.saved = event


async def _select_omnivoice(step: VoiceSetupStep, pilot) -> None:
    step._select_preset_button("setup-voice-preset-omnivoice")
    await pilot.pause(0.2)


def _state(monkeypatch: pytest.MonkeyPatch, value: str) -> list:
    calls: list = []
    monkeypatch.setattr(
        wizard_module, "omnivoice_setup_state",
        lambda model_root, **_: calls.append(model_root) or value,
    )
    return calls


async def test_selecting_omnivoice_swaps_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert step.query_one("#setup-voice-omnivoice-panel").display is False
        await _select_omnivoice(step, pilot)
        assert step.query_one("#setup-voice-omnivoice-panel").display is True
        assert step.query_one("#setup-voice-advanced", Collapsible).display is False
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause(0.2)
        assert step.query_one("#setup-voice-omnivoice-panel").display is False
        assert step.query_one("#setup-voice-advanced", Collapsible).display is True


async def test_custom_edit_survives_a_round_trip_through_omnivoice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix round 1, Important 1, Scenario A: Custom -> Pocket -> Custom ->
    edit -> OmniVoice -> Custom must keep the edit, not revert it."""
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        step.query_one("#setup-voice-endpoint", Input).value = (
            "http://example.test/v1/audio/speech"
        )
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        assert (
            step.query_one("#setup-voice-endpoint", Input).value
            == "http://example.test/v1/audio/speech"
        )


async def test_custom_edit_survives_omnivoice_via_a_third_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix round 1, Important 1, Scenario B: Custom (edited) -> OmniVoice ->
    Pocket -> Custom must keep the edit, not lose it."""
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        step.query_one("#setup-voice-endpoint", Input).value = (
            "http://example.test/v1/audio/speech"
        )
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()
        step._select_preset_button("setup-voice-preset-custom")
        await pilot.pause()
        assert (
            step.query_one("#setup-voice-endpoint", Input).value
            == "http://example.test/v1/audio/speech"
        )


@pytest.mark.parametrize(
    ("state", "copy", "install_enabled", "test_enabled"),
    [
        ("engine_missing", vs.OMNIVOICE_ENGINE_MISSING_COPY, False, False),
        ("model_missing", vs.OMNIVOICE_MODEL_MISSING_COPY, True, False),
        ("ready", vs.OMNIVOICE_READY_COPY, False, True),
    ],
)
async def test_each_state_renders_copy_and_buttons(
    monkeypatch, state, copy, install_enabled, test_enabled
) -> None:
    _state(monkeypatch, state)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        assert str(step.query_one("#setup-voice-omnivoice-status", Static).renderable) == copy
        install = step.query_one("#setup-voice-omnivoice-install", Button)
        assert (install.display and not install.disabled) is install_enabled
        assert step.query_one("#setup-voice-test", Button).disabled is (not test_enabled)


async def test_install_runs_consent_then_provision_then_rereads(monkeypatch) -> None:
    states = iter(["model_missing", "ready"])
    monkeypatch.setattr(wizard_module, "omnivoice_setup_state", lambda *_a, **_k: next(states))
    order: list = []

    async def preflight(**_):
        order.append("preflight")
        return "REPORT"

    async def provision(report, *, progress=None, **_):
        order.append(("provision", report))
        return None

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(wizard_module, "run_omnivoice_provision", provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(True))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(40):
            await pilot.pause(0.05)
        assert order == ["preflight", ("provision", "REPORT")]
        assert str(step.query_one("#setup-voice-omnivoice-status", Static).renderable) == vs.OMNIVOICE_READY_COPY


async def test_declining_consent_leaves_model_missing_and_reenables_install(
    monkeypatch,
) -> None:
    """Fix round 1, Minor 4: declining the consent modal must not provision,
    and must land back on an actionable model_missing state."""
    _state(monkeypatch, "model_missing")
    provision_calls: list = []

    async def preflight(**_):
        return "REPORT"

    async def provision(report, *, progress=None, **_):
        provision_calls.append(report)
        return None

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(wizard_module, "run_omnivoice_provision", provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(False))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(40):
            await pilot.pause(0.05)
        assert provision_calls == []
        assert step._omnivoice_state == "model_missing"
        install = step.query_one("#setup-voice-omnivoice-install", Button)
        assert install.display is True
        assert install.disabled is False


async def test_provision_failure_shows_message_and_reenables_install(
    monkeypatch,
) -> None:
    """Fix round 1, Minor 4: a provision failure must surface
    install_failure_message's text and leave Install re-enabled to retry."""
    _state(monkeypatch, "model_missing")

    async def preflight(**_):
        return "REPORT"

    async def failing_provision(report, *, progress=None, **_):
        raise RuntimeError("disk full")

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(wizard_module, "run_omnivoice_provision", failing_provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(True))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(40):
            await pilot.pause(0.05)
        status = str(step.query_one("#setup-voice-omnivoice-status", Static).renderable)
        assert status and status != vs.OMNIVOICE_MODEL_MISSING_COPY
        install = step.query_one("#setup-voice-omnivoice-install", Button)
        assert install.disabled is False


async def test_install_double_press_runs_one_preflight(monkeypatch) -> None:
    _state(monkeypatch, "model_missing")
    count = {"preflight": 0}

    async def slow_preflight(**_):
        count["preflight"] += 1
        await asyncio.sleep(0.3)
        raise RuntimeError("offline")

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", slow_preflight)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        # Button.press() is a no-op on a disabled button, so pressing twice
        # would pass even without the guard: drive the handler directly.
        pressed = SimpleNamespace(stop=lambda: None)
        step._on_omnivoice_install(pressed)
        step._on_omnivoice_install(pressed)
        for _ in range(20):
            await pilot.pause(0.05)
        assert count["preflight"] == 1


async def test_preflight_failure_shows_message_and_allows_retry(monkeypatch) -> None:
    _state(monkeypatch, "model_missing")

    async def failing(**_):
        raise RuntimeError("network down")

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", failing)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
        status = str(step.query_one("#setup-voice-omnivoice-status", Static).renderable)
        assert status and status != vs.OMNIVOICE_MODEL_MISSING_COPY
        assert step.query_one("#setup-voice-omnivoice-install", Button).disabled is False


async def test_reshow_rereads_state_without_cancelling_install(monkeypatch) -> None:
    calls = _state(monkeypatch, "model_missing")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        before = len(calls)
        step._omnivoice_installing = True
        step.on_hide()
        step.on_show()
        await pilot.pause(0.2)
        assert len(calls) == before + 1
        assert step._omnivoice_installing is True


async def test_commit_without_default_saves_nothing(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    host = _Host(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        assert await step.commit() == (True, "")
        assert host.saved is None


async def test_commit_default_without_model_is_refused(monkeypatch) -> None:
    _state(monkeypatch, "model_missing")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-default", Checkbox).value = True
        assert await step.commit() == (False, vs.OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY)


async def test_commit_default_names_the_engine_when_the_engine_is_missing(
    monkeypatch,
) -> None:
    _state(monkeypatch, "engine_missing")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-default", Checkbox).value = True
        assert await step.commit() == (False, vs.OMNIVOICE_ENGINE_MISSING_COPY)


async def test_commit_default_while_state_is_still_checking(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step._omnivoice_state = None
        step.query_one("#setup-voice-default", Checkbox).value = True
        assert await step.commit() == (False, vs.OMNIVOICE_CHECKING_COPY)


async def test_state_read_import_error_means_engine_missing(monkeypatch) -> None:
    """A broken engine import must not offer a model download that can't help."""

    def broken(*_a, **_k):
        raise ImportError("onnxruntime is broken")

    monkeypatch.setattr(wizard_module, "omnivoice_setup_state", broken)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        status = step.query_one("#setup-voice-omnivoice-status", Static)
        assert str(status.renderable) == vs.OMNIVOICE_ENGINE_MISSING_COPY
        install = step.query_one("#setup-voice-omnivoice-install", Button)
        assert not (install.display and not install.disabled)


async def test_commit_default_saves_omnivoice_with_the_sampled_seed(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    seeds: list = []

    async def sample(text, *, speed, seed, **_):
        seeds.append(seed)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(b"\x00\x00" * 240)
        return vs.VoiceSampleResult(buffer.getvalue(), "audio/wav", "wav", True)

    monkeypatch.setattr(vs, "run_omnivoice_sample", sample)
    step = _step({"COMPREHENSIVE_CONFIG_RAW": {"OmniVoiceSettings": {"seed": 777}}})
    host = _Host(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-test", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
        step.query_one("#setup-voice-default", Checkbox).value = True
        commit = asyncio.create_task(step.commit())
        await pilot.pause(0.1)
        assert host.saved is not None
        assert dict(host.saved.settings) == {"OMNIVOICE_SEED": 777}
        assert seeds == [777]
        step.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=host.saved.request_id,
                persisted=True,
                provider_statuses={"omnivoice": "applied"},
                provider_configuration_revisions={"omnivoice": 1},
                provider_runtime_revisions={"omnivoice": 1},
                defaults_activated=True,
                defaults_activation_status="committed",
            )
        )
        assert await commit == (True, "")


async def test_step_data_records_the_preset(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        assert step.get_step_data()["preset"] == "omnivoice"


async def test_service_row_fits_at_80_columns(monkeypatch) -> None:
    import html
    import re

    _state(monkeypatch, "ready")
    step = _step()
    host = _Host(step)
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        painted = html.unescape(
            "".join(re.findall(r">([^<>]*)</text>", host.export_screenshot()))
        ).replace("\xa0", " ")
        # Every service label paints whole — clipping would end it in "…".
        for label in ("PocketTTS", "OpenAI", "Custom", "OmniVoice"):
            assert label in painted
            assert f"{label[:6]}…" not in painted
