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

# TASK-33921: the step moved to its own module; patch where it looks names up.
# TASK-34100.8: the OmniVoice half moved again, into first_run_voice_omnivoice.
import tldw_chatbook.UI.Wizards.first_run_voice_omnivoice as voice_step_module
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
        voice_step_module, "omnivoice_setup_state",
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
        ("path_invalid", vs.OMNIVOICE_PATH_INVALID_COPY, False, False),
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
    monkeypatch.setattr(voice_step_module, "omnivoice_setup_state", lambda *_a, **_k: next(states))
    order: list = []

    async def preflight(**_):
        order.append("preflight")
        return "REPORT"

    async def provision(report, *, progress=None, **_):
        order.append(("provision", report))
        return None

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(voice_step_module, "run_omnivoice_provision", provision)
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

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(voice_step_module, "run_omnivoice_provision", provision)
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

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(voice_step_module, "run_omnivoice_provision", failing_provision)
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

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", slow_preflight)
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

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", failing)
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


def _default_box(step) -> tuple[bool, bool, str]:
    box = step.query_one("#setup-voice-default", Checkbox)
    help_line = str(step.query_one("#setup-voice-default-help", Static).render())
    return box.value, box.disabled, help_line


_OMNIVOICE_READS_REPLIES = {
    "COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"default_provider": "omnivoice"}}
}


async def test_a_saved_omnivoice_reply_voice_cannot_be_unticked_away(
    monkeypatch,
) -> None:
    """Review round 2 (R2-F2): the prefill preselected a saved OmniVoice voice
    with the box ticked but free. Unticked, Next posted nothing, so OmniVoice
    kept reading replies while the box said it would not."""
    _state(monkeypatch, "ready")
    step = _step(_OMNIVOICE_READS_REPLIES)
    host = _Host(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        assert step._preset == vs.VOICE_PRESET_OMNIVOICE

        ticked, locked, help_line = _default_box(step)
        assert (ticked, locked) == (True, True)
        assert help_line.startswith(
            "Replies use OmniVoice now — pick another service to change that."
        )
        step.query_one("#setup-voice-default", Checkbox).value = False
        await pilot.pause()
        assert _default_box(step)[:2] == (True, True)
        assert await step.commit() == (True, "")
        assert host.saved is None


async def test_another_service_over_a_saved_omnivoice_names_it(monkeypatch) -> None:
    """Review round 2 (R2-F1): over a saved OmniVoice reply voice, an
    OpenAI-slot pick read "Replies will use this voice instead of ." -- the
    provider name was empty. Like kokoro, the box starts unticked."""
    _state(monkeypatch, "ready")
    step = _step(_OMNIVOICE_READS_REPLIES)
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause()

        ticked, locked, help_line = _default_box(step)
        assert (ticked, locked) == (False, False)
        assert help_line.startswith("Saved for later; replies keep using OmniVoice.")
        step.query_one("#setup-voice-default", Checkbox).value = True
        await pilot.pause()
        assert _default_box(step)[2].startswith(
            "Replies will use this voice instead of OmniVoice."
        )


async def test_a_blank_sample_says_why_test_is_off(monkeypatch) -> None:
    """Review round 2 (G8-R2-F5): OmniVoice shares the status line, so a
    blank sample says why Test and Hear is off there too."""
    from tldw_chatbook.UI.Wizards import first_run_voice_status as voice_status

    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-sample", Input).value = ""
        await pilot.pause()

        assert step.query_one("#setup-voice-test", Button).disabled
        assert str(step.query_one("#setup-voice-status", Static).render()) == (
            voice_status.BLANK_SAMPLE_COPY
        )


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

    monkeypatch.setattr(voice_step_module, "omnivoice_setup_state", broken)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        status = step.query_one("#setup-voice-omnivoice-status", Static)
        assert str(status.renderable) == vs.OMNIVOICE_ENGINE_MISSING_COPY
        install = step.query_one("#setup-voice-omnivoice-install", Button)
        assert not (install.display and not install.disabled)


async def test_commit_default_with_an_invalid_configured_root(monkeypatch) -> None:
    _state(monkeypatch, "path_invalid")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-default", Checkbox).value = True
        assert await step.commit() == (False, vs.OMNIVOICE_PATH_INVALID_COPY)


async def test_a_superseded_state_check_cannot_overwrite_the_current_one(
    monkeypatch,
) -> None:
    """Enter OmniVoice (slow check), leave, re-enter (fast check): the first
    check finishing last must not replace the newer result."""
    import threading

    release = threading.Event()
    calls: list = []

    def state(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            release.wait(5)
            return "ready"
        return "model_missing"

    monkeypatch.setattr(voice_step_module, "omnivoice_setup_state", state)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause(0.1)
        await _select_omnivoice(step, pilot)
        assert step._omnivoice_state == "model_missing"
        release.set()
        await pilot.pause(0.3)
        assert step._omnivoice_state == "model_missing"
        status = step.query_one("#setup-voice-omnivoice-status", Static)
        assert str(status.renderable) == vs.OMNIVOICE_MODEL_MISSING_COPY


async def test_a_stale_state_check_leaves_other_services_alone(monkeypatch) -> None:
    import threading

    release = threading.Event()

    def state(*_a, **_k):
        release.wait(5)
        return "ready"

    monkeypatch.setattr(voice_step_module, "omnivoice_setup_state", state)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause(0.1)
        release.set()
        await pilot.pause(0.3)
        assert step._omnivoice_state is None


async def test_an_older_sample_finishing_playback_keeps_the_newer_status(
    monkeypatch,
) -> None:
    _state(monkeypatch, "ready")
    playing = asyncio.Event()
    finish = asyncio.Event()

    async def sample(*_a, **_k):
        return vs.VoiceSampleResult(b"RIFF....WAVE", "audio/wav", "wav", True)

    class SlowPlayer:
        async def play(self, _path):
            playing.set()
            await finish.wait()
            return True

    monkeypatch.setattr(vs, "run_omnivoice_sample", sample)
    step = _step()
    host = _Host(step)
    host.audio_player = SlowPlayer()
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-test", Button).press()
        await asyncio.wait_for(playing.wait(), timeout=2)
        step._test_generation += 1  # a newer test/control change took over
        step.query_one("#setup-voice-status", Static).update("newer")
        finish.set()
        await pilot.pause(0.2)
        assert str(step.query_one("#setup-voice-status", Static).renderable) == "newer"


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


@pytest.mark.parametrize("size", [(100, 30), (80, 24), (72, 24)])
async def test_service_row_fits_at_every_supported_width(monkeypatch, size) -> None:
    """Every service label paints whole -- measured with the real stylesheet
    (TASK-34100.8 review round 1, F8). This test used to run at 80 columns
    without the stylesheet, where the radio is a vertical list of full-width
    rows, so it passed whatever the CSS said. The five labels need 71 cells:
    round 1 fitted them from 100 columns and conceded 80; review round 2
    (R2-F3) wraps the radio to two rows wherever one row is narrower, so
    they fit at 80 and below too."""
    from pathlib import Path

    from textual.widgets import RadioButton

    class _StyledHost(_Host):
        CSS_PATH = str(
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/css/tldw_cli_modular.tcss"
        )

    _state(monkeypatch, "ready")
    step = _step()
    host = _StyledHost(step)
    async with host.run_test(size=size) as pilot:
        step.add_class("active")  # what the wizard's show_step does
        await pilot.pause()
        buttons = list(step.query_one("#setup-voice-preset").query(RadioButton))
        assert len({button.region.y for button in buttons}) <= 2
        for button in buttons:
            assert button.content_size.width >= button.get_content_width(
                button.size, host.size
            ), str(button.label)


async def test_omnivoice_install_names_itself_waits_in_line_and_can_be_cancelled(
    monkeypatch,
) -> None:
    """TASK-34100.14: the 1.1 GB OmniVoice download shares the Speech step's
    download control -- it names itself for a second install that has to
    wait, waits in line itself, and stops from the progress widget's Cancel.
    """
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionCancelledError,
        AcquisitionProgress,
    )

    _state(monkeypatch, "model_missing")
    seen: dict = {}

    async def preflight(**_):
        return "REPORT"

    async def provision(report, *, progress=None, control=None, **_):
        seen["control"] = control
        progress(
            AcquisitionProgress(
                "fetch", ArtifactRef("omnivoice-onnx-int8hq", "r1", "int8"),
                "model.onnx", 1, 2,
            )
        )
        for _ in range(100):
            if control is not None and control.cancel_requested:
                raise AcquisitionCancelledError()
            await asyncio.sleep(0.05)
        raise AssertionError("the download was never cancelled")

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(voice_step_module, "run_omnivoice_provision", provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(True))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        cancel = None
        for _ in range(40):
            await pilot.pause(0.05)
            found = step.query("#setup-voice-omnivoice-progress .model-install-cancel")
            if found and found.first(Button).display:
                cancel = found.first(Button)
                break
        assert cancel is not None, "no Cancel download on the OmniVoice progress"
        await pilot.click(cancel)
        for _ in range(40):
            await pilot.pause(0.05)
            if not step._omnivoice_installing:
                break
        status = str(step.query_one("#setup-voice-omnivoice-status", Static).renderable)
        install = step.query_one("#setup-voice-omnivoice-install", Button)

    control = seen["control"]
    assert control.label == "OmniVoice voice model"
    assert control.wait_if_busy is True
    assert control.cancel_requested is True
    assert status == "Download cancelled — nothing was saved."
    assert install.disabled is False


async def test_omnivoice_download_ending_under_a_focused_cancel_keeps_focus_alive(
    monkeypatch,
) -> None:
    """TASK-34100.14 live recheck: the step hides the whole progress widget
    when the download ends. Textual blurs a hidden focused widget, and the
    Cancel button had no focusable sibling, so focus fell to None and the
    wizard's Ctrl+N/Ctrl+B (bound on its container) did nothing until Tab.
    """
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress

    _state(monkeypatch, "model_missing")
    release = asyncio.Event()

    async def preflight(**_):
        return "REPORT"

    async def provision(report, *, progress=None, control=None, **_):
        progress(
            AcquisitionProgress(
                "fetch", ArtifactRef("omnivoice-onnx-int8hq", "r1", "int8"),
                "model.onnx", 1, 2,
            )
        )
        for _ in range(100):
            if release.is_set():
                break
            await asyncio.sleep(0.05)
        raise RuntimeError("upstream went away")

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(voice_step_module, "run_omnivoice_provision", provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(True))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        cancel = None
        for _ in range(40):
            await pilot.pause(0.05)
            found = step.query("#setup-voice-omnivoice-progress .model-install-cancel")
            if found and found.first(Button).display:
                cancel = found.first(Button)
                break
        assert cancel is not None, "no Cancel download on the OmniVoice progress"
        cancel.focus()
        await pilot.pause()
        assert host.focused is cancel
        host.call_later(release.set)
        for _ in range(60):
            await pilot.pause(0.05)
            if not step._omnivoice_installing:
                break
        await pilot.pause()
        focused = host.focused
        alive = focused is not None and focused.is_attached
        node = focused
        while alive and node is not None and node is not host.screen:
            alive = bool(node.display)
            node = node.parent
        in_step = focused is not None and step in focused.ancestors

    assert not step._omnivoice_installing
    assert alive, f"focus stranded after the download ended: {focused!r}"
    assert in_step, f"focus left the step: {focused!r}"


async def test_omnivoice_second_download_after_a_cancel_can_be_cancelled_too(
    monkeypatch,
) -> None:
    """TASK-34100.14 review F1: the Voice step composes its OmniVoice
    progress widget once and only hides it between downloads. After one
    cancel, the next download used to read "Cancelling download…" for the
    whole 1.1 GB transfer with Cancel disabled -- it could not be stopped.
    """
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionCancelledError,
        AcquisitionProgress,
    )

    _state(monkeypatch, "model_missing")
    controls: list = []

    async def preflight(**_):
        return "REPORT"

    async def provision(report, *, progress=None, control=None, **_):
        controls.append(control)
        progress(
            AcquisitionProgress(
                "fetch", ArtifactRef("omnivoice-onnx-int8hq", "r1", "int8"),
                "model.onnx", 1, 2,
            )
        )
        for _ in range(100):
            if control is not None and control.cancel_requested:
                raise AcquisitionCancelledError()
            await asyncio.sleep(0.05)
        raise AssertionError("the download was never cancelled")

    async def start_and_find_cancel(step, pilot) -> tuple:
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(60):
            await pilot.pause(0.05)
            found = step.query("#setup-voice-omnivoice-progress .model-install-cancel")
            if found and found.first(Button).display and step._omnivoice_download:
                break
        cancel = step.query_one(
            "#setup-voice-omnivoice-progress .model-install-cancel", Button
        )
        phase = str(
            step.query_one("#model-install-progress-phase", Static).render()
        )
        status = str(
            step.query_one("#setup-voice-omnivoice-status", Static).renderable
        )
        return cancel, phase, cancel.disabled, status

    async def wait_finished(step, pilot) -> None:
        for _ in range(60):
            await pilot.pause(0.05)
            if not step._omnivoice_installing:
                return

    monkeypatch.setattr(voice_step_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(voice_step_module, "run_omnivoice_provision", provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(True))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        cancel, first_phase, first_disabled, first_status = (
            await start_and_find_cancel(step, pilot)
        )
        await pilot.click(cancel)
        await wait_finished(step, pilot)
        assert not step._omnivoice_installing

        cancel, second_phase, second_disabled, second_status = (
            await start_and_find_cancel(step, pilot)
        )
        await pilot.click(cancel)
        await wait_finished(step, pilot)
        status = str(step.query_one("#setup-voice-omnivoice-status", Static).renderable)

    assert len(controls) == 2
    assert (first_phase, first_disabled) == ("Downloading model", False)
    assert (second_phase, second_disabled) == ("Downloading model", False)
    # Found live (g14v5-voice/08): the last outcome stayed on the status
    # line above the running download.
    assert first_status == vs.OMNIVOICE_MODEL_MISSING_COPY
    assert second_status == first_status
    assert controls[1].cancel_requested is True
    assert status == "Download cancelled — nothing was saved."


@pytest.mark.parametrize("started", [False, True])
async def test_leaving_setup_abandons_an_omnivoice_download_still_waiting_in_line(
    monkeypatch, started
) -> None:
    """TASK-34100.14 review F4: see the Speech step's twin test."""
    from tldw_chatbook.Model_Artifacts.download_control import DownloadControl

    _state(monkeypatch, "model_missing")
    step = _step()
    host = _Host(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        control = DownloadControl(label="OmniVoice voice model", wait_if_busy=True)
        if started:
            control.mark_started()
        step._omnivoice_download = control
        await step.remove()
        await pilot.pause()

    assert control.cancel_requested is (not started)
