"""TASK-34100.13 (voice-speech-15, voice-speech-12): the Speech step's engine
choice and its choice-then-action Parakeet layout.

The step offered only a 633 MiB Parakeet download, a model folder or a GGUF,
with the actions above the choices they depend on and jargon throughout. It
now opens with an engine choice built from what this machine has ("No
dictation for now" preselected, writing nothing), and the Parakeet path reads
choice-then-action with the expert options under Advanced.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Collapsible, RadioButton, Select, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_speech_engine as engines
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_speech_step import SpeechSetupStep

pytestmark = pytest.mark.bootstrap_profile

_BUNDLE = str(
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
)

FACTS = engines.EngineFacts(
    installed_engine="parakeet-mlx", openai_key=True, parakeet_mib=633
)


class _FakeService:
    def get_installed(self, ref):
        return None

    def list_installed(self, kind=None):
        return []


class _Host(ConsolidatedCSSApp):
    CSS_PATH = _BUNDLE

    def __init__(self, step) -> None:
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _step(*, facts=FACTS, engine=None, runtime=True, app_config=None):
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config or {}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = SpeechSetupStep(
        wizard=wizard,
        config=WizardStepConfig(id="speech", title="Speech", step_number=6),
        service_factory=_FakeService,
        runtime_installed=lambda: runtime,
        engine=engine,
        engine_facts=facts,
    )
    return wizard, step


def _labels(step) -> list[str]:
    return [str(b.label) for b in step.query("#setup-speech-engine RadioButton")]


# -- AC#8: the engine choice ---------------------------------------------------


@pytest.mark.asyncio
async def test_engine_choice_is_built_from_what_is_detected() -> None:
    _wizard, step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        labels = _labels(step)
        pressed = [
            b.id for b in step.query("#setup-speech-engine RadioButton") if b.value
        ]
        parakeet_shown = step.query_one("#setup-speech-parakeet").display
    assert labels == [
        "No dictation for now",
        "Already available — Parakeet MLX (installed)",
        "On this computer — Parakeet (633 MiB download)",
        "Cloud — OpenAI Whisper (audio is sent to OpenAI)",
    ]
    assert pressed == ["setup-speech-engine-none"]
    assert parakeet_shown is False


@pytest.mark.asyncio
async def test_no_cloud_row_without_an_openai_key_and_no_installed_row_without_an_engine() -> None:
    _wizard, step = _step(facts=engines.EngineFacts(parakeet_mib=633))
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        labels = _labels(step)
    assert labels == [
        "No dictation for now",
        "On this computer — Parakeet (633 MiB download)",
    ]


@pytest.mark.asyncio
async def test_no_dictation_for_now_writes_nothing() -> None:
    wizard, step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, error = await step.commit()
    assert (ok, error) == (True, "")
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_available_writes_the_engine_and_its_own_model_on_next() -> None:
    wizard, step = _step(
        app_config={
            "transcription": {
                "default_provider": "faster-whisper",
                "default_model": "distil-large-v3",
            }
        }
    )
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-speech-engine-installed", RadioButton).value = True
        await pilot.pause()
        wizard.commit_config.assert_not_awaited()  # only on Next
        ok, _ = await step.commit()
    assert ok
    assert wizard.commit_config.await_args.args[0] == {
        "transcription": {
            "default_provider": "parakeet-mlx",
            "default_model": "mlx-community/parakeet-tdt-0.6b-v2",
        }
    }


@pytest.mark.asyncio
async def test_cloud_whisper_points_remote_whisper_at_openai_without_a_key_copy() -> None:
    wizard, step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-speech-engine-cloud", RadioButton).value = True
        await pilot.pause()
        detail = str(step.query_one("#setup-speech-engine-detail", Static).render())
        ok, _ = await step.commit()
    assert ok
    committed = wizard.commit_config.await_args.args[0]
    assert committed == {
        "transcription": {"default_provider": "remote-whisper", "default_model": "whisper-1"},
        "transcription.remote_whisper": {
            "enabled": True,
            "api_endpoint": "https://api.openai.com/v1/audio/transcriptions",
            "model": "whisper-1",
        },
    }
    assert wizard_state.commit_sections_allowed(committed)
    assert "api_key" not in str(committed)
    assert "Live dictation in Console stays on this computer" in detail


def test_a_rerun_with_cloud_saved_starts_on_cloud_and_writes_nothing_unchanged() -> None:
    transcription = {
        "default_provider": "remote-whisper",
        "default_model": "whisper-1",
        "remote_whisper": {
            "enabled": True,
            "api_endpoint": engines.OPENAI_TRANSCRIPTIONS_URL,
        },
    }
    assert engines.saved_engine(transcription, FACTS) == engines.ENGINE_CLOUD
    assert (
        engines.build_engine_commit(
            engines.ENGINE_CLOUD, facts=FACTS, transcription=transcription
        )
        == {}
    )
    # The shipped template default is not a choice the user made.
    assert (
        engines.saved_engine({"default_provider": "parakeet-mlx"}, FACTS)
        == engines.ENGINE_NONE
    )


def test_faster_whisper_writes_the_shipped_template_model() -> None:
    """Dev's TASK-32755.4 made ``base`` the shipped ``[transcription]`` model
    (it is also TranscriptionService's own faster-whisper fallback). Picking
    "Already available -- faster-whisper" on a fresh profile changes nothing,
    and from another engine writes that same model, never the larger
    ``distil-large-v3`` the old template named."""
    import tomllib

    from tldw_chatbook import config as config_module

    shipped = tomllib.loads(config_module.CONFIG_TOML_CONTENT)["transcription"]
    facts = engines.EngineFacts(installed_engine="faster-whisper", parakeet_mib=633)
    fresh = {"default_provider": "faster-whisper", "default_model": shipped["default_model"]}
    assert (
        engines.build_engine_commit(
            engines.ENGINE_INSTALLED, facts=facts, transcription=fresh
        )
        == {}
    )
    assert engines.build_engine_commit(
        engines.ENGINE_INSTALLED,
        facts=facts,
        transcription={"default_provider": "remote-whisper", "default_model": "whisper-1"},
    ) == {
        "transcription": {
            "default_provider": "faster-whisper",
            "default_model": shipped["default_model"],
        }
    }


def test_installed_engine_follows_console_dictations_resolver() -> None:
    installed = ("parakeet-onnx", "parakeet-mlx", "faster-whisper")
    assert engines.pick_installed_engine("faster-whisper", installed) == "faster-whisper"
    assert engines.pick_installed_engine("remote-whisper", installed) == "parakeet-mlx"
    assert engines.pick_installed_engine("", ("parakeet-onnx",)) == ""


@pytest.mark.parametrize(
    ("app_config", "environ", "expected"),
    [
        ({"api_settings": {"openai": {"api_key": "sk-real-looking-key"}}}, {}, True),
        ({"api_settings": {"openai": {"api_key": "<API_KEY_HERE>"}}}, {}, False),
        ({}, {"OPENAI_API_KEY": "sk-env-key"}, True),
        ({}, {}, False),
    ],
)
def test_openai_key_detection_never_counts_a_placeholder(app_config, environ, expected) -> None:
    assert engines.openai_key_configured(app_config, environ) is expected


async def _speech_in_wizard(pilot, wizard, monkeypatch):
    """The Speech step shown in the real wizard frame, its on-show check settled."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

    await pilot.pause(0.3)
    container = wizard.query_one(SetupWizardContainer)
    container.select_track(wizard_state.TRACK_FULL)
    container.show_step(container._step_index_for_id(wizard_state.STEP_SPEECH))
    step = wizard.query_one(SpeechSetupStep)
    step._runtime_installed = lambda: True
    for _ in range(80):
        await pilot.pause(0.05)
        if step._loaded and not step._loading:
            break
    await pilot.pause()
    return step


def _wizard_with_facts(monkeypatch):
    from tldw_chatbook.UI.Wizards import first_run_speech_step as speech_module
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

    monkeypatch.setattr(speech_module.engines, "detect_engine_facts", lambda *a, **k: FACTS)
    monkeypatch.setattr(speech_module, "parakeet_v2_managed_service", _FakeService)
    app_instance = MagicMock()
    app_instance.app_config = {}
    return FirstRunSetupWizard(app_instance)


@pytest.mark.asyncio
async def test_each_down_press_moves_to_the_next_engine(monkeypatch) -> None:
    """Review F7 (TASK-34100.13): an engine change rebuilt the whole step, and
    the new radio's highlight sat on the first row, so every second Down
    press re-selected the engine already chosen. A change no longer rebuilds
    the radio."""
    wizard = _wizard_with_facts(monkeypatch)
    async with _WizardHost(wizard).run_test(size=(120, 40)) as pilot:
        step = await _speech_in_wizard(pilot, wizard, monkeypatch)
        engine = step.query_one("#setup-speech-engine")
        engine.focus()
        await pilot.pause()
        seen = []
        for _ in range(3):
            await pilot.press("down")
            await pilot.pause()
            seen.append(step._engine)
        same_radio = step.query_one("#setup-speech-engine") is engine
    assert seen == [
        engines.ENGINE_INSTALLED,
        engines.ENGINE_PARAKEET,
        engines.ENGINE_CLOUD,
    ]
    assert same_radio


@pytest.mark.asyncio
async def test_arriving_at_speech_leaves_the_keyboard_on_the_engine_choice(
    monkeypatch,
) -> None:
    """Live (TASK-34100.13 resume): reaching Speech with Next, Down and Space
    did nothing until the radio was clicked. The wizard focused the engine
    radio, then the step's first-show reload rebuilt it and the focus went
    with the old widget."""
    wizard = _wizard_with_facts(monkeypatch)
    async with _WizardHost(wizard).run_test(size=(120, 40)) as pilot:
        step = await _speech_in_wizard(pilot, wizard, monkeypatch)
        focused = wizard.app.focused
        engine = step.query_one("#setup-speech-engine")
        on_engine = focused is not None and engine in focused.ancestors_with_self
        await pilot.press("down")
        await pilot.pause()
        chosen = step._engine
    assert on_engine, f"focus is on {focused!r}"
    assert chosen == engines.ENGINE_INSTALLED


@pytest.mark.asyncio
async def test_after_arrival_a_rebuild_leaves_the_keyboard_where_the_user_put_it(
    monkeypatch,
) -> None:
    """The arrival hand-back happens once. A step that arrived while the
    keyboard was still on Next gets the engine radio on its first rebuild;
    once the user moves to Next, a later rebuild (an install, a reload) must
    not pull the keyboard back into the step."""
    wizard = _wizard_with_facts(monkeypatch)
    async with _WizardHost(wizard).run_test(size=(120, 40)) as pilot:
        step = await _speech_in_wizard(pilot, wizard, monkeypatch)
        next_button = wizard.query_one("#wizard-next")
        step._arriving = True  # arrived; the wizard's focus call has not landed
        next_button.focus()
        await pilot.pause()
        await step.recompose()
        await pilot.pause()
        handed_back = wizard.app.focused
        engine = step.query_one("#setup-speech-engine")
        on_engine = handed_back is not None and engine in handed_back.ancestors_with_self
        next_button.focus()  # the user tabs on to Next
        await pilot.pause()
        await step.recompose()
        await pilot.pause()
        after = wizard.app.focused
    assert on_engine, f"arrival left the focus on {handed_back!r}"
    assert after is next_button, f"focus moved to {after!r}"


@pytest.mark.asyncio
async def test_a_rebuilt_engine_radio_keeps_its_highlight_on_the_choice(
    monkeypatch,
) -> None:
    """The step still rebuilds for its own lifecycle (an install, a reload);
    the engine radio's highlight then stays on the chosen engine."""
    wizard = _wizard_with_facts(monkeypatch)
    async with _WizardHost(wizard).run_test(size=(120, 40)) as pilot:
        step = await _speech_in_wizard(pilot, wizard, monkeypatch)
        step._engine = engines.ENGINE_CLOUD
        await step.recompose()
        await pilot.pause()
        radio = step.query_one("#setup-speech-engine")
        highlighted, pressed = radio._selected, radio.pressed_index
    assert pressed == 3
    assert highlighted == pressed


@pytest.mark.asyncio
async def test_the_engine_choice_is_locked_while_a_disk_model_is_checked(
    monkeypatch,
) -> None:
    """Review F7: switching away mid-"Use model from disk…" hid the Cancel
    button, and Next on another engine skipped the busy guard."""
    wizard = _wizard_with_facts(monkeypatch)
    async with _WizardHost(wizard).run_test(size=(120, 40)) as pilot:
        step = await _speech_in_wizard(pilot, wizard, monkeypatch)
        step._engine = engines.ENGINE_PARAKEET
        step._external_busy = True
        await step.recompose()
        await pilot.pause()
        locked = step.query_one("#setup-speech-engine").disabled
        step._engine = engines.ENGINE_NONE  # however it got there
        result = await step.commit()
        step._external_busy = False
    assert locked is True
    assert result == (False, "Wait for external model verification to finish.")


# -- AC#9: choice, then action ---------------------------------------------------


async def _parakeet_step(pilot, step) -> None:
    step.query_one("#setup-speech-engine-parakeet", RadioButton).value = True
    # The engine change shows the Parakeet block in place; the installed-state
    # check may still rebuild the step once; wait for the settled tree.
    for _ in range(60):
        await pilot.pause(0.05)
        installs = list(step.query("#setup-speech-install"))
        if installs and installs[0].region.height and not step._loading:
            break


class _WizardHost(ConsolidatedCSSApp):
    CSS_PATH = _BUNDLE

    def __init__(self, wizard) -> None:
        super().__init__()
        self._wizard = wizard

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard)


@pytest.mark.parametrize("size", [(120, 40)])
@pytest.mark.asyncio
async def test_parakeet_path_reads_choice_then_action(monkeypatch, size) -> None:
    """In the real wizard frame: choice first, the primary action on screen."""
    from tldw_chatbook.UI.Wizards import first_run_speech_step as speech_module
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        FirstRunSetupWizard,
        SetupWizardContainer,
    )

    monkeypatch.setattr(speech_module.engines, "detect_engine_facts", lambda *a, **k: FACTS)
    monkeypatch.setattr(speech_module, "parakeet_v2_managed_service", _FakeService)
    app_instance = MagicMock()
    app_instance.app_config = {}
    wizard = FirstRunSetupWizard(app_instance)
    app = _WizardHost(wizard)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.3)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(wizard_state.TRACK_FULL)
        container.show_step(container._step_index_for_id(wizard_state.STEP_SPEECH))
        step = wizard.query_one(SpeechSetupStep)
        await pilot.pause(0.2)
        step._runtime_installed = lambda: True
        await _parakeet_step(pilot, step)
        card = str(step.query_one("#setup-speech-recommendation", Static).render())
        language = step.query_one("#setup-speech-language", Select)
        install = step.query_one("#setup-speech-install", Button)
        advanced = step.query_one("#setup-speech-advanced", Collapsible)
        inside = {w.id for w in advanced.query("*") if w.id}
        precision_labels = [
            str(b.label) for b in step.query("#setup-speech-precision-choice RadioButton")
        ]
        language_region, install_region = language.region, install.region
        viewport = step.region
    assert card.startswith("Recommended: Parakeet v2")
    assert "Selected:" not in card
    assert "633 MiB" in card
    assert 0 < language_region.height <= 3, "Language must be one row, not a list"
    assert language_region.y < install_region.y
    assert install_region.height > 0
    assert viewport.contains_region(install_region), (
        f"Review and install… is not visible at {size}: {install_region} vs {viewport}"
    )
    assert advanced.collapsed is True
    assert {
        "setup-speech-precision-choice",
        "setup-speech-use-from-disk",
        "setup-speech-choose-transcribe-cpp-gguf",
    } <= inside
    assert precision_labels[0].startswith("Smaller — recommended")
    assert any(label.startswith("Full precision, larger") for label in precision_labels)


@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
@pytest.mark.asyncio
async def test_every_parakeet_control_scrolls_into_view_with_advanced_open(
    monkeypatch, size
) -> None:
    """Review V1 (TASK-34100.13): the Parakeet block was a bare Vertical (1fr),
    so it was sized to the leftover viewport, not its content: the step
    under-reported its height and could not scroll. At 120x40 with Advanced
    open, Precision and both disk options were cut off; at 80x24 even
    "Review and install…" could not be reached."""
    wizard = _wizard_with_facts(monkeypatch)
    async with _WizardHost(wizard).run_test(size=size) as pilot:
        step = await _speech_in_wizard(pilot, wizard, monkeypatch)
        await _parakeet_step(pilot, step)
        step.query_one("#setup-speech-advanced", Collapsible).collapsed = False
        for _ in range(20):
            await pilot.pause(0.05)
        unreachable = []
        for selector in (
            "#setup-speech-language",
            "#setup-speech-install",
            "#setup-speech-precision-choice",
            "#setup-speech-use-from-disk",
            "#setup-speech-choose-transcribe-cpp-gguf",
        ):
            widget = step.query_one(selector)
            step.scroll_to_widget(widget, animate=False, immediate=True)
            await pilot.pause()
            await pilot.pause()
            region, viewport = widget.region, step.region
            if not region.height or not viewport.contains_region(region):
                unreachable.append(f"{selector} {region} vs {viewport}")
    assert unreachable == [], f"out of reach at {size}: {unreachable}"


@pytest.mark.asyncio
async def test_runtime_missing_copy_names_the_extras_package_with_a_copy_action() -> None:
    _wizard, step = _step(runtime=False)
    app = _Host(step)
    copied: list[str] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _parakeet_step(pilot, step)
        status = str(step.query_one("#setup-speech-status", Static).render())
        text = "\n".join(str(s.render()) for s in step.query(Static))
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        step.query_one("#setup-speech-copy-extras", Button).press()
        await pilot.pause()
    assert status == engines.RUNTIME_MISSING_COPY
    assert engines.PARAKEET_EXTRAS_PACKAGE in text
    assert "pip" not in text.lower()
    assert "==" not in text
    assert copied == [engines.PARAKEET_EXTRAS_PACKAGE]


def _real_config(tmp_path, monkeypatch, toml_text: str, environ: dict) -> dict:
    """Point the real loader at a scratch TOML (no stand-in for its reads)."""
    from Tests.Backup_Recovery.config_test_support import install_config_source

    for name in ("OPENAI_API_KEY", "MY_OPENAI"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    config_path = tmp_path / "speech-cloud-config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    fresh = install_config_source(monkeypatch)
    fresh.load_cli_config_and_ensure_existence(force_reload=True)
    return fresh.load_settings(force_reload=True)


_PROVIDER_KEY = '[api_settings.openai]\napi_key = "sk-provider-key"\n'


def test_remote_whisper_reuses_the_openai_provider_key_only_for_openai(
    tmp_path, monkeypatch
) -> None:
    from tldw_chatbook.Local_Ingestion import transcription_service as ts

    _real_config(tmp_path, monkeypatch, _PROVIDER_KEY, {})
    assert ts._openai_key_for_endpoint(engines.OPENAI_TRANSCRIPTIONS_URL) == "sk-provider-key"
    assert ts._openai_key_for_endpoint("http://localhost:8000/v1/audio/transcriptions") == ""


def test_remote_whisper_never_sends_the_key_over_plain_http(tmp_path, monkeypatch) -> None:
    """Review F6 (TASK-34100.13): a hand-set http://api.openai.com endpoint got
    the bearer key in clear text; the host check did not require https."""
    from tldw_chatbook.Local_Ingestion import transcription_service as ts

    _real_config(tmp_path, monkeypatch, _PROVIDER_KEY, {})
    assert ts._openai_key_for_endpoint("http://api.openai.com/v1/audio/transcriptions") == ""


# -- the Summary row reads what the engine choice wrote ---------------------------


def _speech_row(config, **kwargs):
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            config, {}, rag_deps_installed=True, **kwargs
        )
    }
    return rows["Speech transcription"]


def test_summary_names_cloud_whisper_after_the_cloud_choice() -> None:
    row = _speech_row(
        {
            "transcription": {
                "default_provider": "remote-whisper",
                "default_model": "whisper-1",
                "remote_whisper": {
                    "enabled": True,
                    "api_endpoint": engines.OPENAI_TRANSCRIPTIONS_URL,
                },
            },
            "api_settings": {"openai": {"api_key": "sk-provider-key"}},
        }
    )
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail == "OpenAI Whisper (cloud; for audio and video files)"


def test_summary_names_an_installed_local_engine_honestly() -> None:
    config = {"transcription": {"default_provider": "parakeet-mlx"}}
    row = _speech_row(config, installed_stt=("parakeet-mlx",))
    assert row.state == wizard_state.ROW_DEFAULT
    assert row.detail == "Parakeet MLX on this computer"
    missing = _speech_row(config, installed_stt=())
    assert missing.detail == "not set up (optional)"


@pytest.mark.asyncio
async def test_a_key_saved_on_connect_this_run_adds_the_cloud_row_on_show(monkeypatch) -> None:
    """Live: the step is built before Connect saves a key; showing it re-detects."""
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    step = SpeechSetupStep(
        wizard=wizard,
        config=WizardStepConfig(id="speech", title="Speech", step_number=6),
        service_factory=_FakeService,
        runtime_installed=lambda: True,
    )
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not any("Cloud" in label for label in _labels(step))
        # Connect saves the key and mirrors it into app_config.
        wizard.app_instance.app_config["api_settings"] = {
            "openai": {"api_key": "sk-saved-on-connect"}
        }
        step.on_show()
        for _ in range(20):
            await pilot.pause(0.05)
            if any("Cloud" in label for label in _labels(step)):
                break
        labels = _labels(step)
    assert "Cloud — OpenAI Whisper (audio is sent to OpenAI)" in labels


# -- review v5r1 R1: a self-hosted Whisper server is not "Cloud — OpenAI" ------

SELF_HOSTED = {
    "default_provider": "remote-whisper",
    "default_model": "large-v3",
    "remote_whisper": {
        "enabled": True,
        "api_endpoint": "http://whisper.lan:9000/v1/audio/transcriptions",
        "api_key": "selfhosted-secret",
    },
}


def test_a_self_hosted_remote_whisper_is_not_the_cloud_choice() -> None:
    """Review R1 (TASK-34100.13): any enabled remote-whisper counted as the
    Cloud choice, so an untouched re-run's Next repointed a self-hosted
    server at api.openai.com -- with that server's own key as the bearer."""
    assert engines.saved_engine(SELF_HOSTED, FACTS) == engines.ENGINE_NONE
    facts = engines.detect_engine_facts({"transcription": SELF_HOSTED}, environ={})
    assert facts.kept_remote == "whisper.lan"


@pytest.mark.asyncio
async def test_an_untouched_rerun_over_a_self_hosted_server_writes_nothing() -> None:
    from dataclasses import replace

    facts = replace(FACTS, kept_remote="whisper.lan")
    wizard, step = _step(facts=facts, app_config={"transcription": SELF_HOSTED})
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pressed = [
            b.id for b in step.query("#setup-speech-engine RadioButton") if b.value
        ]
        detail = str(step.query_one("#setup-speech-engine-detail", Static).render())
        ok, error = await step.commit()
    assert pressed == ["setup-speech-engine-none"]
    assert "whisper.lan" in detail and "Nothing is changed" in detail
    assert (ok, error) == (True, "")
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_untouched_rerun_over_cloud_with_its_own_model_writes_nothing() -> None:
    """Only a choice made this run writes [transcription] (AC#8): a saved
    OpenAI endpoint with a model of the user's own is not reset to whisper-1
    by a Next that never touched the step."""
    transcription = {
        "default_provider": "remote-whisper",
        "default_model": "gpt-4o-transcribe",
        "remote_whisper": {
            "enabled": True,
            "api_endpoint": engines.OPENAI_TRANSCRIPTIONS_URL,
        },
    }
    wizard, step = _step(app_config={"transcription": transcription})
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pressed = [
            b.id for b in step.query("#setup-speech-engine RadioButton") if b.value
        ]
        ok, _ = await step.commit()
    assert pressed == ["setup-speech-engine-cloud"]
    assert ok
    wizard.commit_config.assert_not_awaited()


def test_choosing_cloud_over_a_self_hosted_server_drops_that_servers_key() -> None:
    """A key issued for the self-hosted server is never sent to OpenAI: the
    Cloud choice blanks it (and an env-var name meant for that server) as it
    repoints the endpoint."""
    commit = engines.build_engine_commit(
        engines.ENGINE_CLOUD, facts=FACTS, transcription=SELF_HOSTED
    )
    assert commit["transcription.remote_whisper"] == {
        "enabled": True,
        "api_endpoint": engines.OPENAI_TRANSCRIPTIONS_URL,
        "model": engines.OPENAI_WHISPER_MODEL,
        "api_key": "",
    }
    by_env = {
        "default_provider": "faster-whisper",
        "remote_whisper": {
            "enabled": False,
            "api_endpoint": "http://localhost:8000/v1/audio/transcriptions",
            "api_key_env_var": "REMOTE_WHISPER_API_KEY",
        },
    }
    remote = engines.build_engine_commit(
        engines.ENGINE_CLOUD, facts=FACTS, transcription=by_env
    )["transcription.remote_whisper"]
    assert remote["api_key_env_var"] == ""
    assert "api_key" not in remote
    # A key the user gave this same OpenAI endpoint stays theirs.
    own_openai = {
        "default_provider": "faster-whisper",
        "remote_whisper": {
            "enabled": False,
            "api_endpoint": engines.OPENAI_TRANSCRIPTIONS_URL,
            "api_key": "sk-their-own",
        },
    }
    remote = engines.build_engine_commit(
        engines.ENGINE_CLOUD, facts=FACTS, transcription=own_openai
    )["transcription.remote_whisper"]
    assert "api_key" not in remote and "api_key_env_var" not in remote


def test_summary_names_openai_only_for_openais_endpoint() -> None:
    row = _speech_row({"transcription": SELF_HOSTED})
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail == "Whisper server at whisper.lan (for audio and video files)"


# -- review v5r1 R4: "Already available" keeps a model the user chose -----------


def test_already_available_keeps_a_model_the_user_chose_for_that_engine() -> None:
    facts = engines.EngineFacts(installed_engine="faster-whisper")
    assert (
        engines.build_engine_commit(
            engines.ENGINE_INSTALLED,
            facts=facts,
            transcription={"default_provider": "faster-whisper", "default_model": "large-v3"},
        )
        == {}
    )


@pytest.mark.parametrize("model", ["base", "distil-large-v3", ""])
def test_already_available_replaces_a_model_the_engine_cannot_load(model) -> None:
    """parakeet-mlx asked for a Whisper id (the shipped template's ``base``,
    or the older ``distil-large-v3``) 404s; that model is not a choice."""
    facts = engines.EngineFacts(installed_engine="parakeet-mlx")
    assert engines.build_engine_commit(
        engines.ENGINE_INSTALLED,
        facts=facts,
        transcription={"default_provider": "parakeet-mlx", "default_model": model},
    ) == {"transcription": {"default_model": "mlx-community/parakeet-tdt-0.6b-v2"}}


# -- review v5r1 R2: Cloud follows the provider's own credential rule ------------


@pytest.mark.parametrize(
    ("openai_table", "environ", "expected"),
    [
        ({"credential_source": "none", "api_key": "sk-table-key"}, {"OPENAI_API_KEY": "sk-env"}, False),
        ({"credential_source": "stored"}, {"OPENAI_API_KEY": "sk-env-key"}, False),
        ({"api_key_env_var": "MY_OPENAI"}, {"OPENAI_API_KEY": "sk-env-key"}, False),
        ({"api_key_env_var": "MY_OPENAI"}, {"MY_OPENAI": "sk-named-key"}, True),
        ({"credential_source": "environment", "api_key": "sk-table-key"}, {}, False),
        ({"api_key": "sk-table-key"}, {}, True),
    ],
)
def test_cloud_is_offered_only_on_a_key_the_provider_resolves(
    openai_table, environ, expected
) -> None:
    """Review R2 (TASK-34100.13): the F6 resolver ignored dev's ADR-012
    ``credential_source`` and a named ``api_key_env_var``, so Cloud was
    offered (and remote-whisper spent) a key the provider config excludes."""
    app_config = {"api_settings": {"openai": openai_table}}
    assert engines.openai_key_configured(app_config, environ) is expected


_REAL_TOML_CASES = [
    # Review F6: a legacy-only key is offered and sent alike.
    ('[API]\nopenai_api_key = "sk-legacy-only-key"\n', {}, "sk-legacy-only-key"),
    # The provider's own table outranks its environment variable.
    (_PROVIDER_KEY, {"OPENAI_API_KEY": "sk-env-key"}, "sk-provider-key"),
    ("", {"OPENAI_API_KEY": "sk-env-key"}, "sk-env-key"),
    (
        '[api_settings.openai]\napi_key = "<API_KEY_HERE>"\n'
        '[API]\nopenai_api_key = "sk-legacy-only-key"\n',
        {},
        "sk-legacy-only-key",
    ),
    ('[API]\nopenai_api_key = "<API_KEY_HERE>"\n', {}, ""),
    # Review R2: dev's ADR-012 credential_source and a named variable decide.
    (
        '[api_settings.openai]\ncredential_source = "none"\n'
        '[API]\nopenai_api_key = "sk-legacy-only-key"\n',
        {"OPENAI_API_KEY": "sk-env-key"},
        "",
    ),
    (
        '[api_settings.openai]\ncredential_source = "stored"\n',
        {"OPENAI_API_KEY": "sk-env-key"},
        "",
    ),
    (
        '[api_settings.openai]\napi_key_env_var = "MY_OPENAI"\n',
        {"OPENAI_API_KEY": "sk-env-key", "MY_OPENAI": "sk-named-key"},
        "sk-named-key",
    ),
]


@pytest.mark.parametrize(("toml_text", "environ", "expected"), _REAL_TOML_CASES)
def test_readiness_and_spend_read_one_real_config_alike(
    tmp_path, monkeypatch, toml_text, environ, expected
) -> None:
    """T-G13-6: both halves read the same real TOML through the real loader
    (no stand-in for get_cli_setting): the Speech step's Cloud offer and the
    key remote-whisper sends to api.openai.com agree."""
    import os

    from tldw_chatbook.Local_Ingestion import transcription_service as ts

    settings = _real_config(tmp_path, monkeypatch, toml_text, environ)
    offered = engines.openai_key_configured(settings, os.environ)
    sent = ts._openai_key_for_endpoint(engines.OPENAI_TRANSCRIPTIONS_URL)
    assert sent == expected
    assert offered is bool(expected)


# -- review v5r2 LV1-F1: OpenAI Whisper saved, but no key to send -------------

_SAVED_OPENAI = {
    "default_provider": "remote-whisper",
    "default_model": "whisper-1",
    "remote_whisper": {
        "enabled": True,
        "api_endpoint": engines.OPENAI_TRANSCRIPTIONS_URL,
        "api_key_env_var": "",
    },
}
#: The live case: credential_source = "stored" with nothing stored, and the
#: key only in the environment -- the provider rule excludes it.
_STORED_NOTHING = {"openai": {"credential_source": "stored"}}
_ENV_ONLY = {"OPENAI_API_KEY": "sk-env-only-key"}


def test_saved_openai_whisper_without_a_key_says_so_on_the_step() -> None:
    """Review LV1-F1 (TASK-34100.13): the step hid Cloud and said "Nothing is
    changed. Set dictation up later…" -- as if no dictation were saved --
    while [transcription] still sent audio files to OpenAI with no key."""
    app_config = {"transcription": _SAVED_OPENAI, "api_settings": _STORED_NOTHING}
    facts = engines.detect_engine_facts(app_config, environ=_ENV_ONLY)
    assert facts.openai_key is False
    assert facts.kept_openai == engines.KEPT_OPENAI_NO_KEY
    assert engines.saved_engine(_SAVED_OPENAI, facts) == engines.ENGINE_NONE
    none = engines.engine_choices(facts)[0]
    assert none.key == engines.ENGINE_NONE
    assert none.detail.startswith("Nothing is changed: ")
    assert "OpenAI Whisper" in none.detail
    assert "no OpenAI key is available" in none.detail
    assert "Settings ▸ Providers & Models" in none.detail
    assert "Set dictation up later" not in none.detail


def test_saved_openai_whisper_without_a_key_is_no_tick_on_the_summary() -> None:
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            {"transcription": _SAVED_OPENAI, "api_settings": _STORED_NOTHING},
            _ENV_ONLY,
            rag_deps_installed=True,
        )
    }
    row = rows["Speech transcription"]
    assert row.state == wizard_state.ROW_ATTENTION
    assert "OpenAI Whisper" in row.detail
    assert "no OpenAI key is available" in row.detail
    assert "Settings ▸ Providers & Models" in row.detail


@pytest.mark.parametrize(
    ("remote_extra", "environ", "app_extra"),
    [
        # A key given to the endpoint itself, or through the variable it names.
        ({"api_key": "sk-their-own"}, {}, {}),
        ({"api_key_env_var": "REMOTE_WHISPER_API_KEY"}, {"REMOTE_WHISPER_API_KEY": "sk-rw"}, {}),
        # A legacy [API] key, which the Summary's raw config read has not folded.
        ({}, {}, {"API": {"openai_api_key": "sk-legacy-only-key"}}),
    ],
)
def test_saved_openai_whisper_with_a_key_it_can_send_keeps_its_tick(
    remote_extra, environ, app_extra
) -> None:
    """The controls: any key the transcription service would send (its own
    key, its named variable, or the OpenAI provider's, a legacy one included)
    keeps the ✓ -- the attention mark is only for a key nothing resolves."""
    transcription = {
        **_SAVED_OPENAI,
        "remote_whisper": {**_SAVED_OPENAI["remote_whisper"], **remote_extra},
    }
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            {"transcription": transcription, **app_extra},
            environ,
            rag_deps_installed=True,
        )
    }
    row = rows["Speech transcription"]
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail == "OpenAI Whisper (cloud; for audio and video files)"


def test_saved_openai_whisper_with_its_own_key_says_it_is_kept() -> None:
    transcription = {
        **_SAVED_OPENAI,
        "remote_whisper": {**_SAVED_OPENAI["remote_whisper"], "api_key": "sk-their-own"},
    }
    facts = engines.detect_engine_facts(
        {"transcription": transcription, "api_settings": _STORED_NOTHING},
        environ=_ENV_ONLY,
    )
    assert facts.kept_openai == engines.KEPT_OPENAI_KEYED
    detail = engines.engine_choices(facts)[0].detail
    assert detail.startswith("Nothing is changed: ")
    assert "keep going to OpenAI Whisper" in detail
    assert "no OpenAI key" not in detail


@pytest.mark.asyncio
async def test_an_untouched_step_over_keyless_openai_whisper_shows_it_and_writes_nothing() -> None:
    from dataclasses import replace

    facts = replace(FACTS, openai_key=False, kept_openai=engines.KEPT_OPENAI_NO_KEY)
    wizard, step = _step(facts=facts, app_config={"transcription": _SAVED_OPENAI})
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pressed = [
            b.id for b in step.query("#setup-speech-engine RadioButton") if b.value
        ]
        detail = str(step.query_one("#setup-speech-engine-detail", Static).render())
        ok, _ = await step.commit()
    assert pressed == ["setup-speech-engine-none"]
    assert "no OpenAI key is available" in detail
    assert ok
    wizard.commit_config.assert_not_awaited()


# -- review v5r2 LV1-F2: the Language Select survives a recompose mid-mount ------


@pytest.mark.asyncio
async def test_the_language_select_survives_being_pruned_mid_mount() -> None:
    """Review LV1-F2 (TASK-34100.13): under load a mounted walk and a Speech
    test raised NoMatches('#label' on SelectCurrent) from Select._on_mount.
    The step recomposes from a dozen places (show, reload, every install
    state), and a recompose or teardown that prunes the Language Select
    inside its own mount cascade hits Textual's unguarded #label lookup
    (TASK-1960). Dev's Speech step had no Select; this branch added it, so it
    uses the project's PruneSafeSelect. The state is reconstructed as in
    Tests/Widgets/test_prune_safe_select.py (a stock Select raises in it)."""
    from textual import events
    from textual.widgets._select import SelectCurrent

    _wizard, step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _parakeet_step(pilot, step)
        select = step.query_one("#setup-speech-language", Select)
        current = select.query_one(SelectCurrent)
        await current.query_children("*").remove()
        select.set_reactive(Select.value, Select.NULL)
        select._pruning = True
        current._pruning = True
        select._on_mount(events.Mount())
