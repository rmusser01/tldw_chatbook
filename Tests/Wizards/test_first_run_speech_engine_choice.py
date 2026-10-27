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


def _raw_settings(monkeypatch, *, openai=None, legacy_key=None) -> None:
    """Stand in for get_cli_setting's raw-TOML reads (no load_settings fold)."""

    def get_cli_setting(section, key=None, default=None):
        if section == "api_settings.openai":
            return dict(openai or {})
        if (section, key) == ("API", "openai_api_key"):
            return legacy_key if legacy_key is not None else default
        return default if key is not None else key

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", get_cli_setting)


def test_remote_whisper_reuses_the_openai_provider_key_only_for_openai(monkeypatch) -> None:
    from tldw_chatbook.Local_Ingestion import transcription_service as ts

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _raw_settings(monkeypatch, openai={"api_key": "sk-provider-key"})
    assert ts._openai_key_for_endpoint(engines.OPENAI_TRANSCRIPTIONS_URL) == "sk-provider-key"
    assert ts._openai_key_for_endpoint("http://localhost:8000/v1/audio/transcriptions") == ""


def test_remote_whisper_never_sends_the_key_over_plain_http(monkeypatch) -> None:
    """Review F6 (TASK-34100.13): a hand-set http://api.openai.com endpoint got
    the bearer key in clear text; the host check did not require https."""
    from tldw_chatbook.Local_Ingestion import transcription_service as ts

    _raw_settings(monkeypatch, openai={"api_key": "sk-provider-key"})
    assert ts._openai_key_for_endpoint("http://api.openai.com/v1/audio/transcriptions") == ""


def test_a_legacy_only_key_is_offered_and_used_alike(monkeypatch) -> None:
    """Review F6 (TASK-34100.13): the step counted a legacy ``[API]
    openai_api_key`` and offered Cloud, but transcription read only the raw
    ``[api_settings.openai]`` table and the environment, so remote-whisper
    sent no key (401). Readiness and spend now share one resolver."""
    from tldw_chatbook.Local_Ingestion import transcription_service as ts

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app_config = {"API": {"openai_api_key": "sk-legacy-only-key"}}
    assert engines.openai_key_configured(app_config, {}) is True
    _raw_settings(monkeypatch, legacy_key="sk-legacy-only-key")
    assert ts._openai_key_for_endpoint(engines.OPENAI_TRANSCRIPTIONS_URL) == (
        "sk-legacy-only-key"
    )


@pytest.mark.parametrize(
    ("openai", "legacy", "environ", "expected"),
    [
        ({"api_key": "sk-table"}, "sk-legacy", {"OPENAI_API_KEY": "sk-env"}, "sk-table"),
        ({}, "sk-legacy", {"OPENAI_API_KEY": "sk-env"}, "sk-env"),
        ({"api_key_env_var": "MY_KEY"}, None, {"MY_KEY": "sk-mine"}, "sk-mine"),
        ({"api_key": "<API_KEY_HERE>"}, "sk-legacy", {}, "sk-legacy"),
        ({}, "<API_KEY_HERE>", {}, None),
    ],
)
def test_one_resolver_orders_table_then_environment_then_legacy(
    openai, legacy, environ, expected
) -> None:
    from tldw_chatbook.config import resolve_openai_provider_key

    assert resolve_openai_provider_key(openai, legacy, environ) == expected


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
                "remote_whisper": {"enabled": True},
            }
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
