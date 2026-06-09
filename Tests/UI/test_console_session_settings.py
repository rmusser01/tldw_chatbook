from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text as _screen_visible_text,
)
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_default_console_session_settings,
    build_console_settings_summary_state,
    validate_console_session_settings,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens import provider_model_resolution
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal
from tldw_chatbook.Widgets.Console import console_settings_summary as settings_summary_module
from tldw_chatbook.Widgets.Console.console_settings_summary import ConsoleSettingsSummary
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry


class SummaryHarness(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self, state: ConsoleSettingsSummaryState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield ConsoleSettingsSummary(self.state)


class ModalHarness(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
                "openai": {"api_key": "test-key"},
            },
        }
        self.saved_settings: ConsoleSessionSettings | None = None

    def capture_saved_settings(self, settings: ConsoleSessionSettings | None) -> None:
        self.saved_settings = settings


class StyledModalHarness(ModalHarness):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
    )


class FakeConsoleModelDiscoveryScope:
    def __init__(self, entries: tuple[MergedModelEntry, ...]) -> None:
        self.entries = entries
        self.merge_calls = []

    async def merge_saved_and_discovered_models(self, **kwargs):
        self.merge_calls.append(kwargs)
        return self.entries


class FailingConsoleModelDiscoveryScope:
    async def merge_saved_and_discovered_models(self, **kwargs):
        raise RuntimeError("merge failed")


def _visible_text(app: App[None]) -> str:
    return " ".join(str(widget.renderable) for widget in app.screen.query(Static))


def _summary_text(console) -> str:
    summary = console.query_one("#console-settings-summary")
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in summary.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


async def _wait_for_console_settings_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-settings-modal")
            and host.screen_stack[-1].query("#console-settings-provider")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console settings modal did not open")


async def _wait_for_console_top_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            return
        await pilot.pause(0.05)
    raise AssertionError("Console settings modal did not dismiss")


async def _wait_for_focused_id(host: App[None], pilot, widget_id: str) -> None:
    for _ in range(40):
        focused_id = getattr(host.focused, "id", None)
        if focused_id == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Expected focus on {widget_id!r}, found {getattr(host.focused, 'id', None)!r}")


async def _press_new_console_tab(console, store, pilot) -> str:
    previous_session_id = store.active_session_id
    console.query_one("#console-new-chat-tab", Button).press()
    for _ in range(40):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError("New Console tab did not activate")


async def _click_console_session_tab(console, store, pilot, session_id: str) -> None:
    await pilot.click(f"#console-session-tab-{session_id}")
    for _ in range(40):
        if store.active_session_id == session_id:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    console._ensure_console_chat_controller().switch_session(session_id)
    await console._sync_native_console_chat_ui()
    if store.active_session_id == session_id:
        await pilot.pause()
        return
    raise AssertionError(f"Console tab {session_id!r} did not activate")


def _select_values(select: Select) -> set[str]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    values: set[str] = set()
    for option in options:
        value = getattr(option, "value", None)
        if value is None and isinstance(option, tuple) and len(option) >= 2:
            value = option[1]
        if value is not None:
            values.add(str(value))
    return values


def _merged_model(
    model_id: str,
    *,
    source: str = "saved",
    capability_status: str = "known",
    persisted: bool = True,
) -> MergedModelEntry:
    return MergedModelEntry(
        provider="openai",
        provider_list_key="openai",
        model_id=model_id,
        display_name=model_id,
        source=source,
        capability_status=capability_status,
        persisted=persisted,
    )


@pytest.mark.asyncio
async def test_console_settings_summary_renders_rows_and_button() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama.cpp",
        model_row="Model: model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Persona: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Session Settings" in text
        assert "Provider: llama.cpp" in text
        assert "Model: model-a" in text
        assert "Context: 12 / 4k" in text
        assert "Sampling: T 0.70, P 0.95" in text
        assert "Persona: General" in text
        header = app.query_one("#console-settings-header", Horizontal)
        title = app.query_one("#console-settings-title", Static)
        button = app.query_one("#console-settings-open", Button)
        assert title.parent is header
        assert button.parent is header
        assert title.region.y == button.region.y
        assert str(button.label) == "Configure"
        assert button.tooltip == "Configure Console settings"


@pytest.mark.asyncio
async def test_console_settings_summary_uses_direct_choose_model_action_when_setup_blocked() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama.cpp",
        model_row="Model: Missing",
        context_row="Context: unavailable",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Persona: General",
        readiness_label="Missing model",
        action_label="Choose Model",
        action_tooltip="Choose a model for this Console session",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Provider: llama.cpp" in text
        assert "Model: Missing" in text
        button = app.query_one("#console-settings-open", Button)
        assert str(button.label) == "Choose Model"
        assert button.tooltip == "Choose a model for this Console session"


@pytest.mark.asyncio
async def test_console_settings_summary_treats_missing_provider_row_as_blank() -> None:
    state = ConsoleSettingsSummaryState(
        provider_row=None,  # type: ignore[arg-type]
        model_row="Model: model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Persona: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        provider_row = app.query_one("#console-settings-provider-row", Static)
        assert str(provider_row.renderable) == ""
        assert "None" not in _visible_text(app)

        updated_state = ConsoleSettingsSummaryState(
            provider_row=None,  # type: ignore[arg-type]
            model_row="Model: model-b",
            context_row="Context: 20 / 4k",
            sampling_row="Sampling: T 0.20, P 0.90",
            identity_row="Persona: Analyst",
            readiness_label="Ready",
        )
        app.query_one(ConsoleSettingsSummary).sync_state(updated_state)
        await pilot.pause()

        assert str(provider_row.renderable) == ""
        assert "None" not in _visible_text(app)


def test_console_settings_summary_button_sizing_uses_named_constants() -> None:
    assert settings_summary_module.CONSOLE_SETTINGS_SUMMARY_MAX_HEIGHT == 9
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING == 2
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_MIN_WIDTH == 9
    assert settings_summary_module.CONSOLE_SETTINGS_BUTTON_MAX_WIDTH == 14
    assert settings_summary_module.CONSOLE_SETTINGS_ROW_HEIGHT == 1


def test_pending_launch_inspector_auto_open_docstring_is_google_style() -> None:
    docstring = ChatScreen._apply_pending_launch_inspector_auto_open.__doc__

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring


def test_summary_state_appends_non_ready_readiness_to_model_row() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="WIP", detail="Provider not wired yet.", native_send_supported=False),
    )

    assert state.provider_row == "Provider: llama_cpp"
    assert state.model_row == "Model: model-a (WIP)"
    assert state.readiness_label == "WIP"


def test_default_console_session_settings_prefers_provider_model_profile() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": False,
                "model_defaults": {
                    "gpt-4.1": {
                        "temperature": 0.2,
                        "top_p": 0.88,
                        "min_p": 0.04,
                        "top_k": 40,
                        "max_tokens": 1234,
                        "seed": 101,
                        "presence_penalty": 0.2,
                        "frequency_penalty": 0.3,
                        "reasoning_effort": "high",
                        "reasoning_summary": "auto",
                        "verbosity": "high",
                        "streaming": True,
                    },
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.provider == "openai"
    assert settings.model == "gpt-4.1"
    assert settings.temperature == 0.2
    assert settings.top_p == 0.88
    assert settings.min_p == 0.04
    assert settings.top_k == 40
    assert settings.max_tokens == 1234
    assert settings.seed == 101
    assert settings.presence_penalty == 0.2
    assert settings.frequency_penalty == 0.3
    assert settings.reasoning_effort == "high"
    assert settings.reasoning_summary == "auto"
    assert settings.verbosity == "high"
    assert settings.streaming is True


def test_default_console_session_settings_prefers_chat_defaults_over_provider_scalars() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": True,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.9
    assert settings.top_p == 0.8
    assert settings.streaming is False


def test_console_session_settings_accepts_documented_effort_values() -> None:
    app_config = {
        "api_settings": {
            "openai": {"api_key": "test-key", "model": "gpt-5.1"},
            "anthropic": {"api_key": "test-key", "model": "claude-opus-4-8"},
        }
    }
    openai_settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-5.1",
        reasoning_effort="none",
    )
    anthropic_settings = ConsoleSessionSettings(
        provider="anthropic",
        model="claude-opus-4-8",
        thinking_effort="max",
    )

    assert validate_console_session_settings(openai_settings, app_config=app_config) == []
    assert validate_console_session_settings(anthropic_settings, app_config=app_config) == []


def test_default_console_session_settings_reads_enable_streaming_as_compatibility_fallback() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "enable_streaming": False,
        },
        "api_settings": {
            "openai": {
                "streaming": True,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.streaming is False


def test_default_console_session_settings_prefers_canonical_streaming_over_enable_streaming() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "streaming": True,
            "enable_streaming": False,
        },
        "api_settings": {
            "openai": {
                "streaming": False,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.streaming is True


def test_default_console_session_settings_uses_global_fallbacks_when_profile_is_absent() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.33,
            "top_p": 0.81,
            "max_tokens": 2048,
            "streaming": False,
        },
        "api_settings": {
            "openai": {},
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.33
    assert settings.top_p == 0.81
    assert settings.max_tokens == 2048
    assert settings.streaming is False


def test_default_console_session_settings_skips_blank_model_profile_values() -> None:
    app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1",
            "temperature": 0.9,
            "top_p": 0.8,
            "streaming": False,
        },
        "api_settings": {
            "openai": {
                "temperature": 0.7,
                "top_p": 0.95,
                "streaming": True,
                "model_defaults": {
                    "gpt-4.1": {
                        "temperature": "",
                        "top_p": " ",
                    },
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config,
        provider="openai",
        model="gpt-4.1",
    )

    assert settings.temperature == 0.9
    assert settings.top_p == 0.8


def test_summary_state_keeps_missing_model_row_compact() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model=None),
        ConsoleSettingsContextEstimate(used_tokens=None, token_limit=None, label="unknown"),
        ConsoleSettingsReadiness(
            label="Missing model",
            detail="Select a model before sending.",
            native_send_supported=False,
        ),
    )

    assert state.provider_row == "Provider: llama_cpp"
    assert state.model_row == "Model: Missing"
    assert state.readiness_label == "Missing model"
    assert state.action_label == "Choose Model"
    assert state.action_tooltip == "Choose a model for this Console session"


def test_summary_state_appends_optional_sampling_fields_only_when_set() -> None:
    without_optional = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a", temperature=0.7, top_p=0.95),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )
    with_optional = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            temperature=0.7,
            top_p=0.95,
            min_p=0.05,
            top_k=40,
            max_tokens=512,
        ),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )

    assert without_optional.sampling_row == "Sampling: T 0.70, P 0.95"
    assert with_optional.sampling_row == "Sampling: T 0.70, P 0.95, min_p 0.05, top_k 40, max_tokens 512"


def test_summary_state_normalizes_unknown_context_label() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(used_tokens=None, token_limit=None, label="Context: unknown"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )

    assert state.context_row == "Context: unavailable"


def test_summary_state_prefers_character_label_over_persona_label() -> None:
    character = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            persona_label="General",
            character_label="Ada",
        ),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )
    persona = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a", persona_label="Mentor"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )
    fallback = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a", persona_label=""),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )

    assert character.identity_row == "Character: Ada"
    assert persona.identity_row == "Persona: Mentor"
    assert fallback.identity_row == "Persona: General"


def test_choose_model_action_label_normalization() -> None:
    assert ChatScreen._is_console_choose_model_action(" Choose Model ")
    assert ChatScreen._is_console_choose_model_action("choose model")
    assert ChatScreen._is_console_choose_model_action("CHOOSE MODEL")
    assert not ChatScreen._is_console_choose_model_action("Configure")


@pytest.mark.asyncio
async def test_console_model_resolution_includes_runtime_discovered_models() -> None:
    scope = FakeConsoleModelDiscoveryScope(
        (
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
            _merged_model("gpt-4.1"),
        )
    )
    app = SimpleNamespace(
        providers_models={"openai": ["gpt-4.1"]},
        llm_provider_catalog_scope_service=scope,
    )

    options = await provider_model_resolution.resolve_provider_model_options(
        app,
        provider="OpenAI",
    )

    assert [option.model_id for option in options] == ["gpt-4.1", "gpt-5"]
    assert options[1].warning == "Capabilities unknown until saved or verified; text chat is assumed."
    assert scope.merge_calls == [
        {
            "mode": "local",
            "provider": "openai",
        }
    ]


@pytest.mark.asyncio
async def test_console_model_resolution_failure_logs_provider_context(monkeypatch) -> None:
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.llm_provider_catalog_scope_service = FailingConsoleModelDiscoveryScope()
    console = ChatScreen(app)
    logged = []

    def fake_exception(message, *args, **kwargs):
        logged.append((message, args, kwargs))

    monkeypatch.setattr(chat_screen_module.logger, "exception", fake_exception)

    models = await console._providers_models_for_console_settings(
        "OpenAI",
        current_model="gpt-5",
    )

    assert models == {"openai": ["gpt-4.1"]}
    assert logged == [
        (
            "Unable to resolve Console runtime-discovered models for provider=%s model=%s",
            ("openai", "gpt-5"),
            {},
        )
    ]


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_discards_draft() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="should-clear")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a", "model-b"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    used_tokens=10,
                    token_limit=4096,
                    label="10 / 4k",
                ),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        await pilot.click("#console-settings-cancel")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_escape_dismisses_none() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="should-clear")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        await pilot.press("escape")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_save_returns_validated_settings() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-provider-model-section")
        assert "Choose a model to enable sending." not in str(readiness.renderable)
        assert provider_model_section.has_class("console-settings-primary-section") is False
        app.screen.query_one("#console-settings-temperature", Input).value = "0.42"
        app.screen.query_one("#console-settings-top-p", Input).value = "0.88"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"
    assert app.saved_settings.temperature == 0.42
    assert app.saved_settings.top_p == 0.88


@pytest.mark.parametrize(
    ("field_id", "attribute", "backspace_count", "typed_suffix", "expected"),
    [
        ("console-settings-temperature", "temperature", 0, "1", 0.71),
        ("console-settings-top-p", "top_p", 1, "6", 0.96),
        ("console-settings-min-p", "min_p", 1, "6", 0.06),
        ("console-settings-top-k", "top_k", 1, "1", 41),
        ("console-settings-max-tokens", "max_tokens", 1, "5", 65),
    ],
)
@pytest.mark.asyncio
async def test_console_settings_modal_accepts_keyboard_edited_sampling_inputs(
    field_id: str,
    attribute: str,
    backspace_count: int,
    typed_suffix: str,
    expected: float | int,
) -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.70,
        top_p=0.95,
        min_p=0.05,
        top_k=40,
        max_tokens=64,
    )

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        target_input = app.screen.query_one(f"#{field_id}", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(target_input)
        await pilot.pause()
        await pilot.click(target_input)
        await pilot.press("end")
        for _ in range(backspace_count):
            await pilot.press("backspace")
        await pilot.press(typed_suffix)
        assert str(expected) in target_input.value

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert getattr(app.saved_settings, attribute) == expected


@pytest.mark.asyncio
async def test_console_settings_modal_shows_inherited_provider_endpoint() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", base_url=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert base_url_input.display is True
        assert base_url_input.disabled is False
        assert base_url_input.value == "http://127.0.0.1:9099"


@pytest.mark.asyncio
async def test_console_settings_modal_prefers_api_base_url_alias_over_default_api_url() -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://localhost:8080/completion",
        "api_base_url": "http://127.0.0.1:9099/v1",
    }
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", base_url=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        assert base_url_input.value == "http://127.0.0.1:9099"
        assert "Provider blocked" not in str(readiness.renderable)
        assert "localhost:8080" not in str(readiness.renderable)


@pytest.mark.asyncio
async def test_console_settings_modal_replaces_stale_lower_priority_endpoint_with_alias() -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://localhost:8080/completion",
        "api_base_url": "http://127.0.0.1:9099/v1",
    }
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://localhost:8080",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        assert base_url_input.value == "http://127.0.0.1:9099"
        assert "Provider blocked" not in str(readiness.renderable)
        assert "localhost:8080" not in str(readiness.renderable)


@pytest.mark.asyncio
async def test_console_settings_modal_focus_mode_uses_ready_copy_when_model_selected() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-provider-model-section")
        assert str(readiness.renderable) == "llama_cpp is ready. No API key is required."
        assert provider_model_section.has_class("console-settings-primary-section") is False


@pytest.mark.asyncio
async def test_console_settings_modal_clears_setup_copy_when_dropdown_model_is_available() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": ["freeform-model"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-provider-model-section")
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        readiness_copy = str(readiness.renderable)
        assert "Choose a model to enable sending." not in readiness_copy
        assert "not wired yet" not in readiness_copy
        assert "custom is ready" in str(readiness.renderable)
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert provider_model_section.has_class("console-settings-primary-section") is False


@pytest.mark.asyncio
async def test_console_settings_modal_setup_copy_preserves_blocking_readiness_detail() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model=None,
        base_url="ftp://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                focus_model=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        readiness = app.screen.query_one("#console-settings-readiness", Static)
        readiness_copy = str(readiness.renderable)
        assert "Choose a model to enable sending." in readiness_copy
        assert "Provider blocked: invalid llama.cpp base URL." in readiness_copy


@pytest.mark.asyncio
async def test_console_settings_modal_invalid_temperature_stays_open_and_renders_error() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-temperature", Input).value = "3.0"
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Temperature must be between 0 and 2." in str(
            app.screen.query_one("#console-settings-error", Static).renderable
        )

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_blank_temperature_stays_open_and_renders_error() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", temperature=0.7)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-temperature", Input).value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Temperature is required." in str(app.screen.query_one("#console-settings-error", Static).renderable)

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_blank_top_p_stays_open_and_renders_error() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", top_p=0.95)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-top-p", Input).value = ""
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert app.screen.query_one("#console-settings-modal") is not None
        assert "Top P is required." in str(app.screen.query_one("#console-settings-error", Static).renderable)

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_save_disabled_when_cannot_save() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=False,
            )
        )
        await pilot.pause()

        assert app.screen.query_one("#console-settings-save", Button).disabled is True


@pytest.mark.asyncio
async def test_console_settings_modal_has_stable_body_error_and_footer_regions() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(100, 32)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        body = app.screen.query_one("#console-settings-body")
        error = app.screen.query_one("#console-settings-error", Static)
        actions = app.screen.query_one("#console-settings-actions")
        temperature = app.screen.query_one("#console-settings-temperature", Input)

        assert "console-settings-body" in body.classes
        assert "console-settings-error-summary" in error.classes
        assert "console-settings-modal-actions" in actions.classes
        assert "console-settings-control" in temperature.classes
        assert error.region.y < body.region.y < actions.region.y


@pytest.mark.asyncio
async def test_console_settings_modal_renders_context_and_identity_read_only_rows() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        persona_label="Planner",
        character_label="Ada",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    10,
                    4096,
                    "10 / 4k",
                    staged_source_count=2,
                    staged_context_summary="2 staged sources",
                ),
                can_save=True,
            )
        )
        await pilot.pause()

        text = _visible_text(app)
        assert "Current" in str(app.screen.query_one("#console-settings-context-current", Static).renderable)
        assert "10 / 4k tokens" in text
        assert "2 staged sources" in str(app.screen.query_one("#console-settings-context-sources", Static).renderable)
        assert "Estimate only; no truncation changes in this version." in str(
            app.screen.query_one("#console-settings-context-note", Static).renderable
        )
        assert "Planner / Ada" in str(app.screen.query_one("#console-settings-identity-current", Static).renderable)
        assert "Planner [read-only]" in str(
            app.screen.query_one("#console-settings-persona-readonly", Static).renderable
        )
        assert "Ada [read-only]" in str(app.screen.query_one("#console-settings-character-readonly", Static).renderable)
        assert not app.screen.query("#console-settings-persona-input")
        assert not app.screen.query("#console-settings-character-input")


@pytest.mark.asyncio
async def test_console_settings_modal_provider_select_lists_all_configured_providers() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "openai": ["gpt-4"],
                    "custom": [],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_values = _select_values(app.screen.query_one("#console-settings-provider", Select))
        assert {"custom", "llama_cpp", "openai"}.issubset(provider_values)


@pytest.mark.asyncio
async def test_console_settings_modal_uses_model_dropdown_without_configured_models() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model="freeform-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert "freeform-model" in _select_values(model_select)
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_uses_first_model_when_initial_model_missing() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "gpt-4.1"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_settings_modal_keyboard_selects_model_from_dropdown() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1", "gpt-5"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_select.focus()
        await pilot.press("enter")
        assert model_select.expanded is True

        await pilot.press("down")
        await pilot.press("enter")
        assert model_select.expanded is False
        assert model_select.value == "gpt-5"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-5"


@pytest.mark.asyncio
async def test_console_settings_modal_keyboard_selects_provider_and_refreshes_models() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "local_llamacpp": ["local-model"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert provider_select.value == "llama_cpp"
        assert model_select.value == "model-a"

        provider_select.focus()
        await pilot.press("enter")
        assert provider_select.expanded is True

        await pilot.press("down")
        await pilot.press("enter")
        assert provider_select.expanded is False
        assert provider_select.value == "local_llamacpp"
        assert model_select.disabled is False
        assert model_select.value == "local-model"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "local_llamacpp"
    assert app.saved_settings.model == "local-model"


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_missing_registry_model_for_current_provider() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="custom-openai-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "custom-openai-model"
        assert {"custom-openai-model", "gpt-4.1"}.issubset(_select_values(model_select))
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "custom-openai-model"


@pytest.mark.asyncio
async def test_console_settings_modal_can_select_runtime_discovered_model_with_warning() -> None:
    app = _build_test_app()
    app.providers_models = {"openai": ["gpt-4.1"]}
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": "test-key"}}
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (
            _merged_model("gpt-4.1"),
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 60)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-open")
        await pilot.click("#console-settings-open")
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        model_select = modal_screen.query_one("#console-settings-model-select", Select)
        assert {"gpt-4.1", "gpt-5"}.issubset(_select_values(model_select))

        model_select.value = "gpt-5"
        await pilot.pause()
        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        for _ in range(40):
            summary_text = _summary_text(console)
            if "Model: gpt-5 (Capabilities unknown)" in summary_text:
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(f"Console summary did not show discovered-model warning: {summary_text}")

        _settings, readiness = console._active_console_settings_readiness()
        assert readiness.native_send_supported is True


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_to_no_models_allows_freeform_model_entry() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "custom": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is False
        assert model_select.disabled is True
        assert "model-a" not in _select_values(model_select)
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_input.value == ""
        assert model_input.placeholder == "Enter model id"
        model_input.value = "freeform-model"
        await pilot.pause()
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "custom"
    assert app.saved_settings.model == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_accepts_keyboard_edited_freeform_model_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "koboldcpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "koboldcpp"
        await pilot.pause()

        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_input.placeholder == "Enter model id"

        await pilot.click(model_input)
        for character in "local-model":
            await pilot.press(character)
        assert model_input.value == "local-model"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "koboldcpp"
    assert app.saved_settings.model == "local-model"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_target_provider_model() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "gpt-4.1"
        assert "model-a" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.model == "gpt-4.1"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_round_trip_ignores_none_model_sentinel() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="koboldcpp",
        model=None,
        base_url="http://localhost:5001/api/v1/generate",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "koboldcpp": ["None"],
                    "Llama_cpp": ["None"],
                    "llama_cpp": ["model-a"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.disabled is False
        assert model_select.value == "model-a"
        assert "None" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_console_settings_modal_existing_none_model_sentinel_is_not_saved() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="None",
        base_url="http://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Llama_cpp": ["None"],
                    "llama_cpp": ["model-a"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.value == "model-a"
        assert "None" not in _select_values(model_select)
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "model-a"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_does_not_carry_base_url_to_non_url_provider() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "openai"
        await pilot.pause()

        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert base_url_input.disabled is True or base_url_input.display is False
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "openai"
    assert app.saved_settings.base_url is None


@pytest.mark.asyncio
async def test_console_settings_modal_restores_freeform_model_after_provider_round_trip() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="custom", model="freeform-model")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"custom": [], "llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert app.screen.query_one("#console-settings-model-select", Select).value == "model-a"

        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "freeform-model"
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "freeform-model"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "custom"
    assert app.saved_settings.model == "freeform-model"


@pytest.mark.asyncio
async def test_console_left_rail_renders_settings_below_staged_context() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        assert staged_context.region.y < settings.region.y < workspace_context.region.y
        assert settings.region.width == staged_context.region.width


@pytest.mark.asyncio
async def test_console_left_rail_body_scrolls_below_fixed_header() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        left_rail = console.query_one("#console-left-rail")
        header = console.query_one(".console-rail-header")
        body = console.query_one("#console-left-rail-body")
        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        assert body.region.y >= header.region.y + header.region.height
        assert body.region.height <= left_rail.region.height - header.region.height
        assert settings.parent is body
        assert workspace_context.parent is body
        assert staged_context.region.width == settings.region.width
        assert settings.region.width == workspace_context.region.width
        assert settings.region.width <= body.region.width
        assert body.region.width - settings.region.width <= 2


@pytest.mark.asyncio
async def test_console_settings_modal_save_updates_active_summary_only() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(first.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a"))
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(second_id, ConsoleSessionSettings(provider="llama_cpp", model="model-a"))
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        console.query_one("#console-settings-open", Button).press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(ConsoleSessionSettings(provider="openai", model="gpt-4.1"))
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        summary_text = _summary_text(console)
        assert "Provider: openai" in summary_text
        assert "Model: gpt-4.1" in summary_text
        assert store.session_settings(second_id).provider == "openai"
        assert store.session_settings(first.id).provider == "llama_cpp"

        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        summary_text = _summary_text(console)
        assert "Provider: llama_cpp" in summary_text
        assert "Model: model-a" in summary_text


@pytest.mark.asyncio
async def test_console_settings_are_isolated_between_native_tabs() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        store.replace_session_settings(first.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a"))
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        store.replace_session_settings(second_id, ConsoleSessionSettings(provider="llama_cpp", model="model-a"))
        await console._sync_native_console_chat_ui()
        console.query_one("#console-settings-open", Button).press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(ConsoleSessionSettings(provider="openai", model="gpt-4.1"))
        await _wait_for_console_top_screen(host, console, pilot)
        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert console._build_console_provider_selection().provider == "llama_cpp"
        await _click_console_session_tab(console, store, pilot, second_id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        assert console._build_console_provider_selection().provider == "openai"


@pytest.mark.asyncio
async def test_console_provider_selection_includes_generation_controls() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="openai",
                model="gpt-4.1",
                seed=17,
                presence_penalty=0.4,
                frequency_penalty=0.5,
                reasoning_effort="high",
                reasoning_summary="auto",
                verbosity="medium",
                thinking_effort="low",
                thinking_budget_tokens=2048,
            ),
        )
        await console._sync_native_console_chat_ui()

        selection = console._build_console_provider_selection()

    assert selection.seed == 17
    assert selection.presence_penalty == 0.4
    assert selection.frequency_penalty == 0.5
    assert selection.reasoning_effort == "high"
    assert selection.reasoning_summary == "auto"
    assert selection.verbosity == "medium"
    assert selection.thinking_effort == "low"
    assert selection.thinking_budget_tokens == 2048


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_keeps_original_summary() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-4.1"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(session.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a"))
        await console._sync_native_console_chat_ui()
        original_summary = _summary_text(console)

        console.query_one("#console-settings-open", Button).press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(None)
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert _summary_text(console) == original_summary
        assert store.session_settings(session.id).provider == "llama_cpp"


@pytest.mark.asyncio
async def test_console_settings_modal_save_disabled_during_active_run() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"}}
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(session.id, ConsoleSessionSettings(provider="llama_cpp", model="model-a"))
        await console._sync_native_console_chat_ui()
        controller = console._ensure_console_chat_controller()
        controller.run_state = ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")

        console.query_one("#console-settings-open", Button).press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        assert modal_screen.query_one("#console-settings-save", Button).disabled is True


@pytest.mark.asyncio
async def test_console_send_blocker_uses_saved_unsupported_session_provider() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(session.id, ConsoleSessionSettings(provider="wip_provider", model="test-model"))
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        for _ in range(40):
            if "Provider blocked" in _screen_visible_text(console):
                break
            await pilot.pause(0.05)

        assert "Provider blocked: 'wip_provider' is not available in Console yet." in _screen_visible_text(console)


@pytest.mark.asyncio
async def test_console_missing_model_opens_console_settings_from_summary() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-open")
        await _wait_for_selector(console, pilot, "#console-open-provider-settings")

        recovery_button = console.query_one("#console-open-provider-settings", Button)
        assert str(recovery_button.label) == "Choose model"
        assert recovery_button.display is False

        console.query_one("#console-settings-open", Button).press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        await _wait_for_focused_id(host, pilot, "console-settings-model-select")

        assert modal_screen.query_one("#console-settings-provider", Select).value == "llama_cpp"
        assert modal_screen.query_one("#console-settings-model-select", Select).value == "model-a"
        readiness = modal_screen.query_one("#console-settings-readiness", Static)
        provider_model_section = modal_screen.query_one("#console-settings-provider-model-section")
        assert str(readiness.renderable) == "llama_cpp is ready. No API key is required."
        assert provider_model_section.has_class("console-settings-primary-section") is False

        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        text = _screen_visible_text(console)
        assert "Model: model-a" in _summary_text(console)
        assert "Setup required: choose a model in Console Settings." not in text
        assert console._console_send_blocked_reason() == ""


@pytest.mark.asyncio
async def test_console_llamacpp_saved_missing_model_blocks_before_send() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(session.id, ConsoleSessionSettings(provider="llama_cpp", model=None))
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        send_button = console.query_one("#console-send-message", Button)
        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        assert send_button.disabled is False
        assert send_button.tooltip == "Choose a model in Console Settings before sending."
        assert "Console send blocked: Select a model before sending." not in _screen_visible_text(console)
        assert "Setup required: choose a model in Console Settings." not in _screen_visible_text(console)
        assert composer.draft_text() == "hello"


def test_console_default_settings_keep_configured_model_without_legacy_model() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)

    settings = screen._default_console_session_settings()

    assert settings.provider == "llama_cpp"
    assert settings.model == "configured-model"


def test_console_settings_summary_uses_effective_config_endpoint_for_llamacpp_defaults() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.endpoint_row == "Endpoint: http://127.0.0.1:9099"


def test_console_readiness_uses_saved_session_settings_over_stale_global_provider() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099/v1", "model": "local-model"},
        "openai": {},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(session.id, ConsoleSessionSettings(provider="llama_cpp", model="local-model"))

    control_state = screen._build_console_control_state(None)
    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")

    assert screen._console_provider_blocker_copy() == ""
    assert control_state.provider_label == "Provider: llama_cpp"
    assert control_state.model_label == "Model: local-model"
    assert provider_row.value == "ready"
    assert provider_row.recovery == ""


def test_console_saved_openai_with_key_shows_ready_readiness() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(session.id, ConsoleSessionSettings(provider="openai", model="gpt-4.1"))

    summary_state = screen._build_console_settings_summary_state()
    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    blocker_copy = screen._console_provider_blocker_copy()

    assert summary_state.readiness_label == "Ready"
    assert provider_row.value == "ready"
    assert provider_row.recovery == ""
    assert blocker_copy == ""
    assert screen._console_send_blocked_reason() == ""


def test_console_missing_key_recovery_action_is_provider_specific() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "gpt-4.1"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(session.id, ConsoleSessionSettings(provider="openai", model="gpt-4.1"))

    label, target, tooltip = screen._console_provider_recovery_action()

    assert screen._console_provider_blocker_copy() == "Provider setup needed: OpenAI missing API key"
    assert label == "Add API Key"
    assert target == "settings"
    assert tooltip == "Add an API key for OpenAI"
    assert screen._console_setup_blocked_reason() == "Add API Key in Settings before sending."


def test_console_unsaved_generic_endpoint_blocks_inspector_with_endpoint_details() -> None:
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "ollama": {"api_url": "http://127.0.0.1:11434", "model": "llama3"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(
            provider="ollama",
            model="llama3",
            base_url="http://127.0.0.1:9999/v1",
        ),
    )

    inspector_state = screen._build_console_inspector_state(None)
    provider_row = next(row for row in inspector_state.rows if row.label == "Provider")
    label, target, tooltip = screen._console_provider_recovery_action()

    assert provider_row.value == "blocked"
    assert "Selected endpoint: http://127.0.0.1:9999/v1" in provider_row.recovery
    assert "Saved endpoint: http://127.0.0.1:11434" in provider_row.recovery
    assert "save the endpoint in Settings" in screen._console_provider_blocker_copy()
    assert label == "Configure endpoint"
    assert target == "settings"
    assert tooltip == "Save the ollama endpoint in Settings"
    assert screen._console_setup_blocked_reason() == "Save provider endpoint in Settings before sending."


def test_console_saved_llamacpp_missing_model_summary_is_not_ready_without_fallback() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(session.id, ConsoleSessionSettings(provider="llama_cpp", model=None))

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.readiness_label != "Ready"
    assert summary_state.provider_row == "Provider: llama_cpp"
    assert summary_state.model_row == "Model: Missing"
    assert screen._console_send_blocked_reason() == "Console send blocked: Select a model before sending."


def test_console_saved_llamacpp_missing_model_summary_ready_with_configured_fallback() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "configured-model",
        },
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.replace_session_settings(session.id, ConsoleSessionSettings(provider="llama_cpp", model=None))

    summary_state = screen._build_console_settings_summary_state()

    assert summary_state.readiness_label == "Ready"
    assert "Select a model before sending" not in summary_state.model_row


@pytest.mark.asyncio
async def test_console_new_native_tab_receives_default_settings_snapshot() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = None
    app.app_config["chat_defaults"] = {"provider": "llama_cpp"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099/v1",
            "model": "configured-model",
        },
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id

        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert second_id != first_id
        settings = store.session_settings(second_id)
        assert settings is not None
        assert settings.provider == "llama_cpp"
        assert settings.model == "configured-model"


@pytest.mark.asyncio
async def test_console_model_switch_inherits_selected_model_default_profile() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    app.providers_models = {"openai": ["gpt-4.1", "gpt-4.1-mini"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()

        initial_settings = store.session_settings(session.id)
        assert initial_settings is not None
        assert initial_settings.model == "gpt-4.1"
        assert initial_settings.temperature == 0.2

        console._sync_compact_shell_controls(model="gpt-4.1-mini")
        await pilot.pause()

        updated_settings = store.session_settings(session.id)
        assert updated_settings is not None
        assert updated_settings.model == "gpt-4.1-mini"
        assert updated_settings.temperature == 0.45
        assert updated_settings.top_p == 0.9
        assert updated_settings.streaming is False


@pytest.mark.asyncio
async def test_console_model_switch_preserves_explicit_session_overrides() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_defaults": {
                "gpt-4.1": {"temperature": 0.2, "top_p": 0.8, "streaming": True},
                "gpt-4.1-mini": {"temperature": 0.45, "top_p": 0.9, "streaming": False},
            },
        },
    }
    app.providers_models = {"openai": ["gpt-4.1", "gpt-4.1-mini"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()

        console._sync_compact_shell_controls(temperature="0.33")
        console._sync_compact_shell_controls(model="gpt-4.1-mini")
        await pilot.pause()

        updated_settings = store.session_settings(session.id)
        assert updated_settings is not None
        assert updated_settings.model == "gpt-4.1-mini"
        assert updated_settings.temperature == 0.33
        assert updated_settings.top_p == 0.9
        assert updated_settings.streaming is False
