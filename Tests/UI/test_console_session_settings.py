from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.geometry import Region
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
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
    ChatScreen,
)
from tldw_chatbook.UI.Screens import provider_model_resolution
from tldw_chatbook.Chat.local_server_discovery import LocalModelProbeResult
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    MODAL_BODY_MIN_HEIGHT,
    MODAL_CONTROL_HEIGHT,
    MODEL_DISCOVER_BUTTON_ID,
    MODEL_DISCOVER_STATUS_ID,
    ConsoleSettingsInput,
    ConsoleSettingsModal,
    _settings_screen_region,
)
from tldw_chatbook.Widgets.Console import console_settings_summary as settings_summary_module
from tldw_chatbook.Widgets.Console.console_settings_summary import ConsoleSettingsSummary
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry
from tldw_chatbook.config import API_MODELS_BY_PROVIDER, DEFAULT_CONFIG_FROM_TOML


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


def test_console_settings_screen_region_prefers_absolute_region() -> None:
    absolute_region = Region(10, 20, 30, 1)
    widget = SimpleNamespace(
        region=Region(1, 2, 30, 1),
        screen_region=absolute_region,
    )

    assert _settings_screen_region(widget) == absolute_region


def test_console_settings_screen_region_falls_back_to_mounted_region() -> None:
    mounted_region = Region(3, 4, 30, 1)
    widget = SimpleNamespace(region=mounted_region)

    assert _settings_screen_region(widget) == mounted_region


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


class StyledConsoleHarness(ConsoleHarness):
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


def test_groq_console_default_uses_current_catalog_model() -> None:
    groq_settings = DEFAULT_CONFIG_FROM_TOML["api_settings"]["groq"]

    assert groq_settings["model"] == "llama-3.3-70b-versatile"
    assert groq_settings["model"] in API_MODELS_BY_PROVIDER["Groq"]
    assert groq_settings["model"] not in {"llama3-70b-8192", "llama3-8b-8192"}


def test_console_remote_defaults_use_smoke_verified_models() -> None:
    expected_defaults = {
        "anthropic": ("Anthropic", "claude-sonnet-4-20250514"),
        "cohere": ("Cohere", "command-a-03-2025"),
        "google": ("Google", "gemini-2.5-flash"),
        "huggingface": ("HuggingFace", "openai/gpt-oss-120b"),
    }

    for config_key, (catalog_key, expected_model) in expected_defaults.items():
        provider_settings = DEFAULT_CONFIG_FROM_TOML["api_settings"][config_key]

        assert provider_settings["model"] == expected_model
        assert expected_model in API_MODELS_BY_PROVIDER[catalog_key]


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


async def _visible_console_settings_button(console: ChatScreen, pilot) -> Button:
    """Open the inspector rail and return the actionable settings summary button."""
    rail_state = replace(
        console._current_console_rail_state(),
        right_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    assert rail_state.right_open is True
    await _wait_for_selector(console, pilot, "#console-settings-open")
    for _ in range(40):
        button = console.query_one("#console-settings-open", Button)
        if button.display and button.region.width > 0 and button.region.height > 0:
            return button
        await pilot.pause(0.05)
    button = console.query_one("#console-settings-open", Button)
    raise AssertionError(
        "Console settings button is not visible/actionable: "
        f"display={button.display!r} region={button.region!r}"
    )


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


def test_console_settings_modal_sizing_uses_named_constants() -> None:
    assert MODAL_BODY_MIN_HEIGHT == 0
    assert MODAL_CONTROL_HEIGHT == 3
    assert f"min-height: {MODAL_BODY_MIN_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS
    assert f"height: {MODAL_CONTROL_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS
    assert f"min-height: {MODAL_CONTROL_HEIGHT};" in ConsoleSettingsModal.DEFAULT_CSS


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


def test_summary_state_exposes_safe_credential_source() -> None:
    """Show safe env/config credential sources without exposing secret values."""
    env_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="OpenAI is ready. API key found via env:OPENAI_API_KEY.",
            native_send_supported=True,
        ),
    )
    config_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-20250514"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Anthropic is ready. API key found via config:api_settings.anthropic.api_key.",
            native_send_supported=True,
        ),
    )

    assert env_state.credential_row == "Credential: env OPENAI_API_KEY"
    assert config_state.credential_row == "Credential: config api_settings.anthropic.api_key"


def test_summary_state_handles_empty_credential_source_names() -> None:
    """Collapse empty env/config credential-source identifiers without padding."""
    env_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="OpenAI is ready. API key found via env:   .",
            native_send_supported=True,
        ),
    )
    config_state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-20250514"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(
            label="Ready",
            detail="Anthropic is ready. API key found via config:   .",
            native_send_supported=True,
        ),
    )

    assert env_state.credential_row == "Credential: env"
    assert config_state.credential_row == "Credential: config"


def test_summary_state_ignores_warning_lines_after_credential_source() -> None:
    """Keep appended readiness warnings out of the credential summary row."""
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(
            label="Ready",
            detail=(
                "OpenAI is ready. API key found via env:OPENAI_API_KEY.\n"
                "Model warning: selected model may not support native tools."
            ),
            native_send_supported=True,
        ),
    )

    assert state.credential_row == "Credential: env OPENAI_API_KEY"


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
async def test_console_settings_model_resolution_preserves_configured_alternatives() -> None:
    app = _build_test_app()
    app.providers_models = {
        "local_llamacpp": ["uat-local-model", "uat-alt-local-model"],
    }
    app.llm_provider_catalog_scope_service = FakeConsoleModelDiscoveryScope(
        (
            _merged_model(
                "uat-local-model",
                source="runtime_discovered",
                capability_status="known",
                persisted=False,
            ),
        )
    )
    console = ChatScreen(app)

    models = await console._providers_models_for_console_settings(
        "local_llamacpp",
        current_model="uat-local-model",
    )

    assert models["local_llamacpp"] == ["uat-local-model", "uat-alt-local-model"]


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


@pytest.mark.asyncio
async def test_console_settings_modal_single_model_uses_readonly_value_not_dead_dropdown() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

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

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        model_custom = app.screen.query_one("#console-settings-model-custom", Button)

        assert model_select.display is False
        assert model_select.disabled is True
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "model-a"
        assert model_custom.display is True
        assert model_custom.disabled is False

        model_custom.press()
        await pilot.pause()
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_custom.label == "Model list"

        model_custom.press()
        await pilot.pause()
        assert model_select.display is False
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "model-a"
        assert app.focused is model_custom


@pytest.mark.asyncio
async def test_console_settings_modal_saves_replaced_temperature_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.60,
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
        temperature = app.screen.query_one("#console-settings-temperature", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(temperature)
        await pilot.pause()

        await pilot.click(temperature)
        await pilot.press("ctrl+a")
        await pilot.press("0")
        await pilot.press(".")
        await pilot.press("7")
        assert temperature.value == "0.7"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.temperature == 0.7


@pytest.mark.asyncio
async def test_console_settings_modal_replaces_focused_sampling_input() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.60,
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
        temperature = app.screen.query_one("#console-settings-temperature", Input)
        body = app.screen.query_one("#console-settings-body")
        body.scroll_to_widget(temperature)
        await pilot.pause()

        await pilot.click(temperature)
        await pilot.press("0")
        await pilot.press(".")
        await pilot.press("7")
        await pilot.press("2")

        assert temperature.value == "0.72"


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
async def test_console_settings_modal_body_is_scrollable_container_for_overflow_controls() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        seed=17,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(140, 32)) as pilot:
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

        body = app.screen.query_one("#console-settings-body")
        assert isinstance(body, ScrollableContainer)


@pytest.mark.asyncio
async def test_console_settings_modal_preserves_provider_specific_generation_controls() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        seed=17,
        presence_penalty=0.4,
        frequency_penalty=0.5,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
    )

    async with app.run_test(size=(140, 60)) as pilot:
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

        for selector in (
            "#console-settings-seed",
            "#console-settings-presence-penalty",
            "#console-settings-frequency-penalty",
            "#console-settings-reasoning-effort",
            "#console-settings-reasoning-summary",
            "#console-settings-verbosity",
            "#console-settings-thinking-effort",
            "#console-settings-thinking-budget-tokens",
        ):
            input_widget = app.screen.query_one(selector, Input)
            body = app.screen.query_one("#console-settings-body")
            body.scroll_to_widget(input_widget)
            await pilot.pause()

            assert input_widget.display is True
            assert input_widget.disabled is False
            assert input_widget.value
            assert input_widget.content_region.height >= 1

        app.screen.query_one("#console-settings-seed", Input).value = "23"
        app.screen.query_one("#console-settings-presence-penalty", Input).value = "0.6"
        app.screen.query_one("#console-settings-frequency-penalty", Input).value = "0.7"
        app.screen.query_one("#console-settings-reasoning-effort", Input).value = "medium"
        app.screen.query_one("#console-settings-reasoning-summary", Input).value = "concise"
        app.screen.query_one("#console-settings-verbosity", Input).value = "high"
        app.screen.query_one("#console-settings-thinking-effort", Input).value = "medium"
        app.screen.query_one("#console-settings-thinking-budget-tokens", Input).value = "4096"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.seed == 23
    assert app.saved_settings.presence_penalty == 0.6
    assert app.saved_settings.frequency_penalty == 0.7
    assert app.saved_settings.reasoning_effort == "medium"
    assert app.saved_settings.reasoning_summary == "concise"
    assert app.saved_settings.verbosity == "high"
    assert app.saved_settings.thinking_effort == "medium"
    assert app.saved_settings.thinking_budget_tokens == 4096


@pytest.mark.asyncio
async def test_console_settings_modal_normalizes_provider_specific_choices() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-4.1",
        temperature=0.70,
        top_p=0.95,
        reasoning_effort="medium",
        reasoning_summary="concise",
        verbosity="low",
        thinking_effort="medium",
    )

    async with app.run_test(size=(140, 60)) as pilot:
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

        app.screen.query_one("#console-settings-reasoning-effort", Input).value = " HIGH "
        app.screen.query_one("#console-settings-reasoning-summary", Input).value = " AUTO "
        app.screen.query_one("#console-settings-verbosity", Input).value = " Medium "
        app.screen.query_one("#console-settings-thinking-effort", Input).value = " LOW "
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.reasoning_effort == "high"
    assert app.saved_settings.reasoning_summary == "auto"
    assert app.saved_settings.verbosity == "medium"
    assert app.saved_settings.thinking_effort == "low"


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
async def test_console_settings_modal_inputs_keep_visible_content_row_when_unfocused() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )

    async with app.run_test(size=(140, 60)) as pilot:
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

        for selector in (
            "#console-settings-base-url",
            "#console-settings-temperature",
            "#console-settings-top-p",
            "#console-settings-max-tokens",
        ):
            input_widget = app.screen.query_one(selector, Input)

            assert input_widget.value
            assert input_widget.content_region.height >= 1


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
async def test_console_settings_modal_tabs_to_model_select_after_provider_change() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        model_select = app.screen.query_one("#console-settings-model-select", Select)

        provider_select.focus()
        provider_select.value = "groq"
        await pilot.pause()

        assert model_select.disabled is False
        assert model_select.display is True
        assert model_select.value == "llama-3.3-70b-versatile"

        await pilot.press("tab")
        await _wait_for_focused_id(app, pilot, "console-settings-model-select")
        await pilot.press("enter")

        assert model_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_reopens_provider_select_after_input_edit() -> None:
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

        temperature = app.screen.query_one("#console-settings-temperature", Input)
        provider_select = app.screen.query_one("#console-settings-provider", Select)

        temperature.focus()
        temperature.value = "0.22"
        await pilot.pause()

        provider_select.focus()
        await pilot.press("enter")

        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_opens_provider_select_click_after_input_edit() -> None:
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

        temperature = app.screen.query_one("#console-settings-temperature", ConsoleSettingsInput)
        provider_select = app.screen.query_one("#console-settings-provider", Select)

        await pilot.click("#console-settings-temperature")
        temperature.value = "0.72"
        await pilot.pause()
        await pilot.click("#console-settings-provider")

        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_opens_screen_routed_select_click_after_input_edit() -> None:
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

        temperature = app.screen.query_one("#console-settings-temperature", ConsoleSettingsInput)
        provider_select = app.screen.query_one("#console-settings-provider", Select)

        temperature.focus()
        temperature.value = "0.72"
        await pilot.pause()

        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            app.screen,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_input_releases_mouse_capture_after_click_to_replace() -> None:
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(140, 60)) as pilot:
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

        temperature = app.screen.query_one("#console-settings-temperature", ConsoleSettingsInput)
        temperature.capture_mouse()

        assert app.mouse_captured is temperature

        temperature.on_click()

        assert app.mouse_captured is None
        assert temperature.selected_text == temperature.value


@pytest.mark.asyncio
async def test_console_settings_modal_opens_provider_select_from_redirected_input_click(monkeypatch) -> None:
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
            )
        )
        await pilot.pause()

        temperature = app.screen.query_one("#console-settings-temperature", ConsoleSettingsInput)
        provider_select = app.screen.query_one("#console-settings-provider", Select)
        temperature.capture_mouse()
        temperature.value = "0.22"

        provider_screen_region = provider_select.region.translate((10, 0))
        monkeypatch.setattr(
            Select,
            "screen_region",
            property(
                lambda widget: provider_screen_region
                if widget is provider_select
                else widget.region
            ),
            raising=False,
        )
        click = events.Click(
            temperature,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_screen_region.x + provider_screen_region.width - 1,
            screen_y=provider_screen_region.y,
        )

        temperature.on_click(click)

        assert app.mouse_captured is None
        assert provider_select.expanded is True


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_plain_select_click_without_redirected_input() -> None:
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
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            provider_select,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert app.mouse_captured is None
        assert provider_select.expanded is False


@pytest.mark.asyncio
async def test_console_settings_modal_ignores_screen_routed_select_click_without_input_focus() -> None:
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
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        cancel_button = app.screen.query_one("#console-settings-cancel", Button)
        cancel_button.focus()
        await pilot.pause()
        provider_region = _settings_screen_region(provider_select)
        click = events.Click(
            app.screen,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.x + provider_region.width - 1,
            screen_y=provider_region.y,
        )

        app.screen.on_click(click)

        assert getattr(app.focused, "id", None) == "console-settings-cancel"
        assert app.mouse_captured is None
        assert provider_select.expanded is False


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
async def test_console_settings_modal_allows_manual_model_when_registry_has_stale_options() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="anthropic", model="claude-3-haiku-20240307")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"anthropic": ["claude-3-haiku-20240307"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        custom_button = app.screen.query_one("#console-settings-model-custom", Button)
        assert model_select.display is True
        assert model_input.display is False
        assert custom_button.display is True

        await pilot.click("#console-settings-model-custom")
        await pilot.pause()

        assert model_select.display is False
        assert model_input.display is True
        assert model_input.disabled is False
        model_input.value = "claude-haiku-4-5-20251001"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "anthropic"
    assert app.saved_settings.model == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_console_settings_modal_refreshes_readiness_after_returning_to_model_list() -> None:
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

        await pilot.click("#console-settings-model-custom")
        await pilot.pause()

        model_input = app.screen.query_one("#console-settings-model-input", Input)
        readiness = app.screen.query_one("#console-settings-readiness", Static)
        provider_model_section = app.screen.query_one("#console-settings-provider-model-section")
        model_input.value = ""
        app.screen._sync_readiness_display()
        await pilot.pause()

        assert "Choose a model to enable sending." in str(readiness.renderable)
        assert provider_model_section.has_class("console-settings-primary-section") is True

        app.screen._toggle_manual_model_input()
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.value == "model-a"
        assert str(readiness.renderable) == "llama_cpp is ready. No API key is required."
        assert provider_model_section.has_class("console-settings-primary-section") is False


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_configured_provider_model() -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = {
        "api_url": "http://127.0.0.1:9099",
        "model": "gemma-local-config-model",
    }
    settings = ConsoleSessionSettings(
        provider="custom",
        model="custom-model-beta",
        base_url="http://localhost:1234/v1/chat/completions",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Custom": ["custom-model-alpha", "custom-model-beta"],
                    "Llama_cpp": ["None"],
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
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        base_url_input = app.screen.query_one("#console-settings-base-url", Input)
        assert model_select.display is True
        assert model_select.disabled is False
        assert model_select.value == "gemma-local-config-model"
        assert model_input.display is False
        assert model_input.disabled is True
        assert model_input.value == "gemma-local-config-model"
        assert base_url_input.value == "http://127.0.0.1:9099"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "gemma-local-config-model"
    assert app.saved_settings.base_url == "http://127.0.0.1:9099"


@pytest.mark.parametrize(
    ("provider_settings", "expected_model"),
    (
        (
            {
                "api_url": "http://127.0.0.1:9099",
                "api_model": "gemma-api-model",
            },
            "gemma-api-model",
        ),
        (
            {
                "api_url": "http://127.0.0.1:9099",
                "model": "None",
                "api_model": "null",
                "default_model": "gemma-default-model",
            },
            "gemma-default-model",
        ),
    ),
)
@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_uses_model_alias_fallbacks(
    provider_settings: dict[str, str],
    expected_model: str,
) -> None:
    app = ModalHarness()
    app.app_config["api_settings"]["llama_cpp"] = provider_settings
    settings = ConsoleSessionSettings(
        provider="custom",
        model="custom-model-beta",
        base_url="http://localhost:1234/v1/chat/completions",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={
                    "Custom": ["custom-model-alpha", "custom-model-beta"],
                    "Llama_cpp": ["None"],
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
        assert model_select.display is True
        assert model_select.value == expected_model
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == expected_model


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
        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        model_select = modal_screen.query_one("#console-settings-model-select", Select)
        assert {"gpt-4.1", "gpt-5"}.issubset(_select_values(model_select))

        model_select.value = "gpt-5"
        await pilot.pause()
        await pilot.click("#console-settings-save")
        await _wait_for_console_top_screen(host, console, pilot)
        await _visible_console_settings_button(console, pilot)
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
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.display is False
        assert model_select.disabled is True
        assert model_select.value == "gpt-4.1"
        assert model_input.display is True
        assert model_input.disabled is True
        assert model_input.value == "gpt-4.1"
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
async def test_console_left_rail_orders_session_before_staged_context() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        # Phase 1 rail restructure: the rail is four sections in order
        # Session (workspace context), Context (staged context), Model,
        # Details -- so workspace context renders above staged context.
        assert workspace_context.region.y < staged_context.region.y
        assert settings.parent.id == "console-run-inspector"
        assert workspace_context.parent.id == "console-rail-section-body-session"
        assert staged_context.parent.id == "console-rail-section-body-context"


@pytest.mark.asyncio
async def test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        left_rail = console.query_one("#console-left-rail")
        header = console.query_one(".console-rail-header")
        body = console.query_one("#console-left-rail-body")
        session_body = console.query_one("#console-rail-section-body-session")
        context_body = console.query_one("#console-rail-section-body-context")
        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        assert header.region.height == 1
        assert body.region.y >= header.region.y + header.region.height
        assert body.region.height <= left_rail.region.height - header.region.height
        assert settings.parent.id == "console-run-inspector"
        # Phase 1 nests each tray inside its own rail-section body, which is
        # itself a direct child of the scrolling rail body.
        assert workspace_context.parent is session_body
        assert staged_context.parent is context_body
        assert session_body.parent is body
        assert context_body.parent is body
        assert staged_context.region.width == workspace_context.region.width
        assert workspace_context.region.width <= body.region.width
        assert body.region.width - workspace_context.region.width <= 2


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

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(ConsoleSessionSettings(provider="openai", model="gpt-4.1"))
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

        summary_text = _summary_text(console)
        assert "Provider: openai" in summary_text
        assert "Model: gpt-4.1" in summary_text
        assert store.session_settings(second_id).provider == "openai"
        assert store.session_settings(first.id).provider == "llama_cpp"

        await _click_console_session_tab(console, store, pilot, first.id)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await _visible_console_settings_button(console, pilot)

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
        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
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
async def test_console_native_tab_click_switches_without_programmatic_fallback() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{first_id}")

        first_tab = console.query_one(f"#console-session-tab-{first_id}", Button)
        assert await pilot.click(first_tab, offset=(1, 0))
        for _ in range(10):
            if store.active_session_id == first_id:
                break
            await pilot.pause(0.05)

        assert store.active_session_id == first_id
        assert store.active_session_id != second_id


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_switches_native_tab() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
    }
    app.providers_models = {"llama_cpp": ["model-a"]}
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-1")

        first_conversation = console.query_one("#console-workspace-conversation-1", Button)
        assert getattr(first_conversation, "conversation_id", None) == f"native:{first_id}"
        assert await pilot.click(first_conversation, offset=(1, 0))
        for _ in range(10):
            if store.active_session_id == first_id:
                break
            await pilot.pause(0.05)

        assert store.active_session_id == first_id
        assert store.active_session_id != second_id


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
        await _visible_console_settings_button(console, pilot)
        original_summary = _summary_text(console)

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
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

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)

        assert modal_screen.query_one("#console-settings-save", Button).disabled is True


@pytest.mark.asyncio
async def test_console_settings_save_clears_stale_terminal_run_status() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
        "custom": {
            "api_url": "http://localhost:1234/v1/chat/completions",
            "model": "custom-model-beta",
        },
    }
    app.providers_models = {
        "llama_cpp": ["model-a"],
        "custom": ["custom-model-alpha", "custom-model-beta"],
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        await console._sync_native_console_chat_ui()

        controller = console._ensure_console_chat_controller()
        stale_copy = "Provider blocked: old llama.cpp failure."
        controller.run_state = ConsoleRunState.blocked(stale_copy)
        console._sync_console_mode_bar()
        assert stale_copy in str(console.query_one("#console-mode-bar", Static).renderable)

        settings_button = await _visible_console_settings_button(console, pilot)
        settings_button.press()
        modal_screen = await _wait_for_console_settings_modal(host, pilot)
        modal_screen.dismiss(
            ConsoleSessionSettings(
                provider="custom",
                model="custom-model-beta",
                base_url="http://localhost:1234/v1/chat/completions",
            )
        )
        await _wait_for_console_top_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert console._build_console_provider_selection().provider == "custom"
        assert controller.run_state.status is ConsoleRunStatus.IDLE
        assert stale_copy not in str(console.query_one("#console-mode-bar", Static).renderable)


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
        await _visible_console_settings_button(console, pilot)
        # The shared Workbench recovery banner stays hidden — the setup
        # card's action button carries this recovery instead (Phase 2 spec,
        # section 2).
        await _wait_for_selector(console, pilot, "#console-setup-modal-action")

        recovery_button = console.query_one("#console-setup-modal-action", Button)
        assert str(recovery_button.label) == "Choose model"
        assert recovery_button.display is True

        recovery_button.press()
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
        await _visible_console_settings_button(console, pilot)

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
    assert label == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
    assert target == "settings"
    assert tooltip == "Configure OpenAI API and API key in Settings"
    assert screen._console_provider_recovery_field() == "api_key"
    assert (
        screen._console_setup_blocked_reason()
        == "Add API key in Settings > Providers & Models before sending."
    )


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
    assert screen._console_provider_recovery_field() == "endpoint"
    assert (
        screen._console_setup_blocked_reason()
        == "Save provider endpoint in Settings > Providers & Models before sending."
    )


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
async def test_console_new_native_tab_inherits_active_session_settings_snapshot() -> None:
    app = _build_test_app()
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key", "model": "gpt-4.1"},
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "local-model",
        },
    }
    app.providers_models = {
        "openai": ["gpt-4.1"],
        "local_llamacpp": ["local-model"],
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_id = store.ensure_session().id
        active_settings = ConsoleSessionSettings(
            provider="local_llamacpp",
            model="local-model",
            base_url="http://127.0.0.1:9099",
            temperature=0.2,
            top_p=0.8,
            streaming=False,
        )
        store.replace_session_settings(first_id, active_settings)
        await console._sync_native_console_chat_ui()

        second_id = await _press_new_console_tab(console, store, pilot)
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        assert second_id != first_id
        assert store.session_settings(second_id) == active_settings


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


# --- task-177: readiness must follow Settings saves without an app restart ---


def _disk_loaded_snapshot(**overrides) -> dict:
    """Snapshot shaped like a real ``load_settings()`` boot config."""
    snapshot = {
        "general": {},
        "logging": {},
        "splash_screen": {},
        "api_settings": {"openai": {"api_key": ""}},
    }
    snapshot.update(overrides)
    return snapshot


def test_provider_readiness_config_refreshes_disk_loaded_snapshot(monkeypatch) -> None:
    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot()
    console = ChatScreen(app)
    fresh = _disk_loaded_snapshot(api_settings={"openai": {"api_key": "sk-fresh"}})
    monkeypatch.setattr(chat_screen_module, "load_settings", lambda: fresh)

    assert console._provider_readiness_app_config() is fresh


def test_provider_readiness_config_honors_injected_test_snapshot(monkeypatch) -> None:
    """Fakes without the disk-loaded marker sections stay authoritative."""
    app = _build_test_app()
    app.app_config = {"api_settings": {"openai": {"api_key": "injected"}}}
    console = ChatScreen(app)

    def _fail_load_settings():
        raise AssertionError("load_settings must not be consulted for injected snapshots")

    monkeypatch.setattr(chat_screen_module, "load_settings", _fail_load_settings)

    assert console._provider_readiness_app_config() is app.app_config


def test_provider_readiness_config_falls_back_when_load_settings_fails(monkeypatch) -> None:
    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot()
    console = ChatScreen(app)

    def _boom():
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(chat_screen_module, "load_settings", _boom)

    assert console._provider_readiness_app_config() is app.app_config


def test_console_readiness_unblocks_after_provider_save_without_restart(
    monkeypatch, tmp_path
) -> None:
    """Save a provider key via the config API after boot; readiness must see it.

    Mirrors the live UAT failure: Settings saved the key, the config module
    cache reloaded, but Console kept reading the boot-time ``app_config``
    snapshot until restart.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    config_path = tmp_path / "console-readiness-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _build_test_app()
        # Boot-time snapshot: disk-loaded shape, but captured before the save.
        app.app_config = _disk_loaded_snapshot()
        console = ChatScreen(app)
        settings = ConsoleSessionSettings(provider="openai", model="gpt-4o")

        readiness_before = build_console_settings_readiness(
            settings,
            app_config=console._provider_readiness_app_config(),
            environ={},
        )
        assert readiness_before.native_send_supported is False

        # The Settings screen save path: config API write + cache reload.
        assert config_module.save_setting_to_cli_config(
            "api_settings.openai", "api_key", "sk-saved-after-boot"
        )

        readiness_after = build_console_settings_readiness(
            settings,
            app_config=console._provider_readiness_app_config(),
            environ={},
        )
        assert readiness_after.native_send_supported is True
        assert readiness_after.label == "Ready"
        # The stale snapshot alone would still be blocked - proving the fresh
        # read (not the snapshot) unblocked readiness.
        readiness_stale = build_console_settings_readiness(
            settings,
            app_config=app.app_config,
            environ={},
        )
        assert readiness_stale.native_send_supported is False
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


# --- task-178: settings modal persistence affordance, boolean streaming, focus artifact ---


def _basic_modal(settings: ConsoleSessionSettings, app: "ModalHarness", **kwargs) -> ConsoleSettingsModal:
    return ConsoleSettingsModal(
        settings=settings,
        app_config=app.app_config,
        providers_models=kwargs.pop("providers_models", {"llama_cpp": ["model-a"]}),
        context_estimate=kwargs.pop(
            "context_estimate", ConsoleSettingsContextEstimate(10, 4096, "10 / 4k")
        ),
        can_save=kwargs.pop("can_save", True),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_console_settings_modal_streaming_is_boolean_toggle() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a", streaming=False)

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(_basic_modal(settings, app), callback=app.capture_saved_settings)
        await pilot.pause()
        toggle = app.screen.query_one("#console-settings-streaming", Button)
        assert str(toggle.label) == "Off"

        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "On"

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.streaming is True


@pytest.mark.asyncio
async def test_console_settings_modal_enumerated_inputs_list_accepted_values() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(_basic_modal(settings, app), callback=app.capture_saved_settings)
        await pilot.pause()
        placeholders = {
            "console-settings-reasoning-effort": "none, minimal, low, medium, high, xhigh",
            "console-settings-reasoning-summary": "auto, concise, detailed, none",
            "console-settings-verbosity": "low, medium, high",
            "console-settings-thinking-effort": "off, low, medium, high, xhigh, max",
        }
        for input_id, expected in placeholders.items():
            assert app.screen.query_one(f"#{input_id}", Input).placeholder == expected


@pytest.mark.asyncio
async def test_console_settings_modal_scope_line_names_session_and_default_scopes() -> None:
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        CONSOLE_SETTINGS_SCOPE_COPY,
    )

    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(_basic_modal(settings, app), callback=app.capture_saved_settings)
        await pilot.pause()
        scope = app.screen.query_one("#console-settings-scope", Static)
        assert str(scope.renderable) == CONSOLE_SETTINGS_SCOPE_COPY
        assert "session" in CONSOLE_SETTINGS_SCOPE_COPY.lower()
        assert "default" in CONSOLE_SETTINGS_SCOPE_COPY.lower()


@pytest.mark.asyncio
async def test_console_settings_modal_save_as_default_writes_through_config(monkeypatch) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module

    captured: list[dict] = []

    def fake_save(sections):
        captured.append(sections)
        return True

    monkeypatch.setattr(modal_module, "save_settings_to_cli_config", fake_save)
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        temperature=0.6,
        streaming=False,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(_basic_modal(settings, app), callback=app.capture_saved_settings)
        await pilot.pause()
        await pilot.click("#console-settings-save-default")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "model-a"
    assert len(captured) == 1
    sections = captured[0]
    provider_section = sections["api_settings.llama_cpp"]
    assert provider_section["model"] == "model-a"
    # llama_cpp already persists its endpoint under api_url in ModalHarness config.
    assert provider_section["api_url"] == "http://127.0.0.1:9099"
    assert provider_section["temperature"] == 0.6
    # Streaming persists on the canonical chat_defaults key (bridged legacy key),
    # and the provider itself becomes the default (PR #606 review finding:
    # chat_defaults.provider is the ONLY source of the default provider).
    assert sections["chat_defaults"] == {"streaming": False, "provider": "llama_cpp"}
    # Never persist None-valued optionals.
    assert "min_p" not in provider_section
    assert "seed" not in provider_section


@pytest.mark.asyncio
async def test_console_settings_modal_save_as_default_failure_keeps_modal_open(monkeypatch) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        CONSOLE_SETTINGS_SAVE_DEFAULT_FAILED_COPY,
    )

    monkeypatch.setattr(modal_module, "save_settings_to_cli_config", lambda sections: False)
    app = ModalHarness()
    app.saved_settings = ConsoleSessionSettings(provider="openai", model="sentinel")
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(_basic_modal(settings, app), callback=app.capture_saved_settings)
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()
        error = app.screen.query_one("#console-settings-error", Static)
        assert str(error.renderable) == CONSOLE_SETTINGS_SAVE_DEFAULT_FAILED_COPY
        # Modal stays open (dismiss would pop it and fire the callback).
        assert isinstance(app.screen, ConsoleSettingsModal)
        await pilot.click("#console-settings-cancel")

    assert app.saved_settings is None


@pytest.mark.asyncio
async def test_console_settings_modal_body_scroll_container_is_not_focusable() -> None:
    """The focused scroll body painted stray focus-outline fragments ("|")
    through the section margins with the real app CSS; keeping it out of the
    focus chain removes the artifact and lands first focus on a real control."""
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(_basic_modal(settings, app), callback=app.capture_saved_settings)
        await pilot.pause()
        body = app.screen.query_one("#console-settings-body", ScrollableContainer)
        assert body.can_focus is False
        assert app.focused is not body


# --- task-177 live regression: REAL journey (boot -> Settings save -> Console) ---


def _build_live_config_test_app():
    """Real TldwCli booted against the REAL (test-sandboxed) config file.

    Unlike ``_build_test_app`` this does NOT stub ``load_settings`` /
    ``get_cli_setting``: ``app.app_config`` is the genuine template config from
    the sandbox ``TLDW_CONFIG_PATH``, so the disk-loaded snapshot path (and the
    stale-snapshot bug it guards against) is exercised end to end.
    """
    import tempfile
    from contextlib import ExitStack
    from unittest.mock import MagicMock, patch

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    user_data_dir = Path(tempfile.mkdtemp(prefix="tldw-chatbook-live-config-test-"))

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app.current_runtime_source = "local"
        app.current_runtime_backend = "local"
        return context

    with ExitStack() as stack:
        stack.enter_context(
            patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None)
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.ServerNotesWorkspaceService.from_config",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.ServerCharacterPersonaService.from_config",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_notes_service",
                lambda self, _user: setattr(self, "notes_service", None),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_prompts_service",
                lambda self: setattr(self, "prompts_service_initialized", False),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_providers_models",
                lambda self: setattr(self, "providers_models", {}),
            )
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_media_db",
                lambda self: (
                    setattr(self, "media_db", None),
                    setattr(self, "_media_types_for_ui", ["All Media"]),
                ),
            )
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.load_runtime_policy_for_app",
                side_effect=fake_runtime_policy,
            )
        )
        for db_path_getter in (
            "get_notifications_db_path",
            "get_subscriptions_db_path",
            "get_research_db_path",
            "get_writing_db_path",
        ):
            stack.enter_context(
                patch(f"tldw_chatbook.app.{db_path_getter}", return_value=":memory:")
            )
        stack.enter_context(
            patch("tldw_chatbook.app.get_user_data_dir", return_value=user_data_dir)
        )
        stack.enter_context(
            patch(
                "tldw_chatbook.app.get_workspaces_db_path",
                return_value=user_data_dir / "workspaces.sqlite",
            )
        )
        return TldwCli()


async def _wait_for_screen(app, pilot, screen_type_name: str, *, attempts: int = 250):
    for _ in range(attempts):
        if type(app.screen).__name__ == screen_type_name:
            return app.screen
        await pilot.pause(0.02)
    raise AssertionError(
        f"Never reached {screen_type_name}; current screen: {type(app.screen).__name__}"
    )


@pytest.mark.asyncio
async def test_real_journey_settings_save_unblocks_console_without_restart(
    monkeypatch,
) -> None:
    """Live-UAT regression: boot -> blocked Console -> Settings save -> Console.

    Mirrors the exact live failure: the Settings adapter saves
    chat_defaults.provider/model + the llama.cpp endpoint (config caches reload),
    the user clicks the Console nav tab (fresh ChatScreen composes, prior screen
    state restores), and the setup card must NOT still be blocking.
    """
    from tldw_chatbook import config as config_module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.Widgets.Console import ConsoleSetupModal

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    # Prime the sandbox template config and keep the boot fast/deterministic.
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    assert config_module.save_setting_to_cli_config("splash_screen", "enabled", False)
    config_module.load_settings(force_reload=True)

    app = _build_live_config_test_app()
    # Sanity: the boot snapshot must look disk-loaded (markers present) so the
    # fresh-config branch is the one under test.
    assert ChatScreen._console_config_snapshot_is_disk_loaded(app.app_config)

    async with app.run_test(size=(180, 50)) as pilot:
        # 1) First-run landing: Console blocked on the template OpenAI default.
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        assert console._build_console_setup_card_state().mode == "card"

        # 2) Leave Console (screen state, including session settings, is saved).
        app.post_message(NavigateToScreen("home"))
        await _wait_for_screen(app, pilot, "HomeScreen")

        # 3) The real Settings save path (same three values as the live run).
        adapter = SettingsConfigAdapter()
        assert adapter.save_values(
            "chat_defaults",
            {"provider": "llama_cpp", "model": "Qwen3-Coder-Test.gguf"},
        )
        assert adapter.save_values(
            "api_settings.llama_cpp",
            {"api_url": "http://127.0.0.1:9099"},
        )

        # 4) Back to Console: a fresh ChatScreen composes and restores state.
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-setup-modal")

        card_state = console._build_console_setup_card_state()
        assert card_state.mode != "card", (
            "Setup card still blocking after a provider save; "
            f"steps={[(step.state, step.label) for step in card_state.steps]}"
        )
        settings, readiness = console._active_console_settings_readiness()
        assert settings.provider == "llama_cpp"
        assert readiness.native_send_supported is True

        # The blocking modal must clear once guidance syncs.
        for _ in range(100):
            modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
            if not modal.is_blocking:
                break
            await pilot.pause(0.02)
        assert not console.query_one("#console-setup-modal", ConsoleSetupModal).is_blocking


def test_console_resolution_view_suppresses_boot_echo_reactives(monkeypatch) -> None:
    """Post-save, reactives echoing the boot template defaults must not win."""
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    app = _build_test_app()
    app.app_config = _disk_loaded_snapshot(
        chat_defaults={"provider": "OpenAI", "model": "gpt-4o"}
    )
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4o"
    console = ChatScreen(app)
    fresh = _disk_loaded_snapshot(
        chat_defaults={"provider": "llama_cpp", "model": "Qwen3-Test.gguf"},
        api_settings={"llama_cpp": {"api_url": "http://127.0.0.1:9099"}},
    )
    monkeypatch.setattr(chat_screen_module, "load_settings", lambda: fresh)

    provider, model = console._effective_console_provider_model()
    assert provider_config_key(str(provider)) == "llama_cpp"
    assert str(model) == "Qwen3-Test.gguf"

    # A reactive value the user actually changed (differs from the boot echo)
    # still wins over fresh chat_defaults.
    app.chat_api_provider_value = "Anthropic"
    provider_after_user_pick, _model = console._effective_console_provider_model()
    assert provider_config_key(str(provider_after_user_pick)) == "anthropic"


def test_console_stale_default_refresh_respects_user_marked_settings() -> None:
    """Blocked derived defaults refresh; explicit user selections never do."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": ""},
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    session = store.ensure_session()

    user_choice = ConsoleSessionSettings(provider="openai", model="gpt-4o", source="user")
    store.replace_session_settings(session.id, user_choice)
    assert console._ensure_active_console_session_settings() == user_choice

    stale_derived = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    store.replace_session_settings(session.id, stale_derived)
    refreshed = console._ensure_active_console_session_settings()
    assert refreshed.provider == "llama_cpp"
    assert refreshed.source == "derived"


# --- task-188/191: provider display names + Discover models -----------------


def _select_labels(select: Select) -> set[str]:
    options = getattr(select, "options", None)
    if options is None:
        options = getattr(select, "_options", [])
    labels: set[str] = set()
    for option in options:
        prompt = getattr(option, "prompt", None)
        if prompt is None and isinstance(option, tuple) and option:
            prompt = option[0]
        if prompt is not None:
            labels.add(str(getattr(prompt, "plain", prompt)))
    return labels


@pytest.mark.asyncio
async def test_console_settings_modal_provider_labels_use_catalog_display_names() -> None:
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
            )
        )
        await pilot.pause()

        provider_select = app.screen.query_one("#console-settings-provider", Select)
        labels = _select_labels(provider_select)
        values = _select_values(provider_select)

    # Labels render shared-catalog display names; values stay raw config keys.
    assert "llama.cpp" in labels
    assert "OpenAI" in labels
    assert "Ollama" in labels
    assert "llama_cpp" not in labels
    assert {"llama_cpp", "openai", "ollama"}.issubset(values)


class _RecordingProber:
    def __init__(self, result: LocalModelProbeResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, base_url: str, provider_key: str) -> LocalModelProbeResult:
        self.calls.append((base_url, provider_key))
        return self.result


async def _wait_for_discover_status(app, pilot, fragment: str) -> Static:
    status = app.screen.query_one(f"#{MODEL_DISCOVER_STATUS_ID}", Static)
    for _ in range(60):
        if fragment in str(status.renderable):
            return status
        await pilot.pause(0.05)
    raise AssertionError(
        f"discover status never showed {fragment!r}; last: {str(status.renderable)!r}"
    )


@pytest.mark.asyncio
async def test_console_settings_modal_discover_models_success_swaps_input_for_select() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model=None)
    prober = _RecordingProber(
        LocalModelProbeResult(
            ok=True,
            base_url="http://127.0.0.1:9099",
            model_ids=("srv-a", "srv-b"),
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()

        app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).press()
        await _wait_for_discover_status(app, pilot, "Found 2 models at http://127.0.0.1:9099.")

        assert prober.calls == [("http://127.0.0.1:9099", "llama_cpp")]
        model_select = app.screen.query_one("#console-settings-model-select", Select)
        assert model_select.display is True
        assert model_select.disabled is False
        assert _select_values(model_select) == {"srv-a", "srv-b"}
        assert model_select.value == "srv-a"
        # Free-text fallback stays available after discovery.
        model_custom = app.screen.query_one("#console-settings-model-custom", Button)
        assert model_custom.display is True
        assert model_custom.disabled is False

        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.model == "srv-a"


@pytest.mark.asyncio
async def test_console_settings_modal_discover_models_failure_shows_inline_copy() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model=None)
    prober = _RecordingProber(
        LocalModelProbeResult(
            ok=False,
            base_url="http://127.0.0.1:9099",
            detail="No models endpoint at http://127.0.0.1:9099.",
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": []},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
                model_prober=prober,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        discover.press()
        await _wait_for_discover_status(
            app, pilot, "No models endpoint at http://127.0.0.1:9099."
        )

        # Honest inline line, button usable again, manual entry still works.
        assert discover.disabled is False
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_input.display is True
        assert model_input.disabled is False


@pytest.mark.asyncio
async def test_console_settings_modal_discover_button_only_for_url_based_providers() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4.1"], "llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        discover = app.screen.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button)
        assert discover.display is False
        assert discover.disabled is True

        app.screen.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert discover.display is True
        assert discover.disabled is False
