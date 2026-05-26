import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_console_settings_summary_state,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal
from tldw_chatbook.Widgets.Console.console_settings_summary import ConsoleSettingsSummary


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


def _visible_text(app: App[None]) -> str:
    return " ".join(str(widget.renderable) for widget in app.screen.query(Static))


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


@pytest.mark.asyncio
async def test_console_settings_summary_renders_four_rows_and_button() -> None:
    state = ConsoleSettingsSummaryState(
        model_row="Model: llama.cpp / model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Persona: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Console Settings" in text
        assert "Model: llama.cpp / model-a" in text
        assert "Context: 12 / 4k" in text
        assert "Sampling: T 0.70, P 0.95" in text
        assert "Persona: General" in text
        assert app.query_one("#console-settings-open", Button).tooltip == "Open Console settings"


def test_summary_state_appends_non_ready_readiness_to_model_row() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(used_tokens=12, token_limit=4096, label="12 / 4k"),
        ConsoleSettingsReadiness(label="WIP", detail="Provider not wired yet.", native_send_supported=False),
    )

    assert state.model_row == "Model: llama_cpp / model-a (WIP)"
    assert state.readiness_label == "WIP"


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


def test_summary_state_preserves_prefixed_context_label() -> None:
    state = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        ConsoleSettingsContextEstimate(used_tokens=None, token_limit=None, label="Context: unknown"),
        ConsoleSettingsReadiness(label="Ready", detail="Ready.", native_send_supported=True),
    )

    assert state.context_row == "Context: unknown"


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


@pytest.mark.asyncio
async def test_console_settings_modal_cancel_discards_draft() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

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
            )
        )
        await pilot.pause()
        await pilot.click("#console-settings-cancel")

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
        app.screen.query_one("#console-settings-temperature", Input).value = "0.42"
        app.screen.query_one("#console-settings-top-p", Input).value = "0.88"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"
    assert app.saved_settings.temperature == 0.42
    assert app.saved_settings.top_p == 0.88


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
async def test_console_settings_modal_uses_model_input_without_configured_models() -> None:
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
        assert model_select.disabled is True
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_input.value == "freeform-model"


@pytest.mark.asyncio
async def test_console_settings_modal_provider_change_to_no_models_switches_to_input() -> None:
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
            )
        )
        await pilot.pause()
        app.screen.query_one("#console-settings-provider", Select).value = "custom"
        await pilot.pause()

        model_select = app.screen.query_one("#console-settings-model-select", Select)
        model_input = app.screen.query_one("#console-settings-model-input", Input)
        assert model_select.disabled is True
        assert "model-a" not in _select_values(model_select)
        assert model_input.display is True
        assert model_input.disabled is False
        assert model_input.value == ""
