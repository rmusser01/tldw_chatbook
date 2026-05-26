import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_console_settings_summary_state,
)
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


def _visible_text(app: App[None]) -> str:
    return " ".join(str(widget.renderable) for widget in app.query(Static))


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
