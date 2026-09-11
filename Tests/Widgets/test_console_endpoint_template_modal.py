"""Component tests for the Console "New endpoint from template" modal (ADR-146)."""

import tomllib
from pathlib import Path

import pytest
from textual.widgets import Button, Static

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.custom_endpoint_registry import load_custom_endpoints
from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
    ConsoleEndpointTemplateModal,
)


class _TemplateModalHarness(ConsolidatedCSSApp):
    """StyledModalHarness-equivalent minimal host for the template modal."""

    CSS = """
    Screen {
        layout: vertical;
    }
    """
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, app_config: dict) -> None:
        super().__init__()
        self.app_config = app_config
        self.created_provider_id: str | None = None

    def on_console_endpoint_template_modal_endpoint_created(
        self, event: ConsoleEndpointTemplateModal.EndpointCreated
    ) -> None:
        self.created_provider_id = event.provider_id


@pytest.mark.asyncio
async def test_template_modal_creates_entry_and_dismisses_with_id(
    tmp_path, monkeypatch
):
    # app_config fixtures follow Tests/UI/test_console_session_settings.py's
    # tmp-path config pattern (isolate_config style)
    config_path = tmp_path / "endpoint-template-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _TemplateModalHarness(app_config={})
        async with app.run_test(size=(100, 40)) as pilot:
            modal = ConsoleEndpointTemplateModal(
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                template_provider="llama_cpp",
            )
            await app.push_screen(modal)
            await pilot.click("#endpoint-template-name")
            await pilot.press(*"GPU box")
            await pilot.click("#endpoint-template-url")
            # pilot.press types nothing for multi-char strings (char-less Key
            # events), so unpack the URL into single-character keys.
            await pilot.press("ctrl+a", *"http://192.168.1.9:8080")
            await pilot.click("#endpoint-template-create")
            await pilot.pause()
        assert app.created_provider_id == "custom-ep:gpu-box"
        entry = load_custom_endpoints(app.app_config)["gpu-box"]
        assert entry.family == "llama_cpp"
        assert entry.base_url == "http://192.168.1.9:8080"
        # The [custom_endpoints.<slug>] section round-trips through the real
        # (unmonkeypatched) atomic writer into the temp config file itself.
        raw = tomllib.loads(config_path.read_text())
        assert raw["custom_endpoints"]["gpu-box"] == {
            "display_name": "GPU box",
            "family": "llama_cpp",
            "base_url": "http://192.168.1.9:8080",
            "models": ["model-a"],
            "created_from": "llama_cpp",
        }
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


@pytest.mark.asyncio
async def test_template_modal_shows_validation_inline(tmp_path):
    # app_config fixtures follow Tests/UI/test_console_session_settings.py's
    # tmp-path config pattern (isolate_config style)
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 40)) as pilot:
        modal = ConsoleEndpointTemplateModal(
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            template_provider="llama_cpp",
        )
        await app.push_screen(modal)
        await pilot.click("#endpoint-template-name")
        await pilot.press(*"GPU box")
        await pilot.click("#endpoint-template-url")
        # Unpacked for the same multi-char press limitation as above.
        await pilot.press("ctrl+a", *"ftp://192.168.1.9:8080")
        error = app.screen.query_one("#endpoint-template-error", Static)
        assert "http(s)" in error.renderable
        create = app.screen.query_one("#endpoint-template-create", Button)
        assert create.disabled is True
    assert app.created_provider_id is None
    assert load_custom_endpoints(app.app_config) == {}
