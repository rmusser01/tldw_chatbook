"""Params table editor on the Console endpoint template modal (ADR-147,
TASK-32477 task 9).

Harness and config-isolation patterns mirror
Tests/Widgets/test_console_endpoint_template_modal.py; the harness class is
imported from there rather than duplicated.
"""

import tomllib

import pytest
from textual.widgets import Button, Static, TextArea

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.custom_endpoint_registry import load_custom_endpoints
from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
    ConsoleEndpointTemplateModal,
)

from Tests.Widgets.test_console_endpoint_template_modal import _TemplateModalHarness


async def _push_modal_and_fill_form(app) -> None:
    modal = ConsoleEndpointTemplateModal(
        app_config=app.app_config,
        providers_models={"llama_cpp": ["model-a"]},
        template_provider="llama_cpp",
    )
    await app.push_screen(modal)


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    config_path = tmp_path / "endpoint-params-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        yield config_path
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


async def _fill_name_and_url(pilot) -> None:
    await pilot.click("#endpoint-template-name")
    await pilot.press(*"GPU box")
    await pilot.click("#endpoint-template-url")
    # pilot.press types nothing for multi-char strings (char-less Key
    # events), so unpack the URL into single-character keys.
    await pilot.press("ctrl+a", *"http://192.168.1.9:8080")


@pytest.mark.asyncio
async def test_template_modal_persists_params_table(isolated_config):
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 50)) as pilot:
        await _push_modal_and_fill_form(app)
        await _fill_name_and_url(pilot)
        app.screen.query_one("#endpoint-template-params", TextArea).text = (
            "temperature = 0.2\ntop_k = 40"
        )
        await pilot.pause()
        await pilot.click("#endpoint-template-create")
        await pilot.pause()
    assert app.created_provider_id == "custom-ep:gpu-box"
    entry = load_custom_endpoints(app.app_config)["gpu-box"]
    assert entry.params == (("temperature", 0.2), ("top_k", 40))
    raw = tomllib.loads(isolated_config.read_text())
    assert raw["custom_endpoints"]["gpu-box"]["params"] == {
        "temperature": 0.2,
        "top_k": 40,
    }


@pytest.mark.asyncio
async def test_template_modal_blocks_unknown_param_key():
    # No TLDW_CONFIG_PATH override needed: Create must stay gated, so no
    # config write can happen (same shape as the existing inline-validation
    # test).
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 50)) as pilot:
        await _push_modal_and_fill_form(app)
        await _fill_name_and_url(pilot)
        app.screen.query_one(
            "#endpoint-template-params", TextArea
        ).text = "temprature = 0.2"
        await pilot.pause()
        error = app.screen.query_one("#endpoint-template-error", Static)
        assert "temprature" in error.renderable
        create = app.screen.query_one("#endpoint-template-create", Button)
        assert create.disabled is True
    assert app.created_provider_id is None
    assert load_custom_endpoints(app.app_config) == {}


@pytest.mark.asyncio
async def test_template_modal_empty_params_omit_table(isolated_config):
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 50)) as pilot:
        await _push_modal_and_fill_form(app)
        await _fill_name_and_url(pilot)
        await pilot.click("#endpoint-template-create")
        await pilot.pause()
    assert app.created_provider_id == "custom-ep:gpu-box"
    entry = load_custom_endpoints(app.app_config)["gpu-box"]
    assert entry.params == ()
    raw = tomllib.loads(isolated_config.read_text())
    assert "params" not in raw["custom_endpoints"]["gpu-box"]
