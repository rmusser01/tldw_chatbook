"""Settings ▸ Agents: preset routing, sub-agent defaults, override policy,
and the "Test routing" dry-run (ADR-147, TASK-32477 task 9).

Harness conventions follow Tests/UI/test_settings_agents_category.py
(PanelHarness + runs_db fixture); config isolation follows
Tests/Widgets/test_console_endpoint_template_modal.py (TLDW_CONFIG_PATH at a
tmp path + force reloads), plus the TLDW_AGENTS_* env tier pinned off so the
loader's env-first precedence (Agents/run_log.py ``_setting``) cannot leak
the developer machine's environment into the assertions.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static, TextArea

from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_destination_shells import _static_text
from tldw_chatbook import config as config_module
from tldw_chatbook.Agents.agent_models import AgentDefinition, definition_from_row
from tldw_chatbook.Agents.agent_routing import load_agents_routing_config
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Widgets.settings_agents_panel import AgentsSettingsPanel

pytestmark = pytest.mark.bootstrap_profile


#: A registry entry (llama.cpp family is keyless) plus a configured built-in
#: model, shaped like Tests/Agents/test_agent_routing.py's APP_CFG.
APP_CONFIG = {
    "custom_endpoints": {
        "qwen-local": {
            "display_name": "Qwen Local",
            "family": "llama_cpp",
            "base_url": "http://127.0.0.1:8080",
        }
    },
    "api_settings": {"llama_cpp": {"model": "llama-3-8b"}},
}

_AGENTS_ENV_KEYS = (
    "TLDW_AGENTS_SUBAGENT_DEFAULT_PROVIDER",
    "TLDW_AGENTS_SUBAGENT_DEFAULT_MODEL",
    "TLDW_AGENTS_SPAWN_OVERRIDE_ENABLED",
    "TLDW_AGENTS_SPAWN_OVERRIDE_ALLOWLIST",
)

#: Readiness fake: every provider resolves as ready (the resolver's own
#: ``readiness=`` seam, mirrored from Tests/Agents/test_agent_routing.py).
READY = lambda _cfg, _provider: None  # noqa: E731


@pytest.fixture()
def runs_db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    """Route the atomic config writer/loader at a tmp file; pin env tier off."""
    for env_key in _AGENTS_ENV_KEYS:
        monkeypatch.delenv(env_key, raising=False)
    # Keep the source-bound profile selected before config import.
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    original = config_path.read_bytes()
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        yield config_path
    finally:
        config_path.write_bytes(original)
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


class PanelHarness(App):
    """Use the actual scrollable Settings form and control geometry."""

    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, panel):
        super().__init__()
        self._panel = panel

    def compose(self):
        yield self._panel


def _make_panel(runs_db, app_config=None, **kwargs) -> AgentsSettingsPanel:
    return AgentsSettingsPanel(
        app_instance=SimpleNamespace(
            app_config=APP_CONFIG if app_config is None else app_config
        ),
        runs_db=runs_db,
        **kwargs,
    )


def _fill_valid_preset_form(panel) -> None:
    panel.query_one("#agents-name-input", Input).value = "researcher"
    panel.query_one("#agents-instructions-area", TextArea).text = "Cite sources."


async def _click_save(panel, pilot) -> None:
    panel.query_one("#agents-save-button", Button).scroll_visible(animate=False)
    await pilot.pause()
    assert await pilot.click("#agents-save-button")


@pytest.mark.asyncio
async def test_panel_saves_preset_routing(runs_db, isolated_config):
    panel = _make_panel(runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _fill_valid_preset_form(panel)
        panel.query_one(
            "#agents-provider-select", Select
        ).value = "custom-ep:qwen-local"
        panel.query_one("#agents-model-input", Input).value = "qwen3.8-27b"
        panel.query_one(
            "#agents-params-area", TextArea
        ).text = "temperature = 0.2"
        panel.query_one("#agents-fallback-models-area", TextArea).text = (
            "llama_cpp/backup\ncustom-ep:qwen-local/alternate"
        )
        await _click_save(panel, pilot)
        await pilot.pause()
    rows = runs_db.list_agent_definitions()
    assert len(rows) == 1
    loaded = definition_from_row(rows[0])
    assert loaded.provider == "custom-ep:qwen-local"
    assert loaded.model == "qwen3.8-27b"
    assert loaded.params == (("temperature", 0.2),)
    assert loaded.fallback_models == (
        ("llama_cpp", "backup"),
        ("custom-ep:qwen-local", "alternate"),
    )


@pytest.mark.asyncio
async def test_panel_rejects_bad_param_key_on_save(runs_db, isolated_config):
    panel = _make_panel(runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _fill_valid_preset_form(panel)
        panel.query_one(
            "#agents-params-area", TextArea
        ).text = "temprature = 0.2"
        await _click_save(panel, pilot)
        await pilot.pause()
        status = panel.query_one("#agents-status", Static)
        assert "temprature" in _static_text(status)
    # Save is gated: neither the preset row nor the [agents] keys persist
    # (the ensured config file may carry an [agents] section from defaults,
    # so assert on the routing keys themselves, not the section).
    assert runs_db.list_agent_definitions() == []
    raw = (
        tomllib.loads(isolated_config.read_text())
        if isolated_config.exists()
        else {}
    )
    assert "subagent_default_provider" not in raw.get("agents", {})


@pytest.mark.asyncio
async def test_panel_flags_stale_allowlist_slug(runs_db, isolated_config):
    isolated_config.write_text(
        '[agents]\nspawn_override_allowlist = ["custom-ep:deleted-slug"]\n'
    )
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    panel = _make_panel(runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # The stale entry loads into the editor for the user to see.
        assert "custom-ep:deleted-slug" in panel.query_one(
            "#agents-override-allowlist-area", TextArea
        ).text
        _fill_valid_preset_form(panel)
        panel.query_one(
            "#agents-default-model-input", Input
        ).value = "qwen3.8-27b"
        await _click_save(panel, pilot)
        await pilot.pause()
        warning = panel.query_one("#agents-allowlist-warning", Static)
        assert "custom-ep:deleted-slug" in _static_text(warning)
    raw = tomllib.loads(isolated_config.read_text())
    # The stale allowlist value is left UNTOUCHED — never silently rewritten —
    # while the unaffected keys still save.
    assert raw["agents"]["spawn_override_allowlist"] == ["custom-ep:deleted-slug"]
    assert raw["agents"]["subagent_default_model"] == "qwen3.8-27b"


@pytest.mark.asyncio
async def test_test_routing_reports_readiness(runs_db, isolated_config):
    runs_db.create_agent_definition(
        AgentDefinition(
            name="alpha",
            instructions="A.",
            provider="custom-ep:qwen-local",
            model="qwen3.8-27b",
        )
    )
    runs_db.create_agent_definition(
        AgentDefinition(
            name="beta",
            instructions="B.",
            provider="custom-ep:deleted-slug",
        )
    )
    panel = _make_panel(runs_db, routing_readiness=READY)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # The routing block sits at the bottom of the form's VerticalScroll;
        # pilot.click refuses out-of-view targets, so scroll it in first.
        # animate=False: the default animated scroll races the click — the
        # pilot can compute target coordinates mid-animation and miss.
        panel.query_one("#agents-test-routing-button", Button).scroll_visible(
            animate=False
        )
        await pilot.pause()
        await pilot.click("#agents-test-routing-button")
        await pilot.pause()
        report = _static_text(panel.query_one("#agents-routing-report", Static))
    lines = report.splitlines()
    alpha_line = next(line for line in lines if line.startswith("alpha"))
    assert alpha_line == "alpha -> custom-ep:qwen-local / qwen3.8-27b — ready"
    beta_line = next(line for line in lines if line.startswith("beta"))
    assert "[unknown_endpoint_slug] (preset)" in beta_line
    # The configured default is always reported too (nothing configured here).
    assert any(line.startswith("(default)") for line in lines)


@pytest.mark.asyncio
async def test_allowlist_model_glob_entries_round_trip(runs_db, isolated_config):
    panel = _make_panel(runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _fill_valid_preset_form(panel)
        panel.query_one("#agents-override-allowlist-area", TextArea).text = (
            "llama_cpp/qwen3.8-*\ncustom-ep:qwen-local/qwen3.*"
        )
        await _click_save(panel, pilot)
        await pilot.pause()
    raw = tomllib.loads(isolated_config.read_text())
    assert raw["agents"]["spawn_override_allowlist"] == [
        "llama_cpp/qwen3.8-*",
        "custom-ep:qwen-local/qwen3.*",
    ]
    # And the loader the spawn path actually uses reads them back unchanged.
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    routing = load_agents_routing_config()
    assert routing.spawn_override_allowlist == (
        "llama_cpp/qwen3.8-*",
        "custom-ep:qwen-local/qwen3.*",
    )


@pytest.mark.parametrize("level", ["override", "preset", "default", "inherit"])
def test_routing_report_names_the_failing_level(monkeypatch, level):
    from tldw_chatbook.Agents.agent_routing import AgentsRoutingConfig, RoutingError
    from tldw_chatbook.Widgets import settings_agents_panel

    def refuse(*args, **kwargs):
        raise RoutingError("provider_not_ready", "provider is unavailable", level=level)

    monkeypatch.setattr(settings_agents_panel, "resolve_spawn_target", refuse)
    report = AgentsSettingsPanel._routing_report_line(
        SimpleNamespace(_routing_readiness=READY),
        "child", APP_CONFIG, AgentsRoutingConfig(), preset=None,
    )
    assert f"({level})" in report
    if level != "inherit":
        assert "[provider_not_ready]" in report
