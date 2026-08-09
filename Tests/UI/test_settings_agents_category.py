"""Settings ▸ Agents: category registration + panel CRUD (fleet spec §4)."""

import pytest
from textual.app import App

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Widgets.settings_agents_panel import AgentsSettingsPanel

from Tests.UI.test_destination_shells import _static_text


@pytest.fixture()
def runs_db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


class PanelHarness(App):
    def __init__(self, panel):
        super().__init__()
        self._panel = panel

    def compose(self):
        yield self._panel


@pytest.mark.asyncio
async def test_panel_creates_definition_via_form(runs_db):
    panel = AgentsSettingsPanel(app_instance=None, runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        panel.query_one("#agents-name-input").value = "researcher"
        panel.query_one("#agents-description-input").value = "Searches sources."
        panel.query_one("#agents-instructions-area").text = "Cite sources."
        await pilot.click("#agents-save-button")
        await pilot.pause()
    rows = runs_db.list_agent_definitions()
    assert [r["name"] for r in rows] == ["researcher"]


@pytest.mark.asyncio
async def test_panel_surfaces_validation_error(runs_db):
    panel = AgentsSettingsPanel(app_instance=None, runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        panel.query_one("#agents-name-input").value = "subagent"  # reserved
        panel.query_one("#agents-instructions-area").text = "x"
        await pilot.click("#agents-save-button")
        await pilot.pause()
        status = panel.query_one("#agents-status")
        # Rendered-geometry guard, not just DOM presence (Library-UAT
        # lesson: unbounded-width Statics are invisible to headless
        # queries while "present").
        assert status.region.width > 0
        assert "reserved" in _static_text(status)
    assert runs_db.list_agent_definitions() == []


@pytest.mark.asyncio
async def test_panel_without_db_shows_notice(tmp_path):
    panel = AgentsSettingsPanel(app_instance=None, runs_db=None)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        notice = panel.query_one("#agents-no-db-notice")
        assert notice.region.width > 0


@pytest.mark.asyncio
async def test_agents_category_renders_in_settings_screen():
    # The category sweep (test_settings_category_sweep.py) already visits
    # every category; this pins OUR panel specifically: selecting Agents
    # renders either the editor or the no-DB notice (test app runs with a
    # :memory: ChaChaNotes, so the notice is the expected branch).
    import Tests.UI.test_settings_category_sweep as sweep

    app = sweep._build_test_app()
    host = sweep.DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await sweep._settle_settings(pilot)
        await sweep._click_settings_category(pilot, "agents")
        screen = sweep._active_destination_screen(host)
        assert screen.query("#settings-agents-panel") or screen.query(
            "#agents-no-db-notice"
        )
