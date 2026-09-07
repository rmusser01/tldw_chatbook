"""Settings ▸ Agents: category registration + panel CRUD (fleet spec §4)."""

import pytest
from textual.app import App
from textual.widgets import ListView

from tldw_chatbook.Agents.agent_models import AgentDefinition
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
async def test_panel_selection_round_trip_updates_in_place(runs_db):
    # Review finding (task-6 fix round 1): the ListView selection round
    # trip (select -> form populates -> Save updates in place) had zero
    # coverage, and _reload_list() used to fire ListView.clear()/append()
    # without awaiting the AwaitRemove/AwaitMount they return -- a freshly
    # appended row could sit at Region(0,0,0,0) for a tick, so a
    # select/click right after a reload could miss it. This pins both the
    # round trip AND (implicitly, by working at all with a single
    # `pilot.pause()` settle) the await fix.
    seeded_id = runs_db.create_agent_definition(
        AgentDefinition(
            name="researcher",
            description="Searches sources.",
            instructions="Cite sources.",
        )
    )
    panel = AgentsSettingsPanel(app_instance=None, runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        list_view = panel.query_one("#agents-definition-list", ListView)
        list_view.focus()
        list_view.index = 0
        list_view.action_select_cursor()
        await pilot.pause()

        assert panel.query_one("#agents-name-input").value == "researcher"
        assert (
            panel.query_one("#agents-description-input").value
            == "Searches sources."
        )
        assert panel.query_one("#agents-instructions-area").text == "Cite sources."

        panel.query_one(
            "#agents-description-input"
        ).value = "Now cites primary sources."
        await pilot.click("#agents-save-button")
        await pilot.pause()

    rows = runs_db.list_agent_definitions()
    assert len(rows) == 1
    assert rows[0]["id"] == seeded_id
    assert rows[0]["name"] == "researcher"
    assert rows[0]["description"] == "Now cites primary sources."


@pytest.mark.asyncio
async def test_panel_inputs_carry_the_compact_class_that_makes_them_paint(runs_db):
    # Live-verification finding (task-13154.1, 2026-08-09): the Name/
    # Description/Model override/Tools Input widgets were missing
    # classes="settings-compact-input" -- the class every other Settings
    # Input in this screen carries so it can live inside
    # .settings-input-row's height:1. Without it, Textual's default 3-row
    # bordered Input chrome ate the panel's single available row and NEVER
    # painted placeholder or value text on screen, though .value/Save/DB
    # were always correct -- which is exactly why no earlier test here (all
    # of which set .value directly) could catch it. This guard asserts the
    # class directly rather than the paint itself, since the harness above
    # does not load the real CSS bundle (`.settings-input-row`'s height:1
    # rule lives there, not in DEFAULT_CSS) and so cannot reproduce the
    # collapse either way -- see lessons-live-verification.md for the two
    # entries this incident produced.
    panel = AgentsSettingsPanel(app_instance=None, runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for widget_id in (
            "#agents-name-input",
            "#agents-description-input",
            "#agents-model-input",
            "#agents-tools-input",
        ):
            widget = panel.query_one(widget_id)
            assert widget.has_class("settings-compact-input"), (
                f"{widget_id} is missing settings-compact-input -- it will "
                "not paint its placeholder or value inside "
                ".settings-input-row's height:1"
            )


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
