"""Saved review teams use stable preset IDs without editing destinations."""

import tomllib
from types import SimpleNamespace

import pytest
from textual.widgets import Input, ListView, Select, Static

from Tests.UI import test_settings_agents_category as category_tests
from Tests.UI.test_destination_shells import _static_text
from Tests.Widgets import test_settings_agents_panel_routing as routing_tests
from Tests.Widgets.test_settings_agents_panel_routing import PanelHarness
from tldw_chatbook import config as config_module
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Widgets.settings_agents_panel import AgentsSettingsPanel

runs_db = category_tests.runs_db
isolated_config = routing_tests.isolated_config

pytestmark = pytest.mark.bootstrap_profile


def preset(db, name, *, enabled=True):
    return db.create_agent_definition(
        AgentDefinition(
            name=name,
            instructions="Review.",
            enabled=enabled,
            provider="llama_cpp",
            model="local-test",
        )
    )


def panel_for(db, path):
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    return AgentsSettingsPanel(
        SimpleNamespace(app_config=tomllib.loads(path.read_text())), runs_db=db
    )


@pytest.mark.asyncio
async def test_load_rename_disable_delete_preserves_order_ids_and_diagnostics(
    runs_db, isolated_config
):
    first = preset(runs_db, "first")
    second = preset(runs_db, "second", enabled=False)
    missing = preset(runs_db, "gone")
    runs_db.soft_delete_agent_definition(missing)
    runs_db.update_agent_definition(
        first,
        AgentDefinition(
            name="renamed",
            instructions="Review.",
            provider="llama_cpp",
            model="local-test",
        ),
    )
    isolated_config.write_text(f'''[agents.code_review]
verifier_preset_id = "{second}"
reviewers = [{{preset_id="{first}",focus="security"}}, {{preset_id="{second}",focus="tests"}}, {{preset_id="{missing}",focus="API"}}]
''')
    before = isolated_config.read_bytes()
    panel = panel_for(runs_db, isolated_config)
    async with PanelHarness(panel).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert [r.preset_id for r in panel._review_team.reviewers] == [
            first,
            second,
            missing,
        ]
        labels = " ".join(
            _static_text(w) for w in panel.query("#agents-review-team-list Static")
        )
        assert "renamed" in labels and "disabled" in labels and "missing" in labels
        status = _static_text(panel.query_one("#agents-review-team-status", Static))
        assert "duplicate" in status.lower()
        assert panel.query_one("#agents-review-verifier", Select).value == second
    assert isolated_config.read_bytes() == before


@pytest.mark.asyncio
async def test_team_add_focus_reorder_remove_save_cancel_and_preserve_settings(
    runs_db, isolated_config
):
    first = preset(runs_db, "first")
    second = preset(runs_db, "second")
    isolated_config.write_text(
        "[agents]\nspawn_override_enabled = false\n[tools]\ncode_review_enabled = true\n"
    )
    panel = panel_for(runs_db, isolated_config)
    original_presets = runs_db.list_agent_definitions()
    async with PanelHarness(panel).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert panel._review_team.reviewers == ()
        assert "Choose" in _static_text(
            panel.query_one("#agents-review-team-status", Static)
        )
        for pid, focus in ((first, "security"), (second, "tests")):
            panel.query_one("#agents-review-preset", Select).value = pid
            panel.query_one("#agents-review-focus", Input).value = focus
            await panel._edit_review_team("add")
        panel.query_one("#agents-review-team-list", ListView).index = 1
        await panel._edit_review_team("up")
        assert [r.preset_id for r in panel._review_team.reviewers] == [second, first]
        panel.query_one("#agents-review-focus", Input).value = "API"
        await panel._edit_review_team("apply")
        assert panel._review_team.reviewers[0].focus == "API"
        panel.query_one("#agents-review-verifier", Select).value = first
        await panel._save_review_team()
        saved = tomllib.loads(isolated_config.read_text())
        assert saved["agents"]["spawn_override_enabled"] is False
        assert saved["tools"]["code_review_enabled"] is True
        assert saved["agents"]["code_review"]["verifier_preset_id"] == first
        assert saved["agents"]["code_review"]["reviewers"] == [
            {"preset_id": second, "focus": "API"},
            {"preset_id": first, "focus": "security"},
        ]
        await panel._edit_review_team("remove")
        assert len(panel._review_team.reviewers) == 1
        await panel._load_review_team()
        assert len(panel._review_team.reviewers) == 2
        assert panel.query_one("#agents-review-focus", Input).value == "API"
        assert runs_db.list_agent_definitions() == original_presets
        before = isolated_config.read_bytes()
        panel.query_one("#agents-review-verifier", Select).value = Select.NULL
        await panel._save_review_team()
        assert (
            "verifier_preset_id"
            not in tomllib.loads(isolated_config.read_text())["agents"]["code_review"]
        )
        assert before != isolated_config.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (70, 40)])
async def test_keyboard_team_controls_at_narrow_width(
    runs_db, isolated_config, tmp_path, size
):
    pid = preset(runs_db, "local")
    panel = panel_for(runs_db, isolated_config)
    async with PanelHarness(panel).run_test(size=size) as pilot:
        await pilot.pause()
        select = panel.query_one("#agents-review-preset", Select)
        select.focus()
        await pilot.press("enter", "down", "enter", "tab")
        await pilot.press(*"security")
        await pilot.press("tab", "enter")
        await pilot.pause()
        assert [(r.preset_id, r.focus) for r in panel._review_team.reviewers] == [
            (pid, "security")
        ]
        for widget_id in (
            "agents-review-preset",
            "agents-review-focus",
            "agents-review-add",
            "agents-review-save",
            "agents-review-cancel",
        ):
            widget = panel.query_one("#" + widget_id)
            widget.focus()
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            assert widget.has_focus
            assert widget.region.width > 0 and widget.region.right <= size[0]
            assert widget.region.y >= 0 and widget.region.bottom <= size[1], (
                widget.id,
                widget.region,
                panel.query_one("#agents-form").region,
                panel.query_one("#agents-form").scroll_y,
                panel.query_one("#agents-form").max_scroll_y,
            )
        panel.app.save_screenshot(
            f"review-team-{size[0]}x{size[1]}.svg", path=str(tmp_path)
        )


@pytest.mark.asyncio
async def test_contextless_team_load_and_cancel_do_not_read_or_write_profile(
    runs_db, monkeypatch
):
    from tldw_chatbook.Widgets import settings_agents_panel as module

    def forbidden(*args, **kwargs):
        pytest.fail("Contextless review preferences must not access the real profile")

    monkeypatch.setattr(module, "get_code_review_team", forbidden)
    monkeypatch.setattr(module, "save_settings_to_cli_config", forbidden)
    panel = AgentsSettingsPanel(SimpleNamespace(), runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(80, 24)):
        await panel._load_review_team()
        await panel._save_review_team()
        assert panel._review_team.reviewers == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["load", "load_refused", "render", "save"])
async def test_deferred_review_team_work_survives_panel_removal(
    runs_db, isolated_config, monkeypatch, operation
):
    import asyncio
    import threading

    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Widgets import settings_agents_panel as module

    pid = preset(runs_db, "retained")
    isolated_config.write_text(
        f'[agents.code_review]\nreviewers = [{{preset_id="{pid}",focus="saved"}}]\n'
    )
    panel = panel_for(runs_db, isolated_config)
    async with PanelHarness(panel).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        status = panel.query_one("#agents-review-team-status", Static)
        before_status = _static_text(status)
        entered, release = threading.Event(), threading.Event()
        name = (
            "get_code_review_team"
            if operation.startswith("load")
            else "load_agents_routing_config"
            if operation == "render"
            else "save_settings_to_cli_config"
        )
        original = getattr(module, name)

        def deferred(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            if operation == "load_refused":
                raise ReviewRefused("invalid_review_team")
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, deferred)
        if operation == "save":
            panel.query_one("#agents-review-verifier", Select).value = pid
        task = asyncio.create_task(
            panel._load_review_team()
            if operation.startswith("load")
            else panel._render_review_team()
            if operation == "render"
            else panel._save_review_team()
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            await panel.remove()
            assert not status.is_attached
        finally:
            release.set()
        await task
        assert _static_text(status) == before_status
        if operation == "save":
            saved = tomllib.loads(isolated_config.read_text())["agents"]["code_review"]
            assert saved["reviewers"][0]["preset_id"] == pid
            assert saved["verifier_preset_id"] == pid
            assert panel._review_team.verifier_preset_id == pid


@pytest.mark.asyncio
async def test_missing_preset_id_with_markup_renders_literally(
    runs_db, isolated_config
):
    """A raw config id is user text; Select prompts must not parse it as markup."""
    named = preset(runs_db, "named")
    isolated_config.write_text(
        f'''[agents.code_review]
verifier_preset_id = "[b]verify"
reviewers = [{{preset_id="[/]bad"}}, {{preset_id="{named}"}}]
'''
    )
    panel = panel_for(runs_db, isolated_config)
    async with PanelHarness(panel).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        select = panel.query_one("#agents-review-preset", Select)
        select.expanded = True
        await pilot.pause()
        prompts = {str(prompt) for prompt, _ in select._options}
        assert {
            "[/]bad (missing)",
            "[b]verify (missing)",
            "named",
        } <= prompts
