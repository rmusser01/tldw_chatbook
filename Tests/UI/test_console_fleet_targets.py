"""Resolved child targets reach the painted rail, including saved runs."""

import json

import pytest

from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_console_inspector_section import _SectionHarness
from Tests.UI.test_console_parallel_runs import _assert_painted_at_own_region
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    _subagent_summaries_from_fleet,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Console_Modules.agent import (
    _fleet_row_from_handle,
    _fleet_row_from_record,
    _fleet_row_from_summary,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
)

pytestmark = pytest.mark.bootstrap_profile


def _painted_text(app, widget):
    _assert_painted_at_own_region(app, widget)
    region = widget.region
    return " ".join(
        " ".join(
            strip.crop(region.x, region.right).text
            for strip in app.screen._compositor.render_strips()[
                region.y : region.bottom
            ]
        ).split()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [30, 38])
async def test_routed_live_and_finished_child_paints_complete_frozen_target(
    width, monkeypatch
):
    fleet = FleetCoordinator(1, lambda: 1.0)
    handle = fleet.reserve(
        "Review logs",
        "implementer",
        resolved_provider="custom-ep:qwen-local",
        resolved_model="qwen3.8-27b",
    )
    fleet.attach_run(handle.handle_id, "child")
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="targets",
        collapsible=False,
        rows=(_fleet_row_from_handle(fleet.get(handle.handle_id), now=2.0),),
    )
    monkeypatch.setattr(
        _SectionHarness, "CSS_PATH", [str(path) for path in APP_STYLESHEETS]
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        selector = "#console-inspector-section-targets-row-0-secondary"
        assert "custom-ep:qwen-local · qwen3.8-27b" in _painted_text(
            app, app.query_one(selector)
        )
        assert not fleet.set_resolved_target("foreign", "ollama", "foreign-model")
        assert fleet.set_resolved_target("child", "ollama", "fallback-model")
        section.sync_state(
            ConsoleInspectorSectionState(
                rows=(_fleet_row_from_handle(fleet.get(handle.handle_id), now=3.0),),
                summary="",
            )
        )
        await pilot.pause()
        assert "ollama · fallback-model" in _painted_text(app, app.query_one(selector))
        fleet.finish(handle.handle_id, "done", result="Reviewed")
        assert not fleet.set_resolved_target("child", "openai", "late-model")
        summaries = _subagent_summaries_from_fleet(fleet.snapshot(), [])
        section.sync_state(
            ConsoleInspectorSectionState(
                rows=(_fleet_row_from_summary(summaries[0], 0),),
                summary="",
            )
        )
        await pilot.pause()
        assert "ollama · fallback-model" in _painted_text(app, app.query_one(selector))


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_reopened_child_paints_saved_target_or_honest_unavailable(
    tmp_path, legacy, monkeypatch
):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path, client_id="test")
    run_id = db.create_run(
        conversation_id="c",
        agent_kind="subagent",
        task="Review logs",
        agent_definition="edited-preset",
        resolved_provider=None if legacy else "custom-ep:qwen-local",
        resolved_model=None if legacy else "qwen3.8-27b",
    )
    db.set_status(run_id, "done")
    db.create_agent_definition(
        AgentDefinition(
            name="edited-preset",
            instructions="Edited after launch.",
            provider="llama_cpp",
            model="edited-model",
        )
    )
    db.close()
    reopened = AgentRunsDB(path, client_id="reopened")
    try:
        record = reopened.get_run(run_id)
        summary = ConsoleAgentBridge.historical_subagent_summary(record)
        section = ConsoleInspectorSection(
            title="Agents",
            section_id="targets",
            collapsible=False,
            rows=(_fleet_row_from_summary(summary, 0), _fleet_row_from_record(record)),
        )
        monkeypatch.setattr(
            _SectionHarness, "CSS_PATH", [str(p) for p in APP_STYLESHEETS]
        )
        app = _SectionHarness(section)
        async with app.run_test(size=(30, 24)) as pilot:
            await pilot.pause()
            for index in range(2):
                painted = _painted_text(
                    app,
                    app.query_one(
                        f"#console-inspector-section-targets-row-{index}-secondary"
                    ),
                )
                expected = (
                    "Target unavailable"
                    if legacy
                    else "custom-ep:qwen-local · qwen3.8-27b"
                )
                assert expected in painted
                assert "edited-model" not in painted
    finally:
        reopened.close()


def test_saved_child_target_projects_the_active_frozen_fallback():
    record = {
        "id": "child",
        "resolved_provider": "openai",
        "resolved_model": "original",
        "fallback_targets_json": json.dumps(
            [
                {"provider": "openai", "model": "original"},
                {"provider": "custom-ep:saved", "model": "active"},
            ]
        ),
        "active_fallback_index": 1,
    }
    summary = ConsoleAgentBridge.historical_subagent_summary(record)
    assert summary.resolved_provider == "custom-ep:saved"
    assert summary.resolved_model == "active"
    assert "custom-ep:saved · active" in _fleet_row_from_record(record).secondary_text
    record["active_fallback_index"] = 99
    assert "Target unavailable" in _fleet_row_from_record(record).secondary_text
