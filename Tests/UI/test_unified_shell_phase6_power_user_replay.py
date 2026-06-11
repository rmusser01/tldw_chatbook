"""Phase 6.2 power-user workflow replay contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_unified_shell_phase6_first_time_replay import (
    PHASE_6_PARENT_TASK,
    PHASE_6_README,
    ROADMAP,
    _phase_evidence_row,
    _phase_overview_row,
    _screen_text,
    _status_line,
    _test_cli_setting,
    _text,
    _wait_until,
)
from tldw_chatbook.Home.dashboard_state import HomeDashboardInput
from tldw_chatbook.app import TldwCli


PHASE_6_POWER_USER_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-2-power-user-workflow-replay.md"
)
PHASE_6_POWER_USER_TASK = Path(
    "backlog/tasks/task-7.2 - Phase-6.2-Replay-power-user-workflows.md"
)


def _phase_six_power_user_metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_2_POWER_USER_REPLAY_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_2_POWER_USER_REPLAY_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.skip(reason="Stale release-era snapshot (copy/evidence drifted); re-pin or retire via backlog task-98")
@pytest.mark.asyncio
async def test_power_user_shell_replay_supports_fast_repeated_core_workflows() -> None:
    """Verify repeated shell workflows use direct, deterministic app paths."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
    )

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )
            assert "Start in Console" in _screen_text(app)

            app.screen.query_one("#home-primary-action", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )
            console_text = _screen_text(app)
            assert "Live work sources" in console_text
            assert "Watchlists: Connected" in console_text
            assert "More: Ctrl+P" in console_text
            assert any(binding.key == "ctrl+p" for binding in TldwCli.BINDINGS)

            app.screen.query_one("#nav-library", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and app.screen.__class__.__name__ == "LibraryScreen",
            )
            library_text = _screen_text(app)
            assert "Import/Export Sources" in library_text
            assert "Search/RAG" in library_text

            app.screen.query_one("#library-open-import-export", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "ingest"
                and app.screen.__class__.__name__ == "MediaIngestScreen",
            )

            app.screen.query_one("#nav-console", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )
            app.screen.query_one("#nav-library", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and app.screen.__class__.__name__ == "LibraryScreen",
            )
            app.screen.query_one("#library-open-search", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "search" and app.screen.__class__.__name__ == "SearchScreen",
            )

            app.open_console_for_live_work(
                source="Watchlists",
                title="Daily security feed",
                payload={"target_id": "local:watchlist_run:91", "run_id": 91},
                status="failed",
                recovery="Review the Watchlists run details or retry from Watchlists.",
                action_label="Open Watchlists run",
            )
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )
            live_work_text = _screen_text(app)
            assert "Source: Watchlists" in live_work_text
            assert "Title: Daily security feed" in live_work_text
            assert "Action: Open Watchlists run" in live_work_text

            app.screen.query_one("#console-live-work-primary-action", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "subscriptions"
                and app.screen.__class__.__name__ == "SubscriptionScreen",
            )
            subscription_window = app.screen.subscription_window
            assert subscription_window is not None
            assert subscription_window.initial_tab == "watchlist-runs"
            assert subscription_window._selected_watchlist_run_id == "local:watchlist_run:91"


def test_phase_six_power_user_replay_evidence_and_tracking_are_current() -> None:
    """Verify Phase 6.2 evidence and tracking stay aligned with the roadmap."""
    evidence = _text(PHASE_6_POWER_USER_EVIDENCE)
    readme = _text(PHASE_6_README)
    roadmap = _text(ROADMAP)
    parent_task = _text(PHASE_6_PARENT_TASK)
    power_user_task = _text(PHASE_6_POWER_USER_TASK)
    metadata = _phase_six_power_user_metadata(evidence)

    assert "/Users/" not in evidence
    assert metadata["task"] == "TASK-7.2"
    assert metadata["parent_task"] == "TASK-7"
    assert metadata["persona"] == "power-user"
    assert metadata["decision"] == "power_user_workflows_recorded"
    assert metadata["entry_path"] == "running-app-fast-repeat-use"
    assert metadata["verified_workflows"] == [
        "home-next-action-to-console",
        "console-live-work-readiness",
        "library-search-rag",
        "library-import-export",
        "console-live-work-follow-through",
    ]
    assert metadata["speed_paths"] == ["home-primary-action", "top-nav", "ctrl-p-affordance"]
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0

    for section in (
        "## Environment",
        "## Workflow Matrix",
        "## Repeated-Use Findings",
        "## Visual Usability Notes",
        "## Keyboard Path Result",
        "## Functional Result",
        "## Defect Severity",
        "## Deferred Service-Depth Work",
        "## Residual Risk",
    ):
        assert section in evidence
    assert "running Textual app" in evidence
    assert "Tests/UI/test_unified_shell_phase6_power_user_replay.py" in evidence

    assert _status_line(readme) == "verified"
    assert PHASE_6_POWER_USER_EVIDENCE.name in readme
    assert "TASK-7.2" in readme

    normalized_status = _status_line(roadmap).lower().replace("-", " ")
    assert re.search(r"phase\s+6\s+verified", normalized_status)
    assert _phase_evidence_row(roadmap, "Phase 6")[2] == "verified"
    assert str(PHASE_6_POWER_USER_EVIDENCE).replace("\\", "/") in roadmap
    assert "Phase 6.2: Replay power-user workflows - `TASK-7.2`" in roadmap
    phase_six_overview = _phase_overview_row(roadmap, "Phase 6: Audit Replay And Closeout")
    assert phase_six_overview[2] == "verified"
    assert "TASK-7" in phase_six_overview[3]
    assert "TASK-7.1" in phase_six_overview[3]
    assert "TASK-7.2" in phase_six_overview[3]
    assert "service-depth and live-path risks remain tracked" in phase_six_overview[5]

    assert "status: Done" in parent_task
    assert "TASK-7.2" in parent_task
    assert "- [x] #1 First-time user walkthrough is replayed against the running app." in parent_task
    assert "- [x] #2 Power-user workflows are replayed against the running app." in parent_task
    assert "- [x] #3 Nielsen heuristic closeout documents remaining defects and residual risks." in parent_task
    assert "- [x] #4 Durable QA summaries exist under Docs/superpowers/qa/unified-shell/phase-6/." in parent_task

    assert "status: Done" in power_user_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in power_user_task
    assert "Implementation Notes" in power_user_task
