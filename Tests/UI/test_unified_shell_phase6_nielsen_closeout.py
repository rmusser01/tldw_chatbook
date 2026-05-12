"""Phase 6.3 Nielsen heuristic closeout contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_unified_shell_phase6_first_time_replay import (
    EXPECTED_NAV,
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


PHASE_6_NIELSEN_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-3-nielsen-heuristic-closeout.md"
)
PHASE_6_NIELSEN_TASK = Path(
    "backlog/tasks/task-7.3 - Phase-6.3-Replay-Nielsen-heuristic-closeout.md"
)

NIELSEN_HEURISTICS = [
    "Visibility of system status",
    "Match between system and the real world",
    "User control and freedom",
    "Consistency and standards",
    "Error prevention",
    "Recognition rather than recall",
    "Flexibility and efficiency of use",
    "Aesthetic and minimalist design",
    "Help users recognize diagnose and recover from errors",
    "Help and documentation",
]


def _phase_six_nielsen_metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_3_NIELSEN_CLOSEOUT_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_3_NIELSEN_CLOSEOUT_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.asyncio
async def test_nielsen_closeout_replays_core_heuristic_signals_in_running_app() -> None:
    """Verify visible shell behavior maps to the closeout's heuristic evidence."""
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            nav_buttons = list(app.screen.query("MainNavigationBar").first().query(Button))
            assert [(button.id, str(button.label).strip()) for button in nav_buttons] == EXPECTED_NAV

            home_text = _screen_text(app)
            assert "Status" in home_text
            assert "Model: Blocked" in home_text
            assert "Attention" in home_text
            assert "Next Best Action" in home_text
            assert "More: Ctrl+P" in home_text

            app.screen.query_one("#nav-console", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )
            console_text = _screen_text(app)
            assert "Live work sources" in console_text
            assert "ACP: Not wired" in console_text
            assert "MCP: Not wired" in console_text
            assert "RAG: Connected" in console_text

            app.screen.query_one("#nav-acp", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "acp" and app.screen.__class__.__name__ == "ACPScreen",
            )
            acp_text = _screen_text(app)
            assert "Runtime not configured" in acp_text
            assert "Why: no ACP-compatible runtime is configured" in acp_text
            assert "Next: Configure ACP runtime setup in ACP before launch." in acp_text
            assert app.screen.query_one("#acp-launch-agent", Button).disabled is True

            app.screen.query_one("#nav-library", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and app.screen.__class__.__name__ == "LibraryScreen",
            )
            library_text = _screen_text(app)
            assert "Library" in library_text
            assert "Import/Export Sources" in library_text
            assert "Search/RAG" in library_text

            app.screen.query_one("#nav-settings", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "settings" and app.screen.__class__.__name__ == "SettingsScreen",
            )
            settings_text = _screen_text(app)
            assert "Settings owns global preferences" in settings_text
            assert "MCP and tool-control settings live under MCP, not global Settings." in settings_text


def test_phase_six_nielsen_closeout_evidence_and_tracking_are_current() -> None:
    """Verify Phase 6.3 evidence closes the phase without losing residual-risk detail."""
    evidence = _text(PHASE_6_NIELSEN_EVIDENCE)
    readme = _text(PHASE_6_README)
    roadmap = _text(ROADMAP)
    parent_task = _text(PHASE_6_PARENT_TASK)
    nielsen_task = _text(PHASE_6_NIELSEN_TASK)
    metadata = _phase_six_nielsen_metadata(evidence)

    assert "/Users/" not in evidence
    assert metadata["task"] == "TASK-7.3"
    assert metadata["parent_task"] == "TASK-7"
    assert metadata["persona"] == "senior-ux-designer"
    assert metadata["decision"] == "nielsen_closeout_recorded"
    assert metadata["entry_path"] == "running-app-nielsen-heuristic-closeout"
    assert metadata["heuristics_reviewed"] == NIELSEN_HEURISTICS
    assert metadata["unresolved_findings"] == [
        "library-subroute-return-affordance",
        "full-keyboard-sweep-needs-manual-terminal-pass",
        "service-depth-live-paths-remain-deferred",
    ]
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0

    for section in (
        "## Environment",
        "## Heuristic Audit",
        "## Prioritized Findings",
        "## UX Decision",
        "## Deferred Service-Depth Work",
        "## Residual Risk",
        "## Verification",
    ):
        assert section in evidence
    for heuristic in NIELSEN_HEURISTICS:
        assert heuristic in evidence
    assert "running Textual app" in evidence
    assert "Tests/UI/test_unified_shell_phase6_nielsen_closeout.py" in evidence

    assert _status_line(readme) == "verified"
    assert PHASE_6_NIELSEN_EVIDENCE.name in readme
    assert "TASK-7.3" in readme

    normalized_status = _status_line(roadmap).lower().replace("-", " ")
    assert re.search(r"phase\s+6\s+verified", normalized_status)
    assert _phase_evidence_row(roadmap, "Phase 6")[2] == "verified"
    assert str(PHASE_6_NIELSEN_EVIDENCE).replace("\\", "/") in roadmap
    assert "Phase 6.3: Replay Nielsen heuristic closeout - `TASK-7.3`" in roadmap
    phase_six_overview = _phase_overview_row(roadmap, "Phase 6: Audit Replay And Closeout")
    assert phase_six_overview[2] == "verified"
    assert "TASK-7" in phase_six_overview[3]
    assert "TASK-7.1" in phase_six_overview[3]
    assert "TASK-7.2" in phase_six_overview[3]
    assert "TASK-7.3" in phase_six_overview[3]
    assert "service-depth and live-path risks remain tracked" in phase_six_overview[5]

    assert "status: Done" in parent_task
    assert "TASK-7.3" in parent_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in parent_task
    assert "Implementation Notes" in parent_task

    assert "status: Done" in nielsen_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in nielsen_task
    assert "Implementation Notes" in nielsen_task
