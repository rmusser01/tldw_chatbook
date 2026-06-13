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

            # Import/Export is now an in-Library mode; the ingest screen is
            # one press deeper via the mode's "Open Ingest" action.
            app.screen.query_one("#library-open-import-export", Button).press()
            await _wait_until(
                pilot,
                lambda: bool(app.screen.query("#library-import-export-open-ingest")),
            )
            app.screen.query_one("#library-import-export-open-ingest", Button).press()
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
            # Search/RAG is likewise an in-Library mode now.
            app.screen.query_one("#library-open-search", Button).press()
            await _wait_until(
                pilot,
                lambda: bool(app.screen.query("#library-search-rag-panel")),
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

            # The Watchlists-run detail lives on the subscriptions screen,
            # which is gated behind optional dependencies (feedparser etc.);
            # mirror the app's own route gating in environments without them.
            from tldw_chatbook.UI.Navigation import screen_registry

            subscriptions_route = screen_registry._SCREEN_ROUTES.get("subscriptions")
            if subscriptions_route is not None and subscriptions_route.dependencies_available():
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


