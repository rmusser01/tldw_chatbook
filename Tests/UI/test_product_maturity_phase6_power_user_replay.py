"""Product maturity Phase 6.3 power-user workflow release replay."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Home.dashboard_state import HomeDashboardInput
from tldw_chatbook.app import TldwCli


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_6_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-3-power-user-workflow-release-replay.md"
)
TASK_13 = Path(
    "backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md"
)
TASK_13_3 = Path(
    "backlog/tasks/task-13.3 - Phase-6.3-Power-user-workflow-release-replay.md"
)

LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)

REQUIRED_WORKFLOWS = {
    "grounded-answer-loop",
    "source-to-artifact-loop",
    "agent-run-loop",
    "monitoring-loop",
    "study-loop",
    "recovery-loop",
}


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, f"evidence contains local filesystem prefix(es): {leaked_prefixes}"


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(pieces)


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_3_POWER_USER_RELEASE_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_3_POWER_USER_RELEASE_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _markdown_table_row(markdown: str, first_cell_text: str) -> list[str]:
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or first_cell_text not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == first_cell_text:
            return cells
    raise AssertionError(f"Missing markdown table row for {first_cell_text!r}")


def _workflow_matrix_rows(evidence: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for raw_line in evidence.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] in REQUIRED_WORKFLOWS:
            rows[cells[0]] = cells
    return rows


@pytest.mark.asyncio
async def test_phase6_power_user_release_replay_exposes_fast_repeat_paths() -> None:
    """Verify power-user paths can be repeated through deterministic shell seams."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=[],
    )

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )
            home_text = _screen_text(app)
            assert "Start in Console" in home_text
            assert "Ctrl+P" in home_text
            assert any(binding.key == "ctrl+p" for binding in TldwCli.BINDINGS)

            app.screen.query_one("#home-primary-action", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
            )
            console_text = _screen_text(app)
            assert "Console" in console_text
            assert "Chat/RAG/Follow" in console_text
            assert "Live work sources" in console_text
            assert "RAG:" in console_text
            assert "Artifacts:" in console_text

            app.screen.query_one("#nav-library", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and app.screen.__class__.__name__ == "LibraryScreen",
            )
            # The retired hub's "Search/RAG" / "Study Dashboard" /
            # "Import/Export Sources" action-region copy is gone; the rail
            # row titles are the surviving discoverable surface for the same
            # capabilities. The placeholder Import/Export row itself is
            # retired outright (see the inventory verdict); Import media
            # absorbed its rail slot.
            library_text = _screen_text(app)
            assert "Search / RAG" in library_text
            assert "Study decks" in library_text
            assert "Import media" in library_text
            assert "Collections" in library_text
            assert not app.screen.query("#library-row-ingest-import-export")

            app.screen.query_one("#library-row-browse-search", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and "Library Search/RAG" in _screen_text(app),
            )

            app.screen.query_one("#nav-library", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library" and app.screen.__class__.__name__ == "LibraryScreen",
            )
            # Import media is now a first-class canvas row (not a deep-link
            # to the standalone Ingest screen): pressing it mounts the
            # ingest canvas stub in place rather than navigating away from
            # Library (Task 4 replaces the stub with the real canvas).
            app.screen.query_one("#library-row-ingest-import-media", Button).press()
            await _wait_until(
                pilot,
                lambda: app.current_tab == "library"
                and bool(app.screen.query("#library-ingest-canvas-placeholder")),
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
            assert "Recovery: Review the Watchlists run details or retry from Watchlists." in live_work_text

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

