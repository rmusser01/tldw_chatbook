"""Product maturity Phase 6.4 keyboard/focus/accessibility and visual sweep."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.css.query import NoMatches, QueryError, TooManyMatches
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar, NavigateToScreen
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER
from tldw_chatbook.app import TldwCli

if TYPE_CHECKING:
    from textual.pilot import Pilot


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_6_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-4-keyboard-focus-accessibility-visual-sweep.md"
)
TASK_13 = Path(
    "backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md"
)
TASK_13_4 = Path(
    "backlog/tasks/task-13.4 - Phase-6.4-Keyboard-focus-accessibility-and-visual-sweep.md"
)

TOP_LEVEL_DESTINATION_IDS = tuple(destination.destination_id for destination in SHELL_DESTINATION_ORDER)
TERMINAL_SIZE_MATRIX = (
    ("compact", (100, 32)),
    ("default", (140, 42)),
    ("wide", (180, 50)),
)
DESTINATION_BODY_SELECTORS: dict[str, tuple[str, ...]] = {
    "home": ("#home-dashboard",),
    "console": ("#console-shell", "#console-transcript-region"),
    "library": ("#library-shell",),
    "artifacts": ("#artifacts-shell",),
    "personas": ("#personas-shell",),
    "watchlists_collections": ("#watchlists-collections-shell",),
    "schedules": ("#schedules-shell",),
    "workflows": ("#workflows-shell",),
    "mcp": ("#mcp-shell", "#unified-mcp-panel"),
    "acp": ("#acp-shell",),
    "skills": ("#skills-shell",),
    "settings": ("#settings-shell",),
}
BROKEN_TEXT_PATTERNS = (
    "Traceback",
    "Unhandled exception",
    "No screens installed",
    "Unable to mount",
    "Internal Error",
)
RAW_OBJECT_REPR = re.compile(r"<[\w.]+ object at 0x[0-9a-fA-F]+>")
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, f"evidence contains local filesystem prefix(es): {leaked_prefixes}"


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _build_release_sweep_app() -> TldwCli:
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"
    return app


def _screen_text(app: TldwCli) -> str:
    content = app.screen.query_one("#screen-content")
    pieces: list[str] = []
    for widget in content.query(Static):
        pieces.append(str(widget.renderable))
    for widget in content.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


async def _wait_until(
    pilot: "Pilot",
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
    context: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    suffix = f" for {context}" if context else ""
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s{suffix}")


def _assert_destination_body_mounted(app: TldwCli, destination_id: str, size_label: str) -> None:
    selectors = DESTINATION_BODY_SELECTORS[destination_id]
    content = app.screen.query_one("#screen-content")
    missing_selectors = [selector for selector in selectors if not list(content.query(selector))]
    assert not missing_selectors, (
        f"{destination_id} missing primary body selector(s) at {size_label}: "
        f"{', '.join(missing_selectors)}"
    )


def _visual_chrome_ready(app: TldwCli, destination_id: str) -> bool:
    try:
        nav_bar = app.screen.query_one(MainNavigationBar)
        nav_ids = tuple(button.id.removeprefix("nav-") for button in nav_bar.query(Button))
        if nav_ids != TOP_LEVEL_DESTINATION_IDS:
            return False
        if not nav_bar.query_one(f"#nav-{destination_id}", Button).has_class("is-active"):
            return False
        content = app.screen.query_one("#screen-content")
        return any(list(content.query(selector)) for selector in DESTINATION_BODY_SELECTORS[destination_id])
    except (NoMatches, QueryError, TooManyMatches):
        # During async recomposition, expected query misses mean the chrome is not ready yet.
        return False


def _assert_visual_snapshot_is_healthy(app: TldwCli, destination_id: str, size_label: str) -> None:
    nav_bar = app.screen.query_one(MainNavigationBar)
    nav_ids = tuple(button.id.removeprefix("nav-") for button in nav_bar.query(Button))
    assert nav_ids == TOP_LEVEL_DESTINATION_IDS
    assert nav_bar.query_one(f"#nav-{destination_id}", Button).has_class("is-active")
    assert "Ctrl+P" in str(app.screen.query_one("#nav-overflow-hint", Static).renderable)
    _assert_destination_body_mounted(app, destination_id, size_label)

    text = _screen_text(app)
    svg = app.export_screenshot(title=f"Phase 6.4 {size_label} {destination_id}", simplify=True)

    assert text.strip(), f"{destination_id} rendered empty content at {size_label}"
    assert "<svg" in svg
    assert "</svg>" in svg
    assert len(svg) > 1_000
    for broken_text in BROKEN_TEXT_PATTERNS:
        assert broken_text not in text
        assert broken_text not in svg
    assert RAW_OBJECT_REPR.search(text) is None
    assert RAW_OBJECT_REPR.search(svg) is None


def _metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_4_FOCUS_VISUAL_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_4_FOCUS_VISUAL_METADATA:END -->",
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


@pytest.mark.parametrize(("size_label", "size"), TERMINAL_SIZE_MATRIX)
@pytest.mark.asyncio
async def test_phase6_visual_chrome_survives_release_terminal_size_matrix(
    size_label: str,
    size: tuple[int, int],
) -> None:
    app = _build_release_sweep_app()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=size) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
                context=f"{size_label}:home",
            )

            for destination in SHELL_DESTINATION_ORDER:
                _screen_name, expected_tab, expected_screen_class = app._resolve_screen_navigation_target(
                    destination.primary_route
                )
                assert expected_screen_class is not None, destination.primary_route

                if app.current_tab != expected_tab:
                    await app.handle_screen_navigation(NavigateToScreen(destination.primary_route))
                    await _wait_until(
                        pilot,
                        lambda expected_tab=expected_tab, expected_screen_class=expected_screen_class: (
                            app.current_tab == expected_tab
                            and isinstance(app.screen, expected_screen_class)
                        ),
                        context=f"{size_label}:{destination.destination_id}:navigation",
                    )
                await _wait_until(
                    pilot,
                    lambda destination_id=destination.destination_id: _visual_chrome_ready(app, destination_id),
                    context=f"{size_label}:{destination.destination_id}:visual-chrome",
                )

                _assert_visual_snapshot_is_healthy(app, destination.destination_id, size_label)


@pytest.mark.asyncio
async def test_phase6_home_keyboard_focus_reaches_navigation_and_primary_action() -> None:
    app = _build_release_sweep_app()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            if not isinstance(app.focused, Button):
                await pilot.press("tab")

            expected_focus_ids = [
                *(f"nav-{destination_id}" for destination_id in TOP_LEVEL_DESTINATION_IDS),
                "home-primary-action",
            ]
            observed_focus_ids: list[str] = []
            for index, expected_focus_id in enumerate(expected_focus_ids):
                await _wait_until(
                    pilot,
                    lambda: isinstance(app.focused, Button) and app.focused.id == expected_focus_id,
                    context=f"focus:{expected_focus_id}",
                )
                focused = app.focused
                assert isinstance(focused, Button)
                observed_focus_ids.append(focused.id or "")
                assert str(focused.label).strip()
                if index < len(expected_focus_ids) - 1:
                    await pilot.press("tab")

            assert observed_focus_ids == expected_focus_ids
            assert any(binding.key == "ctrl+p" for binding in TldwCli.BINDINGS)


@pytest.mark.skip(reason="Stale release-era snapshot (copy/evidence drifted); re-pin or retire via backlog task-98")
def test_phase6_focus_visual_evidence_and_tracking_are_current() -> None:
    evidence = _text(EVIDENCE)
    readme = _text(PHASE_6_README)
    tracker = _text(TRACKER)
    parent_task = _text(TASK_13)
    task = _text(TASK_13_4)
    metadata = _metadata(evidence)

    _assert_no_local_path_prefixes(evidence)
    assert metadata["task"] == "TASK-13.4"
    assert metadata["parent_task"] == "TASK-13"
    assert metadata["decision"] == "keyboard_focus_accessibility_visual_sweep_recorded"
    assert metadata["terminal_sizes"] == ["compact", "default", "wide"]
    assert metadata["destinations_checked"] == list(TOP_LEVEL_DESTINATION_IDS)
    assert metadata["p0_p1_findings"] == []
    assert metadata["screenshot_gate"] == "not_required_no_visible_ui_changes"
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0

    for section in (
        "## Environment",
        "## Size Matrix",
        "## Focus And Keyboard Sweep",
        "## Accessibility And Readability Sweep",
        "## Visual Sweep Result",
        "## P0/P1 Decision",
        "## Residual Risk",
        "## Verification",
    ):
        assert section in evidence
    assert "running Textual app" in evidence
    assert "Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py" in evidence

    assert EVIDENCE.name in readme
    assert "Phase 6.4 Keyboard/focus/accessibility and visual sweep" in readme
    assert "Status: TASK-13.1 through TASK-13.4 done; TASK-13.5 through TASK-13.7 not started" in readme

    phase6_row = _markdown_table_row(tracker, "Phase 6: Release Hardening And Documentation")
    assert "in-progress; TASK-13.1 through TASK-13.4 done" in phase6_row[2]
    assert "TASK-13.4" in phase6_row[3]
    assert EVIDENCE.name in phase6_row[4]
    assert "keyboard/focus/accessibility and visual sweep verified" in phase6_row[5].lower()

    qa_row = _markdown_table_row(tracker, "Phase 6.4")
    assert EVIDENCE.as_posix() in qa_row[1]
    assert "verified; TASK-13.4 done" == qa_row[2]

    assert "TASK-13.4" in parent_task
    assert "status: Done" in task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in task
    assert "## Implementation Plan" in task
    assert "## Implementation Notes" in task
