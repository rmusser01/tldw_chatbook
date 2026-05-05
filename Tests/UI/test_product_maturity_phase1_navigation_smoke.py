"""Product maturity Phase 1.3 top-level navigation smoke contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-3-navigation-smoke.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.3 - Product-Maturity-Phase-1.3-Top-Level-Navigation-Smoke-Walkthrough.md")
TOP_LEVEL_DESTINATION_IDS = tuple(destination.destination_id for destination in SHELL_DESTINATION_ORDER)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, path_name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        path = tmp_path / path_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))


def _build_clean_navigation_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


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


@pytest.mark.asyncio
async def test_clean_run_top_level_navigation_reaches_every_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_navigation_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 50)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home" and app.screen.__class__.__name__ == "HomeScreen",
            )

            nav_bar = app.screen.query_one(MainNavigationBar)
            nav_ids = tuple(button.id.removeprefix("nav-") for button in nav_bar.query(Button))
            assert nav_ids == TOP_LEVEL_DESTINATION_IDS
            assert "Ctrl+P" in str(app.screen.query_one("#nav-overflow-hint", Static).renderable)

            reached: dict[str, str] = {}
            for destination in SHELL_DESTINATION_ORDER:
                expected_screen_name, expected_tab, expected_screen_class = app._resolve_screen_navigation_target(
                    destination.primary_route
                )
                assert expected_screen_class is not None, destination.primary_route

                app.screen.query_one(f"#nav-{destination.destination_id}", Button).press()
                await _wait_until(
                    pilot,
                    lambda expected_tab=expected_tab, expected_screen_class=expected_screen_class: (
                        app.current_tab == expected_tab
                        and app.screen.__class__.__name__ == expected_screen_class.__name__
                    ),
                )

                screen_text = _screen_text(app)
                assert screen_text.strip(), destination.destination_id
                reached[destination.destination_id] = expected_screen_name

            assert set(reached) == set(TOP_LEVEL_DESTINATION_IDS)


def test_phase_one_three_evidence_records_top_level_navigation_smoke() -> None:
    evidence = _text(EVIDENCE)

    for required_text in (
        "Phase 1.3",
        "TASK-8.3",
        "Top-Level Navigation Smoke",
        "Fresh HOME",
        "Ctrl+P",
        "usable, not merely rendered",
        "Remaining Phase 1 gates",
        "No P0/P1 navigation blockers",
    ):
        assert required_text in evidence
    for destination_id in TOP_LEVEL_DESTINATION_IDS:
        assert f"`{destination_id}`" in evidence


def test_phase_one_three_tracking_and_task_closeout_are_current() -> None:
    tracker = _text(TRACKER)
    readme = _text(PHASE_1_README)
    task = _text(TASK)

    assert "Phase 1.3" in tracker
    assert "TASK-8.3" in tracker
    assert EVIDENCE.name in tracker
    assert EVIDENCE.name in readme
    assert "Phase 1.3 top-level navigation status: verified" in readme
    assert "status: Done" in task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in task
    assert "Implementation Notes" in task
