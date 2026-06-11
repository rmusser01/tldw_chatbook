"""Product maturity Phase 1.5 visual broken-state audit contract."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar, NavigateToScreen
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from tldw_chatbook.app import TldwCli


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-5-visual-broken-state-audit.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.5 - Product-Maturity-Phase-1.5-Visual-Broken-State-Audit.md")
TOP_LEVEL_DESTINATION_IDS = tuple(destination.destination_id for destination in SHELL_DESTINATION_ORDER)
DESTINATION_BODY_SELECTORS: dict[str, tuple[str, ...]] = {
    "home": ("#home-dashboard",),
    # The live-work readiness card only mounts while a launch is pending and
    # the legacy #chat-window surface is no longer composed; pin the
    # always-present control bar + native session surface instead.
    "console": ("#console-control-bar", "#console-session-surface"),
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
TERMINAL_SIZE_MATRIX = (
    ("compact", (100, 32)),
    ("laptop", (140, 40)),
    ("large", (180, 50)),
)
MIN_VALID_SVG_LENGTH = 1_000
BROKEN_TEXT_PATTERNS = (
    "Traceback",
    "Unhandled exception",
    "No screens installed",
    "Unable to mount",
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


def _build_clean_visual_audit_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> "TldwCli":
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


def _screen_text(app: "TldwCli") -> str:
    content = app.screen.query_one("#screen-content")
    pieces: list[str] = []
    for widget in content.query(Static):
        pieces.append(str(widget.renderable))
    for widget in content.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


def _assert_destination_body_mounted(app: "TldwCli", destination_id: str, size_label: str) -> None:
    selectors = DESTINATION_BODY_SELECTORS[destination_id]
    content = app.screen.query_one("#screen-content")
    missing_selectors = [selector for selector in selectors if not list(content.query(selector))]
    assert not missing_selectors, (
        f"{destination_id} missing primary body selector(s) at {size_label}: "
        f"{', '.join(missing_selectors)}"
    )


def _has_visual_chrome(app: "TldwCli") -> bool:
    nav_bars = list(app.screen.query(MainNavigationBar))
    if not nav_bars or not list(app.screen.query("#screen-content")):
        return False
    nav_buttons = tuple(button.id for button in nav_bars[0].query(Button))
    return nav_buttons == tuple(f"nav-{destination_id}" for destination_id in TOP_LEVEL_DESTINATION_IDS)


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
    context_suffix = f" for {context}" if context else ""
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s{context_suffix}")


def _assert_visual_snapshot_is_healthy(app: "TldwCli", destination_id: str, size_label: str) -> None:
    nav_bar = app.screen.query_one(MainNavigationBar)
    nav_ids = tuple(button.id.removeprefix("nav-") for button in nav_bar.query(Button))
    assert nav_ids == TOP_LEVEL_DESTINATION_IDS
    assert nav_bar.query_one(f"#nav-{destination_id}", Button).has_class("is-active")
    assert "Ctrl+P" in str(app.screen.query_one("#nav-overflow-hint", Static).renderable)
    _assert_destination_body_mounted(app, destination_id, size_label)

    text = _screen_text(app)
    svg = app.export_screenshot(title=f"Phase 1.5 {size_label} {destination_id}", simplify=True)

    assert text.strip(), f"{destination_id} rendered empty content at {size_label}"
    assert "<svg" in svg
    assert "</svg>" in svg
    assert len(svg) > MIN_VALID_SVG_LENGTH
    for broken_text in BROKEN_TEXT_PATTERNS:
        assert broken_text not in text
        assert broken_text not in svg
    assert RAW_OBJECT_REPR.search(text) is None
    assert RAW_OBJECT_REPR.search(svg) is None


@pytest.mark.parametrize(("size_label", "size"), TERMINAL_SIZE_MATRIX)
@pytest.mark.asyncio
async def test_clean_run_top_level_visual_snapshots_survive_terminal_size(
    size_label: str,
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_visual_audit_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=size) as pilot:
            _initial_screen_name, initial_tab, initial_screen_class = app._resolve_screen_navigation_target("home")
            assert initial_screen_class is not None
            await _wait_until(
                pilot,
                lambda: app.current_tab == initial_tab and isinstance(app.screen, initial_screen_class),
                context=f"{size_label}:home:initial",
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
                    lambda: _has_visual_chrome(app),
                    context=f"{size_label}:{destination.destination_id}:chrome",
                )

                _assert_visual_snapshot_is_healthy(app, destination.destination_id, size_label)


def test_visual_audit_destination_body_selectors_cover_top_level_destinations() -> None:
    assert set(DESTINATION_BODY_SELECTORS) == set(TOP_LEVEL_DESTINATION_IDS)


def test_phase_one_five_evidence_records_visual_broken_state_audit() -> None:
    evidence = _text(EVIDENCE)

    _assert_no_local_path_prefixes(evidence)
    for required_text in (
        "Phase 1.5",
        "TASK-8.5",
        "Visual Broken-State Audit",
        "compact",
        "laptop",
        "large",
        "SVG screenshot export",
        "No P0/P1 visual broken-state blockers",
        "Remaining Phase 1 gates",
    ):
        assert required_text in evidence
    for destination_id in TOP_LEVEL_DESTINATION_IDS:
        assert f"`{destination_id}`" in evidence


def test_phase_one_five_tracking_and_task_closeout_are_current() -> None:
    tracker = _text(TRACKER)
    readme = _text(PHASE_1_README)
    task = _text(TASK)

    assert "Phase 1.5" in tracker
    assert "TASK-8.5" in tracker
    assert EVIDENCE.name in tracker
    assert EVIDENCE.name in readme
    assert "Phase 1.5 visual broken-state status: verified" in readme
    assert "status: Done" in task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in task
    assert "Implementation Notes" in task
