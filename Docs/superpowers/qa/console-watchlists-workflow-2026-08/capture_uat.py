#!/usr/bin/env python3
"""Capture the TASK-22868 production-shaped Textual evidence frames."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SANDBOX = Path(tempfile.mkdtemp(prefix="chatbook-watchlists-uat-"))
for sandbox_dir in ("home", "config", "data", "cache"):
    (SANDBOX / sandbox_dir).mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(SANDBOX / "home")
os.environ["XDG_CONFIG_HOME"] = str(SANDBOX / "config")
os.environ["XDG_DATA_HOME"] = str(SANDBOX / "data")
os.environ["XDG_CACHE_HOME"] = str(SANDBOX / "cache")
os.environ["TLDW_CONFIG_PATH"] = str(SANDBOX / "config" / "config.toml")

from Tests.Skills.test_skills_library_flow import (  # noqa: E402
    _wire_empty_non_skill_services,
)
from Tests.UI.app_factory import (  # noqa: E402
    _build_test_app,
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.test_console_workbench_contract import (  # noqa: E402
    ConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector  # noqa: E402
from Tests.UI.test_library_shell import LibraryHarness  # noqa: E402
from Tests.UI.test_watchlists_destination_shell import (  # noqa: E402
    DestinationHarness,
)
from tldw_chatbook.Library.library_shell_state import (  # noqa: E402
    LIBRARY_ROW_BROWSE_SKILLS,
)
from tldw_chatbook.Skills_Interop.skill_package_inspection import (  # noqa: E402
    FRAMEWORK_MESSAGE,
    SkillPackageKind,
)
from tldw_chatbook.UI.Library_Modules.library_skill_import_controller import (  # noqa: E402
    ensure_library_skill_import_coordinator,
)
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState  # noqa: E402
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen  # noqa: E402
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import (  # noqa: E402
    ArtifactsPane,
)
from textual.widgets import Select  # noqa: E402


def _save(app, name: str) -> None:
    svg = app.export_screenshot(title=f"TASK-22868 · {name}", simplify=True)
    svg = "\n".join(line.rstrip() for line in svg.splitlines())
    (HERE / name).write_text(
        svg,
        encoding="utf-8",
    )


def _frame_name(stem: str, size: tuple[int, int]) -> str:
    return f"{stem}-{size[0]}x{size[1]}.svg"


async def _capture_console(size: tuple[int, int]) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app, model="existing-selected-model")
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        console._watchlists_operation_rows = {
            "local:watchlist_run:1": {
                "id": "local:watchlist_run:1",
                "status_detail": "completed",
                "destination": "runs",
            },
            "local:briefing:1": {
                "id": "local:briefing:1",
                "status_detail": "complete",
                "destination": "artifacts",
            },
        }
        console.set_task_resume_state(
            TaskResumeState(
                followed_watchlists_operations=(
                    "local:watchlist_run:1",
                    "local:briefing:1",
                )
            )
        )
        await pilot.pause()
        await pilot.pause()
        _save(host, _frame_name("console", size))


async def _capture_watchlists(size: tuple[int, int]) -> None:
    app = _build_test_app()
    bundles = app.watchlist_bundle_service
    database = bundles.db
    source_id = database.add_subscription(
        name="Threat feed 1",
        type="rss",
        source="https://public.example/feed-1.xml",
    )
    watchlist = bundles.create("Daily threat intelligence")
    bundles.add_source(watchlist["id"], source_id)
    briefing = database.accept_briefing(
        watchlist["id"], created_at="2026-08-28T18:00:00+00:00"
    )
    database.update_briefing(
        briefing["id"],
        status="complete",
        body_markdown="## Daily signals\n\nThree cited campaign signals are ready.",
    )
    database.set_watchlist_briefing_settings(
        watchlist["id"], briefing_cadence_seconds=86_400
    )
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        screen = host.screen_stack[-1]
        screen.apply_navigation_context(
            {
                "section": "artifacts",
                "backend": "local",
                "briefing_id": f"local:briefing:{briefing['id']}",
            }
        )
        for _ in range(250):
            await pilot.pause(0.02)
            if not screen.query(ArtifactsPane):
                continue
            pane = screen.query_one(ArtifactsPane)
            cadence_rows = screen.query("#artifacts-cadence-select")
            if (
                (pane.selected_briefing or {}).get("id") == briefing["id"]
                and cadence_rows
                and cadence_rows.first(Select).value == 86_400
            ):
                break
        else:
            cadence_rows = screen.query("#artifacts-cadence-select")
            cadence_value = (
                cadence_rows.first(Select).value if cadence_rows else "missing"
            )
            raise AssertionError(
                "exact briefing and stored 86,400-second cadence were not both "
                f"visible: briefing={(pane.selected_briefing or {}).get('id')!r}, "
                f"cadence={cadence_value!r}"
            )
        _save(app, _frame_name("watchlists", size))


async def _capture_library(size: tuple[int, int]) -> None:
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    coordinator = ensure_library_skill_import_coordinator(app)
    coordinator.update(
        row_open=True,
        path="https://example.invalid",
        status=FRAMEWORK_MESSAGE,
        in_flight=False,
        package_kind=SkillPackageKind.FRAMEWORK_REPOSITORY.value,
        recovery_actions=(
            "Choose a repository subdirectory that contains SKILL.md.",
            "Use its project instructions when that is the intended integration.",
            "Use the framework's external CLI outside Chatbook.",
            "Create a separately reviewed wrapper skill.",
        ),
        generation=7,
    )
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
    host = LibraryHarness(app, screen=screen)
    async with host.run_test(size=size) as pilot:
        await _wait_for_selector(screen, pilot, "#library-skills-import-status")
        await pilot.pause()
        _save(host, _frame_name("library-skill-classification", size))


async def main() -> None:
    try:
        for size in ((180, 50), (160, 42)):
            await _capture_console(size)
            await _capture_watchlists(size)
            await _capture_library(size)
    finally:
        drain_active_service_patches()
        drain_created_dirs()


if __name__ == "__main__":
    asyncio.run(main())
