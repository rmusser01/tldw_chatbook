"""Critique-9 shell fixes: density, narrow Escape, Conversations footer, trust.

Covers tasks 32217, 32223, 32225 and 32228 -- the shell group of the
critique-9 fix wave.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_COLLECTIONS_READER_PROFILE,
    LIBRARY_CONVERSATION_READER_PROFILE,
    LIBRARY_PROMPTS_READER_PROFILE,
    LIBRARY_SKILLS_READER_PROFILE,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LibraryAdaptiveReaderShell,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _FakeSkillsScopeService,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


pytestmark = pytest.mark.asyncio


WIDE_TEST_SIZE = (235, 52)
NARROW_TEST_SIZE = (60, 24)


def _library_host() -> LibraryHarness:
    return LibraryHarness(_build_test_app())


@pytest.mark.parametrize(
    ("row_id", "shell_id", "profile"),
    [
        ("browse-prompts", "#library-prompts-reader-shell", LIBRARY_PROMPTS_READER_PROFILE),
        ("browse-skills", "#library-skills-reader-shell", LIBRARY_SKILLS_READER_PROFILE),
        (
            "browse-collections",
            "#library-collections-reader-shell",
            LIBRARY_COLLECTIONS_READER_PROFILE,
        ),
        (
            "browse-conversations",
            "#library-conversations-reader-shell",
            LIBRARY_CONVERSATION_READER_PROFILE,
        ),
    ],
)
async def test_a_canvas_with_nothing_open_gives_its_columns_to_the_list(
    row_id: str, shell_id: str, profile
) -> None:
    """task-32217 AC#1: an empty work pane hands its columns to the list.

    The rule Media and Notes already carry (``resolve_adaptive_reader_layout``'s
    ``reader_has_item`` block): a work pane showing only "Select ... to read it
    here." keeps its own floor and nothing more. Measured at the width the
    critique-9 live review ran at.
    """
    host = _library_host()
    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{row_id}").press()
        await _wait_for_selector(screen, pilot, shell_id)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(shell_id, LibraryAdaptiveReaderShell).work.region.width
            > 0,
            message=f"{shell_id} never settled its allocation.",
        )
        shell = screen.query_one(shell_id, LibraryAdaptiveReaderShell)
        items = shell.items
        work = shell.work
        print(
            f"MEASURED {row_id}: items={items.region.width} work={work.region.width} "
            f"work_min={profile.work_min_width} shell={shell.region.width}"
        )
        assert work.region.width <= profile.work_min_width + 1, (
            items.region,
            work.region,
        )
        assert items.region.width > work.region.width, (items.region, work.region)


async def test_the_landing_hub_keeps_a_readable_measure_on_a_wide_terminal() -> None:
    """task-32217 AC#3: the landing is capped, not stretched to 190 cells."""
    host = _library_host()
    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = await _wait_for_selector(screen, pilot, "#library-landing-canvas")
        await pilot.pause()
        canvas = screen.query_one("#library-canvas")
        print(
            f"MEASURED landing: landing={landing.region.width} "
            f"canvas={canvas.region.width}"
        )
        assert landing.region.width <= 96, (landing.region, canvas.region)


async def test_the_skills_list_shows_each_row_s_trust_state() -> None:
    """task-32223: an approved and an unapproved skill no longer paint alike."""
    app = _build_test_app()
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review", "trust_status": "trusted"}],
        blocked=[{"name": "summarize", "trust_status": "quarantined_added"}],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await _wait_for_selector(screen, pilot, "#library-skill-row-code-review")
        labels = {
            str(button.label)
            for button in screen.query("#library-skills-list Button")
        }
        print(f"MEASURED skills rows: {sorted(labels)}")
        assert any("code-review · trusted" in label for label in labels), labels
        assert any("summarize · needs review" in label for label in labels), labels
