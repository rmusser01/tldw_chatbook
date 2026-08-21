"""Console dispatches its per-visit refreshes ONCE per visit (task-15475).

The input-latency audit (Docs/Design/2026-08-11-input-latency-audit.md) found
Console doing every per-visit refresh twice on the FIRST visit: Textual posts
``ScreenResume`` when a screen is pushed, so ``on_mount`` and the mount's own
``on_screen_resume`` both fire, and both dispatched
``_refresh_console_skill_candidates`` (a non-exclusive worker, so neither call
cancelled the other) and both synced task-resume state.

These tests pin the whole contract, not just the halved count: a LATER resume
-- returning from a pushed screen, where a skill may have been installed while
Console was suspended -- must still refresh. A dedupe that suppressed that
would be a correctness regression wearing a performance badge.

Spy shape mirrors ``test_console_agent_fleet_sync_coalescing.py``: the CLASS
method is patched before the screen is pushed, so the counters observe the
real first mount rather than a replayed one.
"""

from __future__ import annotations

import pytest
from textual import events

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Console_Modules.skill import ConsoleSkillController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def visit_spy(monkeypatch):
    """Count the two per-visit Console refreshes, PER screen instance.

    Per-instance, not global: ``_build_test_app`` already stands a Console
    screen up for the app's initial tab, so a global counter would conflate
    two screens' visits and could not tell "one each" from "two on one".
    """
    counts: dict[str, dict[int, int]] = {"skills": {}, "task_resume": {}}

    real_skills = ConsoleSkillController._refresh_console_skill_candidates
    real_sync = ChatScreen.sync_task_resume_state

    async def counting_skills(self):
        counts["skills"][id(self)] = counts["skills"].get(id(self), 0) + 1
        return await real_skills(self)

    def counting_sync(self):
        counts["task_resume"][id(self)] = counts["task_resume"].get(id(self), 0) + 1
        return real_sync(self)

    monkeypatch.setattr(
        ConsoleSkillController, "_refresh_console_skill_candidates", counting_skills
    )
    monkeypatch.setattr(ChatScreen, "sync_task_resume_state", counting_sync)
    return counts


async def test_console_first_visit_refreshes_skills_and_task_resume_once(visit_spy):
    """AC#3: the mount and its own ScreenResume collapse into ONE of each."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        # Long enough for on_mount's 0.05s task-resume timer to fire.
        for _ in range(10):
            await pilot.pause(0.05)

        skill_count = visit_spy["skills"][id(screen._skill)]
        task_resume_count = visit_spy["task_resume"][id(screen)]
        assert skill_count == 1, (
            "Console dispatched the skill-candidate refresh "
            f"{skill_count}x for one visit."
        )
        assert task_resume_count == 1, (
            f"Console synced task-resume state {task_resume_count}x for one visit."
        )


async def test_console_later_resume_still_refreshes(visit_spy):
    """The dedupe is per-visit, not permanent: a real suspend/resume refreshes."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        for _ in range(10):
            await pilot.pause(0.05)
        baseline_skills = visit_spy["skills"][id(screen._skill)]
        baseline_sync = visit_spy["task_resume"][id(screen)]

        # The event Textual delivers on every return to this screen (pop of a
        # pushed screen, tab switch back). Posted directly rather than driven
        # through push/pop: this app runs several Console screens on the stack
        # at once in the test harness, so a pop does not reliably re-activate
        # THIS instance -- and the message is the seam under test either way.
        screen.post_message(events.ScreenResume())
        for _ in range(10):
            await pilot.pause(0.05)

        assert visit_spy["skills"][id(screen._skill)] == baseline_skills + 1, (
            "Returning to Console must re-read the skill catalog: a skill may "
            "have been installed while it was suspended."
        )
        assert visit_spy["task_resume"][id(screen)] == baseline_sync + 1, (
            "Returning to Console must re-sync task-resume state."
        )
