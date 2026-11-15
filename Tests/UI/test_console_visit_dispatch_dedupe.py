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
method is patched before the app is built, so the counters observe the app's
own first Console mount rather than a replayed one.
"""

from __future__ import annotations

import pytest
from textual import events

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Console_Modules.skill import ConsoleSkillController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

# TASK-40323: `_build_test_app` loads config (`build_test_app_config` ->
# `load_settings`), which the per-test sandbox refuses with RecoveryRequired.
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.fixture()
def visit_spy(monkeypatch):
    """Count the two per-visit Console refreshes, PER screen instance.

    Per-instance, not global: ``_build_test_app`` already stands a Console
    screen up for the app's initial tab, so a global counter would conflate
    two screens' visits and could not tell "one each" from "two on one".
    """
    counts: dict[str, dict[int, int]] = {"skills": {}, "task_resume": {}}
    pushing: dict[int, int] = {}

    real_skills = ConsoleSkillController._refresh_console_skill_candidates
    real_sync = ChatScreen.sync_task_resume_state
    real_set = ChatScreen.set_task_resume_state

    async def counting_skills(self):
        counts["skills"][id(self)] = counts["skills"].get(id(self), 0) + 1
        return await real_skills(self)

    def counting_sync(self):
        # TASK-41139: a state push (a question or decision remount through
        # set_task_resume_state, 524de00360 / 11150b849d) renders its own new
        # state; only a bare sync is the per-visit refresh this file pins.
        if not pushing.get(id(self)):
            counts["task_resume"][id(self)] = (
                counts["task_resume"].get(id(self), 0) + 1
            )
        return real_sync(self)

    def pushing_set(self, task_state):
        pushing[id(self)] = pushing.get(id(self), 0) + 1
        try:
            return real_set(self, task_state)
        finally:
            pushing[id(self)] -= 1

    monkeypatch.setattr(
        ConsoleSkillController, "_refresh_console_skill_candidates", counting_skills
    )
    monkeypatch.setattr(ChatScreen, "sync_task_resume_state", counting_sync)
    monkeypatch.setattr(ChatScreen, "set_task_resume_state", pushing_set)
    return counts


async def _settled(pilot, done) -> None:
    """Wait for ``done`` (loaded hosts take seconds), then 0.5 s for repeats."""
    for _ in range(1200):
        if done():
            break
        await pilot.pause(0.05)
    for _ in range(10):
        await pilot.pause(0.05)


async def _first_console_visit(app, pilot, visit_spy) -> ChatScreen:
    """The app's own first Console visit, settled.

    TASK-41139: a ChatScreen the test pushes can land under the app's own
    initial Console on a loaded host, so it is never the visited screen.
    """

    def visited() -> bool:
        screen = app.screen
        return (
            isinstance(screen, ChatScreen)
            and screen._console_attach_reconciled
            and id(screen._skill) in visit_spy["skills"]
            and id(screen) in visit_spy["task_resume"]
        )

    await _settled(pilot, visited)
    assert visited(), "Console never finished its first visit"
    return app.screen


async def test_console_first_visit_refreshes_skills_and_task_resume_once(visit_spy):
    """AC#3: the mount and its own ScreenResume collapse into ONE of each."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _first_console_visit(app, pilot, visit_spy)

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
        screen = await _first_console_visit(app, pilot, visit_spy)
        baseline_skills = visit_spy["skills"][id(screen._skill)]
        baseline_sync = visit_spy["task_resume"][id(screen)]

        # The event Textual delivers on every return to this screen (pop of a
        # pushed screen, tab switch back). Posted directly rather than driven
        # through push/pop: this app runs several Console screens on the stack
        # at once in the test harness, so a pop does not reliably re-activate
        # THIS instance -- and the message is the seam under test either way.
        screen.post_message(events.ScreenResume())
        await _settled(
            pilot,
            lambda: visit_spy["skills"][id(screen._skill)] > baseline_skills
            and visit_spy["task_resume"][id(screen)] > baseline_sync,
        )

        assert visit_spy["skills"][id(screen._skill)] == baseline_skills + 1, (
            "Returning to Console must re-read the skill catalog: a skill may "
            "have been installed while it was suspended."
        )
        assert visit_spy["task_resume"][id(screen)] == baseline_sync + 1, (
            "Returning to Console must re-sync task-resume state."
        )
