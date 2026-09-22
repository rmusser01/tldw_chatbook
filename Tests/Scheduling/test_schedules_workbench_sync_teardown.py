"""TASK-32892 item 7: `_run_sync`'s `finally:` body is unprotected.

A `finally:` block is NOT covered by its own `try` statement's `except`
handlers. `SchedulesWorkbench._run_sync` re-enables the owner buttons and
refreshes the owner select from its `finally:`, so a sync that outlives the
screen (the user navigates away while the server call is in flight) raises
`NoMatches` straight out of the worker -- whose `exit_on_error` defaults to
True, i.e. the app exits -- and `_sync_running` is left set, wedging the Sync
action for the rest of the session.

Gate-free: the screen is allocated with `__new__`, and only `_run_sync`'s own
seams are driven.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from textual.css.query import NoMatches

from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)


def _screen_that_unmounts_mid_sync():
    """A workbench whose DOM disappears while `sync_now` is awaited."""
    screen = SchedulesWorkbench.__new__(SchedulesWorkbench)
    screen._is_mounted = True
    screen._sync_running = True
    posted: list[object] = []
    buttons = {
        "#scheduling-owner-local": SimpleNamespace(disabled=False),
        "#scheduling-owner-server": SimpleNamespace(disabled=False),
    }

    def _query_one(selector, _expect=None):
        if not screen._is_mounted:
            raise NoMatches(selector)
        return buttons[selector]

    screen.query_one = _query_one
    screen.post_message = posted.append
    screen._refresh_owner_select = lambda: screen.query_one(
        "#scheduling-owner-local"
    )

    class _Service:
        owner_id = "local"
        db = SimpleNamespace(get_conflicts=lambda *_a, **_k: [])

        async def sync_now(self, _owner_id):
            # The user navigates away while the server call is in flight.
            screen._is_mounted = False
            return SimpleNamespace(status="ok")

    screen._service = lambda: _Service()
    return screen, posted


def test_a_sync_that_outlives_the_screen_does_not_raise_out_of_the_worker():
    screen, _posted = _screen_that_unmounts_mid_sync()

    asyncio.run(screen._run_sync())


def test_the_sync_running_flag_always_clears():
    """Without this the Sync action stays refused until the app restarts."""
    screen, _posted = _screen_that_unmounts_mid_sync()

    try:
        asyncio.run(screen._run_sync())
    except NoMatches:
        pass

    assert screen._sync_running is False
