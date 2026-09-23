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


# --------------------------------------------------------------------------
# Qodo review of #2799: the guard above protects the `finally:` body, but the
# worker it runs in was still launched with `exit_on_error` at its default,
# and the two lookups at the TOP of `_run_sync` sit outside the `try` -- so
# the same defect class was still live in the function this task fixed.
# --------------------------------------------------------------------------


def test_the_sync_worker_refuses_to_exit_the_app_on_error():
    """The launch site, pinned by AST so it survives edits above it.

    `ProtectKeysStep._apply_password_worker` and `SummaryStep._render_rows`
    got `exit_on_error=False` in this same task; `_run_sync` did not, and its
    own comment describes exactly that default taking the app down.
    """
    import ast
    from pathlib import Path

    from tldw_chatbook.UI.Screens.scheduling import schedules_workbench

    source = Path(schedules_workbench.__file__).read_text(encoding="utf-8")
    launches = [
        {kw.arg: getattr(kw.value, "value", None) for kw in node.keywords if kw.arg}
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_worker"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "_run_sync"
    ]

    assert launches, "the _run_sync worker launch moved"
    for kwargs in launches:
        assert kwargs.get("exit_on_error") is False, (
            "_run_sync still runs with exit_on_error defaulting to True"
        )


def test_a_sync_whose_screen_is_already_gone_still_clears_the_flag():
    """The worker's first slice can run after the user has navigated away.

    The two owner-button lookups ran BEFORE the `try:`, so a `NoMatches`
    there escaped the worker and skipped the `finally:` -- leaving
    `_sync_running` set and the Sync action refused for the rest of the
    session, which is the wedge this task exists to prevent.
    """
    screen, _posted = _screen_that_unmounts_mid_sync()
    screen._is_mounted = False  # gone before the worker's first slice

    asyncio.run(screen._run_sync())

    assert screen._sync_running is False
