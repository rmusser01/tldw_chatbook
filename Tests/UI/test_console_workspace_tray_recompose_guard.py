"""`ConsoleWorkspaceContextTray` must recompose on change -- and only on change.

TASK-15454, re-opening a guard TASK-251 deliberately reverted.

History, from the source rather than from memory: TASK-251's brief asked for
`ConsoleRunInspector`'s `if state == self.state: return` guard here too. It was
implemented, it broke click targeting on grouped browser rows
(`test_console_workspace_conversation_search_selection_keeps_query_active` and
`..._invalidates_pending_worker` both failed with "row not found"), and it was
withdrawn before landing -- the reason is inline in `sync_state` and pinned by
`Tests/UI/test_console_tick_gating.py`. The screen-side comment in
`_sync_console_workspace_context` records the mechanism: a tray can hold state
X while its DOM shows something else (a fresh instance from a full-screen
recompose whose rows were superseded before they settled), and the
unconditional recompose was what healed that.

So the failure was never "equal states must still repaint". It was "state
equality is not evidence about the DOM". The guard added in task-15454 supplies
that evidence directly: `compose()` records the (row id, row key) sequence it
builds, and the guard skips only when the rows actually mounted still match it,
on an instance the rail has already pushed to, with no recompose latched. Row
id + row key is the identity Console click routing dispatches on, so a match
means every click target is where it was.

Measured while writing this: applying the naive full-equality guard to today's
dev and running the 309-test `test_console_native_chat_flow.py` plus
`test_console_rail_sections.py` no longer reproduces the historical failure
(only the two tick-gating pins fail). The regression that forced the revert has
been dissolved by later work -- most likely TASK-1900's non-echoing search
input and TASK-1191's fit-pass rework. That is a reason to re-guard, not a
reason to guard loosely, so the DOM check below stands regardless.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)

APP_SIZE = (160, 48)

TRAY_SELECTOR = "#console-workspace-context"
SEARCH_SELECTOR = "#console-workspace-conversation-search"


class _RecomposeCounter:
    """Count `refresh(recompose=True)` calls on every tray instance."""

    def __init__(self) -> None:
        self.calls = 0
        self._original = ConsoleWorkspaceContextTray.refresh

    def __enter__(self) -> "_RecomposeCounter":
        counter = self
        original = self._original

        def counting_refresh(tray, *args, **kwargs):
            if kwargs.get("recompose"):
                counter.calls += 1
            return original(tray, *args, **kwargs)

        ConsoleWorkspaceContextTray.refresh = counting_refresh
        return self

    def __exit__(self, *_exc) -> None:
        ConsoleWorkspaceContextTray.refresh = self._original


async def _settled_tray(host, pilot):
    """Return the Console screen and its conversations tray, fully settled."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, TRAY_SELECTOR)
    for _ in range(4):
        await pilot.pause()
    return console, console.query_one(TRAY_SELECTOR, ConsoleWorkspaceContextTray)


async def _seed_rows(console, pilot) -> tuple:
    """Type a query so the tray renders grouped browser rows, and return them."""
    search = console.query_one(SEARCH_SELECTOR, Input)
    console.on_console_workspace_conversation_search_changed(
        type("E", (), {"value": "a", "input": search, "stop": staticmethod(lambda: None)})()
    )
    await pilot.pause(0.4)
    for _ in range(3):
        await pilot.pause()
    return tuple(console.query(".console-workspace-conversation-row"))


# ---------------------------------------------------------------------------
# The guard skips a proven no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_proven_noop_sync_does_not_recompose():
    """An equal state pushed into a tray whose DOM matches it changes nothing."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, tray = await _settled_tray(host, pilot)
        assert tray._composed_row_signature is not None
        assert getattr(tray, "_console_workspace_context_synced", False) is True

        with _RecomposeCounter() as counter:
            tray.sync_state(replace(tray.state))
            assert counter.calls == 0


# ---------------------------------------------------------------------------
# ... and recomposes for every kind of change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_structural_change_still_recomposes():
    """A state whose row set differs must rebuild the rows (and click targets)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, tray = await _settled_tray(host, pilot)
        browser = tray.state.conversation_browser
        assert browser is not None

        # Drop every section: a maximal structural change.
        changed = replace(tray.state, conversation_browser=replace(browser, sections=()))
        with _RecomposeCounter() as counter:
            tray.sync_state(changed)
            assert counter.calls == 1
        await pilot.pause()
        assert not list(console.query(".console-workspace-conversation-row"))


@pytest.mark.asyncio
async def test_a_text_only_change_still_recomposes():
    """The guard is not "structure only" -- any value change repaints.

    A heading change moves no click target, but the tray renders it, so
    skipping would leave stale text on screen. The structural signature is an
    ADDITIONAL requirement on top of value equality, never a replacement for it.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        _console, tray = await _settled_tray(host, pilot)

        with _RecomposeCounter() as counter:
            tray.sync_state(replace(tray.state, heading="Changed heading"))
            assert counter.calls == 1


@pytest.mark.asyncio
async def test_a_fresh_tray_instance_always_recomposes_once():
    """TASK-344/349's one-time healing push survives the guard.

    A tray the rail has never pushed into has no evidence its DOM is settled,
    so its first sync always rebuilds -- exactly the case
    `test_console_workspace_context_fresh_tray_still_synced_mid_run` pins from
    the screen side.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        _console, tray = await _settled_tray(host, pilot)
        if hasattr(tray, "_console_workspace_context_synced"):
            del tray._console_workspace_context_synced

        with _RecomposeCounter() as counter:
            tray.sync_state(replace(tray.state))
            assert counter.calls == 1


# ---------------------------------------------------------------------------
# The regression the revert was about: state says X, DOM says otherwise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tray_whose_rows_went_missing_still_heals_and_stays_clickable():
    """THE historical click-targeting case, reproduced directly.

    The reverted guard's failure mode was: the tray's `.state` lists rows, the
    DOM does not have them, an equal state arrives, the guard skips, and the
    rows are never rebuilt -- so a click can find no row to target ("row not
    found"). Here that desync is created deliberately (the rows are pruned
    behind the tray's back, leaving `.state` untouched) and a value-equal state
    is then pushed, as a poll tick would.

    A full-equality guard skips and the rows stay gone. This guard reads the
    DOM, sees the mismatch, and rebuilds -- and the assertions below require
    the rebuilt rows to be real, addressable click targets with the same
    identities they had before, not merely "some widgets".
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, tray = await _settled_tray(host, pilot)
        rows_before = await _seed_rows(console, pilot)
        assert rows_before, "fixture must render at least one grouped browser row"
        identities_before = tuple(
            (str(row.id), str(getattr(row, "row_key", ""))) for row in rows_before
        )
        pinned_state = tray.state

        for row in rows_before:
            await row.remove()
        await pilot.pause()
        assert not list(console.query(".console-workspace-conversation-row"))
        # The desync the revert was about: state still claims the rows.
        assert tray.state is pinned_state

        with _RecomposeCounter() as counter:
            tray.sync_state(replace(pinned_state))
            assert counter.calls == 1, (
                "a tray whose DOM has lost its rows must rebuild them even "
                "though the incoming state is value-equal"
            )
        for _ in range(3):
            await pilot.pause()

        rows_after = tuple(console.query(".console-workspace-conversation-row"))
        identities_after = tuple(
            (str(row.id), str(getattr(row, "row_key", ""))) for row in rows_after
        )
        assert identities_after == identities_before
        for row in rows_after:
            assert isinstance(row, Button)
            # An addressable click target: resolvable by id from the screen.
            assert console.query_one(f"#{row.id}") is row


@pytest.mark.asyncio
async def test_the_recorded_signature_matches_the_mounted_rows():
    """The guard's two sides must be derived from the same rows.

    If `compose` recorded one thing and the DOM reader read another, the guard
    would simply never skip (safe but pointless) -- or, worse, drift into
    skipping on a mismatch it could not see. Pin the agreement.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, tray = await _settled_tray(host, pilot)
        await _seed_rows(console, pilot)

        assert tray._mounted_row_signature() == tray._composed_row_signature
        assert tray._can_skip_recompose(replace(tray.state)) is True
