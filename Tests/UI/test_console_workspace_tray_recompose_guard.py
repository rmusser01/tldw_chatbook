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
    ConsoleWorkspaceStatusPair,
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


# ---------------------------------------------------------------------------
# The DOM evidence must not be vacuous for the row-less projections
# ---------------------------------------------------------------------------


def _fixed_projection_signature_is_complete(tray) -> None:
    """Assert every control a fixed projection MOUNTED is in its signature.

    Review round 1 caught that the row half of the DOM evidence is vacuous for
    `#console-workspaces-context` (TASK-23199 retired its Sessions peer): neither
    builds grouped-browser rows, so their row signature is `()` and matches
    anything, degenerating the guard toward the reverted full-equality shape.
    The fix records their fixed controls too — and this is the tripwire that
    keeps it honest, so that a future dynamic control added to either
    projection cannot silently sit outside the evidence.

    `ConsoleWorkspaceStatusPair` composes its own label/value Statics; that
    subtree belongs to the pair, not to this tray's signature, so it is
    excluded — the ONE deliberate exemption. A new composed sub-component
    would have to be added here consciously.
    """
    recorded = set(tray._composed_fixed_signature or ())
    unrecorded: list[str] = []
    for node in tray.query("*"):
        node_id = str(getattr(node, "id", "") or "")
        if not node_id or node_id in recorded:
            continue
        owner = node
        owned_by_pair = False
        while owner is not None and owner is not tray:
            if isinstance(owner, ConsoleWorkspaceStatusPair):
                owned_by_pair = True
                break
            owner = owner.parent
        if not owned_by_pair:
            unrecorded.append(node_id)
    assert not unrecorded, (
        f"{tray.id}: {unrecorded} are mounted in a row-less projection but "
        "absent from `_composed_fixed_signature`, so `_can_skip_recompose` "
        "has no DOM evidence about them -- route them through "
        "`_record_composed_node`"
    )


@pytest.mark.asyncio
async def test_the_fixed_projections_record_every_control_they_build():
    """Both row-less projections carry real, non-empty DOM evidence."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, _tray = await _settled_tray(host, pilot)

        # TASK-23199 retired the Sessions tray; Workspaces is the only
        # ROW-LESS projection left (the Conversations tray builds grouped
        # browser rows, so it is not one of these).
        for selector in ("#console-workspaces-context",):
            projection = console.query_one(selector, ConsoleWorkspaceContextTray)
            # Vacuous row half -- this is exactly the review finding.
            assert projection._composed_row_signature == ()
            # ... so the fixed half must be doing the work.
            assert projection._composed_fixed_signature, (
                f"{selector} records no fixed controls, so its DOM evidence "
                "is empty and the guard degenerates to full equality"
            )
            _fixed_projection_signature_is_complete(projection)

        # The Workspaces projection's click targets specifically.
        workspaces = console.query_one(
            "#console-workspaces-context", ConsoleWorkspaceContextTray
        )
        recorded = set(workspaces._composed_fixed_signature or ())
        assert {
            "console-change-workspace",
            "console-new-workspace",
            "console-workspace-rag-scope-open",
        } <= recorded


@pytest.mark.asyncio
async def test_an_unrecorded_dynamic_control_reds_the_pin_while_the_guard_skips():
    """Mutation control: the pin is what discriminates, and it does.

    Give the Sessions projection a per-row button that does NOT go through
    `_record_composed_node` -- exactly the future change the review was
    worried about. Two things must then be true at once, and both are
    asserted: the guard happily skips (the hole is real, because the row half
    of the evidence is vacuous for this projection), and the pin above reds
    (so the hole cannot be introduced unnoticed).
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, _tray = await _settled_tray(host, pilot)
        sessions = console.query_one(
            "#console-workspaces-context", ConsoleWorkspaceContextTray
        )
        _fixed_projection_signature_is_complete(sessions)  # clean before

        # TASK-23199 retired the Sessions projection this mutation used to
        # target. Workspaces is the surviving row-less one, so the control
        # now mutates its compose instead -- the property under test (an
        # unrecorded dynamic control reds the pin while the guard skips) is
        # a property of row-less projections, not of Sessions specifically.
        original = ConsoleWorkspaceContextTray._compose_workspace_context

        def mutated(self):
            yield from original(self)
            if self.content != "workspace":
                return
            browser = self.state.conversation_browser
            sections = browser.sections if browser is not None else ()
            for index, section in enumerate(sections):
                # A dynamic, data-derived click target, deliberately NOT
                # routed through `_record_composed_node`.
                yield Button(
                    str(section.section_id),
                    id=f"unrecorded-dynamic-control-{index}",
                    classes="console-workspace-action",
                    compact=True,
                )

        ConsoleWorkspaceContextTray._compose_workspace_context = mutated
        try:
            sessions.refresh(recompose=True)
            for _ in range(4):
                await pilot.pause()
            injected = list(console.query("#console-workspaces-context Button"))
            assert injected, "the mutation must actually mount dynamic controls"

            # (i) the hole is real -- the guard skips despite the new targets
            assert sessions._can_skip_recompose(replace(sessions.state)) is True

            # (ii) ... and the pin is what catches it
            with pytest.raises(AssertionError) as caught:
                _fixed_projection_signature_is_complete(sessions)
            assert "unrecorded-dynamic-control-0" in str(caught.value)
        finally:
            ConsoleWorkspaceContextTray._compose_workspace_context = original
            sessions.refresh(recompose=True)
            for _ in range(3):
                await pilot.pause()

        _fixed_projection_signature_is_complete(sessions)  # clean after


@pytest.mark.asyncio
async def test_a_fixed_projection_whose_control_went_missing_still_heals():
    """The row-less projections now get the same self-heal the rows do.

    Prune a recorded control out of the Workspaces projection behind its back
    and push a value-equal state: before the fixed signature existed, the
    guard skipped (its row signature was `()` on both sides) and the control
    stayed gone.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, _tray = await _settled_tray(host, pilot)
        workspaces = console.query_one(
            "#console-workspaces-context", ConsoleWorkspaceContextTray
        )
        assert workspaces._can_skip_recompose(replace(workspaces.state)) is True

        await console.query_one("#console-change-workspace", Button).remove()
        await pilot.pause()
        assert not list(console.query("#console-change-workspace"))

        with _RecomposeCounter() as counter:
            workspaces.sync_state(replace(workspaces.state))
            assert counter.calls == 1, (
                "a row-less projection that lost a recorded control must "
                "rebuild it even though the incoming state is value-equal"
            )
        for _ in range(3):
            await pilot.pause()

        assert console.query_one("#console-change-workspace", Button) is not None


@pytest.mark.asyncio
async def test_an_out_of_band_mount_does_not_defeat_the_guard():
    """Extra nodes the screen mounts into a tray are not treated as drift.

    The screen mounts the transitional `#console-new-workspace-conversation`
    alias straight into the Conversations tray. Requiring exact DOM equality
    would make that tray permanently unskippable, so the reader collects only
    the ids compose recorded.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, tray = await _settled_tray(host, pilot)
        assert tray._can_skip_recompose(replace(tray.state)) is True

        await tray.mount(Button("Out of band", id="out-of-band-probe"))
        await pilot.pause()
        assert console.query_one("#out-of-band-probe", Button) is not None

        assert tray._can_skip_recompose(replace(tray.state)) is True


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

        mounted_rows, mounted_fixed = tray._mounted_signatures(
            tray._composed_fixed_signature or ()
        )
        assert mounted_rows == tray._composed_row_signature
        assert mounted_fixed == tray._composed_fixed_signature
        assert tray._can_skip_recompose(replace(tray.state)) is True
