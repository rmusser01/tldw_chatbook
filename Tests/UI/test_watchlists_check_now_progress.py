"""Check now gives progress and a completion signal — TASK-2309.

UAT F19: pressing "Check now" produced ~5 seconds of dead air (no
acknowledgment, no busy state, no completion signal), and a confused second
press queued a duplicate check. Every scenario here gates the fake run
executor behind an `asyncio.Event` so the "check is still running" window is
deterministic rather than a race against a real fetch.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Button

from Tests.UI.full_app_destination_context import (
    FullAppDestinationContext as DestinationHarness,
)
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    CheckNowRequested,
    InspectorPane,
)
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RerunRunRequested
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.watchlist_normalizers import (
    build_watchlist_item_id,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)


class Notified:
    """Capture what the app told the user, since the toast itself is transient."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.options: list[dict] = []

    def __call__(self, message, *args, severity: str = "information", **kwargs) -> None:
        self.calls.append((str(message), severity))
        self.options.append(dict(kwargs))

    def messages(self, severity: str | None = None) -> list[str]:
        if severity is None:
            return [message for message, _ in self.calls]
        return [message for message, sev in self.calls if sev == severity]


def _seed_source(app, *, name: str = "Summit Route") -> int:
    db = app.local_watchlists_service._db()
    return db.add_subscription(
        name=name, type="rss", source="https://summitroute.com/blog/feed.xml"
    )


async def _open_sources(pilot, host):
    screen = host.screen_stack[-1]
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    for _ in range(40):
        await pilot.pause()
        if pane.sources:
            break
    return screen, pane


def _gated_run_executor(gate: asyncio.Event, *, items: list[dict] | None = None):
    """A `run_executor` that blocks on `gate` before returning.

    Lets a test hold a check "in flight" for exactly as long as it needs to
    press a second time / inspect the busy state, then release it
    deterministically -- no sleeps, no timing races.
    """

    async def _executor(subscription):
        await gate.wait()
        return {"items": list(items or [])}

    return _executor


def test_local_check_and_rerun_share_the_canonical_source_operation_key():
    screen = WatchlistsCollectionsScreen(_build_test_app())
    source_id = screen._check_source_id({"id": "local:subscription:5"})

    expected = build_watchlist_item_id("local", "subscription", 5)

    assert screen._check_operation_key("local", source_id) == expected
    assert screen._rerun_operation_key("local", 5) == expected


def test_server_check_and_rerun_use_distinct_source_and_job_namespaces():
    screen = WatchlistsCollectionsScreen(_build_test_app())

    source_key = screen._check_operation_key("server", 5)
    job_key = screen._rerun_operation_key("server", 5)

    assert source_key == build_watchlist_item_id("server", "watchlist_source", 5)
    assert job_key == build_watchlist_item_id("server", "watchlist_job", 5)
    assert source_key != job_key


@pytest.mark.asyncio
async def test_pressing_check_now_gives_an_immediate_toast_and_a_busy_button():
    """AC#1: acknowledgment before the worker even finishes, and a busy
    state (disabled + relabelled) that outlives the toast."""
    app = _build_test_app()
    _seed_source(app)
    notified = Notified()
    app.notify = notified

    gate = asyncio.Event()
    app.local_watchlists_service.run_executor = _gated_run_executor(gate)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)

        button = pane.query_one("#sources-check-now-button", Button)
        assert str(button.label) == "Check now"
        assert not button.disabled

        button.press()
        for _ in range(40):
            await pilot.pause()
            if notified.calls:
                break

        # The immediate acknowledgment (AC#1) -- posted before the gated
        # executor has returned anything at all.
        assert any("checking" in m.lower() for m in notified.messages()), (
            f"no immediate acknowledgment toast, got: {notified.calls!r}"
        )
        assert str(button.label) == "Checking...", (
            f"the button must relabel while busy, got: {button.label!r}"
        )
        assert button.disabled, "the button must disable while a check is running"

        # Release the gate and let the check finish.
        gate.set()
        for _ in range(60):
            await pilot.pause()
            if str(button.label) == "Check now":
                break

        assert str(button.label) == "Check now", "the busy state must clear on completion"
        assert not button.disabled
        assert any(
            "complete" in m.lower() or "started" in m.lower()
            for m in notified.messages("information")
        ), f"no completion signal, got: {notified.calls!r}"


@pytest.mark.asyncio
async def test_a_second_press_while_checking_is_refused_not_duplicated():
    """AC#2: a second activation while the same source is mid-check is
    refused with a stated toast, and never starts a second run executor
    call."""
    app = _build_test_app()
    _seed_source(app)
    notified = Notified()
    app.notify = notified

    gate = asyncio.Event()
    call_count = 0

    async def _counting_executor(subscription):
        nonlocal call_count
        call_count += 1
        await gate.wait()
        return {"items": []}

    app.local_watchlists_service.run_executor = _counting_executor

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)

        button = pane.query_one("#sources-check-now-button", Button)
        button.press()
        for _ in range(40):
            await pilot.pause()
            if call_count >= 1:
                break
        assert call_count == 1, "the precondition: the first check must have started"

        # A confused second press while it is still running. The button is
        # disabled by this point (proven by the previous test), so drive the
        # SAME message a second click would post, exactly as
        # `on_button_pressed` does, to prove the screen-level debounce holds
        # even if some other path posted it.
        screen.post_message(CheckNowRequested(pane.selected_source))
        await pilot.pause(0.3)

        assert call_count == 1, (
            "a second activation while the same source is mid-check must "
            "NOT start a second run"
        )
        assert any(
            "already checking" in m.lower() for m in notified.messages("warning")
        ), f"the refusal must be stated, not silent: {notified.calls!r}"

        gate.set()
        await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_local_check_now_blocks_rerun_for_the_same_raw_source():
    app = _build_test_app()
    _seed_source(app, name="Source [five]")
    notified = Notified()
    app.notify = notified
    gate = asyncio.Event()
    app.local_watchlists_service.run_executor = _gated_run_executor(gate)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        source = pane.sources[0]
        pane.select_source_by_id(str(source["id"]))
        await pilot.pause()
        screen._controller.launch_run = AsyncMock()

        pane.query_one("#sources-check-now-button", Button).press()
        expected_key = screen._check_operation_key("local", source["source_id"])
        for _ in range(40):
            await pilot.pause()
            if expected_key in screen._checks_in_flight:
                break

        screen.post_message(
            RerunRunRequested(
                runtime_backend="local",
                target_id=source["source_id"],
                name="Source [five]",
            )
        )
        await pilot.pause()

        screen._controller.launch_run.assert_not_awaited()
        assert notified.calls[-1] == (
            "Already checking Source [five].",
            "warning",
        )
        assert notified.options[-1].get("markup") is False

        gate.set()
        for _ in range(40):
            await pilot.pause()
            if expected_key not in screen._checks_in_flight:
                break


@pytest.mark.asyncio
async def test_local_rerun_blocks_check_now_for_the_same_raw_source():
    app = _build_test_app()
    _seed_source(app, name="Source five")
    notified = Notified()
    app.notify = notified
    started = asyncio.Event()
    release = asyncio.Event()

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        source = pane.sources[0]

        async def launch(**kwargs):
            started.set()
            await release.wait()
            return {"status": "completed"}

        screen._controller.launch_run = AsyncMock(side_effect=launch)
        screen._controller.check_now = AsyncMock()
        screen._request_runs_refresh = Mock()
        screen.post_message(
            RerunRunRequested(
                runtime_backend="local",
                target_id=source["source_id"],
                name="Source five",
            )
        )
        for _ in range(40):
            await pilot.pause()
            if started.is_set():
                break
        assert started.is_set(), "the gated Re-run must have started"

        screen.post_message(CheckNowRequested(source))
        await pilot.pause()

        screen._controller.check_now.assert_not_awaited()
        assert notified.calls[-1] == ("Already checking Source five.", "warning")
        assert notified.options[-1].get("markup") is False

        release.set()
        for _ in range(40):
            await pilot.pause()
            if screen._request_runs_refresh.call_count:
                break
        screen._controller.launch_run.assert_awaited_once_with(
            runtime_backend="local",
            source_id=source["source_id"],
            job_id=None,
        )


@pytest.mark.asyncio
async def test_a_different_source_can_still_be_checked_while_one_is_busy():
    """The debounce is per-source, not global: `run_worker` uses a named
    group instead of the old screen-wide `exclusive=True`, specifically so
    checking source B does not touch source A's in-flight run."""
    app = _build_test_app()
    _seed_source(app, name="Source A")
    _seed_source(app, name="Source B")
    app.notify = Notified()

    gate_a = asyncio.Event()
    calls: dict[str, int] = {"a": 0, "b": 0}

    async def _executor(subscription):
        # `subscription` here is the RAW db row (`db.get_subscription`), not
        # the normalized entity `pane.sources` holds -- it carries "name",
        # the actual DB column.
        name = subscription.get("name") if hasattr(subscription, "get") else None
        key = "a" if name == "Source A" else "b"
        calls[key] += 1
        if key == "a":
            await gate_a.wait()
        return {"items": []}

    app.local_watchlists_service.run_executor = _executor

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        # `normalize_local_subscription_row` surfaces the display name as
        # "title", not "name" -- `pane.sources` holds the normalized shape.
        source_a = next(s for s in pane.sources if s.get("title") == "Source A")
        source_b = next(s for s in pane.sources if s.get("title") == "Source B")

        pane.select_source_by_id(str(source_a["id"]))
        await pilot.pause(0.2)
        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(40):
            await pilot.pause()
            if calls["a"] >= 1:
                break
        assert calls["a"] == 1

        pane.select_source_by_id(str(source_b["id"]))
        await pilot.pause(0.2)
        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(40):
            await pilot.pause()
            if calls["b"] >= 1:
                break

        assert calls["b"] == 1, (
            "checking a DIFFERENT source must not be blocked by source A's "
            "in-flight check"
        )
        gate_a.set()
        await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_a_failed_check_still_clears_the_busy_state():
    """AC#1's failure half: the busy state must clear in a `finally`, so a
    raising check cannot strand the button permanently disabled."""
    app = _build_test_app()
    _seed_source(app)
    notified = Notified()
    app.notify = notified

    async def dead_host(subscription):
        raise ConnectionError("Name or service not known: summitroute.com")

    app.local_watchlists_service.run_executor = dead_host

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)

        button = pane.query_one("#sources-check-now-button", Button)
        button.press()
        for _ in range(60):
            await pilot.pause()
            if notified.messages("error"):
                break

        assert notified.messages("error"), (
            f"a failed check must still be reported as a failure: {notified.calls!r}"
        )
        for _ in range(40):
            await pilot.pause()
            if not button.disabled:
                break
        assert not button.disabled, (
            "a check that failed must not leave the button permanently disabled"
        )
        assert str(button.label) == "Check now"


@pytest.mark.asyncio
async def test_the_inspector_check_now_button_shows_the_same_busy_state():
    """Both activation sites (Sources pane, Inspector) post the identical
    `CheckNowRequested`, so both must show the same busy state -- otherwise
    the Inspector's copy stays enabled while a duplicate run is already
    refused elsewhere, inviting exactly the confused click AC#2 is about."""
    app = _build_test_app()
    _seed_source(app)
    app.notify = Notified()

    gate = asyncio.Event()
    app.local_watchlists_service.run_executor = _gated_run_executor(gate)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)

        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(40):
            await pilot.pause()
            inspector = screen.query_one(
                "#watchlists-entity-inspector", InspectorPane
            )
            try:
                inspector_button = inspector.query_one(
                    "#inspector-check-now-button", Button
                )
            except Exception:
                continue
            if str(inspector_button.label) == "Checking...":
                break

        inspector_button = screen.query_one(
            "#watchlists-entity-inspector", InspectorPane
        ).query_one("#inspector-check-now-button", Button)
        assert str(inspector_button.label) == "Checking...", (
            "the Inspector's Check now must show the same busy state as the "
            "Sources pane's own button"
        )
        assert inspector_button.disabled

        gate.set()
        for _ in range(60):
            await pilot.pause()
            inspector_button = screen.query_one(
                "#watchlists-entity-inspector", InspectorPane
            ).query_one("#inspector-check-now-button", Button)
            if str(inspector_button.label) == "Check now":
                break
        assert str(inspector_button.label) == "Check now"
        assert not inspector_button.disabled


@pytest.mark.asyncio
async def test_leaving_the_screen_mid_check_reaches_a_terminal_status_not_stuck_running():
    """C1 (batch-4 whole-branch review, CRITICAL).

    Textual's `Widget._on_unmount` cancels every worker registered on the
    widget being torn down (`self.workers.cancel_node(self)`), regardless of
    the worker's group name -- the named "wc_check_now" group only protects
    against a SECOND `run_worker` call in the same group; it does nothing
    about the screen itself going away. This app's screens are never cached
    (`app.py`'s own `_create_navigation_screen` docstring), so switching tabs
    while a check is running -- an entirely ordinary action -- reaches
    exactly this path.

    `asyncio.CancelledError` is `BaseException`, not `Exception` (Python
    >=3.8), so before this fix neither `LocalWatchlistsService.execute_run`'s
    nor `WatchlistScopeService.launch_run`'s `except Exception` guard ever
    saw it: the run row `_mark_run_started` set to `running` moments earlier
    was never transitioned to anything else, silently, forever.
    """
    app = _build_test_app()
    source_id = _seed_source(app)
    app.notify = Notified()

    gate = asyncio.Event()
    app.local_watchlists_service.run_executor = _gated_run_executor(gate)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)

        pane.query_one("#sources-check-now-button", Button).press()

        # Precondition: the run really is at `running` before we leave.
        runs: list = []
        for _ in range(60):
            await pilot.pause()
            runs = await app.local_watchlists_service.list_runs(source_id=source_id)
            if runs and runs[0]["status"] == "running":
                break
        assert runs and runs[0]["status"] == "running", (
            f"precondition not met -- the run never reached 'running': {runs!r}"
        )

        # Leave the screen the way the app actually does: unmount it, the
        # same teardown `switch_screen` performs on the outgoing screen.
        await screen.remove()
        await pilot.pause(0.3)

        # Release the gate anyway, in case anything survived the unmount --
        # the worker should already be cancelled and nothing should be
        # listening on the other side of it.
        gate.set()
        await pilot.pause(0.3)

        runs = await app.local_watchlists_service.list_runs(source_id=source_id)
        assert runs, "the run must not vanish just because the screen did"
        assert runs[0]["status"] != "running", (
            "a run must never be left 'running' after the user navigates "
            f"away; got {runs[0]!r}"
        )
        assert runs[0]["status"] == "failed"
        assert "cancel" in str(runs[0].get("error_msg") or "").lower(), (
            "the recorded reason must say the check was cancelled, not "
            f"blame the feed: {runs[0].get('error_msg')!r}"
        )

        db = app.local_watchlists_service._db()
        row = db.get_subscription(source_id)
        assert row["last_error"], (
            "the source must record that the attempt failed too, the same "
            "as any other failed check"
        )
