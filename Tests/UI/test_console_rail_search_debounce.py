"""The Console rail conversation search must do its DB work behind its debounce.

TASK-15454. `#console-workspace-conversation-search`'s `Input.Changed` handler
has had a 0.2 s debounce since it was written -- but only the persisted-row
worker was ever behind it. In front of the timer, once per KEYSTROKE, it ran:

    _invalidate_console_persisted_rows_cache()   drops the TASK-251 TTL cache
    _native_console_browser_rows()               workspace labels + starred ids
    _membership_console_browser_rows()           labels + starred ids +
                                                 list_workspace_conversations
                                                 once per workspace
    _sync_console_workspace_context()            all of the above AGAIN, plus
                                                 the persisted-row chain (its
                                                 cache having just been
                                                 dropped) and a recompose of
                                                 up to three tray instances

`_console_browser_workspace_records()` also calls `ensure_default_workspace()`,
which can open a WRITE transaction, and Workspace_DB is DELETE-journalled --
so a keystroke could block on another connection's lock.

These tests call the handler synchronously and assert on the state of the world
*before yielding to the event loop at all*, so nothing else can have run in
between. Each "zero" is paired with a control that lets the debounce fire and
requires the same counter to move -- a handler that had simply stopped
searching would pass the first half and fail the second.

Part 4 (folded in from task-15452's review) covers the composer memo: the
draft-edit keystroke path calls `_console_composer_or_none()` twice, and it
used to walk the whole Console DOM each time.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.dom import DOMNode
from textual.widgets import Input

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)

APP_SIZE = (160, 48)

SEARCH_SELECTOR = "#console-workspace-conversation-search"


class _KeystrokeWorkCounter:
    """Count the DB-touching seams and tray recomposes a keystroke provokes.

    The registry seams are patched on their CLASS, so an indirect call from
    anywhere inside the screen still counts. `ConsoleWorkspaceContextTray.
    refresh` counts the tray rebuild for all three mounted projections.
    """

    _REGISTRY_METHODS = (
        "ensure_default_workspace",
        "list_workspaces",
        "list_workspace_conversations",
    )

    def __init__(self, console) -> None:
        self._console = console
        self._service = console.app_instance.workspace_registry_service
        self._patched: list[tuple[type, str, object]] = []
        self.registry_calls = 0
        self.tray_refreshes = 0
        self.context_syncs = 0

    def __enter__(self) -> "_KeystrokeWorkCounter":
        counter = self
        service_type = type(self._service)
        for name in self._REGISTRY_METHODS:
            original = getattr(service_type, name, None)
            if original is None:
                continue

            def make(original=original):
                def counting(*args, **kwargs):
                    counter.registry_calls += 1
                    return original(*args, **kwargs)

                return counting

            setattr(service_type, name, make())
            self._patched.append((service_type, name, original))

        original_refresh = ConsoleWorkspaceContextTray.refresh

        def counting_refresh(tray, *args, **kwargs):
            if kwargs.get("recompose"):
                counter.tray_refreshes += 1
            return original_refresh(tray, *args, **kwargs)

        ConsoleWorkspaceContextTray.refresh = counting_refresh
        self._patched.append((ConsoleWorkspaceContextTray, "refresh", original_refresh))

        original_sync = type(self._console)._sync_console_workspace_context

        def counting_sync(screen, *args, **kwargs):
            counter.context_syncs += 1
            return original_sync(screen, *args, **kwargs)

        type(self._console)._sync_console_workspace_context = counting_sync
        self._patched.append(
            (type(self._console), "_sync_console_workspace_context", original_sync)
        )
        return self

    def __exit__(self, *_exc) -> None:
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched.clear()

    @property
    def total(self) -> int:
        return self.registry_calls + self.tray_refreshes + self.context_syncs


def _changed_event(value: str, search_input: Input):
    """Build the `Input.Changed` payload the handler reads."""
    return SimpleNamespace(
        value=value,
        input=search_input,
        stop=lambda: None,
    )


async def _mounted_rail_search(host, pilot):
    """Return the settled Console screen and its rail search input."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, SEARCH_SELECTOR)
    # Let the boot-time sync burst finish so the counters below only ever see
    # work this test provoked.
    for _ in range(4):
        await pilot.pause()
    return console, console.query_one(SEARCH_SELECTOR, Input)


# ---------------------------------------------------------------------------
# 1. The keystroke path itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_keystroke_does_no_db_work_and_no_tray_recompose():
    """The synchronous part of a keystroke must touch nothing expensive."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search_input = await _mounted_rail_search(host, pilot)

        with _KeystrokeWorkCounter(console) as counter:
            console.on_console_workspace_conversation_search_changed(
                _changed_event("a", search_input)
            )
            # Deliberately NO await here: the assertions describe the state of
            # the world at the instant the handler returned.
            assert counter.registry_calls == 0
            assert counter.tray_refreshes == 0
            assert counter.context_syncs == 0

        assert console._console_conversation_browser_query == "a"
        assert console._console_conversation_browser_search_timer is not None


@pytest.mark.asyncio
async def test_the_debounced_pass_still_does_that_work():
    """Control for the test above: the same counters move once the timer fires.

    Without this, "zero work per keystroke" would also pass against a handler
    that had stopped searching altogether.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search_input = await _mounted_rail_search(host, pilot)

        with _KeystrokeWorkCounter(console) as counter:
            console.on_console_workspace_conversation_search_changed(
                _changed_event("a", search_input)
            )
            await pilot.pause(0.35)
            assert counter.context_syncs > 0
            assert counter.registry_calls > 0


@pytest.mark.asyncio
async def test_a_burst_of_keystrokes_runs_the_debounced_pass_once():
    """Six characters typed inside the debounce window search once, for "alpha6".

    This is the point of the change: pre-task-15454 each of these six calls
    ran the full derivation chain, and only the persisted-row worker was
    spared.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search_input = await _mounted_rail_search(host, pilot)

        with _KeystrokeWorkCounter(console) as counter:
            for text in ("a", "al", "alp", "alph", "alpha", "alpha6"):
                console.on_console_workspace_conversation_search_changed(
                    _changed_event(text, search_input)
                )
            assert counter.context_syncs == 0
            await pilot.pause(0.35)
            # One debounced pass; its own worker syncs again as results land.
            # What matters is that six keystrokes did not cost six passes.
            assert counter.context_syncs <= 3

        assert console._console_conversation_browser_query == "alpha6"


@pytest.mark.asyncio
async def test_a_superseded_timer_never_searches_for_the_stale_query():
    """Cancellation-token semantics survive the move behind the timer.

    The debounced callback re-asserts the token/query contract that only the
    worker used to assert, so a timer that somehow outlives its keystroke
    cannot start a search for text the user has already replaced.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search_input = await _mounted_rail_search(host, pilot)

        console.on_console_workspace_conversation_search_changed(
            _changed_event("stale", search_input)
        )
        stale_token = console._console_conversation_browser_search_token
        console.on_console_workspace_conversation_search_changed(
            _changed_event("fresh", search_input)
        )

        with _KeystrokeWorkCounter(console) as counter:
            # Fire the superseded callback directly, as its stopped timer
            # would have.
            console._workspace._start_console_conversation_browser_search(
                "stale", stale_token
            )
            assert counter.total == 0

        assert console._console_conversation_browser_query == "fresh"


@pytest.mark.asyncio
async def test_visible_rows_never_contradict_the_search_box_mid_debounce():
    """The cheap in-memory re-filter keeps the rail honest inside the window.

    A poll tick can sync the tray between the keystroke and the debounce. The
    rows it would paint come from `_console_conversation_browser_rows`, so
    those are narrowed to the new query synchronously -- with no service call
    (asserted by the counter) because it is a pure filter over rows already in
    memory.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, search_input = await _mounted_rail_search(host, pilot)

        console.on_console_workspace_conversation_search_changed(
            _changed_event("a", search_input)
        )
        await pilot.pause(0.35)
        seeded = console._console_conversation_browser_rows

        with _KeystrokeWorkCounter(console) as counter:
            console.on_console_workspace_conversation_search_changed(
                _changed_event("zzzz-no-such-conversation", search_input)
            )
            assert counter.total == 0

        remaining = console._console_conversation_browser_rows
        assert len(remaining) <= len(seeded)
        for row in remaining:
            assert row in seeded


# ---------------------------------------------------------------------------
# 2. The composer memo (task-15452 review follow-up)
# ---------------------------------------------------------------------------


class _ComposerQueryCounter:
    """Count full-DOM `query()` walks for the composer selector."""

    def __init__(self) -> None:
        self.calls = 0
        self._original = DOMNode.query

    def __enter__(self) -> "_ComposerQueryCounter":
        counter = self
        original = self._original

        def counting_query(node, selector=None, *args, **kwargs):
            if selector == "#console-native-composer":
                counter.calls += 1
            return original(node, selector, *args, **kwargs)

        DOMNode.query = counting_query
        return self

    def __exit__(self, *_exc) -> None:
        DOMNode.query = self._original


@pytest.mark.asyncio
async def test_the_composer_lookup_is_resolved_once_then_memoized():
    """Repeat lookups must not re-walk the largest widget tree in the app."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console._console_composer_ref = None
        with _ComposerQueryCounter() as counter:
            first = console._console_composer_or_none()
            assert counter.calls == 1
            for _ in range(9):
                assert console._console_composer_or_none() is first
            assert counter.calls == 1

        assert isinstance(first, ConsoleComposerBar)


@pytest.mark.asyncio
async def test_the_composer_memo_can_never_return_a_detached_widget():
    """Mutation control: a removed composer must not survive in the memo.

    The memo revalidates on every hit rather than relying on an invalidation
    hook, so this holds even though nothing told it the widget went away.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        stale = console._console_composer_or_none()
        assert stale is not None
        assert console._console_composer_ref is stale

        await stale.remove()
        await pilot.pause()

        resolved = console._console_composer_or_none()
        assert resolved is not stale
        assert console._console_composer_ref is not stale


@pytest.mark.asyncio
async def test_the_composer_memo_follows_a_recomposed_composer():
    """After the composer is replaced, the memo resolves the NEW instance."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        original = console._console_composer_or_none()
        assert original is not None
        parent = original.parent
        await original.remove()
        replacement = ConsoleComposerBar(id="console-native-composer")
        await parent.mount(replacement)
        await pilot.pause()

        assert console._console_composer_or_none() is replacement
