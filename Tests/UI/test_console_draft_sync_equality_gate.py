"""The Console draft-edit sync must not repaint an unchanged Workbench.

TASK-15452. `ConsoleComposerBar.DraftChanged` runs
`_sync_console_workbench_actions_from_draft` on every printable keystroke.
Before the equality gate that path rebuilt and re-pushed Workbench state
unconditionally, bypassing the guard the coalesced control-bar sync has --
measured on dev (c0c4753f8), per keystroke:

    Workbench `Static.update` calls          12
    `sort_children` calls                     2
    screen `_nodes._updates` bumps            2   (evicts the query_one cache)
    `_build_console_provider_selection`       7
    `_provider_readiness_app_config`         63
    handler wall cost                     6.405 ms

`Static.update` has no equality check of its own and `NodeList._sort` bumps
the update counter on every ancestor up to the screen -- and the screen's
counter is part of the `query_one` LRU cache key, so two no-op sorts a
keystroke evict every cached `#id` lookup on the largest tree in the app.

These tests assert observable end state on the real mounted Console, and
each "zero writes" claim is paired with a control that proves the counter
can still see a write. The behavior these must NOT disturb (slash popup
open/close, guidance dismissal, Workbench readiness across all six draft
edit keys) is pinned separately and unchanged in
`Tests/UI/test_console_composer_draft_changed.py`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widget import Widget
from textual.widgets import Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.UI.Workbench.workbench_widgets import CommandStrip
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_control_bar import ConsoleControlBar

APP_SIZE = (140, 42)

#: The four Console-mounted Workbench primitives the draft path pushes into.
WORKBENCH_SELECTORS = (
    "#console-workbench-header",
    "#console-workbench-mode-strip",
    "#console-workbench-command-strip",
    "#workbench-recovery-callout",
)


async def _console(host, pilot):
    """Mount the ready Console and focus its composer."""
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    await pilot.pause()
    return console, composer


def _within(widget, roots) -> bool:
    """Return True when `widget` is one of `roots` or sits under one."""
    node = widget
    while node is not None:
        if any(node is root for root in roots):
            return True
        node = node._parent
    return False


class _WorkbenchWriteCounter:
    """Count real writes into the mounted Console Workbench primitives.

    Covers all three write shapes the four sync methods use: `Static.update`
    (header copy, mode chips, recovery copy -- Textual's `Static.update` has
    no equality check of its own), `CommandStrip._sync_button` (the action
    buttons, which are not Statics), and `Widget.sort_children` (which bumps
    the DOM version up the ancestor chain). Restored on exit.
    """

    def __init__(self, console) -> None:
        self._roots = [console.query_one(selector) for selector in WORKBENCH_SELECTORS]
        self.static_updates = 0
        self.button_syncs = 0
        self.sorts = 0
        self._original_update = Static.update
        self._original_sort = Widget.sort_children
        self._original_sync_button = CommandStrip._sync_button

    def __enter__(self) -> "_WorkbenchWriteCounter":
        counter = self
        original_update = self._original_update
        original_sort = self._original_sort
        original_sync_button = self._original_sync_button

        def counting_update(widget, *args, **kwargs):
            if _within(widget, counter._roots):
                counter.static_updates += 1
            return original_update(widget, *args, **kwargs)

        def counting_sort(widget, *args, **kwargs):
            counter.sorts += 1
            return original_sort(widget, *args, **kwargs)

        def counting_sync_button(strip, *args, **kwargs):
            if _within(strip, counter._roots):
                counter.button_syncs += 1
            return original_sync_button(strip, *args, **kwargs)

        Static.update = counting_update
        Widget.sort_children = counting_sort
        CommandStrip._sync_button = counting_sync_button
        return self

    def __exit__(self, *_exc) -> None:
        Static.update = self._original_update
        Widget.sort_children = self._original_sort
        CommandStrip._sync_button = self._original_sync_button

    @property
    def writes(self) -> int:
        return self.static_updates + self.button_syncs + self.sorts


# ---------------------------------------------------------------------------
# 1. An unchanged derived state writes nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_repeat_keystroke_writes_nothing_into_the_workbench():
    """Typing the second letter of a word must not touch the Workbench.

    The first character flips `can_send`, so it legitimately repaints. The
    second changes no derived state at all -- and used to cost 12
    `Static.update` calls plus two `sort_children` anyway.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)

        await pilot.press("a")
        for _ in range(3):
            await pilot.pause()

        with _WorkbenchWriteCounter(console) as counter:
            await pilot.press("b")
            for _ in range(3):
                await pilot.pause()

        assert composer.draft_text() == "ab"
        assert counter.static_updates == 0
        assert counter.button_syncs == 0
        assert counter.sorts == 0


@pytest.mark.asyncio
async def test_the_state_changing_keystroke_still_repaints_the_workbench():
    """Control for the test above: the counter can still see a real write.

    The first character flips the Workbench Send action from disabled to
    primary, so the command strip must still be pushed -- and be observably
    correct afterwards. Without this control, "zero writes" would also pass
    against a counter that was simply never wired up, or against a gate that
    had muted the Workbench entirely.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        send_action = console.query_one("#workbench-action-send")
        assert send_action.disabled is True

        with _WorkbenchWriteCounter(console) as counter:
            await pilot.press("a")
            for _ in range(3):
                await pilot.pause()

        assert composer.draft_text() == "a"
        assert counter.writes > 0
        assert send_action.disabled is False


@pytest.mark.asyncio
async def test_a_genuinely_new_workbench_state_still_repaints_every_static():
    """Mutation control for the widget early-outs: not a blanket mute.

    Pushes a state whose header subtitle and every mode label differ, and
    requires the Statics to be rewritten. A `sync_*` that returned early on
    a state it had never seen would pass every "zero writes" assertion above
    and fail here.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _composer = await _console(host, pilot)

        control_state = console._build_console_control_state(
            console._pending_console_launch_context
        )
        state = console._build_console_workbench_state(control_state)
        changed = replace(
            state,
            header=replace(state.header, subtitle="Changed subtitle"),
            modes=tuple(replace(mode, label=f"{mode.label}!") for mode in state.modes),
        )

        with _WorkbenchWriteCounter(console) as counter:
            console._sync_console_workbench_state(
                control_state, workbench_state=changed
            )

        # Three header Statics plus one per mode chip.
        assert counter.static_updates >= 3 + len(state.modes)


@pytest.mark.asyncio
async def test_a_repeat_keystroke_leaves_the_screen_query_cache_intact():
    """An idle keystroke must not evict the screen-wide `query_one` cache.

    `query_one` keys its LRU on `screen._nodes._updates`, and
    `NodeList.updated` walks the ancestor chain -- so one no-op
    `sort_children` on a Workbench strip invalidates every cached `#id`
    lookup on the Console, turning the next keystroke's screen-rooted
    queries into full DOM walks.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _composer = await _console(host, pilot)

        await pilot.press("a")
        for _ in range(3):
            await pilot.pause()

        console.query_one("#console-native-composer")
        version_before = console._nodes._updates
        cache_key = (version_before, "#console-native-composer", None)
        assert console._query_one_cache.get(cache_key) is not None

        await pilot.press("b")
        for _ in range(3):
            await pilot.pause()

        assert console._nodes._updates == version_before
        assert console._query_one_cache.get(cache_key) is not None


# ---------------------------------------------------------------------------
# 2. The popup stays outside the gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_gated_keystroke_still_refilters_the_command_popup():
    """The slash popup filters on the draft, which moves every keystroke.

    This is the test that fails if anyone folds
    `_sync_console_command_popup` inside the equality gate: the Workbench
    state is byte-identical between "/" and "/p" (a non-empty draft either
    way), so a gate that swallowed the popup sync too would leave every
    command listed after the user narrowed the search.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        popup = console.query_one("#console-command-popup")

        await pilot.press("/")
        for _ in range(3):
            await pilot.pause()
        assert popup.is_open is True
        labels_before = [suggestion.label for suggestion in popup._suggestions]
        assert len(labels_before) > 1
        workbench_state_before = console._last_console_workbench_state

        with _WorkbenchWriteCounter(console) as counter:
            await pilot.press("p")
            for _ in range(3):
                await pilot.pause()

        assert composer.draft_text() == "/p"
        # The gate held: no derived Workbench state moved, nothing repainted.
        assert console._last_console_workbench_state == workbench_state_before
        assert counter.writes == 0
        # ...and the popup still narrowed to the typed prefix.
        assert popup.is_open is True
        labels_after = [suggestion.label for suggestion in popup._suggestions]
        assert labels_after != labels_before
        assert len(labels_after) < len(labels_before)
        assert all(label.startswith("/p") for label in labels_after)


# ---------------------------------------------------------------------------
# 3. The gate cannot desync the control bar from the recorded last state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_draft_edit_keeps_the_control_bar_in_step_with_last_state():
    """Recording `_last_console_*` obliges the recorder to push everything.

    `_last_console_workbench_state` is what makes the next
    `_sync_console_control_bar` decide it owes no refresh. A draft path that
    recorded it after pushing only the four Workbench widgets would strand
    the control bar (which consumes `workbench_state.actions` too) on stale
    actions until some unrelated state moved. Pin the invariant: after a
    draft edit, the control bar's actions equal the recorded state's.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        control_bar = console.query_one("#console-control-bar", ConsoleControlBar)

        await pilot.press("a")
        for _ in range(3):
            await pilot.pause()

        recorded = console._last_console_workbench_state
        assert recorded is not None
        assert composer.draft_text() == "a"
        assert control_bar.actions == recorded.actions
        assert control_bar.state == console._last_console_control_state


# ---------------------------------------------------------------------------
# 4. Provider state is derived once per pass, not once per leg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_draft_sync_derives_the_provider_selection_once():
    """One draft-edit sync used to build the provider selection 7 times."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _composer = await _console(host, pilot)
        screen_type = type(console)
        original = screen_type._build_console_provider_selection_uncached
        calls = []

        def counting(self, *args, **kwargs):
            calls.append(args)
            return original(self, *args, **kwargs)

        screen_type._build_console_provider_selection_uncached = counting
        try:
            console._sync_console_workbench_actions_from_draft()
        finally:
            screen_type._build_console_provider_selection_uncached = original

        assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_derivation_memo_is_torn_down_even_when_a_leg_raises():
    """A raising leg must not leave a stale selection cached for later."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _composer = await _console(host, pilot)

        with pytest.raises(RuntimeError):
            with console._console_derivation_scope():
                console._provider_selection._build_console_provider_selection()
                assert console._console_derivation_memo
                raise RuntimeError("leg blew up")

        assert console._console_derivation_memo is None


@pytest.mark.asyncio
async def test_the_derivation_memo_is_off_outside_a_scope():
    """Outside a scope every provider lookup is live, exactly as before."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, _composer = await _console(host, pilot)
        screen_type = type(console)
        original = screen_type._build_console_provider_selection_uncached
        calls = []

        def counting(self, *args, **kwargs):
            calls.append(args)
            return original(self, *args, **kwargs)

        screen_type._build_console_provider_selection_uncached = counting
        try:
            console._provider_selection._build_console_provider_selection()
            console._provider_selection._build_console_provider_selection()
        finally:
            screen_type._build_console_provider_selection_uncached = original

        assert len(calls) == 2
