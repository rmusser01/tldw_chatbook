"""Deterministic regression tests for `PruneSafeSelect` (TASK-1960).

The bug these pin is a race in the live app: a DOM prune landing inside a
`Select`'s own mount cascade leaves its `SelectCurrent` reporting
`is_mounted=True` with zero children, because `Widget.mount` silently
returns an empty `AwaitMount` while `_pruning` is set and
`MessagePump._pre_process`'s `finally` sets `_mounted_event` regardless.
`Select._on_mount` then reaches into children that were never mounted and
raises `NoMatches: No nodes match '#label' on SelectCurrent(...)`, which
Textual surfaces as an app-level crash.

Reproducing that *by timing* is exactly the intermittent failure TASK-1960
was filed against (`Tests/UI/test_watchlists_source_create_form.py::
test_a_source_can_be_created_end_to_end_through_the_form`). These tests
instead reconstruct the resulting DOM state directly — the state captured
from the real failure by instrumenting `App._prune`, `Widget.mount` and
`SelectCurrent.update`:

    SelectCurrent  is_mounted=True  _pruning=True  children=0
    Select         _pruning=True

and drive `_on_mount` against it. That is deterministic: no sleeps, no
retries, no dependence on which asyncio task wins.

`test_stock_select_still_raises_in_this_state` is the control. Without it
these tests could silently stop testing anything if upstream Textual ever
fixes the underlying escape hatch; if that day comes, that test fails and
tells us `PruneSafeSelect` can be retired.
"""

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Select
from textual.widgets._select import SelectCurrent

from tldw_chatbook.Widgets.prune_safe_select import PruneSafeSelect

OPTIONS = [("All", "all"), ("RSS", "rss")]


class _Host(App):
    """Mounts one Select of whichever class the test asks for."""

    def __init__(self, select_class):
        super().__init__()
        self._select_class = select_class

    def compose(self) -> ComposeResult:
        yield self._select_class(OPTIONS, value="all", allow_blank=False, id="probe")


async def _mounted_then_pruned_mid_compose(app, *, flag: str = "_pruning"):
    """Reconstruct "pruned between registration and Compose" on a live Select.

    A reconstruction, not a replay: zero children is reached by removing
    them and the pre-mount `value` by rewinding the reactive, rather than by
    a suppressed `mount()`. The resulting DOM state is equivalent for this
    crash — a mounted `SelectCurrent` with no `#label` under a flagged
    `Select` — which is what the guard is specified against.

    Args:
        flag: `"_pruning"` or `"_closing"` — the two flags the guard tests.
            Both are set by real, distinct Textual paths (`App._prune` and
            `MessagePump._close_messages`), so both need covering.
    """
    select = app.query_one("#probe", Select)
    current = select.query_one(SelectCurrent)
    # What `Widget.mount` returning an empty `AwaitMount` leaves behind:
    # a registered, "mounted" SelectCurrent that never composed `#label`.
    await current.query_children("*").remove()
    assert current.is_mounted, "the harness must keep SelectCurrent mounted"
    assert not current.query("#label"), "the harness must remove #label"
    # Rewind `value` to its pre-mount state *without* firing watchers, so the
    # `_on_mount` below is the widget's FIRST initialisation the way it is in
    # the real crash. Re-running `_on_mount` against an already-initialised
    # Select assigns the value it already holds, the reactive sees no change,
    # and `_watch_value` -- the crash site -- never runs at all.
    select.set_reactive(Select.value, Select.NULL)
    # What `App._prune` stamps on every node in its walk (or, for `_closing`,
    # what `MessagePump._close_messages` sets on the way out).
    setattr(select, flag, True)
    setattr(current, flag, True)
    return select


@pytest.mark.asyncio
async def test_prune_safe_select_survives_mount_against_a_half_composed_current():
    """AC#1/AC#2: the confirmed crash chain is a no-op, not an exception."""
    app = _Host(PruneSafeSelect)
    async with app.run_test():
        select = await _mounted_then_pruned_mid_compose(app)
        # The exact call the real crash came from:
        #   Select._on_mount -> _setup_options_renderables / _init_selected_option
        #     -> self.value = hint -> _watch_value -> SelectCurrent.update
        #     -> query_one("#label")
        select._on_mount(events.Mount())


@pytest.mark.asyncio
async def test_stock_select_still_raises_in_this_state():
    """Control: proves the harness reproduces a state that really does crash.

    If upstream Textual ever guards this, this test fails and
    `PruneSafeSelect` has become dead weight.
    """
    app = _Host(Select)
    async with app.run_test():
        select = await _mounted_then_pruned_mid_compose(app)
        with pytest.raises(NoMatches):
            select._on_mount(events.Mount())


async def _pruned_before_its_own_children_mounted(app, *, flag: str = "_pruning"):
    """The sibling shape: the prune caught the `Select` one level higher.

    When `Widget.mount` is suppressed on the `Select` itself, neither
    `SelectCurrent` nor `SelectOverlay` is ever registered. Upstream guards
    `query_one(SelectCurrent)` in `_watch_value` but nothing guards
    `query_one(SelectOverlay)` in `_setup_options_renderables`, which
    `_on_mount` calls first.
    """
    select = app.query_one("#probe", Select)
    await select.query_children("*").remove()
    assert not select.query(SelectCurrent)
    select.set_reactive(Select.value, Select.NULL)
    setattr(select, flag, True)
    return select


@pytest.mark.asyncio
async def test_prune_safe_select_survives_mount_with_no_children_at_all():
    """The `_setup_options_renderables` half of the same escape hatch."""
    app = _Host(PruneSafeSelect)
    async with app.run_test():
        select = await _pruned_before_its_own_children_mounted(app)
        select._on_mount(events.Mount())


@pytest.mark.asyncio
async def test_stock_select_still_raises_with_no_children_at_all():
    """Control for the sibling shape."""
    app = _Host(Select)
    async with app.run_test():
        select = await _pruned_before_its_own_children_mounted(app)
        with pytest.raises(NoMatches):
            select._on_mount(events.Mount())


def _clear_closing(select) -> None:
    """Undo a hand-set `_closing` across the subtree before teardown.

    `MessagePump._close_messages` early-returns when `_closing` is already
    set (`message_pump.py:530-532`), so it never enqueues the `None` sentinel
    that stops the widget's message loop — and `run_test`'s teardown then
    waits on that task forever. Real `_closing` always arrives *from*
    `_close_messages`, so this asymmetry only bites a test that sets the flag
    by hand, which is what these two do in order to exercise the flag in
    isolation from `_pruning`.
    """
    for node in select.walk_children(with_self=True):
        node._closing = False


@pytest.mark.asyncio
async def test_closing_alone_is_enough_to_disarm_the_watcher():
    """The `_closing` half of the `_watch_value` guard (review, Minor 2).

    `_closing` is a genuinely separate path — `MessagePump._close_messages`
    sets it on app shutdown and on `Widget.on_prune`, without `App._prune`
    necessarily having stamped `_pruning` on this widget. Without this test,
    deleting `or self._closing` leaves the suite green.
    """
    app = _Host(PruneSafeSelect)
    async with app.run_test():
        select = await _mounted_then_pruned_mid_compose(app, flag="_closing")
        try:
            assert not select._pruning, "this test must exercise _closing alone"
            select._on_mount(events.Mount())
        finally:
            _clear_closing(select)


@pytest.mark.asyncio
async def test_closing_alone_is_enough_to_disarm_the_options_rebuild():
    """The `_closing` half of the `_setup_options_renderables` guard."""
    app = _Host(PruneSafeSelect)
    async with app.run_test():
        select = await _pruned_before_its_own_children_mounted(app, flag="_closing")
        try:
            assert not select._pruning, "this test must exercise _closing alone"
            select._on_mount(events.Mount())
        finally:
            _clear_closing(select)


@pytest.mark.asyncio
async def test_the_guard_keeps_value_and_its_shadow_in_step():
    """Review, Minor 1: skip the DOM work, not the bookkeeping.

    `Select._watch_value`'s first statement is `self._value = value`; every
    statement after it queries a child. Returning before both left `value`
    and `_value` divergent, and `_on_mount` reads the shadow
    (`_init_selected_option(self._value)`). The guard must drop only the
    child lookups.
    """
    app = _Host(PruneSafeSelect)
    async with app.run_test():
        select = await _mounted_then_pruned_mid_compose(app)
        # Put the shadow somewhere the incoming value is NOT, so "did the
        # guard advance it?" is answerable. `_on_mount` reads the shadow
        # (`_init_selected_option(self._value)`), so `NULL` here is exactly
        # the divergence the review measured on a guarded widget.
        select._value = Select.NULL
        select.set_reactive(Select.value, Select.NULL)

        select._on_mount(events.Mount())

        assert select.value == "all", "the reactive should still have advanced"
        assert select._value == select.value, (
            f"shadow diverged: value={select.value!r} _value={select._value!r}"
        )


@pytest.mark.asyncio
async def test_prune_safe_select_is_a_normal_select_when_not_pruning():
    """The guard must not change behaviour in the ordinary case.

    Without this, "never raises" could be bought by never initialising —
    which would ship the stale-placeholder defect TASK-1960 explicitly
    refused.
    """
    app = _Host(PruneSafeSelect)
    async with app.run_test() as pilot:
        select = app.query_one("#probe", PruneSafeSelect)
        current = select.query_one(SelectCurrent)
        assert select.value == "all"
        assert str(current.query_one("#label").renderable) == "All"
        select.value = "rss"
        await pilot.pause()
        assert str(current.query_one("#label").renderable) == "RSS"


@pytest.mark.asyncio
async def test_prune_safe_select_keeps_matching_the_select_css_type_selector():
    """CSS must be unaffected: every `Select { ... }` rule still applies.

    Textual matches type selectors against every CSS-inheriting base class
    (`DOMNode._css_bases`), so the subclass keeps the stock Select styling
    the Watchlists filter strips depend on for their pinned heights.
    """
    app = _Host(PruneSafeSelect)
    async with app.run_test():
        select = app.query_one("#probe", PruneSafeSelect)
        assert "Select" in select._css_types
        assert app.query(Select), "query(Select) must still find the subclass"
