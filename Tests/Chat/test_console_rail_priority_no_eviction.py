"""An automatic Inspector open must never evict a visible Context rail.

TASK-23197. A 2026-08-29 UX audit measured the Console across widths and
found an 11-column dead zone: at 117 columns the Context rail was visible
and the Inspector closed; at 118 the Context rail was gone, replaced by a
13-column stub, and the Inspector had opened itself. It returned at 129. A
one-column resize swapped which sidebar the user had, with no explanation
anywhere.

The cause is an interaction between two rules that are each reasonable
alone. ``CONSOLE_INSPECTOR_AUTO_OPEN_MIN/MAX_COLUMNS`` auto-opens the
Inspector between 118 and 128; ``resolve_console_rail_priority`` then
collapses Context whenever BOTH rails are open between 100 and 150. So the
app opened a panel the user never asked for, and paid for it by taking away
one the user was already using.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS,
    CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS,
    CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS,
    CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS,
    ConsoleRailState,
    console_auto_open_would_evict_context,
    resolve_console_rail_priority,
)


def _state(**overrides) -> ConsoleRailState:
    base = dict(
        left_open=True,
        right_open=False,
        preferred_left_open=True,
        preferred_right_open=False,
    )
    base.update(overrides)
    return ConsoleRailState(**base)


@pytest.mark.unit
@pytest.mark.parametrize(
    "columns",
    range(CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS, CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS + 1),
)
def test_auto_open_is_declined_across_the_whole_band_when_context_is_visible(columns):
    """Every column of the former dead zone must decline the automatic open."""
    assert console_auto_open_would_evict_context(_state(left_open=True), columns) is True


@pytest.mark.unit
def test_auto_open_is_allowed_when_context_is_already_closed():
    """With no Context rail to evict there is nothing to protect."""
    assert (
        console_auto_open_would_evict_context(_state(left_open=False), 120) is False
    )


@pytest.mark.unit
@pytest.mark.parametrize("columns", [None, 99, 150, 160, 200])
def test_auto_open_is_allowed_outside_the_priority_conflict_band(columns):
    """Outside 100..149 both rails coexist, so the guard must not fire."""
    assert console_auto_open_would_evict_context(_state(left_open=True), columns) is False


@pytest.mark.unit
def test_the_guard_band_matches_the_rule_it_is_protecting_against():
    """The guard must cover exactly the widths priority resolution acts on."""
    for columns in (
        CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS,
        CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS - 1,
    ):
        both_open = _state(left_open=True, right_open=True)
        assert resolve_console_rail_priority(both_open, columns).left_open is False
        assert console_auto_open_would_evict_context(_state(left_open=True), columns)


@pytest.mark.unit
def test_priority_resolution_still_applies_to_two_explicit_opens():
    """The guard covers automatic opens only; explicit ones keep the old rule.

    A user who deliberately opens both rails in compact geometry still gets
    Inspector priority -- that behaviour is unchanged, and TASK-23197 only
    stops the app doing it to them uninvited.
    """
    both_open = _state(left_open=True, right_open=True)
    resolved = resolve_console_rail_priority(both_open, 120)
    assert resolved.left_open is False
    assert resolved.right_compact_override is True


@pytest.mark.unit
def test_priority_eviction_records_that_it_was_forced():
    """An evicted rail must be distinguishable from one the user closed.

    This started as "show the reason on the 13-column stub", and the badge
    was built -- then measured. Rewriting the badge re-renders the handle,
    which replaces the focused reveal button and drops keyboard focus
    (Tests/UI/test_console_edge_rail_geometry.py caught it). Trading focus
    stability for a one-word label is a bad deal, and the label had little
    left to explain: with the automatic open now declined, eviction only
    happens right after the user opened the Inspector themselves, where the
    cause is immediate and self-evident.

    So the eviction records itself in STATE, which costs no re-render and
    keeps the distinction available to any surface that later wants it.
    """
    both_open = _state(left_open=True, right_open=True, left_badge="workspace")
    resolved = resolve_console_rail_priority(both_open, 120)

    assert resolved.left_open is False
    assert resolved.left_forced_collapsed is True, (
        "an evicted rail must record that it was forced, not merely closed"
    )


@pytest.mark.unit
def test_eviction_does_not_disturb_the_stub_badge():
    """The handle must not re-render on eviction; that is what drops focus."""
    resolved = resolve_console_rail_priority(
        _state(left_open=True, right_open=True, left_badge="workspace"), 120
    )
    assert resolved.left_badge == "workspace"


@pytest.mark.unit
def test_a_rail_that_was_not_evicted_is_not_marked_forced():
    resolved = resolve_console_rail_priority(
        _state(left_open=True, right_open=False, left_badge="workspace"), 120
    )
    assert resolved.left_badge == "workspace"
    assert resolved.left_forced_collapsed is False
