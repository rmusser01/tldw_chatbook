"""The TieAwareStylesheet reparse-arm counter counts (TASK-22222).

`css/tie_aware_stylesheet.py` reparses the whole ~814 KB boot bundle
whenever a stored source's tie-breaker is lowered (the TASK-21115
correctness fix). Finding 22222 asked for that cost to be instrumented:
`tie_breaker_lowering_rearm_count` is the permanent cheap counter, and a
real-boot measurement is recorded in its docstring (14 arms -> 8 actual
extra full reparses to `_ui_ready`, 2026-08-25). This test keeps the
counter honest so the recorded number stays reproducible.

Blind spots: this exercises the counter's seam directly, not a real boot --
the boot-time NUMBER in the docstring is a recorded measurement, not an
assertion (pinning it would flake on any widget-tree change; re-measure
with the probe recipe in task-22222's Implementation Notes when it matters).
The counter itself counts ARMS, not reparses paid: consecutive arms with no
`apply()` between them coalesce into one reparse, so it is an upper bound.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet

_CSS = "Vertical { width: 1fr; height: 1fr; }"
_LOCATION = ("synthetic.tcss", "synthetic")


@pytest.mark.unit
def test_lowering_a_tie_breaker_arms_a_reparse_and_counts_it() -> None:
    """A lowered tie-breaker arms the reparse AND increments the counter."""
    sheet = TieAwareStylesheet()
    assert sheet.tie_breaker_lowering_rearm_count == 0

    sheet.add_source(_CSS, read_from=_LOCATION, tie_breaker=0)
    # Force a parse so the arm below is observable as a state change.
    _ = sheet.rules
    assert sheet.tie_breaker_lowering_rearm_count == 0

    # Same source, LOWER tie-breaker: upstream keeps the minimum silently;
    # the subclass arms the reparse and the counter must record the arm.
    sheet.add_source(_CSS, read_from=_LOCATION, tie_breaker=-1)
    assert sheet.tie_breaker_lowering_rearm_count == 1
    assert sheet._require_parse, "the arm must actually force a reparse"
    assert sheet._rules_map is None, "the stale rules map must be dropped"

    # And again: the counter accumulates per arm.
    _ = sheet.rules
    sheet.add_source(_CSS, read_from=_LOCATION, tie_breaker=-2)
    assert sheet.tie_breaker_lowering_rearm_count == 2


@pytest.mark.unit
def test_non_lowering_paths_do_not_count() -> None:
    """New sources, equal offers, and HIGHER offers never touch the counter."""
    sheet = TieAwareStylesheet()
    sheet.add_source(_CSS, read_from=_LOCATION, tie_breaker=-1)  # new source
    sheet.add_source(_CSS, read_from=_LOCATION, tie_breaker=-1)  # equal
    sheet.add_source(_CSS, read_from=_LOCATION, tie_breaker=0)  # higher
    assert sheet.tie_breaker_lowering_rearm_count == 0
