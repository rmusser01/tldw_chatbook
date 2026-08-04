"""Painted-geometry baseline for the Console shell's regions.

Written BEFORE the wave-1 extractions (spec rule 3, screen-decomposition
design). Every extraction task must keep this file green and byte-identical.
If an extraction needs this file to change, the extraction changed behaviour
-- stop and treat that as a finding.

The expectation table pins what the shell DOES at each size as of the
baseline commit, including regions that are legitimately hidden in compact
mode. It does not pin what anyone thinks it should do.

Three sizes are pinned:

- 160x45 and 235x52 both sit ABOVE the shell's own ``-console-compact``
  height threshold (``CONSOLE_COMPACT_HEIGHT_ROWS = 35``, see
  ``chat_screen.py``) and above the right-rail width-forced-collapse
  threshold (``CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150``, see
  ``Chat/console_rail_state.py``). Against a freshly built harness (no
  stored rail preferences, no active run, right rail closed by default),
  every region is hidden/hittable identically at both -- an observed fact
  about this id list and these two sizes, not an assumption.
- 120x30 crosses BOTH thresholds (30 < 35 rows, 120 < 150 columns) AND
  lands inside a third, narrower, deliberate rule:
  ``_should_open_standard_width_inspector`` auto-opens the Inspector rail
  whenever available columns fall in ``118..128`` and the Inspector already
  has a "Run recipe" row plus a companion row (Blocked impact / Next
  action / Sources / Tools / Approvals / Artifacts) -- the fresh harness's
  setup-blocked state supplies exactly that, so at 120x30 the Inspector is
  OPEN by default where it is closed at the other two sizes. This flips
  ``#console-inspector-rail-handle`` from hittable to hidden (the "open"
  handle only shows when the rail is closed) and makes
  ``#console-run-inspector`` newly present in the DOM with ``display=True``
  -- but see the "clipped" state below before assuming that means visible.

``#console-mode-bar`` is hidden unconditionally at any size: it is a legacy
compatibility seam retained only for older selectors and is composed via
``_hidden_static`` regardless of geometry (see the comment directly above
its ``compose()`` yield in chat_screen.py).

A third expectation state, ``"clipped"``, exists alongside "hittable" and
"hidden" for exactly one row: at 120x30 the auto-opened Inspector's
scrollable body (``#console-inspector-rail-body``) has a real viewport only
3 rows tall against ~28 rows of virtual content. Textual still reports a
non-empty, ``display=True`` ``.region`` for ``#console-run-inspector`` (a
child scrolled below that 3-row viewport), but that region's *unclipped*
screen coordinates coincidentally overlap unrelated, actually-painted
widgets elsewhere on screen (verified reproducible across repeated fresh
mounts: the reported center always resolves to ``ConsoleModelChip``, which
is nowhere near ``#console-run-inspector`` in the tree). So this region is
neither cleanly "hidden" (``display`` is True) nor cleanly "hittable" (its
own reported center never resolves to itself or a descendant) -- pinning it
as a fabricated "hittable" or "hidden" would misrepresent what the shell
does today. "clipped" asserts the node is mounted+displayed with a
purported region, AND that a hit-test at its own center does NOT resolve to
it or a descendant -- both halves of the observed reality.
"""

from contextlib import asynccontextmanager

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

# (id, expected_at_160x45, expected_at_235x52, expected_at_120x30) where
# expected is "hittable" | "hidden" | "clipped" -- filled from throwaway
# probes run against the unmodified shell (see task-1 report, including its
# fix-round section, for the raw observations).
_REGIONS: list[tuple[str, str, str, str]] = [
    ("#console-shell", "hittable", "hittable", "hittable"),
    ("#console-left-rail", "hittable", "hittable", "hittable"),
    ("#console-left-rail-body", "hittable", "hittable", "hittable"),
    ("#console-main-column", "hittable", "hittable", "hittable"),
    ("#console-context-rail-handle", "hidden", "hidden", "hidden"),
    ("#console-inspector-rail-handle", "hittable", "hittable", "hidden"),
    ("#console-control-bar", "hittable", "hittable", "hittable"),
    ("#console-mode-bar", "hidden", "hidden", "hidden"),
    ("#console-native-composer", "hittable", "hittable", "hittable"),
    ("#console-run-inspector", "hidden", "hidden", "clipped"),
]

_EXPECTED_BY_SIZE = {
    (160, 45): 0,
    (235, 52): 1,
    (120, 30): 2,
}


@asynccontextmanager
async def make_console_pilot(*, size):
    """Mount a fresh Console (ChatScreen) at ``size`` via the production harness.

    Mirrors the idiom used throughout ``test_console_internals_decomposition.py``
    and friends: build a fresh ``TldwCli`` with every real I/O seam faked out
    (``_build_test_app``), push a ``ChatScreen`` onto a minimal host app
    (``ConsoleHarness``), and wait for the composer -- the same "the shell is
    up" signal every other Console test in this suite waits on -- before
    handing control to the caller.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52), (120, 30)])
@pytest.mark.parametrize(
    "region_id,expect_160x45,expect_235x52,expect_120x30", _REGIONS
)
async def test_region_geometry_is_stable(
    region_id, expect_160x45, expect_235x52, expect_120x30, size
):
    expected = (expect_160x45, expect_235x52, expect_120x30)[_EXPECTED_BY_SIZE[size]]
    async with make_console_pilot(size=size) as pilot:
        nodes = pilot.app.screen.query(region_id)
        if expected == "hidden":
            # Tightened per review: a query returning zero nodes is NOT the
            # same fact as a mounted-but-display:none node, and Tasks 3/4
            # extract exactly the blocks holding these ids -- a silently
            # dropped/renamed id must fail this baseline, not sail through
            # the same branch as "legitimately hidden".
            assert len(nodes) == 1 and not nodes[0].display
            return
        if expected == "clipped":
            # See the module docstring's "clipped" paragraph: mounted and
            # displayed with a purported region, but scrolled below its
            # container's real viewport so nothing is actually painted at
            # its own reported center -- pin both halves of that fact.
            assert len(nodes) == 1
            node = nodes[0]
            assert node.display and node.region.width > 0
            hit = pilot.app.screen.get_widget_at(*node.region.center)[0]
            assert not (
                hit is node or node in hit.ancestors or hit in node.walk_children()
            )
            return
        node = nodes[0]
        assert node.display and node.region.width > 0
        hit = pilot.app.screen.get_widget_at(*node.region.center)[0]
        assert hit is node or node in hit.ancestors or hit in node.walk_children()
