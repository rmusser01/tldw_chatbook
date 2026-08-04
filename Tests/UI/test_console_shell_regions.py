"""Painted-geometry baseline for the Console shell's regions.

Written BEFORE the wave-1 extractions (spec rule 3, screen-decomposition
design). Every extraction task must keep this file green and byte-identical.
If an extraction needs this file to change, the extraction changed behaviour
-- stop and treat that as a finding.

The expectation table pins what the shell DOES at each size as of the
baseline commit, including regions that are legitimately hidden in compact
mode. It does not pin what anyone thinks it should do.

Both sizes chosen here (160x45, 235x52) sit above the shell's own
``-console-compact`` height threshold (``CONSOLE_COMPACT_HEIGHT_ROWS = 35``,
see ``chat_screen.py``) and above the right-rail width-forced-collapse
threshold (``CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150``, see
``Chat/console_rail_state.py``). Against a freshly built harness (no stored
rail preferences, no active run, right rail closed by default), every region
below is therefore hidden/hittable identically at both sizes -- that is an
observed fact about this particular id list and these two sizes, not an
assumption. ``#console-mode-bar`` is additionally hidden unconditionally at
any size: it is a legacy compatibility seam retained only for older
selectors and is composed via ``_hidden_static`` regardless of geometry
(see the comment directly above its ``compose()`` yield in chat_screen.py).
"""

from contextlib import asynccontextmanager

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

# (id, expected_at_160x45, expected_at_235x52) where expected is
# "hittable" | "hidden" -- filled from a throwaway probe run against the
# unmodified shell (see task-1 report for the raw observations).
_REGIONS: list[tuple[str, str, str]] = [
    ("#console-shell", "hittable", "hittable"),
    ("#console-left-rail", "hittable", "hittable"),
    ("#console-left-rail-body", "hittable", "hittable"),
    ("#console-main-column", "hittable", "hittable"),
    ("#console-context-rail-handle", "hidden", "hidden"),
    ("#console-inspector-rail-handle", "hittable", "hittable"),
    ("#console-control-bar", "hittable", "hittable"),
    ("#console-mode-bar", "hidden", "hidden"),
    ("#console-native-composer", "hittable", "hittable"),
    ("#console-run-inspector", "hidden", "hidden"),
]


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
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
@pytest.mark.parametrize("region_id,expect_small,expect_large", _REGIONS)
async def test_region_geometry_is_stable(region_id, expect_small, expect_large, size):
    expected = expect_small if size == (160, 45) else expect_large
    async with make_console_pilot(size=size) as pilot:
        nodes = pilot.app.screen.query(region_id)
        if expected == "hidden":
            assert not nodes or not nodes[0].display
            return
        node = nodes[0]
        assert node.display and node.region.width > 0
        hit = pilot.app.screen.get_widget_at(*node.region.center)[0]
        assert hit is node or node in hit.ancestors or hit in node.walk_children()
