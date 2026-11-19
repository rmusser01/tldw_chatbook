"""A Console rail's edge focus cue restyles only the nodes its rules reach.

TASK-33620.15.1.4. Measured live on origin/dev 3be493413a (isolated
profiles, Anthropic claude-haiku-4-5, 235x52, the Inspect rail opened with
Alt+I during a reply): ``sync_console_focus_paint`` held the UI loop a
median 47 ms on each rail open and 62 ms on each close, up to 293 ms in one
callback at load 12-16. It toggles ``console-edge-region-focused`` on the
rail with a plain ``set_class``, and Textual then restyles the rail's whole
subtree, although the only rules naming the class style the rail's own
divider and its collapse/open control.

The transcript's focus cue already restyles only what its rules can reach
(``_paint_transcript_region_focus``, TASK-33628.5.1). These tests pin the
same bound for the four edge-owned regions, and the fidelity of that bound
against every loaded rule. They count restyled nodes, not time.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from textual.css.model import CombinatorType, SelectorType
from textual.scrollbar import ScrollBar, ScrollBarCorner

from Tests.UI.test_console_shell_regions import make_console_pilot
from tldw_chatbook.UI.Console_Modules.frame import sync_console_focus_paint

# The real ChatScreen goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired); keep the collection-time
# profile, as the other mounted Console tests do.
pytestmark = pytest.mark.bootstrap_profile

_FOCUS_CLASS = "console-edge-region-focused"

#: (region id, the control the region's focus rule styles) -- the pairs
#: ``sync_console_focus_paint`` paints.
_REGIONS = (
    ("console-left-rail", "console-context-rail-collapse"),
    ("console-context-rail-handle", "console-context-rail-open"),
    ("console-right-rail", "console-inspector-rail-collapse"),
    ("console-inspector-rail-handle", "console-inspector-rail-open"),
)


@contextmanager
def _restyles(app: Any) -> Iterator[list[Any]]:
    """Record every node the stylesheet restyles inside the block."""
    applied: list[Any] = []
    stylesheet = app.stylesheet
    real = stylesheet.apply

    def apply(node: Any, *args: Any, **kwargs: Any) -> Any:
        applied.append(node)
        return real(node, *args, **kwargs)

    stylesheet.apply = apply
    try:
        yield applied
    finally:
        del stylesheet.apply


def _subject_start(selectors: list[Any]) -> int:
    """Index of the first selector in the rightmost compound."""
    start = len(selectors) - 1
    while start > 0 and selectors[start].combinator is CombinatorType.SAME:
        start -= 1
    return start


@pytest.mark.asyncio
async def test_an_edge_focus_cue_restyles_only_its_region_and_control() -> None:
    """Moving a cue onto and off a region restyles two nodes, not its subtree.

    On origin/dev each toggle restyled the whole region: 160 nodes in the
    Inspect rail and 136 in the Context rail at 160x45, and each collapsed
    handle's badge.
    """
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        outside = screen.query_one("#console-native-composer")
        sync_console_focus_paint(screen, outside)
        for region_id, control_id in _REGIONS:
            region = screen.query_one(f"#{region_id}")
            control = screen.query_one(f"#{control_id}")
            subtree = sum(1 for _ in region.walk_children())
            assert subtree, region_id
            edge = (
                "right"
                if region_id in ("console-left-rail", "console-context-rail-handle")
                else "left"
            )
            assert not region.has_class(_FOCUS_CLASS)
            resting_divider = getattr(region.styles, f"border_{edge}")
            resting_control = (control.styles.background, control.styles.color)

            for focused, painted in ((control, True), (outside, False)):
                with _restyles(pilot.app) as restyled:
                    sync_console_focus_paint(screen, focused)
                # ``Stylesheet.update_nodes`` also restyles a restyled node's
                # visible scrollbars; those belong to the two nodes it names.
                restyled = [
                    node
                    for node in restyled
                    if not (
                        isinstance(node, (ScrollBar, ScrollBarCorner))
                        and node.parent in (region, control)
                    )
                ]
                inside = {
                    id(node)
                    for node in restyled
                    if region in node.ancestors and node is not control
                }
                assert not inside, (
                    f"the {region_id} cue restyled {len(inside)} nodes inside "
                    f"it (it holds {subtree} widgets)"
                )
                assert {id(node) for node in restyled} <= {id(region), id(control)}
                # The cue itself still paints, both ways: the region's
                # divider and the control's CSS-driven colours, not only the
                # inline bold/underline the paint sets on the control.
                assert region.has_class(_FOCUS_CLASS) is painted
                assert bool(control.styles.text_style.bold) is painted
                assert bool(control.styles.text_style.underline) is painted
                divider = getattr(region.styles, f"border_{edge}")
                assert (divider != resting_divider) is painted, (region_id, divider)
                colours = (control.styles.background, control.styles.color)
                assert (colours[0] != resting_control[0]) is painted, (
                    control_id,
                    colours,
                )
                assert (colours[1] != resting_control[1]) is painted, (
                    control_id,
                    colours,
                )
            # A repeat with nothing to change restyles nothing.
            with _restyles(pilot.app) as restyled:
                sync_console_focus_paint(screen, outside)
            assert not restyled, region_id


@pytest.mark.asyncio
async def test_no_rule_styles_another_node_by_an_edge_regions_focus() -> None:
    """Fidelity pin for the scoped restyle above.

    Restyling only the region and its control is exact only while no loaded
    rule makes any other node's style depend on the region's focus class.
    Checked two ways: every loaded rule, and the computed style of every
    node in each region after a full subtree restyle.
    """
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        controls = {control_id for _region_id, control_id in _REGIONS}
        offenders = []
        named = 0
        for rule in pilot.app.stylesheet.rules:
            for selector_set in rule.selector_set:
                selectors = selector_set.selectors
                start = _subject_start(selectors)
                for selector in selectors[:start]:
                    if (
                        selector.type is SelectorType.CLASS
                        and selector.name == _FOCUS_CLASS
                    ):
                        named += 1
                        subject_ids = {
                            s.name
                            for s in selectors[start:]
                            if s.type is SelectorType.ID
                        }
                        if not subject_ids & controls:
                            offenders.append(selector_set.css)
        assert named, "no rule names the edge focus class through a descendant"
        assert not offenders, offenders

        outside = screen.query_one("#console-native-composer")
        for region_id, control_id in _REGIONS:
            region = screen.query_one(f"#{region_id}")
            control = screen.query_one(f"#{control_id}")
            for focused in (control, outside):
                sync_console_focus_paint(screen, focused)
                scoped = {
                    id(node): node.styles.css
                    for node in region.walk_children(with_self=True)
                }
                pilot.app.stylesheet.update_nodes(region.walk_children(with_self=True))
                full = {
                    id(node): node.styles.css
                    for node in region.walk_children(with_self=True)
                }
                assert full == scoped, region_id
