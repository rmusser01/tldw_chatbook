"""Compositor-level guards for the TASK-31665 minors batch.

Everything here is measured through `Screen._compositor.render_strips()` on
the real generated stylesheets, for the reason the 2026-09-05 critique's own
"background banding" finding demonstrates: reading `styles.*` cannot see what
a viewer sees. That finding reported a `#2d2d2d` band "originating in the
left rail and bleeding full-width to col 233" and blamed it for the row
secondary measuring 3.44:1 on one line and 5.24:1 on the next.

Re-measured here and live (tmux `capture-pane -e`, 235x52, textual-dark), the
mechanism is different from the report in a way that changes the fix:

* There is no stray full-width paint. The Console shell paints
  `$ds-surface-panel` (#242f38) as the backdrop for BOTH rails; a left-rail
  row's own `$surface` (#1e1e1e) ends where that row's widget ends (cols
  ~20-32 at 235x52) and the backdrop simply continues to the last content
  column. That continuous run IS the reported "band".
* Rail rows really are split across two backgrounds -- but by their own
  controls: `Button`/`Input` inside a row carry Textual's stock
  `background: $surface`, so e.g. the "Environment ▾" header paints #242f38
  for the title and #1e1e1e under the chevron. That is a deliberate control
  affordance (a control should read as a control), so it is documented as
  intended -- and this module holds the contrast implication that had to be
  resolved with it: every string a rail row paints must clear AA against
  whichever of the two it actually lands on.
* The 3.44 / 5.24 pair was never two backgrounds. It was two FOREGROUNDS on
  one: `.console-inspector-section-row-secondary` carried
  `text-style: dim` on top of the already-muted `$ds-text-muted`, painting
  #7a8086 (3.42:1) where every other muted string in the same rail painted
  #a7abaf (5.91:1).
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import app_css_text
from Tests.UI.test_console_environment_wiring import _snapshot
from Tests.UI.test_console_inspector_focus_carriers import (
    SUPPORTED_SIZES,
    FocusCarrierHarness,
    _contrast,
    _open_rail,
)
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionRow,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile


#: WCAG AA for body text.
AA_TEXT = 4.5


def _painted_strings(app, region):
    """Yield ``(text, fg, bg)`` for every non-blank run inside ``region``.

    Args:
        app: The running app (its screen's compositor is sampled).
        region: Screen region to restrict the sample to.

    Yields:
        One tuple per painted, non-whitespace segment that carries both a
        foreground and a background colour.
    """
    strips = app.screen._compositor.render_strips()
    for y in range(region.y, min(region.bottom, len(strips))):
        x = 0
        for segment in strips[y]:
            width = len(segment.text)
            if (
                x + width > region.x
                and x < region.right
                and segment.text.strip()
                and segment.style is not None
                and segment.style.color is not None
                and segment.style.bgcolor is not None
            ):
                yield (
                    segment.text.strip(),
                    segment.style.color.get_truecolor(),
                    segment.style.bgcolor.get_truecolor(),
                )
            x += width


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SUPPORTED_SIZES)
async def test_inspector_row_text_meets_aa_on_the_background_it_paints_on(size):
    """AC#1 + AC#9: no rail-row string is under AA on its ACTUAL background.

    Scoped to `ConsoleInspectorSectionRow` subtrees -- primary and secondary
    alike -- rather than the whole rail, because that is what AC#9 names and
    because a whole-rail sweep would also pick up borders and glyph
    furniture, which are non-text and answer to the 3:1 floor instead.

    The sweep is over what the compositor PAINTED, so it sees both
    backgrounds a row can land on (the shell's `$ds-surface-panel` backdrop
    and the `$surface` island under an embedded control) without this test
    having to know which is which.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console, _rail = await _open_rail(host, pilot)
        rows = [
            row
            for row in console.query(ConsoleInspectorSectionRow)
            if row.display and row.region.width and row.region.height
        ]
        assert rows, f"no inspector rows painted at {size}; the sweep proves nothing"

        failures = []
        for row in rows:
            for text, fg, bg in _painted_strings(host, row.region):
                ratio = _contrast(fg, bg)
                if ratio < AA_TEXT:
                    failures.append(
                        f"{text!r} fg={fg.hex} bg={bg.hex} {ratio:.2f}:1"
                    )
        assert not failures, (
            f"Inspect-rail row text under {AA_TEXT}:1 at {size}: {failures}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SUPPORTED_SIZES)
async def test_the_refresh_tail_is_attached_to_the_section_it_refreshes(size):
    """AC#5: no blank line between the last row and the tail button.

    The critique found "Refresh" floating with "its own margin + blank
    neighbour lines" and no visual owner. Measured on GEOMETRY, not on the
    stylesheet: `margin` is only one of several ways a gap could reappear
    (a spacer row, a section `padding-bottom`), and the user-visible
    contract is "the button's first line is the line after the section's
    last row".
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console, _rail = await _open_rail(host, pilot)
        section = console.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        tail = console.query_one("#console-inspector-section-environment-view-all")
        rows = [row for row in section.query(ConsoleInspectorSectionRow) if row.display]
        assert rows, "fixture assumption: the Environment section has rows"
        last_row_bottom = max(row.region.bottom for row in rows)
        assert tail.region.y == last_row_bottom, (
            f"the Refresh tail starts at y={tail.region.y} but the section's "
            f"last row ends at y={last_row_bottom} at {size}: "
            f"{tail.region.y - last_row_bottom} blank line(s) detach it from "
            "the data it refreshes"
        )
        assert tail.tooltip and section.title in str(tail.tooltip), (
            "the tail button's tooltip must name its scope; got "
            f"{tail.tooltip!r} for section {section.title!r}"
        )


#: The five scrollers TASK-31663's review (M6) found still shipping the
#: invisible thumb it had just fixed on `#console-inspector-rail-body`.
_THUMB_SCROLLER_IDS = (
    "console-left-rail-body",
    "console-settings-body",
    "settings-impact-pane-body",
    "library-media-viewer",
    "prompt-variables-scroll",
)


def test_no_shipped_scrollbar_thumb_is_painted_in_the_grid_line_token():
    """AC#15: the 1.01:1 thumb is gone from all five remaining scrollers.

    A stylesheet assertion, deliberately, and it is not a weaker one here:
    the DEFECT is entirely in which token the rule names -- 31663 already
    measured `$ds-grid-line` on `$ds-surface-panel` at 1.01:1 through the
    compositor, and `$ds-text-muted` at the same tracks in the same run.
    Re-deriving that measurement five more times would pin the theme, not
    the rules; what can still regress is a rule quietly going back to
    `$ds-grid-line`, and that is what this reads.
    """

    stylesheet = app_css_text()
    offenders = []
    for scroller_id in _THUMB_SCROLLER_IDS:
        start = stylesheet.find(f"#{scroller_id} {{")
        if start == -1:
            start = stylesheet.find(f"#{scroller_id},")
        assert start != -1, f"#{scroller_id} has no rule in the bundle"
        block = stylesheet[start : stylesheet.find("}", start)]
        for declaration in ("scrollbar-color", "scrollbar-color-hover"):
            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{declaration}:") and "$ds-grid-line" in stripped:
                    offenders.append(f"#{scroller_id}: {stripped}")
    assert not offenders, (
        "these scrollbars still paint their thumb in $ds-grid-line, which "
        f"measures 1.01:1 on a $ds-surface-panel track: {offenders}"
    )
