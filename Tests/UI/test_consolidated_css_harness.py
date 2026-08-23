# test_consolidated_css_harness.py
# Description: Pins Tests/UI/consolidated_css.py's own CSS_PATH-merge behavior
# (TASK-15995).
#
# `ConsolidatedCSSApp.CSS_PATH` carries the two generated screen/modal sheets
# (TASK-15450) so a harness that pushes one of the seven `BUNDLED_SCREEN_CSS`
# modals gets its class-level CSS. But it is an ordinary class attribute, and
# ~27 real test harnesses declare their own `CSS_PATH` (most to also load the
# app bundle) -- which, absent a merge, shadows it wholesale via normal Python
# attribute lookup and drops the screen sheets. Textual's `App.__init__` also
# accepts a `css_path=` constructor kwarg that short-circuits the
# `self.CSS_PATH` class-attribute branch entirely (`css_path or
# self.CSS_PATH`), so the merge has to intercept both forms. This module pins
# both rather than relying on any of the 27 real harnesses, since none of them
# currently happens to push a `BUNDLED_SCREEN_CSS` modal (a vacuous-pass trap
# noted in the task).

from __future__ import annotations

import pytest
from textual.color import Color

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
    NoteSelectionDialog,
)


class _StyledHarness(ConsolidatedCSSApp):
    """Mirrors the real combiners: a subclass that declares its own CSS_PATH."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


@pytest.mark.asyncio
async def test_css_path_class_attr_override_still_loads_screen_sheets():
    """A subclass ``CSS_PATH`` class attribute must not drop the screen sheets.

    `NoteSelectionDialog` (one of the seven `BUNDLED_SCREEN_CSS` modals) gives
    `#note-selection-container` a fixed `width: 80` only via its screen sheet
    entry; absent that sheet the container falls back to `Container`'s own
    `width: 1fr` default and fills the whole content area instead. This is a
    computed-geometry consequence of the CSS actually applying, not merely a
    check that pushing the modal raised no exception.
    """
    app = _StyledHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog([]))
        await pilot.pause()

        container = app.screen.query_one("#note-selection-container")
        assert container.region.width == 80, (
            "NoteSelectionDialog's BUNDLED_SCREEN_CSS width:80 rule for "
            "#note-selection-container did not apply (region.width="
            f"{container.region.width}) -- a subclass CSS_PATH class "
            "attribute override dropped ConsolidatedCSSApp's screen sheets"
        )


@pytest.mark.asyncio
async def test_css_path_kwarg_override_loads_screen_sheets_and_keeps_own_entry(
    tmp_path,
):
    """A ``css_path=`` constructor kwarg override must survive the merge too.

    Textual's ``App.__init__`` resolves ``css_path or self.CSS_PATH`` -- a
    kwarg short-circuits the class-attribute branch entirely, so the merge
    has to intercept it independently of a subclass's ``CSS_PATH`` attribute.
    None of the real harnesses use this form today, but the mechanism must
    still compose with it. Asserts both halves: the harness's own supplied
    stylesheet still applies, and the screen sheets are still merged in
    alongside it.
    """
    custom_css = tmp_path / "harness_only.tcss"
    custom_css.write_text("#note-search-input { background: red; }\n", encoding="utf-8")

    app = ConsolidatedCSSApp(css_path=str(custom_css))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog([]))
        await pilot.pause()

        search_input = app.screen.query_one("#note-search-input")
        assert search_input.styles.background == Color.parse("red"), (
            "the harness's own css_path= entry did not apply -- the merge "
            "must not replace it, only bracket it with the screen sheets"
        )

        container = app.screen.query_one("#note-selection-container")
        assert container.region.width == 80, (
            "a css_path= constructor kwarg override dropped the screen "
            "sheets (region.width=" + str(container.region.width) + ")"
        )


# --- TASK-21115: dynamic first-mount vs the stale-tie-breaker parse ----------


@pytest.mark.asyncio
async def test_dynamic_first_mount_of_a_consolidated_class_keeps_its_geometry():
    """A consolidated class first-mounted AFTER boot must still get its sheet.

    Measured failure shape (TASK-21115): a bare ``Vertical`` mounts at boot,
    registering Textual's ``Vertical { width: 1fr; height: 1fr }`` at
    tie-breaker 0 (it is that widget's OWN class). A consolidated
    Vertical-subclass then first-mounts dynamically: its registration lowers
    the stored ``Vertical`` tie-breaker to -1, but Textual's ``add_source``
    does not arm a reparse for a tie-breaker change, so the parsed rules
    still carry 0 -- exactly tying the consolidated sheet's
    ``ConsoleSelectionMenu { width: auto; ... }`` (specificity (0,0,1),
    tie-breaker 0) and beating it on source order. The menu mounted
    full-screen (measured: 80x40 instead of 24x6). Compose-time mounts never
    showed it, because the widget's registration and the first parse happen
    in the same mount batch.

    ``TieAwareStylesheet`` (used by both ``TldwCli`` and
    ``ConsolidatedCSSApp``) closes that window by treating a lowered
    tie-breaker as a CSS change. Born red against the plain ``Stylesheet``.
    """
    from textual.containers import Vertical

    from tldw_chatbook.Widgets.Console.console_selection_menu import (
        ConsoleSelectionMenu,
    )

    class _DynamicMountApp(ConsolidatedCSSApp):
        def compose(self):
            yield Vertical(id="boot-time-vertical")

    app = _DynamicMountApp()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        await app.mount(ConsoleSelectionMenu(screen_x=2, screen_y=10))
        await pilot.pause()
        menu = app.query_one(ConsoleSelectionMenu)
        assert str(menu.styles.width) == "auto" and str(menu.styles.height) == "auto", (
            f"menu resolved {menu.styles.width}x{menu.styles.height} -- the "
            "consolidated sheet lost to a stale base-class tie-breaker on a "
            "dynamic first mount (see css/tie_aware_stylesheet.py)"
        )
        assert menu.region.width < 40 and menu.region.height < 10, (
            f"menu mounted at {menu.region} -- full-container geometry means "
            "its BUNDLED_CSS did not apply"
        )


def test_tie_aware_stylesheet_arms_reparse_when_a_tie_breaker_lowers():
    """Unit pin for the mechanism itself, independent of any app boot.

    Upstream ``Stylesheet.add_source`` keeps the LOWEST tie-breaker ever
    offered for an existing source but leaves ``_require_parse`` unset when
    lowering it -- the exact staleness the test above measures end-to-end.
    """
    from textual.css.stylesheet import Stylesheet

    from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet

    for cls, expect_armed in ((Stylesheet, False), (TieAwareStylesheet, True)):
        sheet = cls()
        sheet.add_source("X { height: 1; }", read_from=("probe", "X"), tie_breaker=0)
        sheet._require_parse = False  # simulate the post-parse steady state
        sheet.add_source("X { height: 1; }", read_from=("probe", "X"), tie_breaker=-1)
        assert sheet.source[("probe", "X")].tie_breaker == -1, (
            "both classes must keep upstream's lowest-offer-wins contract"
        )
        assert sheet._require_parse is expect_armed, (
            f"{cls.__name__}: _require_parse should be {expect_armed} after a "
            "tie-breaker lowering (upstream leaves the stale parse in place; "
            "the subclass exists to arm the reparse)"
        )
