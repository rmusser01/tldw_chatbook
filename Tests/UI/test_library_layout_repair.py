"""Keep Library's wide collapse targets and usable panes during resizing."""

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.test_library_adaptive_reader_shell import _ProbeApp
from Tests.UI.test_library_media_side_by_side import (
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
)
from tldw_chatbook.Library.library_media_reader_state import MEDIA_READER_LAYOUT_PROFILE
from tldw_chatbook.UI.Library_Modules import screen_constants
from tldw_chatbook.Utils.adaptive_reader_state import (
    AdaptiveReaderLayoutPreferences,
    AdaptiveReaderLayoutProfile,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LibraryAdaptiveReaderShell,
)


PROFILES = {
    name: profile
    for name, profile in vars(screen_constants).items()
    if isinstance(profile, AdaptiveReaderLayoutProfile)
}
PROFILES["media"] = MEDIA_READER_LAYOUT_PROFILE


@pytest.mark.parametrize("profile", PROFILES.values(), ids=PROFILES.keys())
@pytest.mark.parametrize("width", [100, 120, 160, 235])
async def test_reader_keeps_usable_panes_and_five_cell_click_targets(profile, width):
    layout = resolve_adaptive_reader_layout(
        width, AdaptiveReaderLayoutPreferences(), profile
    )
    app = _ProbeApp(layout, grip_width=layout.grip_width)
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", LibraryAdaptiveReaderShell)
        assert shell.items.display, "The list can still fit beside the reader"
        if width > 120 or (width == 120 and profile.work_min_width <= 46):
            assert shell.library.display, "All three panes can still fit"
        assert shell.work.region.width >= profile.work_min_width
        assert sum(child.region.width for child in shell.children) == width
        for grip in (shell.library_grip, shell.items_grip):
            assert grip.region.width == 5
            # The outer column must be clickable, not just the arrow's cell.
            await pilot.click(grip, offset=(4, grip.region.height // 2))
        assert app.toggles == ["library", "items"]


@pytest.mark.parametrize("width", [100, 124, 160])
async def test_real_media_reader_preserves_visible_list_after_opening_item(
    width, tmp_path
):
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(160, 35)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.reader_session.loaded_id is not None,
            message="Media reader did not open",
        )
        for _ in range(3):
            await pilot.pause()
        await pilot.resize_terminal(width, 35)
        for _ in range(3):
            await pilot.pause()
        shell = screen.query_one(".library-media-route", LibraryAdaptiveReaderShell)
        Path(tmp_path, f"media-{width}.svg").write_text(host.export_screenshot())
        assert shell.items.display
        assert shell.items.region.width >= MEDIA_READER_LAYOUT_PROFILE.list_min_width
        assert shell.work.region.width >= MEDIA_READER_LAYOUT_PROFILE.work_min_width
        if width >= 120:
            assert shell.library.display
        assert shell.library_grip.region.width == shell.items_grip.region.width == 5
        for pane, grip in (
            ("library", shell.library_grip),
            ("items", shell.items_grip),
        ):
            if not getattr(shell, pane).display:
                continue
            await pilot.click(grip, offset=(4, grip.region.height // 2))
            await _wait_for_condition(
                pilot,
                lambda: not getattr(shell, pane).display,
                message=f"Click did not collapse {pane}",
            )
            grip.focus()
            # Textual ignores another activation during a Button's click effect.
            await _wait_for_condition(
                pilot,
                lambda: grip.has_focus and not grip.has_class("-active"),
                message="Collapse button did not finish its click effect",
            )
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: getattr(shell, pane).display,
                message=f"Enter did not restore {pane}",
            )
