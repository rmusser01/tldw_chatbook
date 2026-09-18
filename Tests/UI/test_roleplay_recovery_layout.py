"""Roleplay partial-save recovery remains readable inside a compact modal."""

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftRecoveryDialog,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(52, 20), (80, 24), (170, 48)])
@private_profile_test
async def test_recovery_failure_and_actions_fit_centered_frame(request, theme, size):
    """Missing frame styles must fail even when bare full-screen buttons work."""
    domains = ("character form", "character visuals", "Persona visuals", "attachments")
    app = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    app.theme = theme
    results = []
    async with app.run_test(size=size) as pilot:
        modal = RoleplayDraftRecoveryDialog(domains)
        app.push_screen(modal, results.append)
        await pilot.pause()
        frame = modal.query_one("#roleplay-draft-recovery-dialog")
        assert 0 < frame.region.width < size[0]
        assert 0 < frame.region.height < size[1]
        assert abs(frame.region.x * 2 + frame.region.width - size[0]) <= 1
        assert abs(frame.region.y * 2 + frame.region.height - size[1]) <= 1

        def assert_painted(widget, text):
            region, clip = modal._compositor.visible_widgets[widget]
            assert region.intersection(clip) == region
            assert frame.content_region.contains_region(region)
            content = widget.content_region
            painted = " ".join(
                strip.crop(content.x, content.right).text
                for strip in modal._compositor.render_strips()[
                    content.y : content.bottom
                ]
            )
            assert text in " ".join(painted.split())

        title = modal.query_one("Static")
        assert_painted(title, "Some Roleplay drafts could not be saved")
        failed = modal.query_one("#roleplay-draft-recovery-domains")
        assert_painted(failed, "Failed: " + ", ".join(domains))
        retry = modal.query_one("#roleplay-draft-retry", Button)
        stay = modal.query_one("#roleplay-draft-recovery-stay", Button)
        assert app.focused is retry
        assert_painted(retry, "Retry")
        assert_painted(stay, "Stay")
        assert (
            stay.region.right + stay.styles.margin.right == frame.content_region.right
        )
        await pilot.press("tab")
        assert app.focused is stay
        assert_painted(stay, "Stay")
        await pilot.press("enter")
        await pilot.pause()
        assert modal not in app.screen_stack and results == [None]

        modal = RoleplayDraftRecoveryDialog(domains)
        app.push_screen(modal, results.append)
        await pilot.pause()
        assert await pilot.click("#roleplay-draft-retry")
        await pilot.pause()
        assert modal not in app.screen_stack and results == [None, "retry"]

        modal = RoleplayDraftRecoveryDialog(domains)
        app.push_screen(modal, results.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert modal not in app.screen_stack and results == [None, "retry", None]
