"""Full-size image viewer modal + clickable avatar box (task-1534)."""

import pytest
PILImage = pytest.importorskip("PIL.Image")
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Console.console_image_viewer_modal import (
    AvatarViewRequested,
    ClickableAvatarBox,
    ConsoleImageViewerModal,
)


class _BoxApp(App):
    def __init__(self):
        super().__init__()
        self.seen: list[AvatarViewRequested] = []

    def compose(self) -> ComposeResult:
        yield ClickableAvatarBox(id="avatar-box")

    def on_avatar_view_requested(self, message: AvatarViewRequested) -> None:
        self.seen.append(message)


@pytest.mark.asyncio
async def test_clickable_box_posts_view_request_on_click():
    """Clicking the avatar box posts AvatarViewRequested to the app."""
    app = _BoxApp()
    async with app.run_test() as pilot:
        await pilot.click("#avatar-box")
        await pilot.pause()
    assert len(app.seen) == 1


class _ViewerApp(App):
    pass


@pytest.mark.asyncio
async def test_viewer_modal_renders_image_and_escape_dismisses():
    """The viewer paints a real-sized image region and Escape closes it."""
    image = PILImage.new("RGB", (64, 64), (0, 119, 226))
    app = _ViewerApp()
    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleImageViewerModal(image, title="World RP (edited)"))
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConsoleImageViewerModal)
        body = modal.query_one("#console-image-viewer-body")
        # The mosaic fallback paints a renderable child sized well beyond
        # the 16-col avatar thumb (full-size means tens of columns here).
        assert body.children
        # Renderable != painted: the body must occupy real screen area, not
        # collapse to zero under a 100%-height default inside an auto parent.
        image_widget = modal.query_one("#console-image-viewer-image")
        assert image_widget.region.width >= 20
        assert image_widget.region.height >= 10
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ConsoleImageViewerModal)


@pytest.mark.asyncio
async def test_viewer_modal_click_dismisses():
    """Clicking anywhere in the open viewer dismisses it."""
    image = PILImage.new("RGB", (32, 32), (200, 30, 30))
    app = _ViewerApp()
    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleImageViewerModal(image, title="x"))
        await pilot.pause()
        assert isinstance(app.screen, ConsoleImageViewerModal)
        await pilot.click("#console-image-viewer-body")
        await pilot.pause()
        assert not isinstance(app.screen, ConsoleImageViewerModal)
