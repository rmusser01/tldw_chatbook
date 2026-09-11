"""Artifacts screen share wiring tests."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
from tldw_chatbook.Web_Server.artifact_share import ArtifactShareController, ShareStatus

pytestmark = pytest.mark.ui


@asynccontextmanager
async def _open_artifacts(app, *, size=(160, 50)):
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        yield screen, pilot


async def _wait_until(pilot, predicate, *, attempts: int = 100, what: str = "state"):
    """Bounded wait, mirroring test_artifacts_screen_reports.py's helper.

    The share dialog is pushed from a thread worker via ``call_from_thread``,
    so a single ``pilot.pause()`` can return before the listing finishes.
    """
    for _ in range(attempts):
        await pilot.pause(0.05)
        if predicate():
            return
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def stub_controller():
    class _Stub:
        def __init__(self):
            self.status = None
            self.stopped = 0

        def stop_share(self):
            self.stopped += 1

    return _Stub()


async def test_share_button_and_binding_open_dialog(tmp_path):
    app = _build_test_app()
    async with _open_artifacts(app) as (screen, pilot):
        assert screen.query_one("#artifacts-share")
        await pilot.click("#artifacts-share")
        # `screen.app` is the running app (the DestinationHarness); the
        # factory-built TldwCli never runs, so its screen_stack stays empty.
        await _wait_until(
            pilot,
            lambda: any(
                isinstance(s, ArtifactShareDialog) for s in screen.app.screen_stack
            ),
            what="share dialog",
        )


async def test_share_dialog_publication_guard_invalidated_on_unmount(tmp_path):
    """Qodo #13: unmount bumps the dialog-open generation, so an in-flight
    listing can no longer push the dialog (or fire error notifies) onto an
    app the screen no longer belongs to."""
    app = _build_test_app()
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        stale = screen._share_dialog_generation
        assert screen._share_dialog_publish_allowed(stale) is True

        await pilot.app.pop_screen()  # real unmount
        await pilot.pause()

        assert screen._share_dialog_generation != stale
        assert screen._share_dialog_publish_allowed(stale) is False
        # the unmounted flag dominates even a "current" generation number
        assert screen._share_dialog_publish_allowed(stale + 1) is False


async def test_banner_reflects_active_share(tmp_path, stub_controller):
    app = _build_test_app()
    async with _open_artifacts(app) as (screen, pilot):
        app.artifact_share_controller = stub_controller
        stub_controller.status = ShareStatus(
            share_name="Kit",
            urls=("http://127.0.0.1:8123",),
            artifact_count=3,
            share_dir=Path(tmp_path),
        )
        screen._render_share_banner()
        banner = screen.query_one("#artifacts-share-status")
        assert "http://127.0.0.1:8123" in str(banner.renderable)
        assert screen.query_one("#artifacts-share-stop").display
        stub_controller.status = None
        screen._render_share_banner()
        assert not screen.query_one("#artifacts-share-stop").display


async def test_stop_button_calls_controller(tmp_path, stub_controller):
    app = _build_test_app()
    async with _open_artifacts(app) as (screen, pilot):
        app.artifact_share_controller = stub_controller
        stub_controller.status = ShareStatus(
            share_name="Kit",
            urls=("http://127.0.0.1:8123",),
            artifact_count=1,
            share_dir=Path(tmp_path),
        )
        screen._render_share_banner()
        # `.press()` rather than `pilot.click`: the stop button was hidden at
        # compose and only just re-shown, so a mouse-path click races the
        # layout pass (observed flake in-order); every test in the sibling
        # test_artifacts_screen_reports.py presses buttons the same way.
        screen.query_one("#artifacts-share-stop", Button).press()
        await _wait_until(
            pilot, lambda: stub_controller.stopped == 1, what="stop_share call"
        )
        assert stub_controller.stopped == 1


async def test_app_shutdown_stops_share(tmp_path, stub_controller):
    app = _build_test_app()
    app.artifact_share_controller = stub_controller
    app._shutdown_artifact_share()
    assert stub_controller.stopped == 1
    app._shutdown_artifact_share()  # idempotent
    assert stub_controller.stopped == 1
