"""Tests for the THROWAWAY Image Gen (dev) demo screen (Phase 1 proof surface).

The screen's real production dependencies are `BaseAppScreen` (which needs a
back-reference to the running app + a screen_name) and Textual's live app
loop for `self.app.call_from_thread`. Mounting the full `TldwCli` app for a
unit test is expensive (it pulls in DB/service wiring across the whole app);
instead we mount the screen under a minimal bare `App` host, which is enough
for `BaseAppScreen.compose()` (nav bar + footer) and this screen's own
widgets to build and query successfully. The worker-thread generate path is
exercised directly (off the Textual event loop, like the real `@work(thread=True)`
callable is) with `run_generation` monkeypatched, so no backend is required.
"""
from __future__ import annotations

import pytest


def _tiny_png_bytes() -> bytes:
    """A real (not merely well-formed-looking) 1x1 PNG.

    The demo screen's render path decodes the result with PIL/rich_pixels
    (see `_render_result`), so the fake `run_generation` must return actual
    image bytes -- an arbitrary placeholder like `b"x"` would make PIL raise
    `UnidentifiedImageError` once the worker thread reaches the render step.
    """
    import io
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_demo_screen_lists_backends_and_generates(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    import tldw_chatbook.UI.Screens.image_gen_demo_screen as scr
    from tldw_chatbook.UI.Screens.image_gen_demo_screen import ImageGenDemoScreen

    # Stub listing + generation so the test needs no real backend.
    monkeypatch.setattr(
        scr,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )
    captured_requests = []
    png_bytes = _tiny_png_bytes()

    def fake_run_generation(req):
        captured_requests.append(req)
        return ImageGenResult(
            content=png_bytes, content_type="image/png", bytes_len=len(png_bytes)
        )

    monkeypatch.setattr(scr, "run_generation", fake_run_generation)

    from textual.app import App
    from textual.widgets import Select, Static, TextArea, Input

    class Host(App):
        def on_mount(self) -> None:
            self.push_screen(ImageGenDemoScreen(self))

    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()

        # Backend select is present and populated from the (stubbed) catalog.
        select = app.screen.query_one("#imagegen-backend", Select)
        assert select.value == "swarmui"

        # Drive a generate: fill the prompt, press Generate, wait for the
        # worker thread (@work(thread=True)) to finish (status flips to
        # "Done" once `_render_result` has decoded the image and updated the
        # status/image/meta widgets via call_from_thread) and call the
        # stubbed run_generation with a request built from the form fields.
        app.screen.query_one("#imagegen-prompt", TextArea).text = "a cat"
        app.screen.query_one("#imagegen-seed", Input).value = "42"
        await pilot.click("#imagegen-generate")

        status = app.screen.query_one("#imagegen-status", Static)
        for _ in range(50):
            await pilot.pause(0.05)
            if str(status.renderable) == "Done":
                break

    assert str(status.renderable) == "Done"
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.backend == "swarmui"
    assert req.prompt == "a cat"
    assert req.seed == 42
