"""Real Generate workers may finish after cancellation; only live requests publish."""

import asyncio
import inspect
import io
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from PIL import Image
from rich_pixels import Pixels
from textual.worker import get_current_worker
from textual.widgets import Button, Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(90)]


async def _until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


def _thread_running(screen):
    code = inspect.unwrap(type(screen)._generate).__code__
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code is code and frame.f_locals.get("self") is screen:
                return True
            frame = frame.f_back
    return False


@asynccontextmanager
async def _demo(monkeypatch, *, old_error=False, hold="provider", new_error=False):
    # Built only inside a private-profile child. Keep request construction,
    # registry resolution/validation, PNG decode, and widget publication real.
    from tldw_chatbook.Image_Generation import adapter_registry, worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    from tldw_chatbook.Image_Generation.config import get_image_generation_config
    from tldw_chatbook.UI import image_gen_command_provider
    from tldw_chatbook.UI.Screens import image_gen_demo_screen as module

    entered, release = threading.Event(), threading.Event()
    queued, publish = asyncio.Event(), asyncio.Event()
    calls, decodes = {}, []
    ui_thread = threading.get_ident()
    config = replace(get_image_generation_config(), enabled_backends=["swarmui"])
    monkeypatch.setattr(adapter_registry, "get_image_generation_config", lambda: config)
    registry = adapter_registry.ImageAdapterRegistry()

    class Adapter:
        def generate(self, req):
            calls[req.prompt] = get_current_worker()
            assert threading.get_ident() != ui_thread
            if req.prompt == "OLD":
                entered.set()
                if hold == "provider":
                    assert release.wait(20), "OLD provider was never released"
            if (req.prompt == "OLD" and old_error) or (
                req.prompt == "NEW" and new_error
            ):
                raise RuntimeError(f"{req.prompt} provider failed")
            content = io.BytesIO()
            Image.new("RGB", (3, 2), "red" if req.prompt == "OLD" else "blue").save(
                content, format="PNG"
            )
            payload = content.getvalue()
            return ImageGenResult(
                content=payload, content_type="image/png", bytes_len=len(payload)
            )

    registry.register_adapter("swarmui", Adapter)
    monkeypatch.setattr(worker, "get_registry", lambda: registry)
    monkeypatch.setattr(
        module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )
    decode = module.ImageGenDemoScreen._decode_result

    def observed_decode(screen, result, req):
        assert threading.get_ident() != ui_thread
        pixels, meta = decode(screen, result, req)
        assert isinstance(pixels, Pixels)
        decodes.append(req.prompt)
        return pixels, meta

    monkeypatch.setattr(module.ImageGenDemoScreen, "_decode_result", observed_decode)
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(140, 70)) as pilot:
        # Exercise the real provider search and Hit callback, rather than
        # constructing the demo through a test-only route.
        provider = image_gen_command_provider.ImageGenCommandProvider(app.screen)
        hits = [hit async for hit in provider.search("Image Gen")]
        assert len(hits) == 1
        hits[0].command()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, module.ImageGenDemoScreen)
        original_call = app.call_from_thread

        def call_from_thread(callback, *args, **kwargs):
            if hold == "publication" and get_current_worker() is calls.get("OLD"):

                async def delayed():
                    queued.set()
                    await publish.wait()
                    return callback(*args, **kwargs)

                return original_call(delayed)
            return original_call(callback, *args, **kwargs)

        monkeypatch.setattr(app, "call_from_thread", call_from_thread)
        try:
            yield app, pilot, screen, calls, decodes, entered, release, queued, publish
        finally:
            release.set()
            publish.set()
            # A canceled Worker.wait() only witnesses the canceled asyncio
            # task. Wait for this screen's actual thread body before teardown.
            await _until(lambda: not _thread_running(screen))
            await pilot.pause()


async def _generate(pilot, screen, prompt):
    # Textual ignores a second click until Button's -active timer retires.
    await pilot.pause(0.2)
    button = screen.query_one("#imagegen-generate", Button)
    await _until(lambda: not button.has_class("-active"))
    screen.query_one("#imagegen-prompt", TextArea).text = prompt
    screen.query_one("#imagegen-seed", Input).value = "42"
    assert await pilot.click(button)


def _status(screen):
    return str(screen.query_one("#imagegen-status", Static).renderable)


def _view(screen):
    return tuple(
        screen.query_one(selector, Static).renderable
        for selector in ("#imagegen-status", "#imagegen-image", "#imagegen-meta")
    )


@pytest.mark.parametrize("old_error", [False, True], ids=["success", "error"])
@pytest.mark.parametrize("hold", ["provider", "publication"])
@private_profile_test
async def test_superseded_demo_cannot_publish_after_new_request(
    request, monkeypatch, hold, old_error
):
    async with _demo(monkeypatch, hold=hold, old_error=old_error) as (
        app,
        pilot,
        screen,
        calls,
        decodes,
        entered,
        release,
        queued,
        publish,
    ):
        await _generate(pilot, screen, "OLD")
        await _until(entered.is_set)
        if hold == "publication":
            await _until(queued.is_set)
        old = calls["OLD"]
        await _generate(pilot, screen, "NEW")
        await _until(lambda: "NEW" in calls)
        assert old.is_cancelled, "exclusive NEW admission must cancel actual OLD Worker"
        if hold == "provider":
            await _until(lambda: _status(screen) == "Done")
            assert "prompt='NEW'" in str(_view(screen)[2])
            newest = _view(screen)
        else:
            # Release the already queued OLD callback after NEW admission,
            # while NEW is still the only request allowed to publish.
            await _until(lambda: _status(screen) == "Done")
            newest = _view(screen)
        release.set()
        publish.set()
        await _until(lambda: not _thread_running(screen))
        await pilot.pause()
        assert _view(screen) == newest
        assert "prompt='NEW'" in str(_view(screen)[2])
        assert "NEW" in decodes
        if not old_error:
            assert "OLD" in decodes


@pytest.mark.parametrize("old_error", [False, True], ids=["success", "error"])
@pytest.mark.parametrize("retire", ["cancel", "unmount"])
@private_profile_test
async def test_canceled_demo_without_replacement_cannot_publish(
    request, monkeypatch, old_error, retire
):
    async with _demo(monkeypatch, old_error=old_error) as (
        app,
        pilot,
        screen,
        calls,
        decodes,
        entered,
        release,
        queued,
        publish,
    ):
        await _generate(pilot, screen, "OLD")
        await _until(entered.is_set)
        before = _view(screen)
        if retire == "cancel":
            calls["OLD"].cancel()
        else:
            app.pop_screen()
            await _until(lambda: not screen.is_attached)
        assert calls["OLD"].is_cancelled
        release.set()
        await _until(lambda: not _thread_running(screen))
        await pilot.pause()
        if retire == "cancel":
            assert _view(screen) == before
        assert not app._exception


@pytest.mark.parametrize("new_error", [False, True], ids=["success", "error"])
@private_profile_test
async def test_serial_demo_and_palette_route_publish_current_outcome(
    request, monkeypatch, new_error
):
    async with _demo(monkeypatch, hold=None, new_error=new_error) as (
        app,
        pilot,
        screen,
        calls,
        decodes,
        entered,
        release,
        queued,
        publish,
    ):
        await _generate(pilot, screen, "OLD")
        await _until(lambda: _status(screen) == "Done")
        assert "prompt='OLD'" in str(_view(screen)[2])
        old_image = _view(screen)[1]
        await calls["OLD"].wait()
        await _generate(pilot, screen, "NEW")
        expected = "Error: NEW provider failed" if new_error else "Done"
        await _until(lambda: "NEW" in calls and _status(screen) == expected)
        await calls["NEW"].wait()
        if new_error:
            assert _view(screen)[1] is old_image
        else:
            assert "prompt='NEW'" in str(_view(screen)[2])
            assert _view(screen)[1] is not old_image
        assert not calls["NEW"].is_cancelled
