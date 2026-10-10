"""Original mounted lifecycle joins the image owner's real payload producer."""

import asyncio
from io import BytesIO
import threading
import time

import pytest

from Tests.private_profile import private_profile_test

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.timeout(300)]


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True], ids=["current", "detached"])
@private_profile_test
async def test_app_shutdown_joins_recovered_image_native_read(
    request, tmp_path, monkeypatch, detached
):
    from PIL import Image
    from textual.screen import Screen

    from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
    from Tests.UI.background_signals import await_background_task
    from Tests.UI.test_console_button_routing import _mounted_console
    from Tests.UI.test_original_character_teardown_lifetime import _stock_private_app
    from tldw_chatbook.Backup_Recovery import recovered_media
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )

    app = _stock_private_app()
    media = recovered_media.RecoveredMedia(tmp_path / "recovered")
    source = tmp_path / "source.png"
    data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG")
    source.write_bytes(data.getvalue())
    asset = media.retain(
        source, profile="p", message="m", slug="shutdown", media_type="image/png"
    )
    payload_path = media.resolve(asset)[1]
    app.recovered_media_root = media.root
    app.recovered_media_profile = "p"
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original_read = recovered_media._read
    reads = []
    image_task = shutdown = second_waiter = switch = None
    facts = {}

    def held_read(path):
        if path != payload_path:
            return original_read(path)
        assert threading.current_thread() is not threading.main_thread()
        reads.append(path)
        entered.set()
        assert release.wait(10), "Image native producer was never released"
        payload = original_read(path)
        assert payload == data.getvalue()
        returned.set()  # Original private stream's context has retired.
        return payload

    try:
        async with app.run_test(size=(160, 44)) as pilot:
            await await_background_task(
                app._initial_screen_setup_task, what="original initial Console setup"
            )
            console = await _mounted_console(app, pilot, "#console-native-composer")
            runtime = app.console_runtime
            assert runtime.view is console and console.app_instance is app
            image = console._image
            # Retire only prior image selection work; no all-worker idle oracle.
            image._recovered_images_close_admission()
            assert await image._recovered_images_drain(time.monotonic() + 10)
            monkeypatch.setattr(recovered_media, "_read", held_read)
            message = ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="Recovered image",
                status="complete",
            )
            message.persisted_message_id = "m"
            image._recovered_images_resume()
            image._build_console_image_specs([message])
            assert len(image._recovered_image_tasks) == 1
            image_task = next(iter(image._recovered_image_tasks))
            try:
                async with asyncio.timeout(10):
                    while not entered.is_set():
                        await asyncio.sleep(0.01)
                assert not returned.is_set() and not image_task.done()
                if detached:
                    # Actual Textual removal invokes original on_unmount. The
                    # old screen leaves the stack synchronously; uninstalling
                    # it then permits the normal replacement to remove it.
                    switch = app.switch_screen(Screen())
                    if app.is_screen_installed(console):
                        app.uninstall_screen(console)
                    async with asyncio.timeout(10):
                        while runtime.view is console:
                            await asyncio.sleep(0.01)
                    assert runtime.view is None
                    facts["unmount_waited"] = not switch.is_done
                shutdown = asyncio.create_task(app._shutdown_console_runtime())
                async with asyncio.timeout(10):
                    while app._console_runtime_shutdown_task is None:
                        await asyncio.sleep(0)
                terminal = app._console_runtime_shutdown_task
                await asyncio.sleep(0.03)
                facts["sealed"] = image._recovered_image_paused
                facts["terminal_waited"] = not terminal.done()
                # Concurrent callers join the same original app-owned task.
                second_waiter = asyncio.create_task(app._shutdown_console_runtime())
                shutdown.cancel()
                await asyncio.sleep(0)
                shutdown.cancel()
                image_task.cancel()
                await asyncio.sleep(0)
                image_task.cancel()
                await asyncio.sleep(0.03)
                facts["native_waited"] = not image_task.done() and not returned.is_set()
                facts["cancelled_caller_waited"] = not shutdown.done()
                facts["same_terminal"] = app._console_runtime_shutdown_task is terminal
            finally:
                release.set()
                if image_task is not None:
                    await asyncio.gather(image_task, return_exceptions=True)
                if shutdown is not None:
                    await asyncio.gather(shutdown, return_exceptions=True)
                if second_waiter is not None:
                    await second_waiter
                if switch is not None:
                    await switch
            assert returned.is_set() and reads == [payload_path]
            assert image_task.done() and not image._recovered_image_tasks
            assert terminal.done() and terminal.result() is None
            await app._shutdown_console_runtime()
            assert app._console_runtime_shutdown_task is terminal
        assert app._exception is None
    finally:
        release.set()
        drain_active_service_patches()
        drain_created_dirs()
    # Original-source RED occurs only after the real read and owner cleanup.
    assert all(facts.values()), facts
