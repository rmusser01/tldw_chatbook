"""Unchanged in-flight send status must not repaint visible Console text."""

import asyncio
from collections import Counter
from contextlib import closing
from dataclasses import replace

import pytest
from textual._compositor import Compositor
from textual.geometry import Region

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _persist_console_provider_config
from Tests.UI.test_console_rail_refresh_scope import count_compositor_updates
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("size", [(80, 24), (128, 24), (160, 45)])
async def test_waiting_send_does_not_repaint_unchanged_status(
    monkeypatch, tmp_path, size
):
    app = _build_test_app()
    with closing(CharactersRAGDB(tmp_path / "chat.sqlite", "send-refresh")) as database:
        entered, release = asyncio.Event(), asyncio.Event()
        console = None
        try:
            app.chachanotes_db = database
            for index in range(4):
                database.add_conversation({"title": f"Existing {index}"})
            _persist_console_provider_config(
                app,
                provider="openai",
                model="gpt-4.1",
                provider_settings={"api_key": "synthetic-test-key"},
            )
            host = ConsoleHarness(app)
            host.CSS_PATH = TldwCli.CSS_PATH
            async with host.run_test(size=size) as pilot:
                console = host.screen_stack[-1]
                await _wait_for_selector(console, pilot, "#console-native-composer")
                controller = console._ensure_console_chat_controller()
                gateway = controller.provider_gateway
                original_resolve = gateway.resolve_for_send

                async def resolve(selection):
                    entered.set()
                    await release.wait()
                    return replace(await original_resolve(selection), streaming=False)

                monkeypatch.setattr(gateway, "resolve_for_send", resolve)
                monkeypatch.setattr(
                    gateway,
                    "_chat_api_call_fn",
                    lambda **kw: {
                        "choices": [{"message": {"content": "Synthetic response"}}],
                    },
                )
                composer = console._console_composer_or_none()
                composer.load_draft("Hello")
                composer.focus()
                await pilot.pause()
                await pilot.press("enter")
                await asyncio.wait_for(entered.wait(), timeout=2)
                await pilot.pause(0.5)
                assert controller.run_state.status is ConsoleRunStatus.VALIDATING
                assert console._console_transcript_sync_timer is not None
                regions = {
                    widget_id: console.query_one(f"#{widget_id}").content_region
                    for widget_id in (
                        "console-transcript-title",
                        "console-send-disabled-reason",
                        "footer-key-quit",
                    )
                }
                assert regions["console-transcript-title"].area
                assert regions["footer-key-quit"].area
                # The reason strip intentionally hides when the draft needs its width.
                regions = {
                    key: region for key, region in regions.items() if region.area
                }
                painted = Counter()
                original_partial = Compositor.render_partial_update

                def partial(compositor):
                    update = original_partial(compositor)
                    if update is not None and compositor is console._compositor:
                        for widget_id, region in regions.items():
                            if any(
                                region.overlaps(Region(start, y, end - start, 1))
                                for y, start, end in update.spans
                            ):
                                painted[widget_id] += 1
                    return update

                monkeypatch.setattr(Compositor, "render_partial_update", partial)
                counts = {"full": 0, "partial": 0}
                with count_compositor_updates(counts):
                    for _ in range(3):
                        await console._sync_native_console_chat_ui()
                        await pilot.pause(0.05)
                assert counts["full"] == 0, counts
                assert not painted, painted

                release.set()
                await asyncio.wait_for(host.workers.wait_for_complete(), timeout=10)
                await pilot.pause(0.3)
                assert controller.run_state.status is ConsoleRunStatus.COMPLETED
                assert console._console_transcript_sync_timer is None
        finally:
            release.set()
            if console is not None:
                await console._console_runtime().dispose()
            await asyncio.to_thread(app.ui_responsiveness_monitor.close)
