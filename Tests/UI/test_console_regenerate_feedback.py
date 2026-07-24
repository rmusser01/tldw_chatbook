"""TASK-343: Regenerate/Retry/Continue must show in-flight status.

The transcript sync timer used to start only on the send path
(`_submit_console_native_draft`); the regenerate/retry/continue handlers
awaited the whole generation with zero UI sync, so nothing on screen changed
until the provider finished (75s+ against a slow local model — UX review
finding j6-regenerate-zero-feedback). These tests hold the fake provider
open mid-stream and assert the first chunk is already visible.
"""

import asyncio  # noqa: F401  (GatedGateway release event)

import pytest

from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
    _select_llamacpp_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleTranscript


class GatedGateway(_ReadyResolutionGateway):
    """Stream one chunk immediately, then hold until the test releases."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def stream_chat(self, resolution, messages):
        yield "first-chunk"
        await self.release.wait()
        yield " final-chunk"


async def _seed_selected_assistant_message(console, pilot):
    store = console._ensure_console_chat_store()
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    source = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="seed"
    )
    await console._sync_native_console_chat_ui()
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.select_message(source.id)
    await console._sync_native_console_chat_ui()
    return source


@pytest.mark.asyncio
async def test_console_regenerate_streams_first_chunk_while_in_flight():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        source = await _seed_selected_assistant_message(console, pilot)
        await _wait_for_selector(
            console, pilot, f"#console-message-action-regenerate-{source.id}"
        )

        await pilot.click(f"#console-message-action-regenerate-{source.id}")

        # The provider is held open — the first chunk must already be
        # visible and the sync timer running while the run is in flight.
        await _wait_for_text(console, pilot, "first-chunk")
        assert console._console_transcript_sync_timer is not None

        gateway.release.set()
        await _wait_for_text(console, pilot, "final-chunk")


@pytest.mark.asyncio
async def test_console_retry_streams_first_chunk_while_in_flight():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        # Retry is only offered for failed messages: seed a pending
        # assistant reply and mark it failed (run-gate recipe).
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="question"
        )
        pending = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        source = store.mark_message_failed(pending.id)
        await console._sync_native_console_chat_ui()
        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-retry-{source.id}"
        )

        await pilot.click(f"#console-message-action-retry-{source.id}")

        await _wait_for_text(console, pilot, "first-chunk")
        assert console._console_transcript_sync_timer is not None

        gateway.release.set()
        await _wait_for_text(console, pilot, "final-chunk")


@pytest.mark.asyncio
async def test_console_continue_streams_first_chunk_while_in_flight():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        source = await _seed_selected_assistant_message(console, pilot)
        await _wait_for_selector(
            console, pilot, f"#console-message-action-continue-{source.id}"
        )

        await pilot.click(f"#console-message-action-continue-{source.id}")

        await _wait_for_text(console, pilot, "first-chunk")
        assert console._console_transcript_sync_timer is not None

        gateway.release.set()
        await _wait_for_text(console, pilot, "final-chunk")
