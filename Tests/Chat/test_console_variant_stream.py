"""Regenerate is unified onto the streaming reply engine (Task 1)."""
import asyncio

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _store_with_answer():
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original")
    # Non-empty content already yields status "complete" via _initial_status;
    # this mirrors the existing regenerate tests that seed via append_message alone.
    return store, session, assistant.id


def test_begin_variant_stream_resets_buffer_and_keeps_base():
    store, _session, mid = _store_with_answer()
    streaming = store.begin_variant_stream(mid)
    assert streaming.status == "streaming"
    assert streaming.content == ""            # visible row cleared for the new take
    store.append_stream_chunk(mid, "re")
    store.append_stream_chunk(mid, "generated")
    final = store.finalize_variant_stream(mid)
    assert final.status == "complete"
    assert final.content == "regenerated"     # new variant selected
    assert final.variants is not None
    contents = [v.content for v in final.variants.variants]
    assert contents == ["original", "regenerated"]  # base preserved, no concat
    assert final.variants.selected_index == 1


def test_finalize_variant_stream_appends_to_existing_set():
    store, _session, mid = _store_with_answer()
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "second")
    store.finalize_variant_stream(mid)
    store.begin_variant_stream(mid)
    store.append_stream_chunk(mid, "third")
    final = store.finalize_variant_stream(mid)
    assert [v.content for v in final.variants.variants] == ["original", "second", "third"]
    assert final.variants.selected_index == 2


class _ScriptedGateway:
    """Async stream_chat that yields scripted chunks; resolve_for_send ready."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def resolve_for_send(self, selection):
        class _R:  # noqa: D401 - tiny stub
            ready = True
            visible_copy = ""
        return _R()

    async def stream_chat(self, resolution, messages):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_regenerate_delegates_and_streams_incrementally():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, _session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ScriptedGateway(["Paris", " is", " the", " answer."]),
        provider="llama_cpp",
        model="test-model",
    )
    result = await controller.regenerate_message(mid)
    assert result.accepted is True
    message = store.get_message(mid)
    assert message.content == "Paris is the answer."
    assert [v.content for v in message.variants.variants] == [
        "original", "Paris is the answer."]
    assert message.variants.selected_index == 1


@pytest.mark.asyncio
async def test_regenerate_empty_stream_restores_prior_status_and_keeps_context():
    """Plan-B Task 1 finding: a zero-chunk (empty-stream) regenerate of a
    previously-COMPLETE assistant message must not end up excluded from the
    model's context for the rest of the session. Every send path builds
    provider context via `_provider_messages_for_session(..., skip_failed=
    True)`; pre-refactor, a failed regenerate was a pure no-op, so this
    turn stayed "complete" and stayed in context. The regression: `mark_
    message_failed` was restoring the base CONTENT but flipping status to
    "failed", which silently drops the turn from context on every later
    send/regenerate/retry in this session even though the visible content
    is fully intact.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ScriptedGateway([]),  # zero chunks: empty stream
        provider="llama_cpp",
        model="test-model",
    )

    result = await controller.regenerate_message(mid)

    assert result.accepted is True
    message = store.get_message(mid)
    assert message.status == "complete"
    assert message.content == "original"

    provider_messages = controller._provider_messages_for_session(session.id)
    assert {"role": "assistant", "content": "original"} in provider_messages


@pytest.mark.asyncio
async def test_regenerate_stop_mid_stream_restores_original_answer():
    """Plan-B final-review Medium-2: stopping a regenerate mid-stream must
    restore the pre-regenerate answer exactly like a failed regenerate --
    not replace it with the partial streamed buffer marked "stopped". Pre-
    branch, Stop could not even reach the regenerate loop (no interruptible
    task was ever set during the old inline regenerate loop); post-
    unification onto the shared streaming engine, Stop is live during
    regenerate and this pinned a real regression.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    class WaitingGateway:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def resolve_for_send(self, selection):
            class _R:  # noqa: D401 - tiny stub
                ready = True
                visible_copy = ""
            return _R()

        async def stream_chat(self, resolution, messages):
            self.started.set()
            yield "partial regen "
            await self.release.wait()
            yield "ignored"

    gateway = WaitingGateway()
    store, _session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model")

    task = asyncio.create_task(controller.regenerate_message(mid))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert controller.stop_active_run() is True
    message = store.get_message(mid)
    assert message.content == "original"
    assert message.status == "complete"
    assert message.variants is None
    assert mid not in store._variant_stream_bases

    gateway.release.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.accepted is True
    message = store.get_message(mid)
    assert message.content == "original"
    assert message.status == "complete"
