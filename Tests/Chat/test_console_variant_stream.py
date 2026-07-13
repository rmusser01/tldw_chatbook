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
