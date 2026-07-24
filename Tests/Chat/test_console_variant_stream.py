"""Regenerate is unified onto the streaming reply engine (Task 1).

TASK-6 (Phase A): ``regenerate_message`` no longer streams a replacement
*variant* into the anchor message in place -- it forks a persisted SIBLING
node under the anchor's own parent and streams into that NEW node
(``variant_mode=False``). The ``begin_variant_stream``/``finalize_variant_
stream``/``add_variant`` store primitives below are exercised directly
(store-level, not through the controller) and remain valid, but the
controller-driven tests further down were rewritten to assert against the
new sibling node rather than the untouched anchor -- see
``Tests/Chat/test_console_regenerate_branching.py`` for the full branching
contract.
"""

import asyncio

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _store_with_answer():
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    # Non-empty content already yields status "complete" via _initial_status;
    # this mirrors the existing regenerate tests that seed via append_message alone.
    return store, session, assistant.id


def test_begin_variant_stream_resets_buffer_and_keeps_base():
    store, _session, mid = _store_with_answer()
    streaming = store.begin_variant_stream(mid)
    assert streaming.status == "streaming"
    assert streaming.content == ""  # visible row cleared for the new take
    store.append_stream_chunk(mid, "re")
    store.append_stream_chunk(mid, "generated")
    final = store.finalize_variant_stream(mid)
    assert final.status == "complete"
    assert final.content == "regenerated"  # new variant selected
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
    assert [v.content for v in final.variants.variants] == [
        "original",
        "second",
        "third",
    ]
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

    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ScriptedGateway(["Paris", " is", " the", " answer."]),
        provider="llama_cpp",
        model="test-model",
    )
    result = await controller.regenerate_message(mid)
    assert result.accepted is True

    # The anchor is untouched and off the active path; a NEW sibling node
    # streamed incrementally and carries the fresh answer.
    unchanged = store.get_message(mid)
    assert unchanged.content == "original"
    assert unchanged.variants is None
    assert mid not in store.active_path_message_ids(session.id)

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != mid
    message = store.get_message(new_leaf_id)
    assert message.content == "Paris is the answer."
    assert message.variants is None


@pytest.mark.asyncio
async def test_regenerate_empty_stream_leaves_anchor_untouched_and_new_sibling_failed():
    """TASK-6 supersedes the original Plan-B Task 1 finding here: under the
    old in-place regenerate, a zero-chunk (empty-stream) regenerate had to
    restore the anchor's own prior status so the turn did not silently drop
    out of context (``_provider_messages_for_session(..., skip_failed=
    True)``). Under the sibling/branching model that concern moves with the
    node: regenerate ALWAYS forks a new node and makes it the active leaf,
    so an empty-stream (zero-chunk) regenerate leaves that NEW node
    "failed" (empty, retryable) as the active tip -- the anchor itself is
    a completely separate, untouched node that simply drops off the active
    path (not deleted, and not silently mutated to a failed status either).
    Swiping back to the anchor (``set_active_leaf``) restores it -- and its
    "complete" status/content -- to the active path and provider context.
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
    unchanged = store.get_message(mid)
    assert unchanged.status == "complete"
    assert unchanged.content == "original"
    assert mid not in store.active_path_message_ids(session.id)

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != mid
    new_sibling = store.get_message(new_leaf_id)
    assert new_sibling.status == "failed"
    assert new_sibling.content == ""

    # The failed, empty new sibling is correctly excluded from context...
    provider_messages = controller._provider_messages_for_session(session.id)
    assert {"role": "assistant", "content": "original"} not in provider_messages
    assert {"role": "assistant", "content": ""} not in provider_messages

    # ...but swiping back to the anchor restores it to the active path
    # (and therefore to context) exactly as before.
    store.set_active_leaf(session.id, mid)
    provider_messages = controller._provider_messages_for_session(session.id)
    assert {"role": "assistant", "content": "original"} in provider_messages


@pytest.mark.asyncio
async def test_regenerate_stop_mid_stream_leaves_anchor_untouched_new_sibling_stopped():
    """Plan-B final-review Medium-2, superseded by TASK-6's branching model:
    stopping a regenerate mid-stream must not touch the anchor's own
    pre-regenerate answer at all -- it is a completely separate node. The
    NEW sibling that was streaming into is the one left "stopped" with
    whatever partial buffer it had accumulated; the anchor stays "complete"
    with its original content throughout, on the active path only for as
    long as the stop leaves the new (stopped) sibling's `set_active_leaf`
    untouched -- i.e. the new sibling remains the active leaf, and the
    anchor is reachable by swiping back.
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
    store, session, mid = _store_with_answer()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model"
    )

    task = asyncio.create_task(controller.regenerate_message(mid))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != mid

    assert controller.stop_active_run() is True
    # The anchor was never touched by the regenerate attempt at all.
    anchor = store.get_message(mid)
    assert anchor.content == "original"
    assert anchor.status == "complete"
    assert anchor.variants is None
    assert mid not in store._variant_stream_bases
    assert new_leaf_id not in store._variant_stream_bases

    gateway.release.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.accepted is True

    # The anchor is still untouched; the NEW sibling carries the partial
    # buffer and is marked "stopped" as the active leaf.
    anchor = store.get_message(mid)
    assert anchor.content == "original"
    assert anchor.status == "complete"
    stopped_sibling = store.get_message(new_leaf_id)
    assert stopped_sibling.content == "partial regen "
    assert stopped_sibling.status == "stopped"
    # (stop_active_run's own "Response stopped by user." system row becomes
    # the new active leaf, parented under the stopped sibling above --
    # pre-existing behavior, unrelated to Task 6, not asserted here.)
