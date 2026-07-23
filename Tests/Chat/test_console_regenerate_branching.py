"""Controller-level tests for regenerate-as-sibling (Phase A, Task 6).

``regenerate_message`` used to stream a replacement variant into the SAME
assistant message (``variant_mode=True`` -> ``begin_variant_stream`` /
``finalize_variant_stream``). It now forks a persisted SIBLING assistant
node under the anchor's own parent (``store.create_sibling``) and streams
into that new node normally (``variant_mode=False``), so a mid-conversation
regenerate creates a real branch instead of mutating history in place.
"""

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class StreamingGateway:
    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
            },
        )()

    async def stream_chat(self, resolution, messages):
        for chunk in ("hel", "lo"):
            yield chunk


class FailingBeforeAnyChunkGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages):
        raise RuntimeError("regen exploded")
        yield ""  # pragma: no cover - unreachable, keeps this an async generator


@pytest.mark.asyncio
async def test_regenerate_creates_sibling_and_streams_into_new_active_leaf():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="seed"
    )

    result = await controller.regenerate_message(a1.id)

    assert result.accepted is True

    # Two assistant children now live under a1's own parent.
    siblings, _index, count = store.siblings_at(a1.id)
    assert count == 2
    sibling_ids = {s.id for s in siblings}
    assert a1.id in sibling_ids

    # The active leaf moved to the NEW child, not a1.
    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != a1.id
    assert new_leaf_id in sibling_ids

    # The new child carries the freshly streamed text.
    new_message = store.get_message(new_leaf_id)
    assert new_message.content == "hello"
    assert new_message.status == "complete"

    # a1 is untouched and now off the active path.
    unchanged_a1 = store.get_message(a1.id)
    assert unchanged_a1.content == "seed"
    assert a1.id not in store.active_path_message_ids(session.id)


@pytest.mark.asyncio
async def test_regenerate_mid_conversation_forks_branch_and_preserves_old_tail():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a1-seed"
    )
    u2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="q2")
    a2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a2-seed"
    )

    result = await controller.regenerate_message(a1.id)

    assert result.accepted is True
    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id not in {a1.id, a2.id}

    # Active path now runs straight from u1 to the new sibling -- the old
    # tail (u2, a2) has dropped off the visible branch.
    assert store.active_path_message_ids(session.id) == [u1.id, new_leaf_id]

    # The old tail is not deleted -- it is still reachable by swiping back.
    assert store.get_message(a1.id).content == "a1-seed"
    assert store.get_message(u2.id).content == "q2"
    assert store.get_message(a2.id).content == "a2-seed"
    store.set_active_leaf(session.id, a2.id)
    assert store.active_path_message_ids(session.id) == [u1.id, a1.id, u2.id, a2.id]


@pytest.mark.asyncio
async def test_regenerate_stream_failure_marks_new_sibling_failed_not_a1():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="seed"
    )

    controller.provider_gateway = FailingBeforeAnyChunkGateway()
    result = await controller.regenerate_message(a1.id)

    assert result.accepted is True
    assert "Provider stream failed:" in result.visible_copy

    # a1 is completely untouched by the failed regenerate.
    unchanged_a1 = store.get_message(a1.id)
    assert unchanged_a1.content == "seed"
    assert unchanged_a1.status == "complete"

    # The NEW sibling is the one left in the "failed" (retryable) state --
    # this is the intended node-model behavior: a failed regenerate does not
    # restore the old reply in place, it leaves a retryable failed node as
    # the new branch tip. (The active leaf itself has since moved to the
    # failure system row that `_stream_assistant_response` appends, which is
    # pre-existing, unrelated-to-Task-6 behavior -- not asserted here.)
    messages = store.messages_for_session(session.id)
    new_sibling = next(
        m
        for m in messages
        if m.role is ConsoleMessageRole.ASSISTANT and m.id != a1.id
    )
    assert new_sibling.status == "failed"


@pytest.mark.asyncio
async def test_regenerate_on_leading_greeting_still_blocks_without_mutating_tree():
    """Blocking a regenerate (no user turn yet) must not fork a stray node."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.create_session(title="Chat with Elara")
    greeting = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Greetings.",
        persist=False,
    )

    result = await controller.regenerate_message(greeting.id)

    assert result.accepted is False
    # No sibling was created: the greeting still has no siblings at all.
    siblings, _index, count = store.siblings_at(greeting.id)
    assert count == 1
    assert store.get_message(greeting.id).content == "Greetings."
