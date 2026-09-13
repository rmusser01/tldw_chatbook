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
from Tests.console_provider_doubles import persisted_console_store, provider_resolution


class StreamingGateway:
    async def resolve_for_send(self, selection):
        return provider_resolution(base_url="http://127.0.0.1:9099")

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in ("hel", "lo"):
            yield chunk


class FailingBeforeAnyChunkGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages, **kwargs):
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
async def test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hi")
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

    siblings, _index, count = store.siblings_at(a1.id)
    assert count == 2
    new_sibling = next(sibling for sibling in siblings if sibling.id != a1.id)
    assert new_sibling.status == "failed"
    notice = store.messages_for_session(session.id)[-1]
    assert notice.role is ConsoleMessageRole.SYSTEM
    assert notice.content == result.visible_copy
    assert notice.persisted_message_id is None
    assert store.active_leaf(session.id) == notice.id
    assert store.active_path_message_ids(session.id) == [u1.id, a1.id, notice.id]
    assert {"role": "assistant", "content": "seed"} in (
        controller._provider_messages_for_session(session.id)
    )


@pytest.mark.asyncio
async def test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeAnyChunkGateway()
    )
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
    assert "Provider stream failed:" in result.visible_copy
    notice = store.messages_for_session(session.id)[-1]
    assert notice.role is ConsoleMessageRole.SYSTEM
    assert notice.content == result.visible_copy
    assert notice.persisted_message_id is None
    assert store.active_path_message_ids(session.id) == [u1.id, a1.id, notice.id]
    assert store.get_message(u2.id).content == "q2"
    assert store.get_message(a2.id).content == "a2-seed"
    assert u2.id not in store.active_path_message_ids(session.id)
    assert a2.id not in store.active_path_message_ids(session.id)

    siblings, _index, count = store.siblings_at(a1.id)
    assert count == 2
    failed_sibling = next(sibling for sibling in siblings if sibling.id != a1.id)
    assert failed_sibling.status == "failed"

    provider_messages = controller._provider_messages_for_session(session.id)
    assert provider_messages[-1] == {
        "role": "assistant",
        "content": "a1-seed",
    }


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


@pytest.mark.asyncio
async def test_regenerate_persists_new_sibling_when_store_has_persistence(tmp_path):
    """Critical regression: the regenerated sibling must be durably persisted.

    ``create_sibling`` defaults ``persist=False``. Before this fix,
    ``regenerate_message`` called it with no ``persist`` kwarg at all, so the
    freshly forked node was never registered in
    ``_pending_persistence_message_ids``. When the stream completed,
    ``mark_message_complete`` -> ``_persist_existing_message`` saw
    ``persisted_message_id is None`` and deferred to
    ``_persist_pending_message_if_ready``, which no-opped because the node's
    id was never marked pending -- so on a persistence-backed session (the
    normal production configuration, since ``chat_screen.py`` always wires
    real persistence when ``chachanotes_db`` exists), the regenerated reply
    was silently never written to the DB and vanished on resume.
    """
    store = persisted_console_store(database_path=tmp_path / "regenerate.sqlite")
    db = store.persistence.db
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    try:
        session = store.create_session(title="t")
        store.active_session_id = session.id
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Hi", persist=True
        )
        a1 = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="seed", persist=True
        )
        result = await controller.regenerate_message(a1.id)

        assert result.accepted is True
        new_leaf_id = store.active_leaf(session.id)
        assert new_leaf_id != a1.id
        new_message = store.get_message(new_leaf_id)
        assert new_message.content == "hello"
        assert new_message.status == "complete"

        # Verify the durable row, not merely an invocation of a recording fake.
        assert new_message.persisted_message_id is not None
        persisted = db.get_message_by_id(new_message.persisted_message_id)
        assert persisted["content"] == "hello"
        assert persisted["parent_message_id"] == a1.parent_message_id
        assert persisted["assistant_generation_state"] == "complete"
        assert db.get_message_by_id(a1.persisted_message_id)["content"] == "seed"
    finally:
        await controller.shutdown()
        with db.quiesce_connections(timeout_seconds=2.0):
            pass
