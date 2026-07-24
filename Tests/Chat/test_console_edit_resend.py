"""Controller-level tests for edit-and-resend-as-sibling (Phase B, Task 2).

``edit_and_resend_message`` forks a NEW USER sibling branch alongside the
edited message (``store.create_sibling``, mirroring how
``regenerate_message`` forks assistant siblings) and appends a fresh empty
ASSISTANT node under it to stream a reply into. The anchor USER message
(and any old tail beneath it -- its prior assistant reply, and anything
after it) is left untouched and simply drops off the active path.
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
        for chunk in ("edi", "ted-reply"):
            yield chunk


class NotReadyGateway:
    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": False,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "WIP: provider offline",
            },
        )()

    async def stream_chat(self, resolution, messages):  # pragma: no cover - unreachable
        yield ""


class _EditResendPersistence:
    """Records create_message calls; hands back predictable ``msg-N`` ids.

    Mirrors ``_RegeneratePersistence`` in
    ``test_console_regenerate_branching.py``.
    """

    def __init__(self):
        self.created_messages = []

    def create_conversation(self, **kwargs):
        return "conv-1"

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
    ):
        pid = f"msg-{len(self.created_messages) + 1}"
        self.created_messages.append(
            {
                "id": pid,
                "conversation_id": conversation_id,
                "sender": sender,
                "content": content,
                "parent_message_id": parent_message_id,
            }
        )
        return pid

    def update_message_content(self, **kwargs):
        return True


@pytest.mark.asyncio
async def test_edit_and_resend_forks_user_sibling_and_streams_reply():
    """Core contract: two USER children under u1's parent, new sibling active,
    an assistant reply forked under it, and u1's old tail preserved off-path.
    """
    persistence = _EditResendPersistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.create_session(title="t")
    store.active_session_id = session.id
    u1 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="original", persist=True
    )
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original-reply", persist=True
    )
    persistence.created_messages.clear()

    result = await controller.edit_and_resend_message(u1.id, "edited")

    assert result.accepted is True

    # u1's parent now has TWO user children (u1 and the new sibling).
    siblings, _index, count = store.siblings_at(u1.id)
    assert count == 2
    sibling_ids = {s.id for s in siblings}
    assert u1.id in sibling_ids

    # The new sibling carries the edited content and is on the active path.
    new_leaf_ancestor_ids = set(store.active_path_message_ids(session.id))
    new_user_id = next(iter(sibling_ids - {u1.id}))
    assert new_user_id in new_leaf_ancestor_ids
    new_user = store.get_message(new_user_id)
    assert new_user.content == "edited"
    assert new_user.role is ConsoleMessageRole.USER
    assert new_user.persisted_message_id is not None

    # An assistant reply node exists under it with the streamed text, and is
    # the new active leaf.
    active_leaf_id = store.active_leaf(session.id)
    assistant_reply = store.get_message(active_leaf_id)
    assert assistant_reply.role is ConsoleMessageRole.ASSISTANT
    assert assistant_reply.content == "edited-reply"
    assert assistant_reply.status == "complete"
    assert assistant_reply.persisted_message_id is not None

    # u1 and its old tail (a1) are preserved, untouched, off the active path.
    assert store.get_message(u1.id).content == "original"
    assert store.get_message(a1.id).content == "original-reply"
    active_path_ids = store.active_path_message_ids(session.id)
    assert u1.id not in active_path_ids
    assert a1.id not in active_path_ids
    assert active_path_ids == [new_user_id, active_leaf_id]

    # The old branch is still reachable by swiping back.
    store.set_active_leaf(session.id, a1.id)
    assert store.active_path_message_ids(session.id) == [u1.id, a1.id]


@pytest.mark.asyncio
async def test_edit_and_resend_mid_conversation_preserves_rest_of_old_branch():
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

    result = await controller.edit_and_resend_message(u1.id, "edited q1")

    assert result.accepted is True
    active_path_ids = store.active_path_message_ids(session.id)
    assert len(active_path_ids) == 2
    new_user_id, new_assistant_id = active_path_ids
    assert store.get_message(new_user_id).content == "edited q1"
    assert store.get_message(new_assistant_id).content == "edited-reply"

    # The entire old branch (a1, u2, a2) is preserved off-path.
    assert store.get_message(a1.id).content == "a1-seed"
    assert store.get_message(u2.id).content == "q2"
    assert store.get_message(a2.id).content == "a2-seed"
    store.set_active_leaf(session.id, a2.id)
    assert store.active_path_message_ids(session.id) == [u1.id, a1.id, u2.id, a2.id]


@pytest.mark.asyncio
async def test_edit_and_resend_blocks_non_user_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="seed"
    )

    result = await controller.edit_and_resend_message(a1.id, "edited")

    assert result.accepted is False
    assert "Only your messages can be edited and re-sent." in result.visible_copy
    # No sibling was created: a1 still has no siblings at all.
    siblings, _index, count = store.siblings_at(a1.id)
    assert count == 1
    assert store.get_message(a1.id).content == "seed"


@pytest.mark.asyncio
async def test_edit_and_resend_blank_content_blocks_without_mutating_tree():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hi")

    result = await controller.edit_and_resend_message(u1.id, "   ")

    assert result.accepted is False
    siblings, _index, count = store.siblings_at(u1.id)
    assert count == 1
    assert store.get_message(u1.id).content == "Hi"


@pytest.mark.asyncio
async def test_edit_and_resend_provider_not_ready_blocks_without_orphan_sibling():
    """Phase A Task-6 lesson: a blocked resend must leave no orphan sibling."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=NotReadyGateway())
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hi")

    result = await controller.edit_and_resend_message(u1.id, "edited")

    assert result.accepted is False
    siblings, _index, count = store.siblings_at(u1.id)
    assert count == 1
    assert store.get_message(u1.id).content == "Hi"


@pytest.mark.asyncio
async def test_edit_and_resend_blocks_off_active_path_anchor():
    """Qodo PR #811 finding 2: ``_provider_messages_for_session`` scans the
    ACTIVE-PATH transcript until ``before_message_id`` is seen -- if the
    anchor is not on the active path, that scan never breaks and the resend
    payload would be built from the wrong branch. Edit is only exposed on
    active-path rows today, but the controller must refuse a
    directly-called off-path anchor rather than silently mis-building the
    payload."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    # Forking a sibling makes IT the new active leaf, leaving u1 off-path.
    sibling = store.create_sibling(u1.id, role=ConsoleMessageRole.USER, content="q1-b")
    assert store.active_leaf(session.id) == sibling.id
    active_path_ids = store.active_path_message_ids(session.id)
    assert u1.id not in active_path_ids
    _siblings, _index, count_before = store.siblings_at(u1.id)
    assert count_before == 2

    result = await controller.edit_and_resend_message(u1.id, "edited")

    assert result.accepted is False
    assert (
        "Switch to that branch before editing and re-sending this message."
        in result.visible_copy
    )
    # No third sibling was forked, and no pending assistant node anywhere.
    _siblings, _index, count_after = store.siblings_at(u1.id)
    assert count_after == count_before
    assert all(
        m.status != "pending" for m in store.messages_for_session(session.id)
    )
    # u1 itself is untouched, and neither u1 nor the sibling gained a new
    # USER/ASSISTANT child -- `_block` only appends a SYSTEM notice under
    # the (unrelated) active leaf, same as every other block gate.
    assert store.get_message(u1.id).content == "q1"
    assert store.get_message(sibling.id).content == "q1-b"


@pytest.mark.asyncio
async def test_edit_and_resend_skill_refusal_leaves_no_pending_node():
    """Critical (Phase B Task 2 review): a skill-substitution refusal
    discovered while resending an edited message must not leave a stray
    ``pending`` ASSISTANT node forked into the tree.

    Before the fix, ``new_user``/``assistant`` were created BEFORE the
    skill-substitution check ran, so a refusal returned ``_block(...)``
    while the empty ``assistant`` node was already on the active path --
    it never streams (so it never becomes ``"failed"``), and
    ``retry_message`` requires ``"failed"``, so it was permanently stuck
    tree litter. The fix builds and transforms the provider payload FIRST
    and only creates ``new_user``/``assistant`` after every transform
    (including skill substitution) has succeeded, mirroring
    ``regenerate_message``'s "mutate last" discipline.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="original")

    async def _refuse(provider_messages):
        # 5-tuple contract: (messages, refuse, notes, skill_bindings,
        # skill_bundle_block) — widened by the skill_file reachability work.
        return provider_messages, "Skill refused for testing.", (), (), ""

    controller._apply_skill_substitution = _refuse

    result = await controller.edit_and_resend_message(u1.id, "edited")

    assert result.accepted is False
    assert result.visible_copy == "Skill refused for testing."

    # No new USER sibling was forked: u1 still has no siblings at all.
    siblings, _index, count = store.siblings_at(u1.id)
    assert count == 1
    assert store.get_message(u1.id).content == "original"

    # No pending assistant node anywhere on the active path (or off it --
    # nothing was ever created).
    assert all(
        m.status != "pending" for m in store.messages_for_session(session.id)
    )
