"""End-to-end integration test for Console "Edit & resend" branching (Phase B, Task 5).

Exercises the full edit-and-resend lifecycle over the REAL stack -- an
in-memory ``CharactersRAGDB`` behind the real ``ChatPersistenceService`` /
``ChatConversationService``, a real ``ConsoleChatStore``, and the real
``ConsoleChatController`` (with a fake streaming provider gateway, mirroring
the harness used by ``Tests/Chat/test_console_edit_resend.py``) -- through:
linear send, edit-and-resend (forking a new USER sibling branch), swipe
between the old and new USER branches, persist -> drop the store -> resume
(via the real ``ChatScreen`` flatten + ``restore_persisted_session`` path
used by ``Tests/integration/test_console_branching_e2e.py``), and a
plain in-place "Save" edit that must NOT fork a new branch.

No shortcuts: every step drives the same store/controller methods the
production ``ChatScreen`` drives, and resume is a genuine persist -> drop ->
reload round-trip against the real database, not a hand-rolled fake.
"""

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_destination_shells import _build_test_app


class _SequencedGateway:
    """Fake provider gateway: streams one scripted full-text reply per call.

    Mirrors ``_SequencedGateway`` in ``test_console_branching_e2e.py``: each
    controller call (``submit_draft``/``edit_and_resend_message``) consumes
    the next scripted reply in order, so the E2E scenario can tell A1 and the
    edited branch's own reply apart.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self._calls = 0

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
        text = self._replies[self._calls]
        self._calls += 1
        yield text


def _new_controller(db: CharactersRAGDB, replies):
    """Real store (real DB-backed persistence) + real controller + fake gateway."""
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    controller = ConsoleChatController(
        store=store, provider_gateway=_SequencedGateway(replies)
    )
    session = store.create_session(title="Edit-Resend E2E")
    store.active_session_id = session.id
    return store, controller, session


def _resume_into_fresh_store(db: CharactersRAGDB, conversation_id: str):
    """Genuine persist -> drop -> resume round-trip via the real resume path.

    Mirrors ``_resume_into_fresh_store`` in ``test_console_branching_e2e.py``:
    the real ``ChatConversationService.get_conversation_tree`` full-tree read,
    the real ``ChatScreen._console_messages_from_conversation_tree`` flatten,
    the real ``db.get_conversation_active_leaf`` pointer read, and a brand
    new ``ConsoleChatStore`` fed through ``restore_persisted_session`` -- the
    exact same plumbing ``_resume_console_workspace_conversation`` drives.
    """
    service = ChatConversationService(db)
    tree = service.get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = db
    all_nodes = screen._console_messages_from_conversation_tree(tree)
    active_leaf_id = db.get_conversation_active_leaf(conversation_id)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Edit-Resend E2E",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
    )
    return store, session


@pytest.mark.asyncio
async def test_console_edit_and_resend_full_lifecycle_persist_resume_swipe():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        store, controller, session = _new_controller(
            db, replies=["A1", "edited-reply"]
        )

        # ---- Step 1: send U1 -> A1 (linear) ----
        result1 = await controller.submit_draft("U1")
        assert result1.accepted is True
        transcript = store.messages_for_session(session.id)
        assert [m.content for m in transcript] == ["U1", "A1"]
        u1, a1 = transcript
        _sibs, _idx, count = store.siblings_at(u1.id)
        assert count == 1  # no siblings yet -- linear

        # ---- Step 2: edit U1 & resend -> forks a NEW user sibling branch ----
        result2 = await controller.edit_and_resend_message(u1.id, "edited prompt")
        assert result2.accepted is True

        # u1's parent now has TWO user children.
        siblings, _index, sib_count = store.siblings_at(u1.id)
        assert sib_count == 2
        sibling_ids = {s.id for s in siblings}
        assert u1.id in sibling_ids
        new_user_id = next(iter(sibling_ids - {u1.id}))

        # The new sibling carries the edited text and sits on the active path.
        active_path_ids = store.active_path_message_ids(session.id)
        assert new_user_id in active_path_ids
        new_user = store.get_message(new_user_id)
        assert new_user.content == "edited prompt"

        # It has its own assistant reply, which is the new active leaf.
        active_leaf_id = store.active_leaf(session.id)
        new_reply = store.get_message(active_leaf_id)
        assert new_reply.content == "edited-reply"
        assert new_reply.status == "complete"
        assert active_path_ids == [new_user_id, active_leaf_id]

        # U1 (and its old subtree A1) are preserved OFF the active path.
        assert store.get_message(u1.id).content == "U1"
        assert store.get_message(a1.id).content == "A1"
        assert u1.id not in active_path_ids
        assert a1.id not in active_path_ids

        transcript_after_edit = [
            m.content for m in store.messages_for_session(session.id)
        ]
        assert transcript_after_edit == ["edited prompt", "edited-reply"]

        # ---- Step 3: swipe the USER row: `<` to the old branch, `>` back ----
        store.set_active_leaf(session.id, store._leaf_under(u1.id))
        old_branch_view = [
            m.content for m in store.messages_for_session(session.id)
        ]
        assert old_branch_view == ["U1", "A1"]

        store.set_active_leaf(session.id, store._leaf_under(new_user_id))
        new_branch_view = [
            m.content for m in store.messages_for_session(session.id)
        ]
        assert new_branch_view == ["edited prompt", "edited-reply"]

        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None  # real persistence engaged throughout

        # ---- Step 4: persist, DROP the store, resume via the real resume path ----
        resumed_store, resumed_session = _resume_into_fresh_store(db, conversation_id)
        resumed_transcript = [
            m.content for m in resumed_store.messages_for_session(resumed_session.id)
        ]
        # The active branch (the edited one, left active above) is restored.
        assert resumed_transcript == ["edited prompt", "edited-reply"]
        resumed_leaf_id = resumed_store.active_leaf(resumed_session.id)
        assert resumed_store.get_message(resumed_leaf_id).content == "edited-reply"
        assert resumed_store.get_message(resumed_leaf_id).persisted_message_id == (
            new_reply.persisted_message_id
        )

        # The old branch (U1/A1) is still reachable off-path after resume.
        resumed_new_user = resumed_store.messages_for_session(resumed_session.id)[0]
        resumed_sibs, _idx, resumed_count = resumed_store.siblings_at(
            resumed_new_user.id
        )
        assert resumed_count == 2
        resumed_u1 = next(
            s for s in resumed_sibs if s.id != resumed_new_user.id
        )
        assert resumed_u1.content == "U1"
        resumed_store.set_active_leaf(
            resumed_session.id, resumed_store._leaf_under(resumed_u1.id)
        )
        assert [
            m.content
            for m in resumed_store.messages_for_session(resumed_session.id)
        ] == ["U1", "A1"]
        # Swipe back to the edited branch before moving on.
        resumed_store.set_active_leaf(
            resumed_session.id, resumed_store._leaf_under(resumed_new_user.id)
        )

        # ---- Step 5: in-place "Save" (NOT resend) still edits in place ----
        resumed_edited_user_id = resumed_new_user.id
        _before_sibs, _idx, before_count = resumed_store.siblings_at(
            resumed_edited_user_id
        )
        resumed_store.update_message_content(resumed_edited_user_id, "typo fix")
        after_sibs, _idx, after_count = resumed_store.siblings_at(
            resumed_edited_user_id
        )
        assert after_count == before_count  # no new branch created
        assert resumed_store.get_message(resumed_edited_user_id).content == "typo fix"
        assert [
            m.content
            for m in resumed_store.messages_for_session(resumed_session.id)
        ] == ["typo fix", "edited-reply"]
    finally:
        db.close_connection()
