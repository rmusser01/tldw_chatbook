"""End-to-end integration test for Console conversation branching (Phase A, Task 9).

Exercises the whole branching lifecycle over the REAL stack -- an in-memory
``CharactersRAGDB`` behind the real ``ChatPersistenceService`` /
``ChatConversationService``, a real ``ConsoleChatStore``, and the real
``ConsoleChatController`` (with a fake streaming provider gateway, mirroring
the harness used by ``Tests/Chat/test_console_regenerate_branching.py``) --
through: linear send, regenerate-as-sibling, continue, persist -> drop the
store -> resume (via the real ``ChatScreen`` flatten + ``restore_persisted_session``
path used by ``Tests/UI/test_console_resume_active_path.py``), swipe, and a
second resume. Also asserts the local-only active-leaf pointer never emits a
``sync_log`` row (Task 1's contract).

No shortcuts: every step drives the same store/controller methods the
production `ChatScreen` drives, and resume is a genuine persist -> drop ->
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

    Mirrors ``StreamingGateway`` in ``test_console_regenerate_branching.py``
    but supports multiple, distinguishable replies (one per controller call)
    so the E2E scenario can tell U1/A1, the regenerated A1', and A2 apart.
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
    session = store.create_session(title="Branch E2E")
    store.active_session_id = session.id
    return store, controller, session


def _resume_into_fresh_store(db: CharactersRAGDB, conversation_id: str):
    """Genuine persist -> drop -> resume round-trip via the real resume path.

    Mirrors ``_resume_into_store`` in ``Tests/UI/test_console_resume_active_path.py``:
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
        title="Branch E2E",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
    )
    return store, session


@pytest.mark.asyncio
async def test_console_branching_full_lifecycle_persist_resume_swipe():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        store, controller, session = _new_controller(
            db, replies=["A1", "A1-prime", "A2"]
        )

        # ---- Step 1: send U1 -> A1 (linear); one active path ----
        result1 = await controller.submit_draft("U1")
        assert result1.accepted is True
        transcript = store.messages_for_session(session.id)
        assert [m.content for m in transcript] == ["U1", "A1"]
        u1, a1 = transcript
        _sibs, _idx, count = store.siblings_at(a1.id)
        assert count == 1  # no siblings yet -- linear

        # ---- Step 2: regenerate A1 -> A1'; two siblings, active leaf = A1' ----
        result2 = await controller.regenerate_message(a1.id)
        assert result2.accepted is True
        a1_prime_id = store.active_leaf(session.id)
        assert a1_prime_id != a1.id
        a1_prime = store.get_message(a1_prime_id)
        assert a1_prime.content == "A1-prime"
        # `1/2` <-> `2/2` navigable from EITHER sibling's own perspective.
        a1_prime_sibs, a1_prime_index, a1_prime_count = store.siblings_at(a1_prime_id)
        assert a1_prime_count == 2 and a1_prime_index == 1  # "2/2"
        a1_sibs, a1_index, a1_count = store.siblings_at(a1.id)
        assert a1_count == 2 and a1_index == 0  # "1/2"
        assert {s.id for s in a1_prime_sibs} == {s.id for s in a1_sibs} == {
            a1.id,
            a1_prime_id,
        }
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "U1",
            "A1-prime",
        ]

        # ---- Step 3: continue on A1' -> U2 -> A2 ----
        result3 = await controller.submit_draft("U2")
        assert result3.accepted is True
        transcript = store.messages_for_session(session.id)
        assert [m.content for m in transcript] == ["U1", "A1-prime", "U2", "A2"]
        a2 = transcript[3]

        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None  # real persistence engaged throughout

        # ---- Step 4: persist, DROP the store, resume via the real resume path ----
        resumed_store, resumed_session = _resume_into_fresh_store(db, conversation_id)
        resumed_transcript = resumed_store.messages_for_session(resumed_session.id)
        assert [m.content for m in resumed_transcript] == [
            "U1",
            "A1-prime",
            "U2",
            "A2",
        ]
        resumed_leaf_id = resumed_store.active_leaf(resumed_session.id)
        assert resumed_store.get_message(resumed_leaf_id).content == "A2"
        assert resumed_store.get_message(resumed_leaf_id).persisted_message_id == (
            a2.persisted_message_id
        )

        # Off-path sibling (A1) was loaded too -- swipe is navigable post-resume.
        resumed_a1_prime = resumed_transcript[1]
        assert resumed_a1_prime.content == "A1-prime"
        resumed_sibs, _idx, resumed_count = resumed_store.siblings_at(
            resumed_a1_prime.id
        )
        assert resumed_count == 2
        resumed_a1 = next(s for s in resumed_sibs if s.id != resumed_a1_prime.id)
        assert resumed_a1.content == "A1"

        # ---- Step 5: swipe A1' <-> A1; the tail swaps ----
        resumed_store.set_active_leaf(
            resumed_session.id, resumed_store._leaf_under(resumed_a1.id)
        )
        swapped_view = [
            m.content for m in resumed_store.messages_for_session(resumed_session.id)
        ]
        assert swapped_view == ["U1", "A1"]  # A1's own tail (U2/A2) never existed

        # The swipe choice survives a SECOND resume (a fresh store, again).
        resumed_store_2, resumed_session_2 = _resume_into_fresh_store(
            db, conversation_id
        )
        second_view = [
            m.content
            for m in resumed_store_2.messages_for_session(resumed_session_2.id)
        ]
        assert second_view == ["U1", "A1"]
        # ...and A1-prime's own branch (U2/A2) is still off-path but intact.
        second_a1 = resumed_store_2.messages_for_session(resumed_session_2.id)[1]
        second_sibs, _idx, second_count = resumed_store_2.siblings_at(second_a1.id)
        assert second_count == 2
        second_a1_prime = next(s for s in second_sibs if s.id != second_a1.id)
        resumed_store_2.set_active_leaf(
            resumed_session_2.id,
            resumed_store_2._leaf_under(second_a1_prime.id),
        )
        assert [
            m.content
            for m in resumed_store_2.messages_for_session(resumed_session_2.id)
        ] == ["U1", "A1-prime", "U2", "A2"]

        # ---- Step 6: the local-only active-leaf pointer never touches sync_log ----
        # By this point the store has write-through'd the active-leaf pointer
        # many times over (every append that lands on the leaf, every
        # create_sibling, and the explicit swipe above) -- none of that may
        # ever surface as a `sync_log` row (Task 1's contract: a bare UPDATE
        # that bumps neither `version` nor `last_modified`, so the
        # `conversations_sync_update` trigger's WHEN clause never fires).
        with db.get_connection() as conn:
            conversation_sync_rows = conn.execute(
                "SELECT operation, payload FROM sync_log "
                "WHERE entity = 'conversations' AND entity_id = ?",
                (conversation_id,),
            ).fetchall()
        # Exactly the ONE legitimate row: the conversation's own creation.
        # No 'update' rows at all -- this flow never calls update_conversation.
        assert [row["operation"] for row in conversation_sync_rows] == ["create"]
        for row in conversation_sync_rows:
            assert "active_leaf_message_id" not in row["payload"]
        # Belt-and-braces: the active-leaf column must never appear in ANY
        # sync_log payload (messages included), for any entity.
        with db.get_connection() as conn:
            all_payloads = conn.execute("SELECT payload FROM sync_log").fetchall()
        assert all(
            "active_leaf_message_id" not in row["payload"] for row in all_payloads
        )
    finally:
        db.close_connection()
