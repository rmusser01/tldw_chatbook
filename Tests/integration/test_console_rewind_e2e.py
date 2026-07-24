"""End-to-end integration test for Console `/rewind` (SP2, Task 4).

Exercises the whole `/rewind` lifecycle over the REAL stack -- an in-memory
``CharactersRAGDB`` behind the real ``ChatPersistenceService`` /
``ChatConversationService``, a real ``ConsoleChatStore``, and the real
``ConsoleChatController`` (with a fake streaming provider gateway that also
serves the non-streaming summarize seam, mirroring
``Tests/Chat/test_console_rewind_summarize.py``'s ``SummaryGateway``) --
through: converse (U1/A1/U2/A2, persisted) -> restore-to-here (Task 1's
id-lookup + ``set_active_leaf`` rule, driven directly as the screen callback
drives it) -> send an edited prompt (forks a sibling -- SP1 interplay) ->
summarize-up-to-here (Task 3) -> a compacted next-send payload built via the
real send path with a gateway spy -> persist -> DROP the store -> resume (via
the real ``ChatScreen`` flatten + ``restore_persisted_session`` path used by
``Tests/integration/test_console_branching_e2e.py``) -> restore to before the
boundary (the leak rule's inert case) -> ``sync_log`` purity for the summary
writes (Task 2's contract).

No shortcuts: every step drives the same store/controller methods the
production ``ChatScreen`` drives, and resume is a genuine persist -> drop ->
reload round-trip against the real database, not a hand-rolled fake. The
composer-refill step (screen-only) is simulated per the brief: asserting the
full original prompt text is retrievable via ``store.get_message(...).content``,
since ``_insert_prompt_text_into_composer`` lives on ``ChatScreen``, not the
store/controller pair this harness exercises directly.
"""

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import (
    NATIVE_MESSAGE_ID_KEY,
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_destination_shells import _build_test_app


class _SequencedCapturingGateway:
    """Fake provider gateway: streams one scripted reply per call, in order,
    and records the exact outgoing message list for EVERY call.

    Both real sends (``submit_draft``) and the non-streaming summarize seam
    (``summarize_up_to`` -> ``_collect_summary_completion``) flow through the
    same ``stream_chat`` method on the real gateway protocol, so a single
    sequenced/capturing fake -- mirroring ``_SequencedGateway`` in
    ``test_console_branching_e2e.py`` plus the message-capture from
    ``SummaryGateway`` in ``test_console_rewind_summarize.py`` -- covers both.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "max_tokens": 512,
                "visible_copy": "",
            },
        )()

    async def stream_chat(self, resolution, messages):
        self.calls.append(messages)
        text = self._replies[len(self.calls) - 1]
        yield text


def _new_controller(db: CharactersRAGDB, replies):
    """Real store (real DB-backed persistence) + real controller + fake gateway."""
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    gateway = _SequencedCapturingGateway(replies)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.create_session(title="Rewind E2E")
    store.active_session_id = session.id
    return store, controller, session, gateway


def _resume_into_fresh_store(db: CharactersRAGDB, conversation_id: str):
    """Genuine persist -> drop -> resume round-trip via the real resume path.

    Mirrors ``_resume_into_fresh_store`` in ``test_console_branching_e2e.py``
    / ``test_console_edit_resend_e2e.py``: the real
    ``ChatConversationService.get_conversation_tree`` full-tree read, the real
    ``ChatScreen._console_messages_from_conversation_tree`` flatten, the real
    ``db.get_conversation_active_leaf`` pointer read, and a brand new
    ``ConsoleChatStore`` fed through ``restore_persisted_session`` -- which
    also internally maps the persisted context-summary boundary back to a
    native id (``_resolve_context_summary_on_resume``, Task 2).
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
        title="Rewind E2E",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
    )
    return store, session


def _restore_to_prompt(store: ConsoleChatStore, session_id: str, prompt_id: str):
    """Drives the exact restore primitive ``_apply_console_rewind_choice`` uses.

    Task 1's id-lookup rule (never positional against ``messages_for_session``'s
    view): the target leaf is the node BEFORE ``prompt_id`` on the CURRENT
    active path, or ``None`` when ``prompt_id`` is the path's first node.
    """
    path = store.active_path_message_ids(session_id)
    index = path.index(prompt_id)
    target = path[index - 1] if index > 0 else None
    store.set_active_leaf(session_id, target)
    return target


def _payload_texts(messages):
    """Flatten each provider-message row's content to plain text."""
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.append(
                "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            )
        else:
            texts.append("")
    return texts


@pytest.mark.asyncio
async def test_console_rewind_restore_edit_summarize_resume_leak_rule():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        store, controller, session, gateway = _new_controller(
            db, replies=["A1", "A2", "A2-prime", "SUMMARY TEXT", "A3"]
        )

        # ---- Step 1: converse U1 -> A1 -> U2 -> A2 (persisted) ----
        result1 = await controller.submit_draft("U1")
        assert result1.accepted is True
        result2 = await controller.submit_draft("U2")
        assert result2.accepted is True
        transcript = store.messages_for_session(session.id)
        assert [m.content for m in transcript] == ["U1", "A1", "U2", "A2"]
        u1, a1, u2, a2 = transcript
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None  # real persistence engaged throughout

        # ---- Step 2: restore to U2 ----
        # id-lookup -> set_active_leaf, exactly what the screen callback does.
        target = _restore_to_prompt(store, session.id, u2.id)
        assert target == a1.id
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "U1",
            "A1",
        ]
        # Composer-refill step simulated: the full original prompt text is
        # retrievable (the screen would feed this into
        # `_insert_prompt_text_into_composer(..., replace=True)`).
        assert store.get_message(u2.id).content == "U2"
        # U2/A2 preserved OFF the active path, not deleted.
        assert u2.id not in store.active_path_message_ids(session.id)
        u2_siblings, _idx, u2_sib_count = store.siblings_at(u2.id)
        assert u2_sib_count == 1  # no fork yet -- still the only child of A1

        # ---- Step 3: send an edited prompt -> forks a sibling (SP1) ----
        # Active leaf is A1 (from the restore), so a plain send parents the
        # new USER node alongside U2 -- exactly SP1's fork-a-sibling behavior,
        # driven by the real send path (not `create_sibling`/`edit_and_resend`
        # directly).
        result3 = await controller.submit_draft("U2-edited")
        assert result3.accepted is True
        siblings, _idx, sib_count = store.siblings_at(u2.id)
        assert sib_count == 2
        sibling_ids = {s.id for s in siblings}
        u2_prime_id = next(iter(sibling_ids - {u2.id}))
        u2_prime = store.get_message(u2_prime_id)
        assert u2_prime.content == "U2-edited"
        active_path_ids = store.active_path_message_ids(session.id)
        assert u2_prime_id in active_path_ids
        assert u2.id not in active_path_ids  # old U2/A2 branch preserved off-path
        a2_prime_id = store.active_leaf(session.id)
        assert store.get_message(a2_prime_id).content == "A2-prime"
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "U1",
            "A1",
            "U2-edited",
            "A2-prime",
        ]

        # ---- Step 4: summarize up to the new tip's prompt ----
        summarize_result = await controller.summarize_up_to(u2_prime_id)
        assert summarize_result.accepted is True
        assert store.session_context_summary(session.id) == (
            "SUMMARY TEXT",
            u2_prime_id,
        )
        # The summarize span covered exactly U1/A1 (everything before the new
        # tip's prompt) -- never the edited prompt or its reply.
        summarize_span_text = gateway.calls[3][1]["content"]
        assert "User: U1" in summarize_span_text
        assert "Assistant: A1" in summarize_span_text
        assert "U2-edited" not in summarize_span_text

        # ---- Step 5: next-send payload is compacted (built via the real
        # send path, inspected through the gateway spy) ----
        result4 = await controller.submit_draft("U3")
        assert result4.accepted is True
        outgoing = gateway.calls[4]
        assert all(NATIVE_MESSAGE_ID_KEY not in row for row in outgoing)
        outgoing_texts = _payload_texts(outgoing)
        # Pre-boundary rows (U1/A1) are gone; boundary + tail (edited U2,
        # A2-prime, the new U3) are kept.
        assert "U1" not in outgoing_texts
        assert "A1" not in outgoing_texts
        assert "U2-edited" in outgoing_texts
        assert "A2-prime" in outgoing_texts
        assert "U3" in outgoing_texts
        assert outgoing[0]["role"] == "system"
        assert "[Summary of earlier conversation]" in outgoing[0]["content"]
        assert "SUMMARY TEXT" in outgoing[0]["content"]
        # Meanwhile the store's own transcript view is the FULL, uncompacted
        # history -- compaction only ever touches the provider payload.
        full_transcript = [
            m.content for m in store.messages_for_session(session.id)
        ]
        assert full_transcript == [
            "U1",
            "A1",
            "U2-edited",
            "A2-prime",
            "U3",
            "A3",
        ]

        # ---- Step 6: persist -> DROP the store -> resume ----
        resumed_store, resumed_session = _resume_into_fresh_store(
            db, conversation_id
        )
        resumed_transcript = resumed_store.messages_for_session(resumed_session.id)
        assert [m.content for m in resumed_transcript] == [
            "U1",
            "A1",
            "U2-edited",
            "A2-prime",
            "U3",
            "A3",
        ]
        resumed_u2_prime = resumed_transcript[2]
        assert resumed_u2_prime.content == "U2-edited"

        # Summary + boundary restored, mapped to the RESUMED store's new
        # native id for the same underlying (persisted) message.
        resumed_summary, resumed_boundary_id = resumed_store.session_context_summary(
            resumed_session.id
        )
        assert resumed_summary == "SUMMARY TEXT"
        assert resumed_boundary_id == resumed_u2_prime.id
        assert resumed_boundary_id != u2_prime_id  # a genuinely new native id

        # Banner state is derivable: the transcript renderer's own gate is
        # "boundary set AND on the active path" -- true here.
        assert resumed_boundary_id in resumed_store.active_path_message_ids(
            resumed_session.id
        )

        # Next payload (post-resume) is still compacted.
        resumed_controller = ConsoleChatController(
            store=resumed_store,
            provider_gateway=_SequencedCapturingGateway(["unused"]),
        )
        resumed_payload = resumed_controller._provider_messages_for_session(
            resumed_session.id, annotate_ids=True
        )
        resumed_compacted = resumed_controller._apply_context_summary_compaction(
            resumed_session.id, resumed_payload
        )
        resumed_texts = _payload_texts(resumed_compacted)
        assert "U1" not in resumed_texts
        assert "A1" not in resumed_texts
        assert "U2-edited" in resumed_texts
        assert resumed_compacted[0]["role"] == "system"
        assert "[Summary of earlier conversation]" in resumed_compacted[0]["content"]

        # ---- Step 7: restore to before the boundary -> summary inert ----
        restore_target = _restore_to_prompt(
            resumed_store, resumed_session.id, resumed_u2_prime.id
        )
        assert [
            m.content
            for m in resumed_store.messages_for_session(resumed_session.id)
        ] == ["U1", "A1"]
        # The stored summary/boundary is left in place (not cleared)...
        assert resumed_store.session_context_summary(resumed_session.id) == (
            "SUMMARY TEXT",
            resumed_boundary_id,
        )
        # ...but the boundary is no longer on the active path, so it is INERT.
        assert resumed_boundary_id not in resumed_store.active_path_message_ids(
            resumed_session.id
        )
        inert_payload = resumed_controller._provider_messages_for_session(
            resumed_session.id, annotate_ids=True
        )
        inert_compacted = resumed_controller._apply_context_summary_compaction(
            resumed_session.id, inert_payload
        )
        # Byte-identical to the uncompacted payload -- the leak rule: a
        # summary covering LATER turns never reaches this EARLIER point.
        assert inert_compacted == inert_payload
        inert_texts = _payload_texts(inert_compacted)
        assert "U1" in inert_texts and "A1" in inert_texts
        assert not any(
            "[Summary of earlier conversation]" in text for text in inert_texts
        )

        # ---- Step 8: sync_log purity for the summary writes ----
        # `set_conversation_context_summary` is a bare UPDATE (no version /
        # last_modified bump), so it must never emit a sync_log row -- same
        # local-only contract as the active-leaf pointer (Task 1/2).
        with db.get_connection() as conn:
            conversation_sync_rows = conn.execute(
                "SELECT operation, payload FROM sync_log "
                "WHERE entity = 'conversations' AND entity_id = ?",
                (conversation_id,),
            ).fetchall()
        assert [row["operation"] for row in conversation_sync_rows] == ["create"]
        for row in conversation_sync_rows:
            assert "context_summary" not in row["payload"]
            assert "summary_boundary_message_id" not in row["payload"]
        with db.get_connection() as conn:
            all_payloads = conn.execute("SELECT payload FROM sync_log").fetchall()
        assert all(
            "context_summary" not in row["payload"]
            and "summary_boundary_message_id" not in row["payload"]
            for row in all_payloads
        )
    finally:
        db.close_connection()
