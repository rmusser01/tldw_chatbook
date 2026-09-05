"""End-to-end integration test for Console `/rewind` (SP2, Task 4).

Exercises the whole `/rewind` lifecycle over the REAL stack -- a per-test,
file-backed ``CharactersRAGDB`` shared by event-loop and worker-thread
connections behind the real ``ChatPersistenceService`` /
``ChatConversationService``, a real ``ConsoleChatStore``, and the real
``ConsoleChatController`` (with a fake streaming provider gateway that also
serves the non-streaming summarize seam, mirroring
``Tests/Chat/test_console_rewind_summarize.py``'s ``SummaryGateway``).

Coverage includes the persisted U1/A1/U2/A2 restore, edit, summarize, and
resume flow, plus the before-first cursor lifecycle across fresh stores:
restoring an empty active path, hydrating the current durable prompt directly
into the session draft, keeping unsent draft edits session-only, accepting an
edited resend as a new canonical root while clearing the marker, and recovering
both the selected branch and the unchanged original branch after restart.

No shortcuts: every step drives the same store/controller methods the
production ``ChatScreen`` drives, and resume is a genuine persist -> drop ->
reload round-trip against the real database, not a hand-rolled fake. The
screen-only composer widget insertion is outside this harness, but the
production resume state is covered directly through assertions on the restored
session draft.
"""

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import (
    NATIVE_MESSAGE_ID_KEY,
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prepared_request import (
    MEMORY_OPEN_TAG,
    PreparedConsoleRequest,
    PreparedProviderRequest,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_destination_shells import _build_test_app
from Tests.console_provider_doubles import provider_resolution


class _SequencedCapturingGateway:
    """Fake provider gateway: streams one scripted reply per call, in order,
    and records the exact outgoing message list for EVERY call.

    Real sends stream; summaries use auxiliary completion. Both consume the
    same scripted sequence and retain the exact outgoing payload for assertions.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []
        self.prepared_calls: list[PreparedProviderRequest] = []

    async def resolve_for_send(self, selection):
        destination = provider_resolution(
            base_url="http://127.0.0.1:9099"
        ).resolved_destination
        return ConsoleProviderResolution(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            max_tokens=512,
            resolved_destination=destination,
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        if isinstance(messages, PreparedProviderRequest):
            self.prepared_calls.append(messages)
            messages = thaw_json(messages.messages_payload)
        self.calls.append(messages)
        text = self._replies[len(self.calls) - 1]
        yield text

    def prepare_chat_request(
        self, resolution, messages, *, tools=None, apply_safety_window=True, **kwargs
    ):
        semantic = (
            messages
            if isinstance(messages, PreparedConsoleRequest)
            else build_console_request(messages, tools=tools or ())
        )
        return prepare_provider_request(
            semantic,
            wire_style="single_preamble",
            model=resolution.model,
            provider=resolution.provider,
            capacity=resolve_request_capacity(
                context_window_tokens=50_000,
                requested_response_tokens=resolution.max_tokens or 512,
            ),
            count_fn=lambda rows, _model: sum(
                len(str(row.get("content", "")).split()) + 2 for row in rows
            ),
            apply_safety_window=apply_safety_window,
        )

    async def complete_auxiliary(self, request):
        self.calls.append([dict(message) for message in request.messages])
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=request.resolution.model,
            text=self._replies[len(self.calls) - 1],
        )


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
    ``db.get_conversation_active_cursor`` cursor-pair read, and a brand new
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
    all_nodes = screen._message._console_messages_from_conversation_tree(tree)
    active_leaf_id, before_message_id = db.get_conversation_active_cursor(
        conversation_id
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Rewind E2E",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
        active_leaf_before_persisted_id=before_message_id,
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
async def test_before_first_survives_restart_then_resend_clears_marker(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chat.db"), "test_client")
    try:
        store, controller, session, _gateway = _new_controller(db, ["A1"])
        assert (await controller.submit_draft("U1")).accepted is True
        await store.hydrate_session_library_policy(session.id)
        original = store.messages_for_session(session.id)
        root = original[0]
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        assert store.set_active_path_before(session.id, root.id) is True
        assert db.get_conversation_active_cursor(conversation_id) == (
            None,
            root.persisted_message_id,
        )

        resumed, resumed_session = _resume_into_fresh_store(db, conversation_id)
        assert resumed.active_path_message_ids(resumed_session.id) == []
        assert resumed.session_draft(resumed_session.id) == "U1"
        resumed.set_session_draft(resumed_session.id, "U1 edited")
        await resumed.hydrate_session_library_policy(resumed_session.id)

        resumed_controller = ConsoleChatController(
            store=resumed,
            provider_gateway=_SequencedCapturingGateway(["A1 edited"]),
        )
        assert (await resumed_controller.submit_draft("U1 edited")).accepted is True
        active_leaf, before = db.get_conversation_active_cursor(conversation_id)
        assert active_leaf is not None
        assert before is None

        restarted, restarted_session = _resume_into_fresh_store(db, conversation_id)
        assert [
            message.content
            for message in restarted.messages_for_session(restarted_session.id)
        ] == ["U1 edited", "A1 edited"]
        active_root = restarted.messages_for_session(restarted_session.id)[0]
        roots, _index, count = restarted.siblings_at(active_root.id)
        assert count == 2
        old_root = next(root for root in roots if root.content == "U1")
        restarted.set_active_leaf(
            restarted_session.id,
            restarted._leaf_under(old_root.id),
        )
        assert [
            message.content
            for message in restarted.messages_for_session(restarted_session.id)
        ] == ["U1", "A1"]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_before_first_unsent_draft_edit_is_session_only(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chat.db"), "test_client")
    try:
        store, controller, session, _gateway = _new_controller(db, ["A1"])
        assert (await controller.submit_draft("U1")).accepted is True
        root = store.messages_for_session(session.id)[0]
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None
        assert store.set_active_path_before(session.id, root.id) is True

        resumed, resumed_session = _resume_into_fresh_store(db, conversation_id)
        assert resumed.active_path_message_ids(resumed_session.id) == []
        assert resumed.session_draft(resumed_session.id) == "U1"
        resumed.set_session_draft(resumed_session.id, "unsent local edit")
        assert resumed.session_draft(resumed_session.id) == "unsent local edit"

        del resumed, resumed_session
        restarted, restarted_session = _resume_into_fresh_store(db, conversation_id)
        assert restarted.active_path_message_ids(restarted_session.id) == []
        assert restarted.session_draft(restarted_session.id) == "U1"
        assert db.get_conversation_active_cursor(conversation_id) == (
            None,
            root.persisted_message_id,
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_console_rewind_restore_edit_summarize_resume_leak_rule(tmp_path):
    # The real compaction guard requires savings after the memory preamble.
    u1_text = "U1 " + "original user context " * 40
    a1_text = "A1 " + "original assistant context " * 40
    db = CharactersRAGDB(str(tmp_path / "chat.db"), "test_client")
    try:
        store, controller, session, gateway = _new_controller(
            db, replies=[a1_text, "A2", "A2-prime", "SUMMARY TEXT", "A3"]
        )

        # ---- Step 1: converse U1 -> A1 -> U2 -> A2 (persisted) ----
        result1 = await controller.submit_draft(u1_text)
        assert result1.accepted is True
        await store.hydrate_session_library_policy(session.id)
        result2 = await controller.submit_draft("U2")
        assert result2.accepted is True
        transcript = store.messages_for_session(session.id)
        assert [m.content for m in transcript] == [u1_text, a1_text, "U2", "A2"]
        u1, a1, u2, a2 = transcript
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None  # real persistence engaged throughout

        # ---- Step 2: restore to U2 ----
        # id-lookup -> set_active_leaf, exactly what the screen callback does.
        target = _restore_to_prompt(store, session.id, u2.id)
        assert target == a1.id
        assert [m.content for m in store.messages_for_session(session.id)] == [
            u1_text,
            a1_text,
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
            u1_text,
            a1_text,
            "U2-edited",
            "A2-prime",
        ]

        # ---- Step 4: summarize up to the new tip's prompt ----
        snapshots = controller._durable_context_snapshots(session.id)
        assert snapshots is not None
        assert snapshots[0].parent_message_id is None
        assert all(
            child.parent_message_id == parent.message_id
            for parent, child in zip(snapshots[:-1], snapshots[1:], strict=True)
        ), [(row.message_id, row.parent_message_id) for row in snapshots]
        summarize_result = await controller.summarize_up_to(u2_prime_id)
        assert summarize_result.accepted is True
        repository = controller._context_repository
        memory = repository.list_active_memories(conversation_id)[0]
        scope = repository.load_memory_scope(memory.memory_id)
        assert memory.summary_text == "SUMMARY TEXT"
        assert memory.boundary_message_id == a1.persisted_message_id
        assert scope.selection_anchor_message_id == u2_prime.persisted_message_id
        assert scope.coverage_kind.value == "prefix"
        assert store.session_context_summary(session.id) == (None, None)
        # The summarize span covered exactly U1/A1 (everything before the new
        # tip's prompt) -- never the edited prompt or its reply.
        summarize_span_text = gateway.calls[3][1]["content"]
        assert '"content":"U1 ' in summarize_span_text
        assert '"content":"A1 ' in summarize_span_text
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
        assert u1_text not in outgoing_texts
        assert a1_text not in outgoing_texts
        assert "U2-edited" in outgoing_texts
        assert "A2-prime" in outgoing_texts
        assert "U3" in outgoing_texts
        preamble = gateway.prepared_calls[-1].system_message
        assert MEMORY_OPEN_TAG in preamble
        assert "SUMMARY TEXT" in preamble
        # Meanwhile the store's own transcript view is the FULL, uncompacted
        # history -- compaction only ever touches the provider payload.
        full_transcript = [m.content for m in store.messages_for_session(session.id)]
        assert full_transcript == [
            u1_text,
            a1_text,
            "U2-edited",
            "A2-prime",
            "U3",
            "A3",
        ]

        # ---- Step 6: persist -> DROP the store -> resume ----
        resumed_store, resumed_session = _resume_into_fresh_store(db, conversation_id)
        resumed_transcript = resumed_store.messages_for_session(resumed_session.id)
        assert [m.content for m in resumed_transcript] == [
            u1_text,
            a1_text,
            "U2-edited",
            "A2-prime",
            "U3",
            "A3",
        ]
        resumed_u2_prime = resumed_transcript[2]
        assert resumed_u2_prime.content == "U2-edited"

        # Branch memory keeps durable ownership across newly allocated native IDs.
        assert resumed_u2_prime.id != u2_prime_id
        assert (
            resumed_u2_prime.persisted_message_id == scope.selection_anchor_message_id
        )
        assert resumed_store.session_context_summary(resumed_session.id) == (None, None)

        # Next payload (post-resume) is still compacted.
        resumed_controller = ConsoleChatController(
            store=resumed_store,
            provider_gateway=_SequencedCapturingGateway(["unused"]),
        )
        resumed_payload = resumed_controller._provider_messages_for_session(
            resumed_session.id, annotate_ids=True
        )
        effective, resumed_projection = (
            resumed_controller._project_session_effective_memory(
                resumed_session.id, resumed_payload
            )
        )
        assert effective.memory.memory_id == memory.memory_id
        assert effective.memory.summary_text == "SUMMARY TEXT"
        assert (
            effective.scope.selection_anchor_message_id
            == resumed_u2_prime.persisted_message_id
        )
        resumed_compacted = [*resumed_projection.memory, *resumed_projection.rows]
        resumed_texts = _payload_texts(resumed_compacted)
        assert u1_text not in resumed_texts
        assert a1_text not in resumed_texts
        assert "U2-edited" in resumed_texts
        assert resumed_compacted[0]["role"] == "system"
        assert MEMORY_OPEN_TAG in resumed_compacted[0]["content"]

        # ---- Step 7: restore to before the boundary -> summary inert ----
        _restore_to_prompt(resumed_store, resumed_session.id, resumed_u2_prime.id)
        assert [
            m.content for m in resumed_store.messages_for_session(resumed_session.id)
        ] == [u1_text, a1_text]
        # Durable memory remains stored, but its selection anchor is off-path.
        assert repository.list_active_memories(conversation_id)[0] == memory
        assert resumed_u2_prime.id not in resumed_store.active_path_message_ids(
            resumed_session.id
        )
        inert_payload = resumed_controller._provider_messages_for_session(
            resumed_session.id, annotate_ids=True
        )
        inert_effective, inert_projection = (
            resumed_controller._project_session_effective_memory(
                resumed_session.id, inert_payload
            )
        )
        assert inert_effective.memory is None
        assert inert_projection.memory == ()
        inert_compacted = list(inert_projection.rows)
        # Byte-identical to the uncompacted payload -- the leak rule: a
        # summary covering LATER turns never reaches this EARLIER point.
        assert inert_compacted == inert_payload
        inert_texts = _payload_texts(inert_compacted)
        assert u1_text in inert_texts and a1_text in inert_texts
        assert not any(MEMORY_OPEN_TAG in text for text in inert_texts)

        # ---- Step 8: sync_log purity for the summary writes ----
        # Branch-memory writes remain local and do not rewrite legacy summary
        # fields or emit synchronization payloads.
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
