"""TASK-33621.10: Fork works on freshly sent saved turns and past command notes.

Every normal (non-temporary) Console send goes through the durable-commit path
(``commit_durable_turn`` -> ``publish_durable_turn_owners``). These tests drive
that path with the real ``ConsoleChatController``, the real ``ConsoleChatStore``
and ``ChatPersistenceService`` on a temporary SQLite file -- only the provider
gateway is a double -- and then fork the freshly sent pair WITHOUT reloading the
conversation, all the way through the durable fork bundle commit.

The bug they pin: the live USER/ASSISTANT owners got their persisted ids but no
persisted parent (``parent_message_id`` stayed ``None``), so the assistant
boundary failed with "Saved Console fork parent is unavailable." and the user
boundary with "Saved active leaf lineage is unavailable." until a restart. And
unsaved command feedback (/help, /doctor, unknown-command hints) sat on the
active path as a SYSTEM node that refused every later fork with "Only user and
assistant messages can be forked." although ADR-092 defines the fork source as
the USER/ASSISTANT parent chain.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller as _base_controller
from Tests.Chat.test_console_provider_continuation import _active_checkpoint
from tldw_chatbook.Agents.agent_models import ContinuationEventContext, ToolBatchReady
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_fork import ConsoleForkEligibility
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_message_actions import (
    ConsoleMessageActionService,
    action_row_guide,
)
from tldw_chatbook.Chat.console_project_instructions import (
    encode_project_context_json,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

# The real ChatScreen/store goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired); keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile

_SESSION = "session-1"
# The literal body `/help` appends through `_append_native_console_system_message`
# is irrelevant here; what matters is the call shape: an unsaved SYSTEM node
# appended to the native tree at the active leaf.
_HELP_NOTE = "Commands: /help, /new, /clear, /settings, /doctor"


def _controller(tmp_path):
    # A real Console session always carries settings (the fork copies them).
    return _base_controller(
        tmp_path,
        initial_settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="test-model",
            streaming=False,
        ),
    )


def _active_messages(store: ConsoleChatStore) -> list[ConsoleChatMessage]:
    return [
        store.get_message(native_id)
        for native_id in store.active_path_message_ids(_SESSION)
    ]


def _last(store: ConsoleChatStore, role: ConsoleMessageRole) -> ConsoleChatMessage:
    return [message for message in _active_messages(store) if message.role is role][-1]


async def _send(controller, text: str) -> None:
    result = await controller.submit_draft(text, session_id=_SESSION)
    assert result.accepted is True, result


def _append_help_note(store: ConsoleChatStore) -> ConsoleChatMessage:
    # Same store call `_append_native_console_system_message` makes for /help,
    # /doctor and the unknown-command hint (persist defaults to False).
    note = store.append_message(
        _SESSION,
        role=ConsoleMessageRole.SYSTEM,
        content=_HELP_NOTE,
    )
    assert note.persisted_message_id is None
    return note


def _fork_and_commit(
    db: CharactersRAGDB,
    store: ConsoleChatStore,
    boundary: ConsoleChatMessage,
) -> list[dict[str, object]]:
    """Fork ``boundary`` exactly as the Fork dialog does, then commit it."""
    eligibility = store.fork_eligibility(boundary.id)
    assert eligibility.eligible is True, eligibility.reason
    fence = store.issue_fork_fence(boundary.id)
    snapshot = store.stage_fork_snapshot(
        fence,
        title="Fork",
        fork_session_id=str(uuid4()),
        fork_conversation_id=str(uuid4()),
    )
    persistence = store.persistence
    assert isinstance(persistence, ChatPersistenceService)
    result = persistence.fork_console_conversation_bundle(
        snapshot=snapshot,
        conversation_kwargs=ConsoleSessionController._fork_conversation_kwargs(
            snapshot
        ),
        policy_candidate=snapshot.configuration.library_policy,
        project_context_json=encode_project_context_json(
            snapshot.configuration.project_instruction_state
        ),
    )
    assert result is not None
    fork_session = store.register_fork_snapshot(snapshot, activate=False)
    assert fork_session.persisted_conversation_id == result.conversation_id
    conversation = db.get_conversation_by_id(result.conversation_id)
    assert conversation["forked_from_message_id"] == boundary.persisted_message_id
    rows = [
        dict(row)
        for row in db.get_messages_for_conversation(result.conversation_id, limit=50)
    ]
    # The fork is one linear USER/ASSISTANT parent chain.
    by_id = {row["id"]: row for row in rows}
    leaf = db.get_conversation_active_leaf(result.conversation_id)
    chain: list[dict[str, object]] = []
    while leaf is not None:
        chain.append(by_id[leaf])
        leaf = by_id[leaf]["parent_message_id"]
    chain.reverse()
    assert len(chain) == len(rows)
    return chain


@pytest.mark.asyncio
async def test_fresh_saved_pair_forks_from_both_boundaries_without_reload(
    tmp_path,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)

    await _send(controller, "Reply with exactly: hello there friend")

    assert gateway.calls == 1
    user = _last(store, ConsoleMessageRole.USER)
    assistant = _last(store, ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "complete"

    assistant_fork = _fork_and_commit(db, store, assistant)
    user_fork = _fork_and_commit(db, store, user)

    assert [(row["sender"], row["content"]) for row in assistant_fork] == [
        ("user", "Reply with exactly: hello there friend"),
        ("assistant", "done"),
    ]
    assert [(row["sender"], row["content"]) for row in user_fork] == [
        ("user", "Reply with exactly: hello there friend"),
    ]
    # The live owners mirror the durable rows' parent chain -- no reload.
    assert user.persisted_message_id is not None
    assert assistant.persisted_message_id is not None
    assert user.parent_message_id is None
    assert assistant.parent_message_id == user.persisted_message_id
    assert (
        db.get_message_by_id(assistant.persisted_message_id)["parent_message_id"]
        == user.persisted_message_id
    )


@pytest.mark.asyncio
async def test_second_fresh_turn_links_to_the_first_and_forks_mid_chat(
    tmp_path,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)

    await _send(controller, "first question")
    first_answer = _last(store, ConsoleMessageRole.ASSISTANT)
    await _send(controller, "second question")
    second_user = _last(store, ConsoleMessageRole.USER)

    # A mid-chat boundary also walks the fresh tail after it to the saved leaf.
    chain = _fork_and_commit(db, store, first_answer)
    assert [row["content"] for row in chain] == ["first question", "done"]
    chain = _fork_and_commit(db, store, _last(store, ConsoleMessageRole.ASSISTANT))
    assert [row["content"] for row in chain] == [
        "first question",
        "done",
        "second question",
        "done",
    ]
    assert second_user.parent_message_id == first_answer.persisted_message_id


@pytest.mark.asyncio
async def test_help_then_send_then_fork_excludes_the_help_note(tmp_path) -> None:
    db, store, controller, _gateway = _controller(tmp_path)

    help_note = _append_help_note(store)
    await _send(controller, "Reply with exactly: hello there friend")

    assert help_note.id in store.active_path_message_ids(_SESSION)
    for boundary in (
        _last(store, ConsoleMessageRole.ASSISTANT),
        _last(store, ConsoleMessageRole.USER),
    ):
        chain = _fork_and_commit(db, store, boundary)
        assert _HELP_NOTE not in [row["content"] for row in chain]
        assert all(row["sender"] in {"user", "assistant"} for row in chain)


@pytest.mark.asyncio
async def test_help_between_turns_never_blocks_a_later_or_earlier_fork(
    tmp_path,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)

    await _send(controller, "first question")
    first_user = _last(store, ConsoleMessageRole.USER)
    _append_help_note(store)
    await _send(controller, "second question")
    # Command feedback AFTER the last reply, too (the active leaf is a note).
    _append_help_note(store)

    later = _fork_and_commit(db, store, _last(store, ConsoleMessageRole.ASSISTANT))
    assert [row["content"] for row in later] == [
        "first question",
        "done",
        "second question",
        "done",
    ]
    # An earlier boundary walks PAST the note to the saved active leaf.
    earlier = _fork_and_commit(db, store, first_user)
    assert [row["content"] for row in earlier] == ["first question"]


@pytest.mark.asyncio
async def test_saved_system_note_between_turns_is_left_out_of_the_fork(
    tmp_path,
) -> None:
    """A SYSTEM row that IS saved (a promoted /help note, image-edit failure
    guidance) sits in the durable parent chain; the fork still copies only the
    USER/ASSISTANT chain, and the atomic commit's source re-check accepts it.
    """
    db, store, controller, _gateway = _controller(tmp_path)

    await _send(controller, "first question")
    note = store.append_message(
        _SESSION,
        role=ConsoleMessageRole.SYSTEM,
        content="Image edit failed: the provider rejected the mask.",
        persist=True,
    )
    await _send(controller, "second question")
    second_user = _last(store, ConsoleMessageRole.USER)
    assert note.persisted_message_id is not None
    assert (
        db.get_message_by_id(second_user.persisted_message_id)["parent_message_id"]
        == note.persisted_message_id
    )

    chain = _fork_and_commit(db, store, _last(store, ConsoleMessageRole.ASSISTANT))

    assert [row["content"] for row in chain] == [
        "first question",
        "done",
        "second question",
        "done",
    ]


@pytest.mark.asyncio
async def test_fork_commit_recheck_walks_past_saved_system_rows_only(
    tmp_path,
) -> None:
    """Negative control for the commit-time source re-check's relaxation."""
    db, store, controller, _gateway = _controller(tmp_path)
    await _send(controller, "first question")
    first_answer = _last(store, ConsoleMessageRole.ASSISTANT)
    note = store.append_message(
        _SESSION,
        role=ConsoleMessageRole.SYSTEM,
        content="Image edit failed: the provider rejected the mask.",
        persist=True,
    )
    await _send(controller, "second question")
    second_user = _last(store, ConsoleMessageRole.USER)
    conversation_id = store.sessions()[0].persisted_conversation_id
    saved_parent = db.get_message_by_id(second_user.persisted_message_id)[
        "parent_message_id"
    ]
    walk = ChatPersistenceService._fork_source_parent
    connection = db.get_connection()

    assert saved_parent == note.persisted_message_id
    assert (
        walk(
            connection.cursor(),
            conversation_id,
            saved_parent,
            first_answer.persisted_message_id,
        )
        == first_answer.persisted_message_id
    )
    # A USER row is conversation content: the walk never skips it, so a copied
    # row whose saved parent is an UNcopied user message still fails the check.
    assert (
        walk(
            connection.cursor(),
            conversation_id,
            second_user.persisted_message_id,
            first_answer.persisted_message_id,
        )
        == second_user.persisted_message_id
    )


def _fail_provider_call(gateway, failing_call: int) -> None:
    """Make one provider stream fail the way a dropped request does."""
    stream = gateway.stream_chat
    calls = 0

    async def stream_chat(resolution, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise RuntimeError("provider dropped the request")
        async for chunk in stream(resolution, messages, **kwargs):
            yield chunk

    gateway.stream_chat = stream_chat


@pytest.mark.asyncio
async def test_refusal_names_the_blocking_row_and_the_nearest_boundary(
    tmp_path,
) -> None:
    """A failed reply with no text is a genuine blocker (it cannot be copied).

    The refusal must say which row blocks and where the user can fork instead,
    not a generic sentence about the selected message.
    """
    _db, store, controller, gateway = _controller(tmp_path)
    _fail_provider_call(gateway, 1)

    await controller.submit_draft("first question", session_id=_SESSION)
    failed = _last(store, ConsoleMessageRole.ASSISTANT)
    assert (failed.status, failed.content) == ("failed", "")
    await _send(controller, "second question")
    later = _last(store, ConsoleMessageRole.ASSISTANT)
    assert later.status == "complete"

    eligibility = store.fork_eligibility(later.id)

    assert eligibility == ConsoleForkEligibility(
        False,
        "The failed Assistant reply above this message has no text to copy. "
        'Fork from the User message "first question" instead.',
    )
    with pytest.raises(ValueError) as refused:
        store.issue_fork_fence(later.id)
    assert str(refused.value) == eligibility.reason
    # The named fallback really is forkable.
    assert store.fork_eligibility(_active_messages(store)[0].id) == (
        ConsoleForkEligibility(True)
    )
    # Selecting the blocking row itself names the same fallback: the reason
    # alone ("no content") would leave the user with no next step.
    assert store.fork_eligibility(failed.id) == ConsoleForkEligibility(
        False,
        "This partial response has no content to fork. "
        'Fork from the User message "first question" instead.',
    )


@pytest.mark.asyncio
async def test_refusal_names_the_nearest_boundary_not_the_first(tmp_path) -> None:
    _db, store, controller, gateway = _controller(tmp_path)
    _fail_provider_call(gateway, 2)

    await _send(controller, "first question")
    await controller.submit_draft("second question", session_id=_SESSION)
    failed = _last(store, ConsoleMessageRole.ASSISTANT)
    assert (failed.status, failed.content) == ("failed", "")
    await _send(controller, "third question")

    assert store.fork_eligibility(
        _last(store, ConsoleMessageRole.ASSISTANT).id
    ) == ConsoleForkEligibility(
        False,
        "The failed Assistant reply above this message has no text to copy. "
        'Fork from the User message "second question" instead.',
    )


def test_discarded_reply_is_named_by_its_state_not_its_placeholder_text() -> None:
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        ephemeral=True,
    )
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    discarded = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Response discarded."
    )
    store._nodes_by_session[session.id][discarded.id].status = "discarded"
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q2")
    later = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )

    assert store.fork_eligibility(later.id) == ConsoleForkEligibility(
        False,
        "The discarded Assistant reply above this message can't be copied into "
        'a fork. Fork from the User message "q1" instead.',
    )
    assert store.fork_eligibility(discarded.id) == ConsoleForkEligibility(
        False,
        'Discarded messages cannot be forked. Fork from the User message "q1" instead.',
    )


def test_quoted_row_text_is_never_parsed_as_markup() -> None:
    """A refusal quotes user/model text into markup-parsing surfaces.

    The action guide ``Static``, the Fork button's tooltip and ``notify``
    toasts all render a plain string through ``Content.from_markup``, so an
    unescaped ``[/]`` raised ``MarkupError`` and ``[@click=...]`` added a link.
    """
    from textual.content import Content

    question = "[/] [b]b[/b] [@click=app.quit]x[/]"
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        ephemeral=True,
    )
    store.append_message(session.id, role=ConsoleMessageRole.USER, content=question)
    discarded = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Response discarded."
    )
    store._nodes_by_session[session.id][discarded.id].status = "discarded"
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q2")
    later = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )

    for message_id, lead in (
        (discarded.id, "Discarded messages cannot be forked."),
        (
            later.id,
            "The discarded Assistant reply above this message can't be copied "
            "into a fork.",
        ),
    ):
        painted = Content.from_markup(store.fork_eligibility(message_id).reason)
        assert painted.plain == (
            f'{lead} Fork from the User message "{question}" instead.'
        )
        assert painted.spans == []


@pytest.mark.asyncio
async def test_temporary_chat_help_between_turns_forks_the_whole_exchange(
    tmp_path,
) -> None:
    """AC#2 covers temporary chats too: the note is left out, never a blocker."""
    _db, store, _controller_, _gateway = _controller(tmp_path)
    session = store.create_session(
        session_id="temporary-1",
        title="Temporary",
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="test-model", streaming=False
        ),
        ephemeral=True,
    )
    for role, content in (
        (ConsoleMessageRole.USER, "q1"),
        (ConsoleMessageRole.ASSISTANT, "a1"),
        (ConsoleMessageRole.SYSTEM, _HELP_NOTE),
        (ConsoleMessageRole.USER, "q2"),
        (ConsoleMessageRole.ASSISTANT, "a2"),
    ):
        boundary = store.append_message(session.id, role=role, content=content)

    assert store.fork_eligibility(boundary.id) == ConsoleForkEligibility(True)
    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(boundary.id),
        title="Fork",
        fork_session_id=str(uuid4()),
        fork_conversation_id=None,
    )
    fork = store.register_fork_snapshot(snapshot, activate=False)

    assert [
        (store.get_message(native_id).role, store.get_message(native_id).content)
        for native_id in store.active_path_message_ids(fork.id)
    ] == [
        (ConsoleMessageRole.USER, "q1"),
        (ConsoleMessageRole.ASSISTANT, "a1"),
        (ConsoleMessageRole.USER, "q2"),
        (ConsoleMessageRole.ASSISTANT, "a2"),
    ]


@pytest.mark.asyncio
async def test_help_between_sends_leaves_one_parent_chain_for_compaction(
    tmp_path,
) -> None:
    """The same missing live lineage broke automatic compaction (GAP2-01)."""
    _db, store, controller, _gateway = _controller(tmp_path)

    await _send(controller, "first question")
    _append_help_note(store)
    await _send(controller, "second question")
    snapshots = controller._durable_context_snapshots(_SESSION)

    assert snapshots is not None
    assert [snapshot.role for snapshot in snapshots] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert snapshots[0].parent_message_id is None
    for parent, child in zip(snapshots, snapshots[1:]):
        assert child.parent_message_id == parent.message_id


@pytest.mark.asyncio
async def test_continuation_owner_saved_by_its_first_tool_batch_mirrors_its_parent(
    tmp_path,
) -> None:
    """The other first-save path (a tool batch before any text) links too."""
    db = CharactersRAGDB(tmp_path / "continuation.sqlite", "fork-continuation")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(
            title="Durable continuation",
            settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        )
        user = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Use it", persist=True
        )
        owner = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        assert owner.persisted_message_id is None
        store.persist_provider_continuation_event(
            ToolBatchReady(
                ContinuationEventContext(owner.id, "run", "primary", "persistent"),
                _active_checkpoint(),
                None,
            )
        )

        live = store.get_message(owner.id)
        assert live.persisted_message_id == owner.id
        assert live.parent_message_id == user.persisted_message_id
        assert (
            db.get_message_by_id(owner.id)["parent_message_id"]
            == user.persisted_message_id
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_eligibility_refuses_whatever_the_fence_would_refuse(tmp_path) -> None:
    """The Fork button is enabled only when opening the dialog would succeed.

    Before TASK-33621.10 eligibility never checked a row's saved parent link,
    so an assistant row whose link was missing advertised Fork and then failed
    with "Saved Console fork parent is unavailable." Break the live link the
    way the old durable-commit path left it and require both to refuse.
    """
    _db, store, controller, _gateway = _controller(tmp_path)
    await _send(controller, "first question")
    first_answer = _last(store, ConsoleMessageRole.ASSISTANT)
    await _send(controller, "second question")
    later = _last(store, ConsoleMessageRole.ASSISTANT)
    store._nodes_by_session[_SESSION][later.id].parent_message_id = None

    boundary = store.fork_eligibility(later.id)
    # Plain language, no "reopen the chat" promise a reload cannot keep (a
    # legacy flat-parent chat reloads with the same missing link).
    assert boundary == ConsoleForkEligibility(
        False,
        "This message isn't linked to the message before it in the saved chat, "
        "so it can't be forked. No earlier message can be forked.",
    )
    with pytest.raises(ValueError) as refused:
        store.issue_fork_fence(later.id)
    assert str(refused.value) == boundary.reason
    # An earlier boundary walks the broken link on its way to the saved leaf;
    # the refusal names that row rather than "saved active leaf lineage".
    assert store.fork_eligibility(first_answer.id) == ConsoleForkEligibility(
        False,
        'The Assistant reply "done" further down isn\'t linked to the message '
        "before it in the saved chat, so no message up to it can be forked.",
    )


def test_guide_never_advertises_a_refused_fork() -> None:
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")
    reason = "The failed Assistant reply above this message has no text to copy."

    refused = action_row_guide(
        service.available_actions(
            message,
            fork_eligibility=ConsoleForkEligibility(False, reason),
        )
    )
    allowed = action_row_guide(service.available_actions(message))

    assert "f Fork" not in refused
    assert "e Edit" in refused and "r ♻ Regenerate" in refused
    assert "f Fork" in allowed
