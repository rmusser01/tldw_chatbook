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


def _fail_first_provider_call(gateway) -> None:
    """Make the first provider stream fail the way a dropped request does."""
    stream = gateway.stream_chat
    calls = 0

    async def stream_chat(resolution, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
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
    _fail_first_provider_call(gateway)

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
    assert boundary.eligible is False
    assert "saved history" in boundary.reason
    with pytest.raises(ValueError, match="saved history"):
        store.issue_fork_fence(later.id)
    # An earlier boundary walks the broken link on its way to the saved leaf.
    assert store.fork_eligibility(first_answer.id).eligible is False


def test_guide_never_advertises_a_refused_fork() -> None:
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")
    reason = 'The System note "x" above this message can\'t be copied into a fork.'

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
