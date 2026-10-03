"""TASK-33662: a dispatch-recovery card settles after a REAL relaunch.

A relaunch rebuilds the session through the production hydration
(``hydrate_console_session``: the conversation tree, then the durable Library
policy), which gives restored nodes fresh native ids. The recovery addresses
its owner rows by persisted id, so before the fix Retry and Discard both
refused with "That response recovery action is unavailable." ``_restored_store`` in
``test_console_dispatch_recovery.py`` hand-builds nodes whose native id IS the
persisted id, which is why that suite never saw it.

"Crash mid-reply" is real here: a real controller sends a turn whose provider
hangs, and the SQLite file is copied at that instant (what a killed process
leaves on disk). The relaunch restores from that copy.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _continuation,
    _deepseek_acceptance,
    _install_legacy_owner,
)
from Tests.Chat.test_console_dispatch_recovery import (
    DISCARD_COPY,
    _acceptance,
    _authority,
    _database,
    _insert,
)
from Tests.Chat.test_console_turn_resend import (
    REPLY,
    _Gateway,
    _console,
    _controller,
    _path,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryKind,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import hydrate_console_session
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_turn_resend import resend_target_id, resend_turn
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# Drives the real controller and store (lessons-testing-evidence.md).
pytestmark = pytest.mark.bootstrap_profile

USER = ConsoleMessageRole.USER
ASSISTANT = ConsoleMessageRole.ASSISTANT


@pytest.fixture
def databases():
    opened: list[CharactersRAGDB] = []
    yield opened
    for db in reversed(opened):
        db.close()


async def _relaunch(db, conversation_id) -> SimpleNamespace:
    """A fresh store and controller, hydrated the way the app opens a chat."""
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = await hydrate_console_session(
        app=SimpleNamespace(chachanotes_db=db),
        store=store,
        conversation_id=conversation_id,
        tree=ChatConversationService(db).get_conversation_tree(
            conversation_id, root_limit=100, depth_cap=100
        ),
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
    )
    gateway = _Gateway("ok")
    return SimpleNamespace(
        db=db,
        store=store,
        session_id=session.id,
        gateway=gateway,
        controller=_controller(store, gateway),
    )


async def _crash_mid_reply(tmp_path, databases):
    """Send one healthy turn, then copy the database while turn two hangs."""
    console = _console(tmp_path, databases)
    assert (await console.controller.submit_draft("first question")).accepted
    console.gateway.mode = "hang"
    task = asyncio.create_task(console.controller.submit_draft("second question"))
    await asyncio.wait_for(console.gateway.started.wait(), 5)
    conversation_id = console.store._sessions[
        console.session_id
    ].persisted_conversation_id
    crashed = sqlite3.connect(tmp_path / "crashed.sqlite")
    console.db.get_connection().backup(crashed)
    crashed.close()
    console.gateway.release.set()
    await asyncio.wait_for(task, 5)
    db = CharactersRAGDB(tmp_path / "crashed.sqlite", client_id="relaunch-test")
    databases.append(db)
    return await _relaunch(db, conversation_id)


def _assert_fresh_native_ids_for_the_settled_turn(relaunched) -> None:
    """The production hydration really ran: the healthy turn got new ids."""
    first_user, first_reply = _path(relaunched)[:2]
    assert first_user.content == "first question"
    assert first_user.id != first_user.persisted_message_id
    assert first_reply.id != first_reply.persisted_message_id


@pytest.mark.asyncio
async def test_relaunch_after_a_crash_mid_reply_retry_streams_into_the_pending_reply(
    tmp_path, databases
):
    relaunched = await _crash_mid_reply(tmp_path, databases)
    _assert_fresh_native_ids_for_the_settled_turn(relaunched)
    recovery = relaunched.store.dispatch_recovery_for_session(relaunched.session_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.DISPATCH_STARTED
    pending = _path(relaunched)[-1]
    assert pending.persisted_message_id == recovery.assistant_message_id

    result = await relaunched.controller.retry_dispatch_recovery(relaunched.session_id)

    assert result.accepted, result.visible_copy
    path = _path(relaunched)
    assert [(row.role, row.content) for row in path] == [
        (USER, "first question"),
        (ASSISTANT, REPLY),
        (USER, "second question"),
        (ASSISTANT, REPLY),
    ]
    assert path[-1].persisted_message_id == pending.persisted_message_id
    assert path[-1].sibling_count == 1
    # The provider saw the whole history before the pending reply.
    assert [m["content"] for m in relaunched.gateway.seen[-1]][-3:] == [
        "first question",
        REPLY,
        "second question",
    ]
    row = relaunched.db.get_message_by_id(pending.persisted_message_id)
    assert (row["content"], row["assistant_generation_state"]) == (REPLY, "complete")
    assert relaunched.store.dispatch_recovery_for_session(relaunched.session_id) is None


@pytest.mark.asyncio
async def test_relaunch_after_a_crash_mid_reply_discard_keeps_the_user_message(
    tmp_path, databases
):
    relaunched = await _crash_mid_reply(tmp_path, databases)
    _assert_fresh_native_ids_for_the_settled_turn(relaunched)
    user_id = _path(relaunched)[2].id

    result = await relaunched.controller.discard_dispatch_recovery(
        relaunched.session_id
    )

    assert result.accepted, result.visible_copy
    path = _path(relaunched)
    assert [(row.role, row.content) for row in path] == [
        (USER, "first question"),
        (ASSISTANT, REPLY),
        (USER, "second question"),
        (ASSISTANT, DISCARD_COPY),
    ]
    assert path[2].id == user_id
    assert path[-1].assistant_generation_state == "discarded"
    assert relaunched.db.get_message_by_id(path[2].persisted_message_id) is not None
    row = relaunched.db.get_message_by_id(path[-1].persisted_message_id)
    assert (row["content"], row["assistant_generation_state"]) == (
        DISCARD_COPY,
        "discarded",
    )
    assert relaunched.store.dispatch_recovery_for_session(relaunched.session_id) is None
    assert relaunched.gateway.seen == []


@pytest.mark.asyncio
async def test_relaunched_accepted_recovery_retry_response_settles(
    tmp_path, databases
):
    """The ACCEPTED shape ("Retry response"), restored with fresh native ids."""
    db, conversation_id, repository = _database(tmp_path / "accepted.sqlite")
    databases.append(db)
    # Frozen as this controller's own send would freeze it, so the retry's
    # authority check passes and only the relaunch is under test.
    authority = replace(_authority(), direct_library_tools=True)
    acceptance = replace(_acceptance(conversation_id), frozen_authority=authority)
    _insert(db, repository, acceptance)
    relaunched = await _relaunch(db, conversation_id)
    recovery = relaunched.store.dispatch_recovery_for_session(relaunched.session_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.ACCEPTED
    assert recovery.actions[0].label == "Retry response"

    result = await relaunched.controller.retry_dispatch_recovery(relaunched.session_id)

    assert result.accepted, result.visible_copy
    assert [(row.role, row.content) for row in _path(relaunched)] == [
        (USER, "hello"),
        (ASSISTANT, REPLY),
    ]
    row = db.get_message_by_id("assistant-1")
    assert (row["content"], row["assistant_generation_state"]) == (REPLY, "complete")
    assert relaunched.store.dispatch_recovery_for_session(relaunched.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("relaunch_again", [False, True])
async def test_relaunch_discard_then_resend_re_runs_the_turn_in_place(
    tmp_path, databases, relaunch_again
):
    relaunched = await _crash_mid_reply(tmp_path, databases)
    assert resend_target_id(_path(relaunched)) is None
    discarded = await relaunched.controller.discard_dispatch_recovery(
        relaunched.session_id
    )
    assert discarded.accepted, discarded.visible_copy
    if relaunch_again:
        conversation_id = relaunched.store._sessions[
            relaunched.session_id
        ].persisted_conversation_id
        relaunched = await _relaunch(relaunched.db, conversation_id)
    user = _path(relaunched)[2]
    assert resend_target_id(_path(relaunched)) == user.id

    result = await resend_turn(relaunched.controller, user.id)

    assert result.accepted, result.visible_copy
    path = _path(relaunched)
    assert [(row.role, row.content) for row in path] == [
        (USER, "first question"),
        (ASSISTANT, REPLY),
        (USER, "second question"),
        (ASSISTANT, REPLY),
    ]
    assert path[2].id == user.id
    assert path[3].parent_message_id == user.persisted_message_id
    assert path[3].sibling_count == 1
    live_children = (
        relaunched.db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM messages WHERE parent_message_id = ? AND deleted = 0",
            (user.persisted_message_id,),
        )
        .fetchone()[0]
    )
    assert live_children == 1


@pytest.mark.asyncio
async def test_relaunched_legacy_continuation_owner_is_normalized_before_actions(
    tmp_path, databases
):
    """Same root cause, continuation owner: restore must find it by its id.

    ``test_legacy_continuation_normalizes_and_rebinds_before_actions`` pins this
    with hand-built nodes. Through the production hydration the owner was never
    found, so it was never normalized and its actions stayed disabled.
    """
    db, conversation_id, repository = _database(tmp_path / "legacy.sqlite")
    databases.append(db)
    _insert(db, repository, _deepseek_acceptance(conversation_id))
    _install_legacy_owner(db, conversation_id, state="accepted")

    relaunched = await _relaunch(db, conversation_id)

    row = db.get_message_by_id("assistant-1")
    assert (row["assistant_generation_state"], row["version"]) == (
        "continuation_active",
        8,
    )
    owner = _path(relaunched)[-1]
    assert owner.provider_continuation == _continuation()
    assert owner.provider_continuation_message_version == 8
    assert owner.provider_continuation_actions_enabled is True


@pytest.mark.asyncio
async def test_a_second_open_of_the_same_chat_never_settles_the_first_ones_reply(
    tmp_path, databases
):
    """The owner keeps its persisted id in the FIRST session only.

    A second session restoring the same conversation in the same store gets
    fresh ids for the owner too. Its card must refuse instead of claiming the
    first session's node through the shared message index.
    """
    first = await _crash_mid_reply(tmp_path, databases)
    conversation_id = first.store._sessions[first.session_id].persisted_conversation_id
    second_session = await hydrate_console_session(
        app=SimpleNamespace(chachanotes_db=first.db),
        store=first.store,
        conversation_id=conversation_id,
        tree=ChatConversationService(first.db).get_conversation_tree(
            conversation_id, root_limit=100, depth_cap=100
        ),
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
    )
    owner = _path(first)[-1]

    refused = await first.controller.discard_dispatch_recovery(second_session.id)

    assert not refused.accepted
    assert refused.visible_copy == "That response recovery action is unavailable."
    assert first.store.get_message(owner.id).content == owner.content
    assert first.store.dispatch_recovery_for_session(first.session_id) is not None
    settled = await first.controller.discard_dispatch_recovery(first.session_id)
    assert settled.accepted, settled.visible_copy
    assert _path(first)[-1].content == DISCARD_COPY
