"""Provider-context epoch contracts for ``ConsoleChatStore``.

The queue coordinator uses this token to detect history changes that happen
outside ordinary linear turn growth. These tests deliberately distinguish the
broader payload revision from changes that must pause deferred prompts.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore


def _store_with_session() -> tuple[ConsoleChatStore, str]:
    store = ConsoleChatStore()
    session = store.create_session(title="Epoch")
    return store, session.id


def _generation_meta(seed: int) -> GenerationVariantMeta:
    return GenerationVariantMeta(
        prompt="prompt",
        negative_prompt="",
        backend="test",
        model=None,
        seed=seed,
        style=None,
        params={},
    )


def _failed_assistant(store: ConsoleChatStore, session_id: str, text: str):
    message = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(message.id, text)
    store.mark_message_failed(message.id)
    return message


def test_epoch_starts_at_zero_is_session_isolated_and_is_purged_on_close():
    store = ConsoleChatStore()
    first = store.create_session(title="First")
    second = store.create_session(title="Second")
    first_message = store.append_message(
        first.id,
        role=ConsoleMessageRole.USER,
        content="first",
    )

    assert store.conversation_context_epoch(first.id) == 0
    assert store.conversation_context_epoch(second.id) == 0

    store.update_message_content(first_message.id, "changed")

    assert store.conversation_context_epoch(first.id) == 1
    assert store.conversation_context_epoch(second.id) == 0

    store.close_session(first.id)

    with pytest.raises(KeyError):
        store.conversation_context_epoch(first.id)
    assert store.conversation_context_epoch(second.id) == 0


def test_restore_initializes_fresh_process_local_epochs():
    store = ConsoleChatStore()
    restored = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="restored",
        persisted_message_id="message-1",
    )

    session = store.restore_persisted_session(
        title="Restored",
        workspace_id=None,
        persisted_conversation_id="conversation-1",
        all_nodes=[restored],
        active_leaf_persisted_id="message-1",
    )

    assert store.conversation_context_epoch(session.id) == 0
    assert "conversation_context_epoch" not in vars(session)


def test_restore_state_replaces_and_reinitializes_epoch_state():
    store, old_session_id = _store_with_session()
    old_message = store.append_message(
        old_session_id,
        role=ConsoleMessageRole.USER,
        content="old",
    )
    store.update_message_content(old_message.id, "changed")
    restored_session = ConsoleChatSession(title="Restored state")
    restored_message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="restored",
    )

    store.restore_state(
        sessions=[restored_session],
        messages_by_session={restored_session.id: [restored_message]},
        active_session_id=restored_session.id,
    )

    with pytest.raises(KeyError):
        store.conversation_context_epoch(old_session_id)
    assert store.conversation_context_epoch(restored_session.id) == 0


@pytest.mark.parametrize(
    "terminal_method",
    ["mark_message_complete", "mark_message_failed", "mark_message_stopped"],
)
def test_linear_append_stream_and_terminal_status_do_not_advance_epoch(
    terminal_method: str,
):
    store, session_id = _store_with_session()
    store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )

    store.append_stream_chunk(assistant.id, "answer")
    getattr(store, terminal_method)(assistant.id)

    assert store.conversation_context_epoch(session_id) == 0


def test_feedback_tool_markers_and_same_content_write_do_not_advance_epoch():
    store, session_id = _store_with_session()
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
    )

    store.set_message_feedback(assistant.id, "up")
    store.append_message(
        session_id,
        role=ConsoleMessageRole.TOOL,
        content="tool marker",
    )
    store.update_message_content(assistant.id, "answer")

    assert store.conversation_context_epoch(session_id) == 0


def test_active_path_edit_advances_once_while_off_path_edit_is_stable():
    store, session_id = _store_with_session()
    user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    original = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="original",
    )
    sibling = store.create_sibling(
        original.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="sibling",
    )
    baseline = store.conversation_context_epoch(session_id)

    store.update_message_content(original.id, "off path edit")

    assert store.conversation_context_epoch(session_id) == baseline

    store.update_message_content(sibling.id, "active edit")

    assert store.conversation_context_epoch(session_id) == baseline + 1
    assert store.active_path_message_ids(session_id) == [user.id, sibling.id]


def test_active_leaf_changes_advance_but_idempotent_selection_is_stable():
    store, session_id = _store_with_session()
    user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
    )

    store.set_active_leaf(session_id, user.id)
    assert store.conversation_context_epoch(session_id) == 1

    store.set_active_leaf(session_id, user.id)
    assert store.conversation_context_epoch(session_id) == 1

    store.set_active_leaf(session_id, assistant.id)
    assert store.conversation_context_epoch(session_id) == 2


def test_summary_changes_advance_but_identical_pair_is_stable():
    store, session_id = _store_with_session()
    user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="question",
    )

    store.set_session_context_summary(session_id, "summary", user.id)
    assert store.conversation_context_epoch(session_id) == 1

    store.set_session_context_summary(session_id, "summary", user.id)
    assert store.conversation_context_epoch(session_id) == 1

    store.set_session_context_summary(session_id, None, None)
    assert store.conversation_context_epoch(session_id) == 2


def test_create_sibling_advances_once_and_following_linear_append_is_stable():
    store, session_id = _store_with_session()
    user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
    )

    sibling = store.create_sibling(
        assistant.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="replacement",
    )

    assert store.conversation_context_epoch(session_id) == 1

    store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="next",
    )

    assert store.conversation_context_epoch(session_id) == 1
    assert store.active_path_message_ids(session_id)[:2] == [user.id, sibling.id]


def test_delete_advances_only_when_deleted_subtree_is_on_active_path():
    store, session_id = _store_with_session()
    store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    original = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="original",
    )
    sibling = store.create_sibling(
        original.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="sibling",
    )
    baseline = store.conversation_context_epoch(session_id)

    store.delete_message(original.id)
    assert store.conversation_context_epoch(session_id) == baseline

    store.delete_message(sibling.id)
    assert store.conversation_context_epoch(session_id) == baseline + 1


def test_text_variant_changes_advance_only_for_effective_active_path_text():
    store, session_id = _store_with_session()
    message = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="one",
    )

    store.add_variant(message.id, "two")
    assert store.conversation_context_epoch(session_id) == 1

    store.add_variant(message.id, "two")
    store.select_variant(message.id, 2)
    store.update_message_content(message.id, "two")
    assert store.conversation_context_epoch(session_id) == 1

    store.select_variant(message.id, 0)
    assert store.conversation_context_epoch(session_id) == 2

    store.create_sibling(
        message.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="active sibling",
    )
    baseline = store.conversation_context_epoch(session_id)
    store.select_variant(message.id, 1)

    assert store.conversation_context_epoch(session_id) == baseline


def test_variant_stream_advances_only_when_success_changes_selected_text():
    store, session_id = _store_with_session()
    message = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="base",
    )

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "replacement")
    assert store.conversation_context_epoch(session_id) == 0
    store.finalize_variant_stream(message.id)
    assert store.conversation_context_epoch(session_id) == 1

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "replacement")
    store.finalize_variant_stream(message.id)
    assert store.conversation_context_epoch(session_id) == 1

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "discarded")
    store.mark_message_failed(message.id)
    assert store.conversation_context_epoch(session_id) == 1


def test_failed_retry_advances_when_complete_or_stopped_but_not_when_failed():
    complete_store, complete_session_id = _store_with_session()
    completed_retry = _failed_assistant(
        complete_store,
        complete_session_id,
        "same text",
    )
    complete_store.prepare_message_retry(completed_retry.id)
    complete_store.append_stream_chunk(completed_retry.id, "same text")
    complete_store.mark_message_complete(completed_retry.id)
    assert complete_store.conversation_context_epoch(complete_session_id) == 1

    failed_store, failed_session_id = _store_with_session()
    failed_retry = _failed_assistant(failed_store, failed_session_id, "first")
    failed_store.prepare_message_retry(failed_retry.id)
    failed_store.append_stream_chunk(failed_retry.id, "second")
    failed_store.mark_message_failed(failed_retry.id)
    assert failed_store.conversation_context_epoch(failed_session_id) == 0

    stopped_store, stopped_session_id = _store_with_session()
    stopped_retry = _failed_assistant(stopped_store, stopped_session_id, "first")
    stopped_store.prepare_message_retry(stopped_retry.id)
    stopped_store.append_stream_chunk(stopped_retry.id, "partial")
    stopped_store.mark_message_stopped(stopped_retry.id)
    assert stopped_store.conversation_context_epoch(stopped_session_id) == 1


def test_generation_attachment_changes_advance_only_on_active_path():
    store, session_id = _store_with_session()
    message = store.append_generation_message(
        session_id,
        content="[image] prompt",
        variants=[
            (b"one", "image/png", _generation_meta(1)),
            (b"two", "image/png", _generation_meta(2)),
        ],
    )

    store.append_generation_variant(
        session_id,
        message.id,
        data=b"three",
        mime_type="image/png",
        meta=_generation_meta(3),
        persist=False,
    )
    store.keep_generation_variant(
        session_id,
        message.id,
        position=1,
        persist=False,
    )
    assert store.conversation_context_epoch(session_id) == 2

    store.create_sibling(
        message.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="active sibling",
    )
    baseline = store.conversation_context_epoch(session_id)

    store.append_generation_variant(
        session_id,
        message.id,
        data=b"four",
        mime_type="image/png",
        meta=_generation_meta(4),
        persist=False,
    )
    store.keep_generation_variant(
        session_id,
        message.id,
        position=2,
        persist=False,
    )

    assert store.conversation_context_epoch(session_id) == baseline
