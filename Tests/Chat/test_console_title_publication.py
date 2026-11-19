"""A committed title updates all exact aliases, not unrelated conversations."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def test_committed_title_publication_updates_every_exact_runtime_alias() -> None:
    """Dropping alias publication leaves duplicated bound tabs inconsistent."""
    store = ConsoleChatStore()
    aliases = [
        store.restore_persisted_session(
            title="Old",
            workspace_id=None,
            persisted_conversation_id="same",
            all_nodes=[],
            active_leaf_persisted_id=None,
        )
        for _ in range(2)
    ]
    unrelated = store.restore_persisted_session(
        title="Old",
        workspace_id=None,
        persisted_conversation_id="different",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )
    active = store.active_session_id
    store.publish_conversation_title("same", "Renamed")
    assert [session.title for session in aliases] == ["Renamed", "Renamed"]
    assert unrelated.title == "Old"
    assert store.active_session_id == active


def test_title_transition_reserves_all_aliases_before_durable_write(
    monkeypatch,
) -> None:
    """Admission refusal must occur before any caller can commit the title.

    Args:
        monkeypatch: Refuse the second alias's existing mutation admission.
    """
    store = ConsoleChatStore()
    aliases = [
        store.restore_persisted_session(
            title="Old",
            workspace_id=None,
            persisted_conversation_id="same",
            all_nodes=[],
        )
        for _ in range(2)
    ]
    original = store._fork_source_transition

    @contextmanager
    def refuse_second(session_id):
        if session_id == aliases[1].id:
            raise RuntimeError("A voice promotion owns this session.")
        with original(session_id):
            yield

    monkeypatch.setattr(store, "_fork_source_transition", refuse_second)
    writes = []
    with (
        pytest.raises(RuntimeError),
        store.conversation_title_transition("same") as publish,
    ):
        writes.append("committed")
        publish("Renamed")
    assert writes == []
    assert all(session.title == "Old" for session in aliases)
    assert not store._fork_source_transitions


def test_title_transition_refuses_new_alias_until_publication_finishes() -> None:
    """A stale hydrate cannot join the exact conversation's durable/live gap."""
    store = ConsoleChatStore()
    with store.conversation_title_transition("same") as publish:
        with pytest.raises(RuntimeError, match="title"):
            store.restore_persisted_session(
                title="Old",
                workspace_id=None,
                persisted_conversation_id="same",
                all_nodes=[],
            )
        publish("New")
    assert (
        store.restore_persisted_session(
            title="New",
            workspace_id=None,
            persisted_conversation_id="same",
            all_nodes=[],
        ).title
        == "New"
    )


def test_title_publication_never_changes_an_alias_rebound_to_another_chat() -> None:
    """Even retained object identity is not permission to rename another chat."""
    store = ConsoleChatStore()
    alias = store.restore_persisted_session(
        title="Old",
        workspace_id=None,
        persisted_conversation_id="same",
        all_nodes=[],
    )
    with store.conversation_title_transition("same") as publish:
        store.rebind_persisted_conversation(alias.id, "another")
        publish("New")
    assert alias.title == "Old"


def test_rebinding_into_an_active_title_transition_is_refused() -> None:
    """A rebind cannot bypass the same exact-conversation hydrate fence."""
    store = ConsoleChatStore()
    alias = store.restore_persisted_session(
        title="Other",
        workspace_id=None,
        persisted_conversation_id="other",
        all_nodes=[],
    )
    with (
        store.conversation_title_transition("same"),
        pytest.raises(RuntimeError, match="title"),
    ):
        store.rebind_persisted_conversation(alias.id, "same")
    assert alias.persisted_conversation_id == "other"


@pytest.mark.parametrize(
    "order", ["cancel-first", "publish-first", "after-publication"]
)
def test_cancelling_saved_preparation_during_rename_keeps_cleanup_working(
    order: str,
) -> None:
    """Cancellation cleans transient inputs without undoing a committed rename.

    Args:
        order: Cancellation before, during, or after committed title publication.
    """
    from Tests.Chat.test_console_turn_preparation import _preparation_values
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsoleTurnPreparation,
        ConsoleTurnPreparationState,
    )

    store = ConsoleChatStore()
    session = store.restore_persisted_session(
        title="Old",
        workspace_id=None,
        persisted_conversation_id="same",
        all_nodes=[],
    )
    transient = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="draft", persist=False
    )
    values = _preparation_values(session_id=session.id)
    values.update(
        pre_send_conversation_id="same",
        pre_send_title="Old",
        transient_user_message_id=transient.id,
    )
    preparation = ConsoleTurnPreparation(**values)
    store.begin_preparation(preparation)

    def cancel():
        cancelled = store.cancel_preparation(
            session.id, preparation.preparation_id, expected_state=preparation.state
        )
        assert (
            cancelled is not None
            and cancelled.state is ConsoleTurnPreparationState.CANCELLED
        )
        assert transient.id not in store._message_session_index
        assert session.draft == preparation.executed_draft
        assert session.persisted_conversation_id == "same"

    with store.conversation_title_transition("same") as publish:
        if order == "cancel-first":
            cancel()
        publish("New")
        if order == "publish-first":
            cancel()
    if order == "after-publication":
        cancel()
    assert session.title == "New"


@pytest.mark.parametrize("rollback", ["preparation", "optimistic"])
@pytest.mark.parametrize("binding", ["same", None, "other"])
def test_send_rollback_preserves_only_the_unchanged_saved_title(
    rollback: str, binding: str | None
) -> None:
    """Saved titles survive rollback; scratch or rebound identities still restore.

    Args:
        rollback: Preparation cancellation or earlier optimistic-send rollback.
        binding: Captured pre-send conversation identity.
    """
    from Tests.Chat.test_console_turn_preparation import _preparation_values
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparation

    store = ConsoleChatStore()
    session = store.restore_persisted_session(
        title="Old",
        workspace_id=None,
        persisted_conversation_id="same",
        all_nodes=[],
    )
    transient = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="draft", persist=False
    )
    if rollback == "preparation":
        values = _preparation_values(session_id=session.id)
        values.update(
            pre_send_conversation_id=binding,
            pre_send_title="Old",
            transient_user_message_id=transient.id,
        )
        preparation = ConsoleTurnPreparation(**values)
        store.begin_preparation(preparation)
    store.publish_conversation_title("same", "New")
    if rollback == "preparation":
        assert (
            store.cancel_preparation(
                session.id, preparation.preparation_id, expected_state=preparation.state
            )
            is not None
        )
    else:
        store.rollback_transient_send(
            session.id, transient.id, title="Old", persisted_conversation_id=binding
        )
    assert session.title == ("New" if binding == "same" else "Old")
    assert session.persisted_conversation_id == binding
    assert transient.id not in store._message_session_index


def test_title_publication_callback_expires_with_its_admission_scope() -> None:
    """An escaped callback cannot bypass later voice-promotion ownership."""
    store = ConsoleChatStore()
    with store.conversation_title_transition("same") as publish:
        publish("New")
    with pytest.raises(RuntimeError, match="expired"):
        publish("Late")


@pytest.mark.parametrize("already_saved", [True, False])
def test_committed_send_identity_preserves_a_saved_title_but_names_first_save(
    already_saved: bool,
) -> None:
    """A staged send title is authoritative only for first persistence.

    Args:
        already_saved: Existing saved binding or an unbound first-send session.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleStagedConversationIdentity

    store = ConsoleChatStore()
    if already_saved:
        session = store.restore_persisted_session(
            title="Old",
            workspace_id=None,
            persisted_conversation_id="same",
            all_nodes=[],
        )
        staged = ConsoleStagedConversationIdentity("same", "Old")
        store.publish_conversation_title("same", "New")
    else:
        session = store.create_session(title="Chat 1")
        staged = ConsoleStagedConversationIdentity("same", "First message")
    store.publish_committed_identity(session.id, staged)
    assert session.title == ("New" if already_saved else "First message")
    assert session.persisted_conversation_id == "same"
    store.publish_conversation_title("same", "Later rename")
    store.publish_committed_identity(session.id, staged)
    assert session.title == "Later rename"
