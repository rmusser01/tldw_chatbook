"""Fleet PR 3b Task 5, Qodo #1808 finding 3: closing a session kills its fleet.

`ConsoleChatStore.close_session` is DESTRUCTIVE -- it purges every
message the session owns and drops the session outright. With Stop
decoupled from the children (this branch), a session's live decoupled
children would otherwise outlive their own conversation: the panel has
no rows left to cancel them from, wake delivery would target a dead
conversation, and its unseen-mark would leak. Navigation-away preserves
the conversation and its survivors rightly continue; DESTRUCTION must
take the fleet with it.

The seam pinned here: `ConsoleChatController.close_session` calls the
explicit whole-fleet path this branch built --
`ConsoleAgentBridge.cancel_all_subagents` -- with the conversation id
derived WHILE THE SESSION STILL EXISTS (the persisted conversation id
when set; the bridge keys everything by it). That method's own walk,
per-handle cancel reuse, approval revocation, and
cancelled-is-never-retained semantics are pinned by execution in
`Tests/Chat/test_console_agent_bridge_cancel_all.py` /
`Tests/Agents/test_fleet_stop_semantics.py`; the recording bridge here
asserts THIS layer's own seam (the call, its target id, its timing),
per the layered-guard rule.

All three were committed RED at the pre-fix branch (`cancel_all_calls
== []`: the close path made no attempt to stop the fleet -- the child
"keeps running with no cancel path", exactly the bot's framing).
"""

from __future__ import annotations

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

from Tests.Chat.test_console_chat_controller import StreamingGateway


class _RecordingFleetBridge:
    """Exposes exactly the seam `close_session` must call, recording it.

    `sessions_present_at_call` captures whether the closed session was
    still in the store when the cancel fired -- the conversation-id
    derivation (`_agent_conversation_id`) reads the store, so a cancel
    issued after the purge would silently target the bare session id
    even for a persisted conversation.
    """

    def __init__(self, store: ConsoleChatStore) -> None:
        self._store = store
        self.cancel_all_calls: list[str] = []
        self.sessions_present_at_call: list[bool] = []

    def cancel_all_subagents(self, conversation_id: str) -> int:
        self.cancel_all_calls.append(conversation_id)
        self.sessions_present_at_call.append(bool(self._store.sessions()))
        return 1


def _controller_with_bridge(store: ConsoleChatStore):
    bridge = _RecordingFleetBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=bridge,
    )
    return controller, bridge


def test_closing_a_session_cancels_its_fleet_through_cancel_all():
    """The active-close path: one close, one whole-fleet cancel, keyed by
    the session's own conversation id, issued while the session still
    exists in the store."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller, bridge = _controller_with_bridge(store)

    controller.close_session(session.id)

    assert bridge.cancel_all_calls == [session.id]
    assert bridge.sessions_present_at_call == [True], (
        "the cancel must fire while the session (and its conversation-id "
        "mapping) still exists"
    )
    assert store.sessions() == []


def test_closing_an_ephemeral_session_cancels_its_fleet_too():
    """The ephemeral-teardown path: a temporary session's close is the
    same destruction (nothing was ever persisted to return to), so its
    fleet dies with it identically."""
    store = ConsoleChatStore()
    session = store.new_session(title="temp", ephemeral=True)
    assert store.session_is_ephemeral(session.id)
    controller, bridge = _controller_with_bridge(store)

    controller.close_session(session.id)

    assert bridge.cancel_all_calls == [session.id]
    assert store.sessions() == []


def test_close_cancel_targets_the_persisted_conversation_id_when_set():
    """The bridge keys fleets by the DURABLE conversation id
    (`_agent_conversation_id`: persisted id when set, else the session
    id). A close must cancel under that key, or a persisted
    conversation's fleet would be looked up under a key it never used."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-durable"
    controller, bridge = _controller_with_bridge(store)

    controller.close_session(session.id)

    assert bridge.cancel_all_calls == ["conv-durable"]


def test_close_survives_a_bridge_without_the_cancel_all_seam():
    """A bare bridge double (or none at all) must never break a close --
    the fleet cancel degrades to a no-op, exactly like every other
    getattr-guarded bridge read on the controller."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=object(),  # no cancel_all_subagents at all
    )

    assert controller.close_session(session.id) is None
    assert store.sessions() == []
