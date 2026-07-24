"""End-to-end: agent TOOL markers anchor to the reply that produced them,
across branches and a real persist -> drop -> resume round trip (Console
branching Phase C, Task 4).

Exercises the REAL stack throughout: an in-memory ``CharactersRAGDB`` behind
the real ``ChatPersistenceService``/``ChatConversationService``, a real
``AgentRunsDB`` file, a real ``ConsoleChatStore`` + ``ConsoleAgentBridge`` +
``ConsoleChatController`` (fake provider gateway only), and the real
``ChatScreen`` resume plumbing (``_console_messages_from_conversation_tree``
+ ``restore_persisted_session`` + ``_inject_resume_agent_markers`` +
``store.apply_resume_marker_overlay`` -- the exact two-step wiring
``_resume_console_workspace_conversation`` drives). No hand-rolled fakes for
any of the pieces under test; only the streaming provider is scripted.

Construction note (see the module docstring on
``test_retry_supersedes_prior_run_dropping_its_resume_markers`` below for the
full investigation): the task brief speculated that ``regenerate_message``
sets ``supersede_previous=True`` and that the two-sibling-branch scenario
would therefore need edit-and-resend instead. Reading
``console_chat_controller.py`` end to end shows the opposite: post-Task-6,
``regenerate_message`` forks a sibling and streams into it with
``variant_mode=False`` and no ``prepare_retry``, so
``_run_agent_reply``'s ``supersede_previous=bool(prepare_retry or
variant_mode)`` is always ``False`` there -- ``variant_mode=True`` is never
passed anywhere in the file any more (it is pre-Task-6, dead-in-practice
plumbing only referenced in a docstring). The ONLY caller that ever sets
``supersede_previous=True`` is ``retry_message`` (``prepare_retry=True``).
Regenerate therefore genuinely produces two independently-anchored, live
(non-superseded) primary runs -- exactly the brief's original Step 1 shape
-- so the main scenario below uses ``regenerate_message`` directly, and the
brief's speculation is corrected (with a dedicated pinning test) rather than
routed around.
"""

import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_ERROR,
    STEP_ERROR,
    STEP_TOOL_RESULT,
    AgentStep,
    RunOutcome,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_destination_shells import _build_test_app


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


class _Gateway:
    """Scripted streaming gateway: each ``stream_chat`` call consumes the
    next script entry in order (mirrors ``_Gateway`` in
    ``Tests/Chat/test_console_agent_swap.py``)."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def resolve_for_send(self, _selection):
        return SimpleNamespace(ready=True, provider="llama_cpp", visible_copy="")

    async def stream_chat(self, _resolution, _messages):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


def _all_runs(agent_db: AgentRunsDB) -> list[dict]:
    """Read every persisted run record directly (steps stays a JSON string)."""
    with agent_db.connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM agent_runs").fetchall()]


def _user_and_assistant(messages):
    """Return ``(user_message, assistant_message)``, ignoring any live TOOL
    marker rows interleaved by the agent bridge (``_append_marker`` appends
    those straight into the live view; they are not tree nodes and are not
    part of the simple two-message shape a fresh turn's transcript has)."""
    user = next(m for m in messages if m.role is ConsoleMessageRole.USER)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    return user, assistant


def _new_agent_controller(db: CharactersRAGDB, agent_db: AgentRunsDB, scripts):
    """Real persisted store + real controller + real agent bridge (TOOL steps)."""
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    gateway = _Gateway(scripts)
    bridge = ConsoleAgentBridge(
        agent_runs_db=agent_db, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    session = store.create_session(title="Marker Anchoring E2E")
    store.active_session_id = session.id
    return store, controller, session


def _resume_with_markers(
    db: CharactersRAGDB,
    agent_db: AgentRunsDB,
    conversation_id: str,
    *,
    active_leaf_persisted_id: str | None,
):
    """Genuine persist -> drop -> resume round trip, WITH the TOOL-marker overlay.

    Mirrors ``_resume_console_workspace_conversation``'s two-step wiring:
    ``_console_messages_from_conversation_tree`` + ``restore_persisted_session``
    reconstruct the active-path view for ``active_leaf_persisted_id``, then
    ``_inject_resume_agent_markers`` (driven directly on a bare ``ChatScreen``,
    per the task brief) + ``store.apply_resume_marker_overlay`` re-derive and
    interleave that branch's agent TOOL markers.

    A FRESH ``ConsoleAgentBridge`` is built for the marker re-derivation
    (``provider_gateway=None`` -- ``resume_marker_messages`` never touches
    it), simulating an app restart exactly as
    ``test_resume_marker_messages_reproduces_live_markers_after_simulated_restart``
    does at the bridge level, and is installed on the screen's private
    ``_console_agent_bridge`` slot so ``_ensure_console_agent_bridge``'s own
    ``:memory:``-DB short-circuit (irrelevant here -- this harness never
    routes through it) never comes into play.
    """
    db.set_conversation_active_leaf(conversation_id, active_leaf_persisted_id)
    service = ChatConversationService(db)
    tree = service.get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = db
    all_nodes = screen._console_messages_from_conversation_tree(tree)
    resumed_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    resumed_session = resumed_store.restore_persisted_session(
        title="Marker Anchoring E2E",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_persisted_id,
    )
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=agent_db, store=resumed_store, provider_gateway=None
    )
    screen._console_agent_bridge = fresh_bridge
    markers = screen._inject_resume_agent_markers(
        resumed_store.messages_for_session(resumed_session.id), conversation_id
    )
    resumed_store.apply_resume_marker_overlay(resumed_session.id, markers)
    return resumed_store, resumed_session


@pytest.mark.asyncio
async def test_agent_marker_anchoring_across_two_sibling_branches_on_resume(tmp_path):
    """Two sibling agent replies to the SAME user turn, each its own real
    agent run with its own TOOL step: regenerate forks A1' as a sibling of
    A1 under the same parent U1 (see the module docstring for why regenerate
    -- not edit-and-resend -- is the right construction). Persisting, then
    resuming onto EACH branch in turn, must show only that branch's own
    run's marker, placed immediately after its own reply -- never the
    sibling's.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    agent_db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    try:
        scripts = [
            [_fence("calculator", {"expression": "6*7"})],  # R1 turn 1: tool fence
            ["42."],  # R1 turn 2: final answer
            [_fence("calculator", {"expression": "7*8"})],  # R2 turn 1: tool fence
            ["56."],  # R2 turn 2: final answer
        ]
        store, controller, session = _new_agent_controller(db, agent_db, scripts)

        # ---- A1: U1 -> A1, agent run R1 with its own TOOL step ----
        result1 = await controller.submit_draft("what is 6*7?")
        assert result1.accepted is True
        u1, a1 = _user_and_assistant(store.messages_for_session(session.id))
        assert a1.content == "42."
        assert a1.status == "complete"
        assert a1.persisted_message_id is not None

        # ---- A1': regenerate forks a SIBLING under the SAME parent (u1) ----
        result2 = await controller.regenerate_message(a1.id)
        assert result2.accepted is True
        a1_prime_id = store.active_leaf(session.id)
        assert a1_prime_id != a1.id
        a1_prime = store.get_message(a1_prime_id)
        assert a1_prime.content == "56."
        assert a1_prime.persisted_message_id is not None
        siblings, _index, sibling_count = store.siblings_at(a1.id)
        assert sibling_count == 2
        assert {s.id for s in siblings} == {a1.id, a1_prime_id}
        # a1 is untouched, just off the active path (never deleted).
        assert store.get_message(a1.id).content == "42."

        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        # ---- Both runs are real, distinct, NON-superseded, correctly anchored ----
        primary_runs = [
            r for r in _all_runs(agent_db) if r["agent_kind"] == "primary"
        ]
        assert len(primary_runs) == 2
        assert all(r["status"] != "superseded" for r in primary_runs)
        by_anchor = {r["assistant_message_id"]: r for r in primary_runs}
        assert set(by_anchor) == {a1.persisted_message_id, a1_prime.persisted_message_id}
        r1_steps = json.loads(by_anchor[a1.persisted_message_id]["steps"])
        r2_steps = json.loads(by_anchor[a1_prime.persisted_message_id]["steps"])
        assert any(s["tool_name"] == "calculator" for s in r1_steps)
        assert any(s["tool_name"] == "calculator" for s in r2_steps)

        # ---- Resume ON BRANCH A1: R1's marker shows, R2's is entirely absent ----
        resumed_a_store, resumed_a_session = _resume_with_markers(
            db,
            agent_db,
            conversation_id,
            active_leaf_persisted_id=a1.persisted_message_id,
        )
        view_a = resumed_a_store.messages_for_session(resumed_a_session.id)
        assert [m.content for m in view_a[:2]] == ["what is 6*7?", "42."]
        assert len(view_a) == 3  # U1, A1, exactly one TOOL marker
        assert view_a[2].role is ConsoleMessageRole.TOOL
        assert "calculator" in view_a[2].content
        assert "6*7" in view_a[2].content
        # Branch A1''s run is off-path -- absent, not just its own marker
        # (checked on the TOOL row only: U1's own question text legitimately
        # contains "6*7" regardless of which branch is active).
        tool_rows_a = [m for m in view_a if m.role is ConsoleMessageRole.TOOL]
        assert len(tool_rows_a) == 1
        assert "7*8" not in tool_rows_a[0].content

        # ---- Fresh resume ON BRANCH A1': the reverse ----
        resumed_b_store, resumed_b_session = _resume_with_markers(
            db,
            agent_db,
            conversation_id,
            active_leaf_persisted_id=a1_prime.persisted_message_id,
        )
        view_b = resumed_b_store.messages_for_session(resumed_b_session.id)
        assert [m.content for m in view_b[:2]] == ["what is 6*7?", "56."]
        assert len(view_b) == 3
        assert view_b[2].role is ConsoleMessageRole.TOOL
        assert "calculator" in view_b[2].content
        assert "7*8" in view_b[2].content
        tool_rows_b = [m for m in view_b if m.role is ConsoleMessageRole.TOOL]
        assert len(tool_rows_b) == 1
        assert "6*7" not in tool_rows_b[0].content
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_legacy_null_anchor_run_places_via_ordinal_fallback_on_resume(tmp_path):
    """A run recorded before Phase C (or one whose terminal write never
    landed a persisted id) has ``assistant_message_id`` NULL forever --
    ``set_run_assistant_message_id`` is never retroactively called for it.
    Task 3's fallback must still place it: the single null-anchored block
    matched ordinally against the single unclaimed ASSISTANT message, rather
    than silently dropped.

    Built directly against ``AgentRunsDB`` (bypassing the live bridge, which
    -- since Task 2 -- always records SOME id, NULL or persisted, on every
    terminal path) to simulate genuinely pre-Phase-C data: a plain
    (non-agent) turn with no bridge wired at all, then one run hand-inserted
    after the fact.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    agent_db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        gateway = _Gateway([["a plain answer."]])
        controller = ConsoleChatController(store=store, provider_gateway=gateway)
        session = store.create_session(title="Legacy Fallback")
        store.active_session_id = session.id

        result = await controller.submit_draft("hello")
        assert result.accepted is True
        u1, a1 = _user_and_assistant(store.messages_for_session(session.id))
        assert a1.content == "a plain answer."
        assert a1.persisted_message_id is not None
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None
        # The legacy/no-bridge path never touches AgentRunsDB at all.
        assert _all_runs(agent_db) == []

        legacy_run_id = agent_db.create_run(
            conversation_id=conversation_id, agent_kind="primary"
        )
        agent_db.append_steps(
            legacy_run_id,
            [
                {
                    "index": 0,
                    "kind": STEP_TOOL_RESULT,
                    "tool_name": "calculator",
                    "result": "9*9=81",
                    "summary": "",
                    "args": None,
                    "created_at": "",
                },
            ],
        )
        agent_db.set_status(legacy_run_id, "done", result="a plain answer.")
        legacy_record = agent_db.get_run(legacy_run_id)
        assert legacy_record["assistant_message_id"] is None  # the fallback's trigger

        resumed_store, resumed_session = _resume_with_markers(
            db,
            agent_db,
            conversation_id,
            active_leaf_persisted_id=a1.persisted_message_id,
        )
        view = resumed_store.messages_for_session(resumed_session.id)
        assert [m.content for m in view[:2]] == ["hello", "a plain answer."]
        assert len(view) == 3
        assert view[2].role is ConsoleMessageRole.TOOL
        assert view[2].content == "⚙ calculator → 9*9=81"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_retry_supersedes_prior_run_dropping_its_resume_markers(tmp_path):
    """The investigated finding, pinned directly: ``retry_message`` (NOT
    ``regenerate_message`` -- see the module docstring) is the only caller
    that sets ``supersede_previous=True``. It reruns the SAME assistant
    message id in place, so without superseding, a retried run's stale TOOL
    marker would reappear ALONGSIDE the fresh one on resume (same anchor id,
    both blocks would match the same message index) -- ``supersede_run_tree``
    prevents that duplication by excluding the superseded run from
    ``resume_marker_messages`` entirely.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    agent_db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    try:
        scripts = [
            [_fence("calculator", {"expression": "1*1"})],  # R1 turn 1: tool fence
            ["1."],  # R1 turn 2: real completion (outcome discarded below)
            [_fence("calculator", {"expression": "9*9"})],  # R2 turn 1: tool fence
            ["81."],  # R2 turn 2: final answer
        ]
        store, controller, session = _new_agent_controller(db, agent_db, scripts)
        real_run_reply = controller._agent_bridge.run_reply

        def fail_after_real_run(**kwargs):
            # Execute the REAL turn (a genuine TOOL step lands on R1 in
            # AgentRunsDB), but report failure so the assistant message ends
            # up "failed" -> retryable (mirrors test_console_agent_swap.py's
            # error_after_real_run pattern).
            run_id, _outcome = real_run_reply(**kwargs)
            return run_id, RunOutcome(
                status=RUN_ERROR,
                steps=[AgentStep(index=0, kind=STEP_ERROR, summary="boom")],
            )

        controller._agent_bridge.run_reply = fail_after_real_run
        result1 = await controller.submit_draft("what is 1*1?")
        assert result1.accepted is True
        controller._agent_bridge.run_reply = real_run_reply

        _u1, a1 = _user_and_assistant(store.messages_for_session(session.id))
        assert a1.status == "failed"
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        result2 = await controller.retry_message(a1.id)
        assert result2.accepted is True
        retried = store.get_message(a1.id)
        assert retried.status == "complete"
        assert retried.content == "81."

        primary_runs = [
            r for r in _all_runs(agent_db) if r["agent_kind"] == "primary"
        ]
        assert len(primary_runs) == 2
        statuses = sorted(r["status"] for r in primary_runs)
        assert "superseded" in statuses
        # Retry reruns IN PLACE -- both runs anchor to the SAME message id.
        assert {r["assistant_message_id"] for r in primary_runs} == {
            retried.persisted_message_id
        }

        resumed_store, resumed_session = _resume_with_markers(
            db,
            agent_db,
            conversation_id,
            active_leaf_persisted_id=retried.persisted_message_id,
        )
        view = resumed_store.messages_for_session(resumed_session.id)
        tool_rows = [m for m in view if m.role is ConsoleMessageRole.TOOL]
        # Exactly ONE marker -- the fresh run's (9*9), never a stale
        # duplicate from the superseded one (1*1).
        assert len(tool_rows) == 1
        assert "9*9" in tool_rows[0].content
        assert "1*1" not in tool_rows[0].content
    finally:
        db.close_connection()
