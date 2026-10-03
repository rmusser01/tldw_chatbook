"""Confirm rounds for agent-initiated chat creation (fork_chat / new_chat)."""
import threading
from collections.abc import Callable

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until  # reuse fakes


#: Event waits and worker joins share the existing bounded race-test deadline.
_CHAT_CREATE_SYNC_TIMEOUT_SECONDS = 5


@pytest.fixture
def make_controller():
    made = []

    def _make() -> ConsoleChatController:
        store = ConsoleChatStore()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.pending_chat_create_payloads = []
        controller.set_pending_chat_create = controller.pending_chat_create_payloads.append
        made.append(controller)
        return controller

    yield _make
    for controller in made:
        controller.begin_shutdown()


def _payload(tool="fork_chat", **extra):
    return {"tool": tool, "title": "W: db", "opening_prompt": "go", "instructions": ""}


def test_allow_round_trip(make_controller):
    controller = make_controller()
    result = {}
    t = threading.Thread(target=lambda: result.update(
        decision=controller.request_chat_create_confirm(_payload(), session_id="s1")))
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(True, False, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert result["decision"] == {"allow": True, "remember": False}


def test_deny_round_trip(make_controller):
    controller = make_controller()
    result = {}
    t = threading.Thread(target=lambda: result.update(
        decision=controller.request_chat_create_confirm(_payload(), session_id="s1")))
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(False, False, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert result["decision"] == {"allow": False, "remember": False}


def test_remember_grants_session_scope(make_controller):
    controller = make_controller()
    # TASK-32531: grants ride only for VERIFIED primary requesters.
    class _PDB:
        @staticmethod
        def get_run(run_id):
            return {"agent_kind": "primary", "parent_run_id": None, "task": ""}

    controller._agent_bridge = type("B", (), {"agent_runs_db": _PDB()})
    results = []

    from tldw_chatbook.Agents.run_context import use_run_id

    def first():
        with use_run_id("run-x"):
            results.append(controller.request_chat_create_confirm(_payload(), session_id="s1"))
    t = threading.Thread(target=first)
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(True, True, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)

    # Second call in the same session: no card, straight allow. ("s1" is
    # not the store's active session, so round 1 PARKED -- it never
    # marshaled a payload. The brief's `payloads == [payloads[0]]` phrasing
    # IndexError'd on that empty list; a before/after snapshot asserts the
    # same intent -- the granted call marshals NO new payload -- without
    # assuming round 1 ever mounted.)
    payloads_before_second = list(controller.pending_chat_create_payloads)
    with use_run_id("run-x"):
        decision = controller.request_chat_create_confirm(_payload(), session_id="s1")
    assert decision == {"allow": True, "remember": True}
    assert controller.pending_chat_create_payloads == payloads_before_second

    # Different tool in the same session still confirms.
    def second_tool():
        with use_run_id("run-x"):
            results.append(controller.request_chat_create_confirm(
                _payload(tool="new_chat"), session_id="s1"))

    t2 = threading.Thread(target=second_tool)
    t2.start()
    _wait_until(lambda: len(controller.pending_chat_create_ids()) > 0)
    controller.resolve_pending_chat_create(True, False, request_id=controller.pending_chat_create_ids()[-1])
    t2.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert results[-1] == {"allow": True, "remember": False}


def test_no_ui_fails_closed_immediately(make_controller):
    controller = make_controller()
    controller.app = None
    controller.set_pending_chat_create = None
    decision = controller.request_chat_create_confirm(_payload())
    assert decision == {"allow": False, "remember": False}


def test_revoking_a_run_denies_its_chat_create_confirm(make_controller):
    """The approval kill-switch must sweep chat-create rounds too: a
    cancelled/abandoned run's confirm card dies with it, failing closed --
    mirroring the skill-script suite's
    ``test_revoking_a_run_denies_its_skill_script_confirm_and_spares_a_sibling``
    (see `revoke_approval_rounds_for_run`)."""
    from tldw_chatbook.Agents.run_context import use_run_id

    controller = make_controller()
    results = {}

    def worker():
        with use_run_id("run-chat-a"):
            results["decision"] = controller.request_chat_create_confirm(
                _payload(), session_id="s1"
            )

    t = threading.Thread(target=worker)
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))

    assert controller.revoke_approval_rounds_for_run("run-chat-a") == 1

    t.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert not t.is_alive(), "the revoked confirm never released its thread"
    assert results["decision"] == {"allow": False, "remember": False}
    assert controller.pending_chat_create_ids() == []  # torn down, not armed


# ---------------------------------------------------------------------------
# Final-review fix wave (Finding 1): the controller enriches the card payload
# BEFORE arming -- fork_source_title / fork_message_count (the card's fork
# line), a default title when the agent omitted one, and run-id attribution.
# Real-DB fixture: the enrichment reads the source conversation's tree.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_db_confirm(tmp_path):
    """Confirm controller over a real SQLite store, with the UI sink wired.

    Construction mirrors ``test_console_chat_create_integration.py``'s
    ``real_db_controller`` (same class, same file-per-test pattern).
    """
    db = CharactersRAGDB(tmp_path / "confirm-enrich.sqlite", "confirm-enrich-test")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.app = _FakeApp()
    controller.pending_chat_create_payloads = []
    controller.set_pending_chat_create = controller.pending_chat_create_payloads.append
    try:
        yield controller, db
    finally:
        controller.begin_shutdown()
        db.close_connection()


def _forked_source(controller, db, *, title="Src"):
    """An ACTIVE session with a persisted 2-message active path plus one
    off-path sibling root -- pins the count to the ACTIVE-PATH definition
    (2), not total tree nodes (3)."""
    session = controller.store.create_session(title=title)
    conv = controller.store.persistence.create_conversation(conversation_title=title)
    session.persisted_conversation_id = conv
    m1 = db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    m2 = db.add_message(
        {
            "conversation_id": conv,
            "sender": "assistant",
            "content": "ok",
            "parent_message_id": m1,
        }
    )
    db.set_conversation_active_leaf(conv, str(m2))
    db.add_message({"conversation_id": conv, "sender": "user", "content": "off-path"})
    return session, conv


def _arm_and_capture(controller, payload, session_id):
    """Arm one confirm round, capture its marshaled card payload, deny it."""
    result = {}
    t = threading.Thread(
        target=lambda: result.update(
            decision=controller.request_chat_create_confirm(
                payload, session_id=session_id
            )
        )
    )
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_payloads))
    card = controller.pending_chat_create_payloads[0]
    controller.resolve_pending_chat_create(
        False, False, request_id=controller.pending_chat_create_ids()[0]
    )
    t.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert not t.is_alive(), "the confirm round never released its thread"
    return card, result["decision"]


def test_fork_card_payload_is_enriched_with_fork_facts(real_db_confirm):
    controller, db = real_db_confirm
    session, conv = _forked_source(controller, db)
    card, decision = _arm_and_capture(
        controller,
        {
            "tool": "fork_chat",
            "session_id": session.id,
            "run_id": "run-77",
            "title": "",
            "opening_prompt": "go",
            "instructions": "",
        },
        session.id,
    )
    assert decision == {"allow": False, "remember": False}  # round behaved normally
    assert card["fork_source_title"] == "Src"
    # ACTIVE-PATH length (2), not total tree nodes (3): the sibling root the
    # fork drops must not be counted.
    assert card["fork_message_count"] == 2
    assert card["title"] == "Fork of Src"  # default filled; header never empty
    assert card["run_id"] == "run-77"  # run attribution for the card


def test_fork_card_payload_keeps_explicit_title(real_db_confirm):
    controller, db = real_db_confirm
    session, conv = _forked_source(controller, db)
    card, _ = _arm_and_capture(
        controller,
        {
            "tool": "fork_chat",
            "session_id": session.id,
            "run_id": "run-77",
            "title": "W: db",
            "opening_prompt": "go",
            "instructions": "",
        },
        session.id,
    )
    assert card["title"] == "W: db"  # agent-provided title is never overwritten


def test_new_chat_card_payload_gets_default_title(real_db_confirm):
    controller, db = real_db_confirm
    session = controller.store.create_session(title="Any")
    card, _ = _arm_and_capture(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "run_id": "run-78",
            "title": "",
            "opening_prompt": "",
            "instructions": "",
        },
        session.id,
    )
    assert card["title"] == "New Chat"
    assert "fork_source_title" not in card
    assert "fork_message_count" not in card


def test_fork_enrichment_degrades_when_tree_read_fails(real_db_confirm, monkeypatch):
    """A raising tree read must degrade (no fork keys, title defaulted from
    the owning SESSION's title) and never block or deny the round."""
    controller, db = real_db_confirm
    session, conv = _forked_source(controller, db, title="Fallback")

    def _boom(*args, **kwargs):
        raise RuntimeError("tree read exploded")

    monkeypatch.setattr(ChatConversationService, "get_conversation_tree", _boom)
    card, decision = _arm_and_capture(
        controller,
        {
            "tool": "fork_chat",
            "session_id": session.id,
            "run_id": "run-79",
            "title": "",
            "opening_prompt": "",
            "instructions": "",
        },
        session.id,
    )
    assert decision == {"allow": False, "remember": False}  # round armed + resolved
    assert "fork_message_count" not in card  # no fork line
    assert "fork_source_title" not in card
    assert card["title"] == "Fork of Fallback"  # degraded default from session title


def test_confirm_payload_run_id_is_the_true_run(make_controller):
    """PR review #13: the card displays the round's TRUE run id, not the
    assistant-message placeholder the bridge closure rode in on."""
    from tldw_chatbook.Agents.run_context import use_run_id

    controller = make_controller()
    # Arm with the run context INSIDE the worker (the round stamps
    # current_run_id there); legacy no-session caller so the payload
    # MOUNTS to the sink instead of parking for a background session.
    def worker():
        with use_run_id("run-TRUE-1"):
            controller.request_chat_create_confirm(_payload())

    t2 = threading.Thread(target=worker)
    t2.start()
    _wait_until(lambda: bool(controller.pending_chat_create_payloads))
    controller.resolve_pending_chat_create(
        True, False, request_id=controller.pending_chat_create_ids()[0]
    )
    t2.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    payload = controller.pending_chat_create_payloads[0]
    assert payload["run_id"] == "run-TRUE-1"


# ---------------------------------------------------------------------------
# TASK-32531: sub-agent requesters stamp identity and never ride grants.
# ---------------------------------------------------------------------------


class _FakeAgentDB:
    def __init__(self, row):
        self._row = row

    def get_run(self, run_id):
        return {"agent_kind": self._row, "parent_run_id": "parent-1",
                "task": "do the thing"} if run_id == "run-9" else None


def test_subagent_requester_stamps_identity_and_skips_session_grant(make_controller):
    controller = make_controller()
    real = controller.store.create_session(title="S")
    sid = real.id  # active session: the round MOUNTS (no park)
    controller._chat_create_session_grants[sid] = {"fork_chat"}
    controller._agent_bridge = type("B", (), {"agent_runs_db": _FakeAgentDB("subagent")})

    decisions = []

    import threading
    from tldw_chatbook.Agents.run_context import use_run_id

    def worker():
        with use_run_id("run-9"):
            decisions.append(
                controller.request_chat_create_confirm(_payload(), session_id=sid)
            )

    t = threading.Thread(target=worker)
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    payload_seen = dict(controller.pending_chat_create_payloads[-1])
    controller.resolve_pending_chat_create(
        True, False, request_id=controller.pending_chat_create_ids()[0]
    )
    t.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    # No silent grant ride: a card was armed (round existed) and decided.
    assert decisions == [{"allow": True, "remember": False}]
    assert payload_seen["agent_kind"] == "subagent"
    assert payload_seen["parent_run_id"] == "parent-1"
    assert payload_seen["agent_task"].startswith("do the thing")


@pytest.mark.parametrize("closing", [False, True], ids=["open-session", "committed-close"])
def test_primary_requester_still_rides_session_grant(make_controller, closing):
    """A remembered grant survives only while its source session is live.

    Args:
        make_controller: Existing standalone confirmation controller fixture.
        closing: Commit the real Close ticket before consulting the grant.
    """
    controller = make_controller()
    real = controller.store.create_session(title="S")
    sid = real.id
    controller._chat_create_session_grants[sid] = {"fork_chat"}
    controller._agent_bridge = type("B", (), {"agent_runs_db": _FakeAgentDB("primary")})

    from tldw_chatbook.Agents.run_context import use_run_id

    if closing:
        controller.begin_session_close(
            sid, expected_revision=controller.lifecycle_impact(session_id=sid).revision
        )
        assert sid in controller._chat_create_session_grants
        assert any(session.id == sid for session in controller.store.sessions())
    with use_run_id("run-9"):
        decision = controller.request_chat_create_confirm(_payload(), session_id=sid)
    assert decision == {"allow": not closing, "remember": not closing}
    assert not controller.pending_chat_create_ids()


def test_close_cannot_resurrect_a_remembered_chat_create_grant(
    make_controller: Callable[[], ConsoleChatController],
) -> None:
    """A decided confirmation cannot recreate a grant after real Close.

    Args:
        make_controller: Existing standalone confirmation controller fixture.
    """
    controller = make_controller()
    session = controller.store.create_session(title="Source")
    entered = threading.Event()
    release = threading.Event()
    results = {}

    class PausedGrants(dict):
        def setdefault(self, key, default=None):
            """Pause only an unprotected write so Close can settle first.

            Args:
                key: Owning session's grant key.
                default: Grant set created by the confirmation.

            Returns:
                The existing or newly inserted grant set for the session.
            """
            protected = controller._pending_chat_create_lock.locked()
            results["protected_write"] = protected
            entered.set()
            # A protected write must finish before Close's cancellation sweep;
            # waiting for Close while holding its lock would deadlock the test.
            if not protected:
                assert release.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
            return super().setdefault(key, default)

    controller._chat_create_session_grants = PausedGrants()
    worker = threading.Thread(
        target=lambda: results.update(
            decision=controller.request_chat_create_confirm(
                _payload(), session_id=session.id
            )
        )
    )
    worker.start()
    try:
        _wait_until(lambda: bool(controller.pending_chat_create_ids()))
        controller.resolve_pending_chat_create(
            True, True, request_id=controller.pending_chat_create_ids()[0]
        )
        assert entered.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        if results["protected_write"]:
            worker.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
            assert not worker.is_alive()
        ticket = controller.begin_session_close(
            session.id,
            expected_revision=controller.lifecycle_impact(
                session_id=session.id
            ).revision,
        )
        controller.finalize_session_close(ticket)
        assert not any(s.id == session.id for s in controller.store.sessions())
    finally:
        release.set()
        worker.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert not worker.is_alive()
    assert results["decision"] == {"allow": True, "remember": True}
    assert session.id not in controller._chat_create_session_grants
    assert controller.pending_chat_create_ids() == []
    assert controller._parked_chat_create_payloads == {}


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("closing", [False, True], ids=["navigate", "close"])
def test_legacy_chat_create_marshal_keeps_its_unscoped_contract(
    make_controller: Callable[[], ConsoleChatController],
    monkeypatch: pytest.MonkeyPatch,
    closing: bool,
) -> None:
    """An unparked legacy round remains visible after navigation, or denies Close.

    Args:
        make_controller: Existing real confirmation registry with a fake UI sink.
        monkeypatch: Pauses only the original worker-to-UI marshal boundary.
        closing: Complete source Close instead of merely changing the viewed tab.
    """
    controller = make_controller()
    source = controller.new_session(title="Legacy source")
    entered = threading.Event()
    release = threading.Event()
    marshalled = threading.Event()
    results = {}
    original_marshal = controller._marshal_pending_chat_create

    def delayed_marshal(payload):
        """Hold the legacy initial projection before its real UI callback.

        Args:
            payload: Original controller confirmation or teardown payload.
        """
        if payload:
            entered.set()
            assert release.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        original_marshal(payload)
        if payload:
            marshalled.set()

    monkeypatch.setattr(controller, "_marshal_pending_chat_create", delayed_marshal)
    worker = threading.Thread(
        target=lambda: results.update(
            decision=controller.request_chat_create_confirm(_payload())
        )
    )
    worker.start()
    try:
        assert entered.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        request_id = controller.pending_chat_create_ids()[0]
        controller.new_session(title="Viewed sibling")
        if closing:
            ticket = controller.begin_session_close(
                source.id,
                expected_revision=controller.lifecycle_impact(
                    session_id=source.id
                ).revision,
            )
            controller.finalize_session_close(ticket)
        release.set()
        assert marshalled.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        captured = [
            payload
            for payload in controller.pending_chat_create_payloads
            if payload and payload.get("request_id") == request_id
        ]
        assert bool(captured) is not closing
        assert controller._parked_chat_create_payloads == {}
        if not closing:
            controller.resolve_pending_chat_create(True, False, request_id=request_id)
        worker.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        assert not worker.is_alive()
        assert results == {"decision": {"allow": not closing, "remember": False}}
    finally:
        release.set()
        controller.begin_shutdown()
        worker.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
