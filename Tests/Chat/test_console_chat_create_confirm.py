"""Confirm rounds for agent-initiated chat creation (fork_chat / new_chat)."""
import threading

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until  # reuse fakes


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
    t.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": False}


def test_deny_round_trip(make_controller):
    controller = make_controller()
    result = {}
    t = threading.Thread(target=lambda: result.update(
        decision=controller.request_chat_create_confirm(_payload(), session_id="s1")))
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(False, False, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=5)
    assert result["decision"] == {"allow": False, "remember": False}


def test_remember_grants_session_scope(make_controller):
    controller = make_controller()
    results = []

    def first():
        results.append(controller.request_chat_create_confirm(_payload(), session_id="s1"))
    t = threading.Thread(target=first)
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(True, True, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=5)

    # Second call in the same session: no card, straight allow. ("s1" is
    # not the store's active session, so round 1 PARKED -- it never
    # marshaled a payload. The brief's `payloads == [payloads[0]]` phrasing
    # IndexError'd on that empty list; a before/after snapshot asserts the
    # same intent -- the granted call marshals NO new payload -- without
    # assuming round 1 ever mounted.)
    payloads_before_second = list(controller.pending_chat_create_payloads)
    decision = controller.request_chat_create_confirm(_payload(), session_id="s1")
    assert decision == {"allow": True, "remember": True}
    assert controller.pending_chat_create_payloads == payloads_before_second

    # Different tool in the same session still confirms.
    t2 = threading.Thread(target=lambda: results.append(
        controller.request_chat_create_confirm(_payload(tool="new_chat"), session_id="s1")))
    t2.start()
    _wait_until(lambda: len(controller.pending_chat_create_ids()) > 0)
    controller.resolve_pending_chat_create(True, False, request_id=controller.pending_chat_create_ids()[-1])
    t2.join(timeout=5)
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

    t.join(timeout=5)
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
    t.join(timeout=5)
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
