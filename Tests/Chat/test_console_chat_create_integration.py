"""End-to-end bridge closures for fork_chat / new_chat with fakes + real SQLite.

Task 7 of the agent-chat-fork-spawn plan (TASK-32482):

1. The ``build_chat_create_tool_closures`` module helper -- happy path,
   deny, the two-denials terminal guard, and the per-run remember memo
   (fakes only; the executor is stubbed).
2. ``ConsoleChatController.execute_agent_chat_create`` against a REAL
   in-memory-schema SQLite DB (construction cribbed from
   ``Tests/Chat/test_chat_persistence_service.py``'s ``db_instance``
   fixture) with the UI completion callback stubbed to record kwargs.
"""
import json

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import (
    build_chat_create_tool_closures,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_console_skill_script_confirm import _FakeApp


class _FakeConfirm:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.decisions.pop(0)


class _FakeExecutor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"ok": True, "title": "W", "conversation_id": "c2",
                                 "workspace_id": None, "copied_messages": 3, "draft_set": True}

    def __call__(self, payload):
        self.calls.append(payload)
        return dict(self.result)


def test_fork_chat_tool_happy_path():
    confirm, executor = _FakeConfirm([{"allow": True, "remember": False}]), _FakeExecutor()
    fork_tool, new_tool = build_chat_create_tool_closures(
        confirm=confirm, execute=executor, session_id="s1", run_id="r1"
    )
    result = fork_tool({"title": "W: db", "opening_prompt": "go", "instructions": ""})
    assert result.ok
    data = json.loads(result.content)
    assert data["conversation_id"] == "c2" and data["draft_set"] is True
    assert "user" in data["note"].lower()
    assert executor.calls[0]["tool"] == "fork_chat"


def test_deny_returns_error_without_execution():
    confirm, executor = _FakeConfirm([{"allow": False, "remember": False}]), _FakeExecutor()
    fork_tool, _ = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                   session_id="s1", run_id="r1")
    result = fork_tool({"title": "x"})
    assert not result.ok and "declined" in result.error.lower()
    assert executor.calls == []


def test_denial_guard_terminals_after_two():
    confirm = _FakeConfirm([{"allow": False, "remember": False}, {"allow": False, "remember": False}])
    executor = _FakeExecutor()
    fork_tool, _ = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                   session_id="s1", run_id="r1")
    assert not fork_tool({}).ok
    assert not fork_tool({}).ok
    third = fork_tool({})
    assert not third.ok and "denied_repeatedly" in third.error
    assert len(confirm.payloads) == 2  # no third card


def test_remember_skips_confirm_for_that_tool_only():
    confirm, executor = _FakeConfirm([{"allow": True, "remember": True}]), _FakeExecutor()
    fork_tool, new_tool = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                          session_id="s1", run_id="r1")
    assert fork_tool({}).ok                      # card shown, remembered
    assert fork_tool({}).ok                      # no card (per-run memo)
    assert not new_tool({}).ok                   # confirm has no decisions left -> deny fail-closed


def test_raising_executor_maps_to_execution_failed():
    """Fix round 1, part 1: an executor that RAISES (instead of returning
    an outcome dict) must surface through the outcome contract --
    ``ok=False`` with an ``execution_failed:`` prefix -- never as an
    uncaught worker-thread exception escaping into the agent run loop
    (mirrors run_skill_script_tool's broad-catch execute phase)."""
    confirm = _FakeConfirm([{"allow": True, "remember": False}])

    def _raising_executor(payload):
        raise RuntimeError("boom during create")

    fork_tool, _ = build_chat_create_tool_closures(
        confirm=confirm, execute=_raising_executor, session_id="s1", run_id="r1"
    )
    result = fork_tool({"title": "x"})
    assert not result.ok
    assert "execution_failed" in result.error
    assert "boom during create" in result.error


# -- executor tests (real SQLite) ------------------------------------------


@pytest.fixture
def real_db_controller(tmp_path):
    """Controller + real SQLite store, with UI completion stubbed.

    The DB construction mirrors ``Tests/Chat/test_chat_persistence_service.py``'s
    ``db_instance`` fixture (same class, same file-per-test pattern) so the
    schema/migrations initialize identically; the controller/controller-store
    wiring mirrors ``Tests/Chat/test_console_chat_create_confirm.py``'s
    ``make_controller``.
    """
    db = CharactersRAGDB(tmp_path / "test_chat_create.sqlite", "test_chat_create_client")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.app = _FakeApp()
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    try:
        yield controller, db
    finally:
        controller.begin_shutdown()
        db.close_connection()


def test_execute_fork_copies_history_and_lineage(real_db_controller):
    controller, db = real_db_controller
    src_session = controller.store.create_session(title="Src", activate=True)
    src_conv = controller.store.persistence.create_conversation(conversation_title="Src")
    src_session.persisted_conversation_id = src_conv
    ids = [db.add_message({"conversation_id": src_conv, "sender": "user", "content": "hi"})]
    db.set_conversation_active_leaf(src_conv, str(ids[0]))

    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": src_session.id, "title": "W: db",
         "opening_prompt": "go", "instructions": ""}
    )
    assert outcome["ok"], outcome
    forked = db.get_conversation_by_id(outcome["conversation_id"])
    assert forked["parent_conversation_id"] == src_conv
    msgs = db.get_messages_for_conversation(outcome["conversation_id"])
    assert [m["content"] for m in msgs] == ["hi"]
    assert outcome["copied_messages"] == 1
    metadata = json.loads(forked["metadata"])
    assert metadata["console_agent_handoff"]["created_via"] == "fork_chat"
    assert metadata["console_agent_handoff"]["draft"] == "go"
    assert metadata["console_agent_handoff"]["source_run_id"] == ""
    assert outcome["draft_set"] is True


def test_execute_fork_refuses_instructions_on_character_chat(real_db_controller):
    controller, db = real_db_controller
    # FK-enforced: the conversation's character_id must reference a REAL
    # character row (never a synthetic id).
    char_id = db.add_character_card({"name": "Rex"})
    session = controller.store.create_session(
        title="Char", character_id=char_id, character_name="Rex"
    )
    conv = controller.store.persistence.create_conversation(
        conversation_title="Char", character_id=char_id
    )
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x",
         "opening_prompt": "", "instructions": "be terse"}
    )
    assert not outcome["ok"] and outcome["kind"] == "character_conflict"


def test_execute_fork_ephemeral_source_error(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="Tmp", ephemeral=True)
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"})
    assert not outcome["ok"] and outcome["kind"] == "source_not_persisted"


def test_execute_fork_empty_history_error(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="Empty")
    conv = controller.store.persistence.create_conversation(conversation_title="Empty")
    session.persisted_conversation_id = conv
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"})
    assert not outcome["ok"] and outcome["kind"] == "empty_history"


def _live_conversation_count(db) -> int:
    return int(
        db.execute_query(
            "SELECT COUNT(*) AS c FROM conversations WHERE deleted = 0"
        ).fetchone()["c"]
    )


def test_execute_fork_empty_source_creates_no_conversation_row(real_db_controller):
    """Final-review fix wave (Finding 3a): the empty-history refusal fires
    BEFORE create_conversation, so no orphaned row is left behind -- the
    spec's "transaction rolled back; nothing created" contract."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="Empty")
    conv = controller.store.persistence.create_conversation(conversation_title="Empty")
    session.persisted_conversation_id = conv
    before = _live_conversation_count(db)
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"})
    assert not outcome["ok"] and outcome["kind"] == "empty_history"
    assert _live_conversation_count(db) == before  # only the source row exists
    assert not db.execute_query(
        "SELECT 1 AS hit FROM conversations WHERE parent_conversation_id = ?",
        (conv,),
    ).fetchone()  # no fork lineage row was ever created


def test_execute_fork_copy_failure_discards_created_conversation(
    real_db_controller, monkeypatch
):
    """Final-review fix wave (Finding 3b): a mid-copy failure (Task 7's
    read-failure test style) best-effort discards the already-created row
    via soft-delete, keeps the original error kind, and leaves no orphan
    in the live listing."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="Src")
    conv = controller.store.persistence.create_conversation(conversation_title="Src")
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    before = _live_conversation_count(db)

    created: list[str] = []
    original_create = controller.store.persistence.create_conversation

    def _spy_create(**kwargs):
        conversation_id = original_create(**kwargs)
        created.append(conversation_id)
        return conversation_id

    def _boom(*args, **kwargs):
        raise RuntimeError("copy exploded mid-fork")

    monkeypatch.setattr(controller.store.persistence, "create_conversation", _spy_create)
    monkeypatch.setattr(
        ChatConversationService, "copy_conversation_active_path", _boom
    )
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"}
    )
    assert not outcome["ok"]
    assert outcome["kind"] == "execution_failed"  # original kind preserved
    assert "copy exploded mid-fork" in outcome["error"]
    assert len(created) == 1  # the row WAS created, then discarded
    assert db.get_conversation_by_id(created[0]) is None  # excluded when deleted
    discarded = db.get_conversation_by_id(created[0], include_deleted=True)
    assert discarded is not None and discarded["deleted"] == 1
    assert _live_conversation_count(db) == before  # no orphan in the listing


def test_execute_fork_prefers_provided_enriched_title(real_db_controller):
    """Final-review fix wave (Finding 1): the executor prefers a non-empty
    payload title -- including a title the confirm-side enrichment already
    defaulted ("Fork of <source>") -- and only falls back to its own
    default formula when none was provided."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="Src")
    conv = controller.store.persistence.create_conversation(conversation_title="Src")
    session.persisted_conversation_id = conv
    ids = [db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})]
    db.set_conversation_active_leaf(conv, str(ids[0]))
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id,
         "title": "Fork of Src", "opening_prompt": "", "instructions": ""}
    )
    assert outcome["ok"], outcome
    assert outcome["title"] == "Fork of Src"  # used verbatim, not re-wrapped
    assert db.get_conversation_by_id(outcome["conversation_id"])["title"] == (
        "Fork of Src"
    )


def test_execute_payload_too_large(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="S")
    outcome = controller.execute_agent_chat_create(
        {"tool": "new_chat", "session_id": session.id, "title": "x",
         "opening_prompt": "y" * 20_001, "instructions": ""})
    assert not outcome["ok"] and outcome["kind"] == "payload_too_large"


def test_execute_new_chat_creates_conversation_and_completion(real_db_controller):
    controller, db = real_db_controller
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    session = controller.store.create_session(title="S")
    outcome = controller.execute_agent_chat_create(
        {"tool": "new_chat", "session_id": session.id, "title": "Fresh",
         "opening_prompt": "hello there", "instructions": "be brief", "run_id": "run-9"}
    )
    assert outcome["ok"], outcome
    assert outcome["copied_messages"] == 0
    assert outcome["title"] == "Fresh"
    created = db.get_conversation_by_id(outcome["conversation_id"])
    assert created is not None and created["system_prompt"] == "be brief"
    metadata = json.loads(created["metadata"])
    assert metadata["console_agent_handoff"] == {
        "draft": "hello there", "created_via": "new_chat", "source_run_id": "run-9"
    }
    # The executor marshaled exactly one UI completion with the marshal
    # kwargs (_FakeApp's call_from_thread invokes immediately).
    assert completed == [{
        "session_id": session.id,
        "conversation_id": outcome["conversation_id"],
        "title": "Fresh",
        "tool": "new_chat",
        "opening_prompt": "hello there",
        "workspace_id": session.workspace_id,
    }]
    # The new conversation is NOT wired onto any session yet (Task 8 owns
    # session placement) -- verify no session claims it.
    assert all(
        s.persisted_conversation_id != outcome["conversation_id"]
        for s in controller.store.sessions()
    )

def test_execute_fork_read_failure_maps_to_execution_failed(
    real_db_controller, monkeypatch
):
    """Fix round 1, part 2: the fork source-tree/active-leaf DB reads run
    INSIDE the executor's try, so a read failure returns a kind-bearing
    outcome (`execution_failed`) instead of escaping the worker thread."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="Src")
    conv = controller.store.persistence.create_conversation(conversation_title="Src")
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})

    def _boom(*args, **kwargs):
        raise RuntimeError("db read exploded")

    monkeypatch.setattr(ChatConversationService, "get_conversation_tree", _boom)
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"}
    )
    assert not outcome["ok"]
    assert outcome["kind"] == "execution_failed"
    assert "db read exploded" in outcome["error"]
