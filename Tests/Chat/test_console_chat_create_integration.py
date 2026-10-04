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
import threading

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


#: Event waits and worker joins share the existing bounded race-test deadline.
_CHAT_CREATE_SYNC_TIMEOUT_SECONDS = 5


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


def test_remember_decision_is_not_cached_in_the_closure():
    """Qodo 2761 finding 1: there is deliberately NO run-local remember
    memo -- every call goes through the confirm callback, whose
    session-grant short-circuit is the single remember authority (and
    refuses to ride for sub-agent requesters)."""
    confirm, executor = _FakeConfirm([
        {"allow": True, "remember": True},
        {"allow": True, "remember": False},
    ]), _FakeExecutor()
    fork_tool, _ = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                   session_id="s1", run_id="r1")
    assert fork_tool({}).ok                      # card 1: allow + remember
    assert fork_tool({}).ok                      # card 2 STILL armed (no memo)
    assert len(confirm.payloads) == 2


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
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "x",
            "opening_prompt": "y" * 20_001,
            "instructions": "",
        }
    )
    assert not outcome["ok"] and outcome["kind"] == "creation_refused"


def test_execute_new_chat_requires_live_prepared_approval(real_db_controller):
    controller, db = real_db_controller
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    session = controller.store.create_session(title="S")
    outcome = controller.execute_agent_chat_create(
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "Fresh",
            "opening_prompt": "hello there",
            "instructions": "be brief",
            "run_id": "run-9",
        }
    )
    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not completed
    assert len(controller.store.sessions()) == 1


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


@pytest.mark.asyncio
async def test_chat_create_callbacks_reach_bridge_when_ui_sinks_wired(tmp_path):
    """Live-UAT regression probe: with both UI sinks wired, the controller
    must forward confirm+execute to the bridge (the closures -- and the
    advertised schemas -- depend on them)."""
    from Tests.Chat.test_console_skill_script_confirm import _bridged_controller

    controller, captured = _bridged_controller(tmp_path)
    assert controller.set_pending_chat_create is None
    controller.set_pending_chat_create = lambda payload: None
    controller.complete_agent_chat_create = lambda **kw: None

    result = await controller.submit_draft("hi")

    assert result.accepted is True, result
    assert captured[0]["request_chat_create_confirm"] is not None
    assert captured[0]["execute_agent_chat_create"] is not None


@pytest.mark.asyncio
async def test_chat_create_callbacks_remain_available_for_remembered_grants(tmp_path):
    from Tests.Chat.test_console_skill_script_confirm import _bridged_controller

    controller, captured = _bridged_controller(tmp_path)
    result = await controller.submit_draft("hi")
    assert result.accepted is True, result
    assert captured[0]["request_chat_create_confirm"] is not None
    assert captured[0]["execute_agent_chat_create"] is not None


def test_execute_fork_merges_source_metadata(real_db_controller):
    """PR review #9: the fork keeps the source's own metadata (speech/
    roleplay prefs) and overlays the handoff key -- never replaces."""
    import json as _json

    controller, db = real_db_controller
    session = controller.store.create_session(title="S")
    conv = controller.store.persistence.create_conversation(
        conversation_title="S",
        metadata={"console_speech": {"voice": "alto"}, "roleplay": {"mood": 1}},
    )
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "F",
         "opening_prompt": "", "instructions": ""}
    )
    assert outcome["ok"], outcome
    row = db.get_conversation_by_id(outcome["conversation_id"])
    meta = _json.loads(row["metadata"])
    assert meta["console_speech"] == {"voice": "alto"}
    assert meta["roleplay"] == {"mood": 1}
    assert meta["console_agent_handoff"]["created_via"] == "fork_chat"


def test_execute_fork_global_workspace_resolves(real_db_controller):
    """PR review #8: a globally scoped fork's completion carries the
    Console GLOBAL workspace id, never None."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="G")
    conv = controller.store.persistence.create_conversation(
        conversation_title="G", scope_type="global"
    )
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "F",
         "opening_prompt": "", "instructions": ""}
    )
    assert outcome["ok"], outcome
    assert outcome["workspace_id"] not in (None, "")
    row = db.get_conversation_by_id(outcome["conversation_id"])
    assert row["scope_type"] == "global"


def test_execute_fork_passes_identity_and_hydrated_nodes(real_db_controller):
    """PR review #1/#6: the completion carries the conversation's identity
    and transcript nodes hydrated on the worker thread."""
    controller, db = real_db_controller
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    session = controller.store.create_session(title="I")
    conv = controller.store.persistence.create_conversation(
        conversation_title="I", assistant_kind="persona", assistant_id="winter"
    )
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "F",
         "opening_prompt": "go", "instructions": ""}
    )
    assert outcome["ok"], outcome
    kw = completed[0]
    assert kw["assistant_kind"] == "persona"
    assert kw["assistant_id"] == "winter"
    assert len(kw["nodes"]) == 1  # worker-side hydration, ready for restore
    assert kw["active_leaf_persisted_id"] is not None


def test_execute_fork_lineage_uses_effective_leaf(real_db_controller):
    """PR review #3: a dangling durable leaf pointer must never reach the
    FK-enforced lineage column; the effective (latest) row does."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="L")
    conv = controller.store.persistence.create_conversation(conversation_title="L")
    session.persisted_conversation_id = conv
    first = db.add_message({"conversation_id": conv, "sender": "user", "content": "a"})
    db.set_conversation_active_leaf(conv, "bogus-pointer")
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "F",
         "opening_prompt": "", "instructions": ""}
    )
    assert outcome["ok"], outcome
    row = db.get_conversation_by_id(outcome["conversation_id"])
    assert row["forked_from_message_id"] == str(first)


def test_closure_rejects_non_string_args():
    """PR review #14: a non-string tool arg is a clear error, never a str()
    coercion that stringifies a mapping into the payload."""
    from tldw_chatbook.Chat.console_agent_bridge import build_chat_create_tool_closures

    confirm, executor = _FakeConfirm([{"allow": True, "remember": False}]), _FakeExecutor()
    fork_tool, _ = build_chat_create_tool_closures(
        confirm=confirm, execute=executor, session_id="s1", run_id="r1"
    )
    result = fork_tool({"title": {"nested": "mapping"}})
    assert not result.ok and "invalid_args" in result.error
    assert executor.calls == []


# ---------------------------------------------------------------------------
# TASK-32874: routing for the CREATED chat (ADR-147 gates + vocabulary).
# ---------------------------------------------------------------------------

_FAKE_APP_CONFIG = {
    "chat_defaults": {},
    "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}},
}


def _routing(monkeypatch, *, enabled, allowlist):
    from tldw_chatbook.Agents import agent_routing

    cfg = agent_routing.AgentsRoutingConfig(
        spawn_override_enabled=enabled,
        spawn_override_allowlist=tuple(allowlist),
    )
    monkeypatch.setattr(agent_routing, "load_agents_routing_config", lambda: cfg)


def _app_config(monkeypatch):
    import tldw_chatbook.config as config_mod

    monkeypatch.setattr(config_mod, "load_settings", lambda: _FAKE_APP_CONFIG)


def _routed_session(controller):
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    return controller.store.create_session(
        title="R",
        settings=ConsoleSessionSettings(provider="llama_cpp", model="local-model"),
    )


def _prepared_execute(controller, payload):
    """Exercise actual preparation and an exact controller approval round."""
    from types import SimpleNamespace
    from threading import Event
    from tldw_chatbook.Agents.run_context import use_run_id
    from tldw_chatbook.Agents.agent_routing import RoutingError
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    session = controller.store._sessions[payload["session_id"]]
    if not session.persisted_conversation_id:
        session.persisted_conversation_id = (
            controller.store.persistence.create_conversation(
                conversation_title="source"
            )
        )
    controller._active_cancel_events[session.id] = Event()
    controller._active_assistant_message_ids[session.id] = "source-message"
    bridge = controller._agent_bridge or SimpleNamespace()
    bridge.live_primary_run_id = lambda conversation: "source-run"
    bridge.runs_db = SimpleNamespace(
        get_run=lambda run_id: {
            "id": run_id,
            "conversation_id": session.persisted_conversation_id,
            "agent_kind": "primary",
            "status": "running",
        }
    )
    controller._agent_bridge = bridge
    controller.app.app_config = _FAKE_APP_CONFIG
    runtime = SimpleNamespace(_app=controller.app)
    runtime._resolve_new_console_assistant = lambda workspace, settings: (
        ConsoleRuntime._resolve_new_console_assistant(runtime, workspace, settings)
    )
    controller.app.console_runtime = runtime
    controller._default_session_settings = lambda: ConsoleSessionSettings(
        provider="llama_cpp",
        model="destination-model",
        system_prompt="destination instructions",
    )

    def approve(pending):
        if pending:
            controller.resolve_pending_chat_create(True, False, pending["request_id"])

    controller.set_pending_chat_create = approve
    with use_run_id("source-run"):
        try:
            prepared = controller.prepare_agent_chat_create(
                {
                    **payload,
                    "source_run_id": "source-run",
                    "source_message_id": "source-message",
                }
            )
        except RoutingError as error:
            return {"ok": False, "kind": error.code}
        except ValueError as error:
            return {"ok": False, "kind": str(error).split(":", 1)[0]}
        controller._last_routing_prepared = dict(prepared)
        assert controller.request_chat_create_confirm(prepared)["allow"]
        return controller.execute_agent_chat_create(prepared)


def test_execute_routing_override_disabled(real_db_controller, monkeypatch):
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=False, allowlist=())
    _app_config(monkeypatch)
    session = controller.store.create_session(title="S")
    conv = controller.store.persistence.create_conversation(conversation_title="S")
    session.persisted_conversation_id = conv
    outcome = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "provider": "llama_cpp",
        },
    )
    assert not outcome["ok"] and outcome["kind"] == "override_disabled"


def test_execute_routing_not_allowlisted(real_db_controller, monkeypatch):
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=True, allowlist=("openai",))
    _app_config(monkeypatch)
    session = controller.store.create_session(title="S")
    outcome = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "provider": "llama_cpp",
        },
    )
    assert not outcome["ok"] and outcome["kind"] == "provider_not_allowlisted"


def test_execute_routing_final_guard_on_model_only(real_db_controller, monkeypatch):
    """A model-only override must still match the allowlist glob
    (ADR-147's final-provider guard)."""
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=True, allowlist=("llama_cpp/m1",))
    _app_config(monkeypatch)
    session = _routed_session(controller)
    outcome = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "model": "m2",
        },
    )
    assert not outcome["ok"] and outcome["kind"] == "provider_not_allowlisted"


def test_execute_routing_override_builds_settings(real_db_controller, monkeypatch):
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=True, allowlist=("llama_cpp",))
    _app_config(monkeypatch)
    session = _routed_session(controller)
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    outcome = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "model": "m2",
        },
    )
    assert outcome["ok"], outcome
    settings = next(
        s.settings
        for s in controller.store.sessions()
        if s.persisted_conversation_id == outcome["conversation_id"]
    )
    assert settings is not None
    assert settings.provider == "llama_cpp"
    assert settings.model == "m2"


def test_execute_routing_unknown_and_unrouted_preset(real_db_controller, monkeypatch):
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=False, allowlist=())
    _app_config(monkeypatch)
    session = _routed_session(controller)

    class _Bridge:
        class agent_runs_db:
            @staticmethod
            def list_agent_definitions(enabled_only=True):
                return [
                    {"name": "cloudy", "description": "d", "instructions": "", "tool_allowlist": [], "model": "", "enabled": 1},
                    {"name": "localy", "description": "d", "instructions": "", "tool_allowlist": [], "provider": "llama_cpp", "model": "m1", "enabled": 1},
                ]

    controller._agent_bridge = _Bridge()
    out1 = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "preset": "ghost",
        },
    )
    assert not out1["ok"] and out1["kind"] == "unknown_preset"
    out2 = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "preset": "cloudy",
        },
    )
    assert not out2["ok"] and out2["kind"] == "preset_unrouted"


def test_execute_routing_preset_rides_definition(real_db_controller, monkeypatch):
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=False, allowlist=())
    _app_config(monkeypatch)
    session = _routed_session(controller)

    class _Bridge:
        class agent_runs_db:
            @staticmethod
            def list_agent_definitions(enabled_only=True):
                return [{"name": "localy", "description": "d", "instructions": "", "tool_allowlist": [], "provider": "llama_cpp", "model": "m1", "enabled": 1}]

    controller._agent_bridge = _Bridge()
    conv = controller.store.persistence.create_conversation(conversation_title="R")
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "F",
         "opening_prompt": "", "instructions": "", "preset": "localy"})
    assert outcome["ok"], outcome
    settings = completed[0]["settings"]
    assert settings.provider == "llama_cpp" and settings.model == "m1"


def test_execute_routing_snapshot_is_durable(real_db_controller, monkeypatch):
    """Qodo round finding 3: the routed generation snapshot is merged into
    the conversation's metadata -- the routing survives reopen/crash."""
    import json as _json

    controller, db = real_db_controller
    _routing(monkeypatch, enabled=True, allowlist=("llama_cpp",))
    _app_config(monkeypatch)
    session = _routed_session(controller)
    conv = controller.store.persistence.create_conversation(conversation_title="R")
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "F",
         "opening_prompt": "", "instructions": "", "model": "m2"})
    assert outcome["ok"], outcome
    row = db.get_conversation_by_id(outcome["conversation_id"])
    meta = _json.loads(row["metadata"])
    assert any("m2" in str(v) for v in meta.values()), meta


def test_execute_routing_rejects_non_string_inputs(real_db_controller):
    """Qodo round finding 4: executor-side strict types (callable is
    reachable from callers other than the closures)."""
    controller, db = real_db_controller
    session = controller.store.create_session(title="S")
    outcome = _prepared_execute(
        controller,
        {
            "tool": "new_chat",
            "session_id": session.id,
            "title": "N",
            "opening_prompt": "",
            "instructions": "",
            "provider": ["x"],
        },
    )
    assert not outcome["ok"] and outcome["kind"] == "invalid_args"


def test_fork_from_child_run_context_copies_parent_conversation(real_db_controller):
    """TASK-32531 end to end: a sub-agent's fork_chat call (child run
    context) forks the PARENT conversation's active path with lineage."""
    from tldw_chatbook.Agents.run_context import use_run_id

    controller, db = real_db_controller
    session = controller.store.create_session(title="P")
    conv = controller.store.persistence.create_conversation(conversation_title="P")
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "root"})
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)

    # The executor keys off the SESSION (shared by parent and child runs),
    # so a child-run context lands on the same conversation.
    with use_run_id("child-run-1"):
        outcome = controller.execute_agent_chat_create(
            {"tool": "fork_chat", "session_id": session.id, "title": "C",
             "opening_prompt": "go", "instructions": "", "run_id": "child-run-1"})
    assert outcome["ok"], outcome
    row = db.get_conversation_by_id(outcome["conversation_id"])
    assert row["parent_conversation_id"] == conv
    copied = db.get_messages_for_conversation(outcome["conversation_id"])
    assert [m["content"] for m in copied] == ["root"]
    handoff = completed[0]
    assert handoff["conversation_id"] == outcome["conversation_id"]


@pytest.mark.parametrize("tool", ["new_chat", "fork_chat"])
def test_confirmed_create_refuses_a_source_retained_during_close(
    real_db_controller: tuple[ConsoleChatController, CharactersRAGDB],
    tool: str,
) -> None:
    """A committed Close prevents rows and UI completion before deletion.

    Args:
        real_db_controller: Controller over real SQLite with a UI bridge.
        tool: Confirmed new-chat or fork-chat executor path.
    """
    controller, db = real_db_controller
    source = controller.store.create_session(title="Source")
    conversation_id = controller.store.persistence.create_conversation(
        conversation_title="Source"
    )
    source.persisted_conversation_id = conversation_id
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "hi"}
    )
    db.set_conversation_active_leaf(conversation_id, str(message_id))
    completed = []
    controller.complete_agent_chat_create = lambda **kwargs: completed.append(kwargs)
    before = _live_conversation_count(db)
    ticket = controller.begin_session_close(
        source.id,
        expected_revision=controller.lifecycle_impact(session_id=source.id).revision,
    )
    assert any(session.id == source.id for session in controller.store.sessions())
    outcome = controller.execute_agent_chat_create(
        {"tool": tool, "session_id": source.id, "title": "Late creation"}
    )
    assert not outcome["ok"] and outcome["kind"] == "session_gone"
    assert _live_conversation_count(db) == before
    assert completed == []
    controller.finalize_session_close(ticket)
    assert not any(s.id == source.id for s in controller.store.sessions())


@pytest.mark.parametrize("tool", ["new_chat", "fork_chat"])
@pytest.mark.parametrize(
    "close_at", ["durable-create", "ui-handoff", "ui-handoff-error"]
)
def test_inflight_create_cannot_publish_after_its_source_closes(
    real_db_controller: tuple[ConsoleChatController, CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    close_at: str,
) -> None:
    """Actual Close suppresses a started creation and its queued UI handoff.

    Args:
        real_db_controller: Controller over real SQLite with a UI bridge.
        monkeypatch: Pause the real create or queued completion boundary.
        tool: New-chat or fork-chat executor path.
        close_at: Boundary held while the source's actual Close completes.
    """
    controller, db = real_db_controller
    source = controller.store.create_session(title="Source")
    source_conv = controller.store.persistence.create_conversation(
        conversation_title="Source"
    )
    source.persisted_conversation_id = source_conv
    message_id = db.add_message(
        {"conversation_id": source_conv, "sender": "user", "content": "hi"}
    )
    db.set_conversation_active_leaf(source_conv, str(message_id))
    entered, release = threading.Event(), threading.Event()
    created, completed, queued = [], [], []
    result = {}
    original_create = controller.store.persistence.create_conversation
    original_marshal = controller.app.call_from_thread
    before = _live_conversation_count(db)
    controller.complete_agent_chat_create = lambda **kwargs: completed.append(kwargs)

    def paused_create(**kwargs):
        """Create a real row after the optional Close interleaving.

        Args:
            kwargs: Original persistence inputs.

        Returns:
            The actual created conversation ID.
        """
        if close_at == "durable-create":
            entered.set()
            assert release.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        conversation_id = original_create(**kwargs)
        created.append(conversation_id)
        return conversation_id

    def queued_marshal(callback, *args, **kwargs):
        """Hand worker callbacks to the test's UI thread across Close.

        Args:
            callback: Controller completion gate or original completion sink.
            args: Positional callback inputs.
            kwargs: Keyword callback inputs.

        Returns:
            The result produced by the actual callback on the UI thread.
        """
        if close_at.startswith("ui-handoff") and threading.current_thread() is worker:
            queued.append((callback, args, kwargs))
            entered.set()
            assert release.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
            if close_at == "ui-handoff-error":
                raise RuntimeError("App is not running")
            return result["ui_result"]
        return original_marshal(callback, *args, **kwargs)

    monkeypatch.setattr(
        controller.store.persistence, "create_conversation", paused_create
    )
    monkeypatch.setattr(controller.app, "call_from_thread", queued_marshal)

    def execute():
        """Capture executor failure without abandoning test-owned thread cleanup."""
        try:
            result["outcome"] = controller.execute_agent_chat_create(
                {"tool": tool, "session_id": source.id, "title": "Late creation"}
            )
        except Exception as exc:  # noqa: BLE001 -- surface owned thread failures
            result["error"] = str(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    try:
        assert entered.wait(_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
        ticket = controller.begin_session_close(
            source.id,
            expected_revision=controller.lifecycle_impact(
                session_id=source.id
            ).revision,
        )
        controller.finalize_session_close(ticket)
        assert not any(s.id == source.id for s in controller.store.sessions())
        if queued and close_at != "ui-handoff-error":
            callback, args, kwargs = queued[0]
            result["ui_result"] = callback(*args, **kwargs)
    finally:
        release.set()
        worker.join(timeout=_CHAT_CREATE_SYNC_TIMEOUT_SECONDS)
    assert not worker.is_alive()
    assert _live_conversation_count(db) == before
    assert "error" not in result
    assert not result["outcome"]["ok"]
    assert result["outcome"]["kind"] == "session_gone"
    assert completed == []
    assert len(created) == 1
    assert db.get_conversation_by_id(created[0]) is None
    assert db.get_conversation_by_id(created[0], include_deleted=True)["deleted"] == 1


@pytest.mark.parametrize("view_change", ["detached", "reattached"])
def test_chat_create_completion_uses_the_current_view_sink(
    real_db_controller: tuple[ConsoleChatController, CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
    view_change: str,
) -> None:
    """View detachment preserves durable results and reattachment owns UI.

    Args:
        real_db_controller: Controller over real SQLite with a UI bridge.
        monkeypatch: Change the view sink during the UI handoff.
        view_change: Detach the old view or replace it with a new one.
    """
    controller, db = real_db_controller
    source = controller.store.create_session(title="Source")
    old_completed, new_completed = [], []
    controller.complete_agent_chat_create = lambda **kwargs: old_completed.append(
        kwargs
    )

    def handoff_after_view_change(callback, *args, **kwargs):
        """Execute the actual queued callback after changing its UI sink.

        Args:
            callback: Controller completion gate or original completion sink.
            args: Positional callback inputs.
            kwargs: Keyword callback inputs.

        Returns:
            The actual callback result.
        """
        controller.complete_agent_chat_create = (
            None
            if view_change == "detached"
            else lambda **fields: new_completed.append(fields)
        )
        return callback(*args, **kwargs)

    monkeypatch.setattr(controller.app, "call_from_thread", handoff_after_view_change)
    outcome = controller.execute_agent_chat_create(
        {"tool": "new_chat", "session_id": source.id, "title": "Live source"}
    )
    assert outcome["ok"]
    assert db.get_conversation_by_id(outcome["conversation_id"]) is not None
    assert old_completed == []
    assert len(new_completed) == (view_change == "reattached")


def test_admitted_chat_create_completion_error_keeps_the_placed_chat(
    real_db_controller: tuple[ConsoleChatController, CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later source Close cannot discard an already-admitted UI result.

    Args:
        real_db_controller: Controller over real SQLite and the real chat store.
        monkeypatch: Deliver a UI exception after retiring its source.
    """
    controller, db = real_db_controller
    source = controller.store.create_session(title="Source")
    placed = []

    def place_then_fail(**kwargs):
        """Place the durable chat before a subsequent UI completion failure.

        Args:
            kwargs: Real executor completion inputs.
        """
        placed.append(
            controller.store.restore_persisted_session(
                title=kwargs["title"],
                workspace_id=kwargs["workspace_id"],
                persisted_conversation_id=kwargs["conversation_id"],
                all_nodes=kwargs["nodes"],
                active_leaf_persisted_id=kwargs["active_leaf_persisted_id"],
                activate=False,
            )
        )
        raise RuntimeError("completion failed after placement")

    def retire_before_worker_receives_error(callback, *args, **kwargs):
        """Retire the source after the admitted UI callback fails.

        Args:
            callback: The actual controller completion gate.
            args: Positional callback inputs.
            kwargs: Keyword callback inputs.

        Returns:
            The actual callback result if it succeeds.
        """
        try:
            return callback(*args, **kwargs)
        except RuntimeError:
            ticket = controller.begin_session_close(
                source.id,
                expected_revision=controller.lifecycle_impact(
                    session_id=source.id
                ).revision,
            )
            controller.finalize_session_close(ticket)
            raise

    controller.complete_agent_chat_create = place_then_fail
    monkeypatch.setattr(
        controller.app, "call_from_thread", retire_before_worker_receives_error
    )
    with pytest.raises(RuntimeError, match="completion failed after placement"):
        controller.execute_agent_chat_create(
            {"tool": "new_chat", "session_id": source.id, "title": "Placed chat"}
        )
    assert len(placed) == 1
    assert placed[0] in controller.store.sessions()
    assert db.get_conversation_by_id(placed[0].persisted_conversation_id) is not None


def test_prepared_new_chat_destination_routing_disclosed_and_durable(
    real_db_controller, monkeypatch
):
    controller, db = real_db_controller
    _routing(monkeypatch, enabled=True, allowlist=("llama_cpp",))
    _app_config(monkeypatch)
    source = _routed_session(controller)
    result = _prepared_execute(
        controller, {"tool": "new_chat", "session_id": source.id, "model": "m2"}
    )
    assert result["ok"], result
    prepared = controller._last_routing_prepared
    assert prepared["provider"] == "llama_cpp" and prepared["model"] == "m2"
    assert prepared["resolved_instructions"] == "destination instructions"
    row = db.get_conversation_by_id(result["conversation_id"])
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        parse_console_generation_settings,
    )

    snapshot = parse_console_generation_settings(row["metadata"]).snapshot
    assert snapshot.provider == "llama_cpp" and snapshot.model == "m2"


def test_prepared_new_chat_model_only_uses_destination_provider(
    real_db_controller, monkeypatch
):
    from dataclasses import replace

    controller, db = real_db_controller
    _routing(monkeypatch, enabled=True, allowlist=("llama_cpp",))
    _app_config(monkeypatch)
    source = _routed_session(controller)
    source.settings = replace(source.settings, provider="openai", model="source-only")
    result = _prepared_execute(
        controller, {"tool": "new_chat", "session_id": source.id, "model": "m2"}
    )
    assert result["ok"], result
    assert controller._last_routing_prepared["provider"] == "llama_cpp"


def test_prepared_new_chat_preset_params_win_over_config_and_keep_destination_identity(
    real_db_controller, monkeypatch
):
    from types import SimpleNamespace
    from tldw_chatbook import config

    controller, db = real_db_controller
    _routing(monkeypatch, enabled=False, allowlist=())
    configured = {
        **_FAKE_APP_CONFIG,
        "chat_defaults": {"temperature": 0.9},
        "console": {"provider_defaults": {"llama_cpp": {"temperature": 0.8}}},
    }
    monkeypatch.setattr(config, "load_settings", lambda: configured)
    controller._agent_bridge = SimpleNamespace(
        agent_runs_db=SimpleNamespace(
            list_agent_definitions=lambda enabled_only: [
                {
                    "name": "tuned",
                    "description": "d",
                    "instructions": "preset persona must not ride",
                    "tool_allowlist": [],
                    "provider": "llama_cpp",
                    "model": "m1",
                    "enabled": 1,
                    "params": {"temperature": 0.2, "top_p": 0.4, "max_tokens": 123},
                }
            ]
        )
    )
    source = _routed_session(controller)
    result = _prepared_execute(
        controller, {"tool": "new_chat", "session_id": source.id, "preset": "tuned"}
    )
    assert result["ok"], result
    target = next(
        s
        for s in controller.store.sessions()
        if s.persisted_conversation_id == result["conversation_id"]
    )
    assert target.settings.temperature == 0.2
    assert target.settings.top_p == 0.4 and target.settings.max_tokens == 123
    assert target.settings.system_prompt == "destination instructions"
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        parse_console_generation_settings,
    )

    row = db.get_conversation_by_id(result["conversation_id"])
    durable = parse_console_generation_settings(row["metadata"]).snapshot
    assert (
        durable.temperature == 0.2
        and durable.top_p == 0.4
        and durable.max_tokens == 123
    )


@pytest.fixture
def child_new_chat_rig(real_db_controller, tmp_path):
    """A trusted child actor backed by genuine native run and conversation rows."""
    from threading import Event
    from types import SimpleNamespace
    from tldw_chatbook.Agents.run_context import CurrentRunActor
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    controller, db = real_db_controller
    source = controller.store.create_session(title="Parent chat")
    source.persisted_conversation_id = controller.store.persistence.create_conversation(
        conversation_title="Parent chat"
    )
    runs = AgentRunsDB(tmp_path / "child-runs.sqlite")
    parent = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary",
        assistant_message_id="parent-message",
    )
    child = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="subagent",
        parent_run_id=parent,
        task="Follow-up work",
    )
    controller._agent_bridge = SimpleNamespace(
        runs_db=runs,
        agent_runs_db=runs,
        live_primary_run_id=lambda conversation: parent,
    )
    controller._active_cancel_events[source.id] = Event()
    controller._active_assistant_message_ids[source.id] = "parent-message"
    controller.app.app_config = _FAKE_APP_CONFIG
    runtime = SimpleNamespace(_app=controller.app)
    runtime._resolve_new_console_assistant = lambda workspace, settings: (
        ConsoleRuntime._resolve_new_console_assistant(runtime, workspace, settings)
    )
    controller.app.console_runtime = runtime
    controller._default_session_settings = lambda: ConsoleSessionSettings(
        provider="llama_cpp", model="destination-model"
    )
    actor = CurrentRunActor("subagent", child, parent)
    payload = {
        "tool": "new_chat",
        "session_id": source.id,
        "source_run_id": child,
        "source_message_id": "parent-message",
        "title": "Child draft",
        "opening_prompt": "literal /new @helper",
        "instructions": "",
    }
    try:
        yield controller, db, runs, source, actor, payload
    finally:
        runs.close()


def test_child_new_chat_prepares_confirms_executes_and_requires_fresh_approval(
    child_new_chat_rig,
):
    from tldw_chatbook.Agents.run_context import use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    cards = []

    def approve(card):
        if card:
            cards.append(dict(card))
            controller.resolve_pending_chat_create(True, True, card["request_id"])

    controller.set_pending_chat_create = approve
    controller._start_created_chat = lambda *args: pytest.fail(
        "child draft acquired start authority"
    )
    _, new_chat = build_chat_create_tool_closures(
        confirm=lambda prepared: controller.request_chat_create_confirm(
            prepared, session_id=source.id
        ),
        execute=controller.execute_agent_chat_create,
        prepare=controller.prepare_agent_chat_create,
        session_id=source.id,
        run_id="parent-message",
    )
    source.draft = "parent composer custody"
    with use_run_actor(actor):
        first = new_chat(
            {"title": "Child draft", "opening_prompt": "literal /new @helper"}
        )
        assert first.ok, first.error
        second = new_chat({"title": "Child draft 2", "opening_prompt": "second draft"})
        assert second.ok, second.error
    assert len(cards) == 2
    assert cards[0]["request_id"] != cards[1]["request_id"]
    for card in cards:
        assert card["run_id"] == actor.run_id
        assert (
            card["agent_kind"] == "subagent"
            and card["parent_run_id"] == actor.parent_run_id
        )
        assert card["destination"] == "same_workspace" and card["mode"] == "draft"
    for result, prompt in ((first, "literal /new @helper"), (second, "second draft")):
        outcome = json.loads(result.content)
        assert outcome["launch_status"] == "draft"
        assert outcome["draft_set"] and outcome["copied_messages"] == 0
        row = db.get_conversation_by_id(outcome["conversation_id"])
        metadata = json.loads(row["metadata"])
        assert metadata["console_agent_handoff"]["draft"] == prompt
        assert metadata["console_agent_handoff"]["source_run_id"] == actor.run_id
        assert not db.get_messages_for_conversation(outcome["conversation_id"])
    assert controller.store.active_session_id == source.id
    assert source.draft == "parent composer custody"
    assert not controller._chat_create_session_grants.get(source.id)
    assert (
        not controller._chat_creation_records
        and not controller.pending_chat_create_ids()
    )
    with runs.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM automatic_chat_start_attempts"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("extra", [{"destination": "casual"}, {"mode": "start"}])
def test_child_new_chat_refuses_new_destination_and_start_authority(
    child_new_chat_rig, extra
):
    from tldw_chatbook.Agents.run_context import use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    with use_run_actor(actor), pytest.raises(PermissionError):
        controller.prepare_agent_chat_create({**payload, **extra})
    assert not controller._chat_creation_records


@pytest.mark.parametrize(
    "mutation", ["terminal_child", "wrong_actor", "source_incarnation"]
)
def test_child_prepared_create_rechecks_currentness_before_execution(
    child_new_chat_rig, mutation
):
    from dataclasses import replace
    from tldw_chatbook.Agents.run_context import use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    controller.set_pending_chat_create = lambda card: (
        controller.resolve_pending_chat_create(True, False, card["request_id"])
        if card
        else None
    )
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        assert controller.request_chat_create_confirm(prepared)["allow"]
        if mutation == "terminal_child":
            with runs.transaction() as connection:
                connection.execute(
                    "UPDATE agent_runs SET status='cancelled' WHERE id=?",
                    (actor.run_id,),
                )
        elif mutation == "source_incarnation":
            source.incarnation_id = "replacement-incarnation"
        acting = (
            replace(actor, parent_run_id="other-parent")
            if mutation == "wrong_actor"
            else actor
        )
        with use_run_actor(acting):
            outcome = controller.execute_agent_chat_create(prepared)
    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records


def test_survivor_child_draft_ignores_primary_slot_and_standing_grant(
    child_new_chat_rig,
):
    from threading import Event
    from tldw_chatbook.Agents.run_context import use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    with runs.transaction() as connection:
        connection.execute(
            "UPDATE agent_runs SET status='done' WHERE id=?", (actor.parent_run_id,)
        )
    controller._agent_bridge.live_primary_run_id = lambda conversation: None
    controller._active_cancel_events.pop(source.id)
    controller._active_assistant_message_ids.pop(source.id)
    scope = (source.incarnation_id, "new_chat", "global", None, "draft")
    controller._chat_create_session_grants[source.id] = {scope}
    cards = []

    def approve(card):
        if card:
            cards.append(dict(card))
            controller.resolve_pending_chat_create(True, True, card["request_id"])

    controller.set_pending_chat_create = approve
    with use_run_actor(actor):
        first = controller.prepare_agent_chat_create(payload)
        assert not controller._chat_creation_records[first["_creation_token"]][
            "approved"
        ]
        assert controller.request_chat_create_confirm(first, session_id=source.id) == {
            "allow": True,
            "remember": False,
        }
        assert controller.execute_agent_chat_create(first)["ok"]
        controller._active_cancel_events[source.id] = Event()
        controller._active_assistant_message_ids[source.id] = "unrelated-next-turn"
        second = controller.prepare_agent_chat_create(
            {**payload, "title": "Later child draft"}
        )
        assert controller.request_chat_create_confirm(second, session_id=source.id) == {
            "allow": True,
            "remember": False,
        }
        assert controller.execute_agent_chat_create(second)["ok"]
    assert len(cards) == 2 and cards[0]["request_id"] != cards[1]["request_id"]
    assert controller._chat_create_session_grants[source.id] == {scope}
    assert not controller._chat_creation_records


@pytest.mark.parametrize("before_prepare", [True, False])
def test_child_draft_observes_its_parent_turn_stop(child_new_chat_rig, before_prepare):
    from tldw_chatbook.Agents.run_context import use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    cancel = controller._active_cancel_events[source.id]
    controller.set_pending_chat_create = lambda card: (
        controller.resolve_pending_chat_create(True, False, card["request_id"])
        if card
        else None
    )
    with use_run_actor(actor):
        if before_prepare:
            cancel.set()
            with pytest.raises(PermissionError):
                controller.prepare_agent_chat_create(payload)
        else:
            prepared = controller.prepare_agent_chat_create(payload)
            assert controller.request_chat_create_confirm(prepared)["allow"]
            cancel.set()
            # Completion can pop this turn's slot; its captured Stop still wins.
            controller._active_cancel_events.pop(source.id)
            controller._active_assistant_message_ids.pop(source.id)
            outcome = controller.execute_agent_chat_create(prepared)
            assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records


def test_primary_remembered_bridge_still_confirms_each_child_request(
    child_new_chat_rig,
):
    """A shared bridge memo must never inherit primary approval into a child."""
    from tldw_chatbook.Agents.run_context import use_run_id, use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    cards = []

    def approve(card):
        if card:
            cards.append(dict(card))
            controller.resolve_pending_chat_create(True, True, card["request_id"])

    controller.set_pending_chat_create = approve
    _, new_chat = build_chat_create_tool_closures(
        confirm=lambda prepared: controller.request_chat_create_confirm(
            prepared, session_id=source.id
        ),
        execute=controller.execute_agent_chat_create,
        prepare=controller.prepare_agent_chat_create,
        session_id=source.id,
        run_id="parent-message",
    )
    source.draft = "primary composer sentinel"
    with use_run_id(actor.parent_run_id):
        primary = new_chat({"title": "Primary", "opening_prompt": "primary draft"})
        assert primary.ok, primary.error
        remembered = new_chat(
            {
                "title": "Remembered",
                "opening_prompt": "another",
                "source_agent_kind": "subagent",
            }
        )
        assert remembered.ok, remembered.error
    assert len(cards) == 1
    assert controller._chat_create_session_grants[source.id]
    with use_run_actor(actor):
        first = new_chat(
            {
                "title": "Child one",
                "opening_prompt": "child one",
                "source_agent_kind": "primary",
            }
        )
        assert first.ok, first.error
        second = new_chat({"title": "Child two", "opening_prompt": "child two"})
        assert second.ok, second.error
    assert len(cards) == 3
    assert len({card["request_id"] for card in cards}) == 3
    assert [card["agent_kind"] for card in cards] == ["primary", "subagent", "subagent"]
    for card in cards[1:]:
        assert card["run_id"] == actor.run_id
        assert card["parent_run_id"] == actor.parent_run_id
    for result, prompt in ((first, "child one"), (second, "child two")):
        outcome = json.loads(result.content)
        assert outcome["launch_status"] == "draft"
        assert not db.get_messages_for_conversation(outcome["conversation_id"])
        handoff = json.loads(
            db.get_conversation_by_id(outcome["conversation_id"])["metadata"]
        )["console_agent_handoff"]
        assert handoff["draft"] == prompt and handoff["source_run_id"] == actor.run_id
    assert source.draft == "primary composer sentinel"
    assert controller.store.active_session_id == source.id
    assert (
        not controller._chat_creation_records
        and not controller.pending_chat_create_ids()
    )


@pytest.mark.parametrize(
    ("arguments", "category"),
    [
        ({"title": "x" * 121}, "payload_too_large"),
        ({"opening_prompt": "x" * 20_001}, "payload_too_large"),
        ({"instructions": "x" * 20_001}, "payload_too_large"),
        ({"destination": ["secret-payload"]}, "invalid_args"),
        ({"mode": None}, "invalid_args"),
        ({"mode": "start", "opening_prompt": " \n\t"}, "invalid_args"),
        ({"provider": {"secret-payload": "private"}}, "invalid_args"),
    ],
)
def test_new_chat_tool_and_controller_reject_before_authority_or_execution(
    real_db_controller, arguments, category
):
    controller, _db = real_db_controller
    confirm, executor = _FakeConfirm([]), _FakeExecutor()
    preparations = []
    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm,
        execute=executor,
        session_id="s1",
        run_id="r1",
        prepare=lambda payload: preparations.append(payload),
    )
    result = new_chat(arguments)
    assert not result.ok and result.error.startswith(category + ":")
    assert "secret-payload" not in result.error
    assert confirm.payloads == executor.calls == preparations == []
    with pytest.raises(ValueError) as captured:
        controller.prepare_agent_chat_create(arguments)
    assert str(captured.value).startswith(category + ":")
    assert "secret-payload" not in str(captured.value)
