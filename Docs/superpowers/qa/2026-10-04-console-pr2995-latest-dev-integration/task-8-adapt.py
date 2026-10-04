from pathlib import Path
p=Path('Tests/Chat/test_console_chat_create_integration.py');s=p.read_text();a=s.index('@pytest.mark.parametrize("tool", ["new_chat", "fork_chat"])\ndef test_confirmed_create')
s=s[:a]+'''def _prepare_close_new_chat(controller, session_id, title):
    """Prepare the exact live primary request used by Close integration fixtures."""
    from types import SimpleNamespace
    from tldw_chatbook.Agents.run_context import use_run_id
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    source = controller.store._sessions[session_id]
    if not source.persisted_conversation_id:
        source.persisted_conversation_id = controller.store.persistence.create_conversation(
            conversation_title=source.title
        )
    message_id = controller._active_assistant_message_ids.get(session_id)
    if message_id is None:
        message_id = controller.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        ).id
        controller._active_assistant_message_ids[session_id] = message_id
    controller._active_cancel_events.setdefault(session_id, threading.Event())
    bridge = controller._agent_bridge
    if bridge is None:
        bridge = SimpleNamespace(_live_primary_runs={}, rows={})
        bridge.runs_db = SimpleNamespace(get_run=lambda run: bridge.rows.get(run))
        bridge.live_primary_run_id = lambda conversation: bridge._live_primary_runs.get(conversation)
        controller._agent_bridge = bridge
    run = f"close-create-{session_id}"
    if hasattr(bridge.runs_db, "create_run"):
        run = bridge.runs_db.create_run(
            conversation_id=source.persisted_conversation_id, agent_kind="primary"
        )
    else:
        bridge.rows[run] = {"id": run, "conversation_id": source.persisted_conversation_id,
                            "agent_kind": "primary", "status": "running"}
    bridge._live_primary_runs[source.persisted_conversation_id] = run
    if not hasattr(controller.app, "console_runtime"):
        controller.app.app_config = {}
        runtime = SimpleNamespace(_app=controller.app)
        runtime._resolve_new_console_assistant = lambda workspace, settings: (
            ConsoleRuntime._resolve_new_console_assistant(runtime, workspace, settings)
        )
        controller.app.console_runtime = runtime
    with use_run_id(run):
        prepared = controller.prepare_agent_chat_create({
            "tool": "new_chat", "session_id": session_id, "title": title,
            "source_run_id": run, "source_message_id": message_id,
        })
    return prepared, run


def _approve_close_new_chat(controller, source, title):
    from tldw_chatbook.Agents.run_context import use_run_id

    prepared, run = _prepare_close_new_chat(controller, source.id, title)
    def approve(pending):
        if pending:
            controller.resolve_pending_chat_create(True, False, pending["request_id"])
    controller.set_pending_chat_create = approve
    with use_run_id(run):
        assert controller.request_chat_create_confirm(prepared)["allow"]
    return prepared


'''+s[a:]
a=s.index('def test_confirmed_create_refuses');b=s.index('def test_inflight_create',a);c=s[a:b];c=c.replace('    before = _live_conversation_count(db)','    payload = (_approve_close_new_chat(controller, source, "Late creation")\n               if tool == "new_chat" else {"tool": tool, "session_id": source.id, "title": "Late creation"})\n    before = _live_conversation_count(db)',1);c=c.replace('{"tool": tool, "session_id": source.id, "title": "Late creation"}\n    )','payload\n    )',1);s=s[:a]+c+s[b:]
a=s.index('def test_inflight_create');b=s.index('@pytest.mark.parametrize("view_change"',a);c=s[a:b];c=c.replace('    entered, release =', '    payload = (_approve_close_new_chat(controller, source, "Late creation")\n               if tool == "new_chat" else {"tool": tool, "session_id": source.id, "title": "Late creation"})\n    entered, release =',1);c=c.replace('    original_create = controller.store.persistence.create_conversation','    persistence_method = ("persist_console_conversation_with_policy" if tool == "new_chat" else "create_conversation")\n    original_create = getattr(controller.store.persistence, persistence_method)',1);c=c.replace('        created.append(conversation_id)','        if tool == "new_chat":\n            conversation_id = kwargs["conversation_id"]\n        created.append(conversation_id)',1);c=c.replace('if close_at.startswith("ui-handoff") and threading.current_thread() is worker:', 'if close_at.startswith("ui-handoff") and threading.current_thread() is worker and not queued:',1);c=c.replace('controller.store.persistence, "create_conversation", paused_create','controller.store.persistence, persistence_method, paused_create',1);c=c.replace('{"tool": tool, "session_id": source.id, "title": "Late creation"}\n            )','payload\n            )',1);c=c.replace('    assert _live_conversation_count(db) == before\n    assert "error" not in result','    assert _live_conversation_count(db) == before + (tool == "new_chat")\n    assert "error" not in result',1);a2=c.index('    assert not result["outcome"]["ok"]');c=c[:a2]+'''    assert completed == []
    assert len(created) == 1
    if tool == "new_chat":
        # ADR219: an approved saved draft remains durable across source Close.
        assert result["outcome"]["ok"]
        assert result["outcome"]["launch_status"] == "draft"
        assert result["outcome"]["reason"] == "source_unavailable"
        row = db.get_conversation_by_id(created[0])
        assert row and not row["deleted"]
        assert json.loads(row["metadata"])["console_agent_handoff"]["state"] == "pending"
        assert not controller._chat_creation_records
        assert not controller.store.sessions()
    else:
        assert not result["outcome"]["ok"]
        assert result["outcome"]["kind"] == "session_gone"
        assert db.get_conversation_by_id(created[0]) is None
        assert db.get_conversation_by_id(created[0], include_deleted=True)["deleted"] == 1


''';s=s[:a]+c+s[b:]
a=s.index('def test_chat_create_completion_uses_the_current_view_sink');b=s.index('def test_admitted_chat_create_completion_error',a);c=s[a:b];c=c.replace('    old_completed, new_completed =', '    prepared = _approve_close_new_chat(controller, source, "Live source")\n    old_completed, new_completed =',1);c=c.replace('{"tool": "new_chat", "session_id": source.id, "title": "Live source"}','prepared',1);s=s[:a]+c+s[b:]
a=s.index('def test_admitted_chat_create_completion_error');b=s.index('def test_prepared_new_chat_destination',a);c=s[a:b];c=c.replace('    placed = []','    prepared = _approve_close_new_chat(controller, source, "Placed chat")\n    original_restore = controller.store.restore_persisted_session\n    placed = []',1);x=c.index('        placed.append(');y=c.index('        raise RuntimeError',x);c=c[:x]+'''        placed.append(original_restore(**kwargs))
'''+c[y:];c=c.replace('    controller.complete_agent_chat_create = place_then_fail','    monkeypatch.setattr(controller.store, "restore_persisted_session", place_then_fail)',1);x=c.index('    with pytest.raises(RuntimeError, match="completion failed after placement"):');y=c.index('    assert len(placed)',x);c=c[:x]+'''    outcome = controller.execute_agent_chat_create(prepared)
    assert outcome["ok"] and outcome["reason"] == "source_unavailable"
'''+c[y:];s=s[:a]+c+s[b:];p.write_text(s)
