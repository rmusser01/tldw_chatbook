from pathlib import Path
p=Path('Tests/UI/test_console_session_tab_close.py');s=p.read_text();s=s.replace('def _pending_close_app(request, kind):','def _pending_close_app(request, kind, *, surviving_child=False):',1);s=s.replace('''        try:
            yield app
        finally:
            db.close_connection()


async def _arm_pending_round(controller, kind: str, session_id: str):''','''        runs = None
        if surviving_child:
            from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

            runs = AgentRunsDB(Path(directory) / "runs.sqlite")
            app._pending_close_runs = runs
        try:
            yield app
        finally:
            if runs is not None:
                runs.close()
            db.close_connection()


def _prepare_surviving_child(controller, session_id):
    """Keep existing child authority live after the primary turn has ended."""
    from types import SimpleNamespace
    from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor

    source = controller.store._sessions[session_id]
    source.persisted_conversation_id = controller.store.persistence.create_conversation(
        conversation_title=source.title
    )
    runs = controller.app._pending_close_runs
    parent = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary", assistant_message_id="finished-parent-message",
    )
    child = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="subagent", parent_run_id=parent,
        run_id=f"close-chat_create-{session_id}",
    )
    controller._agent_bridge = SimpleNamespace(
        runs_db=runs, agent_runs_db=runs,
        live_primary_run_id=lambda conversation: parent,
    )
    actor = CurrentRunActor("subagent", child, parent)
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create({
            "tool": "new_chat", "session_id": session_id,
            "source_run_id": child, "source_message_id": "finished-parent-message",
            "title": "private close chat", "destination": "same_workspace", "mode": "draft",
        })
    return prepared, actor


def _assert_surviving_creation_is_live(controller, pending):
    from tldw_chatbook.Agents.run_context import use_run_actor

    prepared = pending._prepared_creation_payload
    with use_run_actor(pending._prepared_creation_actor):
        assert controller._chat_creation_source_live(prepared)
        record = controller._chat_creation_record(prepared)
        assert record is controller._chat_creation_records[prepared["_creation_token"]]
        assert record["payload"] == prepared
        assert not record["approved"]


async def _arm_pending_round(controller, kind: str, session_id: str, *, surviving_child=False):''',1)
s=s.replace('''    if kind == "chat_create":
        from Tests.Chat.test_console_chat_create_integration import _prepare_close_new_chat

        prepared, creation_run = _prepare_close_new_chat(
            controller, session_id, "private close chat"
        )
''','''    actor = None
    if kind == "chat_create":
        if surviving_child:
            prepared, actor = _prepare_surviving_child(controller, session_id)
            creation_run = actor.run_id
        else:
            from Tests.Chat.test_console_chat_create_integration import _prepare_close_new_chat

            prepared, creation_run = _prepare_close_new_chat(
                controller, session_id, "private close chat"
            )
''',1)
s=s.replace('''    with use_run_id(creation_run if kind == "chat_create" else f"close-{kind}-{session_id}"):
        return asyncio.create_task(asyncio.to_thread(request))
''','''    from tldw_chatbook.Agents.run_context import use_run_actor

    context = (use_run_actor(actor) if actor is not None else
               use_run_id(creation_run if kind == "chat_create" else f"close-{kind}-{session_id}"))
    with context:
        pending = asyncio.create_task(asyncio.to_thread(request))
    if kind == "chat_create":
        pending._prepared_creation_payload = prepared
        pending._prepared_creation_actor = actor
    return pending
''',1)
a=s.index('async def _verify_background_pending_close_releases_round_without_an_active_turn');b=s.index('async def _verify_chat_create_enrichment',a);c=s[a:b];c=c.replace('with _pending_close_app(request, kind) as app:', 'with _pending_close_app(request, kind, surviving_child=kind == "chat_create") as app:',1).replace('pending = await _arm_pending_round(controller, kind, doomed.id)','pending = await _arm_pending_round(controller, kind, doomed.id, surviving_child=kind == "chat_create")',1);x=c.index('                    if kind == "chat_create":\n');y=c.index('                    assert doomed.id not in controller._active_cancel_events',x);c=c[:x]+'''                    if kind == "chat_create":
                        assert await _settle(pilot, lambda: bool(controller._parked_chat_create_payloads))
                        request_id = controller.pending_chat_create_ids()[0]
                        card = controller._parked_chat_create_payloads[request_id]
                        assert card["session_id"] == doomed.id
                        assert card["request_id"] == request_id
                        assert card["_creation_token"] is pending._prepared_creation_payload["_creation_token"]
                        assert not controller._pending_chat_create_rounds[request_id]["event"].is_set()
                        assert controller.pending_round_kinds(doomed.id) == {"chat_create"}
                        _assert_surviving_creation_is_live(controller, pending)
'''+c[y:];s=s[:a]+c+s[b:]
a=s.index('async def _verify_chat_create_enrichment');b=s.index('async def _verify_failed_confirmed_close',a);c=s[a:b];c=c.replace('with _pending_close_app(request, "chat_create") as app:', 'with _pending_close_app(request, "chat_create", surviving_child=True) as app:',1).replace('pending = await _arm_pending_round(controller, "chat_create", doomed.id)','pending = await _arm_pending_round(controller, "chat_create", doomed.id, surviving_child=True)',1).replace('                assert not controller.pending_round_kinds(doomed.id)\n','                _assert_surviving_creation_is_live(controller, pending)\n                assert not controller.pending_round_kinds(doomed.id)\n',1).replace('                assert doomed.id not in controller._active_cancel_events','                assert doomed.id not in controller._active_assistant_message_ids\n                assert doomed.id not in controller._active_cancel_events',1);s=s[:a]+c+s[b:];p.write_text(s)
