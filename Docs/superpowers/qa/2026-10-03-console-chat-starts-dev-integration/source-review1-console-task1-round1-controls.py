from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
p=root/'Tests/Chat/test_console_chat_create_confirm.py';s=p.read_text();s=s.replace('from Tests.Chat.test_console_chat_create_integration import child_new_chat_rig, real_db_controller\n','');s=s.replace('from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until  # reuse fakes\n','from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until  # reuse fakes\nfrom Tests.Chat.test_console_chat_create_integration import child_new_chat_rig, real_db_controller\n');p.write_text(s+'''

def test_child_approved_creation_revocation_closes_prepared_token(child_new_chat_rig):
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    controller.set_pending_chat_create = lambda card: (
        controller.resolve_pending_chat_create(True, False, card["request_id"]) if card else None)
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        assert controller.request_chat_create_confirm(prepared)["allow"]
        assert controller.revoke_approval_rounds_for_run(actor.run_id) == 0
        outcome = controller.execute_agent_chat_create(prepared)
    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records


def test_child_declined_creation_releases_prepared_token(child_new_chat_rig):
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    controller.set_pending_chat_create = lambda card: (
        controller.resolve_pending_chat_create(False, True, card["request_id"]) if card else None)
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        assert not controller.request_chat_create_confirm(prepared)["allow"]
        outcome = controller.execute_agent_chat_create(prepared)
    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records and not controller._chat_create_session_grants.get(source.id)
''')
p=root/'Tests/Chat/test_console_chat_create_integration.py';p.write_text(p.read_text()+'''

def test_survivor_child_draft_ignores_primary_slot_and_standing_grant(child_new_chat_rig):
    from threading import Event
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    with runs.transaction() as connection:
        connection.execute("UPDATE agent_runs SET status='done' WHERE id=?", (actor.parent_run_id,))
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
        assert not controller._chat_creation_records[first["_creation_token"]]["approved"]
        assert controller.request_chat_create_confirm(first, session_id=source.id) == {"allow": True, "remember": False}
        assert controller.execute_agent_chat_create(first)["ok"]
        controller._active_cancel_events[source.id] = Event()
        controller._active_assistant_message_ids[source.id] = "unrelated-next-turn"
        second = controller.prepare_agent_chat_create({**payload, "title": "Later child draft"})
        assert controller.request_chat_create_confirm(second, session_id=source.id) == {"allow": True, "remember": False}
        assert controller.execute_agent_chat_create(second)["ok"]
    assert len(cards) == 2 and cards[0]["request_id"] != cards[1]["request_id"]
    assert controller._chat_create_session_grants[source.id] == {scope}
    assert not controller._chat_creation_records
''')
