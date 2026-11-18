from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
p=root/'Tests/Chat/test_console_chat_create_confirm.py';p.write_text(p.read_text()+'''

def test_child_revocation_between_record_check_and_arm_does_not_show_card(child_new_chat_rig, monkeypatch):
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    cards = []
    def approve(card):
        if card:
            cards.append(dict(card))
            controller.resolve_pending_chat_create(True, False, card["request_id"])
    controller.set_pending_chat_create = approve
    bind_visit = controller._bind_visit_cancel_signal
    def revoke_before_arm():
        controller.revoke_approval_rounds_for_run(actor.run_id)
        return bind_visit()
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        monkeypatch.setattr(controller, "_bind_visit_cancel_signal", revoke_before_arm)
        assert not controller.request_chat_create_confirm(prepared, session_id=source.id)["allow"]
    assert cards == []
    assert not controller._chat_creation_records and not controller.pending_chat_create_ids()
''')
p=root/'Tests/Chat/test_console_chat_create_integration.py';p.write_text(p.read_text()+'''

@pytest.mark.parametrize("before_prepare", [True, False])
def test_child_draft_observes_its_parent_turn_stop(child_new_chat_rig, before_prepare):
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    cancel = controller._active_cancel_events[source.id]
    controller.set_pending_chat_create = lambda card: (
        controller.resolve_pending_chat_create(True, False, card["request_id"]) if card else None)
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
''')
