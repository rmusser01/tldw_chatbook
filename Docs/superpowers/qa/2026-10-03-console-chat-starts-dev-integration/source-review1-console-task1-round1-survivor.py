from pathlib import Path
p=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/Tests/Chat/test_console_chat_create_confirm.py')
p.write_text(p.read_text()+'''

def test_survivor_child_confirm_does_not_bind_next_turn_stop(child_new_chat_rig, monkeypatch):
    from threading import Event
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    unrelated_cancel = Event()
    unrelated_cancel.set()
    controller._active_cancel_events[source.id] = unrelated_cancel
    controller._active_assistant_message_ids[source.id] = "next-primary-message"
    cards = []
    controller.set_pending_chat_create = lambda card: cards.append(dict(card)) if card else None
    original = controller._is_session_cancelled
    polls = []
    def inspect_cancel(session_id, *, cancel_event, visit_event):
        polls.append(cancel_event)
        assert cancel_event is None, "survivor approval bound another turn's Stop"
        assert not original(session_id, cancel_event=cancel_event, visit_event=visit_event)
        controller.resolve_pending_chat_create(True, False, cards[0]["request_id"])
        return False
    monkeypatch.setattr(controller, "_is_session_cancelled", inspect_cancel)
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        assert controller.request_chat_create_confirm(prepared, session_id=source.id)["allow"]
        assert controller.execute_agent_chat_create(prepared)["ok"]
    assert polls == [None]
    assert not controller._chat_creation_records and not controller.pending_chat_create_ids()
''')
