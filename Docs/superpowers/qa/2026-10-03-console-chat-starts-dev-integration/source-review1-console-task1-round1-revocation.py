from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
p=root/'tldw_chatbook/Chat/console_chat_controller.py';s=p.read_text()
def swap(old,new):
 global s
 assert s.count(old)==1,s.count(old);s=s.replace(old,new)
swap('''        self._chat_creation_records: dict[object, dict[str, Any]] = {}
''','''        self._chat_creation_records: dict[object, dict[str, Any]] = {}
        self._chat_creation_revoked_runs: set[str] = set()
''')
swap('''        with self._pending_chat_create_lock:
            for request_id, state in list(self._pending_chat_create_rounds.items()):
''','''        with self._pending_chat_create_lock:
            self._chat_creation_revoked_runs.add(run_id)
            for token, record in list(self._chat_creation_records.items()):
                if record["payload"].get("source_run_id") == run_id:
                    self._chat_creation_records.pop(token, None)
            for request_id, state in list(self._pending_chat_create_rounds.items()):
''')
swap('''        if self._disposed or session is None or not run_id or bridge is None:
''','''        if (
            self._disposed or session is None or not run_id or bridge is None
            or run_id in self._chat_creation_revoked_runs
        ):
''')
swap('''            if len(self._chat_creation_records) >= 64:
                raise PermissionError("creation_capacity")
''','''            if payload["source_run_id"] in self._chat_creation_revoked_runs:
                raise PermissionError("source_unavailable")
            if len(self._chat_creation_records) >= 64:
                raise PermissionError("creation_capacity")
''')
# This exact creation round owns an earlier child turn, never the next turn's Stop.
start=s.index('    def request_chat_create_confirm(');end=s.index('    def _remount_parked_chat_create(',start)
section=s[start:end];old='''        round_cancel_event = self._bind_round_cancel_signal(session_id)
''';assert section.count(old)==1
section=section.replace(old,'''        round_cancel_event = self._bind_round_cancel_signal(session_id)
        if (
            record is not None
            and requesting_kind == "subagent"
            and self._active_assistant_message_ids.get(owning_session_id)
            != record["payload"]["source_message_id"]
        ):
            round_cancel_event = None
''')
s=s[:start]+section+s[end:];p.write_text(s)
p=root/'Tests/Chat/test_console_chat_create_confirm.py';s=p.read_text();needle='''    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records


def test_child_declined''';assert needle in s;s=s.replace(needle,'''    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records
    with use_run_actor(actor), pytest.raises(PermissionError):
        controller.prepare_agent_chat_create(payload)


def test_child_declined''');p.write_text(s)
