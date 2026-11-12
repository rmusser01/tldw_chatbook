from pathlib import Path
p=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py');s=p.read_text()
def swap(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
swap('''                parent = bridge.runs_db.get_run(parent_id) if parent_id else None
                return bool(
''','''                parent = bridge.runs_db.get_run(parent_id) if parent_id else None
                cancel = self._active_cancel_events.get(session.id)
                owns_parent_turn = self._active_assistant_message_ids.get(session.id) == payload.get("source_message_id")
                return bool(
                    (not owns_parent_turn or (cancel is not None and not cancel.is_set()))
''')
swap('''                "startup": startup,
                "approved": not child
''','''                "startup": startup,
                "source_cancel_event": (
                    self._active_cancel_events.get(source.id)
                    if child and self._active_assistant_message_ids.get(source.id) == payload["source_message_id"]
                    else None
                ),
                "approved": not child
''')
swap('''                not self._chat_creation_source_live(record["payload"])
                or source is None
''','''                not self._chat_creation_source_live(record["payload"])
                or (
                    record.get("source_cancel_event") is not None
                    and record["source_cancel_event"].is_set()
                )
                or source is None
''')
start=s.index('    def request_chat_create_confirm(');end=s.index('    def _remount_parked_chat_create(',start);section=s[start:end]
old='''        with self._pending_chat_create_lock:
            self._pending_chat_create_rounds[request_id] = chat_create_round_state
''';assert section.count(old)==1
section=section.replace(old,'''        with self._pending_chat_create_lock:
            if record is not None and (
                true_run_id in self._chat_creation_revoked_runs
                or self._chat_creation_records.get(payload["_creation_token"]) is not record
            ):
                return {"allow": False, "remember": False}
            self._pending_chat_create_rounds[request_id] = chat_create_round_state
''');s=s[:start]+section+s[end:];p.write_text(s)
