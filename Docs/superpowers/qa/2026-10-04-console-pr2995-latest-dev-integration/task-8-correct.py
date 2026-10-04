from pathlib import Path
p=Path('tldw_chatbook/Chat/console_chat_controller.py');s=p.read_text()
s=s.replace('''            self._disposed
            or session is None
            or not run_id
''','''            self._disposed
            or session is None
            or not self._chat_create_source_is_open(session.id)
            or not run_id
''',1)
a=s.index('    def _chat_creation_record(');b=s.index('    def _start_created_chat(',a)
old=s[a:b];body=old[old.index('            record ='):];body='\n'.join(line[4:] if line.startswith('    ') else line for line in body.split('\n'))
s=s[:a]+'''    def _chat_creation_record(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        with self._pending_chat_create_lock:
            return self._chat_creation_record_locked(payload)

    def _chat_creation_record_locked(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Validate exact preparation while the caller holds the registry lock."""
'''+body+s[b:]
a=s.index('        # Session-scoped remember:',s.index('    def request_chat_create_confirm'));b=s.index('        if self.app is None',a)
s=s[:a]+'''        # Decide a remembered grant atomically with Close and revocation.
        with self._pending_chat_create_lock:
            record = (
                self._chat_creation_record_locked(payload)
                if tool == "new_chat"
                else None
            )
            refused = owning_session_id in self._session_close_generations or (
                tool == "new_chat" and record is None
            )
            grant = payload["_grant_scope"] if record is not None else tool
            if (
                not refused
                and requesting_kind == AGENT_KIND_PRIMARY
                and grant in self._chat_create_session_grants.get(owning_session_id, set())
            ):
                if record is not None:
                    record["approved"] = True
                return {"allow": True, "remember": True}
        if refused:
            token = payload.get("_creation_token")
            if isinstance(token, _ChatCreationToken):
                token.close()
            return {"allow": False, "remember": False}
'''+s[b:]
s=s.replace('allow = self._chat_creation_record(payload) is record','allow = self._chat_creation_record_locked(payload) is record',1)
s=s.replace('''            self.store.persistence.persist_console_conversation_with_policy(
                conversation_id=conversation_id,''','''            # No registry lock spans SQLite I/O. A successful save owns the
            # durable draft even if Close retires its source during that write.
            if self._chat_creation_record(payload) is not record or not record["approved"]:
                raise PermissionError("source_unavailable")
            self.store.persistence.persist_console_conversation_with_policy(
                conversation_id=conversation_id,''',1)
s=s.replace('''                def restore():
                    created_session''','''                def restore():
                    if not self._chat_create_source_is_open(approved["session_id"]):
                        return None
                    created_session''',1)
s=s.replace('''                if approved["mode"] == "start":
                    result.update(self._start_created_chat(approved, target))
            except Exception:
                result["reason"] = "restore_unavailable"''','''                if target is None:
                    result["reason"] = "source_unavailable"
                elif approved["mode"] == "start":
                    result.update(self._start_created_chat(approved, target))
            except Exception:
                result["reason"] = (
                    "restore_unavailable"
                    if self._chat_create_source_is_open(approved["session_id"])
                    else "source_unavailable"
                )''',1)
s=s.replace('if result.get("reason") in {"restore_unavailable", "runtime_unavailable"}:','if result.get("reason") in {"restore_unavailable", "runtime_unavailable", "source_unavailable"}:',1)
a=s.index('            if self.complete_agent_chat_create is not None:',s.index('    def _execute_prepared_chat_create'));b=s.index('            return result',a)
s=s[:a]+'''            def complete_if_source_open():
                if not self._chat_create_source_is_open(approved["session_id"]):
                    return
                complete = self.complete_agent_chat_create
                if complete is not None:
                    complete(
                        session_id=approved["session_id"],
                        conversation_id=conversation_id,
                        title=title,
                        tool="new_chat",
                        opening_prompt=approved["opening_prompt"],
                        workspace_id=approved["workspace_id"] or CONSOLE_GLOBAL_WORKSPACE_ID,
                        launch_status=result["launch_status"],
                        reason=result.get("reason"),
                    )

            try:
                self.app.call_from_thread(complete_if_source_open)
            except Exception:
                pass
'''+s[b:]
a=s.index('    def execute_agent_chat_create');b=s.index('        session = next',a);chunk=s[a:b];chunk=chunk.replace('''        if payload.get("tool") == "new_chat":
            return self._execute_prepared_chat_create(payload)

''','');chunk+='''        if tool == "new_chat":
            return self._execute_prepared_chat_create(payload)

''';chunk=chunk.replace('''        refuses UI placement and discards its newly committed row through
        the existing best-effort orphan soft-delete helper.''','''        refuses UI placement. Legacy forks discard their orphan row; prepared
        new chats retain their successfully saved draft before launch.''');s=s[:a]+chunk+s[b:];p.write_text(s)
