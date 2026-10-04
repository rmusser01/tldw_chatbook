from pathlib import Path
p=Path('tldw_chatbook/Chat/console_chat_controller.py');s=p.read_text()
pos=s.index('\n\nclass ConsoleChatController:')
s=s[:pos]+'''

@dataclass(frozen=True, slots=True)
class _ChatCreationObservation:
    """One validation phase's identities and unlocked durable observations."""

    payload: dict[str, Any]
    record: dict[str, Any] | None
    source: Any
    bridge: Any
    runs_db: Any
    incarnation: str
    conversation_id: str | None
    workspace_id: str | None
    actor: Any
    row: dict[str, Any] | None = None
    parent: dict[str, Any] | None = None
'''+s[pos:]
start=s.index('    def _chat_creation_source_live(');end=s.index('    def _chat_creation_destination_available(',start)
old=s[start:end]
body=old[old.index('        session = '):]
body=body.replace('        session = self.store._sessions.get(str(payload.get("session_id") or ""))\n','''        payload = observation.payload
        session = self.store._sessions.get(str(payload.get("session_id") or ""))
''',1)
body=body.replace('            self._disposed\n','''            self._disposed
            or session is not observation.source
            or bridge is not observation.bridge
            or getattr(bridge, "runs_db", None) is not observation.runs_db
            or current_run_actor() != observation.actor
''',1)
body=body.replace('        if (\n            payload.get("source_incarnation", session.incarnation_id)','''        if (
            session.incarnation_id != observation.incarnation
            or session.persisted_conversation_id != observation.conversation_id
            or session.workspace_id != observation.workspace_id
            or payload.get("source_incarnation", session.incarnation_id)''',1)
body=body.replace('row = bridge.runs_db.get_run(run_id)','row = observation.row',1)
body=body.replace('parent = bridge.runs_db.get_run(parent_id) if parent_id else None','parent = observation.parent',1)
s=s[:start]+'''    def _capture_chat_creation_source(
        self, payload: Mapping[str, Any], record: dict[str, Any] | None = None
    ) -> _ChatCreationObservation | None:
        """Capture local identities without reading storage."""
        source = self.store._sessions.get(str(payload.get("session_id") or ""))
        bridge = self._agent_bridge
        runs_db = getattr(bridge, "runs_db", None)
        if source is None or runs_db is None:
            return None
        return _ChatCreationObservation(
            payload=dict(payload),
            record=record,
            source=source,
            bridge=bridge,
            runs_db=runs_db,
            incarnation=source.incarnation_id,
            conversation_id=source.persisted_conversation_id,
            workspace_id=source.workspace_id,
            actor=current_run_actor(),
        )

    def _read_chat_creation_source(
        self, observation: _ChatCreationObservation | None
    ) -> _ChatCreationObservation | None:
        """Observe run rows with no registry lock; never reuse across phases."""
        if observation is None:
            return None
        payload = observation.payload
        try:
            row = observation.runs_db.get_run(payload.get("source_run_id"))
            parent_id = payload.get("source_parent_run_id")
            parent = (
                observation.runs_db.get_run(parent_id)
                if payload.get("source_agent_kind") == "subagent" and parent_id
                else None
            )
            fields = ("conversation_id", "agent_kind", "parent_run_id", "status")
            return replace(
                observation,
                row={key: row.get(key) for key in fields} if row else None,
                parent={key: parent.get(key) for key in fields} if parent else None,
            )
        except Exception:
            return None

    def _chat_creation_source_live(self, payload: Mapping[str, Any]) -> bool:
        """Read anew, then require the captured source and current runtime owner."""
        observation = self._read_chat_creation_source(
            self._capture_chat_creation_source(payload)
        )
        return observation is not None and self._chat_creation_source_matches(
            observation
        )

    def _chat_creation_source_matches(
        self, observation: _ChatCreationObservation
    ) -> bool:
        """Check copied row fields and live memory only; never enter SQLite."""
'''+body+s[end:]
start=s.index('    def _chat_creation_record(\n');end=s.index('    def _start_created_chat(',start)
s=s[:start]+'''    def _observe_chat_creation_record(
        self, payload: Mapping[str, Any]
    ) -> _ChatCreationObservation | None:
        """Capture the exact registry entry, release its lock, then read rows."""
        with self._pending_chat_create_lock:
            record = self._chat_creation_records.get(payload.get("_creation_token"))
            if record is None or any(
                payload.get(key) != value for key, value in record["payload"].items()
            ):
                return None
            observation = self._capture_chat_creation_source(record["payload"], record)
        return self._read_chat_creation_source(observation)

    def _chat_creation_record(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        observation = self._observe_chat_creation_record(payload)
        with self._pending_chat_create_lock:
            return self._chat_creation_record_locked(payload, observation)

    def _chat_creation_record_locked(
        self,
        payload: Mapping[str, Any],
        observation: _ChatCreationObservation | None,
    ) -> dict[str, Any] | None:
        """Recheck exact identity and current fences with no storage under lock."""
        record = self._chat_creation_records.get(payload.get("_creation_token"))
        if (
            observation is None
            or record is None
            or record is not observation.record
            or record["payload"] != observation.payload
            or any(payload.get(key) != value for key, value in record["payload"].items())
        ):
            return None
        source = self.store._sessions.get(record["payload"]["session_id"])
        if (
            not self._chat_creation_source_matches(observation)
            or (
                record.get("source_cancel_event") is not None
                and record["source_cancel_event"].is_set()
            )
            or source is None
            or source.workspace_id != record["payload"]["source_workspace_id"]
        ):
            return None
        return record

'''+s[end:]
s=s.replace('''        # Decide a remembered grant atomically with Close and revocation.
        with self._pending_chat_create_lock:''','''        observation = (
            self._observe_chat_creation_record(payload) if tool == "new_chat" else None
        )
        # Decide a remembered grant atomically with Close and revocation.
        with self._pending_chat_create_lock:''',1)
s=s.replace('self._chat_creation_record_locked(payload)\n','self._chat_creation_record_locked(payload, observation)\n',1)
s=s.replace('''            with self._pending_chat_create_lock:
                if (
                    chat_create_round_state.get("revoked")''','''            # Human approval is a new observation phase; never reuse its entry rows.
            observation = (
                self._observe_chat_creation_record(payload)
                if tool == "new_chat" and decision.get("allow", False)
                else None
            )
            with self._pending_chat_create_lock:
                if (
                    chat_create_round_state.get("revoked")''',1)
s=s.replace('allow = self._chat_creation_record_locked(payload) is record','allow = self._chat_creation_record_locked(payload, observation) is record',1)
p.write_text(s)
