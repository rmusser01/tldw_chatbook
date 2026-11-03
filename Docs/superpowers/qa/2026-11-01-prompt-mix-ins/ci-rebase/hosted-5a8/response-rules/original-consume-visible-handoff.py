    def _consume_visible_agent_handoff(self, store: Any, composer: Any) -> None:
        """Apply an accepted handoff receipt only to its exact visible revision."""
        captured = self._visible_agent_handoff_draft
        if captured is None:
            return
        session_id, incarnation, revision, snapshot = captured
        session = next((row for row in store.sessions() if row.id == session_id), None)
        if session is None or session.incarnation_id != incarnation:
            self._visible_agent_handoff_draft = None
            return
        if session.agent_handoff_state != "consumed":
            return
        self._visible_agent_handoff_draft = None
        live_snapshot = composer.capture_draft_snapshot()
        if (
            self._console_visible_draft_session_id == session_id
            and session.agent_handoff_revision == revision + 1
            and live_snapshot.generation == snapshot.generation
            and live_snapshot.edit_serial == snapshot.edit_serial
            and live_snapshot.segments == snapshot.segments
        ):
            # Caret/selection navigation does not author a new draft. Keep
            # widget generation, authored revision and segment identity fenced.
            if composer.commit_captured_draft(composer.capture_draft_for_send()):
                self._console_draft_switch_snapshot = None
