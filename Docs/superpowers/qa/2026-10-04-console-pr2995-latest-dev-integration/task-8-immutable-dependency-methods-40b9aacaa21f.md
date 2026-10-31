# Immutable Task8 dependencies at 40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77

Named risks: committed Close atomicity; no lock reentry in source-open predicate; actual child/prepared liveness before Close; native preaccept source fence.

## tldw_chatbook/Chat/console_chat_controller.py:14940 — begin_session_close

```python
    def begin_session_close(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> ConsoleSessionCloseTicket:
        """Fence and cancel one session without deleting it yet.

        Args:
            session_id: Native Console session ID to close.
            expected_revision: Revision of the impact the user approved.

        Returns:
            Opaque ticket required by :meth:`finalize_session_close`.

        Raises:
            ConsoleLifecycleRevisionChanged: The approved impact changed.
            RuntimeError: Recovery or an unreconciled close fence blocks close.
        """
        if session_id in self._failed_session_close_generations:
            raise RuntimeError(CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL)
        if session_id in self._session_close_generations:
            raise RuntimeError(CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL)
        impact = self.lifecycle_impact(session_id=session_id)
        if impact.revision != expected_revision:
            raise ConsoleLifecycleRevisionChanged(
                "Console session activity changed during close."
            )
        recovery = self.store.dispatch_recovery_for_session(session_id)
        if (
            recovery is not None
            and recovery.recovery_needed
            and recovery.kind
            in {
                ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
                ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
            }
        ):
            raise RuntimeError(
                "Finish or discard the pending turn before closing this chat."
            )
        self._session_close_generation += 1
        generation = self._session_close_generation
        fleet_conversation_id = self._agent_conversation_id(session_id)
        fence_fleet = (
            getattr(self._agent_bridge, "fence_fleet", None)
            if self._agent_bridge is not None
            else None
        )
        fleet_fence_acquired = False
        if callable(fence_fleet):
            fleet_fence_acquired = bool(
                fence_fleet(fleet_conversation_id, generation=generation)
            )

        def abort_provisional_fleet_fence() -> None:
            if not fleet_fence_acquired:
                return
            abort_fleet_fence = getattr(
                self._agent_bridge,
                "abort_fleet_fence",
                None,
            )
            try:
                if callable(abort_fleet_fence) and abort_fleet_fence(
                    fleet_conversation_id,
                    generation=generation,
                ):
                    return
            except Exception as exc:  # noqa: BLE001 -- failed rollback stays fenced
                logger.warning(
                    "close_session provisional fleet rollback failed (error_type={})",
                    type(exc).__name__,
                )
            # An uncertain rollback must never be replaced by a retry's new
            # generation. Keep this close fail-closed even without a ticket;
            # surviving children still own valid usage in the retained session.
            self._failed_session_close_generations[session_id] = generation
            logger.warning("close_session provisional fleet fence stayed latched")

        # A reservation publishes its lifecycle revision before the fleet
        # fence can acquire the coordinator lock. Recheck only after that
        # admission boundary is closed; a child admitted while the dialog was
        # open must refresh consent rather than silently widening it.
        try:
            current_impact = self.lifecycle_impact(session_id=session_id)
            if current_impact.revision != expected_revision:
                raise ConsoleLifecycleRevisionChanged(
                    "Console session activity changed during close."
                )
            if self._cancel_raw_cli_session is not None:
                try:
                    self._cancel_raw_cli_session(session_id)
                except Exception:  # noqa: BLE001 -- teardown remains best-effort
                    logger.warning("close_session could not cancel raw CLI commands")
            owns_active_stream = self._active_stream_belongs_to_session(session_id)
            active_assistant_message_id = self._active_assistant_message_ids.get(
                session_id
            )
            if owns_active_stream and active_assistant_message_id is not None:
                # Closing is an explicit cancellation boundary. Settle the durable
                # dispatch before removing its in-memory owner so the cancelled
                # task cannot leave a restart-visible ``dispatch_started`` row.
                self._signal_stop(session_id=session_id)
                try:
                    self._mark_stream_stopped(
                        active_assistant_message_id,
                        visible_copy="Session closed.",
                    )
                except ConsoleDispatchSettlementError:
                    self._restore_dispatch_recovery_after_settlement_failure(
                        session_id,
                        active_assistant_message_id,
                    )
                    raise
            # Progress ownership must release before child cancellation, but a
            # failed callback must not commit wake, scratch or queue teardown.
            close_progress = getattr(self._agent_bridge, "close_progress", None)
            if callable(close_progress):
                close_progress(session_id, conversation_id=fleet_conversation_id)
        except BaseException as exc:
            abort_provisional_fleet_fence()
            if (
                isinstance(exc, Exception)
                and session_id in self._failed_session_close_generations
            ):
                frames = traceback.extract_tb(exc.__traceback__)
                logger.warning(
                    "close_session provisional failure (error_type={}, origin={})",
                    type(exc).__name__,
                    frames[-1].name if frames else "unknown",
                )
                raise RuntimeError(CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL) from None
            raise
        # Commit under the question registry's shared host lock. An earlier
        # registration is swept below; a later one observes this fence.
        with self._approval_state_lock:
            self._session_close_generations[session_id] = generation
        # Admission fences are the first irreversible close action after the
        # durable stream gate has settled successfully. They must beat every
        # cancellation snapshot and precede queue/file teardown, so a stale
        # parent cannot reserve a child in the cancellation-to-drain window.
        fence_wake = getattr(self._fleet_wake, "fence_conversation", None)
        if callable(fence_wake):
            fence_wake(fleet_conversation_id, generation=generation)
        self._discard_approval_rows_for_closing_session(session_id)
        # Revoke file authority before any close action can wake a worker or
        # remove the owning session from the store.
        self._scratch_spaces.close(session_id)
        forget_file_authority = getattr(
            self._agent_bridge,
            "forget_session_file_authority",
            None,
        )
        if callable(forget_file_authority):
            try:
                forget_file_authority(session_id)
            except Exception:  # noqa: BLE001 -- teardown remains best-effort
                logger.warning("close_session could not forget run-log authority")
        # Queue tombstone MUST precede stop/cancel: cancellation can wake a
        # terminal callback, which must observe that no next claim is legal.
        self.prompt_queue_coordinator.mark_closing(session_id)
        preparation = self.store.preparation_for_session(session_id)
        if preparation is not None and preparation.state in {
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.PAUSED,
        }:
            self._abandon_preparation(preparation.preparation_id)
        # PR3b Task 5 (Qodo #1808 finding 3): closing a session is
        # DESTRUCTIVE -- `ConsoleChatStore.close_session` purges every
        # message and drops the session -- so its fleet must die with it.
        # Navigation-away teardowns (`leave_console`/`shutdown`) preserve
        # the conversation and its survivors rightly continue; here a
        # surviving child would outlive its own conversation with no
        # panel row left to cancel it from, a wake targeting a dead
        # conversation, and a leaked unseen-mark. The conversation id is
        # derived NOW, while the session still exists (persisted id when
        # set -- the key the bridge's fleet state actually lives under),
        # and every live child goes through the explicit whole-fleet
        # path: `cancel_all_subagents` reuses the per-handle cancel, so
        # approval-card revocation and cancelled-is-never-retained ride
        # along. getattr-guarded and wrapped: a bare bridge double, no
        # bridge, or a raising cancel must never break a close.
        fleet_conversation_id = self._agent_conversation_id(session_id)
        cancel_all = (
            getattr(self._agent_bridge, "cancel_all_subagents", None)
            if self._agent_bridge is not None
            else None
        )
        if callable(cancel_all):
            try:
                cancelled_children = int(cancel_all(fleet_conversation_id))
                if cancelled_children:
                    logger.info(
                        "close_session cancelled {} sub-agent(s) of the closed conversation",
                        cancelled_children,
                    )
            except Exception:  # noqa: BLE001 -- teardown never fails on a fleet read
                logger.warning(
                    "close_session could not cancel the conversation's sub-agents"
                )
        repair_session = self._active_citation_repair_sessions.get(session_id)
        self.clear_original_attempts_for_session(session_id)
        submit_tasks = self._submit_tasks_for_session(session_id)
        if repair_session is not None and owns_active_stream:
            repair_session.cancel_reason = "session_close"
        if owns_active_stream:
            self._signal_stop(session_id=session_id)
            task = self._active_stream_tasks.get(session_id)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, SESSION_CLOSED_COPY),
                # `session_id` here is the session being CLOSED, which the
                # `_active_stream_belongs_to_session` guard above confirms
                # owns the active stream -- not necessarily the currently
                # ACTIVE session (you can close a background tab while
                # viewing another one), so this must be explicit rather
                # than falling back to the active-session default.
                session_id=session_id,
            )
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        for submit_task in submit_tasks:
            if submit_task is current_task:
                continue
            self._signal_stop(session_id=session_id)
            self._cancel_task_on_owner_loop(submit_task)
        if preparation is not None:
            self._preparation_outcomes.pop(preparation.preparation_id, None)
            self._prepared_send_continuations.pop(preparation.preparation_id, None)
        previous_active_id = self.store.active_session_id
        if previous_active_id == session_id:
            self.set_answerable_decision(session_id, None)
        self._cancel_pending_decisions_for_session(session_id)
        with self.store.durable_preparation_lock:
            durable_continuations = tuple(
                continuation
                for continuation in self._durable_postcommit_continuations.values()
                if continuation.session_id == session_id
            )
            for continuation in durable_continuations:
                self._durable_postcommit_continuations.pop(
                    continuation.preparation_id, None
                )
                self._release_retired_prepared_evidence(continuation)
                self.store.retire_durable_acceptance(
                    continuation.preparation_id, continuation.fingerprint
                )
        ticket = ConsoleSessionCloseTicket(
            close_id=str(uuid4()),
            session_id=session_id,
            conversation_id=fleet_conversation_id,
            expected_revision=expected_revision,
            generation=generation,
        )
        self._session_close_states[ticket.close_id] = (
            ticket,
            owns_active_stream,
            repair_session,
            previous_active_id,
        )
        return ticket
```

## tldw_chatbook/Chat/console_chat_controller.py:15207 — finalize_session_close

```python
    def finalize_session_close(
        self,
        ticket: ConsoleSessionCloseTicket,
    ) -> ConsoleChatSession | None:
        """Delete a session only after its runtime-owned work was drained."""

        state = self._session_close_states.pop(ticket.close_id, None)
        if state is None or state[0] != ticket:
            raise RuntimeError("Console session close ticket is stale.")
        if self._session_close_generations.get(ticket.session_id) != ticket.generation:
            raise RuntimeError("Console session close generation changed.")
        _stored_ticket, owns_active_stream, repair_session, previous_active_id = state
        session_id = ticket.session_id
        # Session-scoped grants die only after the close ticket is validated.
        self._chat_create_session_grants.pop(session_id, None)
        closed = self.store.close_session(session_id)
        self.prompt_queue_coordinator.remove_session(session_id)
        self._clear_project_instruction_delivery(session_id)
        new_active_id = self.store.active_session_id
        if (
            owns_active_stream
            and repair_session is not None
            and self._active_citation_repair_sessions.get(session_id) is repair_session
        ):
            self._active_citation_repair_sessions.pop(session_id, None)
        # Parallel-agents spec §6: closing the ACTIVE session auto-activates
        # a neighbor (`ConsoleChatStore.close_session`, console_chat_store.py
        # ~594-604) -- that neighbor is now the VIEWED session exactly as if
        # `switch_session` had navigated to it, so its unvisited outcome
        # must clear the same way, AND (Task 9) its parked approval card
        # (if any) must mount the same way too -- closing a background tab
        # must never leave the newly-viewed session's own pending approval
        # invisible just because it arrived here via auto-activation rather
        # than an explicit switch. Closing a BACKGROUND (non-active) session
        # leaves `active_session_id` unchanged, so this is a no-op in that
        # case.
        if new_active_id is not None and new_active_id != previous_active_id:
            self.mark_session_visited(new_active_id)
            if self.set_pending_decision is None:
                self._reproject_pending_decision_for_session(new_active_id)
            self._remount_session_kinds(new_active_id)
        self._remount_task_panel(new_active_id)
        return closed
```

## tldw_chatbook/Chat/console_chat_controller.py:20603 — _chat_create_source_is_open

```python
    def _chat_create_source_is_open(self, session_id: str) -> bool:
        """Check the existing lifetime fence and live source ownership.

        Args:
            session_id: Exact source session of the confirmed chat creation.

        Returns:
            True while the source exists without committed Close or disposal.
        """
        return (
            not self._disposed
            and session_id not in self._session_close_generations
            and any(session.id == session_id for session in self.store.sessions())
        )
```

## tldw_chatbook/Chat/console_chat_controller.py:19415 — prepare_agent_chat_create

```python
    def prepare_agent_chat_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Freeze source authority, destination and defaults before approval."""
        from .console_agent_bridge import validate_new_chat_arguments
        from .console_session_settings import blank_console_session_settings

        public = validate_new_chat_arguments(payload)
        actor = current_run_actor()
        child = actor is not None and actor.kind == "subagent"
        # Children retain draft creation in the parent's workspace only;
        # destination selection and bounded starts remain primary authority.
        if child and (
            public["destination"] != "same_workspace" or public["mode"] != "draft"
        ):
            raise PermissionError("primary_creation_authority_required")
        payload = {
            **payload,
            **public,
            "source_agent_kind": "subagent" if child else "primary",
            "source_parent_run_id": actor.parent_run_id if child else None,
        }
        if not self._chat_creation_source_live(
            payload
        ) or current_run_id() != payload.get("source_run_id"):
            raise PermissionError("source_unavailable")
        source = self.store._sessions.get(payload["session_id"])
        workspace = (
            CONSOLE_GLOBAL_WORKSPACE_ID
            if public["destination"] == "casual"
            else source.workspace_id
        )
        scope = (
            "global"
            if workspace in (None, CONSOLE_GLOBAL_WORKSPACE_ID)
            else "workspace"
        )
        workspace = None if scope == "global" else workspace
        persistence = self.store.persistence
        if persistence is None:
            raise PermissionError("persistence_unavailable")
        if not self._chat_creation_destination_available(
            scope_type=scope, workspace_id=workspace
        ):
            raise PermissionError("destination_unavailable")
        runtime = getattr(self.app, "console_runtime", None)
        if runtime is None:
            raise PermissionError("runtime_unavailable")
        settings = (
            self._default_session_settings()
            if self._default_session_settings
            else blank_console_session_settings(getattr(self.app, "app_config", {}))
        )
        if public["instructions"].strip():
            settings = replace(settings, system_prompt=public["instructions"])
        startup = runtime._resolve_new_console_assistant(
            workspace or CONSOLE_GLOBAL_WORKSPACE_ID, settings
        )
        startup = replace(
            startup, settings=self._resolve_new_chat_routing(startup.settings, public)
        )
        grant = (source.incarnation_id, "new_chat", scope, workspace, public["mode"])
        prepared = {
            **public,
            "tool": "new_chat",
            "session_id": source.id,
            "source_run_id": payload["source_run_id"],
            "source_agent_kind": payload["source_agent_kind"],
            "source_parent_run_id": payload["source_parent_run_id"],
            "source_message_id": payload["source_message_id"],
            "source_incarnation": source.incarnation_id,
            "source_workspace_id": source.workspace_id,
            "scope_type": scope,
            "workspace_id": workspace,
            "_grant_scope": grant,
            "assistant": startup.assistant_id,
            "assistant_default_notice": startup.notice,
            "provider": startup.settings.provider,
            "model": startup.settings.model or "",
            "resolved_instructions": startup.settings.system_prompt,
            "instructions": public["instructions"],
        }
        with self._pending_chat_create_lock:
            if payload["source_run_id"] in self._chat_creation_revoked_runs:
                raise PermissionError("source_unavailable")
            if len(self._chat_creation_records) >= 64:
                raise PermissionError("creation_capacity")
            token = _ChatCreationToken(self)
            prepared["_creation_token"] = token
            self._chat_creation_records[token] = {
                "payload": dict(prepared),
                "startup": startup,
                "source_cancel_event": (
                    self._active_cancel_events.get(source.id)
                    if child
                    and self._active_assistant_message_ids.get(source.id)
                    == payload["source_message_id"]
                    else None
                ),
                "approved": not child
                and grant in self._chat_create_session_grants.get(source.id, set()),
            }
        return prepared
```

## tldw_chatbook/Chat/console_chat_controller.py:19268 — _chat_creation_source_live

```python
    def _chat_creation_source_live(self, payload: Mapping[str, Any]) -> bool:
        """Require the captured primary or child execution and source lifetime."""
        session = self.store._sessions.get(str(payload.get("session_id") or ""))
        run_id = payload.get("source_run_id")
        bridge = self._agent_bridge
        if (
            self._disposed
            or session is None
            or not self._chat_create_source_is_open(session.id)
            or not run_id
            or bridge is None
            or run_id in self._chat_creation_revoked_runs
        ):
            return False
        if (
            payload.get("source_incarnation", session.incarnation_id)
            != session.incarnation_id
        ):
            return False
        try:
            row = bridge.runs_db.get_run(run_id)
            if not row or row["conversation_id"] != session.persisted_conversation_id:
                return False
            if payload.get("source_agent_kind") == "subagent":
                # TASK32531 children may survive their parent's turn. Bind to
                # the trusted child actor, never the session's next primary.
                actor = current_run_actor()
                parent_id = payload.get("source_parent_run_id")
                parent = bridge.runs_db.get_run(parent_id) if parent_id else None
                cancel = self._active_cancel_events.get(session.id)
                owns_parent_turn = self._active_assistant_message_ids.get(
                    session.id
                ) == payload.get("source_message_id")
                return bool(
                    (
                        not owns_parent_turn
                        or (cancel is not None and not cancel.is_set())
                    )
                    and actor is not None
                    and actor.kind == row["agent_kind"] == "subagent"
                    and actor.run_id == run_id
                    and actor.parent_run_id == row.get("parent_run_id") == parent_id
                    and row["status"] == "running"
                    and parent
                    and parent["conversation_id"] == session.persisted_conversation_id
                    and payload.get("destination") == "same_workspace"
                    and payload.get("mode") == "draft"
                )
            cancel = self._active_cancel_events.get(session.id)
            return bool(
                cancel is not None
                and not cancel.is_set()
                and self._active_assistant_message_ids.get(session.id)
                == payload.get("source_message_id")
                and bridge.live_primary_run_id(session.persisted_conversation_id)
                == run_id
                and row["agent_kind"] == "primary"
                and row["status"] not in {"done", "error", "cancelled", "abandoned"}
            )
        except Exception:
            return False
```

## tldw_chatbook/Chat/console_chat_store.py:5215 — sessions

```python
    def sessions(self) -> list[ConsoleChatSession]:
        """Return native Console sessions in creation order."""
        return list(self._sessions.values())
```

## tldw_chatbook/Chat/console_chat_start.py:128 — _source_live

```python
    def _source_live(self, request: AgentChatStartRequest) -> bool:
        controller = self._controller
        return controller._chat_creation_source_live(
            {
                "session_id": request.source_session_id,
                "source_incarnation": request.source_session_incarnation,
                "source_run_id": request.source_run_id,
                "source_message_id": controller._active_assistant_message_ids.get(
                    request.source_session_id
                ),
            }
        )
```
