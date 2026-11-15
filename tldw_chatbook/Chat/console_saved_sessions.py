"""Saved-session resolution and restoration under the store's existing ownership."""

from __future__ import annotations

from . import console_chat_store as host


def _saved_operation(self, conversation_id: str) -> host._SavedSessionOperation | None:
    with self._preparation_lock:
        operation = self._saved_session_operations.get(conversation_id)
        if (
            operation is not None
            and operation.session is not None
            and (
                self._sessions.get(operation.session.id) is not operation.session
                or (
                    operation.ready
                    and operation.session.persisted_conversation_id != conversation_id
                )
            )
        ):
            self._saved_session_operations.pop(conversation_id, None)
            operation = None
        if operation is None:
            # First Send/manual Save already owns a live slot before SQL. Join
            # that staged identity even after commit but before native publication.
            staged_session = next(
                (
                    self._sessions.get(sid)
                    for sid, reservation in self._first_identity_reservations.items()
                    if reservation[1].conversation_id == conversation_id
                ),
                None,
            )
            if staged_session is not None:
                operation = host._SavedSessionOperation(
                    conversation_id,
                    session=staged_session,
                    native_producer=True,
                    published=True,
                )
                self._saved_session_operations[conversation_id] = operation
        if operation is None:
            session = next(
                (
                    s
                    for s in self.sessions()
                    if s.persisted_conversation_id == conversation_id
                ),
                None,
            )
            if session is not None:
                # Existing first-persisted sessions keep their original live ID.
                operation = host._SavedSessionOperation(
                    conversation_id, session=session, ready=True, published=True
                )
                self._saved_session_operations[conversation_id] = operation
        return operation


def reserve_saved_creation(self, conversation_id: str) -> host._SavedSessionOperation:
    """Reserve a native preallocated ID before its first durable commit."""
    if self._saved_session_admission_closed:
        raise RuntimeError("Saved-session admission is closed.")
    if self._saved_operation(conversation_id) is not None:
        raise ValueError("Saved conversation already has a native owner.")
    operation = host._SavedSessionOperation(conversation_id, native_producer=True)
    self._saved_session_operations[conversation_id] = operation
    return operation


def abort_saved_creation(
    self, operation: host._SavedSessionOperation, error: BaseException
) -> None:
    """Fail only this unpublished native producer and wake its observers."""
    with self._preparation_lock:
        operation.error = error
    try:
        if self._saved_session_operations.get(operation.conversation_id) is operation:
            if operation.session is not None and not operation.published:
                self.rollback_restored_session(
                    operation.session.id,
                    expected_session=operation.session,
                    prior_active_session_id=self.active_session_id,
                    _operation=operation,
                )
            self._saved_session_operations.pop(operation.conversation_id, None)
    finally:
        # The native hydration failure may already have purged its exact
        # row/entry. Its detached observers still own this outcome future.
        future = operation.future
        if future is not None:

            def settle():
                if not future.done():
                    if isinstance(error, host.asyncio.CancelledError):
                        future.cancel()
                    else:
                        future.set_exception(error)

            future.get_loop().call_soon_threadsafe(settle)


def publish_restored_session(
    self, session: host.ConsoleChatSession, cleanup_owner: object | None = None
) -> None:
    operation = self._saved_operation(session.persisted_conversation_id)
    if (
        operation is None
        or operation.session is not session
        or not operation.ready
        or self._sessions.get(session.id) is not session
    ):
        raise RuntimeError("Saved conversation is not ready.")
    operation.published = True
    operation.cleanup_owner = None
    operation.abandoned = False


def restored_cleanup_session(
    self, cleanup_owner: object
) -> host.ConsoleChatSession | None:
    return next(
        (
            op.session
            for op in self._saved_session_operations.values()
            if op.cleanup_owner is cleanup_owner and not op.published
        ),
        None,
    )


async def resolve_saved_session(
    self,
    conversation_id: str,
    factory: host.Callable[[host._SavedSessionOperation], host.Any] | None = None,
    *,
    cleanup_owner: object | None = None,
    session_id: str | None = None,
    producer: host._SavedSessionOperation | None = None,
) -> host.ConsoleChatSession:
    """Await one authoritative ready owner; no observer owns its task."""
    if self._saved_session_admission_closed:
        raise RuntimeError("Saved-session admission is closed.")
    operation = self._saved_operation(conversation_id)
    with self._preparation_lock:
        if producer is not None and operation is not producer:
            raise RuntimeError("Saved producer ownership changed.")
        if operation is None:
            if factory is None:
                raise ValueError("Saved conversation has no native owner.")
            operation = host._SavedSessionOperation(
                conversation_id,
                future=host.asyncio.get_running_loop().create_future(),
                cleanup_owner=cleanup_owner,
            )
            operation.future.add_done_callback(host._consume_saved_outcome)
            self._saved_session_operations[conversation_id] = operation
        if operation.error is not None:
            raise operation.error
        if not operation.ready and operation.future is None:
            operation.future = host.asyncio.get_running_loop().create_future()
            operation.future.add_done_callback(host._consume_saved_outcome)
    if (
        not operation.ready
        and operation.task is None
        and (not operation.native_producer or producer is operation)
    ):
        if factory is None:
            raise RuntimeError("Saved conversation is not ready.")

        async def build():
            try:
                session = await factory(operation)
                if (
                    self._saved_session_admission_closed
                    or self._sessions.get(session.id) is not session
                    or session.persisted_conversation_id != conversation_id
                ):
                    raise RuntimeError("Saved conversation owner changed.")
                operation.session = session
                operation.ready = True
                if operation.cleanup_owner is None:
                    operation.published = True
                operation.future.set_result(session)
            except BaseException as error:  # noqa: BLE001 - settle detached native ownership even on cancellation.
                self.abort_saved_creation(operation, error)
            finally:
                operation.task = None
                if (
                    operation.abandoned
                    and operation.waiters == 0
                    and operation.session is not None
                    and not operation.published
                ):
                    self.rollback_restored_session(
                        operation.session.id,
                        expected_session=operation.session,
                        prior_active_session_id=self.active_session_id,
                        _operation=operation,
                    )

        operation.task = host.asyncio.create_task(
            build(), name="hydrate-canonical-console"
        )
        operation.task.add_done_callback(host._consume_saved_outcome)
    operation.waiters += 1
    try:
        session = (
            operation.session
            if operation.ready
            else await host.asyncio.shield(operation.future)
        )
        if (
            not operation.ready
            or self._sessions.get(session.id) is not session
            or session.persisted_conversation_id != conversation_id
        ):
            raise RuntimeError("Saved conversation owner changed.")
        if session_id is not None and session.id != session_id:
            raise ValueError("Conflicting native saved-session identity.")
        if cleanup_owner is None:
            self.publish_restored_session(session)
        return session
    except host.asyncio.CancelledError:
        if cleanup_owner is not None and operation.cleanup_owner is cleanup_owner:
            operation.abandoned = True
        raise
    finally:
        operation.waiters -= 1
        if (
            operation.abandoned
            and operation.waiters == 0
            and operation.task is None
            and operation.session is not None
            and not operation.published
        ):
            self.rollback_restored_session(
                operation.session.id,
                expected_session=operation.session,
                prior_active_session_id=self.active_session_id,
                _operation=operation,
            )


def _retire_first_saved_owner_locked(
    self,
    session: host.ConsoleChatSession | None,
    identity: host.ConsoleStagedConversationIdentity,
    *,
    session_id: str,
    error: BaseException | None = None,
) -> host._SavedSessionOperation | None:
    """Retire one exact staged identity with no I/O while preparation is locked."""
    reservation = self._first_identity_reservations.get(session_id)
    if reservation is None or reservation[1] is not identity:
        return None
    operation = self._saved_session_operations.get(identity.conversation_id)
    if (
        operation is not None
        and operation.native_producer
        and operation.published
        and operation.session is session
    ):
        if (
            error is None
            and session is not None
            and self._sessions.get(session.id) is session
            and session.persisted_conversation_id == identity.conversation_id
        ):
            operation.ready = True
        else:
            operation.error = (
                error
                if error is not None
                else RuntimeError("First saved owner did not publish.")
            )
            if (
                self._saved_session_operations.get(identity.conversation_id)
                is operation
            ):
                self._saved_session_operations.pop(identity.conversation_id, None)
    else:
        operation = None
    self._first_identity_reservations.pop(session_id, None)
    return operation


def _notify_first_saved_owner(operation: host._SavedSessionOperation | None) -> None:
    """Deliver terminal state without holding the native preparation lock."""
    if operation is None or operation.future is None:
        return
    future, session, error = operation.future, operation.session, operation.error

    def settle():
        if not future.done():
            if isinstance(error, host.asyncio.CancelledError):
                future.cancel()
            elif error is not None:
                future.set_exception(error)
            else:
                future.set_result(session)

    future.get_loop().call_soon_threadsafe(settle)


def _finish_first_saved_owner(
    self,
    session: host.ConsoleChatSession,
    identity: host.ConsoleStagedConversationIdentity,
    *,
    error: BaseException | None = None,
) -> None:
    with self._preparation_lock:
        operation = self._retire_first_saved_owner_locked(
            session, identity, session_id=session.id, error=error
        )
    self._notify_first_saved_owner(operation)


async def restore_persisted_session_owned(
    self, *, producer=None, **kwargs
) -> host.ConsoleChatSession:
    """Publish direct native creation through the same policy-ready operation."""

    async def restore(operation):
        session = self.restore_persisted_session(
            **dict(kwargs, _operation=operation, activate=False, prepare_progress=False)
        )
        from tldw_chatbook.Agents.fleet_messages import MessageError

        try:
            await self.prepare_progress_inbox_owned(session.id)
        except MessageError:
            pass  # Preserve the native optional saved-report refusal.
        await self.hydrate_session_library_policy(session.id)
        await self.reconcile_pending_workspace_projection(session.id)
        return session

    return await self.resolve_saved_session(
        kwargs["persisted_conversation_id"],
        restore,
        session_id=kwargs.get("session_id"),
        producer=producer,
    )


async def close_saved_session_operations(self) -> None:
    """Fence publication and drain physical hydration before native DB teardown."""
    self._saved_session_admission_closed = True
    tasks = []
    for operation in tuple(self._saved_session_operations.values()):
        if operation.task is not None:
            tasks.append(operation.task)
        elif not operation.ready:
            self.abort_saved_creation(
                operation, RuntimeError("Saved owner is closing.")
            )
    if tasks:
        drain = host.asyncio.gather(*tasks, return_exceptions=True)
        cancelled = False
        while not drain.done():
            try:
                await host.asyncio.shield(drain)
            except host.asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise host.asyncio.CancelledError


def rollback_restored_session(
    self,
    session_id: str,
    *,
    expected_session: host.ConsoleChatSession,
    prior_active_session_id: str | None,
    cleanup_owner: object | None = None,
    _operation: host._SavedSessionOperation | None = None,
) -> bool:
    """Remove only an unpublished runtime under exact creator cleanup custody.

    Args:
        session_id: Runtime session identifier to remove.
        expected_session: Exact restored session instance that still owns cleanup.
        prior_active_session_id: Session to reactivate when it still exists.

    Returns:
        True when the exact restored session was removed; False when ownership
        had changed and no cleanup was performed.
    """

    if self._sessions.get(session_id) is not expected_session:
        return False
    operation = self._saved_session_operations.get(
        expected_session.persisted_conversation_id
    )
    if (
        operation is None
        or operation.session is not expected_session
        or operation.published
        or not (
            _operation is operation
            or (cleanup_owner is not None and operation.cleanup_owner is cleanup_owner)
        )
    ):
        return False
    self._purge_session_runtime_state(session_id)
    if self.active_session_id == session_id:
        self._activate_session(
            prior_active_session_id
            if prior_active_session_id in self._sessions
            else None
        )
    return True


def restore_persisted_session(
    self,
    *,
    title: str,
    session_id: str | None = None,
    workspace_id: str | None,
    persisted_conversation_id: str,
    all_nodes: host.Iterable[host.ConsoleChatMessage],
    active_leaf_persisted_id: str | None = None,
    active_leaf_before_persisted_id: str | None = None,
    settings: host.ConsoleSessionSettings | None = None,
    generation_durable_snapshot: host.ConsoleGenerationSettingsSnapshot | None = None,
    generation_metadata_status: host.ConsoleGenerationSettingsReadStatus = (
        host.ConsoleGenerationSettingsReadStatus.ABSENT
    ),
    runtime_backend: str = "local",
    assistant_kind: str | None = "generic",
    assistant_id: str | None = "console",
    assistant_authority_id: str | None = None,
    persona_memory_mode: str | None = None,
    assistant_default_notice: str = "",
    character_id: int | None = None,
    character_name: str | None = None,
    user_display_name_override: str | None = None,
    character_system_template: str | None = None,
    ephemeral: bool = False,
    remote_active: bool = False,
    activate: bool = True,
    prepared_data: host.ConsoleConversationHydrationData | None = None,
    initial_project_instruction_state: host.ProjectInstructionControlState
    | None = None,
    prepare_progress: bool = True,
    cleanup_owner: object | None = None,
    _operation: host._SavedSessionOperation | None = None,
) -> host.ConsoleChatSession:
    """Create and activate a native session from persisted conversation data.

    Task 8: a restored conversation arrives as the WHOLE persisted tree --
    every branch, on- and off-path -- so off-path siblings are navigable
    (swipe) immediately after resume. ``all_nodes`` is the flattened node
    set (pre-order, every node), each carrying its own
    ``persisted_message_id`` and its persisted ``parent_message_id``; the
    full in-memory tree is rebuilt from those links and the active-path
    VIEW is derived from the two-component durable cursor. A valid
    before-message marker with no active leaf restores immediately before
    that root user prompt; otherwise the active leaf wins, with invalid or
    unset state falling back to the most-recent-child leaf and repairing
    the cursor.

    task-558: also hydrates every restored node's ``generation_metadata``
    from the ``message_generation_metadata`` sidecar table, via one
    batched ``persistence.get_generation_metadata_for_messages`` call
    feeding ``hydrate_generation_metadata`` (see
    ``_hydrate_generation_metadata_from_persistence``) -- callers do not
    need to drive that seam themselves. Silently skipped (nodes stay
    metadata-only) when ``persistence`` is ``None`` or doesn't implement
    the batch-fetch method.

    Args:
        title: Display title for the restored Console session.
        workspace_id: Workspace scope recorded on the persisted conversation,
            or ``None`` to use the current store workspace context.
        persisted_conversation_id: Durable Chat conversation identifier.
        all_nodes: Every native Console node reconstructed from the
            persisted conversation tree (all branches), each carrying its
            ``persisted_message_id`` and persisted ``parent_message_id``.
        active_leaf_persisted_id: Persisted id of the stored active-leaf
            pointer, or ``None``. A non-null value is authoritative over
            the companion before-message marker.
        active_leaf_before_persisted_id: Persisted id of the root user
            prompt immediately after an explicitly empty active path, or
            ``None`` for ordinary selected/unset cursor state.
        settings: Optional provider/model settings snapshot for the session.
        initial_project_instruction_state: Owner-supplied controls for a newly
            created conversation, applied at session construction. When absent,
            retain the durable controls or the legacy disabled default.
        prepared_data: Optional unpublished bulk reads for this conversation.
            Policy reconciliation and all store publication remain on the caller.

    Returns:
        The newly created and activated Console session.
    """
    if initial_project_instruction_state is not None and not isinstance(
        initial_project_instruction_state, host.ProjectInstructionControlState
    ):
        raise TypeError(
            "initial_project_instruction_state must be ProjectInstructionControlState"
        )
    if persisted_conversation_id in self._conversation_title_transitions:
        raise RuntimeError("A conversation title change is still publishing.")
    # A restored session comes FROM durable storage, so it is by
    # definition not temporary. Refuse rather than silently produce a
    # session that is both temporary and persisted -- the one state the
    # gate's invariant does not allow.
    if (
        prepared_data is not None
        and prepared_data.conversation_id != persisted_conversation_id
    ):
        raise ValueError("Conversation hydration identity changed")
    if ephemeral:
        raise ValueError(
            "Cannot restore a persisted session as temporary: a temporary "
            "session has no persisted conversation."
        )
    from .console_conversation_hydration import restored_project_instruction_state

    project_instruction_state = restored_project_instruction_state(
        self.persistence,
        persisted_conversation_id,
        initial_project_instruction_state,
    )
    operation = self._saved_operation(str(persisted_conversation_id))
    if operation is not None and operation is not _operation:
        if not operation.ready or operation.session is None:
            raise RuntimeError("Saved conversation is still hydrating.")
        if session_id is not None and operation.session.id != session_id:
            raise ValueError("Conflicting native saved-session identity.")
        self.publish_restored_session(operation.session)
        if activate:
            self.switch_session(operation.session.id)
        return operation.session
    if operation is None:
        operation = host._SavedSessionOperation(
            str(persisted_conversation_id), cleanup_owner=cleanup_owner
        )
        self._saved_session_operations[str(persisted_conversation_id)] = operation
    prior_active_session_id = self.active_session_id
    session = self.create_session(
        session_id=session_id or str(persisted_conversation_id),
        title=title,
        workspace_id=workspace_id,
        settings=settings,
        runtime_backend=runtime_backend,
        assistant_kind=assistant_kind,
        assistant_id=assistant_id,
        assistant_authority_id=assistant_authority_id,
        persona_memory_mode=persona_memory_mode,
        assistant_default_notice=assistant_default_notice,
        character_id=character_id,
        character_name=character_name,
        project_instruction_state=project_instruction_state,
        thinking_history_policy=self._durable_thinking_history_policy(
            str(persisted_conversation_id)
        ),
        activate=activate,
    )
    operation.session = session
    try:
        session.user_display_name_override = user_display_name_override
        session.character_system_template = character_system_template
        self.rebind_persisted_conversation(
            session.id,
            str(persisted_conversation_id),
        )
        handoff_draft = None
        database = getattr(self.persistence, "db", None) if self.persistence else None
        reader = getattr(database, "get_conversation_by_id", None)
        if callable(reader):
            try:
                row = reader(str(persisted_conversation_id))
            except Exception:
                row = None
            raw_metadata = (row or {}).get("metadata") or "{}"
            try:
                metadata_obj = (
                    host.json.loads(raw_metadata)
                    if isinstance(raw_metadata, str)
                    else {}
                )
            except ValueError:
                metadata_obj = {}
            handoff = (
                metadata_obj.get("console_agent_handoff")
                if isinstance(metadata_obj, dict)
                else None
            )
            if isinstance(handoff, dict):
                session.agent_handoff_launch = host.AgentHandoffLaunchMetadata.read(
                    handoff.get("launch")
                )
                if "version" not in handoff and handoff.get("draft"):
                    handoff_draft = str(handoff["draft"])
                    self._pending_agent_handoff_clears[
                        str(persisted_conversation_id)
                    ] = None
                elif (
                    handoff.get("version") == 2
                    and handoff.get("state") in {"pending", "consumed"}
                    and type(handoff.get("draft_revision")) is int
                    and handoff["draft_revision"] >= 1
                    and isinstance(handoff.get("draft"), str)
                ):
                    session.agent_handoff_revision = handoff["draft_revision"]
                    session.agent_handoff_state = handoff["state"]
                    if handoff["state"] == "pending":
                        handoff_draft = handoff["draft"]
                        self._agent_handoff_writes[session.id] = {
                            "conversation_id": str(persisted_conversation_id),
                            "durable_revision": handoff["draft_revision"],
                            "revision": handoff["draft_revision"],
                            "draft": handoff_draft,
                            "task": None,
                            "failed": False,
                        }
                elif "version" in handoff:
                    session.agent_handoff_state = "review_required"
        if handoff_draft is not None:
            with self._preparation_lock:
                host._replace_session_draft_locked(session, handoff_draft)
        if self.active_session_id == session.id:
            # An ACTIVATING restore (the post-restart open path) activates
            # inside `create_session` -- before the pending clear above was
            # registered -- so consume it directly here.
            self._consume_pending_agent_handoff_clear(session.id)
        with self._progress_identity_lock:
            sibling = next(
                (
                    other
                    for other in self._sessions.values()
                    if other is not session
                    and other.persisted_conversation_id
                    == session.persisted_conversation_id
                    and other._progress_owner_id is not None
                ),
                None,
            )
            if sibling is not None:
                session._progress_owner_id = sibling._progress_owner_id
        session.generation_durable_snapshot = generation_durable_snapshot
        session.generation_metadata_status = generation_metadata_status
        self.hydrate_session_capture_policy(session.id)
        recovery = self._hydrate_dispatch_recovery(
            session.id,
            str(persisted_conversation_id),
        )
        if recovery is not None and recovery.checkpoint is not None:
            # Reconciliation may have repaired an older stranded cursor.
            # Publish the committed cursor, not the pre-reconcile snapshot.
            database = getattr(self.persistence, "db", None)
            if database is not None:
                active_leaf_persisted_id, active_leaf_before_persisted_id = (
                    database.get_conversation_active_cursor(
                        str(persisted_conversation_id)
                    )
                )
        session.library_policy_hydrated = False
        coordinator = self.library_policy_coordinator
        if coordinator is not None:
            session.library_policy_holder.snapshot = host.normalize_policy_read(
                None
            ).snapshot
            session.library_policy_holder.explicitly_staged = False
            coordinator.register_holder(
                session.id,
                session.persisted_conversation_id,
                session.library_policy_holder,
            )
        if session.workspace_id != host.CONSOLE_GLOBAL_WORKSPACE_ID:
            self._pending_workspace_projections[session.id] = (
                session.persisted_conversation_id
            )
        mixin_reader = getattr(
            self.persistence, "get_conversation_mixin_preferences", None
        )
        if prepared_data is not None and prepared_data.mixin_preferences is not None:
            session.prompt_mixin_preferences = prepared_data.mixin_preferences
        elif callable(mixin_reader):
            session.prompt_mixin_preferences = mixin_reader(
                str(persisted_conversation_id)
            )
        prefs = session.prompt_mixin_preferences.preferences
        session.prompt_mixin_durable_revision = (
            prefs.revision if prefs is not None else 0
        )
        self._restore_speech_preferences(session)
        self._resolve_context_policy_on_resume(session.id)
        restored_nodes = self._hydrate_provider_continuations_from_persistence(
            session.id,
            persisted_conversation_id,
            list(all_nodes),
            remote_active=remote_active,
            prepared_rows=prepared_data.continuation_rows
            if prepared_data is not None
            else host._HYDRATION_NOT_PREPARED,
        )
        for node in restored_nodes:
            node.mixin_usage_count = None
            if prepared_data is not None:
                node.mixin_usage_count = prepared_data.mixin_usage_counts.get(
                    node.persisted_message_id
                )
            elif node.role is host.ConsoleMessageRole.ASSISTANT:
                reader = getattr(
                    self.persistence, "get_conversation_mixin_attempt", None
                )
                if callable(reader) and node.persisted_message_id is not None:
                    try:
                        replay = reader(
                            str(persisted_conversation_id),
                            assistant_message_id=node.persisted_message_id,
                        )
                        node.mixin_usage_count = (
                            len(replay.snapshot.items)
                            if replay.status == "recorded"
                            and replay.snapshot is not None
                            else 0
                            if replay.status == "legacy"
                            else None
                        )
                    except Exception:  # noqa: BLE001 - unreadable records stay unqualified.
                        node.mixin_usage_count = None
        self._ingest_full_tree(
            session.id,
            restored_nodes,
            active_leaf_persisted_id=active_leaf_persisted_id,
            active_leaf_before_persisted_id=(active_leaf_before_persisted_id),
        )
        self._normalize_restored_provider_continuation(
            session.id, str(persisted_conversation_id)
        )
        self._reconcile_restored_chat_sync_intents(
            session.id, str(persisted_conversation_id)
        )
        if prepared_data is None:
            self._hydrate_generation_metadata_from_persistence(session.id)
        elif isinstance(prepared_data.generation_rows, dict):
            self.hydrate_generation_metadata(session.id, prepared_data.generation_rows)
        self._seed_console_settings_owned_bases(session)
        self._bump_payload_revision(session.id)
        from tldw_chatbook.Agents.fleet_messages import MessageError

        if prepare_progress:
            try:
                self.prepare_progress_inbox(session.id)
            except MessageError:
                pass  # Saved capacity/load refusal must not roll back the chat.
        if _operation is None:
            operation.ready = True
            operation.published = cleanup_owner is None
        return session
    except BaseException:
        self.rollback_restored_session(
            session.id,
            expected_session=session,
            prior_active_session_id=prior_active_session_id,
            _operation=operation,
        )
        raise
