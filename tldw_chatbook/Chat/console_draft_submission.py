"""Console draft submission, loaded only when a send starts.

The live controller module remains the owner of every free-name patch seam;
qualifying those reads preserves replacements made during an awaited send.
Controller state, permission admission and recovery ordering stay on the same owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable  # noqa: UP035

if TYPE_CHECKING:
    from . import console_chat_controller as owner


async def submit_draft_body(
    self,
    draft: str,
    *,
    session_id: str | None = None,
    origin: owner.ConsoleSubmissionOrigin,
    queue_entry_id: str | None = None,
    queue_authorization: owner.QueueGenerationAuthorization | None = None,
    wake_authorization: owner.AgentWakeAuthorization | None = None,
    preserve_composer: bool = False,
    configuration: owner.ConsoleTurnConfigurationSnapshot | None = None,
    accepted_attachments: tuple[owner.PendingAttachment, ...] | None = None,
    captured_one_shot_prefill: str | None = None,
    captured_one_shot_prefill_revision: int | None = None,
    staged_evidence_launch: Any | None = None,
    staged_evidence_capture: (Callable[[str, Any, Any], Awaitable[Any]] | None) = None,
    staged_evidence_release: Callable[[Any, Any], None] | None = None,
    custody_acceptance_hook: Callable[[], None] | None = None,
    _resume_preparation_id: str | None = None,
    _resume_resolution: Any | None = None,
) -> owner.ConsoleSubmitResult:
    """Submit a composer draft through native Console validation and provider resolution.

    PR3a-2 Task 5: ``origin=AGENT_WAKE`` (requires a coordinator-issued
    ``wake_authorization``, the queue-token precedent) submits a
    machine-injected auto-wake notice instead of a user draft. The
    wake branch: no USER transcript row is echoed (a SYSTEM-class row
    carrying ``MessageMetadata(origin="agent_wake")`` is appended at
    the acceptance point instead); the composer hook is never invoked
    (non-MANUAL); pending attachments and the one-shot prefill are
    left untouched (they are the USER's staged state); auto-titling,
    RAG capture and prompt history are skipped; and the notice
    reaches the model as a payload-only trailing user-role entry
    appended after every per-send transform (see
    ``console_fleet_wake``'s module docstring for the delivery-path
    decision record). Skill substitution and dictionary/world-info
    transforms still run -- they are HISTORY transforms every send
    re-applies, and the wake's own notice is appended after them, so
    it is never itself substituted.

    F4 fix (Qodo wave, parallel-agents spec §2): sends are dispatched
    per-session -- ``chat_screen._dispatch_console_draft_send`` captures
    the target session at DISPATCH time and threads it through
    ``run_worker``'s coroutine args (see ``_submit_console_native_
    draft``). Before this fix, this method always re-resolved "the
    session to submit into" via ``store.ensure_session()``/
    ``store.active_session_id`` at EXECUTION time instead -- a session
    switch during the scheduling gap between ``run_worker(...)`` and
    this coroutine's body actually running could silently submit the
    draft into whichever session the user switched TO, not the one
    that was showing when Send was pressed.

    Args:
        draft: The raw composer text to submit.
        session_id: The session this draft was dispatched for, captured
            by the caller at dispatch time. ``None`` (the default)
            preserves the pre-fix behavior -- resolve/create the
            CURRENTLY active session -- for direct-call test idioms and
            other callers that have no per-session dispatch to capture.
            An empty string is treated the same as ``None`` (the
            dispatch-time sentinel for "no session existed yet").

    Returns:
        The submission outcome: ``accepted`` False (with an explanatory
        ``visible_copy``) when blocked before any provider call, or
        when ``session_id`` names a session that no longer exists by
        the time this runs (see ``_session_closed_result``); ``True``
        once the turn actually proceeds.
    """
    from . import console_chat_controller as owner

    if not isinstance(origin, owner.ConsoleSubmissionOrigin):
        raise ValueError(  # noqa: TRY004 - preserve the admission contract.
            "origin must be an explicit ConsoleSubmissionOrigin"
        )
    target_id = session_id or self.store.active_session_id or ""
    resumed_preparation = (
        self._preparation_by_id(_resume_preparation_id)
        if _resume_preparation_id is not None
        else None
    )
    prepared_continuation = (
        self._prepared_send_continuations.get(_resume_preparation_id)
        if _resume_preparation_id is not None
        else None
    )
    if prepared_continuation is not None:
        preserve_composer = prepared_continuation.preserve_composer
    if preserve_composer and not session_id:
        return owner.ConsoleSubmitResult(
            False, False, "Choose an explicit conversation."
        )
    if preserve_composer and str(draft).lstrip().startswith(
        (owner.COMMAND_PREFIX, owner.MENTION_SIGIL)
    ):
        return owner.ConsoleSubmitResult(
            False,
            False,
            "Use Console for slash commands and @ references.",
            session_id=session_id,
        )
    if _resume_preparation_id is not None and resumed_preparation is None:
        return owner.ConsoleSubmitResult(
            False, False, "Prepared turn is no longer available."
        )
    if _resume_preparation_id is not None and prepared_continuation is None:
        return owner.ConsoleSubmitResult(
            False, False, "Prepared turn is no longer available."
        )
    if origin is owner.ConsoleSubmissionOrigin.QUEUED:
        if not queue_entry_id or (
            _resume_preparation_id is None
            and not self.prompt_queue_coordinator.authorizes(
                queue_authorization, target_id
            )
        ):
            raise PermissionError(
                "queued sends require coordinator-issued generation authority"
            )
    elif origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        # PR3a-2 Task 5: only the wake coordinator can mint the token
        # (queue-token precedent) -- no other code path can fabricate
        # a machine-origin send.
        if not self._fleet_wake.authorizes(wake_authorization, target_id):
            raise PermissionError(
                "agent-wake sends require coordinator-issued wake authority"
            )
        if target_id and self.prompt_queue_coordinator.controls_generation(target_id):
            # Defense-in-depth twin of the coordinator's own gate: a
            # queue-owned session's next turn belongs to the queue.
            # Refused WITHOUT a transcript row -- a machine deferral
            # is not user-visible news; the wake retries later.
            return owner.ConsoleSubmitResult(
                False,
                False,
                "Queued messages control the next turn.",
            )
    elif target_id and self.prompt_queue_coordinator.controls_generation(target_id):
        visible_copy = (
            "Queued messages control the next turn. Resume or manage the queue first."
        )
        if target_id and any(
            session.id == target_id for session in self.store.sessions()
        ):
            self.store.append_message(
                target_id,
                role=owner.ConsoleMessageRole.SYSTEM,
                content=visible_copy,
            )
        return owner.ConsoleSubmitResult(False, False, visible_copy)

    active_rejection = self._active_run_rejection(
        session_id=session_id,
        # A raced wake refusal is machine-internal (retried later);
        # only user-facing origins get the explanatory SYSTEM row.
        append_row=origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE,
        queue_authorization=queue_authorization,
    )
    if active_rejection is not None and resumed_preparation is None:
        return active_rejection

    if (
        target_id
        and resumed_preparation is None
        and self.store.dispatch_recovery_blocks_submission(target_id)
    ):
        return owner.ConsoleSubmitResult(
            False,
            False,
            "Finish or discard the pending response before sending another message.",
            session_id=target_id,
            origin=origin,
            queue_entry_id=queue_entry_id,
        )

    if session_id:
        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        if session is None:
            # The dispatching session was closed during the gap between
            # dispatch and this coroutine actually running -- there is
            # nothing left to submit into. Stamp the (now-orphaned)
            # session id, never whatever is active now (see
            # `_session_closed_result`'s own docstring). `dispatch_gap`
            # is what makes THIS call site (uniquely among ~19) toast --
            # every other one fires mid-run, after the user already
            # confirmed closing that session themselves.
            return self._session_closed_result(session_id=session_id, dispatch_gap=True)
    else:
        # Task 4 (D2 fix wave, "bonus race"): mirror the mount-time
        # creator (`ConsoleSessionController._ensure_active_console_session_settings`),
        # which always passes `settings=` -- without this, a session
        # bootstrapped from THIS branch (no dispatch-captured session id
        # at all) got `settings=None` while every other creator gave the
        # first session a real snapshot, and whichever creator ran first
        # decided the outcome.
        session = self._ensure_default_session()
    active_task = owner.asyncio.current_task()
    if active_task is not None:
        self._rebind_submit_task(active_task, session.id)
        if resumed_preparation is not None:
            self._bind_submit_preparation(
                active_task, resumed_preparation.preparation_id
            )
    if preserve_composer and (
        self.store.pending_attachments(session.id)
        or self.store.session_one_shot_prefill(session.id)
        or self._has_explicit_staged_evidence(session.id) is not False
    ):
        return owner.ConsoleSubmitResult(
            False,
            False,
            "This conversation has staged Console attachments, evidence or prefill. "
            "Review them in Console before sending from Buddy.",
            session_id=session.id,
        )
    # PR3a-2 Task 5: a wake never touches the user's staged state --
    # pending attachments belong to the USER's next send and must be
    # neither embedded nor cleared by a machine turn.
    custodied_inputs = configuration is not None
    admitted_prefill: str | None = None
    admitted_prefill_from_one_shot = False
    admitted_prefill_revision: int | None = None
    if origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE and custodied_inputs:
        if captured_one_shot_prefill:
            admitted_prefill = captured_one_shot_prefill
            admitted_prefill_from_one_shot = True
            admitted_prefill_revision = captured_one_shot_prefill_revision
        elif configuration.session_settings is not None:
            admitted_prefill = configuration.session_settings.pinned_prefill
    pendings = (
        list(prepared_continuation.attachments)
        if prepared_continuation is not None
        else list(accepted_attachments or ())
        if custodied_inputs
        else self.store.pending_attachments(session.id)
        if origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
        and not preserve_composer
        else []
    )
    attachment_mode_pendings = [
        pending
        for pending in pendings
        if pending.insert_mode == "attachment" and pending.data is not None
    ]
    has_pending_attachment = bool(attachment_mode_pendings)
    if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        # The notice is machine-composed from DB text and bounded by
        # `compose_wake_notice`'s own result budget; `_validated_draft`
        # exists to validate USER drafts (its length cap and markup
        # rules are composer policy, not payload policy).
        clean_draft = str(draft or "").strip()
        validation_error = None if clean_draft else "Empty wake notice."
    else:
        clean_draft, validation_error = self._validated_draft(
            draft, allow_empty=has_pending_attachment
        )
    if validation_error is not None:
        return self._block(session.id, validation_error)
    # TASK-27021: expand @-references (files/folders/diff) into the text
    # the PROVIDER sees, before the one preparation construction below.
    # The user echo keeps the RAW draft; a compact system row records what
    # expanded/refused (26020 AC#6). Expansion failures never block the
    # send -- the raw draft goes through with a note. AGENT_WAKE drafts
    # are machine-composed and never expanded.
    executed_draft_text = clean_draft
    reference_records: tuple = ()
    # Qodo #6 (PR #2313): a RESUMED preparation re-enters this seam with
    # the already-expanded executed_draft; expansion is not idempotent
    # (each raw @token survives ahead of its inserted block), so re-running
    # it would inject every referenced file a second time. Expand only on
    # the first pass.
    if (
        origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
        and resumed_preparation is None
    ):
        try:
            from tldw_chatbook.Chat.console_references import (
                build_console_reference_resolver,
                expand_references,
                find_reference_candidates,
                run_git_reference,
            )

            if find_reference_candidates(clean_draft):

                def _expand() -> object:
                    return expand_references(
                        clean_draft,
                        resolve=build_console_reference_resolver(),
                        git_runner=run_git_reference,
                    )

                expansion = await owner.asyncio.to_thread(_expand)
                executed_draft_text = expansion.expanded_text
                reference_records = tuple(expansion.records)
        except Exception:  # noqa: BLE001 - references must never block a send
            owner.logger.opt(exception=True).warning(
                "@-reference expansion failed; sending the raw draft"
            )
            executed_draft_text = clean_draft
            reference_records = ()
    configuration = (
        resumed_preparation.execution_context.configuration
        if resumed_preparation is not None
        else configuration
        if configuration is not None
        else self.resolve_turn_configuration_snapshot(session.id)
    )
    turn_selection = configuration.provider_selection
    if has_pending_attachment:
        vision_model = configuration.effective_model
        # ONE capability check decides the gate AND the copy: this
        # module's is_vision_capable (the documented monkeypatch seam) is
        # injected into vision_block_reason instead of being re-checked
        # around it — the two seams could otherwise disagree under test.
        block_reason = owner.vision_block_reason(
            turn_selection.provider,
            vision_model,
            is_capable=lambda _provider, _model: bool(
                configuration.capabilities.get("vision", False)
            ),
        )
        if block_reason is not None:
            return self._block(session.id, block_reason)
    if turn_selection.workspace_context.has_policy_blocks:
        return self._block(session.id, turn_selection.workspace_context.recovery_copy)
    library_authority = (
        resumed_preparation.execution_context.library_authority
        if resumed_preparation is not None
        else await self._capture_turn_library_authority(session.id, configuration)
    )
    existing_preparation = self.store.preparation_for_session(session.id)
    if (
        existing_preparation is not None
        and existing_preparation is not resumed_preparation
        and existing_preparation.state
        not in {
            owner.ConsoleTurnPreparationState.CANCELLED,
            owner.ConsoleTurnPreparationState.SETTLED,
        }
    ):
        return owner.ConsoleSubmitResult(
            False,
            False,
            "Another send is still preparing for this conversation.",
        )
    pre_send_title = (
        resumed_preparation.pre_send_title
        if resumed_preparation is not None
        else session.title
    )
    pre_send_conversation_id = (
        resumed_preparation.pre_send_conversation_id
        if resumed_preparation is not None
        else session.persisted_conversation_id
    )
    explicit_evidence_staged = (
        staged_evidence_launch is not None
        if custodied_inputs
        else self._has_explicit_staged_evidence(session.id)
    )

    # TASK-457(a): echo the USER message BEFORE resolving the provider, so a
    # slow/cold readiness probe no longer leaves the transcript blank while
    # the composer clears — the message reads as "sent", not lost. On a
    # not-ready provider the row persists next to the honest block-row below
    # (the message is no longer silently dropped) and the draft is kept (the
    # composer clears only on the accepted path via
    # `_notify_submission_accepted`), so the user can re-attempt. Staged
    # attachments are embedded on the row here but only CLEARED on the
    # success path below, so a blocked attempt leaves them staged for retry.
    #
    # Auto-title BEFORE the append: a persisting append creates the durable
    # conversation from `session.title` (persist_session_if_needed) and sets
    # `persisted_conversation_id`, after which `_maybe_auto_title_session`
    # early-returns. Titling first means the conversation is created as the
    # derived title (e.g. "hello") instead of the default "Chat 1", so the
    # workspace rail shows it immediately after persistence.
    durable_commit = getattr(self.store.persistence, "commit_durable_turn", None)
    durable_turn = bool(
        not session.ephemeral
        and origin
        in {owner.ConsoleSubmissionOrigin.MANUAL, owner.ConsoleSubmissionOrigin.QUEUED}
    )
    if durable_turn and not callable(durable_commit):
        # TASK-22030: a refusal the user cannot see is indistinguishable
        # from a broken app. `_block_undurable_turn` writes the run state,
        # the transcript row, and the toast that `56db75386` dropped.
        return self._block_undurable_turn(
            session.id,
            origin=origin,
            queue_entry_id=queue_entry_id,
        )
    staged_title = session.title
    if (
        origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
        and resumed_preparation is None
    ):
        derived_title = (
            owner.derive_console_session_title(clean_draft)
            if session.persisted_conversation_id is None
            and owner.is_default_console_session_title(session.title)
            else ""
        )
        if durable_turn:
            staged_title = derived_title or session.title
        else:
            self._maybe_auto_title_session(session, clean_draft)
    staged_attachments = tuple(
        owner.MessageAttachment(
            data=pending.data,
            mime_type=pending.mime_type or "image/png",
            display_name=pending.display_name,
            position=index,
        )
        for index, pending in enumerate(attachment_mode_pendings)
    )
    # TASK-485: the optimistic echo is appended WITHOUT persistence. A send
    # that is blocked/fails before it reaches the provider must leave no
    # durable record — otherwise the resume path (which reconstructs every
    # row as "complete") would silently drop the row's failed state and let a
    # never-sent message re-enter the next send's context, and the orphan
    # would render as a lonely user prompt. The row is flushed to storage
    # only once the turn is confirmed to proceed (below).
    #
    # PR3a-2 Task 5: a wake echoes NOTHING here -- invariant 5 forbids
    # a USER row for machine input, and the SYSTEM notice row is
    # appended only at the acceptance point below (TASK-457(a)'s
    # "reads as sent, not lost" concern protects a HUMAN's typed
    # message during a slow readiness probe; a machine notice has no
    # one watching for it, and appending late means a blocked wake
    # leaves no orphaned notice row to clean up).
    echoed_user = (
        self.store.get_message(resumed_preparation.transient_user_message_id)
        if resumed_preparation is not None
        and resumed_preparation.transient_user_message_id is not None
        else self.store.append_message(
            session.id,
            role=owner.ConsoleMessageRole.USER,
            content=clean_draft,
            attachments=staged_attachments,
            persist=False,
        )
        if origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
        else None
    )
    if reference_records:
        # TASK-27021 / 26020 AC#6 (placement per Qodo #7, PR #2313): the
        # audit row is written adjacent to the raw user echo and
        # UNCONDITIONALLY -- a leading-@ draft with trace capture off
        # skips ordinary preparation entirely, but its expansion still
        # reaches the payload and must still be visible.
        summary_lines = []
        for record in reference_records:
            mark = "included" if record.ok else "REFUSED"
            summary_lines.append(f"{record.raw}: {mark} — {record.detail}")
        self.store.append_message(
            session.id,
            role=owner.ConsoleMessageRole.SYSTEM,
            content="@-references:\n" + "\n".join(summary_lines),
        )

    self._set_run_state(
        owner.ConsoleRunState(
            owner.ConsoleRunStatus.VALIDATING, "Validating provider."
        ),
        session_id=session.id,
    )
    try:
        resolution = (
            _resume_resolution
            if resumed_preparation is not None
            else await self._resolve_for_send_bounded(turn_selection)
        )
    except BaseException as exc:
        # A readiness probe that raises or is cancelled AFTER the optimistic
        # USER echo must still fail that row — otherwise a never-sent USER
        # message leaks into the NEXT send's provider context (`skip_failed`
        # only drops "failed" rows). Fail it, then re-raise so the caller
        # still sees the probe failure. (A wake echoed nothing: None guard.)
        if echoed_user is not None:
            if self._shutdown_requested.is_set():
                self.store.rollback_transient_send(
                    session.id,
                    echoed_user.id,
                    title=pre_send_title,
                    persisted_conversation_id=pre_send_conversation_id,
                )
            else:
                self._mark_transient_echo_blocked(echoed_user.id)
        # Validation owns a busy slot even before a provider starts. Release
        # it on failure so retry and the view's idle polling cleanup can run.
        # A closed session or an already-stopped run keeps its owner's state.
        # Wakes own a separate retry loop; releasing their slot here would
        # immediately retry the same failed wake ahead of other pending wakes.
        if (
            origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
            and self.run_state_for(session.id).status
            is owner.ConsoleRunStatus.VALIDATING
            and any(item.id == session.id for item in self.store.sessions())
        ):
            cancelled = isinstance(exc, owner.asyncio.CancelledError)
            self._set_run_state(
                owner.ConsoleRunState(
                    owner.ConsoleRunStatus.STOPPED
                    if cancelled
                    else owner.ConsoleRunStatus.BLOCKED,
                    "Provider validation was cancelled."
                    if cancelled
                    else "Provider validation failed. Try sending again.",
                ),
                session_id=session.id,
            )
        raise
    if not getattr(resolution, "ready", False):
        visible_copy = self._blocked_visible_copy(
            getattr(resolution, "visible_copy", "")
        )
        # The echoed row stays visible but never reached a provider — fail it
        # so it is excluded from the NEXT send's provider context
        # (`skip_failed`) and reads honestly as unsent rather than polluting
        # the history. (A wake echoed nothing: None guard.)
        if echoed_user is not None:
            self._mark_transient_echo_blocked(echoed_user.id)
        return self._block(session.id, visible_copy)

    thinking_block = self._thinking_persistence_preflight(
        session_id=session.id,
        resolution=resolution,
    )
    if thinking_block is not None:
        if resumed_preparation is not None:
            # This echo is the preparation's existing owner, not a fresh
            # optimistic row. Keep its frozen inputs and any newer session
            # identity intact, and return the state machine to the same
            # retryable persistence pause used by durable commit failures.
            self._pause_prepared_commit(
                resumed_preparation.preparation_id,
                owner.ConsolePreparationPauseKind.PERSISTENCE,
            )
            return thinking_block
        # The optimistic echo exists only to cover a slow readiness probe.
        # Compatibility is still pre-acceptance: leave neither a synthetic
        # transcript owner nor an auto-derived title behind, so the exact
        # draft can be retried after the persistent backend is upgraded.
        if echoed_user is not None:
            self.store.rollback_transient_send(
                session.id,
                echoed_user.id,
                title=pre_send_title,
                persisted_conversation_id=pre_send_conversation_id,
            )
        return thinking_block

    if resumed_preparation is not None:
        turn_context = resumed_preparation.execution_context
    else:
        try:
            turn_context = self._finalize_turn_execution_context(
                configuration,
                library_authority,
                resolution,
            )
        except (TypeError, ValueError):
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
                self.store.delete_message(echoed_user.id)
            return self._block(session.id, "Provider destination is incomplete.")

    admission_policy: owner.CapturePolicySnapshot | None = None
    if resumed_preparation is not None:
        capture_mode = resumed_preparation.capture_mode
    else:
        try:
            admission_policy = self.capture_policy_snapshot(session.id)
            capture_mode = (
                owner.ConsoleTraceCaptureMode.CAPTURE_ON
                if origin
                in {
                    owner.ConsoleSubmissionOrigin.MANUAL,
                    owner.ConsoleSubmissionOrigin.QUEUED,
                }
                and admission_policy.effective_capture_enabled
                # TASK-25814: policy alone is not enough -- the RUNTIME has
                # to be able to honour it. The gateway's durable-capture
                # seam is optional and unsupplied in production, so
                # preparing Capture-On against a gateway without one
                # guaranteed a pre-dispatch refusal on EVERY send
                # (`_reserve_trace_call` raises on its first statement).
                # Capture Off is the app's own modelled outcome for "no
                # capture" (`one_shot_capture_off`), so fall back to it
                # rather than promise something that cannot be recorded.
                # The dispatch guard itself is a deliberate invariant and
                # is untouched.
                and bool(
                    getattr(
                        self.provider_gateway,
                        "supports_durable_capture",
                        False,
                    )
                )
                else owner.ConsoleTraceCaptureMode.CAPTURE_OFF
            )
        except Exception as exc:  # noqa: BLE001 - preserve the fail-soft capture-policy gate.
            owner.logger.bind(error_type=type(exc).__name__).warning(
                "capture_policy_preparation_failed"
            )
            capture_mode = owner.ConsoleTraceCaptureMode.CAPTURE_OFF
    from .console_send_diagnostics import record_send_stage

    record_send_stage(
        "capture_policy",
        capture_enabled=capture_mode is owner.ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    if resumed_preparation is not None:
        pii_redaction_enabled = resumed_preparation.pii_redaction_enabled
        pii_ruleset_revision_id = resumed_preparation.pii_ruleset_revision_id
        next_trace_privacy_revision = resumed_preparation.next_trace_privacy_revision
    else:
        pii_redaction_enabled = (
            capture_mode is owner.ConsoleTraceCaptureMode.CAPTURE_ON
            and admission_policy is not None
            and admission_policy.pii_redaction_enabled
        )
        pii_ruleset_revision_id = (
            admission_policy.pii_ruleset_revision_id
            if pii_redaction_enabled and admission_policy is not None
            else None
        )
        next_trace_privacy_revision = (
            admission_policy.next_privacy_revision
            if admission_policy is not None
            and (
                admission_policy.next_capture_enabled is not None
                or admission_policy.next_pii_redaction_enabled is not None
            )
            else None
        )
    if (
        resumed_preparation is None
        and session.ephemeral
        and origin is owner.ConsoleSubmissionOrigin.QUEUED
        and capture_mode is owner.ConsoleTraceCaptureMode.CAPTURE_ON
    ):
        visible_copy = (
            "Queued Capture On needs a durable conversation. Save the chat or "
            "turn Capture Off before resuming the queue."
        )
        if echoed_user is not None:
            self.store.rollback_transient_send(
                session.id,
                echoed_user.id,
                title=pre_send_title,
                persisted_conversation_id=pre_send_conversation_id,
            )
        self._set_run_state(
            owner.ConsoleRunState.blocked(visible_copy),
            session_id=session.id,
        )
        return owner.ConsoleSubmitResult(
            False,
            False,
            visible_copy,
            session_id=session.id,
            origin=origin,
            queue_entry_id=queue_entry_id,
            provider_started=False,
        )

    preparation: owner.ConsoleTurnPreparation | None = resumed_preparation
    preparation_outcome: owner.ConsolePreparationOutcome | None = (
        self._preparation_outcomes.get(resumed_preparation.preparation_id)
        if resumed_preparation is not None
        else None
    )
    ordinary_library_text = self._ordinary_library_text(
        clean_draft,
        origin,
        has_pending_attachment=has_pending_attachment,
    )
    if (
        ordinary_library_text
        or capture_mode is owner.ConsoleTraceCaptureMode.CAPTURE_ON
    ) and resumed_preparation is None:
        automatic_eligible = (
            ordinary_library_text
            and library_authority.policy.auto_retrieve
            is owner.ConsoleAutoRetrieve.AUTOMATIC
            and explicit_evidence_staged is False
        )
        if not ordinary_library_text:
            initial_state = owner.ConsoleTurnPreparationState.READY
        else:
            initial_state = (
                owner.initial_preparation_state(library_authority.policy.auto_retrieve)
                if automatic_eligible
                or library_authority.policy.auto_retrieve
                is owner.ConsoleAutoRetrieve.NEVER
                else owner.ConsoleTurnPreparationState.READY
            )
        queue_generation = None
        if origin is owner.ConsoleSubmissionOrigin.QUEUED:
            queue_generation = self.prompt_queue_registry.snapshot(session.id).revision
        if custodied_inputs:
            frozen_prefill = admitted_prefill
            frozen_prefill_from_one_shot = admitted_prefill_from_one_shot
            captured_prefill_revision = admitted_prefill_revision
        elif preserve_composer:
            frozen_prefill = self._pinned_prefill_for_session(session.id)
            frozen_prefill_from_one_shot = False
            captured_prefill_revision = None
        else:
            one_shot_prefill, captured_prefill_revision = (
                self.store.session_one_shot_prefill_snapshot(session.id)
            )
            frozen_prefill, frozen_prefill_from_one_shot = self._resolve_submit_prefill(
                session.id
            )
        one_shot_prefill = frozen_prefill if frozen_prefill_from_one_shot else None
        if custodied_inputs:
            staged_evidence_frozen = staged_evidence_launch is not None
            staged_evidence = staged_evidence_launch
        elif preserve_composer:
            staged_evidence_frozen, staged_evidence, staged_evidence_release = (
                True,
                None,
                None,
            )
        else:
            (
                staged_evidence_frozen,
                staged_evidence,
                staged_evidence_release,
            ) = self._snapshot_staged_evidence()
        preparation = owner.ConsoleTurnPreparation(
            preparation_id=str(owner.uuid4()),
            attempt_id=library_authority.attempt_id,
            session_id=session.id,
            origin=origin.value,
            queue_entry_id=queue_entry_id,
            executed_draft=executed_draft_text,
            execution_context=turn_context,
            transient_user_message_id=(
                echoed_user.id if echoed_user is not None else None
            ),
            attachment_ids=tuple(pending.attachment_id for pending in pendings),
            evidence_ids=(
                ("explicit-staged-evidence",) if explicit_evidence_staged else ()
            ),
            prefill_id=(
                "prefill-"
                + owner.hashlib.sha256(one_shot_prefill.encode("utf-8")).hexdigest()[
                    :24
                ]
                if one_shot_prefill is not None
                else None
            ),
            queue_generation=queue_generation,
            pre_send_title=pre_send_title,
            pre_send_conversation_id=pre_send_conversation_id,
            state=initial_state,
            pause_kind=None,
            one_shot_bypass=False,
            ephemeral=session.ephemeral,
            capture_mode=capture_mode,
            pii_redaction_enabled=pii_redaction_enabled,
            pii_ruleset_revision_id=pii_ruleset_revision_id,
            next_trace_privacy_revision=next_trace_privacy_revision,
        )
        preparation = owner.pause_temporary_capture_on(preparation)
        if self._begin_submit_preparation(active_task, preparation) is None:
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            return owner.ConsoleSubmitResult(
                False,
                False,
                "Another send is still preparing for this conversation.",
            )
        if origin is owner.ConsoleSubmissionOrigin.QUEUED and (
            queue_entry_id is None
            or not self.prompt_queue_coordinator.bind_claimed_preparation(
                session.id,
                entry_id=queue_entry_id,
                preparation_id=preparation.preparation_id,
            )
        ):
            self._abandon_preparation(preparation.preparation_id)
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            return owner.ConsoleSubmitResult(
                False,
                False,
                "Queued preparation could not bind its exact entry.",
                session_id=session.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )
        self._prepared_send_continuations[preparation.preparation_id] = (
            owner._PreparedSendContinuation(
                preparation_id=preparation.preparation_id,
                preserve_composer=preserve_composer,
                attachments=tuple(pendings),
                prefill=frozen_prefill,
                prefill_from_one_shot=frozen_prefill_from_one_shot,
                one_shot_prefill_revision=(
                    captured_prefill_revision if frozen_prefill_from_one_shot else None
                ),
                staged_evidence_frozen=staged_evidence_frozen,
                staged_evidence=(
                    owner._PreparedEvidenceLease(
                        staged_evidence,
                        capture=staged_evidence_capture,
                        release=staged_evidence_release,
                    )
                    if staged_evidence is not None
                    else None
                ),
            )
        )
        prepared_continuation = self._prepared_send_continuations[
            preparation.preparation_id
        ]
        if (
            preparation.state is owner.ConsoleTurnPreparationState.PAUSED
            and preparation.pause_kind
            is owner.ConsolePreparationPauseKind.TEMPORARY_CAPTURE
        ):
            visible_copy = (
                "Trace capture needs a saved chat. Choose Save & Send, "
                "Send without capture, or Cancel."
            )
            self._set_run_state(
                owner.ConsoleRunState.blocked(visible_copy),
                session_id=session.id,
            )
            return owner.ConsoleSubmitResult(
                False,
                False,
                visible_copy,
                session_id=session.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
                preparation_id=preparation.preparation_id,
                provider_started=False,
            )
        if preparation.state is owner.ConsoleTurnPreparationState.PREPARING:
            preparation_outcome = await self.prepare_library_for_turn(
                preparation.preparation_id
            )
            if preparation_outcome.state is not owner.ConsoleTurnPreparationState.READY:
                self._set_run_state(
                    owner.ConsoleRunState.blocked(
                        "Library preparation paused before provider dispatch."
                    ),
                    session_id=session.id,
                )
                return owner.ConsoleSubmitResult(
                    False,
                    False,
                    "Library preparation paused before provider dispatch.",
                    session_id=session.id,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                )

    if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

        try:
            accepted = await self._fleet_wake.accept(wake_authorization, session.id)
        except (owner.SQLiteError, OSError, AutomaticWorkRefused):
            return self._block(
                session.id,
                "Automatic work paused. Results are saved; send a message to continue.",
            )
        if not accepted:
            return self._block(
                session.id,
                "Manual work has priority. Background results are saved.",
            )
    citation_context: str | None = None
    citation_trace_builder: owner.CitationTraceBuilder | None = None
    prompt_evidence_set_id: str | None = None
    citation_repair_contract: owner.CitationRepairContract | None = None
    terminal_citation_finalizer: owner.TerminalCitationFinalizer | None = None
    try:
        provider_messages = self._provider_messages_for_session(
            session.id, annotate_ids=True, turn_context=turn_context
        )
        trace_source_messages = tuple(dict(row) for row in provider_messages)
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages, turn_context)
        if refuse is not None:
            # A substitution refusal is a block outcome like any other
            # (provider not ready, probe raise): fail the echoed row so the
            # refused command never enters the next send's provider context.
            # (A wake echoed nothing: None guard.)
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            if preparation is not None:
                self._abandon_preparation(preparation.preparation_id)
            return self._block(session.id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session.id, role=owner.ConsoleMessageRole.SYSTEM, content=note
            )
        if (
            preparation_outcome is not None
            and preparation_outcome.evidence_bundle is not None
        ):
            citation_context = owner.format_evidence_for_cited_answer(
                preparation_outcome.evidence_bundle
            )
        elif (
            origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
            and prepared_continuation is not None
            and prepared_continuation.staged_evidence_frozen
        ):
            (
                citation_context,
                citation_trace_builder,
                prompt_evidence_set_id,
                citation_repair_contract,
            ) = await self._capture_frozen_rag_context(
                clean_draft,
                turn_context,
                prepared_continuation,
            )
        elif origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE:
            # PR3a-2 Task 5: a wake notice is a delivery, not a query
            # -- retrieving evidence "about" a machine notice would
            # inject RAG context the user never asked for. The
            # pre-initialized Nones above stand.
            (
                citation_context,
                citation_trace_builder,
                prompt_evidence_set_id,
                citation_repair_contract,
            ) = await self._capture_rag_context(
                clean_draft,
                turn_context=turn_context,
                origin=origin,
            )
        has_exact_citation_context = (
            citation_trace_builder is not None or citation_repair_contract is not None
        )
        if citation_context and not has_exact_citation_context:
            provider_messages = self._prepend_evidence_context(
                provider_messages,
                citation_context,
            )
        if reference_records and executed_draft_text != clean_draft:
            # TASK-27021: the store echo keeps the RAW draft; the payload
            # carries the @-reference expansion. Swap the just-echoed last
            # user message. Runs BEFORE dictionaries/world-info -- the
            # expanded text is the user's composed message; note the
            # accepted hazard that dictionary keywords inside included
            # file content will also match.
            for _i in range(len(provider_messages) - 1, -1, -1):
                if provider_messages[_i].get("role") == "user":
                    provider_messages = (
                        provider_messages[:_i]
                        + [{**provider_messages[_i], "content": executed_draft_text}]
                        + provider_messages[_i + 1 :]
                    )
                    break
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session.id, turn_context
        )
        provider_messages = await self._apply_world_info(
            provider_messages, session.id, turn_context
        )
        if citation_context and has_exact_citation_context:
            provider_messages = self._prepend_evidence_context(
                provider_messages,
                citation_context,
            )
        if citation_context and echoed_user is not None:
            trace_prefix = f"console-trace:{echoed_user.id}:retrieval"
            retrieval_event_id = f"{trace_prefix}:retrieval_completed"
            attached_event_id = f"{trace_prefix}:context_attached"
            self.store.record_trace_event(
                session.id,
                anchor_message_id=echoed_user.id,
                event_kind="context_attached",
                summary="Retrieved context attached",
                status="completed",
                event_id=attached_event_id,
                parent_event_id=retrieval_event_id,
                source_event_id=retrieval_event_id,
                sensitivity="system_context",
            )
            self.store.record_trace_event(
                session.id,
                anchor_message_id=echoed_user.id,
                event_kind="context_injected",
                summary="Retrieved context injected into provider request",
                status="completed",
                event_id=f"{trace_prefix}:context_injected",
                parent_event_id=attached_event_id,
                source_event_id=attached_event_id,
                sensitivity="system_context",
            )
        if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
            # The one-shot prefill is USER-staged state; a wake must
            # not consume (and thereby destroy) it.
            prefill, prefill_from_one_shot, one_shot_prefill_revision = (
                None,
                False,
                None,
            )
        elif prepared_continuation is not None:
            prefill = prepared_continuation.prefill
            prefill_from_one_shot = prepared_continuation.prefill_from_one_shot
            one_shot_prefill_revision = prepared_continuation.one_shot_prefill_revision
        elif custodied_inputs:
            prefill = admitted_prefill
            prefill_from_one_shot = admitted_prefill_from_one_shot
            one_shot_prefill_revision = admitted_prefill_revision
        else:
            prefill, prefill_from_one_shot = self._resolve_submit_prefill(session.id)
            one_shot_prefill_revision = (
                self.store.session_one_shot_prefill_snapshot(session.id)[1]
                if prefill_from_one_shot
                else None
            )
        terminal_citation_finalizer = self._build_terminal_citation_finalizer(
            context=citation_context,
            builder=citation_trace_builder,
            prompt_evidence_set_id=prompt_evidence_set_id,
        )
    except BaseException:
        # Any failure between the optimistic echo and the confirmed turn
        # (dictionary/world-info application, prefill resolution) must also
        # fail the echoed row, or a never-sent message leaks into the next
        # send's provider context (`skip_failed` only drops "failed" rows).
        # (A wake echoed nothing: None guard.)
        if echoed_user is not None:
            self._mark_transient_echo_blocked(echoed_user.id)
        if preparation is not None:
            self._abandon_preparation(preparation.preparation_id)
        raise
    # The accepted-hook fires only once the turn is confirmed to
    # actually proceed (Qodo finding 3, PR #636 bot review): it used to
    # fire right after the USER row was appended, BEFORE this skill
    # substitution/trust check ran. In the real ChatScreen this hook
    # clears the composer, so firing it before a substitution refusal
    # ate the refused draft the user needs to correct. A substitution
    # refusal is a `_block()` outcome exactly like any other (provider
    # not ready, policy block, validation failure) and those already
    # never reach this hook -- this ordering just extends that same
    # rule to cover it too.
    if origin is owner.ConsoleSubmissionOrigin.QUEUED and not (
        self.prompt_queue_coordinator.authorizes(queue_authorization, session.id)
    ):
        # Close/shutdown can tombstone the chain while this claimed turn
        # awaits readiness/substitution/RAG. Revalidate immediately before
        # acceptance so cancellation cannot turn that stale claim into a
        # durable user message or provider dispatch. (A wake echoed
        # nothing: None guard.)
        if echoed_user is not None:
            self._mark_transient_echo_blocked(echoed_user.id)
        if preparation is not None:
            self._abandon_preparation(preparation.preparation_id)
        return owner.ConsoleSubmitResult(
            False,
            False,
            "Queued turn canceled before it could start.",
        )
    # PR3a-2 Task 5: the wake notice enters the MODEL PAYLOAD here, as
    # a payload-only trailing user-role entry -- appended AFTER every
    # per-send transform (substitution/dictionaries/world-info ran on
    # the history above and must never rewrite the notice) and never
    # written to the store (the transcript's record is the SYSTEM
    # machine-origin row at the acceptance point below). Trailing
    # user-role is deliberate: SYSTEM transcript rows are dropped from
    # payloads by design, and a payload ending on an assistant row is
    # a prefill to strict providers -- see console_fleet_wake's
    # delivery-path decision record for why neither turn_bundle_block
    # nor the system fold can carry this.
    if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        provider_messages = [
            *provider_messages,
            {
                "role": owner.ConsoleMessageRole.USER.value,
                "content": clean_draft,
            },
        ]
    # This await remains before acceptance. A refusal or cancellation must
    # release the exact optimistic echo and preparation, preserving custody.
    hook_context = ""
    if origin is owner.ConsoleSubmissionOrigin.MANUAL:
        try:
            from tldw_chatbook.Agents.run_hooks import truncate_hook_text

            hooks_engine = self._run_hooks_engine()
            outcome = (
                await hooks_engine.fire_async(
                    "UserPromptSubmit",
                    session_id=session.id,
                    data={"prompt": truncate_hook_text(clean_draft)},
                )
                if hooks_engine is not None
                else None
            )
        except BaseException:
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            if preparation is not None:
                self._abandon_preparation(preparation.preparation_id)
            self._set_run_state(
                owner.ConsoleRunState(
                    owner.ConsoleRunStatus.STOPPED, "Send cancelled before acceptance."
                ),
                session_id=session.id,
            )
            raise
        if outcome is not None and outcome.blocked:
            if echoed_user is not None:
                self._mark_transient_echo_blocked(echoed_user.id)
            if preparation is not None:
                self._abandon_preparation(preparation.preparation_id)
            self._set_run_state(
                owner.ConsoleRunState.blocked(f"Blocked by hook: {outcome.reason}"),
                session_id=session.id,
            )
            self.store.append_message(
                session.id,
                role=owner.ConsoleMessageRole.SYSTEM,
                content=f"Send blocked by hook: {outcome.reason}",
                persist=self.store.persistence is not None,
                metadata=owner.MessageMetadata(origin=owner.MESSAGE_ORIGIN_HOOK),
            )
            return owner.ConsoleSubmitResult(
                False,
                False,
                f"Blocked by hook: {outcome.reason}",
                session_id=session.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )
        hook_context = outcome.context if outcome is not None else ""
    if hook_context:
        # Freeze the model-visible half before either acceptance path seals
        # its request. The SYSTEM audit row is excluded from future history.
        provider_messages = [
            *provider_messages,
            {"role": owner.ConsoleMessageRole.USER.value, "content": hook_context},
        ]
    if preparation is not None:
        current_preparation = self._preparation_by_id(preparation.preparation_id)
        if current_preparation is None or (
            current_preparation.state
            is not owner.ConsoleTurnPreparationState.COMMITTING
            and not self._transition_preparation(
                preparation.preparation_id,
                owner.ConsoleTurnPreparationState.READY,
                owner.ConsoleTurnPreparationState.COMMITTING,
            )
        ):
            return owner.ConsoleSubmitResult(
                False,
                False,
                "Prepared turn changed before provider dispatch.",
                session_id=session.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
            )
    committed_context_epoch = self.store.conversation_context_epoch(session.id)
    if durable_turn and preparation is not None and echoed_user is not None:
        return await self._accept_durable_turn(
            session=session,
            preparation=preparation,
            preparation_outcome=preparation_outcome,
            prepared_continuation=prepared_continuation,
            echoed_user=echoed_user,
            staged_title=staged_title,
            staged_attachments=staged_attachments,
            resolution=resolution,
            provider_messages=provider_messages,
            trace_source_messages=trace_source_messages,
            prefill=prefill,
            prefill_from_one_shot=prefill_from_one_shot,
            one_shot_prefill_revision=one_shot_prefill_revision,
            skill_bindings=tuple(skill_bindings),
            skill_bundle_block=skill_bundle_block,
            citation_repair_contract=citation_repair_contract,
            terminal_citation_finalizer=terminal_citation_finalizer,
            turn_context=turn_context,
            origin=origin,
            queue_entry_id=queue_entry_id,
            committed_context_epoch=committed_context_epoch,
            custody_acceptance_hook=custody_acceptance_hook,
            hook_context=hook_context,
        )
    # TASK-1364: record the accepted send to the shared prompt history.
    # Same placement rule as the accepted-hook above: only a send that is
    # confirmed to proceed is recorded -- every `_block`/refusal path
    # returns before this point, and `_record_prompt_history` itself
    # skips empty (attachment-only) drafts. A wake notice is not a
    # prompt the user typed and never enters their prompt history.
    if origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        try:
            await self._record_prompt_history(clean_draft)
        except BaseException:
            if preparation is not None:
                self._rollback_committing_preparation(preparation.preparation_id)
            raise
    if self._disposed or (
        self._shutdown_requested.is_set()
        and origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
    ):
        if echoed_user is not None:
            try:
                self._mark_transient_echo_blocked(echoed_user.id)
            except KeyError:
                pass
        if preparation is not None:
            self._rollback_committing_preparation(preparation.preparation_id)
        return owner.ConsoleSubmitResult(
            False,
            False,
            "Console shut down before turn acceptance.",
            session_id=session.id,
            origin=origin,
            queue_entry_id=queue_entry_id,
        )
    # TASK-485: the turn is confirmed to proceed — flush the deferred USER
    # echo to durable storage now (creating the conversation), BEFORE the
    # assistant row, so a reload shows the user's prompt ahead of its reply.
    #
    # PR3a-2 Task 5, the wake half: the SYSTEM-class notice row is
    # appended HERE, only once the turn is confirmed -- so a blocked
    # wake leaves no orphaned notice -- ahead of the assistant row,
    # persisted, and carrying the machine-origin metadata that marks
    # it as not-user-input for every machine consumer.
    if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        echoed_user = self.store.append_message(
            session.id,
            role=owner.ConsoleMessageRole.SYSTEM,
            content=clean_draft,
            persist=self.store.persistence is not None,
            metadata=owner.MessageMetadata(origin=owner.MESSAGE_ORIGIN_AGENT_WAKE),
        )
    else:
        try:
            self.store.persist_message_if_needed(echoed_user.id)
        except BaseException:
            if preparation is not None:
                self._rollback_committing_preparation(preparation.preparation_id)
            raise
        if hook_context:
            # Run hooks (spec 2026-09-11, Task 7): a UserPromptSubmit
            # hook's captured stdout is recorded as its own hook-origin
            # SYSTEM row -- appended only once the user echo is
            # confirmed proceeding (same placement rule as the wake
            # notice above) and BEFORE the assistant row, persisted
            # like it, and never merged into the user's message.
            self.store.append_message(
                session.id,
                role=owner.ConsoleMessageRole.SYSTEM,
                content=hook_context,
                persist=self.store.persistence is not None,
                metadata=owner.MessageMetadata(origin=owner.MESSAGE_ORIGIN_HOOK),
            )
    assistant: owner.ConsoleChatMessage | None = None
    citation_repair_session = (
        owner.ConsoleCitationRepairSession(
            contract=citation_repair_contract,
            resolution=resolution,
        )
        if citation_repair_contract is not None
        else None
    )
    # task-15860: a wake turn in flight is exempt from `leave_console()`
    # (owner ruling -- see that method). Registered here, released in
    # the `finally` below, so the exemption cannot outlive the turn.
    if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
        self._agent_wake_turn_sessions.add(session.id)
    try:
        assistant = self.store.append_message(
            session.id,
            role=owner.ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
            terminal_citation_finalizer=terminal_citation_finalizer,
            defer_terminal_persistence=citation_repair_session is not None,
        )
        if (
            session.ephemeral
            and origin
            in {
                owner.ConsoleSubmissionOrigin.MANUAL,
                owner.ConsoleSubmissionOrigin.QUEUED,
            }
            and preparation is not None
        ):
            self.store.register_ephemeral_dispatch_recovery(
                session.id,
                user_message_id=echoed_user.id,
                assistant_message_id=assistant.id,
                preparation_id=preparation.preparation_id,
                attempt_id=turn_context.library_authority.attempt_id,
                checkpoint_state=owner.ConsoleDispatchCheckpointState.ACCEPTED,
                origin=origin.value,
                queue_entry_id=queue_entry_id,
                frozen_authority=turn_context.library_authority,
                resolved_destination=turn_context.resolved_destination,
                reconstructability=owner.ConsoleDispatchReconstructability(
                    attachments_reconstructable=True,
                    evidence_reconstructable=not bool(
                        prepared_continuation is not None
                        and (
                            prepared_continuation.staged_evidence_frozen
                            or prepared_continuation.staged_evidence is not None
                        )
                    ),
                    prefill_reconstructable=(
                        prefill is None and not prefill_from_one_shot
                    ),
                    opaque_reference=(f"opaque:{preparation.preparation_id}"),
                ),
                runtime_active=True,
            )
        if preparation is not None and not self._transition_preparation(
            preparation.preparation_id,
            owner.ConsoleTurnPreparationState.COMMITTING,
            owner.ConsoleTurnPreparationState.ACCEPTED,
        ):
            raise RuntimeError("Prepared turn changed before acceptance.")
        stream_signals = self._admit_capture_policy(
            session.id,
            origin,
            frozen_capture_enabled=(
                capture_mode is owner.ConsoleTraceCaptureMode.CAPTURE_ON
            ),
            frozen_pii_redaction_enabled=pii_redaction_enabled,
            frozen_pii_ruleset_revision_id=pii_ruleset_revision_id,
            frozen_next_trace_privacy_revision=next_trace_privacy_revision,
        )
        self._release_prepared_evidence(prepared_continuation)
        if not custodied_inputs:
            for pending in pendings:
                self.store.consume_pending_attachment(session.id, pending.attachment_id)
        if custody_acceptance_hook is not None:
            custody_acceptance_hook()
        self._notify_submission_accepted(
            session_id=session.id,
            preserve_composer=preserve_composer,
            origin=origin,
            entry_id=queue_entry_id,
            context_epoch=committed_context_epoch,
            preparation_id=(
                preparation.preparation_id if preparation is not None else None
            ),
            assistant_message_id=assistant.id,
            defer_queued_settlement=(
                resumed_preparation is not None
                and origin is owner.ConsoleSubmissionOrigin.QUEUED
            ),
        )

        async def enter_ephemeral_provider_dispatch() -> None:
            if (
                session.ephemeral
                and self.store.dispatch_recovery_for_session(session.id) is not None
                and self.store.begin_ephemeral_dispatch(
                    session.id,
                    assistant_message_id=assistant.id,
                    new_attempt_id=turn_context.library_authority.attempt_id,
                )
                is None
            ):
                raise RuntimeError(
                    "Ephemeral dispatch checkpoint changed before provider entry."
                )
            if preparation is not None and not self._transition_preparation(
                preparation.preparation_id,
                owner.ConsoleTurnPreparationState.ACCEPTED,
                owner.ConsoleTurnPreparationState.DISPATCH_STARTED,
            ):
                raise RuntimeError("Prepared turn changed before provider dispatch.")

        deferred_provider_dispatch = bool(
            getattr(self.provider_gateway, "deferred_dispatch_boundary", False)
        )
        if (
            not deferred_provider_dispatch
            and session.ephemeral
            and self.store.dispatch_recovery_for_session(session.id) is not None
            and self.store.begin_ephemeral_dispatch(
                session.id,
                assistant_message_id=assistant.id,
                new_attempt_id=turn_context.library_authority.attempt_id,
            )
            is None
        ):
            raise RuntimeError(
                "Ephemeral dispatch checkpoint changed before provider entry."
            )

        stream_result = await self._stream_assistant_response(
            route=owner.ConsoleRequestRoute.FRESH,
            resolution=resolution,
            work_origin=(
                owner.WorkOrigin.AUTOMATIC
                if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE
                else owner.WorkOrigin.MANUAL
            ),
            work_chain_id=(
                wake_authorization.work_chain_id
                if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE
                else None
            ),
            provider_messages=provider_messages,
            assistant_message_id=assistant.id,
            prefill=prefill,
            prefill_from_one_shot=prefill_from_one_shot,
            one_shot_prefill_revision=one_shot_prefill_revision,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            citation_repair_session=citation_repair_session,
            turn_context=turn_context,
            preparation_id=(
                preparation.preparation_id if preparation is not None else None
            ),
            stream_signals=stream_signals,
            before_provider_dispatch=(
                enter_ephemeral_provider_dispatch
                if deferred_provider_dispatch
                else None
            ),
            trusted_profile_user_message_id=(
                echoed_user.id
                if echoed_user.role is owner.ConsoleMessageRole.USER
                else None
            ),
        )
        if (
            not stream_result.accepted
            and origin is not owner.ConsoleSubmissionOrigin.AGENT_WAKE
        ):
            self._mark_transient_echo_blocked(echoed_user.id)
        result = owner.replace(
            stream_result,
            session_id=session.id,
            user_message_id=echoed_user.id,
            assistant_message_id=assistant.id,
            terminal_status=self.run_state_for(session.id).status,
            origin=origin,
            queue_entry_id=queue_entry_id,
            committed_context_epoch=committed_context_epoch,
        )
        if preparation is not None:
            self._settle_accepted_preparation(preparation.preparation_id)
        return result
    except BaseException as exc:
        if isinstance(exc, owner.ConsoleDispatchSettlementError):
            if assistant is not None:
                self.store.release_dispatch_recovery_action(
                    session.id,
                    assistant.id,
                )
            raise
        if (
            isinstance(exc, owner.TraceCallPersistenceError)
            and origin is owner.ConsoleSubmissionOrigin.MANUAL
            and preparation is not None
        ):
            current = self._preparation_by_id(preparation.preparation_id)
            if (
                current is not None
                and current.state is owner.ConsoleTurnPreparationState.ACCEPTED
            ):
                paused_shape = owner.pause_for_trace_call_failure(current, exc)
                paused = self.store.compare_and_set_preparation(
                    current.session_id,
                    owner.ConsolePreparationTransition(
                        preparation_id=current.preparation_id,
                        expected_state=current.state,
                        new_state=paused_shape.state,
                        pause_kind=paused_shape.pause_kind,
                        new_attempt_id=None,
                    ),
                )
                if paused is not None:
                    visible_copy = (
                        "Trace capture could not start. Retry, Send without "
                        "capture, or Cancel."
                    )
                    self._set_run_state(
                        owner.ConsoleRunState.blocked(visible_copy),
                        session_id=session.id,
                    )
                    return owner.ConsoleSubmitResult(
                        True,
                        True,
                        visible_copy,
                        session_id=session.id,
                        user_message_id=(
                            echoed_user.id if echoed_user is not None else None
                        ),
                        assistant_message_id=(
                            assistant.id if assistant is not None else None
                        ),
                        terminal_status=owner.ConsoleRunStatus.BLOCKED,
                        origin=origin,
                        queue_entry_id=queue_entry_id,
                        committed_context_epoch=committed_context_epoch,
                        preparation_id=preparation.preparation_id,
                        provider_started=False,
                    )
        accepted_cancellation = isinstance(exc, owner.asyncio.CancelledError) and (
            assistant is not None and echoed_user is not None
        )
        explicit_cancellation = accepted_cancellation and (
            self._accepted_cancellation_was_requested(session.id)
        )
        if assistant is not None:
            try:
                if explicit_cancellation:
                    self._mark_stream_stopped(
                        assistant.id,
                        visible_copy="Response stopped.",
                    )
                else:
                    self.store.mark_message_failed(assistant.id)
                    self._set_run_state(
                        owner.ConsoleRunState(
                            owner.ConsoleRunStatus.FAILED,
                            "Accepted turn failed before provider dispatch.",
                        ),
                        session_id=session.id,
                    )
            except KeyError:
                pass
        if preparation is not None:
            current = self._preparation_by_id(preparation.preparation_id)
            if (
                current is not None
                and current.state is owner.ConsoleTurnPreparationState.COMMITTING
            ):
                self._rollback_committing_preparation(preparation.preparation_id)
            elif current is not None and current.state in {
                owner.ConsoleTurnPreparationState.ACCEPTED,
                owner.ConsoleTurnPreparationState.DISPATCH_STARTED,
                owner.ConsoleTurnPreparationState.DISPATCHED,
            }:
                self._settle_accepted_preparation(preparation.preparation_id)
        if accepted_cancellation:
            terminal_state = self.run_state_for(session.id)
            return owner.ConsoleSubmitResult(
                True,
                True,
                terminal_state.visible_copy
                or "Accepted turn failed before provider dispatch.",
                session_id=session.id,
                user_message_id=echoed_user.id,
                assistant_message_id=assistant.id,
                terminal_status=terminal_state.status,
                origin=origin,
                queue_entry_id=queue_entry_id,
                committed_context_epoch=committed_context_epoch,
            )
        raise
    finally:
        if origin is owner.ConsoleSubmissionOrigin.AGENT_WAKE:
            self._agent_wake_turn_sessions.discard(session.id)
        if assistant is not None:
            self.store.clear_terminal_citation_state(assistant.id)
        del terminal_citation_finalizer
        del citation_trace_builder
