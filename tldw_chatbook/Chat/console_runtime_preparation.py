"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from . import console_runtime as native_host


async def _run_custodied_turn(
    self,
    record: native_host._ConsoleTurnCustodyRecord,
    *,
    origin: native_host.ConsoleSubmissionOrigin,
    queue_entry_id: str | None,
    queue_authorization: native_host.Any | None,
    wake_authorization: native_host.Any | None,
    goal_authorization: native_host.Any | None,
    raise_on_refusal: bool,
    hook_read: native_host.Any = None,
    first_send_source: tuple,
) -> native_host.Any:
    """Run one screen-free turn using only its frozen custody record.

    ``hook_read`` is the received intent's one full hook authority read
    (``ConsoleHookAttemptRead``); it goes, as an argument, only to this
    record's own initial submission (ADR-230 decision 3).
    """
    request = record.request
    controller = self._chat_controller
    if request is None or controller is None:
        raise RuntimeError("Console controller is unavailable for runtime custody.")

    def require_received():
        self._require_received_custody_current(record, controller)

    require_received()
    if self._archive_recovery is not None and record.received_intent is not None:
        self._archive_recovery.accept_user_proof(
            request.configuration.recovery_user_origin,
            request.session_id,
            request.turn_id,
        )
    if self._archive_recovery is not None and origin in {
        native_host.ConsoleSubmissionOrigin.MANUAL,
        native_host.ConsoleSubmissionOrigin.QUEUED,
    }:
        await self._archive_recovery.prepare_for_admission(request.session_id)
        require_received()
    if record.archive_conversation_id:
        from tldw_chatbook.Chat.conversation_archive_actions import (
            conversation_send_refusal,
        )

        refusal = await conversation_send_refusal(
            self._app, record.archive_conversation_id
        )
        if refusal:
            notify = getattr(self._app, "notify", None)
            if callable(notify):
                notify(refusal, severity="warning")
            # TASK-33621.2: the parked turn's shelf entry states this
            # refusal too, like a controller refusal's below.
            raise native_host._ConsoleTurnRefusedError(
                "Console conversation is unavailable for submission.",
                reason=refusal,
            )

    def mark_durable_acceptance() -> None:
        from .console_received_dispatch import publish_received_acceptance

        publish_received_acceptance(self, record, controller)

    async def submit() -> native_host.Any:
        require_received()
        controller.prompt_queue_coordinator.bind_turn_request(request, origin=origin)
        from .console_received_turn import bind_received_turn_claim

        from .console_chat_controller import ConsoleChatController

        shared = (
            {"_hook_read": hook_read}
            if hook_read is not None
            and getattr(controller.submit_draft, "__func__", None)
            is ConsoleChatController.submit_draft
            else {}
        )
        issuing_store, issuing_claim = record.store, record.received_claim
        try:
            with bind_received_turn_claim(issuing_store, issuing_claim):
                from .console_received_dispatch import received_attachment_metadata

                return await controller.submit_draft(
                    request.draft,
                    session_id=request.session_id,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                    queue_authorization=queue_authorization,
                    wake_authorization=wake_authorization,
                    goal_authorization=goal_authorization,
                    configuration=native_host.replace(
                        request.configuration,
                        skill_context_maximum={
                            **request.configuration.skill_context_maximum,
                            "plugin_turn_id": request.turn_id,
                        },
                    ),
                    accepted_attachments=received_attachment_metadata(
                        controller, request.session_id
                    ),
                    captured_one_shot_prefill=request.one_shot_prefill,
                    captured_one_shot_prefill_revision=(
                        request.one_shot_prefill_revision
                    ),
                    staged_evidence_launch=request.staged_evidence_launch,
                    staged_evidence_capture=self._capture_frozen_console_staged_rag,
                    staged_evidence_release=native_host.functools.partial(
                        self.release_console_staged_evidence,
                        revision=record.inputs.staged_evidence_revision,
                        session_id=record.session_id,
                    ),
                    custody_acceptance_hook=mark_durable_acceptance,
                    **shared,
                )
        finally:
            # Nonpromoting routes release after physical submit completion,
            # before the enclosing prompt chain admits its next request.
            if issuing_store.release_received_turn(issuing_claim):
                self._note_received_admission_changed(issuing_store, request.session_id)

    result = (
        await submit()
        if origin
        in {
            native_host.ConsoleSubmissionOrigin.AGENT_WAKE,
            native_host.ConsoleSubmissionOrigin.GOAL_ITERATION,
        }
        else await controller.run_prompt_chain(
            session_id=request.session_id,
            initial_turn=submit,
        )
    )
    held_preparation_id = getattr(result, "preparation_id", None)
    if (
        not bool(getattr(result, "accepted", False))
        and not record.inputs.durable_accepted
        and raise_on_refusal
        # TASK-34350: a send held at the compaction threshold is waiting
        # on the user's answer in its own card, not refused; recording it
        # as a turn recovery too showed one send on two surfaces.
        and not (
            held_preparation_id is not None
            and controller.context_compaction_hold(held_preparation_id) is not None
        )
    ):
        from .console_unsent_turn import card_holds_refusal

        raise native_host._ConsoleTurnRefusedError(
            "Console turn was refused before durable acceptance.",
            reason=str(getattr(result, "visible_copy", "") or ""),
            preparation_id=held_preparation_id,
            card_held=card_holds_refusal(controller, held_preparation_id),
        )
    run_state_for = getattr(controller, "run_state_for", None)
    run_state = run_state_for(request.session_id) if callable(run_state_for) else None
    if (
        origin is native_host.ConsoleSubmissionOrigin.MANUAL
        and bool(getattr(result, "accepted", False))
        and getattr(run_state, "status", None) is native_host.ConsoleRunStatus.COMPLETED
    ):
        await self._record_successful_first_send(first_send_source)
    return result


async def _persist_first_send_when_quiescent(
    self, config, identity, app_config, controller, store
) -> None:
    """Run the original one-key save only beyond all original lifetimes."""
    import sys

    from . import console_first_send_persistence as persistence
    from .console_ordinary_tree import _function_unchanged

    check = persistence._NATIVE_CONFIG_CHECK

    def original_config():
        return (
            persistence._native_config_current is check[0]
            and _function_unchanged(check)
            and check[0]()
        )

    wakeup = self._first_send_wakeup
    try:
        while True:
            wakeup.clear()
            registry = getattr(controller, "prompt_queue_registry", None)
            queued = bool(
                registry is not None
                and any(state.total_count for state in registry._states.values())
            )
            with self._execution_capacity_lock:
                if (
                    not self._first_send_sources
                    and not self._ordinary_turns
                    and not self._turn_custody
                    and not queued
                ):
                    self._first_send_publishing = True
                    break
            await wakeup.wait()

        def current():
            return (
                sys.modules.get("tldw_chatbook.config") is config
                and original_config()
                and vars(self._app).get("console_runtime") is self
                and vars(self._app).get("app_config") is app_config
                and self._chat_controller is controller
                and self._chat_store is store
                and config.current_config_identity() == identity
            )

        if not current():
            raise RuntimeError("Original first-send source changed.")
        producer = native_host.asyncio.create_task(
            native_host.asyncio.to_thread(
                config.apply_settings_mutation_to_cli_config,
                {"console.onboarding": {"first_send_completed": True}},
                mutation_precondition=current,
            )
        )
        cancelled = False
        while not producer.done():
            try:
                await native_host.asyncio.shield(producer)
            except native_host.asyncio.CancelledError:
                cancelled = True
        result = producer.result()
        if cancelled:
            raise native_host.asyncio.CancelledError
        if result.conflict or result.failure_phase is not None:
            raise RuntimeError("Original first-send persistence failed.")
    except Exception as exc:  # noqa: BLE001 - the completed turn stays valid
        native_host.logger.warning(
            "Failed to persist Console first-send completion (exception_type={})",
            type(exc).__name__,
        )
    finally:
        with self._execution_capacity_lock:
            self._first_send_publishing = False
        self._wake_first_send()


async def prepare_hooks_v2(
    self,
    session_id: str,
    *,
    reason="startup",
    initiator="manual",
    configuration=None,
    _hook_read=None,
    test_only=False,
):
    """Initialize only at validated execution admission, never at view access.

    ``_hook_read`` is the submitting attempt's own earlier full consent
    read (``ConsoleHookAttemptRead``). While it stands for this session
    and owner, and only when it answers "no v2 hook to prepare"
    (``_hook_read_prepares_nothing``), it replaces the
    ``v2_configuration()`` re-read (ADR-230 decision 3). Everything that
    builds, keeps or replaces an engine still reads fresh, as before.
    """
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.run_hooks import load_hooks_config

    self._raise_if_disposed_or_session_fenced(session_id)
    permissions = self.ensure_hook_permissions()
    source = self._capture_hook_preparation_source(session_id, permissions)
    from .console_hook_preparation import attempt_v2_configuration_for

    shared = attempt_v2_configuration_for(permissions, session_id, _hook_read)
    if shared is not None and self._hook_read_prepares_nothing(
        session_id, shared[0], configuration
    ):
        # The early return below, reached without the re-read: nothing in
        # the attempt's read or this session's state needs an engine.
        self._require_hook_preparation_source(source)
        self._raise_if_disposed_or_session_fenced(session_id)
        return None
    review, targets = await self._read_hook_preparation(
        permissions.v2_configuration, source
    )
    self._require_hook_preparation_source(source)
    configured = load_hooks_config(
        {"hooks": review.config.section} if review.config.section_present else {}
    )
    engine = self.get_hooks_v2(session_id)
    native = getattr(engine, "native_plugins", None) if configuration is None else None
    plugin_request = None
    async with native_host.contextlib.AsyncExitStack() as preparation_handoff:
        if configuration is not None:
            maximum = configuration.skill_context_maximum
            if maximum.get("_plugin_requirement_ceiling") or any(
                row.get("plugin_owned") for row in maximum.get("available_skills", ())
            ):
                local = getattr(
                    self._chat_controller._skills_service, "local_service", None
                )
                service = getattr(local, "plugin_service", None)
                if service is None:
                    raise PermissionError("plugin_hook_authority_unavailable")
                if self._native_plugin_hook_preparation_required:
                    preparation = self._native_plugin_hook_preparation
                    if preparation is None:
                        raise RuntimeError(
                            "Native plugin hook preparation unavailable."
                        )
                    (
                        native,
                        plugin_request,
                    ) = await preparation_handoff.enter_async_context(
                        preparation.capture(source, service, configuration, review)
                    )
                else:
                    native = await service.hook_configuration(
                        maximum,
                        scratch_manager=self._scratch_spaces,
                        scratch_snapshot=self._scratch_spaces.snapshot(session_id),
                    )
                self._require_hook_preparation_source(source)
        signature = (
            configured,
            targets,
            native.signature if native is not None else None,
        )
        engine = self.get_hooks_v2(session_id)
        previous = self._hooks_v2_configured.get(session_id)
        if (
            review.ready
            and not configured.v2_handlers
            and not configured.v2_invalid_admissions
            and (native is None or not native.definitions)
            and engine is None
            and session_id not in self._hooks_v2_engines
            and session_id not in self._hooks_v2_lifecycles
            and session_id not in self._hooks_v2_configured
        ):
            self._require_hook_preparation_source(source)
            return None
        context_key = await self._read_hook_preparation(
            native_host.functools.partial(self._hooks_v2_context_key, session_id),
            source,
        )
        self._require_hook_preparation_source(source)
        self._raise_if_disposed_or_session_fenced(session_id)
        owner = self._hooks_v2_lifecycles.get(session_id)
        context_changed = owner is not None and owner.context_key != context_key
        if context_changed and native is not None and configuration is None:
            raise PermissionError("plugin_hook_workspace_admission_required")
        if (
            test_only
            and engine is not None
            and (
                previous != signature
                or context_changed
                or owner is None
                or not owner.live
                or owner.turn_scope is not None
            )
        ):
            raise PermissionError("plugin_test_requires_compatible_idle_hook_session")
        if (previous is not None and previous != signature) or context_changed:
            # The controller owns a reversible validation slot at this point.
            owner = self._hooks_v2_lifecycles.get(session_id)
            if owner is not None and getattr(owner, "turn_scope", None) is not None:
                raise RuntimeError("hook replacement requires idle session")
            await self.close_hooks_v2(session_id)
            self._require_hook_preparation_source(source)
            if self._hooks_v2_lifecycles.get(session_id) is owner:
                self._hooks_v2_lifecycles.pop(session_id, None)
            if context_changed and previous is None and engine is not None:
                # Host-injected definitions retain their authority resolver.
                engine = self.ensure_hooks_v2(
                    session_id,
                    engine.definitions,
                    engine.authority_check,
                    _native_preparation=plugin_request,
                    enabled=engine.enabled,
                    invalid_admissions=engine.invalid_admissions,
                    process_owner=engine.processes.owner,
                    process_owner_for=engine._process_owner_for,
                    environment=engine._environment,
                    host_environment=engine._host_environment,
                    command_launch=engine._command_launch,
                    dependency_required=engine._dependency_required,
                    event_projector=engine._event_projector,
                    launch_guard=engine.processes._launch_guard,
                    effect_authority_check=engine._effect_authority_check,
                )
            else:
                engine = None
            reason = "configuration_changed"
        if plugin_request is not None:
            plugin_request.check(service)
        if engine is None:
            if not review.ready:
                raise RuntimeError("Review enabled hooks before execution.")
            if (
                not configured.v2_handlers
                and not configured.v2_invalid_admissions
                and (native is None or not native.definitions)
            ):
                return None
            captured = {target.spec.id: target for target in targets}

            def session_end_current(handler, event):
                owner = getattr(engine, "lifecycle_owner", None)
                return bool(
                    self._disposed
                    and handler.type == "command"
                    and handler.event == "SessionEnd"
                    and not handler.effects
                    and not handler.required
                    and owner is not None
                    and self.get_hooks_v2(session_id) is engine
                    and owner._session_end_current(event)
                )

            def authority(handler, _event, _stage):
                if native is not None and handler.id in native.owners:
                    return (
                        not self._disposed
                        and permissions.configuration_current(review)
                        and native.authority(handler, _event, _stage)
                        and permissions.configuration_current(review)
                    )
                target = captured.get(handler.id)
                return bool(
                    target is not None
                    and (
                        not self._disposed
                        and permissions.target_current(target)
                        or session_end_current(handler, _event)
                        and permissions._session_end_current(
                            review, target, lambda: session_end_current(handler, _event)
                        )
                    )
                )

            def effects_current(handler, _event, _stage):
                if native is not None and handler.id in native.owners:
                    return (
                        not self._disposed
                        and permissions.configuration_current(review)
                        and native.effects_current(handler, _event, _stage)
                    )
                target = captured.get(handler.id)
                return bool(
                    target is not None
                    and not self._disposed
                    and permissions.configuration_current(review)
                    and permissions.target_current(target, refresh=False)
                )

            @native_host.contextlib.contextmanager
            def launch_guard(handler, event):
                if native is not None and handler.id in native.owners:
                    from tldw_chatbook import config

                    with config.locked_hooks_config_snapshot() as current:
                        if (
                            current.section_stamp != review.config.section_stamp
                            or not effects_current(handler, event, "launch")
                        ):
                            raise PermissionError("plugin_hook_authority_changed")
                        yield
                else:
                    target = captured[handler.id]
                    guard = (
                        permissions._session_end_launch_guard(
                            review, target, lambda: session_end_current(handler, event)
                        )
                        if session_end_current(handler, event)
                        else permissions.launch_guard(target, tool_name=None)
                    )
                    with guard:
                        yield

            engine = self.ensure_hooks_v2(
                session_id,
                (configured.v2_handlers if configured.enabled else ())
                + (native.definitions if native is not None else ()),
                authority,
                _native_preparation=plugin_request,
                _native_plugins=native,
                _configured_signature=signature,
                process_owner=native,
                process_owner_for=native.process_owner if native is not None else None,
                host_environment=native.host_environment
                if native is not None
                else None,
                event_projector=native.project_event if native is not None else None,
                command_launch=native.command_launch if native is not None else None,
                dependency_required=native.dependency_required
                if native is not None
                else None,
                launch_guard=launch_guard,
                effect_authority_check=effects_current,
                enabled=configured.enabled or bool(native and native.definitions),
                invalid_admissions=(
                    configured.v2_invalid_admissions if configured.enabled else ()
                ),
            )
            if test_only:
                created = getattr(self, "_plugin_test_created_engines", None)
                if created is None:
                    created = self._plugin_test_created_engines = {}
                created[native_host.asyncio.current_task()] = engine
    lifecycle = self._hooks_v2_lifecycles.get(session_id)
    if lifecycle is None:

        def current():
            try:
                return (
                    not self._disposed
                    and self.get_hooks_v2(session_id) is engine
                    and self._hooks_v2_context_key(session_id) == context_key
                    and (
                        session_id not in self._hooks_v2_configured
                        or permissions.configuration_current(review)
                        and all(
                            permissions.target_current(target, refresh=False)
                            for target in targets
                        )
                    )
                )
            except Exception:  # noqa: BLE001 -- hook boundary
                return False

        lifecycle = HookSessionLifecycle(
            engine,
            session_id,
            current=current,
            render_context=native.render_context if native is not None else None,
        )
        lifecycle.context_key = context_key
        self._hooks_v2_lifecycles[session_id] = lifecycle
        engine.lifecycle_owner = lifecycle
    pending_scope = None
    if configuration is not None:
        pending_scope = lifecycle.open_scope()
        lifecycle.turn_scope = pending_scope
        self._chat_controller._hooks_v2_submissions[
            native_host.asyncio.current_task()
        ] = (
            lifecycle,
            pending_scope,
            session_id,
        )
        try:
            if engine.mcp_executor is not None:
                context = await self._chat_controller.compose_prospective_hook_context(
                    configuration, lifecycle, pending_scope
                )
                self._require_hook_preparation_source(source)
                engine.mcp_executor.bind_context(context)
        except BaseException:
            lifecycle.close_scope(pending_scope)
            lifecycle.turn_scope = None
            raise
    if not lifecycle.live:
        start_event = lifecycle.event(
            "SessionStart", data={"reason": reason}, initiator=initiator
        )
        if test_only:
            start_event = start_event.model_copy(update={"origin": "host_plugin_test"})
        token = lifecycle.reserve(start_event)
        try:
            await lifecycle.initialize(token)
            self._require_hook_preparation_source(source)
            lifecycle.publish(token)
        except BaseException:
            lifecycle.cancel(token)
            if pending_scope is not None:
                lifecycle.close_scope(pending_scope)
                lifecycle.turn_scope = None
            # A failed provisional initialization has no live session effects.
            if self._hooks_v2_lifecycles.get(session_id) is lifecycle:
                self._hooks_v2_lifecycles.pop(session_id, None)
            raise
    if configuration is not None and engine.mcp_executor is not None:
        engine.mcp_executor.retain_runtime(session_id)
    return lifecycle


def accept_turn(
    self,
    request: native_host.ConsoleTurnCustodyRequest,
    *,
    origin: native_host.ConsoleSubmissionOrigin = native_host.ConsoleSubmissionOrigin.MANUAL,
    queue_entry_id: str | None = None,
    queue_authorization: native_host.Any | None = None,
    wake_authorization: native_host.Any | None = None,
    goal_authorization: native_host.Any | None = None,
    terminal_callback: native_host.Callable[[bool], None] | None = None,
    recover_before_acceptance: bool = True,
) -> str:
    """Synchronously retain input metadata for asynchronous physical admission."""
    with self._first_send_source_scope():
        from tldw_chatbook import config

        first_send_source = (
            config.current_config_identity(),
            getattr(self._app, "app_config", None),
            self._chat_controller,
            self._chat_store,
        )
        self._raise_if_disposed_or_session_fenced(request.session_id)
        if request.configuration.session_id != request.session_id:
            raise ValueError("Turn configuration belongs to another session.")
        store = self._chat_store
        if store is None:
            raise RuntimeError("Console chat store is unavailable.")
        pressed = request._pressed_inputs
        if pressed is not None:
            from .console_received_intent import ConsoleSessionInputSnapshot

            if (
                type(pressed) is not ConsoleSessionInputSnapshot
                or pressed._store_ref() is not store
                or pressed.session_id != request.session_id
                or not store.session_inputs_are_current(pressed, include_draft=False)
            ):
                raise RuntimeError("Pressed Console input changed before admission.")
        claim = store.claim_received_turn(
            request.session_id,
            request.turn_id,
            origin=origin,
            draft_revision=pressed.draft_revision
            if pressed is not None
            else request.draft_revision,
            _allow_draft_change=pressed is not None,
        )
        if claim is None:
            raise RuntimeError("Console session already has an admission owner.")
        attachments = ()
        try:
            self._note_received_admission_changed(store, request.session_id)
            if (
                origin is native_host.ConsoleSubmissionOrigin.MANUAL
                and self._response_rules is not None
            ):
                self._response_rules.invalidate(request.session_id, "user_input")
            record = self._register_custody(
                request,
                attachments,
                self._staged_evidence_lease_revision(
                    request.staged_evidence_launch, request.session_id
                ),
                store=store,
                received_claim=claim,
            )
        except BaseException:
            try:
                store.restore_transferred_pending_attachments(
                    request.session_id, attachments
                )
            finally:
                store.release_received_turn(claim)
            raise
        if self._archive_recovery is not None:
            self._archive_recovery.accept_user_proof(
                request.configuration.recovery_user_origin,
                request.session_id,
                request.turn_id,
            )
        coroutine = self._run_custodied_turn(
            record,
            first_send_source=first_send_source,
            origin=origin,
            queue_entry_id=queue_entry_id,
            queue_authorization=queue_authorization,
            wake_authorization=wake_authorization,
            goal_authorization=goal_authorization,
            raise_on_refusal=recover_before_acceptance,
        )
        try:
            record.task = self._create_custody_task(coroutine)
        except BaseException:
            coroutine.close()
            try:
                store.restore_transferred_pending_attachments(
                    request.session_id, attachments
                )
            finally:
                self._release_custody(record.turn_id)
            raise
        record.task.add_done_callback(
            native_host.functools.partial(
                self._finish_custodied_turn,
                turn_id=record.turn_id,
                recover_before_acceptance=recover_before_acceptance,
                terminal_callback=terminal_callback,
            )
        )
        return record.turn_id
