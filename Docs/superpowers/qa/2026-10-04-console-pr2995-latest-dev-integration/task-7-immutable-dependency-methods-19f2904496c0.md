# Immutable Task7 dependency methods
Source 19f2904496c05682ef20f5f21bd75fc172ed2bae; scoped risks: canonical rebase semantics, controller/summary preparation side effects and existing endpoint adoption/compensation. This is unchanged dependency context, not another whole-feature review.

## tldw_chatbook/UI/Screens/chat_screen.py:2810 — ChatScreen._console_settings_initial_draft

```python
    @staticmethod
    def _console_settings_initial_draft(
        settings: ConsoleSessionSettings,
        context_policy: ConsoleContextPolicyOverrides,
        *,
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState:
        """Build one process-local transaction from an exact live snapshot."""

        return ConsoleSettingsDraftState(
            settings=settings,
            context_policy_overrides=context_policy,
            field_drafts=tuple(
                ConsoleSettingsFieldDraft(
                    name=name,
                    effective_value=getattr(settings, name),
                    profile_override=getattr(settings, name),
                    provenance=ConsoleSettingsFieldProvenance.INHERITED,
                    dirty=False,
                )
                for name in sorted(exposed_fields)
            ),
            model_drafts=(),
            endpoint_draft=None,
        )
```

## tldw_chatbook/UI/Screens/chat_screen.py:9747 — ChatScreen._ensure_console_chat_controller

```python
    def _ensure_console_chat_controller(self) -> ConsoleChatController:
        """Return the native Console chat controller with fresh selection state.

        task-15860 Task 1: CONSTRUCTED by the app-owned `ConsoleRuntime`,
        with post-custody domain dependencies replaced by app/runtime-owned
        seams there. The disposable projections, wake app wiring, and core
        state sync still run here on every call.
        """
        runtime = self._console_runtime()
        if getattr(self, "_console_runtime_attachment_retired", False):
            return runtime.chat_controller
        if self._console_chat_controller is None:
            selection = self._build_console_provider_selection()
            runtime.ensure_chat_controller(
                store=self._ensure_console_chat_store(),
                provider_gateway=self._ensure_console_provider_gateway(),
                provider=selection.provider,
                model=selection.explicit_model,
                configured_model=selection.configured_model,
                base_url=selection.base_url,
                temperature=selection.temperature,
                top_p=selection.top_p,
                min_p=selection.min_p,
                top_k=selection.top_k,
                max_tokens=selection.max_tokens,
                seed=selection.seed,
                presence_penalty=selection.presence_penalty,
                frequency_penalty=selection.frequency_penalty,
                reasoning_effort=selection.reasoning_effort,
                reasoning_summary=selection.reasoning_summary,
                verbosity=selection.verbosity,
                thinking_effort=selection.thinking_effort,
                thinking_budget_tokens=selection.thinking_budget_tokens,
                streaming=selection.streaming,
                system_prompt=selection.system_prompt,
                agent_bridge=self._ensure_console_agent_bridge(),
                agent_runtime_enabled=self._console_agent_runtime_enabled(),
                skills_service=getattr(self.app_instance, "skills_scope_service", None),
                chat_dictionary_applier=self._console_chat_dictionary_applier,
                world_info_applier=self._console_world_info_applier,
                rag_capture_provider=self._retrieval._capture_console_staged_rag,
                default_session_settings=self._session._blank_console_session_settings,
                library_provider_factory=self._library_activity.build_provider,
                global_user_display_name=self._global_chat_display_name,
                turn_context_provider=(
                    self._session._build_console_turn_execution_context
                ),
                provider_config=self._provider_readiness_app_config,
            )
        # task-15860: every screen-owned slot on the controller, the store
        # and the wake coordinator is (re)bound HERE, through the single
        # enumerated `CONSOLE_VIEW_HOOK_SLOTS` list, so that the same list
        # can clear all of them at detach. This block used to assign each
        # one by hand and had no counterpart anywhere.
        generation = runtime.attach_view(
            self,
            prior_generation=getattr(
                self, "_console_runtime_attachment_generation", None
            ),
        )
        if generation is None:
            return self._console_chat_controller
        self._console_runtime_attachment_generation = generation
        # MCP batch-approval bridge (task-5): `request_mcp_approvals` runs
        # on the agent bridge's worker thread and needs a
        # `call_from_thread`-capable App handle. Deliberately NOT a
        # view-hook slot: this is the APP, which outlives every view, and
        # clearing it at detach would break the bridge a surviving turn
        # still needs.
        self._console_chat_controller.app = self.app_instance
        # PR3a-2 Task 5 (auto-wake): the app object (durable-mark clear
        # seam + marks reads). getattr-guarded because several UI tests
        # swap in hand-built controller doubles before re-running this
        # wiring block. `delivery_ui_hook` is a view-hook slot and is
        # bound by `attach_view` above.
        wake = getattr(self._console_chat_controller, "fleet_wake", None)
        if wake is not None:
            wake.wire(app=self.app_instance)
        self._sync_console_chat_core_state()
        return self._console_chat_controller
```

## tldw_chatbook/UI/Screens/chat_screen.py:8205 — ChatScreen._build_console_settings_summary_state

```python
    def _build_console_settings_summary_state(self) -> ConsoleSettingsSummaryState:
        """Build compact summary state for the active Console session settings."""
        settings, readiness = self._active_console_settings_readiness()
        estimate = self._active_console_settings_context_estimate()
        try:
            self._last_console_context_control_state = (
                self._active_console_context_control_state(estimate=estimate)
            )
        except (KeyError, ValueError):
            self._last_console_context_control_state = None
        return build_console_settings_summary_state(
            settings,
            estimate,
            readiness,
        )
```

## tldw_chatbook/UI/Screens/chat_screen.py:7967 — ChatScreen._console_settings_context_estimate_for_session

```python
    def _console_settings_context_estimate_for_session(
        self,
        session_id: str,
        *,
        settings: ConsoleSessionSettings | None = None,
    ) -> ConsoleSettingsContextEstimate:
        """Return settings context derived from one captured session only."""
        store = self._ensure_console_chat_store()
        settings = settings or store.session_settings(session_id)
        if settings is None:
            raise KeyError(session_id)
        include_active_staging = store.active_session_id == session_id
        controller = self._ensure_console_chat_controller()
        run_status = controller.run_state_for(session_id).status
        recovery = store.dispatch_recovery_for_session(session_id)
        preparation = store.preparation_for_session(session_id)
        composer = self._console_composer_or_none() if include_active_staging else None
        draft_text = composer.draft_text() if composer is not None else ""
        pending_owner = (
            self._pending_console_launch_context if include_active_staging else None
        )
        # TASK-33081: keep the one-second streaming bound, but check its key
        # before any transcript snapshots. Ordinary edits invalidate through
        # payload/display revisions; growing stream text waits at most one second.
        estimate_cache_key = (
            store,
            session_id,
            store.message_count(session_id),
            store.payload_revision(session_id),
            None
            if run_status in CONSOLE_ACTIVE_RUN_STATUSES
            else store.display_projection_revision(session_id),
            store.session_settings_revision(session_id),
            settings.provider,
            settings.model,
            settings.max_tokens,
            settings.system_prompt,
            draft_text,
            id(pending_owner),
            id(recovery),
            id(preparation),
            run_status,
            bool(controller._submit_tasks_for_session(session_id)),
        )
        now = time.monotonic()
        cached_estimate = getattr(self, "_console_estimate_cache", None)
        if (
            cached_estimate is not None
            and cached_estimate[0] == estimate_cache_key
            and now - cached_estimate[1] < CONSOLE_SETTINGS_ESTIMATE_TTL_SECONDS
        ):
            return cached_estimate[2]

        def remember(
            estimate: ConsoleSettingsContextEstimate,
        ) -> ConsoleSettingsContextEstimate:
            self._console_estimate_cache = (estimate_cache_key, now, estimate)
            # Keep immutable owners alive while their IDs are in the cache key.
            self._console_estimate_cache_owners = (pending_owner, recovery, preparation)
            return estimate

        workspace_context = (
            self._workspace._current_console_workspace_context()
            if include_active_staging
            else None
        )
        pending_launch = (
            self._pending_console_launch_context if include_active_staging else None
        )
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        greeting = ""
        composer = self._console_composer_or_none() if include_active_staging else None
        if include_active_staging:
            controller = self._ensure_console_chat_controller()
            history_key, session_messages, history = self._console_display_history(
                store, session_id, controller
            )
            staged_text = console_prompted_evidence_text(pending_launch)
            context_window = (
                self._ensure_console_provider_gateway().cached_context_window(settings)
            )
            cache_key = (
                history_key,
                settings.provider,
                settings.model,
                settings.max_tokens,
                settings.system_prompt,
                len(workspace_context.staged_sources)
                if workspace_context is not None
                else 0,
                staged_context_state.summary,
                staged_text,
                context_window,
            )
            cached_context = getattr(self, "_console_settled_context_cache", None)
            if cached_context is None or cached_context[0] != cache_key:
                greeting = controller._seeded_greeting_text(
                    session_id, session_messages
                )
                settled_messages = spend.build_console_context_messages(
                    session_messages, history.request_ids, ""
                )
                settled_estimate = build_console_context_estimate(
                    settled_messages,
                    settings.provider,
                    settings.model,
                    staged_source_count=cache_key[5],
                    staged_context_summary=staged_context_state.summary,
                    max_tokens_response=settings.max_tokens,
                    system_prompt=spend.fold_system_prompt(
                        settings.system_prompt, greeting
                    ),
                    staged_text=staged_text,
                    context_window=context_window,
                )
                self._console_settled_context_cache = (cache_key, settled_estimate)
            else:
                settled_estimate = cached_context[1]
            draft = composer.draft_text() if composer is not None else ""
            if not draft.strip() or settled_estimate.used_tokens is None:
                return remember(settled_estimate)
            estimate = build_console_context_estimate(
                [{"role": "user", "content": draft}],
                settings.provider,
                settings.model,
                staged_source_count=cache_key[5],
                staged_context_summary=staged_context_state.summary,
                max_tokens_response=settings.max_tokens,
                context_window=context_window,
                history_used_tokens=settled_estimate.used_tokens,
            )
            return remember(estimate)
        else:
            try:
                session_messages = store.messages_for_session(session_id)
            except KeyError:
                session_messages = []
            messages = spend.build_console_context_messages(session_messages, None, "")
        estimate = build_console_context_estimate(
            messages,
            settings.provider,
            settings.model,
            staged_source_count=(
                len(workspace_context.staged_sources)
                if workspace_context is not None
                else 0
            ),
            staged_context_summary=staged_context_state.summary,
            max_tokens_response=settings.max_tokens,
            system_prompt=spend.fold_system_prompt(settings.system_prompt, greeting),
            # task-6: staged evidence used to move only the label's "; N
            # sources staged" suffix (`staged_source_count` above) while
            # `used_tokens` silently reported zero for content the send
            # is likely to carry. `console_prompted_evidence_text` reads
            # the same in-memory, zero-I/O staged bundle and produces the
            # formatted pre-authority estimate
            # `_current_console_workspace_context` already parses above --
            # no extra DB round trip. The actual send may shrink this after
            # its authority check.
            staged_text=console_prompted_evidence_text(pending_launch),
            context_window=self._ensure_console_provider_gateway().cached_context_window(
                settings
            ),
        )
        return remember(estimate)
```

## tldw_chatbook/Chat/console_chat_controller.py:14602 — ConsoleChatController.rebase_console_settings_draft

```python
    def rebase_console_settings_draft(
        self,
        state: ConsoleSettingsDraftState,
        *,
        provider: str,
        model: str | None,
        app_config: Mapping[str, object],
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState:
        """Rebase one settings draft onto an exact provider/model target.

        The current or remembered exact target retains its conversation snapshot,
        including fields hidden by the calling surface. An unseen target starts
        from its established default chain and carries only supported dirty
        fields. Explicit Inherit edits resolve current lower-precedence defaults.

        Args:
            state: The draft being switched away from; never mutated.
            provider: Exact target provider IDENTITY. Registry entries use
                the dashed ``custom-ep:<slug>`` spelling (config-table keys
                such as ``custom_ep:<slug>`` are canonicalized back to it);
                every other provider is a plain config key.
            model: Literal target model ID, or ``None`` to take the target
                provider's default model.
            app_config: The live application configuration snapshot the
                target's default chain resolves against.
            exposed_fields: Exact field names the calling surface can carry;
                only exposed dirty fields carry to an unseen target.

        Returns:
            A new ``ConsoleSettingsDraftState`` rebased onto the target:
            ``settings.provider`` carries the target's canonical identity
            spelling, remembered drafts stay keyed by provider identity, and
            the input ``state`` is left untouched.
        """

        target_defaults = build_target_default_console_session_settings(
            app_config,
            provider,
            model,
        )
        target_provider = provider_config_key(target_defaults.provider)
        # CE-001: ``settings.provider`` and the remembered-draft keys are
        # provider IDENTITY values, not config-table lookup keys -- a dashed
        # ``custom-ep:<slug>`` id canonicalized here would arrive at the
        # provider Select as an illegal underscored value and crash the app.
        # Registry ids keep their dashed spelling; config-key lookups below
        # still use ``target_provider``.
        target_provider_id = provider_identity_key(target_defaults.provider)
        target_model = normalize_console_model_value(target_defaults.model)
        target_key = (target_provider_id, target_model)
        current_key = (
            provider_identity_key(state.settings.provider),
            normalize_console_model_value(state.settings.model),
        )
        remembered_target = next(
            (
                draft
                for draft in state.model_drafts
                if (draft.provider, draft.model) == target_key
            ),
            None,
        )
        restoring_remembered_target = (
            current_key != target_key and remembered_target is not None
        )
        preserve_snapshot = current_key == target_key or restoring_remembered_target
        source_settings = (
            remembered_target.settings
            if restoring_remembered_target
            else state.settings
        )
        source_fields = (
            remembered_target.field_drafts
            if restoring_remembered_target
            else state.field_drafts
        )
        source_endpoint = (
            remembered_target.endpoint_draft
            if restoring_remembered_target
            else state.endpoint_draft
        )

        quick_surface = exposed_fields == QUICK_MODEL_DEFAULT_FIELDS
        inherited_dirty_fields = (
            frozenset(
                field.name
                for field in source_fields
                if field.dirty
                and field.profile_override is None
                and field.name in exposed_fields
            )
            if not quick_surface
            else frozenset()
        )
        if inherited_dirty_fields:
            target_defaults = build_target_default_console_session_settings(
                app_config,
                # Identity spelling: the dashed registry id must resolve
                # through entry_for, which the mangled key does not for
                # hyphenated slugs (CE-001).
                target_provider_id,
                target_model,
                excluded_model_profile_fields=inherited_dirty_fields,
            )
        settings_base = source_settings if preserve_snapshot else target_defaults

        supported_fields = supported_generation_fields(
            target_provider_id, target_model, app_config
        )
        exposed_supported_fields = exposed_fields & supported_fields
        profile = normalized_console_model_profile_overrides(
            app_config,
            target_provider,
            target_model,
        )
        rebased_fields: dict[str, ConsoleSettingsFieldDraft] = {}
        for name in _CONSOLE_SETTINGS_FIELD_ORDER:
            if name not in exposed_supported_fields:
                continue
            effective_value = getattr(settings_base, name)
            has_profile_override = name in profile
            rebased_fields[name] = ConsoleSettingsFieldDraft(
                name=name,
                effective_value=effective_value,
                profile_override=(
                    effective_value
                    if quick_surface
                    else profile.get(name)
                    if has_profile_override
                    else None
                ),
                provenance=(
                    ConsoleSettingsFieldProvenance.EXPLICIT
                    if has_profile_override
                    else ConsoleSettingsFieldProvenance.INHERITED
                ),
                dirty=False,
            )

        field_values: dict[str, object | None] = {}
        carrying_to_unseen_target = (
            current_key != target_key and remembered_target is None
        )
        for source_field in source_fields:
            if source_field.name not in exposed_supported_fields or (
                not preserve_snapshot and not source_field.dirty
            ):
                continue
            if quick_surface and source_field.effective_value is None:
                effective_value, profile_override = quick_blank_field_default(
                    app_config, target_defaults, source_field.name
                )
                field_values[source_field.name] = effective_value
                rebased_fields[source_field.name] = replace(
                    rebased_fields[source_field.name],
                    effective_value=effective_value,
                    profile_override=profile_override,
                )
                continue
            inherits_target_default = (
                source_field.dirty
                and not quick_surface
                and source_field.profile_override is None
            )
            effective_value = (
                getattr(target_defaults, source_field.name)
                if inherits_target_default
                else source_field.effective_value
            )
            field_values[source_field.name] = effective_value
            rebased_fields[source_field.name] = replace(
                source_field,
                effective_value=effective_value,
                profile_override=(
                    effective_value if quick_surface else source_field.profile_override
                ),
                provenance=(
                    ConsoleSettingsFieldProvenance.CARRIED
                    if carrying_to_unseen_target
                    else source_field.provenance
                ),
                dirty=source_field.dirty,
            )

        unsupported_provider_fields = FULL_MODEL_DEFAULT_FIELDS - supported_fields
        settings_changes: dict[str, object | None] = {
            "provider": target_provider_id,
            "model": target_model,
            "character_label": state.settings.character_label,
            "system_prompt": state.settings.system_prompt,
            "source": state.settings.source,
            "pinned_prefill": state.settings.pinned_prefill,
            **{name: None for name in unsupported_provider_fields},
            **field_values,
        }

        endpoint_draft: ConsoleEndpointDraft | None = None
        target_base_url = target_defaults.base_url
        if preserve_snapshot and (
            source_endpoint is None
            or (
                not source_endpoint.dirty
                and source_endpoint.bound_provider_config_key
                == provider_config_key(target_provider)
            )
        ):
            target_base_url = source_settings.base_url
        if (
            exposed_fields == FULL_MODEL_DEFAULT_FIELDS
            and source_endpoint is not None
            and source_endpoint.dirty
            and source_endpoint.bound_provider_config_key
            == provider_config_key(target_provider)
        ):
            endpoint_draft = source_endpoint
            target_base_url = source_endpoint.value or None
        elif target_base_url is not None and not (
            preserve_snapshot and source_endpoint is None
        ):
            endpoint_draft = ConsoleEndpointDraft(
                value=target_base_url,
                bound_provider_config_key=provider_config_key(target_provider),
                dirty=False,
                checked=False,
            )
        settings_changes["base_url"] = target_base_url

        return replace(
            state,
            settings=replace(settings_base, **settings_changes),
            field_drafts=tuple(
                rebased_fields[name]
                for name in _CONSOLE_SETTINGS_FIELD_ORDER
                if name in rebased_fields
            ),
            endpoint_draft=endpoint_draft,
        )
```

## tldw_chatbook/Chat/console_chat_store.py:9605 — ConsoleChatStore.adopt_session_ephemeral_endpoint

```python
    def adopt_session_ephemeral_endpoint(
        self,
        session_id: str,
        *,
        settings: ConsoleSessionSettings,
        policy: ConsoleEphemeralEndpointPolicy,
    ) -> ConsoleEndpointAdoptionReceipt | None:
        """Publish one verified live endpoint without serializing that endpoint.

        Existing conversations persist the endpoint-safe ``settings`` first.
        Only after that durable write returns is the raw settings snapshot and
        live-only endpoint policy published together in the session.
        """

        if not isinstance(settings, ConsoleSessionSettings) or not isinstance(
            policy, ConsoleEphemeralEndpointPolicy
        ):
            raise TypeError("Exact Console endpoint adoption state is required")
        if policy.state is not ConsoleEndpointPolicyState.ACTIVE:
            raise ValueError("A new Console endpoint policy must be active")
        if settings.provider != policy.provider or settings.model != policy.model:
            raise ValueError("Console endpoint policy must match durable settings")

        with self._fork_source_transition(session_id):
            session = self._session_or_raise(session_id)
            receipt: ConsoleEndpointAdoptionReceipt | None = None
            if session.persisted_conversation_id is not None:
                persistence = self.persistence
                writer = getattr(
                    persistence,
                    "adopt_console_session_endpoint_settings",
                    None,
                )
                if not callable(writer):
                    raise RuntimeError(
                        "Console persistence cannot safely adopt a live endpoint"
                    )
                receipt = writer(
                    conversation_id=session.persisted_conversation_id,
                    settings=settings,
                )
                if not isinstance(receipt, ConsoleEndpointAdoptionReceipt):
                    raise RuntimeError("Console endpoint adoption receipt is invalid")

            changed = (
                session.settings != settings
                or session.ephemeral_endpoint_policy != policy
            )
            if changed:
                session.has_user_work = True
            session.settings = settings
            session.ephemeral_endpoint_policy = policy
            session.generation_settings_revision += 1
            session.settings_persistence_failures.pop(
                ConsoleSettingsComponent.GENERATION_SETTINGS,
                None,
            )
            self._bump_payload_revision(session_id)
            if changed:
                self._bump_settings_revision(session_id)
            return receipt
```

## tldw_chatbook/Chat/console_chat_store.py:9667 — ConsoleChatStore.rollback_session_ephemeral_endpoint_adoption

```python
    def rollback_session_ephemeral_endpoint_adoption(
        self,
        session_id: str,
        *,
        expected_settings: ConsoleSessionSettings,
        expected_policy: ConsoleEphemeralEndpointPolicy,
        prior_settings: ConsoleSessionSettings,
        prior_policy: ConsoleEphemeralEndpointPolicy | None,
        prior_has_user_work: bool,
        receipt: ConsoleEndpointAdoptionReceipt | None,
    ) -> ConsoleEndpointRollbackOutcome:
        """Restore exact adoption state or block the endpoint on durable failure."""

        if (
            not isinstance(expected_settings, ConsoleSessionSettings)
            or not isinstance(expected_policy, ConsoleEphemeralEndpointPolicy)
            or not isinstance(prior_settings, ConsoleSessionSettings)
            or (
                prior_policy is not None
                and not isinstance(prior_policy, ConsoleEphemeralEndpointPolicy)
            )
            or type(prior_has_user_work) is not bool
            or (
                receipt is not None
                and not isinstance(receipt, ConsoleEndpointAdoptionReceipt)
            )
        ):
            raise TypeError("Exact Console endpoint rollback state is required")

        with self._fork_source_transition(session_id):
            session = self._session_or_raise(session_id)
            if (
                session.settings != expected_settings
                or session.ephemeral_endpoint_policy != expected_policy
            ):
                return ConsoleEndpointRollbackOutcome.LOST_SESSION_FENCE

            if receipt is not None:
                persistence = self.persistence
                rollback = getattr(
                    persistence,
                    "rollback_console_session_endpoint_adoption",
                    None,
                )
                try:
                    restored = callable(rollback) and rollback(receipt=receipt)
                except Exception:
                    restored = False
                    logger.bind(
                        session_id=session_id,
                        conversation_id=receipt.conversation_id,
                    ).exception(
                        "Failed to restore durable Console metadata after "
                        "endpoint adoption."
                    )
                if not restored:
                    session.ephemeral_endpoint_policy = replace(
                        expected_policy,
                        state=ConsoleEndpointPolicyState.BLOCKED,
                    )
                    self._bump_payload_revision(session_id)
                    self._bump_settings_revision(session_id)
                    return ConsoleEndpointRollbackOutcome.BLOCKED_DURABLE_RESTORE

            session.settings = prior_settings
            session.ephemeral_endpoint_policy = prior_policy
            session.has_user_work = prior_has_user_work
            session.generation_settings_revision += 1
            self._bump_payload_revision(session_id)
            self._bump_settings_revision(session_id)
            return ConsoleEndpointRollbackOutcome.RESTORED
```
