"""Settings persistence drains for the existing Console store.

The caller retains identity, state, and public entry points. Original module
bindings are read live through host, including after awaits and in callbacks.
"""

from __future__ import annotations

from . import console_chat_store as host


async def persist_console_settings_commit_serialized(
    self,
    commit: host.ConsoleSettingsLiveCommit,
    *,
    policy_failure_label: host.ConsoleSettingsPolicyFailureLabel | None = None,
) -> host.ConsoleSettingsPersistenceOutcome:
    """Join one session drain that converges durability to newest live state."""
    if not isinstance(commit, host.ConsoleSettingsLiveCommit):
        raise TypeError("commit must be ConsoleSettingsLiveCommit")
    if policy_failure_label is None:
        accepted = commit.accepted_submission
        policy_failure_label = self._console_settings_policy_failure_label(
            accepted.surface if accepted is not None else None
        )
    if not isinstance(policy_failure_label, host.ConsoleSettingsPolicyFailureLabel):
        raise TypeError(
            "policy_failure_label must be ConsoleSettingsPolicyFailureLabel"
        )
    return await self._join_console_settings_persistence_drain(
        session_id=commit.session_id,
        persisted_conversation_id=commit.persisted_conversation_id,
        conversation_binding_revision=commit.conversation_binding_revision,
        generation_revision=commit.generation_revision,
        context_policy_revision=commit.context_policy_revision,
        generation_snapshot=host.snapshot_from_session_settings(commit.settings),
        context_policy_overrides=commit.context_policy_overrides,
        policy_failure_label=policy_failure_label,
        components=frozenset(
            {
                host.ConsoleSettingsComponent.GENERATION_SETTINGS,
                host.ConsoleSettingsComponent.CONTEXT_POLICY,
            }
        ),
    )


def _console_settings_policy_failure_label(
    surface: host.ConsoleSettingsSurface | None,
) -> host.ConsoleSettingsPolicyFailureLabel:
    """Map an exact typed submission surface to durable failure copy."""
    if surface is host.ConsoleSettingsSurface.QUICK_POPOVER:
        return host.ConsoleSettingsPolicyFailureLabel.COMPACTION
    return host.ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS


async def retry_console_settings_persistence(
    self,
    *,
    session_id: str,
    component: host.ConsoleSettingsComponent,
    revision: int,
    _expected_failure: host.ConsoleSettingsPersistenceFailure | None = None,
) -> bool:
    """Retry one still-current failed component without advancing revisions."""
    if not isinstance(component, host.ConsoleSettingsComponent):
        raise TypeError("component must be ConsoleSettingsComponent")
    session = self._sessions.get(session_id)
    failure = (
        session.settings_persistence_failures.get(component)
        if session is not None
        else None
    )
    if component is host.ConsoleSettingsComponent.PROMPT_MIXINS:
        if _expected_failure is not None and failure is not _expected_failure:
            return False
        preferences = (
            session.prompt_mixin_preferences.preferences
            if session is not None
            else None
        )
        if (
            failure is None
            or preferences is None
            or preferences.revision != revision
            or failure.revision != revision
            or session.persisted_conversation_id != failure.persisted_conversation_id
            or session.conversation_binding_revision
            != failure.conversation_binding_revision
        ):
            return False
        outcome = await self.persist_prompt_mixin_preferences(session_id)
        return component in outcome.written_components
    current_revision = (
        session.generation_settings_revision
        if session is not None
        and component is host.ConsoleSettingsComponent.GENERATION_SETTINGS
        else session.context_policy_revision
        if session is not None
        else None
    )
    if (
        session is None
        or failure is None
        or failure.revision != revision
        or current_revision != revision
    ):
        return False
    outcome = await self._join_console_settings_persistence_drain(
        session_id=session_id,
        persisted_conversation_id=failure.persisted_conversation_id,
        conversation_binding_revision=failure.conversation_binding_revision,
        generation_revision=(
            revision
            if component is host.ConsoleSettingsComponent.GENERATION_SETTINGS
            else session.generation_settings_revision
        ),
        context_policy_revision=(
            revision
            if component is host.ConsoleSettingsComponent.CONTEXT_POLICY
            else session.context_policy_revision
        ),
        generation_snapshot=failure.generation_snapshot,
        context_policy_overrides=failure.context_policy_overrides,
        policy_failure_label=(
            failure.policy_failure_label
            or host.ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS
        ),
        components=frozenset({component}),
        retry_components=frozenset({component}),
    )
    return component in outcome.written_components


async def _join_console_settings_persistence_drain(
    self,
    *,
    session_id: str,
    persisted_conversation_id: str | None,
    conversation_binding_revision: int,
    generation_revision: int,
    context_policy_revision: int,
    generation_snapshot: host.ConsoleGenerationSettingsSnapshot | None,
    context_policy_overrides: host.ConsoleContextPolicyOverrides | None,
    policy_failure_label: host.ConsoleSettingsPolicyFailureLabel,
    components: frozenset[host.ConsoleSettingsComponent],
    retry_components: frozenset[host.ConsoleSettingsComponent] = frozenset(),
) -> host.ConsoleSettingsPersistenceOutcome:
    """Start or join the one in-flight durability drain for a live slot."""
    if not isinstance(policy_failure_label, host.ConsoleSettingsPolicyFailureLabel):
        raise TypeError(
            "policy_failure_label must be ConsoleSettingsPolicyFailureLabel"
        )
    lifecycle = self._settings_persistence_lifecycles.setdefault(
        session_id,
        host._ConsoleSettingsPersistenceLifecycle(),
    )
    incarnation = self._settings_session_incarnations.get(session_id, 0)
    drain = lifecycle.drain
    if drain is not None and drain.task is not None and drain.task.done():
        lifecycle.drain = None
        drain = None
    if (
        drain is not None
        and drain.session_incarnation == incarnation
        and drain.persisted_conversation_id == persisted_conversation_id
        and drain.conversation_binding_revision == conversation_binding_revision
    ):
        drain.requested_components.update(components)
        drain.retry_components.update(retry_components)
        drain.initial_components = drain.initial_components | components
        if (
            host.ConsoleSettingsComponent.GENERATION_SETTINGS in components
            and generation_revision >= drain.generation_revision
        ):
            drain.generation_revision = generation_revision
            drain.generation_snapshot = generation_snapshot
        if (
            host.ConsoleSettingsComponent.CONTEXT_POLICY in components
            and context_policy_revision >= drain.context_policy_revision
        ):
            drain.context_policy_revision = context_policy_revision
            drain.context_policy_overrides = context_policy_overrides
            drain.policy_failure_label = policy_failure_label
        assert drain.task is not None
        return await host.asyncio.shield(drain.task)

    drain = host._ConsoleSettingsPersistenceDrain(
        session_incarnation=incarnation,
        persisted_conversation_id=persisted_conversation_id,
        conversation_binding_revision=conversation_binding_revision,
        generation_revision=generation_revision,
        context_policy_revision=context_policy_revision,
        generation_snapshot=generation_snapshot,
        context_policy_overrides=context_policy_overrides,
        policy_failure_label=policy_failure_label,
        initial_components=components,
        requested_components=set(components),
        retry_components=set(retry_components),
    )
    task = host.asyncio.create_task(
        self._run_console_settings_persistence_drain(
            session_id,
            drain,
            lifecycle,
        )
    )
    drain.task = task
    lifecycle.drain = drain
    lifecycle.tasks.add(task)
    task.add_done_callback(
        lambda completed: self._retire_console_settings_persistence_drain(
            session_id,
            drain,
            lifecycle,
            completed,
        )
    )
    return await host.asyncio.shield(task)


def _retire_console_settings_persistence_drain(
    self,
    session_id: str,
    drain: host._ConsoleSettingsPersistenceDrain,
    lifecycle: host._ConsoleSettingsPersistenceLifecycle,
    task: host.asyncio.Task[host.ConsoleSettingsPersistenceOutcome],
) -> None:
    """Forget a completed drain only when it still owns the live registry."""
    lifecycle.tasks.discard(task)
    if drain.task is task and lifecycle.drain is drain:
        lifecycle.drain = None
    if self._settings_persistence_lifecycles.get(session_id) is lifecycle:
        self._cleanup_console_settings_lifecycle_if_idle(session_id)


async def _run_console_settings_persistence_drain(
    self,
    session_id: str,
    drain: host._ConsoleSettingsPersistenceDrain,
    lifecycle: host._ConsoleSettingsPersistenceLifecycle,
) -> host.ConsoleSettingsPersistenceOutcome:
    async with lifecycle.lock:
        return await self._persist_console_settings_components_locked(
            session_id=session_id,
            drain=drain,
            lifecycle=lifecycle,
        )


async def _persist_console_settings_components_locked(
    self,
    *,
    session_id: str,
    drain: host._ConsoleSettingsPersistenceDrain,
    lifecycle: host._ConsoleSettingsPersistenceLifecycle,
) -> host.ConsoleSettingsPersistenceOutcome:
    """Drain selected components to their newest live revisions.

    A transaction already executing in ``to_thread`` cannot be recalled. If
    Apply advances while that transaction is in flight, its successful CAS
    result becomes only the private base for the next CAS. It is never
    published into the session, and this coroutine keeps reconciling until
    the newest live revision is durable (or fails) before any joined waiter
    returns.
    """
    written: set[host.ConsoleSettingsComponent] = set()
    failed: set[host.ConsoleSettingsComponent] = set()
    stale: set[host.ConsoleSettingsComponent] = set()
    session = self._sessions.get(session_id)
    if not self._console_settings_drain_identity_matches(
        session_id,
        session,
        drain,
    ):
        return host.ConsoleSettingsPersistenceOutcome(
            session_id=session_id,
            stale_components=frozenset(drain.requested_components),
        )
    assert session is not None
    if session.ephemeral or drain.persisted_conversation_id is None:
        if session.context_policy_revision == drain.context_policy_revision:
            session.staged_context_policy_failure_label = drain.policy_failure_label
            session.staged_context_policy_failure_revision = (
                drain.context_policy_revision
            )
        return host.ConsoleSettingsPersistenceOutcome(
            session_id=session_id,
            staged=True,
        )

    excluded: set[host.ConsoleSettingsComponent] = set()
    if (
        host.ConsoleSettingsComponent.GENERATION_SETTINGS in drain.initial_components
        and (
            session.generation_settings_revision != drain.generation_revision
            or drain.generation_snapshot is None
        )
    ):
        excluded.add(host.ConsoleSettingsComponent.GENERATION_SETTINGS)
    if host.ConsoleSettingsComponent.CONTEXT_POLICY in drain.initial_components and (
        session.context_policy_revision != drain.context_policy_revision
        or drain.context_policy_overrides is None
    ):
        excluded.add(host.ConsoleSettingsComponent.CONTEXT_POLICY)
    stale.update(excluded)

    conversation_id = drain.persisted_conversation_id
    assert conversation_id is not None
    private_generation_base = (
        lifecycle.generation_bases[conversation_id]
        if conversation_id in lifecycle.generation_bases
        else session.generation_durable_snapshot
    )
    private_context_base = (
        lifecycle.context_bases[conversation_id]
        if conversation_id in lifecycle.context_bases
        else session.context_policy_durable_revision
    )
    persisted_revisions = dict(lifecycle.component_revisions)
    failed_revisions: dict[host.ConsoleSettingsComponent, int] = {}
    mixin_failed_value = None

    while True:
        current = self._sessions.get(session_id)
        if not self._console_settings_drain_identity_matches(
            session_id,
            current,
            drain,
        ):
            stale.update(drain.requested_components)
            break
        assert current is not None
        requested = set(drain.requested_components)
        for retried_component in tuple(drain.retry_components):
            failed_revisions.pop(retried_component, None)
        drain.retry_components.clear()
        generation_needs_reconcile = False

        generation = host.ConsoleSettingsComponent.GENERATION_SETTINGS
        if generation in requested and generation not in excluded:
            revision = current.generation_settings_revision
            if persisted_revisions.get(generation) == revision:
                written.add(generation)
                failed.discard(generation)
                stale.discard(generation)
            elif failed_revisions.get(generation) == revision:
                written.discard(generation)
                failed.add(generation)
            elif current.settings is None:
                written.discard(generation)
                stale.add(generation)
                excluded.add(generation)
            else:
                written.discard(generation)
                snapshot = host.snapshot_from_session_settings(current.settings)
                writer = getattr(
                    self.persistence,
                    "update_conversation_generation_settings",
                    None,
                )
                try:
                    result = (
                        await host.asyncio.to_thread(
                            writer,
                            conversation_id=drain.persisted_conversation_id,
                            snapshot=snapshot,
                            expected_snapshot=private_generation_base,
                        )
                        if callable(writer)
                        else object()
                    )
                except Exception:
                    result = object()
                current = self._sessions.get(session_id)
                identity_matches = self._console_settings_drain_identity_matches(
                    session_id,
                    current,
                    drain,
                )
                generation_written = bool(
                    isinstance(result, host.ConsoleGenerationSettingsWriteResult)
                    and result.status
                    is host.ConsoleGenerationSettingsWriteStatus.WRITTEN
                )
                if generation_written:
                    assert isinstance(
                        result,
                        host.ConsoleGenerationSettingsWriteResult,
                    )
                    private_generation_base = result.snapshot
                    lifecycle.generation_bases[conversation_id] = result.snapshot
                    persisted_revisions[generation] = revision
                    if (
                        identity_matches
                        and current is not None
                        and current.generation_settings_revision == revision
                    ):
                        current.generation_durable_snapshot = result.snapshot
                        lifecycle.component_revisions[generation] = revision
                        current.settings_persistence_failures.pop(generation, None)
                        failed_revisions.pop(generation, None)
                        failed.discard(generation)
                        stale.discard(generation)
                        written.add(generation)
                    else:
                        stale.add(generation)
                        generation_needs_reconcile = True
                elif (
                    identity_matches
                    and current is not None
                    and current.generation_settings_revision == revision
                ):
                    self._record_console_settings_failure(
                        current,
                        component=generation,
                        revision=revision,
                        generation_snapshot=snapshot,
                    )
                    failed_revisions[generation] = revision
                    failed.add(generation)
                    stale.discard(generation)
                else:
                    stale.add(generation)
                    generation_needs_reconcile = True

        if generation_needs_reconcile:
            continue

        context = host.ConsoleSettingsComponent.CONTEXT_POLICY
        current = self._sessions.get(session_id)
        if (
            current is not None
            and context in lifecycle.component_revisions
            and conversation_id in lifecycle.context_bases
            and self._console_settings_drain_identity_matches(
                session_id,
                current,
                drain,
            )
        ):
            published_context_revision = lifecycle.component_revisions[context]
            if published_context_revision == current.context_policy_revision:
                persisted_revisions[context] = published_context_revision
                private_context_base = lifecycle.context_bases[conversation_id]
        if (
            context in requested
            and context not in excluded
            and self._console_settings_drain_identity_matches(
                session_id,
                current,
                drain,
            )
        ):
            assert current is not None
            revision = current.context_policy_revision
            if persisted_revisions.get(context) == revision:
                written.add(context)
                failed.discard(context)
                stale.discard(context)
            elif failed_revisions.get(context) == revision:
                written.discard(context)
                failed.add(context)
            else:
                written.discard(context)
                overrides = current.context_policy_overrides
                writer = getattr(
                    self.persistence,
                    "update_conversation_context_policy",
                    None,
                )
                try:
                    result = (
                        await host.asyncio.to_thread(
                            writer,
                            conversation_id=drain.persisted_conversation_id,
                            overrides=overrides,
                            expected_revision=private_context_base,
                        )
                        if callable(writer)
                        else object()
                    )
                except Exception:
                    result = object()
                current = self._sessions.get(session_id)
                identity_matches = self._console_settings_drain_identity_matches(
                    session_id,
                    current,
                    drain,
                )
                context_written, durable_revision = self._context_write_result(result)
                if context_written:
                    private_context_base = durable_revision
                    lifecycle.context_bases[conversation_id] = durable_revision
                    persisted_revisions[context] = revision
                    if (
                        identity_matches
                        and current is not None
                        and current.context_policy_revision == revision
                    ):
                        current.context_policy_durable_revision = durable_revision
                        lifecycle.component_revisions[context] = revision
                        current.settings_persistence_failures.pop(context, None)
                        if current.staged_context_policy_failure_revision == revision:
                            current.staged_context_policy_failure_label = None
                            current.staged_context_policy_failure_revision = None
                        failed_revisions.pop(context, None)
                        failed.discard(context)
                        stale.discard(context)
                        written.add(context)
                    else:
                        stale.add(context)
                elif (
                    identity_matches
                    and current is not None
                    and current.context_policy_revision == revision
                ):
                    self._record_console_settings_failure(
                        current,
                        component=context,
                        revision=revision,
                        context_policy_overrides=overrides,
                        policy_failure_label=drain.policy_failure_label,
                    )
                    failed_revisions[context] = revision
                    failed.add(context)
                    stale.discard(context)
                else:
                    stale.add(context)

        mixins = host.ConsoleSettingsComponent.PROMPT_MIXINS
        current = self._sessions.get(session_id)
        if (
            mixins in requested
            and mixins not in excluded
            and self._console_settings_drain_identity_matches(
                session_id, current, drain
            )
        ):
            prefs = current.prompt_mixin_preferences.preferences
            if prefs is None:
                excluded.add(mixins)
                stale.add(mixins)
            elif lifecycle.mixin_values.get(conversation_id) == prefs:
                written.add(mixins)
            elif (
                mixin_failed_value == prefs
                and failed_revisions.get(mixins) == prefs.revision
            ):
                failed.add(mixins)
            else:
                revision = prefs.revision
                base = lifecycle.mixin_bases.get(
                    conversation_id, current.prompt_mixin_durable_revision
                )
                try:
                    result = await host.asyncio.to_thread(
                        self.persistence.save_conversation_mixin_preferences,
                        conversation_id=conversation_id,
                        expected_revision=base,
                        value=prefs,
                    )
                    if type(result) is not host.MixinPreferences:
                        raise ValueError("mixin_preference_save_failed")
                except Exception:  # noqa: BLE001 - failed persistence retains exact retry state.
                    result = None
                current = self._sessions.get(session_id)
                same = self._console_settings_drain_identity_matches(
                    session_id, current, drain
                )
                if result is not None:
                    lifecycle.mixin_bases[conversation_id] = result.revision
                    lifecycle.mixin_values[conversation_id] = prefs
                    persisted_revisions[mixins] = revision
                    if same and current.prompt_mixin_preferences.preferences == prefs:
                        current.prompt_mixin_durable_revision = result.revision
                        lifecycle.component_revisions[mixins] = revision
                        current.settings_persistence_failures.pop(mixins, None)
                        written.add(mixins)
                        failed.discard(mixins)
                elif same and current.prompt_mixin_preferences.preferences == prefs:
                    current.settings_persistence_failures[mixins] = (
                        host.ConsoleSettingsPersistenceFailure(
                            mixins,
                            revision,
                            conversation_id,
                            current.conversation_binding_revision,
                            None,
                            mixin_preferences=prefs,
                        )
                    )
                    failed_revisions[mixins] = revision
                    mixin_failed_value = prefs
                    failed.add(mixins)
                    written.discard(mixins)
        current = self._sessions.get(session_id)
        if not self._console_settings_drain_identity_matches(
            session_id,
            current,
            drain,
        ):
            stale.update(drain.requested_components)
            break
        assert current is not None
        converged = True
        for component in set(drain.requested_components) - excluded:
            revision = (
                current.generation_settings_revision
                if component is host.ConsoleSettingsComponent.GENERATION_SETTINGS
                else current.prompt_mixin_preferences.preferences.revision
                if component is host.ConsoleSettingsComponent.PROMPT_MIXINS
                else current.context_policy_revision
            )
            if component is host.ConsoleSettingsComponent.PROMPT_MIXINS:
                prefs = current.prompt_mixin_preferences.preferences
                if (
                    lifecycle.mixin_values.get(conversation_id) != prefs
                    and mixin_failed_value != prefs
                ):
                    converged = False
                    break
            elif (
                persisted_revisions.get(component) != revision
                and failed_revisions.get(component) != revision
            ):
                converged = False
                break
        if converged:
            break

    return host.ConsoleSettingsPersistenceOutcome(
        session_id=session_id,
        written_components=frozenset(written),
        failed_components=frozenset(failed),
        stale_components=frozenset(stale),
    )


def _context_write_result(result: object) -> tuple[bool, int | None]:
    if isinstance(result, host.ContextPolicyWriteResult):
        return (
            result.status is host.ContextPolicyWriteStatus.WRITTEN,
            result.revision,
        )
    if type(result) is int or result is None:
        return True, result
    return False, None


def _console_settings_drain_identity_matches(
    self,
    session_id: str,
    session: host.ConsoleChatSession | None,
    drain: host._ConsoleSettingsPersistenceDrain,
) -> bool:
    """Require both scalar origin identity and store-private incarnation."""
    return bool(
        self._settings_session_incarnations.get(session_id) == drain.session_incarnation
        and self._console_settings_identity_matches(
            session,
            persisted_conversation_id=drain.persisted_conversation_id,
            conversation_binding_revision=drain.conversation_binding_revision,
        )
    )


def _console_settings_identity_matches(
    session: host.ConsoleChatSession | None,
    *,
    persisted_conversation_id: str | None,
    conversation_binding_revision: int,
) -> bool:
    return bool(
        session is not None
        and session.persisted_conversation_id == persisted_conversation_id
        and session.conversation_binding_revision == conversation_binding_revision
    )


def _record_console_settings_failure(
    session: host.ConsoleChatSession,
    *,
    component: host.ConsoleSettingsComponent,
    revision: int,
    generation_snapshot: host.ConsoleGenerationSettingsSnapshot | None = None,
    context_policy_overrides: host.ConsoleContextPolicyOverrides | None = None,
    policy_failure_label: host.ConsoleSettingsPolicyFailureLabel | None = None,
) -> None:
    conversation_id = session.persisted_conversation_id
    if conversation_id is None:
        return
    session.settings_persistence_failures[component] = (
        host.ConsoleSettingsPersistenceFailure(
            component=component,
            revision=revision,
            persisted_conversation_id=conversation_id,
            conversation_binding_revision=session.conversation_binding_revision,
            generation_snapshot=generation_snapshot,
            context_policy_overrides=context_policy_overrides,
            policy_failure_label=policy_failure_label,
        )
    )


def schedule_prompt_mixin_preferences_save(
    self, session_id: str, *, owner=None, lease=None
) -> None:
    """Retain an exact-origin save in the existing settings lifecycle."""
    session = self._session_or_raise(session_id)
    if session.ephemeral or session.persisted_conversation_id is None:
        if lease is not None:
            lease.release()
        return
    if owner is None:
        preferences = session.prompt_mixin_preferences.preferences
        if preferences is not None:
            session.settings_persistence_failures[
                host.ConsoleSettingsComponent.PROMPT_MIXINS
            ] = host.ConsoleSettingsPersistenceFailure(
                host.ConsoleSettingsComponent.PROMPT_MIXINS,
                preferences.revision,
                session.persisted_conversation_id,
                session.conversation_binding_revision,
                None,
                mixin_preferences=preferences,
            )
        return
    lifecycle = self._settings_persistence_lifecycles.setdefault(
        session_id, host._ConsoleSettingsPersistenceLifecycle()
    )
    if lifecycle.mixin_save_task is not None and not lifecycle.mixin_save_task.done():
        if lease is not None:
            lease.release()
        return
    lease = lease or owner.try_acquire()
    if lease is None:
        return
    task = owner.launch(
        lease,
        self.persist_prompt_mixin_preferences(
            session_id,
            _session=session,
            _origin=(
                session.incarnation_id,
                session.persisted_conversation_id,
                session.conversation_binding_revision,
            ),
        ),
        name=f"console-mixin-preferences-{session_id}",
    )
    lifecycle.mixin_save_task = task
    lifecycle.tasks.add(task)
    task.add_done_callback(lifecycle.tasks.discard)


async def persist_prompt_mixin_preferences(
    self, session_id: str, *, _session=None, _origin=None
) -> host.ConsoleSettingsPersistenceOutcome:
    """Join the existing bounded per-session settings drain."""
    session = self._sessions.get(session_id)
    if (
        session is None
        or (_session is not None and session is not _session)
        or (
            _origin is not None
            and _origin
            != (
                session.incarnation_id,
                session.persisted_conversation_id,
                session.conversation_binding_revision,
            )
        )
    ):
        return host.ConsoleSettingsPersistenceOutcome(
            session_id=session_id,
            stale_components=frozenset({host.ConsoleSettingsComponent.PROMPT_MIXINS}),
        )
    return await self._join_console_settings_persistence_drain(
        session_id=session_id,
        persisted_conversation_id=session.persisted_conversation_id,
        conversation_binding_revision=session.conversation_binding_revision,
        generation_revision=session.generation_settings_revision,
        context_policy_revision=session.context_policy_revision,
        generation_snapshot=None,
        context_policy_overrides=None,
        policy_failure_label=host.ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS,
        components=frozenset({host.ConsoleSettingsComponent.PROMPT_MIXINS}),
    )
