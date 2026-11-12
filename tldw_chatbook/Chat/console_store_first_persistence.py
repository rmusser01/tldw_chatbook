"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from . import console_chat_store as native_host


def _persist_session_if_needed(
    self,
    session_id: str,
    *,
    strict_roleplay_context: bool = False,
    staged_identity: native_host.ConsoleStagedConversationIdentity | None = None,
) -> str | None:
    """Persist a session once, returning its persisted conversation ID.

    Returns:
        The persisted conversation ID; ``None`` when no persistence
        adapter is configured, or when the session is temporary.

    Raises:
        ValueError: If ``runtime_backend`` is not exactly ``"local"`` or
            ``"server"``.
    """
    session = self._session_or_raise(session_id)
    # Temporary conversations (spec 2026-07-31) stop here, BEFORE the
    # already-persisted check and before the adapter is consulted. This
    # single early return is the entire durability mechanism: with no
    # conversation id, `persist_message_if_needed` and every other
    # conversation-scoped write returns early on its own.
    if session.ephemeral:
        return None
    if session.persisted_conversation_id is not None:
        self.retry_pending_workspace_projection(session_id)
        return session.persisted_conversation_id
    if self.persistence is None:
        return None

    self.invalidate_response_rule_source(session_id, "saved")
    if type(session.runtime_backend) is not str or session.runtime_backend not in {
        "local",
        "server",
    }:
        native_host.logger.bind(
            session_id=session_id,
            runtime_backend=native_host._invalid_runtime_backend_diagnostic(
                session.runtime_backend
            ),
        ).error("Cannot persist Console session with invalid runtime backend.")
        raise ValueError(
            "Cannot persist Console session: runtime_backend must be "
            "'local' or 'server'."
        )
    scope_type, persisted_workspace_id = self._persistence_scope(session)
    local_character_id = session.local_character_id()
    identity_kwargs = {
        "runtime_backend": session.runtime_backend,
        "assistant_kind": session.assistant_kind,
        "assistant_id": session.assistant_id,
        "assistant_authority_id": session.assistant_authority_id,
        "persona_memory_mode": session.persona_memory_mode,
        "character_id": local_character_id,
        "character_name": (
            session.character_name if local_character_id is not None else None
        ),
    }
    create_conversation = self.persistence.create_conversation
    create_kwargs: dict[str, native_host.Any] = dict(
        conversation_title=session.title,
        workspace_id=persisted_workspace_id,
        scope_type=scope_type,
        system_prompt=session.settings.system_prompt
        if session.settings is not None
        else None,
        thinking_history_policy=session.thinking_history_policy,
        **identity_kwargs,
    )
    staged_generation_snapshot = (
        native_host.snapshot_from_session_settings(session.settings)
        if session.settings is not None and session.generation_settings_revision > 0
        else None
    )
    if staged_generation_snapshot is not None:
        create_kwargs["metadata"] = native_host.merge_console_generation_settings(
            create_kwargs.get("metadata"),
            staged_generation_snapshot,
        )
    if session.speech_preferences != native_host.ConsoleSpeechPreferences():
        if not self._persistence_accepts_kwarg(
            create_conversation,
            "speech_preferences",
        ):
            raise RuntimeError(
                "Persistence adapter cannot store staged reply-speech preferences."
            )
        create_kwargs["speech_preferences"] = session.speech_preferences
    staged_identity = staged_identity or self.stage_first_persistence(session_id)
    policy_snapshot = session.library_policy_holder.snapshot
    policy_candidate = native_host.ConsoleLibraryPolicyCandidate(
        auto_retrieve=policy_snapshot.auto_retrieve,
        assistant_access=policy_snapshot.assistant_access,
    )
    atomic_first_persist = getattr(
        self.persistence,
        "persist_console_conversation_with_policy",
        None,
    )
    if callable(atomic_first_persist):
        committed_policy = atomic_first_persist(
            conversation_id=staged_identity.conversation_id,
            policy_candidate=policy_candidate,
            conversation_kwargs=create_kwargs,
        )
        committed_identity = staged_identity
    else:
        created_id = create_conversation(**create_kwargs)
        committed_identity = native_host.ConsoleStagedConversationIdentity(
            conversation_id=created_id,
            title=staged_identity.title,
        )
        committed_policy = None
    self.publish_committed_identity(session_id, committed_identity)
    if staged_generation_snapshot is not None:
        session.generation_durable_snapshot = staged_generation_snapshot
    if isinstance(committed_policy, native_host.ConsoleLibraryPolicySnapshot):
        session.library_policy_holder.snapshot = committed_policy
        session.library_policy_holder.explicitly_staged = False
        if self.library_policy_coordinator is not None:
            self.library_policy_coordinator.register_holder(
                session.id,
                committed_identity.conversation_id,
                session.library_policy_holder,
            )
    self._project_workspace_membership_after_commit(session)
    if (
        session.user_display_name_override is not None
        or session.character_system_template is not None
        or session.assistant_kind == "character"
        or session.persona_system_template is not None
    ) and not self._persist_roleplay_context(session):
        native_host.logger.warning(
            "Failed to flush Console roleplay context on first persist."
        )
        if strict_roleplay_context:
            raise RuntimeError(
                "Failed to flush Console roleplay context while promoting "
                "a temporary session."
            )
    self._persist_project_instruction_state(session)
    pinned_prefill = (
        session.settings.pinned_prefill if session.settings is not None else None
    )
    if pinned_prefill:
        update_pinned = getattr(
            self.persistence, "update_conversation_pinned_prefill", None
        )
        if callable(update_pinned):
            try:
                update_pinned(
                    conversation_id=session.persisted_conversation_id,
                    pinned_prefill=pinned_prefill,
                )
            except Exception:
                native_host.logger.bind(
                    session_id=session_id,
                    conversation_id=session.persisted_conversation_id,
                ).exception("Failed to flush pinned prefill on first persist.")
    self._flush_context_policy_on_first_persist(session)
    # task-9: flush a session-held RAG retrieval scope (unpersisted-
    # session lifecycle, ``SessionScopeHolder``) through to durable
    # storage now that the conversation row exists. ``flush_to`` itself
    # no-ops when nothing was held, so this is safe to call
    # unconditionally. Requires the underlying ``CharactersRAGDB`` --
    # ``self.persistence`` is the ``ChatPersistenceService`` wrapper, so
    # the raw DB is reached via its ``db`` attribute, now a declared
    # (not merely probed) member of ``ConsoleChatPersistence`` (PR #747
    # review); persistence adapters without one (e.g. test fakes) still
    # simply skip the flush, matching every other durable write in this
    # method degrading gracefully when the seam it needs is absent --
    # but that skip must be OBSERVABLE (see the ``else`` branch below)
    # rather than a silent loss of the user's scope selection.
    persistence_db = getattr(self.persistence, "db", None)
    # Captured BEFORE the flush -- `flush_to` empties the holder on
    # success, so this is the only chance to learn what was actually
    # held (task-9 review finding 1; PR #747 review).
    held_scope = session.rag_scope_holder.scope
    if persistence_db is not None:
        flushed_scope = held_scope
        try:
            session.rag_scope_holder.flush_to(
                persistence_db, session.persisted_conversation_id
            )
        except Exception:
            native_host.logger.bind(
                session_id=session_id,
                conversation_id=session.persisted_conversation_id,
            ).exception("Failed to flush RAG retrieval scope on first persist.")
        else:
            if flushed_scope is not None and self.on_scope_flushed is not None:
                try:
                    self.on_scope_flushed(
                        session.persisted_conversation_id, flushed_scope
                    )
                except Exception:
                    native_host.logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception(
                        "on_scope_flushed callback raised after a "
                        "successful RAG retrieval scope flush."
                    )
    elif held_scope is not None:
        # A scope WAS held but the persistence adapter exposes no raw
        # `db` seam to flush it through -- the holder is left untouched
        # (not emptied) so a later flush attempt could still succeed,
        # but the loss must not be silent: warn, naming the
        # conversation, so it is observable.
        native_host.logger.bind(
            session_id=session_id,
            conversation_id=session.persisted_conversation_id,
        ).warning(
            "Skipped RAG retrieval scope flush for conversation {} on "
            "first persist: persistence adapter exposes no `db` seam. "
            "The scope remains held in-memory only and was not "
            "written to durable storage.",
            session.persisted_conversation_id,
        )
    return session.persisted_conversation_id


def _flush_context_policy_on_first_persist(
    self, session: native_host.ConsoleChatSession
) -> native_host.ContextPolicyWriteResult | None:
    """Write one staged policy and publish its durable owned revision."""
    if (
        session.persisted_conversation_id is None
        or session.context_policy_revision < 1
        or self.persistence is None
    ):
        return None
    failure_label = (
        session.staged_context_policy_failure_label
        if session.staged_context_policy_failure_revision
        == session.context_policy_revision
        and session.staged_context_policy_failure_label is not None
        else native_host.ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS
    )
    writer = getattr(self.persistence, "update_conversation_context_policy", None)
    if not callable(writer):
        native_host.logger.warning(
            "Skipped staged Console context-policy flush: persistence "
            "adapter exposes no policy write seam."
        )
        result = native_host.ContextPolicyWriteResult(
            native_host.ContextPolicyWriteStatus.MISSING,
            session.context_policy_durable_revision,
        )
        self._record_console_settings_failure(
            session,
            component=native_host.ConsoleSettingsComponent.CONTEXT_POLICY,
            revision=session.context_policy_revision,
            context_policy_overrides=session.context_policy_overrides,
            policy_failure_label=failure_label,
        )
        return result
    try:
        raw_result = writer(
            conversation_id=session.persisted_conversation_id,
            overrides=session.context_policy_overrides,
            expected_revision=session.context_policy_durable_revision,
        )
    except Exception:
        native_host.logger.error(
            "Failed to flush Console context policy on first persist."
        )
        result = native_host.ContextPolicyWriteResult(
            native_host.ContextPolicyWriteStatus.CONFLICT,
            session.context_policy_durable_revision,
        )
        self._record_console_settings_failure(
            session,
            component=native_host.ConsoleSettingsComponent.CONTEXT_POLICY,
            revision=session.context_policy_revision,
            context_policy_overrides=session.context_policy_overrides,
            policy_failure_label=failure_label,
        )
        return result
    written, revision = self._context_write_result(raw_result)
    if written:
        session.context_policy_durable_revision = revision
        session.settings_persistence_failures.pop(
            native_host.ConsoleSettingsComponent.CONTEXT_POLICY,
            None,
        )
        session.staged_context_policy_failure_label = None
        session.staged_context_policy_failure_revision = None
        return native_host.ContextPolicyWriteResult(
            native_host.ContextPolicyWriteStatus.WRITTEN,
            revision,
        )
    self._record_console_settings_failure(
        session,
        component=native_host.ConsoleSettingsComponent.CONTEXT_POLICY,
        revision=session.context_policy_revision,
        context_policy_overrides=session.context_policy_overrides,
        policy_failure_label=failure_label,
    )
    return (
        raw_result
        if isinstance(raw_result, native_host.ContextPolicyWriteResult)
        else native_host.ContextPolicyWriteResult(
            native_host.ContextPolicyWriteStatus.CONFLICT,
            session.context_policy_durable_revision,
        )
    )
