from __future__ import annotations


class ConsoleChatController:
    def __init__(self):
        """Proposed insertion into the existing controller constructor."""
        from tldw_chatbook.Chat.console_context_compaction import (
            ConsoleCompactionPreflight,
        )

        self._compaction_preflight = ConsoleCompactionPreflight(
            read_controller__agent_bridge=lambda: self._agent_bridge,
            read_controller__agent_runtime_enabled=lambda: self._agent_runtime_enabled,
            read_controller__append_failure_system_row=lambda: (
                self._append_failure_system_row
            ),
            read_controller__apply_conversation_memory_preflight=lambda: (
                self._apply_conversation_memory_preflight
            ),
            read_controller__assess_request_capacity_only=lambda: (
                self._assess_request_capacity_only
            ),
            read_controller__automatic_memory_admission=lambda: (
                self._automatic_memory_admission
            ),
            read_controller__auxiliary_compaction_resolution=lambda: (
                self._auxiliary_compaction_resolution
            ),
            read_controller__block_context_preflight=lambda: (
                self._block_context_preflight
            ),
            read_controller__blocked_visible_copy=lambda: self._blocked_visible_copy,
            read_controller__compaction_admission=lambda: self._compaction_admission,
            read_controller__compaction_service=lambda: self._compaction_service,
            read_controller__context_accounting_by_session=lambda: (
                self._context_accounting_by_session
            ),
            read_controller__context_overflow_alert=lambda: (
                self._context_overflow_alert
            ),
            read_controller__context_repository=lambda: self._context_repository,
            read_controller__durable_context_snapshots=lambda: (
                self._durable_context_snapshots
            ),
            read_controller__global_context_policy_overrides=lambda: (
                self._global_context_policy_overrides
            ),
            read_controller__hooks_for_compaction=lambda: self._hooks_for_compaction,
            read_controller__provider_continuation_history_for_resolution=lambda: (
                self._provider_continuation_history_for_resolution
            ),
            read_controller__provider_messages_for_session=lambda: (
                self._provider_messages_for_session
            ),
            read_controller__provider_selection_for_session=lambda: (
                self._provider_selection_for_session
            ),
            read_controller__resolve_for_send_bounded=lambda: (
                self._resolve_for_send_bounded
            ),
            read_controller__select_session_effective_memory=lambda: (
                self._select_session_effective_memory
            ),
            read_controller_context_control_inputs=lambda: self.context_control_inputs,
            read_controller_provider_gateway=lambda: self.provider_gateway,
            read_controller_run_state_for=lambda: self.run_state_for,
            read_controller_store=lambda: self.store,
            read_global_Any=lambda: Any,
            read_global_CompactionDecision=lambda: CompactionDecision,
            read_global_CompactionFailureBehavior=lambda: CompactionFailureBehavior,
            read_global_CompactionPromptSnapshot=lambda: CompactionPromptSnapshot,
            read_global_CompactionTerminal=lambda: CompactionTerminal,
            read_global_ConsoleContextCapacity=lambda: ConsoleContextCapacity,
            read_global_ConsoleSubmitResult=lambda: ConsoleSubmitResult,
            read_global_ContextCarryForwardMode=lambda: ContextCarryForwardMode,
            read_global_ContextCompactionHold=lambda: ContextCompactionHold,
            read_global_ContextCompactionRepresentation=lambda: (
                ContextCompactionRepresentation
            ),
            read_global_ContinuationConflictError=lambda: ContinuationConflictError,
            read_global_EffectiveMemoryKind=lambda: EffectiveMemoryKind,
            read_global_Mapping=lambda: Mapping,
            read_global_NATIVE_MESSAGE_ID_KEY=lambda: NATIVE_MESSAGE_ID_KEY,
            read_global_PROVIDER_CONTINUATION_RECOVERY_REQUIRED=lambda: (
                PROVIDER_CONTINUATION_RECOVERY_REQUIRED
            ),
            read_global_PreparedConsoleRequest=lambda: PreparedConsoleRequest,
            read_global_ProviderArtifactTraceProvenance=lambda: (
                ProviderArtifactTraceProvenance
            ),
            read_global_TraceProvenanceSource=lambda: TraceProvenanceSource,
            read_global_TraceTransformKind=lambda: TraceTransformKind,
            read_global__context_overflow_cause=lambda: _context_overflow_cause,
            read_global__flatten_preflight_messages=lambda: _flatten_preflight_messages,
            read_global_asyncio=lambda: asyncio,
            read_global_compactable_units_after=lambda: compactable_units_after,
            read_global_compaction_retry_fence=lambda: compaction_retry_fence,
            read_global_compaction_transform_provenance=lambda: (
                compaction_transform_provenance
            ),
            read_global_complete_durable_units=lambda: complete_durable_units,
            read_global_decide_compaction=lambda: decide_compaction,
            read_global_effective_memory_identity=lambda: effective_memory_identity,
            read_global_frozen_policy_from_provenance=lambda: (
                frozen_policy_from_provenance
            ),
            read_global_get_internal_prompt=lambda: get_internal_prompt,
            read_global_is_vision_capable=lambda: is_vision_capable,
            read_global_logger=lambda: logger,
            read_global_max_history_images=lambda: max_history_images,
            read_global_merge_context_policy=lambda: merge_context_policy,
            read_global_plan_compaction=lambda: plan_compaction,
            read_global_project_effective_memory=lambda: project_effective_memory,
            read_global_replace=lambda: replace,
            read_global_resolve_context_policy=lambda: resolve_context_policy,
            read_global_resolve_micro_escalation=lambda: resolve_micro_escalation,
            read_global_tagged_memory_message=lambda: tagged_memory_message,
            read_global_tagged_visual_memory_message=lambda: (
                tagged_visual_memory_message
            ),
        )

    @_maintenance_boundary("compact")
    async def compact_context_now(
        self, session_id: str, *, micro: bool = False
    ) -> tuple[bool, str]:
        """Forward to the documented compaction preflight implementation."""
        return await self._compaction_preflight.compact_context_now(
            session_id, micro=micro
        )

    async def _apply_conversation_memory_preflight(
        self,
        *,
        session_id: str,
        resolution: ConsoleProviderResolution,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        agent_tools_enabled: bool,
        force_compaction: bool = False,
        manual_action: bool = False,
        micro_compaction: bool = False,
        continuation_sidecar: tuple[ProviderContinuationSidecar, ...] = (),
        continuation_target: ContinuationRestoreTarget | None = None,
        thinking_sidecar: tuple[ProviderThinkingSidecar, ...] = (),
        thinking_policy: ThinkingHistoryPolicy = "auto",
        assessment_sink: "Callable[[ContextCompactionHold, CompactionDecision, str | None], None] | None" = None,
        uncommitted_user_message_id: str | None = None,
        ask_bypassed: bool = False,
    ) -> tuple[list[dict[str, Any]], ConsoleSubmitResult | None]:
        """Forward to the documented compaction preflight implementation."""
        return await self._compaction_preflight._apply_conversation_memory_preflight(
            session_id=session_id,
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=assistant_message_id,
            agent_tools_enabled=agent_tools_enabled,
            force_compaction=force_compaction,
            manual_action=manual_action,
            micro_compaction=micro_compaction,
            continuation_sidecar=continuation_sidecar,
            continuation_target=continuation_target,
            thinking_sidecar=thinking_sidecar,
            thinking_policy=thinking_policy,
            assessment_sink=assessment_sink,
            uncommitted_user_message_id=uncommitted_user_message_id,
            ask_bypassed=ask_bypassed,
        )

    async def _assess_context_compaction(
        self,
        *,
        session_id: str,
        resolution: ConsoleProviderResolution,
        provider_messages: list[dict[str, Any]],
        uncommitted_user_message_id: str | None,
    ) -> tuple[ContextCompactionHold | None, str | None]:
        """Forward to the documented compaction preflight implementation."""
        return await self._compaction_preflight._assess_context_compaction(
            session_id=session_id,
            resolution=resolution,
            provider_messages=provider_messages,
            uncommitted_user_message_id=uncommitted_user_message_id,
        )

    def _assess_request_capacity_only(
        self,
        *,
        session_id: str,
        owner: Any,
        resolution: ConsoleProviderResolution,
        provider_messages: list[dict[str, Any]],
        prepare: Callable[..., Any],
        agent_tools_enabled: bool,
        assessment_sink: Callable[..., None],
    ) -> None:
        """Forward to the documented compaction preflight implementation."""
        return self._compaction_preflight._assess_request_capacity_only(
            session_id=session_id,
            owner=owner,
            resolution=resolution,
            provider_messages=provider_messages,
            prepare=prepare,
            agent_tools_enabled=agent_tools_enabled,
            assessment_sink=assessment_sink,
        )

    def _context_overflow_alert(
        self,
        decision: CompactionDecision,
        resolved: Any,
        capacity: Any,
        prepared_before: Any,
        resolution: ConsoleProviderResolution,
    ) -> str | None:
        """Forward to the documented compaction preflight implementation."""
        return self._compaction_preflight._context_overflow_alert(
            decision, resolved, capacity, prepared_before, resolution
        )
