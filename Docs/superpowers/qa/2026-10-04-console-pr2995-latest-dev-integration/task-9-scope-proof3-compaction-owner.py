from __future__ import annotations
from typing import Any, Callable


class ConsoleCompactionPreflight:
    """Proposed policy/preflight owner; every live dependency is named."""

    def __init__(
        self,
        *,
        read_controller__agent_bridge: Callable[[], Any],
        read_controller__agent_runtime_enabled: Callable[[], Any],
        read_controller__append_failure_system_row: Callable[[], Any],
        read_controller__apply_conversation_memory_preflight: Callable[[], Any],
        read_controller__assess_request_capacity_only: Callable[[], Any],
        read_controller__automatic_memory_admission: Callable[[], Any],
        read_controller__auxiliary_compaction_resolution: Callable[[], Any],
        read_controller__block_context_preflight: Callable[[], Any],
        read_controller__blocked_visible_copy: Callable[[], Any],
        read_controller__compaction_admission: Callable[[], Any],
        read_controller__compaction_service: Callable[[], Any],
        read_controller__context_accounting_by_session: Callable[[], Any],
        read_controller__context_overflow_alert: Callable[[], Any],
        read_controller__context_repository: Callable[[], Any],
        read_controller__durable_context_snapshots: Callable[[], Any],
        read_controller__global_context_policy_overrides: Callable[[], Any],
        read_controller__hooks_for_compaction: Callable[[], Any],
        read_controller__provider_continuation_history_for_resolution: Callable[
            [], Any
        ],
        read_controller__provider_messages_for_session: Callable[[], Any],
        read_controller__provider_selection_for_session: Callable[[], Any],
        read_controller__resolve_for_send_bounded: Callable[[], Any],
        read_controller__select_session_effective_memory: Callable[[], Any],
        read_controller_context_control_inputs: Callable[[], Any],
        read_controller_provider_gateway: Callable[[], Any],
        read_controller_run_state_for: Callable[[], Any],
        read_controller_store: Callable[[], Any],
        read_global_Any: Callable[[], Any],
        read_global_CompactionDecision: Callable[[], Any],
        read_global_CompactionFailureBehavior: Callable[[], Any],
        read_global_CompactionPromptSnapshot: Callable[[], Any],
        read_global_CompactionTerminal: Callable[[], Any],
        read_global_ConsoleContextCapacity: Callable[[], Any],
        read_global_ConsoleSubmitResult: Callable[[], Any],
        read_global_ContextCarryForwardMode: Callable[[], Any],
        read_global_ContextCompactionHold: Callable[[], Any],
        read_global_ContextCompactionRepresentation: Callable[[], Any],
        read_global_ContinuationConflictError: Callable[[], Any],
        read_global_EffectiveMemoryKind: Callable[[], Any],
        read_global_Mapping: Callable[[], Any],
        read_global_NATIVE_MESSAGE_ID_KEY: Callable[[], Any],
        read_global_PROVIDER_CONTINUATION_RECOVERY_REQUIRED: Callable[[], Any],
        read_global_PreparedConsoleRequest: Callable[[], Any],
        read_global_ProviderArtifactTraceProvenance: Callable[[], Any],
        read_global_TraceProvenanceSource: Callable[[], Any],
        read_global_TraceTransformKind: Callable[[], Any],
        read_global__context_overflow_cause: Callable[[], Any],
        read_global__flatten_preflight_messages: Callable[[], Any],
        read_global_asyncio: Callable[[], Any],
        read_global_compactable_units_after: Callable[[], Any],
        read_global_compaction_retry_fence: Callable[[], Any],
        read_global_compaction_transform_provenance: Callable[[], Any],
        read_global_complete_durable_units: Callable[[], Any],
        read_global_decide_compaction: Callable[[], Any],
        read_global_effective_memory_identity: Callable[[], Any],
        read_global_frozen_policy_from_provenance: Callable[[], Any],
        read_global_get_internal_prompt: Callable[[], Any],
        read_global_is_vision_capable: Callable[[], Any],
        read_global_logger: Callable[[], Any],
        read_global_max_history_images: Callable[[], Any],
        read_global_merge_context_policy: Callable[[], Any],
        read_global_plan_compaction: Callable[[], Any],
        read_global_project_effective_memory: Callable[[], Any],
        read_global_replace: Callable[[], Any],
        read_global_resolve_context_policy: Callable[[], Any],
        read_global_resolve_micro_escalation: Callable[[], Any],
        read_global_tagged_memory_message: Callable[[], Any],
        read_global_tagged_visual_memory_message: Callable[[], Any],
    ) -> None:
        self.read_controller__agent_bridge = read_controller__agent_bridge
        self.read_controller__agent_runtime_enabled = (
            read_controller__agent_runtime_enabled
        )
        self.read_controller__append_failure_system_row = (
            read_controller__append_failure_system_row
        )
        self.read_controller__apply_conversation_memory_preflight = (
            read_controller__apply_conversation_memory_preflight
        )
        self.read_controller__assess_request_capacity_only = (
            read_controller__assess_request_capacity_only
        )
        self.read_controller__automatic_memory_admission = (
            read_controller__automatic_memory_admission
        )
        self.read_controller__auxiliary_compaction_resolution = (
            read_controller__auxiliary_compaction_resolution
        )
        self.read_controller__block_context_preflight = (
            read_controller__block_context_preflight
        )
        self.read_controller__blocked_visible_copy = (
            read_controller__blocked_visible_copy
        )
        self.read_controller__compaction_admission = (
            read_controller__compaction_admission
        )
        self.read_controller__compaction_service = read_controller__compaction_service
        self.read_controller__context_accounting_by_session = (
            read_controller__context_accounting_by_session
        )
        self.read_controller__context_overflow_alert = (
            read_controller__context_overflow_alert
        )
        self.read_controller__context_repository = read_controller__context_repository
        self.read_controller__durable_context_snapshots = (
            read_controller__durable_context_snapshots
        )
        self.read_controller__global_context_policy_overrides = (
            read_controller__global_context_policy_overrides
        )
        self.read_controller__hooks_for_compaction = (
            read_controller__hooks_for_compaction
        )
        self.read_controller__provider_continuation_history_for_resolution = (
            read_controller__provider_continuation_history_for_resolution
        )
        self.read_controller__provider_messages_for_session = (
            read_controller__provider_messages_for_session
        )
        self.read_controller__provider_selection_for_session = (
            read_controller__provider_selection_for_session
        )
        self.read_controller__resolve_for_send_bounded = (
            read_controller__resolve_for_send_bounded
        )
        self.read_controller__select_session_effective_memory = (
            read_controller__select_session_effective_memory
        )
        self.read_controller_context_control_inputs = (
            read_controller_context_control_inputs
        )
        self.read_controller_provider_gateway = read_controller_provider_gateway
        self.read_controller_run_state_for = read_controller_run_state_for
        self.read_controller_store = read_controller_store
        self.read_global_Any = read_global_Any
        self.read_global_CompactionDecision = read_global_CompactionDecision
        self.read_global_CompactionFailureBehavior = (
            read_global_CompactionFailureBehavior
        )
        self.read_global_CompactionPromptSnapshot = read_global_CompactionPromptSnapshot
        self.read_global_CompactionTerminal = read_global_CompactionTerminal
        self.read_global_ConsoleContextCapacity = read_global_ConsoleContextCapacity
        self.read_global_ConsoleSubmitResult = read_global_ConsoleSubmitResult
        self.read_global_ContextCarryForwardMode = read_global_ContextCarryForwardMode
        self.read_global_ContextCompactionHold = read_global_ContextCompactionHold
        self.read_global_ContextCompactionRepresentation = (
            read_global_ContextCompactionRepresentation
        )
        self.read_global_ContinuationConflictError = (
            read_global_ContinuationConflictError
        )
        self.read_global_EffectiveMemoryKind = read_global_EffectiveMemoryKind
        self.read_global_Mapping = read_global_Mapping
        self.read_global_NATIVE_MESSAGE_ID_KEY = read_global_NATIVE_MESSAGE_ID_KEY
        self.read_global_PROVIDER_CONTINUATION_RECOVERY_REQUIRED = (
            read_global_PROVIDER_CONTINUATION_RECOVERY_REQUIRED
        )
        self.read_global_PreparedConsoleRequest = read_global_PreparedConsoleRequest
        self.read_global_ProviderArtifactTraceProvenance = (
            read_global_ProviderArtifactTraceProvenance
        )
        self.read_global_TraceProvenanceSource = read_global_TraceProvenanceSource
        self.read_global_TraceTransformKind = read_global_TraceTransformKind
        self.read_global__context_overflow_cause = read_global__context_overflow_cause
        self.read_global__flatten_preflight_messages = (
            read_global__flatten_preflight_messages
        )
        self.read_global_asyncio = read_global_asyncio
        self.read_global_compactable_units_after = read_global_compactable_units_after
        self.read_global_compaction_retry_fence = read_global_compaction_retry_fence
        self.read_global_compaction_transform_provenance = (
            read_global_compaction_transform_provenance
        )
        self.read_global_complete_durable_units = read_global_complete_durable_units
        self.read_global_decide_compaction = read_global_decide_compaction
        self.read_global_effective_memory_identity = (
            read_global_effective_memory_identity
        )
        self.read_global_frozen_policy_from_provenance = (
            read_global_frozen_policy_from_provenance
        )
        self.read_global_get_internal_prompt = read_global_get_internal_prompt
        self.read_global_is_vision_capable = read_global_is_vision_capable
        self.read_global_logger = read_global_logger
        self.read_global_max_history_images = read_global_max_history_images
        self.read_global_merge_context_policy = read_global_merge_context_policy
        self.read_global_plan_compaction = read_global_plan_compaction
        self.read_global_project_effective_memory = read_global_project_effective_memory
        self.read_global_replace = read_global_replace
        self.read_global_resolve_context_policy = read_global_resolve_context_policy
        self.read_global_resolve_micro_escalation = read_global_resolve_micro_escalation
        self.read_global_tagged_memory_message = read_global_tagged_memory_message
        self.read_global_tagged_visual_memory_message = (
            read_global_tagged_visual_memory_message
        )

    async def compact_context_now(
        self, session_id: str, *, micro: bool = False
    ) -> tuple[bool, str]:
        """Run one user-initiated bounded compaction without sending a turn.

        TASK-25910: ``micro=True`` is the per-turn micro-compaction pass --
        same assembly, but the preflight only escalates a below-trigger
        AUTOMATIC decision (never ASK) and caps the plan at the single
        oldest exchange; every refusal is silent for that caller.
        """
        if not self.read_controller_run_state_for()(session_id).is_send_allowed:
            return False, "Wait for the active run to finish before compacting."
        owner = next(
            (
                item
                for item in self.read_controller_store().sessions()
                if item.id == session_id
            ),
            None,
        )
        if owner is None or owner.persisted_conversation_id is None:
            return False, "Send or save this conversation before compacting it."
        try:
            # Review #2: the OWNING session's provider, never the viewed
            # tab's -- a background micro fold can fire on a session the
            # user has switched away from.
            main_selection = self.read_controller__provider_selection_for_session()(
                session_id
            )
            resolution = await self.read_controller__resolve_for_send_bounded()(
                main_selection
            )
        except Exception:
            return False, "The active provider could not be prepared for compaction."
        if not getattr(resolution, "ready", False):
            return False, self.read_controller__blocked_visible_copy()(
                getattr(resolution, "visible_copy", "")
            )
        # TASK-26024: route the compaction summary to a cheaper auxiliary
        # model when configured (fallback to this resolution otherwise).
        resolution = await self.read_controller__auxiliary_compaction_resolution()(
            main_selection, resolution
        )
        overrides, global_overrides, before_memory = (
            self.read_controller_context_control_inputs()(session_id)
        )
        requested_representation = self.read_global_merge_context_policy()(
            global_overrides=global_overrides,
            conversation_overrides=overrides,
        ).compaction_representation
        try:
            continuation_sidecar, continuation_target = (
                self.read_controller__provider_continuation_history_for_resolution()(
                    session_id, resolution
                )
            )
        except self.read_global_ContinuationConflictError():
            return False, self.read_global_PROVIDER_CONTINUATION_RECOVERY_REQUIRED()
        (
            _messages,
            blocked_result,
        ) = await self.read_controller__apply_conversation_memory_preflight()(
            session_id=session_id,
            resolution=resolution,
            provider_messages=self.read_controller__provider_messages_for_session()(
                session_id, annotate_ids=True
            ),
            assistant_message_id="",
            agent_tools_enabled=False,
            force_compaction=not micro,
            manual_action=True,
            micro_compaction=micro,
            continuation_sidecar=continuation_sidecar,
            continuation_target=continuation_target,
        )
        if blocked_result is not None:
            return False, blocked_result.visible_copy
        _overrides, _global, after_memory = (
            self.read_controller_context_control_inputs()(session_id)
        )
        if after_memory.kind is self.read_global_EffectiveMemoryKind().RAW or (
            self.read_global_effective_memory_identity()(before_memory)
            == self.read_global_effective_memory_identity()(after_memory)
        ):
            if (
                requested_representation
                is self.read_global_ContextCompactionRepresentation().VISUAL_TRANSCRIPT
                and self.read_global_is_vision_capable()(
                    resolution.provider, resolution.model or ""
                )
            ):
                return (
                    True,
                    "Visual transcript fits and will be regenerated locally for each request; transcript unchanged.",
                )
            return False, "There are not enough older complete turns to compact yet."
        return True, "Conversation memory updated; transcript messages were unchanged."

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
        """Revalidate memory and optionally run one automatic summary call.

        TASK-34350: ``assessment_sink`` turns this into a side-effect-free
        probe -- it receives the decision and its numbers, and the request is
        returned unchanged before anything compacts or blocks.
        ``ask_bypassed`` is the user's one-shot "Send without compacting" for
        a send held at the threshold.
        """
        from tldw_chatbook.Chat.console_visual_transcript import (
            count_semantic_images,
            plan_visual_compaction,
            render_visual_transcript,
            resolve_effective_compaction_representation,
        )

        def blocked(visible_copy: str) -> self.read_global_ConsoleSubmitResult():
            if manual_action:
                return self.read_global_ConsoleSubmitResult()(False, True, visible_copy)
            return self.read_controller__block_context_preflight()(
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                visible_copy=visible_copy,
            )

        repository = self.read_controller__context_repository()
        service = self.read_controller__compaction_service()
        prepare = getattr(
            self.read_controller_provider_gateway(), "prepare_chat_request", None
        )
        owner = next(
            (
                item
                for item in self.read_controller_store().sessions()
                if item.id == session_id
            ),
            None,
        )
        if (
            repository is None
            or service is None
            or not callable(prepare)
            or owner is None
        ):
            return provider_messages, None
        snapshots = (
            self.read_controller__durable_context_snapshots()(
                session_id, uncommitted_user_message_id=uncommitted_user_message_id
            )
            if owner.persisted_conversation_id is not None
            else None
        )
        if not snapshots:
            if assessment_sink is not None and not owner.ephemeral:
                # TASK-34350: a durable chat's first message is assessed for
                # capacity alone (nothing to compact yet), so a request that
                # cannot fit is refused before commit, like a later one.
                self.read_controller__assess_request_capacity_only()(
                    session_id=session_id,
                    owner=owner,
                    resolution=resolution,
                    provider_messages=provider_messages,
                    prepare=prepare,
                    agent_tools_enabled=agent_tools_enabled,
                    assessment_sink=assessment_sink,
                )
            return provider_messages, None
        conversation_id = owner.persisted_conversation_id
        effective = self.read_controller__select_session_effective_memory()(
            session_id,
            conversation_id,
            snapshots,
        )
        projection = self.read_global_project_effective_memory()(
            provider_messages, effective
        )
        memory = effective.memory
        retained_messages = list(projection.rows)
        memory_rows = projection.memory

        tools: list[self.read_global_Mapping()[str, self.read_global_Any()]] = []
        if agent_tools_enabled and self.read_controller__agent_bridge() is not None:
            preview = getattr(
                self.read_controller__agent_bridge(), "preview_tool_schemas", None
            )
            if callable(preview):
                try:
                    tools = list(preview())
                except Exception:
                    tools = []
        prepared_before = prepare(
            resolution,
            retained_messages,
            tools=tools,
            apply_safety_window=False,
            continuation_target=continuation_target,
            continuation_sidecar=continuation_sidecar,
            continuation_owner_key=(
                self.read_global_NATIVE_MESSAGE_ID_KEY()
                if continuation_sidecar
                else None
            ),
            thinking_sidecar=thinking_sidecar,
            thinking_policy=thinking_policy,
            thinking_owner_key=(
                self.read_global_NATIVE_MESSAGE_ID_KEY() if thinking_sidecar else None
            ),
        )
        semantic = prepared_before.semantic
        if memory_rows:
            memory_provenance = (
                self.read_global_replace()(
                    semantic.provenance,
                    memory=(
                        self.read_global_ProviderArtifactTraceProvenance()(
                            self.read_global_TraceProvenanceSource().CONVERSATION_MEMORY,
                            self.read_global_frozen_policy_from_provenance()(
                                semantic.provenance
                            ),
                        ),
                    ),
                )
                if semantic.provenance is not None
                else None
            )
            semantic = self.read_global_replace()(
                semantic,
                memory=memory_rows,
                provenance=memory_provenance,
            )
            prepared_before = prepare(
                resolution,
                semantic,
                apply_safety_window=False,
                continuation_target=continuation_target,
            )
        capacity = prepared_before.capacity
        # TASK-26019: this accounting IS the request's own (AC#2); the
        # breakdown surface reads the latest copy, no re-estimation.
        self.read_controller__context_accounting_by_session()[session_id] = (
            prepared_before.accounting
        )
        mandatory_tokens = (
            prepared_before.accounting.non_compactable_tokens
            - prepared_before.accounting.memory_tokens
        )
        try:
            global_overrides = self.read_controller__global_context_policy_overrides()()
        except Exception:
            global_overrides = None
        resolved = self.read_global_resolve_context_policy()(
            capacity=self.read_global_ConsoleContextCapacity()(
                model_context_window_tokens=capacity.context_window_tokens,
                model_window_verified=capacity.safety_verified,
                provider_input_cap_tokens=capacity.provider_input_cap_tokens,
                response_reservation_tokens=capacity.effective_response_tokens,
                safety_margin_tokens=capacity.safety_margin_tokens,
                mandatory_input_tokens=mandatory_tokens,
            ),
            global_overrides=global_overrides,
            conversation_overrides=owner.context_policy_overrides,
        )

        def prepare_main(request: self.read_global_PreparedConsoleRequest()):
            return prepare(
                resolution,
                request,
                apply_safety_window=False,
                continuation_target=continuation_target,
            )

        units = (
            self.read_global_complete_durable_units()(snapshots)
            if effective.kind is self.read_global_EffectiveMemoryKind().GENERATED_RANGE
            else self.read_global_compactable_units_after()(
                snapshots,
                boundary_message_id=(
                    memory.boundary_message_id if memory is not None else None
                ),
            )
        )
        decision = self.read_global_decide_compaction()(
            resolved,
            conversation_tokens=(
                prepared_before.accounting.memory_tokens
                + prepared_before.accounting.compactable_tokens
            ),
            compactable_units=len(units),
        )
        if effective.kind is self.read_global_EffectiveMemoryKind().LEGACY_PREFIX and (
            force_compaction
            or decision
            in {
                self.read_global_CompactionDecision().ASK,
                self.read_global_CompactionDecision().AUTOMATIC,
            }
        ):
            decision = self.read_global_CompactionDecision().NON_COMPACTABLE
        elif force_compaction and units:
            decision = self.read_global_CompactionDecision().AUTOMATIC
        # TASK-25910: the escalation ruling is a pure, pinned function in
        # console_context_compaction (review Critical 2026-09-01: the
        # inline version was an uncovered runtime NameError). Every micro
        # pass is capped to the single oldest exchange or silently no-ops.
        micro_escalated = False
        if micro_compaction:
            ruling = self.read_global_resolve_micro_escalation()(
                decision,
                units_present=bool(units),
                compaction_mode=resolved.policy.compaction_mode,
                effective_kind=effective.kind,
            )
            if ruling is None:
                return self.read_global__flatten_preflight_messages()(semantic), None
            decision, micro_escalated = ruling
        if assessment_sink is not None:
            budget = resolved.effective_conversation_budget_tokens or 0
            assessment_sink(
                self.read_global_ContextCompactionHold()(
                    session_id=session_id,
                    used_tokens=(
                        prepared_before.accounting.memory_tokens
                        + prepared_before.accounting.compactable_tokens
                    ),
                    trigger_tokens=int(budget * resolved.policy.trigger_ratio),
                    budget_tokens=budget,
                    # Only a budget the window sets rests on its estimate; a
                    # custom budget below capacity does not (live 2026-10-04).
                    estimated=(
                        capacity.limit_source == "estimated"
                        and resolved.effective_conversation_budget_tokens
                        == resolved.available_conversation_capacity_tokens
                    ),
                ),
                decision,
                self.read_controller__context_overflow_alert()(
                    decision, resolved, capacity, prepared_before, resolution
                ),
            )
            return provider_messages, None
        self.read_global_logger().info("console_context_policy_decision")
        if decision in {
            self.read_global_CompactionDecision().OFF,
            self.read_global_CompactionDecision().BELOW_TRIGGER,
        }:
            return self.read_global__flatten_preflight_messages()(semantic), None
        if ask_bypassed and decision in {
            self.read_global_CompactionDecision().ASK,
            self.read_global_CompactionDecision().AUTOMATIC,
        }:
            # The user's answer was "do not compact this send" (or "compacted
            # already"); a policy changed to Automatic meanwhile must not
            # compact it anyway (Qodo #2 on PR #3003).
            return self.read_global__flatten_preflight_messages()(semantic), None
        if decision is self.read_global_CompactionDecision().ASK:
            # A composer send is held before it is committed (TASK-34350);
            # this is the path for sends that cannot show the hold card.
            result = blocked(
                (
                    "Conversation context reached its compaction threshold. "
                    "Use Compact now in Conversation settings > Context and "
                    "memory, then send again."
                )
            )
            return provider_messages, result
        if decision in {
            self.read_global_CompactionDecision().UNKNOWN_WINDOW,
            self.read_global_CompactionDecision().NON_COMPACTABLE,
        }:
            # A missing compaction threshold or an empty set of replaceable
            # units is not itself a provider overflow.  Unknown/new models
            # historically remained sendable with an explicit unverified
            # label, and reaching a policy high-water mark while the exact
            # request still fits must not turn that advisory threshold into
            # an admission failure.  Block only when the immutable prepared
            # request proves that the effective input ceiling is exceeded.
            alert = self.read_controller__context_overflow_alert()(
                decision, resolved, capacity, prepared_before, resolution
            )
            if alert is None:
                return self.read_global__flatten_preflight_messages()(semantic), None
            return provider_messages, blocked(alert)

        requested_representation = resolved.policy.compaction_representation
        if effective.kind is self.read_global_EffectiveMemoryKind().GENERATED_RANGE:
            requested_representation = (
                self.read_global_ContextCompactionRepresentation().TEXT_SUMMARY
            )
        vision_available = False
        if (
            requested_representation
            is not self.read_global_ContextCompactionRepresentation().TEXT_SUMMARY
        ):
            try:
                vision_available = self.read_global_is_vision_capable()(
                    resolution.provider, resolution.model or ""
                )
            except Exception:
                vision_available = False
        effective_representation, visual_fallback_reason = (
            resolve_effective_compaction_representation(
                requested_representation,
                vision_available=vision_available,
            )
        )

        if (
            effective_representation
            is self.read_global_ContextCompactionRepresentation().VISUAL_TRANSCRIPT
        ):
            budget = resolved.effective_conversation_budget_tokens
            visual_plan = None
            if budget is not None:
                try:
                    visual_plan = await self.read_global_asyncio().to_thread(
                        plan_visual_compaction,
                        semantic=semantic,
                        prepared_before=prepared_before,
                        durable_units=units,
                        budget_tokens=budget,
                        target_ratio=resolved.policy.target_ratio,
                        max_images=self.read_global_max_history_images()(
                            resolution.provider, resolution.model or ""
                        ),
                        keep_latest_exchange=(
                            resolved.policy.carry_forward_mode
                            is self.read_global_ContextCarryForwardMode().MEMORY_WITH_LATEST_EXCHANGE
                        ),
                        prepare_main=prepare_main,
                    )
                except Exception:
                    visual_fallback_reason = "local_visual_render_failed"
            if visual_plan is not None and visual_plan.plan is not None:
                self.read_global_logger().info("console_visual_compaction_prepared")
                return self.read_global__flatten_preflight_messages()(
                    visual_plan.plan.semantic
                ), None
            effective_representation = (
                self.read_global_ContextCompactionRepresentation().TEXT_SUMMARY
            )
            if visual_fallback_reason is None:
                visual_fallback_reason = (
                    visual_plan.reason
                    if visual_plan is not None
                    else "visual_compaction_unavailable"
                )

        if visual_fallback_reason is not None:
            self.read_global_logger().info(
                "console_visual_compaction_fell_back_to_text"
            )

        prompt = self.read_global_CompactionPromptSnapshot()(
            self.read_global_get_internal_prompt()("console.rewind_summarize")
        )

        def prepare_auxiliary(messages, output_cap):
            return prepare(
                self.read_global_replace()(
                    resolution,
                    streaming=False,
                    max_tokens=output_cap,
                ),
                list(messages),
                apply_safety_window=False,
            )

        try:
            max_visual_inputs = (
                self.read_global_max_history_images()(
                    resolution.provider, resolution.model or ""
                )
                if self.read_global_is_vision_capable()(
                    resolution.provider, resolution.model or ""
                )
                else 0
            )
        except Exception:
            max_visual_inputs = 0

        planned = self.read_global_plan_compaction()(
            semantic=semantic,
            prepared_before=prepared_before,
            durable_units=units,
            resolved_policy=resolved,
            prompt=prompt,
            effective_memory=effective,
            max_visual_inputs=max_visual_inputs,
            prepare_main=prepare_main,
            prepare_auxiliary=prepare_auxiliary,
            max_units=1 if micro_escalated else None,
        )
        if micro_escalated and planned.plan is None:
            # An unprofitable single-exchange fold (too small to beat the
            # summary cap) just waits for a later tick -- never a blocked
            # notice from a background pass.
            return self.read_global__flatten_preflight_messages()(semantic), None
        if planned.plan is None:
            if not manual_action and (
                resolved.policy.failure_behavior
                is self.read_global_CompactionFailureBehavior().OMIT_OLDER_CONTEXT
            ):
                return self.read_global__flatten_preflight_messages()(semantic), None
            # ADR-097: failure copy is first-use work, outside boot/mount.
            from .console_compaction_failure import compaction_failure_copy

            return provider_messages, blocked(
                compaction_failure_copy(
                    planned.reason or "plan_unreachable", manual=manual_action
                )
            )

        admission = self.read_controller__compaction_admission()(
            session_id=session_id,
            resolution=resolution,
            prompt=prompt,
        )
        if admission is None:
            return provider_messages, blocked(
                "Conversation changed before compaction could start."
            )
        branch_commit = self.read_controller__automatic_memory_admission()(
            session_id=session_id,
            snapshots=snapshots,
            plan=planned.plan,
            effective=effective,
            resolution=resolution,
            prompt=prompt,
        )
        if branch_commit is None:
            return provider_messages, blocked(
                "Conversation changed before compaction could start."
            )
        boundary_index = next(
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot.message_id == planned.plan.boundary_message_id
        )
        hooks = await self.read_controller__hooks_for_compaction()(
            session_id,
            resolution,
            reason="automatic",
            current=lambda: (
                self.read_controller__compaction_admission()(
                    session_id=session_id,
                    resolution=resolution,
                    prompt=prompt,
                )
                == admission
            ),
        )
        transaction = await service.compact(
            admission=admission,
            branch_commit=branch_commit,
            plan=planned.plan,
            resolution=resolution,
            prompt=prompt,
            current_admission=lambda: self.read_controller__compaction_admission()(
                session_id=session_id,
                resolution=resolution,
                prompt=prompt,
            ),
            prepare_main=prepare_main,
            prefix_messages=snapshots[: boundary_index + 1],
            retry_fence=self.read_global_compaction_retry_fence()(
                conversation_id,
                resolution,
                prompt,
                resolved,
                effective,
                snapshots,
                active_request=not manual_action,
            ),
            honor_failure_latch=micro_compaction or not manual_action,
            hooks=hooks,
        )
        if hooks is not None:
            try:
                await hooks.finish()
                await hooks.lifecycle.wait(hooks.owner)
            except Exception:  # noqa: BLE001 -- hook boundary
                return provider_messages, blocked(
                    "Required compaction hook failed; committed memory is retained."
                )
            finally:
                if hooks.reason == "manual" and (
                    hooks.lifecycle._handoff is None
                    or hooks.lifecycle._handoff[0] != hooks.owner
                ):
                    hooks.lifecycle.close_scope(hooks.owner)
        if transaction.terminal is self.read_global_CompactionTerminal().SUCCEEDED:
            memory_rows_after: tuple[
                self.read_global_Mapping()[str, self.read_global_Any()], ...
            ] = (
                self.read_global_tagged_memory_message()(
                    transaction.memory.summary_text
                ),
            )
            remaining_provenance = planned.plan.remaining_semantic.provenance
            text_memory_provenance = (
                self.read_global_compaction_transform_provenance()(
                    semantic.provenance,
                    selected_units=len(planned.plan.selected_units),
                    transform=self.read_global_TraceTransformKind().TEXT_COMPACTION,
                    source=self.read_global_TraceProvenanceSource().CONTEXT_SUMMARY,
                )
                if semantic.provenance is not None
                else None
            )
            hybrid_visual_added = False
            if (
                effective_representation
                is self.read_global_ContextCompactionRepresentation().HYBRID
            ):
                try:
                    image_limit = self.read_global_max_history_images()(
                        resolution.provider, resolution.model or ""
                    )
                    remaining_image_capacity = image_limit - count_semantic_images(
                        planned.plan.remaining_semantic
                    )
                    if remaining_image_capacity > 0:
                        artifact = await self.read_global_asyncio().to_thread(
                            render_visual_transcript,
                            planned.plan.selected_units,
                            summarized_prefix_digest=(
                                transaction.memory.summarized_prefix_digest
                            ),
                            max_pages=remaining_image_capacity,
                        )
                        visual_row = self.read_global_tagged_visual_memory_message()(
                            [page.png_bytes for page in artifact.pages],
                            # Wire integrity (exact PNG bytes), not renderer identity.
                            page_hashes=[page.png_sha256 for page in artifact.pages],
                        )
                        visual_memory_provenance = (
                            self.read_global_compaction_transform_provenance()(
                                semantic.provenance,
                                selected_units=len(planned.plan.selected_units),
                                transform=self.read_global_TraceTransformKind().HYBRID_COMPACTION,
                                source=self.read_global_TraceProvenanceSource().VISUAL_TRANSCRIPT,
                                include_memory=False,
                            )
                            if semantic.provenance is not None
                            else None
                        )
                        hybrid_semantic = self.read_global_replace()(
                            planned.plan.remaining_semantic,
                            memory=memory_rows_after + (visual_row,),
                            provenance=(
                                self.read_global_replace()(
                                    remaining_provenance,
                                    memory=(
                                        text_memory_provenance,
                                        visual_memory_provenance,
                                    ),
                                )
                                if remaining_provenance is not None
                                and text_memory_provenance is not None
                                and visual_memory_provenance is not None
                                else None
                            ),
                        )
                        hybrid_prepared = prepare_main(hybrid_semantic)
                        hybrid_conversation_tokens = (
                            hybrid_prepared.accounting.memory_tokens
                            + hybrid_prepared.accounting.compactable_tokens
                        )
                        if (
                            not hybrid_prepared.known_overflow
                            and hybrid_conversation_tokens
                            <= planned.plan.target_conversation_tokens
                        ):
                            memory_rows_after += (visual_row,)
                            hybrid_visual_added = True
                except Exception:
                    pass
                if not hybrid_visual_added:
                    self.read_global_logger().info(
                        "console_visual_compaction_fell_back_to_text"
                    )
            final_memory_provenance = (
                (
                    text_memory_provenance,
                    self.read_global_compaction_transform_provenance()(
                        semantic.provenance,
                        selected_units=len(planned.plan.selected_units),
                        transform=self.read_global_TraceTransformKind().HYBRID_COMPACTION,
                        source=self.read_global_TraceProvenanceSource().VISUAL_TRANSCRIPT,
                        include_memory=False,
                    ),
                )
                if hybrid_visual_added
                and semantic.provenance is not None
                and text_memory_provenance is not None
                else (
                    (text_memory_provenance,)
                    if text_memory_provenance is not None
                    else ()
                )
            )
            after = self.read_global_replace()(
                planned.plan.remaining_semantic,
                memory=memory_rows_after,
                provenance=(
                    self.read_global_replace()(
                        remaining_provenance, memory=final_memory_provenance
                    )
                    if remaining_provenance is not None
                    else None
                ),
            )
            return self.read_global__flatten_preflight_messages()(after), None
        omit = not manual_action and (
            resolved.policy.failure_behavior
            is self.read_global_CompactionFailureBehavior().OMIT_OLDER_CONTEXT
        )
        from .console_compaction_failure import transaction_failure_copy

        note = transaction_failure_copy(transaction, manual=manual_action, omitted=omit)
        if not omit:
            return provider_messages, blocked(note)
        if transaction.attempted:  # TASK-33621.3: disclose the billed call once.
            self.read_controller__append_failure_system_row()(session_id, note)
        return self.read_global__flatten_preflight_messages()(semantic), None

    async def _assess_context_compaction(
        self,
        *,
        session_id: str,
        resolution: ConsoleProviderResolution,
        provider_messages: list[dict[str, Any]],
        uncommitted_user_message_id: str | None,
    ) -> tuple[ContextCompactionHold | None, str | None]:
        """Probe whether this exact request would stop at Ask or cannot fit.

        Runs the real preflight as a side-effect-free assessment (TASK-34350).
        A probe failure never blocks the send: the stream preflight still
        runs and owns every refusal.

        Args:
            session_id: The sending session.
            resolution: The resolved provider for this send.
            provider_messages: The assembled request, current draft included.
            uncommitted_user_message_id: The send's own optimistic echo.

        Returns:
            ``(hold, alert)``: the hold numbers when the decision is Ask, and
            the alert copy when compacting cannot make the request fit.
        """

        captured: list[
            tuple[
                self.read_global_ContextCompactionHold(),
                self.read_global_CompactionDecision(),
                str | None,
            ]
        ] = []
        try:
            continuation_sidecar, continuation_target = (
                self.read_controller__provider_continuation_history_for_resolution()(
                    session_id, resolution
                )
            )
            await self.read_controller__apply_conversation_memory_preflight()(
                session_id=session_id,
                resolution=resolution,
                provider_messages=list(provider_messages),
                assistant_message_id="",
                agent_tools_enabled=(
                    self.read_controller__agent_runtime_enabled()
                    and self.read_controller__agent_bridge() is not None
                ),
                continuation_sidecar=continuation_sidecar,
                continuation_target=continuation_target,
                assessment_sink=lambda hold, decision, alert: captured.append(
                    (hold, decision, alert)
                ),
                uncommitted_user_message_id=uncommitted_user_message_id,
            )
        except Exception as exc:  # noqa: BLE001 - the stream preflight decides
            self.read_global_logger().warning(
                "Console compaction hold assessment unavailable; exception_type={}",
                type(exc).__name__,
            )
            return None, None
        if not captured:
            return None, None
        hold, decision, alert = captured[0]
        return (
            hold if decision is self.read_global_CompactionDecision().ASK else None
        ), alert

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
        """Report a request's capacity verdict when there is no history yet."""

        tools: list[self.read_global_Mapping()[str, self.read_global_Any()]] = []
        if agent_tools_enabled and self.read_controller__agent_bridge() is not None:
            preview = getattr(
                self.read_controller__agent_bridge(), "preview_tool_schemas", None
            )
            if callable(preview):
                try:
                    tools = list(preview())
                except Exception:
                    tools = []
        prepared = prepare(
            resolution, list(provider_messages), tools=tools, apply_safety_window=False
        )
        capacity = prepared.capacity
        try:
            global_overrides = self.read_controller__global_context_policy_overrides()()
        except Exception:
            global_overrides = None
        resolved = self.read_global_resolve_context_policy()(
            capacity=self.read_global_ConsoleContextCapacity()(
                model_context_window_tokens=capacity.context_window_tokens,
                model_window_verified=capacity.safety_verified,
                provider_input_cap_tokens=capacity.provider_input_cap_tokens,
                response_reservation_tokens=capacity.effective_response_tokens,
                safety_margin_tokens=capacity.safety_margin_tokens,
                mandatory_input_tokens=(
                    prepared.accounting.non_compactable_tokens
                    - prepared.accounting.memory_tokens
                ),
            ),
            global_overrides=global_overrides,
            conversation_overrides=owner.context_policy_overrides,
        )
        decision = self.read_global_decide_compaction()(
            resolved,
            conversation_tokens=(
                prepared.accounting.memory_tokens
                + prepared.accounting.compactable_tokens
            ),
            compactable_units=0,
        )
        budget = resolved.effective_conversation_budget_tokens or 0
        assessment_sink(
            self.read_global_ContextCompactionHold()(
                session_id=session_id,
                used_tokens=(
                    prepared.accounting.memory_tokens
                    + prepared.accounting.compactable_tokens
                ),
                trigger_tokens=int(budget * resolved.policy.trigger_ratio),
                budget_tokens=budget,
                estimated=False,
            ),
            decision,
            self.read_controller__context_overflow_alert()(
                decision, resolved, capacity, prepared, resolution
            ),
        )

    def _context_overflow_alert(
        self,
        decision: CompactionDecision,
        resolved: Any,
        capacity: Any,
        prepared_before: Any,
        resolution: ConsoleProviderResolution,
    ) -> str | None:
        """Alert copy for a request compacting cannot make fit, else None.

        Shared by the stream preflight and the pre-commit assessment so the
        two cannot disagree (TASK-34350).
        """

        if decision not in {
            self.read_global_CompactionDecision().UNKNOWN_WINDOW,
            self.read_global_CompactionDecision().NON_COMPACTABLE,
        }:
            return None
        if not prepared_before.known_overflow:
            return None
        if (
            resolved.policy.failure_behavior
            is self.read_global_CompactionFailureBehavior().OMIT_OLDER_CONTEXT
        ):
            return None
        # Lazy (ADR-097 UI-ready census): needed only when a send cannot fit.
        from tldw_chatbook.Chat.console_context_budget_copy import (
            MODEL_WINDOW_SETTING,
            context_overflow_alert_copy,
        )

        cause = self.read_global__context_overflow_cause()(decision, resolved, capacity)
        if cause is not None:
            return context_overflow_alert_copy(
                cause,
                model=resolution.model or "the selected model",
                window_tokens=capacity.context_window_tokens,
                window_estimated=capacity.limit_source == "estimated",
                response_tokens=capacity.effective_response_tokens,
                input_ceiling_tokens=capacity.effective_input_ceiling_tokens,
            )
        limiting_reason = (
            resolved.validation_errors[0]
            if resolved.validation_errors
            else "The effective model input ceiling is unavailable."
        )
        return (
            "This request cannot fit the selected model. "
            f"{limiting_reason} Summarizing older turns cannot make "
            "enough room. Set the model's context window in "
            f"{MODEL_WINDOW_SETTING}, or reduce mandatory context or "
            "the response maximum."
        )
