"""Run request assembly and admission with the service compatibility seams."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .agent_service import (
        AgentConfig,
        ContinuationOwnerGroup,
        CurrentRunActor,
        InstructionActivationLedger,
        InstructionChainPayloadState,
        InstructionDeliveryReceipt,
        ModelTurn,
        ProviderContinuationCheckpoint,
        ToolCall,
        ToolResult,
    )

#: Every tool-admission refusal (capacity or automatic budget) starts with this.
TOOL_ADMISSION_REFUSED = "tool call refused: "


def make_call_model(
    self,
    config: AgentConfig,
    api_endpoint: str,
    runtime_schemas: list,
    log_active: bool = False,
    continuation_groups: tuple[ContinuationOwnerGroup, ...] = (),
    continuation_owner_key: str | None = None,
    continuation_owner_message_id: str | None = None,
    *,
    trusted_role: Literal["primary", "subagent"] = "primary",
    project_instruction_context: InstructionActivationLedger | None = None,
    chain_id: str = "primary",
    payload_state: InstructionChainPayloadState | None = None,
    staged_delivery: dict[str, InstructionDeliveryReceipt] | None = None,
    on_context_assembled: Callable[[tuple[str, ...]], None] | None = None,
    input_budget_tokens: int | None = None,
    first_request_fits: bool = True,
    run_id: str | None = None,
):
    from tldw_chatbook.Agents import agent_service as runtime

    native = config.native_tools and runtime.provider_supports_native_tools(
        config.execution_provider or api_endpoint,
        reasoning_replay=config.reasoning_replay,
    )
    initial_context_checked = False
    context_observed = False
    staged = staged_delivery if staged_delivery is not None else {}
    # TASK-1272 (Phase 3): the ONLY gate on whether eviction may run at
    # all is (a) `log_active` -- the SAME condition, reused verbatim,
    # that gates the search_run_log tool and the prompt section above,
    # so eviction is never offered a run that has nothing durable to
    # recover from -- and (b) the opt-in `[agents] run_log_evict_
    # enabled` flag, off by default so existing runs stay byte-identical
    # until a user turns it on (requirement #5). Resolved once here,
    # not per turn: neither operand can change during a run.
    evict_requested = self._run_log_evict_enabled
    # TASK-1272 follow-up (live-verified 2026-07-28): the minimum-
    # recent-rounds floor, resolved once alongside evict_enabled for
    # the same reason -- it cannot change during a run. Unused when
    # evict_enabled is False, but resolving it unconditionally keeps
    # this closure's config reads in one place rather than split
    # across a conditional.
    min_recent_rounds = self._run_log_min_recent_rounds
    # task-245: one render per active-set change, not per turn. Keyed by
    # schema NAMES (the set only ever grows via load_tools — AC #2), and
    # scoped to this closure = this run, so sub-agents (their own
    # _run_one -> their own closure) never share a cache. Byte-stable
    # repeated turns are the precondition for provider-side prompt
    # caching (see Docs/superpowers/reviews/
    # 2026-07-17-provider-prompt-caching-note.md).
    protocol_key: tuple | None = None
    protocol_text = ""
    run_log_schema_names = {
        runtime.SEARCH_RUN_LOG_TOOL_SCHEMA.name,
        runtime.RUN_LOG_STATS_TOOL_SCHEMA.name,
        runtime.RUN_LOG_SLICE_TOOL_SCHEMA.name,
    }

    def call_model(
        messages: list[dict],
        active_schemas: tuple,
        current_continuation: ProviderContinuationCheckpoint | None = None,
    ) -> ModelTurn:
        nonlocal protocol_key, protocol_text, initial_context_checked, context_observed
        if (
            project_instruction_context is not None
            and payload_state is not None
            and not initial_context_checked
        ):
            payload_state.capture(messages, active_schemas, ())
            try:
                initial = project_instruction_context.initial_context_for_chain(
                    chain_id, payload_state
                )
            except Exception:  # noqa: BLE001 - content-free boundary
                raise runtime._ProjectInstructionPayloadError(
                    "project_instruction_delivery_failed"
                ) from None
            initial_context_checked = True
            if initial.status == "retry_with_context":
                messages.extend(dict(row) for row in initial.ephemeral_rows)
                staged["receipt"] = initial.delivery_receipt
        effective_log_active = bool(
            log_active
            and self.run_log_writer is not None
            and self.run_log_writer.is_active
        )
        effective_runtime_schemas = [
            schema
            for schema in runtime_schemas
            if effective_log_active or schema.name not in run_log_schema_names
        ]
        schemas = effective_runtime_schemas + list(active_schemas)
        runtime_names = {schema.name for schema in effective_runtime_schemas}
        runtime.logger.info(
            "agent schema request active={} runtime={} find_tools={} "
            "load_tools={} native={} run={}",
            len(active_schemas),
            len(effective_runtime_schemas),
            int(runtime.FIND_TOOLS_NAME in runtime_names),
            int(runtime.LOAD_TOOLS_NAME in runtime_names),
            int(native),
            run_id,
        )
        system_content = config.system_prompt
        call_kwargs: dict = {}
        if native:
            # Native mode: the provider carries the tool catalog in
            # tools= — no fence-protocol section in the system prompt.
            tools = runtime.schemas_to_openai_tools(schemas)
            if tools:
                call_kwargs["tools"] = tools
        else:
            key = tuple(repr(schema) for schema in schemas)
            if key != protocol_key:
                protocol_text = runtime.render_tool_protocol(schemas)
                protocol_key = key
            if protocol_text:
                system_content = f"{config.system_prompt}\n\n{protocol_text}"
        if effective_log_active:
            system_content = f"{system_content}\n\n{runtime.RUN_LOG_PROMPT_SECTION}"
        from .canvas_tool_provider import build_canvas_runtime_guidance

        canvas_guidance = build_canvas_runtime_guidance(
            schemas, messages=messages, system_prompt=config.system_prompt
        )
        if canvas_guidance:
            system_content = f"{system_content}\n\n{canvas_guidance}"
        from tldw_chatbook.Notes.agent_lessons import (
            build_agent_lessons_runtime_guidance,
        )

        from .agent_lesson_promotion import build_agent_lesson_promotion_guidance

        guidance = build_agent_lessons_runtime_guidance(
            schemas, trusted_role=trusted_role
        )
        if guidance:
            system_content = f"{system_content}\n\n{guidance}"
        promotion_guidance = build_agent_lesson_promotion_guidance(
            schemas,
            trusted_role=trusted_role,
            repository_target_enabled=(self.project_instruction_context is not None),
        )
        if promotion_guidance:
            system_content = f"{system_content}\n\n{promotion_guidance}"
        # Non-default workspace: append the environment note LAST, as a
        # stable per-turn suffix (cache-friendly, like the sections
        # above). ``_is_subagent`` prefix-matches the SENT system content;
        # keeping the identity prompt first leaves detection unaffected.
        system_content = runtime._append_workspace_context_note(
            system_content, config.workspace_context_note
        )
        system_content = runtime.append_personal_context(
            system_content, config.personal_context_block
        )
        # TASK-1272 (Phase 3): bound the SEND payload, never
        # `run_agent_loop`'s own `messages` -- that list is untouched,
        # see `bound_history_for_send`'s docstring. A no-op (returns
        # `raw_payload` unchanged) whenever `evict_enabled` is False.
        effective_groups = continuation_groups
        payload_messages = [dict(message) for message in messages]
        if current_continuation is not None:
            if not continuation_owner_key or not continuation_owner_message_id:
                raise ValueError("Active provider continuation requires an owner.")
            if any(
                continuation_owner_key in message
                and (
                    type(message[continuation_owner_key]) is not str
                    or not message[continuation_owner_key].strip()
                )
                for message in payload_messages
            ):
                raise ValueError("Provider continuation owner marker is malformed.")
            current_group = runtime.ContinuationOwnerGroup(
                owner_message_id=continuation_owner_message_id,
                checkpoint=current_continuation,
                rounds=current_continuation.rounds,
            )
            effective_groups = tuple(
                group
                for group in continuation_groups
                if group.owner_message_id != continuation_owner_message_id
            ) + (current_group,)
            native_batches = [
                [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in round_.calls
                ]
                for round_ in current_continuation.rounds
            ]
            anchor = next(
                (
                    index
                    for index in range(len(payload_messages) - 1, -1, -1)
                    if native_batches[-1]
                    and payload_messages[index].get("role") == "assistant"
                    and payload_messages[index].get("tool_calls") == native_batches[-1]
                ),
                None,
            )
            if anchor is None or (
                continuation_owner_key in payload_messages[anchor]
                and payload_messages[anchor][continuation_owner_key]
                != continuation_owner_message_id
            ):
                raise ValueError(
                    "Active provider continuation requires its exact native batch."
                )
            # The checkpoint owns all rounds, but attaches only at the newest
            # batch in this send copy. Leave malformed/duplicate/foreign echoes
            # marked so the existing request-owner guard still refuses them.
            for batch in native_batches[:-1]:
                echoes = [
                    message
                    for message in payload_messages[:anchor]
                    if batch
                    and message.get("role") == "assistant"
                    and message.get(continuation_owner_key)
                    == continuation_owner_message_id
                    and message.get("tool_calls") == batch
                ]
                if len(echoes) == 1:
                    echoes[0].pop(continuation_owner_key)
            payload_messages[anchor][continuation_owner_key] = (
                continuation_owner_message_id
            )
        raw_payload = [{"role": "system", "content": system_content}] + payload_messages
        if config.review_binding is not None:
            raw_payload = self._review_history_for_request(
                config, raw_payload, native=native, continuation_groups=effective_groups
            )
        else:
            raw_payload = self._prune_send_payload(raw_payload, native=native)
        gateway_prepares_continuation = bool(
            effective_groups and self.prepare_provider_continuation_request
        )
        payload = runtime.bound_history_for_send(
            raw_payload,
            model=config.model,
            provider=api_endpoint,
            native=native,
            enabled=effective_log_active and evict_requested,
            min_recent_rounds=min_recent_rounds,
            count_fn=lambda rows, _model: runtime._count_model_messages(
                rows,
                config.model,
                api_endpoint,
                reasoning_replay=config.reasoning_replay,
            ),
            continuation_groups=(
                () if gateway_prepares_continuation else effective_groups
            ),
            continuation_owner_key=continuation_owner_key or "id",
        )
        if continuation_owner_key is not None and not gateway_prepares_continuation:
            payload = [
                {
                    key: value
                    for key, value in message.items()
                    if key != continuation_owner_key
                }
                for message in payload
            ]
        if gateway_prepares_continuation:
            call_kwargs["continuation_groups"] = effective_groups
        receipt = staged.get("receipt")
        if receipt is not None:
            request = runtime.ModelRequest(
                messages=tuple(dict(message) for message in payload),
                tools=tuple(call_kwargs.get("tools", ())),
            )
            request_fits = self._project_instruction_request_fits(
                config,
                api_endpoint,
                request,
                input_budget_tokens=input_budget_tokens,
            )
            row_keys = tuple(
                row.get(runtime.PROJECT_INSTRUCTION_ROW_KEY)
                for row in payload
                if row.get(runtime.PROJECT_INSTRUCTION_ROW_KEY) in receipt.row_keys
            )
            if not request_fits or row_keys != receipt.row_keys:
                raise runtime._ProjectInstructionPayloadError(
                    "project instruction context could not fit"
                )
            assert project_instruction_context is not None
            try:
                project_instruction_context.mark_payload_sent(receipt, payload)
            except Exception:  # noqa: BLE001 - content-free boundary
                raise runtime._ProjectInstructionPayloadError(
                    "project_instruction_delivery_failed"
                ) from None
            staged.pop("receipt", None)
        if (
            config.preset_fallback_targets or input_budget_tokens is not None
        ) and not self._project_instruction_request_fits(
            config,
            config.execution_provider or api_endpoint,
            runtime.ModelRequest(
                messages=tuple(dict(row) for row in payload),
                tools=tuple(call_kwargs.get("tools", ())),
            ),
            input_budget_tokens=input_budget_tokens,
        ):
            raise runtime._ProjectInstructionPayloadError(
                "request exceeds model context budget"
            )
        if not first_request_fits:
            raise runtime._ProjectInstructionPayloadError(
                "first request exceeds model context budget"
            )
        if on_context_assembled is not None and not context_observed:
            categories = ["system"]
            if config.workspace_context_note:
                categories.append("workspace")
            if any(
                row.get(runtime.PROJECT_INSTRUCTION_ROW_KEY)
                or row.get(runtime.EPHEMERAL_ORIGIN_KEY)
                for row in payload
            ):
                categories.insert(0, "project")
            try:
                on_context_assembled(tuple(categories))
            except Exception:  # noqa: BLE001 — capture never fails a request
                runtime.logger.warning("agent_context_capture_failed")
            context_observed = True
        # Review F1: while a checkpointed call is on the wire the
        # redirect stream-cut is disarmed (see _primary_cut_suppress).
        suppress_cut = current_continuation is not None and run_id is not None
        if suppress_cut:
            self._primary_cut_suppress.add(run_id)
        try:
            # ADR-147 (Task 6): this run's OWN resolved sampling params
            # and base_url ride the call -- for a spawned child they are
            # the resolver's target (never the parent's params, which
            # simply are not on its config); both stay absent for a
            # primary, leaving every pre-existing call byte-identical.
            # Keys are drawn from KNOWN_SAMPLING_PARAM_KEYS by the
            # upstream validators, so the map lookup is total.
            for param_key, param_value in config.sampling_params:
                for request_key in runtime.GENERATION_FIELD_REQUEST_KEYS[param_key]:
                    call_kwargs[request_key] = param_value
            if config.base_url:
                call_kwargs["api_base_url"] = config.base_url
            if config.execution_provider:
                call_kwargs["execution_provider"] = config.execution_provider
            checked_payload = runtime.check_host_context(payload)
            review_capture = getattr(self.registry, "_review_service", None)
            admission_scope = runtime.contextlib.nullcontext()
            if config.review_binding is not None:
                from tldw_chatbook.Chat.console_prepared_request import (
                    PreparedProviderRequest,
                )

                from .code_review_models import ReviewRefused

                def admit_review_request(request):
                    if isinstance(request, PreparedProviderRequest):
                        capacity = request.capacity
                        known = (
                            capacity.limit_source
                            in {"detected", "user_override", "provider_input_cap"}
                            or capacity.provider_input_cap_tokens is not None
                        )
                        remaining = (
                            capacity.effective_input_ceiling_tokens
                            - request.accounting.total_input_tokens
                            if known
                            and capacity.effective_input_ceiling_tokens is not None
                            else None
                        )
                        provider, model = request.provider, request.model
                    else:
                        remaining = self.review_input_capacity(
                            config, api_endpoint, request
                        )
                        provider, model = api_endpoint, config.model
                    if remaining is None:
                        review_capture.record_verification_gap(
                            config.review_binding.review_id,
                            "review_context_capacity_unknown",
                        )
                        raise ReviewRefused("review_context_capacity_unknown")
                    if remaining < 0:
                        review_capture.record_verification_gap(
                            config.review_binding.review_id,
                            "review_context_reduction_blocked",
                        )
                        raise ReviewRefused("review_context_reduction_blocked")
                    if run_id is not None:
                        review_capture.store.observe_review_route(
                            run_id, config.provider or provider, model
                        )
                    review_capture.verifier_input_tokens = remaining
                    review_capture.verifier_required_continuation = isinstance(
                        request, PreparedProviderRequest
                    ) and bool(request.continuation_groups)
                    review_capture.verifier_estimate = lambda text: (
                        runtime.estimate_tokens(text, model, provider=provider)
                    )
                    prior_capacity = review_capture.capacity()
                    review_capture.capacity = lambda: runtime.dataclasses.replace(
                        prior_capacity, remaining_input_tokens=remaining
                    )

                gateway_owner = getattr(self.chat_call, "__self__", None)
                scoped_admission = getattr(
                    gateway_owner, "review_request_admission", None
                )
                if callable(scoped_admission):
                    admission_scope = scoped_admission(admit_review_request)
                else:
                    if effective_groups:
                        review_capture.record_verification_gap(
                            config.review_binding.review_id,
                            "review_context_reduction_blocked",
                        )
                        raise ReviewRefused("review_context_reduction_blocked")
                    admit_review_request(
                        runtime.ModelRequest(
                            messages=tuple(checked_payload),
                            tools=tuple(call_kwargs.get("tools", ())),
                        )
                    )
            with admission_scope:
                resp = self.chat_call(
                    api_endpoint=config.provider or api_endpoint,
                    messages_payload=checked_payload,
                    streaming=False,
                    model=config.model,
                    **call_kwargs,
                )
        finally:
            if suppress_cut:
                self._primary_cut_suppress.discard(run_id)
        text = runtime._response_text(resp)
        if review_capture is not None and run_id is not None:
            review_capture.capture_review_output(run_id, text, terminal=False)
        # TASK-18603: cache-aware. Identical to the flat sum whenever
        # this turn read nothing from a prompt cache.
        tokens = runtime._budget_weighted_tokens(
            resp, provider=api_endpoint, model=config.model
        )
        provider_continuation = getattr(resp, "provider_continuation", None)
        if provider_continuation is not None and not isinstance(
            provider_continuation, runtime.ProviderContinuationCheckpoint
        ):
            raise ValueError("Provider continuation metadata is malformed.")
        if tokens is None:
            # Provider reported no usage -> estimate from sent payload +
            # response text (native tool_calls JSON is not separately
            # counted here; the prompt term dominates the per-turn total).
            # Strip a provider prefix ("openai/gpt-4o-mini" -> "gpt-4o-mini")
            # so the tokenizer's model-family framing detection matches, and
            # pass the endpoint as the provider hint for the chars ratio.
            est_model = (
                config.model.split("/", 1)[-1] if "/" in config.model else config.model
            )
            tokens = runtime._count_model_messages(
                payload,
                config.model,
                api_endpoint,
                reasoning_replay=config.reasoning_replay,
                tokenizer_model=est_model,
            ) + runtime.estimate_tokens(text, est_model, provider=api_endpoint)
        thinking_envelope = getattr(resp, "thinking_envelope", None)
        thinking_echo = None
        if thinking_envelope is not None:
            from tldw_chatbook.Chat.console_thinking_capture import (
                CALL_THINKING_KEY,
            )
            from tldw_chatbook.Chat.thinking_blocks import ThinkingEnvelope

            if not isinstance(thinking_envelope, ThinkingEnvelope):
                raise ValueError("Call thinking metadata is malformed.")
            thinking_echo = {
                "role": "assistant",
                "content": text,
                CALL_THINKING_KEY: thinking_envelope,
            }
        if not native:
            return runtime.ModelTurn(
                text=text,
                assistant_message=thinking_echo,
                tokens=tokens,
                provider_continuation=provider_continuation,
            )
        message = runtime._response_message(resp)
        # Id-less entries get synthesized ids BEFORE parsing, and the
        # SAME normalized list feeds the assistant echo — the echo and
        # its role="tool" replies must always pair by id (PR #648
        # review: a split convention 400s on strict providers).
        raw_calls = runtime.ensure_tool_call_ids(message.get("tool_calls"))
        if raw_calls:
            message = {**message, "tool_calls": raw_calls}
        tool_calls = runtime.parse_native_tool_calls(message)
        assistant_message = thinking_echo
        if tool_calls:
            assistant_message = {
                **(thinking_echo or {}),
                "role": "assistant",
                "content": text,
                "tool_calls": raw_calls,
            }
            if (
                provider_continuation is not None
                and continuation_owner_key
                and continuation_owner_message_id
            ):
                assistant_message[continuation_owner_key] = (
                    continuation_owner_message_id
                )
        return runtime.ModelTurn(
            text=text,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            tokens=tokens,
            provider_continuation=provider_continuation,
        )

    return call_model


def make_invoke_tool(
    self,
    config: AgentConfig,
    disclosed_names: set,
    should_cancel: Callable[[], bool] = lambda: False,
    *,
    run_id: str,
    run_actor: CurrentRunActor | None = None,
    validate_prepared_call: Callable | None = None,
    install_post_checkpoint: Callable | None = None,
):
    """Build this run's ``LoopDeps.invoke_tool``.

    Args:
        config: This run's config (allow-list and budgets).
        disclosed_names: The tool names whose schemas this run has.
        should_cancel: Cooperative cancellation probe.
        run_id: THIS run's id, used for the legacy primary-actor fallback.
        run_actor: Exact primary/subagent attribution, bound via
            ``run_context.use_run_actor`` around each invocation so the
            permission gates can find
            this run's own approval stamps (PR2a Task 5). Bound
            INSIDE the callable handed to ``_call_with_timeout``, so
            it is established on the per-call daemon thread that
            actually runs the tool -- a ``ContextVar`` set on this
            thread would not be visible there.

            Keyword-REQUIRED, with no default, deliberately: a
            defaulted ``""`` would turn a future caller that forgets
            it (PR2a Task 6 adds dispatch paths) into a SILENT
            fail-closed degradation -- every approved tool refused
            for want of a stamp it can no longer find -- instead of
            a loud ``TypeError`` at the call site.
    """
    from tldw_chatbook.Agents import agent_service as runtime

    from .execution_capacity import CapacityRefused, current_execution_owner

    # Captured as the deps are built, which is immediately before the
    # loop records its own `started`. Being marginally early makes the
    # remaining-budget clamp conservative, which is the safe direction.
    run_started = self.clock()

    actor = run_actor or runtime.CurrentRunActor("primary", run_id, None)

    owner = current_execution_owner()

    def invoke_tool(call: ToolCall) -> ToolResult:
        if config.review_binding is not None and call.name not in {
            "code_review_read",
            "code_review_submit",
        }:
            return runtime.ToolResult(
                ok=False, error="role_denied", dispatch_state="not_started"
            )
        if should_cancel():
            return runtime.ToolResult(
                ok=False,
                error="tool call cancelled before start",
                outcome=runtime.TOOL_OUTCOME_CANCELLED,
                dispatch_state="not_started",
            )
        refusal = runtime._tool_availability_refusal(
            call.name, config.allowed_tools, disclosed_names, self.registry
        )
        if refusal is not None:
            return refusal

        def _invoke() -> ToolResult:
            from .automatic_work_runtime import current_automatic_work

            with runtime.use_run_actor(actor), runtime.use_tool_call_id(call.call_id):
                automatic_work = current_automatic_work()
                if automatic_work is not None:
                    from .automatic_work_budget import AutomaticWorkRefused

                    try:
                        automatic_work.check()
                    except AutomaticWorkRefused:
                        return runtime.ToolResult.blocked(
                            f"{TOOL_ADMISSION_REFUSED}automatic work refused before dispatch"
                        )
                review_owner = self.registry.resolve_owner_for_name(
                    "code_review_prepare"
                )
                check_scope = runtime.contextlib.nullcontext()
                if review_owner is not None:
                    try:
                        check_scope = review_owner[1].service.prepare_check_process(
                            call
                        )
                    except Exception:  # noqa: BLE001 - specification admission must fail closed
                        return runtime.ToolResult.blocked(
                            "check execution specification refused"
                        )
                if validate_prepared_call is not None:
                    try:
                        definition = validate_prepared_call(call)
                    except Exception:  # noqa: BLE001 -- fail closed
                        return runtime.ToolResult.blocked(
                            "hook: final dispatch identity refused"
                        )
                    with check_scope:
                        result = self.registry.invoke_by_name(
                            call.name, call.args, expected_definition=definition
                        )
                else:
                    with check_scope:
                        result = self.registry.invoke_by_name(call.name, call.args)
                return (
                    runtime.dataclasses.replace(result, dispatch_state="settled")
                    if result.dispatch_state is None
                    else result
                )

        if (
            self.registry.execution_policy_for(call.name)
            is runtime.ToolExecutionPolicy.DEFINITIVE_AFTER_START
        ):
            if should_cancel():
                return runtime.ToolResult(
                    ok=False,
                    error=f"tool call cancelled before start: {call.name}",
                    outcome=runtime.TOOL_OUTCOME_CANCELLED,
                    dispatch_state="not_started",
                )
            try:
                operation = owner.reserve_tool() if owner is not None else None
            except CapacityRefused as exc:
                return runtime.ToolResult.blocked(f"{TOOL_ADMISSION_REFUSED}{exc}")
            result = runtime.ToolResult(
                ok=False, error=f"tool call failed: {call.name}"
            )
            try:
                try:
                    result = _invoke()
                except BaseException:  # noqa: BLE001 - protocol boundary
                    # This branch runs synchronously after the approved
                    # mutation has started.  Every terminal -- including
                    # control-flow BaseExceptions raised by a provider --
                    # must collapse to one scrubbed result instead of
                    # escaping the agent loop or being mistaken for the
                    # pre-start cooperative-cancel outcome above.
                    result = runtime.ToolResult(
                        ok=False,
                        error=f"tool call failed: {call.name}",
                        dispatch_state="settled",
                    )
                return result
            finally:
                try:
                    if install_post_checkpoint is not None:
                        install_post_checkpoint(call, result)
                finally:
                    if operation is not None:
                        operation.finish()
                self._notify_tool_terminal(
                    run_id,
                    call.call_id or call.name,
                    call.name,
                )
                self._notify_tool_result_terminal(
                    run_id,
                    call.call_id or call.name,
                    call.name,
                    result,
                )

        timeout = self.registry.timeout_for(call.name) or (
            config.budget.max_tool_call_seconds
        )
        if timeout and timeout > 0:
            # TASK-25913: the loop only checks max_wall_seconds between
            # iterations, so a call left on its own ceiling could outlive
            # the run's budget by nearly an hour under Console's config.
            # Resolved once here, at dispatch -- an approval wait that
            # happens afterwards must not shrink a running call.
            timeout, clamped_by_wall_budget = runtime._effective_tool_timeout(
                timeout,
                run_started,
                getattr(config.budget, "max_wall_seconds", 0),
                self.clock,
            )
            if timeout is None:
                # Budget already spent: refuse rather than start work that
                # would be abandoned mid-flight and could still take effect.
                return runtime.ToolResult(
                    ok=False,
                    error=(
                        f"tool call not started: {call.name} "
                        "(the run's wall-clock budget is exhausted)"
                    ),
                    outcome=runtime.TOOL_OUTCOME_TIMEOUT,
                    dispatch_state="not_started",
                )
            # task-31386: a per-call abandon request rides the same
            # poll as the run-wide Stop, but only inside this wrapper --
            # `should_cancel` itself stays False, so the loop continues.
            runtime._mark_tool_call_inflight(
                run_id, call.call_id or call.name, call.name
            )
            try:
                return runtime._call_with_timeout(
                    _invoke,
                    timeout,
                    call.name,
                    lambda: (
                        should_cancel() or runtime.tool_call_abandon_requested(run_id)
                    ),
                    # ADR-067: pause the per-call clock while a human
                    # decision is pending for THIS run, so an approval/
                    # confirm wait inside the invoke outlives the ceiling.
                    pauses_deadline=lambda: runtime.human_input_wait_active(run_id),
                    execution_owner=self,
                    clamped_by_wall_budget=clamped_by_wall_budget,
                    owner=owner,
                )
            finally:
                runtime._clear_tool_call_inflight(run_id)
        try:
            operation = owner.reserve_tool() if owner is not None else None
        except CapacityRefused as exc:
            return runtime.ToolResult(
                ok=False,
                error=f"{TOOL_ADMISSION_REFUSED}{exc}",
                dispatch_state="not_started",
            )
        try:
            return _invoke()
        finally:
            if operation is not None:
                operation.finish()

    check_owner = self.registry.resolve_owner_for_name("code_review_read")
    if (
        self.post_tool_dispatch is None
        and install_post_checkpoint is None
        and check_owner is None
    ):
        # AC#5: no observer, no wrapper -- the closure above is returned
        # untouched, so an unconfigured install pays nothing.
        return invoke_tool

    def observed_invoke_tool(call: ToolCall) -> ToolResult:
        observation_started = self.clock()
        result = invoke_tool(call)
        if install_post_checkpoint is not None:
            install_post_checkpoint(call, result)
        self._fire_post_tool_dispatch(
            call, result, self.clock() - observation_started, run_id
        )
        return result

    return observed_invoke_tool


def settle_review_capture(
    self,
    service,
    run_id: str,
    receipt_id: str | None = None,
    *,
    close_capture: bool = True,
) -> None:
    """Close only an actually associated attempt after native status persistence.

    Never raises: callers are teardown seams (a fleet child's ``finally``
    right before its settled signal, end-of-turn abandonment, ``_persist``),
    and a store/DB/UI failure here must not cost them their next step.
    """
    from tldw_chatbook.Agents import agent_service as runtime

    try:
        binding = service.store._tab.runs.get(run_id)
        if binding is None or binding.role == "primary":
            return
        row = self.db.get_run(run_id)
        status = row["status"]
        if status not in runtime.TERMINAL_RUN_STATUSES:
            return
        # A prior cancellation wins over a later outcome's terminal step.
        if receipt_id is None or (
            receipt_id
            and not any(
                f"agent-step:{run_id}:{step['index']}" == receipt_id
                and step.get("status") == status
                for step in row["steps"]
            )
        ):
            receipt_id = next(
                (
                    f"agent-step:{run_id}:{step['index']}"
                    for step in reversed(row["steps"])
                    if step.get("status") == status
                    and step.get("kind") == self._terminal_lifecycle_kind(status)
                ),
                "",
            )
        with (
            service.store._tab.lock,
            service.store.lease(binding.review_id) as context,
        ):
            stream = context.streams[(binding.slot_id, binding.attempt_id)]
            if stream.terminal_status is None:
                if close_capture and not stream.capture_terminal:
                    service.store.capture_review_output(run_id, "", terminal=True)
                service.settle_review_run(
                    run_id, terminal_status=status, receipt_id=receipt_id
                )
        # No review-store lock may cross the wake/DB notification boundary.
        if service.native_coordinator is not None:
            service.native_coordinator.on_settled(binding.review_id)
    except Exception as exc:  # noqa: BLE001 - never break the caller's teardown
        runtime.logger.warning(
            "review_terminal_capture_unavailable run_id={} error_type={}",
            run_id,
            runtime._safe_exception_type(exc),
        )


def end_review_checks(
    self,
    run_id: str,
    reason: str,
    *,
    unstarted_only: bool = False,
    denied_call=None,
) -> None:
    """Signal this run's pending review checks; never raises.

    Callers are containment seams (end-of-turn settling, post-dispatch
    observation): a check-store failure must not skip the fleet settle or
    discard an already executed tool result. With ``denied_call``, checks end
    only when that call is the claimed check's own process.
    """
    from tldw_chatbook.Agents import agent_service as runtime

    try:
        owner = self.registry.resolve_owner_for_name("code_review_read")
        if owner is None:
            return
        from .code_review_tool_provider import CodeReviewToolProvider

        if type(owner[1]) is not CodeReviewToolProvider:
            return
        service = owner[1].service
        if denied_call is None or service.is_claimed_check_call(run_id, denied_call):
            service.end_check_actor(run_id, reason, unstarted_only=unstarted_only)
    except Exception as exc:  # noqa: BLE001 - never break the caller's seam
        runtime.logger.warning(
            "could not end review checks run_id={} error_type={}",
            run_id,
            runtime._safe_exception_type(exc),
        )


def end_checks_after_refusal(self, call, result, run_id: str) -> None:
    """End this run's unstarted review checks only when ``call`` could not start.

    The design's check section has an actual tool-admission refusal resolve
    unstarted checks (including when a waiting verifier occupies the only tool
    worker) and a denial of the claimed execution release the wait. Every
    other failure leaves the claim pending: review-tool errors the verifier
    guide says to retry (``invalid_source_range``,
    ``submission_revision_conflict``), the check gate's block of a
    non-matching ``shell_exec`` (the very claim it protects) and unrelated
    tool failures or denials. A started claimed execution settles through its own
    activation scope; primary finalization ends whatever remains.
    """
    if result.ok:
        return
    reason = result.error or "tool_unavailable"
    if (result.error or "").startswith(TOOL_ADMISSION_REFUSED):
        end_review_checks(self, run_id, reason, unstarted_only=True)
    elif result.outcome == "review_denied":
        # Only a denial of the claimed check's own argv-exact process
        # (CodeReviewService.is_claimed_check_call); any other denied call,
        # including an unrelated shell_exec, leaves the claim pending.
        end_review_checks(self, run_id, reason, unstarted_only=True, denied_call=call)


def attach_review_launch(review_service, review_launch, run_id: str) -> None:
    """Attach a review launch; undo a half-applied attach this call made.

    ``attach_review_run`` binds the attempt (setting ``run_id``) before a
    verifier's ``bind_check_actor``. When that refuses, no model turn has run,
    so the bound attempt is retired through ``retire_unstarted_launch`` and the
    claim unbound: callers then unwind an unstarted launch and the slot can
    launch again. The refusal is re-raised either way.
    """
    from .code_review_models import ReviewRefused

    bound_before = review_launch.run_id
    try:
        review_service.attach_review_run(review_launch, run_id)
    except ReviewRefused:
        if bound_before is None and review_launch.run_id == run_id:
            with contextlib.suppress(ReviewRefused):
                review_service.retire_unstarted_launch(review_launch, run_id)
                review_launch.run_id = None
        raise
