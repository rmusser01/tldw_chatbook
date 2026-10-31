"""Streaming Console request transport; the adapter owns lifecycle and state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .console_agent_bridge import (
        ConsoleProviderCallSignals,
        ConsoleProviderStreamSignals,
        ContinuationOwnerGroup,
        ProviderTurnMetadata,
    )


def chat_call_impl(
    self,
    *,
    messages_payload,
    model=None,
    api_endpoint=None,
    streaming=False,
    tools=None,
    continuation_groups: tuple[ContinuationOwnerGroup, ...] = (),
    **per_call_kwargs,
) -> dict:
    """Stream one turn, honoring ADR-147 per-call routing kwargs.

    Task 6 emits a spawned child's resolved target onto every call:
    ``api_endpoint`` (raw provider id, possibly ``custom-ep:<slug>``),
    ``model``, ``api_base_url``, and the 13 sampling kwargs mapped by
    ``_PER_CALL_SAMPLING_RESOLUTION_FIELDS``. An ``api_endpoint`` naming
    a DIFFERENT provider re-resolves through the gateway's own send
    seam (``resolve_for_send``) so the bytes stream from the child's
    provider; a call on the parent's provider overlays only what it
    carries onto a per-call copy of the parent resolution. A call
    carrying none of these behaves exactly as before, on
    ``self._resolution`` itself -- which is never mutated, since the
    adapter is shared across concurrent children. An unknown/unready
    per-call provider raises through the same channel as a stream
    failure, never a silent fallback to the parent. A routed call on a
    continuation-pinned turn raises the typed ``ContinuationConflictError``
    from request preparation (the restore target pins the parent's
    provider/model/base_url) -- loud, through that same channel.
    """
    from tldw_chatbook.Chat import console_agent_bridge as bridge

    review_admission = getattr(self._thread_loop, "review_admission", None)
    review_prepare_kwargs = (
        {"apply_safety_window": False} if review_admission is not None else {}
    )
    attributed = getattr(self._thread_loop, "run_scope", None)
    usage_sequence = getattr(self._thread_loop, "live_usage_sequence", None)
    if attributed is not None:
        usage_run_id, usage_agent_kind = attributed
    sampling_overlays = {
        field: per_call_kwargs.pop(kwarg)
        for kwarg, field in bridge._PER_CALL_SAMPLING_RESOLUTION_FIELDS.items()
        if per_call_kwargs.get(kwarg) is not None
    }
    api_base_url = per_call_kwargs.pop("api_base_url", None)
    execution_provider = str(per_call_kwargs.pop("execution_provider", "") or "")
    # Any keys still in per_call_kwargs stay ignored, exactly as before.
    # Same-target decision keys on the parent's RAW selection identity,
    # not the flattened execution key: a child explicitly routed to the
    # built-in family of a custom-ep parent (e.g. `llama_cpp` under
    # `custom-ep:qwen-local`) is a DIFFERENT target and must resolve the
    # built-in provider independently, while an inheriting child names
    # the raw slug and reuses this resolution (qodo PR-2651 High).
    # Plain providers leave ``selected_provider`` empty and fall through
    # to the historical execution_key/provider chain unchanged.
    parent_selected_provider = str(
        getattr(self._resolution, "selected_provider", "") or ""
    )
    parent_execution_key = str(getattr(self._resolution, "execution_key", "") or "")
    parent_endpoint = (
        parent_selected_provider
        or parent_execution_key
        or getattr(self._resolution, "provider", "")
        or ""
    )
    requested_endpoint = str(api_endpoint or "").strip()
    # "agent" is build_console_first_request_plan's no-identity
    # fallback, not a provider id: it names THIS resolution by
    # construction.
    if requested_endpoint == "agent":
        requested_endpoint = ""
    # Fourth spelling seam (ADR-179 Qodo follow-up): the bridge-level
    # send names the parent's OWN flattened execution key (the first
    # request plan puts ``execution_key`` on every run-turn call), which
    # for a swapped custom-ep selection is an execution-ONLY spelling
    # ("custom-hosted" under "custom-ep:<slug>") that no caller could
    # have selected as a standalone target -- re-resolving it as a
    # provider id fails "Unknown provider". The resolution's
    # selected_provider<->execution_key pairing is therefore same-target
    # ONLY for spellings Console cannot select independently; a family
    # key that IS selectable ("llama_cpp" under "custom-ep:qwen-local")
    # keeps qodo PR-2651 High's reroute protection.
    same_target = requested_endpoint == parent_endpoint or (
        bool(parent_selected_provider)
        and requested_endpoint == parent_execution_key
        and requested_endpoint not in bridge.CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
    )
    rerouted = bool(requested_endpoint) and (
        not same_target
        or bool(execution_provider and execution_provider != parent_execution_key)
    )

    transport_messages = bridge._serialize_project_instruction_rows_for_transport(
        messages_payload, native_tools=self._native_tools
    )
    is_subagent = self._is_subagent(transport_messages)
    # Resolve sidecars after the trace factory has built the neutral request.
    # The reserved canonical key is valid only once its groups are attached.
    # This temporary scalar also keeps mandatory continuation ownership separate.
    call_thinking_owner_key = "_tldw_call_thinking_owner"
    if not is_subagent and self._thinking_sidecar and self._thinking_owner_key:
        historical_owners = {
            sidecar.owner_message_id for sidecar in self._thinking_sidecar
        }
        for row in transport_messages:
            historical_owner = row.get(self._thinking_owner_key)
            if (
                isinstance(historical_owner, str)
                and historical_owner in historical_owners
            ):
                row[call_thinking_owner_key] = historical_owner
            if self._thinking_owner_key not in {
                call_thinking_owner_key,
                self._continuation_owner_key,
            }:
                row.pop(self._thinking_owner_key, None)
    transport_messages, call_sidecars = bridge.consume_call_thinking(
        transport_messages, owner_key=call_thinking_owner_key
    )
    call_thinking_sidecar = (
        () if is_subagent else self._thinking_sidecar
    ) + call_sidecars
    call_capture = bridge.ThinkingCapture(assistant_owner_id=bridge.new_opaque_id())
    # TASK-28227: a redirect aborts only the PRIMARY's in-flight
    # model stream. Children keep the plain cancel predicate --
    # cutting a fleet child's stream on a primary redirect would
    # truncate its turn with no redirect entry in ITS mailbox to
    # explain it. Never wired into LoopDeps.should_cancel: aborting
    # is not cancelling.
    stream_cut = (
        self._should_cancel
        if is_subagent
        else (lambda: self._should_cancel() or self._primary_stream_abort())
    )
    request_count = int(getattr(self._thread_loop, "request_count", 0))
    self._thread_loop.request_count = request_count + 1
    route = (
        bridge.ConsoleRequestRoute.AGENT_FIRST
        if request_count == 0
        else bridge.ConsoleRequestRoute.TOOL_LOOP
    )
    route_actor_id = getattr(self._thread_loop, "route_actor_id", None)
    route_chain_id = getattr(self._thread_loop, "route_chain_id", None)
    if route_actor_id is None or route_chain_id is None:
        recover_route = getattr(self._gateway, "_trace_recovery_route_identity", None)
        recovered = (
            recover_route(self._provider_stream_signals)
            if request_count == 0 and not is_subagent and callable(recover_route)
            else None
        )
        if recovered is None:
            route_actor_id = bridge.new_opaque_id()
            route_chain_id = bridge.new_opaque_id()
        else:
            route_actor_id, route_chain_id = recovered
        self._thread_loop.route_actor_id = route_actor_id
        self._thread_loop.route_chain_id = route_chain_id
    # Named definitions can override the model while staying on the
    # parent's provider endpoint. The adapter is shared by concurrent
    # parent/child turns, so keep that override immutable and call-local.
    call_resolution = self._resolution
    if model and model != self._resolution.model:
        # Endpoint/model metadata is frozen for the primary target; it is
        # not rediscovered for overrides during a running agent loop.
        call_resolution = bridge.dataclass_replace(
            self._resolution,
            model=model,
            reasoning_replay=None,
            local_structured_thinking=False,
        )
    # ADR-147 Task 6B: per-call sampling/base-url kwargs overlay the
    # call-local resolution (model already handled above). Only the
    # dispatch/stream/usage paths read ``effective_resolution``; a
    # rerouted call re-resolves entirely inside ``_consume``.
    effective_resolution = call_resolution
    if (
        not rerouted
        and isinstance(call_resolution, bridge.ConsoleProviderResolution)
        and (
            sampling_overlays
            or (api_base_url and api_base_url != call_resolution.base_url)
        )
    ):
        # Resolved child params are a complete stack. Clear optional
        # parent-only knobs that are absent from it; legacy calls with no
        # sampling kwargs retain their existing resolution behavior.
        overlays = (
            dict.fromkeys(bridge._PER_CALL_SAMPLING_RESOLUTION_FIELDS.values())
            if is_subagent and sampling_overlays
            else {}
        )
        overlays.update(sampling_overlays)
        if api_base_url and api_base_url != call_resolution.base_url:
            overlays["base_url"] = str(api_base_url)
        if overlays:
            effective_resolution = bridge.dataclass_replace(call_resolution, **overlays)
    call_continuation_target = self._continuation_target
    gate = bridge.StreamGate()
    any_streamed = False
    received_item = False
    native_calls: list[dict] = []
    terminal_metadata: ProviderTurnMetadata | None = None
    call_signals: ConsoleProviderCallSignals | None = None
    live_provider_usage_enabled = True
    gateway_signals: (
        ConsoleProviderStreamSignals | ConsoleProviderCallSignals | None
    ) = self._provider_stream_signals
    if isinstance(self._gateway, bridge.ConsoleProviderGateway):
        aggregate_signals = (
            self._provider_stream_signals or bridge.ConsoleProviderStreamSignals()
        )
        call_signals = aggregate_signals.new_usage_call()
        gateway_signals = call_signals

    @bridge.agent_async_worker_guard(self)
    async def _consume() -> None:
        nonlocal \
            any_streamed, \
            terminal_metadata, \
            effective_resolution, \
            live_provider_usage_enabled
        nonlocal received_item, call_continuation_target
        if rerouted:
            # Task 6B: the child names a provider other than the
            # parent's -- re-resolve through the gateway's send seam
            # BEFORE anything prepares or streams, so the whole turn
            # (request preparation included) speaks to the child.
            effective_resolution = await self._resolve_routed_resolution(
                provider=requested_endpoint,
                model=model,
                api_base_url=api_base_url,
                sampling_overlays=sampling_overlays,
                execution_provider=execution_provider,
            )
        window = getattr(effective_resolution, "context_window", None)
        resolve_window = getattr(self._gateway, "resolve_context_window", None)
        if (
            review_admission is not None
            and window is not None
            and not window.verified
            and callable(resolve_window)
        ):
            # Sends attach the cache's fallback and refresh in the background,
            # but review admission refuses unknown capacity: a first review
            # child to an unprobed destination waits for the bounded probe.
            try:
                effective_resolution = bridge.dataclass_replace(
                    effective_resolution,
                    context_window=await resolve_window(effective_resolution),
                )
            except ValueError:
                pass  # changed target: keep the fallback; admission refuses it
        if is_subagent:
            call_continuation_target = (
                bridge.ContinuationRestoreTarget(
                    provider=effective_resolution.execution_key,
                    protocol=effective_resolution.continuation_protocol,
                    model=effective_resolution.model or "",
                    api_base_url=effective_resolution.base_url,
                    selected_provider=effective_resolution.selected_provider
                    or effective_resolution.provider,
                )
                if effective_resolution.continuation_protocol
                else None
            )
        # Forwarding `tools=` only when it is non-None (rather than
        # always passing the keyword, even as None) keeps every
        # pre-Task-5 gateway fake elsewhere in the test suite — whose
        # `stream_chat(resolution, messages)` signature predates this
        # parameter and has no matching call-site to update under this
        # task's own touched-files constraint — working unchanged for
        # the (still overwhelmingly common) fence path, where `tools`
        # is always None. The real gateway and any fake built against
        # this task's own `tools=None` contract see identical behavior
        # either way, since the callee-side default is also None.
        dispatch_messages = transport_messages
        stream_kwargs = {"tools": tools} if tools is not None else {}
        prepare_request = getattr(self._gateway, "prepare_chat_request", None)
        call_groups = continuation_groups
        incoming_owners = [group.owner_message_id for group in call_groups]
        if len(set(incoming_owners)) != len(incoming_owners):
            raise ValueError("Continuation owner IDs must be unique.")
        message_owners = [
            row.get(self._continuation_owner_key) for row in transport_messages
        ]
        if any(message_owners.count(owner) != 1 for owner in incoming_owners):
            raise ValueError(
                "Every continuation group must attach to one request owner."
            )
        semantic_messages = transport_messages
        if not is_subagent and self._continuation_sidecar:
            from tldw_chatbook.Chat.console_history_budget import (
                provider_continuation_owner_groups,
            )

            selected = {
                row.get(self._continuation_owner_key) for row in transport_messages
            }
            retained = provider_continuation_owner_groups(
                tuple(
                    item
                    for item in self._continuation_sidecar
                    if item.owner_message_id in selected
                ),
                target=call_continuation_target,
            )
            by_owner = {
                group.owner_message_id: group for group in (*retained, *call_groups)
            }
            call_groups = tuple(
                by_owner.pop(owner)
                for row in transport_messages
                if (owner := row.get(self._continuation_owner_key)) in by_owner
            )
        if call_groups:
            if call_continuation_target is None or not self._continuation_owner_key:
                raise ValueError("Provider continuation request is not pinned.")
            owner_ids = {group.owner_message_id for group in call_groups}
            rewritten_messages: list[dict[str, Any]] = []
            for message in transport_messages:
                row = dict(message)
                owner_id = row.get(self._continuation_owner_key)
                if type(owner_id) is str and owner_id in owner_ids:
                    row[bridge.CONTINUATION_OWNER_KEY] = owner_id
                # The pop drops the run's TEMPORARY owner key after its
                # value has been copied onto the canonical key. When the
                # run already uses the canonical key itself (the
                # default, e.g. an auto-derived native continuation),
                # popping would delete the marker the line above just
                # wrote, and every later continuation request would die
                # at build_console_request with "Every continuation
                # group must attach to one request owner" (TASK-19642.7,
                # red for the Kimi/Z.ai tool-error continuation since
                # 798d9c28bf).
                if self._continuation_owner_key != bridge.CONTINUATION_OWNER_KEY and (
                    is_subagent
                    or not self._thinking_sidecar
                    or self._thinking_owner_key != self._continuation_owner_key
                ):
                    row.pop(self._continuation_owner_key, None)
                rewritten_messages.append(row)
            semantic_messages = rewritten_messages
        if self._trace_request_factory is not None:
            if not callable(prepare_request):
                raise ValueError("Capture On agent gateway cannot prepare requests")
            dispatch_messages = prepare_request(
                effective_resolution,
                self._trace_request_factory.build(
                    semantic_messages,
                    tools=tools or (),
                    route=route,
                    actor_id=route_actor_id,
                    chain_id=route_chain_id,
                    continuation_groups=call_groups,
                ),
                route=route,
                route_actor_id=route_actor_id,
                route_chain_id=route_chain_id,
                continuation_target=call_continuation_target,
                thinking_sidecar=call_thinking_sidecar,
                thinking_policy=self._thinking_policy,
                thinking_owner_key=call_thinking_owner_key,
                capture_mode=self._capture_mode,
                **review_prepare_kwargs,
            )
            if not is_subagent and call_groups:
                dispatch_messages = bridge.dataclass_replace(
                    dispatch_messages, live_native_custody=self._live_native_custody
                )
            stream_kwargs.pop("tools", None)
        elif call_groups and callable(prepare_request):
            dispatch_messages = prepare_request(
                effective_resolution,
                bridge.build_console_request(
                    semantic_messages,
                    tools=tools or (),
                    continuation_groups=call_groups,
                    capture_mode=self._capture_mode,
                ),
                route=route,
                route_actor_id=route_actor_id,
                route_chain_id=route_chain_id,
                continuation_target=call_continuation_target,
                thinking_sidecar=call_thinking_sidecar,
                thinking_policy=self._thinking_policy,
                thinking_owner_key=call_thinking_owner_key,
                capture_mode=self._capture_mode,
                **review_prepare_kwargs,
            )
            stream_kwargs.pop("tools", None)
        elif (
            call_thinking_sidecar or (not is_subagent and self._continuation_sidecar)
        ) and callable(prepare_request):
            # The constructor sidecar belongs to the primary turn. A
            # child has its own history and must never consume it.
            dispatch_messages = prepare_request(
                effective_resolution,
                transport_messages,
                tools=tools,
                route=route,
                route_actor_id=route_actor_id,
                route_chain_id=route_chain_id,
                continuation_target=call_continuation_target,
                continuation_sidecar=() if is_subagent else self._continuation_sidecar,
                continuation_owner_key=self._continuation_owner_key,
                thinking_sidecar=call_thinking_sidecar,
                thinking_policy=self._thinking_policy,
                thinking_owner_key=call_thinking_owner_key,
                capture_mode=self._capture_mode,
                **review_prepare_kwargs,
            )
            stream_kwargs.pop("tools", None)
        if review_admission is not None:
            if not isinstance(dispatch_messages, bridge.PreparedProviderRequest):
                if not callable(prepare_request):
                    from tldw_chatbook.Agents.code_review_models import (
                        ReviewRefused,
                    )

                    raise ReviewRefused("review_context_capacity_unknown")
                dispatch_messages = prepare_request(
                    effective_resolution,
                    semantic_messages,
                    tools=tools,
                    route=route,
                    route_actor_id=route_actor_id,
                    route_chain_id=route_chain_id,
                    capture_mode=self._capture_mode,
                    apply_safety_window=False,
                )
                stream_kwargs.pop("tools", None)
            review_admission(dispatch_messages)
        if gateway_signals is not None:
            stream_kwargs["signals"] = gateway_signals
        emission_synthetic: bool | None = None

        def observe_emission(synthetic: bool) -> None:
            nonlocal emission_synthetic
            emission_synthetic = synthetic

        if isinstance(self._gateway, bridge.ConsoleProviderGateway):
            stream_kwargs["emission_observer"] = observe_emission
        owner_session_id = self._store.session_id_for_message(
            self._assistant_message_id
        )
        bridge.require_thinking_persistence_support(
            self._store.persistence,
            persistent=(
                self._store.persistence is not None
                and not self._store.session_is_ephemeral(owner_session_id)
            ),
            may_emit_thinking=bool(
                getattr(effective_resolution, "may_emit_thinking", False)
            ),
        )
        from tldw_chatbook.Chat.stream_stall_watchdog import (
            first_token_timeout_seconds,
            watch_content_stalls,
        )

        stall_timeout = bridge._stall_timeout_seconds()
        async for chunk in watch_content_stalls(
            self._gateway.stream_chat(
                effective_resolution,
                dispatch_messages,
                route=route,
                route_actor_id=route_actor_id,
                route_chain_id=route_chain_id,
                capture_mode=self._capture_mode,
                **stream_kwargs,
            ),
            stall_timeout,
            provider=effective_resolution.provider,
            # TASK-34100.5 AC#5: a cold local model's first token waits
            # longer than the gap between tokens.
            first_item_timeout_seconds=first_token_timeout_seconds(
                effective_resolution.provider, stall_timeout=stall_timeout
            ),
            # TASK-33666: the cuts below are only reached when a chunk
            # arrives. A provider that accepted the request and sends
            # nothing would hold this run's thread for the whole window
            # after a Stop, and the app's exit joins that thread. The
            # plain cancel, not stream_cut: a redirect is not a cancel
            # and still waits for the next chunk (TASK-28227).
            should_stop=self._should_cancel,
        ):
            received_item = True
            synthetic = emission_synthetic is True
            emission_synthetic = None
            if terminal_metadata is not None:
                raise ValueError("Provider terminal metadata must be final.")
            if isinstance(
                chunk,
                (
                    bridge.ProviderThinkingDelta,
                    bridge.ProviderProprietaryThinkingEvidence,
                ),
            ):
                if (
                    isinstance(chunk, bridge.ProviderThinkingDelta)
                    and attributed is not None
                ):
                    thinking_text = getattr(chunk, "text", None)
                    if isinstance(thinking_text, str) and thinking_text:
                        self._emit_live_usage(
                            bridge.AgentLiveUsageEvent(
                                "text",
                                usage_run_id,
                                usage_agent_kind,
                                usage_sequence,
                                bridge.time.monotonic(),
                                text=thinking_text,
                            )
                        )
                call_capture.observe(chunk)
                if not is_subagent:
                    update = self._thinking_capture.observe(chunk)
                    if update.envelope is not None:
                        # task-33081: validated=True -- the capture
                        # enforces the canonical limits incrementally
                        # per delta; re-validating the whole growing
                        # envelope here was quadratic.
                        self._store.replace_message_thinking(
                            self._assistant_message_id,
                            update.envelope,
                            generation_token=self._generation_token,
                            validated=True,
                        )
                if stream_cut():
                    break
                continue
            if isinstance(chunk, bridge.ProviderToolCalls):
                # Plan-B contract: structured deltas never hit the
                # transcript — captured here, surfaced only through the
                # returned message dict's `tool_calls`.
                native_calls.extend(chunk.tool_calls)
                if chunk.metadata is not None:
                    if not isinstance(chunk.metadata, bridge.ProviderTurnMetadata):
                        raise ValueError("Provider terminal metadata is malformed.")
                    terminal_metadata = chunk.metadata
                if not is_subagent:
                    self._thinking_capture.observe(chunk)
                if stream_cut():
                    break
                continue
            visible = gate.feed(chunk)
            if (
                isinstance(chunk, str)
                and chunk
                and attributed is not None
                and not synthetic
            ):
                self._emit_live_usage(
                    bridge.AgentLiveUsageEvent(
                        "text",
                        usage_run_id,
                        usage_agent_kind,
                        usage_sequence,
                        bridge.time.monotonic(),
                        text=chunk,
                    )
                )
            if (
                live_provider_usage_enabled
                and call_signals is not None
                and attributed is not None
            ):
                try:
                    partial_usage = call_signals.usage_snapshot()
                    partial_count = self._provider_output_count(partial_usage)
                    if partial_count is not None:
                        self._emit_live_usage(
                            bridge.AgentLiveUsageEvent(
                                "provider_usage",
                                usage_run_id,
                                usage_agent_kind,
                                usage_sequence,
                                bridge.time.monotonic(),
                                provider_output_tokens=partial_count,
                            )
                        )
                except Exception:  # noqa: BLE001 — optional observation only
                    live_provider_usage_enabled = False
                    bridge.logger.warning(
                        "live provider usage observation failed; "
                        "telemetry disabled for call"
                    )
            if visible and not is_subagent:
                self._thinking_capture.observe_answer(visible)
                self._store.append_stream_chunk(self._assistant_message_id, visible)
                any_streamed = True
            if stream_cut():
                break
        tail = gate.flush_tail()
        if tail and not is_subagent:
            self._thinking_capture.observe_answer(tail)
            self._store.append_stream_chunk(self._assistant_message_id, tail)
            any_streamed = True

    # PR2a Task 6.5: submit to the run's loop rather than driving it
    # from this thread. Every caller — the primary run's worker thread
    # and each fleet child's own thread — may be inside `chat_call`
    # simultaneously, and only ONE thread may ever drive a loop.
    # `run_coroutine_threadsafe` is the documented cross-thread entry
    # point; `.result()` blocks THIS thread (exactly as
    # `run_until_complete` did, including re-raising the coroutine's
    # exception) while the loop, on its own thread, interleaves this
    # turn with whatever other agents of this run have in flight.
    #
    # The wait is BOUNDED. A bare `.result()` deadlocks forever in one
    # real case: a child ABANDONED by `_settle_fleet` (which gives a
    # wedged child `FLEET_JOIN_TIMEOUT_SECONDS` and then walks away)
    # can still be sitting here when `run_reply`'s finally stops the
    # loop -- after which its coroutine is never scheduled again and
    # nothing will ever complete this future. The child's thread is a
    # daemon, so the process still exits, but it would hold its
    # provider connection and any locks it owns for the life of the
    # session. Timing out turns that into an ordinary turn failure the
    # run loop already knows how to report.
    #
    # PR3a-1 Task 1: `_submit_loop`, not `self._loop`. A fleet child
    # submits to the lifeline IT owns (parked on this thread by
    # `child_lifeline`), so `run_reply`'s end-of-turn teardown of the
    # TURN's loop can no longer strand a child mid-call -- the case the
    # timeout above was the last line of defence against.
    from tldw_chatbook.Chat.stream_stall_watchdog import StreamStallError

    future = bridge.asyncio.run_coroutine_threadsafe(_consume(), self._submit_loop)
    _stall_session_id = self._store.session_id_for_message(self._assistant_message_id)
    try:
        future.result(timeout=bridge._CHAT_CALL_TIMEOUT_SECONDS)
    except bridge.FuturesTimeoutError:
        future.cancel()
        raise TimeoutError(
            f"provider turn did not complete within {bridge._CHAT_CALL_TIMEOUT_SECONDS}s"
        ) from None
    except StreamStallError:
        # AC#3: a content stall is reported distinctly from a network error
        # and from a user cancel. AC#4: repeated stalls against the same
        # provider in a session surface a warning instead of silent retries.
        provider_label = self._resolution.provider
        repeated = bridge.record_session_stall(_stall_session_id, provider_label)
        # A stall is terminal for the turn (StreamStallError is non-transient,
        # so the run loop does not retry it). The threshold branch just marks
        # a provider that keeps stalling across turns in this session (AC#4).
        bridge.logger.warning(
            "provider stream stalled: no content within the stall window "
            "(provider={}, repeated_this_session={})",
            provider_label,
            repeated,
        )
        raise
    # A productive turn clears this session's stall streak (AC#4). A
    # stream the stop poll ended before anything arrived is not one
    # (TASK-33666): the same Stop used to run out the window as a stall.
    if received_item:
        bridge.reset_session_stalls(_stall_session_id, self._resolution.provider)
    _visible, tool_call = gate.result()
    if tool_call is not None and not native_calls and not is_subagent:
        self._thinking_capture.observe_tool()
    if any_streamed and not is_subagent:
        # Finding A: this turn leaked prose to the store before it was
        # known to be a tool call (a well-behaved fence-first tool call
        # never streams anything, so any_streamed stays False and this
        # never fires for the common case). Now that the full buffer is
        # in and the authoritative parse is available, discard that
        # leaked prose so it doesn't survive to garble the next turn's
        # chunks on the same message. Extended for native tool-calls
        # (Task 5): a native turn that streamed leaked prose before its
        # ProviderToolCalls sentinel arrived must be reset the same way.
        if tool_call is not None or native_calls:
            self._store.reset_stream_content(self._assistant_message_id)
        elif self._primary_stream_abort():
            # TASK-28227 review F2: this prose turn was cut by a
            # redirect and the re-asked turn will stream into the SAME
            # message -- without a separator the transcript glues
            # "...theRight — ..." together. The loop drains the redirect
            # only after this call returns, so the flag is still up here.
            self._store.append_stream_chunk(self._assistant_message_id, "\n\n")
    message: dict = {"content": gate.full_text}
    if native_calls:
        message["tool_calls"] = native_calls
    response: dict[str, Any] = {"choices": [{"message": message}]}
    # Design decision (TASK-16270): usage accounting is pure
    # observability. By the time this block runs, the provider turn has
    # already streamed and completed successfully — a failure HERE is
    # bookkeeping failing, not the model call failing, so it must never
    # convert the finished turn into an error outcome (PR #1612 hoisted
    # this accounting out of the signals-only guard, which made an
    # accounting crash fatal to an otherwise-successful run). A genuine
    # provider failure still classifies as an error: it raises from the
    # streaming/consume path above, OUTSIDE this wrap. On failure the
    # turn completes with the usage simply missing, and the cause is
    # logged.
    usage: dict[str, Any] | None = None
    raw_usage_payload: object = None
    provider_count: int | None = None
    try:
        raw_usage_payload = (
            terminal_metadata.usage if terminal_metadata is not None else None
        )
        if raw_usage_payload is None and call_signals is not None:
            raw_usage_payload = call_signals.usage_snapshot()
        usage = bridge._openai_usage_from_provider_call(
            raw_usage_payload,
            provider=effective_resolution.provider,
            model=effective_resolution.model or model or "",
        )
        if usage is None and call_signals is not None:
            usage = bridge._openai_usage_from_provider_call(
                call_signals.usage_snapshot(),
                provider=effective_resolution.provider,
                model=effective_resolution.model or model or "",
            )
        provider_count = self._provider_output_count(raw_usage_payload)
    except Exception as exc:  # noqa: BLE001 — observability is never fatal
        usage = None
        provider_count = None
        bridge.logger.warning(
            "usage accounting failed after a successful provider turn; "
            "completing the turn without usage (exception_type={})",
            type(exc).__name__,
        )
    if usage is not None:
        response["usage"] = usage
    if attributed is not None and provider_count is not None:
        self._emit_live_usage(
            bridge.AgentLiveUsageEvent(
                "provider_usage",
                usage_run_id,
                usage_agent_kind,
                usage_sequence,
                bridge.time.monotonic(),
                provider_output_tokens=provider_count,
            )
        )
    call_envelope = call_capture.settle(
        "stopped" if stream_cut() else "complete"
    ).envelope
    if (
        terminal_metadata is not None
        and terminal_metadata.provider_continuation is not None
    ):
        target = bridge.ContinuationRestoreTarget(
            provider=effective_resolution.execution_key,
            protocol=effective_resolution.continuation_protocol,
            model=effective_resolution.model or "",
            api_base_url=effective_resolution.base_url,
            selected_provider=effective_resolution.selected_provider
            or effective_resolution.provider,
        )
        terminal_metadata = bridge.dataclass_replace(
            terminal_metadata,
            provider_continuation=bridge.bind_continuation_to_target(
                terminal_metadata.provider_continuation, target
            ),
        )
    result = bridge._StreamingProviderResponse(
        response, terminal_metadata, call_envelope
    )
    return result
