"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from . import console_provider_gateway as native_host


def prepare_chat_request(
    self,
    resolution: native_host.ConsoleProviderResolution,
    messages: list[native_host.Mapping[str, native_host.Any]]
    | native_host.PreparedConsoleRequest,
    *,
    tools: list[native_host.Mapping[str, native_host.Any]] | None = None,
    context_window_override_tokens: int | None = None,
    apply_safety_window: bool = True,
    response_format: native_host.Mapping[str, native_host.Any] | None = None,
    continuation_target: native_host.ContinuationRestoreTarget | None = None,
    continuation_sidecar: tuple[native_host.ProviderContinuationSidecar, ...] = (),
    continuation_owner_key: str | None = None,
    thinking_sidecar: tuple[native_host.ProviderThinkingSidecar, ...] = (),
    thinking_policy: native_host.ThinkingHistoryPolicy | None = None,
    thinking_owner_key: str | None = None,
    route: native_host.ConsoleRequestRoute | None = None,
    route_actor_id: str | None = None,
    route_chain_id: str | None = None,
    capture_mode: native_host.ConsoleTraceCaptureMode = native_host.ConsoleTraceCaptureMode.CAPTURE_OFF,
    ephemeral: bool = False,
) -> native_host.PreparedProviderRequest:
    """Prepare the one immutable payload later consumed by dispatch.

    Model capability facts are read once here.  Unknown models remain
    explicitly unverified; an optional user override is enforced as a
    bound but never labeled as provider-verified.
    """

    native_host.require_durable_capture_admission(
        capture_mode=capture_mode,
        ephemeral=ephemeral,
    )
    if isinstance(messages, native_host.PreparedConsoleRequest) and tools is not None:
        raise ValueError("tools are already owned by PreparedConsoleRequest")
    sidecar = tuple(continuation_sidecar)
    thinking_sidecars = tuple(thinking_sidecar)
    if sidecar and (continuation_target is None or not continuation_owner_key):
        raise ValueError(
            "continuation target and owner key are required for private history"
        )
    if thinking_sidecars and not thinking_owner_key:
        raise ValueError("thinking owner key is required for thinking history")
    if continuation_target is not None:
        from .console_continuation_binding import require_target_matches_resolution

        require_target_matches_resolution(continuation_target, resolution)
    if isinstance(messages, native_host.PreparedConsoleRequest):
        continuation_groups = (
            tuple(
                group
                for unit in messages.compactable
                for group in unit.continuation_groups
            )
            + messages.active_continuation_groups
        )
        if continuation_groups and continuation_target is None:
            raise ValueError(
                "continuation_target is required for provider continuation history"
            )
        if continuation_target is not None:
            for group in continuation_groups:
                native_host.validate_continuation_restore(
                    group.checkpoint, continuation_target
                )
        semantic = (
            native_host.replace(messages, effective_thinking_policy="required")
            if continuation_groups and messages.effective_thinking_policy != "required"
            else messages
        )
        if thinking_sidecars:
            assert thinking_owner_key is not None
            selected_thinking_owner_ids = {
                message.get(thinking_owner_key)
                for message in semantic.flattened_messages()
                if type(message.get(thinking_owner_key)) is str
            }
            thinking = native_host.resolve_thinking_history(
                target=native_host.ThinkingReplayTarget(
                    provider=resolution.execution_key or resolution.provider,
                    model=resolution.model or "",
                    protocol=native_host._thinking_protocol(resolution),
                    disposition=(
                        "displayable"
                        if resolution.local_structured_thinking
                        else resolution.thinking_stream_disposition
                    ),
                    round_trip_version=(
                        native_host.THINKING_ENVELOPE_VERSION
                        if resolution.local_structured_thinking
                        else resolution.thinking_round_trip_version
                    ),
                    reasoning_replay=resolution.reasoning_replay,
                ),
                policy=thinking_policy,
                sidecars=tuple(
                    item
                    for item in thinking_sidecars
                    if item.owner_message_id in selected_thinking_owner_ids
                ),
                continuation_required=bool(continuation_groups),
            )
            semantic = native_host.attach_thinking_history(
                semantic,
                groups=thinking.groups,
                owner_key=thinking_owner_key,
                thinking_policy=thinking.saved_policy,
                effective_thinking_policy=thinking.effective_policy,
            )
    elif not sidecar and not thinking_sidecars:
        if any("provider_continuation" in message for message in messages):
            raise ValueError(
                "continuation_target is required for provider continuation history"
            )
        semantic = native_host.build_console_request(
            messages,
            tools=tools or (),
            capture_mode=capture_mode,
        )
    else:
        continuation_groups = ()
        if sidecar:
            assert continuation_target is not None
            assert continuation_owner_key is not None
            selected_owner_ids = {
                message.get(continuation_owner_key)
                for message in messages
                if not native_host.is_deleted_history_value(message.get("deleted"))
                and type(message.get(continuation_owner_key)) is str
            }
            continuation_groups = native_host.provider_continuation_owner_groups(
                tuple(
                    item
                    for item in sidecar
                    if item.owner_message_id in selected_owner_ids
                ),
                target=continuation_target,
            )
        selected_thinking_owner_ids = {
            message.get(thinking_owner_key)
            for message in messages
            if thinking_owner_key is not None
            and not native_host.is_deleted_history_value(message.get("deleted"))
            and type(message.get(thinking_owner_key)) is str
        }
        thinking = native_host.resolve_thinking_history(
            target=native_host.ThinkingReplayTarget(
                provider=resolution.execution_key or resolution.provider,
                model=resolution.model or "",
                protocol=native_host._thinking_protocol(resolution),
                disposition=(
                    "displayable"
                    if resolution.local_structured_thinking
                    else resolution.thinking_stream_disposition
                ),
                round_trip_version=(
                    native_host.THINKING_ENVELOPE_VERSION
                    if resolution.local_structured_thinking
                    else resolution.thinking_round_trip_version
                ),
                reasoning_replay=resolution.reasoning_replay,
            ),
            policy=thinking_policy,
            sidecars=tuple(
                item
                for item in thinking_sidecars
                if item.owner_message_id in selected_thinking_owner_ids
            ),
            continuation_required=bool(continuation_groups),
        )
        continuation_owner_ids = {
            group.owner_message_id for group in continuation_groups
        }
        thinking_owner_ids = {group.owner_message_id for group in thinking.groups}
        visible_messages: list[dict[str, native_host.Any]] = []
        for message in messages:
            if native_host.is_deleted_history_value(message.get("deleted")):
                continue
            row = dict(message)
            if (
                continuation_owner_key is not None
                and continuation_owner_key == thinking_owner_key
            ):
                shared_owner_id = row.pop(continuation_owner_key, None)
                continuation_owner_id = shared_owner_id
                thinking_owner_id = shared_owner_id
            else:
                continuation_owner_id = (
                    row.pop(continuation_owner_key, None)
                    if continuation_owner_key is not None
                    else None
                )
                thinking_owner_id = (
                    row.pop(thinking_owner_key, None)
                    if thinking_owner_key is not None
                    else None
                )
            row.pop("provider_continuation", None)
            row.pop("deleted", None)
            if (
                type(continuation_owner_id) is str
                and continuation_owner_id in continuation_owner_ids
            ):
                row[native_host.CONTINUATION_OWNER_KEY] = continuation_owner_id
            if (
                type(thinking_owner_id) is str
                and thinking_owner_id in thinking_owner_ids
            ):
                row[native_host.THINKING_OWNER_KEY] = thinking_owner_id
            visible_messages.append(row)
        semantic = native_host.build_console_request(
            visible_messages,
            tools=tools or (),
            continuation_groups=continuation_groups,
            thinking_groups=thinking.groups,
            thinking_policy=thinking.saved_policy,
            effective_thinking_policy=thinking.effective_policy,
            capture_mode=capture_mode,
        )

    native_host._validate_request_trace_binding(
        semantic,
        route=route,
        route_actor_id=route_actor_id,
        route_chain_id=route_chain_id,
        capture_mode=capture_mode,
    )

    capabilities: native_host.Mapping[str, native_host.Any] = {}
    try:
        from tldw_chatbook.model_capabilities import get_model_capabilities

        capabilities = get_model_capabilities().get_model_capabilities(
            resolution.provider,
            resolution.model or "",
        )
    except Exception:
        native_host.logger.debug("console_request_capability_lookup_failed")

    def positive_cap(*names: str) -> int | None:
        return native_host.positive_token_cap(capabilities, *names)

    from tldw_chatbook.Utils.token_counter import resolve_context_window

    context_window = resolution.context_window or resolve_context_window(
        resolution.provider, resolution.model or ""
    )
    if (
        resolution.context_window is not None
        and getattr(resolution.context_window, "source", None) == "server metadata"
    ):
        # TASK-34100.2 AC#2: readiness checks capacity against the window
        # the serving endpoint reported, exactly as this preflight does.
        # Only a server's report is kept: a catalog window is re-read from
        # the catalog, so a window saved in Settings is never shadowed.
        native_host.CONSOLE_SEND_OUTCOMES.record_window(
            resolution.provider,
            resolution.model,
            context_window.tokens,
            context_window.verified,
            endpoint=resolution.base_url,
        )
    capacity = native_host.resolve_request_capacity(
        context_window_tokens=context_window.tokens,
        context_window_verified=context_window.verified,
        provider_input_cap_tokens=positive_cap(
            "max_input_tokens", "input_token_limit", "provider_input_cap"
        ),
        provider_output_cap_tokens=positive_cap(
            "max_output_tokens", "output_token_limit", "provider_output_cap"
        ),
        requested_response_tokens=resolution.max_tokens,
        context_window_override_tokens=context_window_override_tokens,
    )
    wire_style: native_host.WireStyle = (
        "distinct_roles"
        if resolution.provider in {"llama_cpp", "local_llamacpp"}
        else "single_preamble"
    )
    return native_host.prepare_provider_request(
        semantic,
        reasoning_replay=resolution.reasoning_replay,
        wire_style=wire_style,
        model=resolution.model or "",
        provider=resolution.provider,
        capacity=capacity,
        per_image_tokens=(
            positive_cap("image_input_tokens", "image_tokens", "per_image_tokens")
            or native_host.DEFAULT_PER_IMAGE_TOKENS
        ),
        apply_safety_window=apply_safety_window,
        response_format=response_format,
    )
