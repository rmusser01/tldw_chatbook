"""Host binding of provider continuation to Console owners and destinations.

Pure checks shared by the Console store and agent bridge: how a runtime event
transitions an owner's checkpoint, what a durable write must prove about its
host-captured target and dispatch destination, when a resume may claim its
owner, and whether a run may write durable continuation state at all.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Agents.agent_models import (
    FinalContinuation,
    ProviderContinuationEvent,
    ToolBatchReady,
    ToolCallExecuting,
    ToolCallFinished,
)
from tldw_chatbook.Chat.console_endpoint_provenance import ConsoleEndpointProvenance
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationConflictError,
    ContinuationRestoreTarget,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    transition_provider_call,
    validate_continuation_restore,
)


def require_target_matches_resolution(
    target: ContinuationRestoreTarget, resolution: Any
) -> None:
    """Refuse a pinned continuation target the current resolution does not match.

    A target without ``selected_provider`` is an unbound legacy target: it is
    accepted as validate_continuation_restore accepts it, except that it cannot
    prove a custom endpoint selection.

    Raises:
        ContinuationConflictError: If any pinned field differs.
    """
    from tldw_chatbook.Chat.console_provider_endpoints import (
        normalize_generic_endpoint_for_compare as endpoint,
    )
    from tldw_chatbook.Chat.provider_readiness import provider_config_key as key

    # One normalization on both sides; either the execution key or the
    # identity spelling may pin the target (a swapped custom-ep pins the
    # execution key; split thinking/continuation doubles pin the identity).
    if (
        key(target.provider)
        not in {
            key(resolution.execution_key or resolution.provider),
            key(resolution.provider),
        }
        or (
            target.selected_provider
            != (resolution.selected_provider or resolution.provider)
            if target.selected_provider is not None
            else target.provider == "custom-hosted"
        )
        or target.model != (resolution.model or "")
        or endpoint(target.api_base_url) != endpoint(resolution.base_url)
        or (
            resolution.continuation_protocol is not None
            and target.protocol != resolution.continuation_protocol
        )
    ):
        raise ContinuationConflictError("Continuation restore target mismatch.")


def continuation_event_value(
    message: Any,
    event: ProviderContinuationEvent,
) -> tuple[ProviderContinuationCheckpoint, str]:
    """Resolve a typed event without logging its private payload."""
    current = message.provider_continuation
    if isinstance(event, (ToolBatchReady, FinalContinuation)):
        if event.expected_checkpoint_revision is None:
            if current is not None:
                raise RuntimeError("Continuation revision conflict.")
        elif (
            current is None
            or current.checkpoint_revision != event.expected_checkpoint_revision
        ):
            raise RuntimeError("Continuation revision conflict.")
        if isinstance(event, ToolBatchReady):
            return event.checkpoint, message.content
        return event.checkpoint, event.assistant_content
    if current is None:
        raise RuntimeError("Durable continuation owner is unavailable.")
    if isinstance(event, ToolCallExecuting):
        checkpoint = transition_provider_call(
            current,
            call_id=event.call_id,
            expected_revision=event.expected_checkpoint_revision,
            target="executing",
        )
        return checkpoint, message.content
    if isinstance(event, ToolCallFinished):
        checkpoint = transition_provider_call(
            current,
            call_id=event.call_id,
            expected_revision=event.expected_checkpoint_revision,
            target=event.target_state,
            result=event.result,
        )
        return checkpoint, message.content
    raise RuntimeError("Unsupported continuation event.")


def validate_durable_continuation_binding(
    message: Any,
    checkpoint: ProviderContinuationCheckpoint,
    continuation_target: ContinuationRestoreTarget | None,
    dispatch_recovery: Any,
    *,
    has_finalizer: bool,
) -> None:
    """Prove a durable write matches its host target and dispatch destination.

    Raises:
        RuntimeError: If the target is missing or any binding differs.
    """
    if continuation_target is None:
        raise RuntimeError("Durable continuation host target is unavailable.")
    try:
        validate_continuation_restore(checkpoint, continuation_target)
        current = message.provider_continuation
        if current is not None:
            if current.schema_version == 3 and checkpoint.schema_version != 3:
                raise RuntimeError("Provider continuation binding cannot be erased.")
            validate_continuation_restore(current, continuation_target)
            validate_continuation_restore(
                checkpoint,
                ContinuationRestoreTarget(
                    provider=current.provider,
                    protocol=current.protocol,
                    model=current.model,
                    api_base_url=current.api_base_url,
                    selected_provider=current.selected_provider
                    or continuation_target.selected_provider,
                ),
            )
            if current.managed_resume is not None and not has_finalizer:
                raise RuntimeError("Managed continuation finalizer is unavailable.")
        if dispatch_recovery is not None and dispatch_recovery.checkpoint is not None:
            if (
                dispatch_recovery.assistant_message_id != message.id
                or checkpoint.schema_version != 3
            ):
                raise RuntimeError("Durable continuation destination is unavailable.")
            destination = dispatch_recovery.checkpoint.resolved_destination
            validate_continuation_restore(
                checkpoint,
                ContinuationRestoreTarget(
                    provider=destination.continuation_provider,
                    protocol=destination.continuation_protocol,
                    model=destination.model,
                    api_base_url=destination.continuation_base_url,
                    selected_provider=destination.provider,
                ),
            )
    except ValueError:
        raise RuntimeError("Provider continuation destination mismatch.") from None


def claims_resume_owner(
    message: Any,
    expected: ProviderContinuationCheckpoint | None,
    expected_version: int | None,
    persistence: Any,
) -> bool:
    """Whether an attempt resumes exactly the active checkpoint it read.

    Returns:
        False for an ordinary attempt (no expectation), True for a resume.

    Raises:
        RuntimeError: If the owner moved since the resume read it.
    """
    if expected is None and expected_version is None:
        return False
    if (
        expected is None
        or expected.state != "active"
        or any(
            call.state == "executing"
            for round_ in expected.rounds
            for call in round_.calls
        )
        or message.provider_continuation != expected
        or not message.provider_continuation_actions_enabled
        or type(expected_version) is not int
        or message.provider_continuation_message_version != expected_version
        or message.persisted_message_id is None
        or persistence is None
        or persistence.get_message_version(message.persisted_message_id)
        != expected_version
    ):
        raise RuntimeError("Provider continuation owner changed before resume.")
    return True


def require_live_native_owner(
    cursor: Any,
    message: Any,
    *,
    checkpoint: ProviderContinuationCheckpoint,
    message_version: int,
    current: bool,
) -> None:
    """Recheck an acknowledged live owner, in memory and in its saved row.

    Raises:
        ValueError: ``live_native_owner_changed`` when either side moved.
    """
    if (
        not current
        or message.provider_continuation_remote
        or not message.provider_continuation_actions_enabled
        or message.provider_continuation != checkpoint
        or message.provider_continuation_message_version != message_version
        or message.assistant_generation_state != "continuation_active"
        or checkpoint.state != "active"
    ):
        raise ValueError("live_native_owner_changed")
    row = cursor.execute(
        "SELECT version, provider_continuation_json, assistant_generation_state "
        "FROM messages WHERE id = ? AND deleted = 0",
        (message.persisted_message_id,),
    ).fetchone()
    if row is None or tuple(row) != (
        message_version,
        dump_provider_continuation_json(checkpoint),
        "continuation_active",
    ):
        raise ValueError("live_native_owner_changed")


def continuation_can_be_durable(
    store: Any, session_id: str, assistant_message_id: str, resolution: Any
) -> bool:
    """Whether this run may write durable provider continuation state.

    A temporary session or a session-only endpoint never is. Neither is a
    dispatch whose checkpoint recorded no continuation tuple (pre-upgrade, or
    a destination it could not record): the tool loop still runs, but its
    continuation is not resumable.
    """
    if (
        store.session_is_ephemeral(session_id)
        or getattr(resolution, "endpoint_provenance", None)
        == ConsoleEndpointProvenance.EPHEMERAL_SESSION
    ):
        return False
    lookup = getattr(store, "dispatch_recovery_for_session", None)
    recovery = lookup(session_id) if callable(lookup) else None
    checkpoint = getattr(recovery, "checkpoint", None)
    return (
        checkpoint is None
        or recovery.assistant_message_id != assistant_message_id
        or not getattr(resolution, "continuation_protocol", None)
        or checkpoint.resolved_destination.continuation_provider is not None
    )


def continuation_restore_target_for_resolution(
    resolution: Any,
) -> ContinuationRestoreTarget | None:
    """Build an exact target only from explicitly pinned resolution fields.

    ``api_base_url`` is carried through VERBATIM. It used to be passed
    through ``normalize_generic_endpoint_for_compare`` first, which expands
    a base URL to its full chat-completions endpoint
    (``https://api.moonshot.ai/v1`` -> ``.../v1/chat/completions``). That
    silently broke every Resume: the checkpoint pins whatever
    ``ConsoleAgentBridge`` recorded, and the bridge records
    ``resolution.base_url`` RAW (see its ``ContinuationRestoreTarget(...)``
    construction), so recovery compared an expanded URL against a
    non-expanded one and ``validate_continuation_restore`` -- which is
    byte-exact by design, down to a trailing slash -- rejected it with
    "Pinned provider settings no longer match".

    Normalizing BOTH sides instead would have been the wrong repair: the
    exactness is deliberate (``test_provider_continuation`` pins that
    ``https://api.deepseek.com/v1/`` and ``https://api.deepseek.com/v1``
    are a MISMATCH), and this comparison is what stops a private
    continuation from being replayed against a different endpoint than the
    one that produced it. The two writers simply have to agree, and the
    checkpoint's writer is the one that defines the format.

    The provider field follows the same writer-agreement rule (ADR-179,
    Qodo follow-up): checkpoints are pinned under the EXECUTION key
    (engine records use their ``record.key`` -- "custom-hosted" for a
    swapped custom-ep selection -- and moonshot/zai/deepseek spell theirs
    identically), and ``ConsoleAgentBridge`` records targets from
    ``resolution.execution_key`` VERBATIM. Building the target from
    ``provider_config_key(resolution.provider)`` instead pinned the
    IDENTITY spelling ("custom_ep:paid" for ``custom-ep:paid``), which no
    checkpoint ever carries, so the durable resume/crash-recovery pre-gate
    byte-compared a spelling that could never match. ``execution_key`` is
    therefore preferred verbatim; only a resolution with an EMPTY
    execution key (legacy doubles) falls back to the normalized
    ``provider`` spelling, which is byte-identical to the old behavior for
    every checkpoint-producing provider.
    """
    protocol = getattr(resolution, "continuation_protocol", None) or getattr(
        resolution, "api_mode", None
    )
    base_url = getattr(resolution, "base_url", None)
    model = getattr(resolution, "model", None)
    if (
        not protocol
        or not isinstance(base_url, str)
        or not base_url.startswith(("http://", "https://"))
        or not model
    ):
        return None
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    execution_key = getattr(resolution, "execution_key", None)
    provider_value = (
        str(execution_key)
        if isinstance(execution_key, str) and execution_key
        else provider_config_key(str(getattr(resolution, "provider", "")))
    )
    return ContinuationRestoreTarget(
        provider=provider_value,
        model=str(model),
        protocol=str(protocol),
        api_base_url=base_url,
        selected_provider=getattr(resolution, "selected_provider", None)
        or resolution.provider,
    )
