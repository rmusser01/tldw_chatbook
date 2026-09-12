"""Frozen Console capture-policy bindings shared by live policy surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from tldw_chatbook.Chat.console_chat_controller import (
    CapturePolicyMutationResult,
    CapturePolicyMutationStatus,
    CapturePurgeResult,
    CapturePurgeStatus,
)
from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
    CapturePolicyBindings,
)


class CapturePurgeRefreshError(Exception):
    """A committed purge whose post-commit repaint failed."""

    def __init__(self, result: CapturePurgeResult) -> None:
        super().__init__("capture purge committed; refresh failed")
        self.result = result


class InspectorCapturePolicyWiring(NamedTuple):
    """Frozen policy dependencies for one Inspector instance."""

    target_session_id: str
    target_conversation_id: str | None
    bindings: CapturePolicyBindings
    capture_revision: Callable[[], int | None]
    bind_inspector: Callable[[Any], None]


def build_capture_policy_bindings(
    controller: Any,
    session_id: str,
    conversation_id: str | None,
    *,
    purge_success: Callable[[], Awaitable[None]] | None = None,
) -> CapturePolicyBindings:
    """Freeze policy/count/purge callbacks to one live Console session.

    Args:
        controller: Live Console controller that owns capture policy.
        session_id: Immutable session targeted by the surface.
        conversation_id: Immutable persisted conversation target, if any.
        purge_success: Optional post-commit Inspector repaint callback.

    Returns:
        Shared policy bindings whose purge revision is always an integer.
        Inspector read freshness uses the separate nullable callback below.
    """

    def read():
        return controller.capture_policy_snapshot(session_id)

    def apply_next(detail, expected_policy_revision):
        return controller.set_next_capture_detail(
            session_id,
            detail,
            expected_policy_revision=expected_policy_revision,
        )

    async def apply_conversation(detail, expected_policy_revision):
        return await controller.replace_conversation_capture_detail(
            session_id,
            detail,
            expected_policy_revision=expected_policy_revision,
        )

    def apply_next_privacy(capture_enabled, pii_enabled, expected_policy_revision):
        return controller.set_next_trace_privacy(
            session_id,
            capture_enabled=capture_enabled,
            pii_redaction_enabled=pii_enabled,
            expected_policy_revision=expected_policy_revision,
        )

    async def apply_conversation_privacy(
        capture_enabled, pii_enabled, expected_policy_revision
    ):
        return await controller.replace_conversation_trace_privacy(
            session_id,
            capture_enabled=capture_enabled,
            pii_redaction_enabled=pii_enabled,
            expected_policy_revision=expected_policy_revision,
        )

    def apply_global(enabled, detail, config_generation, policy_revision):
        if controller.store.active_session_id != session_id:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.TARGET_MISSING,
                read(),
                True,
                "opener_session_not_active",
            )
        return controller.apply_global_capture_settings(
            enabled=enabled,
            detail=detail,
            expected_config_generation=config_generation,
            expected_policy_revision=policy_revision,
        )

    def apply_global_privacy(
        capture_enabled, pii_enabled, config_generation, policy_revision
    ):
        if controller.store.active_session_id != session_id:
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.TARGET_MISSING,
                read(),
                True,
                "opener_session_not_active",
            )
        snapshot = read()
        return controller.apply_global_capture_settings(
            enabled=capture_enabled,
            detail=snapshot.global_detail,
            pii_redaction_enabled=pii_enabled,
            expected_config_generation=config_generation,
            expected_policy_revision=policy_revision,
        )

    async def count_full() -> int:
        stage = await asyncio.to_thread(
            controller.store.stage_full_capture_purge, session_id
        )
        return stage.removed_count

    async def purge_full(expected_capture_revision):
        result = await controller.purge_full_captures(
            session_id, expected_capture_revision
        )
        if result.status is CapturePurgeStatus.DELETED and purge_success is not None:
            try:
                await purge_success()
            except Exception as exc:
                raise CapturePurgeRefreshError(result) from exc
        return result

    return CapturePolicyBindings(
        target_session_id=session_id,
        target_conversation_id=conversation_id,
        read=read,
        apply_next=apply_next,
        apply_conversation=apply_conversation,
        apply_global=apply_global,
        count_full=count_full,
        purge_full=purge_full,
        capture_revision=lambda: controller.capture_revision(session_id),
        purge_availability=lambda: controller.capture_purge_availability(session_id),
        apply_next_privacy=apply_next_privacy,
        apply_conversation_privacy=apply_conversation_privacy,
        apply_global_privacy=apply_global_privacy,
    )


def build_inspector_capture_policy_wiring(
    controller: Any,
) -> InspectorCapturePolicyWiring | None:
    """Resolve and freeze the active Inspector target and repaint callback.

    Args:
        controller: Live Console controller whose active session is opening
            the Inspector.

    Returns:
        Immutable Inspector wiring, or ``None`` when no session is active.
        Its freshness callback returns ``None`` only while capture is
        quiescent; purge mutation continues to use the non-null binding.
    """

    session_id = controller.store.active_session_id
    if session_id is None:
        return None
    session = next(
        (item for item in controller.store.sessions() if item.id == session_id),
        None,
    )
    conversation_id = session.persisted_conversation_id if session else None
    inspector_holder: list[Any] = []

    async def refresh() -> None:
        if inspector_holder:
            await inspector_holder[0]._invalidate_stale_exchange_mounts()

    def capture_revision() -> int | None:
        if controller.store.capture_quiescent(session_id):
            return None
        return controller.capture_revision(session_id)

    return InspectorCapturePolicyWiring(
        session_id,
        conversation_id,
        build_capture_policy_bindings(
            controller,
            session_id,
            conversation_id,
            purge_success=refresh,
        ),
        capture_revision,
        inspector_holder.append,
    )


__all__ = [
    "CapturePurgeRefreshError",
    "InspectorCapturePolicyWiring",
    "build_capture_policy_bindings",
    "build_inspector_capture_policy_wiring",
]
