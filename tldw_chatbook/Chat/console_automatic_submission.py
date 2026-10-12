"""Live automatic submission authority shared by wake, chat-start and goal owners."""

from __future__ import annotations

import asyncio
from typing import Any

from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin


def resolve_automatic_authorization(controller, origin, session_id, wake, goal):
    if origin is not ConsoleSubmissionOrigin.GOAL_ITERATION:
        return controller._fleet_wake, wake
    coordinator = getattr(controller, "_goal_coordinator", None)
    if coordinator is None or not coordinator.authorizes(goal, session_id):
        raise PermissionError("Goal turns require live coordinator authority.")
    return coordinator, goal


def automatic_source_conversation(
    controller,
    session_id,
    work_chain_id,
    *,
    wake_authorization,
    chat_start_authorization,
):
    """Resolve a live automatic owner; stored provenance grants no authority."""
    if getattr(
        controller, "_goal_coordinator", None
    ) is not None and controller._goal_coordinator.authorizes(
        wake_authorization, session_id
    ):
        goal = wake_authorization
        coordinator = getattr(controller, "_goal_coordinator", None)
        if (
            chat_start_authorization is not None
            or coordinator is None
            or not coordinator.authorizes(goal, session_id)
            or not goal.accepted
            or goal.work_chain_id != work_chain_id
        ):
            raise PermissionError("Automatic source requires live goal authority.")
        return goal.conversation_id
    if chat_start_authorization is not None:
        start = chat_start_authorization
        if (
            wake_authorization is not None
            or not controller._chat_start.authorizes(start, session_id)
            or not start.accepted
            or not start.receipted
            or start.bridge is not controller._agent_bridge
            or start.database is not getattr(controller._agent_bridge, "runs_db", None)
            or start.capacity_owner is not controller._fleet_wake
            or start.runtime_owner_id != controller._fleet_wake.runtime_owner_id
            or start.context.chain_id != work_chain_id
        ):
            raise PermissionError(
                "Automatic source requires live chat start authority."
            )
        return start.request.conversation_id
    if (
        not controller._fleet_wake.authorizes(wake_authorization, session_id)
        or not wake_authorization.accepted
        or wake_authorization.work_chain_id != work_chain_id
    ):
        raise PermissionError("Automatic source requires live wake authority.")
    return wake_authorization.conversation_id


def capture_execution_roots(
    controller,
    session_id,
    project_selection,
    project_authority_guard,
    remote_session_key,
):
    """Capture workspace authority, then apply the goal's frozen narrowing."""
    from .console_chat_controller import capture_run_admitted_workspace_roots

    run_admitted_roots = capture_run_admitted_workspace_roots(
        session=next(
            (item for item in controller.store.sessions() if item.id == session_id),
            None,
        ),
        registry=getattr(controller.app, "workspace_registry_service", None),
        project_selection=project_selection,
        project_authority_guard=project_authority_guard,
        remote_session_key=remote_session_key,
    )
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    automatic = current_automatic_work()
    if automatic is not None and automatic.goal is not None:
        run_admitted_roots, project_selection, project_authority_guard = (
            automatic.goal.execution_roots(project_selection, project_authority_guard)
        )
    return run_admitted_roots, project_selection, project_authority_guard


def submission_scope(controller, origin, session_id):
    """Manual sends pause their goal and reset inherited automatic context."""
    from contextlib import nullcontext

    from tldw_chatbook.Agents.automatic_work_runtime import manual_work_scope

    if not origin.automatic:
        goals = getattr(controller, "_goal_coordinator", None)
        target = next(
            (
                item
                for item in controller.store.sessions()
                if item.id == (session_id or controller.store.active_session_id)
            ),
            None,
        )
        if goals is not None and target is not None:
            goals.manual_send(target.persisted_conversation_id)
    return (
        nullcontext()
        if origin
        in {
            ConsoleSubmissionOrigin.AGENT_WAKE,
            ConsoleSubmissionOrigin.GOAL_ITERATION,
            ConsoleSubmissionOrigin.AGENT_CHAT_START,
        }
        else manual_work_scope()
    )


def agent_conversation_id(controller, session_id):
    """Bind the session to its durable run-store conversation identity."""
    for session in controller.store.sessions():
        if session.id == session_id:
            conversation_id = session.persisted_conversation_id or session_id
            with controller._lifecycle_revision_lock:
                controller._lifecycle_session_by_conversation[str(conversation_id)] = (
                    session_id
                )
            return conversation_id
    return session_id


def hook_busy_sessions(
    controller,
    origin,
    session_id,
    queue_authorization,
    chat_start_authorization,
    recovery,
):
    """Exclude only this submission's proven held slot from hook preflight."""
    busy = controller._live_busy_session_ids()
    if (
        origin is ConsoleSubmissionOrigin.QUEUED
        and controller.prompt_queue_coordinator.reuses_claimed_slot(
            queue_authorization, session_id, recovery=recovery
        )
    ):
        # Claimed turns and authorized recovery reuse the queue's
        # held capacity instead of competing with themselves.
        busy = [identity for identity in busy if identity != session_id]
    if (
        origin is ConsoleSubmissionOrigin.AGENT_CHAT_START
        and controller._chat_start.authorizes(chat_start_authorization, session_id)
    ):
        busy = [identity for identity in busy if identity != session_id]
    if origin is ConsoleSubmissionOrigin.GOAL_ITERATION:
        from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

        automatic = current_automatic_work()
        if (
            automatic is not None
            and automatic.goal is not None
            and automatic.goal._coordinator.authorizes(automatic.goal, session_id)
        ):
            busy = [identity for identity in busy if identity != session_id]
    return busy


async def prepare_submission_hooks(
    controller,
    session,
    configuration,
    origin,
    queue_authorization=None,
    *,
    recovery=False,
    chat_start_authorization=None,
):
    """Reuse the submission's existing hook scope and capacity reservation."""
    import asyncio

    from .console_chat_controller import ConsoleSubmitResult
    from .console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
    )

    session_id = session.id
    hook_runtime = getattr(controller, "_hooks_v2_runtime", None)
    if hook_runtime is not None:
        # Recheck after awaited reference expansion; reserve without an await.
        busy = hook_busy_sessions(
            controller,
            origin,
            session_id,
            queue_authorization,
            chat_start_authorization,
            recovery,
        )
        if session_id in busy or len(busy) >= controller.max_parallel_runs:
            return ConsoleSubmitResult(False, False, "A run is already preparing.")
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Initializing hooks."),
            session_id=session_id,
        )
        submissions = getattr(controller, "_hooks_v2_submissions", None)
        if submissions is None:
            submissions = controller._hooks_v2_submissions = {}
        submissions[asyncio.current_task()] = (None, None, session_id)
        try:
            lifecycle = await hook_runtime.prepare_hooks_v2(
                session_id,
                configuration=configuration,
                reason="resume" if session.persisted_conversation_id else "startup",
                initiator=(
                    "manual"
                    if origin is ConsoleSubmissionOrigin.MANUAL
                    else "scheduled"
                ),
            )
            if lifecycle is not None:
                scope = lifecycle.turn_scope or lifecycle.open_scope()
                lifecycle.turn_scope = scope
                submissions[asyncio.current_task()] = (lifecycle, scope, session_id)
                await lifecycle.wait(scope)
        except asyncio.CancelledError:
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Initialization cancelled."),
                session_id=session_id,
            )
            raise
        except Exception:  # noqa: BLE001 -- hook boundary
            controller._set_run_state(
                ConsoleRunState(
                    ConsoleRunStatus.BLOCKED, "Required hook initialization failed."
                ),
                session_id=session_id,
            )
            return ConsoleSubmitResult(
                False, False, "Required hook initialization failed."
            )
    return None


async def submission_hook_input(
    controller, draft, origin, provider_messages=()
) -> list[dict[str, Any]]:
    """Fire input under the exact task's existing scope and deliver its context."""
    hook_owner = getattr(controller, "_hooks_v2_submissions", {}).get(
        asyncio.current_task()
    )
    if hook_owner is None or hook_owner[0] is None:
        return []
    lifecycle, scope, _session_key = hook_owner
    if origin is ConsoleSubmissionOrigin.MANUAL:
        await lifecycle.fire(
            lifecycle.event("UserPromptSubmit", data={"prompt": draft}), scope
        )
    await lifecycle.wait(scope)
    rows = lifecycle.context.blocks(scope, "model")
    from tldw_chatbook.Agents.agent_models import check_host_context

    check_host_context([*provider_messages, *rows], strip=False)
    return rows


async def legacy_submission_hook_input(controller, draft, session_id):
    """Use the existing legacy engine and bounded prompt projection."""
    from tldw_chatbook.Agents.run_hooks import truncate_hook_text

    hooks_engine = controller._run_hooks_engine()
    return (
        await hooks_engine.fire_async(
            "UserPromptSubmit",
            session_id=session_id,
            data={"prompt": truncate_hook_text(draft)},
        )
        if hooks_engine is not None
        else None
    )


def hook_admission_reason(controller) -> str | None:
    from tldw_chatbook.Agents.run_hooks import inspect_hooks_config
    from tldw_chatbook.config import read_hooks_config_snapshot

    try:
        if controller._hook_permissions_accessor is not None:
            return controller._hook_permissions_accessor().snapshot().blocked_reason
        saved = read_hooks_config_snapshot()
        inventory = inspect_hooks_config(
            {"hooks": saved.section} if saved.section_present else {}
        )
        if not inventory.requires_authority:
            return None
        return "Hook review required; permission owner unavailable."
    except Exception:  # noqa: BLE001 -- unavailable authority must refuse admission
        return "Hooks unavailable; review or disable hooks before sending."


def run_hooks_engine(controller):
    """Resolve the app-owned run-hooks engine for this send, or ``None``.

    Spec 2026-09-11 (Task 7): the submit path reaches the engine through
    the optional ``ensure_run_hooks`` accessor (the Task 5 bridge seam --
    a bound ``ConsoleRuntime.ensure_run_hooks``, or a test double).
    ``None`` -- no accessor wired, or no ``[hooks]`` configured -- means
    the fire site skips entirely; the accessor is consulted per send, so
    an engine built after the first-ever ``[hooks]`` entry appears is
    picked up without rebuilding this controller.
    """
    accessor = controller._ensure_run_hooks
    return accessor() if accessor is not None else None
