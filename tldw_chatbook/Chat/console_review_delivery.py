"""Console controller seams for multi-model code review (ADR-281).

Moved out of ``ConsoleChatController`` (module size ratchet): presenting
reports, offers and status on the owner loop, admitting an accepted review
wake, composing a turn's review tools, and attaching the review ledger. Each
function takes the controller explicitly; the controller keeps thin methods.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


def _on_owner_loop(controller: Any) -> tuple[Any, bool]:
    loop = controller._fleet_wake._loop
    try:
        return loop, asyncio.get_running_loop() is loop
    except RuntimeError:
        return loop, False


def present_review_status(controller: Any, session_id, review_id, status) -> None:
    """Post a review status line on the owner loop."""

    def present():
        if controller._disposed or session_id in controller._session_close_generations:
            return
        controller.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"Review {review_id}: {status}. Captured progress is retained; a new explicit request can resume an available stage.",
        )

    loop, on_loop = _on_owner_loop(controller)
    if on_loop:
        present()
    elif loop is not None and loop.is_running():
        loop.call_soon_threadsafe(present)


def present_review_offer(controller: Any, session_id, review_id) -> bool:
    """Offer verification; never block the owner loop on its own coroutine."""

    def present():
        if controller._disposed or session_id in controller._session_close_generations:
            return False
        controller.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"Verification is available for review {review_id}. Reply yes to verify this review, or no to decline.",
        )
        return True

    async def hop():
        return present()

    loop, on_loop = _on_owner_loop(controller)
    if on_loop:
        # Blocking on a coroutine scheduled onto our own loop deadlocks.
        return present()
    if loop is not None and loop.is_running():
        return asyncio.run_coroutine_threadsafe(hop(), loop).result()
    return False


def present_review_report(controller: Any, session_id, review_id, report) -> bool:
    """Acknowledge mounted native originals before an authorized verifier starts."""
    from ..Widgets.Console.console_code_review_modal import ConsoleCodeReviewModal

    def available():
        cancel = controller._active_cancel_events.get(session_id)
        return (
            not controller._disposed
            and session_id not in controller._session_close_generations
            and controller.store.active_session_id == session_id
            and not (cancel is not None and cancel.is_set())
        )

    async def present():
        if not available():
            return False
        try:
            _, service = controller.code_review.report_service(session_id, review_id)
            page = await asyncio.to_thread(
                ConsoleCodeReviewModal.literal_page, report["original_page"]
            )
            modal = controller.app.screen
            if (
                not isinstance(modal, ConsoleCodeReviewModal)
                or modal._review_id != review_id
            ):
                modal = ConsoleCodeReviewModal(service=service, review_id=review_id)
                await controller.app.push_screen(modal)
            if not available():
                await modal.request_safe_cancel(source="review_closed")
                return False
            if not modal.present_original(report["slot_id"], page):
                return False
            controller.store.append_message(
                session_id,
                role=ConsoleMessageRole.ASSISTANT,
                content=f"Original report {report['slot_id']} for review {review_id} is open in Review reports (u). Full originals and raw output are retained for this session; export to keep them. Saved chat references cannot restore an expired report bundle.",
            )
            return True
        except Exception:  # noqa: BLE001 - closed UI never acknowledges
            # A closed/unavailable screen does not acknowledge presentation.
            return False

    loop, on_loop = _on_owner_loop(controller)
    if not on_loop and loop is not None and loop.is_running():
        call_from_thread = getattr(controller.app, "call_from_thread", None)
        if callable(call_from_thread):
            return bool(call_from_thread(present))
    return False


async def resolve_accepted_review_capacity(
    controller: Any, session, authorization, resolution, turn_context
):
    """Resolve only this accepted review target before capacity/schema freeze."""
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.DB.base_db import run_owned_db_call

    receipt = authorization.review_boundary
    context = authorization.context

    def validate_binding():
        if (
            not controller._fleet_wake.authorizes(authorization, session.id)
            or not authorization.accepted
            or controller._disposed
            or controller._shutdown_requested.is_set()
            or session.id in controller._session_close_generations
            or controller.store._session_or_raise(session.id) is not session
            or authorization.review_boundary is not receipt
            or authorization.context is not context
            or context.chain_id != receipt.chain_id
            or context.attempt_id != authorization.attempt_id
            or context.owner_id != authorization.owner_id
            or receipt.session_id != session.id
            or receipt.conversation_id != authorization.conversation_id
            or resolution.resolved_destination != turn_context.resolved_destination
            or not controller._fleet_wake._priority_allows(session.id, accepting=True)
        ):
            raise ReviewRefused("review_boundary_unavailable")

    def validate_attempt():
        snapshot = context.check()
        attempt = context.ledger.read_attempt(
            context.attempt_id, owner_id=context.owner_id
        )
        if (
            attempt.state != "accepted"
            or attempt.review_boundary_id != receipt.boundary_id
            or attempt.session_id != session.id
            or attempt.conversation_id != receipt.conversation_id
        ):
            raise ReviewRefused("review_boundary_unavailable")
        return snapshot

    async def validate():
        validate_binding()
        snapshot = await run_owned_db_call(context.ledger._db, validate_attempt)
        # Source/native guards use composition's off-loop owner; all locks
        # are released before the optional metadata await.
        await asyncio.to_thread(
            controller.code_review._accepted_boundary_service, receipt
        )
        validate_binding()
        return snapshot

    snapshot = await validate()
    ceiling = controller.PROVIDER_VALIDATION_TIMEOUT_SECONDS
    if snapshot.deadline_at is not None:
        ceiling = min(ceiling, max(0.0, snapshot.deadline_at - time.time()))
    async with asyncio.timeout(ceiling):
        window = await controller.provider_gateway.resolve_context_window(resolution)
        await validate()
    if not window.verified:
        raise ReviewRefused("review_context_capacity_unknown")
    return replace(resolution, context_window=window)


async def admit_review_wake(
    controller: Any, session, authorization, resolution, turn_context
):
    """Return the capacity-pinned resolution, or the block result to return.

    Accepted automatic work fails closed: any refusal pauses the chain and
    keeps the reports.
    """
    try:
        return await controller._resolve_accepted_review_capacity(
            session, authorization, resolution, turn_context
        )
    except asyncio.CancelledError:
        await controller._fleet_wake._pause(
            authorization.conversation_id,
            authorization.work_chain_id,
            "review_preflight_cancelled",
        )
        raise
    except Exception as exc:  # noqa: BLE001 - accepted work must fail closed
        from tldw_chatbook.Agents.code_review_models import ReviewRefused

        reason = (
            exc.reason
            if isinstance(exc, ReviewRefused)
            else "review_capacity_unavailable"
        )
        controller.code_review.diagnostic = reason
        await controller._fleet_wake._pause(
            authorization.conversation_id, authorization.work_chain_id, reason
        )
        return controller._block(
            session.id, f"Review continuation blocked: {reason}. Reports are saved."
        )


async def compose_turn_review_provider(
    controller: Any,
    *,
    session_id,
    message,
    roots,
    resolution,
    conversation_id,
    wake_authorization,
    tool_configuration,
    skill_bindings,
    project_selection,
    project_authority_guard,
    confirm_project_dispatch,
    startup_candidate,
):
    """Compose this turn's review tools; review is optional for ordinary turns."""
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    from .console_code_review import compose_native_review_provider

    def review_destination_check(target):
        if (
            project_selection is None
            or project_authority_guard is None
            or not project_authority_guard()
        ):
            return "source_authority_unavailable"
        if confirm_project_dispatch is None or startup_candidate is None:
            return "destination_consent_required"
        from tldw_chatbook.Agents.project_instruction_resolver import (
            InstructionChainDelivery,
            InstructionSnapshot,
        )

        snapshot = InstructionSnapshot(
            binding_id=startup_candidate.binding_id,
            binding_root=startup_candidate.binding_root,
            locator_fingerprint=startup_candidate.locator_fingerprint,
            dispatch_started_wall_ns=startup_candidate.dispatch_started_wall_ns,
            startup_source=startup_candidate.source,
            global_outcomes=startup_candidate.outcomes,
            primary_delivery=InstructionChainDelivery(
                source_digests=(), outcomes=startup_candidate.outcomes
            ),
            warning_codes=(),
        )
        selected = replace(
            resolution,
            provider=target.provider,
            execution_key=target.execution_provider or target.provider,
            model=target.model,
            base_url=target.base_url,
        )
        return (
            None
            if confirm_project_dispatch(snapshot, selected) == "proceed"
            else "destination_consent_required"
        )

    try:
        return await asyncio.to_thread(
            compose_native_review_provider,
            controller,
            session_id=session_id,
            message=message,
            roots=roots,
            resolution=resolution,
            destination_check=review_destination_check,
            conversation_id=conversation_id,
            wake_authorization=wake_authorization,
            tool_configuration=tool_configuration,
            trusted_skill_bindings=skill_bindings,
        )
    except ReviewRefused as exc:
        controller.code_review.diagnostic = exc.reason
        controller.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"Review needs attention: {exc.reason}. No review stage was admitted.",
        )
    except Exception as exc:  # noqa: BLE001 - review tools are optional
        logger.warning(
            "Console review tools unavailable; continuing the agent turn "
            "without them | exception_type={}",
            type(exc).__name__,
        )
    return None


def attach_review_ledger(controller: Any, bridge: Any) -> None:
    """Move review-boundary authority to this bridge's ledger, if loaded."""
    with controller._code_review_lock:
        database = getattr(bridge, "runs_db", None)
        ledger = (
            getattr(database, "automatic_work", None)
            if controller._code_review is not None
            else None
        )
        if ledger is controller._review_ledger:
            return
        owner_id = controller._fleet_wake.runtime_owner_id
        if controller._review_ledger is not None:
            controller._review_ledger.unregister_review_boundary_authority(owner_id)
            controller.code_review.reset_runtime()
        controller._review_ledger = ledger
        if ledger is not None:
            review = controller.code_review
            ledger.register_review_boundary_authority(
                owner_id,
                validate=review.validate_boundary,
                transition=review.boundary_transition,
                acceptance_scope=review.boundary_acceptance_scope,
            )
