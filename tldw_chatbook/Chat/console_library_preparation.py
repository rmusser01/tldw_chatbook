"""Frozen Library preparation delegated from the size-governed controller."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from loguru import logger

from tldw_chatbook.Chat.library_preparation import (
    LibraryPreparationContribution,
    library_preparation_event_for_outcome,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchRequest
from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
    build_library_rag_evidence_bundle,
)

from .console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)

if TYPE_CHECKING:
    from .console_chat_controller import (
        ConsoleChatController,
        ConsolePreparationOutcome,
    )

#: TASK-33621.51: a paused search's error code -> the reason its "Send once
#: without Library" record keeps (the card already says it; the turn now too).
_BYPASS_REASONS = MappingProxyType(
    {
        "library_retrieval_timeout": "timed_out",
        "library_retrieval_stopped": "stopped",
        "library_retrieval_failed": "failed",
    }
)


def preparation_contribution(
    controller: ConsoleChatController,
    outcome: Literal["zero_matches", "bypassed", "evidence_sent"],
    preparation: ConsoleTurnPreparation,
    *,
    result_count: int = 0,
    bypass_reason: str | None = None,
) -> LibraryPreparationContribution:
    """Build the bounded record the send's durable acceptance writes.

    Args:
        controller: The Console chat controller that prepared the send.
        outcome: What the Automatic Library search did for this send.
        preparation: The send's READY preparation.
        result_count: Evidence references the send carries (evidence only).
        bypass_reason: Why a bypassed send's search paused, when known.

    Returns:
        The contribution, owned by the send's USER message.
    """
    event = library_preparation_event_for_outcome(
        outcome,
        attempt_id=preparation.attempt_id,
        result_count=result_count,
        source_types=controller._frozen_rag_source_types(
            preparation.execution_context
        ),
        bypass_reason=bypass_reason,
    )
    assert event is not None
    return LibraryPreparationContribution(event=event)


def bypass_contribution(
    controller: ConsoleChatController, ready: ConsoleTurnPreparation
) -> LibraryPreparationContribution:
    """Record a "Send once without Library", keeping why its search paused.

    Args:
        controller: The Console chat controller holding the paused outcome.
        ready: The preparation the bypass has just moved to READY.

    Returns:
        A ``bypassed`` contribution; its reason is None for an unknown code.
    """
    paused = controller._preparation_outcomes.get(ready.preparation_id)
    reason = _BYPASS_REASONS.get(str(getattr(paused, "error_code", None) or ""))
    return preparation_contribution(
        controller, "bypassed", ready, bypass_reason=reason
    )


async def prepare_library_for_turn(
    controller: ConsoleChatController, preparation_id: str
) -> ConsolePreparationOutcome:
    """Run one frozen automatic retrieval attempt and CAS its result."""

    from .console_chat_controller import ConsolePreparationOutcome

    preparation = controller._preparation_by_id(preparation_id)
    if preparation is None:
        raise KeyError(preparation_id)
    if preparation.state is not ConsoleTurnPreparationState.PREPARING:
        existing = controller._preparation_outcomes.get(preparation_id)
        if existing is not None and existing.attempt_id == preparation.attempt_id:
            return existing
        return ConsolePreparationOutcome(
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            state=preparation.state,
            evidence_bundle=None,
            contribution=None,
            error_code=None,
        )

    authority = preparation.execution_context.library_authority
    request = LibraryRagSearchRequest(
        query=preparation.executed_draft,
        source_types=controller._frozen_rag_source_types(preparation.execution_context),
        mode="rag",
        top_k=controller._frozen_rag_top_k(preparation.execution_context),
        include_citations=True,
        scope=controller._automatic_scope_for_authority(authority),
    )
    # TASK-33621.20: bounded, shown ("Searching Library…") and stoppable.
    from .console_library_search import automatic_search_outcome

    result, error_code = await automatic_search_outcome(
        controller, preparation.session_id, request
    )
    results = tuple(getattr(result, "results", ()) or ()) if result else ()
    status = str(getattr(result, "status", "") or "") if result else ""
    if error_code is None and status not in {"ready", "empty"}:
        error_code = "library_retrieval_failed"
    if error_code is not None:
        logger.warning("Console Library preparation paused | error_code={}", error_code)
        paused = controller.store.compare_and_set_preparation(
            preparation.session_id,
            ConsolePreparationTransition(
                preparation_id=preparation.preparation_id,
                expected_state=ConsoleTurnPreparationState.PREPARING,
                new_state=ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
                new_attempt_id=None,
            ),
        )
        current = paused or controller._preparation_by_id(preparation_id)
        if current is None:
            outcome = ConsolePreparationOutcome(
                preparation_id=preparation.preparation_id,
                attempt_id=preparation.attempt_id,
                state=ConsoleTurnPreparationState.CANCELLED,
                evidence_bundle=None,
                contribution=None,
                error_code=None,
            )
            return outcome
        outcome = ConsolePreparationOutcome(
            preparation_id=current.preparation_id,
            attempt_id=current.attempt_id,
            state=current.state,
            evidence_bundle=None,
            contribution=None,
            error_code=error_code if paused is not None else None,
        )
        controller._preparation_outcomes[preparation_id] = outcome
        return outcome

    bundle = (
        build_library_rag_evidence_bundle(
            results,
            query=preparation.executed_draft,
        )
        if results
        else None
    )
    ready = controller.store.compare_and_set_preparation(
        preparation.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.PREPARING,
            new_state=ConsoleTurnPreparationState.READY,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )
    current = ready or controller._preparation_by_id(preparation_id)
    if current is None:
        outcome = ConsolePreparationOutcome(
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            state=ConsoleTurnPreparationState.CANCELLED,
            evidence_bundle=None,
            contribution=None,
            error_code=None,
        )
        return outcome
    # TASK-33621.51: every completed search leaves a record on the turn: how
    # many evidence references it sent, or that it found none.
    sent = len(bundle.references) if bundle is not None else 0
    contribution = (
        preparation_contribution(
            controller,
            "evidence_sent" if sent else "zero_matches",
            current,
            result_count=sent,
        )
        if ready is not None
        else None
    )
    outcome = ConsolePreparationOutcome(
        preparation_id=current.preparation_id,
        attempt_id=current.attempt_id,
        state=current.state,
        evidence_bundle=bundle if ready is not None else None,
        contribution=contribution,
        error_code=None,
    )
    controller._preparation_outcomes[preparation_id] = outcome
    return outcome
