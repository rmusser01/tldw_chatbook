"""Redacted real-provider smoke for Console conversation-memory closeout.

This helper never prints API keys, prompts, transcript text, or summary text.
It exercises the production auxiliary provider boundary once successfully and
once with a deliberately invalid model, then reports the policy-decision matrix
using the same configured provider/model identity and canonical policy code.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json

from tldw_chatbook.Chat.console_context_compaction import decide_compaction
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextCapacity,
    ConsoleContextPolicyOverrides,
    ConsoleContextPolicyDefaults,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    resolve_context_policy,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_repository import ConsoleContextRepository
from tldw_chatbook.Chat.console_provider_support import (
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ChatProviderError,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Utils.token_counter import get_table_model_token_limit
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.config import get_api_key


def _resolution(provider: str, model: str, api_key: str) -> ConsoleProviderResolution:
    identity = resolve_console_provider_identity(provider)
    if not identity.is_supported:
        raise RuntimeError(f"Console does not support provider {provider}.")
    return ConsoleProviderResolution(
        provider=provider,
        ready=True,
        readiness_key=identity.readiness_key,
        execution_key=identity.execution_key,
        model=model,
        api_key=api_key,
        base_url="",
        max_tokens=96,
        streaming=False,
    )


def _decision(
    mode: ContextCompactionMode,
    *,
    window: int | None,
    conversation_tokens: int,
    compactable_units: int = 2,
) -> tuple[str, tuple[str, ...]]:
    resolved = resolve_context_policy(
        capacity=ConsoleContextCapacity(
            model_context_window_tokens=window,
            response_reservation_tokens=96,
            safety_margin_tokens=128 if window is not None else 0,
            mandatory_input_tokens=64,
        ),
        application_defaults=ConsoleContextPolicyDefaults(
            compaction_mode=mode,
            trigger_ratio=0.80,
            target_ratio=0.55,
            summary_max_tokens=96,
        ),
    )
    decision = decide_compaction(
        resolved,
        conversation_tokens=conversation_tokens,
        compactable_units=compactable_units,
    )
    return decision.value, tuple(
        "context_policy_validation_error" for _ in resolved.validation_errors
    )


async def _controller_scenario(
    gateway: ConsoleProviderGateway,
    resolution: ConsoleProviderResolution,
    mode: ContextCompactionMode,
    *,
    budget_tokens: int,
) -> dict[str, object]:
    """Drive production Console preflight with private in-memory persistence."""

    db = CharactersRAGDB(":memory:", client_id=f"live-{mode.value}")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title=f"live-{mode.value}")
    store.persist_session_if_needed(session.id)
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=budget_tokens,
            compaction_mode=mode,
            trigger_ratio=0.80,
            target_ratio=0.55,
            summary_max_tokens=96,
            failure_behavior=CompactionFailureBehavior.STOP_AND_ASK,
            carry_forward_mode=ContextCarryForwardMode.MEMORY_WITH_RECENT_TURNS,
        ),
    )
    for index in range(2):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=f"neutral question {index} " + "alpha " * 450,
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"neutral answer {index} " + "beta " * 450,
            persist=True,
        )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="current neutral request",
        persist=True,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    repository = ConsoleContextRepository(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        context_repository=repository,
    )
    transcript_count = len(store.messages_for_session(session.id))
    _projected, blocked = await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=resolution,
        provider_messages=controller._provider_messages_for_session(
            session.id,
            annotate_ids=True,
        ),
        assistant_message_id=assistant.id,
        agent_tools_enabled=False,
    )
    owner = next(item for item in store.sessions() if item.id == session.id)
    conversation_id = owner.persisted_conversation_id
    attempts = (
        repository.list_auxiliary_attempts(conversation_id)
        if conversation_id is not None
        else ()
    )
    usage = (
        ProviderUsage.from_json(attempts[0]["provider_usage_json"])
        if attempts and attempts[0]["provider_usage_json"]
        else None
    )
    result = {
        "blocked": blocked is not None,
        "auxiliary_attempt_count": len(attempts),
        "auxiliary_status": attempts[0]["status"] if attempts else None,
        "usage_total_tokens": usage.total_tokens if usage is not None else None,
        "pricing_provenance_present": bool(
            attempts and attempts[0]["pricing_provenance_json"]
        ),
        "memory_count": (
            len(repository.list_active_memories(conversation_id))
            if conversation_id is not None
            else 0
        ),
        "transcript_unchanged": (
            len(store.messages_for_session(session.id)) == transcript_count
        ),
    }
    db.close_connection()
    return result


async def _run(provider: str, model: str) -> dict[str, object]:
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No configured credential for {provider}.")
    resolution = _resolution(provider, model, api_key)
    gateway = ConsoleProviderGateway()
    request = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=(
            {
                "role": "system",
                "content": "Summarize the supplied neutral facts in one short sentence.",
            },
            {
                "role": "user",
                "content": "Fact A is alpha. Fact B is beta. Preserve both facts.",
            },
        ),
        response_format=None,
        max_output_tokens=96,
    )
    below_threshold = await _controller_scenario(
        gateway,
        resolution,
        ContextCompactionMode.AUTOMATIC,
        budget_tokens=20_000,
    )
    ask = await _controller_scenario(
        gateway,
        resolution,
        ContextCompactionMode.ASK,
        budget_tokens=1_800,
    )
    automatic = await _controller_scenario(
        gateway,
        resolution,
        ContextCompactionMode.AUTOMATIC,
        budget_tokens=1_800,
    )
    off = await _controller_scenario(
        gateway,
        resolution,
        ContextCompactionMode.OFF,
        budget_tokens=1_800,
    )

    failure_status = "unexpected_success"
    try:
        await gateway.complete_auxiliary(
            replace(
                request,
                resolution=replace(
                    resolution,
                    model="tldw-live-invalid-model-id-for-failure-check",
                ),
            )
        )
    except ChatProviderError:
        failure_status = "provider_error_redacted"

    window = get_table_model_token_limit(model, provider)
    matrix = {
        "below_threshold": _decision(
            ContextCompactionMode.AUTOMATIC,
            window=window,
            conversation_tokens=100,
        )[0],
        "ask": _decision(
            ContextCompactionMode.ASK,
            window=2_000,
            conversation_tokens=1_500,
        )[0],
        "automatic": _decision(
            ContextCompactionMode.AUTOMATIC,
            window=2_000,
            conversation_tokens=1_500,
        )[0],
        "off": _decision(
            ContextCompactionMode.OFF,
            window=2_000,
            conversation_tokens=1_500,
        )[0],
        "unknown_model_window": _decision(
            ContextCompactionMode.AUTOMATIC,
            window=None,
            conversation_tokens=1_500,
        )[0],
        "overhead_exceeds_budget": _decision(
            ContextCompactionMode.AUTOMATIC,
            window=128,
            conversation_tokens=1_500,
        ),
    }
    return {
        "provider": provider,
        "model": model,
        "detected_context_window_tokens": window,
        "auxiliary_call": {
            "status": automatic["auxiliary_status"],
            "usage_present": automatic["usage_total_tokens"] is not None,
            "usage_total_tokens": automatic["usage_total_tokens"],
            "pricing_provenance_present": automatic["pricing_provenance_present"],
        },
        "summary_failure": failure_status,
        "policy_matrix": matrix,
        "console_preflight": {
            "below_threshold": below_threshold,
            "ask": ask,
            "automatic": automatic,
            "off": off,
        },
        "private_content_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--confirm-live-provider-cost",
        action="store_true",
        help="Confirm that this smoke test may make billable provider requests.",
    )
    args = parser.parse_args()
    if not args.confirm_live_provider_cost:
        parser.error("--confirm-live-provider-cost is required")
    evidence = asyncio.run(_run(args.provider.strip(), args.model.strip()))
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
