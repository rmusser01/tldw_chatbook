from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _frozen_context(
    *,
    auto_retrieve: ConsoleAutoRetrieve = ConsoleAutoRetrieve.NEVER,
) -> ConsoleTurnExecutionContext:
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=auto_retrieve,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=4,
        source="durable",
    )
    scope = ConsoleLibraryItemScopeSnapshot((), (), True)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-voice",
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-test",
            system_prompt="Frozen system prompt",
        ),
        tool_configuration={"agent_runtime_enabled": True},
        library_policy_maximum=policy,
        library_scope_maximum=scope,
    )
    authority = ConsoleTurnLibraryAuthority(
        policy=policy,
        direct_library_tools=True,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=scope,
        provider_intent=ConsoleProviderIntent("openai", "gpt-test", None),
        attempt_id="authority-attempt",
    )
    return ConsoleTurnExecutionContext(
        configuration=configuration,
        library_authority=authority,
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="gpt-test",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _prepared_request(
    *, tools: list[dict[str, Any]] | None = None
) -> PreparedProviderRequest:
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    return ConsoleProviderGateway(  # type: ignore[arg-type]
        http_client=object()
    ).prepare_chat_request(
        resolution,
        [
            {"role": "system", "content": "Frozen system prompt"},
            {"role": "user", "content": "Exact rolling transcript"},
        ],
        tools=tools,
    )


def _eligibility_api():
    from tldw_chatbook.Chat.console_voice_eligibility import (
        VoiceSpeculationDecision,
        classify_voice_speculation,
    )

    return VoiceSpeculationDecision, classify_voice_speculation


def test_plain_frozen_provider_turn_is_provisional_even_with_tool_schemas() -> None:
    decision_type, classify = _eligibility_api()
    prepared = _prepared_request(tools=TOOLS)

    decision = classify(
        frozen_session_context=_frozen_context(),
        prepared_request=prepared,
    )

    assert decision is decision_type.PROVISIONAL
    assert prepared.tools[0]["function"]["name"] == "weather"


def test_automatic_library_retrieval_waits_for_the_ordinary_pipeline() -> None:
    decision_type, classify = _eligibility_api()

    decision = classify(
        frozen_session_context=_frozen_context(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC
        ),
        prepared_request=_prepared_request(),
    )

    assert decision is decision_type.WAIT_FOR_STABLE_TURN


@pytest.mark.parametrize(
    "effect_flag",
    ["requires_citation_creation", "requires_pre_dispatch_authority"],
)
def test_other_pre_dispatch_effects_wait_for_the_ordinary_pipeline(
    effect_flag: str,
) -> None:
    decision_type, classify = _eligibility_api()

    decision = classify(
        frozen_session_context=_frozen_context(),
        prepared_request=_prepared_request(tools=TOOLS),
        **{effect_flag: True},
    )

    assert decision is decision_type.WAIT_FOR_STABLE_TURN
