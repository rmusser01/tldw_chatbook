"""TASK-34350: the Console preflight routes an over-budget send three ways.

Owner ruling (2026-10-03): prompt to compact (compaction mode Ask), compact
automatically (Automatic), or alert when compacting cannot make room. An
estimated (fallback) context window takes the same routes as a verified one;
its alert says it is an estimate and where to set the real window.

These drive the real ``_apply_conversation_memory_preflight`` with the real
capacity resolver (``resolve_request_capacity``) and context policy, each in a
fresh real profile (``private_profile_test``).
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from Tests.Chat.test_console_context_compaction import (
    _controller_preflight_fixture,
    _resolution,
)
from tldw_chatbook.Chat.console_context_budget_copy import (
    MAX_TOKENS_SETTING,
    MODEL_WINDOW_SETTING,
    ContextOverflowCause,
    context_overflow_alert_copy,
)
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode


async def _preflight(controller, session, assistant, provider_messages):
    return await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=_resolution(),
        provider_messages=provider_messages,
        assistant_message_id=assistant.id,
        agent_tools_enabled=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("estimated", [False, True])
@private_profile_test
async def test_a_window_the_reservation_fills_alerts_with_max_tokens(request, estimated) -> None:
    # 600-token window: Max tokens (120) plus the 512-token minimum safety
    # margin leave no input capacity. Compacting cannot help.
    controller, store, session, assistant, gateway, provider_messages = (
        _controller_preflight_fixture(
            ContextCompactionMode.AUTOMATIC,
            context_window_tokens=600,
            context_window_verified=not estimated,
        )
    )

    _output, result = await _preflight(
        controller, session, assistant, provider_messages
    )

    assert result is not None
    assert gateway.calls == 0
    assert result.visible_copy == context_overflow_alert_copy(
        ContextOverflowCause.NO_INPUT_CAPACITY,
        model="gpt-test",
        window_tokens=600,
        window_estimated=estimated,
        response_tokens=120,
        input_ceiling_tokens=0,
    )
    assert MAX_TOKENS_SETTING in result.visible_copy
    assert (MODEL_WINDOW_SETTING in result.visible_copy) is estimated
    assert store.get_message(assistant.id).status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("estimated", [False, True])
@private_profile_test
async def test_mandatory_context_over_the_window_alerts_with_what_fills_it(
    request,
    estimated,
) -> None:
    controller, _store, session, assistant, gateway, provider_messages = (
        _controller_preflight_fixture(
            ContextCompactionMode.AUTOMATIC,
            context_window_tokens=635,
            context_window_verified=not estimated,
        )
    )

    _output, result = await _preflight(
        controller, session, assistant, provider_messages
    )

    assert result is not None
    assert gateway.calls == 0
    assert result.visible_copy.startswith("Your message was not sent:")
    assert "system prompt, tools and attached context" in result.visible_copy
    assert MAX_TOKENS_SETTING in result.visible_copy
    assert ("an estimate" in result.visible_copy) is estimated


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [ContextCompactionMode.ASK, ContextCompactionMode.AUTOMATIC])
@private_profile_test
async def test_an_estimated_window_takes_the_same_route_as_a_verified_one(request, mode) -> None:
    outcomes = []
    for verified in (True, False):
        controller, store, session, assistant, gateway, provider_messages = (
            _controller_preflight_fixture(mode, context_window_verified=verified)
        )
        _output, result = await _preflight(
            controller, session, assistant, provider_messages
        )
        outcomes.append(
            (
                result is not None,
                gateway.calls,
                store.get_message(assistant.id).status,
            )
        )

    assert outcomes[0] == outcomes[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget_mode", "expected"),
    [("custom", False), ("automatic", True)],
)
@private_profile_test
async def test_the_hold_calls_its_budget_an_estimate_only_when_the_window_sets_it(
    request, budget_mode, expected
) -> None:
    """Live 2026-10-04: a custom 1,500-token budget on an estimated window
    was described as coming "from an estimated context window". A custom
    budget below capacity does not come from the window at all."""
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextBudgetMode,
    )

    overrides = (
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=1_800,
            compaction_mode=ContextCompactionMode.ASK,
            summary_max_tokens=100,
        )
        if budget_mode == "custom"
        else ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.AUTOMATIC,
            compaction_mode=ContextCompactionMode.ASK,
            summary_max_tokens=100,
        )
    )
    # 4,000-token estimated window: capacity after Max tokens and the margin
    # is well above the 1,800 custom budget.
    controller, _store, session, assistant, _gateway, provider_messages = (
        _controller_preflight_fixture(
            ContextCompactionMode.ASK,
            context_window_tokens=2_600 if budget_mode == "automatic" else 4_000,
            overrides=overrides,
            context_window_verified=False,
        )
    )
    captured = []

    await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=_resolution(),
        provider_messages=provider_messages,
        assistant_message_id=assistant.id,
        agent_tools_enabled=False,
        assessment_sink=lambda hold, decision, _alert: captured.append(
            (hold, decision)
        ),
    )

    assert len(captured) == 1
    hold, _decision = captured[0]
    assert hold.estimated is expected
