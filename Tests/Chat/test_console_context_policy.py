from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextCapacity,
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
    ContextPolicyError,
    context_policy_overrides_from_console_config,
    merge_context_policy,
    resolve_context_policy,
)


def test_policy_precedence_is_field_by_field() -> None:
    global_overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=40_000,
        compaction_mode=ContextCompactionMode.AUTOMATIC,
        compaction_representation=ContextCompactionRepresentation.HYBRID,
        trigger_ratio=0.85,
        target_ratio=0.60,
        failure_behavior=CompactionFailureBehavior.OMIT_OLDER_CONTEXT,
    )
    conversation_overrides = ConsoleContextPolicyOverrides(
        custom_budget_tokens=24_000,
        compaction_mode=ContextCompactionMode.OFF,
        carry_forward_mode=ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE,
    )

    policy = merge_context_policy(
        global_overrides=global_overrides,
        conversation_overrides=conversation_overrides,
    )

    assert policy.budget_mode is ContextBudgetMode.CUSTOM
    assert policy.custom_budget_tokens == 24_000
    assert policy.compaction_mode is ContextCompactionMode.OFF
    assert policy.compaction_representation is ContextCompactionRepresentation.HYBRID
    assert policy.trigger_ratio == 0.85
    assert policy.target_ratio == 0.60
    assert (
        policy.failure_behavior
        is CompactionFailureBehavior.OMIT_OLDER_CONTEXT
    )
    assert (
        policy.carry_forward_mode
        is ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE
    )
    assert policy.summary_max_tokens == 1024


def test_unknown_model_window_blocks_automatic_budget() -> None:
    resolved = resolve_context_policy(
        capacity=ConsoleContextCapacity(model_context_window_tokens=None)
    )

    assert resolved.effective_conversation_budget_tokens is None
    assert resolved.safety_verified is False
    assert resolved.can_compact is False
    assert any(
        "known model context window" in message
        for message in resolved.validation_errors
    )


def test_unknown_model_window_allows_bounded_custom_threshold_but_not_safety() -> None:
    overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=12_000,
        compaction_mode=ContextCompactionMode.AUTOMATIC,
    )

    resolved = resolve_context_policy(
        capacity=ConsoleContextCapacity(model_context_window_tokens=None),
        conversation_overrides=overrides,
    )

    assert resolved.effective_conversation_budget_tokens == 12_000
    assert resolved.safety_verified is False
    assert resolved.validation_errors == ()
    assert resolved.can_compact is True
    assert any("safety is unverified" in warning for warning in resolved.warnings)


def test_model_switch_reduces_only_effective_budget_and_preserves_override() -> None:
    overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=100_000,
    )
    resolved = resolve_context_policy(
        capacity=ConsoleContextCapacity(
            model_context_window_tokens=32_000,
            provider_input_cap_tokens=30_000,
            response_reservation_tokens=4_000,
            safety_margin_tokens=1_000,
            mandatory_input_tokens=2_000,
        ),
        conversation_overrides=overrides,
    )

    assert resolved.safe_input_ceiling_tokens == 27_000
    assert resolved.available_conversation_capacity_tokens == 25_000
    assert resolved.effective_conversation_budget_tokens == 25_000
    assert overrides.custom_budget_tokens == 100_000
    assert resolved.policy.custom_budget_tokens == 100_000
    assert any("saved intent was preserved" in item for item in resolved.warnings)


def test_mandatory_material_exhaustion_is_an_actionable_error() -> None:
    resolved = resolve_context_policy(
        capacity=ConsoleContextCapacity(
            model_context_window_tokens=8_000,
            response_reservation_tokens=2_000,
            safety_margin_tokens=1_000,
            mandatory_input_tokens=5_000,
        )
    )

    assert resolved.safe_input_ceiling_tokens == 5_000
    assert resolved.available_conversation_capacity_tokens == 0
    assert resolved.effective_conversation_budget_tokens is None
    assert "Mandatory request material leaves no conversation capacity." in (
        resolved.validation_errors
    )


def test_policy_rejects_missing_hysteresis() -> None:
    with pytest.raises(ContextPolicyError, match="differ by at least"):
        merge_context_policy(
            conversation_overrides=ConsoleContextPolicyOverrides(
                trigger_ratio=0.80,
                target_ratio=0.70,
            )
        )


def test_global_console_config_uses_canonical_keys() -> None:
    overrides = context_policy_overrides_from_console_config(
        {
            "conversation_budget_mode": "custom",
            "conversation_budget_tokens": "16000",
            "compaction_mode": "automatic",
            "compaction_representation": "visual_transcript",
            "compaction_trigger_ratio": 0.9,
            "compaction_target_ratio": 0.6,
            "compaction_summary_max_tokens": 512,
            "compaction_failure_behavior": "omit_older_context",
            "compaction_carry_forward_mode": "memory_with_latest_exchange",
        }
    )

    assert overrides.budget_mode is ContextBudgetMode.CUSTOM
    assert overrides.custom_budget_tokens == 16_000
    assert overrides.compaction_mode is ContextCompactionMode.AUTOMATIC
    assert (
        overrides.compaction_representation
        is ContextCompactionRepresentation.VISUAL_TRANSCRIPT
    )
    assert overrides.failure_behavior is CompactionFailureBehavior.OMIT_OLDER_CONTEXT
    assert (
        overrides.carry_forward_mode
        is ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE
    )


def test_sparse_serialization_round_trip_rejects_boolean_integer() -> None:
    original = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=8_000,
        compaction_mode=ContextCompactionMode.ASK,
    )
    assert ConsoleContextPolicyOverrides.from_mapping(original.to_dict()) == original

    with pytest.raises(ContextPolicyError, match="integer"):
        ConsoleContextPolicyOverrides.from_mapping(
            {"custom_budget_tokens": True}
        )

    with pytest.raises(ContextPolicyError, match="ContextBudgetMode"):
        ConsoleContextPolicyOverrides(budget_mode="custom")  # type: ignore[arg-type]
