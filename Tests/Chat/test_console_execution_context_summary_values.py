"""Context rail values preserve absence and cannot carry authority objects."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from Tests.Chat.test_console_execution_readiness_values import _readiness
from tldw_chatbook.Chat.console_execution_context_summary_values import (
    ConsoleContextSummaryProjection,
    decode_context_summary,
    encode_context_summary,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCacheState, ConsoleCostState
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsSummaryState,
    build_console_settings_summary_state,
)


def _empty_summary():
    return ConsoleSettingsSummaryState("", "", "", "", action_label="", action_tooltip="")


def test_cost_roundtrip_keeps_absence_zero_and_original_cache_state_distinct():
    estimate, summary = ConsoleSettingsContextEstimate(None, None, ""), _empty_summary()
    for cost, cache in (
        (None, ConsoleCacheState.NONE),
        (ConsoleCostState("$0.00", "$0.00", "Empty", False, False), ConsoleCacheState.NONE),
        (ConsoleCostState("$18.00", "$18.00", "Actual usage", True, False), ConsoleCacheState.WARM),
        (ConsoleCostState("$18.00", "$18.00", "Expired", False, True), ConsoleCacheState.EXPIRED),
    ):
        payload = encode_context_summary(
            estimate, summary, compaction_pause_expected=False,
            compaction_forecast_unavailable=False, cost=cost, cost_cache_state=cache,
        )
        restored = decode_context_summary(json.loads(json.dumps(payload)))
        assert restored.cost == cost and restored.cost_cache_state is cache
        bad = deepcopy(payload)
        bad["cost_cache_state"] = "ready"
        with pytest.raises(ConsoleProtocolError):
            decode_context_summary(bad)
        if cost is not None:
            bad = deepcopy(payload)
            bad["cost"]["alert"] = 1
            with pytest.raises(ConsoleProtocolError):
                decode_context_summary(bad)
            bad = deepcopy(payload)
            bad["cost"]["authority"] = {}
            with pytest.raises(ConsoleProtocolError):
                decode_context_summary(bad)


def test_unknown_zero_and_signed_estimates_keep_explicit_flags():
    unknown = ConsoleSettingsContextEstimate(None, None, "")
    encoded = encode_context_summary(
        unknown, _empty_summary(),
        compaction_pause_expected=True, compaction_forecast_unavailable=False,
    )
    restored = decode_context_summary(json.loads(json.dumps(encoded, allow_nan=False)))
    assert type(restored) is ConsoleContextSummaryProjection
    assert type(restored.estimate) is ConsoleSettingsContextEstimate
    assert type(restored.summary) is ConsoleSettingsSummaryState
    assert restored.estimate == unknown
    assert restored.summary == _empty_summary()
    assert restored.compaction_pause_expected is True
    assert restored.compaction_forecast_unavailable is False
    assert restored.estimate.token_limit_verified is None
    assert len(encoded["estimate"]) == 7 and len(encoded["summary"]) == 15

    zero = replace(unknown, used_tokens=0, token_limit=0, token_limit_verified=False)
    restored_zero = decode_context_summary(encode_context_summary(
        zero, _empty_summary(),
        compaction_pause_expected=False, compaction_forecast_unavailable=True,
    ))
    assert restored_zero.estimate == zero and restored_zero.estimate != unknown
    assert restored_zero.compaction_pause_expected is False
    assert restored_zero.compaction_forecast_unavailable is True

    # The original constructors/builders do not clamp injected integer values.
    signed = replace(zero, used_tokens=-3, token_limit=-1, staged_source_count=-2)
    assert decode_context_summary(encode_context_summary(
        signed, _empty_summary(),
        compaction_pause_expected=False, compaction_forecast_unavailable=False,
    )).estimate == signed


def test_original_summary_and_nested_readiness_roundtrip_without_shared_containers():
    estimate = ConsoleSettingsContextEstimate(
        12, 4096, "12 / 4,096 tokens", 1, "one source", True, "provided resolver"
    )
    readiness = _readiness()
    summary = build_console_settings_summary_state(
        ConsoleSessionSettings(provider="custom", model="model-a", streaming=False),
        estimate, readiness,
    )
    encoded = encode_context_summary(
        estimate, summary,
        compaction_pause_expected=False, compaction_forecast_unavailable=False,
    )
    restored = decode_context_summary(encoded)
    assert restored.estimate == estimate and restored.summary == summary
    assert restored.summary.readiness == readiness
    assert restored.summary.readiness is not readiness
    assert restored.summary.readiness.evidence is not readiness.evidence
    encoded["summary"]["readiness"]["connection"]["connection_identity"][0] = "changed"
    encoded["estimate"]["label"] = "changed"
    assert restored.summary.readiness.connection == readiness.connection
    assert restored.estimate.label == estimate.label
    with pytest.raises(FrozenInstanceError):
        restored.compaction_pause_expected = True


def test_closed_shapes_exact_primitives_and_authority_objects_are_refused():
    estimate, summary = ConsoleSettingsContextEstimate(None, None, ""), _empty_summary()
    payload = encode_context_summary(
        estimate, summary,
        compaction_pause_expected=False, compaction_forecast_unavailable=False,
    )
    with pytest.raises(ConsoleProtocolError):
        decode_context_summary(MappingProxyType(payload))
    extra = deepcopy(payload)
    extra["context_control_state"] = object()
    with pytest.raises(ConsoleProtocolError):
        decode_context_summary(extra)
    wrong_integer = deepcopy(payload)
    wrong_integer["estimate"]["used_tokens"] = True
    with pytest.raises(ConsoleProtocolError):
        decode_context_summary(wrong_integer)
    wrong_flag = deepcopy(payload)
    wrong_flag["compaction_pause_expected"] = 0
    with pytest.raises(ConsoleProtocolError):
        decode_context_summary(wrong_flag)
    nested_extra = deepcopy(payload)
    nested_extra["summary"]["settings_policy"] = {}
    with pytest.raises(ConsoleProtocolError):
        decode_context_summary(nested_extra)

    class Authority:
        def __str__(self) -> str:
            raise AssertionError("Authority must never be rendered or serialized")

    with pytest.raises(ConsoleProtocolError):
        encode_context_summary(
            estimate, replace(summary, model_row=Authority()),
            compaction_pause_expected=False, compaction_forecast_unavailable=False,
        )
    with pytest.raises(ConsoleProtocolError):
        encode_context_summary(
            estimate, replace(summary, readiness=Authority()),
            compaction_pause_expected=False, compaction_forecast_unavailable=False,
        )
