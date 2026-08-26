"""Canonical provider-continuation checkpoint contracts."""

from __future__ import annotations

import copy
import json
import logging
import traceback
import tracemalloc
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

import tldw_chatbook.Chat.provider_continuation as continuation_module
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationConflictError,
    ContinuationOwnerGroup,
    ContinuationResult,
    ContinuationRestoreTarget,
    ContinuationRound,
    ContinuationValidationError,
    ProviderContinuationCheckpoint,
    continuation_owner_group,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
    read_provider_continuation_json,
    transition_provider_call,
    validate_continuation_restore,
)


def _call(
    call_id: str = "call_1",
    *,
    name: str = "calculator",
    arguments: str = '{"expression":"2+2"}',
    state: str = "pending",
    result: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "state": state,
    }
    if result is not None:
        value["result"] = result
    return value


def _checkpoint(
    *,
    provider: str = "deepseek",
    protocol: str = "responses",
    model: str = "deepseek-v4-flash",
    api_base_url: str = "https://api.deepseek.com/v1",
    state: str = "active",
    rounds: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": provider,
        "protocol": protocol,
        "model": model,
        "api_base_url": api_base_url,
        "state": state,
        "rounds": rounds
        if rounds is not None
        else [
            {
                "assistant_content": "",
                "reasoning_blocks": ["private reasoning"],
                "calls": [_call()],
            }
        ],
    }


@pytest.fixture
def active_tool_checkpoint() -> dict[str, object]:
    return _checkpoint()


@pytest.fixture
def result_checkpoint() -> dict[str, object]:
    return _checkpoint(
        protocol="chat_completions",
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["first", "second"],
                "calls": [
                    _call("call_1", state="completed", result="4"),
                    _call("call_2", state="failed", result="tool denied"),
                ],
            }
        ],
    )


@pytest.fixture
def kimi_k3_reasoning_checkpoint() -> dict[str, object]:
    return _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "Visible final answer",
                "reasoning_blocks": ["preserved K3 reasoning"],
                "calls": [],
            }
        ],
    )


def test_schema_accepts_canonical_v1_fixtures_and_dumps_deterministically(
    active_tool_checkpoint: dict[str, object],
    result_checkpoint: dict[str, object],
    kimi_k3_reasoning_checkpoint: dict[str, object],
) -> None:
    for value in (
        active_tool_checkpoint,
        result_checkpoint,
        kimi_k3_reasoning_checkpoint,
    ):
        parsed = parse_provider_continuation_json(value)
        dumped = dump_provider_continuation_json(parsed)

        assert dumped is not None
        assert json.loads(dumped) == value
        assert dump_provider_continuation_json(parsed) == dumped
        assert parse_provider_continuation_json(dumped) == parsed

    assert dump_provider_continuation_json(None) is None


def test_schema_builds_frozen_immutable_canonical_values(
    result_checkpoint: dict[str, object],
) -> None:
    parsed = parse_provider_continuation_json(result_checkpoint)

    assert isinstance(parsed, ProviderContinuationCheckpoint)
    assert isinstance(parsed.rounds, tuple)
    assert isinstance(parsed.rounds[0], ContinuationRound)
    assert isinstance(parsed.rounds[0].reasoning_blocks, tuple)
    assert isinstance(parsed.rounds[0].calls, tuple)
    assert isinstance(parsed.rounds[0].calls[0], ContinuationCall)
    assert parsed.rounds[0].calls[0].result == ContinuationResult("4")
    with pytest.raises(FrozenInstanceError):
        parsed.checkpoint_revision = 2  # type: ignore[misc]


def test_schema_preserves_call_order_and_allows_repeated_function_names() -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call("a"), _call("b")],
            }
        ]
    )

    parsed = parse_provider_continuation_json(value)

    assert [call.call_id for call in parsed.rounds[0].calls] == ["a", "b"]
    assert [call.name for call in parsed.rounds[0].calls] == [
        "calculator",
        "calculator",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "ab",
        "bad name",
        "bad/name",
        "café",
        "1abc",
        "-abc",
        "bad.name",
        "$abc",
    ],
)
def test_invalid_function_name_grammar_is_rejected(name: str) -> None:
    value = _checkpoint()
    value["rounds"][0]["calls"][0]["name"] = name

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "name",
    [
        "_ab",
        "a-b",
        "A_0",
        "a" + "b" * 63,
    ],
)
def test_schema_accepts_provider_safe_function_name_boundaries(name: str) -> None:
    value = _checkpoint()
    value["rounds"][0]["calls"][0]["name"] = name

    assert parse_provider_continuation_json(value).rounds[0].calls[0].name == name


@pytest.mark.parametrize("prior_state", ["pending", "executing"])
def test_invalid_nonfinal_round_cannot_retain_nonterminal_calls(
    prior_state: str,
) -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["prior"],
                "calls": [_call("prior", state=prior_state)],
            },
            {
                "assistant_content": "",
                "reasoning_blocks": ["current"],
                "calls": [_call("current")],
            },
        ]
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


def test_schema_allows_later_round_after_prior_calls_are_terminal() -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["prior"],
                "calls": [
                    _call("completed", state="completed", result="ok"),
                    _call("failed", state="failed", result="denied"),
                ],
            },
            {
                "assistant_content": "",
                "reasoning_blocks": ["current"],
                "calls": [_call("current")],
            },
        ]
    )

    parsed = parse_provider_continuation_json(value)

    assert [call.state for call in parsed.rounds[0].calls] == ["completed", "failed"]
    assert parsed.rounds[1].calls[0].state == "pending"


@pytest.mark.parametrize(
    ("mutation", "case"),
    [
        (lambda value: value.update(extra="unknown"), "top-level extra"),
        (lambda value: value.pop("state"), "top-level missing"),
        (
            lambda value: value["rounds"][0].update(vendor_id="opaque"),
            "round extra",
        ),
        (
            lambda value: value["rounds"][0]["calls"][0].update(timestamp=1),
            "call extra",
        ),
        (lambda value: value.update(schema_version=2), "unknown version"),
        (lambda value: value.update(checkpoint_revision=True), "bool revision"),
        (lambda value: value.update(checkpoint_revision=0), "zero revision"),
        (lambda value: value.update(checkpoint_revision=1.0), "float revision"),
        (lambda value: value.update(provider=True), "provider bool"),
        (lambda value: value.update(state="discarded"), "unknown state"),
        (lambda value: value.update(rounds="not-a-list"), "rounds scalar"),
        (
            lambda value: value["rounds"][0].update(reasoning_blocks=[1]),
            "reasoning scalar type",
        ),
        (
            lambda value: value["rounds"][0]["calls"][0].update(call_id=""),
            "blank call id",
        ),
        (
            lambda value: value["rounds"][0]["calls"][0].update(name=" "),
            "blank function name",
        ),
        (
            lambda value: value["rounds"][0]["calls"][0].update(state="done"),
            "unknown call state",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_invalid_schema_and_scalar_types_are_rejected(
    active_tool_checkpoint: dict[str, object],
    mutation: Any,
    case: str,
) -> None:
    del case
    mutation(active_tool_checkpoint)

    with pytest.raises(ContinuationValidationError, match="(?i)invalid continuation"):
        parse_provider_continuation_json(active_tool_checkpoint)


@pytest.mark.parametrize(
    ("provider", "protocol", "valid"),
    [
        ("moonshot", "chat_completions", True),
        ("zai", "chat_completions", True),
        ("deepseek", "chat_completions", True),
        ("deepseek", "responses", True),
        ("moonshot", "responses", False),
        ("zai", "responses", False),
        ("openai", "responses", False),
        ("deepseek", "messages", False),
    ],
)
def test_schema_enforces_closed_provider_protocol_pairings(
    provider: str, protocol: str, valid: bool
) -> None:
    value = _checkpoint(provider=provider, protocol=protocol)

    if valid:
        assert parse_provider_continuation_json(value).provider == provider
    else:
        with pytest.raises(ContinuationValidationError):
            parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "ftp://api.example.test",
        "https://user@api.example.test",
        "https://user:password@api.example.test",
        "https://api.example.test?key=secret",
        "https://api.example.test/#private",
    ],
)
def test_invalid_or_credential_bearing_base_urls_are_rejected(url: str) -> None:
    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(_checkpoint(api_base_url=url))


def test_invalid_duplicate_call_ids_across_rounds_are_rejected() -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call("duplicate")],
            },
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call("duplicate")],
            },
        ]
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize("state", ["pending", "executing"])
def test_invalid_nonterminal_calls_cannot_have_results(state: str) -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call(state=state, result="unexpected")],
            }
        ]
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize("state", ["completed", "failed"])
def test_invalid_terminal_calls_require_results(state: str) -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call(state=state)],
            }
        ]
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "arguments",
    [
        "not json",
        "[]",
        '"scalar"',
        "null",
        '{"value":NaN}',
        '{"value":Infinity}',
    ],
)
def test_invalid_arguments_must_be_exact_finite_json_objects(arguments: str) -> None:
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call(arguments=arguments)],
            }
        ]
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    ("provider", "model", "state", "round_index", "reasoning"),
    [
        ("deepseek", "deepseek-v4", "complete", 0, ["reasoning"]),
        ("zai", "glm-5", "complete", 0, ["reasoning"]),
        # kimi-latest returned no reasoning_content on the wire
        # (TASK-19170, chatcmpl-6a8768a616ceb0c0ae780f2c): outside the
        # reasoning family, no-call rounds stay invalid.
        ("moonshot", "kimi-latest", "complete", 0, ["reasoning"]),
        ("moonshot", "moonshot-v1-8k", "complete", 0, ["reasoning"]),
        ("moonshot", "kimi-k3", "active", 0, ["reasoning"]),
        ("moonshot", "kimi-k3", "complete", 0, []),
        ("moonshot", "kimi-k3", "complete", 0, ["   "]),
        ("moonshot", "kimi-k2.6", "complete", 0, []),
        ("moonshot", "kimi-k2.6", "complete", 0, ["   "]),
        ("moonshot", "kimi-k3", "complete", 0, ["reasoning"]),
    ],
)
def test_invalid_empty_call_rounds_are_rejected_outside_final_k3_exception(
    provider: str,
    model: str,
    state: str,
    round_index: int,
    reasoning: list[str],
) -> None:
    rounds: list[dict[str, object]] = [
        {
            "assistant_content": "",
            "reasoning_blocks": reasoning,
            "calls": [],
        }
    ]
    if (
        provider == "moonshot"
        and model == "kimi-k3"
        and state == "complete"
        and reasoning == ["reasoning"]
    ):
        rounds.append(
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call("later")],
            }
        )
    value = _checkpoint(
        provider=provider,
        protocol="chat_completions",
        model=model,
        state=state,
        rounds=rounds,
    )

    assert round_index == 0
    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


def test_schema_accepts_complete_k3_tool_round_followed_by_final_reasoning_round() -> (
    None
):
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["tool reasoning"],
                "calls": [_call("tool", state="completed", result="ok")],
            },
            {
                "assistant_content": "Visible final answer",
                "reasoning_blocks": ["final reasoning"],
                "calls": [],
            },
        ],
    )

    parsed = parse_provider_continuation_json(value)

    assert parsed.rounds[0].calls[0].state == "completed"
    assert parsed.rounds[-1].calls == ()
    assert parsed.rounds[-1].reasoning_blocks == ("final reasoning",)


def test_schema_accepts_active_k3_tool_round_without_final_reasoning_round() -> None:
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["tool reasoning"],
                "calls": [_call("tool")],
            }
        ],
    )

    assert parse_provider_continuation_json(value).rounds[0].calls[0].state == "pending"


def test_invalid_complete_k3_tool_only_checkpoint_requires_final_reasoning_round() -> (
    None
):
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["tool reasoning"],
                "calls": [_call("tool", state="completed", result="ok")],
            }
        ],
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "model",
    [
        "not-kimi-k3-fake",
        "not-kimi-k3",
        "kimi-latest",  # no reasoning_content on the wire (TASK-19170 probe A)
        "kimi",
        "moonshot-v1-8k",
        "kimik3",
    ],
)
def test_invalid_reasoning_only_exception_requires_versioned_kimi_family(
    model: str,
) -> None:
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model=model,
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "Visible final answer",
                "reasoning_blocks": ["final reasoning"],
                "calls": [],
            }
        ],
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "model",
    [
        "kimi-k3",
        # TASK-19170 probes: every versioned kimi id returns reasoning_content
        # (kimi-k2.5 chatcmpl-6a8768d3666d8454604d8b5f, kimi-k2.6
        # chatcmpl-6a8768a3b5c429b466fbc42d, kimi-k2.7-code
        # chatcmpl-6a8768d705f910ba798aeca0), so the reasoning-only final
        # round is a family shape, not a kimi-k3 exception.
        "kimi-k2",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2.7-code",
        "kimi-k3-turbo",
        "kimi-k4",
        "moonshot/kimi-k3",
        "KIMI-K3",
        "kimi_k3",
        "kimi-k30",
    ],
)
def test_schema_reasoning_only_exception_accepts_versioned_kimi_family(
    model: str,
) -> None:
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model=model,
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "Visible final answer",
                "reasoning_blocks": ["final reasoning"],
                "calls": [],
            }
        ],
    )

    assert parse_provider_continuation_json(value).model == model


def test_schema_family_complete_tool_only_checkpoint_stays_valid_off_k3() -> None:
    """Durable-data pin (TASK-19170): pre-19170 versioned-kimi (non-k3)
    tool-loop checkpoints were persisted complete WITHOUT a final reasoning
    round. That stored shape must keep parsing forever; only the exact
    kimi-k3 id carries the must-end-with-reasoning-round invariant."""
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2.6",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["tool reasoning"],
                "calls": [_call("tool", state="completed", result="ok")],
            }
        ],
    )

    parsed = parse_provider_continuation_json(value)

    assert parsed.model == "kimi-k2.6"
    assert parsed.rounds[-1].calls[0].state == "completed"


def test_schema_family_complete_tool_round_then_final_reasoning_round_off_k3() -> None:
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2.6",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": ["tool reasoning"],
                "calls": [_call("tool", state="completed", result="ok")],
            },
            {
                "assistant_content": "Visible final answer",
                "reasoning_blocks": ["final reasoning"],
                "calls": [],
            },
        ],
    )

    parsed = parse_provider_continuation_json(value)

    assert parsed.rounds[-1].calls == ()
    assert parsed.rounds[-1].reasoning_blocks == ("final reasoning",)


def test_invalid_family_reasoning_only_round_still_must_be_final() -> None:
    value = _checkpoint(
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2.6",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=[
            {
                "assistant_content": "Visible final answer",
                "reasoning_blocks": ["final reasoning"],
                "calls": [],
            },
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [_call("later", state="completed", result="ok")],
            },
        ],
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(rounds=[]),
        lambda value: value.update(
            rounds=[
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
                    "calls": [_call(str(index))],
                }
                for index in range(129)
            ]
        ),
        lambda value: value.update(model="m" * 4097),
        lambda value: value.update(api_base_url="https://" + "a" * 4097 + ".test"),
        lambda value: value["rounds"][0]["calls"][0].update(call_id="c" * 4097),
        lambda value: value["rounds"][0]["calls"][0].update(name="n" * 65),
        lambda value: value["rounds"][0]["calls"][0].update(
            arguments='{"value":"' + "a" * 1_048_576 + '"}'
        ),
        lambda value: value["rounds"][0].update(
            reasoning_blocks=["r" * (4 * 1_048_576 + 1)]
        ),
        lambda value: value["rounds"][0]["calls"][0].update(
            state="completed", result="r" * 16_001
        ),
    ],
)
def test_bounds_are_rejected(mutate: Any) -> None:
    value = _checkpoint()
    mutate(value)

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


def test_bounds_count_identity_limits_as_utf8_bytes() -> None:
    value = _checkpoint(model="é" * 2049)

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


def test_bounds_reject_more_than_128_total_calls() -> None:
    calls = [_call(str(index)) for index in range(129)]
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": calls,
            }
        ]
    )

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


def test_bounds_reject_deep_or_excessive_argument_json() -> None:
    nested: object = 0
    for _ in range(33):
        nested = {"child": nested}
    deep = json.dumps(nested)
    too_many_nodes = json.dumps({"values": [None] * 100_000})

    for arguments in (deep, too_many_nodes):
        value = _checkpoint(
            rounds=[
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
                    "calls": [_call(arguments=arguments)],
                }
            ]
        )
        with pytest.raises(ContinuationValidationError):
            parse_provider_continuation_json(value)


def test_bounds_reject_excessive_nodes_across_the_whole_payload() -> None:
    value = _checkpoint()
    value["rounds"][0]["reasoning_blocks"] = [""] * 99_990

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


@pytest.mark.parametrize(
    "decoded",
    [
        [None] * 100_001,
        {str(index): None for index in range(50_001)},
    ],
    ids=["list", "dict"],
)
def test_bounds_reject_normal_decoded_immediate_children(decoded: object) -> None:
    tracemalloc.start()
    try:
        _, before_peak = tracemalloc.get_traced_memory()
        with pytest.raises(continuation_module._InvalidContinuation):
            continuation_module._json_shape(decoded)
        _, after_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert after_peak - before_peak < 8_192


@pytest.mark.parametrize("field", ["assistant_content", "reasoning_blocks"])
def test_bounds_reject_oversized_decoded_strings_without_copying(
    field: str,
) -> None:
    value = _checkpoint()
    oversized = "x" * (8 * 1_048_576 + 1)
    value["rounds"][0][field] = (
        [oversized] if field == "reasoning_blocks" else oversized
    )

    tracemalloc.start()
    try:
        _, before_peak = tracemalloc.get_traced_memory()
        with pytest.raises(ContinuationValidationError):
            parse_provider_continuation_json(value)
        _, after_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert after_peak - before_peak < 65_536


def test_bounds_reject_aggregate_decoded_payload_without_materializing_dump() -> None:
    arguments = '{"value":"' + "a" * 920_000 + '"}'
    value = _checkpoint(
        rounds=[
            {
                "assistant_content": "",
                "reasoning_blocks": [],
                "calls": [
                    _call(str(index), arguments=arguments) for index in range(10)
                ],
            }
        ]
    )

    tracemalloc.start()
    try:
        _, before_peak = tracemalloc.get_traced_memory()
        with pytest.raises(ContinuationValidationError):
            parse_provider_continuation_json(value)
        _, after_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert after_peak - before_peak < 3 * 1_048_576


def test_bounds_accept_exact_canonical_payload_limit_and_reject_one_more() -> None:
    value = _checkpoint()
    empty_dump = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    content_length = 8 * 1_048_576 - len(empty_dump.encode("utf-8"))
    value["rounds"][0]["assistant_content"] = "x" * content_length

    checkpoint = parse_provider_continuation_json(value)
    dumped = dump_provider_continuation_json(checkpoint)

    assert dumped is not None
    assert len(dumped.encode("utf-8")) == 8 * 1_048_576

    value["rounds"][0]["assistant_content"] += "x"
    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(value)


def test_bounds_reject_payloads_over_eight_mib_before_json_decode() -> None:
    raw = "{" + "x" * (8 * 1_048_576) + "}"

    with pytest.raises(ContinuationValidationError):
        parse_provider_continuation_json(raw)


def test_invalid_tolerant_read_discards_private_data_with_one_safe_warning() -> None:
    invalid = _checkpoint()
    invalid["credential"] = "TOP-SECRET-CREDENTIAL"

    result = read_provider_continuation_json(invalid)

    assert result.checkpoint is None
    assert result.warning == "Exact tool continuation was discarded."
    assert "TOP-SECRET-CREDENTIAL" not in repr(result)


def test_schema_parse_does_not_mutate_input(
    result_checkpoint: dict[str, object],
) -> None:
    before = copy.deepcopy(result_checkpoint)

    parse_provider_continuation_json(result_checkpoint)

    assert result_checkpoint == before


def test_call_transition_state_machine_is_revision_checked(
    active_tool_checkpoint: dict[str, object],
) -> None:
    pending = parse_provider_continuation_json(active_tool_checkpoint)

    executing = transition_provider_call(
        pending,
        call_id="call_1",
        expected_revision=1,
        target="executing",
    )
    assert executing.checkpoint_revision == 2
    assert executing.rounds[0].calls[0].state == "executing"
    assert pending.rounds[0].calls[0].state == "pending"

    completed = transition_provider_call(
        executing,
        call_id="call_1",
        expected_revision=2,
        target="completed",
        result=ContinuationResult("4"),
    )
    assert completed.checkpoint_revision == 3
    assert completed.rounds[0].calls[0].state == "completed"
    assert completed.rounds[0].calls[0].result == ContinuationResult("4")
    assert (
        transition_provider_call(
            completed,
            call_id="call_1",
            expected_revision=3,
            target="completed",
            result=ContinuationResult("4"),
        )
        is completed
    )

    failed = transition_provider_call(
        executing,
        call_id="call_1",
        expected_revision=2,
        target="failed",
        result=ContinuationResult("denied"),
    )
    assert failed.checkpoint_revision == 3
    assert failed.rounds[0].calls[0].state == "failed"

    refused = transition_provider_call(
        pending,
        call_id="call_1",
        expected_revision=1,
        target="failed",
        result=ContinuationResult("review refused"),
    )
    assert refused.checkpoint_revision == 2
    assert refused.rounds[0].calls[0].state == "failed"
    assert refused.rounds[0].calls[0].result == ContinuationResult("review refused")

    with pytest.raises(ContinuationConflictError, match="revision conflict"):
        transition_provider_call(
            executing,
            call_id="call_1",
            expected_revision=1,
            target="completed",
            result=ContinuationResult("4"),
        )

    illegal = [
        (pending, "pending", None),
        (pending, "completed", ContinuationResult("4")),
        (pending, "failed", None),
        (executing, "pending", None),
        (executing, "executing", None),
        (executing, "completed", None),
        (completed, "executing", None),
        (completed, "completed", ContinuationResult("different")),
        (completed, "failed", ContinuationResult("failed")),
    ]
    for checkpoint, target, result in illegal:
        with pytest.raises(ContinuationValidationError):
            transition_provider_call(
                checkpoint,
                call_id="call_1",
                expected_revision=checkpoint.checkpoint_revision,
                target=target,  # type: ignore[arg-type]
                result=result,
            )

    with pytest.raises(ContinuationValidationError):
        transition_provider_call(
            pending,
            call_id="missing",
            expected_revision=1,
            target="executing",
        )


def test_restore_target_requires_all_four_exact_frozen_fields(
    active_tool_checkpoint: dict[str, object],
) -> None:
    checkpoint = parse_provider_continuation_json(active_tool_checkpoint)
    target = ContinuationRestoreTarget(
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
    )

    validate_continuation_restore(checkpoint, target)
    validate_continuation_restore(
        checkpoint,
        replace(target, api_base_url="https://api.deepseek.com/v1/chat/completions"),
    )
    with pytest.raises(FrozenInstanceError):
        target.model = "changed"  # type: ignore[misc]

    mismatches = [
        ContinuationRestoreTarget(
            provider="zai",
            protocol=target.protocol,
            model=target.model,
            api_base_url=target.api_base_url,
        ),
        ContinuationRestoreTarget(
            provider=target.provider,
            protocol="chat_completions",
            model=target.model,
            api_base_url=target.api_base_url,
        ),
        ContinuationRestoreTarget(
            provider=target.provider,
            protocol=target.protocol,
            model="deepseek-v4-flash-next",
            api_base_url=target.api_base_url,
        ),
        ContinuationRestoreTarget(
            provider=target.provider,
            protocol=target.protocol,
            model=target.model,
            api_base_url="https://api.deepseek.com/v1/",
        ),
    ]
    for mismatch in mismatches:
        with pytest.raises(ContinuationConflictError, match="restore target mismatch"):
            validate_continuation_restore(checkpoint, mismatch)


def test_owner_group_binds_one_assistant_id_to_only_canonical_immutable_rounds(
    active_tool_checkpoint: dict[str, object],
) -> None:
    checkpoint = parse_provider_continuation_json(active_tool_checkpoint)
    visible = {
        "id": "assistant-1",
        "role": "assistant",
        "content": "",
        "public_metadata": {"variant": 2},
    }
    before = copy.deepcopy(visible)

    group = continuation_owner_group(visible, checkpoint)

    assert isinstance(group, ContinuationOwnerGroup)
    assert group.owner_message_id == "assistant-1"
    assert group.checkpoint == checkpoint
    assert group.checkpoint is not checkpoint
    assert group.rounds == checkpoint.rounds
    assert group.rounds is not checkpoint.rounds
    assert isinstance(group.rounds, tuple)
    assert "tool_calls" not in vars(group.rounds[0])
    assert "response_id" not in vars(group.rounds[0])
    assert visible == before
    with pytest.raises(FrozenInstanceError):
        group.owner_message_id = "changed"  # type: ignore[misc]

    invalid_owners = [
        {"id": "", "role": "assistant"},
        {"id": "assistant-1", "role": "user"},
        {"role": "assistant"},
    ]
    for invalid_owner in invalid_owners:
        with pytest.raises(ContinuationValidationError):
            continuation_owner_group(invalid_owner, checkpoint)
    with pytest.raises(ContinuationValidationError):
        continuation_owner_group(visible, None)


def test_invalid_private_data_is_context_free_and_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canaries = (
        "CONTINUATION-CREDENTIAL-CANARY",
        "CONTINUATION-RAW-BODY-CANARY",
    )
    value = _checkpoint()
    value["credential"] = canaries[0]
    value["raw_provider_body"] = canaries[1]

    caplog.set_level(logging.DEBUG)
    try:
        parse_provider_continuation_json(value)
    except ContinuationValidationError as exc:
        rendered = "\n".join(
            (
                str(exc),
                repr(exc),
                "".join(traceback.format_exception(exc)),
                repr(read_provider_continuation_json(value)),
                caplog.text,
            )
        )
        assert exc.__cause__ is None
        assert exc.__context__ is None
    else:  # pragma: no cover - makes a surprising acceptance explicit
        pytest.fail("credential-bearing private continuation was accepted")

    for canary in canaries:
        assert canary not in rendered


def test_repr_redacts_every_canonical_private_string(
    result_checkpoint: dict[str, object],
) -> None:
    call = result_checkpoint["rounds"][0]["calls"][0]
    call["arguments"] = '{"secret":"ARGUMENT-CANARY"}'
    call["result"] = "RESULT-CANARY"
    result_checkpoint["rounds"][0]["reasoning_blocks"] = ["REASONING-CANARY"]
    result_checkpoint["rounds"][0]["assistant_content"] = "CONTENT-CANARY"
    checkpoint = parse_provider_continuation_json(result_checkpoint)
    target = ContinuationRestoreTarget(
        provider=checkpoint.provider,
        protocol=checkpoint.protocol,
        model="MODEL-CANARY",
        api_base_url="https://BASE-CANARY.example.test",
    )
    group = continuation_owner_group(
        {"id": "OWNER-CANARY", "role": "assistant"}, checkpoint
    )

    rendered = "\n".join(
        repr(value)
        for value in (
            checkpoint,
            checkpoint.rounds[0],
            checkpoint.rounds[0].calls[0],
            checkpoint.rounds[0].calls[0].result,
            target,
            group,
        )
    )

    for canary in (
        "ARGUMENT-CANARY",
        "RESULT-CANARY",
        "REASONING-CANARY",
        "CONTENT-CANARY",
        "MODEL-CANARY",
        "BASE-CANARY",
    ):
        assert canary not in rendered
    assert "OWNER-CANARY" in rendered


def test_input_aliases_cannot_mutate_canonical_checkpoint(
    active_tool_checkpoint: dict[str, object],
) -> None:
    checkpoint = parse_provider_continuation_json(active_tool_checkpoint)

    active_tool_checkpoint["model"] = "mutated"
    active_tool_checkpoint["rounds"][0]["reasoning_blocks"].append("mutated")
    active_tool_checkpoint["rounds"][0]["calls"][0]["arguments"] = "{}"

    assert checkpoint.model == "deepseek-v4-flash"
    assert checkpoint.rounds[0].reasoning_blocks == ("private reasoning",)
    assert checkpoint.rounds[0].calls[0].arguments == '{"expression":"2+2"}'


def test_dump_revalidates_forged_dataclass_instances_without_leaking() -> None:
    forged = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="MODEL-SECRET-CANARY",
        api_base_url="https://user:password@api.example.test",
        state="active",
        rounds=(),
    )

    with pytest.raises(ContinuationValidationError) as caught:
        dump_provider_continuation_json(forged)

    assert "SECRET-CANARY" not in str(caught.value)
    assert "password" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_owner_validation_does_not_chain_forged_private_checkpoint_data() -> None:
    forged = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="FORGED-CONTEXT-CANARY",
        api_base_url="https://api.example.test",
        state="active",
        rounds=(),
    )

    with pytest.raises(ContinuationValidationError) as caught:
        continuation_owner_group({"id": "assistant-1", "role": "assistant"}, forged)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "FORGED-CONTEXT-CANARY" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_private_mutation_revision_rejects_bool_as_invalid_not_conflict(
    active_tool_checkpoint: dict[str, object],
) -> None:
    checkpoint = parse_provider_continuation_json(active_tool_checkpoint)

    with pytest.raises(ContinuationValidationError):
        transition_provider_call(
            checkpoint,
            call_id="call_1",
            expected_revision=True,
            target="executing",
        )
