"""Canonical local replay policy, wire accounting, and causal provenance."""

from pathlib import Path

import pytest

from tldw_chatbook.Chat import local_reasoning as replay
from tldw_chatbook.Chat.console_prepared_request import (
    THINKING_OWNER_KEY,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_thinking_history import (
    ProviderThinkingSidecar,
    ThinkingHistorySerializationError,
    ThinkingReplayTarget,
    resolve_thinking_history,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    DerivedTraceProvenance,
    SavedRevisionTraceProvenance,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "reasoning_templates"


def sidecar(
    owner, text, source_format="reasoning_content", provider="llama_cpp", model="alias"
):
    return ProviderThinkingSidecar(
        owner,
        ThinkingEnvelope(
            (
                DisplayableThinkingBlock(
                    block_id=owner,
                    round_ordinal=0,
                    provider=provider,
                    model=model,
                    protocol="chat_completions",
                    source_format=source_format,
                    status="complete",
                    text=text,
                ),
            )
        ),
    )


def prepare(mode="current", preference="auto", family="", rows=None, sidecars=None):
    policy = replay.ReasoningReplayPolicy(mode, "test", family)
    target = ThinkingReplayTarget(
        "llama_cpp",
        "alias",
        "chat_completions",
        "displayable",
        1,
        reasoning_replay=policy,
    )
    resolved = resolve_thinking_history(
        target=target,
        policy=preference,
        sidecars=tuple(
            sidecars
            or (sidecar("old", "OLD_THOUGHT"), sidecar("active", "ACTIVE_THOUGHT"))
        ),
    )
    rows = rows or [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", THINKING_OWNER_KEY: "old"},
        {"role": "user", "content": "current question"},
        {
            "role": "assistant",
            "content": "",
            THINKING_OWNER_KEY: "active",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "found", "tool_call_id": "a"},
        {
            "role": "user",
            "content": "keep going",
            replay.EXCHANGE_CONTINUATION_KEY: True,
        },
    ]
    eligible = {group.owner_message_id for group in resolved.groups}
    rows = [
        {
            key: value
            for key, value in row.items()
            if key != THINKING_OWNER_KEY or value in eligible
        }
        for row in rows
    ]
    revisions = tuple(SavedRevisionTraceProvenance(new_opaque_id()) for _ in rows)
    semantic = build_console_request(
        rows,
        thinking_groups=resolved.groups,
        thinking_policy=resolved.saved_policy,
        effective_thinking_policy=resolved.effective_policy,
        message_provenance=revisions,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=FrozenTracePolicy(new_opaque_id(), "v1", False, None),
    )
    counted = []

    def counter(messages, model):
        counted.append(messages)
        return len(str(messages))

    result = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="alias",
        provider="llama_cpp",
        capacity=resolve_request_capacity(context_window_tokens=None),
        reasoning_replay=policy,
        count_fn=counter,
    )
    assert counted[-1] == [thaw_json(row) for row in result.messages]
    return result, semantic, revisions


def _render_reviewed_template(filename, messages):
    """Render the exact captured server template with its native input shape."""
    import copy
    import json

    from jinja2 import Environment

    def fail(message):
        raise AssertionError(message)

    wire = copy.deepcopy(messages)
    for row in wire:
        for call in row.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(
                call["function"]["arguments"]
            )
    return (
        Environment()
        .from_string((FIXTURES / f"{filename}.jinja").read_text())
        .render(
            messages=wire,
            tools=[],
            bos_token="<bos>",
            add_generation_prompt=True,
            enable_thinking=True,
            raise_exception=fail,
        )
    )


@pytest.mark.parametrize(
    "mode,preference,owners",
    [
        ("current", "auto", ["active"]),
        ("all", "auto", ["old", "active"]),
        ("off", "auto", []),
        ("all", "exclude", []),
        ("off", "include", ["old", "active"]),
    ],
)
def test_policy_filters_whole_groups_with_matching_provenance(mode, preference, owners):
    result, semantic, _ = prepare(mode, preference)
    assert [group.owner_message_id for group in result.thinking_groups] == owners
    assert len(result.provenance.thinking) == len(owners)
    assert result.semantic.active_request[0]["content"] == "current question"
    for row in result.messages:
        assert replay.EXCHANGE_CONTINUATION_KEY not in row
    for row in result.messages:
        if row["role"] == "assistant":
            owner = "old" if row["content"] == "old answer" else "active"
            assert row.get("reasoning_content") == (
                f"{owner.upper()}_THOUGHT" if owner in owners else None
            )
    assert semantic.compactable[0].messages[1]["content"] == "old answer"


def test_structured_fields_rewrite_trace_without_rewriting_answer():
    result, _, revisions = prepare("all")
    assert result.messages[1]["content"] == "old answer"
    descriptor = result.provenance.messages[1]
    assert isinstance(descriptor, DerivedTraceProvenance)
    assert descriptor.inputs[0] == revisions[1]
    assert descriptor.inputs[1] == result.provenance.thinking[0]


@pytest.mark.parametrize(
    "provider,source",
    [
        ("llama_cpp", "reasoning_content"),
        ("local_vllm", "reasoning_content"),
        ("ollama", "reasoning"),
    ],
)
def test_structured_source_round_trips_only_with_eligible_local_family(
    provider, source
):
    target = ThinkingReplayTarget(
        provider, "alias", "chat_completions", "displayable", 1
    )
    resolved = resolve_thinking_history(
        target=target,
        policy="include",
        sidecars=(sidecar("a", "thought", source, provider),),
    )
    assert len(resolved.groups) == 1
    incompatible = resolve_thinking_history(
        target=target,
        policy="include",
        sidecars=(sidecar("a", "thought", source, "openai"),),
    )
    assert incompatible.groups == ()


def test_saved_tool_trace_does_not_become_final_answer_reasoning():
    result, _, _ = prepare(
        "all",
        sidecars=[
            sidecar("old", "TOOL_PRIVATE", "reasoning_content:tool_call"),
            sidecar("active", "FINAL_PRIVATE"),
        ],
    )
    assert "TOOL_PRIVATE" not in str(result.messages)
    assert result.messages[3]["reasoning_content"] == "FINAL_PRIVATE"


def test_include_keeps_strict_serialization_even_when_device_setting_off():
    target = ThinkingReplayTarget(
        "llama_cpp",
        "alias",
        "chat_completions",
        "displayable",
        1,
        reasoning_replay=replay.ReasoningReplayPolicy("off", "test"),
    )
    with pytest.raises(ThinkingHistorySerializationError):
        resolve_thinking_history(
            target=target,
            policy="include",
            sidecars=(sidecar("a", "unsafe", "unknown"),),
        )


def test_gemma_control_merge_preserves_both_causes_and_tool_overlay():
    result, semantic, revisions = prepare(family="Gemma 4")
    assert len(result.messages) == 5
    assert result.messages[-1]["role"] == "tool"
    assert result.messages[-1]["content"] == "found\n\n[Runtime guidance]\nkeep going"
    assert result.provenance.messages[-1].inputs == revisions[-2:]
    assert result.provenance.tool_loop == (3, 4)
    assert len(semantic.active_request) == 4


def test_qwen_runtime_guidance_has_exact_wrapper_and_rewrite_provenance():
    result, _, revisions = prepare(family="Qwen3.6")
    assert (
        result.messages[-1]["content"]
        == "<tool_response>\nkeep going\n</tool_response>"
    )
    assert result.provenance.messages[-1].inputs == (revisions[-1],)


@pytest.mark.parametrize(
    "filename,mode",
    [
        ("gemma4", "current"),
        ("qwen35", "current"),
        ("qwen36", "current"),
        ("qwen38", "all"),
    ],
)
def test_reviewed_templates_resolve_without_alias_guessing(filename, mode):
    assert (
        replay.resolve_reasoning_policy(
            "auto", template=(FIXTURES / f"{filename}.jinja").read_text()
        ).mode
        == mode
    )


def test_unknown_template_uses_server_default_and_scoped_keys_strip_credentials():
    assert (
        replay.resolve_reasoning_policy("auto", template="unreviewed").mode
        == "server_default"
    )
    assert replay.reasoning_override_key(
        "local_llamacpp", "http://u:p@HOST:8/v1/chat/completions", "x"
    ) == replay.reasoning_override_key("llama_cpp", "http://host:8", "x")


def test_budget_counter_includes_structured_reasoning(monkeypatch):
    from tldw_chatbook.Chat import console_history_budget as budget

    received = []

    def count(rows, model):
        received.extend(rows)
        return sum(len(row["content"]) for row in rows)

    monkeypatch.setattr(budget, "count_tokens_messages", count)
    assert budget.count_console_messages_tokens(
        [{"role": "assistant", "content": "answer", "reasoning_content": "thought"}],
        "alias",
    ) == len("answer thought")
    assert received[0]["content"] == "answer thought"


def test_runtime_guidance_transform_refuses_thinking_or_unrelated_second_input():
    from tldw_chatbook.Chat.console_trace_provenance import (
        ProviderArtifactTraceProvenance,
        TraceProvenanceAlignmentError,
        TraceProvenanceSource,
        TraceTransformKind,
    )

    policy = FrozenTracePolicy(new_opaque_id(), "v1", False, None)
    result = ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
    thinking = ProviderArtifactTraceProvenance(TraceProvenanceSource.THINKING, policy)
    with pytest.raises(TraceProvenanceAlignmentError):
        DerivedTraceProvenance(TraceTransformKind.RUNTIME_GUIDANCE, (result, thinking))
    with pytest.raises(TraceProvenanceAlignmentError):
        DerivedTraceProvenance(TraceTransformKind.MESSAGE_REWRITE, (result, result))


@pytest.mark.parametrize(
    "filename,family",
    [
        ("gemma4", "Gemma 4"),
        ("qwen35", "Qwen3.5"),
        ("qwen36", "Qwen3.6"),
        ("qwen38", "Qwen3.8"),
    ],
)
def test_reviewed_render_retains_active_tool_thought_across_runtime_guidance(
    filename, family
):
    result, _, _ = prepare(family=family)
    wire = [thaw_json(row) for row in result.messages]
    rendered = _render_reviewed_template(filename, wire)
    assert "ACTIVE_THOUGHT" in rendered
    assert "OLD_THOUGHT" not in rendered
    assert "keep going" in rendered


def test_gemma_all_sends_prior_final_but_template_keeps_only_tool_thinking():
    """All is request-side; Gemma still omits reasoning from prior final answers."""
    result, _, _ = prepare(mode="all", family="Gemma 4")
    wire = [thaw_json(row) for row in result.messages]

    assert wire[1]["reasoning_content"] == "OLD_THOUGHT"
    assert wire[3]["reasoning_content"] == "ACTIVE_THOUGHT"

    rendered = _render_reviewed_template("gemma4", wire)
    assert "OLD_THOUGHT" not in rendered
    assert "ACTIVE_THOUGHT" in rendered


def test_gemma_fenced_tool_result_starts_new_user_turn_and_omits_active_thinking():
    """Gemma retains active tool thinking only with the native tool protocol."""
    wire = [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": "old answer",
            "reasoning_content": "OLD_THOUGHT",
        },
        {"role": "user", "content": "current question"},
        {
            "role": "assistant",
            "content": '```tool\n{"name":"lookup","arguments":{}}\n```',
            "reasoning_content": "ACTIVE_THOUGHT",
        },
        {
            "role": "user",
            "content": "Tool result: found\n\n[Runtime guidance]\nkeep going",
        },
    ]

    assert wire[3]["reasoning_content"] == "ACTIVE_THOUGHT"
    rendered = _render_reviewed_template("gemma4", wire)
    assert "```tool" in rendered
    assert "Tool result: found" in rendered
    assert "ACTIVE_THOUGHT" not in rendered


def test_ollama_and_openai_shape_local_fields_are_not_cross_translated():
    target = ThinkingReplayTarget(
        "ollama", "alias", "chat_completions", "displayable", 1
    )
    assert (
        resolve_thinking_history(
            target=target, policy="auto", sidecars=(sidecar("a", "private"),)
        ).groups
        == ()
    )


def test_incompatible_source_encodings_in_one_owner_are_skipped_or_strict():
    envelope = ThinkingEnvelope(
        (
            sidecar("a", "first").envelope.blocks[0],
            DisplayableThinkingBlock(
                block_id="b",
                round_ordinal=1,
                provider="llama_cpp",
                model="alias",
                protocol="chat_completions",
                source_format="start_anchored_think",
                status="complete",
                text="second",
            ),
        )
    )
    target = ThinkingReplayTarget(
        "llama_cpp", "alias", "chat_completions", "displayable", 1
    )
    assert (
        resolve_thinking_history(
            target=target,
            policy="auto",
            sidecars=(ProviderThinkingSidecar("a", envelope),),
        ).groups
        == ()
    )
    with pytest.raises(ThinkingHistorySerializationError):
        resolve_thinking_history(
            target=target,
            policy="include",
            sidecars=(ProviderThinkingSidecar("a", envelope),),
        )


@pytest.mark.parametrize(
    "provider,source", [("llama_cpp", "reasoning"), ("ollama", "reasoning_content")]
)
def test_declared_source_must_match_local_wire_field(provider, source):
    target = ThinkingReplayTarget(
        provider, "alias", "chat_completions", "displayable", 1
    )
    assert (
        resolve_thinking_history(
            target=target,
            policy="auto",
            sidecars=(sidecar("a", "private", source, provider),),
        ).groups
        == ()
    )
    with pytest.raises(ThinkingHistorySerializationError):
        resolve_thinking_history(
            target=target,
            policy="include",
            sidecars=(sidecar("a", "private", source, provider),),
        )


def test_agent_projection_uses_canonical_per_call_envelope_and_off_keeps_protocol():
    from tldw_chatbook.Chat.console_thinking_capture import CALL_THINKING_KEY

    rows = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            CALL_THINKING_KEY: sidecar("call", "EXACT_CALL").envelope,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "found"},
        {"role": "user", "content": "continue", replay.EXCHANGE_CONTINUATION_KEY: True},
    ]
    current = replay.project_reasoning_history(
        rows,
        provider="llama_cpp",
        model="alias",
        policy=replay.ReasoningReplayPolicy("current", "test", "Gemma 4"),
    )
    off = replay.project_reasoning_history(
        rows,
        provider="llama_cpp",
        model="alias",
        policy=replay.ReasoningReplayPolicy("off", "test", "Gemma 4"),
    )
    assert current[1]["reasoning_content"] == "EXACT_CALL"
    assert "reasoning_content" not in off[1]
    assert off[1]["tool_calls"] == current[1]["tool_calls"]
    assert (
        off[-1]["content"]
        == current[-1]["content"]
        == "found\n\n[Runtime guidance]\ncontinue"
    )
    assert CALL_THINKING_KEY not in str(current)
    assert (
        replay.project_reasoning_history([], provider="llama_cpp", model="alias") == []
    )
