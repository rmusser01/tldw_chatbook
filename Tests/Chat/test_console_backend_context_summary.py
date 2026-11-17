"""Backend context arithmetic uses the original display builders and unknowns."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_execution_readiness_values import _readiness
from tldw_chatbook.Chat import console_session_settings as settings_source
from tldw_chatbook.Chat.console_backend_context_summary import (
    ConsoleBackendContextInputs,
    build_backend_context_summary,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    build_console_context_estimate,
    build_console_settings_summary_state,
)
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.Utils.token_counter import ContextWindowResolution
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
    compaction_forecast_rows,
)


def _inputs():
    messages = (
        ConsoleChatMessage(ConsoleMessageRole.USER, "question", id="user"),
        ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer", id="assistant"),
        ConsoleChatMessage(ConsoleMessageRole.USER, "unadmitted", id="not-in-request"),
    )
    return ConsoleBackendContextInputs(
        settings=ConsoleSessionSettings(
            provider="custom", model="model-a", system_prompt="system"
        ),
        summary_settings=ConsoleSessionSettings(
            provider="custom", model="resolved-model", system_prompt="system"
        ),
        readiness=_readiness(),
        messages=messages,
        request_ids=frozenset({"user", "assistant"}),
        active_session=True,
        seeded_greeting="greeting",
        draft_text="new draft",
        staged_source_count=1,
        staged_context_summary="one source",
        staged_text="staged evidence",
        context_window=ContextWindowResolution(4096, "captured", True),
        tokenizer_ready=True,
        overrides=ConsoleContextPolicyOverrides(),
        global_overrides=None,
        memory=EffectiveMemoryResult(EffectiveMemoryKind.RAW),
        accounting=None,
        thinking_history_policy="auto",
        thinking_history_effective_policy=None,
        provider_token_caps=(None, None),
        agent_runtime_enabled=False,
        agent_bridge_present=False,
        has_submit_prefill=False,
        has_pending_attachments=False,
    )


def _display_counter(messages, _model, _provider):
    # A deterministic numerical dependency of the original pure builder;
    # this is not a tokenizer-readiness or native-ownership qualification.
    return 2 + sum(2 + len(row["content"]) for row in messages)


def test_unknown_tokenizer_never_estimates_or_resolves_a_window(monkeypatch):
    def unavailable(*_args, **_kwargs):
        pytest.fail("Unprepared display must not enter token counting or resolution")

    monkeypatch.setattr(settings_source, "_estimate_tokens_locally", unavailable)
    monkeypatch.setattr(
        settings_source, "_resolve_token_limit_locally_with_provenance", unavailable
    )
    inputs = replace(_inputs(), tokenizer_ready=False, context_window=None, memory=None)
    result = build_backend_context_summary(inputs)
    expected = ConsoleSettingsContextEstimate(None, None, "Context unavailable")
    assert result.estimate == expected
    assert result.summary == build_console_settings_summary_state(
        inputs.summary_settings, expected, inputs.readiness
    )
    assert result.compaction_pause_expected is False
    assert result.compaction_forecast_unavailable is False
    # The original display gate reports ready for an absent model; the builder
    # returns unknown before it needs either a tokenizer or a window resolver.
    missing = build_backend_context_summary(
        replace(
            inputs, tokenizer_ready=True, settings=replace(inputs.settings, model=None)
        )
    )
    assert missing.estimate == ConsoleSettingsContextEstimate(
        None, None, "Context: unavailable", 1, "one source"
    )


def test_settled_history_plus_live_draft_preserves_one_framing_base(monkeypatch):
    monkeypatch.setattr(settings_source, "_estimate_tokens_locally", _display_counter)
    monkeypatch.setattr(
        settings_source,
        "_resolve_token_limit_locally_with_provenance",
        lambda *_: pytest.fail(
            "All windows must come from the captured original result"
        ),
    )
    inputs = _inputs()
    rows = spend.build_console_context_messages(inputs.messages, inputs.request_ids, "")
    prompt = spend.fold_system_prompt(
        inputs.settings.system_prompt, inputs.seeded_greeting
    )
    expected = build_console_context_estimate(
        [*rows, {"role": "user", "content": inputs.draft_text}],
        inputs.settings.provider,
        inputs.settings.model,
        staged_source_count=1,
        staged_context_summary="one source",
        system_prompt=prompt,
        staged_text=inputs.staged_text,
        context_window=inputs.context_window,
    )
    result = build_backend_context_summary(inputs)
    assert result.estimate == expected
    assert result.summary == build_console_settings_summary_state(
        inputs.summary_settings, expected, inputs.readiness
    )
    settled = build_backend_context_summary(replace(inputs, draft_text="  "))
    assert result.estimate.used_tokens - settled.estimate.used_tokens == 2 + len(
        inputs.draft_text
    )
    with pytest.raises(ValueError, match="context window"):
        build_backend_context_summary(replace(inputs, context_window=None))


def test_ask_forecast_uses_original_unknown_mandatory_input_result(monkeypatch):
    monkeypatch.setattr(settings_source, "_estimate_tokens_locally", _display_counter)
    inputs = replace(_inputs(), agent_runtime_enabled=True, agent_bridge_present=True)
    result = build_backend_context_summary(inputs)
    admitted = [row for row in inputs.messages if row.id in inputs.request_ids]
    rows, units = compaction_forecast_rows(admitted, inputs.memory)
    forecast = build_console_context_estimate(
        list(rows),
        inputs.settings.provider,
        inputs.settings.model,
        context_window=inputs.context_window,
    )
    state = build_console_context_control_state(
        settings=inputs.settings,
        estimate=result.estimate,
        overrides=inputs.overrides,
        global_overrides=inputs.global_overrides,
        active_memory=inputs.memory,
        accounting=inputs.accounting,
        thinking_history_policy=inputs.thinking_history_policy,
        thinking_history_effective_policy=inputs.thinking_history_effective_policy,
        forecast_tokens=None,
        forecast_units=units,
        request_overhead_tokens=None,
        provider_input_cap_tokens=None,
        provider_output_cap_tokens=None,
        forecast_unavailable_reason="Current mandatory input or tool overhead is not measured.",
    )
    assert forecast.used_tokens is not None and units == 1
    assert result.compaction_pause_expected is (
        state.compaction_pause_forecast is not None
    )
    assert result.compaction_forecast_unavailable is bool(
        state.forecast_unavailable_reason
    )
    assert result.compaction_pause_expected is False
    assert result.compaction_forecast_unavailable is True
