import inspect
from dataclasses import replace
from functools import cache

import pytest

from tldw_chatbook.Chat import console_provider_support as support_module
from tldw_chatbook.Chat.Chat_Functions import (
    PROVIDER_PARAM_MAP,
    project_chat_handler_kwargs,
)
from tldw_chatbook.Chat.console_prepared_request import (
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    build_llamacpp_chat_payload,
)
from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    ConsoleControlSupport,
    ConsoleGenerationControl,
    ConsoleProviderIdentity,
    console_generation_control_support,
    resolve_console_provider_identity,
    supported_console_provider_catalog,
    supported_console_provider_readiness_keys,
    supported_generation_fields,
)
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
)
from tldw_chatbook.Chat.console_settings_apply import FULL_MODEL_DEFAULT_FIELDS


@pytest.mark.parametrize(
    ("provider", "model", "control", "expected"),
    (
        ("llama_cpp", "model-a", "reasoning_effort", "supported"),
        ("llama_cpp", "model-a", "thinking_budget_tokens", "supported"),
        ("llama_cpp", "model-a", "reasoning_summary", "unsupported"),
        ("llama_cpp", "model-a", "verbosity", "unsupported"),
        ("llama_cpp", "model-a", "thinking_effort", "unsupported"),
        ("local_vllm", "model-a", "reasoning_effort", "supported"),
        ("local_vllm", "model-a", "thinking_budget_tokens", "unsupported"),
        ("openai", "gpt-5.6-terra", "thinking_effort", "unsupported"),
        ("openai", "gpt-5.6-terra", "thinking_budget_tokens", "unsupported"),
        ("moonshot", "kimi-k3", "reasoning_effort", "supported"),
        ("zai", "glm-5.2", "reasoning_effort", "supported"),
        # Engine presets whose record refuses reasoning effort (TASK-33501),
        # except a model with a thinking toggle (TASK-33502).
        ("nvidia", "meta/llama-3.3-70b-instruct", "reasoning_effort", "unsupported"),
        ("nvidia", "qwen/qwen3.5-397b-a17b", "reasoning_effort", "supported"),
        ("fireworks", "accounts/fireworks/models/qwen3p8", "reasoning_effort", "unsupported"),
        ("together", "moonshotai/Kimi-K3", "reasoning_effort", "unsupported"),
        ("cerebras", "gpt-oss-120b", "reasoning_effort", "unsupported"),
        ("anthropic", "claude-sonnet-5", "thinking_effort", "supported"),
        (
            "anthropic",
            "claude-sonnet-5",
            "thinking_budget_tokens",
            "unsupported",
        ),
    ),
)
def test_generation_control_support_uses_existing_authoritative_facts(
    provider: str,
    model: str,
    control: ConsoleGenerationControl,
    expected: ConsoleControlSupport,
) -> None:
    assert console_generation_control_support(provider, model, control) == expected


@pytest.mark.parametrize(
    ("provider", "model", "control"),
    (
        ("openai", "future-custom-model", "reasoning_effort"),
        ("openai", "future-custom-model", "reasoning_summary"),
        ("openai", None, "verbosity"),
        ("openai", "gpt-5.6-terra", "reasoning_effort"),
        ("openai", "gpt-5.6-terra", "reasoning_summary"),
        ("openai", "gpt-5.6-terra", "verbosity"),
        ("custom", "private-model", "reasoning_effort"),
        ("definitely-not-real", "private-model", "thinking_effort"),
    ),
)
def test_generation_control_support_keeps_unproved_models_unknown(
    provider: str,
    model: str | None,
    control: ConsoleGenerationControl,
) -> None:
    assert console_generation_control_support(provider, model, control) == "unknown"


def test_generation_control_support_keeps_mlx_reasoning_pending_verification() -> None:
    assert (
        console_generation_control_support(
            "local_mlx_lm",
            "mlx-community/Qwen3-4B",
            "reasoning_effort",
        )
        == "unknown"
    )


def test_aliases_resolve_to_readiness_and_execution_keys() -> None:
    cases = {
        "Custom": ("custom", "custom-openai-api"),
        "custom": ("custom", "custom-openai-api"),
        "custom-openai-api": ("custom", "custom-openai-api"),
        "Custom-2": ("custom_2", "custom-openai-api-2"),
        "custom_2": ("custom_2", "custom-openai-api-2"),
        "custom-2": ("custom_2", "custom-openai-api-2"),
        "custom-openai-api-2": ("custom_2", "custom-openai-api-2"),
        "local_llm": ("local_llm", "local-llm"),
        "local-llm": ("local_llm", "local-llm"),
        "mlx_lm": ("local_mlx_lm", "local_mlx_lm"),
        "local_mlx_lm": ("local_mlx_lm", "local_mlx_lm"),
        "MistralAI": ("mistralai", "mistralai"),
        "mistralai": ("mistralai", "mistralai"),
    }

    for raw, expected in cases.items():
        identity = resolve_console_provider_identity(raw)

        assert (identity.readiness_key, identity.execution_key) == expected
        assert identity.is_supported is True


def test_normalized_handler_key_resolves_to_hyphenated_execution_key() -> None:
    identity = resolve_console_provider_identity(
        "custom_openai_api",
        handler_keys={"custom-openai-api"},
    )

    assert identity == ConsoleProviderIdentity(
        display_key="custom_openai_api",
        readiness_key="custom",
        execution_key="custom-openai-api",
        is_supported=True,
    )


def test_numbered_normalized_handler_key_resolves_to_hyphenated_execution_key() -> None:
    identity = resolve_console_provider_identity(
        "custom_openai_api_2",
        handler_keys={"custom-openai-api-2"},
    )

    assert identity == ConsoleProviderIdentity(
        display_key="custom_openai_api_2",
        readiness_key="custom_2",
        execution_key="custom-openai-api-2",
        is_supported=True,
    )


def test_direct_console_provider_keys_are_not_generic_adapter() -> None:
    for provider in DIRECT_CONSOLE_PROVIDER_KEYS:
        identity = resolve_console_provider_identity(provider)

        assert identity.uses_direct_llama_path is True
        assert identity.execution_key == provider
        assert identity.readiness_key == provider
        assert identity.is_supported is True


def test_direct_console_provider_identity_does_not_require_handler_keys(
    monkeypatch,
) -> None:
    def fail_handler_lookup(*_args, **_kwargs):
        raise AssertionError("direct providers should not load chat_api_call handlers")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_provider_support._handler_keys",
        fail_handler_lookup,
    )

    identity = resolve_console_provider_identity("llama_cpp")

    assert identity.uses_direct_llama_path is True
    assert identity.execution_key == "llama_cpp"


def test_preserves_exact_execution_key_when_present_in_handler_keys() -> None:
    identity = resolve_console_provider_identity(
        "provider-with-hyphen",
        handler_keys={"provider-with-hyphen"},
    )

    assert identity == ConsoleProviderIdentity(
        display_key="provider_with_hyphen",
        readiness_key="provider_with_hyphen",
        execution_key="provider-with-hyphen",
        is_supported=True,
    )


def test_all_chat_api_call_handlers_are_known_to_console_support() -> None:
    keys = supported_console_provider_readiness_keys()

    assert "openai" in keys
    assert "anthropic" in keys
    assert "local_vllm" in keys
    assert "custom" in keys
    assert "custom_2" in keys


def test_supported_console_provider_catalog_describes_handler_identities() -> None:
    catalog = supported_console_provider_catalog(
        handler_keys={
            "openai",
            "anthropic",
            "custom-openai-api",
            "custom-openai-api-2",
            "llama_cpp",
            "local_vllm",
            "mlx_lm",
            "local_mlx_lm",
            "qwencloud",
        }
    )
    by_readiness_key = {entry.readiness_key: entry for entry in catalog}

    assert set(by_readiness_key) == {
        "anthropic",
        "custom",
        "custom_2",
        "llama_cpp",
        "local_mlx_lm",
        "local_vllm",
        "openai",
        "qwencloud",
    }
    assert by_readiness_key["custom"].execution_key == "custom-openai-api"
    assert by_readiness_key["custom_2"].execution_key == "custom-openai-api-2"
    assert by_readiness_key["llama_cpp"].requires_api_key is False
    assert by_readiness_key["openai"].requires_api_key is True
    assert by_readiness_key["openai"].display_name == "OpenAI"
    assert by_readiness_key["qwencloud"].execution_key == "qwencloud"
    assert by_readiness_key["qwencloud"].display_name == "QwenCloud"


def test_all_chat_api_call_handlers_resolve_to_supported_console_identity() -> None:
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    handler_keys = frozenset(API_CALL_HANDLERS)

    for handler_key in sorted(handler_keys):
        identity = resolve_console_provider_identity(
            handler_key,
            handler_keys=handler_keys,
        )

        assert identity.is_supported is True, handler_key
        assert identity.execution_key in handler_keys, handler_key


def test_supported_readiness_key_sweep_deduplicates_aliases() -> None:
    keys = supported_console_provider_readiness_keys(
        handler_keys={
            "custom-openai-api",
            "custom-openai-api-2",
            "local-llm",
            "mlx_lm",
            "local_mlx_lm",
            "mistralai",
        }
    )

    assert keys == frozenset(
        {"custom", "custom_2", "local_llm", "local_mlx_lm", "mistralai"}
    )


def test_unsupported_provider_returns_not_supported_without_crashing() -> None:
    identity = resolve_console_provider_identity(
        "definitely-not-real",
        handler_keys={"openai"},
    )

    assert identity == ConsoleProviderIdentity(
        display_key="definitely_not_real",
        readiness_key="definitely_not_real",
        execution_key="definitely_not_real",
        is_supported=False,
    )


# ---------------------------------------------------------------------------
# TASK-33001.2: one field-support decision that matches what the request
# forwards.
# ---------------------------------------------------------------------------

_ANTHROPIC_DROPPED = frozenset(
    {"min_p", "seed", "presence_penalty", "frequency_penalty"}
)
_TABLE_MODELS = (
    None,
    "gpt-5",
    "claude-sonnet-4-5",
    "claude-sonnet-5",
    "kimi-k3",
    "moonshot-v1-8k",
    "glm-5.2",
    "glm-4.6",
)
# One non-default value per field; each is sent on its own and compared with
# a request that leaves the field unset, so nothing depends on value
# collisions between fields.
_PROBE_VALUES: dict[str, object] = {
    "temperature": 0.37,
    "top_p": 0.61,
    "min_p": 0.07,
    "top_k": 23,
    "max_tokens": 777,
    "seed": 4242,
    "presence_penalty": 0.31,
    "frequency_penalty": 0.29,
    "reasoning_effort": "low",
    "reasoning_summary": "detailed",
    "verbosity": "high",
    "thinking_effort": "xhigh",
    "thinking_budget_tokens": 4096,
    "streaming": False,
}


def production_send_kwargs(resolution: ConsoleProviderResolution) -> dict:
    """The ``chat_api_call`` kwargs a real Console send builds for ``resolution``.

    Main sends and tool rounds go through ``prepare_provider_request`` and
    ``ConsoleProviderGateway._chat_api_kwargs_from_prepared``; this drives both.
    """
    prepared = prepare_provider_request(
        build_console_request([{"role": "user", "content": "hi"}]),
        wire_style="distinct_roles",
        model=resolution.model or "probe-model",
        provider=resolution.provider,
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=lambda messages, _model: len(messages),
    )
    return ConsoleProviderGateway._chat_api_kwargs_from_prepared(resolution, prepared)


def _forwarded(request) -> frozenset[str]:
    """Fields whose probe value changes what ``request(**fields)`` delivers."""
    unset = request()
    return frozenset(
        name
        for name, value in _PROBE_VALUES.items()
        if request(**{name: value}) != unset
    )


@cache
def _fields_the_request_forwards(provider: str) -> frozenset[str]:
    """Measure which fields a real Console send delivers, field by field.

    Direct llama.cpp providers send through ``build_llamacpp_chat_payload``;
    every other provider through ``_chat_api_kwargs_from_prepared`` and the
    ``chat_api_call`` projection (``project_chat_handler_kwargs``).
    """

    identity = resolve_console_provider_identity(provider)
    if identity.uses_direct_llama_path:
        accepted = inspect.signature(build_llamacpp_chat_payload).parameters

        def request(**fields: object) -> dict[str, object]:
            stream = fields.pop("streaming", True)
            return build_llamacpp_chat_payload(
                model="probe-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=stream,
                **{name: value for name, value in fields.items() if name in accepted},
            )

        return _forwarded(request)

    base = _probe_resolution(provider)

    def request(**fields: object) -> dict[str, object]:
        kwargs = production_send_kwargs(replace(base, **fields))
        return project_chat_handler_kwargs(kwargs.pop("api_endpoint"), kwargs)

    return _forwarded(request)


def _probe_resolution(provider: str) -> ConsoleProviderResolution:
    return ConsoleProviderResolution(
        provider=provider,
        base_url="",
        model="probe-model",
        ready=True,
        execution_key=resolve_console_provider_identity(provider).execution_key,
    )


@pytest.mark.parametrize("model", _TABLE_MODELS)
@pytest.mark.parametrize("provider", sorted(CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS))
def test_supported_generation_fields_match_what_the_request_forwards(
    provider: str, model: str | None
) -> None:
    """AC#7: supported == capability rules ∩ fields the request really sends."""

    forwarded = _fields_the_request_forwards(provider)
    supported = supported_generation_fields(provider, model)

    dropped_but_reported = supported - forwarded
    assert not dropped_but_reported, (
        f"{provider}/{model}: the request drops {sorted(dropped_but_reported)}"
    )
    assert supported == (
        support_module._capability_generation_fields(provider, model) & forwarded
    )


@pytest.mark.parametrize(
    "provider",
    sorted(
        provider
        for provider in CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
        if not resolve_console_provider_identity(provider).uses_direct_llama_path
    ),
)
def test_auxiliary_requests_forward_the_same_generation_fields(provider: str) -> None:
    """AC#7: the auxiliary builder (``_auxiliary_chat_api_kwargs``) carries the
    same resolution fields as a main send. Its max tokens come from the
    request and it never streams, so those two are the only difference."""

    base = _probe_resolution(provider)
    auxiliary = AuxiliaryCompletionRequest(
        resolution=base,
        messages=({"role": "user", "content": "hi"},),
        response_format=None,
        max_output_tokens=64,
    )

    def request(**fields: object) -> dict[str, object]:
        kwargs = ConsoleProviderGateway._auxiliary_chat_api_kwargs(
            auxiliary, replace(base, **fields)
        )
        return project_chat_handler_kwargs(kwargs.pop("api_endpoint"), kwargs)

    assert _forwarded(request) == _fields_the_request_forwards(provider) - {
        "max_tokens",
        "streaming",
    }


def test_anthropic_hides_only_the_fields_its_request_drops() -> None:
    """AC#4: Min P, Seed and both penalties are gone; the rest keeps support."""

    fields = supported_generation_fields("anthropic", "claude-sonnet-4-5")

    assert fields.isdisjoint(_ANTHROPIC_DROPPED)
    assert fields == frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "streaming",
            "thinking_effort",
            "thinking_budget_tokens",
        }
    )
    # The model-gated fixed thinking budget keeps its existing rule.
    assert "thinking_budget_tokens" not in supported_generation_fields(
        "Anthropic", "claude-sonnet-5"
    )


def test_openai_top_p_rides_maxp() -> None:
    """AC#5: OpenAI's map forwards top_p only as ``maxp``."""

    fields = supported_generation_fields("openai", "gpt-5")

    assert "top_p" in fields
    assert "maxp" in PROVIDER_PARAM_MAP["openai"]
    assert "topp" not in PROVIDER_PARAM_MAP["openai"]


def test_provider_without_a_request_map_keeps_todays_fields(monkeypatch) -> None:
    """AC#6 (TASK-30012 AC#3): no map entry means no intersection."""

    everyday = FULL_MODEL_DEFAULT_FIELDS - frozenset(
        {
            "reasoning_effort",
            "reasoning_summary",
            "verbosity",
            "thinking_effort",
            "thinking_budget_tokens",
        }
    )
    assert supported_generation_fields("acme-private-llm", "any") == everyday

    monkeypatch.delitem(PROVIDER_PARAM_MAP, "anthropic")
    assert supported_generation_fields(
        "anthropic", "claude-sonnet-4-5"
    ) == support_module._capability_generation_fields("anthropic", "claude-sonnet-4-5")
    assert _ANTHROPIC_DROPPED <= supported_generation_fields(
        "anthropic", "claude-sonnet-4-5"
    )


def test_every_engine_preset_refusing_reasoning_effort_hides_the_control() -> None:
    """TASK-33501: the send would fail locally, so the Console must not offer it."""
    from tldw_chatbook.provider_registry import ALL_RECORDS

    refusing = [r.key for r in ALL_RECORDS if r.engine_driven and not r.reasoning_effort]
    assert refusing
    for key in refusing:
        assert (
            console_generation_control_support(key, "no-toggle-model", "reasoning_effort")
            == "unsupported"
        ), key


def test_engine_presets_accepting_reasoning_effort_are_not_hidden() -> None:
    from tldw_chatbook.provider_registry import ALL_RECORDS

    accepting = [r.key for r in ALL_RECORDS if r.engine_driven and r.reasoning_effort]
    for key in accepting:
        assert (
            console_generation_control_support(key, "any-model", "reasoning_effort")
            != "unsupported"
        ), key


def test_nvidia_qwen_thinking_toggle_survives_the_draft_rebase() -> None:
    """TASK-33502: the draft rebase carries reasoning effort only where it is sent."""
    assert "reasoning_effort" in supported_generation_fields("nvidia", "qwen/qwen3.5-397b-a17b")
    assert "reasoning_effort" not in supported_generation_fields(
        "nvidia", "meta/llama-3.3-70b-instruct"
    )

