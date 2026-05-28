from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    ConsoleProviderIdentity,
    resolve_console_provider_identity,
    supported_console_provider_readiness_keys,
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


def test_direct_console_provider_keys_are_not_generic_adapter() -> None:
    for provider in DIRECT_CONSOLE_PROVIDER_KEYS:
        identity = resolve_console_provider_identity(provider)

        assert identity.uses_direct_llama_path is True
        assert identity.execution_key == provider
        assert identity.readiness_key == provider
        assert identity.is_supported is True


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
