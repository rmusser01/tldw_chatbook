from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from tldw_chatbook.Chat import provider_endpoint_contract as contract


@pytest.mark.parametrize(
    ("value", "persisted", "chat", "models", "form"),
    [
        (
            "http://127.0.0.1:9000",
            "http://127.0.0.1:9000/v1/chat/completions",
            "http://127.0.0.1:9000/v1/chat/completions",
            "http://127.0.0.1:9000/v1/models",
            "origin",
        ),
        (
            "http://127.0.0.1:9000/v1",
            "http://127.0.0.1:9000/v1/chat/completions",
            "http://127.0.0.1:9000/v1/chat/completions",
            "http://127.0.0.1:9000/v1/models",
            "api_base",
        ),
        (
            "https://example.test/proxy/v1/chat/completions",
            "https://example.test/proxy/v1/chat/completions",
            "https://example.test/proxy/v1/chat/completions",
            "https://example.test/proxy/v1/models",
            "chat_url",
        ),
        (
            "https://example.test/proxy/v1/models",
            "https://example.test/proxy/v1/chat/completions",
            "https://example.test/proxy/v1/chat/completions",
            "https://example.test/proxy/v1/models",
            "models_url",
        ),
    ],
)
def test_custom_endpoint_forms_resolve_to_openai_routes(
    value: str,
    persisted: str,
    chat: str,
    models: str,
    form: str,
) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.provider_key == "custom"
    assert result.persisted_endpoint == persisted
    assert result.chat_url == chat
    assert result.models_url == models
    assert result.persisted_display == persisted
    assert result.chat_display == chat
    assert result.models_display == models
    assert result.form == form
    assert result.errors == ()


@pytest.mark.parametrize(
    ("value", "form"),
    [
        ("http://127.0.0.1:8080/v1/chat/completions", "chat_url"),
        ("http://127.0.0.1:8080/completion", "legacy_local"),
    ],
)
def test_llama_cpp_persists_root_and_derives_openai_routes(
    value: str, form: str
) -> None:
    result = contract.resolve_provider_endpoint("llama_cpp", value)

    assert result.provider_key == "llama_cpp"
    assert result.persisted_endpoint == "http://127.0.0.1:8080"
    assert result.chat_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert result.models_url == "http://127.0.0.1:8080/v1/models"
    assert result.form == form
    assert result.errors == ()


def test_local_llamacpp_uses_the_same_endpoint_rules_as_llama_cpp() -> None:
    value = "http://127.0.0.1:8080/v1/models"

    llama_cpp = contract.resolve_provider_endpoint("llama_cpp", value)
    local_llamacpp = contract.resolve_provider_endpoint("local_llamacpp", value)

    assert local_llamacpp.provider_key == "local_llamacpp"
    assert local_llamacpp.persisted_endpoint == llama_cpp.persisted_endpoint
    assert local_llamacpp.chat_url == llama_cpp.chat_url
    assert local_llamacpp.models_url == llama_cpp.models_url
    assert local_llamacpp.form == llama_cpp.form == "models_url"


def test_other_provider_uses_generic_full_chat_persistence() -> None:
    result = contract.resolve_provider_endpoint(
        "vllm", "https://inference.example.test/openai/v1"
    )

    assert result.provider_key == "vllm"
    assert result.persisted_endpoint == (
        "https://inference.example.test/openai/v1/chat/completions"
    )
    assert result.models_url == "https://inference.example.test/openai/v1/models"
    assert result.form == "api_base"


def test_whitespace_trailing_slashes_and_local_schemeless_input_are_normalized() -> (
    None
):
    result = contract.resolve_provider_endpoint(
        " custom ", "  localhost:9000/proxy/v1/  "
    )

    assert result.provider_key == "custom"
    assert result.normalized_input == "http://localhost:9000/proxy/v1"
    assert result.persisted_endpoint == (
        "http://localhost:9000/proxy/v1/chat/completions"
    )
    assert result.models_url == "http://localhost:9000/proxy/v1/models"
    assert result.form == "api_base"
    assert result.warnings == ()


def test_explicit_remote_http_is_accepted_with_one_bounded_safe_warning() -> None:
    result = contract.resolve_provider_endpoint(
        "custom", "http://example.test/v1"
    )

    assert result.persisted_endpoint == "http://example.test/v1/chat/completions"
    assert len(result.warnings) == 1
    assert "HTTP" in result.warnings[0]
    assert len(result.warnings[0]) <= 100
    assert "example.test" not in result.warnings[0]


@pytest.mark.parametrize(
    "value",
    [
        "example.test:9000/v1",
        "https://user:secret@example.test/v1",
        "https://example.test/v1?token=secret",
        "https://example.test/v1#secret",
        "ftp://example.test/v1",
        "http:///v1",
        "http://:8080/v1",
        "http://localhost:0/v1",
        "http://localhost:99999/v1",
        "http://bad_host/v1",
        "https://example.test/proxy%2Fv1%2Fchat%2Fcompletions",
        "https://example.test/proxy/v1%2fchat%2fcompletions",
        "https://example.test/proxy%5Cv1%5Cchat%5Ccompletions",
        "https://example.test/proxy\\v1\\chat\\completions",
        "https://example.test/v1/chat/completions/v1/models",
        "https://example.test/v1/v1/chat/completions",
        123,
        None,
    ],
)
def test_unsafe_or_ambiguous_inputs_are_rejected_without_echoing_secrets(
    value: object,
) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.chat_url is None
    assert result.models_url is None
    assert result.normalized_input == ""
    assert result.form is None
    assert result.errors
    assert all(len(message) <= 100 for message in result.errors)
    combined_display = (
        f"{result.persisted_display} {result.chat_display} "
        f"{result.models_display}"
    )
    assert "user" not in combined_display
    assert "secret" not in combined_display
    assert "token" not in combined_display
    assert "?" not in combined_display
    assert "#" not in combined_display
    assert "@" not in combined_display


def test_empty_provider_is_rejected_with_bounded_copy() -> None:
    result = contract.resolve_provider_endpoint("  ", "http://localhost:9000")

    assert result.provider_key == ""
    assert result.persisted_endpoint is None
    assert result.errors
    assert all(len(message) <= 100 for message in result.errors)


@pytest.mark.parametrize(
    "value",
    [
        "http://127.0.0.1:9000",
        "http://127.0.0.1:9000/",
        "http://127.0.0.1:9000/v1",
        "http://127.0.0.1:9000/v1/",
        "http://127.0.0.1:9000/v1/chat/completions",
        "http://127.0.0.1:9000/v1/chat/completions/",
        "http://127.0.0.1:9000/v1/models",
    ],
)
def test_custom_equivalent_forms_have_one_canonical_identity(value: str) -> None:
    identity = contract.canonical_connection_identity("custom", value)

    assert identity == (
        "custom",
        "http://127.0.0.1:9000/v1/chat/completions",
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8080/v1/",
        "http://127.0.0.1:8080/v1/chat/completions/",
        "http://127.0.0.1:8080/v1/models",
        "http://127.0.0.1:8080/completion/",
    ],
)
def test_llama_equivalent_forms_have_one_canonical_identity(value: str) -> None:
    identity = contract.canonical_connection_identity("llama_cpp", value)

    assert identity == ("llama_cpp", "http://127.0.0.1:8080")


def test_proxy_prefixes_remain_distinct_canonical_identities() -> None:
    direct = contract.canonical_connection_identity(
        "custom", "https://example.test/v1"
    )
    proxied = contract.canonical_connection_identity(
        "custom", "https://example.test/proxy/v1"
    )

    assert direct != proxied


def test_invalid_endpoint_has_no_canonical_identity() -> None:
    assert (
        contract.canonical_connection_identity(
            "custom", "https://user:secret@example.test/v1"
        )
        is None
    )


def test_public_resolution_type_is_frozen_slotted_and_form_type_is_complete() -> (
    None
):
    result = contract.resolve_provider_endpoint("custom", "localhost:9000")

    assert get_args(contract.EndpointForm) == (
        "origin",
        "api_base",
        "chat_url",
        "models_url",
        "legacy_local",
    )
    assert hasattr(contract.ProviderEndpointResolution, "__slots__")
    with pytest.raises(FrozenInstanceError):
        result.provider_key = "changed"
