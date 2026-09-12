from dataclasses import FrozenInstanceError, fields
from typing import get_args

import pytest

from tldw_chatbook.Chat import provider_endpoint_contract as contract


@pytest.mark.parametrize(
    ("provider", "provider_key"),
    [
        ("Custom", "custom"),
        ("Custom OpenAI", "custom"),
        ("Custom OpenAI API", "custom"),
        ("custom-openai", "custom"),
        ("custom_openai", "custom"),
        ("custom-openai-api", "custom"),
        ("custom_openai_api", "custom"),
        ("Custom-2", "custom_2"),
        ("Custom 2", "custom_2"),
        ("custom-2", "custom_2"),
        ("Custom OpenAI 2", "custom_2"),
        ("custom_openai_2", "custom_2"),
        ("Custom OpenAI API-2", "custom_2"),
        ("custom-openai-api-2", "custom_2"),
        ("custom_openai_api_2", "custom_2"),
        ("llama.cpp", "llama_cpp"),
        ("local llama.cpp", "local_llamacpp"),
        ("vllm", "vllm"),
        ("openrouter", "openrouter"),
        ("OpenRouter", "openrouter"),
        ("local_llamacpp", "local_llamacpp"),
        ("local-llamacpp", "local_llamacpp"),
        ("custom__openai__api", "custom__openai__api"),
    ],
)
def test_provider_aliases_resolve_to_canonical_config_keys(
    provider: str, provider_key: str
) -> None:
    result = contract.resolve_provider_endpoint(provider, "http://localhost:9000")

    assert result.provider_key == provider_key
    assert contract.canonical_connection_identity(provider, "localhost:9000") == (
        provider_key,
        result.persisted_endpoint,
    )


@pytest.mark.parametrize(
    "provider",
    [
        "custom---openai",
        "open router",
        "local--vllm",
    ],
)
def test_unknown_provider_keys_must_already_be_canonical(provider: str) -> None:
    result = contract.resolve_provider_endpoint(provider, "http://localhost:9000")

    assert result.provider_key == ""
    assert result.persisted_endpoint is None
    assert result.errors


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


@pytest.mark.parametrize("provider", ["custom", "vllm"])
def test_completion_is_an_origin_prefix_for_non_llama_providers(
    provider: str,
) -> None:
    result = contract.resolve_provider_endpoint(
        provider, "https://example.test/completion"
    )

    assert result.normalized_input == "https://example.test/completion"
    assert result.persisted_endpoint == (
        "https://example.test/completion/v1/chat/completions"
    )
    assert result.chat_url == "https://example.test/completion/v1/chat/completions"
    assert result.models_url == "https://example.test/completion/v1/models"
    assert result.form == "origin"


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
        "https://example.test/v1?",
        "https://example.test/proxy/v1?",
        "https://example.test/v1#",
        "https://example.test/proxy/v1#",
    ],
)
def test_empty_query_and_fragment_markers_are_rejected(value: str) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.chat_url is None
    assert result.models_url is None
    assert result.form is None
    assert result.errors


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test:/v1",
        "http://localhost:/v1",
        "https://[::1]:/v1",
    ],
)
def test_explicit_empty_ports_are_rejected(value: str) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.chat_url is None
    assert result.models_url is None
    assert result.errors


@pytest.mark.parametrize(
    ("value", "chat_url"),
    [
        (
            "https://[::1]/v1",
            "https://[::1]/v1/chat/completions",
        ),
        (
            "http://[::1]:8080/v1/chat/completions",
            "http://[::1]:8080/v1/chat/completions",
        ),
    ],
)
def test_valid_explicit_ipv6_endpoints_are_bracket_safe(
    value: str, chat_url: str
) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.chat_url == chat_url
    assert result.persisted_endpoint == chat_url
    assert result.errors == ()


@pytest.mark.parametrize(
    "unsafe_character",
    [
        "\u0085",
        "\u202e",
        "\u2066",
        "\u200d",
        "\ud800",
    ],
)
def test_unsafe_unicode_categories_are_rejected_before_parsing(
    unsafe_character: str,
) -> None:
    result = contract.resolve_provider_endpoint(
        "custom", f"https://example.test/proxy{unsafe_character}/v1"
    )

    assert result.persisted_endpoint is None
    assert result.normalized_input == ""
    assert result.errors
    assert all(len(message) <= 100 for message in result.errors)


@pytest.mark.parametrize("control", [chr(code) for code in range(32)] + ["\x7f"])
def test_c0_controls_and_del_are_rejected_before_normalization(
    control: str,
) -> None:
    result = contract.resolve_provider_endpoint(
        "custom", f"https://example.test/v1{control}"
    )

    assert result.persisted_endpoint is None
    assert result.chat_url is None
    assert result.models_url is None
    assert result.errors


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test/proxy/%ZZ",
        "https://example.test/proxy/%",
        "https://example.test/proxy/%0",
        "https://example.test/proxy%2Fv1",
        "https://example.test/proxy%5cv1",
        "https://example.test/proxy/%00/v1",
        "https://example.test/proxy/%1f/v1",
        "https://example.test/proxy/%7F/v1",
        "https://example.test/proxy/%85/v1",
        "https://example.test/proxy/%C2%85/v1",
        "https://example.test/proxy/%E2%80%AE/v1",
        "https://example.test/proxy/%ED%A0%80/v1",
        "https://example.test/proxy/%C0%AF/v1",
        "https://example.test/proxy/%C0%AE%C0%AE/v1",
        "https://example.test/proxy/%FF%C2%AD/v1",
        "https://example.test/proxy/%2e%2e/v1",
        "https://example.test/proxy/.%2E/v1",
        "https://example.test/proxy/%2E/v1",
    ],
)
def test_malformed_or_unsafe_percent_encoded_paths_are_rejected(value: str) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.chat_url is None
    assert result.models_url is None
    assert result.errors


@pytest.mark.parametrize(
    ("value", "normalized_input"),
    [
        (
            "https://example.test/%70roxy/v1",
            "https://example.test/proxy/v1",
        ),
        (
            "https://example.test/proxy/%7emodel/v1",
            "https://example.test/proxy/~model/v1",
        ),
        (
            "https://example.test/proxy/%3a/v1",
            "https://example.test/proxy/%3A/v1",
        ),
    ],
)
def test_valid_percent_escapes_are_canonicalized(
    value: str, normalized_input: str
) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.normalized_input == normalized_input
    assert result.errors == ()


def test_percent_encoded_unreserved_path_has_same_canonical_identity() -> None:
    encoded = contract.canonical_connection_identity(
        "custom", "https://example.test/%70roxy/v1"
    )
    plain = contract.canonical_connection_identity(
        "custom", "https://example.test/proxy/v1"
    )

    assert encoded == plain


def test_raw_unicode_path_is_serialized_as_uppercase_utf8_percent_escapes() -> None:
    raw = contract.resolve_provider_endpoint(
        "custom", "https://example.test/café/v1"
    )
    encoded = contract.resolve_provider_endpoint(
        "custom", "https://example.test/caf%C3%A9/v1"
    )

    assert raw.normalized_input == "https://example.test/caf%C3%A9/v1"
    assert raw.persisted_endpoint == (
        "https://example.test/caf%C3%A9/v1/chat/completions"
    )
    assert raw.chat_url == encoded.chat_url
    assert raw.models_url == encoded.models_url
    assert raw == encoded
    assert contract.canonical_connection_identity(
        "custom", "https://example.test/café/v1"
    ) == contract.canonical_connection_identity(
        "custom", "https://example.test/caf%C3%A9/v1"
    )


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
        "192.168.1.10:8000/v1",
        "10.0.0.1/v1",
        "127.0.0.2:9000/v1",
        "example.test/v1",
        "service.local:8000/v1",
    ],
)
def test_schemeless_remote_ip_and_dns_endpoints_are_rejected(value: str) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.chat_url is None
    assert result.models_url is None
    assert result.errors


def test_exact_schemeless_ipv6_loopback_is_accepted() -> None:
    result = contract.resolve_provider_endpoint("custom", "[::1]:8080/v1")

    assert result.errors == ()
    assert result.normalized_input == "http://[::1]:8080/v1"
    assert result.chat_url == "http://[::1]:8080/v1/chat/completions"
    assert result.models_url == "http://[::1]:8080/v1/models"


@pytest.mark.parametrize(
    "value",
    [
        "LOCALHOST:9000/v1",
        "localhost.:9000/v1",
        "127.000.000.001:9000/v1",
    ],
)
def test_schemeless_local_policy_requires_exact_raw_host(value: str) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.errors


@pytest.mark.parametrize(
    ("value", "normalized_input"),
    [
        (
            "https://EXAMPLE.TEST./v1",
            "https://example.test/v1",
        ),
        (
            "http://LOCALHOST.:8080/v1",
            "http://localhost:8080/v1",
        ),
    ],
)
def test_explicit_dns_hosts_are_canonicalized(
    value: str, normalized_input: str
) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.normalized_input == normalized_input
    assert result.errors == ()
    if normalized_input.startswith("http://localhost"):
        assert result.warnings == ()


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test../v1",
        "http://localhost..:8080/v1",
    ],
)
def test_dns_hosts_with_multiple_trailing_dots_are_rejected(value: str) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.errors


def test_equivalent_ipv6_spellings_share_serialization_and_identity() -> None:
    expanded = "https://[2001:0DB8:0:0:0:0:0:1]/v1"
    compressed = "https://[2001:db8::1]/v1"

    expanded_result = contract.resolve_provider_endpoint("custom", expanded)
    compressed_result = contract.resolve_provider_endpoint("custom", compressed)

    assert expanded_result.normalized_input == "https://[2001:db8::1]/v1"
    assert expanded_result == compressed_result
    assert contract.canonical_connection_identity(
        "custom", expanded
    ) == contract.canonical_connection_identity("custom", compressed)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test/proxy//",
        "https://example.test///",
        "https://example.test/proxy///",
        "https://example.test//v1",
    ],
)
def test_original_path_rejects_doubled_separators_before_trimming(
    value: str,
) -> None:
    result = contract.resolve_provider_endpoint("custom", value)

    assert result.persisted_endpoint is None
    assert result.errors


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


@pytest.mark.parametrize(
    ("provider", "explicit", "implicit", "expected_identity"),
    [
        (
            "custom",
            "http://example.test:80/v1",
            "http://example.test/v1",
            ("custom", "http://example.test/v1/chat/completions"),
        ),
        (
            "llama_cpp",
            "https://example.test:443/v1",
            "https://example.test/v1",
            ("llama_cpp", "https://example.test"),
        ),
        (
            "custom",
            "http://[::1]:80/v1",
            "http://[::1]/v1",
            ("custom", "http://[::1]/v1/chat/completions"),
        ),
        (
            "custom",
            "https://[::1]:443/v1",
            "https://[::1]/v1",
            ("custom", "https://[::1]/v1/chat/completions"),
        ),
    ],
)
def test_canonical_identity_drops_explicit_default_ports(
    provider: str,
    explicit: str,
    implicit: str,
    expected_identity: tuple[str, str],
) -> None:
    explicit_result = contract.resolve_provider_endpoint(provider, explicit)

    assert explicit_result.persisted_endpoint is not None
    assert ":80" in explicit_result.persisted_endpoint or ":443" in (
        explicit_result.persisted_endpoint
    )
    assert contract.canonical_connection_identity(provider, explicit) == (
        expected_identity
    )
    assert contract.canonical_connection_identity(provider, implicit) == (
        expected_identity
    )


def test_invalid_endpoint_has_no_canonical_identity() -> None:
    assert (
        contract.canonical_connection_identity(
            "custom", "https://user:secret@example.test/v1"
        )
        is None
    )


def test_provider_and_endpoint_length_bounds_are_checked_before_parsing() -> None:
    valid_provider = "a" * 128
    oversized_provider = "a" * 129
    endpoint_prefix = "https://example.test/"
    endpoint_at_limit = endpoint_prefix + "a" * (4096 - len(endpoint_prefix))
    oversized_endpoint = endpoint_at_limit + "a"

    assert contract.resolve_provider_endpoint(
        valid_provider, "http://localhost:9000"
    ).provider_key == valid_provider

    provider_result = contract.resolve_provider_endpoint(
        oversized_provider, "http://localhost:9000"
    )
    endpoint_result = contract.resolve_provider_endpoint(
        "custom", oversized_endpoint
    )

    assert provider_result.persisted_endpoint is None
    assert endpoint_result.persisted_endpoint is None
    assert all(len(message) <= 100 for message in provider_result.errors)
    assert all(len(message) <= 100 for message in endpoint_result.errors)
    assert contract.resolve_provider_endpoint(
        "custom", endpoint_at_limit
    ).persisted_endpoint is not None


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
    assert tuple(field.name for field in fields(contract.ProviderEndpointResolution)) == (
        "provider_key",
        "normalized_input",
        "persisted_endpoint",
        "chat_url",
        "models_url",
        "persisted_display",
        "chat_display",
        "models_display",
        "form",
        "warnings",
        "errors",
    )
    assert hasattr(contract.ProviderEndpointResolution, "__slots__")
    with pytest.raises(FrozenInstanceError):
        result.provider_key = "changed"


@pytest.mark.parametrize(
    "provider",
    ("custom", "llama_cpp", "ollama", "vllm", "tabbyapi"),
)
def test_connection_probe_availability_accepts_valid_url_provider_models_routes(
    provider: str,
) -> None:
    """Removing URL-provider eligibility must hide a useful bounded probe."""
    assert contract.connection_probe_availability(
        provider,
        "http://127.0.0.1:9099/v1",
    ) is contract.ConnectionProbeAvailability.MODELS_ROUTE


@pytest.mark.parametrize("provider", ("openai", "anthropic", "google"))
def test_connection_probe_availability_rejects_cloud_providers_without_a_declared_probe(
    provider: str,
) -> None:
    """A derived URL alone must not invent a live-check contract for cloud APIs."""
    assert contract.connection_probe_availability(
        provider,
        "https://api.example.test/v1",
    ) is contract.ConnectionProbeAvailability.UNAVAILABLE


@pytest.mark.parametrize("endpoint", (None, "", "not a url", "ftp://localhost/v1"))
def test_connection_probe_availability_rejects_missing_or_invalid_routes(
    endpoint: str | None,
) -> None:
    """Invalid drafts must not expose an action that cannot issue a safe request."""
    assert contract.connection_probe_availability(
        "custom",
        endpoint,
    ) is contract.ConnectionProbeAvailability.UNAVAILABLE
