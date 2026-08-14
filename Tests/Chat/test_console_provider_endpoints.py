from tldw_chatbook.Chat.console_provider_endpoints import (
    effective_provider_endpoint,
    first_configured_endpoint,
    generic_endpoint_differs,
    normalize_generic_endpoint_for_compare,
    safe_endpoint_display,
)
from tldw_chatbook.Chat.local_server_discovery import endpoint_display


def test_safe_endpoint_display_redacts_schemeless_credentials_and_query_tokens() -> (
    None
):
    display = safe_endpoint_display(
        "user:secret@127.0.0.1:9999/v1?token=unit-test-token"
    )

    assert display == "127.0.0.1:9999/v1"
    assert "user" not in display
    assert "secret" not in display
    assert "token" not in display
    assert "unit-test-token" not in display


def test_safe_endpoint_display_strips_query_secret_from_url() -> None:
    display = safe_endpoint_display(
        "https://api.example.test/v1?api_key=unit-test-token"
    )

    assert display == "https://api.example.test/v1"
    assert "api_key" not in display
    assert "unit-test-token" not in display


def test_safe_endpoint_display_does_not_return_malformed_secret_input() -> None:
    display = safe_endpoint_display("http://[::1?api_key=unit-test-token")

    assert display == "invalid endpoint"
    assert "unit-test-token" not in display


def test_safe_endpoint_display_never_echoes_contract_invalid_paths() -> None:
    rejected_values = (
        ("https://example.test/%ZZ-secret", "%ZZ-secret"),
        ("https://example.test/proxy//secret", "proxy//secret"),
        ("https://example.test/path\u202esecret", "\u202e"),
        ("https://example.test/path\ud800secret", "\ud800"),
    )

    for entered, rejected_text in rejected_values:
        display = safe_endpoint_display(entered)

        assert display == "invalid endpoint"
        assert rejected_text not in display


def test_normalize_generic_endpoint_for_compare_handles_schemeless_credentials() -> (
    None
):
    normalized = normalize_generic_endpoint_for_compare(
        "user:secret@127.0.0.1:9999/v1?token=unit-test-token"
    )

    assert normalized == "127.0.0.1:9999/v1"


def test_safe_endpoint_display_rejects_schemeless_single_label_token() -> None:
    display = safe_endpoint_display("unit-test-key")

    assert display == "invalid endpoint"
    assert "unit-test-key" not in display


def test_safe_endpoint_display_rejects_single_label_token_with_scheme() -> None:
    display = safe_endpoint_display("http://unit-test-key")

    assert display == "invalid endpoint"
    assert "unit-test-key" not in display


def test_normalize_generic_endpoint_for_compare_rejects_single_label_token() -> None:
    normalized = normalize_generic_endpoint_for_compare("unit-test-key")

    assert normalized == "invalid endpoint"
    assert "unit-test-key" not in normalized


def test_safe_endpoint_display_allows_schemeless_single_label_host_with_port() -> None:
    display = safe_endpoint_display("local-service:9090/v1")

    assert display == "local-service:9090/v1"


def test_first_configured_endpoint_accepts_api_base_url_alias() -> None:
    endpoint = first_configured_endpoint(
        {
            "api_url": "http://localhost:8080/completion",
            "api_base_url": "http://127.0.0.1:9099/v1",
        }
    )

    assert endpoint == "http://127.0.0.1:9099/v1"


def test_generic_endpoint_differs_accepts_api_base_url_alias() -> None:
    differs = generic_endpoint_differs(
        "http://127.0.0.1:9099/v1",
        {"api_base_url": "http://127.0.0.1:9099/v1"},
    )

    assert differs is False


def test_effective_custom_endpoint_is_the_chat_url_actually_used() -> None:
    endpoint = effective_provider_endpoint(
        "custom",
        "https://api.example.test/proxy/v1",
        {},
    )

    assert endpoint == "https://api.example.test/proxy/v1/chat/completions"


def test_local_discovery_endpoint_display_never_echoes_malformed_input() -> None:
    entered = "http://[::1?api_key=unit-test-token"

    assert endpoint_display(entered) == "invalid endpoint"
    assert "unit-test-token" not in endpoint_display(entered)
