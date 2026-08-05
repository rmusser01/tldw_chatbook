from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError


DEFAULT_CONFIG = {
    "mode": "external",
    "base_url": "http://127.0.0.1:8080",
    "connect_timeout_seconds": 5.0,
    "synthesis_timeout_seconds": 600.0,
    "max_input_characters": 10_000,
    "max_response_bytes": 128 * 1024 * 1024,
    "max_metadata_bytes": 1024 * 1024,
    "max_catalog_models": 1000,
    "max_voices_per_model": 1000,
    "max_identifier_characters": 256,
}
TIMEOUT_FIELDS = (
    "connect_timeout_seconds",
    "synthesis_timeout_seconds",
)
LIMIT_FIELDS = (
    "max_input_characters",
    "max_response_bytes",
    "max_metadata_bytes",
    "max_catalog_models",
    "max_voices_per_model",
    "max_identifier_characters",
)
URL_DIAGNOSTIC = "audio.cpp base_url must be an absolute HTTP or HTTPS origin"


def _config_api() -> tuple[Any, Any]:
    from tldw_chatbook.TTS.audio_cpp_config import (
        AudioCppConfig,
        project_audio_cpp_config,
    )

    return AudioCppConfig, project_audio_cpp_config


def test_audio_cpp_config_is_a_pydantic_validation_model() -> None:
    AudioCppConfig, _ = _config_api()

    assert issubclass(AudioCppConfig, BaseModel)


def test_defaults_are_immutable_and_bounded() -> None:
    AudioCppConfig, project_audio_cpp_config = _config_api()

    config = project_audio_cpp_config({})

    assert config == AudioCppConfig()
    assert config.to_mapping() == DEFAULT_CONFIG
    with pytest.raises(ValidationError):
        config.base_url = "http://example.test"  # type: ignore[misc]


def test_raw_nested_configuration_has_exact_precedence() -> None:
    _, project_audio_cpp_config = _config_api()
    source = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "audio_cpp": {
                    "base_url": "https://raw.example.test:8443/",
                    "connect_timeout_seconds": 2,
                }
            }
        },
        "APP_TTS_CONFIG": {
            "audio_cpp": {
                "base_url": "https://normalized.example.test",
                "connect_timeout_seconds": 3,
                "synthesis_timeout_seconds": 9,
            }
        },
    }

    config = project_audio_cpp_config(source)

    assert config.base_url == "https://raw.example.test:8443"
    assert config.connect_timeout_seconds == 2.0
    assert config.synthesis_timeout_seconds == 600.0


def test_normalized_configuration_is_used_when_raw_entry_is_absent() -> None:
    _, project_audio_cpp_config = _config_api()
    source = {
        "COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"default_format": "wav"}},
        "APP_TTS_CONFIG": {
            "audio_cpp": {
                "base_url": "http://normalized.example.test:9090",
                "max_catalog_models": 42,
            }
        },
    }

    config = project_audio_cpp_config(source)

    assert config.base_url == "http://normalized.example.test:9090"
    assert config.max_catalog_models == 42


def test_present_raw_entry_must_be_a_mapping() -> None:
    _, project_audio_cpp_config = _config_api()

    with pytest.raises(
        ValueError,
        match=r"^audio\.cpp configuration must be a mapping$",
    ):
        project_audio_cpp_config(
            {
                "COMPREHENSIVE_CONFIG_RAW": {
                    "app_tts": {"audio_cpp": ["not", "a", "mapping"]}
                },
                "APP_TTS_CONFIG": {"audio_cpp": {}},
            }
        )


def test_projection_and_mapping_are_defensive_snapshots() -> None:
    _, project_audio_cpp_config = _config_api()
    raw_config = {
        "base_url": "https://snapshot.example.test",
        "max_input_characters": 123,
    }
    source = {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": raw_config}}}

    config = project_audio_cpp_config(source)
    first_mapping = config.to_mapping()
    raw_config["base_url"] = "https://mutated.example.test"
    raw_config["max_input_characters"] = 999
    first_mapping["base_url"] = "https://mapping-mutated.example.test"

    assert config.base_url == "https://snapshot.example.test"
    assert config.max_input_characters == 123
    assert config.to_mapping()["base_url"] == "https://snapshot.example.test"


def test_environment_variables_do_not_override_external_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project_audio_cpp_config = _config_api()
    monkeypatch.setenv("AUDIO_CPP_MODE", "managed")
    monkeypatch.setenv("AUDIO_CPP_BASE_URL", "https://environment.example.test")
    monkeypatch.setenv("AUDIO_CPP_CONNECT_TIMEOUT_SECONDS", "99")
    monkeypatch.setenv("AUDIO_CPP_AUTHORIZATION", "Bearer secret")

    config = project_audio_cpp_config({})

    assert config.to_mapping() == DEFAULT_CONFIG


def test_projection_does_not_retain_managed_or_authentication_fields() -> None:
    _, project_audio_cpp_config = _config_api()
    source = {
        "APP_TTS_CONFIG": {
            "audio_cpp": {
                "mode": "external",
                "binary_path": "/secret/bin/audiocpp_server",
                "server_config_path": "/secret/server.json",
                "startup_timeout_seconds": 300,
                "shutdown_timeout_seconds": 10,
                "log_ring_lines": 200,
                "headers": {"Authorization": "Bearer secret"},
                "auth_headers": {"X-Token": "secret"},
            }
        }
    }

    config = project_audio_cpp_config(source)

    assert config.to_mapping() == DEFAULT_CONFIG


@pytest.mark.parametrize(
    "base_url",
    (
        "localhost:8080",
        "//localhost:8080",
        "ftp://localhost:8080",
        "http://user:secret@localhost:8080",
        "http://@localhost:8080",
        "http://user:@localhost:8080",
        "http:///",
        "http://localhost:not-a-port",
        "http://localhost:65536",
        "http://localhost:",
        "http://localhost:0",
        "http://localhost:8080/v1",
        "http://localhost:8080/?token=secret",
        "http://localhost:8080/#secret",
        " http://localhost:8080",
        "http://localhost:8080 ",
        "http://local\nhost:8080",
        "http://[::1",
    ),
)
def test_invalid_url_categories_are_rejected_with_one_safe_diagnostic(
    base_url: str,
) -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"base_url": base_url})

    assert str(raised.value) == URL_DIAGNOSTIC
    assert base_url not in str(raised.value)


@pytest.mark.parametrize(
    ("base_url", "canonical_origin"),
    (
        ("http://localhost", "http://localhost"),
        ("http://127.0.0.1:8080/", "http://127.0.0.1:8080"),
        ("HTTP://EXAMPLE.COM", "http://example.com"),
        ("http://example.com:80/", "http://example.com"),
        ("https://EXAMPLE.COM:443/", "https://example.com"),
        ("https://EXAMPLE.COM:444/", "https://example.com:444"),
        ("http://example.com.", "http://example.com."),
        (
            "http://bücher.example/",
            "http://xn--bcher-kva.example",
        ),
        (
            "HTTP://BÜCHER.EXAMPLE.",
            "http://xn--bcher-kva.example.",
        ),
        (
            "HTTP://[2001:0DB8:0:0:0:0:0:1]:80/",
            "http://[2001:db8::1]",
        ),
        ("http://[::1]:8080", "http://[::1]:8080"),
    ),
)
def test_http_and_https_origins_are_stored_canonically(
    base_url: str,
    canonical_origin: str,
) -> None:
    AudioCppConfig, _ = _config_api()

    config = AudioCppConfig.from_mapping({"base_url": base_url})

    assert config.base_url == canonical_origin


def test_semantically_equivalent_origins_produce_equal_configurations() -> None:
    AudioCppConfig, _ = _config_api()
    spellings = (
        "HTTP://EXAMPLE.COM",
        "http://example.com",
        "http://example.com:80/",
    )

    configurations = tuple(
        AudioCppConfig.from_mapping({"base_url": spelling}) for spelling in spellings
    )

    assert configurations[1:] == configurations[:-1]
    assert configurations[0].base_url == "http://example.com"


def test_rooted_and_dotless_dns_origins_remain_distinct() -> None:
    AudioCppConfig, _ = _config_api()

    rooted = AudioCppConfig.from_mapping({"base_url": "HTTP://EXAMPLE.COM."})
    dotless = AudioCppConfig.from_mapping({"base_url": "http://example.com"})

    assert rooted.base_url == "http://example.com."
    assert rooted != dotless


@pytest.mark.parametrize(
    "base_url",
    (
        "http://exa\u200bmple.example",
        "http://exa\u2066mple.example",
        "http://exa\ud800mple.example",
        "http://exa\u0378mple.example",
        "http://exa\ue000mple.example",
        "http://0127.0.0.1",
        "http://999.1.1.1",
        "http://[v1.fe80]",
        "http://example.com..",
        "http://example..com",
    ),
)
def test_client_incompatible_host_forms_are_rejected_safely(
    base_url: str,
) -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"base_url": base_url})

    assert str(raised.value) == URL_DIAGNOSTIC
    assert base_url not in str(raised.value)


@pytest.mark.parametrize(
    "base_url",
    (
        "http://example.com:",
        "http://example.com:+80",
        "http://example.com:-80",
        "http://example.com:1_0",
        "http://example.com:１２",
        "http://example.com:\t80",
    ),
)
def test_raw_port_syntax_requires_non_empty_ascii_decimal_digits(
    base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.audio_cpp_config as config_module

    AudioCppConfig, _ = _config_api()
    netloc = base_url.removeprefix("http://")
    monkeypatch.setattr(
        config_module,
        "urlsplit",
        lambda _value: SimpleNamespace(
            scheme="http",
            netloc=netloc,
            path="",
            query="",
            fragment="",
            hostname="example.com",
            port=80,
        ),
    )

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"base_url": base_url})

    assert str(raised.value) == URL_DIAGNOSTIC
    assert base_url not in str(raised.value)


@pytest.mark.parametrize("mode", ("managed", "EXTERNAL", "", None, 1))
def test_mode_must_be_exactly_external(mode: object) -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"mode": mode})

    assert str(raised.value) == "audio.cpp mode must be external"
    if str(mode):
        assert str(mode) not in str(raised.value)


@pytest.mark.parametrize("field", TIMEOUT_FIELDS + LIMIT_FIELDS)
@pytest.mark.parametrize("value", (True, False))
def test_all_numeric_settings_reject_booleans(
    field: str,
    value: bool,
) -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError):
        AudioCppConfig.from_mapping({field: value})


@pytest.mark.parametrize("field", TIMEOUT_FIELDS)
@pytest.mark.parametrize(
    "value",
    (0, -1, float("inf"), float("-inf"), float("nan"), "5", None),
)
def test_timeouts_require_finite_positive_real_numbers(
    field: str,
    value: object,
) -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({field: value})

    assert str(raised.value) == (f"audio.cpp {field} must be a finite positive number")


@pytest.mark.parametrize("field", LIMIT_FIELDS)
@pytest.mark.parametrize(
    "value",
    (0, -1, 1.5, float("inf"), "1", None),
)
def test_limits_require_positive_integers(
    field: str,
    value: object,
) -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({field: value})

    assert str(raised.value) == (f"audio.cpp {field} must be a positive integer")


def test_diagnostics_never_echo_submitted_urls_or_secrets() -> None:
    AudioCppConfig, _ = _config_api()
    submitted = "https://user:super-secret@example.test/private?token=hidden"

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"base_url": submitted})

    diagnostic = str(raised.value)
    assert diagnostic == URL_DIAGNOSTIC
    assert submitted not in diagnostic
    assert "super-secret" not in diagnostic
    assert "hidden" not in diagnostic


def test_url_parser_failures_do_not_retain_an_unsafe_exception_cause() -> None:
    AudioCppConfig, _ = _config_api()

    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"base_url": "https://[super-secret.example.test"})

    assert str(raised.value) == URL_DIAGNOSTIC
    assert raised.value.__cause__ is None


def test_mapping_round_trip_revalidates_and_ignores_unapproved_fields() -> None:
    AudioCppConfig, _ = _config_api()
    original = AudioCppConfig.from_mapping(
        {
            "mode": "external",
            "base_url": "https://round-trip.example.test:8443",
            "connect_timeout_seconds": 1,
            "synthesis_timeout_seconds": 7.5,
            "max_input_characters": 101,
            "max_response_bytes": 102,
            "max_metadata_bytes": 103,
            "max_catalog_models": 104,
            "max_voices_per_model": 105,
            "max_identifier_characters": 106,
            "binary_path": "/must/not/be/retained",
            "headers": {"Authorization": "must-not-be-retained"},
        }
    )

    mapping = original.to_mapping()
    reconstructed = AudioCppConfig.from_mapping(mapping)

    assert reconstructed == original
    assert mapping == {
        "mode": "external",
        "base_url": "https://round-trip.example.test:8443",
        "connect_timeout_seconds": 1.0,
        "synthesis_timeout_seconds": 7.5,
        "max_input_characters": 101,
        "max_response_bytes": 102,
        "max_metadata_bytes": 103,
        "max_catalog_models": 104,
        "max_voices_per_model": 105,
        "max_identifier_characters": 106,
    }
