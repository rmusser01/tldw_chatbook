"""Tests for the dependency-free batch speech-to-text routing policy."""

import pytest

from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
    BatchSTTRoutingError,
    resolve_batch_stt_route,
)


def test_default_english_uses_faster_whisper_while_promotion_gate_is_closed() -> None:
    route = resolve_batch_stt_route(provider="default", language="en")

    assert route.provider == "faster-whisper"
    assert route.reason == "parakeet_promotion_gate_closed"
    assert route.precision == "int8"
    assert route.local_files_only is True


@pytest.mark.parametrize(
    ("language", "target_language", "provider", "model"),
    [
        ("en", None, "parakeet-onnx", PARAKEET_V2_MODEL),
        ("de", None, "parakeet-onnx", PARAKEET_V3_MODEL),
        ("auto", None, "faster-whisper", None),
        ("ja", None, "faster-whisper", None),
        ("en", "fr", "faster-whisper", None),
    ],
)
def test_enabled_default_routes_by_language_and_task(
    language: str,
    target_language: str | None,
    provider: str,
    model: str | None,
) -> None:
    route = resolve_batch_stt_route(
        provider="default",
        language=language,
        target_language=target_language,
        parakeet_defaults_enabled=True,
    )

    assert route.provider == provider
    assert route.model == model


@pytest.mark.parametrize("language", [None, "", "en", "EN"])
def test_exact_parakeet_defaults_english_inputs_to_v2(language: str | None) -> None:
    route = resolve_batch_stt_route(provider="parakeet-onnx", language=language)

    assert route.provider == "parakeet-onnx"
    assert route.model == PARAKEET_V2_MODEL
    assert route.requested_language == "en"


@pytest.mark.parametrize("language", ["de", "es", "fr", "uk"])
def test_exact_parakeet_routes_supported_non_english_to_v3(language: str) -> None:
    route = resolve_batch_stt_route(provider="parakeet-onnx", language=language)

    assert route.model == PARAKEET_V3_MODEL
    assert route.requested_language == language


@pytest.mark.parametrize(
    ("language", "target_language"),
    [("auto", None), ("ja", None), ("en", "fr")],
)
def test_exact_parakeet_rejects_unsupported_or_translation_requests(
    language: str,
    target_language: str | None,
) -> None:
    with pytest.raises(BatchSTTRoutingError, match="Retry with faster-whisper"):
        resolve_batch_stt_route(
            provider="parakeet-onnx",
            language=language,
            target_language=target_language,
        )


def test_exact_faster_whisper_retains_requested_language_and_task() -> None:
    route = resolve_batch_stt_route(
        provider="faster-whisper",
        language="ja",
        target_language="en",
    )

    assert route.provider == "faster-whisper"
    assert route.requested_language == "ja"
    assert route.target_language == "en"


@pytest.mark.parametrize(
    "provider",
    [
        "parakeet",
        "faster",
        "faster-whisper-large",
        " PARAKEET-ONNX ",
        "Faster-Whisper",
    ],
)
def test_unknown_providers_are_rejected_without_prefix_matching(provider: str) -> None:
    with pytest.raises(BatchSTTRoutingError):
        resolve_batch_stt_route(provider=provider, language="en")


def test_language_codes_are_normalized_and_missing_language_defaults_to_english() -> None:
    route = resolve_batch_stt_route(
        provider="parakeet-onnx",
        language=" DE ",
    )
    missing = resolve_batch_stt_route(provider="faster-whisper", language=None)

    assert route.requested_language == "de"
    assert missing.requested_language == "en"
