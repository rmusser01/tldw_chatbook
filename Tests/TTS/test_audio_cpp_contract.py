from __future__ import annotations

import json
import math
import struct
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.TTS import audio_cpp_contract
from tldw_chatbook.TTS.audio_cpp_contract import (
    AudioCppContractError,
    AudioCppHealth,
    AudioCppModel,
    Pcm16WavInfo,
    parse_health_response,
    parse_models_response,
    parse_timing_headers,
    parse_voices_response,
    validate_pcm16_wav,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio_cpp_http_v1"
MAX_METADATA_BYTES = 4096
MAX_IDENTIFIER_CHARACTERS = 64
MAX_ITEMS = 16


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _model(**updates: Any) -> dict[str, Any]:
    model = {
        "id": "pocket-tts",
        "object": "model",
        "owned_by": "engine",
        "family": "pocket_tts",
        "task": "tts",
        "mode": "offline",
    }
    model.update(updates)
    return model


def _models_body(*models: Any, **root_updates: Any) -> bytes:
    root: dict[str, Any] = {"object": "list", "data": list(models)}
    root.update(root_updates)
    return _json_bytes(root)


def _parse_surface(surface: str, body: bytes, max_bytes: int) -> object:
    if surface == "health":
        return parse_health_response(
            body,
            max_metadata_bytes=max_bytes,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=MAX_ITEMS,
        )
    if surface == "models":
        return parse_models_response(
            body,
            max_metadata_bytes=max_bytes,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=MAX_ITEMS,
        )
    return parse_voices_response(
        body,
        max_metadata_bytes=max_bytes,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_voices=MAX_ITEMS,
    )


def _surface_body_with_extra_number(surface: str, number: bytes) -> bytes:
    if surface == "health":
        prefix = b'{"status":"ok","backend":"cuda","models":0,"ignored":'
    elif surface == "models":
        prefix = b'{"object":"list","data":[],"ignored":'
    else:
        prefix = b'{"voices":[],"ignored":'
    return prefix + number + b"}"


def _exception_graph(error: BaseException) -> list[BaseException]:
    pending = [error]
    seen: set[int] = set()
    graph: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.append(current)
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return graph


def _assert_unchained_contract_error(
    call: Callable[[], object],
) -> AudioCppContractError:
    with pytest.raises(AudioCppContractError) as captured:
        call()
    error = captured.value
    assert _exception_graph(error) == [error]
    assert error.__context__ is None
    assert error.__cause__ is None
    return error


def _chunk(
    chunk_id: bytes,
    payload: bytes,
    *,
    include_padding: bool = True,
    padding: bytes = b"\x00",
) -> bytes:
    result = chunk_id + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2 and include_padding:
        result += padding
    return result


def _riff_payload(payload: bytes, *, declared_size: int | None = None) -> bytes:
    size = len(payload) + 4 if declared_size is None else declared_size
    return b"RIFF" + struct.pack("<I", size) + b"WAVE" + payload


def _fmt_chunk(
    *,
    format_tag: int = 1,
    channels: int = 1,
    sample_rate: int = 24_000,
    byte_rate: int | None = None,
    block_align: int | None = None,
    bits_per_sample: int = 16,
    payload_suffix: bytes = b"",
) -> bytes:
    resolved_block_align = channels * 2 if block_align is None else block_align
    resolved_byte_rate = (
        sample_rate * resolved_block_align if byte_rate is None else byte_rate
    )
    payload = struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        sample_rate,
        resolved_byte_rate,
        resolved_block_align,
        bits_per_sample,
    )
    return _chunk(b"fmt ", payload + payload_suffix)


def _wav(
    *,
    channels: int = 1,
    sample_rate: int = 24_000,
    data: bytes = b"\x00\x00\x01\x00",
    before_fmt: bytes = b"",
    between: bytes = b"",
    after_data: bytes = b"",
    fmt: bytes | None = None,
) -> bytes:
    return _riff_payload(
        before_fmt
        + (
            _fmt_chunk(channels=channels, sample_rate=sample_rate)
            if fmt is None
            else fmt
        )
        + between
        + _chunk(b"data", data)
        + after_data
    )


def test_pinned_fixtures_capture_reviewed_upstream_contract() -> None:
    provenance = json.loads(_fixture_bytes("provenance.json"))
    assert provenance == {
        "repository": "https://github.com/0xShug0/audio.cpp",
        "commit": "d3d748179e5ace353386fbf17bcaedfacf482d75",
        "reviewed": "2026-07-23",
        "endpoints": [
            "GET /health",
            "GET /v1/models",
            "GET /v1/audio/voices",
            "POST /v1/audio/speech",
        ],
        "source_paths": [
            "app/server/README.md",
            "app/server/runtime.cpp",
            "app/server/http.cpp",
            "app/server/busy_guard.h",
        ],
    }

    health = parse_health_response(
        _fixture_bytes("health.json"),
        max_metadata_bytes=MAX_METADATA_BYTES,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_models=MAX_ITEMS,
    )
    assert health == AudioCppHealth(status="ok", backend="cuda", models=2)

    models = parse_models_response(
        _fixture_bytes("models.json"),
        max_metadata_bytes=MAX_METADATA_BYTES,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_models=MAX_ITEMS,
    )
    assert models == (
        AudioCppModel(
            model_id="pocket-tts",
            family="pocket_tts",
            task="tts",
            mode="offline",
        ),
    )

    voices = parse_voices_response(
        _fixture_bytes("voices.json"),
        max_metadata_bytes=MAX_METADATA_BYTES,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_voices=MAX_ITEMS,
    )
    assert voices == ("alba", "cosette", "marius")

    assert audio_cpp_contract.parse_server_busy_response(
        _fixture_bytes("server_busy.json"),
        max_metadata_bytes=MAX_METADATA_BYTES,
    )


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b'{"error":null}',
        b'{"error":{"message":1,"type":"server_busy"}}',
        b'{"error":{"message":"busy","type":"server_error"}}',
        b'{"error":{"message":"busy","type":"server_busy","type":"server_busy"}}',
        b'{"error":{"message":"busy","type":"server_busy"},"ignored":'
        + b"7" * 129
        + b"}",
    ],
)
def test_server_busy_parser_rejects_malformed_or_extreme_envelopes(
    body: bytes,
) -> None:
    error = _assert_unchained_contract_error(
        lambda: audio_cpp_contract.parse_server_busy_response(
            body,
            max_metadata_bytes=len(body),
        )
    )

    assert error.surface == "server_busy"


def test_server_busy_parser_enforces_bound_before_decoding() -> None:
    body = _fixture_bytes("server_busy.json")

    error = _assert_unchained_contract_error(
        lambda: audio_cpp_contract.parse_server_busy_response(
            body,
            max_metadata_bytes=len(body) - 1,
        )
    )

    assert error.surface == "server_busy"
    assert error.category == "size"


@pytest.mark.parametrize("surface", ["health", "models", "voices"])
@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"\xff",
        b"[]",
        b'"not-an-object"',
        b"null",
        b'{"duplicate":1,"duplicate":2}',
        b'{"constant":NaN}',
        b'{"constant":Infinity}',
        b'{"constant":-Infinity}',
    ],
)
def test_json_surfaces_reject_malformed_or_non_object_documents(
    surface: str,
    body: bytes,
) -> None:
    with pytest.raises(AudioCppContractError):
        _parse_surface(surface, body, MAX_METADATA_BYTES)


@pytest.mark.parametrize("surface", ["health", "models", "voices"])
def test_json_surfaces_enforce_byte_limit_before_decoding(surface: str) -> None:
    body = b"\xff"

    with pytest.raises(AudioCppContractError) as error:
        _parse_surface(surface, body, 0)

    assert error.value.category == "size"


@pytest.mark.parametrize("surface", ["health", "models", "voices"])
def test_json_surfaces_wrap_huge_integer_errors_with_stable_diagnostics(
    surface: str,
) -> None:
    diagnostics: list[str] = []

    for digit_count in (4_501, 5_003):
        body = _surface_body_with_extra_number(surface, b"7" * digit_count)
        with pytest.raises(AudioCppContractError) as error:
            _parse_surface(surface, body, len(body))
        assert error.value.category == "json"
        diagnostics.append(str(error.value))

    assert diagnostics[0] == diagnostics[1]
    assert "4300" not in diagnostics[0]
    assert "4501" not in diagnostics[0]
    assert "5003" not in diagnostics[0]


@pytest.mark.parametrize("surface", ["health", "models", "voices"])
@pytest.mark.parametrize(
    "number",
    [
        b"9" * 129,
        b"1." + b"9" * 129,
        b"1e309",
        b"1e-309",
        b"0e9999",
    ],
)
def test_json_surfaces_reject_bounded_or_nonfinite_numeric_tokens(
    surface: str,
    number: bytes,
) -> None:
    body = _surface_body_with_extra_number(surface, number)

    with pytest.raises(AudioCppContractError) as error:
        _parse_surface(surface, body, len(body))

    assert error.value.category == "json"


@pytest.mark.parametrize("surface", ["health", "models", "voices"])
@pytest.mark.parametrize("number", [b"0", b"-1", b"1.25", b"1e3", b"1e308"])
def test_json_surfaces_accept_reasonable_finite_numbers_in_extra_fields(
    surface: str,
    number: bytes,
) -> None:
    body = _surface_body_with_extra_number(surface, number)

    _parse_surface(surface, body, len(body))


@pytest.mark.parametrize(
    ("case", "body"),
    [
        (
            "utf8",
            b'{"status":"ok","backend":"cuda","models":0,'
            b'"secret":"REMOTE_UTF8_SECRET"}\xff',
        ),
        (
            "json",
            b'{"status":"ok","backend":"cuda","models":0,"secret":"REMOTE_JSON_SECRET"',
        ),
        (
            "integer",
            b'{"status":"ok","backend":"cuda","models":0,'
            b'"secret":"REMOTE_INTEGER_SECRET","ignored":' + b"7" * 5_003 + b"}",
        ),
        (
            "exponent",
            b'{"status":"ok","backend":"cuda","models":0,'
            b'"secret":"REMOTE_EXPONENT_SECRET","ignored":1e9999}',
        ),
        (
            "duplicate",
            b'{"status":"ok","backend":"cuda","models":0,'
            b'"REMOTE_DUPLICATE_SECRET":1,"REMOTE_DUPLICATE_SECRET":2}',
        ),
        (
            "structure",
            b'{"status":"REMOTE_STRUCTURE_SECRET","backend":"cuda","models":0}',
        ),
    ],
    ids=["utf8", "json", "integer", "exponent", "duplicate", "structure"],
)
def test_secret_bearing_contract_errors_have_no_exception_chain(
    case: str,
    body: bytes,
) -> None:
    error = _assert_unchained_contract_error(
        lambda: parse_health_response(
            body,
            max_metadata_bytes=len(body),
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=MAX_ITEMS,
        )
    )

    assert error.surface == "health", case
    assert "REMOTE_" not in str(error)


def test_deeply_nested_json_is_translated_to_unchained_contract_error() -> None:
    depth = sys.getrecursionlimit() * 20
    body = (
        b'{"status":"ok","backend":"cuda","models":0,'
        b'"secret":"REMOTE_DEPTH_SECRET","ignored":'
        + b"[" * depth
        + b"0"
        + b"]" * depth
        + b"}"
    )

    error = _assert_unchained_contract_error(
        lambda: parse_health_response(
            body,
            max_metadata_bytes=len(body),
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=MAX_ITEMS,
        )
    )

    assert error.category == "json"
    assert "REMOTE_DEPTH_SECRET" not in str(error)


def test_wav_structural_error_has_no_exception_chain() -> None:
    body = b"REMOTE_AUDIO_SECRET"

    error = _assert_unchained_contract_error(lambda: validate_pcm16_wav(body))

    assert error.surface == "wav"
    assert "REMOTE_AUDIO_SECRET" not in str(error)


def test_health_accepts_extra_fields_and_boundary_values() -> None:
    body = _json_bytes(
        {
            "status": "ok",
            "backend": "cuda",
            "models": 0,
            "future": {"nested": True},
        }
    )

    result = parse_health_response(
        body,
        max_metadata_bytes=len(body),
        max_identifier_characters=4,
        max_models=0,
    )

    assert result == AudioCppHealth(status="ok", backend="cuda", models=0)
    with pytest.raises(FrozenInstanceError):
        result.backend = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize("missing", ["status", "backend", "models"])
def test_health_requires_each_pinned_field(missing: str) -> None:
    value: dict[str, Any] = {"status": "ok", "backend": "cuda", "models": 2}
    del value[missing]

    with pytest.raises(AudioCppContractError):
        parse_health_response(
            _json_bytes(value),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=2,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "ready"},
        {"status": True},
        {"backend": 3},
        {"models": True},
        {"models": -1},
        {"models": 1.0},
        {"models": "2"},
        {"models": 3},
    ],
)
def test_health_rejects_wrong_or_out_of_range_fields(
    updates: dict[str, Any],
) -> None:
    value: dict[str, Any] = {"status": "ok", "backend": "cuda", "models": 2}
    value.update(updates)

    with pytest.raises(AudioCppContractError):
        parse_health_response(
            _json_bytes(value),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=2,
        )


@pytest.mark.parametrize(
    "backend",
    [
        "",
        " cuda",
        "cuda ",
        "cu\x00da",
        "cu\u200bda",
        "cu\ud800da",
        "cu\ue000da",
        "cu\u0378da",
        "toolong",
    ],
)
def test_health_rejects_unsafe_or_overlong_backend(backend: str) -> None:
    with pytest.raises(AudioCppContractError):
        parse_health_response(
            _json_bytes({"status": "ok", "backend": backend, "models": 0}),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=6,
            max_models=0,
        )


def test_models_filter_tts_case_insensitively_without_mutating_identity() -> None:
    body = _models_body(
        _model(
            id="Opaque.Model/v1",
            family="Pocket_Family",
            task="TTS",
            mode="Offline.Mode",
            future=True,
        ),
        _model(id="asr", family="asr_family", task="ASR"),
        future_root=True,
    )

    result = parse_models_response(
        body,
        max_metadata_bytes=MAX_METADATA_BYTES,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_models=2,
    )

    assert result == (
        AudioCppModel(
            model_id="Opaque.Model/v1",
            family="Pocket_Family",
            task="TTS",
            mode="Offline.Mode",
        ),
    )
    with pytest.raises(FrozenInstanceError):
        result[0].family = "other"  # type: ignore[misc]


def test_models_allow_zero_tts_entries() -> None:
    result = parse_models_response(
        _models_body(_model(task="asr")),
        max_metadata_bytes=MAX_METADATA_BYTES,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_models=1,
    )

    assert result == ()


@pytest.mark.parametrize(
    "body",
    [
        _json_bytes({"object": "collection", "data": []}),
        _json_bytes({"data": []}),
        _json_bytes({"object": "list"}),
        _json_bytes({"object": "list", "data": {}}),
        _models_body("not-an-object"),
    ],
)
def test_models_reject_wrong_root_or_entry_shape(body: bytes) -> None:
    with pytest.raises(AudioCppContractError):
        parse_models_response(
            body,
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", 1),
        ("family", False),
        ("task", None),
        ("mode", []),
        ("object", "not-model"),
        ("owned_by", "not-engine"),
    ],
)
def test_models_reject_wrong_required_fields(field: str, value: Any) -> None:
    with pytest.raises(AudioCppContractError):
        parse_models_response(
            _models_body(_model(**{field: value})),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=1,
        )


@pytest.mark.parametrize(
    "missing", ["id", "family", "task", "mode", "object", "owned_by"]
)
def test_models_require_all_pinned_entry_fields(missing: str) -> None:
    model = _model()
    del model[missing]

    with pytest.raises(AudioCppContractError):
        parse_models_response(
            _models_body(model),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=1,
        )


@pytest.mark.parametrize("field", ["id", "family", "task", "mode"])
@pytest.mark.parametrize(
    "value",
    [
        "",
        " leading",
        "trailing ",
        "unsafe\x00value",
        "unsafe\u200bvalue",
        "unsafe\ud800value",
        "unsafe\ue000value",
        "unsafe\u0378value",
        "123456789",
    ],
)
def test_models_reject_unsafe_whitespace_or_overlong_identifiers(
    field: str,
    value: str,
) -> None:
    with pytest.raises(AudioCppContractError):
        parse_models_response(
            _models_body(_model(**{field: value})),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=8,
            max_models=1,
        )


def test_models_enforce_unfiltered_count_and_reject_duplicate_ids() -> None:
    too_many = _models_body(_model(id="tts"), _model(id="asr", task="asr"))
    with pytest.raises(AudioCppContractError):
        parse_models_response(
            too_many,
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=1,
        )

    duplicate = _models_body(_model(id="same"), _model(id="same", task="asr"))
    with pytest.raises(AudioCppContractError):
        parse_models_response(
            duplicate,
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_models=2,
        )


def test_voices_preserve_opaque_order_and_accept_extra_fields() -> None:
    body = _json_bytes({"voices": ["zeta", "alpha"], "future": 1})

    result = parse_voices_response(
        body,
        max_metadata_bytes=MAX_METADATA_BYTES,
        max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
        max_voices=2,
    )

    assert result == ("zeta", "alpha")


@pytest.mark.parametrize(
    "body",
    [
        _json_bytes({}),
        _json_bytes({"voices": "alba"}),
        _json_bytes({"voices": ["alba", 1]}),
        _json_bytes({"voices": ["same", "same"]}),
        _json_bytes({"voices": ["one", "two", "three"]}),
    ],
)
def test_voices_reject_wrong_shape_duplicates_and_excess_count(body: bytes) -> None:
    with pytest.raises(AudioCppContractError):
        parse_voices_response(
            body,
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
            max_voices=2,
        )


@pytest.mark.parametrize(
    "voice",
    [
        "",
        " alba",
        "alba ",
        "bad\x00voice",
        "bad\u200bvoice",
        "bad\ud800voice",
        "bad\ue000voice",
        "bad\u0378voice",
        "123456789",
    ],
)
def test_voices_reject_unsafe_or_overlong_identifiers(voice: str) -> None:
    with pytest.raises(AudioCppContractError):
        parse_voices_response(
            _json_bytes({"voices": [voice]}),
            max_metadata_bytes=MAX_METADATA_BYTES,
            max_identifier_characters=8,
            max_voices=1,
        )


def test_timing_headers_are_case_insensitive_bounded_and_immutable() -> None:
    result = parse_timing_headers(
        {
            "x-audiocpp-wall-ms": "12.5",
            "X-AUDIOCPP-AUDIO-DURATION-MS": "250",
            "X-AudioCPP-RTF": "0.05",
            "X-Untrusted": "do-not-retain",
        }
    )

    assert result == {
        "wall_ms": 12.5,
        "audio_duration_ms": 250.0,
        "rtf": 0.05,
    }
    assert all(
        type(value) is float and math.isfinite(value) for value in result.values()
    )
    with pytest.raises(TypeError):
        result["wall_ms"] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-number",
        "NaN",
        "Infinity",
        "-Infinity",
        "-0.1",
        " 1",
        "1 ",
        "1e3",
        "1e309",
        "١",
        "123456",
    ],
)
def test_timing_headers_ignore_each_malformed_value(value: str) -> None:
    result = parse_timing_headers(
        {
            "X-AudioCPP-Wall-Ms": value,
            "X-AudioCPP-Audio-Duration-Ms": "2.5",
        },
        max_value_characters=5,
    )

    assert result == {"audio_duration_ms": 2.5}


def test_timing_headers_ignore_missing_unknown_and_non_string_values() -> None:
    result = parse_timing_headers(  # type: ignore[arg-type]
        {
            "X-Unknown": "7",
            "X-AudioCPP-Wall-Ms": None,
            "X-AudioCPP-RTF": True,
        }
    )

    assert result == {}


@pytest.mark.parametrize(
    ("channels", "sample_rate", "data", "expected_frames"),
    [
        (1, 24_000, b"\x00\x00\x01\x00", 2),
        (2, 44_100, b"\x00\x00\x01\x00\x02\x00\x03\x00", 2),
    ],
)
def test_validate_pcm16_wav_returns_authoritative_immutable_info(
    channels: int,
    sample_rate: int,
    data: bytes,
    expected_frames: int,
) -> None:
    result = validate_pcm16_wav(
        _wav(channels=channels, sample_rate=sample_rate, data=data)
    )

    assert result == Pcm16WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        frame_count=expected_frames,
        data_size=len(data),
        byte_rate=sample_rate * channels * 2,
        block_align=channels * 2,
        bits_per_sample=16,
    )
    with pytest.raises(FrozenInstanceError):
        result.channels = 3  # type: ignore[misc]


def test_wav_accepts_complete_odd_ancillary_chunks_without_interpreting_padding() -> (
    None
):
    ancillary = _chunk(b"JUNK", b"abc", padding=b"\xff")

    result = validate_pcm16_wav(_wav(before_fmt=ancillary))

    assert result.frame_count == 2


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"RIFF",
        b"NOPE" + struct.pack("<I", 4) + b"WAVE",
        b"RIFF" + struct.pack("<I", 4) + b"NOPE",
        _riff_payload(b"JUNK"),
        _riff_payload(b"JUNK" + struct.pack("<I", 0xFFFFFFFF)),
        _riff_payload(_chunk(b"JUNK", b"abc", include_padding=False)),
        _riff_payload(_fmt_chunk() + b"data" + struct.pack("<I", 4) + b"\x00\x00"),
    ],
)
def test_wav_rejects_bad_signatures_incomplete_chunks_and_truncation(
    body: bytes,
) -> None:
    with pytest.raises(AudioCppContractError):
        validate_pcm16_wav(body)


def test_wav_requires_exact_riff_declared_size_and_no_trailing_bytes() -> None:
    valid = _wav()
    declared = struct.unpack_from("<I", valid, 4)[0]

    for body in (
        valid[:4] + struct.pack("<I", declared - 1) + valid[8:],
        valid[:4] + struct.pack("<I", declared + 1) + valid[8:],
        valid + b"x",
    ):
        with pytest.raises(AudioCppContractError):
            validate_pcm16_wav(body)


@pytest.mark.parametrize(
    "fmt",
    [
        _fmt_chunk(format_tag=3),
        _chunk(b"fmt ", b"\x00" * 15),
        _fmt_chunk(payload_suffix=b"\x00"),
        _fmt_chunk(channels=0),
        _fmt_chunk(sample_rate=0),
        _fmt_chunk(byte_rate=1),
        _fmt_chunk(block_align=1),
        _fmt_chunk(bits_per_sample=8),
    ],
)
def test_wav_rejects_invalid_pcm16_format(fmt: bytes) -> None:
    with pytest.raises(AudioCppContractError):
        validate_pcm16_wav(_wav(fmt=fmt))


@pytest.mark.parametrize(
    "body",
    [
        _wav(data=b""),
        _wav(data=b"\x00\x00\x00"),
        _riff_payload(_chunk(b"data", b"\x00\x00") + _fmt_chunk()),
        _riff_payload(_fmt_chunk() + _fmt_chunk() + _chunk(b"data", b"\x00\x00")),
        _riff_payload(
            _fmt_chunk() + _chunk(b"data", b"\x00\x00") + _chunk(b"data", b"\x00\x00")
        ),
        _riff_payload(_fmt_chunk()),
        _riff_payload(_chunk(b"data", b"\x00\x00")),
        _riff_payload(_chunk(b"JUNK", b"even")),
    ],
)
def test_wav_rejects_empty_or_misaligned_data_order_duplicates_and_missing_chunks(
    body: bytes,
) -> None:
    with pytest.raises(AudioCppContractError):
        validate_pcm16_wav(body)


def test_validation_diagnostics_are_value_independent_and_never_echo_payloads() -> None:
    secret_one = "REMOTE_SECRET_ONE"
    secret_two = "REMOTE_SECRET_TWO"
    errors: list[AudioCppContractError] = []

    for secret in (secret_one, secret_two):
        invalid_calls = (
            lambda: parse_health_response(
                _json_bytes({"status": secret, "backend": "cuda", "models": 0}),
                max_metadata_bytes=MAX_METADATA_BYTES,
                max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
                max_models=0,
            ),
            lambda: parse_models_response(
                _models_body(_model(id=f"{secret}\x00")),
                max_metadata_bytes=MAX_METADATA_BYTES,
                max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
                max_models=1,
            ),
            lambda: parse_voices_response(
                _json_bytes({"voices": [f"{secret}\x00"]}),
                max_metadata_bytes=MAX_METADATA_BYTES,
                max_identifier_characters=MAX_IDENTIFIER_CHARACTERS,
                max_voices=1,
            ),
            lambda: validate_pcm16_wav(secret.encode("ascii")),
        )
        for call in invalid_calls:
            with pytest.raises(AudioCppContractError) as error:
                call()
            errors.append(error.value)

    diagnostics = [str(error) for error in errors]
    assert diagnostics[:4] == diagnostics[4:]
    assert all(secret_one not in diagnostic for diagnostic in diagnostics)
    assert all(secret_two not in diagnostic for diagnostic in diagnostics)
    assert all(
        error.surface in {"health", "models", "voices", "wav"} for error in errors
    )
