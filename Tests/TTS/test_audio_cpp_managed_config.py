from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from tldw_chatbook.TTS import audio_cpp_managed_config as managed_config_module
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_managed_config import (
    AudioCppManagedLaunchConfig,
    validate_audio_cpp_managed_launch,
)


EXTERNAL_DEFAULT_MAPPING: dict[str, str | float | int] = {
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

COMMON_MAPPING: dict[str, float | int] = {
    "connect_timeout_seconds": 2.5,
    "synthesis_timeout_seconds": 45.0,
    "max_input_characters": 101,
    "max_response_bytes": 102,
    "max_metadata_bytes": 103,
    "max_catalog_models": 104,
    "max_voices_per_model": 105,
    "max_identifier_characters": 106,
}

EXPECTED_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "HOME",
        "USER",
        "LOGNAME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "HIP_PATH",
        "ROCM_PATH",
        "VK_ICD_FILENAMES",
        "VK_LAYER_PATH",
        "GGML_METAL_PATH_RESOURCES",
        "GGML_VK_VISIBLE_DEVICES",
    }
)


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _write_server_json(
    path: Path,
    *,
    host: object = "127.0.0.1",
    port: object = 8080,
) -> Path:
    path.write_text(json.dumps({"host": host, "port": port}), encoding="utf-8")
    return path


def _launch_config(
    binary_path: Path | str,
    server_json_path: Path | str,
) -> AudioCppConfig:
    return AudioCppConfig.from_mapping(
        {
            "mode": "managed",
            "managed_binary_path": str(binary_path),
            "managed_server_json_path": str(server_json_path),
        }
    )


def _validate_launch(config: AudioCppConfig) -> AudioCppManagedLaunchConfig:
    return validate_audio_cpp_managed_launch(config)


def test_missing_mode_projects_the_existing_external_mapping() -> None:
    config = AudioCppConfig.from_mapping({})

    assert config.to_mapping() == EXTERNAL_DEFAULT_MAPPING
    assert not any(key.startswith("managed_") for key in config.to_mapping())


def test_external_projection_ignores_malformed_dormant_managed_fields() -> None:
    config = AudioCppConfig.from_mapping(
        {
            "mode": "external",
            "base_url": "https://external.example.test",
            "managed_binary_path": {"not": "a path"},
            "managed_server_json_path": ["not", "a", "path"],
            "managed_startup_timeout_seconds": False,
            "managed_health_check_interval_seconds": float("nan"),
            "managed_termination_grace_seconds": "five",
        }
    )

    assert config.to_mapping() == {
        **EXTERNAL_DEFAULT_MAPPING,
        "base_url": "https://external.example.test",
    }


def test_managed_projection_ignores_malformed_dormant_external_origin() -> None:
    config = AudioCppConfig.from_mapping(
        {
            "mode": "managed",
            "base_url": "contains secret and is not an origin",
            "managed_binary_path": "/opt/homebrew/bin/audiocpp_server",
            "managed_server_json_path": "/srv/audio.cpp/server.json",
        }
    )

    assert config.mode == "managed"
    assert "base_url" not in config.to_mapping()


def test_to_mapping_contains_only_active_mode_fields_and_common_limits() -> None:
    external = AudioCppConfig.from_mapping(
        {
            "mode": "external",
            "base_url": "http://127.0.0.1:18080",
            **COMMON_MAPPING,
            "managed_binary_path": "/ignored/bin",
            "managed_server_json_path": "/ignored/server.json",
            "unknown": "ignored",
        }
    )
    managed = AudioCppConfig.from_mapping(
        {
            "mode": "managed",
            "base_url": "not-an-origin",
            "managed_binary_path": "/approved/bin/audiocpp_server",
            "managed_server_json_path": "/approved/server.json",
            "managed_startup_timeout_seconds": 7,
            "managed_health_check_interval_seconds": 8,
            "managed_termination_grace_seconds": 9,
            **COMMON_MAPPING,
            "unknown": "ignored",
        }
    )

    assert external.to_mapping() == {
        "mode": "external",
        "base_url": "http://127.0.0.1:18080",
        **COMMON_MAPPING,
    }
    assert managed.to_mapping() == {
        "mode": "managed",
        "managed_binary_path": "/approved/bin/audiocpp_server",
        "managed_server_json_path": "/approved/server.json",
        "managed_startup_timeout_seconds": 7.0,
        "managed_health_check_interval_seconds": 8.0,
        "managed_termination_grace_seconds": 9.0,
        **COMMON_MAPPING,
    }


def test_managed_timing_defaults_and_finite_bounds_are_exact() -> None:
    defaults = AudioCppConfig.from_mapping({"mode": "managed"})

    assert defaults.managed_startup_timeout_seconds == 30.0
    assert defaults.managed_health_check_interval_seconds == 10.0
    assert defaults.managed_termination_grace_seconds == 5.0

    accepted = AudioCppConfig.from_mapping(
        {
            "mode": "managed",
            "managed_startup_timeout_seconds": 300,
            "managed_health_check_interval_seconds": 2,
            "managed_termination_grace_seconds": 0.1,
        }
    )

    assert accepted.managed_startup_timeout_seconds == 300.0
    assert accepted.managed_health_check_interval_seconds == 2.0
    assert accepted.managed_termination_grace_seconds == 0.1

    rejected = (
        ("managed_startup_timeout_seconds", 0.999),
        ("managed_startup_timeout_seconds", 300.001),
        ("managed_health_check_interval_seconds", 1.999),
        ("managed_health_check_interval_seconds", 300.001),
        ("managed_termination_grace_seconds", 0.099),
        ("managed_termination_grace_seconds", 60.001),
    )
    for field_name, value in rejected:
        with pytest.raises(ValueError):
            AudioCppConfig.from_mapping({"mode": "managed", field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("managed_startup_timeout_seconds", True),
        ("managed_startup_timeout_seconds", float("nan")),
        ("managed_startup_timeout_seconds", float("inf")),
        ("managed_health_check_interval_seconds", False),
        ("managed_health_check_interval_seconds", float("-inf")),
        ("managed_termination_grace_seconds", True),
        ("managed_termination_grace_seconds", float("nan")),
    ),
)
def test_managed_timing_rejects_booleans_nan_and_infinities(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError) as raised:
        AudioCppConfig.from_mapping({"mode": "managed", field_name: value})

    diagnostic = str(raised.value)
    assert field_name in diagnostic


def test_managed_binary_requires_absolute_executable_regular_file(
    tmp_path: Path,
) -> None:
    server_json = _write_server_json(tmp_path / "server.json")
    directory = tmp_path / "server-dir"
    directory.mkdir()
    not_executable = tmp_path / "not-executable"
    not_executable.write_text("not executable", encoding="utf-8")

    candidates = (
        "",
        "relative/audiocpp_server",
        tmp_path / "missing",
        directory,
        not_executable,
    )
    for candidate in candidates:
        with pytest.raises(ValueError) as raised:
            _validate_launch(_launch_config(candidate, server_json))

        diagnostic = str(raised.value)
        assert diagnostic == (
            "audio.cpp managed_binary_path must be an absolute executable file"
        )
        if str(candidate):
            assert str(candidate) not in diagnostic


def test_managed_binary_preserves_an_approved_symlink_path(tmp_path: Path) -> None:
    binary_target = _write_executable(tmp_path / "audiocpp_server-v1")
    selected_link = tmp_path / "audiocpp_server"
    selected_link.symlink_to(binary_target)
    server_json = _write_server_json(tmp_path / "server.json")

    launch = _validate_launch(_launch_config(selected_link, server_json))

    assert launch.binary_path == selected_link
    assert launch.binary_path != binary_target


def test_windows_managed_binary_uses_the_shared_pe_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as guided_launch

    binary = tmp_path / "audiocpp_server.exe"
    binary.write_bytes(b"bounded-pe-fixture")
    server_json = _write_server_json(tmp_path / "server.json")
    calls: list[str] = []
    monkeypatch.setattr(managed_config_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        guided_launch,
        "_validate_binary",
        lambda value: calls.append(value) or binary,
    )

    launch = _validate_launch(_launch_config(binary, server_json))

    assert launch.binary_path == binary
    assert calls == [str(binary)]


def test_server_json_requires_readable_regular_utf8_object(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"

    invalid_paths: list[Path | str] = [
        "relative/server.json",
        tmp_path / "missing",
        f"{tmp_path}/secret\x00server.json",
    ]
    directory = tmp_path / "config-dir"
    directory.mkdir()
    invalid_paths.append(directory)
    for candidate in invalid_paths:
        with pytest.raises(ValueError) as raised:
            _validate_launch(_launch_config(binary, candidate))
        assert str(raised.value) == (
            "audio.cpp managed_server_json_path must be an absolute readable file"
        )

    server_json.write_bytes(b"\xff")
    with pytest.raises(
        ValueError, match=r"^audio\.cpp server\.json must be UTF-8 JSON$"
    ):
        _validate_launch(_launch_config(binary, server_json))

    server_json.write_text("[]", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"^audio\.cpp server\.json must contain one JSON object$",
    ):
        _validate_launch(_launch_config(binary, server_json))

    _write_server_json(server_json)
    server_json.chmod(0)
    try:
        if not os.access(server_json, os.R_OK):
            with pytest.raises(ValueError) as raised:
                _validate_launch(_launch_config(binary, server_json))
            assert str(raised.value) == (
                "audio.cpp managed_server_json_path must be an absolute readable file"
            )
    finally:
        server_json.chmod(0o600)


def test_server_json_path_uses_the_central_path_safety_policy(
    tmp_path: Path,
) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    unsafe_directory = tmp_path / "configs;private"
    unsafe_directory.mkdir()
    server_json = _write_server_json(unsafe_directory / "server.json")

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == (
        "audio.cpp managed_server_json_path must be an absolute readable file"
    )
    assert str(server_json) not in str(raised.value)


def test_server_json_owned_fields_have_a_strict_pydantic_boundary() -> None:
    schema_type = getattr(managed_config_module, "_AudioCppServerConfig", None)

    assert isinstance(schema_type, type), "managed server schema is missing"
    assert issubclass(schema_type, BaseModel)
    assert schema_type.model_config["extra"] == "allow"
    assert schema_type.model_config["strict"] is True

    validated = schema_type.model_validate(
        {
            "host": "127.0.0.1",
            "port": 8080,
            "models": [{"id": "pocket-tts", "path": "models/model.gguf"}],
        },
        strict=True,
    )
    assert validated.host == "127.0.0.1"
    assert validated.port == 8080
    assert validated.model_extra == {
        "models": [{"id": "pocket-tts", "path": "models/model.gguf"}]
    }

    with pytest.raises(ValidationError):
        schema_type.model_validate(
            {"host": "127.0.0.1", "port": "8080"},
            strict=True,
        )


def test_server_json_rejects_more_than_one_mib_before_parsing(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"
    server_json.write_bytes(b"{" + (b" " * 1_048_576))

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == "audio.cpp server.json must be at most 1048576 bytes"


def test_server_json_rejects_duplicate_keys_at_every_depth(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"
    server_json.write_text(
        '{"host":"127.0.0.1","port":8080,"nested":{"value":1,"value":2}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == "audio.cpp server.json must be strict JSON"


def test_server_json_normalizes_parser_depth_failures(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"
    server_json.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == "audio.cpp server.json must be strict JSON"
    assert raised.value.__cause__ is None


def test_server_json_normalizes_numeric_parser_limits(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"
    server_json.write_text(
        '{"host":"127.0.0.1","port":' + "1" * 5_000 + "}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == "audio.cpp server.json must be strict JSON"
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_server_json_rejects_non_json_numeric_constants_at_every_depth(
    constant: str,
    tmp_path: Path,
) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"
    server_json.write_text(
        (f'{{"host":"127.0.0.1","port":8080,"nested":{{"value":{constant}}}}}'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == "audio.cpp server.json must be strict JSON"
    assert constant not in str(raised.value)


@pytest.mark.parametrize(
    "host",
    ["localhost", "::1", "0.0.0.0", "127.0.0.2", "example.test"],
)
def test_server_json_requires_exact_ipv4_loopback(
    host: str,
    tmp_path: Path,
) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = _write_server_json(tmp_path / "server.json", host=host)

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == ("audio.cpp server.json host must be exactly 127.0.0.1")
    assert host not in str(raised.value)


@pytest.mark.parametrize("port", [None, True, False, 0, 65536, "8080", 3.5])
def test_server_json_requires_explicit_integer_port(
    port: object,
    tmp_path: Path,
) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = _write_server_json(tmp_path / "server.json", port=port)

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    assert str(raised.value) == (
        "audio.cpp server.json port must be an integer from 1 through 65535"
    )


@pytest.mark.parametrize("missing_field", ["host", "port"])
def test_server_json_requires_explicit_host_and_port(
    missing_field: str,
    tmp_path: Path,
) -> None:
    binary = _write_executable(tmp_path / "audiocpp_server")
    server_json = tmp_path / "server.json"
    payload = {"host": "127.0.0.1", "port": 8080}
    del payload[missing_field]
    server_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        _validate_launch(_launch_config(binary, server_json))

    expected = (
        "audio.cpp server.json host must be exactly 127.0.0.1"
        if missing_field == "host"
        else "audio.cpp server.json port must be an integer from 1 through 65535"
    )
    assert str(raised.value) == expected


def test_launch_snapshot_uses_json_parent_as_cwd_and_derives_origin(
    tmp_path: Path,
) -> None:
    binary = _write_executable(tmp_path / "bin" / "audiocpp_server")
    config_directory = tmp_path / "runtime"
    config_directory.mkdir()
    server_json = _write_server_json(
        config_directory / "server.json",
        port=19_876,
    )
    config = AudioCppConfig.from_mapping(
        {
            "mode": "managed",
            "managed_binary_path": str(binary),
            "managed_server_json_path": str(server_json),
            "managed_startup_timeout_seconds": 31,
            "managed_health_check_interval_seconds": 11,
            "managed_termination_grace_seconds": 6,
        }
    )

    launch = _validate_launch(config)

    assert launch.binary_path == binary
    assert launch.server_json_path == server_json
    assert launch.working_directory == config_directory
    assert launch.base_url == "http://127.0.0.1:19876"
    assert launch.startup_timeout_seconds == 31.0
    assert launch.health_check_interval_seconds == 11.0
    assert launch.termination_grace_seconds == 6.0


def test_credential_inventory_includes_fixed_and_configured_provider_names() -> None:
    app_config = {
        "api_settings": {
            "custom": {"api_key_env_var": "CUSTOM_PROVIDER_CREDENTIAL"},
            "blank": {"api_key_env_var": "   "},
            "digit": {"api_key_env_var": "9INVALID"},
            "hyphen": {"api_key_env_var": "INVALID-NAME"},
            "dot": {"api_key_env_var": "INVALID.NAME"},
            "non_string": {"api_key_env_var": 123},
            "not_a_mapping": "ignored",
        }
    }

    inventory = managed_config_module.collect_provider_credential_environment_names(
        app_config
    )

    assert {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "HUGGINGFACE_API_KEY",
        "MISTRAL_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENROUTER_API_KEY",
        "ZAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "CUSTOM_PROVIDER_CREDENTIAL",
    } <= inventory
    assert inventory.isdisjoint(
        {"", "   ", "9INVALID", "INVALID-NAME", "INVALID.NAME", "123"}
    )


def test_child_environment_copies_only_the_exact_allowlist() -> None:
    source: dict[str, Any] = {
        name: f"safe-value-{index}"
        for index, name in enumerate(sorted(EXPECTED_CHILD_ENV_ALLOWLIST))
    }
    source.update(
        {
            "APPLICATION_INTERNAL_STATE": "private-state",
            "OPENAI_API_KEY": "synthetic-openai-secret",
            "HTTP_PROXY": "http://private-proxy.invalid",
            "path": "lowercase-path-is-not-the-allowlisted-name",
            "LC_TIME": "not-explicitly-allowed",
        }
    )
    source["TEMP"] = 123

    child = managed_config_module.build_audio_cpp_child_environment(
        source,
        provider_credential_names=frozenset(),
    )

    assert child == {
        name: source[name]
        for name in EXPECTED_CHILD_ENV_ALLOWLIST
        if isinstance(source[name], str)
    }


def test_child_environment_drops_casefolded_credential_collisions() -> None:
    child = managed_config_module.build_audio_cpp_child_environment(
        {"PATH": "/safe/bin", "LANG": "en_US.UTF-8"},
        provider_credential_names=frozenset({"path"}),
    )

    assert child == {"LANG": "en_US.UTF-8"}


@pytest.mark.parametrize(
    "secretish_name",
    [
        "FUTURE_API_KEY_SOCKET",
        "FUTURE_APIKEY_SOCKET",
        "FUTURE_TOKEN_SOCKET",
        "FUTURE_SECRET_SOCKET",
        "FUTURE_PASSWORD_SOCKET",
        "FUTURE_CREDENTIAL_SOCKET",
        "FUTURE_AUTHORIZATION_SOCKET",
        "FUTURE_AUTH_SOCKET",
    ],
)
def test_child_environment_drops_secretish_names_even_if_allowlisted(
    secretish_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        managed_config_module,
        "_AUDIO_CPP_CHILD_ENV_ALLOWLIST",
        frozenset({secretish_name, "LANG"}),
        raising=False,
    )

    child = managed_config_module.build_audio_cpp_child_environment(
        {secretish_name: "synthetic-secret", "LANG": "C"},
        provider_credential_names=frozenset(),
    )

    assert child == {"LANG": "C"}
