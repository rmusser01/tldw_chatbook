from __future__ import annotations

import json
import os
import platform
import re
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.Utils.path_validation import validate_path_simple

_MAX_SERVER_JSON_BYTES = 1_048_576
_BINARY_DIAGNOSTIC = "audio.cpp managed_binary_path must be an absolute executable file"
_SERVER_PATH_DIAGNOSTIC = (
    "audio.cpp managed_server_json_path must be an absolute readable file"
)
_SERVER_SIZE_DIAGNOSTIC = "audio.cpp server.json must be at most 1048576 bytes"
_SERVER_UTF8_DIAGNOSTIC = "audio.cpp server.json must be UTF-8 JSON"
_SERVER_JSON_DIAGNOSTIC = "audio.cpp server.json must be strict JSON"
_SERVER_OBJECT_DIAGNOSTIC = "audio.cpp server.json must contain one JSON object"
_SERVER_HOST_DIAGNOSTIC = "audio.cpp server.json host must be exactly 127.0.0.1"
_SERVER_PORT_DIAGNOSTIC = (
    "audio.cpp server.json port must be an integer from 1 through 65535"
)
_AUDIO_CPP_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "APPDATA",
        "BLIS_NUM_THREADS",
        "COMSPEC",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "GGML_METAL_PATH_RESOURCES",
        "GGML_VK_VISIBLE_DEVICES",
        "HIP_PATH",
        "HIP_VISIBLE_DEVICES",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOCALAPPDATA",
        "LOGNAME",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "ROCM_PATH",
        "ROCR_VISIBLE_DEVICES",
        "SYSTEMROOT",
        "SystemRoot",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
        "USERPROFILE",
        "VECLIB_MAXIMUM_THREADS",
        "VK_ICD_FILENAMES",
        "VK_LAYER_PATH",
        "WINDIR",
    }
)
_FIXED_PROVIDER_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "DEEPSEEK_API_KEY",
        "ELEVENLABS_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "HUGGINGFACE_API_KEY",
        "MISTRAL_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ZAI_API_KEY",
    }
)
_ENVIRONMENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SECRET_ENV_NAME_FRAGMENTS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "authorization",
    "auth",
)


@dataclass(frozen=True, slots=True)
class AudioCppExpectedModel:
    """Exact generated model evidence retained with a managed launch."""

    model_id: str
    family: str
    task: Literal["tts", "clone"]
    mode: str
    speech_capabilities: tuple[Literal["tts", "clone"], ...]


class AudioCppLaunchArtifact(Protocol):
    """Exact private artifact ownership retained by a generated launch."""

    def validate(self) -> None:
        """Raise a stable error unless the exact artifact is unchanged."""

    def cleanup(self) -> None:
        """Remove only the exact owned artifact and release ownership."""


@dataclass(frozen=True, slots=True)
class AudioCppManagedLaunchConfig:
    """Validated immutable inputs for one managed audio.cpp launch."""

    binary_path: Path
    server_json_path: Path
    working_directory: Path
    base_url: str
    startup_timeout_seconds: float
    health_check_interval_seconds: float
    termination_grace_seconds: float
    expected_models: tuple[AudioCppExpectedModel, ...] = ()
    generated_artifact: AudioCppLaunchArtifact | None = None


class _StrictJSONError(ValueError):
    """Internal marker for JSON extensions forbidden by the managed contract."""


class _AudioCppServerConfig(BaseModel):
    """Strict Chatbook-owned fields from an otherwise server-owned document."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    host: Literal["127.0.0.1"]
    port: Annotated[int, Field(ge=1, le=65_535)]


def collect_provider_credential_environment_names(
    app_config: Mapping[str, Any],
) -> frozenset[str]:
    """Collect fixed and valid configured provider credential variable names.

    Args:
        app_config: Application configuration containing optional ``api_settings``.

    Returns:
        Credential environment names to exclude from the managed child process.
    """
    names = set(_FIXED_PROVIDER_CREDENTIAL_ENV_NAMES)
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return frozenset(names)

    for provider_settings in api_settings.values():
        if not isinstance(provider_settings, Mapping):
            continue
        candidate = provider_settings.get("api_key_env_var")
        if isinstance(candidate, str) and _ENVIRONMENT_NAME_RE.fullmatch(candidate):
            names.add(candidate)
    return frozenset(names)


def build_audio_cpp_child_environment(
    source: Mapping[str, Any],
    *,
    provider_credential_names: AbstractSet[str],
) -> dict[str, str]:
    """Build the minimal non-secret environment for a managed audio.cpp child.

    Args:
        source: Parent-process environment mapping.
        provider_credential_names: Credential variable names to exclude.

    Returns:
        A new environment containing allowlisted string values only.
    """
    credential_names = {
        name.casefold() for name in provider_credential_names if isinstance(name, str)
    }
    child: dict[str, str] = {}
    for name in _AUDIO_CPP_CHILD_ENV_ALLOWLIST:
        folded_name = name.casefold()
        if folded_name in credential_names or any(
            fragment in folded_name for fragment in _SECRET_ENV_NAME_FRAGMENTS
        ):
            continue
        value = source.get(name)
        if isinstance(value, str):
            child[name] = value
    return child


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError
        result[key] = value
    return result


def _reject_non_json_constant(_constant: str) -> NoReturn:
    raise _StrictJSONError


def _expanded_path(value: str, diagnostic: str) -> Path:
    try:
        path = Path(value).expanduser()
        validate_path_simple(path, probe_existing=False)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(diagnostic) from None
    if not path.is_absolute():
        raise ValueError(diagnostic)
    return path


def _validated_binary_path(value: str) -> Path:
    path = _expanded_path(value, _BINARY_DIAGNOSTIC)
    if platform.system().casefold() == "windows":
        from tldw_chatbook.TTS.audio_cpp_guided_launch import _validate_binary

        if _validate_binary(value) is None:
            raise ValueError(_BINARY_DIAGNOSTIC)
        return path
    try:
        valid = path.is_file() and os.access(path, os.X_OK)
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise ValueError(_BINARY_DIAGNOSTIC)
    return path


def _read_server_json(value: str) -> tuple[Path, _AudioCppServerConfig]:
    path = _expanded_path(value, _SERVER_PATH_DIAGNOSTIC)
    try:
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(_SERVER_PATH_DIAGNOSTIC)
        if path.stat().st_size > _MAX_SERVER_JSON_BYTES:
            raise ValueError(_SERVER_SIZE_DIAGNOSTIC)
        with path.open("rb") as stream:
            raw = stream.read(_MAX_SERVER_JSON_BYTES + 1)
    except ValueError:
        raise
    except OSError:
        raise ValueError(_SERVER_PATH_DIAGNOSTIC) from None

    if len(raw) > _MAX_SERVER_JSON_BYTES:
        raise ValueError(_SERVER_SIZE_DIAGNOSTIC)
    try:
        document = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(_SERVER_UTF8_DIAGNOSTIC) from None
    try:
        parsed = json.loads(
            document,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except (ValueError, RecursionError):
        raise ValueError(_SERVER_JSON_DIAGNOSTIC) from None
    if not isinstance(parsed, dict):
        raise ValueError(_SERVER_OBJECT_DIAGNOSTIC)

    invalid_fields: set[str] | None = None
    try:
        validated = _AudioCppServerConfig.model_validate(parsed, strict=True)
    except ValidationError as error:
        invalid_fields = {
            str(item["loc"][0])
            for item in error.errors(include_url=False, include_input=False)
            if item["loc"]
        }
    if invalid_fields is not None:
        if "host" in invalid_fields:
            raise ValueError(_SERVER_HOST_DIAGNOSTIC)
        raise ValueError(_SERVER_PORT_DIAGNOSTIC)
    return path, validated


def validate_audio_cpp_managed_launch(
    config: AudioCppConfig,
) -> AudioCppManagedLaunchConfig:
    """Validate active managed launch files and derive the loopback endpoint.

    Args:
        config: Validated active-mode audio.cpp configuration.

    Returns:
        An immutable side-effect-free launch snapshot.

    Raises:
        ValueError: If Managed mode, either path, or required JSON content is invalid.
    """
    if config.mode != "managed":
        raise ValueError("audio.cpp managed launch requires managed mode")

    binary_path = _validated_binary_path(config.managed_binary_path)
    server_json_path, server_config = _read_server_json(config.managed_server_json_path)
    port = server_config.port

    return AudioCppManagedLaunchConfig(
        binary_path=binary_path,
        server_json_path=server_json_path,
        working_directory=server_json_path.parent,
        base_url=f"http://127.0.0.1:{port}",
        startup_timeout_seconds=config.managed_startup_timeout_seconds,
        health_check_interval_seconds=config.managed_health_check_interval_seconds,
        termination_grace_seconds=config.managed_termination_grace_seconds,
    )
