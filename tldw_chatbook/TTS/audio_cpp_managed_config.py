from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig

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


class _StrictJSONError(ValueError):
    """Internal marker for JSON extensions forbidden by the managed contract."""


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
    except (OSError, RuntimeError, ValueError):
        raise ValueError(diagnostic) from None
    if not path.is_absolute():
        raise ValueError(diagnostic)
    return path


def _validated_binary_path(value: str) -> Path:
    path = _expanded_path(value, _BINARY_DIAGNOSTIC)
    try:
        valid = path.is_file() and os.access(path, os.X_OK)
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise ValueError(_BINARY_DIAGNOSTIC)
    return path


def _read_server_json(value: str) -> tuple[Path, dict[str, Any]]:
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
    return path, parsed


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
    if server_config.get("host") != "127.0.0.1":
        raise ValueError(_SERVER_HOST_DIAGNOSTIC)
    port = server_config.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise ValueError(_SERVER_PORT_DIAGNOSTIC)

    return AudioCppManagedLaunchConfig(
        binary_path=binary_path,
        server_json_path=server_json_path,
        working_directory=server_json_path.parent,
        base_url=f"http://127.0.0.1:{port}",
        startup_timeout_seconds=config.managed_startup_timeout_seconds,
        health_check_interval_seconds=config.managed_health_check_interval_seconds,
        termination_grace_seconds=config.managed_termination_grace_seconds,
    )
