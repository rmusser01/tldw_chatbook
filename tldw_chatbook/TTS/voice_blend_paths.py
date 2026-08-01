"""Profile-owned persistence for UI Kokoro voice blends."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tldw_chatbook.config import (
    application_owned_config_directory,
    get_cli_config_path,
)
from tldw_chatbook.Utils.private_paths import (
    PrivatePathResult,
    atomic_private_write_text,
)


def kokoro_ui_blend_file() -> Path:
    """Return the active profile's UI Kokoro blend file."""
    return get_cli_config_path().parent / "kokoro_voice_blends.json"


def default_kokoro_backend_blend_directory() -> Path:
    """Return the active profile's default Kokoro backend blend directory."""
    return get_cli_config_path().parent / "kokoro_voice_blends"


def write_private_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    application_owned_directory: Path | None = None,
) -> PrivatePathResult:
    """Serialize and atomically persist a private JSON document."""
    serialized = json.dumps(payload, indent=2) + "\n"
    return atomic_private_write_text(
        path,
        serialized,
        application_owned_directory=application_owned_directory,
    )


def write_kokoro_ui_blends(payload: Mapping[str, Any]) -> PrivatePathResult:
    """Atomically persist UI Kokoro blends for the active profile."""
    config_path = get_cli_config_path()
    return write_private_json(
        config_path.parent / "kokoro_voice_blends.json",
        payload,
        application_owned_directory=application_owned_config_directory(config_path),
    )
