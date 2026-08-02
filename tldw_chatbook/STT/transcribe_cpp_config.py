"""Path-safe configuration for one user-selected transcribe.cpp GGUF."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Model_Artifacts.gguf_admission import validate_local_gguf
from tldw_chatbook.config import save_settings_to_cli_config

CONFIG_SECTION = "transcription.transcribe_cpp"


class TranscribeCppConfigError(RuntimeError):
    """A path-safe failure to persist the selected local GGUF."""


def is_gguf_file(path: Path) -> bool:
    """Return whether a picker entry has the GGUF suffix.

    Args:
        path: Candidate picker entry.

    Returns:
        True when the candidate filename ends in ``.gguf`` case-insensitively.
    """
    return path.suffix.casefold() == ".gguf"


def configure_model_path(path: str | Path) -> None:
    """Admit one local GGUF and atomically persist only its dedicated key.

    Args:
        path: User-selected local GGUF path.

    Raises:
        GGUFError: If the selected file fails bounded GGUF admission.
        TranscribeCppConfigError: If the admitted path cannot be persisted.
    """
    admission = validate_local_gguf(path)
    saved = save_settings_to_cli_config(
        {CONFIG_SECTION: {"model_path": str(admission.path)}}
    )
    if not saved:
        raise TranscribeCppConfigError(
            "The transcribe.cpp GGUF configuration could not be saved."
        )
