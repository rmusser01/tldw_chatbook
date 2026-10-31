"""Download-free transcription readiness checks, run off the UI thread."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class TranscriptionReadinessError(RuntimeError):
    """A path-private readiness failure with an actionable remedy."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def parakeet_readiness_error(error: BaseException) -> TranscriptionReadinessError:
    """Describe missing model/VAD separately without exposing native paths.

    Args:
        error: The source-resolution or verification failure.

    Returns:
        A path-private readiness failure and its specific repair instruction.
    """
    code = getattr(getattr(error, "code", None), "value", None)
    if code == "managed_vad_unavailable":
        return TranscriptionReadinessError(
            "vad_missing",
            "Parakeet requires the managed Silero VAD dependency. Install Silero "
            "VAD in Lab > Models, or choose an external Parakeet folder "
            "and review its VAD install plan, then retry.",
        )
    if isinstance(error, FileNotFoundError) or code == "managed_source_unavailable":
        return TranscriptionReadinessError(
            "model_missing",
            "Parakeet model files are not installed for the selected language "
            "and precision. Install the matching model in Lab > Models "
            "(or Install verified Parakeet v2 INT8 for English/INT8), or select "
            "a complete local model folder, then retry.",
        )
    return TranscriptionReadinessError(
        "model_invalid",
        "The Parakeet model folder could not be verified. Select a complete "
        "model matching the language and precision, or install a verified "
        "model in Lab > Models, then retry.",
    )


def check_transcription_readiness(
    options: Mapping[str, Any],
    *,
    default_model: str = "base",
    parakeet_source_service: Any = None,
    transcribe_cpp_model_path: str | None = None,
    scope_id: str | None = None,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> None:
    """Check only the selected runtime and local model; never load weights.

    Args:
        options: Captured audio/video import options.
        default_model: Whisper default from the captured configuration.
        parakeet_source_service: Existing app-owned Parakeet source service.
        transcribe_cpp_model_path: Configured direct-local GGUF path.
        scope_id: Disposable verification scope owned by this submission.
        import_module: Import boundary, injectable for dependency-free tests.

    Raises:
        TranscriptionReadinessError: A runtime, model or VAD is unavailable.
        ValueError: The selected provider or language route is invalid.
    """
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
        resolve_batch_stt_model,
        resolve_batch_stt_route,
    )
    from tldw_chatbook.Library.ingest_capabilities import (
        field_gate_open,
        get_capabilities,
    )

    provider = options.get("transcription_provider") or "default"
    target = options.get("translation_target_language")
    if target is None:
        target = options.get("target_language")
    if (
        target is None
        and options.get("translate_to_english")
        and field_gate_open(
            "audio_video",
            "translate_to_english",
            {**options, "transcription_provider": provider},
        )
    ):
        target = "en"
    route = resolve_batch_stt_route(
        provider=provider,
        language=options.get("language"),
        target_language=target,
        precision=options.get("transcription_precision"),
    )
    module, extra = {
        "faster-whisper": ("faster_whisper", "transcription_faster_whisper"),
        "parakeet-onnx": ("onnx_asr", "transcription_parakeet_onnx"),
        "transcribe-cpp": ("transcribe_cpp", "transcription_transcribe_cpp"),
    }[route.provider]
    try:
        import_module(module)
    except (ImportError, OSError):
        raise TranscriptionReadinessError(
            "package_missing",
            f"The {route.provider} runtime package or a native dependency is "
            "missing or cannot be imported. In the same Python environment "
            f"that launches Chatbook, run: python -m pip install -e '.[{extra}]'. "
            "Restart Chatbook, then retry.",
        ) from None

    if route.provider == "faster-whisper":
        from tldw_chatbook.Utils.path_validation import validate_path_simple

        model = (
            resolve_batch_stt_model(
                route,
                selected_model=options.get("model")
                or options.get("transcription_model"),
                default_model=default_model or "base",
            )
            or "base"
        )
        try:
            if Path(model).is_dir():
                root = Path(model)
            else:
                utilities = import_module("faster_whisper.utils")
                root = Path(utilities.download_model(model, local_files_only=True))
            root = validate_path_simple(
                root, require_exists=True, reject_shell_metacharacters=False
            )
            if not all(
                (root / name).is_file()
                for name in (
                    "model.bin",
                    "config.json",
                    "tokenizer.json",
                )
            ) or not any(
                (root / name).is_file()
                for name in ("vocabulary.json", "vocabulary.txt")
            ):
                raise FileNotFoundError
        except (OSError, ValueError, RuntimeError, ImportError):
            label = (
                model
                if model
                in next(
                    field.options
                    for field in get_capabilities("audio_video").fields
                    if field.name == "transcription_model"
                )
                else "selected"
            )
            raise TranscriptionReadinessError(
                "model_missing",
                f"Faster Whisper {label} model files are missing or incomplete. "
                "Download the selected model into the local Faster Whisper "
                "cache before retrying; installing the Python package does not "
                "download model files. "
                + (
                    "In Chatbook's Python environment, run: python -c "
                    f"\"from faster_whisper.utils import download_model; download_model('{label}')\""
                    if label != "selected"
                    else "Select an installed model, or download the matching model revision."
                ),
            ) from None
        return

    if route.provider == "parakeet-onnx":
        # The existing external-folder workflow verifies the exact files and
        # offers explicit VAD-install consent before it can enqueue anything.
        if str(options.get("transcription_model_dir") or "").strip():
            return
        from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

        try:
            parakeet_source_service.resolve(
                ParakeetSourceKey.from_values(route.model, route.precision),
                scope_id=scope_id,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise parakeet_readiness_error(error) from None
        return

    from tldw_chatbook.Model_Artifacts.gguf_admission import validate_local_gguf

    if not transcribe_cpp_model_path:
        raise TranscriptionReadinessError(
            "model_missing",
            "transcribe.cpp needs a local GGUF model. Use Choose GGUF model "
            "in the Audio & video options, then retry.",
        )
    try:
        validate_local_gguf(transcribe_cpp_model_path)
    except (OSError, ValueError):
        raise TranscriptionReadinessError(
            "model_invalid",
            "The selected transcribe.cpp GGUF model is missing or invalid. "
            "Use Choose GGUF model to select a complete supported file, then retry.",
        ) from None
