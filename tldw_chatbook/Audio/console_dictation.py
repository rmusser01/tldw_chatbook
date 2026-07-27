"""One-shot local microphone dictation for the native Console."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Local_Ingestion.parakeet_v2_installer import (
    PARAKEET_V2_FILES,
    parakeet_v2_install_dir,
    verify_parakeet_v2_bundle,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple


CONSOLE_DICTATION_SAMPLE_RATE = 16_000
CONSOLE_DICTATION_CHANNELS = 1
CONSOLE_DICTATION_SAMPLE_WIDTH = 2
CONSOLE_DICTATION_MAX_SECONDS = 60.0
CONSOLE_DICTATION_MAX_BYTES = int(
    CONSOLE_DICTATION_SAMPLE_RATE
    * CONSOLE_DICTATION_CHANNELS
    * CONSOLE_DICTATION_SAMPLE_WIDTH
    * CONSOLE_DICTATION_MAX_SECONDS
)
PARAKEET_V2_MODEL = "nemo-parakeet-tdt-0.6b-v2"


class ConsoleDictationError(RuntimeError):
    """Raised when a one-shot Console dictation cannot complete."""


class ConsoleDictationSession:
    """Record one bounded PCM buffer and transcribe it with Parakeet v2."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        installed_model_dir: str | Path | None = None,
        verify_installed_bundle: Callable[[str | Path], bool] | None = None,
        recorder_factory: Callable[..., Any] | None = None,
        transcription_service: Any | None = None,
    ) -> None:
        self._configured_model_dir = Path(model_dir) if model_dir else None
        self._installed_model_dir = (
            Path(installed_model_dir)
            if installed_model_dir is not None
            else parakeet_v2_install_dir()
        )
        self._verify_installed_bundle = (
            verify_installed_bundle or verify_parakeet_v2_bundle
        )
        self._recorder_factory = recorder_factory
        self._transcription_service = transcription_service
        self._recorder: Any | None = None
        self.model_dir: Path | None = None

    @staticmethod
    def _required_files_present(directory: Path) -> bool:
        return directory.is_dir() and all(
            (directory / descriptor.filename).is_file()
            for descriptor in PARAKEET_V2_FILES
        )

    def _resolve_model_dir(self) -> Path:
        if self.model_dir is not None and self._required_files_present(self.model_dir):
            return self.model_dir

        configured = self._configured_model_dir
        if configured is None:
            from tldw_chatbook.config import get_cli_setting

            configured_value = get_cli_setting(
                "transcription",
                "parakeet_onnx_model_dir",
                "",
            )
            if str(configured_value or "").strip():
                configured = Path(str(configured_value).strip())

        if configured is not None:
            try:
                configured = validate_path_simple(configured, require_exists=True)
            except (OSError, ValueError) as exc:
                raise ConsoleDictationError(
                    "The configured Parakeet v2 model folder path is invalid."
                ) from exc
            if not self._required_files_present(configured):
                raise ConsoleDictationError(
                    "Parakeet v2 model files are missing from the configured folder."
                )
            return configured

        if self._verify_installed_bundle(self._installed_model_dir):
            return self._installed_model_dir

        raise ConsoleDictationError(
            "Parakeet v2 model files are missing. Install the verified "
            "Parakeet v2 INT8 bundle from Library → Models."
        )

    def _build_recorder(self, *, on_buffer_limit: Callable[[], None] | None) -> Any:
        factory = self._recorder_factory
        if factory is None:
            from tldw_chatbook.Audio.recording_service import AudioRecordingService

            factory = AudioRecordingService
        return factory(
            sample_rate=CONSOLE_DICTATION_SAMPLE_RATE,
            channels=CONSOLE_DICTATION_CHANNELS,
            use_vad=False,
            max_buffer_bytes=CONSOLE_DICTATION_MAX_BYTES,
            on_buffer_limit=on_buffer_limit,
        )

    def _transcriber(self) -> Any:
        if self._transcription_service is not None:
            return self._transcription_service
        try:
            available = importlib.util.find_spec("onnx_asr") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            raise ConsoleDictationError(
                "onnx-asr is not installed. Install the Parakeet ONNX "
                "transcription option and retry."
            )
        from tldw_chatbook.Local_Ingestion.transcription_service import (
            TranscriptionService,
        )

        self._transcription_service = TranscriptionService()
        return self._transcription_service

    def start(
        self,
        *,
        on_buffer_limit: Callable[[], None] | None = None,
    ) -> None:
        """Start capture from the default microphone.

        Args:
            on_buffer_limit: Callback invoked when the bounded PCM buffer is full.

        Raises:
            ConsoleDictationError: The model, dependencies, or microphone are
                unavailable, or a capture is already active.
        """
        if self._recorder is not None:
            raise ConsoleDictationError("Microphone dictation is already recording.")
        self.model_dir = self._resolve_model_dir()
        self._transcriber()
        try:
            recorder = self._build_recorder(on_buffer_limit=on_buffer_limit)
            if not recorder.start_recording():
                raise ConsoleDictationError(
                    "Could not start microphone recording. Check microphone "
                    "permission and the default input device."
                )
        except ConsoleDictationError:
            raise
        except Exception as exc:
            raise ConsoleDictationError(
                f"Could not start microphone recording: {exc}"
            ) from exc
        self._recorder = recorder

    def stop_and_transcribe(self) -> str:
        """Stop capture and return a stripped English transcript.

        Returns:
            The non-empty English transcript.

        Raises:
            ConsoleDictationError: Capture cannot stop, no audio was captured,
                or local transcription fails.
        """
        recorder = self._recorder
        if recorder is None:
            raise ConsoleDictationError("Microphone dictation is not recording.")
        try:
            audio_data = recorder.stop_recording()
        except Exception as exc:
            raise ConsoleDictationError(
                f"Could not stop microphone recording: {exc}"
            ) from exc
        self._recorder = None
        if not audio_data:
            raise ConsoleDictationError("No audio was captured from the microphone.")
        if self.model_dir is None:
            raise ConsoleDictationError("Parakeet v2 model files are missing.")

        try:
            result = self._transcriber().transcribe_buffer(
                audio_data,
                sample_rate=CONSOLE_DICTATION_SAMPLE_RATE,
                channels=CONSOLE_DICTATION_CHANNELS,
                sample_width=CONSOLE_DICTATION_SAMPLE_WIDTH,
                provider="parakeet-onnx",
                model=PARAKEET_V2_MODEL,
                language="en",
                model_dir=str(self.model_dir),
            )
        except ConsoleDictationError:
            raise
        except Exception as exc:
            raise ConsoleDictationError(
                f"Parakeet transcription failed: {exc}"
            ) from exc
        text = str(result.get("text") or "").strip()
        if not text:
            raise ConsoleDictationError("Transcription returned no speech.")
        return text

    def discard(self) -> None:
        """Stop and discard any active capture without transcribing."""
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return
        try:
            recorder.stop_recording()
        except Exception as exc:
            logger.debug("Failed to discard Console dictation cleanly: {}", exc)
