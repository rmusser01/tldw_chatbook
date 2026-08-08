"""Offline Parakeet ONNX runtime for app-owned batch transcription."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)

from .contracts import (
    ExecutionDevice,
    ProducedCapabilities,
    TimestampGranularity,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
    TranscriptionWarningCode,
)


LONG_FORM_SECONDS = 30.0


class ParakeetOnnxCancelled(RuntimeError):
    """Raised when cancellation is observed before an inference batch."""


def _onnx_asr_api() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    """Import the native runtime only when a resident is actually loaded."""
    import onnx_asr
    from onnx_asr.utils import pad_list, read_wav_files

    return onnx_asr, read_wav_files, pad_list


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        return audio.getnframes() / rate if rate else 0.0


@contextmanager
def _prepared_wav(path: Path, ffmpeg_path: str | None) -> Iterator[Path]:
    """Yield a WAV input, converting non-WAV media with local ffmpeg."""
    if path.suffix.lower() == ".wav":
        yield path
        return
    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to transcribe non-WAV media")
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output = Path(temporary.name)
    temporary.close()
    try:
        subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        yield output
    finally:
        output.unlink(missing_ok=True)


class ParakeetOnnxRuntime:
    """One resident Parakeet model plus its optional managed VAD dependency."""

    def __init__(
        self,
        *,
        model: Any,
        vad: Any | None,
        model_id: str,
        precision: str,
        artifact_root: Any | None,
        artifact_dependencies: tuple[Any, ...],
        model_load_seconds: float,
        audio_reader: Callable[..., Any],
        pad_list: Callable[..., Any],
        duration_reader: Callable[[Path], float] = _wav_duration,
    ) -> None:
        if model_id not in {PARAKEET_V2_MODEL, PARAKEET_V3_MODEL}:
            raise ValueError(f"unsupported Parakeet model: {model_id}")
        if precision not in {"int8", "f32"}:
            raise ValueError(f"unsupported Parakeet precision: {precision}")
        self._model = model
        self._vad = vad
        self.model_id = model_id
        self.precision = precision
        self.artifact_root = artifact_root
        self.artifact_dependencies = artifact_dependencies
        self.model_load_seconds = model_load_seconds
        self._audio_reader = audio_reader
        self._pad_list = pad_list
        self._duration_reader = duration_reader

    @classmethod
    def load(
        cls,
        *,
        model_root: Path,
        vad_root: Path | None,
        model_id: str,
        precision: str,
        artifact_root: Any | None,
        artifact_dependencies: tuple[Any, ...],
    ) -> "ParakeetOnnxRuntime":
        """Load only explicit local paths with ONNX Runtime's CPU provider."""
        api, audio_reader, pad_list = _onnx_asr_api()
        started = time.monotonic()
        model = api.load_model(
            model_id,
            path=model_root,
            quantization="int8" if precision == "int8" else None,
            providers=["CPUExecutionProvider"],
            preprocessor_config={
                "use_numpy_preprocessors": True,
                "max_concurrent_workers": 1,
            },
        )
        vad = (
            api.load_vad(
                "silero",
                path=vad_root,
                providers=["CPUExecutionProvider"],
            )
            if vad_root is not None
            else None
        )
        return cls(
            model=model,
            vad=vad,
            model_id=model_id,
            precision=precision,
            artifact_root=artifact_root,
            artifact_dependencies=artifact_dependencies,
            model_load_seconds=time.monotonic() - started,
            audio_reader=audio_reader,
            pad_list=pad_list,
        )

    def transcribe(
        self,
        *,
        audio_path: Path,
        attempt_id: str,
        language: str,
        timestamps: bool,
        batch_id: str | None = None,
        job_id: str | None = None,
        retry_of_attempt_id: str | None = None,
        retry_of_job_id: str | None = None,
        vad: bool = False,
        is_cancelled: Callable[[], bool] | None = None,
        ffmpeg_path: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one local media file and return the normalized contract."""
        started = time.monotonic()
        with _prepared_wav(Path(audio_path), ffmpeg_path) as wav_path:
            duration = self._duration_reader(wav_path)
            use_vad = vad or duration > LONG_FORM_SECONDS
            if use_vad:
                if self._vad is None:
                    raise RuntimeError(
                        "Long-form Parakeet requires the managed VAD dependency. "
                        "Retry with faster-whisper."
                    )
                text, raw_segments = self._transcribe_segments(
                    wav_path,
                    is_cancelled=is_cancelled,
                )
            else:
                self._check_cancelled(is_cancelled)
                text = str(self._model.recognize(wav_path)).strip()
                raw_segments = ((0.0, duration, text),) if text else ()
        inference_seconds = time.monotonic() - started
        granularity = (
            TimestampGranularity.SEGMENT
            if timestamps
            else TimestampGranularity.NONE
        )
        segments = (
            tuple(
                TranscriptionSegment(start, end, segment_text)
                for start, end, segment_text in raw_segments
            )
            if timestamps
            else ()
        )
        is_v3 = self.model_id == PARAKEET_V3_MODEL
        warnings = (
            (TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,)
            if is_v3
            else ()
        )
        provenance = TranscriptionProvenance(
            schema_version=1,
            attempt_id=attempt_id,
            batch_id=batch_id,
            job_id=job_id,
            retry_of_attempt_id=retry_of_attempt_id,
            retry_of_job_id=retry_of_job_id,
            provider_id="parakeet-onnx",
            model_id=self.model_id,
            artifact_root=self.artifact_root,
            artifact_dependencies=self.artifact_dependencies,
            precision=self.precision,
            requested_device=ExecutionDevice.CPU,
            effective_device=ExecutionDevice.CPU,
            requested_language=language,
            effective_language="auto" if is_v3 else "en",
            detected_language=None,
            task=TranscriptionTask.TRANSCRIBE,
        )
        return TranscriptionResult(
            text=text,
            segments=segments,
            provenance=provenance,
            produced_capabilities=ProducedCapabilities(
                timestamps=granularity,
                punctuation=True,
                capitalization=True,
                vad=use_vad,
                diarization=False,
            ),
            duration_seconds=duration,
            timings=TranscriptionTimings(
                preparation_seconds=0.0,
                model_load_seconds=self.model_load_seconds,
                inference_seconds=inference_seconds,
                postprocess_seconds=0.0,
                total_seconds=inference_seconds,
            ),
            warnings=warnings,
        )

    def _transcribe_segments(
        self,
        audio_path: Path,
        *,
        is_cancelled: Callable[[], bool] | None,
    ) -> tuple[str, tuple[tuple[float, float, str], ...]]:
        waveforms, lengths, sample_rate = self._audio_reader(
            audio_path,
            channel="mean",
        )
        waveforms, lengths = self._model.resampler(
            waveforms,
            lengths,
            sample_rate,
        )
        target_rate = self._model.asr._get_sample_rate()
        segment_groups = self._vad.segment_batch(waveforms, lengths, target_rate)
        segments: list[tuple[float, float, str]] = []
        for waveform, ranges in zip(waveforms, segment_groups, strict=True):
            for start, end in ranges:
                self._check_cancelled(is_cancelled)
                batch, batch_lengths = self._pad_list([waveform[start:end]])
                recognized = self._model.asr.recognize_batch(batch, batch_lengths)
                result = next(iter(recognized))
                text = str(result.text).strip()
                if text:
                    segments.append((start / target_rate, end / target_rate, text))
        return " ".join(item[2] for item in segments), tuple(segments)

    @staticmethod
    def _check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
        if is_cancelled is not None and is_cancelled():
            raise ParakeetOnnxCancelled("Parakeet transcription cancelled")

    def close(self) -> None:
        """Release references to resident native sessions."""
        self._model = None
        self._vad = None


__all__ = ["ParakeetOnnxCancelled", "ParakeetOnnxRuntime"]
