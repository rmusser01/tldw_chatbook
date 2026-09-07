"""Offline Parakeet ONNX runtime for app-owned batch transcription."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)

from .contracts import (
    BufferAudioSource,
    ExecutionDevice,
    ProducedCapabilities,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
    TranscriptionWarningCode,
)
from .persistence import (
    FailedTranscriptionAttempt,
    dump_failed_transcription_attempt,
    load_failed_transcription_attempt,
)

LONG_FORM_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class ParakeetBufferResult:
    """Normalized buffer result plus text for each caller-logical segment."""

    normalized: TranscriptionResult
    logical_segments: tuple[str, ...]


class ParakeetOnnxCancelled(RuntimeError):
    """Raised when cancellation is observed before an inference batch."""


class ParakeetOnnxFailure(RuntimeError):
    """Typed path-private failure consumed by the executor boundary."""

    def __init__(
        self,
        code: TranscriptionFailureCode,
        message: str,
        *,
        attempt_id: str,
        batch_id: str | None,
        job_id: str | None,
        model_id: str,
        artifact_root: Any | None,
        artifact_dependencies: tuple[Any, ...],
        precision: str,
        requested_language: str,
        effective_language: str,
        effective_device: ExecutionDevice | None = ExecutionDevice.CPU,
    ) -> None:
        self.error_detail = {
            "category": "stt_failure",
            "code": code.value,
            "message": message,
            "actions": ["retry_faster_whisper"],
        }
        failed_attempt = FailedTranscriptionAttempt(
            attempt_id=attempt_id,
            batch_id=batch_id,
            job_id=job_id,
            provider_id="parakeet-onnx",
            model_id=model_id,
            artifact_root=artifact_root,
            artifact_dependencies=artifact_dependencies,
            precision=precision,
            requested_device=ExecutionDevice.CPU,
            effective_device=effective_device,
            requested_language=requested_language,
            effective_language=effective_language,
            detected_language=None,
            task=TranscriptionTask.TRANSCRIBE,
            error_code=code,
        )
        self.stt_failure_provenance = load_failed_transcription_attempt(
            dump_failed_transcription_attempt(failed_attempt)
        )
        super().__init__(message)


def _onnx_asr_api() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    """Import the native runtime only when a resident is actually loaded."""
    from tldw_chatbook.Utils.optional_deps import parakeet_onnx_deps_installed

    if not parakeet_onnx_deps_installed():
        raise ModuleNotFoundError("onnx-asr runtime is not installed")

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
    ) -> ParakeetOnnxRuntime:
        """Load only explicit local paths with ONNX Runtime's CPU provider.

        Args:
            model_root: Existing local directory containing the ASR model.
            vad_root: Existing local directory containing Silero VAD, if used.
            model_id: Exact supported Parakeet v2 or v3 model identifier.
            precision: Exact ``int8`` or ``f32`` artifact variant.
            artifact_root: Root artifact lease identity for provenance.
            artifact_dependencies: Dependency lease identities for provenance.

        Returns:
            A resident CPU-only Parakeet runtime.

        Raises:
            ModuleNotFoundError: If the optional ``onnx-asr`` runtime is absent.
            ValueError: If the model or precision is unsupported.
        """
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
        """Transcribe one local media file and return the normalized contract.

        Args:
            audio_path: Existing local audio or video path.
            attempt_id: Unique transcription-attempt identifier.
            language: Canonical requested source language.
            timestamps: Whether to include segment timestamps.
            batch_id: Optional owning batch identifier.
            job_id: Optional owning ingest-job identifier.
            retry_of_attempt_id: Optional immediately preceding attempt.
            retry_of_job_id: Optional immediately preceding ingest job.
            vad: Whether to force VAD for short-form input.
            is_cancelled: Optional cancellation probe checked before inference.
            ffmpeg_path: Optional explicit local ffmpeg executable.

        Returns:
            The normalized transcript, capabilities, timings, and provenance.

        Raises:
            ParakeetOnnxCancelled: If cancellation is observed before inference.
            ParakeetOnnxFailure: If long-form input lacks the managed VAD.
        """
        started = time.monotonic()
        model_load_seconds = self._take_model_load_seconds()
        normalized_language = (language or "en").strip().lower()
        effective_language = (
            "auto" if self.model_id == PARAKEET_V3_MODEL else "en"
        )
        with _prepared_wav(Path(audio_path), ffmpeg_path) as wav_path:
            duration = self._duration_reader(wav_path)
            use_vad = vad or duration > LONG_FORM_SECONDS
            if use_vad:
                if self._vad is None:
                    raise ParakeetOnnxFailure(
                        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                        "Long-form Parakeet requires the managed VAD dependency. "
                        "Retry with faster-whisper.",
                        attempt_id=attempt_id,
                        batch_id=batch_id,
                        job_id=job_id,
                        model_id=self.model_id,
                        artifact_root=self.artifact_root,
                        artifact_dependencies=self.artifact_dependencies,
                        precision=self.precision,
                        requested_language=normalized_language,
                        effective_language=effective_language,
                    )
                text, raw_segments = self._transcribe_segments(
                    wav_path,
                    is_cancelled=is_cancelled,
                )
            else:
                self._check_cancelled(is_cancelled)
                text = str(self._model.recognize(wav_path)).strip()
                raw_segments = ((0.0, duration, text),) if text else ()
        return self._build_result(
            text=text,
            raw_segments=raw_segments,
            duration=duration,
            attempt_id=attempt_id,
            batch_id=batch_id,
            job_id=job_id,
            retry_of_attempt_id=retry_of_attempt_id,
            retry_of_job_id=retry_of_job_id,
            language=normalized_language,
            timestamps=timestamps,
            used_vad=use_vad,
            model_load_seconds=model_load_seconds,
            inference_seconds=time.monotonic() - started,
        )

    def transcribe_buffer(
        self,
        *,
        source: BufferAudioSource,
        segment_end_frames: tuple[int, ...],
        attempt_id: str,
        language: str,
        job_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ParakeetBufferResult:
        """Transcribe validated interleaved 16-bit PCM entirely in memory.

        Args:
            source: Bounded PCM bytes and their audio format.
            segment_end_frames: Increasing logical segment boundaries ending at
                the final PCM frame, or an empty tuple for one segment.
            attempt_id: Stable identifier for this inference attempt.
            language: Requested language code.
            job_id: Optional Library job identifier.
            is_cancelled: Optional cancellation probe called before inference.

        Returns:
            Normalized transcription plus text for each logical segment.

        Raises:
            ImportError: NumPy is unavailable for the Parakeet ONNX feature.
            ParakeetOnnxFailure: PCM or artifact capabilities are unsupported.
            ValueError: Logical segment boundaries are invalid.
        """

        if source.sample_width != 2:
            normalized_language = (language or "en").strip().lower()
            raise ParakeetOnnxFailure(
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                "Parakeet ONNX buffer transcription requires 16-bit PCM audio.",
                attempt_id=attempt_id,
                batch_id=None,
                job_id=job_id,
                model_id=self.model_id,
                artifact_root=self.artifact_root,
                artifact_dependencies=self.artifact_dependencies,
                precision=self.precision,
                requested_language=normalized_language,
                effective_language=(
                    "auto" if self.model_id == PARAKEET_V3_MODEL else "en"
                ),
            )

        from tldw_chatbook.Utils.optional_deps import require_dependency

        np = require_dependency("numpy", "transcription_parakeet_onnx")

        started = time.monotonic()
        model_load_seconds = self._take_model_load_seconds()
        samples = np.frombuffer(source.audio, dtype="<i2").reshape(
            -1, source.channels
        )
        mono = samples.astype(np.float32).mean(axis=1) / 32768.0
        ends = segment_end_frames or (len(mono),)
        if (
            any(type(end) is not int or end <= 0 for end in ends)
            or any(a >= b for a, b in pairwise(ends))
            or ends[-1] != len(mono)
        ):
            raise ValueError(
                "segment_end_frames must increase to the final PCM frame"
            )
        starts = (0, *ends[:-1])
        logical_waveforms = tuple(
            mono[start:end] for start, end in zip(starts, ends)
        )
        duration = len(mono) / source.sample_rate
        if (
            duration > LONG_FORM_SECONDS
            and self._vad is None
            and self.model_id != PARAKEET_V2_MODEL
        ):
            normalized_language = (language or "en").strip().lower()
            raise ParakeetOnnxFailure(
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                "Long-form Parakeet v3 requires the managed VAD dependency. "
                "Retry with faster-whisper.",
                attempt_id=attempt_id,
                batch_id=None,
                job_id=job_id,
                model_id=self.model_id,
                artifact_root=self.artifact_root,
                artifact_dependencies=self.artifact_dependencies,
                precision=self.precision,
                requested_language=normalized_language,
                effective_language="auto",
            )
        use_vad = duration > LONG_FORM_SECONDS and self._vad is not None
        if use_vad:
            logical_segments = self._transcribe_buffer_segments(
                logical_waveforms,
                sample_rate=source.sample_rate,
                is_cancelled=is_cancelled,
            )
        else:
            logical_segments_list: list[str] = []
            for waveform in logical_waveforms:
                self._check_cancelled(is_cancelled)
                text = str(
                    self._model.recognize(
                        waveform,
                        sample_rate=source.sample_rate,
                    )
                ).strip()
                logical_segments_list.append(text)
            logical_segments = tuple(logical_segments_list)
        text = " ".join(item for item in logical_segments if item)
        normalized = self._build_result(
            text=text,
            raw_segments=(),
            duration=duration,
            attempt_id=attempt_id,
            batch_id=None,
            job_id=job_id,
            retry_of_attempt_id=None,
            retry_of_job_id=None,
            language=(language or "en").strip().lower(),
            timestamps=False,
            used_vad=use_vad,
            model_load_seconds=model_load_seconds,
            inference_seconds=time.monotonic() - started,
        )
        return ParakeetBufferResult(
            normalized=normalized,
            logical_segments=logical_segments,
        )

    def _build_result(
        self,
        *,
        text: str,
        raw_segments: tuple[tuple[float, float, str], ...],
        duration: float,
        attempt_id: str,
        batch_id: str | None,
        job_id: str | None,
        retry_of_attempt_id: str | None,
        retry_of_job_id: str | None,
        language: str,
        timestamps: bool,
        used_vad: bool,
        model_load_seconds: float,
        inference_seconds: float,
    ) -> TranscriptionResult:
        """Assemble the shared normalized file-or-buffer result contract."""

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
        effective_language = "auto" if is_v3 else "en"
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
            effective_language=effective_language,
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
                vad=used_vad,
                diarization=False,
            ),
            duration_seconds=duration,
            timings=TranscriptionTimings(
                preparation_seconds=0.0,
                model_load_seconds=model_load_seconds,
                inference_seconds=inference_seconds,
                postprocess_seconds=0.0,
                total_seconds=model_load_seconds + inference_seconds,
            ),
            warnings=warnings,
        )

    def _transcribe_buffer_segments(
        self,
        logical_waveforms: tuple[Any, ...],
        *,
        sample_rate: int,
        is_cancelled: Callable[[], bool] | None,
    ) -> tuple[str, ...]:
        """Run resident VAD and ASR for each logical in-memory waveform."""

        logical_texts: list[str] = []
        for logical_waveform in logical_waveforms:
            self._check_cancelled(is_cancelled)
            waveforms, lengths = self._model.resampler(
                [logical_waveform],
                [len(logical_waveform)],
                sample_rate,
            )
            target_rate = self._model.asr._get_sample_rate()
            self._check_cancelled(is_cancelled)
            segment_groups = self._vad.segment_batch(
                waveforms,
                lengths,
                target_rate,
            )
            recognized_segments: list[str] = []
            for waveform, ranges in zip(
                waveforms,
                segment_groups,
                strict=True,
            ):
                for start, end in ranges:
                    self._check_cancelled(is_cancelled)
                    batch, batch_lengths = self._pad_list([waveform[start:end]])
                    recognized = self._model.asr.recognize_batch(
                        batch,
                        batch_lengths,
                    )
                    result = next(iter(recognized))
                    text = str(result.text).strip()
                    if text:
                        recognized_segments.append(text)
            logical_texts.append(" ".join(recognized_segments))
        return tuple(logical_texts)

    def _take_model_load_seconds(self) -> float:
        model_load_seconds = self.model_load_seconds
        self.model_load_seconds = 0.0
        return model_load_seconds

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


__all__ = [
    "ParakeetBufferResult",
    "ParakeetOnnxCancelled",
    "ParakeetOnnxFailure",
    "ParakeetOnnxRuntime",
]
