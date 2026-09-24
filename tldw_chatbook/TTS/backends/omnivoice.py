# omnivoice.py
# Description: OmniVoice ONNX int8hq TTS backend — the in-process engine
#
# Engine port lineage: k2-fsa/omnivoice (Apache-2.0) — prompt construction and
# the diffusion sampling loop live in TTS/omnivoice_prompt.py and
# TTS/omnivoice_sampler.py; this module owns sessions, resolution, threading,
# and audio post-processing. Sessions: ct03/omnivoice-onnx-int8hq (CC-BY-NC
# weights; Boson tokenizer license).
#
# Imports
from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_limits import check_buffered_audio_size
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.audio_service import get_audio_service
from tldw_chatbook.TTS.base_backends import LocalTTSBackend
from tldw_chatbook.TTS.omnivoice_artifact_catalog import OMNIVOICE_ONNX_REQUIRED_PATHS
from tldw_chatbook.TTS.omnivoice_prompt import (
    NUM_AUDIO_CODEBOOK,
    build_prompt_inputs,
    estimate_target_frames,
)
from tldw_chatbook.TTS.omnivoice_sampler import (
    OmniVoiceSamplerConfig,
    OmniVoiceSamplingCancelled,
    run_diffusion_sampling,
)

_SAMPLE_RATE = 24_000
_FRAME_RATE = 75  # one codec frame = 320 samples at 24 kHz
_RTF_HEADROOM = 8.0  # observed worst case ~7; timeout budget multiplier

# Post-processing shape (upstream generate() conventions, seconds/samples at 24 kHz)
_EDGE_TRIM_MARGIN_S = 0.1  # silence retained around the first/last voiced sample
_MID_SILENCE_MAX_S = 0.5  # in-utterance silences longer than this compress to it
_FADE_S = 0.1  # fade-in/out length
_PAD_S = 0.1  # leading/trailing silence pad
_SILENCE_THRESHOLD = 1e-3
_PEAK_TARGET = 0.5  # non-cloned peak normalization target
_DEFAULT_REF_RMS = 0.1  # cloned RMS target when the reference measured silent
_PEAK_CEILING = 0.99  # normalization never clips

_AUDIO_SUFFIXES = (
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".mp4", ".opus", ".aac", ".wma",
)

# Native outputs are wav/pcm; the rest convert through audio_service (kokoro
# sibling routing). soundfile alone can encode wav/flac; mp3/opus/aac need
# pydub (+ ffmpeg).
_NATIVE_FORMATS = ("wav", "pcm")
_SOUNDFILE_FORMATS = ("wav", "flac")
_SUPPORTED_FORMATS = ("mp3", "wav", "opus", "aac", "flac", "pcm")

_OPERATION_ID = "omnivoice"


#######################################################################################################################
#
# Errors
#
class OmniVoiceModelError(TTSOperationError):
    """model_invalid — a resolved OmniVoice layout is incomplete or hostile."""

    def __init__(self, message: str) -> None:
        super().__init__(
            code="model_invalid",
            message=message,
            retryable=False,
            operation_id=_OPERATION_ID,
            recovery_action="reinstall_model",
        )


class OmniVoiceNotConfiguredError(TTSOperationError):
    """not_configured — no model root is configured or installed."""

    def __init__(self, message: str) -> None:
        super().__init__(
            code="not_configured",
            message=message,
            retryable=False,
            operation_id=_OPERATION_ID,
            recovery_action="configure_model_root",
        )


#######################################################################################################################
#
# Resolution
#
def _dependency_available() -> bool:
    """Engine-level truth: both optional runtimes import. (The optional-deps
    feature key ``omnivoice_tts`` declares the same pair.)"""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_model_root(
    config: Mapping[str, Any],
    managed: Callable[[], Path | None] | None = None,
) -> Path:
    """Resolve the OmniVoice ONNX root: explicit config path first, then the
    active managed artifact, else ``not_configured``.

    Args:
        config: Backend configuration mapping (``OMNIVOICE_MODEL_ROOT``).
        managed: Zero-arg callable returning the managed artifact root or None.

    Returns:
        The validated model-root directory.

    Raises:
        OmniVoiceModelError: The resolved layout is missing required files.
        OmniVoiceNotConfiguredError: No source was resolvable at all.
    """
    explicit = config.get("OMNIVOICE_MODEL_ROOT")
    if explicit:
        root = Path(explicit).expanduser()
        _validate_layout(root)
        return root
    if managed is not None:
        found = managed()
        if found is not None:
            _validate_layout(found)
            return found
    raise OmniVoiceNotConfiguredError(
        "omnivoice: not_configured — set OMNIVOICE_MODEL_ROOT (or the "
        "[OmniVoiceSettings] model_root) or install the managed artifact "
        "from the model browser"
    )


def _validate_layout(root: Path) -> None:
    missing = [
        rel for rel in OMNIVOICE_ONNX_REQUIRED_PATHS if not (root / rel).is_file()
    ]
    if missing:
        raise OmniVoiceModelError(
            f"omnivoice: model_invalid — layout at {root} is missing "
            f"{missing[:3]}"
        )


def _managed_root() -> Path | None:
    """Return the active managed OmniVoice artifact root, if installed.

    Mirrors the parakeet managed-first lookup: only the synchronous,
    credential-free ``list_installed()`` surface; every failure (including a
    missing store) means "nothing managed yet", not a hard error.
    """
    try:
        from tldw_chatbook.Model_Artifacts.service import ArtifactError
        from tldw_chatbook.Model_Artifacts.store import managed_service
        from tldw_chatbook.TTS.omnivoice_artifact_catalog import (
            omnivoice_onnx_reference,
        )

        reference = omnivoice_onnx_reference()
        for item in managed_service().list_installed():
            if (
                item.descriptor is not None
                and item.ready
                and item.active
                and item.descriptor.reference == reference
            ):
                return item.path
    except (ArtifactError, TypeError, ValueError, OSError):
        return None
    return None


#######################################################################################################################
#
# ONNX session adapters
#
_ORT_DTYPES: Mapping[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}


def _ort_dtype(declared: str) -> np.dtype:
    try:
        return _ORT_DTYPES[declared]
    except KeyError:
        raise OmniVoiceModelError(
            f"omnivoice: model_invalid — session declared unsupported input "
            f"type {declared!r}"
        ) from None


class _OrtBatchRunner:
    """Adapt the (2,8,S) batched prompt to the LM session's declared inputs.

    ``run`` receives the sampler's contract arrays — ``input_ids`` /
    ``audio_mask`` shaped (2, 8, S) and ``attention_mask`` (2, 1, S, S) — and
    supplies each input the session actually declares, casting to the declared
    dtype (bool masks export as int64 in some graphs).
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._spec = tuple((i.name, i.type) for i in session.get_inputs())

    def run(
        self, input_ids: np.ndarray, audio_mask: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        batch, codebooks, seq_len = input_ids.shape
        feeds: dict[str, np.ndarray] = {}
        for name, declared in self._spec:
            dtype = _ort_dtype(declared)
            if "input_ids" in name:
                feeds[name] = input_ids.astype(dtype, copy=False)
            elif "audio" in name:
                feeds[name] = audio_mask.astype(dtype, copy=False)
            elif "attention" in name:
                feeds[name] = attention_mask.astype(dtype, copy=False)
            elif "position" in name:
                position_ids = np.tile(np.arange(seq_len), (batch, codebooks, 1))
                feeds[name] = position_ids.astype(dtype, copy=False)
            else:
                raise OmniVoiceModelError(
                    f"omnivoice: model_invalid — LM session declares unsupported "
                    f"input {name!r}"
                )
        output = self._session.run(None, feeds)
        return np.asarray(output[0])


class _OrtCodeRunner:
    """Feed (1, 8, T) codebook codes to a codes->waveform decoder session."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._spec = tuple((i.name, i.type) for i in session.get_inputs())

    def run(self, codes: np.ndarray) -> list[np.ndarray]:
        frames = codes.shape[-1]
        feeds: dict[str, np.ndarray] = {}
        assigned = False
        for name, declared in self._spec:
            dtype = _ort_dtype(declared)
            if "position" in name:
                position_ids = np.tile(np.arange(frames), (codes.shape[0], codes.shape[1], 1))
                feeds[name] = position_ids.astype(dtype, copy=False)
            elif not assigned:
                feeds[name] = codes.astype(dtype, copy=False)
                assigned = True
            else:
                raise OmniVoiceModelError(
                    f"omnivoice: model_invalid — decoder session declares "
                    f"unsupported input {name!r}"
                )
        if not assigned:
            raise OmniVoiceModelError(
                "omnivoice: model_invalid — decoder session declares no inputs"
            )
        return self._session.run(None, feeds)


class _OrtWaveRunner:
    """Feed a (T,) 24 kHz float32 mono waveform to the encoder session."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._spec = tuple((i.name, i.type) for i in session.get_inputs())

    def run(self, waveform: np.ndarray) -> list[np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        assigned = False
        for name, declared in self._spec:
            dtype = _ort_dtype(declared)
            if not assigned:
                feeds[name] = waveform[None, :].astype(dtype, copy=False)
                assigned = True
            else:
                raise OmniVoiceModelError(
                    f"omnivoice: model_invalid — encoder session declares "
                    f"unsupported input {name!r}"
                )
        if not assigned:
            raise OmniVoiceModelError(
                "omnivoice: model_invalid — encoder session declares no inputs"
            )
        return self._session.run(None, feeds)


def _first_output(output: Any) -> np.ndarray:
    return np.asarray(output[0] if isinstance(output, (list, tuple)) else output)


def _as_codebooks(codes: np.ndarray) -> np.ndarray:
    """Normalize an encoder output to the (8, T) codebook layout."""
    array = np.asarray(codes)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == NUM_AUDIO_CODEBOOK:
        array = array.reshape(NUM_AUDIO_CODEBOOK, -1) if array.shape[1] != 1 else array[:, 0, :]
    if array.ndim == 2 and array.shape[1] == NUM_AUDIO_CODEBOOK and array.shape[0] != NUM_AUDIO_CODEBOOK:
        array = array.T
    if array.ndim != 2 or array.shape[0] != NUM_AUDIO_CODEBOOK:
        raise OmniVoiceModelError(
            f"omnivoice: model_invalid — encoder returned codes with shape "
            f"{np.asarray(codes).shape}, expected ({NUM_AUDIO_CODEBOOK}, T)"
        )
    return array.astype(np.int64, copy=False)


#######################################################################################################################
#
# Audio helpers
#
def _resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resample (reference-audio grade, not synthesis grade)."""
    if source_rate == target_rate or waveform.size == 0:
        return waveform
    frames = max(1, round(waveform.size * target_rate / source_rate))
    positions = np.linspace(0.0, waveform.size - 1, frames)
    return np.interp(positions, np.arange(waveform.size), waveform).astype(np.float32)


def _load_audio_mono(path: Path) -> np.ndarray:
    """Load an audio file as 24 kHz mono float32.

    soundfile handles every common container when present; without it we
    degrade to stdlib ``wave`` for 16-bit PCM WAV files.
    """
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    if sf is not None:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
    else:
        import wave as _wave

        with _wave.open(str(path), "rb") as reader:
            sample_rate = reader.getframerate()
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            raw = reader.readframes(reader.getnframes())
        if width != 2 or channels < 1:
            raise TTSOperationError(
                code="request_invalid",
                message=(
                    "omnivoice: request_invalid — reference audio must be 16-bit "
                    "PCM WAV when soundfile is unavailable"
                ),
                retryable=False,
                operation_id=_OPERATION_ID,
                recovery_action="use_wav_reference",
            )
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        mono = (
            samples.reshape(-1, channels).mean(axis=1)
            if channels > 1
            else samples
        )
    if sample_rate <= 0:
        raise TTSOperationError(
            code="request_invalid",
            message="omnivoice: request_invalid — reference audio has no frames",
            retryable=False,
            operation_id=_OPERATION_ID,
        )
    return _resample_linear(mono, int(sample_rate), _SAMPLE_RATE)


def _compress_mid_silence(waveform: np.ndarray) -> np.ndarray:
    """Compress in-utterance silence runs longer than 500 ms to 500 ms."""
    max_frames = int(_MID_SILENCE_MAX_S * _SAMPLE_RATE)
    quiet = np.abs(waveform) <= _SILENCE_THRESHOLD
    if not quiet.any():
        return waveform
    padded = np.concatenate(([False], quiet, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    keep = np.ones(waveform.size, dtype=np.bool_)
    for start, end in zip(starts, ends):
        if end - start > max_frames:
            keep[start + max_frames : end] = False
    return waveform[keep]


def _postprocess(waveform: np.ndarray, ref_rms: float | None) -> np.ndarray:
    """Trim, normalize, and pad the decoder output into the final waveform.

    Steps (upstream ``generate()`` conventions): trim lead/trail silence
    keeping a 100 ms margin (a fully silent clip is kept, never emptied);
    compress mid-silence runs over 500 ms; RMS-normalize to the reference
    loudness when cloning (peak 0.5 otherwise, never clipping past 0.99);
    apply 0.1 s fade-in/out and pad 0.1 s of silence on each side.
    """
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise TTSOperationError(
            code="generation_failed",
            message="omnivoice: generation_failed — decoder produced no samples",
            retryable=True,
            operation_id=_OPERATION_ID,
        )

    loud = np.flatnonzero(np.abs(audio) > _SILENCE_THRESHOLD)
    if loud.size:
        margin = int(_EDGE_TRIM_MARGIN_S * _SAMPLE_RATE)
        start = max(0, int(loud[0]) - margin)
        end = min(audio.size, int(loud[-1]) + 1 + margin)
        audio = audio[start:end]

    audio = _compress_mid_silence(audio)

    peak = float(np.max(np.abs(audio)))
    if ref_rms is not None and ref_rms > 0.0:
        target_rms = ref_rms if ref_rms > 0.0 else _DEFAULT_REF_RMS
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms > 0.0:
            gain = target_rms / rms
            if peak > 0.0:
                gain = min(gain, _PEAK_CEILING / peak)
            audio = audio * gain
    elif peak > 0.0:
        audio = audio * (_PEAK_TARGET / peak)

    fade = min(int(_FADE_S * _SAMPLE_RATE), audio.size // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    pad = np.zeros(int(_PAD_S * _SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([pad, audio, pad]).astype(np.float32)


def _wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono waveform as a stdlib-``wave`` 16-bit PCM WAV."""
    data = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _pcm_bytes(waveform: np.ndarray) -> bytes:
    """Encode a float32 mono waveform as raw 16-bit little-endian PCM."""
    data = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    return (data * 32767.0).astype("<i2").tobytes()


#######################################################################################################################
#
# OmniVoice ONNX TTS Backend Implementation
#
class OmniVoiceOnnxTTSBackend(LocalTTSBackend):
    """In-process ONNX OmniVoice engine (batch-only, single-flight).

    The whole utterance synthesizes in one off-thread pass and yields a
    single final chunk. Voice cloning runs the encoder over a 24 kHz mono
    reference and feeds the resulting codes (plus the reference transcript)
    into the prompt; the sampler always receives the prompt's audio mask so
    reference codes are never embedded as text.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._generation_lock = asyncio.Lock()
        self._initialization_lock = asyncio.Lock()
        self._lm: Any = None
        self._decoder: Any = None
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._root: Path | None = None
        self._cancel = asyncio.Event()
        # Set by _encode_reference under the generation lock (single-flight).
        self._last_reference_rms: float | None = None
        self.audio_service = get_audio_service()

    # -- lifecycle ----------------------------------------------------

    async def initialize(self) -> None:
        """Resolve dependencies and the model root (no session creation)."""
        if not _dependency_available():
            raise TTSOperationError(
                code="dependency_missing",
                message=(
                    "omnivoice: dependency_missing — onnxruntime and tokenizers "
                    "are required; install with pip install '.[omnivoice_tts]'"
                ),
                retryable=False,
                operation_id=_OPERATION_ID,
                recovery_action="install_omnivoice_tts",
            )
        self._root = await asyncio.to_thread(resolve_model_root, self.config, _managed_root)
        logger.info(f"OmniVoiceOnnxTTSBackend: model root resolved at {self._root}")

    async def load_model(self) -> None:
        """Lazy: nothing heavy until the first generation."""
        if self._root is None:
            await self.initialize()

    async def close(self) -> None:
        """Cancel in-flight sampling and drop every session."""
        self._cancel.set()
        self._lm = None
        self._decoder = None
        self._encoder = None
        self._tokenizer = None
        self.model_loaded = False
        await super().close()
        logger.info("OmniVoiceOnnxTTSBackend: closed")

    def get_capabilities(self) -> dict[str, Any]:
        """Batch-only engine: one chunk per utterance, cloning on (kokoro set)."""
        return {
            "streaming": False,
            "voice_cloning": True,
            "multi_speaker": False,
            "sample_rate": _SAMPLE_RATE,
            "formats": list(_SUPPORTED_FORMATS),
        }

    # -- generation ---------------------------------------------------

    async def generate_speech_stream(
        self,
        request: OpenAISpeechRequest | None = None,
        *,
        text: str | None = None,
        voice: str = "",
        reference_audio: str | None = None,
        reference_text: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Synthesize one utterance and yield a single audio chunk.

        Accepts either the sibling ``OpenAISpeechRequest`` (higgs/kokoro
        contract — ``input``/``voice``/``response_format``/``speed`` plus
        ``extra_params`` keys ``reference_audio``/``reference_text``) or
        direct ``text``/``voice`` keywords.

        Yields:
            One chunk of audio bytes in the requested format — native WAV or
            raw int16 PCM, or any other supported format converted through
            audio_service.
        """
        extra: dict[str, Any] = {}
        if request is not None:
            extra.update(dict(getattr(request, "extra_params", None) or {}))
            if text is None:
                text = request.input
            if not voice:
                voice = request.voice or ""
            if reference_audio is None:
                reference_audio = extra.get("reference_audio")
            if reference_text is None:
                reference_text = extra.get("reference_text")
            if instruct is None:
                instruct = extra.get("instruct")
            response_format = request.response_format or response_format
            speed = float(request.speed or speed)
        if text is None or not str(text).strip():
            raise TTSOperationError(
                code="request_invalid",
                message="omnivoice: request_invalid — text is required",
                retryable=False,
                operation_id=_OPERATION_ID,
            )

        # higgs convention: a voice that names an audio file is a clone reference.
        reference_source = reference_audio
        if reference_source is None and voice:
            candidate = Path(voice).expanduser()
            if candidate.suffix.lower() in _AUDIO_SUFFIXES and candidate.is_file():
                reference_source = str(candidate)
        # A managed voice profile carries the transcript cloning requires.
        if voice.startswith("profile:"):
            profile_audio, profile_text = self._load_voice_profile(
                voice.removeprefix("profile:")
            )
            if profile_audio is None:
                raise TTSOperationError(
                    code="request_invalid",
                    message=(
                        "omnivoice: request_invalid — "
                        f"voice profile '{voice.removeprefix('profile:')}' not found"
                    ),
                    retryable=False,
                    operation_id=_OPERATION_ID,
                )
            reference_source = profile_audio
            reference_text = profile_text

        # User multiplier on top of the built-in worst-case RTF headroom.
        timeout_factor = float(self.config.get("OMNIVOICE_TIMEOUT_FACTOR", 1.0))
        num_steps = int(self.config.get("OMNIVOICE_NUM_STEPS", 32))
        # Request-level playground knobs beat the config defaults.
        if isinstance(extra.get("num_steps"), (int, float)) and extra["num_steps"] >= 1:
            num_steps = int(extra["num_steps"])
        guidance_override: float | None = None
        if isinstance(extra.get("guidance_scale"), (int, float)):
            guidance_override = float(extra["guidance_scale"])
        started = time.monotonic()
        loop = asyncio.get_running_loop()

        def report_step(step: int, total_steps: int) -> None:
            elapsed = time.monotonic() - started

            def emit() -> None:
                asyncio.ensure_future(
                    self._report_progress(
                        step=step, total_steps=total_steps, elapsed=elapsed
                    )
                )

            loop.call_soon_threadsafe(emit)

        progress = report_step if self.progress_callback is not None else None

        logger.info(
            f"OmniVoiceOnnxTTSBackend: generating {len(text)} chars "
            f"(cloning={reference_source is not None}, steps={num_steps})"
        )
        async with self._generation_lock:
            await self._ensure_loaded()
            self._last_reference_rms = None
            ref_codes = None
            try:
                if reference_source:
                    ref_codes = await asyncio.to_thread(
                        self._encode_reference, reference_source
                    )
                waveform = await asyncio.to_thread(
                    self._synthesize_codes,
                    text,
                    instruct,
                    reference_text or "",
                    ref_codes,
                    num_steps,
                    speed,
                    timeout_factor,
                    guidance_override,
                    lambda: self._cancel.is_set(),
                    progress,
                )
            except OmniVoiceSamplingCancelled:
                logger.info("OmniVoiceOnnxTTSBackend: generation cancelled; no audio")
                return
            except TTSOperationError:
                raise
            except Exception as e:
                logger.opt(exception=True).error(
                    f"OmniVoiceOnnxTTSBackend: generation failed: {e}"
                )
                raise TTSOperationError(
                    code="generation_failed",
                    message="OmniVoice generation failed.",
                    retryable=True,
                    operation_id=_OPERATION_ID,
                    recovery_action="retry",
                ) from e

        elapsed = time.monotonic() - started
        duration = waveform.size / _SAMPLE_RATE
        logger.info(
            f"OmniVoiceOnnxTTSBackend: generated {duration:.2f}s of audio "
            f"in {elapsed:.2f}s ({duration / elapsed if elapsed > 0 else 0:.1f}x realtime)"
        )
        # Format routing happens after the generation lock (kokoro.py:976
        # pattern): wav/pcm are native; everything else converts honestly —
        # no WAV bytes labeled as the requested format.
        requested_format = str(response_format or "wav").lower()
        if requested_format == "pcm":
            yield _pcm_bytes(waveform)
        elif requested_format == "wav":
            yield _wav_bytes(waveform, _SAMPLE_RATE)
        else:
            yield await self._convert_format(waveform, requested_format)

    async def _convert_format(self, waveform: np.ndarray, target_format: str) -> bytes:
        """Encode the waveform in a non-native format via audio_service.

        Raises:
            TTSOperationError: ``request_invalid`` for formats outside the
                supported set, ``dependency_missing`` when no converter for
                the format is installed, ``generation_failed`` when the
                conversion itself fails.
        """
        if target_format not in _SUPPORTED_FORMATS:
            raise TTSOperationError(
                code="request_invalid",
                message=(
                    f"omnivoice: request_invalid — response_format "
                    f"{target_format!r} is not supported; supported formats: "
                    f"{', '.join(_SUPPORTED_FORMATS)}"
                ),
                retryable=False,
                operation_id=_OPERATION_ID,
                recovery_action="choose_supported_format",
            )
        if not self._conversion_available(target_format):
            raise TTSOperationError(
                code="dependency_missing",
                message=(
                    f"omnivoice: dependency_missing — encoding "
                    f"{target_format!r} needs pydub (plus ffmpeg) or soundfile; "
                    f"pip install pydub"
                ),
                retryable=False,
                operation_id=_OPERATION_ID,
                recovery_action="install_conversion_dependency",
            )
        try:
            return await self.audio_service.convert_audio(
                waveform,
                target_format,
                source_format="pcm",
                sample_rate=_SAMPLE_RATE,
            )
        except TTSOperationError:
            raise
        except (RuntimeError, ValueError) as e:
            logger.opt(exception=True).error(
                f"OmniVoiceOnnxTTSBackend: conversion to {target_format} failed: {e}"
            )
            raise TTSOperationError(
                code="generation_failed",
                message=f"OmniVoice audio conversion to {target_format!r} failed.",
                retryable=True,
                operation_id=_OPERATION_ID,
                recovery_action="retry",
            ) from e

    @staticmethod
    def _conversion_available(target_format: str) -> bool:
        """Whether audio_service can encode the format in this environment."""
        from tldw_chatbook.TTS import audio_service as audio_service_module

        if audio_service_module.PYDUB_AVAILABLE:
            return True
        return bool(
            audio_service_module.SOUNDFILE_AVAILABLE
            and target_format in _SOUNDFILE_FORMATS
        )

    # -- lazy loading -------------------------------------------------

    async def _ensure_loaded(self) -> None:
        """Create the LM/decoder sessions and tokenizer exactly once."""
        if self._lm is not None and self._decoder is not None and self._tokenizer is not None:
            return
        async with self._initialization_lock:
            if self._lm is not None and self._decoder is not None and self._tokenizer is not None:
                return
            if self._root is None:
                await self.initialize()
            root = self._root
            assert root is not None

            def _load() -> None:
                if self._tokenizer is None:
                    self._tokenizer = self._load_tokenizer(root / "tokenizer.json")
                if self._lm is None:
                    self._lm = self._create_lm_runner()
                if self._decoder is None:
                    self._decoder = self._create_decoder_session()

            await asyncio.to_thread(_load)
            self.model_loaded = True
            logger.info("OmniVoiceOnnxTTSBackend: LM, decoder, and tokenizer loaded")

    def _session_options(self) -> Any:
        """SessionOptions with the configured intra-op thread count."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        threads = int(self.config.get("OMNIVOICE_INTRA_OP_THREADS", 0) or 0)
        if threads > 0:
            options.intra_op_num_threads = threads
        return options

    def _create_lm_runner(self) -> Any:
        """Create the LM ORT session wrapped as an LMBatchRunner."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(self._root / "omnivoice_lm_int8_hq" / "model.onnx"),
            sess_options=self._session_options(),
            providers=["CPUExecutionProvider"],
        )
        return _OrtBatchRunner(session)

    def _create_decoder_session(self) -> Any:
        """Create the decoder ORT session wrapped as a codes runner."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(self._root / "audio_tokenizer_decoder_int8" / "model.onnx"),
            sess_options=self._session_options(),
            providers=["CPUExecutionProvider"],
        )
        return _OrtCodeRunner(session)

    def _create_encoder_session(self) -> Any:
        """Create the encoder ORT session wrapped as a waveform runner."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(self._root / "audio_tokenizer_encoder_int8" / "model.onnx"),
            sess_options=self._session_options(),
            providers=["CPUExecutionProvider"],
        )
        return _OrtWaveRunner(session)

    def _load_tokenizer(self, path: Path) -> Any:
        """Load tokenizer.json via the ``tokenizers`` library (no transformers)."""
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path))

    # -- reference encoding -------------------------------------------

    def _load_voice_profile(self, profile_name: str) -> tuple[str | None, str | None]:
        """Resolve a managed voice profile to (reference audio, transcript).

        Profiles are the only clone source that carries the transcript the
        model requires; store failures degrade to "not found" rather than
        aborting synthesis with a store error.
        """
        try:
            from tldw_chatbook.TTS.omnivoice_voice_manager import (
                OmniVoiceVoiceManager,
            )

            voices_dir = Path(
                str(
                    self.config.get(
                        "OMNIVOICE_VOICE_SAMPLES_DIR",
                        "~/.config/tldw_cli/omnivoice_voices",
                    )
                )
            ).expanduser()
            manager = OmniVoiceVoiceManager(voices_dir)
            profile = manager.get_profile(profile_name)
            if profile is None:
                return None, None
            audio = manager.get_reference_audio_path(profile_name)
            if audio is None or not audio.is_file():
                return None, None
            return str(audio), str(profile.get("reference_text") or "")
        except Exception as exc:
            logger.warning("OmniVoice voice profile lookup failed: {}", exc)
            return None, None

    def _encode_reference(self, reference_audio: str) -> np.ndarray:
        """Encode reference audio as (8, T_ref) int64 codec codes.

        Loads the file as 24 kHz mono float32, enforces the configured
        maximum reference duration, and runs the encoder session. Also
        records the reference RMS for loudness-matched post-processing.
        """
        path = Path(reference_audio).expanduser()
        if not path.is_file():
            raise TTSOperationError(
                code="request_invalid",
                message=f"omnivoice: request_invalid — reference audio not found: {path}",
                retryable=False,
                operation_id=_OPERATION_ID,
            )
        waveform = _load_audio_mono(path)
        if waveform.size == 0:
            raise TTSOperationError(
                code="request_invalid",
                message="omnivoice: request_invalid — reference audio is empty",
                retryable=False,
                operation_id=_OPERATION_ID,
            )
        self._last_reference_rms = float(np.sqrt(np.mean(np.square(waveform))))

        max_duration = float(
            self.config.get("OMNIVOICE_MAX_REFERENCE_DURATION", 30) or 0
        )
        if max_duration > 0 and waveform.size > max_duration * _SAMPLE_RATE:
            waveform = waveform[: int(max_duration * _SAMPLE_RATE)]
            logger.info(
                f"OmniVoiceOnnxTTSBackend: reference truncated to {max_duration:.1f}s"
            )

        if self._encoder is None:
            self._encoder = self._create_encoder_session()
        output = self._encoder.run(waveform)
        return _as_codebooks(_first_output(output))

    # -- synthesis ----------------------------------------------------

    def _synthesize_codes(
        self,
        text: str,
        instruct: str | None,
        ref_text: str,
        ref_codes: np.ndarray | None,
        num_steps: int,
        speed: float,
        timeout_factor: float,
        guidance_override: float | None = None,
        cancel_check: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Run prompt -> sampler -> decoder -> postprocess; return the waveform.

        The length-scaled timeout folds into the sampler's cooperative
        cancel: past the deadline the loop aborts between steps and maps to
        ``generation_timeout`` (a genuine user cancel stays
        ``OmniVoiceSamplingCancelled`` for the caller to swallow).
        """
        if num_steps < 1:
            raise TTSOperationError(
                code="configuration_invalid",
                message="omnivoice: configuration_invalid — OMNIVOICE_NUM_STEPS must be >= 1",
                retryable=False,
                operation_id=_OPERATION_ID,
                recovery_action="fix_num_steps",
            )
        cloning = ref_codes is not None
        ref_frames = int(ref_codes.shape[1]) if cloning else 0
        prompt_ref_text = ref_text if cloning else ""
        target_len = estimate_target_frames(
            text, prompt_ref_text, ref_frames, speed=speed if speed > 0 else 1.0
        )
        prompt = build_prompt_inputs(
            self._tokenizer,
            text=text,
            ref_text=prompt_ref_text,
            lang=self.config.get("OMNIVOICE_LANGUAGE") or None,
            instruct=instruct,
            ref_codes=ref_codes,
            target_len=target_len,
            num_codebooks=NUM_AUDIO_CODEBOOK,
        )

        budget_seconds = (target_len / _FRAME_RATE) * _RTF_HEADROOM * max(timeout_factor, 0.0)
        deadline = None if timeout_factor <= 0 else time.monotonic() + budget_seconds

        def deadline_cancel() -> bool:
            if cancel_check():
                return True
            return deadline is not None and time.monotonic() >= deadline

        sampler_config = OmniVoiceSamplerConfig(
            num_step=num_steps,
            guidance_scale=(
                guidance_override
                if guidance_override is not None
                else float(self.config.get("OMNIVOICE_GUIDANCE_SCALE", 2.0))
            ),
            t_shift=float(self.config.get("OMNIVOICE_T_SHIFT", 0.1)),
            layer_penalty_factor=float(
                self.config.get("OMNIVOICE_LAYER_PENALTY_FACTOR", 5.0)
            ),
            position_temperature=float(
                self.config.get("OMNIVOICE_POSITION_TEMPERATURE", 5.0)
            ),
            class_temperature=float(self.config.get("OMNIVOICE_CLASS_TEMPERATURE", 0.0)),
            seed=self._optional_int("OMNIVOICE_SEED"),
            class_top_ratio=float(self.config.get("OMNIVOICE_CLASS_TOP_RATIO", 0.1)),
        )
        try:
            codes = run_diffusion_sampling(
                self._lm,
                prompt.input_ids,
                target_len,
                config=sampler_config,
                cancel_check=deadline_cancel,
                progress=progress,
                num_codebook=NUM_AUDIO_CODEBOOK,
                prompt_audio_mask=prompt.audio_mask,
            )
        except OmniVoiceSamplingCancelled:
            if cancel_check():
                raise
            raise TTSOperationError(
                code="generation_timeout",
                message=(
                    f"omnivoice: generation_timeout — sampling exceeded the "
                    f"{budget_seconds:.1f}s budget for ~{target_len / _FRAME_RATE:.1f}s "
                    f"of audio (timeout factor {timeout_factor})"
                ),
                retryable=True,
                operation_id=_OPERATION_ID,
                recovery_action="retry_or_raise_timeout_factor",
            ) from None
        if deadline is not None and time.monotonic() >= deadline and not cancel_check():
            raise TTSOperationError(
                code="generation_timeout",
                message="omnivoice: generation_timeout — decoding exceeded the time budget",
                retryable=True,
                operation_id=_OPERATION_ID,
                recovery_action="retry_or_raise_timeout_factor",
            )

        output = self._decoder.run(codes[None, :, :])  # (1, 8, T)
        waveform = np.asarray(_first_output(output), dtype=np.float32).reshape(-1)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise TTSOperationError(
                code="generation_failed",
                message="omnivoice: generation_failed — decoder returned invalid audio",
                retryable=True,
                operation_id=_OPERATION_ID,
            )
        check_buffered_audio_size(waveform.nbytes)
        return _postprocess(waveform, self._last_reference_rms if cloning else None)

    def _optional_int(self, key: str) -> int | None:
        value = self.config.get(key)
        if value is None or value == "":
            return None
        return int(value)


#
# End of omnivoice.py
#######################################################################################################################
