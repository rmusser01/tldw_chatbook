"""Response-eligibility seam for the streaming PCM sink.

Pure, provider-agnostic decision: given a TTS response's declared audio
format, sample rate, and (for containerized formats) the first bytes of the
body, decide whether the response is eligible for the streaming PCM sink and,
if so, produce a `SinkPlan` describing how to consume it.

The seam branches only on the response's own shape (format / rate / bytes),
never on provider identity -- callers pass plain values, not a provider
object, so this stays true regardless of whether the caller holds an adapter
`TTSAudioResponse` or a backend path's raw format string.

Import weight: `audio_cpp_contract` was read before adding this module-scope
import -- it pulls in only stdlib (`json`, `math`, `re`, `struct`,
`unicodedata`, `dataclasses`, `decimal`, `types`, `typing`), so no lazy-import
is needed. This module itself imports nothing else: no sounddevice, no
Textual, no other TTS submodules.
"""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.TTS.audio_cpp_contract import (
    AudioCppContractError,
    validate_pcm16_wav,
)

_FORMAT_RAW_PCM = "pcm"
_FORMAT_WAV = "wav"
_DEFAULT_CHANNELS = 1

__all__ = ["SinkPlan", "sink_plan"]


@dataclass(frozen=True, slots=True)
class SinkPlan:
    """Parameters the streaming PCM sink needs to consume an eligible response."""

    sample_rate: int
    channels: int
    skip_bytes: int


def sink_plan(
    audio_format: str,
    sample_rate: int | None,
    first_bytes: bytes | None,
    channels: int | None = None,
) -> SinkPlan | None:
    """Decide whether a TTS response is eligible for the streaming PCM sink.

    Returns `None` for anything the sink can't safely consume -- compressed
    formats, raw PCM with no declared rate, or a WAV body that fails
    structural validation -- so the caller can fall back to the legacy
    whole-file path unconditionally.

    Args:
        audio_format: The response's declared audio format, e.g. "pcm",
            "wav", "mp3". Compared case-sensitively against the two formats
            the sink understands; anything else is ineligible.
        sample_rate: The response's declared sample rate. Required for
            "pcm" (returns `None` when absent); ignored for "wav", where the
            validated header is authoritative.
        first_bytes: The response body (or its head). Required for "wav" --
            validated as a canonical PCM16 RIFF/WAVE body via
            `validate_pcm16_wav`. Unused for "pcm".
        channels: Explicit channel count for raw "pcm"; defaults to 1 when
            omitted. Ignored for "wav", where the validated header is
            authoritative.

    Returns:
        A `SinkPlan` if the response is eligible, else `None`.
    """
    if audio_format == _FORMAT_RAW_PCM:
        if sample_rate is None:
            return None
        resolved_channels = _DEFAULT_CHANNELS if channels is None else channels
        return SinkPlan(
            sample_rate=sample_rate, channels=resolved_channels, skip_bytes=0
        )

    if audio_format == _FORMAT_WAV:
        return _wav_plan(first_bytes)

    return None


def _wav_plan(first_bytes: bytes | None) -> SinkPlan | None:
    """Validate a WAV body and derive the plan, or `None` if it's not usable."""
    if not first_bytes:
        return None
    try:
        info = validate_pcm16_wav(first_bytes)
    except AudioCppContractError:
        return None

    # Pcm16WavInfo (tldw_chatbook/TTS/audio_cpp_contract.py) carries no
    # chunk-offset field -- only sample_rate/channels/frame_count/data_size/
    # byte_rate/block_align/bits_per_sample. `validate_pcm16_wav` accepts
    # ancillary RIFF chunks ahead of `data` (see its module docstring), so
    # the header length isn't reliably the canonical 44 bytes. Deriving it as
    # "everything that isn't the validated data payload" stays correct in
    # that case and agrees with 44 for a canonical header (as pinned by
    # test_valid_pcm16_wav_is_eligible_with_header_skip).
    skip_bytes = len(first_bytes) - info.data_size
    return SinkPlan(
        sample_rate=info.sample_rate, channels=info.channels, skip_bytes=skip_bytes
    )
