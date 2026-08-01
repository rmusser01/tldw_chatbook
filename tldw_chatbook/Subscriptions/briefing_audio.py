"""Synthesize one briefing script turn's WAV audio (spec #2 phase 2b).

A cast script (`briefing_cast.generate_script`) produces an ordered list of
`{"speaker", "text"}` turns; `briefing_voices.resolve_roster_voices` turns
each speaker's stored `voice_profile_id` into a concrete `VoiceSelection`.
This module is the piece in between the two and the actual audio: given one
turn's speaker, resolved voice, and text, :func:`synthesize_turn` returns WAV
bytes for that turn, on *any* configured TTS provider. Task 6 calls this once
per turn and stitches the whole script together.

**Two provider paths, because the app has two synthesis contracts.**
`VoiceSelection.is_exact_provider()` is `True` only for `"audio_cpp"`, the
app's own native provider, reached through `TTSService.synthesize_exact`.
That call returns a `TTSAudioResponse` (an async context manager,
`TTS/adapter_types.py:352`) *and* a `TTSRequestedSelectionSnapshot` --
provenance the caller can compare field-for-field against what it asked for.
`TTSEventHandler._generate_tts` (`Event_Handlers/TTS_Events/tts_events.py`,
around its `_validate_exact_selection` at `:983`) is the reference for this
comparison; :func:`_validate_exact_snapshot` below reuses the same idea, but
reports a mismatch as a `TurnSynthesisError` naming the speaker, the turn,
and both the requested and returned voice, since a briefing turn is one of
many synthesis calls in flight, not the single global chat voice that
handler guards.

Every other provider is legacy, reached through
`TTSService.generate_audio_stream`, which yields a bare
`AsyncIterator[bytes]` -- no response object, and therefore no contract to
compare. That is exactly why the legacy path validates what it can, on its
own: a zero-byte result and a payload that fails to decode as WAV are the
two failure modes the exact path gets for free from its snapshot. This
self-validation is the whole reason phase 2b can offer per-speaker voices on
every provider, not just the app's native one.

**Closing the response is load-bearing.** `_generate_tts`'s own `finally`
(`tts_events.py:899-912`) calls `response.aclose()` unconditionally --
success or failure -- because a synthesis lease left open holds a registry
slot. It also takes care that a *second* failure, from `aclose()` itself
while a drain has already failed, does not replace the drain's real error;
it is logged (by type only) and the original error still propagates. The
exact path below manually reproduces that same `try`/`finally` shape rather
than relying on a bare `async with response:` -- **deliberately, not for
style.** Both `async with response:` and `contextlib.aclosing(response)`
call `__aexit__`/`aclose()` from inside their own exception handling, and if
that second call *also* raises, it replaces the body's exception (demoted
to `__context__` on the new one) rather than letting the original
propagate as primary. Reviewed and confirmed against the real
`TTSAudioResponse` (task-5 review round 1). Do not "simplify" this back to
`async with`/`aclosing` -- it would silently reintroduce exactly the bug
this module exists to avoid.

**Long turns are chunked.** Text longer than `MAX_TURN_CHARS` characters is
split with `TTS/text_processing.TextChunker` and the per-chunk WAV results
stitched with `TTS/audio_stitch.concat_wav_segments`. `TextChunker`'s "token"
budget is a `words * 1.3` estimate (`TOKENS_PER_WORD`), not a real tokenizer
-- see `_CHUNK_MAX_TOKENS` below for why this module picks a budget well
under the backends' own default of 500. `TextChunker` groups by
sentence/word boundaries, so it cannot split text with no whitespace at all
(dense CJK, or a pathological space-free run) no matter what budget it is
given; `_hard_split_piece` is the backstop that makes `MAX_TURN_CHARS` an
actual cap on every piece, not just a hint to the chunker.

**An unmapped legacy `provider_id` is rejected before it ever reaches the
shared builder.** `TTS/legacy_request_builder.build_legacy_speech_request`
falls through to using the request's own `model_id` as the internal route
id for any `provider_id` it does not recognize (documented, pre-existing
behavior -- see that module's Task 2 review notes). If a caller ever passed
an unmapped `provider_id` whose `model_id` happened to collide with a
reserved internal id (for example `"local_kokoro_default_onnx"`), that
would silently mis-route to the wrong backend. This module is 2b's only
caller of that builder, so it rejects any `provider_id` outside
`TTS/legacy_bridge.LEGACY_PROVIDER_IDS` -- the same source of truth the
bridge itself uses to resolve routes -- before ever calling the builder.

**ElevenLabs never produces a WAV container on its own, so this module
gives it one.** Every legacy request's `response_format` is forced to
`"wav"` (`briefing_voices._FORCED_RESPONSE_FORMAT`), but
`TTS/legacy_request_builder._LEGACY_FORMAT_OVERRIDES` -- a table shared
with other callers, preserved verbatim, not this module's to change --
unconditionally rewrites elevenlabs's format to `"mp3"` regardless of what
was requested. Asking elevenlabs for `"wav"` directly would not help
either: `TTS/backends/elevenlabs.py`'s `_map_output_format` maps `"wav"` to
the exact same wire format as `"pcm"` -- headerless PCM, not a WAV
container. Decoding the mp3 elevenlabs actually returns would need pydub's
ffmpeg-backed path, which this module deliberately avoids for the common
case (see "pydub stays optional" below). Instead, `_synthesize_legacy_chunk`
asks elevenlabs for `"pcm"` explicitly -- overriding the shared builder's
forced `"mp3"` in this module's own request, after the builder returns it,
never inside the builder itself -- and wraps the headerless samples into a
WAV container with `_wrap_pcm16_mono_as_wav` (no pydub/ffmpeg involved,
same header shape as `TTS/audio_service.py`'s `AudioService._pcm_to_wav`).
The sample rate used for that header is parsed out of
`_ELEVENLABS_PCM_WIRE_FORMAT_NAME`'s own name rather than a second,
separately hardcoded number, matching the wire format
`ElevenLabsTTSBackend._map_output_format`'s fixed `simple_format_map`
entry for `"pcm"` actually returns.

**A provider/adapter failure is named by speaker and turn, not left raw.**
`TTSService.synthesize_exact` and `TTSService.generate_audio_stream` are
the only two calls into code this module does not own; either can raise an
adapter or network error, or (on the legacy path) a route lookup failure
such as `TTS.legacy_bridge.UnknownLegacyModelError` for a profile whose
model id the compatibility bridge does not enumerate. Both call sites wrap
any `Exception` into a `TurnSynthesisError` naming the speaker and turn
index, with the original exception preserved as `__cause__` and its text
capped (`_error_text`, reused from Task 6's row-storage half below).
`asyncio.CancelledError` is a `BaseException`, not an `Exception` (since
Python 3.8), so it is never caught by this -- cancellation always
propagates untouched.

Testing (synthesis half): the only faked seam is the TTS service
(`synthesize_exact` and `generate_audio_stream`) -- everything else,
including the real `TTSAudioResponse` and the real
`TextChunker`/`concat_wav_segments`, is exercised as-is.

Nothing here logs turn text, a speaker name, or a voice id at any level --
only exception types, for the same reason `briefing_cast` and
`briefing_voices` avoid it: this app's log sink runs with `diagnose=True`,
which dumps a failing frame's locals, and the frame at a synthesis failure
holds exactly that content. `TurnSynthesisError` *messages* do name the
speaker and turn index -- that is a caller-facing error, not a log line --
but never a voice id or the turn's text.

**Task 6 adds this module's pipeline half.** `synthesize_turn` above turns
one turn into WAV bytes; `generate_script_audio` is the orchestrator that
turns a whole cast script into one stored, playable `briefing_audio` row:
it loads the script (refusing before any row exists if the script is not
`complete` or has no readable turns), resolves the roster's voices
(`briefing_voices.resolve_roster_voices`), creates the `briefing_audio`
row, synthesizes and stitches every turn
(`TTS/audio_stitch.concat_wav_segments`), and writes the finished payload
once into `briefing_audio_dir()` via
`Utils.private_paths.atomic_private_write_bytes`.

Its error-boundary contract is copied from `briefing_cast.generate_script`
(`Subscriptions/briefing_cast.py:562`) in every respect: a pre-flight
refusal (script not `complete`, no readable turns) raises
`AudioGenerationError` before any `briefing_audio` row exists; once the
pipeline has committed to an attempt, every in-band failure (a voice that
does not resolve, a turn's synthesis error, an unassigned speaker, a
stitch failure, a write failure) updates that SAME row to `failed` rather
than raising; a genuine DB error still propagates uncaught -- the caller's
worker wraps it, matching the spec's "Error handling ethos". The parent
`briefing_scripts` row is never touched by any outcome here, success or
failure -- exactly as a briefing is never touched by a script's own
outcome.

**Storage is buffer-then-write-once, not streaming.** A correct
decode-and-concat (`concat_wav_segments`) must hold every turn's decoded
audio in memory to join them, so by the time there is anything to write,
the whole finished payload already exists in memory; a streaming append
would still need a full re-encode pass over everything written so far,
and `private_paths` has no binary append call at all
(`open_private_binary` is `O_RDONLY`). The whole payload is therefore
written once, atomically, via `atomic_private_write_bytes`. If anything
fails after that write succeeds, the file is removed -- a `failed` row
must never leave an orphan audio file on disk.

Testing (pipeline half): per the brief, the only faked seam is the
per-turn `synthesize` callable itself (the `synthesize=synthesize_turn`
parameter) -- `resolve_roster_voices` runs for real against a fake
*profile service* (mirroring `test_briefing_voices.py`'s own rule), and
everything else, including a real, file-backed `SubscriptionsDB` and the
real stitcher, is exercised as-is.
"""

from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Subscriptions.briefing_cast import STATUS_COMPLETE as _SCRIPT_STATUS_COMPLETE
from tldw_chatbook.Subscriptions.briefing_voices import (
    VoiceResolutionError,
    VoiceSelection,
    dump_voice_snapshot,
    resolve_roster_voices,
)
from tldw_chatbook.TTS.adapter_types import TTSRequest
from tldw_chatbook.TTS.audio_stitch import (
    AudioStitchError,
    concat_wav_segments,
    wav_duration_seconds,
)
from tldw_chatbook.TTS.legacy_bridge import LEGACY_PROVIDER_IDS
from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot
from tldw_chatbook.TTS.text_processing import TextChunker
from tldw_chatbook.Utils.private_paths import (
    atomic_private_write_bytes,
    secure_private_directory,
)

#: Turn text longer than this many characters is split into multiple
#: synthesis requests (see `_split_turn_text`) and the resulting WAV pieces
#: stitched back into one payload with `concat_wav_segments`.
MAX_TURN_CHARS = 1800

#: `TextChunker`'s own per-chunk token budget for a turn that crosses
#: `MAX_TURN_CHARS`. `TextChunker` estimates "tokens" as `words * 1.3`
#: (`TTS/text_processing.py`'s `TOKENS_PER_WORD`), not a real tokenizer. A
#: turn that just crosses `MAX_TURN_CHARS` is roughly 300 words -- about 390
#: estimated tokens -- so the backends' own default budget of 500
#: (`TTS/backends/kokoro.py`, `higgs.py`) would frequently NOT split it at
#: all, silently defeating the point of gating on `MAX_TURN_CHARS` in the
#: first place. 200 is comfortably below that, so any turn that trips the
#: character gate reliably splits into at least two pieces.
_CHUNK_MAX_TOKENS = 200

#: Milliseconds of silence `concat_wav_segments` inserts between stitched
#: chunks of one long turn. Matches that function's own default explicitly
#: so a caller reading this module does not need to cross-reference it.
_CHUNK_GAP_MS = 350

#: The one legacy provider whose shared-builder format override
#: (`TTS.legacy_request_builder._LEGACY_FORMAT_OVERRIDES`) does not produce
#: a WAV container. See the module docstring's "ElevenLabs never produces
#: a WAV container on its own" section.
_ELEVENLABS_PROVIDER_ID = "elevenlabs"

#: `TTS.backends.elevenlabs.ElevenLabsTTSBackend._map_output_format`'s fixed
#: `simple_format_map` mapping for a bare `"pcm"` request -- the wire-level
#: ElevenLabs output format name this module's `"pcm"` override actually
#: resolves to, headerless 16-bit mono PCM. Named as a constant so the
#: sample rate below is parsed out of its own name rather than a second,
#: separately hardcoded number that could quietly drift out of sync with
#: it.
_ELEVENLABS_PCM_WIRE_FORMAT_NAME = "pcm_44100"


def _elevenlabs_pcm_sample_rate() -> int:
    """Sample rate ElevenLabs' bare "pcm" wire format actually returns.

    Parsed out of `_ELEVENLABS_PCM_WIRE_FORMAT_NAME`'s own name
    (`"pcm_44100"` -> `44100`) rather than a second, separately hardcoded
    number, so the two can never quietly drift apart.

    Returns:
        The sample rate, in Hz, encoded in
        `_ELEVENLABS_PCM_WIRE_FORMAT_NAME`'s name.
    """
    _, _, rate = _ELEVENLABS_PCM_WIRE_FORMAT_NAME.partition("_")
    return int(rate)


def _wrap_pcm16_mono_as_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap headerless 16-bit mono PCM samples in a minimal WAV container.

    No `pydub`/ffmpeg involved: byte-for-byte the same header shape as
    `TTS/audio_service.py`'s `AudioService._pcm_to_wav`, reproduced here
    rather than instantiating an `AudioService` for one private method.

    Args:
        pcm_data: Headerless 16-bit little-endian mono PCM samples.
        sample_rate: The sample rate the samples were generated at.

    Returns:
        `pcm_data` prefixed with a RIFF/WAVE header describing it.
    """
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = bytearray()
    header.extend(b"RIFF")
    header.extend((36 + len(pcm_data)).to_bytes(4, "little"))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))
    header.extend((1).to_bytes(2, "little"))
    header.extend(channels.to_bytes(2, "little"))
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(bits_per_sample.to_bytes(2, "little"))
    header.extend(b"data")
    header.extend(len(pcm_data).to_bytes(4, "little"))
    return bytes(header) + pcm_data


class TurnSynthesisError(RuntimeError):
    """Raised when one script turn's audio cannot be synthesized.

    Every raise site in this module names both `selection.speaker` and the
    turn's 0-based `turn_index`, so a caller (Task 6) can surface exactly
    which turn of a long briefing failed rather than a generic "synthesis
    failed" message.
    """


def _hard_split_piece(piece: str) -> list[str]:
    """Force `piece` under `MAX_TURN_CHARS`, splitting whitespace-blind text.

    `TextChunker` groups by sentence/word boundaries and budgets on an
    estimated word/token count, not a character count -- so a piece with no
    sentence-ending punctuation and no whitespace at all (dense CJK text, or
    a pathological space-free run) comes back as a single oversized `str`
    regardless of the token budget it was given (its own word-count
    estimate for such a piece is `1`, nowhere near any reasonable budget).
    This is the hard backstop that makes `MAX_TURN_CHARS` an actual cap on
    every synthesis request, not just a hint to the chunker.

    Args:
        piece: One `TextChunker` piece (or a whole turn, when unchunked).

    Returns:
        `[piece]` unchanged when already at or under `MAX_TURN_CHARS`.
        Otherwise, `piece` cut into consecutive windows of at most
        `MAX_TURN_CHARS` characters each: a cut prefers the last whitespace
        character in the back half of each window (so an ordinary word is
        not sliced mid-token when a break is available), falling back to a
        hard cut at exactly `MAX_TURN_CHARS` when no whitespace exists in
        that window at all.
    """
    if len(piece) <= MAX_TURN_CHARS:
        return [piece]

    pieces: list[str] = []
    remaining = piece
    search_floor = MAX_TURN_CHARS // 2
    while len(remaining) > MAX_TURN_CHARS:
        window = remaining[:MAX_TURN_CHARS]
        break_at = None
        for index in range(len(window) - 1, search_floor, -1):
            if window[index].isspace():
                break_at = index
                break
        if break_at is not None:
            pieces.append(remaining[:break_at].rstrip())
            remaining = remaining[break_at:].lstrip()
        else:
            pieces.append(window)
            remaining = remaining[MAX_TURN_CHARS:]
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_turn_text(text: str) -> list[str]:
    """Split one turn's text into synthesis-sized pieces.

    Args:
        text: The turn's full script text.

    Returns:
        `[text]` unchanged when `text` is at most `MAX_TURN_CHARS`
        characters (the common case, and the only case that never needs
        `concat_wav_segments` -- see the module docstring on why that
        matters for the optional `pydub` dependency). Otherwise, the
        non-empty pieces `TextChunker` splits `text` into, each further
        passed through `_hard_split_piece` so no returned piece ever
        exceeds `MAX_TURN_CHARS` characters.
    """
    if len(text) <= MAX_TURN_CHARS:
        return [text]

    chunker = TextChunker(max_tokens=_CHUNK_MAX_TOKENS)
    pieces = [chunk.text for chunk in chunker.chunk_text(text) if chunk.text.strip()]
    pieces = pieces or [text]

    capped: list[str] = []
    for candidate in pieces:
        capped.extend(_hard_split_piece(candidate))
    return capped


def _looks_like_wav(payload: bytes) -> bool:
    """Return whether `payload` decodes as a well-formed WAV container.

    Uses the stdlib `wave` module rather than `pydub`, so this check -- the
    only validation an unchunked legacy turn gets, per the module docstring
    -- never requires the optional `pydub`/`audioop-lts` extras
    (`TTS/audio_stitch.py`'s module docstring): the common case of a single-
    chunk legacy turn should not need the audio extra installed at all.

    Args:
        payload: Candidate audio bytes.

    Returns:
        `True` if `wave.open` can parse a RIFF/WAVE header and locate a
        `fmt `/`data` chunk pair; `False` for anything else, including empty
        bytes or a non-WAV codec.
    """
    try:
        with wave.open(BytesIO(payload), "rb"):
            return True
    except Exception:
        return False


def _validate_exact_snapshot(
    request: TTSRequest,
    snapshot: object,
    *,
    speaker: str,
    turn_index: int,
) -> None:
    """Reject an exact-path response whose provenance disagrees with the request.

    Mirrors `TTSEventHandler._validate_exact_selection`
    (`Event_Handlers/TTS_Events/tts_events.py:983`) field-for-field. The
    difference is only in how a mismatch is reported: that handler guards
    one global chat voice, so a provider-neutral contract error is enough;
    a briefing turn is one of many synthesis calls in flight, so a caller
    needs to know which speaker and turn misbehaved, and what voice the
    provider actually used instead of the one requested.

    Args:
        request: The `TTSRequest` this turn's chunk was submitted with.
        snapshot: Whatever `TTSService.synthesize_exact` returned as
            provenance -- expected to be a `TTSRequestedSelectionSnapshot`.
        speaker: The roster speaker this turn belongs to.
        turn_index: The turn's 0-based index.

    Raises:
        TurnSynthesisError: If `snapshot` is not a
            `TTSRequestedSelectionSnapshot`, or if any of its fields differ
            from what `request` asked for.
    """
    if type(snapshot) is not TTSRequestedSelectionSnapshot:
        raise TurnSynthesisError(
            f"speaker {speaker!r} turn {turn_index}: TTS provider did not "
            "return a requested-selection snapshot"
        )

    expected = (
        request.provider_id,
        request.model_id,
        request.voice,
        request.response_format,
        request.speed,
        request.options,
    )
    actual = (
        snapshot.provider_id,
        snapshot.model_id,
        snapshot.voice_id,
        snapshot.response_format,
        snapshot.speed,
        snapshot.options,
    )
    if actual != expected:
        raise TurnSynthesisError(
            f"speaker {speaker!r} turn {turn_index}: TTS provider used voice "
            f"{snapshot.voice_id!r}, requested voice {request.voice!r}"
        )


async def _synthesize_exact_chunk(
    tts_service: Any,
    selection: VoiceSelection,
    text: str,
    *,
    turn_index: int,
) -> bytes:
    """Synthesize one chunk of text through the exact `audio_cpp` path.

    Reproduces `TTSEventHandler._generate_tts`'s `finally`
    (`tts_events.py:899-912`) shape: the response is closed exactly once,
    on every exit path -- success, a validation failure, or a drain failure
    -- and a *second* failure from closing itself never replaces the
    primary error; it is logged by type only. This is a manual
    `try`/`except`/`finally`, not `async with response:` -- see the module
    docstring's "Closing the response is load-bearing" section for why
    that substitution is unsafe here.

    Args:
        tts_service: The app's TTS service, duck-typed to `synthesize_exact`.
        selection: The speaker's resolved voice (`provider_id="audio_cpp"`).
        text: The chunk's text (a whole turn, or one `TextChunker` piece of
            a long turn).
        turn_index: The turn's 0-based index, for error messages.

    Returns:
        WAV-encoded audio bytes for this chunk.

    Raises:
        TurnSynthesisError: If the provider's returned provenance snapshot
            disagrees with what was requested, or if `synthesize_exact`
            itself raises (an adapter/network error, wrapped naming the
            speaker and turn, with the original preserved as `__cause__`).
    """
    request = TTSRequest(
        provider_id=selection.provider_id,
        model_id=selection.model_id,
        text=text,
        voice=selection.voice_id,
        response_format="wav",
        speed=selection.speed,
        options=selection.options,
    )
    try:
        response, snapshot = await tts_service.synthesize_exact(request)
    except Exception as exc:  # noqa: BLE001 - every provider failure is named by turn
        # `asyncio.CancelledError` is a `BaseException`, not an `Exception`
        # (since Python 3.8), so cancellation is never caught here.
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: TTS provider "
            f"call failed ({_error_text(exc)})"
        ) from exc

    primary_error: BaseException | None = None
    try:
        _validate_exact_snapshot(
            request,
            snapshot,
            speaker=selection.speaker,
            turn_index=turn_index,
        )
        chunks: list[bytes] = []
        async for piece in response.byte_stream:
            if piece:
                chunks.append(piece)
        return b"".join(chunks)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            await response.aclose()
        except BaseException:
            if primary_error is None:
                raise
            logger.warning(
                "Briefing turn synthesis: response close failed after {}",
                type(primary_error).__name__,
            )


async def _synthesize_legacy_chunk(
    tts_service: Any,
    selection: VoiceSelection,
    text: str,
    *,
    turn_index: int,
) -> bytes:
    """Synthesize one chunk of text through a legacy (non-exact) provider.

    There is no response object on this path -- `generate_audio_stream`
    yields a bare `AsyncIterator[bytes]` -- so this is the only validation a
    legacy chunk gets: a zero-byte result and a payload that fails to decode
    as WAV are the two failure modes `synthesize_exact`'s snapshot contract
    would otherwise have caught.

    Args:
        tts_service: The app's TTS service, duck-typed to
            `generate_audio_stream`.
        selection: The speaker's resolved voice (any non-`audio_cpp`
            provider).
        text: The chunk's text.
        turn_index: The turn's 0-based index, for error messages.

    Returns:
        WAV-encoded audio bytes for this chunk.

    Raises:
        TurnSynthesisError: If `selection.provider_id` is not a known legacy
            provider (see `TTS.legacy_bridge.LEGACY_PROVIDER_IDS`); if
            `selection.voice_id` is empty (legacy providers cannot resolve a
            server-default voice); if the provider/adapter call itself
            raises (wrapped naming the speaker and turn, with the original
            preserved as `__cause__` -- see the module docstring); if the
            provider returned no audio; or if the resulting payload does
            not decode as WAV.
    """
    if selection.provider_id not in LEGACY_PROVIDER_IDS:
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: unsupported "
            f"TTS provider {selection.provider_id!r}"
        )
    if not selection.voice_id:
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: no voice is "
            "selected for this provider"
        )

    request, internal_model_id = build_legacy_speech_request(
        provider_id=selection.provider_id,
        model_id=selection.model_id,
        voice=selection.voice_id,
        text=text,
        response_format=selection.response_format,
        speed=selection.speed,
    )
    if selection.provider_id == _ELEVENLABS_PROVIDER_ID:
        # The shared builder's format-override table forces elevenlabs to
        # "mp3" regardless of what was requested (preserved verbatim here --
        # see the module docstring's "ElevenLabs never produces a WAV
        # container on its own" section). Ask for "pcm" instead so the
        # payload below can be wrapped into a WAV container without pydub
        # or ffmpeg.
        request = request.model_copy(update={"response_format": "pcm"})

    try:
        stream = tts_service.generate_audio_stream(request, internal_model_id)
        payload = b"".join([piece async for piece in stream if piece])
    except Exception as exc:  # noqa: BLE001 - every provider failure is named by turn
        # `asyncio.CancelledError` is a `BaseException`, not an `Exception`
        # (since Python 3.8), so cancellation is never caught here.
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: TTS provider "
            f"call failed ({_error_text(exc)})"
        ) from exc

    if not payload:
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: TTS provider "
            "returned no audio"
        )
    if selection.provider_id == _ELEVENLABS_PROVIDER_ID:
        payload = _wrap_pcm16_mono_as_wav(payload, _elevenlabs_pcm_sample_rate())
    if not _looks_like_wav(payload):
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: TTS provider "
            "returned audio that is not valid WAV"
        )
    return payload


async def synthesize_turn(
    tts_service: Any,
    selection: VoiceSelection,
    text: str,
    *,
    turn_index: int,
) -> bytes:
    """Synthesize one script turn's WAV audio, on any configured provider.

    Routes through `TTSService.synthesize_exact` when
    `selection.is_exact_provider()`, and through
    `TTSService.generate_audio_stream` (via
    `TTS/legacy_request_builder.build_legacy_speech_request`) otherwise. A
    turn longer than `MAX_TURN_CHARS` characters is split into multiple
    chunks (`_split_turn_text`), each synthesized independently and then
    stitched into one payload with `TTS/audio_stitch.concat_wav_segments`.

    Args:
        tts_service: The app's TTS service, duck-typed to `synthesize_exact`
            and `generate_audio_stream` (`TTSService`,
            `TTS/TTS_Generation.py`).
        selection: The speaker's resolved voice
            (`briefing_voices.VoiceSelection`).
        text: The turn's script text.
        turn_index: The turn's 0-based index within its script, used only
            for error messages.

    Returns:
        WAV-encoded audio bytes covering the whole turn.

    Raises:
        TurnSynthesisError: Naming `selection.speaker` and `turn_index` --
            see `_synthesize_exact_chunk` and `_synthesize_legacy_chunk` for
            the path-specific failure modes; also raised if a multi-chunk
            turn's pieces cannot be stitched into one WAV payload.
    """
    pieces = _split_turn_text(text)

    segments: list[bytes] = []
    for piece in pieces:
        if selection.is_exact_provider():
            segment = await _synthesize_exact_chunk(
                tts_service,
                selection,
                piece,
                turn_index=turn_index,
            )
        else:
            segment = await _synthesize_legacy_chunk(
                tts_service,
                selection,
                piece,
                turn_index=turn_index,
            )
        segments.append(segment)

    if len(segments) == 1:
        return segments[0]

    try:
        return concat_wav_segments(segments, gap_ms=_CHUNK_GAP_MS)
    except AudioStitchError as exc:
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: could not "
            f"stitch synthesized audio pieces ({type(exc).__name__})"
        ) from exc


# ---------------------------------------------------------------------------
# Task 6: orchestrating one script's turns into one stored, playable file.
# ---------------------------------------------------------------------------

#: Statuses a `briefing_audio` row can hold. Mirrors `briefing_cast`'s own
#: three-status shape exactly -- there is no "empty" status here either,
#: since a script with no turns is refused before any row is ever written
#: (see `_load_script_for_audio`).
STATUS_GENERATING = "generating"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

#: The error text a zombie `generating` row is failed with by
#: `fail_interrupted_audio` (mirrors `briefing_cast.INTERRUPTED_ERROR`
#: exactly).
INTERRUPTED_ERROR = "interrupted"

#: Every stored error message -- a `TurnSynthesisError`, a
#: `VoiceResolutionError`, an `AudioGenerationError`, or any other in-band
#: failure -- is capped at this many characters before it reaches a
#: `briefing_audio` row (mirrors `briefing_cast.ERROR_CHAR_CAP` exactly):
#: the row is rendered in a status line, not a log file, and some failures
#: (a provider's raw error body) can be arbitrarily long.
ERROR_CHAR_CAP = 1000

#: `briefing_audio_dir()`'s directory name under `get_user_data_dir()`.
_AUDIO_SUBDIR = "briefing_audio"


class AudioGenerationError(RuntimeError):
    """Raised when a script cannot be turned into audio at all.

    Every raise site reached directly from `generate_script_audio` (via
    `_load_script_for_audio`) is a pre-flight refusal -- the script does
    not exist, is not `complete`, or has no turns a synthesis attempt could
    ever read -- so those raise sites all run BEFORE any `briefing_audio`
    row exists (mirrors `briefing_cast.ScriptCastError`'s pre-row-insert
    raise sites in `_start_script`). A turn naming a speaker absent from
    the resolved voice snapshot also raises this type, but that raise site
    runs INSIDE `generate_script_audio`'s own try/except, after the row
    already exists, so it becomes a `failed` row rather than propagating.
    """


def _error_text(exc: BaseException) -> str:
    """The exception's message, capped -- never a traceback.

    Copies `briefing_cast._error_text`'s exact shape (not imported: that
    function is private to its own module).

    Args:
        exc: The exception to render.

    Returns:
        `str(exc)`, stripped, falling back to the exception's class name
        when that is empty, truncated to `ERROR_CHAR_CAP` characters with a
        `" [...]"` suffix when longer.
    """
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > ERROR_CHAR_CAP:
        message = message[:ERROR_CHAR_CAP] + " [...]"
    return message


def _parse_turns(turns_json: str | None) -> list[dict[str, str]]:
    """Decode a script's stored `turns_json` into synthesis-ready turns.

    Args:
        turns_json: A `briefing_scripts.turns_json` value -- expected to be
            `briefing_cast.parse_script_turns`'s own `json.dumps` output, a
            JSON array of `{"speaker": str, "text": str}` objects.

    Returns:
        The decoded turns, in stored order, each a plain
        `{"speaker": str, "text": str}` dict (the speaker name stripped of
        incidental whitespace).

    Raises:
        AudioGenerationError: If `turns_json` is `None`/empty, is not valid
            JSON, is not a non-empty JSON array, or contains an item
            missing a non-empty string `speaker` or a string `text`.
    """
    if not turns_json:
        raise AudioGenerationError("script has no turns to synthesize")
    try:
        payload = json.loads(turns_json)
    except (ValueError, TypeError) as exc:
        raise AudioGenerationError("script turns are not valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise AudioGenerationError("script has no turns to synthesize")

    turns: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        speaker = item.get("speaker") if isinstance(item, Mapping) else None
        text = item.get("text") if isinstance(item, Mapping) else None
        if not isinstance(speaker, str) or not speaker.strip() or not isinstance(text, str):
            raise AudioGenerationError(f"script turn {index} is malformed")
        turns.append({"speaker": speaker.strip(), "text": text})
    return turns


def _parse_roster_snapshot(roster_snapshot_json: str | None) -> list[dict[str, Any]]:
    """Decode a script's stored `roster_snapshot_json` for voice resolution.

    Args:
        roster_snapshot_json: A `briefing_scripts.roster_snapshot_json`
            value -- expected to be `briefing_cast.dump_roster`'s own
            output, a JSON array of speaker objects.

    Returns:
        The decoded roster, in stored order, each entry a plain `dict`.

    Raises:
        AudioGenerationError: If the value is not valid JSON, or is not a
            JSON array of objects.
    """
    try:
        payload = json.loads(roster_snapshot_json or "[]")
    except (ValueError, TypeError) as exc:
        raise AudioGenerationError("script roster snapshot is not valid JSON") from exc
    if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
        raise AudioGenerationError(
            "script roster snapshot must be an array of speaker objects"
        )
    return [dict(item) for item in payload]


def _load_script_for_audio(
    db: Any, script_id: int
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]]]:
    """Everything before any `briefing_audio` row exists: fetch, validate, parse.

    Grouped into one function so `generate_script_audio` can run it as a
    single `asyncio.to_thread` hop (2a's whole-branch ruling): every check
    that must refuse WITHOUT ever creating a row runs here.

    Args:
        db: An open `SubscriptionsDB`.
        script_id: The `briefing_scripts.id` to synthesize audio for.

    Returns:
        `(script, turns, roster_snapshot)`.

    Raises:
        AudioGenerationError: If the script does not exist, is not
            `complete`, or its stored turns/roster snapshot cannot be
            parsed into usable data. No `briefing_audio` row is ever
            written when this raises.
    """
    script = db.get_briefing_script(script_id)
    if script is None:
        raise AudioGenerationError(f"script {script_id} does not exist")
    if script["status"] != _SCRIPT_STATUS_COMPLETE:
        raise AudioGenerationError(
            f"script {script_id} is {script['status']!r}, not complete; audio can "
            "only be generated from a complete script"
        )

    turns = _parse_turns(script.get("turns_json"))
    roster_snapshot = _parse_roster_snapshot(script.get("roster_snapshot_json"))
    return script, turns, roster_snapshot


def _record_voice_resolution_failure(db: Any, script_id: int, message: str) -> dict[str, Any]:
    """Write a `failed` `briefing_audio` row for a voice resolution failure.

    `resolve_roster_voices` must succeed before `generate_script_audio` has
    anything to pass as `voice_snapshot_json` -- `create_briefing_audio`'s
    own write-once contract (Task 1) means a row can never be created
    first and "filled in" with the resolved snapshot afterward. So a
    resolution failure creates its OWN row here, directly, with an empty
    placeholder snapshot -- there is no meaningful voice assignment to
    record for an attempt that never resolved one.

    Args:
        db: An open `SubscriptionsDB`.
        script_id: The script this audio attempt belongs to.
        message: The capped, human-readable failure text (`_error_text`'s
            output) -- already naming the speaker and, for a stored profile
            id that no longer resolves, the id too (see
            `briefing_voices.VoiceResolutionError`).

    Returns:
        The finished (`failed`) `briefing_audio` row as a dict.
    """
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(audio_id, status=STATUS_FAILED, error=message)
    return db.get_briefing_audio(audio_id)


def _finish_audio_failure(db: Any, audio_id: int, message: str) -> dict[str, Any]:
    """Record an in-band synthesis failure on an already-existing row.

    Touches only this `briefing_audio` row -- the `briefing_scripts` row it
    narrates is never written by this module, on any outcome (spec §Error
    handling ethos; the named invariant
    `test_a_failed_synthesis_never_touches_the_script`).

    Args:
        db: An open `SubscriptionsDB`.
        audio_id: The `briefing_audio.id` row to fail.
        message: The capped, human-readable failure text.

    Returns:
        The finished (`failed`) `briefing_audio` row as a dict.
    """
    db.update_briefing_audio(audio_id, status=STATUS_FAILED, error=message)
    return db.get_briefing_audio(audio_id)


def _finish_audio_success(
    db: Any, audio_id: int, file_path: str, duration_seconds: float, turn_count: int
) -> dict[str, Any]:
    """Record a completed render and read the finished row back.

    Args:
        db: An open `SubscriptionsDB`.
        audio_id: The `briefing_audio.id` row to complete.
        file_path: Absolute path of the written WAV file.
        duration_seconds: The stitched payload's duration.
        turn_count: How many turns the finished render covers.

    Returns:
        The finished (`complete`) `briefing_audio` row as a dict.
    """
    db.update_briefing_audio(
        audio_id,
        status=STATUS_COMPLETE,
        file_path=file_path,
        duration_seconds=duration_seconds,
        turn_count=turn_count,
    )
    return db.get_briefing_audio(audio_id)


def _remove_file_quietly(path: Path) -> None:
    """Best-effort delete of a possibly-already-absent file.

    Used to clean up an orphaned audio file after the write succeeded but a
    later step (a duration read, or the finalizing DB write) failed -- a
    `failed` row must never leave an artifact on disk behind it. Never
    raises: a failure to remove a stray file is logged (by type only) and
    swallowed, since the caller is already on a failure path of its own and
    must not lose that original error to a cleanup problem.

    Args:
        path: The audio file to remove.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(
            "Briefing audio cleanup: could not remove an orphaned file ({}).",
            type(exc).__name__,
        )


def briefing_audio_dir() -> Path:
    """Return the secured, application-owned directory for rendered audio.

    Returns:
        `<user data dir>/briefing_audio`, created if missing and hardened
        to this application's own private-directory posture (only this
        user, only this app -- see
        `Utils.private_paths.secure_private_directory`).
    """
    return secure_private_directory(
        get_user_data_dir() / _AUDIO_SUBDIR,
        create=True,
        application_owned=True,
    ).lexical_path


async def generate_script_audio(
    db: Any,
    script_id: int,
    *,
    tts_service: Any,
    profile_service: Any | None,
    synthesize: Callable[..., Any] = synthesize_turn,
) -> dict[str, Any]:
    """Synthesize a cast script's turns into one stored, playable audio file.

    Never raises for an in-band failure once it has committed to an
    attempt (a voice that fails to resolve, a turn's synthesis error, an
    unassigned speaker, a stitch failure, a write failure): the failure
    becomes that SAME `briefing_audio` row's status and error, mirroring
    `briefing_cast.generate_script`'s own contract exactly. It DOES raise
    `AudioGenerationError` for a request that never should have started an
    attempt at all -- the script isn't `complete`, or has no readable turns
    -- and in that case no `briefing_audio` row is ever written. See the
    module docstring's "Task 6 adds this module's pipeline half" section
    for the full error-boundary shape this copies.

    Args:
        db: An open `SubscriptionsDB`.
        script_id: The `briefing_scripts.id` to synthesize. Must be
            `complete`.
        tts_service: The app's TTS service, passed straight through to
            `synthesize` -- never used directly by this function.
        profile_service: The app's TTS profile service, passed straight
            through to `briefing_voices.resolve_roster_voices`.
        synthesize: The per-turn synthesis seam. Defaults to
            `synthesize_turn`; per the module docstring, the only seam a
            test needs to fake to exercise this whole orchestration
            against a real `SubscriptionsDB` and the real stitcher.

    Returns:
        The finished `briefing_audio` row as a dict, whatever its status.

    Raises:
        AudioGenerationError: If the script does not exist, is not
            `complete`, or its stored turns cannot be parsed. No row is
            written in any of these cases.
    """
    script, turns, roster_snapshot = await asyncio.to_thread(
        _load_script_for_audio, db, script_id
    )

    try:
        selections = await resolve_roster_voices(
            roster_snapshot, profile_service=profile_service
        )
    except VoiceResolutionError as exc:
        # No message content logged: see the module docstring's egress
        # note -- a `VoiceResolutionError`'s own message names the speaker
        # (and, for a deleted profile, its id).
        logger.warning(f"script {script_id}: voice resolution failed: {type(exc).__name__}")
        return await asyncio.to_thread(
            _record_voice_resolution_failure, db, script_id, _error_text(exc)
        )

    audio_id = await asyncio.to_thread(
        db.create_briefing_audio,
        script_id,
        voice_snapshot_json=dump_voice_snapshot(selections),
    )

    by_speaker: dict[str, VoiceSelection] = {
        selection.speaker: selection for selection in selections
    }

    try:
        segments: list[bytes] = []
        for index, turn in enumerate(turns):
            speaker = turn["speaker"]
            selection = by_speaker.get(speaker)
            if selection is None:
                raise AudioGenerationError(
                    f"turn {index}: no voice assigned for speaker {speaker!r}"
                )
            segment = await synthesize(tts_service, selection, turn["text"], turn_index=index)
            segments.append(segment)
        payload = concat_wav_segments(segments)
    except Exception as exc:  # noqa: BLE001 - every synthesis failure is a row
        # No message content logged: a `TurnSynthesisError`'s own message
        # names the speaker and turn index -- that is a caller-facing
        # error (stored on the row), not a log line.
        logger.warning(
            f"script {script_id} audio {audio_id}: synthesis failed: {type(exc).__name__}"
        )
        return await asyncio.to_thread(_finish_audio_failure, db, audio_id, _error_text(exc))

    directory = await asyncio.to_thread(briefing_audio_dir)
    path = directory / f"script-{script_id}-audio-{audio_id}.wav"
    try:
        await asyncio.to_thread(
            atomic_private_write_bytes,
            path,
            payload,
            application_owned_directory=directory,
        )
    except Exception as exc:  # noqa: BLE001 - a write failure is a row, not a raise
        logger.warning(
            f"script {script_id} audio {audio_id}: audio write failed: {type(exc).__name__}"
        )
        return await asyncio.to_thread(_finish_audio_failure, db, audio_id, _error_text(exc))

    try:
        duration = wav_duration_seconds(payload)
    except AudioStitchError as exc:
        logger.warning(
            f"script {script_id} audio {audio_id}: duration read failed: {type(exc).__name__}"
        )
        await asyncio.to_thread(_remove_file_quietly, path)
        return await asyncio.to_thread(_finish_audio_failure, db, audio_id, _error_text(exc))

    try:
        return await asyncio.to_thread(
            _finish_audio_success, db, audio_id, str(path), duration, len(turns)
        )
    except Exception:
        # A genuine DB error finalizing the row: propagate uncaught (the
        # caller's worker wraps it -- see the module docstring) but the
        # file must not be left orphaned on disk behind a row stuck
        # `generating` forever.
        await asyncio.to_thread(_remove_file_quietly, path)
        raise


def fail_interrupted_audio(db: Any, script_id: int | None = None) -> int:
    """Fail every `generating` audio row as `interrupted`; return the count.

    Mirrors `briefing_cast.fail_interrupted_scripts` exactly: a worker that
    crashed mid-render leaves a `generating` row that would otherwise wedge
    a one-render-at-a-time guard shut forever. Only `generating` rows are
    touched -- finished history keeps its status, its file, and its own
    error text.

    Args:
        db: An open `SubscriptionsDB`.
        script_id: Scope the sweep to one script's audio rows. `None`
            sweeps every script's audio, which is what a startup pass
            wants.

    Returns:
        How many rows were failed.
    """
    sql = (
        "UPDATE briefing_audio SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE status = ?"
    )
    params: list[Any] = [STATUS_FAILED, INTERRUPTED_ERROR, STATUS_GENERATING]
    if script_id is not None:
        sql += " AND script_id = ?"
        params.append(script_id)

    with db.transaction() as conn:
        count = conn.execute(sql, params).rowcount
    if count:
        logger.info(f"failed {count} interrupted briefing audio row(s)")
    return count
