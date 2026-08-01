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

Testing: the only faked seam is the TTS service (`synthesize_exact` and
`generate_audio_stream`) -- everything else, including the real
`TTSAudioResponse` and the real `TextChunker`/`concat_wav_segments`, is
exercised as-is.

Nothing here logs turn text, a speaker name, or a voice id at any level --
only exception types, for the same reason `briefing_cast` and
`briefing_voices` avoid it: this app's log sink runs with `diagnose=True`,
which dumps a failing frame's locals, and the frame at a synthesis failure
holds exactly that content. `TurnSynthesisError` *messages* do name the
speaker and turn index -- that is a caller-facing error, not a log line --
but never a voice id or the turn's text.
"""

from __future__ import annotations

import wave
from io import BytesIO
from typing import Any

from loguru import logger

from tldw_chatbook.Subscriptions.briefing_voices import VoiceSelection
from tldw_chatbook.TTS.adapter_types import TTSRequest
from tldw_chatbook.TTS.audio_stitch import AudioStitchError, concat_wav_segments
from tldw_chatbook.TTS.legacy_bridge import LEGACY_PROVIDER_IDS
from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot
from tldw_chatbook.TTS.text_processing import TextChunker

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
            disagrees with what was requested.
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
    response, snapshot = await tts_service.synthesize_exact(request)

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
            server-default voice); if the provider returned no audio; or if
            the joined bytes do not decode as WAV.
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
    stream = tts_service.generate_audio_stream(request, internal_model_id)
    payload = b"".join([piece async for piece in stream if piece])

    if not payload:
        raise TurnSynthesisError(
            f"speaker {selection.speaker!r} turn {turn_index}: TTS provider "
            "returned no audio"
        )
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
