"""General-purpose WAV decode-and-concat stitching for multi-turn audio.

Briefing audio is synthesized one turn at a time (one speaker, one voice, one
provider request per turn) and must be joined into a single playable file.
:meth:`tldw_chatbook.TTS.audiobook_generator.AudioBookGenerator._combine_segments`
concatenates *encoded* bytes (``combined += segment``); for WAV that produces
a file whose RIFF header still describes only the first segment, so most
players report the wrong length and stop early. The only correct decode-based
concat in the repo today is welded inside
:meth:`tldw_chatbook.TTS.audio_service.AudioService.create_m4b_with_chapters`
and is M4B-specific. :func:`concat_wav_segments` is that same
decode-then-re-encode idea, generalized: it decodes every input with pydub,
joins the decoded segments (with silence between them), and re-encodes once.

:func:`wav_duration_seconds` reports the duration of a WAV payload via pydub
rather than :meth:`tldw_chatbook.TTS.audiobook_generator.AudioBookGenerator._get_audio_duration`,
which is a private method on a heavyweight class whose mutagen/size-estimate
ladder is built for files already on disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


class AudioStitchError(RuntimeError):
    """Raised when WAV audio cannot be stitched or decoded.

    Every message names the 0-based index of the offending segment (within
    the sequence passed to :func:`concat_wav_segments`) so callers — notably
    per-turn synthesis — can report which turn failed, not merely that one did.
    """


def concat_wav_segments(segments: Sequence[bytes], *, gap_ms: int = 350) -> bytes:
    """Decode WAV segments and re-encode them as one continuous WAV payload.

    Each segment is decoded independently, so the segments may differ in
    sample rate or channel count; pydub resamples/upmixes as needed when the
    decoded segments are joined.

    Args:
        segments: WAV-encoded audio bytes, one entry per turn (or per chunk
            of a turn), in the order they should play.
        gap_ms: Milliseconds of silence to insert between consecutive
            segments. No gap is added after the final segment. A value of
            ``0`` joins segments with no silence at all.

    Returns:
        A single WAV-encoded payload covering every input segment in order,
        separated by ``gap_ms`` of silence.

    Raises:
        AudioStitchError: If ``segments`` is empty, or if any segment fails
            to decode as audio; the message names its 0-based index.
    """
    if not segments:
        raise AudioStitchError("concat_wav_segments received no segments to stitch")

    decoded: list[AudioSegment] = []
    for index, segment in enumerate(segments):
        try:
            decoded.append(AudioSegment.from_file(BytesIO(segment), format="wav"))
        except (CouldntDecodeError, OSError, ValueError) as exc:
            raise AudioStitchError(
                f"Segment {index} could not be decoded as WAV audio: {type(exc).__name__}"
            ) from exc

    combined = decoded[0]
    if len(decoded) > 1:
        gap = (
            AudioSegment.silent(duration=gap_ms, frame_rate=decoded[0].frame_rate)
            if gap_ms > 0
            else None
        )
        for segment in decoded[1:]:
            if gap is not None:
                combined = combined + gap
            combined = combined + segment

    buffer = BytesIO()
    combined.export(buffer, format="wav")
    return buffer.getvalue()


def wav_duration_seconds(payload: bytes) -> float:
    """Return the duration, in seconds, of a WAV payload.

    Args:
        payload: WAV-encoded audio bytes.

    Returns:
        The decoded duration in seconds.

    Raises:
        AudioStitchError: If ``payload`` cannot be decoded as WAV audio.
    """
    try:
        audio = AudioSegment.from_file(BytesIO(payload), format="wav")
    except (CouldntDecodeError, OSError, ValueError) as exc:
        raise AudioStitchError(
            f"Payload could not be decoded as WAV audio: {type(exc).__name__}"
        ) from exc
    return len(audio) / 1000.0
