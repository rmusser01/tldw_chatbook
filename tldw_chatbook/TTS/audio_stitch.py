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

**pydub is optional, and this module must import cleanly without it.** pydub
is only installed via the ``audio``/``local_tts`` extras (see
``pyproject.toml`` and :mod:`tldw_chatbook.Utils.optional_deps`), and
``UI/Screens/screen_registry.py``'s ``load_screen_class`` swallows
``ImportError`` from any module in a screen's import chain and silently
drops the whole screen. A module-scope ``import pydub`` here would therefore
take down the entire Watchlists Collections screen for a user who never
installed the audio extra, not merely disable a Synthesize button. Following
the established pattern in :mod:`tldw_chatbook.TTS.audio_service`
(``PYDUB_AVAILABLE``), the import below is guarded and both public functions
raise :class:`AudioStitchError` at *call* time, naming the missing extra,
rather than failing at import time.

**ffmpeg is not required for this module.** pydub has a general reputation
for shelling out to an ffmpeg/avconv binary, but that is not true of the
pure-WAV-in/WAV-out path used here: ``AudioSegment.from_file(..., format=
"wav")`` decodes through pydub's own stdlib-``wave``-based parser (its
"safe wav" path), ``export(..., format="wav")`` with no codec/parameters
writes through the stdlib ``wave`` module directly (pydub's "easy wav"
path), and segment concatenation/resampling operate on raw PCM bytes via the
stdlib ``audioop`` module (``audioop-lts`` on Python >= 3.13, where stdlib
``audioop`` was removed). None of that shells out to ffmpeg — verified
empirically by pointing ``AudioSegment.converter`` at a nonexistent binary
and confirming stitching (including mixed-sample-rate resampling) still
succeeds. Only ``pydub`` (plus ``audioop-lts`` on Python >= 3.13) is required
to use this module; ffmpeg is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from loguru import logger

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError

    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None  # type: ignore[assignment, misc]
    CouldntDecodeError = Exception  # type: ignore[assignment, misc]
    PYDUB_AVAILABLE = False
    logger.warning(
        "pydub not available. Audio stitching (tldw_chatbook.TTS.audio_stitch) "
        "will be unavailable until it is installed."
    )

_PYDUB_MISSING_MESSAGE = (
    "Audio stitching requires the optional 'pydub' dependency, which is not "
    "installed. Install it with: pip install tldw_chatbook[audio] "
    "(or: pip install pydub audioop-lts, on Python 3.13+)."
)


class AudioStitchError(RuntimeError):
    """Raised when WAV audio cannot be stitched, decoded, or pydub is missing.

    Args:
        message: Human-readable description of what went wrong.
        segment_index: 0-based index, within the sequence passed to
            :func:`concat_wav_segments`, of the segment that failed to
            decode. ``None`` when the error is not about one specific
            segment (an empty input sequence, a missing pydub install, or a
            plain duration lookup via :func:`wav_duration_seconds`).

    Attributes:
        segment_index: See ``Args`` above. Callers — notably per-turn
            synthesis reporting "turn N" to a user — should read this
            attribute rather than parse the message to identify which
            segment failed.
    """

    def __init__(self, message: str, *, segment_index: int | None = None) -> None:
        super().__init__(message)
        self.segment_index = segment_index


def _require_pydub() -> None:
    """Raise :class:`AudioStitchError` if pydub failed to import.

    Raises:
        AudioStitchError: Always, when :data:`PYDUB_AVAILABLE` is ``False``.
    """
    if not PYDUB_AVAILABLE:
        raise AudioStitchError(_PYDUB_MISSING_MESSAGE)


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
        AudioStitchError: If pydub is not installed; if ``segments`` is
            empty; or if any segment fails to decode as audio, in which case
            ``segment_index`` on the raised exception names its 0-based
            index.
    """
    _require_pydub()

    if not segments:
        raise AudioStitchError("concat_wav_segments received no segments to stitch")

    decoded: list[AudioSegment] = []
    for index, segment in enumerate(segments):
        try:
            decoded.append(AudioSegment.from_file(BytesIO(segment), format="wav"))
        except (CouldntDecodeError, OSError, ValueError) as exc:
            raise AudioStitchError(
                f"Segment {index} could not be decoded as WAV audio: {type(exc).__name__}",
                segment_index=index,
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
        AudioStitchError: If pydub is not installed, or if ``payload`` cannot
            be decoded as WAV audio.
    """
    _require_pydub()

    try:
        audio = AudioSegment.from_file(BytesIO(payload), format="wav")
    except (CouldntDecodeError, OSError, ValueError) as exc:
        raise AudioStitchError(
            f"Payload could not be decoded as WAV audio: {type(exc).__name__}"
        ) from exc
    return len(audio) / 1000.0
