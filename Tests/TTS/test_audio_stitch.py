"""Tests for the general WAV decode-and-concat stitcher.

``concat_wav_segments`` exists because the only concat logic already in the
repo is wrong for this job: ``AudioBookGenerator._combine_segments``
(``audiobook_generator.py:638``) does ``combined += segment`` on *encoded*
bytes, which for WAV leaves a RIFF header describing only the first segment
— most players report the wrong length and stop early. The load-bearing test
below (``test_concat_produces_a_header_whose_duration_matches_the_inputs``)
decodes the stitched result and checks its actual frame count/duration
against the sum of the inputs plus gaps; a naive ``b"".join(...)`` decodes to
only the first segment's duration, so that test must fail against it (see
``test_naive_byte_concat_would_fail_the_header_assertion``, which pins that
property directly against a hand-rolled naive join so the mutation check has
a fast, in-suite confirmation independent of manually reverting the module).

Every WAV input is built in-process with ``pydub.AudioSegment.silent(...)``
exported to ``BytesIO`` — no fixtures on disk, no network.
"""

from __future__ import annotations

import importlib
import sys
from io import BytesIO

import pytest
from pydub import AudioSegment

import tldw_chatbook.TTS.audio_stitch as audio_stitch
from tldw_chatbook.TTS.audio_stitch import (
    AudioStitchError,
    concat_wav_segments,
    wav_duration_seconds,
)

pytestmark = pytest.mark.unit

_DURATION_TOLERANCE_SECONDS = 0.05


def _silence_wav(duration_ms: int, frame_rate: int = 22050) -> bytes:
    """Build a real WAV payload of ``duration_ms`` of silence, in-process."""
    segment = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)
    buffer = BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def _decode(payload: bytes) -> AudioSegment:
    """Decode a WAV payload back into a pydub segment for assertions."""
    return AudioSegment.from_file(BytesIO(payload), format="wav")


def test_two_segments_with_gap_sum_to_expected_duration() -> None:
    segments = [_silence_wav(500), _silence_wav(500)]

    result = concat_wav_segments(segments, gap_ms=200)

    assert wav_duration_seconds(result) == pytest.approx(1.2, abs=_DURATION_TOLERANCE_SECONDS)


def test_concat_produces_a_header_whose_duration_matches_the_inputs() -> None:
    """The load-bearing test: decode the result and check its frame count.

    A byte-concatenated (rather than decode-and-concat) WAV file's RIFF
    header still describes only the first segment, so decoding it yields
    only that segment's frames. Asserting on the decoded frame count (not
    just a duration estimate) is what catches that failure mode reliably.
    """
    segments = [_silence_wav(500), _silence_wav(500)]

    result = concat_wav_segments(segments, gap_ms=200)
    decoded = _decode(result)

    expected_ms = 500 + 200 + 500
    expected_frames = decoded.frame_rate * expected_ms / 1000.0
    assert decoded.frame_count() == pytest.approx(expected_frames, rel=0.01)
    assert len(decoded) == pytest.approx(expected_ms, abs=50)


def test_naive_byte_concat_would_fail_the_header_assertion() -> None:
    """Pin the failure mode the header assertion above exists to catch.

    This does not call ``concat_wav_segments`` — it directly reproduces
    ``AudioBookGenerator._combine_segments``'s ``combined += segment`` on
    encoded WAV bytes, so a reader (or the mutation check) can see, in-suite,
    that the naive approach decodes to only the first segment's duration
    rather than the sum.
    """
    first = _silence_wav(500)
    second = _silence_wav(500)

    naive = b"".join([first, second])
    decoded = _decode(naive)

    assert len(decoded) == pytest.approx(500, abs=5)
    assert len(decoded) != pytest.approx(1000, abs=5)


def test_single_segment_is_returned_unchanged_with_no_trailing_gap() -> None:
    segment = _silence_wav(500)

    result = concat_wav_segments([segment], gap_ms=200)

    assert wav_duration_seconds(result) == pytest.approx(0.5, abs=_DURATION_TOLERANCE_SECONDS)


def test_empty_sequence_raises_audio_stitch_error() -> None:
    with pytest.raises(AudioStitchError):
        concat_wav_segments([])


def test_undecodable_segment_names_its_zero_based_index() -> None:
    """Pin the failing index precisely, not via a loose substring check.

    A bare ``"1" in str(exc)`` would also pass for an unrelated "1" anywhere
    in the message (for example a byte count or a different segment's
    index). Task 6 surfaces this index to the user as "turn N", so an
    off-by-one must be catchable: assert both the typed ``segment_index``
    attribute and a specific, unambiguous message phrase.
    """
    segments = [_silence_wav(300), b"not audio", _silence_wav(300)]

    with pytest.raises(AudioStitchError) as exc_info:
        concat_wav_segments(segments)

    assert exc_info.value.segment_index == 1
    assert "Segment 1 " in str(exc_info.value)


def test_mixed_sample_rates_decode_and_sum_durations() -> None:
    low_rate = _silence_wav(400, frame_rate=8000)
    high_rate = _silence_wav(300, frame_rate=44100)

    result = concat_wav_segments([low_rate, high_rate], gap_ms=350)

    expected_seconds = (400 + 350 + 300) / 1000.0
    assert wav_duration_seconds(result) == pytest.approx(
        expected_seconds, abs=_DURATION_TOLERANCE_SECONDS
    )


def test_concat_wav_segments_raises_when_pydub_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both public functions must fail at call time, not import time.

    ``pydub`` is only an optional extra; a bare module-scope import would
    take down the entire Watchlists Collections screen for a user who never
    installed it (``screen_registry.load_screen_class`` swallows
    ``ImportError`` from any module in a screen's import chain). Patching the
    module's ``PYDUB_AVAILABLE`` flag simulates that absence for the call
    path without touching real import machinery.
    """
    monkeypatch.setattr(audio_stitch, "PYDUB_AVAILABLE", False)

    with pytest.raises(AudioStitchError, match="pydub"):
        concat_wav_segments([_silence_wav(100)])


def test_wav_duration_seconds_raises_when_pydub_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_stitch, "PYDUB_AVAILABLE", False)

    with pytest.raises(AudioStitchError, match="pydub"):
        wav_duration_seconds(_silence_wav(100))


def test_module_imports_successfully_without_pydub_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing import-safety test: no import-time dependency on pydub.

    Chosen approach: block ``pydub`` via the ``sys.modules[name] = None``
    convention — Python's import machinery raises ``ImportError`` immediately
    for any ``import``/``from ... import`` of a name mapped to ``None`` in
    ``sys.modules`` — then force a genuinely fresh import of
    ``tldw_chatbook.TTS.audio_stitch``. This is the honest version of the
    check: it does not merely assert the guard *flag* takes the right value
    (which could pass even if a stray, unguarded module-scope ``import
    pydub`` sat elsewhere in the file, so long as some other guarded import
    ran first) — it forces the module's actual top-level code to execute
    with pydub genuinely unreachable, the same condition
    ``screen_registry.load_screen_class`` creates for a user without the
    audio extra.

    ``monkeypatch.setitem``/``delitem`` record and restore ``sys.modules``'s
    exact prior state at teardown, so the real, already-imported module (and
    the names already bound at the top of this test file, which reference
    the original module object) are unaffected once this test ends.
    """
    monkeypatch.setitem(sys.modules, "pydub", None)
    monkeypatch.setitem(sys.modules, "pydub.exceptions", None)
    monkeypatch.delitem(sys.modules, "tldw_chatbook.TTS.audio_stitch", raising=False)

    reloaded = importlib.import_module("tldw_chatbook.TTS.audio_stitch")

    assert reloaded.PYDUB_AVAILABLE is False
