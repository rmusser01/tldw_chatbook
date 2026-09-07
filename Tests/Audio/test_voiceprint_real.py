"""Opt-in: two real TTS voices through the real SpeechBrain worker (spec §7).

Run: TLDW_RUN_VOICEPRINT_TEST=1 pytest Tests/Audio/test_voiceprint_real.py -p no:cacheprovider

ECAPA embeddings of pure tones don't behave like speech, so this generates
two distinct voices with macOS's `say` (`-v Samantha` / `-v Daniel`),
converts each to a 16 kHz mono PCM16 WAV with `afconvert` -- both platform
built-ins, invoked as argument lists via `subprocess.run`, never a shell
string -- enrolls a voiceprint from one voice via `enroll_from_pcm` (the
same op explicit enrollment uses, spec §3.4), then drives a real meeting
session (`Tests/Audio/conftest.py::meeting_session_with_fake_capture` +
a real `SpeechBrainDiarizer(voiceprint=...)`) over both voices and asserts
only the enrolled voice's cluster is matched as the user. All audio lives
under `tmp_path`; nothing is written anywhere else.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

pytestmark = [pytest.mark.real_audio_device, pytest.mark.integration]

SAMPLE_RATE = 16000
_BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono PCM
_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "the autumn leaves drift slowly onto the old wooden bridge below the "
    "hill, and the wind carries the sound of distant church bells across "
    "the valley every morning before the market opens for the day."
)


def _say_wav(voice: str, dest: Path) -> Path:
    """`say -v <voice> -o` an AIFF, then `afconvert` it to a 16 kHz mono
    PCM16 WAV. Argument lists only -- never a shell string."""
    aiff = dest.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), _PARAGRAPH], check=True, capture_output=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(dest)],
        check=True, capture_output=True,
    )
    return dest


def _read_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (SAMPLE_RATE, 1, 2), (
            f"unexpected WAV format from afconvert: {wf.getparams()}"
        )
        return wf.readframes(wf.getnframes())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Both centroids are already unit-normalised (spec §3.1/§3.2, worker's
    own `_unit`), so this is just `1 - diarizer_worker._cos_dist(a, b)`."""
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.skipif(
    os.environ.get("TLDW_RUN_VOICEPRINT_TEST") != "1"
    or sys.platform != "darwin"
    or shutil.which("say") is None
    or shutil.which("afconvert") is None,
    reason="opt-in: TLDW_RUN_VOICEPRINT_TEST=1 on macOS with the diarization extra and the `say`/`afconvert` tools",
)
def test_enrolled_voice_matches_itself_and_not_another(tmp_path, meeting_session_with_fake_capture):
    pytest.importorskip("torch")
    pytest.importorskip("speechbrain")

    from tldw_chatbook.Audio.diarizer_local import READY_TIMEOUT_S, SpeechBrainDiarizer

    pcm_a = _read_pcm16(_say_wav("Samantha", tmp_path / "voice_a.wav"))
    pcm_b = _read_pcm16(_say_wav("Daniel", tmp_path / "voice_b.wav"))
    assert pcm_a and pcm_b, "say/afconvert produced empty audio"

    # Enroll from voice A on a short-lived worker -- the same op explicit
    # enrollment uses (spec §3.4). A separate process from the meeting's own
    # diarizer, matching how the app never reuses a live meeting's worker for
    # enrollment either.
    enroller = SpeechBrainDiarizer(max_speakers=8)
    try:
        assert enroller.wait_ready(READY_TIMEOUT_S), "enrollment worker never reached READY"
        enrolled = enroller.enroll_from_pcm(pcm_a, SAMPLE_RATE)
    finally:
        enroller.close()
    assert enrolled is not None, "enroll_from_pcm returned nothing for voice A"
    voiceprint, enrolled_seconds = enrolled
    assert enrolled_seconds > 0

    diarizer = SpeechBrainDiarizer(
        max_speakers=8, voiceprint=voiceprint, match_threshold=0.2, match_min_seconds=4.0,
    )
    try:
        assert diarizer.wait_ready(READY_TIMEOUT_S), "meeting worker never reached READY"

        session = meeting_session_with_fake_capture(mode="room", diarizer=diarizer)

        # Room mode diarizes every final segment on the "mixed" channel
        # (spec §3.4; `meeting_session._on_final`'s `_label` returns None off
        # call mode) -- feed real audio through the fake capture's
        # `pcm_window` seam. Two 3s windows per voice clears
        # `voice_match_min_seconds=4.0` before the other voice starts.
        windows = [("a", 0.0, 3.0), ("a", 3.0, 6.0), ("b", 0.0, 3.0), ("b", 3.0, 6.0)]
        pcm_by_speaker = {"a": pcm_a, "b": pcm_b}
        current: dict[str, tuple[str, float, float]] = {}

        def _pcm_window(_source: str, _start_s: float, _end_s: float) -> bytes:
            speaker, w_start, w_end = current["window"]
            pcm = pcm_by_speaker[speaker]
            return pcm[int(w_start * _BYTES_PER_SECOND):min(len(pcm), int(w_end * _BYTES_PER_SECOND))]

        session.capture.pcm_window = _pcm_window

        cluster_ids: dict[str, list[str | None]] = {"a": [], "b": []}
        position = 0.0
        for speaker, w_start, w_end in windows:
            current["window"] = (speaker, w_start, w_end)
            position += w_end - w_start
            session.capture.audio_position_s = position
            session.capture.last_speech_position_s = position
            session._on_final_for_test(f"{speaker} speaking")
            cluster_ids[speaker].append(session.segments[-1].speaker_id)

        a_cluster = next((c for c in cluster_ids["a"] if c), None)
        b_cluster = next((c for c in cluster_ids["b"] if c), None)
        assert a_cluster, f"voice A never received a cluster id ({cluster_ids['a']})"
        assert b_cluster, f"voice B never received a cluster id ({cluster_ids['b']})"
        assert a_cluster != b_cluster, "the two distinct voices collapsed into one cluster"

        assert session.meta.matched_self == a_cluster, (
            f"the enrolled voice's cluster ({a_cluster!r}) was not matched as the user "
            f"(matched_self={session.meta.matched_self!r})"
        )
        assert session.meta.speaker_names.get(b_cluster) != session.meta.user_display_name, (
            "the OTHER voice's cluster was named as the user"
        )

        # Calibration only -- never asserted on: the threshold (spec §5) is a
        # documented starting value, not validated until a run like this one.
        mine = diarizer.export_centroid(a_cluster)
        other = diarizer.export_centroid(b_cluster)
        if mine is not None and other is not None:
            print(
                f"voiceprint calibration: self-similarity={_cosine_similarity(voiceprint, mine[0]):.4f} "
                f"other-voice-similarity={_cosine_similarity(voiceprint, other[0]):.4f} "
                "(match_threshold=0.2 distance <=> similarity >= 0.8)"
            )
    finally:
        diarizer.close()
