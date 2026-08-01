"""Tests for the script-to-audio pipeline orchestrator (spec #2 phase 2b, Task 6).

`generate_script_audio` turns one cast script's turns into one stored,
playable `briefing_audio` row: it loads the script, resolves the roster's
voices, synthesizes and stitches every turn, and writes the finished payload
once into `briefing_audio_dir()`. Its error-boundary contract is copied from
`briefing_cast.generate_script` (`Subscriptions/briefing_cast.py:562`) in
every respect -- see this module's own docstring's "Task 6 adds..." section.

Per the brief, the only faked seam is the per-turn `synthesize` callable
(the `synthesize=synthesize_turn` parameter) -- `resolve_roster_voices` runs
for real against a fake *profile service* (mirroring
`test_briefing_voices.py`'s own rule), and everything else, including a
real, file-backed `SubscriptionsDB` and the real stitcher (`pydub`), is
exercised as-is.

Every test that reaches storage patches `get_user_data_dir` to `tmp_path` --
this repo has had three separate incidents of test scaffolding touching live
user files; treat it as the hard rule it is (see the module's own binding
rules).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from loguru import logger
from pydub import AudioSegment

import tldw_chatbook.Subscriptions.briefing_audio as briefing_audio
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.briefing_audio import (
    ERROR_CHAR_CAP,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_GENERATING,
    AudioGenerationError,
    briefing_audio_dir,
    fail_interrupted_audio,
    generate_script_audio,
)
from tldw_chatbook.Subscriptions.briefing_audio import TurnSynthesisError
from tldw_chatbook.Subscriptions.briefing_voices import VoiceSelection
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_service import LoadedTTSProfile
from tldw_chatbook.TTS.profile_types import TTSGenerationProfile
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit

_CREATED_AT = datetime(2026, 7, 31, 12, tzinfo=UTC)
_HOST_PROFILE_ID = UUID("11111111-1111-4111-8111-111111111111")
_GUEST_PROFILE_ID = UUID("22222222-2222-4222-8222-222222222222")


# --- Fixtures / builders -----------------------------------------------------


def _db(tmp_path) -> SubscriptionsDB:
    """A real, file-backed `SubscriptionsDB` -- not `:memory:`.

    `generate_script_audio`'s DB work is dispatched via `asyncio.to_thread`
    (2a's whole-branch ruling): `SubscriptionsDB.conn` is thread-local, so a
    `:memory:` connection would be private to whichever thread first opened
    it and invisible from the executor thread. Matches
    `test_briefing_cast.py`'s own `_db` fixture and its docstring's
    reasoning exactly.
    """
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _script_id(
    db: SubscriptionsDB,
    *,
    roster: list[dict[str, Any]],
    turns: list[dict[str, str]] | None,
    status: str = "complete",
) -> int:
    """Build a watchlist -> briefing -> briefing_scripts chain and return the script id.

    `turns=None` leaves `turns_json` genuinely NULL (never written), the
    "script has no turns" refusal case; a `dict` list is JSON-encoded.
    """
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)
    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Duo",
        roster_snapshot_json=json.dumps(roster),
    )
    fields: dict[str, Any] = {"status": status}
    if turns is not None:
        fields["turns_json"] = json.dumps(turns)
    db.update_briefing_script(script_id, **fields)
    return script_id


def _profile(
    *,
    profile_id: UUID = _HOST_PROFILE_ID,
    voice_id: str | None = "voice-a",
    provider_id: str = "kokoro",
) -> TTSGenerationProfile:
    """A real, fully-validated `TTSGenerationProfile` fixture (legacy provider,
    so tests never need to fake the exact/`audio_cpp` snapshot contract)."""
    return TTSGenerationProfile(
        profile_id=profile_id,
        display_name="Voice",
        normalized_name="voice",
        provider_id=provider_id,
        model_id="model-a",
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
        options={},
        revision=1,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )


class _FakeProfileService:
    """The one faked seam `resolve_roster_voices` needs (mirrors
    `test_briefing_voices.py`'s `_FakeProfileService` exactly)."""

    def __init__(self, profiles: dict[UUID, TTSGenerationProfile]) -> None:
        self._profiles = profiles

    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile:
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ProfileRepositoryError("missing")
        return LoadedTTSProfile(repository_generation=1, profile=profile)


def _roster_entry(*, name: str, voice_profile_id: str | None) -> dict[str, Any]:
    return {"name": name, "voice_profile_id": voice_profile_id}


def _silence_wav(duration_ms: int = 100, frame_rate: int = 22050) -> bytes:
    """Build a real WAV payload of `duration_ms` of silence, in-process."""
    segment = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)
    buffer = BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def _decode(payload: bytes) -> AudioSegment:
    return AudioSegment.from_file(BytesIO(payload), format="wav")


@dataclass
class _RecordingSynthesize:
    """The Task 5 seam Task 6's tests fake, per the brief and module docstring.

    Returns a fixed-duration silent WAV per call, in turn order, unless
    `fail_at` matches the current `turn_index`, in which case `fail_exc` is
    raised instead. Records every call's arguments for wiring assertions.
    """

    duration_ms: int = 100
    frame_rate: int = 22050
    fail_at: int | None = None
    fail_exc: BaseException | None = None
    calls: list[tuple[Any, VoiceSelection, str, int]] = field(
        default_factory=list, init=False
    )

    async def __call__(
        self, tts_service: Any, selection: VoiceSelection, text: str, *, turn_index: int
    ) -> bytes:
        self.calls.append((tts_service, selection, text, turn_index))
        if self.fail_at is not None and turn_index == self.fail_at:
            raise self.fail_exc
        return _silence_wav(self.duration_ms, self.frame_rate)


def _patch_user_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect `briefing_audio_dir()` into `tmp_path` -- never real storage."""
    monkeypatch.setattr(briefing_audio, "get_user_data_dir", lambda: tmp_path)


# --- Happy path ---------------------------------------------------------------


async def test_happy_path_produces_a_complete_row_with_a_working_wav_file(
    tmp_path, monkeypatch
) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [
        {"speaker": "Host", "text": "Welcome back."},
        {"speaker": "Host", "text": "Here is what happened."},
    ]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})
    synth = _RecordingSynthesize(duration_ms=100)

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=synth,
    )

    assert row["status"] == STATUS_COMPLETE
    assert row["turn_count"] == 2
    assert row["error"] is None
    assert len(synth.calls) == 2
    assert [call[3] for call in synth.calls] == [0, 1]
    assert [call[2] for call in synth.calls] == [turns[0]["text"], turns[1]["text"]]

    file_path = Path(row["file_path"])
    assert file_path.exists()
    payload = file_path.read_bytes()
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"

    decoded = _decode(payload)
    expected_ms = 2 * 100 + 350  # two 100ms turns + concat_wav_segments' 350ms gap
    assert len(decoded) == pytest.approx(expected_ms, abs=50)
    assert row["duration_seconds"] == pytest.approx(expected_ms / 1000.0, abs=0.06)


async def test_the_audio_file_lands_under_the_private_data_dir(tmp_path, monkeypatch) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Host", "text": "Hello."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=_RecordingSynthesize(),
    )

    expected_dir = briefing_audio_dir()
    file_path = Path(row["file_path"])
    assert file_path.is_relative_to(expected_dir)
    assert file_path.name == f"script-{script_id}-audio-{row['id']}.wav"


# --- Pre-flight refusals: no row ever written ---------------------------------


async def test_script_not_complete_is_refused_with_no_audio_row(tmp_path, monkeypatch) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    script_id = _script_id(db, roster=roster, turns=None, status="generating")

    with pytest.raises(AudioGenerationError, match="not complete"):
        await generate_script_audio(
            db,
            script_id,
            tts_service=object(),
            profile_service=_FakeProfileService({}),
            synthesize=_RecordingSynthesize(),
        )

    assert db.list_briefing_audio(script_id) == []


async def test_script_with_no_turns_is_refused_with_no_row(tmp_path, monkeypatch) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    script_id = _script_id(db, roster=roster, turns=None, status="complete")

    with pytest.raises(AudioGenerationError, match="no turns"):
        await generate_script_audio(
            db,
            script_id,
            tts_service=object(),
            profile_service=_FakeProfileService({}),
            synthesize=_RecordingSynthesize(),
        )

    assert db.list_briefing_audio(script_id) == []


async def test_script_with_unparsable_turns_json_is_refused_with_no_row(
    tmp_path, monkeypatch
) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    script_id = _script_id(db, roster=roster, turns=None, status="generating")
    db.update_briefing_script(script_id, status="complete", turns_json="not json at all")

    with pytest.raises(AudioGenerationError):
        await generate_script_audio(
            db,
            script_id,
            tts_service=object(),
            profile_service=_FakeProfileService({}),
            synthesize=_RecordingSynthesize(),
        )

    assert db.list_briefing_audio(script_id) == []


# --- In-band failures: a `failed` row, never a raise ---------------------------


async def test_a_turn_raising_turn_synthesis_error_is_a_failed_row_naming_speaker_and_turn(
    tmp_path, monkeypatch
) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Narrator", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [
        {"speaker": "Narrator", "text": "First turn, fine."},
        {"speaker": "Narrator", "text": "Second turn, fails."},
    ]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})
    boom = TurnSynthesisError("speaker 'Narrator' turn 1: " + ("x" * 2000))
    synth = _RecordingSynthesize(fail_at=1, fail_exc=boom)

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=synth,
    )

    assert row["status"] == STATUS_FAILED
    assert "Narrator" in row["error"]
    assert "turn 1" in row["error"]
    assert len(row["error"]) <= ERROR_CHAR_CAP + len(" [...]")
    assert row["error"].endswith(" [...]")
    assert row["file_path"] is None
    # No file was ever written -- the pipeline fails before reaching the
    # write step whenever a turn's own synthesis raises.
    assert list(briefing_audio_dir().glob("*.wav")) == []


async def test_a_failed_synthesis_never_touches_the_script(tmp_path, monkeypatch) -> None:
    """THE named invariant (spec §Error handling ethos): a failed audio
    render must leave the parent `briefing_scripts` row byte-identical."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Host", "text": "This turn fails."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})
    synth = _RecordingSynthesize(
        fail_at=0, fail_exc=TurnSynthesisError("speaker 'Host' turn 0: provider exploded")
    )

    before = db.get_briefing_script(script_id)
    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=synth,
    )
    after = db.get_briefing_script(script_id)

    assert row["status"] == STATUS_FAILED
    assert before == after


async def test_a_turn_naming_a_speaker_absent_from_the_voice_snapshot_is_a_failed_row(
    tmp_path, monkeypatch
) -> None:
    """Turn/speaker mismatch: not a crash, and not silent substitution."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Ghost", "text": "Nobody assigned me a voice."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=_RecordingSynthesize(),
    )

    assert row["status"] == STATUS_FAILED
    assert "Ghost" in row["error"]
    assert row["file_path"] is None


async def test_voice_resolution_failure_for_a_deleted_profile_is_a_failed_row(
    tmp_path, monkeypatch
) -> None:
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Analyst", voice_profile_id=str(_GUEST_PROFILE_ID))]
    turns = [{"speaker": "Analyst", "text": "Some analysis."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({})  # the profile no longer exists

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=_RecordingSynthesize(),
    )

    assert row["status"] == STATUS_FAILED
    assert "Analyst" in row["error"]
    assert str(_GUEST_PROFILE_ID) in row["error"]
    assert row["file_path"] is None
    # Exactly one row -- created directly, never a "generating" row a
    # caller could see.
    assert [audio_row["id"] for audio_row in db.list_briefing_audio(script_id)] == [row["id"]]


async def test_no_file_left_behind_when_something_fails_after_the_write(
    tmp_path, monkeypatch
) -> None:
    """Cleanup mandate: if anything fails after the file is written, remove it."""
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Host", "text": "Hello."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})

    def _boom(payload: bytes):
        raise briefing_audio.AudioStitchError("could not read duration")

    monkeypatch.setattr(briefing_audio, "wav_duration_seconds", _boom)

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=_RecordingSynthesize(),
    )

    assert row["status"] == STATUS_FAILED
    assert row["file_path"] is None
    assert list(briefing_audio_dir().glob("*.wav")) == []


# --- fail_interrupted_audio -----------------------------------------------------


def test_fail_interrupted_audio_flips_orphaned_generating_rows_and_returns_the_count(
    tmp_path,
) -> None:
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    script_a = _script_id(db, roster=roster, turns=[{"speaker": "Host", "text": "hi"}])
    script_b = _script_id(db, roster=roster, turns=[{"speaker": "Host", "text": "hi"}])

    zombie = db.create_briefing_audio(script_a, voice_snapshot_json="[]")
    other_zombie = db.create_briefing_audio(script_b, voice_snapshot_json="[]")
    done = db.create_briefing_audio(script_a, voice_snapshot_json="[]")
    db.update_briefing_audio(done, status="complete", file_path="/tmp/x.wav")
    already_failed = db.create_briefing_audio(script_a, voice_snapshot_json="[]")
    db.update_briefing_audio(already_failed, status="failed", error="provider said no")

    assert fail_interrupted_audio(db, script_id=script_a) == 1

    assert db.get_briefing_audio(zombie)["status"] == "failed"
    assert db.get_briefing_audio(zombie)["error"] == "interrupted"
    assert db.get_briefing_audio(other_zombie)["status"] == "generating"
    assert db.get_briefing_audio(done)["status"] == "complete"
    assert db.get_briefing_audio(already_failed)["error"] == "provider said no"

    assert fail_interrupted_audio(db, script_id=script_a) == 0

    assert fail_interrupted_audio(db) == 1
    assert db.get_briefing_audio(other_zombie)["status"] == "failed"
    assert db.get_briefing_audio(other_zombie)["error"] == "interrupted"


# --- DB error propagation + off-loop threading (binding rules) -----------------


async def test_generate_script_audio_propagates_a_real_db_error(tmp_path, monkeypatch) -> None:
    """A genuine DB failure at the pre-flight load must propagate, never
    degrade into a row -- mirrors `test_generate_script_propagates_a_real_db_error`.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    script_id = _script_id(db, roster=roster, turns=[{"speaker": "Host", "text": "hi"}])

    closed_connection = db._get_connection()
    closed_connection.close()
    monkeypatch.setattr(db, "_get_connection", lambda: closed_connection)
    db._local = threading.local()  # evict every thread's cached (open) connection

    with pytest.raises(sqlite3.ProgrammingError):
        await generate_script_audio(
            db,
            script_id,
            tts_service=object(),
            profile_service=_FakeProfileService({}),
            synthesize=_RecordingSynthesize(),
        )


async def test_a_db_error_finalizing_the_row_propagates_and_still_cleans_up_the_file(
    tmp_path, monkeypatch
) -> None:
    """The finalize DB write is a genuine DB call: its failure must
    propagate (not become a `failed` row) per the binding "DB errors
    propagate" rule -- but the now-orphaned file must still be removed.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Host", "text": "Hello."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})

    original_update = db.update_briefing_audio

    def _spy_update(audio_id, **fields):
        if fields.get("status") == STATUS_COMPLETE:
            raise RuntimeError("finalize boom")
        return original_update(audio_id, **fields)

    monkeypatch.setattr(db, "update_briefing_audio", _spy_update)

    with pytest.raises(RuntimeError, match="finalize boom"):
        await generate_script_audio(
            db,
            script_id,
            tts_service=object(),
            profile_service=profile_service,
            synthesize=_RecordingSynthesize(),
        )

    # The row never reached "complete" (the DB write that would have said
    # so is exactly what raised), and no orphan file remains on disk.
    [row] = db.list_briefing_audio(script_id)
    assert row["status"] == STATUS_GENERATING
    assert list(briefing_audio_dir().glob("*.wav")) == []


async def test_generate_script_audio_db_work_runs_off_the_event_loop_thread(
    tmp_path, monkeypatch
) -> None:
    """Mirrors `test_generate_script_db_work_runs_off_the_event_loop_thread`
    (phase 2a): every DB call, and the audio file write itself, must be
    dispatched off the event loop thread (2a's whole-branch ruling).
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Host", "text": "Hello."}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})

    loop_thread_id = threading.get_ident()
    write_thread_ids: list[int] = []

    for name in (
        "get_briefing_script",
        "create_briefing_audio",
        "update_briefing_audio",
    ):
        original = getattr(db, name)

        def _spy(*args, __original=original, **kwargs):
            write_thread_ids.append(threading.get_ident())
            return __original(*args, **kwargs)

        setattr(db, name, _spy)

    write_call_thread_ids: list[int] = []
    original_write = briefing_audio.atomic_private_write_bytes

    def _spy_write(*args, **kwargs):
        write_call_thread_ids.append(threading.get_ident())
        return original_write(*args, **kwargs)

    monkeypatch.setattr(briefing_audio, "atomic_private_write_bytes", _spy_write)

    row = await generate_script_audio(
        db,
        script_id,
        tts_service=object(),
        profile_service=profile_service,
        synthesize=_RecordingSynthesize(),
    )

    assert row["status"] == STATUS_COMPLETE
    assert len(write_thread_ids) >= 3
    assert all(tid != loop_thread_id for tid in write_thread_ids)
    assert len(write_call_thread_ids) == 1
    assert write_call_thread_ids[0] != loop_thread_id


async def test_generate_script_audio_logs_no_turn_content_on_failure(
    tmp_path, monkeypatch
) -> None:
    """Egress pin: this app's file sink runs `diagnose=True`, which dumps a
    failing frame's locals -- and the frame at a synthesis failure holds
    the turn text. Only the exception TYPE may reach the log.
    """
    _patch_user_data_dir(monkeypatch, tmp_path)
    canary = "ZEBRACANARY"
    db = _db(tmp_path)
    roster = [_roster_entry(name="Host", voice_profile_id=str(_HOST_PROFILE_ID))]
    turns = [{"speaker": "Host", "text": canary}]
    script_id = _script_id(db, roster=roster, turns=turns)
    profile_service = _FakeProfileService({_HOST_PROFILE_ID: _profile()})
    synth = _RecordingSynthesize(
        fail_at=0, fail_exc=TurnSynthesisError(f"speaker 'Host' turn 0: {canary} boom")
    )

    captured: list[str] = []
    handler = logger.add(captured.append, level="DEBUG", diagnose=True, backtrace=True, catch=False)
    try:
        row = await generate_script_audio(
            db,
            script_id,
            tts_service=object(),
            profile_service=profile_service,
            synthesize=synth,
        )
    finally:
        logger.remove(handler)

    assert row["status"] == STATUS_FAILED
    log_text = "".join(captured)
    assert log_text
    assert "synthesis failed" in log_text
    assert "TurnSynthesisError" in log_text
    assert canary not in log_text
