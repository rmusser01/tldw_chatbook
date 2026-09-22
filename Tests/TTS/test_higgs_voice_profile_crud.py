"""Direct characterization of HiggsVoiceProfileManager's profile CRUD.

TASK-32863: the manager adopts VoiceManagerBase; these tests pin the exact
on-disk contract (single voice_profiles.json store, per-profile reference
dirs, backups on save, higgs_voice_ export packages) BEFORE the refactor so
the adoption cannot silently change any of it. In-process only -- no torch,
no engine, no subprocesses.
"""

import json
import wave
from pathlib import Path

import pytest

from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager


def _write_wav(path: Path, seconds: float = 0.1) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * int(8000 * seconds))
    return path


@pytest.fixture()
def manager(tmp_path):
    return HiggsVoiceProfileManager(tmp_path / "voices")


@pytest.fixture()
def sample(tmp_path) -> Path:
    return _write_wav(tmp_path / "sample.wav")


def test_create_persists_record_and_copies_reference(manager, sample):
    ok, message = manager.create_profile(
        "voice-a", str(sample), display_name="Voice A", tags=["proto"]
    )
    assert ok, message

    stored = json.loads((manager.voice_samples_dir / "voice_profiles.json").read_text())
    assert set(stored) == {"voice-a"}
    record = stored["voice-a"]
    assert record["display_name"] == "Voice A"
    assert record["language"] == "en"
    assert record["description"] == ""
    assert record["tags"] == ["proto"]
    assert record["metadata"] == {}
    assert record["created_at"] and record["updated_at"]
    assert record["audio_info"]["path"].endswith("sample.wav")
    reference = Path(record["reference_audio"])
    assert reference.parent == manager.voice_samples_dir / "voice-a"
    assert reference.exists()


def test_create_refuses_empty_duplicate_and_missing_audio(manager, sample):
    assert manager.create_profile("", str(sample)) == (False, "Profile name cannot be empty")
    ok, _ = manager.create_profile("dup", str(sample))
    assert ok
    assert manager.create_profile("dup", str(sample)) == (
        False,
        "Profile 'dup' already exists",
    )
    ok, message = manager.create_profile("gone", str(manager.voice_samples_dir / "nope.wav"))
    assert not ok and "Reference audio not found" in message


def test_list_summaries_filter_tags_and_sort(manager, sample):
    manager.create_profile("zeta", str(sample), display_name="B-second", tags=["x"])
    manager.create_profile("alpha", str(sample), display_name="A-first", tags=["y"])
    listed = manager.list_profiles()
    assert [item["name"] for item in listed] == ["alpha", "zeta"]  # sorted by display
    first = listed[0]
    assert first["display_name"] == "A-first"
    assert first["has_reference"] is True
    assert first["audio_duration"] == 0  # no soundfile in this environment
    assert manager.list_profiles(tags=["y"])[0]["name"] == "alpha"
    assert manager.list_profiles(tags=["nope"]) == []


def test_get_update_and_not_found(manager, sample):
    manager.create_profile("voice", str(sample))
    assert manager.get_profile("missing") is None
    ok, message = manager.update_profile("voice", display_name="Renamed", language="fr")
    assert ok, message
    record = manager.get_profile("voice")
    assert record["display_name"] == "Renamed"
    assert record["language"] == "fr"
    assert not manager.update_profile("missing")[0]


def test_delete_removes_record_and_directory(manager, sample):
    manager.create_profile("voice", str(sample))
    profile_dir = manager.voice_samples_dir / "voice"
    assert profile_dir.exists()
    ok, message = manager.delete_profile("voice")
    assert ok, message
    assert manager.get_profile("voice") is None
    assert not profile_dir.exists()
    assert not manager.delete_profile("voice")[0]


def test_export_import_round_trip(manager, sample, tmp_path):
    manager.create_profile("voice", str(sample), description="keep me")
    ok, message = manager.export_profile("voice", str(tmp_path / "export"))
    assert ok, message

    package = tmp_path / "export" / "higgs_voice_voice"
    profile_json = json.loads((package / "profile.json").read_text())
    assert profile_json["description"] == "keep me"
    assert profile_json["reference_audio"].endswith(".wav")
    assert (package / "README.txt").exists()

    fresh = HiggsVoiceProfileManager(tmp_path / "voices2")
    ok, message = fresh.import_profile(str(package))
    assert ok, message
    record = fresh.get_profile("voice")  # name inferred from package prefix
    assert record["description"] == "keep me"
    assert record["imported_at"]
    assert Path(record["reference_audio"]).exists()

    assert not fresh.import_profile(str(package))[0]  # overwrite refusal
    ok, _ = fresh.import_profile(str(package), overwrite=True)
    assert ok


def test_save_creates_timestamped_backup(manager, sample):
    manager.create_profile("voice", str(sample))
    manager.create_profile("voice2", str(sample))
    backups = list(manager.backup_dir.glob("voice_profiles_backup_*.json"))
    assert len(backups) == 1  # only the second save had a prior file to back up
