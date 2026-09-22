"""Direct characterization of ChatterboxVoiceManager's profile CRUD.

TASK-32863: the shared single-JSON store and update/delete/get CRUD moved
into VoiceManagerBase; these tests pin Chatterbox's side of the contract
(chatterbox_profiles.json store, backend-stamping get_profile, inherited
update/delete) so the adoption cannot silently change it. In-process only.
"""

import json
import wave
from pathlib import Path

import pytest

from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager


def _write_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)
    return path


@pytest.fixture()
def manager(tmp_path):
    return ChatterboxVoiceManager(tmp_path / "voices")


@pytest.fixture()
def sample(tmp_path) -> Path:
    return _write_wav(tmp_path / "sample.wav")


def test_store_uses_chatterbox_profiles_file(manager, sample):
    ok, message = manager.create_profile("voice", str(sample))
    assert ok, message
    assert (manager.voice_samples_dir / "chatterbox_profiles.json").exists()
    stored = json.loads(
        (manager.voice_samples_dir / "chatterbox_profiles.json").read_text()
    )
    assert "voice" in stored


def test_get_profile_stamps_backend_field(manager, sample):
    manager.create_profile("voice", str(sample))
    record = manager.get_profile("voice")
    assert record is not None
    assert record["backend"] == "chatterbox"
    assert manager.get_profile("missing") is None


def test_inherited_update_and_delete(manager, sample):
    manager.create_profile("voice", str(sample))
    ok, message = manager.update_profile("voice", display_name="Renamed")
    assert ok, message
    assert manager.get_profile("voice")["display_name"] == "Renamed"
    assert not manager.update_profile("missing")[0]

    ok, message = manager.delete_profile("voice")
    assert ok, message
    assert manager.get_profile("voice") is None
    assert not (manager.voice_samples_dir / "voice").exists()


def test_create_refuses_duplicate_and_invalid_audio(manager, sample):
    manager.create_profile("voice", str(sample))
    assert manager.create_profile("voice", str(sample)) == (
        False,
        "Profile 'voice' already exists",
    )
    bad = manager.voice_samples_dir / "bad.txt"
    bad.write_text("not audio")
    ok, message = manager.create_profile("bad", str(bad))
    assert not ok and "Invalid audio file" in message


def test_no_backup_directory_unless_enabled(manager, sample):
    manager.create_profile("voice", str(sample))
    manager.create_profile("voice2", str(sample))
    assert not (manager.voice_samples_dir / "backups").exists()


def test_delete_never_escapes_the_samples_root(manager, sample, tmp_path):
    """Traversal-shaped names are refused before any filesystem effect.

    Two independent layers refuse them: the loose-voice admission wrapper
    rejects path-component characters in profile_name on every wrapped
    call, and the shared delete validates the joined directory against the
    samples root before removal (PR #2794 review, security finding).
    """
    outside = tmp_path / "outside-marker"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")

    for malicious in ("../outside-marker", "..", "a/b", "a" + chr(92) + "b"):
        with pytest.raises(ValueError, match="invalid_voice_name"):
            manager.delete_profile(malicious)
        assert sentinel.read_text() == "keep", malicious


def test_create_refuses_an_oversized_reference_before_copying_it(
    manager, sample, monkeypatch, tmp_path
):
    """Chatterbox had no size bound on a user-picked reference file.

    Tier-2 review S03/S04: `VoiceManagerBase.validate_audio_file` checked
    existence and extension only, and `create_profile` then hands the file to
    `loose_voice_lifetime.copy`, which reads it WHOLE into memory in one
    `stream.read()`. Higgs carried a 100 MB bound; Chatterbox, which inherits
    the base unchanged, carried none, so a user picking a multi-GB container
    OOMs the TUI. The bound is checked here on the base so both managers and
    any future subclass get it.
    """
    monkeypatch.setattr(type(manager), "max_reference_audio_bytes", 512)
    oversized = _write_wav(tmp_path / "big.wav")
    assert oversized.stat().st_size > 512

    ok, message = manager.create_profile("too-big", str(oversized))

    assert ok is False
    assert "too large" in message.lower()
    # Nothing was copied: the profile directory must not exist.
    assert not (manager.voice_samples_dir / "too-big").exists()
    assert manager.load_profiles() == {}
