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


def test_saving_backs_up_the_store_it_is_about_to_replace(manager, sample):
    """(TASK-32893) Inverted: this pinned ``keep_backups`` staying off here.

    Every save rewrites the whole store, so with no backup one bad profile
    edit took every Chatterbox profile with it. Higgs already kept backups;
    Chatterbox now sets ``keep_backups`` too. The durability properties
    themselves are pinned in ``test_voice_profile_store_safety.py``.
    """
    manager.create_profile("voice", str(sample))
    manager.create_profile("voice2", str(sample))
    # Only the second save had a prior file to preserve.
    assert len(manager._list_backups()) == 1


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
