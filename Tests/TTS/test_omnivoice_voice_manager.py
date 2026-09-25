"""OmniVoice voice manager — CRUD + transcript requirement.

OmniVoice zero-shot cloning consumes reference audio *and* its transcript
together, so every profile stores a required ``reference_text`` alongside
the copied reference clip.
"""

import json
import time
import wave
from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional

import pytest

from tldw_chatbook.TTS import omnivoice_voice_manager as omnivoice_vm_module
from tldw_chatbook.TTS.omnivoice_voice_manager import OmniVoiceVoiceManager
from tldw_chatbook.TTS.profile_reference_types import MAX_REFERENCE_TEXT_CHARACTERS


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * int(24000 * seconds))


def test_create_list_get_roundtrip(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.create_profile(
        "narrator", str(tmp_path / "ref.wav"), reference_text="hello there"
    )
    assert ok, msg

    names = [p["name"] for p in mgr.list_profiles()]
    assert names == ["narrator"]
    summaries = mgr.list_profiles()
    assert summaries[0]["backend"] == "omnivoice"
    assert summaries[0]["display_name"] == "narrator"

    profile = mgr.get_profile("narrator")
    assert profile is not None
    assert profile["name"] == "narrator"
    assert profile["reference_text"] == "hello there"
    assert profile["language"] == "en"
    assert profile["display_name"] == "narrator"
    # ISO-8601 timestamps
    datetime.fromisoformat(profile["created_at"])
    datetime.fromisoformat(profile["updated_at"])

    # Real file effects: per-profile directory with profile.json + copied clip
    profile_dir = tmp_path / "voices" / "narrator"
    stored = json.loads((profile_dir / "profile.json").read_text())
    assert stored["reference_text"] == "hello there"
    assert stored["reference_audio"] == "reference.wav"
    copied = profile_dir / "reference.wav"
    assert copied.is_file()
    with wave.open(str(copied), "rb") as w:
        assert w.getnframes() == int(24000 * 1.0)
    ref_path = mgr.get_reference_audio_path("narrator")
    assert ref_path == copied


def test_create_requires_reference_text(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")

    ok, msg = mgr.create_profile("x", str(tmp_path / "ref.wav"), reference_text="")
    assert not ok and "reference_text" in msg

    ok, msg = mgr.create_profile(
        "x", str(tmp_path / "ref.wav"), reference_text="   \n\t  "
    )
    assert not ok and "reference_text" in msg

    # Missing entirely (base VoiceManagerBase signature) must also be rejected
    # with a clear message, not a TypeError.
    ok, msg = mgr.create_profile("x", str(tmp_path / "ref.wav"))
    assert not ok and "reference_text" in msg

    # Nothing was created
    assert not (tmp_path / "voices" / "x").exists()
    assert mgr.list_profiles() == []


def test_create_reference_text_length_bounded(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")

    ok, msg = mgr.create_profile(
        "x",
        str(tmp_path / "ref.wav"),
        reference_text="a" * MAX_REFERENCE_TEXT_CHARACTERS,
    )
    assert ok, msg

    ok, msg = mgr.create_profile(
        "y",
        str(tmp_path / "ref.wav"),
        reference_text="a" * (MAX_REFERENCE_TEXT_CHARACTERS + 1),
    )
    assert not ok and "reference_text" in msg


def test_create_rejects_reference_over_duration_limit(tmp_path: Path) -> None:
    _write_wav(tmp_path / "short.wav", seconds=1.0)
    _write_wav(tmp_path / "long.wav", seconds=3.0)
    mgr = OmniVoiceVoiceManager(tmp_path / "voices", max_reference_duration=2.0)

    ok, msg = mgr.create_profile(
        "short", str(tmp_path / "short.wav"), reference_text="ok"
    )
    assert ok, msg

    ok, msg = mgr.create_profile(
        "long", str(tmp_path / "long.wav"), reference_text="too long"
    )
    assert not ok
    assert "duration" in msg.lower()
    assert not (tmp_path / "voices" / "long").exists()

    # Default limit is 30 s
    default_mgr = OmniVoiceVoiceManager(tmp_path / "voices2")
    assert default_mgr.max_reference_duration == 30.0
    _write_wav(tmp_path / "half_minute_plus.wav", seconds=30.5)
    ok, msg = default_mgr.create_profile(
        "over", str(tmp_path / "half_minute_plus.wav"), reference_text="nope"
    )
    assert not ok and "duration" in msg.lower()


def test_create_rejects_missing_reference_audio(tmp_path: Path) -> None:
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.create_profile(
        "x", str(tmp_path / "does-not-exist.wav"), reference_text="hi"
    )
    assert not ok
    assert "not found" in msg.lower()


def test_create_rejects_empty_reference_audio(tmp_path: Path) -> None:
    _write_wav(tmp_path / "silent.wav", seconds=0.0)
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.create_profile(
        "x", str(tmp_path / "silent.wav"), reference_text="hi"
    )
    assert not ok
    assert "frames" in msg.lower()


def test_create_rejects_duplicate_profile(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, _ = mgr.create_profile("x", str(tmp_path / "ref.wav"), reference_text="hi")
    assert ok
    ok, msg = mgr.create_profile("x", str(tmp_path / "ref.wav"), reference_text="hi")
    assert not ok and "already exists" in msg


def test_create_rejects_non_wav_when_soundfile_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mp3 = tmp_path / "ref.mp3"
    fake_mp3.write_bytes(b"\xff\xfb not really an mp3")
    monkeypatch.setattr(omnivoice_vm_module, "SOUNDFILE_AVAILABLE", False)

    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.create_profile("x", str(fake_mp3), reference_text="hi")
    assert not ok
    assert "wav" in msg.lower()


def test_delete_and_update(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, _ = mgr.create_profile("x", str(tmp_path / "ref.wav"), reference_text="hi")
    assert ok
    created = mgr.get_profile("x")

    time.sleep(0.01)
    ok, msg = mgr.update_profile("x", description="desc")
    assert ok, msg
    updated = mgr.get_profile("x")
    assert updated["description"] == "desc"
    assert updated["reference_text"] == "hi"
    assert updated["updated_at"] > created["updated_at"]

    ok, msg = mgr.delete_profile("x")
    assert ok, msg
    assert mgr.get_profile("x") is None
    assert not (tmp_path / "voices" / "x").exists()

    ok, msg = mgr.delete_profile("x")
    assert not ok and "not found" in msg.lower()


def test_export_import_roundtrip(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, _ = mgr.create_profile(
        "narrator",
        str(tmp_path / "ref.wav"),
        reference_text="hello there",
        description="clone source",
    )
    assert ok

    ok, msg = mgr.export_profile("narrator", str(tmp_path / "export"))
    assert ok, msg
    package = tmp_path / "export" / "omnivoice_voice_narrator"
    exported = json.loads((package / "profile.json").read_text())
    assert exported["reference_text"] == "hello there"
    assert exported["reference_audio"] == "reference.wav"
    assert (package / "reference.wav").is_file()
    assert (package / "README.txt").is_file()

    mgr2 = OmniVoiceVoiceManager(tmp_path / "voices2")
    ok, msg = mgr2.import_profile(str(package))
    assert ok, msg
    imported = mgr2.get_profile("narrator")
    assert imported is not None
    assert imported["reference_text"] == "hello there"
    assert (tmp_path / "voices2" / "narrator" / "reference.wav").is_file()
    assert mgr2.get_reference_audio_path("narrator").is_file()

    # Duplicate import is refused without overwrite, allowed with it
    ok, msg = mgr2.import_profile(str(package))
    assert not ok and "already exists" in msg
    ok, msg = mgr2.import_profile(str(package), overwrite=True)
    assert ok, msg


def test_import_requires_reference_text(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, _ = mgr.create_profile("x", str(tmp_path / "ref.wav"), reference_text="hi")
    assert ok
    ok, _ = mgr.export_profile("x", str(tmp_path / "export"))
    assert ok

    package = tmp_path / "export" / "omnivoice_voice_x"
    stored = json.loads((package / "profile.json").read_text())
    del stored["reference_text"]
    (package / "profile.json").write_text(json.dumps(stored))

    mgr2 = OmniVoiceVoiceManager(tmp_path / "voices2")
    ok, msg = mgr2.import_profile(str(package))
    assert not ok and "reference_text" in msg
    assert mgr2.get_profile("x") is None


def test_traversal_names_refused_on_every_path(tmp_path: Path) -> None:
    """Traversal-style names never resolve outside the voices dir.

    The name becomes a directory name on every entry point — including
    the destructive delete — so each must refuse unsafe names and leave
    anything outside the voices dir untouched.
    """
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, _ = mgr.create_profile("x", str(tmp_path / "ref.wav"), reference_text="hi")
    assert ok

    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "sentinel.txt").write_text("do not delete")
    victim = outside / "nested" / "profile.json"
    victim.write_text('{"name": "victim"}')

    for name in ("../outside", "../outside/nested", "x/../../outside", "..", "/etc"):
        ok, msg = mgr.delete_profile(name)
        assert not ok and "profile name" in msg.lower(), (name, msg)
        ok, msg = mgr.update_profile(name, description="nope")
        assert not ok and "profile name" in msg.lower(), (name, msg)
        ok, msg = mgr.export_profile(name, str(tmp_path / "export"))
        assert not ok and "profile name" in msg.lower(), (name, msg)
        ok, msg = mgr.create_profile(
            name, str(tmp_path / "ref.wav"), reference_text="hi"
        )
        assert not ok and "profile name" in msg.lower(), (name, msg)
        assert mgr.get_profile(name) is None
        assert mgr.get_reference_audio_path(name) is None

    # Import with an explicit traversal name is refused too
    ok, _ = mgr.export_profile("x", str(tmp_path / "export"))
    assert ok
    package = tmp_path / "export" / "omnivoice_voice_x"
    ok, msg = mgr.import_profile(str(package), profile_name="../evil")
    assert not ok and "profile name" in msg.lower()

    # Nothing outside the voices dir was touched
    assert (outside / "sentinel.txt").read_text() == "do not delete"
    assert json.loads(victim.read_text())["name"] == "victim"
    # and the legitimate profile is intact
    assert mgr.get_profile("x") is not None


def test_create_accepts_float_wav_via_soundfile(tmp_path: Path) -> None:
    """The stdlib ``wave`` module rejects IEEE-float WAVs; with soundfile
    installed they must still be measurable (and so accepted)."""
    sf = pytest.importorskip("soundfile")
    import numpy as np

    ref = tmp_path / "float.wav"
    sf.write(str(ref), np.zeros(24000, dtype=np.float32), 24000, subtype="FLOAT")
    with pytest.raises(wave.Error):
        wave.open(str(ref), "rb")

    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.create_profile("floaty", str(ref), reference_text="hello there")
    assert ok, msg


def test_overwrite_import_drops_stale_reference_clip(tmp_path: Path) -> None:
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    _write_wav(tmp_path / "ref.wav")
    ok, msg = mgr.create_profile("swap", str(tmp_path / "ref.wav"), reference_text="hi")
    assert ok, msg

    sf = pytest.importorskip("soundfile")
    import numpy as np

    package = tmp_path / "omnivoice_voice_swap"
    package.mkdir()
    sf.write(str(package / "reference.flac"), np.zeros(24000, dtype=np.float32), 24000)
    (package / "profile.json").write_text(
        json.dumps({"reference_audio": "reference.flac", "reference_text": "new words"})
    )
    ok, msg = mgr.import_profile(str(package), overwrite=True)
    assert ok, msg

    profile_dir = tmp_path / "voices" / "swap"
    assert sorted(p.name for p in profile_dir.glob("reference.*")) == ["reference.flac"]
    assert mgr.get_reference_audio_path("swap") == profile_dir / "reference.flac"


# --- Qodo review (PR #2825) -----------------------------------------------------


def _package(tmp_path: Path, clip: Optional[Path], *, transcript: str = "hi there") -> Path:
    package = tmp_path / "omnivoice_voice_pkg"
    package.mkdir()
    payload = {"reference_text": transcript}
    if clip is not None:
        shutil.copy2(clip, package / clip.name)
        payload["reference_audio"] = clip.name
    (package / "profile.json").write_text(json.dumps(payload))
    return package


def test_import_requires_a_reference_clip(tmp_path: Path) -> None:
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.import_profile(str(_package(tmp_path, None)))
    assert not ok and "reference_audio" in msg
    assert not (tmp_path / "voices" / "pkg").exists()


def test_import_applies_create_clip_validation(tmp_path: Path) -> None:
    """Imports once skipped the format/frames/duration checks create applies."""
    long_clip = tmp_path / "long.wav"
    _write_wav(long_clip, seconds=3.0)
    mgr = OmniVoiceVoiceManager(tmp_path / "voices", max_reference_duration=1.0)
    ok, msg = mgr.import_profile(str(_package(tmp_path, long_clip)))
    assert not ok and "exceeds" in msg
    assert list((tmp_path / "voices").glob("*")) == []

    empty_clip = tmp_path / "empty.wav"
    _write_wav(empty_clip, seconds=0.0)
    (tmp_path / "omnivoice_voice_pkg").rename(tmp_path / "old_pkg")
    ok, msg = OmniVoiceVoiceManager(tmp_path / "voices").import_profile(
        str(_package(tmp_path, empty_clip))
    )
    assert not ok and "no audio frames" in msg


def test_failed_copy_leaves_nothing_so_a_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_wav(tmp_path / "ref.wav")
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")

    def failing_copy(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(omnivoice_vm_module.shutil, "copy2", failing_copy)
    ok, _msg = mgr.create_profile("retry", str(tmp_path / "ref.wav"), reference_text="hi")
    assert not ok
    assert not (tmp_path / "voices" / "retry").exists()

    monkeypatch.undo()
    ok, msg = mgr.create_profile("retry", str(tmp_path / "ref.wav"), reference_text="hi")
    assert ok, msg


def test_user_paths_go_through_path_validation(tmp_path: Path) -> None:
    mgr = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, msg = mgr.create_profile("nul", "bad\x00path.wav", reference_text="hi")
    assert not ok and "not a valid local file" in msg
    ok, _msg = mgr.import_profile("bad\x00dir")
    assert not ok
    _write_wav(tmp_path / "ref.wav")
    assert mgr.create_profile("ok", str(tmp_path / "ref.wav"), reference_text="hi")[0]
    ok, msg = mgr.export_profile("ok", "bad\x00out")
    assert not ok and "not a valid local path" in msg
