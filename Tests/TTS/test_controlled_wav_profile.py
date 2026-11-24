"""Controlled readiness and retained WAV admission use real files."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path
import wave

import pytest

from tldw_chatbook.TTS import controlled_wav_profile as profile
from tldw_chatbook.TTS.controlled_wav_playback import ControlledWavPlayback

native = pytest.mark.skipif(os.name != "nt", reason="requires native NTFS pins")


@pytest.fixture(name="retained")
def retained_files(tmp_path, monkeypatch):
    exe = tmp_path / "ffplay.exe"
    exe.write_bytes(b"fake native executable for injected launch")
    monkeypatch.setattr(
        profile, "QUALIFIED_SHA256", hashlib.sha256(exe.read_bytes()).hexdigest()
    )
    path = tmp_path / "article.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 9600 + bytes(48000))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = profile.RetainedWav(
        path, digest, 48000, 4800, 16800, profile.PADDING_POLICY
    )
    return exe, artifact, profile.WavBookmark(digest, 48000, 0)


@native
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "digest",
        "rate",
        "file_frames",
        "padding_policy",
        "bookmark",
        "negative",
        "truncated",
        "partial",
        "tail",
        "shape",
        "hardlink",
    ],
)
async def test_invalid_retained_source_refuses_before_launch(retained, bad):
    exe, artifact, bookmark = retained
    if bad in {"digest", "rate", "file_frames", "padding_policy"}:
        field, value = {
            "digest": ("sha256", "1" * 64),
            "rate": ("sample_rate", 44100),
            "file_frames": ("file_frames", 16500),
            "padding_policy": ("padding_policy", "unknown"),
        }[bad]
        artifact = replace(artifact, **{field: value})
    elif bad == "bookmark":
        bookmark = replace(bookmark, sha256="a" * 64)
    elif bad == "negative":
        bookmark = replace(bookmark, frame=-1)
    elif bad == "hardlink":
        os.link(artifact.path, artifact.path.with_suffix(".link"))
    else:
        payload = bytearray(artifact.path.read_bytes())
        if bad == "shape":
            payload[22] = 1
        elif bad == "tail":
            payload[-1] = 8
        elif bad == "partial":
            payload.extend(b"\0")
        else:
            del payload[-4:]
        artifact.path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        artifact, bookmark = (
            replace(artifact, sha256=digest),
            replace(bookmark, sha256=digest),
        )
    with pytest.raises(profile.ControlledWavError):
        await ControlledWavPlayback.prepare(artifact, bookmark, executable=exe)
    artifact.path.rename(artifact.path.with_suffix(".retired"))


@native
@pytest.mark.asyncio
async def test_readiness_requires_direct_known_payload_and_default_route(
    retained, monkeypatch
):
    exe, _, _ = retained
    assert await profile.controlled_wav_readiness(executable=exe) is None
    batch = exe.with_suffix(".cmd")
    batch.write_bytes(exe.read_bytes())
    assert (
        await profile.controlled_wav_readiness(executable=batch)
        == "qualified_ffplay_unavailable"
    )
    assert await profile.controlled_wav_readiness(executable=Path("ffplay.exe"))
    exe.write_bytes(b"unqualified")
    assert await profile.controlled_wav_readiness(executable=exe)
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    assert (
        await profile.controlled_wav_readiness(executable=exe)
        == "unsupported_audio_route"
    )


@pytest.mark.asyncio
async def test_unsupported_platform_refused_before_any_probe(monkeypatch):
    monkeypatch.setattr(profile.sys, "platform", "linux")
    assert await profile.controlled_wav_readiness() == "unsupported_platform"


@native
@pytest.mark.asyncio
async def test_same_digest_shim_refused_but_later_real_package_discovered(
    retained, tmp_path, monkeypatch
):
    exe, _, _ = retained
    install = tmp_path / "packages"
    shim = install / "bin" / "ffplay.exe"
    wrong = install / "lib" / "ffmpeg-a" / "tools" / "ffplay.exe"
    right = install / "lib" / "ffmpeg-b" / "tools" / "ffplay.exe"
    for path in (shim, wrong, right):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"other" if path == wrong else exe.read_bytes())
    monkeypatch.setenv("ChocolateyInstall", str(install))
    assert (
        await profile.controlled_wav_readiness(executable=shim)
        == "qualified_ffplay_unavailable"
    )
    assert await profile.controlled_wav_readiness() is None
