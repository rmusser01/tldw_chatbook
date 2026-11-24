"""Provenance and synthetic fixtures for the opt-in Windows measurement drivers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import qualification as q
from wasapi_packets import Packet

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
RUNTIME = (
    "TTS/_controlled_wav_windows.py",
    "TTS/controlled_wav_playback.py",
    "TTS/controlled_wav_profile.py",
    "TTS/controlled_wav_progress.py",
    "TTS/audio_output_ownership.py",
    "TTS/audio_output_runs.py",
    "Audio/streaming_sink.py",
    "Utils/windows_files.py",
    "Utils/private_paths.py",
    "Utils/platform_files.py",
)
DRIVERS = (
    "qualification.py",
    "wasapi_packets.py",
    "pulse_fragments.py",
    "controlled_instrument.py",
    "windows_controlled_adapter.py",
    "windows_output_transfers.py",
    "windows_early_exit.py",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def new_output(path):
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "tldw_chatbook" / name for name in RUNTIME] + [
        TOOLS / name for name in DRIVERS
    ]
    metadata = {
        "schema_version": 2,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_file_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in sources
        },
        "python": platform.python_version(),
        "windows_version": platform.version(),
        "machine": platform.machine(),
        "packages": {},
    }
    for package in ("numpy", "scipy", "cffi", "soundcard", "sounddevice"):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][package] = None
    save(directory / "manifest.json", metadata)
    return directory, metadata


def save(path, values):
    text = json.dumps(values, indent=2, allow_nan=False) + "\n"
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


class Fixtures:
    """One new directory below a verified parent, retained on uncertain custody."""

    def __init__(self):
        from tldw_chatbook.Utils.private_paths import (
            _native_close,
            _open_verified_parent,
        )

        parent = Path.home()
        fd, _ = _open_verified_parent(parent / "probe", missing_leaf_allowed=True)
        _native_close(fd)
        self.directory = Path(
            tempfile.mkdtemp(prefix="chatbook-measure-41177-", dir=parent)
        )
        self.identity = self.directory.stat()

    def remove(self):
        path = self.directory
        if (
            path.resolve(strict=True) != path
            or path.parent != Path.home()
            or not self.identity.st_ino
            or not os.path.samestat(path.stat(), self.identity)
        ):
            raise RuntimeError("private_fixture_cleanup_boundary")
        shutil.rmtree(path)


def fixture_mode(path):
    from tldw_chatbook.Utils.platform_files import os as native_files

    return oct(stat.S_IMODE(native_files.stat(path).st_mode))


def make_fixture(directory, duration, seed, channel=None, *, case=""):
    from tldw_chatbook.TTS.controlled_wav_profile import RetainedWav
    from tldw_chatbook.Utils.platform_files import os as native_files

    source = q.marker_signal(duration, seed=seed)
    pcm = q.np.repeat((source * 32767).astype("<i2")[:, None], 2, axis=1)
    if channel is not None:
        pcm[:, 1 - channel] = 0
    path = directory / f"marker-{duration}-{seed}-{channel}-{case}.wav"
    payload = pcm.tobytes()
    with path.open("xb") as stream:
        native_files.fchmod(stream.fileno(), 0o600)
        if stat.S_IMODE(native_files.fstat(stream.fileno()).st_mode) != 0o600:
            raise RuntimeError("fixture_mode")
        with wave.open(stream, "wb") as wav:
            wav.setparams((2, 2, q.RATE, 0, "NONE", "not compressed"))
            wav.writeframes(payload)
            wav.writeframes(bytes(12000 * 4))
    with wave.open(str(path), "rb") as wav:
        artifact = RetainedWav(
            path, sha256(path), q.RATE, len(source), wav.getnframes()
        )
    return artifact, source, payload


def fixture_receipt(artifact):
    return {
        "sha256": artifact.sha256,
        "source_frames": artifact.content_frames,
        "logical_frames": artifact.content_frames,
        "physical_frames": artifact.file_frames,
        "padding_frames": artifact.file_frames - artifact.content_frames,
        "sample_rate": artifact.sample_rate,
        "native_mode": fixture_mode(artifact.path),
    }


def channel_packets(packets, channel):
    return [
        Packet(
            p.frame,
            p.start_s,
            p.flags,
            q.np.repeat(p.data[:, channel, None], 2, axis=1),
        )
        for p in packets
    ]


def separation_check():
    # Feed an independent known mixture through exactly the channel splitter and
    # original matcher. An overlap must remain visible, never be sliced away.
    rate = q.RATE
    old, new = q.marker_signal(0.5, seed=15), q.marker_signal(0.5, seed=16)
    results = []
    for onset in (0.3, 0.7):
        data = q.np.zeros((rate * 2, 2), dtype=q.np.float32)
        data[: len(old), 0] = old
        start = round(onset * rate)
        data[start : start + len(new), 1] = new
        packets = [Packet(0, 10, 0, data)]
        left, _ = q._match_packets(channel_packets(packets, 0), old)
        right, _ = q._match_packets(channel_packets(packets, 1), new)
        gap = right["capture_onset_qpc_s"] - left["capture_end_qpc_s"]
        if (gap < 0) != (onset < 0.5):
            raise RuntimeError("stereo_separation_control_failed")
        results.append(gap)
    return results
