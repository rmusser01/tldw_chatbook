"""Task 6: system-audio tap resolvers, subprocess reader, probe."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tldw_chatbook.Audio import system_audio_tap as sat

pytestmark = pytest.mark.unit

FAKE = Path(__file__).with_name("fake_audiotap.py")


def _fake_cmd(*extra: str) -> tuple[str, ...]:
    return (sys.executable, str(FAKE), *extra)


# ---- pure resolvers -------------------------------------------------------

def test_sink_name_validation():
    assert sat.validate_sink_name("alsa_output.pci-0000_00_1f.3.analog-stereo") == (
        "alsa_output.pci-0000_00_1f.3.analog-stereo"
    )
    for bad in ("", "sink;rm -rf /", "a b", "x$(y)", "über"):
        with pytest.raises(ValueError):
            sat.validate_sink_name(bad)


def test_parse_default_sink_takes_first_line():
    assert sat.parse_default_sink("my_sink\n") == "my_sink"
    with pytest.raises(ValueError):
        sat.parse_default_sink("")


def test_linux_capture_commands():
    assert sat.linux_capture_command("parec", "s1") == (
        "parec", "--device=s1.monitor", "--format=s16le", "--rate=16000",
        "--channels=1", "--latency-msec=20",
    )
    assert sat.linux_capture_command("pw-record", "s1") == (
        "pw-record", "--target", "s1.monitor", "--rate", "16000", "--channels", "1",
        "--format", "s16", "-",
    )
    with pytest.raises(ValueError):
        sat.linux_capture_command("arecord", "s1")


def test_resolve_wasapi_loopback_matches_default_output():
    devices = [
        {"name": "Speakers (Realtek)", "max_input_channels": 0},
        {"name": "Speakers (Realtek) [Loopback]", "max_input_channels": 2},
        {"name": "Headphones [Loopback]", "max_input_channels": 2},
    ]
    assert sat.resolve_wasapi_loopback(devices, "Speakers (Realtek)") == 1
    assert sat.resolve_wasapi_loopback(devices, "Monitor") is None


def test_macos_version_gate():
    assert sat.macos_version_ok("14.2") and sat.macos_version_ok("26.5.2")
    assert not sat.macos_version_ok("14.1.1") and not sat.macos_version_ok("")


# ---- helper lookup --------------------------------------------------------

def test_helper_source_ships_in_package():
    assert sat.helper_source_path().name == "main.swift"


def test_ensure_helper_prefers_bundled_then_dev_then_compiles(tmp_path, monkeypatch):
    bundled_dir = tmp_path / "Contents" / "MacOS"
    bundled_dir.mkdir(parents=True)
    exe = bundled_dir / "python"
    exe.write_text("")
    helper = bundled_dir / "tldw-audiotap"
    helper.write_text("")
    assert sat.ensure_helper(tmp_path / "data", executable=str(exe)) == helper

    data_dir = tmp_path / "data"
    dev = sat.dev_helper_path(data_dir)
    dev.parent.mkdir(parents=True)
    dev.write_text("")
    assert sat.ensure_helper(data_dir, executable=str(tmp_path / "nowhere")) == dev
    dev.unlink()

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        Path(args[args.index("-o") + 1]).write_text("")
        return subprocess.CompletedProcess(args, 0)

    got = sat.ensure_helper(
        data_dir, run=fake_run, which=lambda name: "/usr/bin/swiftc",
        executable=str(tmp_path / "nowhere"),
    )
    assert got == dev and calls[0][0] == "/usr/bin/swiftc" and "-O" in calls[0]
    assert sat.ensure_helper(
        tmp_path / "other", which=lambda name: None, executable=str(tmp_path / "nowhere")
    ) is None


def test_ensure_helper_returns_none_when_source_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sat, "helper_source_path", lambda: tmp_path / "absent.swift")
    assert sat.ensure_helper(
        tmp_path, which=lambda n: "/usr/bin/swiftc", executable=str(tmp_path / "nowhere")
    ) is None
    assert sat.probe(platform="darwin", mac_ver="15.0", data_dir=tmp_path).kind == "unavailable"


# ---- probe ----------------------------------------------------------------

def test_probe_virtual_device_wins_over_platform():
    mode = sat.probe(system_source="BlackHole 2ch", platform="darwin")
    assert mode.kind == "virtual_device" and mode.device_name == "BlackHole 2ch"


def test_probe_macos_old_version_unavailable():
    mode = sat.probe(platform="darwin", mac_ver="13.6", data_dir=Path("/tmp/x"))
    assert mode.kind == "unavailable" and "14.2" in mode.reason


def test_probe_macos_uses_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(sat, "ensure_helper", lambda data_dir, **kw: tmp_path / "tap")
    mode = sat.probe(platform="darwin", mac_ver="15.0", data_dir=tmp_path)
    assert mode.kind == "native_macos" and mode.command == (str(tmp_path / "tap"),)


def test_probe_linux_parec_then_pw_record_then_unavailable():
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="sink_a\n")

    mode = sat.probe(platform="linux", which=lambda n: "/usr/bin/parec" if n == "parec" else None, run=run)
    assert mode.kind == "native_parec" and mode.command[0] == "parec"
    mode = sat.probe(platform="linux", which=lambda n: "/usr/bin/pw-record" if n == "pw-record" else None, run=run)
    assert mode.kind == "native_parec" and mode.command[0] == "pw-record"
    mode = sat.probe(platform="linux", which=lambda n: None, run=run)
    assert mode.kind == "unavailable"


def test_probe_linux_rejects_hostile_sink_name():
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="bad;name\n")

    mode = sat.probe(platform="linux", which=lambda n: "/usr/bin/parec", run=run)
    assert mode.kind == "unavailable" and "sink" in mode.reason.lower()


def test_probe_windows_loopback():
    devices = [{"name": "Speakers [Loopback]", "max_input_channels": 2}]
    mode = sat.probe(platform="win32", query_devices=lambda: devices, default_output_name="Speakers")
    assert mode.kind == "native_wasapi" and mode.device_index == 0
    mode = sat.probe(platform="win32", query_devices=lambda: [], default_output_name="Speakers")
    assert mode.kind == "unavailable"


# ---- SubprocessTap --------------------------------------------------------

def _collect(tap: sat.SubprocessTap, expected: int, timeout: float = 5.0) -> list[bytes]:
    frames: list[bytes] = []
    done = threading.Event()

    def on_frames(frame: bytes) -> None:
        frames.append(frame)
        if len(frames) >= expected:
            done.set()

    assert tap.start(on_frames) is True
    done.wait(timeout)
    return frames


def test_subprocess_tap_delivers_frames_and_stops_cleanly():
    tap = sat.SubprocessTap(_fake_cmd("--frames", "5", "--hold"))
    frames = _collect(tap, 5)
    assert len(frames) == 5 and all(len(f) == 640 for f in frames)
    assert tap.state == "running"
    tap.stop()
    assert tap.state == "stopped" and tap.exit_code == 0 and tap.restarts == 0


def test_subprocess_tap_restarts_once_then_reports_lost():
    tap = sat.SubprocessTap(
        _fake_cmd("--frames", "3", "--exit-code", "1"),
        restart_delay_s=0.01, sleep=lambda s: None,
    )
    frames = _collect(tap, 6)
    deadline = time.monotonic() + 5
    while tap.state != "lost" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(frames) == 6 and tap.restarts == 1 and tap.state == "lost"
    assert tap.exit_code == 1
    tap.stop()


def test_stop_during_restart_delay_reports_stopped():
    tap = sat.SubprocessTap(
        _fake_cmd("--frames", "3", "--exit-code", "1"), restart_delay_s=0.5,
    )
    _collect(tap, 3)
    deadline = time.monotonic() + 2
    while tap.restarts != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tap.restarts == 1
    tap.stop()
    assert tap.state == "stopped" and tap.restarts == 1


def test_subprocess_tap_start_false_when_spawn_fails():
    def boom(*args, **kwargs):
        raise FileNotFoundError("no helper")

    tap = sat.SubprocessTap(("missing",), spawn=boom)
    assert tap.start(lambda f: None) is False and tap.state == "lost"


# ---- DeviceTap + build_tap ------------------------------------------------

class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = None
        self.callback = None
        self.stopped = 0

    def get_audio_devices(self):
        return [{"id": 3, "name": "BlackHole 2ch"}, {"id": 4, "name": "MacBook Pro Microphone"}]

    def set_device(self, device_id):
        self.device = device_id
        return True

    def start_recording(self, callback=None, save_to_file=None):
        self.callback = callback
        return True

    def stop_recording(self):
        self.stopped += 1


def test_device_tap_selects_device_by_name_and_forwards_frames():
    made: list[_Recorder] = []

    def factory(**kwargs):
        made.append(_Recorder(**kwargs))
        return made[-1]

    tap = sat.DeviceTap("BlackHole 2ch", recorder_factory=factory)
    got: list[bytes] = []
    assert tap.start(got.append) is True
    assert made[0].kwargs == {"use_vad": False, "retain_audio": False, "chunk_size": 320}
    assert made[0].device == 3 and tap.state == "running"
    made[0].callback(b"\x00" * 640)
    assert got == [b"\x00" * 640]
    tap.stop()
    assert made[0].stopped == 1 and tap.state == "stopped"


def test_device_tap_refuses_unknown_device():
    # The user named a loopback device that is not plugged in. Falling
    # through to the default INPUT (the old behaviour: device_id stayed
    # None, so set_device was skipped and the recorder opened the default)
    # would record the room through the mic a second time and label it
    # "others" -- worse than having no system audio at all.
    made: list[_Recorder] = []

    def factory(**kwargs):
        made.append(_Recorder(**kwargs))
        return made[-1]

    tap = sat.DeviceTap("VB-Cable (not plugged in)", recorder_factory=factory)
    assert tap.start(lambda frame: None) is False
    assert tap.state == "lost"
    assert made[0].device is None and made[0].callback is None


def test_build_tap_by_kind():
    assert sat.build_tap(sat.TapMode("unavailable", "x")) is None
    assert isinstance(sat.build_tap(sat.TapMode("native_parec", "x", command=("parec",))), sat.SubprocessTap)
    assert isinstance(sat.build_tap(sat.TapMode("native_macos", "x", command=("/h",))), sat.SubprocessTap)
    assert isinstance(
        sat.build_tap(sat.TapMode("virtual_device", "x", device_name="BlackHole"), recorder_factory=_Recorder),
        sat.DeviceTap,
    )
    assert isinstance(
        sat.build_tap(sat.TapMode("native_wasapi", "x", device_index=1), recorder_factory=_Recorder),
        sat.DeviceTap,
    )
