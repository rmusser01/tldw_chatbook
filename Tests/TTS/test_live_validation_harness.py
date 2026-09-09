"""Negative controls for live evidence; these tests never load a speech model."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import struct
import subprocess
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    path = ROOT / "scripts" / f"{name}.py"
    assert path.is_file(), f"Missing runnable harness: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav_file(path, seconds=1):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes(struct.pack("<h", 2400) * int(24000 * seconds))
    return path


def test_help_and_import_do_not_touch_a_profile_or_import_audio(tmp_path):
    # A top-level app/ML import or eager profile setup must break this test.
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    probe = """
import importlib.util, pathlib, sys
class DenyRuntime:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'tldw_chatbook', 'torch', 'kokoro', 'kokoro_onnx', 'sounddevice', 'soundfile', 'faster_whisper'}:
            raise AssertionError('eager runtime import: ' + fullname)
sys.meta_path.insert(0, DenyRuntime())
for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.main(['--help'])
    except SystemExit as error:
        assert error.code == 0
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            probe,
            str(ROOT / "scripts/validate_live_tts.py"),
            str(ROOT / "scripts/verify_live_tts_content.py"),
        ],
        cwd=probe_root,
        env={**os.environ, "TLDW_CONFIG_PATH": str(probe_root / "profile.toml")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(probe_root.iterdir()) == []


def live_args(tmp_path, **overrides):
    runner = script("validate_live_tts")
    model = tmp_path / "model.pth"
    model.write_bytes(b"supplied checkpoint")
    (tmp_path / "config.json").write_text("{}")
    voices = tmp_path / "voices"
    voices.mkdir(exist_ok=True)
    (voices / "af_heart.pt").write_bytes(b"supplied voice")
    package = tmp_path / "tldw_chatbook"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    args = runner.build_parser().parse_args(
        [
            "--engine",
            "pytorch",
            "--model",
            str(model),
            "--voice-dir",
            str(voices),
            "--voice",
            "af_heart",
            "--expected-package-root",
            str(package),
            "--output",
            str(tmp_path / "run"),
            "--play-audio",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return runner, args


@pytest.mark.parametrize(
    "change, match",
    [
        ({"play_audio": False}, "play-audio"),
        ({"repeats": 1000}, "repeats"),
        ({"phase_timeout": float("nan")}, "finite"),
        ({"voice": "../escape"}, "voice"),
        ({"scenarios": "download"}, "scenario"),
        ({"language": "ja"}, "voice"),
    ],
)
def test_invalid_live_request_fails_before_creating_output(tmp_path, change, match):
    runner, args = live_args(tmp_path, **change)
    with pytest.raises(ValueError, match=match):
        runner.validate_args(args)
    assert not args.output.exists()


def test_missing_adjacent_model_config_cannot_trigger_a_download(tmp_path):
    runner, args = live_args(tmp_path)
    (tmp_path / "config.json").unlink()
    with pytest.raises(ValueError, match="config.json"):
        runner.validate_args(args)
    assert not args.output.exists()


def test_existing_output_is_never_reused(tmp_path):
    runner, args = live_args(tmp_path)
    args.output.mkdir()
    evidence = args.output / "evidence.json"
    evidence.write_text("original")
    with pytest.raises(ValueError, match="exists"):
        runner.validate_args(args)
    assert evidence.read_text() == "original"


def test_package_provenance_checks_the_imported_location(tmp_path):
    runner = script("validate_live_tts")
    with pytest.raises(ValueError, match="package"):
        runner.check_package_root(tmp_path / "old", tmp_path / "requested")
    runner.check_package_root(tmp_path / "requested", tmp_path / "requested")


def test_worker_clears_ambient_kokoro_asset_overrides(tmp_path, monkeypatch):
    runner = script("validate_live_tts")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.update(KOKORO_MODEL_PATH="ambient.pth", KOKORO_VOICES_PATH="ambient.bin")
    runner.setup_environment(tmp_path)
    assert "KOKORO_MODEL_PATH" not in os.environ
    assert "KOKORO_VOICES_PATH" not in os.environ


def test_backend_model_and_engine_must_match_before_initialization(tmp_path):
    runner, args = live_args(tmp_path)
    backend = SimpleNamespace(
        use_onnx=False, model_path=str(args.model), voice_dir=str(args.voice_dir)
    )
    runner.check_backend_assets(backend, args)
    backend.model_path = str(tmp_path / "ambient.pth")
    with pytest.raises(ValueError, match="model"):
        runner.check_backend_assets(backend, args)
    backend.model_path = str(args.model)
    backend.use_onnx = True
    with pytest.raises(ValueError, match="engine"):
        runner.check_backend_assets(backend, args)


def test_native_observer_delegates_and_retains_the_original_request():
    runner = script("validate_live_tts")
    calls = []
    entered, release = threading.Event(), threading.Event()
    original_error = RuntimeError("native failure")
    errors = []

    def operation(value):
        assert value == 17
        entered.set()
        assert release.wait(2)
        raise original_error

    observed = runner.observe_native(operation, calls, {"request_id": "old"})

    def work():
        try:
            observed(17)
        except RuntimeError as error:
            errors.append(error)

    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(2)
    try:
        assert calls[0]["request_id"] == "old"
        assert "exit_at" not in calls[0]
        assert calls[0]["thread_id"] != threading.get_ident()
    finally:
        release.set()
        thread.join(2)
    assert errors == [original_error]
    assert calls[0]["exit_at"] >= calls[0]["enter_at"]
    assert calls[0]["exception"] == "RuntimeError"


@pytest.mark.parametrize(
    "stop, end, settled, passes",
    [
        (2.0, 3.0, 4.0, True),
        (0.5, 3.0, 4.0, False),
        (3.5, 3.0, 4.0, False),
        (2.0, None, 4.0, False),
        (2.0, 5.0, 4.0, False),
    ],
)
def test_cancel_requires_overlap_and_native_completion(stop, end, settled, passes):
    runner = script("validate_live_tts")
    call = {"enter_at": 1.0, "thread_id": -1}
    if end is not None:
        call["exit_at"] = end
    row = {"native_calls": [call], "stop_requested_at": stop, "settled_at": settled}
    if passes:
        runner.validate_cancellation(row)
    else:
        with pytest.raises(ValueError):
            runner.validate_cancellation(row)


def test_cleanup_cannot_hide_active_native_work_behind_an_empty_host():
    runner = script("validate_live_tts")
    with pytest.raises(ValueError, match="native_calls"):
        runner.assert_quiescent({"manager_attached": 0, "native_calls": 1})
    runner.assert_quiescent({"native_calls": 0, "leases": 0, "retained_io": 0})


@pytest.mark.parametrize(
    "elapsed, exit_code, passes",
    [
        (2.0, 0, False),
        (6.13, 0, True),
        (6.13, 1, False),
    ],
)
def test_successful_player_exit_does_not_prove_full_playback(
    elapsed, exit_code, passes
):
    runner = script("validate_live_tts")
    row = {
        "audio": {"seconds": 5.717, "frames": 137208},
        "playback": {
            "kind": "file",
            "elapsed_seconds": elapsed,
            "exit_code": exit_code,
        },
    }
    if passes:
        runner.validate_playback(row)
    else:
        with pytest.raises(ValueError):
            runner.validate_playback(row)


def test_waveform_validation_reads_all_declared_frames(tmp_path):
    runner = script("validate_live_tts")
    path = wav_file(tmp_path / "full.wav")
    audio = runner.inspect_wav(path)
    assert audio["frames"] == 24000
    assert audio["seconds"] == 1.0
    assert audio["rms"] == pytest.approx(2400 / 32768)
    path.write_bytes(path.read_bytes()[:-12000])
    with pytest.raises(ValueError, match="truncat"):
        runner.inspect_wav(path)


def test_content_denominator_includes_every_success_and_checks_hashes(tmp_path):
    verifier = script("verify_live_tts_content")
    path = wav_file(tmp_path / "reply.wav")
    audio = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    rows = [
        {"id": str(i), "outcome": "success", "audio": audio, "expected_text": "words"}
        for i in range(6)
    ]
    rows.append({"id": "cancelled", "outcome": "cancelled"})
    evidence = {"phases": rows}
    assert len(verifier.success_audio(evidence, tmp_path.resolve())) == 6
    rows[-2]["audio"] = {**audio, "sha256": "changed"}
    with pytest.raises(ValueError, match="hash"):
        verifier.success_audio(evidence, tmp_path.resolve())


@pytest.mark.parametrize(
    "transcript, complete",
    [
        ("Silver compass. Quiet orchard. Amber sunrise.", True),
        ("Silver compass. Amber sunrise.", False),
        ("Amber sunrise. Quiet orchard. Silver compass.", False),
    ],
)
def test_content_requires_beginning_middle_and_end_in_order(transcript, complete):
    verifier = script("verify_live_tts_content")
    result = verifier.compare_content(
        "Silver compass. Quiet orchard. Amber sunrise.",
        transcript,
        ["silver compass", "quiet orchard", "amber sunrise"],
    )
    assert result["anchors_complete"] is complete


def test_multilingual_content_preserves_lexical_asr_differences():
    verifier = script("verify_live_tts_content")
    raw = "朝の光。静かな森。青い鳥。"
    result = verifier.compare_content(
        "朝の光。静かな森。赤い鳥。", raw, ["朝の光", "静かな森", "赤い鳥"]
    )
    assert result["transcript"] == raw
    assert result["normalized_exact"] is False
    assert result["anchors_complete"] is False


def test_content_normalization_preserves_hindi_vowels_and_nasal_marks():
    verifier = script("verify_live_tts_content")
    assert verifier.normalize("चाँदी।") == "चाँदी"
    assert verifier.normalize("चाँदी") != verifier.normalize("चदी")


def test_missing_success_audio_fails_instead_of_shrinking_the_denominator(tmp_path):
    verifier = script("verify_live_tts_content")
    with pytest.raises(ValueError, match="audio"):
        verifier.success_audio(
            {"phases": [{"id": "retry", "outcome": "success"}]}, tmp_path.resolve()
        )
