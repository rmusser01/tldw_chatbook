"""Untrusted evidence must be rejected before model construction or file access."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.TTS.test_live_validation_harness import live_args, script, wav_file


@pytest.fixture
def verification(tmp_path, monkeypatch):
    verifier = script("verify_live_tts_content")
    run = tmp_path / "run"
    run.mkdir()
    audio = wav_file(run / "reply.wav")
    model = tmp_path / "asr"
    model.mkdir()
    (model / "model.bin").write_bytes(b"local model fixture")
    row = {
        "id": "reply",
        "outcome": "success",
        "expected_text": "First middle last.",
        "content_anchors": ["first", "middle", "last"],
        "audio": {
            "path": str(audio),
            "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        },
    }
    evidence = run / "evidence.json"
    evidence.write_text(json.dumps({"phases": [row]}))
    output = run / "content.json"
    calls = []

    class Model:
        def __init__(self, path, **kwargs):
            calls.append(("model", path, kwargs))

        def transcribe(self, source, **kwargs):
            calls.append(("transcribe", source, kwargs))
            return iter(
                [SimpleNamespace(start=0.0, end=1.0, text="First middle last.")]
            ), SimpleNamespace(language="en")

    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model)
    )
    monkeypatch.setattr(
        verifier.importlib.metadata, "version", lambda _: "test-version"
    )
    argv = ["--evidence", str(evidence), "--model", str(model), "--output", str(output)]
    return SimpleNamespace(
        verifier=verifier,
        run=run,
        audio=audio,
        model=model,
        row=row,
        evidence=evidence,
        output=output,
        argv=argv,
        calls=calls,
        Model=Model,
    )


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"phases": {}},
        {"phases": [None]},
        {"phases": [{"id": 1, "outcome": "success"}]},
        {"phases": [{"id": "reply", "outcome": "success", "expected_text": "text"}]},
    ],
)
def test_malformed_evidence_returns_consistent_cli_error_before_asr(verification, data):
    v = verification
    v.evidence.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert not v.calls and not v.output.exists()


def test_invalid_json_returns_consistent_cli_error(verification):
    v = verification
    v.evidence.write_text("{broken")
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert not v.calls


@pytest.mark.parametrize(
    "kind", ["absolute", "relative", "leaf_link", "parent_link", "fifo"]
)
def test_audio_cannot_escape_run_or_follow_links_or_open_nonregular_files(
    verification, tmp_path, kind
):
    v = verification
    outside = wav_file(tmp_path / "outside.wav")
    if kind == "absolute":
        path = outside
    elif kind == "relative":
        path = Path("../outside.wav")
    elif kind == "leaf_link":
        path = v.run / "linked.wav"
        path.symlink_to(outside)
    elif kind == "parent_link":
        parent = v.run / "linked"
        parent.symlink_to(tmp_path, target_is_directory=True)
        path = parent / "outside.wav"
    else:
        import os

        path = v.run / "pipe.wav"
        os.mkfifo(path)
    v.row["audio"]["path"] = str(path)
    v.row["audio"]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    v.evidence.write_text(json.dumps({"phases": [v.row]}))
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert not v.calls and not v.output.exists()


def test_duplicate_success_ids_cannot_hide_a_clip(verification):
    v = verification
    v.evidence.write_text(json.dumps({"phases": [v.row, v.row]}))
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert not v.calls


def test_asr_uses_verified_open_stream_and_offline_constructor(verification):
    v = verification
    audio_bytes = v.audio.read_bytes()
    reads = []

    class Model(v.Model):
        def transcribe(self, source, **kwargs):
            assert hasattr(source, "read"), "ASR must receive the verified file handle"
            reads.append(source.read())
            source.seek(0)
            return super().transcribe(source, **kwargs)

    sys.modules["faster_whisper"].WhisperModel = Model
    assert v.verifier.main(v.argv) == 0
    assert reads == [audio_bytes]
    assert v.calls[0][2] == {
        "device": "cpu",
        "compute_type": "int8",
        "local_files_only": True,
    }
    assert json.loads(v.output.read_text())["success_audio_count"] == 1


def test_audio_swap_after_admission_cannot_redirect_transcription(
    verification, tmp_path
):
    v = verification
    outside = wav_file(tmp_path / "outside.wav")

    class Model(v.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            v.audio.unlink()
            v.audio.symlink_to(outside)

    sys.modules["faster_whisper"].WhisperModel = Model
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert not any(row[0] == "transcribe" for row in v.calls)
    assert not v.output.exists()


def test_missing_optional_asr_has_project_guidance_before_model_start(
    verification, monkeypatch, capsys
):
    from tldw_chatbook.Utils import optional_deps

    v = verification

    def unavailable(module, feature):
        assert (module, feature) == ("faster_whisper", "transcription_faster_whisper")
        raise ImportError("Install the transcription_faster_whisper extra")

    monkeypatch.setattr(optional_deps, "require_dependency", unavailable)
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert "transcription_faster_whisper" in capsys.readouterr().err
    assert not v.calls


def test_live_runner_uses_central_path_validation_before_reading(tmp_path):
    runner, args = live_args(tmp_path)
    invalid = tmp_path / "bad;model.pth"
    invalid.write_bytes(args.model.read_bytes())
    args.model = invalid
    with pytest.raises(ValueError, match="dangerous"):
        runner.validate_args(args)
    assert not args.output.exists()


def test_live_runner_rejects_dangling_output_link(tmp_path):
    runner, args = live_args(tmp_path)
    args.output.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="exists|symlink"):
        runner.validate_args(args)
    assert not (tmp_path / "missing").exists()


def test_unknown_phase_outcome_cannot_silently_reduce_denominator(verification):
    v = verification
    typo = {**v.row, "id": "another", "outcome": "sucess"}
    v.evidence.write_text(json.dumps({"phases": [v.row, typo]}))
    with pytest.raises(SystemExit) as error:
        v.verifier.main(v.argv)
    assert error.value.code == 2
    assert not v.calls


def test_optional_dependency_lookup_uses_and_removes_a_private_profile(
    verification, tmp_path, monkeypatch
):
    import os

    from tldw_chatbook.Utils import optional_deps

    v = verification
    protected = tmp_path / "user-profile.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(protected))
    observed = []

    def lookup(module, feature):
        private = Path(os.environ["TLDW_CONFIG_PATH"])
        assert private != protected, (
            "Optional dependency import must not access the user profile"
        )
        private.write_text("private lookup fixture")
        observed.append(private)
        return SimpleNamespace(WhisperModel=v.Model)

    monkeypatch.setattr(optional_deps, "require_dependency", lookup)
    assert v.verifier.main(v.argv) == 0
    assert observed and not observed[0].parent.exists()
    assert os.environ["TLDW_CONFIG_PATH"] == str(protected)
    assert not protected.exists()


@pytest.mark.parametrize("device", ["cuda:9", "CUDA", "", None, 3, ["cuda"]])
def test_live_runner_rejects_unadmitted_device_before_output(tmp_path, device):
    runner, args = live_args(tmp_path, device=device)
    with pytest.raises(ValueError, match="device"):
        runner.validate_args(args)
    assert not args.output.exists()


def test_missing_optional_pytorch_has_project_install_guidance(monkeypatch):
    from tldw_chatbook.Utils import optional_deps

    runner = script("validate_live_tts")
    # Simulate only package absence; exercise the real dependency error policy.
    monkeypatch.setattr(optional_deps, "check_dependency", lambda *args: False)
    with pytest.raises(ImportError, match=r"pip install tldw_chatbook\[local_tts\]"):
        runner.load_pytorch_runtime()


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_live_runner_admits_supported_device_without_starting_output(tmp_path, device):
    runner, args = live_args(tmp_path, device=device)
    runner.validate_args(args)
    assert args.device == device
    assert not args.output.exists()
