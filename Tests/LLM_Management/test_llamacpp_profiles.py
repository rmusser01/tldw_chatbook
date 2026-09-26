"""Behavioral checks for device-local llama.cpp tuning profiles."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_chatbook.LLM_Management.llamacpp_profiles import (
    LlamaCppLaunchProfileV1,
    LlamaCppProfileConflict,
    LlamaCppProfileCorrupt,
    LlamaCppProfileFutureVersion,
    LlamaCppProfileRepository,
    LlamaCppProfileValidationError,
    LlamaCppTuning,
    build_tuning_arguments,
    default_llamacpp_profile_path,
)


def profile(name: str = "Balanced", **tuning: object) -> LlamaCppLaunchProfileV1:
    return LlamaCppLaunchProfileV1(
        profile_id=str(uuid4()), name=name, tuning=LlamaCppTuning(**tuning)
    )


def test_default_path_uses_active_user_data_directory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: tmp_path)
    assert default_llamacpp_profile_path() == tmp_path / "llamacpp_launch_profiles.json"


def test_runtime_defaults_emit_no_flags_and_keep_session_expert_arguments():
    assert build_tuning_arguments(LlamaCppTuning(), ("--verbose",)) == ("--verbose",)


def test_typed_flags_include_gpu_all_layers_and_flash_modes():
    tuning = LlamaCppTuning(
        context_size=4096,
        gpu_layers=-1,
        threads=8,
        parallel=2,
        flash_attention="off",
        cache_type_k="q8_0",
        cache_type_v="f16",
        batch_size=512,
        ubatch_size=128,
    )
    assert build_tuning_arguments(tuning, ()) == (
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--threads",
        "8",
        "--parallel",
        "2",
        "--flash-attn",
        "off",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "f16",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
    )
    assert build_tuning_arguments(LlamaCppTuning(flash_attention="auto"), ()) == (
        "--flash-attn",
        "auto",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_size", 0),
        ("context_size", True),
        ("context_size", 4096.0),
        ("gpu_layers", -2),
        ("threads", 0),
        ("parallel", 0),
        ("batch_size", -1),
        ("ubatch_size", 0),
        ("flash_attention", "yes"),
        ("cache_type_k", "unsafe"),
    ],
)
def test_tuning_rejects_invalid_values(field: str, value: object):
    with pytest.raises(LlamaCppProfileValidationError):
        LlamaCppTuning(**{field: value})


@pytest.mark.parametrize(
    "structured,raw",
    [
        (LlamaCppTuning(context_size=4096), ("-c", "8192")),
        (LlamaCppTuning(context_size=4096), ("--ctx-size=8192",)),
        (LlamaCppTuning(gpu_layers=-1), ("-ngl=-1",)),
        (LlamaCppTuning(gpu_layers=0), ("--gpu-layers", "0")),
        (LlamaCppTuning(threads=8), ("-t", "4")),
        (LlamaCppTuning(parallel=2), ("-np=4",)),
        (LlamaCppTuning(flash_attention="on"), ("-fa",)),
        (LlamaCppTuning(flash_attention="off"), ("--flash-attn=auto",)),
        (LlamaCppTuning(cache_type_k="f16"), ("-ctk=q8_0",)),
        (LlamaCppTuning(cache_type_v="f16"), ("--cache-type-v", "q8_0")),
        (LlamaCppTuning(batch_size=512), ("-b=1024",)),
        (LlamaCppTuning(ubatch_size=128), ("-ub", "256")),
    ],
)
def test_structured_field_rejects_raw_alias_conflict(structured, raw):
    with pytest.raises(LlamaCppProfileValidationError, match="conflict"):
        build_tuning_arguments(structured, raw)


@pytest.mark.parametrize(
    "raw",
    [
        ("--model=/private/model.gguf",),
        ("-m", "/private/model.gguf"),
        ("--alias=other",),
        ("-a", "other"),
        ("-a=other",),
        ("--host", "0.0.0.0"),
        ("--port=9000",),
        ("--model-url=https://example.invalid/model",),
        ("-hf", "org/model"),
        ("--hf-repo=org/model",),
    ],
)
def test_source_and_connection_flags_remain_reserved(raw):
    with pytest.raises(LlamaCppProfileValidationError, match="reserved"):
        build_tuning_arguments(LlamaCppTuning(), raw)


def test_unset_structured_field_allows_raw_flag():
    assert build_tuning_arguments(LlamaCppTuning(threads=4), ("--ctx-size=8192",)) == (
        "--threads",
        "4",
        "--ctx-size=8192",
    )


def test_round_trip_saves_tuning_only_and_cas_delete(tmp_path: Path):
    path = tmp_path / "profiles.json"
    repository = LlamaCppProfileRepository(path)
    assert repository.load().revision == 0
    saved = profile("GPU", gpu_layers=-1, flash_attention="on")
    first = repository.save(saved, expected_revision=0)
    assert first.revision == 1
    assert repository.load().profiles == (saved,)
    payload = json.loads(path.read_text())
    assert set(payload) == {"version", "revision", "profiles"}
    assert set(payload["profiles"][0]) == {"profile_id", "name", "tuning"}
    assert set(payload["profiles"][0]["tuning"]) == set(
        LlamaCppTuning.__dataclass_fields__
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert repository.delete(saved.profile_id, expected_revision=1).revision == 2
    assert repository.load().profiles == ()


def test_stale_save_and_delete_preserve_current_bytes(tmp_path: Path):
    path = tmp_path / "profiles.json"
    first = LlamaCppProfileRepository(path)
    second = LlamaCppProfileRepository(path)
    saved = first.save(profile(), expected_revision=0)
    original = path.read_bytes()
    with pytest.raises(LlamaCppProfileConflict):
        second.save(profile("Other"), expected_revision=0)
    with pytest.raises(LlamaCppProfileConflict):
        second.delete(saved.profiles[0].profile_id, expected_revision=0)
    assert path.read_bytes() == original


def test_two_processes_cannot_both_save_the_same_revision(tmp_path: Path):
    path = tmp_path / "profiles.json"
    start = tmp_path / "start"
    script = """
import sys
import time
from pathlib import Path
from uuid import uuid4
from tldw_chatbook.LLM_Management.llamacpp_profiles import (
    LlamaCppLaunchProfileV1, LlamaCppProfileConflict,
    LlamaCppProfileRepository, LlamaCppTuning,
)
path, start, ready, name = map(Path, sys.argv[1:])
ready.touch()
deadline = time.monotonic() + 10
while not start.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('barrier timeout')
    time.sleep(0.01)
try:
    LlamaCppProfileRepository(path).save(
        LlamaCppLaunchProfileV1(str(uuid4()), name.name, LlamaCppTuning()),
        expected_revision=0,
    )
except LlamaCppProfileConflict:
    print('conflict')
else:
    print('saved')
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(start),
                str(tmp_path / f"ready-{index}"),
                f"Profile-{index}",
            ],
            cwd=Path(__file__).parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(2)
    ]
    deadline = time.monotonic() + 10
    while not all((tmp_path / f"ready-{index}").exists() for index in range(2)):
        if time.monotonic() > deadline:
            pytest.fail("race workers did not reach the barrier")
        time.sleep(0.01)
    start.touch()
    outcomes = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        outcomes.append(stdout.strip())
    assert sorted(outcomes) == ["conflict", "saved"]
    document = LlamaCppProfileRepository(path).load()
    assert document.revision == 1
    assert len(document.profiles) == 1


def test_duplicate_casefolded_names_and_ids_are_rejected(tmp_path: Path):
    repo = LlamaCppProfileRepository(tmp_path / "profiles.json")
    first = repo.save(profile("Balanced"), expected_revision=0)
    with pytest.raises(LlamaCppProfileValidationError):
        repo.save(profile(" balanced "), expected_revision=1)
    with pytest.raises(LlamaCppProfileValidationError):
        repo.save(
            replace(first.profiles[0], profile_id="not-a-uuid"), expected_revision=1
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"version":1,"revision":0,"profiles":',
        b'{"version":1,"version":1,"revision":0,"profiles":[]}',
        b'{"version":1,"revision":0,"profiles":[{"profile_id":"x","name":"a","tuning":{"threads":1,"threads":2}}]}',
        b'{"version":2,"revision":0,"profiles":[]}',
        b" " * (256 * 1024 + 1),
    ],
)
def test_bad_documents_refuse_overwrite(tmp_path: Path, raw: bytes):
    path = tmp_path / "profiles.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    error = (
        LlamaCppProfileFutureVersion
        if b'"version":2' in raw
        else LlamaCppProfileCorrupt
    )
    with pytest.raises(error):
        LlamaCppProfileRepository(path).save(profile(), expected_revision=0)
    assert path.read_bytes() == raw


def test_unknown_profile_field_does_not_persist_source_authority(tmp_path: Path):
    path = tmp_path / "profiles.json"
    repo = LlamaCppProfileRepository(path)
    saved = repo.save(profile(), expected_revision=0)
    payload = json.loads(path.read_text())
    payload["profiles"][0]["model_path"] = "/private/model.gguf"
    path.write_text(json.dumps(payload))
    with pytest.raises(LlamaCppProfileCorrupt):
        repo.load()
    assert saved.revision == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(revision=True),
        lambda data: data["profiles"][0]["tuning"].update(threads=4.0),
        lambda data: data["profiles"][0]["tuning"].update(gpu_layers=False),
        lambda data: data["profiles"].append(data["profiles"][0]),
    ],
)
def test_strict_json_ingress_rejects_coercion_and_duplicate_profiles(
    tmp_path: Path, mutate
):
    path = tmp_path / "profiles.json"
    LlamaCppProfileRepository(path).save(profile(), expected_revision=0)
    payload = json.loads(path.read_text())
    mutate(payload)
    original = json.dumps(payload).encode()
    path.write_bytes(original)
    with pytest.raises(LlamaCppProfileCorrupt):
        LlamaCppProfileRepository(path).load()
    assert path.read_bytes() == original


def test_atomic_replace_failure_preserves_existing_document(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "profiles.json"
    repo = LlamaCppProfileRepository(path)
    saved = repo.save(profile(), expected_revision=0)
    original = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("replacement unavailable")

    monkeypatch.setattr("tldw_chatbook.Utils.atomic_file_ops.os.replace", fail_replace)
    with pytest.raises(LlamaCppProfileCorrupt):
        repo.save(replace(saved.profiles[0], name="Renamed"), expected_revision=1)
    assert path.read_bytes() == original


def test_profile_cap_and_document_symlink_refusal(tmp_path: Path):
    path = tmp_path / "profiles.json"
    repo = LlamaCppProfileRepository(path)
    for index in range(32):
        repo.save(profile(f"Profile {index}"), expected_revision=index)
    with pytest.raises(LlamaCppProfileValidationError):
        repo.save(profile("Overflow"), expected_revision=32)
    target = tmp_path / "target.json"
    target.write_bytes(path.read_bytes())
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(LlamaCppProfileCorrupt):
        LlamaCppProfileRepository(link).load()
    with pytest.raises(LlamaCppProfileCorrupt):
        LlamaCppProfileRepository(link).save(profile("Nope"), expected_revision=32)


def test_lock_symlink_refuses_write(tmp_path: Path):
    path = tmp_path / "profiles.json"
    target = tmp_path / "private"
    target.write_text("untouched")
    (tmp_path / "profiles.json.lock").symlink_to(target)
    with pytest.raises(LlamaCppProfileCorrupt):
        LlamaCppProfileRepository(path).save(profile(), expected_revision=0)
    assert target.read_text() == "untouched"
    assert not path.exists()


def test_projector_expert_arguments_keep_existing_owner():
    raw = ("--mmproj", "/private/projector.gguf")
    assert build_tuning_arguments(LlamaCppTuning(), raw) == raw


def test_excessive_json_nesting_is_recoverable_and_preserved(tmp_path):
    path = tmp_path / "profiles.json"
    payload = b"[" * 20000 + b"]" * 20000
    path.write_bytes(payload)
    path.chmod(0o600)
    with pytest.raises(LlamaCppProfileCorrupt):
        LlamaCppProfileRepository(path).load()
    assert path.read_bytes() == payload
