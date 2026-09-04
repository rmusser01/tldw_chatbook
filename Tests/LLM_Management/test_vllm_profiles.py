from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import portalocker
import pytest

from tldw_chatbook.UI.LLM_Management import vllm_profiles as profile_storage
from tldw_chatbook.UI.LLM_Management.vllm_profiles import (
    DEFAULT_PROFILE_NAME,
    MAX_VLLM_PROFILES,
    VllmLaunchProfileV1,
    VllmProfileConflict,
    VllmProfileCorrupt,
    VllmProfileFutureVersion,
    VllmProfileRepository,
    VllmProfileValidationError,
    default_vllm_profile_path,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import VllmModelSource

PROFILE_KEYS = {
    "profile_id",
    "name",
    "python_environment",
    "model_source",
    "model_value",
    "bind_address",
    "port",
    "dtype",
    "tensor_parallel_size",
    "maximum_model_length",
    "gpu_memory_utilization",
    "trust_remote_code",
}


def profile_named(name: str, **changes: object) -> VllmLaunchProfileV1:
    values: dict[str, object] = {
        "profile_id": str(uuid4()),
        "name": name,
        "python_environment": "/opt/venvs/vllm/bin/python",
        "model_source": VllmModelSource.HUGGING_FACE,
        "model_value": "org/model",
        "bind_address": "127.0.0.1",
        "port": 8000,
        "dtype": "auto",
        "tensor_parallel_size": None,
        "maximum_model_length": None,
        "gpu_memory_utilization": None,
        "trust_remote_code": False,
    }
    values.update(changes)
    return VllmLaunchProfileV1(**values)  # type: ignore[arg-type]


def test_default_path_is_device_local_active_profile_data(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "tldw_chatbook.config.get_user_data_dir", lambda: tmp_path / "profile-data"
    )

    assert default_vllm_profile_path() == (
        tmp_path / "profile-data" / "vllm_launch_profiles.json"
    )


def test_profile_round_trip_has_exact_v1_schema_and_excludes_launch_only_fields(
    tmp_path: Path,
):
    path = tmp_path / "vllm_launch_profiles.json"
    repo = VllmProfileRepository(path)
    saved = repo.save(profile_named("GPU 0"), expected_revision=0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"version", "revision", "selected_profile_id", "profiles"}
    assert set(payload["profiles"][0]) == PROFILE_KEYS
    raw = path.read_text(encoding="utf-8")
    for forbidden in (
        "raw_arguments",
        "credential",
        "environment_variables",
        "process_id",
        "pid",
        "http_body",
        "child_output",
    ):
        assert forbidden not in raw
    assert repo.load().profiles == (saved.profile,)
    assert saved.document.revision == 1
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_id", "not-a-uuid"),
        ("name", 7),
        ("python_environment", False),
        ("model_source", "unknown"),
        ("model_value", 4),
        ("bind_address", []),
        ("port", True),
        ("dtype", "float64"),
        ("tensor_parallel_size", 0),
        ("maximum_model_length", True),
        ("gpu_memory_utilization", 1.1),
        ("trust_remote_code", 1),
    ],
)
def test_load_rejects_invalid_profile_field_types_and_values(
    tmp_path: Path, field: str, value: object
):
    profile = {
        "profile_id": str(uuid4()),
        "name": "Strict",
        "python_environment": "python",
        "model_source": "hugging_face",
        "model_value": "org/model",
        "bind_address": "127.0.0.1",
        "port": 8000,
        "dtype": "auto",
        "tensor_parallel_size": None,
        "maximum_model_length": None,
        "gpu_memory_utilization": None,
        "trust_remote_code": False,
    }
    profile[field] = value
    path = tmp_path / "vllm_launch_profiles.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "revision": 1,
                "selected_profile_id": profile["profile_id"],
                "profiles": [profile],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    with pytest.raises(VllmProfileCorrupt):
        VllmProfileRepository(path).load()


def test_load_rejects_unknown_document_or_profile_keys(tmp_path: Path):
    VllmProfileRepository(tmp_path / "source.json").save(
        profile_named("Exact"), expected_revision=0
    )
    # Use the known-good serialized payload so the only mutation is the extra key.
    valid = json.loads((tmp_path / "source.json").read_text(encoding="utf-8"))
    for where in ("document", "profile"):
        mutated = json.loads(json.dumps(valid))
        if where == "document":
            mutated["secret"] = "do-not-accept"
        else:
            mutated["profiles"][0]["raw_arguments"] = "--api-key secret"
        path = tmp_path / f"{where}.json"
        path.write_text(json.dumps(mutated), encoding="utf-8")
        path.chmod(0o600)
        with pytest.raises(VllmProfileCorrupt):
            VllmProfileRepository(path).load()


@pytest.mark.parametrize(
    "name",
    ["x" * 121, "unsafe\nname", "unsafe\u200ename", "unsafe\u2028name"],
)
def test_profile_names_are_bounded_and_reject_control_or_format_characters(
    tmp_path: Path, name: str
):
    with pytest.raises(VllmProfileValidationError):
        VllmProfileRepository(tmp_path / "profiles.json").save(
            profile_named(name), expected_revision=0
        )


@pytest.mark.parametrize(
    "python_environment",
    ["python --api-key secret", "relative/path/python", "", "python\t-m vllm"],
)
def test_profile_rejects_command_like_python_environment(
    tmp_path: Path, python_environment: str
):
    with pytest.raises(VllmProfileValidationError):
        VllmProfileRepository(tmp_path / "profiles.json").save(
            profile_named("Unsafe", python_environment=python_environment),
            expected_revision=0,
        )


@pytest.mark.parametrize(
    ("model_source", "model_value"),
    [
        (VllmModelSource.HUGGING_FACE, "not-a-repository"),
        (VllmModelSource.HUGGING_FACE, "--api-key PROFILE_SECRET_CANARY"),
        (
            VllmModelSource.HUGGING_FACE,
            "https://user:PROFILE_SECRET_CANARY@example.invalid/model",
        ),
        (VllmModelSource.LOCAL_DIRECTORY, "relative/model"),
        (VllmModelSource.LOCAL_DIRECTORY, "--api-key PROFILE_SECRET_CANARY"),
        (
            VllmModelSource.LOCAL_DIRECTORY,
            "https://user:PROFILE_SECRET_CANARY@example.invalid/model",
        ),
        (
            VllmModelSource.LOCAL_DIRECTORY,
            "/tmp/--api-key PROFILE_SECRET_CANARY",
        ),
        (
            VllmModelSource.LOCAL_DIRECTORY,
            "/https://user:PROFILE_SECRET_CANARY@example.invalid/model",
        ),
        (VllmModelSource.LOCAL_DIRECTORY, "/safe/../unsafe/model"),
    ],
)
def test_profile_rejects_invalid_source_values_before_any_write(
    tmp_path: Path,
    model_source: VllmModelSource,
    model_value: str,
):
    path = tmp_path / "profiles.json"

    with pytest.raises(VllmProfileValidationError) as caught:
        VllmProfileRepository(path).save(
            profile_named(
                "Invalid source",
                model_source=model_source,
                model_value=model_value,
            ),
            expected_revision=0,
        )

    assert not path.exists()
    assert "PROFILE_SECRET_CANARY" not in str(caught.value)


def test_profile_accepts_nonexistent_safe_local_directory_for_repair(tmp_path: Path):
    selected = tmp_path / "models" / "not-downloaded-yet"

    saved = VllmProfileRepository(tmp_path / "profiles.json").save(
        profile_named(
            "Repairable local",
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value=str(selected),
        ),
        expected_revision=0,
    )

    assert saved.profile.model_value == str(selected)
    assert not selected.exists()


def test_decode_revalidates_model_source_without_disclosing_rejected_value(
    tmp_path: Path,
):
    path = tmp_path / "profiles.json"
    profile_id = str(uuid4())
    original = json.dumps(
        {
            "version": 1,
            "revision": 1,
            "selected_profile_id": profile_id,
            "profiles": [
                {
                    "profile_id": profile_id,
                    "name": "Unsafe",
                    "python_environment": "python",
                    "model_source": "hugging_face",
                    "model_value": "--api-key PROFILE_SECRET_CANARY",
                    "bind_address": "127.0.0.1",
                    "port": 8000,
                    "dtype": "auto",
                    "tensor_parallel_size": None,
                    "maximum_model_length": None,
                    "gpu_memory_utilization": None,
                    "trust_remote_code": False,
                }
            ],
        }
    ).encode()
    path.write_bytes(original)
    path.chmod(0o600)

    with pytest.raises(VllmProfileCorrupt) as caught:
        VllmProfileRepository(path).load()

    assert path.read_bytes() == original
    assert "PROFILE_SECRET_CANARY" not in str(caught.value)


def test_commit_revalidates_tampered_profile_before_atomic_writer(tmp_path: Path):
    path = tmp_path / "profiles.json"
    repo = VllmProfileRepository(path)
    saved = repo.save(profile_named("Existing"), expected_revision=0)
    original = path.read_bytes()
    object.__setattr__(saved.profile, "model_value", "--api-key PROFILE_SECRET_CANARY")

    with pytest.raises(VllmProfileValidationError) as caught:
        repo.save(saved.profile, expected_revision=saved.document.revision)

    assert path.read_bytes() == original
    assert "PROFILE_SECRET_CANARY" not in str(caught.value)


def test_names_collide_under_unicode_casefold_normalization_and_canonical_whitespace(
    tmp_path: Path,
):
    repo = VllmProfileRepository(tmp_path / "profiles.json")
    first = repo.save(profile_named("  Straße   GPU  "), expected_revision=0)
    assert first.profile.name == "Straße GPU"

    with pytest.raises(VllmProfileValidationError, match="unique"):
        repo.save(profile_named("STRASSE\u00a0GPU"), expected_revision=1)


def test_duplicate_names_use_deterministic_bounded_suffixes(tmp_path: Path):
    repo = VllmProfileRepository(tmp_path / "profiles.json")
    initial = repo.save(profile_named("GPU"), expected_revision=0)
    first = repo.duplicate(initial.profile.profile_id, expected_revision=1)
    second = repo.duplicate(initial.profile.profile_id, expected_revision=2)

    assert first.profile.name == "GPU copy"
    assert second.profile.name == "GPU copy 2"
    assert len(second.profile.name) <= 120


def test_repository_caps_profiles_at_32(tmp_path: Path):
    repo = VllmProfileRepository(tmp_path / "profiles.json")
    receipt = repo.save(profile_named("Profile 0"), expected_revision=0)
    for index in range(1, MAX_VLLM_PROFILES):
        receipt = repo.save(
            profile_named(f"Profile {index}"),
            expected_revision=receipt.document.revision,
        )

    with pytest.raises(VllmProfileValidationError, match="32"):
        repo.save(
            profile_named("Profile overflow"),
            expected_revision=receipt.document.revision,
        )


def test_compare_and_swap_rejects_stale_writer_without_changing_bytes(tmp_path: Path):
    path = tmp_path / "profiles.json"
    first_repo = VllmProfileRepository(path)
    stale_repo = VllmProfileRepository(path)
    stale = stale_repo.load()
    winner = first_repo.save(profile_named("Winner"), expected_revision=0)
    old_bytes = path.read_bytes()

    with pytest.raises(VllmProfileConflict):
        stale_repo.save(profile_named("Loser"), expected_revision=stale.revision)

    assert path.read_bytes() == old_bytes
    assert first_repo.load() == winner.document


def test_compare_and_swap_serializes_separate_process_writers(tmp_path: Path):
    path = tmp_path / "profiles.json"
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.touch(mode=0o600)
    script = """
import sys
from tldw_chatbook.UI.LLM_Management.vllm_profiles import (
    VllmProfileRepository, profile_from_draft
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmLaunchDraft, VllmMode, VllmModelSource
)
draft = VllmLaunchDraft(
    mode=VllmMode.LOCAL,
    python_environment='python',
    model_source=VllmModelSource.HUGGING_FACE,
    model_value='org/model',
)
VllmProfileRepository(sys.argv[1]).save(
    profile_from_draft('Separate process', draft), expected_revision=0
)
print('saved', flush=True)
"""
    with lock_path.open("a+b") as lock_stream:
        portalocker.lock(lock_stream, portalocker.LockFlags.EXCLUSIVE)
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(path)],
            cwd=Path(__file__).parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.5)
        finally:
            portalocker.unlock(lock_stream)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, stderr
    assert stdout.strip() == "saved"
    with pytest.raises(VllmProfileConflict):
        VllmProfileRepository(path).save(profile_named("Stale"), expected_revision=0)
    assert VllmProfileRepository(path).load().revision == 1


def test_same_revision_two_process_race_has_one_winner_and_one_conflict(
    tmp_path: Path,
):
    path = tmp_path / "profiles.json"
    start = tmp_path / "start"
    script = """
import json
import sys
import time
from pathlib import Path
from tldw_chatbook.UI.LLM_Management.vllm_profiles import (
    VllmProfileConflict, VllmProfileRepository, profile_from_draft
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmLaunchDraft, VllmMode, VllmModelSource
)
path, start, ready, name = map(Path, sys.argv[1:])
draft = VllmLaunchDraft(
    mode=VllmMode.LOCAL,
    python_environment='python',
    model_source=VllmModelSource.HUGGING_FACE,
    model_value='org/model',
)
ready.touch()
while not start.exists():
    time.sleep(0.005)
try:
    VllmProfileRepository(path).save(
        profile_from_draft(name.name, draft), expected_revision=0
    )
except VllmProfileConflict:
    print(json.dumps({'result': 'conflict'}), flush=True)
else:
    print(json.dumps({'result': 'success'}), flush=True)
"""
    processes: list[subprocess.Popen[str]] = []
    for index in range(2):
        ready = tmp_path / f"ready-{index}"
        name = tmp_path / f"Writer {index}"
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(path),
                    str(start),
                    str(ready),
                    str(name),
                ],
                cwd=Path(__file__).parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    deadline = time.monotonic() + 10
    while not all((tmp_path / f"ready-{index}").exists() for index in range(2)):
        if time.monotonic() >= deadline:
            pytest.fail("profile race workers did not reach the barrier")
        time.sleep(0.01)
    start.touch()

    results: list[str] = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout)["result"])

    assert sorted(results) == ["conflict", "success"]
    document = VllmProfileRepository(path).load()
    assert document.revision == 1
    assert len(document.profiles) == 1


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires no-follow opens")
def test_document_symlink_fails_closed_without_reading_or_mutating_target(
    tmp_path: Path,
):
    target = tmp_path / "document-target"
    profile_id = str(uuid4())
    original = json.dumps(
        {
            "version": 1,
            "revision": 7,
            "selected_profile_id": profile_id,
            "profiles": [
                {
                    "profile_id": profile_id,
                    "name": "Symlink target",
                    "python_environment": "python",
                    "model_source": "hugging_face",
                    "model_value": "org/model",
                    "bind_address": "127.0.0.1",
                    "port": 8000,
                    "dtype": "auto",
                    "tensor_parallel_size": None,
                    "maximum_model_length": None,
                    "gpu_memory_utilization": None,
                    "trust_remote_code": False,
                }
            ],
        }
    ).encode()
    target.write_bytes(original)
    target.chmod(0o640)
    original_mode = target.stat().st_mode & 0o777
    path = tmp_path / "profiles.json"
    path.symlink_to(target)

    with pytest.raises(VllmProfileCorrupt):
        VllmProfileRepository(path).load()

    assert target.read_bytes() == original
    assert target.stat().st_mode & 0o777 == original_mode


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires no-follow opens")
def test_lock_symlink_fails_closed_without_mutating_target(tmp_path: Path):
    path = tmp_path / "profiles.json"
    lock_target = tmp_path / "lock-target"
    original = b"LOCK_TARGET_SECRET"
    lock_target.write_bytes(original)
    lock_target.chmod(0o640)
    original_mode = lock_target.stat().st_mode & 0o777
    path.with_name(f"{path.name}.lock").symlink_to(lock_target)

    with pytest.raises(VllmProfileCorrupt):
        VllmProfileRepository(path).save(
            profile_named("No lock follow"), expected_revision=0
        )

    assert not path.exists()
    assert lock_target.read_bytes() == original
    assert lock_target.stat().st_mode & 0o777 == original_mode


def test_document_with_broad_permissions_fails_closed_without_chmod(tmp_path: Path):
    source = tmp_path / "source.json"
    VllmProfileRepository(source).save(profile_named("Source"), expected_revision=0)
    original = source.read_bytes()
    path = tmp_path / "profiles.json"
    path.write_bytes(original)
    path.chmod(0o640)

    with pytest.raises(VllmProfileCorrupt):
        VllmProfileRepository(path).load()

    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == 0o640


def test_unavailable_ownership_api_fails_closed_without_mutating_document(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "profiles.json"
    repo = VllmProfileRepository(path)
    repo.save(profile_named("Owned"), expected_revision=0)
    original = path.read_bytes()
    original_mode = path.stat().st_mode & 0o777
    monkeypatch.delattr(profile_storage.os, "geteuid")

    with pytest.raises(VllmProfileCorrupt):
        repo.load()

    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == original_mode


def test_lock_path_replacement_after_acquisition_fails_before_cas_write(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "profiles.json"
    repo = VllmProfileRepository(path)
    saved = repo.save(profile_named("Existing"), expected_revision=0)
    original_document = path.read_bytes()
    lock_path = path.with_name(f"{path.name}.lock")
    held_target = tmp_path / "held-lock-target"
    replacement_source = tmp_path / "replacement-lock-source"
    held_bytes = b"HELD_LOCK_TARGET"
    replacement_bytes = b"REPLACEMENT_LOCK_TARGET"
    lock_path.write_bytes(held_bytes)
    lock_path.chmod(0o600)
    replacement_source.write_bytes(replacement_bytes)
    replacement_source.chmod(0o600)
    real_lock = portalocker.lock

    def replace_named_lock_after_acquisition(stream, flags):
        result = real_lock(stream, flags)
        lock_path.rename(held_target)
        replacement_source.rename(lock_path)
        return result

    monkeypatch.setattr(portalocker, "lock", replace_named_lock_after_acquisition)

    with pytest.raises(VllmProfileCorrupt):
        repo.save(
            replace(saved.profile, port=8001),
            expected_revision=saved.document.revision,
        )

    assert path.read_bytes() == original_document
    assert held_target.read_bytes() == held_bytes
    assert held_target.stat().st_mode & 0o777 == 0o600
    assert lock_path.read_bytes() == replacement_bytes
    assert lock_path.stat().st_mode & 0o777 == 0o600
    with held_target.open("a+b") as held_stream:
        real_lock(
            held_stream,
            portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
        )
        portalocker.unlock(held_stream)


def test_lock_path_replacement_after_cas_validation_fails_before_atomic_write(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "profiles.json"
    repo = VllmProfileRepository(path)
    saved = repo.save(profile_named("Existing"), expected_revision=0)
    original_document = path.read_bytes()
    lock_path = path.with_name(f"{path.name}.lock")
    held_target = tmp_path / "held-lock-target"
    replacement_source = tmp_path / "replacement-lock-source"
    held_bytes = b"HELD_LOCK_TARGET"
    replacement_bytes = b"REPLACEMENT_LOCK_TARGET"
    lock_path.write_bytes(held_bytes)
    lock_path.chmod(0o600)
    replacement_source.write_bytes(replacement_bytes)
    replacement_source.chmod(0o600)
    real_reject_symlink = profile_storage._reject_symlink_leaf
    document_checks = 0

    def replace_lock_at_final_document_check(checked_path: Path) -> None:
        nonlocal document_checks
        real_reject_symlink(checked_path)
        if checked_path != path:
            return
        document_checks += 1
        if document_checks == 2:
            lock_path.rename(held_target)
            replacement_source.rename(lock_path)

    monkeypatch.setattr(
        profile_storage,
        "_reject_symlink_leaf",
        replace_lock_at_final_document_check,
    )

    with pytest.raises(VllmProfileCorrupt):
        repo.save(
            replace(saved.profile, port=8001),
            expected_revision=saved.document.revision,
        )

    assert path.read_bytes() == original_document
    assert held_target.read_bytes() == held_bytes
    assert lock_path.read_bytes() == replacement_bytes


def test_future_version_is_preserved_byte_for_byte(tmp_path: Path):
    path = tmp_path / "profiles.json"
    original = b'{"version":2,"opaque":"keep"}\n'
    path.write_bytes(original)
    path.chmod(0o600)

    with pytest.raises(VllmProfileFutureVersion):
        VllmProfileRepository(path).save(
            profile_named("No overwrite"), expected_revision=0
        )

    assert path.read_bytes() == original


def test_corrupt_document_fails_closed_without_overwrite(tmp_path: Path):
    path = tmp_path / "profiles.json"
    original = b'{"version":1,"profiles":'
    path.write_bytes(original)
    path.chmod(0o600)

    with pytest.raises(VllmProfileCorrupt):
        VllmProfileRepository(path).save(
            profile_named("No overwrite"), expected_revision=0
        )

    assert path.read_bytes() == original


def test_atomic_write_failure_preserves_old_bytes(monkeypatch, tmp_path: Path):
    path = tmp_path / "profiles.json"
    repo = VllmProfileRepository(path)
    saved = repo.save(profile_named("Existing"), expected_revision=0)
    original = path.read_bytes()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "tldw_chatbook.UI.LLM_Management.vllm_profiles.atomic_write_json",
        fail_write,
    )
    with pytest.raises(OSError):
        repo.save(
            replace(saved.profile, bind_address="::1"),
            expected_revision=saved.document.revision,
        )

    assert path.read_bytes() == original


def test_selected_profile_restores_and_delete_last_recreates_default(tmp_path: Path):
    repo = VllmProfileRepository(tmp_path / "profiles.json")
    first = repo.save(profile_named("First"), expected_revision=0)
    second = repo.save(profile_named("Second"), expected_revision=1)
    renamed = repo.rename(first.profile.profile_id, "Renamed", expected_revision=2)
    selected = repo.select(
        first.profile.profile_id, expected_revision=renamed.document.revision
    )

    assert VllmProfileRepository(repo.path).load().selected_profile_id == (
        first.profile.profile_id
    )
    assert selected.profile.name == "Renamed"
    after_second_delete = repo.delete(
        second.profile.profile_id, expected_revision=selected.document.revision
    )
    after_last_delete = repo.delete(
        first.profile.profile_id,
        expected_revision=after_second_delete.document.revision,
    )

    assert len(after_last_delete.document.profiles) == 1
    assert after_last_delete.profile.name == DEFAULT_PROFILE_NAME
    assert after_last_delete.document.selected_profile_id == (
        after_last_delete.profile.profile_id
    )
