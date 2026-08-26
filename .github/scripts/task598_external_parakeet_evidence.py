#!/usr/bin/env python3
"""Bounded, path-private native evidence probe for TASK-598."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_IDS = (
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
)
EXPECTED_REFERENCES = {
    MODEL_IDS[0]: {
        "artifact_id": "parakeet-v2",
        "revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
        "variant": "int8",
    },
    MODEL_IDS[1]: {
        "artifact_id": "parakeet-v3",
        "revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
        "variant": "int8",
    },
}
VAD_REFERENCE = {
    "artifact_id": "silero-vad",
    "revision": "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    "variant": "f32",
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_BOUNDED_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]{0,127}\Z")
_TRANSCRIPTION_FAILURE_CODES = frozenset(
    {
        "model_not_installed",
        "artifact_corrupt",
        "artifact_incompatible",
        "provider_unavailable",
        "provider_removed",
        "unsupported_language",
        "unsupported_capability",
        "insufficient_disk_space",
        "insufficient_memory",
        "inference_failed",
        "engine_crashed",
        "cancelled",
    }
)
_EXTERNAL_VERIFICATION_FAILURE_CODES = frozenset(
    {
        "unsupported_descriptor",
        "missing_file",
        "irregular_file",
        "changed_file",
        "corrupt_file",
        "cancelled",
    }
)
_EXTERNAL_CHANGED_DIAGNOSTIC_CODES = frozenset(
    {
        "ancestor_identity",
        "file_path_identity",
        "open_file_identity",
        "post_read_file_identity",
        "file_read",
        "snapshot_identity",
    }
)
_WINDOWS_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]")
_EMBEDDED_POSIX_ABSOLUTE = re.compile(r"(?:^|[\s=:'\"])/(?!/)[^\s'\"<>]+")


class _SupervisorTermination(BaseException):
    """Internal signal used to unwind the worker through ordered cleanup."""


def _raise_supervisor_termination(_signum: int, _frame: object) -> None:
    raise _SupervisorTermination


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return None


def current_run_identity() -> dict[str, str]:
    """Return the exact checked-out commit and workflow run identity."""

    tested_commit = os.environ.get("TASK598_TESTED_COMMIT", "").strip().lower()
    if not _SHA1.fullmatch(tested_commit):
        try:
            tested_commit = (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .lower()
            )
        except (OSError, subprocess.SubprocessError):
            tested_commit = os.environ.get("GITHUB_SHA", "").strip().lower()
    if not _SHA1.fullmatch(tested_commit):
        tested_commit = "0" * 40
    return {
        "tested_commit": tested_commit,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
    }


def _runtime_providers() -> list[str]:
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except (AttributeError, ImportError, OSError, RuntimeError):
        return []


def _host_result(*, probe_runtime: bool = True) -> dict[str, object]:
    packages = {
        name: version
        for name in ("onnx-asr", "onnxruntime")
        if (version := _package_version(name)) is not None
    }
    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "available_providers": _runtime_providers() if probe_runtime else [],
    }


def failure_result(
    run_identity: Mapping[str, str],
    *,
    failure_code: str,
    failure_stage: str,
    failure_type: str | None = None,
) -> dict[str, object]:
    """Build a stable failure envelope without exception text or paths."""

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "run": dict(run_identity),
        "host": _host_result(probe_runtime=False),
        "models": {},
        "final_store": {},
    }
    if failure_type:
        result["failure_type"] = failure_type
    return result


def _normalized_failure_code(error: BaseException) -> str:
    """Return only a stable transcription code, never exception text."""

    external_module = sys.modules.get("tldw_chatbook.STT.parakeet_external")
    external_error_type = getattr(
        external_module, "ExternalParakeetVerificationError", None
    )
    if external_error_type is not None and type(error) is external_error_type:
        code_value = error.code.value
        if (
            type(code_value) is str
            and code_value in _EXTERNAL_VERIFICATION_FAILURE_CODES
        ):
            diagnostic_code = error.diagnostic_code
            if (
                code_value == "changed_file"
                and type(diagnostic_code) is str
                and diagnostic_code in _EXTERNAL_CHANGED_DIAGNOSTIC_CODES
            ):
                return f"external_changed_{diagnostic_code}"
            return f"external_{code_value}"
    if (
        isinstance(error, RuntimeError)
        and len(error.args) == 1
        and type(error.args[0]) is str
        and error.args[0] in _TRANSCRIPTION_FAILURE_CODES
    ):
        return error.args[0]
    return "probe_failed"


def _write_result(output: Path, result: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether two canonical paths preserve the expected containment."""

    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _walk_values(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_walk_values(key))
            result.extend(_walk_values(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_walk_values(item))
        return result
    return ()


def _reject_local_paths(
    result: Mapping[str, object],
    forbidden_roots: Sequence[Path],
) -> None:
    roots = tuple(str(root.absolute()) for root in forbidden_roots)
    for value in _walk_values(result):
        if any(root and root in value for root in roots):
            raise ValueError("evidence contains a local path")
        if (
            value.startswith(("/", "~/"))
            or _WINDOWS_ABSOLUTE.search(value)
            or _EMBEDDED_POSIX_ABSOLUTE.search(value)
        ):
            raise ValueError("evidence contains a local path")
        if "task598-evidence-" in value:
            raise ValueError("evidence contains a temporary-directory name")


def _reject_sensitive_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                normalized = key.lower()
                if normalized in {"user", "username", "user_name"}:
                    raise ValueError("evidence contains a username field")
                if any(
                    marker in normalized
                    for marker in (
                        "auth",
                        "cookie",
                        "credential",
                        "password",
                        "secret",
                        "token",
                        "api_key",
                    )
                ):
                    raise ValueError("evidence contains a credential field")
            _reject_sensitive_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_keys(item)


def _require_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _require_fields(
    value: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(
            f"{context} contains unexpected fields: {', '.join(unexpected)}"
        )


def validate_result(
    result: Mapping[str, object],
    *,
    require_success: bool = True,
    forbidden_roots: Sequence[Path] = (),
) -> None:
    """Validate a complete pass or a stable pre-probe failure envelope."""

    if not isinstance(result, Mapping):
        raise ValueError("evidence must be an object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if result.get("status") not in {"passed", "failed"}:
        raise ValueError("status must be passed or failed")
    _reject_local_paths(result, forbidden_roots)
    _reject_sensitive_keys(result)
    status = result["status"]
    allowed_result_fields = {
        "schema_version",
        "status",
        "failure_code",
        "failure_stage",
        "run",
        "host",
        "models",
        "final_store",
    }
    if status == "failed" and "failure_type" in result:
        allowed_result_fields.add("failure_type")
    _require_fields(result, allowed_result_fields, "result")
    run = _require_mapping(result, "run")
    _require_fields(
        run,
        {"tested_commit", "workflow_run_id", "workflow_run_attempt"},
        "run",
    )
    tested_commit = run.get("tested_commit")
    if not isinstance(tested_commit, str) or not _SHA1.fullmatch(tested_commit):
        raise ValueError("run.tested_commit must be an exact commit")
    workflow_run_id = run.get("workflow_run_id")
    if not isinstance(workflow_run_id, str) or not (
        workflow_run_id == "local" or workflow_run_id.isdecimal()
    ):
        raise ValueError("run.workflow_run_id must be numeric or local")
    workflow_run_attempt = run.get("workflow_run_attempt")
    if (
        not isinstance(workflow_run_attempt, str)
        or not workflow_run_attempt.isdecimal()
    ):
        raise ValueError("run.workflow_run_attempt must be numeric")

    host = _require_mapping(result, "host")
    _require_fields(
        host,
        {"system", "architecture", "python", "packages", "available_providers"},
        "host",
    )
    for key in ("system", "architecture", "python"):
        value = host.get(key)
        if not isinstance(value, str) or not _BOUNDED_TOKEN.fullmatch(value):
            raise ValueError(f"host.{key} must be a bounded identifier")
    packages = _require_mapping(host, "packages")

    if status == "failed":
        for key in ("failure_code", "failure_stage"):
            value = result.get(key)
            if not isinstance(value, str) or not _BOUNDED_TOKEN.fullmatch(value):
                raise ValueError(f"{key} must be a bounded identifier")
        failure_type = result.get("failure_type")
        if failure_type is not None and (
            not isinstance(failure_type, str)
            or not _BOUNDED_TOKEN.fullmatch(failure_type)
        ):
            raise ValueError("failure_type must be a bounded identifier")
        unexpected_packages = set(packages) - {"onnx-asr", "onnxruntime"}
        if unexpected_packages:
            raise ValueError("host.packages contains unexpected fields")
        for name, value in packages.items():
            if not isinstance(value, str) or not _BOUNDED_TOKEN.fullmatch(value):
                raise ValueError(f"host package {name} must be bounded")
        if host.get("available_providers") != []:
            raise ValueError("failed evidence must not report runtime providers")
        if dict(_require_mapping(result, "models")):
            raise ValueError("failed evidence models must be empty")
        if dict(_require_mapping(result, "final_store")):
            raise ValueError("failed evidence final_store must be empty")
        if require_success:
            raise ValueError("TASK-598 evidence did not pass")
        return
    if tested_commit == "0" * 40:
        raise ValueError("passed evidence requires a real tested commit")

    _require_fields(packages, {"onnx-asr", "onnxruntime"}, "host.packages")
    for name in ("onnx-asr", "onnxruntime"):
        if not isinstance(packages.get(name), str) or not packages[name]:
            raise ValueError(f"host package {name} must be present")
    providers = host.get("available_providers")
    if not isinstance(providers, list) or "CPUExecutionProvider" not in providers:
        raise ValueError("CPUExecutionProvider must be available")

    models = _require_mapping(result, "models")
    if set(models) != set(MODEL_IDS):
        raise ValueError("models must contain exact v2 and v3 INT8 evidence")
    required_true = (
        "descriptor_verified",
        "managed_copy_deleted",
        "external_unchanged",
        "cache_unchanged",
        "store_unchanged",
        "source_preference_unchanged",
        "shutdown_completed",
    )
    for model_id in MODEL_IDS:
        model = _require_mapping(models, model_id)
        _require_fields(
            model,
            {
                "descriptor_verified",
                "managed_copy_deleted",
                "external_unchanged",
                "cache_unchanged",
                "store_unchanged",
                "source_preference_unchanged",
                "execution_provider",
                "artifact_root",
                "artifact_dependencies",
                "shutdown_completed",
                "reference",
                "timings",
            },
            model_id,
        )
        for key in required_true:
            if model.get(key) is not True:
                raise ValueError(f"{model_id}.{key} must be true")
        if model.get("execution_provider") != "CPUExecutionProvider":
            raise ValueError(f"{model_id} must use CPUExecutionProvider")
        if model.get("artifact_root") is not None:
            raise ValueError(f"{model_id}.artifact_root must be null")
        if model.get("artifact_dependencies") != [VAD_REFERENCE]:
            raise ValueError(f"{model_id} must name the exact VAD dependency")
        if model.get("reference") != EXPECTED_REFERENCES[model_id]:
            raise ValueError(f"{model_id}.reference must be exact")
        timings = _require_mapping(model, "timings")
        _require_fields(
            timings,
            {"inference_seconds", "model_total_seconds"},
            f"{model_id}.timings",
        )
        for key in ("inference_seconds", "model_total_seconds"):
            value = timings.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"{model_id}.timings.{key} must be nonnegative")

    final_store = _require_mapping(result, "final_store")
    _require_fields(
        final_store,
        {
            "vad_only",
            "no_parakeet_roots",
            "no_readiness",
            "no_active_selector",
        },
        "final_store",
    )
    for key in (
        "vad_only",
        "no_parakeet_roots",
        "no_readiness",
        "no_active_selector",
    ):
        if final_store.get(key) is not True:
            raise ValueError(f"final_store.{key} must be true")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        process.kill()


def _posix_group_exists(group_id: int) -> bool:
    if os.name != "posix" or group_id <= 1 or group_id == os.getpgrp():
        return False
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_group(group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while _posix_group_exists(group_id) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not _posix_group_exists(group_id)


def _native_process_group(control: Path | None) -> int | None:
    if os.name != "posix" or control is None:
        return None
    try:
        value = json.loads(control.read_text(encoding="utf-8")).get(
            "native_process_group_id"
        )
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if type(value) is not int or not _posix_group_exists(value):
        return None
    return value


def _terminate_native_group(group_id: int, timeout_seconds: float) -> None:
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_for_posix_group(group_id, timeout_seconds):
        return
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_for_posix_group(group_id, timeout_seconds)


def supervise(
    command: Sequence[str],
    *,
    output: Path,
    timeout_seconds: float,
    run_identity: Mapping[str, str],
    forbidden_roots: Sequence[Path],
    env: Mapping[str, str] | None = None,
    control: Path | None = None,
    cleanup_root: Path | None = None,
    cleanup_parent: Path | None = None,
    cleanup_seconds: float = 10.0,
) -> dict[str, object]:
    """Run one bounded child and guarantee a path-private JSON result."""

    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=None if env is None else dict(env),
            start_new_session=os.name == "posix",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
    except OSError:
        result = failure_result(
            run_identity,
            failure_code="worker_start_failed",
            failure_stage="supervisor",
        )
        _write_result(output, result)
        return result
    try:
        process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=max(0.0, cleanup_seconds))
        except subprocess.TimeoutExpired:
            pass
        native_group = _native_process_group(control)
        if native_group is not None:
            _terminate_native_group(native_group, cleanup_seconds)
        _terminate(process)
        try:
            process.wait(timeout=max(0.0, cleanup_seconds))
        except subprocess.TimeoutExpired:
            pass
        process_dead = process.poll() is not None
        native_dead = native_group is None or not _posix_group_exists(native_group)
        cleanup_ok = True
        if cleanup_root is not None or cleanup_parent is not None:
            cleanup_ok = (
                cleanup_root is not None
                and cleanup_parent is not None
                and process_dead
                and native_dead
                and _remove_owned_tree(cleanup_root, cleanup_parent)
            )
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        result = failure_result(
            run_identity,
            failure_code="timeout" if cleanup_ok else "cleanup_failed",
            failure_stage="supervisor",
        )
        _write_result(output, result)
        return result

    result: dict[str, object]
    try:
        loaded = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("worker evidence is not an object")
        result = loaded
        validate_result(
            result,
            require_success=process.returncode == 0,
            forbidden_roots=forbidden_roots,
        )
        if result.get("run") != dict(run_identity):
            raise ValueError("worker run identity changed")
        if process.returncode != 0 and result.get("status") == "passed":
            raise ValueError("failed worker reported success")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        result = failure_result(
            run_identity,
            failure_code="worker_failed",
            failure_stage="worker",
        )
        _write_result(output, result)
    return result


def _remove_owned_tree(root: Path, parent: Path) -> bool:
    """Remove only a probe-owned subtree after its processes are proven dead."""

    canonical_root = root.resolve(strict=False)
    canonical_parent = parent.resolve(strict=False)
    if canonical_root == canonical_parent or not _path_is_within(
        canonical_root, canonical_parent
    ):
        return False
    try:
        shutil.rmtree(canonical_root)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return not canonical_root.exists()


def _tree_token(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        stat_result = path.stat()
        digest.update(relative)
        digest.update(str(stat_result.st_size).encode("ascii"))
        digest.update(str(stat_result.st_mtime_ns).encode("ascii"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _managed_state_token(service: object) -> str:
    root = service.artifacts_path.parent
    digest = hashlib.sha256()
    for name in ("artifacts", "active", "ready", "staging"):
        digest.update(name.encode("ascii"))
        digest.update(_tree_token(root / name).encode("ascii"))
    return digest.hexdigest()


def _cache_token(roots: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for index, root in enumerate(roots):
        digest.update(str(index).encode("ascii"))
        digest.update(_tree_token(root).encode("ascii"))
    return digest.hexdigest()


def _descriptor_token(directory: Path, descriptor: object) -> str:
    digest = hashlib.sha256()
    for item in descriptor.files:
        path = directory / item.path
        stat_result = path.stat()
        digest.update(item.path.encode("utf-8"))
        digest.update(str(stat_result.st_size).encode("ascii"))
        digest.update(str(stat_result.st_mtime_ns).encode("ascii"))
        digest.update(str(stat_result.st_mode).encode("ascii"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _pcm_fixture() -> bytes:
    """Return deterministic four-second signed-16-bit mono PCM."""

    return bytes(16_000 * 4 * 2)


def _reference_dict(reference: object) -> dict[str, str]:
    return {
        "artifact_id": reference.artifact_id,
        "revision": reference.revision,
        "variant": reference.variant,
    }


def _transcription_provenance(payload: Mapping[str, object]) -> dict[str, object]:
    provenance = payload.get("transcription_provenance")
    if type(provenance) is not dict:
        raise RuntimeError("native transcription omitted provenance")
    return provenance


def _copy_external_root(source: Path, destination: Path, descriptor: object) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for item in descriptor.files:
        target = destination / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / item.path, target)


def _publish_executor_identity(
    executor: object,
    control: Path,
    stop: threading.Event,
) -> None:
    while not stop.wait(0.01):
        tree = getattr(executor, "_tree", None)
        identity = getattr(tree, "_identity", None)
        if identity is None:
            continue
        _write_result(
            control,
            {
                "native_pid": identity.pid,
                "native_process_group_id": identity.process_group_id,
            },
        )
        return


def _close_runtime_resources(
    source_service: object,
    coordinator: object,
    executor: object,
) -> bool:
    """Close in contract order and prove native containment plus scratch cleanup."""

    tree = getattr(executor, "_tree", None)
    scratch = getattr(executor, "_scratch_path", None)
    clean = tree is not None and isinstance(scratch, Path)
    for resource in (source_service, coordinator, executor):
        try:
            resource.close()
        except Exception:
            clean = False
    try:
        tree_closed = tree is not None and bool(tree.close())
    except Exception:
        tree_closed = False
    scratch_removed = isinstance(scratch, Path) and not scratch.exists()
    return clean and tree_closed and scratch_removed


def _cleanup_model_runtime(
    facade: object | None,
    source_service: object | None,
    coordinator: object | None,
    executor: object | None,
) -> bool:
    """Clean every constructed runtime resource without short-circuiting."""

    facade_clean = True
    if facade is not None:
        try:
            facade.cleanup()
        except Exception:
            facade_clean = False
    if not all(
        resource is not None for resource in (source_service, coordinator, executor)
    ):
        return False
    resources_clean = _close_runtime_resources(source_service, coordinator, executor)
    return facade_clean and resources_clean


def _run_one_model(
    model_id: str,
    *,
    scratch: Path,
    service: object,
    cache_roots: Sequence[Path],
    control: Path,
    report_stage: Callable[[str], None],
) -> dict[str, object]:
    import asyncio

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_descriptor,
        parakeet_reference,
        run_parakeet_preflight,
        run_parakeet_provision,
    )

    report_stage("descriptor")
    started = time.monotonic()
    descriptor = parakeet_descriptor(model_id, "int8")
    reference = parakeet_reference(model_id, "int8")
    report_stage("provision")
    report = asyncio.run(run_parakeet_preflight(model_id, "int8", core=service))
    managed_root = asyncio.run(
        run_parakeet_provision(model_id, "int8", report, core=service)
    )
    report_stage("external_copy")
    external = scratch / f"external-{MODEL_IDS.index(model_id)}"
    _copy_external_root(managed_root, external, descriptor)
    service.delete(reference)
    external_before = _descriptor_token(external, descriptor)
    cache_before = _cache_token(cache_roots)
    store_before = _managed_state_token(service)

    prior_offline = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "ONNX_ASR_OFFLINE")
    }
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ONNX_ASR_OFFLINE": "1",
        }
    )
    source_service = coordinator = executor = facade = None
    monitor: threading.Thread | None = None
    monitor_stop = threading.Event()
    shutdown_completed = False
    try:
        report_stage("runtime_setup")
        from tldw_chatbook.Local_Ingestion.transcription_service import (
            TranscriptionService,
        )
        from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
        from tldw_chatbook.app import TldwCli

        app = TldwCli()
        source_service = app._ensure_parakeet_source_service()
        coordinator = app._ensure_local_stt_dispatch_coordinator()
        executor = app._ensure_local_stt_executor()
        monitor = threading.Thread(
            target=_publish_executor_identity,
            args=(executor, control, monitor_stop),
            name="task598-containment-observer",
            daemon=True,
        )
        monitor.start()
        key = ParakeetSourceKey.from_values(model_id, "int8")
        records_before = dict(source_service.records())
        report_stage("external_verify")
        prepared = source_service.prepare_external(key, external)
        report_stage("managed_copy")
        plan = source_service.plan_managed_copy(prepared.verified)
        copied = source_service.copy_into_managed(
            prepared.verified,
            plan.grant(),
        )
        if copied != reference or service.readiness_path(reference).exists():
            raise RuntimeError("managed copy changed readiness")
        if dict(source_service.records()) != records_before:
            raise RuntimeError("managed copy changed source preference")
        service.delete(reference)

        report_stage("transcription")
        inference_started = time.monotonic()
        facade = TranscriptionService(
            local_stt_dispatcher=coordinator,
            parakeet_source_service=source_service,
        )
        payload = facade.transcribe_buffer(
            _pcm_fixture(),
            16_000,
            channels=1,
            sample_width=2,
            provider="parakeet-onnx",
            model=model_id,
            language="en",
            precision="int8",
            model_dir=str(external),
        )
        report_stage("provenance")
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise RuntimeError("native transcription returned an invalid payload")
        provenance = _transcription_provenance(payload)
        dependencies = provenance.get("artifact_dependencies")
        if dependencies != [VAD_REFERENCE]:
            raise RuntimeError("native transcription reported wrong dependencies")
        if provenance.get("artifact_root") is not None:
            raise RuntimeError("external transcription reported a managed root")
        if provenance.get("effective_device") != "cpu":
            raise RuntimeError("native transcription did not use CPU")
        if dict(source_service.records()) != records_before:
            raise RuntimeError("native transcription changed source preference")
        inference_seconds = time.monotonic() - inference_started
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(1.0)
        shutdown_completed = _cleanup_model_runtime(
            facade,
            source_service,
            coordinator,
            executor,
        )
        for key, value in prior_offline.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    report_stage("post_invariants")
    external_after = _descriptor_token(external, descriptor)
    cache_after = _cache_token(cache_roots)
    store_after = _managed_state_token(service)
    managed_copy_deleted = not any(
        (
            service.artifact_path(reference).exists(),
            service.readiness_path(reference).exists(),
            service.active_path(reference.artifact_id).exists(),
        )
    )
    shutil.rmtree(external)
    return {
        "descriptor_verified": True,
        "managed_copy_deleted": managed_copy_deleted,
        "external_unchanged": external_before == external_after,
        "cache_unchanged": cache_before == cache_after,
        "store_unchanged": store_before == store_after,
        "source_preference_unchanged": True,
        "execution_provider": "CPUExecutionProvider",
        "artifact_root": None,
        "artifact_dependencies": [copy.deepcopy(VAD_REFERENCE)],
        "shutdown_completed": shutdown_completed,
        "reference": _reference_dict(reference),
        "timings": {
            "inference_seconds": inference_seconds,
            "model_total_seconds": time.monotonic() - started,
        },
    }


def _final_store_result(service: object) -> dict[str, bool]:
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
        parakeet_vad_reference,
    )

    inventory = service.list_installed()
    references = {
        item.descriptor.reference
        for item in inventory
        if item.descriptor is not None and item.error is None
    }
    roots = tuple(parakeet_reference(model_id, "int8") for model_id in MODEL_IDS)
    return {
        "vad_only": references == {parakeet_vad_reference()},
        "no_parakeet_roots": all(
            not service.artifact_path(ref).exists() for ref in roots
        ),
        "no_readiness": all(not service.readiness_path(ref).exists() for ref in roots),
        "no_active_selector": all(
            not service.active_path(ref.artifact_id).exists() for ref in roots
        ),
    }


def run_worker(output: Path, scratch: Path, control: Path) -> int:
    """Run both native model cases sequentially in an isolated profile."""

    run_identity = current_run_identity()
    stage = "imports"
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _raise_supervisor_termination)
    try:
        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            parakeet_v2_managed_service,
        )
        from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root

        stage = "store"
        service = parakeet_v2_managed_service()
        managed_root = service.artifacts_path.parent.resolve(strict=False)
        expected_root = managed_model_artifact_root().resolve(strict=False)
        if managed_root != expected_root:
            raise RuntimeError("managed store root mismatch")
        if not _path_is_within(managed_root, scratch) or not _path_is_within(
            managed_root,
            Path(os.environ["XDG_DATA_HOME"]),
        ):
            raise RuntimeError("managed store escaped the scratch profile")
        cache_roots = (
            Path(os.environ["XDG_CACHE_HOME"]),
            Path(os.environ["HF_HOME"]),
        )
        models: dict[str, object] = {}
        for model_id in MODEL_IDS:
            model_index = MODEL_IDS.index(model_id)
            stage = f"model_{model_index}_setup"

            def report_stage(substage: str, *, index: int = model_index) -> None:
                nonlocal stage
                stage = f"model_{index}_{substage}"

            models[model_id] = _run_one_model(
                model_id,
                scratch=scratch,
                service=service,
                cache_roots=cache_roots,
                control=control,
                report_stage=report_stage,
            )
        stage = "final_store"
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "failure_code": None,
            "failure_stage": None,
            "run": run_identity,
            "host": _host_result(),
            "models": models,
            "final_store": _final_store_result(service),
        }
        validate_result(result, forbidden_roots=(scratch, Path.home()))
        _write_result(output, result)
        return 0
    except BaseException as error:
        result = failure_result(
            run_identity,
            failure_code=_normalized_failure_code(error),
            failure_stage=stage,
            failure_type=type(error).__name__,
        )
        _write_result(output, result)
        return 1


def _isolated_environment(scratch: Path) -> dict[str, str]:
    home = scratch / "home"
    data = scratch / "data"
    cache = scratch / "cache"
    native_temp = scratch / "native-temp"
    for path in (home, data, cache, native_temp):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
    config = scratch / "config.toml"
    config.write_text(
        "[general]\n"
        'users_name = "task598"\n'
        "[paths]\n"
        f"data_dir = {json.dumps(str(data))}\n"
        "[transcription]\n"
        'default_provider = "parakeet-onnx"\n'
        'default_model = "nemo-parakeet-tdt-0.6b-v2"\n'
        'default_precision = "int8"\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(data),
            "XDG_CACHE_HOME": str(cache),
            "HF_HOME": str(cache / "huggingface"),
            "TMPDIR": str(native_temp),
            "TMP": str(native_temp),
            "TEMP": str(native_temp),
            "TLDW_CONFIG_PATH": str(config),
            "TASK598_TESTED_COMMIT": current_run_identity()["tested_commit"],
        }
    )
    return env


def run_parent(output: Path, timeout_seconds: float) -> int:
    run_identity = current_run_identity()
    with tempfile.TemporaryDirectory(prefix="task598-evidence-") as temporary:
        scratch = Path(temporary).resolve(strict=True)
        control = scratch / "control.json"
        env = _isolated_environment(scratch)
        command = [
            sys.executable,
            str(Path(__file__).absolute()),
            "--worker",
            "--output",
            str(output),
            "--scratch",
            str(scratch),
            "--control",
            str(control),
        ]
        result = supervise(
            command,
            output=output,
            timeout_seconds=timeout_seconds,
            run_identity=run_identity,
            forbidden_roots=(scratch, Path.home()),
            env=env,
            control=control,
            cleanup_root=Path(env["TMPDIR"]),
            cleanup_parent=scratch,
        )
    return 0 if result.get("status") == "passed" else 1


def _load_result(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("evidence must be an object")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=4_800)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--record-failure")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--control", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded evidence probe or validate an existing result.

    Args:
        argv: Optional CLI arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Zero when the requested operation succeeds; otherwise one.
    """

    args = _parser().parse_args(argv)
    if args.validate is not None:
        try:
            result = _load_result(args.validate)
            roots = tuple(
                Path(value)
                for value in (
                    os.environ.get("RUNNER_TEMP"),
                    os.environ.get("HOME"),
                )
                if value
            )
            validate_result(result, forbidden_roots=roots)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return 1
        return 0
    if args.output is None:
        raise SystemExit("--output is required")
    if args.initialize:
        result = failure_result(
            current_run_identity(),
            failure_code="not_run",
            failure_stage="initialize",
        )
        _write_result(args.output, result)
        return 0
    if args.record_failure:
        result = failure_result(
            current_run_identity(),
            failure_code=args.record_failure,
            failure_stage="install",
        )
        _write_result(args.output, result)
        return 0
    if args.worker:
        if args.scratch is None or args.control is None:
            raise SystemExit("--scratch and --control are required with --worker")
        return run_worker(args.output, args.scratch, args.control)
    return run_parent(args.output, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
