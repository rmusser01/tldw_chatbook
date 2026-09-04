"""Pure, fail-closed setup validation for Chatbook-owned vLLM launches."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import re
import select
import shlex
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from tldw_chatbook.Utils.path_validation import validate_path_simple

SERVED_MODEL_NAME = "chatbook-vllm"
_HF_REPOSITORY = re.compile(
    r"^(?=.{3,96}$)[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_WINDOWS_DRIVE_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_MANAGED_OR_SECRET_FLAGS = frozenset(
    {
        "--api-key",
        "--config",
        "--dtype",
        "--gpu-memory-utilization",
        "--hf-token",
        "--host",
        "--max-model-len",
        "--model",
        "--no-trust-remote-code",
        "--port",
        "--served-model-name",
        "--tensor-parallel-size",
        "--trust-remote-code",
    }
)
_MANAGED_SHORT_FLAG_ALIASES = {"-tp": "--tensor-parallel-size"}
_MAX_PROBE_OUTPUT_BYTES = 256
_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE_REAP_TIMEOUT_SECONDS = 0.25
_VERSION_OUTPUT = re.compile(r"^[A-Za-z][A-Za-z0-9 ._+\-]{0,120}$")
_DTYPE_VALUES = frozenset({"", "auto", "half", "float16", "bfloat16", "float32"})


class VllmMode(StrEnum):
    LOCAL = "local"
    EXISTING = "existing"


class VllmModelSource(StrEnum):
    HUGGING_FACE = "hugging_face"
    LOCAL_DIRECTORY = "local_directory"


class VllmReadinessState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    CHECKING = "checking"
    READY_TO_START = "ready_to_start"
    LAUNCHING = "launching"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    STOPPING = "stopping"
    NEEDS_ATTENTION = "needs_attention"


@dataclass(frozen=True, slots=True)
class VllmLaunchDraft:
    mode: VllmMode
    python_environment: str
    model_source: VllmModelSource
    model_value: str
    bind_address: str = "127.0.0.1"
    port: int = 8000
    existing_server_url: str = ""
    dtype: str = ""
    tensor_parallel_size: int | None = None
    maximum_model_length: int | None = None
    gpu_memory_utilization: float | None = None
    trust_remote_code: bool = False
    raw_arguments: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class VllmIssue:
    code: str
    field: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class VllmPreflightResult:
    generation: int
    fingerprint: str
    issues: tuple[VllmIssue, ...]
    python_version: str | None = None
    vllm_version: str | None = None
    cli_path: Path | None = field(default=None, repr=False)
    network_exposed: bool = False


@dataclass(frozen=True, slots=True)
class VllmLaunchSnapshot:
    generation: int
    profile_id: str | None
    environment_display: str = field(repr=False)
    model_source_kind: VllmModelSource
    model_source_display: str = field(repr=False)
    bind_address: str
    port: int
    structured_options: tuple[tuple[str, object], ...]
    redacted_argument_summary: str
    raw_arguments_fingerprint: str
    fingerprint: str
    client_api_url: str
    served_model: str
    display_profile_name: str


@dataclass(frozen=True, slots=True)
class VllmConnectionTarget:
    provider_key: Literal["vllm"]
    api_url: str
    model_id: str
    runtime_owner: Literal["chatbook", "external"]
    generation: int
    credential_source: Literal["none", "configured", "environment"]


_CHANGED_FIELD_ORDER: tuple[tuple[str, str], ...] = (
    ("python_environment", "Python environment"),
    ("model_source", "Model source"),
    ("model_value", "Model"),
    ("bind_address", "Bind address"),
    ("port", "Port"),
    ("dtype", "dtype"),
    ("tensor_parallel_size", "Tensor parallel size"),
    ("maximum_model_length", "Maximum model length"),
    ("gpu_memory_utilization", "GPU memory utilization"),
    ("trust_remote_code", "Trust remote code"),
)


def _effective_dtype(value: str) -> str:
    return value or "auto"


def _raw_arguments_fingerprint(raw_arguments: str) -> str:
    return hashlib.sha256(raw_arguments.encode("utf-8")).hexdigest()


def launch_snapshot_from_draft(
    draft: VllmLaunchDraft,
    *,
    generation: int,
    profile_id: str | None = None,
    profile_name: str = "Chatbook-managed vLLM",
) -> VllmLaunchSnapshot:
    """Capture immutable launch truth before reserving its exact process claim."""

    structured_options: tuple[tuple[str, object], ...] = (
        ("dtype", _effective_dtype(draft.dtype)),
        ("tensor_parallel_size", draft.tensor_parallel_size),
        ("maximum_model_length", draft.maximum_model_length),
        ("gpu_memory_utilization", draft.gpu_memory_utilization),
        ("trust_remote_code", draft.trust_remote_code),
    )
    return VllmLaunchSnapshot(
        generation=generation,
        profile_id=profile_id,
        environment_display=draft.python_environment,
        model_source_kind=draft.model_source,
        model_source_display=draft.model_value,
        bind_address=draft.bind_address,
        port=draft.port,
        structured_options=structured_options,
        redacted_argument_summary=(
            "Custom launch arguments" if draft.raw_arguments.strip() else "None"
        ),
        raw_arguments_fingerprint=_raw_arguments_fingerprint(draft.raw_arguments),
        fingerprint=semantic_fingerprint(draft),
        client_api_url=client_api_url(draft.bind_address, draft.port),
        served_model=SERVED_MODEL_NAME,
        display_profile_name=profile_name,
    )


def changed_launch_field_labels(
    snapshot: VllmLaunchSnapshot, draft: VllmLaunchDraft
) -> tuple[str, ...]:
    """Return allowlisted labels for restart changes, never their values."""

    snapshot_values: dict[str, object] = {
        "python_environment": snapshot.environment_display,
        "model_source": snapshot.model_source_kind,
        "model_value": snapshot.model_source_display,
        "bind_address": snapshot.bind_address,
        "port": snapshot.port,
        **dict(snapshot.structured_options),
    }
    labels = [
        label
        for field_name, label in _CHANGED_FIELD_ORDER
        if snapshot_values[field_name]
        != (
            _effective_dtype(draft.dtype)
            if field_name == "dtype"
            else getattr(draft, field_name)
        )
    ]
    if snapshot.raw_arguments_fingerprint != _raw_arguments_fingerprint(
        draft.raw_arguments
    ):
        labels.append("Advanced arguments")
    return tuple(labels)


def semantic_fingerprint(draft: VllmLaunchDraft) -> str:
    """Return a stable identity for every behaviorally relevant launch field."""

    values = asdict(draft)
    values["mode"] = draft.mode.value
    values["model_source"] = draft.model_source.value
    values["dtype"] = _effective_dtype(draft.dtype)
    encoded = repr(tuple(sorted(values.items()))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_raw_arguments(raw_arguments: str) -> tuple[VllmIssue, ...]:
    """Reject malformed shell syntax and managed or credential-bearing flags."""

    if type(raw_arguments) is not str:
        return (VllmIssue("invalid_arguments", "raw_arguments"),)
    if not raw_arguments.strip():
        return ()
    try:
        arguments = shlex.split(raw_arguments)
    except ValueError:
        return (VllmIssue("invalid_arguments", "raw_arguments"),)
    for argument in arguments:
        raw_flag = argument.split("=", 1)[0]
        flag = _MANAGED_SHORT_FLAG_ALIASES.get(raw_flag, raw_flag.replace("_", "-"))
        if flag in _MANAGED_OR_SECRET_FLAGS:
            return (VllmIssue("arguments_conflict", "raw_arguments", flag),)
        if flag.startswith("--") and any(
            protected.startswith(flag) for protected in _MANAGED_OR_SECRET_FLAGS
        ):
            return (VllmIssue("arguments_conflict", "raw_arguments", flag),)
    return ()


def _validate_draft_structure(draft: VllmLaunchDraft) -> tuple[VllmIssue, ...]:
    """Validate every launch field before any value reaches an owning subsystem."""

    issues: list[VllmIssue] = []
    if type(draft.mode) is not VllmMode:
        issues.append(VllmIssue("invalid_mode", "mode"))
    if type(draft.python_environment) is not str:
        issues.append(VllmIssue("invalid_python_environment", "python_environment"))
    if type(draft.model_source) is not VllmModelSource:
        issues.append(VllmIssue("invalid_model_source", "model_source"))
    if type(draft.model_value) is not str:
        issues.append(VllmIssue("invalid_model_value", "model_value"))
    if type(draft.bind_address) is not str:
        issues.append(VllmIssue("invalid_bind_address", "bind_address"))
    if type(draft.port) is not int or not 1 <= draft.port <= 65535:
        issues.append(VllmIssue("invalid_port", "port"))
    if type(draft.existing_server_url) is not str:
        issues.append(VllmIssue("invalid_existing_server_url", "existing_server_url"))
    if type(draft.dtype) is not str or draft.dtype not in _DTYPE_VALUES:
        issues.append(VllmIssue("invalid_dtype", "dtype"))
    if draft.tensor_parallel_size is not None and (
        type(draft.tensor_parallel_size) is not int or draft.tensor_parallel_size < 1
    ):
        issues.append(VllmIssue("invalid_tensor_parallel_size", "tensor_parallel_size"))
    if draft.maximum_model_length is not None and (
        type(draft.maximum_model_length) is not int or draft.maximum_model_length < 1
    ):
        issues.append(VllmIssue("invalid_maximum_model_length", "maximum_model_length"))
    if draft.gpu_memory_utilization is not None and (
        type(draft.gpu_memory_utilization) is not float
        or not math.isfinite(draft.gpu_memory_utilization)
        or not 0 < draft.gpu_memory_utilization <= 1
    ):
        issues.append(
            VllmIssue("invalid_gpu_memory_utilization", "gpu_memory_utilization")
        )
    if type(draft.trust_remote_code) is not bool:
        issues.append(VllmIssue("invalid_trust_remote_code", "trust_remote_code"))
    if type(draft.raw_arguments) is not str:
        issues.append(VllmIssue("invalid_arguments", "raw_arguments"))
    return tuple(issues)


def _invalid_draft_fingerprint(issues: tuple[VllmIssue, ...]) -> str:
    """Return non-sensitive failed-input identity without inspecting invalid values."""

    fields = tuple((issue.code, issue.field) for issue in issues)
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()


def client_api_url(bind_address: str, port: int) -> str:
    """Return the private client endpoint for a local vLLM API server."""

    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(bind_address, bind_address)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}/v1"


def is_port_available(bind_address: str, port: int) -> bool:
    """Check availability without retaining a listening socket."""

    family = socket.AF_INET6 if ":" in bind_address else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_address, port))
        return True
    except OSError:
        return False


def _is_network_exposed(bind_address: str) -> bool:
    try:
        return not ipaddress.ip_address(bind_address).is_loopback
    except ValueError:
        return bind_address.lower() != "localhost"


def _validate_bind_address(bind_address: str) -> VllmIssue | None:
    try:
        ipaddress.ip_address(bind_address)
    except ValueError:
        if bind_address.lower() != "localhost":
            return VllmIssue("invalid_bind_address", "bind_address")
    return None


def is_valid_hugging_face_repository_id(value: object) -> bool:
    """Return whether a value is a namespaced Hugging Face repository ID."""

    return (
        type(value) is str
        and _HF_REPOSITORY.fullmatch(value) is not None
        and ".." not in value
        and "--" not in value
    )


def is_safe_local_model_path_shape(value: object) -> bool:
    """Accept an absolute local path shape without requiring it to exist."""

    if type(value) is not str or not value or value != value.strip():
        return False
    if "\x00" in value or value.startswith("//") or "://" in value:
        return False
    if not (Path(value).is_absolute() or _WINDOWS_DRIVE_ABSOLUTE_PATH.match(value)):
        return False
    components = value.replace("\\", "/").split("/")
    return any(components[1:]) and not any(
        component in {".", ".."} or component.startswith("-")
        for component in components
    )


def _validate_local_model_directory(value: str) -> VllmIssue | None:
    if not is_safe_local_model_path_shape(value):
        return VllmIssue("invalid_model_directory", "model_value")
    try:
        selected = validate_path_simple(value, require_exists=True)
    except ValueError:
        return VllmIssue("invalid_model_directory", "model_value")
    if not selected.is_dir():
        return VllmIssue("invalid_model_directory", "model_value")
    return None


def _validate_existing_url(value: str) -> VllmIssue | None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return VllmIssue("invalid_existing_server_url", "existing_server_url")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return VllmIssue("invalid_existing_server_url", "existing_server_url")
    return None


def _classify_probe_version(output: object) -> str | None:
    """Keep a short printable version classification, never child output."""

    text = str(output).strip()
    if not text or len(text.encode("utf-8")) > _MAX_PROBE_OUTPUT_BYTES:
        return None
    if "\n" in text or "\r" in text or not _VERSION_OUTPUT.fullmatch(text):
        return None
    return text


def _terminate_and_reap_probe(process: subprocess.Popen[bytes]) -> None:
    """Terminate a probe and deterministically reap it without leaking a child."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=_PROBE_REAP_TIMEOUT_SECONDS)
        return
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.kill()
        process.wait(timeout=_PROBE_REAP_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        pass


def _run_default_probe(argv: list[str]) -> tuple[bool, str | None]:
    """Stream default subprocess output through a strict byte ceiling."""

    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False, None
    assert process.stdout is not None
    output = bytearray()
    deadline = time.monotonic() + _PROBE_TIMEOUT_SECONDS
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_and_reap_probe(process)
                return False, None
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if not readable:
                _terminate_and_reap_probe(process)
                return False, None
            chunk = os.read(process.stdout.fileno(), _MAX_PROBE_OUTPUT_BYTES - len(output) + 1)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > _MAX_PROBE_OUTPUT_BYTES:
                _terminate_and_reap_probe(process)
                return False, None
        try:
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _terminate_and_reap_probe(process)
            return False, None
        if returncode != 0:
            return False, None
        return True, _classify_probe_version(output.decode("utf-8", errors="replace"))
    finally:
        process.stdout.close()
        _terminate_and_reap_probe(process)


def _run_probe(run: Callable[..., object], argv: list[str]) -> tuple[bool, str | None]:
    """Run one bounded probe and retain only a classified version string."""

    if run is subprocess.run:
        return _run_default_probe(argv)

    try:
        completed = run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None
    if getattr(completed, "returncode", 1) != 0:
        return False, None
    return True, _classify_probe_version(getattr(completed, "stdout", ""))


def _resolve_python_environment(
    python_environment: str, which: Callable[[str], str | None]
) -> Path | None:
    """Resolve the selected interpreter before deriving its matching CLI."""

    selected = Path(python_environment)
    if selected.parent == Path("."):
        resolved = which(python_environment)
        selected = Path(resolved) if resolved else Path()
    selected = selected.absolute()
    return selected if selected.is_file() and os.access(selected, os.X_OK) else None


def _matching_vllm_cli(python_path: Path) -> Path | None:
    """Resolve vLLM only beside the selected resolved interpreter."""

    candidate = python_path.with_name("vllm")
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def run_vllm_preflight(
    draft: VllmLaunchDraft,
    generation: int,
    *,
    run: Callable[..., object] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    port_available: Callable[[str, int], bool] = is_port_available,
) -> VllmPreflightResult:
    """Run bounded argv-only setup checks without launching vLLM.

    Existing-server setup intentionally performs only endpoint-shape validation;
    health and model readiness belong to the later connection owner.
    """

    structural_issues = _validate_draft_structure(draft)
    if structural_issues:
        return VllmPreflightResult(
            generation,
            _invalid_draft_fingerprint(structural_issues),
            structural_issues,
        )

    issues: list[VllmIssue] = []
    fingerprint = semantic_fingerprint(draft)
    if draft.mode is VllmMode.EXISTING:
        issue = _validate_existing_url(draft.existing_server_url)
        if issue is not None:
            issues.append(issue)
        return VllmPreflightResult(generation, fingerprint, tuple(issues))

    bind_issue = _validate_bind_address(draft.bind_address)
    if bind_issue is not None:
        issues.append(bind_issue)
    if bind_issue is None and not port_available(draft.bind_address, draft.port):
        issues.append(VllmIssue("port_unavailable", "port"))
    if draft.model_source is VllmModelSource.HUGGING_FACE:
        if not is_valid_hugging_face_repository_id(draft.model_value):
            issues.append(VllmIssue("invalid_hugging_face_model", "model_value"))
    else:
        issue = _validate_local_model_directory(draft.model_value)
        if issue is not None:
            issues.append(issue)
    issues.extend(validate_raw_arguments(draft.raw_arguments))

    python_path: Path | None = None
    python_version: str | None = None
    if not draft.python_environment.strip():
        issues.append(VllmIssue("missing_python_environment", "python_environment"))
    else:
        python_path = _resolve_python_environment(draft.python_environment, which)
        if python_path is None:
            issues.append(VllmIssue("python_unavailable", "python_environment"))
        else:
            python_ok, python_version = _run_probe(run, [str(python_path), "--version"])
            if not python_ok or python_version is None:
                issues.append(VllmIssue("python_unavailable", "python_environment"))
    cli_path = _matching_vllm_cli(python_path) if python_path is not None else None
    if cli_path is None:
        issues.append(VllmIssue("vllm_cli_unavailable", "python_environment"))
    vllm_version: str | None = None
    if cli_path is not None:
        version_ok, vllm_version = _run_probe(run, [str(cli_path), "--version"])
        if not version_ok or vllm_version is None:
            issues.append(VllmIssue("vllm_cli_unavailable", "python_environment"))
    import_ok, _ = _run_probe(run, [str(python_path), "-c", "import vllm"]) if python_path else (False, None)
    if not import_ok:
        issues.append(VllmIssue("vllm_import_unavailable", "python_environment"))
    return VllmPreflightResult(
        generation,
        fingerprint,
        tuple(issues),
        python_version=python_version,
        vllm_version=vllm_version,
        cli_path=cli_path,
        network_exposed=_is_network_exposed(draft.bind_address),
    )


def build_vllm_command(
    draft: VllmLaunchDraft,
    preflight: VllmPreflightResult,
    *,
    current_generation: int | None = None,
) -> tuple[str, ...]:
    """Build the public vLLM command after successful current preflight only."""

    if _validate_draft_structure(draft):
        raise ValueError("build_vllm_command requires valid structured launch draft")
    if preflight.issues or preflight.cli_path is None:
        raise ValueError("build_vllm_command requires successful current preflight")
    if preflight.fingerprint != semantic_fingerprint(draft):
        raise ValueError("build_vllm_command requires matching fingerprint")
    if current_generation is not None and preflight.generation != current_generation:
        raise ValueError("build_vllm_command requires current generation")
    if draft.mode is not VllmMode.LOCAL:
        raise ValueError("build_vllm_command is only valid for local mode")
    if validate_raw_arguments(draft.raw_arguments):
        raise ValueError("raw arguments conflict with managed launch settings")
    command = [
        str(preflight.cli_path),
        "serve",
        draft.model_value,
        "--host",
        draft.bind_address,
        "--port",
        str(draft.port),
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]
    if draft.dtype:
        command.extend(("--dtype", draft.dtype))
    if draft.tensor_parallel_size is not None:
        command.extend(("--tensor-parallel-size", str(draft.tensor_parallel_size)))
    if draft.maximum_model_length is not None:
        command.extend(("--max-model-len", str(draft.maximum_model_length)))
    if draft.gpu_memory_utilization is not None:
        command.extend(("--gpu-memory-utilization", str(draft.gpu_memory_utilization)))
    if draft.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(shlex.split(draft.raw_arguments))
    return tuple(command)
