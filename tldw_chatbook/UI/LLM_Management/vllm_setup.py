"""Pure, fail-closed setup validation for Chatbook-owned vLLM launches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import socket
import subprocess
import time
from typing import Callable, Literal
from urllib.parse import urlparse

from tldw_chatbook.Utils.path_validation import validate_path_simple


SERVED_MODEL_NAME = "chatbook-vllm"
_HF_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANAGED_OR_SECRET_FLAGS = frozenset(
    {"--host", "--port", "--model", "--served-model-name", "--api-key", "--hf-token"}
)
_MAX_PROBE_OUTPUT_BYTES = 256
_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE_REAP_TIMEOUT_SECONDS = 0.25
_VERSION_OUTPUT = re.compile(r"^[A-Za-z][A-Za-z0-9 ._+\-]{0,120}$")


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


def semantic_fingerprint(draft: VllmLaunchDraft) -> str:
    """Return a stable identity for every behaviorally relevant launch field."""

    values = asdict(draft)
    values["mode"] = draft.mode.value
    values["model_source"] = draft.model_source.value
    encoded = repr(tuple(sorted(values.items()))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_raw_arguments(raw_arguments: str) -> tuple[VllmIssue, ...]:
    """Reject malformed shell syntax and managed or credential-bearing flags."""

    if not raw_arguments.strip():
        return ()
    try:
        arguments = shlex.split(raw_arguments)
    except ValueError:
        return (VllmIssue("invalid_arguments", "raw_arguments"),)
    for argument in arguments:
        flag = argument.split("=", 1)[0]
        if flag in _MANAGED_OR_SECRET_FLAGS:
            return (VllmIssue("arguments_conflict", "raw_arguments", flag),)
    return ()


def client_api_url(bind_address: str, port: int) -> str:
    """Return the private client endpoint for a local vLLM API server."""

    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(bind_address, bind_address)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}/v1"


def is_port_available(bind_address: str, port: int) -> bool:
    """Check availability without retaining a listening socket."""

    family = socket.AF_INET6 if ":" in bind_address else socket.AF_INET
    host = "::1" if bind_address == "::" else bind_address
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
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


def _validate_local_model_directory(value: str) -> VllmIssue | None:
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
    if not isinstance(draft.port, int) or isinstance(draft.port, bool) or not 1 <= draft.port <= 65535:
        issues.append(VllmIssue("invalid_port", "port"))
    elif bind_issue is None and not port_available(draft.bind_address, draft.port):
        issues.append(VllmIssue("port_unavailable", "port"))
    if draft.model_source is VllmModelSource.HUGGING_FACE:
        if not _HF_REPOSITORY.fullmatch(draft.model_value):
            issues.append(VllmIssue("invalid_hugging_face_model", "model_value"))
    else:
        issue = _validate_local_model_directory(draft.model_value)
        if issue is not None:
            issues.append(issue)
    if draft.tensor_parallel_size is not None and (
        not isinstance(draft.tensor_parallel_size, int)
        or isinstance(draft.tensor_parallel_size, bool)
        or draft.tensor_parallel_size < 1
    ):
        issues.append(VllmIssue("invalid_tensor_parallel_size", "tensor_parallel_size"))
    if draft.maximum_model_length is not None and (
        not isinstance(draft.maximum_model_length, int)
        or isinstance(draft.maximum_model_length, bool)
        or draft.maximum_model_length < 1
    ):
        issues.append(VllmIssue("invalid_maximum_model_length", "maximum_model_length"))
    if draft.gpu_memory_utilization is not None and (
        not isinstance(draft.gpu_memory_utilization, (float, int))
        or isinstance(draft.gpu_memory_utilization, bool)
        or not 0 < draft.gpu_memory_utilization <= 1
    ):
        issues.append(VllmIssue("invalid_gpu_memory_utilization", "gpu_memory_utilization"))
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

    if preflight.issues or preflight.cli_path is None:
        raise ValueError("build_vllm_command requires successful current preflight")
    if preflight.fingerprint != semantic_fingerprint(draft):
        raise ValueError("build_vllm_command requires matching fingerprint")
    if current_generation is not None and preflight.generation != current_generation:
        raise ValueError("build_vllm_command requires current generation")
    if draft.mode is not VllmMode.LOCAL:
        raise ValueError("build_vllm_command is only valid for local mode")
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
