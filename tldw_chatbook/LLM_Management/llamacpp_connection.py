"""Bounded, generation-fenced llama.cpp health and model evidence (ADR-114)."""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

import httpx

from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint

SERVED_MODEL_NAME = "chatbook-llamacpp"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
MAX_RESPONSE_BYTES = 64 * 1024
MAX_MODELS = 100
RuntimeOwner = Literal["lab_process", "external_server"]


def admissible_model_id(value: object) -> bool:
    """Accept exact printable model IDs without filesystem-identifying text."""
    if type(value) is not str or not 1 <= len(value) <= 120:
        return False
    if value != " ".join(value.split()) or not value.isprintable():
        return False
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
        return False
    return not (
        value.casefold().startswith("file:")
        or value.startswith(("/", "./", "../", "~/", "\\\\", "//"))
        or re.match(r"^[A-Za-z]:[/\\]", value)
        or "\\" in value
        or value.casefold().endswith(".gguf")
        or any(part in {".", ".."} for part in value.split("/"))
    )


def canonical_base_url(value: str) -> str:
    """Normalize an explicit credential-free endpoint without adding defaults."""
    if type(value) is not str or not value.strip():
        raise ValueError("Enter a llama.cpp endpoint.")
    resolution = resolve_provider_endpoint("llama_cpp", value)
    if resolution.errors or not resolution.persisted_endpoint:
        raise ValueError("Enter a valid credential-free llama.cpp endpoint.")
    return resolution.persisted_endpoint


@dataclass(frozen=True, slots=True)
class LlamaCppConnectionTarget:
    """The only connection fields permitted across Lab's ownership boundary."""

    base_url: str
    model_id: str
    runtime_owner: RuntimeOwner
    verification_generation: int
    provider_key: Literal["llama_cpp"] = field(default="llama_cpp", init=False)

    def __post_init__(self) -> None:
        if canonical_base_url(self.base_url) != self.base_url:
            raise ValueError("Connection endpoint is not canonical.")
        if not admissible_model_id(self.model_id):
            raise ValueError("Server model ID is unsafe; configure a non-path --alias.")
        if self.runtime_owner not in {"lab_process", "external_server"}:
            raise ValueError("Invalid runtime owner.")
        if (
            type(self.verification_generation) is not int
            or self.verification_generation < 1
        ):
            raise ValueError("Invalid verification generation.")


@dataclass(frozen=True, slots=True, eq=False)
class LlamaCppProbeRequest:
    base_url: str
    model_id: str | None
    runtime_owner: RuntimeOwner
    generation: int
    live_check: Callable[[], bool] | None = field(default=None, repr=False)

    def process_alive(self) -> bool:
        """Require positive process evidence for locally owned launches."""
        if self.runtime_owner == "external_server":
            return True
        try:
            return self.live_check is not None and self.live_check() is True
        except Exception:  # noqa: BLE001 - uncertain liveness cannot authorize adoption
            return False


@dataclass(frozen=True, slots=True)
class LlamaCppProbeResult:
    request: LlamaCppProbeRequest
    code: str
    model_ids: tuple[str, ...] = ()
    selected_model: str | None = None


@dataclass(frozen=True, slots=True)
class LlamaCppConnectionSnapshot:
    request: LlamaCppProbeRequest | None = None
    state: str = "not_configured"
    code: str = "not_configured"
    model_ids: tuple[str, ...] = ()
    target: LlamaCppConnectionTarget | None = None


class LlamaCppConnectionOwner:
    """Retain evidence for exactly one current endpoint and launch generation."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation = 0
        self._snapshot = LlamaCppConnectionSnapshot()

    def begin(
        self,
        base_url: str,
        *,
        runtime_owner: RuntimeOwner,
        model_id: str | None = None,
        live_check: Callable[[], bool] | None = None,
    ) -> LlamaCppProbeRequest:
        base_url = canonical_base_url(base_url)
        if runtime_owner not in {"lab_process", "external_server"}:
            raise ValueError("Invalid runtime owner.")
        if runtime_owner == "lab_process":
            model_id = SERVED_MODEL_NAME
        if model_id is not None and not admissible_model_id(model_id):
            raise ValueError("Server model ID is unsafe; configure a non-path --alias.")
        with self._lock:
            self._generation += 1
            request = LlamaCppProbeRequest(
                base_url, model_id, runtime_owner, self._generation, live_check
            )
            self._snapshot = LlamaCppConnectionSnapshot(request, "checking", "checking")
            return request

    def invalidate(self, code: str = "target_changed") -> None:
        with self._lock:
            self._generation += 1
            self._snapshot = LlamaCppConnectionSnapshot(
                state="not_configured", code=code
            )

    def accept(self, result: LlamaCppProbeResult) -> bool:
        """Publish only evidence for the current exact request."""
        with self._lock:
            request = result.request
            if request is not self._snapshot.request:
                return False
            if not request.process_alive():
                result = LlamaCppProbeResult(request, "process_unavailable")
            target = None
            if result.code == "ready" and result.selected_model is not None:
                if result.selected_model not in result.model_ids or (
                    request.model_id is not None
                    and request.model_id != result.selected_model
                ):
                    return False
                target = LlamaCppConnectionTarget(
                    request.base_url,
                    result.selected_model,
                    request.runtime_owner,
                    request.generation,
                )
            if any(not admissible_model_id(model) for model in result.model_ids):
                return False
            state = (
                "ready"
                if target
                else "loading_model"
                if result.code == "loading_model"
                else "needs_attention"
            )
            self._snapshot = LlamaCppConnectionSnapshot(
                request, state, result.code, result.model_ids, target
            )
            return True

    def snapshot(self) -> LlamaCppConnectionSnapshot:
        with self._lock:
            request = self._snapshot.request
            if (
                self._snapshot.target is not None
                and request is not None
                and not request.process_alive()
            ):
                self.invalidate("process_unavailable")
            return self._snapshot

    def is_current(self, target: LlamaCppConnectionTarget) -> bool:
        return (
            type(target) is LlamaCppConnectionTarget
            and self.snapshot().target == target
        )


async def probe_llamacpp_target(
    request: LlamaCppProbeRequest,
    *,
    credential: str | None = None,
    client: httpx.AsyncClient | None = None,
    timeout_seconds: float = 5.0,
) -> LlamaCppProbeResult:
    """Observe compatible health and safe model identity under one deadline.

    Args:
        request: Immutable exact target to verify.
        credential: Credential already resolved for this exact endpoint, never retained.
        client: Optional injected client for controlled transport tests.
        timeout_seconds: Positive aggregate deadline, at most thirty seconds.

    Returns:
        Bounded safe evidence; arbitrary response bodies and exceptions are omitted.
    """
    if not 0 < timeout_seconds <= 30:
        raise ValueError("Invalid probe deadline.")
    if not request.process_alive():
        return LlamaCppProbeResult(request, "process_unavailable")
    endpoints = resolve_provider_endpoint("llama_cpp", request.base_url)
    models_url = endpoints.models_url
    if endpoints.errors or not models_url or not models_url.endswith("/v1/models"):
        return LlamaCppProbeResult(request, "invalid_endpoint")
    health_url = models_url[: -len("/v1/models")] + "/health"
    headers = {"accept": "application/json", "accept-encoding": "identity"}
    if credential:
        headers["authorization"] = f"Bearer {credential}"
    session = client or httpx.AsyncClient(
        trust_env=False, follow_redirects=False, timeout=timeout_seconds
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            async with session.stream(
                "GET", health_url, headers=headers, follow_redirects=False
            ) as response:
                if response.status_code in {401, 403}:
                    return LlamaCppProbeResult(request, "credential_required")
                if response.status_code == 503:
                    return LlamaCppProbeResult(request, "loading_model")
                if response.status_code != 200:
                    return LlamaCppProbeResult(request, "health_failed")
            if not request.process_alive():
                return LlamaCppProbeResult(request, "process_unavailable")
            body = bytearray()
            async with session.stream(
                "GET", models_url, headers=headers, follow_redirects=False
            ) as response:
                if response.status_code in {401, 403}:
                    return LlamaCppProbeResult(request, "credential_required")
                if response.status_code != 200:
                    return LlamaCppProbeResult(request, "invalid_models_response")
                if (
                    response.headers.get("content-encoding", "identity").lower()
                    != "identity"
                ):
                    return LlamaCppProbeResult(request, "invalid_models_response")
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        return LlamaCppProbeResult(request, "invalid_models_response")
                    body.extend(chunk)
            payload = json.loads(body)
            entries = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_MODELS:
                return LlamaCppProbeResult(request, "invalid_models_response")
            models = []
            for entry in entries:
                model = entry.get("id") if isinstance(entry, dict) else None
                if not admissible_model_id(model) or (
                    credential and credential in model
                ):
                    return LlamaCppProbeResult(request, "unsafe_model_id")
                if model not in models:
                    models.append(model)
            selected = request.model_id
            if selected is not None and selected not in models:
                return LlamaCppProbeResult(request, "model_missing")
            if selected is None and len(models) == 1:
                selected = models[0]
            if not request.process_alive():
                return LlamaCppProbeResult(request, "process_unavailable")
            return LlamaCppProbeResult(
                request,
                "ready" if selected else "choose_model",
                tuple(models),
                selected,
            )
    except RecursionError:
        return LlamaCppProbeResult(request, "invalid_models_response")
    except (TimeoutError, httpx.TimeoutException):
        return LlamaCppProbeResult(request, "timeout")
    except (httpx.HTTPError, ValueError, UnicodeError):
        return LlamaCppProbeResult(request, "connection_failed")
    finally:
        if client is None:
            await session.aclose()


def connection_owner(app: object) -> LlamaCppConnectionOwner:
    """Lazily attach one process-local evidence owner to the application."""
    owner = getattr(app, "_llamacpp_connection_owner", None)
    if owner is None:
        owner = LlamaCppConnectionOwner()
        app._llamacpp_connection_owner = owner
    return owner


def local_launch_url(host: str, port: str) -> str:
    """Validate a local bind and return its client-facing endpoint."""
    import ipaddress

    if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    if host == "localhost":
        host = "127.0.0.1"
    address = ipaddress.ip_address(host)
    if address.is_unspecified:
        host = "::1" if address.version == 6 else "127.0.0.1"
    return f"http://{'[' + host + ']' if ':' in host else host}:{int(port)}"
