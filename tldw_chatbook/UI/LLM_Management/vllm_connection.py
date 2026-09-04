"""Immutable ownership and bounded HTTP verification for vLLM targets."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Literal

import httpx

from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint

from .vllm_setup import (
    SERVED_MODEL_NAME,
    VllmConnectionTarget,
    VllmIssue,
    VllmLaunchDraft,
    VllmLaunchSnapshot,
    VllmReadinessState,
    launch_snapshot_from_draft,
    semantic_fingerprint,
)

RuntimeOwner = Literal["chatbook", "external"]
CredentialSource = Literal["none", "configured", "environment"]

_ACTIVITY_LIMIT = 32
_DISCOVERED_MODELS_LIMIT = 100
_MODELS_RESPONSE_LIMIT_BYTES = 64 * 1024
_MAX_TIMEOUT_SECONDS = 120.0
_WINDOWS_ROOT = re.compile(r"^[A-Za-z]:[/\\]")
_ACTIVITY_CODES = frozenset(
    {
        "cancelled",
        "checking",
        "claim_unavailable",
        "credential_required",
        "health_checking",
        "health_ok",
        "health_timeout",
        "invalid_endpoint",
        "invalid_models_response",
        "invalidated",
        "launch_failed",
        "launch_reserved",
        "loading_model",
        "model_checking",
        "model_missing",
        "models_discovered",
        "process_alive",
        "process_exited",
        "preflight_failed",
        "ready",
        "recomposed",
        "screen_detached",
        "stopped",
        "stopping",
        "target_changed",
    }
)
_ELAPSED_BUCKETS = frozenset(
    {"under_1s", "1_to_4s", "5_to_14s", "15_to_29s", "30s_or_more"}
)
_ISSUE_CODES = frozenset(
    {
        "cancelled",
        "claim_unavailable",
        "credential_required",
        "health_timeout",
        "invalid_endpoint",
        "invalid_models_response",
        "launch_failed",
        "model_missing",
        "process_exited",
        "arguments_conflict",
        "invalid_arguments",
        "invalid_bind_address",
        "invalid_existing_server_url",
        "invalid_gpu_memory_utilization",
        "invalid_hugging_face_model",
        "invalid_maximum_model_length",
        "invalid_model_directory",
        "invalid_port",
        "invalid_tensor_parallel_size",
        "missing_python_environment",
        "port_unavailable",
        "python_unavailable",
        "vllm_cli_unavailable",
        "vllm_import_unavailable",
    }
)


def activity_elapsed_bucket(elapsed_seconds: float) -> str:
    """Bucket elapsed activity against the bounded 30-second readiness window."""

    elapsed = max(0.0, elapsed_seconds)
    if elapsed < 1:
        return "under_1s"
    if elapsed < 5:
        return "1_to_4s"
    if elapsed < 15:
        return "5_to_14s"
    if elapsed < 30:
        return "15_to_29s"
    return "30s_or_more"


def _activity(
    code: str, started_at: float, exit_code: int | None = None
) -> "VllmActivityEvent":
    return VllmActivityEvent(
        code,
        activity_elapsed_bucket(time.monotonic() - started_at),
        exit_code,
    )


def _is_admissible_model_id(value: object) -> bool:
    """Apply ADR-114's exact remote model identifier boundary."""

    if not isinstance(value, str) or not 1 <= len(value) <= 120:
        return False
    if value != " ".join(value.split()) or not value.isprintable():
        return False
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    ):
        return False
    lowered = value.casefold()
    if (
        lowered.startswith("file:")
        or value.startswith(("/", "./", "../", "~/", "\\\\", "//"))
        or _WINDOWS_ROOT.match(value)
        or "\\" in value
        or lowered.endswith(".gguf")
    ):
        return False
    return all(segment not in {".", ".."} for segment in value.split("/"))


@dataclass(frozen=True, slots=True)
class VllmOperationToken:
    generation: int
    fingerprint: str
    runtime_owner: RuntimeOwner

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("generation must be a positive integer")
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) > 64:
            raise ValueError("fingerprint must be bounded")
        if self.runtime_owner not in {"chatbook", "external"}:
            raise ValueError("invalid runtime owner")


@dataclass(frozen=True, slots=True)
class VllmActivityEvent:
    code: str
    elapsed_bucket: str
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if self.code not in _ACTIVITY_CODES:
            raise ValueError("activity code is not allowlisted")
        if self.elapsed_bucket not in _ELAPSED_BUCKETS:
            raise ValueError("elapsed bucket is not allowlisted")
        if self.exit_code is not None and (
            type(self.exit_code) is not int or not -255 <= self.exit_code <= 255
        ):
            raise ValueError("exit code is not bounded")


@dataclass(frozen=True, slots=True)
class VllmProbeRequest:
    token: VllmOperationToken
    api_url: str
    expected_model_id: str | None
    cancellation_requested: Callable[[], bool] | None = field(
        default=None, repr=False, compare=False
    )
    process_alive: Callable[[], bool] | None = field(
        default=None, repr=False, compare=False
    )
    connect_timeout_seconds: float = 1.0
    read_timeout_seconds: float = 2.0
    total_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.token, VllmOperationToken):
            raise TypeError("token must be a VllmOperationToken")
        if not isinstance(self.api_url, str) or not 1 <= len(self.api_url) <= 2048:
            raise ValueError("api_url must be bounded")
        if self.expected_model_id is not None and not _is_admissible_model_id(
            self.expected_model_id
        ):
            raise ValueError("expected_model_id is invalid")
        for timeout in (
            self.connect_timeout_seconds,
            self.read_timeout_seconds,
            self.total_timeout_seconds,
        ):
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or not 0 < timeout <= _MAX_TIMEOUT_SECONDS
            ):
                raise ValueError("timeouts must be finite, positive, and bounded")


@dataclass(frozen=True, slots=True)
class VllmProbeResult:
    token: VllmOperationToken
    state: VllmReadinessState
    target: VllmConnectionTarget | None
    issue: VllmIssue | None
    activity: tuple[VllmActivityEvent, ...] = ()
    discovered_model_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.token, VllmOperationToken):
            raise TypeError("token must be a VllmOperationToken")
        if not isinstance(self.state, VllmReadinessState):
            raise TypeError("state must be a VllmReadinessState")
        if self.issue is not None:
            if not isinstance(self.issue, VllmIssue):
                raise TypeError("issue must be a VllmIssue")
            if (
                self.issue.code not in _ISSUE_CODES
                or self.issue.detail
                or not isinstance(self.issue.field, str)
                or len(self.issue.field) > 32
            ):
                raise ValueError("issue must use a bounded allowlisted classification")
        if not isinstance(self.activity, tuple) or len(self.activity) > _ACTIVITY_LIMIT:
            raise ValueError("activity must be a bounded tuple")
        if any(not isinstance(event, VllmActivityEvent) for event in self.activity):
            raise TypeError("activity contains an invalid event")
        if (
            not isinstance(self.discovered_model_ids, tuple)
            or len(self.discovered_model_ids) > _DISCOVERED_MODELS_LIMIT
            or len(set(self.discovered_model_ids)) != len(self.discovered_model_ids)
            or any(
                not _is_admissible_model_id(model_id)
                for model_id in self.discovered_model_ids
            )
        ):
            raise ValueError("discovered model identifiers must be bounded and unique")
        if self.token.runtime_owner == "chatbook" and self.discovered_model_ids:
            raise ValueError(
                "Chatbook-owned results cannot retain discovery candidates"
            )
        ready = self.state is VllmReadinessState.READY
        if ready != (self.target is not None):
            raise ValueError("a ready result requires exactly one target")
        if ready and self.issue is not None:
            raise ValueError("a ready result cannot include an issue")
        if self.target is not None:
            if not isinstance(self.target, VllmConnectionTarget):
                raise TypeError("target must be a VllmConnectionTarget")
            endpoint = resolve_provider_endpoint("vllm", self.target.api_url)
            if (
                self.target.provider_key != "vllm"
                or self.target.generation != self.token.generation
                or self.target.runtime_owner != self.token.runtime_owner
                or not _is_admissible_model_id(self.target.model_id)
                or self.target.credential_source
                not in {"none", "configured", "environment"}
                or endpoint.errors
                or endpoint.persisted_endpoint != self.target.api_url
                or (
                    self.token.runtime_owner == "chatbook"
                    and self.target.model_id != SERVED_MODEL_NAME
                )
            ):
                raise ValueError("target does not match the operation token")
            if self.discovered_model_ids not in {
                (),
                (self.target.model_id,),
            }:
                raise ValueError(
                    "a ready result can retain only its exact selected model"
                )


@dataclass(frozen=True, slots=True)
class VllmConnectionSnapshot:
    current_token: VllmOperationToken | None
    state: VllmReadinessState
    launch_snapshot: VllmLaunchSnapshot | None
    target: VllmConnectionTarget | None
    issue: VllmIssue | None
    activity: tuple[VllmActivityEvent, ...] = ()
    discovered_model_ids: tuple[str, ...] = ()

    @property
    def token(self) -> VllmOperationToken | None:
        """Compatibility alias for callers that used the initial contract draft."""

        return self.current_token

    @property
    def generation(self) -> int:
        return self.current_token.generation if self.current_token is not None else 0


class VllmConnectionOwner:
    """App-scoped generation authority for vLLM connection evidence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation = 0
        self._launch_claim: object | None = None
        self._claim_launch_snapshot: VllmLaunchSnapshot | None = None
        self._claim_token: VllmOperationToken | None = None
        self._snapshot = VllmConnectionSnapshot(
            current_token=None,
            state=VllmReadinessState.NOT_CONFIGURED,
            launch_snapshot=None,
            target=None,
            issue=None,
        )

    def begin(
        self,
        draft: VllmLaunchDraft,
        *,
        runtime_owner: RuntimeOwner,
        profile_id: str | None = None,
        profile_name: str = "Chatbook-managed vLLM",
    ) -> VllmOperationToken:
        """Begin a new generation and discard all prior readiness evidence."""

        if runtime_owner not in {"chatbook", "external"}:
            raise ValueError("invalid runtime owner")
        with self._lock:
            self._generation += 1
            fingerprint = semantic_fingerprint(draft)
            token = VllmOperationToken(self._generation, fingerprint, runtime_owner)
            launch_snapshot = None
            if runtime_owner == "chatbook":
                launch_snapshot = launch_snapshot_from_draft(
                    draft,
                    generation=token.generation,
                    profile_id=profile_id,
                    profile_name=profile_name,
                )
            self._snapshot = VllmConnectionSnapshot(
                current_token=token,
                state=VllmReadinessState.CHECKING,
                launch_snapshot=launch_snapshot,
                target=None,
                issue=None,
                activity=(VllmActivityEvent("checking", "under_1s"),),
            )
            return token

    def bind_launch_claim(self, token: VllmOperationToken, claim: object) -> bool:
        """Bind an exact lifecycle claim to the snapshot that it launches."""

        cancel_event = getattr(claim, "cancel_event", None)
        with self._lock:
            if (
                token != self._snapshot.current_token
                or token.runtime_owner != "chatbook"
                or self._snapshot.launch_snapshot is None
                or getattr(claim, "provider", None) != "vllm"
                or getattr(claim, "authority", None) != SERVED_MODEL_NAME
                or cancel_event is None
                or not callable(getattr(cancel_event, "is_set", None))
                or cancel_event.is_set()
            ):
                return False
            self._launch_claim = claim
            self._claim_launch_snapshot = self._snapshot.launch_snapshot
            self._claim_token = token
            return True

    def begin_claim_retry(self, claim: object) -> VllmOperationToken | None:
        """Begin verification against only an uncancelled bound live claim."""

        cancel_event = getattr(claim, "cancel_event", None)
        with self._lock:
            launch_snapshot = self._claim_launch_snapshot
            if (
                self._launch_claim is not claim
                or launch_snapshot is None
                or cancel_event is None
                or not callable(getattr(cancel_event, "is_set", None))
                or cancel_event.is_set()
            ):
                return None
            self._generation += 1
            token = VllmOperationToken(
                self._generation,
                launch_snapshot.fingerprint,
                "chatbook",
            )
            self._snapshot = VllmConnectionSnapshot(
                current_token=token,
                state=VllmReadinessState.CHECKING,
                launch_snapshot=launch_snapshot,
                target=None,
                issue=None,
                activity=(VllmActivityEvent("checking", "under_1s"),),
            )
            self._claim_token = token
            return token

    def owns_launch_claim(self, claim: object) -> bool:
        """Return whether ``claim`` is the exact bound launch identity."""

        with self._lock:
            return self._launch_claim is claim

    def bound_launch_snapshot(self, claim: object) -> VllmLaunchSnapshot | None:
        """Return the immutable snapshot bound to one exact launch claim."""

        with self._lock:
            if self._launch_claim is not claim:
                return None
            return self._claim_launch_snapshot

    def release_launch_claim(self, claim: object) -> bool:
        """Forget launch evidence only for the exact bound lifecycle claim."""

        with self._lock:
            if self._launch_claim is not claim:
                return False
            self._launch_claim = None
            self._claim_launch_snapshot = None
            self._claim_token = None
            return True

    def invalidate(self, reason: str) -> int:
        """Advance the generation and clear all connection evidence."""

        code = reason if reason in _ACTIVITY_CODES else "invalidated"
        with self._lock:
            self._generation += 1
            previous = self._snapshot.current_token
            token = VllmOperationToken(
                self._generation,
                previous.fingerprint if previous is not None else "",
                previous.runtime_owner if previous is not None else "external",
            )
            self._snapshot = VllmConnectionSnapshot(
                current_token=token,
                state=VllmReadinessState.NOT_CONFIGURED,
                launch_snapshot=None,
                target=None,
                issue=None,
                activity=(
                    self._snapshot.activity + (VllmActivityEvent(code, "under_1s"),)
                )[-_ACTIVITY_LIMIT:],
            )
            return token.generation

    def settle(self, token: VllmOperationToken, result: VllmProbeResult) -> bool:
        """Publish a result only when both tokens are the current generation."""

        with self._lock:
            if type(result) is not VllmProbeResult:
                return False
            try:
                validated = VllmProbeResult(
                    token=result.token,
                    state=result.state,
                    target=result.target,
                    issue=result.issue,
                    activity=result.activity,
                    discovered_model_ids=result.discovered_model_ids,
                )
            except (TypeError, ValueError):
                return False
            if (
                token != self._snapshot.current_token
                or validated.token != token
                or validated != result
            ):
                return False
            if validated.target is not None and token.runtime_owner == "chatbook":
                launch_snapshot = self._claim_launch_snapshot
                if (
                    self._launch_claim is None
                    or launch_snapshot is None
                    or self._claim_token != token
                    or launch_snapshot.fingerprint != token.fingerprint
                ):
                    return False
                launched_endpoint = resolve_provider_endpoint(
                    "vllm", launch_snapshot.client_api_url
                )
                if (
                    launched_endpoint.errors
                    or launched_endpoint.persisted_endpoint != validated.target.api_url
                ):
                    return False
            activity = (self._snapshot.activity + validated.activity)[-_ACTIVITY_LIMIT:]
            self._snapshot = replace(
                self._snapshot,
                state=validated.state,
                target=validated.target,
                issue=validated.issue,
                activity=activity,
                discovered_model_ids=validated.discovered_model_ids,
            )
            return True

    def snapshot(self) -> VllmConnectionSnapshot:
        """Return an immutable copy of the current evidence snapshot."""

        with self._lock:
            return replace(self._snapshot, activity=tuple(self._snapshot.activity))


def _default_credential_resolver() -> tuple[str | None, CredentialSource]:
    from tldw_chatbook.config import get_api_key

    value = get_api_key("vllm")
    if not value:
        return None, "none"
    source: CredentialSource = (
        "environment" if os.environ.get("VLLM_API_KEY") == value else "configured"
    )
    return value, source


def _failure(
    request: VllmProbeRequest,
    code: str,
    field: str,
    started_at: float,
    *events: VllmActivityEvent,
    discovered_model_ids: tuple[str, ...] = (),
) -> VllmProbeResult:
    return VllmProbeResult(
        token=request.token,
        state=VllmReadinessState.NEEDS_ATTENTION,
        target=None,
        issue=VllmIssue(code, field),
        activity=events + (_activity(code, started_at),),
        discovered_model_ids=discovered_model_ids,
    )


async def probe_vllm_target(
    request: VllmProbeRequest,
    *,
    credential_resolver: Callable[[], tuple[str | None, CredentialSource]]
    | None = None,
    client: httpx.AsyncClient | None = None,
) -> VllmProbeResult:
    """Verify health plus an admissible exact model identity within one deadline."""

    started_at = time.monotonic()
    if request.cancellation_requested and request.cancellation_requested():
        return _failure(request, "cancelled", "connection", started_at)
    if request.process_alive and not request.process_alive():
        return _failure(request, "process_exited", "process", started_at)

    try:
        endpoints = resolve_provider_endpoint("vllm", request.api_url)
        models_url = endpoints.models_url
        completion_url = endpoints.persisted_endpoint
        if (
            not isinstance(models_url, str)
            or not models_url.endswith("/v1/models")
            or not isinstance(completion_url, str)
        ):
            return _failure(request, "invalid_endpoint", "connection", started_at)
        health_url = f"{models_url[: -len('/v1/models')]}/health"
    except (TypeError, ValueError):
        return _failure(request, "invalid_endpoint", "connection", started_at)

    resolver = credential_resolver or _default_credential_resolver
    try:
        credential, credential_source = resolver()
    except Exception:
        credential, credential_source = None, "none"
    if credential_source not in {"none", "configured", "environment"}:
        credential_source = "none"
    headers = {
        "accept": "application/json",
        "accept-encoding": "identity",
    }
    if credential:
        headers["authorization"] = f"Bearer {credential}"

    timeout = httpx.Timeout(
        request.total_timeout_seconds,
        connect=request.connect_timeout_seconds,
        read=request.read_timeout_seconds,
        write=request.read_timeout_seconds,
        pool=request.connect_timeout_seconds,
    )
    owns_client = client is None
    session = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    health_event = _activity("health_checking", started_at)
    try:
        async with asyncio.timeout(request.total_timeout_seconds):
            async with session.stream("GET", health_url, headers=headers) as response:
                if response.status_code in {401, 403}:
                    return _failure(
                        request,
                        "credential_required",
                        "connection",
                        started_at,
                        health_event,
                    )
                if response.status_code < 200 or response.status_code >= 300:
                    return _failure(
                        request,
                        "health_timeout",
                        "connection",
                        started_at,
                        health_event,
                    )

            health_ok = _activity("health_ok", started_at)
            if request.cancellation_requested and request.cancellation_requested():
                return _failure(
                    request,
                    "cancelled",
                    "connection",
                    started_at,
                    health_event,
                    health_ok,
                )
            if request.process_alive and not request.process_alive():
                return _failure(
                    request,
                    "process_exited",
                    "process",
                    started_at,
                    health_event,
                    health_ok,
                )

            model_event = _activity("model_checking", started_at)
            body = bytearray()
            async with session.stream("GET", models_url, headers=headers) as response:
                if response.status_code in {401, 403}:
                    return _failure(
                        request,
                        "credential_required",
                        "connection",
                        started_at,
                        health_event,
                        health_ok,
                        model_event,
                    )
                if response.status_code < 200 or response.status_code >= 300:
                    return _failure(
                        request,
                        "invalid_models_response",
                        "connection",
                        started_at,
                        health_event,
                        health_ok,
                        model_event,
                    )
                async for chunk in response.aiter_raw():
                    if len(body) + len(chunk) > _MODELS_RESPONSE_LIMIT_BYTES:
                        return _failure(
                            request,
                            "invalid_models_response",
                            "connection",
                            started_at,
                            health_event,
                            health_ok,
                            model_event,
                        )
                    body.extend(chunk)

            parsed = json.loads(body)
            data = parsed.get("data") if isinstance(parsed, dict) else None
            if not isinstance(data, list):
                return _failure(
                    request,
                    "invalid_models_response",
                    "connection",
                    started_at,
                    health_event,
                    health_ok,
                    model_event,
                )
            model_ids = tuple(
                dict.fromkeys(
                    entry.get("id")
                    for entry in data
                    if isinstance(entry, dict)
                    and _is_admissible_model_id(entry.get("id"))
                    and entry.get("id") != credential
                )
            )[:_DISCOVERED_MODELS_LIMIT]
            if request.expected_model_id is not None:
                selected_model = (
                    request.expected_model_id
                    if request.expected_model_id in model_ids
                    else None
                )
            else:
                if not model_ids:
                    return _failure(
                        request,
                        "model_missing",
                        "model",
                        started_at,
                        health_event,
                        health_ok,
                        model_event,
                    )
                return VllmProbeResult(
                    token=request.token,
                    state=VllmReadinessState.NOT_CONFIGURED,
                    target=None,
                    issue=None,
                    activity=(
                        health_event,
                        health_ok,
                        model_event,
                        _activity("models_discovered", started_at),
                    ),
                    discovered_model_ids=model_ids,
                )
            if selected_model is None:
                return _failure(
                    request,
                    "model_missing",
                    "model",
                    started_at,
                    health_event,
                    health_ok,
                    model_event,
                )
            if request.cancellation_requested and request.cancellation_requested():
                return _failure(
                    request,
                    "cancelled",
                    "connection",
                    started_at,
                    health_event,
                    health_ok,
                    model_event,
                )
            if request.process_alive and not request.process_alive():
                return _failure(
                    request,
                    "process_exited",
                    "process",
                    started_at,
                    health_event,
                    health_ok,
                    model_event,
                )

            target = VllmConnectionTarget(
                provider_key="vllm",
                api_url=completion_url,
                model_id=selected_model,
                runtime_owner=request.token.runtime_owner,
                generation=request.token.generation,
                credential_source=credential_source,
            )
            return VllmProbeResult(
                token=request.token,
                state=VllmReadinessState.READY,
                target=target,
                issue=None,
                activity=(
                    health_event,
                    health_ok,
                    model_event,
                    _activity("ready", started_at),
                ),
                discovered_model_ids=(
                    (selected_model,)
                    if request.token.runtime_owner == "external"
                    else ()
                ),
            )
    except (TimeoutError, httpx.TimeoutException, httpx.TransportError):
        return _failure(
            request, "health_timeout", "connection", started_at, health_event
        )
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return _failure(
            request,
            "invalid_models_response",
            "connection",
            started_at,
            health_event,
        )
    finally:
        if owns_client:
            await session.aclose()


__all__ = [
    "CredentialSource",
    "RuntimeOwner",
    "VllmActivityEvent",
    "VllmConnectionOwner",
    "VllmConnectionSnapshot",
    "VllmOperationToken",
    "VllmProbeRequest",
    "VllmProbeResult",
    "activity_elapsed_bucket",
    "probe_vllm_target",
]
