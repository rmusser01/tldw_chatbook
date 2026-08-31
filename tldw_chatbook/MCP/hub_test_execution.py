"""Immutable admission primitives for operator-initiated Hub tool tests."""

from __future__ import annotations

import hashlib
import asyncio
import json
import math
import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Coroutine
from typing import Any, Callable, Literal, cast

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Utils.filesystem_identity import DirectoryChain


LocalHubDecision = Literal["allowed", "approved", "denied"]
LocalHubStatus = Literal["success", "blocked", "error", "timeout", "cancelled"]
LocalHubFinalGate = Literal[
    "allow",
    "ask",
    "deny",
    "gate_error",
    "kill_switch",
    "no_callback",
    "not_checked",
    "timeout",
    "unresolved",
]
LocalHubProviderTerminal = Literal["not_started", "returned", "raised"]


_LOCAL_HUB_APPROVAL_BINDING: ContextVar[tuple[str, ...] | None] = ContextVar(
    "local_hub_approval_binding",
    default=None,
)


@dataclass(frozen=True, slots=True)
class ToolTestAdmissionPreview:
    """Public, metadata-only view of one prepared Hub tool test."""

    nonce: str
    server_key: str
    tool_name: str
    definition_hash: str
    rendered_gate: str
    authority_fingerprint: str | None
    safe_authority_label: str | None


@dataclass(frozen=True, slots=True)
class RegisteredToolTestPreview:
    """Private preview record retaining the operational workspace authority."""

    public: ToolTestAdmissionPreview
    authority: DirectoryChain | None
    expires_at: float


@dataclass(frozen=True, slots=True)
class ToolTestAdmissionBlocked:
    """Bounded refusal returned after a prepared click cannot dispatch."""

    reason: str
    refreshed_preview: ToolTestAdmissionPreview | None = None
    status: Literal["blocked"] = field(default="blocked", init=False)


@dataclass(frozen=True, slots=True)
class ToolTestAdmissionStale:
    """Bounded stale-preview result that asks the caller to refresh."""

    reason: str
    refreshed_preview: ToolTestAdmissionPreview | None = None
    status: Literal["stale"] = field(default="stale", init=False)


@dataclass(frozen=True, slots=True)
class LocalHubExecutionOutcome:
    """One bounded terminal shared by local-Hub presentation and audit."""

    decision: LocalHubDecision
    status: LocalHubStatus
    error_category: str | None
    final_gate: LocalHubFinalGate
    approval_consumed: bool
    dispatch_started: bool
    provider_terminal: LocalHubProviderTerminal
    duration_ms: int
    result: ToolResult


class LocalHubExecutionCoordinator:
    """Own active local-Hub tasks beyond any one UI caller's lifetime."""

    def __init__(self) -> None:
        self._active: set[tuple[str, str]] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    def active(self, server_key: str, tool_name: str) -> bool:
        """Return whether the exact local tool has an owning task."""
        return (str(server_key), str(tool_name)) in self._active

    def start(
        self,
        key: tuple[str, str],
        owner: Coroutine[Any, Any, None],
    ) -> asyncio.Task[None] | None:
        """Reserve ``key`` synchronously and retain one owning task."""
        normalized = (str(key[0]), str(key[1]))
        if normalized in self._active:
            owner.close()
            return None
        self._active.add(normalized)

        async def _run() -> None:
            try:
                await owner
            finally:
                self._active.discard(normalized)

        task = asyncio.create_task(_run())
        self._tasks.add(task)

        def _consume(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            try:
                completed.exception()
            except BaseException:
                pass

        task.add_done_callback(_consume)
        return task


class OneShotLocalHubApproval:
    """Private, single-invocation Ask callback with immutable bindings."""

    __slots__ = (
        "_argument_digest",
        "_authority_fingerprint",
        "_consumed",
        "_definition_hash",
        "_invocation_id",
        "_lock",
        "_server_key",
        "_tool_name",
        "_binding",
    )

    def __init__(
        self,
        *,
        invocation_id: str,
        server_key: str,
        tool_name: str,
        definition_hash: str,
        authority_fingerprint: str,
        canonical_arguments: bytes,
    ) -> None:
        self._invocation_id = str(invocation_id)
        self._server_key = str(server_key)
        self._tool_name = str(tool_name)
        self._definition_hash = str(definition_hash)
        self._authority_fingerprint = str(authority_fingerprint)
        self._argument_digest = hashlib.sha256(canonical_arguments).digest()
        self._binding = (
            self._invocation_id,
            self._server_key,
            self._tool_name,
            self._definition_hash,
            self._authority_fingerprint,
            self._argument_digest.hex(),
        )
        self._consumed = False
        self._lock = threading.Lock()

    @contextmanager
    def invocation_scope(self):
        """Bind this callback to its one exact provider invocation thread."""
        token = _LOCAL_HUB_APPROVAL_BINDING.set(self._binding)
        try:
            yield
        finally:
            _LOCAL_HUB_APPROVAL_BINDING.reset(token)

    @property
    def consumed(self) -> bool:
        """Return whether this invocation already spent its one approval."""
        with self._lock:
            return self._consumed

    def __call__(self, gates: list[Any]) -> dict[str, str]:
        """Approve exactly one matching pending gate and then fail closed."""
        with self._lock:
            if (
                self._consumed
                or len(gates) != 1
                or _LOCAL_HUB_APPROVAL_BINDING.get() != self._binding
            ):
                return {}
            gate = gates[0]
            if (
                str(getattr(gate, "server_key", "")) != self._server_key
                or str(getattr(gate, "tool_name", "")) != self._tool_name
            ):
                return {}
            try:
                gate_bytes, _gate_arguments = canonicalize_arguments(
                    getattr(gate, "arguments", None)
                )
            except ValueError:
                return {}
            if hashlib.sha256(gate_bytes).digest() != self._argument_digest:
                return {}
            self._consumed = True
            return {self._tool_name: "approve_once"}


def canonicalize_arguments(value: object) -> tuple[bytes, dict[str, Any]]:
    """Validate and canonicalize an exact JSON-object argument payload.

    Args:
        value: Candidate tool arguments.

    Returns:
        Deterministic UTF-8 JSON bytes and an independent dispatch object.

    Raises:
        ValueError: If the value is not a strict JSON object with finite numbers.
    """
    if type(value) is not dict:
        raise ValueError("tool arguments must be a JSON object")
    _validate_json_value(value, path="$", active_containers=set())
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    dispatch = cast(dict[str, Any], json.loads(encoded))
    return encoded, dispatch


def _validate_json_value(
    value: object,
    *,
    path: str,
    active_containers: set[int],
) -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError(f"JSON number at {path} must be finite")
        return
    if value_type is list:
        _validate_json_container(value, path=path, active_containers=active_containers)
        for index, item in enumerate(cast(list[object], value)):
            _validate_json_value(
                item,
                path=f"{path}[{index}]",
                active_containers=active_containers,
            )
        active_containers.remove(id(value))
        return
    if value_type is dict:
        _validate_json_container(value, path=path, active_containers=active_containers)
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise ValueError(f"JSON object key at {path} must be a string")
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                active_containers=active_containers,
            )
        active_containers.remove(id(value))
        return
    raise ValueError(f"value at {path} is not a JSON value")


def _validate_json_container(
    value: object,
    *,
    path: str,
    active_containers: set[int],
) -> None:
    marker = id(value)
    if marker in active_containers:
        raise ValueError(f"JSON value at {path} contains a circular reference")
    active_containers.add(marker)


def authority_fingerprint(authority: DirectoryChain) -> str:
    """Hash the canonical locator and complete root-first identity chain."""
    payload = {
        "canonical_root": os.fspath(authority.canonical_root),
        "identities": [
            {
                "device": identity.device,
                "inode": identity.inode,
                "mode": identity.mode,
                "reparse": identity.reparse,
            }
            for identity in authority.identities
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ToolTestPreviewRegistry:
    """Synchronized, bounded storage for short-lived single-use previews."""

    def __init__(
        self,
        *,
        max_entries: int = 64,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock if clock is not None else lambda: time.monotonic()
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, RegisteredToolTestPreview] = OrderedDict()

    def issue(
        self,
        *,
        server_key: str,
        tool_name: str,
        definition_hash: str,
        rendered_gate: str,
        authority: DirectoryChain | None,
        safe_authority_label: str | None,
    ) -> ToolTestAdmissionPreview:
        """Mint and retain one immutable preview, evicting oldest entries."""
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            while len(self._entries) >= self._max_entries:
                self._entries.popitem(last=False)
            nonce = secrets.token_urlsafe()
            while nonce in self._entries:
                nonce = secrets.token_urlsafe()
            public = ToolTestAdmissionPreview(
                nonce=nonce,
                server_key=server_key,
                tool_name=tool_name,
                definition_hash=definition_hash,
                rendered_gate=rendered_gate,
                authority_fingerprint=(
                    authority_fingerprint(authority) if authority is not None else None
                ),
                safe_authority_label=safe_authority_label,
            )
            self._entries[nonce] = RegisteredToolTestPreview(
                public=public,
                authority=authority,
                expires_at=now + self._ttl_seconds,
            )
            return public

    def consume(self, nonce: str) -> RegisteredToolTestPreview | None:
        """Atomically remove and return one unexpired preview."""
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            return self._entries.pop(nonce, None)

    def revoke(self, nonce: str) -> None:
        """Remove a preview if it is still registered."""
        with self._lock:
            self._entries.pop(nonce, None)

    def clear(self) -> None:
        """Remove every registered preview."""
        with self._lock:
            self._entries.clear()

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            nonce
            for nonce, registered in self._entries.items()
            if registered.expires_at <= now
        ]
        for nonce in expired:
            del self._entries[nonce]
