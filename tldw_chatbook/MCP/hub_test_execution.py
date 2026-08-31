"""Immutable admission primitives for operator-initiated Hub tool tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from tldw_chatbook.Utils.filesystem_identity import DirectoryChain


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
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
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
        now = time.monotonic()
        with self._lock:
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
        now = time.monotonic()
        with self._lock:
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
