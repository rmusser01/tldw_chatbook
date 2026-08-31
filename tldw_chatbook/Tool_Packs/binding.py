"""Shared lifecycle coordination and policy identity for Tool profiles."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ProfileMutationResult:
    """Identity returned after one complete profile authority mutation."""

    profile_id: str
    revision: int
    policy_digest: str
    store_generation: str


class ProfileMutationError(ValueError):
    """A complete or profile-scoped policy mutation was rejected."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _canonical_json_value(value: Any) -> Any:
    """Copy frozen or ordinary JSON containers into canonicalizable values."""
    if isinstance(value, Mapping):
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def profile_policy_digest(profile: Mapping[str, Any]) -> str:
    """Hash canonical policy fields while excluding lifecycle metadata.

    The profile kind remains covered because it changes the runtime meaning of
    otherwise identical policy. Lifecycle revision/provenance and the store's
    top-level timestamp are deliberately outside this identity.
    """
    policy: dict[str, Any] = {
        "servers": _canonical_json_value(profile.get("servers", {})),
    }
    if "global_default" in profile:
        policy["global_default"] = _canonical_json_value(profile["global_default"])
    if "profile_kind" in profile:
        policy["profile_kind"] = _canonical_json_value(profile["profile_kind"])
    canonical = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


_LIFECYCLE_LOCK = threading.RLock()
_LEASE_CONDITION = threading.Condition(_LIFECYCLE_LOCK)
_ACTIVE_LEASES: dict[str, int] = {}


class ToolProfileLifecycleCoordinator:
    """Process-wide mutation ordering and exact-profile runtime leases."""

    @contextmanager
    def mutation(self) -> Iterator[None]:
        """Serialize lifecycle changes before any permission-store fence."""
        with _LIFECYCLE_LOCK:
            yield

    @contextmanager
    def lease(self, profile_id: str) -> Iterator[None]:
        """Hold one runtime lease for the exact captured profile id."""
        if not profile_id:
            raise ValueError("profile_id must not be empty")
        with _LEASE_CONDITION:
            _ACTIVE_LEASES[profile_id] = _ACTIVE_LEASES.get(profile_id, 0) + 1
        try:
            yield
        finally:
            with _LEASE_CONDITION:
                remaining = _ACTIVE_LEASES.get(profile_id, 0) - 1
                if remaining > 0:
                    _ACTIVE_LEASES[profile_id] = remaining
                else:
                    _ACTIVE_LEASES.pop(profile_id, None)
                _LEASE_CONDITION.notify_all()

    def active_lease_count(self, profile_id: str) -> int:
        """Return the current process-wide lease count for ``profile_id``."""
        with _LEASE_CONDITION:
            return _ACTIVE_LEASES.get(profile_id, 0)
