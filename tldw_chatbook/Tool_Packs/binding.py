"""Shared lifecycle coordination and policy identity for Tool profiles."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from typing import Any, Iterator, Literal, Mapping

from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventorySnapshot,
    capture_v1_inventory,
    thaw_hub_tool,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


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


BindingAction = Literal["create", "set", "replace"]
_TOKEN_TTL = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class ToolProfileBindingSummary:
    """Current policy posture shown before a first imported-profile bind."""

    global_fallback: str
    builtin_fallback: str
    allow_server_fallbacks: tuple[str, ...]
    stored_exact_allows: tuple[tuple[str, str], ...]
    effective_allows: tuple[tuple[str, str], ...]
    unavailable_allows: tuple[tuple[str, str], ...]
    downgraded_allows: tuple[tuple[str, str], ...]
    high_risk_allows: tuple[tuple[str, str], ...]
    allow_count: int
    ask_count: int
    deny_count: int
    inventory_digest: str


@dataclass(frozen=True, slots=True)
class ToolProfileBindingReview:
    """Immutable current authority accepted by one explicit confirmation."""

    workspace_id: str
    action: BindingAction
    intended_defaults_digest: str
    profile_id: str
    policy_digest: str
    revision: int
    expires_at: datetime
    summary: ToolProfileBindingSummary


class ToolProfileConfirmationRequired(ToolPackError):
    """A first imported-profile bind needs a fresh explicit confirmation."""

    def __init__(self) -> None:
        super().__init__("bind", "confirmation_required")


@dataclass(frozen=True, slots=True)
class _ReviewRecord:
    review: ToolProfileBindingReview
    current_defaults_digest: str


@dataclass(frozen=True, slots=True)
class _TokenRecord:
    review: ToolProfileBindingReview
    current_defaults_digest: str


def _canonical_json_value(value: Any) -> Any:
    """Copy frozen or ordinary JSON containers into canonicalizable values."""
    if isinstance(value, Mapping):
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def _defaults_digest(defaults: WorkspaceAssistantDefaults | None) -> str:
    payload: object
    if defaults is None:
        payload = None
    else:
        payload = {
            "assistant_kind": defaults.assistant_kind,
            "assistant_id": defaults.assistant_id,
            "persona_memory_mode": defaults.persona_memory_mode,
            "voice": defaults.voice,
            "style": defaults.style,
            "tool_policy_profile_id": defaults.tool_policy_profile_id,
        }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
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


class ToolProfileBindingGuard:
    """Confirm first imported-profile binding at the registry write boundary."""

    def __init__(
        self,
        *,
        permission_store: object,
        inventory: object,
        workspace_defaults_reader: Callable[[str], WorkspaceAssistantDefaults | None],
        lifecycle: ToolProfileLifecycleCoordinator | None = None,
        now: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._permission_store = permission_store
        self._inventory = inventory
        self._workspace_defaults_reader = workspace_defaults_reader
        self.lifecycle = lifecycle or ToolProfileLifecycleCoordinator()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._token_factory = token_factory or (lambda: token_urlsafe(32))
        self._token_lock = threading.Lock()
        self._reviews: dict[int, _ReviewRecord] = {}
        self._tokens: dict[str, _TokenRecord] = {}

    def review(
        self,
        workspace_id: str,
        intended_defaults: WorkspaceAssistantDefaults,
        *,
        action: str,
    ) -> ToolProfileBindingReview:
        """Capture current policy and inventory for an imported first bind."""
        if (
            type(workspace_id) is not str
            or not workspace_id
            or workspace_id != workspace_id.strip()
            or type(intended_defaults) is not WorkspaceAssistantDefaults
            or action not in {"create", "set", "replace"}
        ):
            raise ToolPackError("bind", "confirmation_invalid")
        profile_id = intended_defaults.tool_policy_profile_id
        if type(profile_id) is not str or not profile_id:
            raise ToolPackError("bind", "confirmation_invalid")

        with self.lifecycle.mutation():
            snapshot, profile, lifecycle = self._imported_authority(profile_id)
            if lifecycle["first_bind_confirmation_required"] is not True:
                raise ToolPackError("bind", "confirmation_invalid")
            inventory = self._capture_inventory()
            now = self._utc_now()
            review = ToolProfileBindingReview(
                workspace_id=workspace_id,
                action=action,  # type: ignore[arg-type]
                intended_defaults_digest=_defaults_digest(intended_defaults),
                profile_id=profile_id,
                policy_digest=lifecycle["policy_digest"],
                revision=lifecycle["revision"],
                expires_at=now + _TOKEN_TTL,
                summary=self._summary(snapshot.payload, profile, profile_id, inventory),
            )
            current_digest = _defaults_digest(
                self._workspace_defaults_reader(workspace_id)
            )
        with self._token_lock:
            self._reviews[id(review)] = _ReviewRecord(review, current_digest)
        return review

    def confirm(self, review: ToolProfileBindingReview) -> str:
        """Issue one opaque process-local token for one genuine live review."""
        if type(review) is not ToolProfileBindingReview:
            raise ToolPackError("bind", "confirmation_invalid")
        now = self._utc_now()
        with self._token_lock:
            record = self._reviews.pop(id(review), None)
            if record is None or record.review is not review:
                raise ToolPackError("bind", "confirmation_invalid")
            if now >= review.expires_at:
                raise ToolPackError("bind", "confirmation_expired")
            token = self._token_factory()
            if type(token) is not str or not token or token in self._tokens:
                raise ToolPackError("bind", "confirmation_invalid")
            self._tokens[token] = _TokenRecord(review, record.current_defaults_digest)
        return token

    @contextmanager
    def mutation_scope(
        self,
        *,
        action: str,
        workspace_id: str,
        current_defaults: WorkspaceAssistantDefaults | None,
        intended_defaults: WorkspaceAssistantDefaults | None,
        confirmation_token: str | None,
    ) -> Iterator[None]:
        """Hold lifecycle and store locks through the registry SQLite commit."""
        needs_marker_clear = False
        expected_revision = 0
        expected_policy_digest = ""
        with self.lifecycle.mutation():
            with self._permission_store.mutation_fence():
                profile_id = (
                    intended_defaults.tool_policy_profile_id
                    if intended_defaults is not None
                    else None
                )
                if profile_id is not None:
                    snapshot = self._strict_snapshot()
                    profile = self._profile(snapshot.payload, profile_id)
                    disposition = self._disposition(profile)
                    if disposition in {"invalid", "tombstone"}:
                        raise ToolPackError("bind", "lifecycle_invalid")

                    if disposition == "imported":
                        lifecycle = profile["tool_pack_lifecycle"]
                        self._validate_policy_identity(profile, lifecycle)
                        if lifecycle["first_bind_confirmation_required"] is True:
                            self._consume_token(
                                confirmation_token,
                                action=action,
                                workspace_id=workspace_id,
                                current_defaults=current_defaults,
                                intended_defaults=intended_defaults,
                                profile=profile,
                                snapshot_payload=snapshot.payload,
                            )
                            needs_marker_clear = True
                            expected_revision = lifecycle["revision"]
                            expected_policy_digest = lifecycle["policy_digest"]
                        elif confirmation_token is not None:
                            raise ToolPackError("bind", "confirmation_invalid")
                    elif confirmation_token is not None:
                        raise ToolPackError("bind", "confirmation_invalid")
                elif confirmation_token is not None:
                    raise ToolPackError("bind", "confirmation_invalid")

                self._validate_registry_state(
                    action=action,
                    workspace_id=workspace_id,
                    current_defaults=current_defaults,
                    intended_defaults=intended_defaults,
                )

                try:
                    yield
                except BaseException as exc:
                    if not needs_marker_clear:
                        raise
                    try:
                        proven = (
                            self._workspace_defaults_reader(workspace_id)
                            == intended_defaults
                        )
                    except Exception:
                        proven = False
                    if proven:
                        self._best_effort_clear_marker(
                            profile_id,
                            expected_revision=expected_revision,
                            expected_policy_digest=expected_policy_digest,
                        )
                    raise ToolPackError("bind", "binding_uncertain") from exc

                if needs_marker_clear:
                    self._best_effort_clear_marker(
                        profile_id,
                        expected_revision=expected_revision,
                        expected_policy_digest=expected_policy_digest,
                    )

    def _validate_registry_state(
        self,
        *,
        action: str,
        workspace_id: str,
        current_defaults: WorkspaceAssistantDefaults | None,
        intended_defaults: WorkspaceAssistantDefaults | None,
    ) -> None:
        if action not in {"create", "set", "replace", "clear"}:
            raise ToolPackError("bind", "confirmation_invalid")
        if type(workspace_id) is not str or not workspace_id:
            raise ToolPackError("bind", "confirmation_invalid")
        if action == "clear" and intended_defaults is not None:
            raise ToolPackError("bind", "confirmation_invalid")
        if action in {"create", "set"} and current_defaults is not None:
            raise ToolPackError("bind", "confirmation_stale")
        if action == "replace" and current_defaults is None:
            raise ToolPackError("bind", "confirmation_stale")
        if self._workspace_defaults_reader(workspace_id) != current_defaults:
            raise ToolPackError("bind", "confirmation_stale")

    def _consume_token(
        self,
        token: str | None,
        *,
        action: str,
        workspace_id: str,
        current_defaults: WorkspaceAssistantDefaults | None,
        intended_defaults: WorkspaceAssistantDefaults | None,
        profile: Mapping[str, Any],
        snapshot_payload: Mapping[str, Any],
    ) -> None:
        if token is None:
            raise ToolProfileConfirmationRequired()
        with self._token_lock:
            record = self._tokens.pop(token, None)
            if record is None:
                raise ToolPackError("bind", "confirmation_invalid")
            review = record.review
            if self._utc_now() >= review.expires_at:
                raise ToolPackError("bind", "confirmation_expired")
            lifecycle = profile["tool_pack_lifecycle"]
            inventory = self._capture_inventory()
            current_summary = self._summary(
                snapshot_payload, profile, review.profile_id, inventory
            )
            stale = (
                review.workspace_id != workspace_id
                or review.action != action
                or review.intended_defaults_digest
                != _defaults_digest(intended_defaults)
                or review.profile_id
                != getattr(intended_defaults, "tool_policy_profile_id", None)
                or review.policy_digest != lifecycle["policy_digest"]
                or review.revision != lifecycle["revision"]
                or review.summary != current_summary
                or record.current_defaults_digest != _defaults_digest(current_defaults)
            )
            if stale:
                raise ToolPackError("bind", "confirmation_stale")

    def _best_effort_clear_marker(
        self,
        profile_id: str | None,
        *,
        expected_revision: int,
        expected_policy_digest: str,
    ) -> None:
        if profile_id is None:
            return
        try:
            snapshot, _profile, lifecycle = self._imported_authority(profile_id)
            if (
                lifecycle["first_bind_confirmation_required"] is not True
                or lifecycle["revision"] != expected_revision
                or lifecycle["policy_digest"] != expected_policy_digest
            ):
                return
            payload = _plain_json(snapshot.payload)
            mutable_profile = payload["profiles"][profile_id]  # type: ignore[index]
            mutable_lifecycle = mutable_profile["tool_pack_lifecycle"]
            mutable_lifecycle["first_bind_confirmation_required"] = False
            mutable_lifecycle["revision"] += 1
            self._permission_store.save(
                payload, expected_generation=snapshot.generation
            )
        except Exception:
            return

    def _imported_authority(
        self, profile_id: str
    ) -> tuple[object, Mapping[str, Any], Mapping[str, Any]]:
        snapshot = self._strict_snapshot()
        profile = self._profile(snapshot.payload, profile_id)
        if self._disposition(profile) != "imported":
            raise ToolPackError("bind", "lifecycle_invalid")
        lifecycle = profile["tool_pack_lifecycle"]
        self._validate_policy_identity(profile, lifecycle)
        return snapshot, profile, lifecycle

    def _strict_snapshot(self):
        try:
            snapshot = self._permission_store.read_snapshot_strict()
        except Exception:
            raise ToolPackError("bind", "lifecycle_invalid") from None
        if not isinstance(getattr(snapshot, "payload", None), Mapping):
            raise ToolPackError("bind", "lifecycle_invalid")
        return snapshot

    @staticmethod
    def _profile(
        payload: Mapping[str, Any], profile_id: str | None
    ) -> Mapping[str, Any] | None:
        profiles = payload.get("profiles")
        if profile_id is None or not isinstance(profiles, Mapping):
            return None
        profile = profiles.get(profile_id)
        return profile if isinstance(profile, Mapping) else None

    @staticmethod
    def _disposition(profile: Mapping[str, Any] | None) -> str:
        if profile is None:
            return "legacy"
        from tldw_chatbook.MCP.permission_store import profile_lifecycle_disposition

        return profile_lifecycle_disposition(profile)

    @staticmethod
    def _validate_policy_identity(
        profile: Mapping[str, Any], lifecycle: Mapping[str, Any]
    ) -> None:
        if lifecycle.get("policy_digest") != profile_policy_digest(profile):
            raise ToolPackError("bind", "lifecycle_invalid")

    def _capture_inventory(self) -> PermissionInventorySnapshot:
        try:
            inventory = capture_v1_inventory(self._inventory)  # type: ignore[arg-type]
        except Exception:
            raise ToolPackError("bind", "confirmation_invalid") from None
        if type(inventory) is not PermissionInventorySnapshot:
            raise ToolPackError("bind", "confirmation_invalid")
        return inventory

    def _summary(
        self,
        payload: Mapping[str, Any],
        profile: Mapping[str, Any],
        profile_id: str,
        inventory: PermissionInventorySnapshot,
    ) -> ToolProfileBindingSummary:
        from tldw_chatbook.MCP.permission_store import (
            BUILTIN_HIGH_RISK_TAGS,
            BUILTIN_TOOL_SERVER_KEY,
            HIGH_RISK_TAGS,
            GatedToolRef,
            resolve_builtin_state,
            resolve_effective_state,
        )

        servers = profile.get("servers")
        servers = servers if isinstance(servers, Mapping) else {}
        builtin = servers.get(BUILTIN_TOOL_SERVER_KEY)
        builtin = builtin if isinstance(builtin, Mapping) else {}
        stored: set[tuple[str, str]] = set()
        allow_server_fallbacks: list[str] = []
        for server_key, server in servers.items():
            if not isinstance(server_key, str) or not isinstance(server, Mapping):
                continue
            if server.get("default") == "allow":
                allow_server_fallbacks.append(server_key)
            tools = server.get("tools")
            if not isinstance(tools, Mapping):
                continue
            stored.update(
                (server_key, tool_name)
                for tool_name, entry in tools.items()
                if isinstance(tool_name, str)
                and isinstance(entry, Mapping)
                and entry.get("state") == "allow"
            )

        live = {
            (item.tool.server_key, item.tool.name): item for item in inventory.tools
        }
        unavailable = stored - set(live)
        effective: set[tuple[str, str]] = set()
        downgraded: set[tuple[str, str]] = set()
        high_risk: set[tuple[str, str]] = set()
        counts = {"allow": 0, "ask": 0, "deny": 0}
        for identity, item in live.items():
            tool = thaw_hub_tool(item.tool)
            if item.authority == "builtin":
                result = resolve_builtin_state(
                    payload,  # type: ignore[arg-type]
                    GatedToolRef(
                        tool.server_key,
                        tool.name,
                        tool.description,
                        tool.input_schema,
                        tool.tags,
                    ),
                    profile_id=profile_id,
                )
                risk_tags = BUILTIN_HIGH_RISK_TAGS
            else:
                result = resolve_effective_state(
                    payload,
                    tool,
                    profile_id=profile_id,  # type: ignore[arg-type]
                )
                risk_tags = HIGH_RISK_TAGS
            counts[result.state] += 1
            if result.state == "allow":
                effective.add(identity)
                if set(tool.tags) & risk_tags:
                    high_risk.add(identity)
            elif identity in stored:
                downgraded.add(identity)

        return ToolProfileBindingSummary(
            global_fallback=str(profile.get("global_default") or "ask"),
            builtin_fallback=str(builtin.get("default") or "allow"),
            allow_server_fallbacks=tuple(sorted(allow_server_fallbacks)),
            stored_exact_allows=tuple(sorted(stored)),
            effective_allows=tuple(sorted(effective)),
            unavailable_allows=tuple(sorted(unavailable)),
            downgraded_allows=tuple(sorted(downgraded)),
            high_risk_allows=tuple(sorted(high_risk)),
            allow_count=counts["allow"],
            ask_count=counts["ask"],
            deny_count=counts["deny"],
            inventory_digest=inventory.digest,
        )

    def _utc_now(self) -> datetime:
        now = self._now()
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise ToolPackError("bind", "confirmation_invalid")
        return now.astimezone(timezone.utc)
