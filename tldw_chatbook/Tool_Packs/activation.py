"""Safe compilation and unbound installation of reviewed Tool Packs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any

from tldw_chatbook.MCP.permission_store import (
    HASH_FREE_SERVER_KEYS,
    PermissionStoreSnapshot,
    definition_hash,
)
from tldw_chatbook.Tool_Packs.binding import (
    ProfileMutationError,
    ProfileMutationResult,
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryRegistry,
    PermissionInventorySnapshot,
    capture_v1_inventory,
    thaw_hub_tool,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.importer import ToolPackImportReview
from tldw_chatbook.Tool_Packs.receipt_store import (
    RECEIPT_SCHEMA,
    ReceiptHandle,
    ToolPackReceipt,
    ToolPackReceiptStore,
)


_MAX_PROFILES = 128
_MAX_STORE_BYTES = 8 * 1024 * 1024


def _fail(category: str) -> ToolPackError:
    return ToolPackError("import", category)


def _utc_seconds(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise _fail("review_stale")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class InstalledToolProfile:
    """Stable identity of one installed imported permission profile."""

    profile_id: str
    policy_digest: str
    revision: int
    receipt_id: str


@dataclass(frozen=True, slots=True)
class ToolPackActivationResult:
    """Known activation outcome after install or strict reconciliation."""

    installed: InstalledToolProfile
    store_generation: str
    reconciled: bool = False


def compile_imported_profile(
    review: ToolPackImportReview,
    inventory: PermissionInventorySnapshot,
    *,
    receipt: ReceiptHandle,
    imported_at: datetime,
) -> dict[str, Any]:
    """Compile one reviewed snapshot into a fail-closed runtime profile."""
    if (
        type(review) is not ToolPackImportReview
        or type(inventory) is not PermissionInventorySnapshot
        or type(receipt) is not ReceiptHandle
        or review.inventory_digest != inventory.digest
    ):
        raise _fail("review_stale")

    fallbacks = {
        (fallback.authority, fallback.server_key): fallback.state
        for fallback in review.fallbacks
    }
    global_default = fallbacks.get(("mcp", "*"))
    builtin_default = fallbacks.get(("builtin", "agent:builtin"))
    if global_default not in {"ask", "deny"} or builtin_default not in {
        "ask",
        "deny",
    }:
        raise _fail("review_stale")

    servers: dict[str, dict[str, Any]] = {"agent:builtin": {"default": builtin_default}}
    for (authority, server_key), state in fallbacks.items():
        if (authority, server_key) in {
            ("mcp", "*"),
            ("builtin", "agent:builtin"),
        }:
            continue
        if authority != "mcp" or state not in {"ask", "deny"}:
            raise _fail("review_stale")
        servers[server_key] = {"default": state}

    destination = {item.identity: item for item in inventory.tools}

    def fallback_for(authority: str, server_key: str) -> str:
        if authority == "builtin":
            return builtin_default
        return servers.get(server_key, {}).get("default", global_default)

    def store_rule(
        authority: str,
        server_key: str,
        tool_name: str,
        state: str,
        *,
        runtime_tool: object | None = None,
    ) -> None:
        if state == fallback_for(authority, server_key):
            return
        server = servers.setdefault(server_key, {})
        entry: dict[str, Any] = {"state": state}
        if state == "allow" and server_key not in HASH_FREE_SERVER_KEYS:
            if runtime_tool is None:
                raise _fail("review_stale")
            tool = thaw_hub_tool(runtime_tool)  # type: ignore[arg-type]
            entry["definition_hash"] = definition_hash(
                tool.description, tool.input_schema
            )
        server.setdefault("tools", {})[tool_name] = entry

    for mapped in review.matched:
        item = destination.get(mapped.destination_identity)
        if (
            item is None
            or item.contract_sha256 != mapped.destination_contract_sha256
            or item.contract_sha256 != mapped.source_rule.contract_sha256
        ):
            raise _fail("review_stale")
        store_rule(
            mapped.authority,
            mapped.server_key,
            mapped.tool_name,
            mapped.state,
            runtime_tool=item.tool,
        )

    for rule in review.pending_denies:
        if rule.state != "deny":
            raise _fail("review_stale")
        store_rule(rule.authority, rule.server_key, rule.tool_name, "deny")

    profile: dict[str, Any] = {
        "global_default": global_default,
        "servers": servers,
        "profile_kind": "tool_pack_imported",
    }
    profile["tool_pack_lifecycle"] = {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": "imported",
        "pack_digest": review.content_digest,
        "imported_at": _utc_seconds(imported_at),
        "first_bind_confirmation_required": True,
        "receipt_id": receipt.receipt_id,
        "receipt_digest": receipt.digest,
        "counts": {
            "matched": len(review.matched),
            "omitted": len(review.omitted_allow_ask),
            "pending_deny": len(review.pending_denies),
        },
        "policy_digest": profile_policy_digest(profile),
        "revision": 1,
    }
    return profile


def _review_evidence(review: ToolPackImportReview) -> tuple[object, ...]:
    return (
        review.archive_path,
        review.archive_sha256,
        review.manifest_sha256,
        review.payload_sha256,
        review.destination_id,
        review.inventory_digest,
        review.mappings,
        review.matched,
        review.changed,
        review.missing,
        review.pending_denies,
        review.omitted_allow_ask,
        review.content_digest,
        review.display_name,
        review.producer,
        review.fallbacks,
    )


def _identity(rule: object) -> tuple[str, str, str]:
    return (
        rule.authority,  # type: ignore[attr-defined,no-any-return]
        rule.server_key,  # type: ignore[attr-defined,no-any-return]
        rule.tool_name,  # type: ignore[attr-defined,no-any-return]
    )


def _receipt_for_review(
    review: ToolPackImportReview, *, imported_at: datetime
) -> ToolPackReceipt:
    return ToolPackReceipt(
        schema=RECEIPT_SCHEMA,
        kind="import",
        profile_id=review.destination_id,
        pack_digest=review.content_digest,
        archive_digest=review.archive_sha256,
        producer=review.producer,
        imported_at=_utc_seconds(imported_at),
        reviewed_mappings=tuple(
            sorted(
                (mapping.source_server_key, mapping.destination_server_key)
                for mapping in review.mappings
            )
        ),
        matched=tuple(sorted(mapped.destination_identity for mapped in review.matched)),
        changed=tuple(sorted(_identity(rule) for rule in review.changed)),
        missing=tuple(sorted(_identity(rule) for rule in review.missing)),
        pending_deny=tuple(sorted(_identity(rule) for rule in review.pending_denies)),
        omitted=tuple(sorted(_identity(rule) for rule in review.omitted_allow_ask)),
    )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


class ToolPackActivationService:
    """Revalidate and install reviewed Tool Packs as unbound profiles."""

    def __init__(
        self,
        *,
        permission_store: object,
        inventory: PermissionInventoryRegistry,
        importer: object,
        reference_checker: object,
        receipt_store: ToolPackReceiptStore,
        lifecycle: ToolProfileLifecycleCoordinator | None = None,
        now: Callable[[], datetime] | None = None,
        max_profiles: int = _MAX_PROFILES,
        max_store_bytes: int = _MAX_STORE_BYTES,
    ) -> None:
        self._permission_store = permission_store
        self._inventory = inventory
        self._importer = importer
        self._reference_checker = reference_checker
        self._receipt_store = receipt_store
        self._lifecycle = lifecycle or ToolProfileLifecycleCoordinator()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._max_profiles = max_profiles
        self._max_store_bytes = max_store_bytes
        self._live_lock = threading.Lock()
        self._live_receipts: set[str] = set()

    def install(self, review: ToolPackImportReview) -> ToolPackActivationResult:
        """Revalidate and install exactly the reviewed id as an unbound profile."""
        now = self._now()
        if (
            type(review) is not ToolPackImportReview
            or type(now) is not datetime
            or now.tzinfo is None
            or now.utcoffset() is None
            or now >= review.expires_at
        ):
            raise _fail("review_stale")

        try:
            fresh = self._importer.inspect_archive(
                review.archive_path,
                destination_id=review.destination_id,
                mappings=review.mappings,
            )
        except ToolPackError:
            raise
        except Exception:
            raise _fail("activation_failed") from None
        if type(fresh) is not ToolPackImportReview:
            raise _fail("review_stale")
        if fresh.store_generation != review.store_generation:
            raise _fail("store_changed")
        if _review_evidence(fresh) != _review_evidence(review):
            raise _fail("review_stale")
        try:
            inventory = capture_v1_inventory(self._inventory)
        except Exception:
            raise _fail("inventory_invalid") from None
        if (
            type(inventory) is not PermissionInventorySnapshot
            or inventory.digest != review.inventory_digest
        ):
            raise _fail("review_stale")

        receipt_bytes = _receipt_for_review(review, imported_at=now).to_bytes()
        with self._receipt_store.reserve(len(receipt_bytes)) as reservation:
            receipt = reservation.commit(receipt_bytes)
        with self._live_lock:
            self._live_receipts.add(receipt.receipt_id)
        try:
            return self._install_authority(review, inventory, receipt, now)
        finally:
            with self._live_lock:
                self._live_receipts.discard(receipt.receipt_id)

    def live_receipt_ids(self) -> frozenset[str]:
        """Return receipt ids currently owned by in-flight activation commits."""
        with self._live_lock:
            return frozenset(self._live_receipts)

    def _install_authority(
        self,
        review: ToolPackImportReview,
        inventory: PermissionInventorySnapshot,
        receipt: ReceiptHandle,
        imported_at: datetime,
    ) -> ToolPackActivationResult:
        with self._lifecycle.mutation():
            with self._permission_store.mutation_fence():
                try:
                    commit_now = self._now()
                    _utc_seconds(commit_now)
                except Exception:
                    raise _fail("review_stale") from None
                if commit_now >= review.expires_at:
                    raise _fail("review_stale")
                snapshot = self._strict_snapshot(category="store_invalid")
                if snapshot.generation != review.store_generation:
                    raise _fail("store_changed")
                profiles = snapshot.payload.get("profiles")
                if not isinstance(profiles, Mapping):
                    raise _fail("store_invalid")
                folded = review.destination_id.casefold()
                if any(
                    type(profile_id) is str and profile_id.casefold() == folded
                    for profile_id in profiles
                ):
                    raise _fail("destination_referenced")
                if self._references_profile(review.destination_id):
                    raise _fail("destination_referenced")

                profile = compile_imported_profile(
                    review,
                    inventory,
                    receipt=receipt,
                    imported_at=imported_at,
                )
                try:
                    mutation = self._permission_store.install_profile_if_absent(
                        review.destination_id,
                        profile,
                        expected_generation=snapshot.generation,
                        max_profiles=self._max_profiles,
                        max_store_bytes=self._max_store_bytes,
                    )
                except Exception as error:
                    return self._reconcile_install(
                        review.destination_id, profile, receipt, error
                    )
        return self._result(mutation, receipt, reconciled=False)

    def _references_profile(self, profile_id: str) -> bool:
        try:
            if callable(self._reference_checker):
                result = self._reference_checker(profile_id)
            else:
                result = self._reference_checker.references_profile(
                    profile_id, include_archived=True
                )
        except Exception:
            raise _fail("destination_referenced") from None
        if type(result) is not bool:
            raise _fail("destination_referenced")
        return result

    def _strict_snapshot(self, *, category: str) -> PermissionStoreSnapshot:
        try:
            snapshot = self._permission_store.read_snapshot_strict()
        except Exception:
            raise _fail(category) from None
        if type(snapshot) is not PermissionStoreSnapshot:
            raise _fail(category)
        return snapshot

    def _reconcile_install(
        self,
        profile_id: str,
        expected_profile: dict[str, Any],
        receipt: ReceiptHandle,
        error: Exception,
    ) -> ToolPackActivationResult:
        snapshot = self._strict_snapshot(category="activation_uncertain")
        profiles = snapshot.payload.get("profiles")
        if not isinstance(profiles, Mapping):
            raise _fail("activation_uncertain")
        actual = profiles.get(profile_id)
        if isinstance(actual, Mapping) and _plain_json(actual) == expected_profile:
            lifecycle = expected_profile["tool_pack_lifecycle"]
            return ToolPackActivationResult(
                InstalledToolProfile(
                    profile_id,
                    lifecycle["policy_digest"],
                    lifecycle["revision"],
                    receipt.receipt_id,
                ),
                snapshot.generation,
                True,
            )
        if actual is not None:
            raise _fail("activation_uncertain")
        if isinstance(error, ProfileMutationError):
            if error.category in {"profile_limit", "store_bytes_limit"}:
                raise _fail("capacity_exceeded") from None
            if error.category == "stale_store":
                raise _fail("store_changed") from None
            if error.category in {"profile_exists", "profile_id_collision"}:
                raise _fail("destination_referenced") from None
        raise _fail("activation_failed") from None

    @staticmethod
    def _result(
        mutation: ProfileMutationResult,
        receipt: ReceiptHandle,
        *,
        reconciled: bool,
    ) -> ToolPackActivationResult:
        return ToolPackActivationResult(
            InstalledToolProfile(
                mutation.profile_id,
                mutation.policy_digest,
                mutation.revision,
                receipt.receipt_id,
            ),
            mutation.store_generation,
            reconciled,
        )


__all__ = [
    "InstalledToolProfile",
    "ToolPackActivationResult",
    "ToolPackActivationService",
    "compile_imported_profile",
]
