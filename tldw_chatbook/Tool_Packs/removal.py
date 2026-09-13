"""Fail-closed removal of imported Tool profiles through permanent tombstones."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    PermissionStoreSnapshot,
    profile_lifecycle_disposition,
)
from tldw_chatbook.Tool_Packs.binding import (
    ProfileMutationError,
    ProfileMutationResult,
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.receipt_store import (
    ReceiptHandle,
    ToolPackReceiptStore,
)


_MAX_STORE_BYTES = 8 * 1024 * 1024


def _fail(category: str) -> ToolPackError:
    return ToolPackError("remove", category)


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RemovedToolProfile:
    """Stable identity of one permanent Tool profile tombstone."""

    profile_id: str
    profile_kind: str
    revision: int
    policy_digest: str
    receipt_id: str
    receipt_digest: str


@dataclass(frozen=True, slots=True)
class ToolProfileRemovalResult:
    """Known removal outcome after replacement or strict reconciliation."""

    tombstone: RemovedToolProfile
    store_generation: str
    reconciled: bool = False


class ToolProfileRemovalService:
    """Replace eligible imported Tool profiles with permanent Deny tombstones."""

    def __init__(
        self,
        *,
        permission_store: object,
        receipt_store: ToolPackReceiptStore,
        reference_checker: object,
        lifecycle: ToolProfileLifecycleCoordinator | None = None,
        max_store_bytes: int = _MAX_STORE_BYTES,
    ) -> None:
        if type(max_store_bytes) is not int or max_store_bytes <= 0:
            raise _fail("non_removable")
        self._permission_store = permission_store
        self._receipt_store = receipt_store
        self._reference_checker = reference_checker
        self._lifecycle = lifecycle or ToolProfileLifecycleCoordinator()
        self._max_store_bytes = max_store_bytes

    def remove(
        self, profile_id: str, *, expected_revision: int
    ) -> ToolProfileRemovalResult:
        """Replace one currently unreferenced imported profile with a tombstone."""
        if (
            type(profile_id) is not str
            or not profile_id
            or profile_id != profile_id.strip()
            or profile_id == "default"
            or profile_id.startswith("ws-")
            or type(expected_revision) is not int
            or isinstance(expected_revision, bool)
            or expected_revision <= 0
        ):
            raise _fail("non_removable")

        with self._lifecycle.mutation():
            with self._permission_store.mutation_fence():
                snapshot = self._strict_snapshot("non_removable")
                profile = self._current_imported(snapshot, profile_id)
                lifecycle = profile["tool_pack_lifecycle"]
                if lifecycle["revision"] != expected_revision:
                    raise _fail("stale")
                if lifecycle["policy_digest"] != profile_policy_digest(profile):
                    raise _fail("non_removable")
                if self._lifecycle.active_lease_count(profile_id):
                    raise _fail("in_use")
                if self._references_profile(profile_id):
                    raise _fail("referenced")

                compact = self._stage_compact_receipt(profile_id, lifecycle)
                tombstone = self._build_tombstone(profile, compact)
                try:
                    mutation = self._permission_store.replace_profile_with_tombstone(
                        profile_id,
                        tombstone,
                        expected_revision=expected_revision,
                        expected_generation=snapshot.generation,
                        expected_profile_digest=lifecycle["policy_digest"],
                        max_store_bytes=self._max_store_bytes,
                    )
                except Exception as error:
                    return self._reconcile_replacement(
                        profile_id,
                        prior_profile=profile,
                        tombstone=tombstone,
                        compact=compact,
                        error=error,
                    )
        return self._result(mutation, compact, reconciled=False)

    def _strict_snapshot(self, category: str) -> PermissionStoreSnapshot:
        try:
            snapshot = self._permission_store.read_snapshot_strict()
        except Exception:
            raise _fail(category) from None
        if type(snapshot) is not PermissionStoreSnapshot:
            raise _fail(category)
        return snapshot

    @staticmethod
    def _current_imported(
        snapshot: PermissionStoreSnapshot, profile_id: str
    ) -> Mapping[str, Any]:
        profiles = snapshot.payload.get("profiles")
        profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
        if not isinstance(profile, Mapping):
            raise _fail("non_removable")
        if profile_lifecycle_disposition(profile) != "imported":
            raise _fail("non_removable")
        return profile

    def _references_profile(self, profile_id: str) -> bool:
        try:
            if callable(self._reference_checker):
                result = self._reference_checker(profile_id)
            else:
                result = self._reference_checker.references_profile(
                    profile_id, include_archived=True
                )
        except Exception:
            raise _fail("referenced") from None
        if type(result) is not bool:
            raise _fail("referenced")
        return result

    def _stage_compact_receipt(
        self, profile_id: str, lifecycle: Mapping[str, Any]
    ) -> ReceiptHandle:
        try:
            source = self._receipt_store.read(
                lifecycle["receipt_id"],
                expected_digest=lifecycle["receipt_digest"],
            ).handle
            return self._receipt_store.write_compact_tombstone(
                source, profile_id=profile_id
            )
        except Exception:
            raise _fail("non_removable") from None

    def _build_tombstone(
        self, profile: Mapping[str, Any], compact: ReceiptHandle
    ) -> dict[str, Any]:
        prior = profile["tool_pack_lifecycle"]
        try:
            compact_receipt = self._receipt_store.read(
                compact.receipt_id, expected_digest=compact.digest
            ).receipt
            removed_at = compact_receipt.removed_at
        except Exception:
            raise _fail("non_removable") from None
        tombstone: dict[str, Any] = {
            "global_default": "deny",
            "servers": {BUILTIN_TOOL_SERVER_KEY: {"default": "deny"}},
            "profile_kind": "tool_pack_tombstone",
            "tool_pack_lifecycle": {
                "schema": "tldw.tool-pack-lifecycle/v1",
                "origin": "tombstone",
                "pack_digest": prior["pack_digest"],
                "imported_at": prior["imported_at"],
                "removed_at": removed_at,
                "first_bind_confirmation_required": False,
                "receipt_id": compact.receipt_id,
                "receipt_digest": compact.digest,
                "policy_digest": "0" * 64,
                "revision": prior["revision"] + 1,
            },
        }
        tombstone["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(
            tombstone
        )
        return tombstone

    def _reconcile_replacement(
        self,
        profile_id: str,
        *,
        prior_profile: Mapping[str, Any],
        tombstone: dict[str, Any],
        compact: ReceiptHandle,
        error: Exception,
    ) -> ToolProfileRemovalResult:
        snapshot = self._strict_snapshot("outcome_uncertain")
        profiles = snapshot.payload.get("profiles")
        actual = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
        if isinstance(actual, Mapping) and _plain_json(actual) == tombstone:
            lifecycle = tombstone["tool_pack_lifecycle"]
            mutation = ProfileMutationResult(
                profile_id,
                lifecycle["revision"],
                lifecycle["policy_digest"],
                snapshot.generation,
            )
            return self._result(mutation, compact, reconciled=True)
        if isinstance(actual, Mapping) and _plain_json(actual) == _plain_json(
            prior_profile
        ):
            if isinstance(error, ProfileMutationError) and error.category.startswith(
                "stale_"
            ):
                raise _fail("stale") from None
            raise _fail("non_removable") from None
        raise _fail("outcome_uncertain") from None

    @staticmethod
    def _result(
        mutation: ProfileMutationResult,
        compact: ReceiptHandle,
        *,
        reconciled: bool,
    ) -> ToolProfileRemovalResult:
        return ToolProfileRemovalResult(
            RemovedToolProfile(
                mutation.profile_id,
                "tool_pack_tombstone",
                mutation.revision,
                mutation.policy_digest,
                compact.receipt_id,
                compact.digest,
            ),
            mutation.store_generation,
            reconciled,
        )


__all__ = [
    "RemovedToolProfile",
    "ToolProfileRemovalResult",
    "ToolProfileRemovalService",
]
