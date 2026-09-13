"""Application-facing orchestration for portable Tool policy packs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal

from tldw_chatbook.MCP.permission_store import (
    PermissionStoreSnapshot,
    profile_lifecycle_disposition,
)
from tldw_chatbook.Tool_Packs.activation import ToolPackActivationService
from tldw_chatbook.Tool_Packs.binding import (
    ToolProfileBindingGuard,
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryRegistry,
    PermissionInventorySnapshot,
    capture_v1_inventory,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.export import (
    ToolPackExportReview,
    ToolPackExportService,
    resolve_portable_tool_state,
)
from tldw_chatbook.Tool_Packs.importer import ServerMapping, ToolPackImportService
from tldw_chatbook.Tool_Packs.publication import publish_tool_pack
from tldw_chatbook.Tool_Packs.receipt_store import ToolPackReceiptStore
from tldw_chatbook.Tool_Packs.removal import ToolProfileRemovalService


_RECEIPT_ID = re.compile(r"tp-[0-9a-f]{32}\Z")


@dataclass(frozen=True, slots=True)
class ToolProfilePresentation:
    """Current, privacy-safe presentation of one stored Tool profile."""

    profile_id: str
    origin: Literal["local", "workspace-managed", "imported"]
    lifecycle_valid: bool
    binding_state: Literal["bound", "unbound"]
    first_bind_confirmation_required: bool
    reference_counts: tuple[int, int]
    posture_counts: tuple[int, int, int]
    receipt_health: Literal["not_applicable", "available", "unavailable"]
    removal_eligible: bool
    removal_blocker: str | None
    revision: int | None
    policy_digest: str | None


@dataclass(frozen=True, slots=True)
class ToolProfileListing:
    """Deeply immutable profile listing or one stable unavailable outcome."""

    profiles: tuple[ToolProfilePresentation, ...] = ()
    unavailable_category: str | None = None

    def by_id(self, profile_id: str) -> ToolProfilePresentation | None:
        """Return the exact visible profile id, if present."""
        return next(
            (row for row in self.profiles if row.profile_id == profile_id), None
        )


@dataclass(frozen=True, slots=True)
class ToolPackReceiptReconciliationResult:
    """Bounded receipt recovery outcome without filesystem details."""

    removed_ids: tuple[str, ...] = ()
    unavailable_category: str | None = None


class _WorkspaceReferences:
    """One adapter for active, archived, and dangling assistant defaults."""

    def __init__(self, registry: object) -> None:
        self._registry = registry

    def capture(self) -> tuple[object, ...]:
        records = self._registry.list_workspaces(include_archived=True)  # type: ignore[attr-defined]
        if type(records) is not tuple:
            raise TypeError("invalid workspace snapshot")
        for record in records:
            if type(getattr(record, "archived", None)) is not bool:
                raise TypeError("invalid workspace snapshot")
        return records

    def counts(self, profile_id: str, records: tuple[object, ...]) -> tuple[int, int]:
        active = archived = 0
        for record in records:
            defaults = getattr(record, "assistant_defaults", None)
            if getattr(defaults, "tool_policy_profile_id", None) != profile_id:
                continue
            if record.archived:  # type: ignore[attr-defined]
                archived += 1
            else:
                active += 1
        return active, archived

    def references_profile(self, profile_id: str, *, include_archived: bool) -> bool:
        active, archived = self.counts(profile_id, self.capture())
        return bool(active or (include_archived and archived))

    def defaults_for(self, workspace_id: str):
        for record in self.capture():
            if getattr(record, "workspace_id", None) == workspace_id:
                return getattr(record, "assistant_defaults", None)
        return None


class ToolPackService:
    """Narrow facade over existing portable Tool Pack authority owners."""

    def __init__(
        self,
        *,
        permission_store: object,
        inventory: object,
        workspace_registry: object,
        receipt_store: ToolPackReceiptStore,
        exporter: object,
        importer: object,
        activation: object,
        binding_guard: object,
        removal: object,
        publisher: Callable[..., object],
        lifecycle: ToolProfileLifecycleCoordinator,
        now: Callable[[], datetime] | None = None,
        references: _WorkspaceReferences | None = None,
    ) -> None:
        self._permission_store = permission_store
        self._inventory = inventory
        self._workspace_registry = workspace_registry
        self._receipt_store = receipt_store
        self._exporter = exporter
        self._importer = importer
        self._activation = activation
        self._binding_guard = binding_guard
        self._removal = removal
        self._publisher = publisher
        self._lifecycle = lifecycle
        self._references = references or _WorkspaceReferences(workspace_registry)
        self._now = now or (lambda: datetime.now(timezone.utc))

    @classmethod
    def compose(
        cls,
        *,
        permission_store: object,
        inventory: PermissionInventoryRegistry,
        workspace_registry: object,
        receipt_root: Path,
    ) -> "ToolPackService":
        """Compose every owner around one lifecycle and reference authority."""
        lifecycle = ToolProfileLifecycleCoordinator()
        references = _WorkspaceReferences(workspace_registry)
        receipts = ToolPackReceiptStore(receipt_root)
        importer = ToolPackImportService(permission_store, inventory, references)
        activation = ToolPackActivationService(
            permission_store=permission_store,
            inventory=inventory,
            importer=importer,
            reference_checker=references,
            receipt_store=receipts,
            lifecycle=lifecycle,
        )
        guard = ToolProfileBindingGuard(
            permission_store=permission_store,
            inventory=inventory,
            workspace_defaults_reader=references.defaults_for,
            lifecycle=lifecycle,
        )
        removal = ToolProfileRemovalService(
            permission_store=permission_store,
            receipt_store=receipts,
            reference_checker=references,
            lifecycle=lifecycle,
        )
        return cls(
            permission_store=permission_store,
            inventory=inventory,
            workspace_registry=workspace_registry,
            receipt_store=receipts,
            exporter=ToolPackExportService(permission_store, inventory),
            importer=importer,
            activation=activation,
            binding_guard=guard,
            removal=removal,
            publisher=publish_tool_pack,
            lifecycle=lifecycle,
            references=references,
        )

    @property
    def binding_guard(self) -> object:
        return self._binding_guard

    @property
    def lifecycle(self) -> ToolProfileLifecycleCoordinator:
        """Return the shared runtime-lease and mutation coordinator."""
        return self._lifecycle

    @property
    def receipt_root(self) -> Path:
        return self._receipt_store.root

    def capture_export(
        self,
        profile_id: str,
        *,
        display_name: str,
        suggested_id: str,
        expected_revision: int | None = None,
        expected_policy_digest: str | None = None,
    ) -> object:
        """Capture an immutable export review for one exact profile revision.

        Args:
            profile_id: Permission profile to export.
            display_name: User-facing pack name.
            suggested_id: Portable destination id suggestion.
            expected_revision: Optional imported-profile revision fence.
            expected_policy_digest: Optional exact policy digest fence.

        Returns:
            The captured Tool Pack export review.

        Raises:
            ToolPackError: If the profile or export authority is invalid.
        """
        return self._delegate(
            "export",
            "profile_invalid",
            self._exporter.capture,
            profile_id=profile_id,
            display_name=display_name,
            suggested_id=suggested_id,
            expected_revision=expected_revision,
            expected_policy_digest=expected_policy_digest,
        )

    def publish_export(
        self,
        review: ToolPackExportReview,
        destination: object,
        *,
        overwrite_token: str | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> object:
        """Publish one captured export review to one captured destination.

        Args:
            review: Exact export review returned by :meth:`capture_export`.
            destination: Captured publication destination.
            overwrite_token: Reserved compatibility token; existing-file overwrite
                is unsupported.
            cancelled: Probe checked immediately before publication.

        Returns:
            The immutable publication result.

        Raises:
            ToolPackError: If review or destination authority is invalid, safe
                publication is unavailable, or publication fails.
        """
        if type(review) is not ToolPackExportReview:
            raise ToolPackError("export", "publication_failed")
        return self._delegate(
            "export",
            "publication_failed",
            self._publisher,
            review.snapshot,
            destination,
            overwrite=overwrite_token is not None,
            overwrite_token=overwrite_token,
            cancelled=cancelled,
        )

    def inspect_import(
        self,
        archive_path: Path,
        *,
        destination_id: str,
        mappings: Sequence[ServerMapping] = (),
    ) -> object:
        """Inspect an archive without mutating permission or workspace state.

        Args:
            archive_path: User-selected Tool Pack archive.
            destination_id: Proposed unbound permission-profile id.
            mappings: Explicit source-to-destination MCP server mappings.

        Returns:
            An immutable, expiring import review.

        Raises:
            ToolPackError: If archive, destination, store, inventory, or mapping
                admission fails.
        """
        return self._delegate(
            "import",
            "archive_invalid",
            self._importer.inspect_archive,
            archive_path,
            destination_id=destination_id,
            mappings=mappings,
        )

    def import_unbound(self, review: object) -> object:
        """Install exactly one approved import review as an unbound profile.

        Args:
            review: Exact import review returned by :meth:`inspect_import`.

        Returns:
            The known activation result.

        Raises:
            ToolPackError: If review revalidation or activation fails or is uncertain.
        """
        return self._delegate(
            "import", "activation_failed", self._activation.install, review
        )

    def review_first_bind(
        self, workspace_id: str, intended_defaults: object, *, action: str
    ) -> object:
        """Capture a first-bind confirmation for current workspace authority.

        Args:
            workspace_id: Workspace whose defaults would change.
            intended_defaults: Exact proposed workspace defaults.
            action: Requested binding action.

        Returns:
            An immutable first-bind review.

        Raises:
            ToolPackError: If current binding authority cannot be confirmed.
        """
        return self._delegate(
            "bind",
            "confirmation_invalid",
            self._binding_guard.review,
            workspace_id,
            intended_defaults,
            action=action,
        )

    def confirm_first_bind(self, review: object) -> object:
        """Consume one current first-bind review.

        Args:
            review: Exact review returned by :meth:`review_first_bind`.

        Returns:
            The confirmed workspace defaults mutation result.

        Raises:
            ToolPackError: If confirmation is invalid, expired, stale, or uncertain.
        """
        return self._delegate(
            "bind", "confirmation_invalid", self._binding_guard.confirm, review
        )

    def remove_profile(self, profile_id: str, *, expected_revision: int) -> object:
        """Remove one eligible imported profile at its exact revision.

        Args:
            profile_id: Imported profile to remove.
            expected_revision: Exact lifecycle revision selected by the user.

        Returns:
            The known removal result.

        Raises:
            ToolPackError: If the profile is referenced, active, stale, invalid,
                non-removable, or has an uncertain mutation outcome.
        """
        return self._delegate(
            "remove",
            "non_removable",
            self._removal.remove,
            profile_id,
            expected_revision=expected_revision,
        )

    @staticmethod
    def _delegate(operation: str, category: str, function: Callable, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ToolPackError:
            raise
        except Exception:
            raise ToolPackError(operation, category) from None

    def list_profiles(self) -> ToolProfileListing:
        """Capture strict current authority and return immutable visible rows."""
        try:
            snapshot = self._permission_store.read_snapshot_strict()  # type: ignore[attr-defined]
            if type(snapshot) is not PermissionStoreSnapshot:
                raise TypeError
            profiles = snapshot.payload.get("profiles")
            if not isinstance(profiles, Mapping):
                raise TypeError
        except Exception:
            return ToolProfileListing(unavailable_category="authority_unavailable")
        try:
            records = self._references.capture()
        except Exception:
            return ToolProfileListing(unavailable_category="references_unavailable")
        try:
            inventory = capture_v1_inventory(self._inventory)  # type: ignore[arg-type]
            if type(inventory) is not PermissionInventorySnapshot:
                raise TypeError
        except Exception:
            return ToolProfileListing(unavailable_category="inventory_unavailable")

        rows: list[ToolProfilePresentation] = []
        for profile_id in sorted(profiles):
            profile = profiles[profile_id]
            if type(profile_id) is not str or not isinstance(profile, Mapping):
                return ToolProfileListing(unavailable_category="authority_unavailable")
            disposition = profile_lifecycle_disposition(profile)
            lifecycle = profile.get("tool_pack_lifecycle")
            lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
            try:
                current_policy_digest = profile_policy_digest(profile)
            except (TypeError, ValueError):
                current_policy_digest = None
            lifecycle_valid = (
                disposition != "invalid"
                and current_policy_digest is not None
                and (
                    disposition not in {"imported", "tombstone"}
                    or lifecycle.get("policy_digest") == current_policy_digest
                )
            )
            if disposition == "tombstone" and lifecycle_valid:
                continue
            references = self._references.counts(profile_id, records)
            origin = self._origin(profile_id, profile, disposition)
            receipt_health = self._receipt_health(profile_id, disposition, lifecycle)
            blocker = self._removal_blocker(
                disposition, lifecycle_valid, references, profile_id, receipt_health
            )
            rows.append(
                ToolProfilePresentation(
                    profile_id=profile_id,
                    origin=origin,
                    lifecycle_valid=lifecycle_valid,
                    binding_state="bound" if any(references) else "unbound",
                    first_bind_confirmation_required=(
                        disposition == "imported"
                        and lifecycle.get("first_bind_confirmation_required") is True
                    ),
                    reference_counts=references,
                    posture_counts=self._posture_counts(
                        snapshot.payload, profile_id, inventory
                    ),
                    receipt_health=receipt_health,
                    removal_eligible=blocker is None,
                    removal_blocker=blocker,
                    revision=(
                        lifecycle.get("revision")
                        if type(lifecycle.get("revision")) is int
                        else None
                    ),
                    policy_digest=(current_policy_digest if lifecycle_valid else None),
                )
            )
        return ToolProfileListing(tuple(rows))

    @staticmethod
    def _origin(profile_id: str, profile: Mapping, disposition: str):
        if disposition == "imported" or profile.get("profile_kind") in {
            "tool_pack_imported",
            "tool_pack_tombstone",
        }:
            return "imported"
        return "workspace-managed" if profile_id.startswith("ws-") else "local"

    def _receipt_health(
        self, profile_id: str, disposition: str, lifecycle: Mapping
    ) -> str:
        if disposition != "imported":
            return "unavailable" if lifecycle else "not_applicable"
        try:
            verified = self._receipt_store.read(
                lifecycle["receipt_id"], expected_digest=lifecycle["receipt_digest"]
            )
            if (
                verified.receipt.kind != "import"
                or verified.receipt.profile_id != profile_id
            ):
                raise ValueError
            return "available"
        except Exception:
            return "unavailable"

    def _removal_blocker(
        self,
        disposition: str,
        lifecycle_valid: bool,
        references: tuple[int, int],
        profile_id: str,
        receipt_health: str,
    ) -> str | None:
        if disposition == "invalid" or not lifecycle_valid:
            return "lifecycle_invalid"
        if disposition != "imported":
            return "not_imported"
        if any(references):
            return "referenced"
        if self._lifecycle.active_lease_count(profile_id):
            return "in_use"
        if receipt_health != "available":
            return "receipt_unavailable"
        return None

    @staticmethod
    def _posture_counts(
        payload: Mapping, profile_id: str, inventory: PermissionInventorySnapshot
    ) -> tuple[int, int, int]:
        counts = {"allow": 0, "ask": 0, "deny": 0}
        for item in inventory.tools:
            state = resolve_portable_tool_state(payload, item, profile_id)
            counts[state] += 1
        return counts["allow"], counts["ask"], counts["deny"]

    def reconcile_receipts(self) -> ToolPackReceiptReconciliationResult:
        """Reclaim only store-eligible orphans after complete owner capture."""
        try:
            snapshot = self._permission_store.read_snapshot_strict()  # type: ignore[attr-defined]
            if type(snapshot) is not PermissionStoreSnapshot:
                raise TypeError
            profiles = snapshot.payload.get("profiles")
            if not isinstance(profiles, Mapping):
                raise TypeError
            referenced = self._linked_receipt_ids(profiles)
        except Exception:
            return ToolPackReceiptReconciliationResult(
                unavailable_category="authority_unavailable"
            )
        try:
            self._references.capture()
        except Exception:
            return ToolPackReceiptReconciliationResult(
                unavailable_category="references_unavailable"
            )
        try:
            live = self._activation.live_receipt_ids()
            if type(live) is not frozenset:
                raise TypeError
        except Exception:
            return ToolPackReceiptReconciliationResult(
                unavailable_category="live_owners_unavailable"
            )
        try:
            now = self._now()
            removed = self._receipt_store.reconcile_orphans(referenced, live, now=now)
        except ToolPackError as error:
            category = (
                "receipt_store_incomplete"
                if error.category == "capacity_exceeded"
                else "receipt_store_unavailable"
            )
            return ToolPackReceiptReconciliationResult(unavailable_category=category)
        except Exception:
            return ToolPackReceiptReconciliationResult(
                unavailable_category="receipt_store_unavailable"
            )
        return ToolPackReceiptReconciliationResult(tuple(removed))

    @staticmethod
    def _linked_receipt_ids(profiles: Mapping) -> frozenset[str]:
        linked: set[str] = set()
        for profile in profiles.values():
            if not isinstance(profile, Mapping):
                raise TypeError
            lifecycle = profile.get("tool_pack_lifecycle")
            if not isinstance(lifecycle, Mapping):
                continue
            receipt_id = lifecycle.get("receipt_id")
            if type(receipt_id) is str and _RECEIPT_ID.fullmatch(receipt_id):
                linked.add(receipt_id)
        return frozenset(linked)


__all__ = [
    "ToolPackReceiptReconciliationResult",
    "ToolPackService",
    "ToolProfileListing",
    "ToolProfilePresentation",
]
