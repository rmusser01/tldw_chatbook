"""Apply installed data-group scope without suppressing storage-ownership checks."""

import hashlib
import json
from dataclasses import replace

from .data_groups import group_for_owner, resolve_inventory_groups
from .models import Inventory, StorageItem


def select_inventory(
    inventory: Inventory, group_ids: tuple[str, ...] | None
) -> Inventory:
    """Keep selected payloads and dependency support, retaining exclusion evidence.

    Full discovery runs first so selection cannot conceal aliases, invalid trees,
    malformed configuration, or unknown owners. Only coverage failures belonging
    to deliberately unselected groups are recomputed for the narrower scope.
    """
    if group_ids is None:
        return inventory
    from .inventory import BLOCKING, classify_entries

    scope = resolve_inventory_groups(inventory.items, group_ids)
    selected = set(scope.member_ids) | set(scope.support_ids)
    items = tuple(
        replace(item, status="intentionally_excluded")
        if group_for_owner(item.owner) is not None and item.logical_id not in selected
        else item
        for item in inventory.items
    )
    result = classify_entries(items)
    issues = set(result.issues) | (
        set(inventory.issues) - BLOCKING - {"dependency_unavailable"}
    )
    digest = hashlib.sha256(
        json.dumps(
            (
                inventory.scope_digest,
                result.scope_digest,
                scope.requested_groups,
                scope.effective_groups,
            ),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return replace(
        result,
        complete=result.complete and not issues,
        issues=tuple(sorted(issues)),
        scope_digest=digest,
    )


def archive_inventory(doc) -> tuple[StorageItem, ...]:
    """Project authenticated declarations for pure grouping, excluding scaffolds.

    Synthetic roots are archive packaging, not separately selected source owners.
    This representation supplies no local paths or filesystem authority.
    """
    synthetic = {row.logical_id for row in doc.directories if row.synthetic}
    return tuple(
        StorageItem(
            row.owner_id,
            row.logical_id,
            None,
            row.status,
            row.dependencies,
            row.shared_group,
        )
        for row in doc.producer_inventory
        if row.logical_id not in synthetic
    )


def validate_archive_group_scope(doc) -> None:
    """Check v2 scope against installed membership and authenticated coverage."""
    scope = doc.group_scope
    if doc.format_version == 1:
        if scope is not None:
            raise ValueError("unexpected_group_scope")
        return
    if scope is None or not doc.producer_inventory:
        raise ValueError("missing_group_scope")
    resolved = resolve_inventory_groups(archive_inventory(doc), scope.requested_groups)
    payloads = {row.logical_id for row in doc.files}
    support = tuple(sorted(set(resolved.support_ids) & payloads))
    if (
        scope.requested_groups != resolved.requested_groups
        or scope.effective_groups != resolved.effective_groups
        or scope.support_ids != support
    ):
        raise ValueError("invalid_group_scope")
    admitted = set(resolved.member_ids) | set(resolved.support_ids)
    # These pre-existing independent options carry no selectable durable group.
    independent = {"diagnostics.logs", "recovery.credentials"}
    if any(
        row.logical_id not in admitted and row.owner_id not in independent
        for row in doc.files
    ):
        raise ValueError("unexpected_group_scope_payload")
    synthetic = {row.logical_id for row in doc.directories if row.synthetic}
    if any(row.logical_id not in admitted | synthetic for row in doc.directories):
        raise ValueError("unexpected_group_scope_directory")
