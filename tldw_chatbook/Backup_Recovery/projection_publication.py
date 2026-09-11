"""Review complete dependent projection roots before any publication authority."""

from dataclasses import replace

from .models import Inventory
from .rag_projection_validation import recognized_path


def projection_groups(items):
    """Check exact declared root topology without opening any live engine."""
    groups = {}
    for item in items:
        if item.owner != "rag.projections":
            continue
        if item.status == "unused":
            if item.path is not None and item.path.exists():
                raise ValueError("projection_scope_unverified")
            continue
        if item.path is None or item.metadata is None:
            raise ValueError("projection_scope_unverified")
        groups.setdefault(item.metadata.root_id, []).append(item)
    result = {}
    for key, group in groups.items():
        roots = [item for item in group if item.logical_id == key]
        if len(roots) != 1:
            raise ValueError("projection_root_incomplete")
        root = roots[0]
        if root.metadata.relative_path or root.metadata.kind != "directory":
            raise ValueError("projection_root_incomplete")
        paths = {item.path for item in group}
        for item in group:
            meta = item.metadata
            if (
                item.status
                not in {"included", "included_directory", "intentionally_excluded"}
                or item.path != root.path / meta.relative_path
                or not recognized_path(meta.relative_path, meta.kind)
                or (
                    item.path.is_dir()
                    if meta.kind == "directory"
                    else item.path.is_file()
                )
                is False
                or item.path.is_symlink()
            ):
                raise ValueError("projection_scope_unverified")
            if meta.relative_path and not any(
                parent.logical_id == meta.parent_id and parent.path == item.path.parent
                for parent in group
            ):
                raise ValueError("projection_root_incomplete")
            if meta.kind == "directory" and any(
                child not in paths for child in item.path.iterdir()
            ):
                raise ValueError("projection_root_incomplete")
        result[key] = tuple(group)
    return result


def dependent_retirements(target, restore, retire, preserve):
    """Follow declared dependencies; omitted dependent roots cannot stay active."""
    changed_paths = {path for _, path in (*restore, *retire)}
    changed = {
        item.logical_id
        for item in target.items
        if item.path is not None
        and any(
            item.path == path or path in item.path.parents for path in changed_paths
        )
    }
    affected = set(changed)
    while True:
        dependents = {
            item.logical_id
            for item in target.items
            if affected.intersection(item.dependencies)
        }
        if dependents <= affected:
            break
        affected.update(dependents)
    projections = [item for item in target.items if item.owner == "rag.projections"]
    roots = {
        item.metadata.root_id
        for item in projections
        if item.logical_id in affected and item.metadata
    }
    # An owner with unknown topology cannot be silently preserved after its source changes.
    if any(
        item.logical_id in affected
        and (item.status != "unused" or item.path is not None and item.path.exists())
        and item.metadata is None
        for item in projections
    ):
        raise ValueError("projection_scope_unverified")
    selected = [
        item for item in projections if item.metadata and item.metadata.root_id in roots
    ]
    groups = projection_groups(selected)
    by_id = {item.logical_id: item for item in target.items}
    paths = {path for _, path in restore}
    retiring, preserving, issues = set(retire), set(preserve), []
    for key, group in groups.items():
        root = by_id[key]
        aliases = [
            item
            for item in projections
            if item.metadata
            and item.logical_id == item.metadata.root_id
            and (
                item.path == root.path
                or root.shared_group
                and item.shared_group == root.shared_group
            )
        ]
        if any(alias.metadata.root_id not in groups for alias in aliases):
            raise ValueError("shared_scope_expansion_required")
        dependencies = {
            dependency
            for item in group
            for dependency in item.dependencies
            if dependency not in {member.logical_id for member in group}
        }
        if dependencies - by_id.keys():
            raise ValueError("projection_source_scope_unverified")
        if not dependencies <= changed:
            raise ValueError("shared_scope_expansion_required")
        if root.path not in paths and any(root.path in path.parents for path in paths):
            raise ValueError("projection_root_incomplete")
        for item in group:
            preserving.discard((item.logical_id, item.path))
            retiring.add((item.logical_id, item.path))
        issues.append("projection_reconciliation_required:" + key)
    return sorted(retiring), sorted(preserving), tuple(sorted(issues))


def normalized_originals(inventory):
    """Qualify whole-root coverage before converting omitted bytes into originals."""
    groups = projection_groups(inventory.items)
    normalized = {
        item.logical_id: replace(
            item,
            status="included_directory"
            if item.metadata.kind == "directory"
            else "included",
        )
        for group in groups.values()
        for item in group
    }
    return Inventory(
        tuple(normalized.get(item.logical_id, item) for item in inventory.items),
        inventory.complete,
        inventory.scope_digest,
        inventory.issues,
    )
