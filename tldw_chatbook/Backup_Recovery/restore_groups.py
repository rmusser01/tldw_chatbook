"""Select archived groups and review dependent target data before planning."""

import hashlib
from dataclasses import replace

from .data_groups import ResolvedGroups, group_for_owner, resolve_inventory_groups
from .group_selection import archive_inventory


def directed_dependency_groups(doc) -> dict[str, str]:
    """Recognize exact per-owner dependency records emitted by our writer.

    A configuration dependency does not make its dependents one atomic group.
    Legacy or modified cohort records still retain atomic membership semantics.
    The ID alone is insufficient: the members must match the producer graph.
    """
    represented = {row.logical_id for row in (*doc.files, *doc.directories)}
    producers = {row.logical_id: row for row in doc.producer_inventory}
    shared = {}
    for row in doc.producer_inventory:
        if row.logical_id in represented and row.shared_group:
            shared.setdefault(row.shared_group, set()).add(row.logical_id)
    expected = {}
    for key in represented:
        row = producers.get(key)
        if row is None:
            continue
        members = {key} | (set(row.dependencies) & represented)
        if row.shared_group:
            members.update(shared[row.shared_group])
        expected["group:" + hashlib.sha256(key.encode()).hexdigest()] = (key, members)
    return {
        group.group_id: expected[group.group_id][0]
        for group in doc.dependency_groups
        if group.group_id in expected
        and set(group.members) == expected[group.group_id][1]
    }


def available_group_ids(doc) -> tuple[str, ...]:
    """Name saved payloads and explicit empty owners, excluding config support."""
    support = set(doc.group_scope.support_ids) if doc.group_scope else set()
    owners = {row.logical_id: row.owner_id for row in doc.producer_inventory}
    pairs = [(row.logical_id, row.owner_id) for row in doc.files]
    pairs.extend(
        (row.logical_id, owners.get(row.logical_id))
        for row in doc.directories
        if not row.synthetic
    )
    pairs.extend(
        (row.logical_id, row.owner_id)
        for row in doc.producer_inventory
        if row.status == "unused"
        and (
            doc.group_scope is None
            or group_for_owner(row.owner_id) in doc.group_scope.effective_groups
        )
    )
    return tuple(
        sorted(
            {
                group
                for key, owner in pairs
                if key not in support
                if (group := group_for_owner(owner)) is not None
            }
        )
    )


def required_target_groups(target, effective_groups) -> frozenset[str]:
    """Return active dependent or shared groups outside the reviewed selection."""
    effective = set(effective_groups)
    items = {item.logical_id: item for item in target.items}
    selected_shared = {
        item.shared_group
        for item in target.items
        if item.shared_group and group_for_owner(item.owner) in effective
    }
    required = set()
    for item in target.items:
        group = group_for_owner(item.owner)
        if (
            group is None
            or group in effective
            or item.status in {"unused", "intentionally_excluded", "missing_required"}
        ):
            continue
        dependencies = (items.get(key) for key in item.dependencies)
        linked = any(
            row is not None
            and row.owner != "config"
            and group_for_owner(row.owner) in effective
            for row in dependencies
        )
        shared = item.shared_group in selected_shared
        if linked or shared:
            required.add(group)
    return frozenset(required)


def resolve_archive_groups(
    doc, group_ids: tuple[str, ...] | None, *, target=None
) -> ResolvedGroups:
    """Close available selections over source and target dependencies.

    Target dependents are explicitly added to the review when their group exists
    in the backup; otherwise restoration refuses. A shared config dependency is
    handled through path/reachability validation, not implicit whole-profile scope.
    """
    available = set(available_group_ids(doc))
    if group_ids is not None:
        resolve_inventory_groups((), group_ids)  # Strict installed choice validation.
        if not set(group_ids) <= available:
            raise ValueError("archive_group_unavailable")
    initial = available if group_ids is None else set(group_ids)
    if not initial or not doc.producer_inventory:
        raise ValueError("archive_group_inventory_required")
    inventory = archive_inventory(doc)
    producers = {row.logical_id: row for row in doc.producer_inventory}
    directed = directed_dependency_groups(doc)
    synthetic = {row.logical_id for row in doc.directories if row.synthetic}
    effective = set(initial)
    while True:
        scope = resolve_inventory_groups(inventory, tuple(sorted(effective)))
        effective.update(scope.effective_groups)
        if not effective <= available:
            raise ValueError("archive_group_dependency_unavailable")
        selected = selected_archive_ids(doc, scope, retain_config=False)
        required = set()
        for cohort in doc.dependency_groups:
            if cohort.group_id in directed or not selected.intersection(cohort.members):
                continue
            if not cohort.complete:
                raise ValueError("dependency_group_incomplete")
            for key in set(cohort.members) - selected - synthetic:
                row = producers.get(key)
                group = group_for_owner(row.owner_id) if row is not None else None
                if group is None or group not in available:
                    raise ValueError("archive_group_dependency_unavailable")
                required.add(group)
        if not required <= effective:
            effective.update(required)
            continue
        if target is None:
            break
        required = required_target_groups(target, effective)
        for group in sorted(required):
            if group not in available:
                raise ValueError("required_target_group_unavailable:" + group)
        if required <= effective:
            break
        effective.update(required)
    return replace(
        scope,
        requested_groups=None if group_ids is None else tuple(sorted(group_ids)),
        required_groups=tuple(sorted(effective - initial)),
    )


def selected_archive_ids(
    doc, scope: ResolvedGroups, *, retain_config: bool
) -> frozenset[str]:
    """Select complete artifact roots without modifying the verified archive."""
    selected = set(scope.member_ids) | set(scope.support_ids)
    if retain_config:
        selected.difference_update(scope.support_ids)
    records = {row.logical_id: row for row in (*doc.files, *doc.directories)}
    selected.intersection_update(records)
    # Package scaffolds carry no group but belong to their selected payload.
    for key in tuple(selected):
        row = records[key]
        selected.add(row.root_id)
        while row.parent_id is not None:
            selected.add(row.parent_id)
            row = records[row.parent_id]
    return frozenset(selected)


def selected_absent_target_ids(doc, scope, target, restore, retained_configs):
    """Map explicit archived owner absence to independently discovered local data.

    Only the selected source profile's exact declaration (or declared tree root)
    can name an empty owner. Missing payloads and excluded declarations do not.
    Existing retirement, shared-scope, and complete-tree checks still apply.
    """
    if target is None or not any(
        row.status == "unused" and row.logical_id in scope.member_ids
        for row in doc.producer_inventory
    ):
        return frozenset()
    producers = {row.logical_id: row for row in doc.producer_inventory}
    represented = {row.logical_id for row in (*doc.files, *doc.directories)}
    config_paths = {
        key: path
        for key, path in restore
        if key in producers and producers[key].owner_id == "config"
    }
    config_paths.update(
        {row.archive_config_id: row.config_path for row in retained_configs}
    )
    from .owner_registry import install_adapters
    from .profile_paths import DATABASE_PATHS, database_path

    installed = {owner.owner_id: owner for owner in install_adapters()}
    sqlite_settings = {
        owner_id: setting
        for owner_id, setting, _, _ in DATABASE_PATHS
        if owner_id in installed
        and (policy := installed[owner_id].schema_policy()) is not None
        and policy.schema_sql
    }
    profiles = {}
    current_config_paths = {}
    current_configs = {}
    for source_id, path in config_paths.items():
        configs = [
            item
            for item in target.items
            if item.owner == "config"
            and item.status == "included"
            and item.path == path
        ]
        if len(configs) != 1:
            raise ValueError("retained_config_relation_unverified")
        current = configs[0].logical_id.split(":")
        source = source_id.split(":")
        if (
            len(current) != 3
            or len(source) != 3
            or current[0] != source[0]
            or source[0] != "profile"
        ):
            raise ValueError("retained_config_relation_unverified")
        if current[1] in profiles and profiles[current[1]] != source[1]:
            raise ValueError("ambiguous_relocation_mapping")
        profiles[current[1]] = source[1]
        current_config_paths[current[1]] = path
    selected = set(scope.member_ids)
    absent = set()
    active_owners = set()
    matched = set()
    for item in target.items:
        parts = item.logical_id.split(":", 2)
        primary_sqlite = (
            item.metadata is None
            and item.status == "included"
            and item.path is not None
            and len(parts) == 3
            and parts[2] == item.owner
            and item.owner in sqlite_settings
        )
        if (
            len(parts) != 3
            or parts[0] != "profile"
            or parts[1] not in profiles
            or item.status not in {"included", "included_directory", "unused"}
            or (
                item.metadata is None
                and item.status != "unused"
                and not primary_sqlite
            )
            or group_for_owner(item.owner) not in scope.effective_groups
            or f"profile:{parts[1]}:config" not in item.dependencies
        ):
            continue
        target_prefix = f"profile:{parts[1]}:"
        source_prefix = f"profile:{profiles[parts[1]]}:"
        if item.status != "unused":
            active_owners.add((parts[1], item.owner))
        keys = {item.logical_id}
        if item.metadata is not None:
            keys.add(item.metadata.root_id)
        for key in keys:
            if not key.startswith(target_prefix):
                continue
            source_key = source_prefix + key[len(target_prefix) :]
            declaration = producers.get(source_key)
            if (
                source_key in selected
                and source_key not in represented
                and declaration is not None
                and declaration.owner_id == item.owner
                and declaration.status == "unused"
                and source_prefix + "config" in declaration.dependencies
            ):
                if primary_sqlite:
                    # A primary without file metadata has no tree/root authority.
                    # Its installed locator must agree with the private selector
                    # belonging to this exact, independently matched profile.
                    if parts[1] not in current_configs:
                        import tomllib

                        from .archive_reader import _regular
                        from .destinations import _config_tables
                        from .restore_plan import (
                            _retained_config_observation,
                            retained_config_observation,
                        )

                        config_path = current_config_paths[parts[1]]
                        observation = retained_config_observation(config_path)
                        with _regular(config_path) as stream:
                            raw = stream.read(16 * 1024**2 + 1)
                        if len(raw) > 16 * 1024**2:
                            raise ValueError("retained_config_limit")
                        try:
                            data = tomllib.loads(raw.decode("utf-8"))
                            _config_tables(data)
                        except (ValueError, UnicodeError, RecursionError):
                            raise ValueError("retained_config_invalid") from None
                        if _retained_config_observation(config_path) != observation:
                            raise ValueError("retained_config_changed")
                        current_configs[parts[1]] = data
                    if database_path(
                        current_configs[parts[1]], sqlite_settings[item.owner]
                    ) != item.path:
                        raise ValueError("selected_absence_locator_changed")
                if item.status != "unused":
                    absent.add(item.logical_id)
                matched.add((parts[1], source_key))
                break
    for current_profile, source_profile in profiles.items():
        prefix = f"profile:{source_profile}:"
        for declaration in doc.producer_inventory:
            if (
                declaration.logical_id in selected
                and declaration.logical_id not in represented
                and declaration.logical_id.startswith(prefix)
                and declaration.status == "unused"
                and prefix + "config" in declaration.dependencies
                and (current_profile, declaration.owner_id) in active_owners
                and (current_profile, declaration.logical_id) not in matched
            ):
                raise ValueError("selected_absence_mapping_required")
    absent_paths = {item.path for item in target.items if item.logical_id in absent}
    if any(
        path == incoming or path in incoming.parents or incoming in path.parents
        for path in absent_paths
        for _, incoming in restore
    ):
        raise ValueError("selected_absence_publication_conflict")
    return frozenset(absent)


def check_preserved_group_effects(plan, scope: ResolvedGroups) -> None:
    """Refuse any planner retirement/publication outside the reviewed groups."""
    if plan.target is None:
        return
    changed = {path for _, path in (*plan.restore, *plan.retire)}
    for item in plan.target.items:
        if item.path is None or item.status == "unused":
            continue
        if item.path not in changed and not any(
            parent in changed for parent in item.path.parents
        ):
            continue
        group = group_for_owner(item.owner)
        if group is not None and group not in scope.effective_groups:
            raise ValueError("unreviewed_group_effect:" + group)
