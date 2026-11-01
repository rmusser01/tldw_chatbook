"""Fixed first-profile enrollment for an explicitly selected mobile start."""

import hashlib
import stat
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Chat.console_trace_read_budget import _mobile_operation_budgets
from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .archive_reader import _check
from .capture import CaptureReviewRequired
from .capture_service import _capture_names
from .control_records import (
    UNBOUND_NAMESPACE,
    _existing_admission_authority,
    bind_profile,
)
from .inventory import (
    _fixed_control_exclusion,
    _service_control_exclusion,
    classify_entries,
    discover,
    read_profile_config,
)
from .mobile_consolidation import _preview_selected_mobile, _source_claim_locations
from .mobile_preview_budget import path_cost
from .native_files import _close_owned, pinned_directory
from .profile_paths import canonical_console_paths, user_data_dir
from .service_storage import ensure_storage
from .storage_admission import (
    _attach_mobile_budgets,
    _CaptureFileDescriptors,
    _preview_reads,
)


def _scope_items(inventory, budget):
    """Compare native user scope, omitting only authenticated fixed controls."""
    budget.container(len(inventory.items))
    return tuple(
        item
        for item in inventory.items
        if not (
            item.owner == "recovery.control"
            and item.logical_id
            in {"recovery.control:fixed-bootstrap", "recovery.control:service"}
            and item.status == "intentionally_excluded"
        )
    )


def _profile_inventory(inventory, selector, budget):
    budget.reserve(512 + path_cost(selector))
    prefix = "profile:" + hashlib.sha256(str(selector).encode()).hexdigest()[:24] + ":"
    budget.container(len(inventory.items))
    selected = tuple(
        item for item in inventory.items if item.logical_id.startswith(prefix)
    )
    budget.value(selected, text_copies=4)
    result = classify_entries(selected)
    if not result.complete:
        raise CaptureReviewRequired(result.issues)
    return result


def _container(root, selector, inventory, budget, custody, cancel, *, controls=()):
    """Observe a complete declared private container without adopting children."""
    budget.container(2 * (len(inventory.items) + len(controls)))
    budget.reserve(512 + path_cost(selector))
    allowed = {
        item.path
        for item in inventory.items
        if item.path is not None
        and (
            item.status in {"included", "included_directory", "intentionally_excluded"}
            or item.status == "unused"
            and item.metadata is not None
        )
    }
    allowed.update(controls)
    lock = selector.with_name(selector.name + ".lock")
    result = []
    count = 0

    def walk(path, fd, depth):
        nonlocal count
        _check(cancel)
        count += 1
        if count > 100_000 or depth > 256:
            raise ValueError("mobile_enrollment_container_limit")
        budget.reserve(4096 + path_cost(path))
        info = os.fstat(fd)
        if (
            info.st_uid != os.geteuid()
            or stat.S_ISREG(info.st_mode)
            and info.st_nlink != 1
        ):
            raise ValueError("mobile_enrollment_container_unsafe")
        if path == root and (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077):
            raise ValueError("mobile_enrollment_container_unsafe")
        if (
            path != root
            and path not in allowed
            and not any(path in item.parents for item in allowed)
        ) and (
            path != lock
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != 0
        ):
            raise ValueError("mobile_enrollment_unknown_sibling")
        result.append(
            (
                str(path),
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size if stat.S_ISREG(info.st_mode) else None,
                info.st_mtime_ns if stat.S_ISREG(info.st_mode) else None,
            )
        )
        if path in controls:
            return
        if stat.S_ISDIR(info.st_mode):
            names = bootstrap._bounded_names(fd, 100_000 - count, budget, custody)
            for name in names:
                _check(cancel)
                budget.reserve(1024 + path_cost(path) + 8 * len(name))
                child = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                custody.fds.append(child)
                try:
                    walk(path / name, child, depth + 1)
                finally:
                    _close_owned(child, custody)
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns) != (
                info.st_dev,
                info.st_ino,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ):
                raise ValueError("mobile_enrollment_container_changed")
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError("mobile_enrollment_container_unsafe")

    with pinned_directory(root, _custody=custody) as fd:
        walk(root, fd, 0)
    return tuple(result)


def _fixed_controls(budget, custody):
    budget.reserve(2048)
    rows = (
        *_fixed_control_exclusion(_budget=budget, _custody=custody),
        *_service_control_exclusion(_budget=budget, _custody=custody),
    )
    if any(row.status != "intentionally_excluded" for row in rows):
        raise ValueError("mobile_enrollment_control_unverified")
    return tuple(row.path for row in rows)


@contextmanager
def _start_session(preview, control_root, cancel):
    """Keep one original allowance and real exclusion through the caller's import."""
    pair = _mobile_operation_budgets()
    budget = pair[0]
    budget.reserve(4096)
    custody = _CaptureFileDescriptors(None)
    root = bootstrap.default_bootstrap_root()
    selectors = tuple(dict.fromkeys((preview.source_selector, preview.target_selector)))
    try:
        _check(cancel)
        authority = _existing_admission_authority(root)
        pending, profiles = bootstrap._records(root, _budget=budget, _custody=custody)
        if pending:
            raise ValueError("recovery_pending")
        registry = bootstrap._registry(root, _budget=budget, _custody=custody)
        budget.container(2 * (len(profiles) + len(selectors)))
        for row in profiles:
            budget.reserve(256 + 16 * len(row["selector"]))
        existing = {Path(row["selector"]): row for row in profiles}
        missing = tuple(selector for selector in selectors if selector not in existing)
        for selector in selectors:
            if (
                selector in existing
                and bootstrap._binding(
                    selector, profiles, registry, _budget=budget, _custody=custody
                )
                != existing[selector]
            ):
                raise ValueError("mobile_enrollment_binding_changed")
        with _preview_reads(_read_budget=budget):
            inventory = discover(selectors, _read_budget=budget, _custody=custody)
        if not inventory.complete:
            raise CaptureReviewRequired(inventory.issues)
        budget.reserve(1024 + 512 * len(missing))
        names_by_selector, observations, data_roots, selected_inventories = (
            {},
            {},
            {},
            {},
        )
        container_roots = {}
        controls = _fixed_controls(budget, custody)
        for selector in missing:
            values = read_profile_config(
                selector, _read_budget=budget, _custody=custody
            )
            selected = _profile_inventory(inventory, selector, budget)
            selected_inventories[selector] = selected
            data = user_data_dir(values)
            data_roots[selector] = data
            canonical = canonical_console_paths(values)
            # The installed optional Runs owner lives next to configured Chat.
            # Its absent leaf needs this exact proved parent, never an arbitrary
            # ancestor inferred from absence or a broadened existing binding.
            for path in canonical:
                budget.reserve(1024 + path_cost(path, copies=3))
            additional = tuple(
                sorted(
                    {
                        path.parent
                        for path in canonical
                        if not path.exists()
                        and path.parent != data
                        and data not in path.parent.parents
                    }
                )
            )
            if any(
                bootstrap._overlap(parent, control)
                for parent in additional
                for control in controls
            ):
                raise ValueError("mobile_enrollment_control_overlap")
            container_roots[selector] = (data, *additional)
            observations[selector] = (
                _container(
                    selector.parent,
                    selector,
                    selected,
                    budget,
                    custody,
                    cancel,
                    controls=controls,
                ),
                _container(data, selector, selected, budget, custody, cancel),
                tuple(
                    (
                        parent,
                        _container(parent, selector, selected, budget, custody, cancel),
                    )
                    for parent in additional
                ),
            )
        if preview.source_selector in missing:
            locations = _source_claim_locations(preview.source_dir)
            budget.container(len(locations) * 2)
            found = [
                (lock, path)
                for lock, path in locations
                if lock.exists() or path.exists()
            ]
            if len(found) > 1:
                raise ValueError("mobile_source_claim_ambiguous")
            selected_claim = (
                found[0]
                if found
                else next(pair for pair in locations if pair[0].parent.is_dir())
            )
            parent = selected_claim[0].parent
            data = data_roots[preview.source_selector]
            if parent != data and data not in parent.parents:
                raise ValueError("mobile_source_claim_outside_scope")
        for selector in missing:
            selected = selected_inventories[selector]
            budget.container(2 * (len(selected.items) + len(container_roots[selector])))
            budget.container(
                2
                * (
                    len(observations[selector][1])
                    + sum(len(rows) for _, rows in observations[selector][2])
                )
            )
            owned = {
                item.path
                for item in selected.items
                if item.path is not None
                and item.status in {"included", "included_directory"}
            }
            owned.update(container_roots[selector])
            observed = (
                *observations[selector][1],
                *(row for _, rows in observations[selector][2] for row in rows),
            )
            identities = {bootstrap.inode_token_for(row[2]) for row in observed}
            for name, entry in registry.items():
                overlaps = any(
                    bootstrap._overlap(container, Path(path))
                    for container in container_roots[selector]
                    for path in (*entry["roots"], *entry["proposed"])
                )
                aliases = bool(
                    identities.intersection(
                        bootstrap.identity_view(entry["historical"])
                    )
                )
                if not overlaps and not aliases:
                    continue
                if any(name in row["namespaces"] for row in profiles) or any(
                    Path(path) not in owned for path in entry["roots"]
                ):
                    raise ValueError("mobile_enrollment_shared_scope_unproved")
        # All container/source closure above precedes namespace publication.
        for selector in missing:
            selected = selected_inventories[selector]
            budget.value(selected, text_copies=4)
            for item in selected.items:
                if item.path is not None:
                    budget.reserve(path_cost(item.path, copies=8))
            names = set(
                _capture_names(
                    authority,
                    selected,
                    _read_budget=budget,
                    _custody=custody,
                    cancel=cancel,
                )
            ) - {UNBOUND_NAMESPACE}
            registry = bootstrap._registry(root, _budget=budget, _custody=custody)
            for container in container_roots[selector]:
                for name, entry in registry.items():
                    if name in names:
                        continue
                    if any(
                        bootstrap._overlap(container, Path(path))
                        for path in (*entry["roots"], *entry["proposed"])
                    ) or any(
                        token.startswith("path:")
                        and bootstrap._overlap(container, Path(token[5:]))
                        for token in entry["historical"]
                    ):
                        raise ValueError("mobile_enrollment_shared_scope_unproved")
                if any(
                    name in row["namespaces"]
                    and any(
                        bootstrap._overlap(container, Path(path))
                        for path in registry[name]["roots"]
                    )
                    for name in names
                    for row in profiles
                ):
                    raise ValueError("mobile_enrollment_shared_scope_unproved")
                name = (
                    "backup.source."
                    + hashlib.sha256(str(container).encode()).hexdigest()
                )
                authority.register(
                    name,
                    (container,),
                    _read_budget=budget,
                    _custody=custody,
                    cancel=cancel,
                )
                names.add(name)
            names_by_selector[selector] = tuple(sorted(names))
        # Capture namespaces are read/exclusion scopes, not additions to an
        # existing profile's roots. Register before the non-nested maintenance.
        capture_names = _capture_names(
            authority, inventory, _read_budget=budget, _custody=custody, cancel=cancel
        )
        registry = bootstrap._registry(root, _budget=budget, _custody=custody)
        budget.container(len(registry) + len(selectors) + 1)
        names = {UNBOUND_NAMESPACE, *capture_names}
        for selector in selectors:
            names.update(
                names_by_selector.get(
                    selector, existing.get(selector, {}).get("namespaces", ())
                )
            )
        # Native Admission expands aliases too. Explicitly include the current
        # candidate writers whose declared roots overlap the reviewed footprint.
        paths = (*preview.destination.paths, preview.source_dir, *selectors)
        for name, entry in registry.items():
            if any(
                bootstrap._overlap(Path(raw), path)
                for raw in entry["roots"]
                for path in paths
            ):
                names.add(name)
        budget.value(registry, text_copies=2)
        expected_registry = {name: registry[name] for name in names}
        with authority.maintenance(tuple(sorted(names)), 30, cancel=cancel) as session:
            session._scopes.append(custody)
            _attach_mobile_budgets(session, pair)
            _check(cancel)
            current_pending, current_profiles = bootstrap._records(
                root, _budget=budget, _custody=custody
            )
            current_registry = bootstrap._registry(
                root, _budget=budget, _custody=custody
            )
            if (
                current_pending
                or any(
                    current_registry.get(name) != row
                    for name, row in expected_registry.items()
                )
                or current_profiles != profiles
            ):
                raise ValueError("mobile_enrollment_binding_changed")
            with session._discovery_reads(private_sqlite=True, _read_budget=budget):
                current = discover(selectors, _read_budget=budget, _custody=custody)
                fresh = _preview_selected_mobile(
                    preview.source_selector,
                    preview.target_selector,
                    _read_budget=budget,
                )
            if not current.complete:
                raise CaptureReviewRequired(current.issues)
            if fresh != preview or _scope_items(current, budget) != _scope_items(
                inventory, budget
            ):
                raise ValueError("mobile_fresh_review_required")
            for selector in missing:
                selected = _profile_inventory(current, selector, budget)
                actual = (
                    _container(
                        selector.parent,
                        selector,
                        selected,
                        budget,
                        custody,
                        cancel,
                        controls=_fixed_controls(budget, custody),
                    ),
                    _container(
                        data_roots[selector],
                        selector,
                        selected,
                        budget,
                        custody,
                        cancel,
                    ),
                    tuple(
                        (
                            parent,
                            _container(
                                parent, selector, selected, budget, custody, cancel
                            ),
                        )
                        for parent in container_roots[selector][1:]
                    ),
                )
                if actual != observations[selector]:
                    raise ValueError("mobile_enrollment_container_changed")
            for selector in missing:
                bind_profile(
                    root,
                    selector,
                    names_by_selector[selector],
                    authority.control_root,
                    session=session,
                )
            destination = ensure_storage(control_root)
            with session._discovery_reads(private_sqlite=True, _read_budget=budget):
                after = discover(selectors, _read_budget=budget, _custody=custody)
                reviewed = _preview_selected_mobile(
                    preview.source_selector,
                    preview.target_selector,
                    _read_budget=budget,
                )
            if not after.complete:
                raise CaptureReviewRequired(after.issues)
            if reviewed != preview or _scope_items(after, budget) != _scope_items(
                inventory, budget
            ):
                raise ValueError("mobile_fresh_review_required")
            yield session, after, destination
    finally:
        custody.retire()
