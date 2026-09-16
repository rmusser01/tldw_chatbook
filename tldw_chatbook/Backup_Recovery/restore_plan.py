"""Explicit local restore planning (ADR-126); no publication authority."""

import hashlib
import json
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Literal

from tldw_chatbook.Utils.platform_files import os

from .archive_models import Metadata, SealedArchive
from .archive_reader import _hash, _manifest, verify_sealed
from .limits import ArchiveLimits
from .models import Inventory
from .native_files import pinned_directory
from .owner_registry import install_adapters


@dataclass(frozen=True)
class LocalSnapshotSource:
    """Local retained-copy association; verified before selecting snapshot policy."""

    control_root: Path
    operation_id: str
    rollback_digest: str

    def record(self):
        return {
            "control_root": str(self.control_root),
            "operation_id": self.operation_id,
            "rollback_digest": self.rollback_digest,
        }


@dataclass(frozen=True)
class RetainedConfig:
    """Bind one imported config dependency to unchanged, checked local settings.

    This relation authorizes dependency validation and selector interpretation,
    never publication of the local file or substitution of any other owner.
    """

    archive_config_id: str
    target_config_id: str
    config_path: Path
    observation: str

    def record(self):
        return {
            "archive_config_id": self.archive_config_id,
            "target_config_id": self.target_config_id,
            "config_path": str(self.config_path),
            "observation": self.observation,
        }


@dataclass(frozen=True)
class RestorePlan:
    """Reviewed artifact paths and independently observed local target state."""

    archive_digest: str
    mode: str
    restore: tuple[tuple[str, Path], ...]
    retire: tuple[tuple[str, Path], ...]
    preserve: tuple[tuple[str, Path], ...]
    target_fingerprint: str
    destinations: tuple[tuple[str, Path], ...] = ()
    target: Inventory | None = None
    selectors: tuple[tuple[str, Path], ...] = ()
    profile_names: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, Metadata | None, Metadata], ...] = ()
    issues: tuple[str, ...] = ()
    containers: tuple[tuple[str, Path], ...] = ()
    safety_scope: tuple[str, ...] = ()
    acknowledged_credential_issues: tuple[str, ...] = ()
    local_snapshot: LocalSnapshotSource | None = None
    retained_configs: tuple[RetainedConfig, ...] = ()
    requested_groups: tuple[str, ...] | None = None
    effective_groups: tuple[str, ...] = ()
    required_groups: tuple[str, ...] = ()


def retained_config_observation(config_path: Path) -> str:
    """Observe a private local selector eligible for existing or first binding.

    The binding/activation checks apply when acquiring review evidence. Later
    byte rechecks leave interrupted activation reconciliation to its journal.
    Semantic dependency validation still requires a native-held private copy.
    """
    import tomllib

    from .archive_reader import _regular
    from .destinations import _config_tables

    before = _retained_config_observation(config_path)
    with _regular(config_path) as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("retained_config_limit")
    try:
        _config_tables(tomllib.loads(raw.decode("utf-8")))
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("retained_config_invalid") from None
    _retained_config_binding(config_path)
    if _retained_config_observation(config_path) != before:
        raise ValueError("retained_config_changed")
    return before


def _retained_config_binding(config_path: Path) -> dict | None:
    """Read binding eligibility without enrolling or repairing a local profile.

    True first use has neither a prior selector binding nor an activation pair.
    Publication still proves the complete current footprint under native leases.
    """
    from . import bootstrap

    root = bootstrap.default_bootstrap_root()
    _, profiles, activations = bootstrap._control_records(root)
    binding = bootstrap._binding(config_path, profiles, bootstrap._registry(root))
    association = next(
        (row for row in activations if row["selector"] == str(config_path)), None
    )
    if binding is None and any(
        row["selector"] == str(config_path) for row in profiles
    ):
        raise ValueError("retained_config_binding_required")
    if (binding.get("activation") if binding is not None else None) != (
        association["activation"] if association is not None else None
    ):
        raise ValueError("retained_config_activation_changed")
    allowed, _ = bootstrap.startup_permission(config_path, root)
    if not allowed:
        raise ValueError("retained_config_binding_required")
    return binding


def _retained_config_observation(path: Path) -> str:
    """Hash private selector identity, ancestor identities, metadata, and bytes."""
    from .archive_reader import _identity, _regular

    _ancestor(path)
    ancestors = []
    for parent in (path.parent, *path.parent.parents):
        with pinned_directory(parent) as descriptor:
            info = os.fstat(descriptor)
            if parent == path.parent and (
                info.st_uid != os.geteuid() or info.st_mode & 0o077
            ):
                raise ValueError("retained_config_not_private")
            ancestors.append((str(parent), info.st_dev, info.st_ino))
    with _regular(path) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.geteuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError("retained_config_not_private")
        if info.st_size > 16 * 1024**2:
            raise ValueError("retained_config_limit")
        raw = stream.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2:
            raise ValueError("retained_config_limit")
        if _identity(info) != _identity(os.fstat(stream.fileno())) or _identity(
            info
        ) != _identity(os.stat(path, follow_symlinks=False)):
            raise ValueError("retained_config_changed")
    observation = (
        str(path),
        ancestors,
        _identity(info),
        info.st_mode,
        hashlib.sha256(raw).hexdigest(),
    )
    return hashlib.sha256(json.dumps(observation).encode()).hexdigest()


def _retained_config_items(plan, document):
    """Verify the finite config relation against both independent inventories."""
    from dataclasses import replace

    from .models import FileMetadata

    relations = plan.retained_configs
    if not relations:
        return {}
    if (
        type(relations) is not tuple
        or plan.mode != "replace"
        or type(plan.target) is not Inventory
        or any(type(row) is not RetainedConfig for row in relations)
        or len({row.archive_config_id for row in relations}) != len(relations)
        or len({row.target_config_id for row in relations}) != len(relations)
        or len({row.config_path for row in relations}) != len(relations)
    ):
        raise ValueError("retained_config_relation_invalid")
    files = {row.logical_id: row for row in document.files}
    producers = {row.logical_id: row for row in document.producer_inventory}
    locals_by_id = {row.logical_id: row for row in plan.target.items}
    result = {}
    for relation in relations:
        source = files.get(relation.archive_config_id)
        producer = producers.get(relation.archive_config_id)
        item = locals_by_id.get(relation.target_config_id)
        path = relation.config_path
        profile = relation.archive_config_id.split(":")
        if (
            source is None
            or source.owner_id != "config"
            or producer is None
            or producer.owner_id != "config"
            or producer.status != "included"
            or producer.dependencies
            or producer.shared_group
            or len(profile) != 3
            or profile[0] != "profile"
            or profile[2] != "config"
            or profile[1] not in document.profile_ids
            or item is None
            or item.owner != "config"
            or item.status != "included"
            or item.path != path
            or item.dependencies
            or item.shared_group
            or not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            or relation.target_config_id
            != "profile:"
            + hashlib.sha256(str(path).encode()).hexdigest()[:24]
            + ":config"
            or sum(row.path == path for row in plan.target.items) != 1
            or item.metadata is None
            or item.metadata.kind != "file"
            or item.metadata.policy != "private"
            or type(relation.observation) is not str
            or len(relation.observation) != 64
            or any(char not in "0123456789abcdef" for char in relation.observation)
            or any(
                selected == path or selected in path.parents
                for _, selected in (*plan.restore, *plan.retire, *plan.containers)
            )
            or relation.archive_config_id in dict(plan.restore)
        ):
            raise ValueError("retained_config_relation_unverified")
        meta = item.metadata
        result[relation.archive_config_id] = replace(
            item,
            logical_id=relation.archive_config_id,
            metadata=FileMetadata(
                1,
                source.root_id,
                source.relative_path,
                source.parent_id,
                "file",
                meta.mode,
                meta.mtime_ns,
                meta.policy,
            ),
        )
    return result


def recheck_retained_configs(plan: RestorePlan) -> None:
    """Recheck untouched local dependencies even after selected data has moved."""
    for relation in plan.retained_configs:
        try:
            actual = _retained_config_observation(relation.config_path)
        except (OSError, ValueError):
            raise ValueError("retained_config_changed") from None
        if actual != relation.observation:
            raise ValueError("retained_config_changed")


def retained_config_names(plan: RestorePlan) -> tuple[str, ...]:
    """Require all current profile namespaces when staging retained dependencies."""
    from . import bootstrap
    from .control_records import UNBOUND_NAMESPACE

    recheck_retained_configs(plan)
    if not plan.retained_configs:
        return ()
    root = bootstrap.default_bootstrap_root()
    _, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    names = set()
    for relation in plan.retained_configs:
        binding = bootstrap._binding(relation.config_path, profiles, registry)
        if binding is None:
            # First use still requires strict eligibility. Established content
            # scopes remain readable while their own recovery operation is pending.
            binding = _retained_config_binding(relation.config_path)
        if binding is None:
            names.add(UNBOUND_NAMESPACE)
        else:
            names.update(binding["namespaces"])
    return tuple(sorted(names))


def required_rollback_dependencies(plan: RestorePlan) -> tuple[str, ...]:
    """List unselected local dependencies of the reviewed original-copy scope.

    This is a preflight explanation, not capture authority. Native preparation
    still verifies exact original paths, whole trees and selected safety sources.
    """
    if plan.mode != "replace" or plan.target is None:
        return ()
    items = {item.logical_id: item for item in plan.target.items}
    restored = {path for _, path in plan.restore}
    retired = {path for _, path in plan.retire}
    preserved = set(plan.preserve)
    trees = {}
    for item in items.values():
        if item.metadata:
            trees.setdefault((item.owner, item.metadata.root_id), set()).add(
                item.logical_id
            )
    available = set(plan.safety_scope)
    available.update(
        item.logical_id
        for item in items.values()
        if item.path is not None
        and (
            item.path in retired
            or any(parent in retired for parent in item.path.parents)
            or item.path in restored
            and (
                item.status == "included"
                or (item.logical_id, item.path) not in preserved
            )
        )
    )
    needed = set(available)
    pending = list(available)
    while pending:
        item = items.get(pending.pop())
        if item is None:
            continue
        dependencies = set(item.dependencies)
        # A preserved declared tree is reviewed as a complete local source.
        if item.metadata and (item.logical_id, item.path) in preserved:
            dependencies.update(trees[(item.owner, item.metadata.root_id)])
        pending.extend(dependencies - needed)
        needed.update(dependencies)
    return tuple(sorted(needed - available))


def _document(archive):
    verify_sealed(archive)
    return _manifest(archive.manifest_bytes, ArchiveLimits(), encrypted=True)


def _shared_directory_aliases(doc):
    """Match declared concrete roots only when their complete trees agree."""
    producer = {row.logical_id: row for row in doc.producer_inventory}
    directories = {row.logical_id: row for row in doc.directories}
    trees = {
        row.logical_id: []
        for row in doc.directories
        if row.parent_id is None
        and not row.synthetic
        and row.logical_id in producer
        and producer[row.logical_id].shared_group
    }
    for row in (*doc.directories, *doc.files):
        if row.root_id not in trees:
            continue
        owner = producer[row.logical_id]
        directory = row.logical_id in directories
        trees[row.root_id].append(
            (
                row.relative_path,
                "directory" if directory else "file",
                directories[row.parent_id].relative_path if row.parent_id else None,
                owner.owner_id,
                owner.shared_group,
                row.metadata,
                None if directory else (row.size, row.sha256),
            )
        )
    representatives, aliases = {}, {}
    for root, members in trees.items():
        closure = tuple(sorted(members, key=lambda row: row[:2]))
        group = (producer[root].shared_group, closure)
        aliases[root] = representatives.setdefault(group, root)
    return aliases


def _ancestor(path):
    """Check existing path components without accepting a symlink spelling."""
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid_destination")
    selected = path
    while not selected.exists() and not selected.is_symlink():
        selected = selected.parent
    if selected.is_symlink():
        raise ValueError("destination_alias")
    parent = selected if selected.is_dir() else selected.parent
    try:
        with pinned_directory(parent):
            pass
    except OSError:
        raise ValueError("destination_alias") from None
    return parent


def _empty_private_container(path: Path) -> bool:
    """Recognize an existing placement parent without adopting directory data."""
    try:
        with pinned_directory(path) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                return False
            with os.scandir(fd) as entries:
                return next(entries, None) is None
    except OSError:
        return False


def _absent_sqlite_paths(item, owner) -> tuple[Path, ...] | None:
    """Check a declared SQLite absence without opening any database content."""
    policy = owner.schema_policy() if owner is not None else None
    if (
        owner is None
        or owner.owner_id != item.owner
        or policy is None
        or not policy.schema_sql
        or item.status not in {"unused", "missing_required"}
        or item.metadata is not None
    ):
        return None
    try:
        _ancestor(item.path)
        suffixes = ("", "-wal", "-shm", "-journal")
        if item.owner in {"tts.profile_store", "tts.references"}:
            suffixes += (".lock",)
        paths = tuple(Path(str(item.path) + suffix) for suffix in suffixes)
        for path in paths:
            try:
                os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                continue
            return None
    except (OSError, ValueError):
        return None
    return paths


def _tokenizer_root_owner(doc, key, path, owners):
    """Match the authenticated concrete root to the installed default owner."""
    from .config_adapter import _Definition
    from .profile_paths import default_config_path

    owner = "tokenizers.custom"
    installed = owners.get(owner)
    root = next((row for row in doc.directories if row.logical_id == key), None)
    producer = {row.logical_id: row for row in doc.producer_inventory}
    if (
        type(installed) is not _Definition
        or installed
        != _Definition(owner, leaf="tokenizers", location="default_config", tree=True)
        or root is None
        or root.synthetic
        or root.parent_id is not None
        or root.root_id != key
        or key not in producer
        or producer[key].owner_id != owner
        or producer[key].status != "included_directory"
        or path != default_config_path().parent / "tokenizers"
        or any(
            row.logical_id not in producer or producer[row.logical_id].owner_id != owner
            for row in (*doc.directories, *doc.files)
            if row.root_id == key
        )
    ):
        return None
    return owner


def _tokenizer_container_owner(doc, key, path, owners):
    """Require an untouched private empty root before isolated publication."""
    owner = _tokenizer_root_owner(doc, key, path, owners)
    return owner if owner is not None and _empty_private_container(path) else None


def _observed(path):
    ancestor = _ancestor(path)
    ancestors = []
    current = ancestor
    while True:
        info = os.stat(current, follow_symlinks=False)
        ancestors.append((str(current), info.st_dev, info.st_ino))
        if current == current.parent:
            break
        current = current.parent
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return (str(path), ancestors, None)
    if stat.S_ISLNK(info.st_mode) or info.st_nlink != 1 and stat.S_ISREG(info.st_mode):
        raise ValueError("destination_alias")
    if stat.S_ISREG(info.st_mode):
        content = _hash(path, Event())
    elif stat.S_ISDIR(info.st_mode):
        with pinned_directory(path) as fd:
            content = sorted(os.listdir(fd))
    else:
        raise ValueError("unsupported_target_kind")
    return (
        str(path),
        ancestors,
        (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
            content,
        ),
    )


def _fingerprint(paths, target):
    observations = []
    for path in sorted(set(paths)):
        observed = _observed(path)
        items = [item for item in target.items if item.path == path] if target else []
        if (
            len(items) == 1
            and items[0].owner == "recovery.control"
            and (
                items[0].status == "intentionally_excluded"
                and items[0].logical_id
                in {"recovery.control:fixed-bootstrap", "recovery.control:service"}
            )
        ):
            from .inventory import _fixed_control_exclusion, _service_control_exclusion

            discover = (
                _fixed_control_exclusion
                if items[0].logical_id == "recovery.control:fixed-bootstrap"
                else _service_control_exclusion
            )
            current = discover()
            if len(current) != 1 or current[0] != items[0] or observed[2] is None:
                raise ValueError("recovery_control_changed")
            # Native recovery records evolve while this preserved directory stays
            # installed. Payloads retain full content/metadata observations.
            identity = observed[2]
            observed = (
                observed[0],
                observed[1],
                (identity[0], identity[1], identity[5]),
            )
            if items[0].logical_id == "recovery.control:service":
                children = []
                for child in (path / "control", path / "control-work"):
                    child_observed = _observed(child)
                    identity = child_observed[2]
                    if identity is None:
                        raise ValueError("recovery_control_changed")
                    children.append(
                        (
                            child_observed[0],
                            child_observed[1],
                            (identity[0], identity[1], identity[5]),
                        )
                    )
                observed = (*observed, children)
        observations.append(observed)
    document = {
        "paths": observations,
        "inventory": [
            (
                item.owner,
                item.logical_id,
                str(item.path),
                item.status,
                item.dependencies,
                item.shared_group,
            )
            for item in target.items
        ]
        if target is not None
        else [],
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _paths(plan):
    return tuple(
        path
        for _, path in (
            *plan.restore,
            *plan.retire,
            *plan.preserve,
            *plan.selectors,
            *plan.destinations,
            *plan.containers,
            *(
                (item.logical_id, item.path)
                for item in (plan.target.items if plan.target else ())
                if item.logical_id in plan.safety_scope
            ),
        )
    )


def recheck_targets(plan: RestorePlan) -> None:
    """Invalidate reviewed paths; never silently recalculate destination mappings."""
    recheck_retained_configs(plan)
    try:
        actual = _fingerprint(_paths(plan), plan.target)
    except (OSError, ValueError):
        raise ValueError("target_changed") from None
    if actual != plan.target_fingerprint:
        raise ValueError("target_changed")


def plan_restore(
    archive: SealedArchive,
    *,
    mode: Literal["isolated", "replace"],
    destinations: Mapping[str, Path],
    target: Inventory | None,
    profile_names: Mapping[str, str] | None = None,
    safety_scope: tuple[str, ...] = (),
    acknowledged_credential_issues: tuple[str, ...] = (),
    local_snapshot: LocalSnapshotSource | None = None,
    retained_configs: tuple[RetainedConfig, ...] = (),
    data_groups: tuple[str, ...] | None = None,
) -> RestorePlan:
    """Plan explicit local roots and independently checked retained selectors.

    Destination keys are archive root IDs. Values are locally selected roots;
    original relocation strings are not destination defaults or authority.
    """
    if mode == "replace" and target is None:
        raise ValueError("target_unverified")
    if mode not in {"isolated", "replace"}:
        raise ValueError("invalid_restore_mode")
    if (
        type(safety_scope) is not tuple
        or type(acknowledged_credential_issues) is not tuple
        or any(
            type(key) is not str
            for key in (*safety_scope, *acknowledged_credential_issues)
        )
        or len(set(safety_scope)) != len(safety_scope)
        or len(set(acknowledged_credential_issues))
        != len(acknowledged_credential_issues)
        or any(
            not issue.startswith("credential_") or len(issue) > 256
            for issue in acknowledged_credential_issues
        )
        or mode != "replace"
        and (safety_scope or acknowledged_credential_issues)
    ):
        raise ValueError("invalid_replacement_review")
    if (
        type(retained_configs) is not tuple
        or any(type(row) is not RetainedConfig for row in retained_configs)
        or retained_configs
        and mode != "replace"
    ):
        raise ValueError("retained_config_relation_invalid")
    safety_scope = tuple(
        sorted(set(safety_scope) | {row.target_config_id for row in retained_configs})
    )
    if safety_scope:
        by_id = {item.logical_id: item for item in target.items}
        if any(row.target_config_id not in by_id for row in retained_configs):
            raise ValueError("retained_config_relation_unverified")
        if set(safety_scope) - by_id.keys() or any(
            by_id[key].path is None
            or by_id[key].status not in {"included", "included_directory"}
            for key in safety_scope
        ):
            raise ValueError("safety_scope_unverified")
    doc = _document(archive)
    group_scope = None
    group_ids = None
    if data_groups is not None or doc.group_scope is not None:
        from .restore_groups import resolve_archive_groups, selected_archive_ids

        if local_snapshot is not None:
            raise ValueError("local_snapshot_group_selection_invalid")
        group_scope = resolve_archive_groups(
            doc, data_groups, target=target if mode == "replace" else None
        )
        retain_config = mode == "replace" and "settings" not in group_scope.effective_groups
        if retain_config and group_scope.support_ids and not retained_configs:
            raise ValueError("retained_config_relation_required")
        group_ids = selected_archive_ids(doc, group_scope, retain_config=retain_config)
    if mode == "replace" and doc.consistency != "coherent":
        raise ValueError("partial_replacement")
    names = dict(profile_names or {})
    if set(names) - set(doc.profile_ids) or any(
        type(name) is not str or not name.strip() for name in names.values()
    ):
        raise ValueError("invalid_profile_identity")
    owners = {adapter.owner_id: adapter for adapter in install_adapters()}
    from tldw_chatbook.Agents.recovery import _RunLogs
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter
    from tldw_chatbook.Persona_Visual.recovery import _Assets
    from tldw_chatbook.TTS.recovery import _Voices

    from .config_adapter import CONFIG_LOCATION_KEYS

    deferred_classes = {
        "agents.history": _RunLogs,
        "eval.definitions": _DefinitionsAdapter,
        "tts.voices": _Voices,
        "persona.visual_identity_builtin": _Assets,
    }
    deferred = {
        name for name, cls in deferred_classes.items() if type(owners.get(name)) is cls
    }
    directories = {row.logical_id: row for row in doc.directories}
    roots = {key: row for key, row in directories.items() if row.parent_id is None}
    shared_roots = _shared_directory_aliases(doc)
    accepted = {
        f"profile:{profile}:" + ".".join(location)
        for profile in doc.profile_ids
        for location in CONFIG_LOCATION_KEYS
    }
    selectors = {key: value for key, value in destinations.items() if key in accepted}
    destinations = {
        key: value for key, value in destinations.items() if key not in selectors
    }
    if group_ids is not None:
        destinations = {key: value for key, value in destinations.items() if key in group_ids}
    retirement_only_selection = (
        mode == "replace"
        and group_scope is not None
        and not group_ids
        and bool(retained_configs)
    )
    if (
        not destinations and local_snapshot is None and not retirement_only_selection
        or set(destinations) - roots.keys()
    ):
        raise ValueError("explicit_destination_required")
    for path in selectors.values():
        _ancestor(path)
        if mode == "isolated" and path.exists():
            raise ValueError("destination_exists")
    for key, path in destinations.items():
        _ancestor(path)
        if (
            mode == "isolated"
            and (path.exists() or path.is_symlink())
            and not (
                roots[key].synthetic
                and _empty_private_container(path)
                or _tokenizer_container_owner(doc, key, path, owners) is not None
            )
        ):
            raise ValueError("destination_exists")
    normalized = [
        unicodedata.normalize("NFC", str(path)).casefold()
        for path in destinations.values()
    ]
    if len(set(normalized)) != len(normalized):
        for key, path in destinations.items():
            for other, other_path in destinations.items():
                if key == other:
                    continue
                if (
                    unicodedata.normalize("NFC", str(path)).casefold()
                    == unicodedata.normalize("NFC", str(other_path)).casefold()
                ) and (
                    path != other_path
                    or not (
                        roots[key].synthetic
                        and roots[other].synthetic
                        or key in shared_roots
                        and shared_roots[key] == shared_roots.get(other)
                    )
                ):
                    raise ValueError("destination_collision")
    for key, left in destinations.items():
        normalized_left = Path(unicodedata.normalize("NFC", str(left)).casefold())
        for other, right in destinations.items():
            normalized_right = Path(unicodedata.normalize("NFC", str(right)).casefold())
            if key == other or normalized_left not in normalized_right.parents:
                continue
            if left not in right.parents or not roots[key].synthetic:
                raise ValueError("destination_overlap")
            # A synthetic root owns selected files, not the containing directory.
            # Its actual members must remain disjoint from the other owner tree.
            for row in (*doc.directories, *doc.files):
                if row.root_id != key or row.logical_id == key:
                    continue
                member = Path(
                    unicodedata.normalize(
                        "NFC", str(left / row.relative_path)
                    ).casefold()
                )
                if (
                    member == normalized_right
                    or member in normalized_right.parents
                    or normalized_right in member.parents
                ):
                    raise ValueError("destination_collision")
    records = (*doc.directories, *doc.files)
    selected = {row.logical_id for row in records if row.root_id in destinations}
    if group_ids is not None and group_ids - selected:
        raise ValueError("data_group_destination_missing")
    if group_ids is not None and selected - group_ids:
        raise ValueError("data_group_root_not_independent")
    retained_template = RestorePlan(
        archive.digest,
        mode,
        tuple(
            (row.logical_id, destinations[row.root_id] / row.relative_path)
            for row in records
            if row.logical_id in selected
            and not (
                row.logical_id in directories and directories[row.logical_id].synthetic
            )
        ),
        (),
        (),
        "",
        target=target,
        retained_configs=retained_configs,
    )
    retained_items = _retained_config_items(retained_template, doc)
    recheck_retained_configs(retained_template)
    for relation in retained_configs:
        if retained_config_observation(relation.config_path) != relation.observation:
            raise ValueError("retained_config_changed")
    empty_owner_retirements = frozenset()
    if retirement_only_selection:
        from .restore_groups import selected_absent_target_ids

        empty_owner_retirements = selected_absent_target_ids(
            doc, group_scope, target, (), retained_configs
        )
        if not empty_owner_retirements:
            raise ValueError("explicit_destination_required")
    selected_owners = {row.owner_id for row in doc.files if row.logical_id in selected}
    if selected_owners - owners.keys():
        raise ValueError("unsupported_owner")
    if doc.required_capabilities:
        raise ValueError("unsupported_restore_capability")
    for owner in doc.owners:
        if owner.owner_id not in selected_owners:
            continue
        policy = owners[owner.owner_id].schema_policy()
        if owner.capabilities:
            raise ValueError("unsupported_restore_capability")
        if policy is None or owner.schema_version not in policy.versions:
            raise ValueError("unsupported_schema_version")
    preserved_snapshot = {}
    if local_snapshot is not None:
        from .later_rollback import _preserved_snapshot_members, _verify_snapshot_source

        _verify_snapshot_source(local_snapshot, mode, archive.digest, archive)
        preserved_snapshot = _preserved_snapshot_members(
            local_snapshot, archive, target
        )
        if any(
            item.logical_id not in safety_scope or key in selected
            for key, item in preserved_snapshot.items()
        ):
            raise ValueError("local_snapshot_preservation_unverified")
    if not destinations and not empty_owner_retirements:
        # The authenticated safety archive also contains fixed credential
        # control material, consumed separately by staging and never published.
        control_files = {
            row.logical_id
            for row in doc.files
            if local_snapshot is not None
            and doc.credential_policy == "rollback"
            and row.logical_id == "credentials"
            and row.owner_id == "recovery.credentials"
            and row.payload == "payload/credential-recovery.json"
            and row.relative_path == "credential-recovery.json"
            and row.parent_id == row.root_id
            and roots[row.root_id].synthetic
            and sum(member.root_id == row.root_id for member in records) == 2
        }
        if not {
            row.logical_id
            for row in records
            if row.logical_id not in directories
            or not directories[row.logical_id].synthetic
        } <= preserved_snapshot.keys() | control_files:
            raise ValueError("explicit_destination_required")
    from .restore_groups import directed_dependency_groups

    directed = directed_dependency_groups(doc)
    for group in doc.dependency_groups:
        needed = (
            directed[group.group_id] in selected
            if group.group_id in directed
            else bool(selected.intersection(group.members))
        )
        if needed and (
            not group.complete
            or not set(group.members)
            <= selected | preserved_snapshot.keys() | retained_items.keys()
        ):
            raise ValueError("dependency_group_incomplete")
    restore = tuple(
        sorted(
            (row.logical_id, destinations[row.root_id] / row.relative_path)
            for row in records
            if row.logical_id in selected
        )
    )
    producer = {row.logical_id: row for row in doc.producer_inventory}
    payloads = {row.logical_id: row for row in doc.files}
    projection_roots = {
        row.root_id
        for row in doc.files
        if row.logical_id in selected and row.owner_id == "rag.projections"
    }
    for key in projection_roots:
        owner = producer.get(key)
        if (
            owner is None
            or owner.owner_id != "rag.projections"
            or roots[key].synthetic
            or any(
                row.owner_id != "rag.projections"
                for row in doc.files
                if row.root_id == key
            )
        ):
            raise ValueError("projection_root_incomplete")
    deferred_selected = {
        row.owner_id
        for row in doc.files
        if row.logical_id in selected and row.owner_id in deferred
    }
    deferred_selected.update(
        row.owner_id
        for row in producer.values()
        if row.logical_id in selected and row.owner_id in deferred
    )
    for record in records:
        item = producer.get(record.logical_id)
        owner = item.owner_id if item else getattr(record, "owner_id", None)
        if (
            record.logical_id in selected
            and owner in deferred_selected
            and destinations[record.root_id].exists()
        ):
            raise ValueError("owner_setup_destination_exists")
    for root_id, destination in destinations.items():
        item = producer.get(root_id)
        if item is None and not any(row.root_id == root_id for row in doc.files):
            raise ValueError("producer_inventory_required")
        external = (
            item is not None
            and item.owner_id == "external.files"
            or any(
                row.root_id == root_id and row.owner_id == "external.files"
                for row in doc.files
            )
        )
        if external and destination.exists():
            raise ValueError("external_destination_exists")
    seen_paths = {}
    trimmed = []
    for key, path in restore:
        directory = directories.get(key)
        if directory is not None and directory.synthetic and path.exists():
            continue  # Container metadata is not ownership of an existing parent.
        norm = unicodedata.normalize("NFC", str(path)).casefold()
        if norm in seen_paths:
            previous = seen_paths[norm]
            previous_directory = directories.get(previous)
            if (
                directory is not None
                and previous_directory is not None
                and key in producer
                and previous in producer
                and producer[key].shared_group
                and producer[key].shared_group == producer[previous].shared_group
                and directory.root_id in shared_roots
                and shared_roots[directory.root_id]
                == shared_roots.get(previous_directory.root_id)
                and path == dict(restore)[previous]
            ):
                trimmed.append((key, path))
                continue
            if (
                key in payloads
                and previous in payloads
                and key in producer
                and previous in producer
            ):
                group = producer[key].shared_group
                if (
                    group
                    and group == producer[previous].shared_group
                    and path == dict(restore)[previous]
                ):
                    if payloads[key].metadata != payloads[previous].metadata:
                        raise ValueError("shared_metadata_mismatch")
                    trimmed.append((key, path))
                    continue
            if (
                directory is None
                or not directory.synthetic
                or previous not in roots
                or not roots[previous].synthetic
            ):
                raise ValueError("destination_collision")
            continue
        seen_paths[norm] = key
        trimmed.append((key, path))
    restore = tuple(trimmed)
    producer = {row.logical_id: row for row in getattr(doc, "producer_inventory", ())}
    for key in selected:
        item = producer.get(key)
        if item is not None:
            if item.owner_id not in owners:
                raise ValueError("unsupported_owner")
            if item.shared_group and any(
                row.logical_id not in selected
                for row in producer.values()
                if row.shared_group == item.shared_group
            ):
                raise ValueError("shared_scope_expansion_required")
    mapped = dict(restore)
    shared_destinations = {}
    for key in selected:
        item = producer.get(key)
        if item and item.shared_group and key in mapped:
            shared_destinations.setdefault(item.shared_group, set()).add(mapped[key])
    if any(len(paths) != 1 for paths in shared_destinations.values()):
        raise ValueError("shared_target_split")
    pending = list(selected)
    visited = set()
    while pending:
        key = pending.pop()
        if key in visited or key not in producer:
            continue
        visited.add(key)
        for dependency in producer[key].dependencies:
            if (
                dependency not in selected
                and dependency not in preserved_snapshot
                and dependency not in retained_items
            ):
                raise ValueError("dependency_group_incomplete")
            pending.append(dependency)
    # Config selectors are installed typed fields. No original value is a key.
    from .profile_paths import DATABASE_PATHS

    for payload in doc.files:
        if payload.logical_id not in selected or payload.owner_id != "config":
            continue
        parts = payload.logical_id.split(":")
        if len(parts) != 3 or parts[0] != "profile" or parts[1] not in doc.profile_ids:
            raise ValueError("config_profile_unverified")
        profile = parts[1]
        if profile not in names:
            raise ValueError("profile_identity_required")
        if mode == "isolated" and f"profile:{profile}:paths.data_dir" not in selectors:
            raise ValueError("config_data_destination_required")
        for owner_id, setting, _, _ in DATABASE_PATHS:
            matches = [
                path for key, path in restore if key == f"profile:{profile}:{owner_id}"
            ]
            if len(matches) == 1:
                key = f"profile:{profile}:database.{setting}"
                if key in selectors and selectors[key] != matches[0]:
                    raise ValueError("ambiguous_relocation_mapping")
                selectors[key] = matches[0]
    if mode == "replace" and not producer:
        raise ValueError("producer_inventory_required")
    retire, preserve = [], []
    absent_ids = empty_owner_retirements
    if mode == "replace" and group_scope is not None and not absent_ids:
        from .restore_groups import selected_absent_target_ids

        absent_ids = selected_absent_target_ids(
            doc, group_scope, target, restore, retained_configs
        )
    if target is not None:
        if type(target) is not Inventory or len(
            {item.logical_id for item in target.items}
        ) != len(target.items):
            raise ValueError("target_unverified")
        replacements = dict(restore)
        selected_owners = {
            producer[key].owner_id for key in selected if key in producer
        }
        selected_owners.update(item.owner for item in target.items if item.logical_id in absent_ids)
        by_target_id = {item.logical_id: item for item in target.items}
        for item in target.items:
            if item.path is None:
                if item.status not in {
                    "unused",
                    "intentionally_excluded",
                    "intentionally_deleted",
                }:
                    raise ValueError("target_unverified")
                continue
            _ancestor(item.path)
            if (
                item.status == "included"
                and not item.path.is_file()
                or item.status == "included_directory"
                and not item.path.is_dir()
            ):
                raise ValueError("target_unverified")
            if mode == "replace" and item.owner == "sqlite.transient":
                if len(item.dependencies) != 1:
                    raise ValueError("target_owner_unclassified")
                main = by_target_id.get(item.dependencies[0])
                adapter = owners.get(main.owner) if main else None
                policy = adapter.schema_policy() if adapter else None
                if (
                    main is None
                    or main.path is None
                    or policy is None
                    or not policy.schema_sql
                    or item.path
                    not in {
                        main.path.with_name(main.path.name + suffix)
                        for suffix in ("-wal", "-shm")
                    }
                    or item.status != "intentionally_excluded"
                ):
                    raise ValueError("target_owner_unclassified")
                if (main.path in replacements.values() or main.logical_id in absent_ids) and item.path.exists():
                    retire.append((item.logical_id, item.path))
                else:
                    preserve.append((item.logical_id, item.path))
                continue
            affected = item.logical_id in absent_ids or item.path in replacements.values() or any(
                not roots[key].synthetic
                and (item.path == root or root in item.path.parents)
                for key, root in destinations.items()
            )
            if mode == "isolated":
                if affected or any(
                    item.path in root.parents for root in destinations.values()
                ):
                    raise ValueError("source_destination_alias")
                preserve.append((item.logical_id, item.path))
                continue
            if not affected:
                preserve.append((item.logical_id, item.path))
                continue
            # A selected file may create an installed SQL store whose complete
            # footprint is absent. Authenticated snapshots also need this for undo.
            create_missing_sqlite = (
                item.status == "missing_required"
                and (group_scope is not None or local_snapshot is not None)
                and any(
                    row.logical_id in replacements
                    and replacements[row.logical_id] == item.path
                    and row.owner_id == item.owner
                    and row.logical_id in producer
                    and producer[row.logical_id].owner_id == item.owner
                    for row in doc.files
                )
                and _absent_sqlite_paths(item, owners.get(item.owner)) is not None
            )
            if item.owner not in owners or (
                item.status in {"unsupported", "unavailable", "missing_required"}
                and not create_missing_sqlite
            ):
                raise ValueError("target_owner_unclassified")
            archived_item = producer.get(item.logical_id)
            if item.owner != "rag.projections" and (
                item.status == "intentionally_excluded"
                or archived_item is not None
                and archived_item.status == "intentionally_excluded"
            ):
                if item.path in replacements.values():
                    raise ValueError("preserved_target_collision")
                preserve.append((item.logical_id, item.path))
                continue
            if item.shared_group:
                aliases = [
                    entry
                    for entry in target.items
                    if entry.shared_group == item.shared_group
                ]
                if any(
                    entry.path is not None
                    and entry.logical_id not in absent_ids
                    and not any(
                        entry.path == root or root in entry.path.parents
                        for root in destinations.values()
                    )
                    for entry in aliases
                ):
                    raise ValueError("shared_scope_expansion_required")
            if item.path in replacements.values():
                matching = [key for key, path in restore if path == item.path]
                if not any(
                    key in producer and producer[key].owner_id == item.owner
                    for key in matching
                ):
                    raise ValueError("target_owner_mismatch")
            elif item.status in {"included", "included_directory"}:
                if item.owner not in selected_owners:
                    raise ValueError("target_owner_unclassified")
                # An explicit producer tree describes the selected owner's full
                # generation; a standalone file never authorizes sibling deletion.
                if item.logical_id not in absent_ids and not any(
                    key in producer
                    and producer[key].owner_id == item.owner
                    and not getattr(roots[key], "synthetic", False)
                    and roots[key].logical_id in selected
                    and (
                        item.path == destinations[key]
                        or destinations[key] in item.path.parents
                    )
                    for key in destinations
                ):
                    raise ValueError("retirement_unclassified")
                retire.append((item.logical_id, item.path))
        declared = {item.path for item in target.items if item.path is not None}
        for _, path in restore:
            if path.exists() and path not in declared:
                raise ValueError("target_unverified")
        # Unknown children are never inferred garbage inside a managed directory.
        for item in target.items:
            if (
                item.path
                and item.path.is_dir()
                and (item.logical_id in absent_ids or any(
                    not roots[key].synthetic
                    and (item.path == root or root in item.path.parents)
                    for key, root in destinations.items()
                ))
                and any(child not in declared for child in item.path.iterdir())
            ):
                raise ValueError("target_owner_unclassified")
    projection_issues = ()
    if mode == "replace" and target is not None:
        from .projection_publication import dependent_retirements

        retire, preserve, projection_issues = dependent_retirements(
            target, restore, retire, preserve, safety_scope=safety_scope,
            preserve_config_dependents=(
                group_scope is not None and "retrieval" not in group_scope.effective_groups
            ),
        )
    projection_issues += tuple(
        "projection_reconciliation_required:" + row.logical_id
        for row in doc.directories
        if row.logical_id in selected
        and row.logical_id in producer
        and producer[row.logical_id].owner_id == "rag.projections"
        and row.parent_id is None
    )
    plan = RestorePlan(
        archive.digest,
        mode,
        restore,
        tuple(sorted(retire)),
        tuple(sorted(preserve)),
        "",
        tuple(sorted(destinations.items())),
        target,
        tuple(sorted(selectors.items())),
        tuple(sorted(names.items())),
        safety_scope=tuple(sorted(safety_scope)),
        acknowledged_credential_issues=tuple(sorted(acknowledged_credential_issues)),
        local_snapshot=local_snapshot,
        retained_configs=retained_configs,
    )
    _retained_config_items(plan, doc)
    if group_scope is not None:
        from dataclasses import replace

        from .restore_groups import check_preserved_group_effects

        plan = replace(plan, requested_groups=group_scope.requested_groups,
                       effective_groups=group_scope.effective_groups,
                       required_groups=group_scope.required_groups)
        check_preserved_group_effects(plan, group_scope)
    if any(
        (row.target_config_id, row.config_path) not in plan.preserve
        for row in retained_configs
    ):
        raise ValueError("retained_config_preservation_unverified")
    if any(
        (item.logical_id, item.path) not in plan.preserve
        or item.path in dict(plan.restore).values()
        for item in preserved_snapshot.values()
    ):
        raise ValueError("local_snapshot_preservation_unverified")
    from dataclasses import replace

    metadata, issues = (
        [],
        ["owner_setup_required:" + owner for owner in sorted(deferred_selected)]
        + sorted(set(projection_issues)),
    )
    restored_ids = {key for key, _ in restore}
    for record in records:
        if record.logical_id not in restored_ids:
            continue
        desired = record.metadata
        private_mode = 0o700 if record.logical_id in directories else 0o600
        if desired is None:
            applied = Metadata(version=1, mode=private_mode, mtime_ns=0)
            issues.append("metadata_unavailable:" + record.logical_id)
        else:
            applied = Metadata(
                version=1,
                mode=(desired.mode & 0o700) | private_mode,
                mtime_ns=desired.mtime_ns,
            )
            if applied != desired:
                issues.append("metadata_normalized:" + record.logical_id)
        metadata.append((record.logical_id, desired, applied))
    parents = set()
    root_paths = set(destinations.values())
    if mode == "isolated":
        supplied_directories = {path for key, path in restore if key in directories}
        for key, path in selectors.items():
            if key.endswith(":paths.data_dir") and path not in supplied_directories:
                if any(root in path.parents for root in root_paths):
                    raise ValueError("isolated_data_container_overlap")
                parents.add(path)
    for root in root_paths | parents.copy():
        parent = root.parent
        while not parent.exists():
            if parent not in root_paths and not any(
                selected in parent.parents and not selected.exists()
                for selected in root_paths
            ):
                parents.add(parent)
            parent = parent.parent
    plan = replace(
        plan,
        containers=tuple(
            ("local:" + hashlib.sha256(str(path).encode()).hexdigest(), path)
            for path in sorted(parents, key=lambda path: (len(path.parts), str(path)))
        ),
    )
    # Check the completed mutation plan before accepting internal service work
    # beneath an existing synthetic container. Every other archive alias refuses.
    for key, path in plan.destinations:
        if (
            path == archive.path
            or path in archive.path.parents
            or archive.path.parent in path.parents
        ):
            from .service_storage import is_preserved_service_container

            if not is_preserved_service_container(
                archive.path, path, plan, synthetic=roots[key].synthetic
            ):
                raise ValueError("archive_destination_alias")
    plan = replace(
        plan,
        target_fingerprint=_fingerprint(_paths(plan), target),
        metadata=tuple(metadata),
        issues=tuple(issues),
    )
    if target is not None and any(
        item.logical_id in absent_ids and item.metadata is None
        for item in target.items
    ):
        # Bind current SQL locator validation to the completed fingerprint.
        # Existing stage/held rechecks then require these same selector bytes,
        # file identities, and ancestors; later edits need a fresh review.
        if selected_absent_target_ids(
            doc, group_scope, target, restore, retained_configs
        ) != absent_ids:
            raise ValueError("selected_absence_mapping_required")
        recheck_targets(plan)
    return plan
