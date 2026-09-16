"""Resolve local profile choices through the installed owner relocation policy."""

import tomllib
import unicodedata
import zipfile
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from . import archive_reader
from .config_adapter import (
    CONFIG_LOCATION_KEYS,
    config_location,
    remap_config_locations,
)
from .owner_registry import install_adapters
from .profile_paths import (
    DATABASE_PATHS,
    custom_database_input,
    data_base,
    database_path,
    user_data_dir,
)
from .restore_plan import _ancestor, _shared_directory_aliases, plan_restore


def _deferred_owners(owners):
    """Keep the planner's installed deferred-owner allowlist unchanged."""
    from tldw_chatbook.Agents.recovery import _RunLogs
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter
    from tldw_chatbook.Persona_Visual.recovery import _Assets
    from tldw_chatbook.TTS.recovery import _Voices

    return {
        name
        for name, cls in {
            "agents.history": _RunLogs,
            "eval.definitions": _DefinitionsAdapter,
            "tts.voices": _Voices,
            "persona.visual_identity_builtin": _Assets,
        }.items()
        if type(owners.get(name)) is cls
    }


def requires_setup_destination(doc, selected_ids=None):
    """Identify deferred content needing an independent replacement destination."""
    deferred = _deferred_owners({owner.owner_id: owner for owner in install_adapters()})
    records = {row.logical_id for row in (*doc.files, *doc.directories)}
    return any(
        row.owner_id in deferred and row.logical_id in records
        and (selected_ids is None or row.logical_id in selected_ids)
        for row in (*doc.files, *doc.producer_inventory)
    )


def check_setup_parent(parent, protected):
    """Require an existing private parent outside managed/control storage."""
    from tldw_chatbook.Utils.platform_files import os

    from .bootstrap import _overlap
    from .native_files import pinned_directory

    if parent is None:
        raise ValueError("restore_setup_parent_required")
    _ancestor(parent)
    if not parent.is_dir() or any(_overlap(parent, path) for path in protected):
        raise ValueError("restore_setup_parent_required")
    with pinned_directory(parent) as fd:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("restore_setup_parent_required")


def destination_slots(doc):
    """Expose profile bases and independent external content, never source paths."""
    configured = {
        row.logical_id.split(":")[1]
        for row in doc.files
        if row.owner_id == "config"
        and row.logical_id.startswith("profile:")
        and len(row.logical_id.split(":")) == 3
    }
    profiles = tuple(profile for profile in doc.profile_ids if profile in configured)
    producer = {row.logical_id: row for row in doc.producer_inventory}
    external = tuple(
        row
        for row in doc.directories
        if row.parent_id is None
        and row.logical_id in producer
        and producer[row.logical_id].owner_id == "external.files"
    )
    return tuple(
        MappingProxyType(
            {
                "logical_id": profile,
                "kind": "profile_base",
                "label": f"Profile {index + 1}",
                "owners": ("config",),
                "group_ids": (),
            }
        )
        for index, profile in enumerate(profiles)
    ) + tuple(
        MappingProxyType(
            {
                "logical_id": row.logical_id,
                "kind": "external_root",
                "label": f"External folder {index + 1}",
                "owners": ("external.files",),
                "group_ids": ("external_files",),
            }
        )
        for index, row in enumerate(external)
    )


def _config_tables(data):
    """Reject scalar tables before reading installed configuration selectors."""
    for location in CONFIG_LOCATION_KEYS:
        config_location(data, location)
    for path in (
        ("general",),
        ("AppRAGSearchConfig",),
        ("AppRAGSearchConfig", "rag"),
        ("AppRAGSearchConfig", "rag", "vector_store"),
        ("AppRAGSearchConfig", "rag", "chroma"),
    ):
        table = data
        for key in path:
            table = table.get(key, {})
            if type(table) is not dict:
                raise ValueError("invalid_config_shape")
    return data


def check_isolated_parents(plan, control_root):
    """Apply the existing publication parent gate without creating directories."""
    from tldw_chatbook.Utils.platform_files import os

    from . import bootstrap
    from .native_files import pinned_directory

    root = bootstrap.default_bootstrap_root()
    ancestors = {_ancestor(path) for _, path in (*plan.destinations, *plan.selectors)}
    if any(
        bootstrap._overlap(parent, control_root) or bootstrap._overlap(parent, root)
        for parent in ancestors
    ):
        raise ValueError("isolated_destination_parent_overlaps_control")
    for parent in sorted(ancestors):
        with pinned_directory(parent) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("private_destination_parent_required")
    return tuple(sorted(ancestors))


def _configs(archive, doc, selected=None):
    """Read only bounded verified configuration payloads from the sealed copy."""
    result = {}
    with (
        archive_reader._regular(archive.path) as stream,
        zipfile.ZipFile(stream) as container,
    ):
        for row in doc.files:
            if (
                row.owner_id != "config"
                or selected is not None
                and row.logical_id not in selected
            ):
                continue
            identity = row.logical_id.split(":")
            if (
                len(identity) != 3
                or identity[0] != "profile"
                or identity[1] not in doc.profile_ids
                or identity[1] in result
            ):
                raise ValueError("config_profile_unverified")
            if row.size > 16 * 1024**2:
                raise ValueError("config_validation_unavailable")
            result[row.logical_id.split(":")[1]] = (
                row,
                _config_tables(
                    tomllib.loads(container.read(row.payload).decode("utf-8"))
                ),
            )
    archive_reader.verify_sealed(archive)
    return result


def retained_config_relations(doc, target, target_configs, data_groups):
    """Bind explicit local config choices for selective low-level destinations."""
    from .restore_groups import resolve_archive_groups
    from .restore_plan import RetainedConfig, retained_config_observation

    if target is None:
        raise ValueError("target_unverified")
    scope = resolve_archive_groups(doc, data_groups, target=target)
    if "settings" in scope.effective_groups:
        return ()
    required = {key.split(":")[1]: key for key in scope.support_ids}
    if set(target_configs) != set(required):
        raise ValueError("retained_config_relation_required")
    relations = []
    for profile, archive_id in sorted(required.items()):
        path = target_configs[profile]
        matches = [item for item in target.items if item.owner == "config"
                   and item.path == path and item.status == "included"]
        if len(matches) != 1:
            raise ValueError("target_unverified")
        relations.append(RetainedConfig(archive_id, matches[0].logical_id, path,
                                        retained_config_observation(path)))
    return tuple(relations)


def check_config_destinations(archive, plan, *, session=None):
    """Run the staging owner-location gate during review without writing files."""
    if plan.local_snapshot is not None and "settings" not in plan.effective_groups:
        return
    from .staging import _config_targets

    doc = archive_reader.verify_sealed(archive)
    selected = dict(plan.restore)
    owners = {owner.owner_id: owner for owner in install_adapters()}
    for profile, (row, data) in _configs(archive, doc, selected).items():
        if row.logical_id not in selected:
            continue
        if plan.local_snapshot is None:
            prefix = f"profile:{profile}:"
            mapping = {
                key[len(prefix) :]: value
                for key, value in plan.selectors
                if key.startswith(prefix)
            }
            data = remap_config_locations(data, mapping)
            data.setdefault("general", {})["users_name"] = dict(plan.profile_names)[profile]
            _config_targets(data, profile, selected[row.logical_id], doc, plan, owners)
        if plan.mode == "replace" and "settings" in plan.effective_groups:
            from .preserved_groups import check_preserved_group_paths
            from .storage_admission import _preview_reads

            with session._discovery_reads(private_sqlite=True) if session is not None else _preview_reads():
                check_preserved_group_paths(plan, data, selected[row.logical_id])


def _new_bases(bases):
    """Reject existing or overlapping profile bases before deriving child paths."""
    normalized = []
    for base in bases.values():
        _ancestor(base)
        if base.exists() or base.is_symlink():
            raise ValueError("destination_exists")
        norm = Path(unicodedata.normalize("NFC", str(base)).casefold())
        if any(
            norm == previous or norm in previous.parents or previous in norm.parents
            for previous in normalized
        ):
            raise ValueError("destination_overlap")
        normalized.append(norm)


def resolve_destinations(
    archive,
    *,
    mode,
    profile_bases,
    external_destinations,
    target,
    profile_names,
    target_configs=None,
    setup_parent=None,
    data_groups=None,
    **review,
):
    """Derive roots, then submit them to the explicit low-level restore planner."""
    from .staging import _config_targets

    doc = archive_reader.verify_sealed(archive)
    scope = None
    selected_ids = None
    retain_config = False
    if data_groups is not None or doc.group_scope is not None:
        from .restore_groups import resolve_archive_groups, selected_archive_ids

        scope = resolve_archive_groups(doc, data_groups, target=target if mode == "replace" else None)
        retain_config = mode == "replace" and "settings" not in scope.effective_groups
        selected_ids = selected_archive_ids(doc, scope, retain_config=retain_config)
    configs = _configs(archive, doc)
    owners = {owner.owner_id: owner for owner in install_adapters()}
    deferred = _deferred_owners(owners)
    target_configs = dict(target_configs or {})
    names = dict(profile_names or {})
    if mode == "isolated":
        if (
            set(profile_bases) != set(configs)
            or set(names) != set(configs)
            or any(
                not isinstance(name, str) or not name.strip() for name in names.values()
            )
        ):
            raise ValueError("profile_identity_required")
        _new_bases(profile_bases)
    elif mode == "replace":
        if target is None or set(target_configs) != set(configs):
            raise ValueError("target_unverified")
    else:
        raise ValueError("invalid_restore_mode")
    if mode == "replace" and requires_setup_destination(doc, selected_ids):
        from . import bootstrap, profile_paths

        check_setup_parent(
            setup_parent,
            (
                archive.path.parent,
                bootstrap.default_bootstrap_root(),
                profile_paths.default_config_path().parent,
                profile_paths.default_base_data_dir(),
                *(item.path for item in target.items if item.path is not None),
            ),
        )
    external_ids = {
        slot["logical_id"]
        for slot in destination_slots(doc)
        if slot["kind"] == "external_root"
        and (selected_ids is None or slot["logical_id"] in selected_ids)
    }
    if set(external_destinations) != external_ids:
        raise ValueError("explicit_destination_required")
    destinations = dict(external_destinations)
    producer = {row.logical_id: row for row in doc.producer_inventory}
    shared_databases = {}
    shared_directories = _shared_directory_aliases(doc)
    inert_destinations = {}
    retained_configs = []
    for profile, (config_record, data) in configs.items():
        prefix = f"profile:{profile}:"
        if mode == "isolated":
            base = profile_bases[profile]
            config_target = base / "config" / Path(config_record.relative_path).name
            data_target = base / "data"
            user_root = user_data_dir(
                {
                    "paths": {"data_dir": str(data_target)},
                    "general": {"users_name": names[profile]},
                }
            )
            mapping = {
                ".".join(location): user_root.joinpath(*location)
                for location in CONFIG_LOCATION_KEYS
                if location[0] != "database" and config_location(data, location)
            }
            mapping.update(
                {"paths.data_dir": data_target, "Paths.data_dir": data_target}
            )
            for owner, setting, leaf, legacy_leaf in DATABASE_PATHS:
                row = next(
                    (row for row in doc.files if row.logical_id == prefix + owner), None
                )
                legacy = (
                    "~/.local/share/tldw_cli/" + legacy_leaf if legacy_leaf else None
                )
                if (
                    row is None
                    and custom_database_input(
                        data.get("database", {}).get(setting), legacy
                    )
                    is None
                ):
                    continue
                path = user_root / (Path(row.relative_path).name if row else leaf)
                declaration = producer.get(row.logical_id) if row else None
                if declaration and declaration.shared_group:
                    path = shared_databases.setdefault(declaration.shared_group, path)
                mapping["database." + setting] = path
        else:
            config_target = target_configs[profile]
            if not any(
                item.owner == "config"
                and item.path == config_target
                and item.status == "included"
                for item in target.items
            ):
                raise ValueError("target_unverified")
            _ancestor(config_target)
            with archive_reader._regular(config_target) as stream:
                raw = stream.read(16 * 1024**2 + 1)
            if len(raw) > 16 * 1024**2:
                raise ValueError("config_validation_unavailable")
            try:
                local = tomllib.loads(raw.decode("utf-8"))
            except (tomllib.TOMLDecodeError, UnicodeError):
                if retain_config:
                    raise ValueError("retained_config_invalid") from None
                # Corrupt configurations are recoverable only from independently
                # classified local database paths with one unambiguous data folder.
                target_profile = next(
                    item.logical_id.split(":")[1]
                    for item in target.items
                    if item.owner == "config" and item.path == config_target
                )
                observed = {
                    item.owner: item.path
                    for item in target.items
                    if item.logical_id.startswith(f"profile:{target_profile}:")
                    and item.path is not None
                    and item.status == "included"
                }
                database_settings = {
                    setting: str(observed[owner])
                    for owner, setting, _, _ in DATABASE_PATHS
                    if owner in observed
                }
                parents = {Path(path).parent for path in database_settings.values()}
                if len(parents) != 1:
                    raise ValueError("target_unverified") from None
                user_root = parents.pop()
                from .profile_paths import user_folder_name

                if user_folder_name(user_root.name) != user_root.name:
                    raise ValueError("target_unverified") from None
                local = {
                    "general": {"users_name": user_root.name},
                    "paths": {"data_dir": str(user_root.parent)},
                    "database": database_settings,
                }
            _config_tables(local)
            if retain_config:
                from .restore_plan import RetainedConfig, retained_config_observation

                target_config = next(
                    item for item in target.items
                    if item.owner == "config" and item.path == config_target
                    and item.status == "included"
                )
                retained_configs.append(RetainedConfig(
                    config_record.logical_id, target_config.logical_id,
                    config_target, retained_config_observation(config_target),
                ))
                data = local
            names[profile] = local.get("general", {}).get("users_name", "default_user")
            data_target = data_base(local)
            user_root = user_data_dir(local)
            base = config_target.parent
            mapping = {
                ".".join(location): user_root.joinpath(*location)
                for location in CONFIG_LOCATION_KEYS
                if location[0] != "database"
                and (
                    config_location(data, location) or config_location(local, location)
                )
            }
            mapping.update(
                {"paths.data_dir": data_target, "Paths.data_dir": data_target}
            )
            for owner, setting, _, legacy_leaf in DATABASE_PATHS:
                legacy = (
                    "~/.local/share/tldw_cli/" + legacy_leaf if legacy_leaf else None
                )
                if (
                    any(row.logical_id == prefix + owner for row in doc.files)
                    or custom_database_input(
                        data.get("database", {}).get(setting), legacy
                    )
                    is not None
                    or custom_database_input(
                        local.get("database", {}).get(setting), legacy
                    )
                    is not None
                ):
                    mapping["database." + setting] = database_path(local, setting)
            for location in CONFIG_LOCATION_KEYS:
                if (
                    location[0] != "database"
                    and location[-1] != "data_dir"
                    and (value := config_location(local, location))
                ):
                    mapping[".".join(location)] = Path(value).expanduser().absolute()
        configured = local if retain_config else remap_config_locations(data, mapping)
        configured.setdefault("general", {})["users_name"] = names[profile]
        # A non-remappable imported projection selector cannot become authority.
        rag = configured.get("AppRAGSearchConfig", {}).get("rag", {})
        if not retain_config and any(
            rag.get(key, {}).get("persist_directory")
            for key in ("vector_store", "chroma")
        ) and any(
            row.owner_id == "rag.projections" and row.logical_id.startswith(prefix)
            and (selected_ids is None or row.logical_id in selected_ids)
            for row in doc.files
        ):
            raise ValueError("owner_relocation_unverified:rag.projections")
        selected = {row.logical_id: base for row in (*doc.files, *doc.directories)
                    if selected_ids is None or row.logical_id in selected_ids}
        template = SimpleNamespace(
            restore=tuple(selected.items()),
            destinations=(),
            issues=tuple("owner_setup_required:" + owner for owner in deferred),
        )
        derived = _config_targets(
            configured, profile, config_target, doc, template, owners, derive=True
        )
        for root in doc.directories:
            if (
                root.parent_id is not None
                or root.logical_id in derived
                or root.logical_id in external_ids
                or selected_ids is not None and root.logical_id not in selected_ids
            ):
                continue
            members = [
                row
                for row in doc.files
                if row.root_id == root.logical_id and row.logical_id.startswith(prefix)
            ]
            owner = producer.get(root.logical_id)
            if (
                members
                and all(row.owner_id in deferred for row in members)
                or owner
                and owner.owner_id in deferred
                and root.logical_id.startswith(prefix)
            ):
                # Separate inert storage keeps deferred owners inactive and disjoint.
                destination = (
                    setup_parent / ("inert-" + hashlib_root(root.logical_id))
                    if mode == "replace"
                    else base / "inert" / hashlib_root(root.logical_id)
                )
                shared = (
                    ["directory:" + shared_directories[root.logical_id]]
                    if root.logical_id in shared_directories
                    else []
                )
                shared.extend(
                    "file:" + producer[row.logical_id].shared_group
                    for row in members
                    if row.logical_id in producer
                    and producer[row.logical_id].shared_group
                )
                prior = {
                    inert_destinations[key]
                    for key in shared
                    if key in inert_destinations
                }
                if len(prior) > 1:
                    raise ValueError("shared_target_split")
                if prior:
                    destination = prior.pop()
                for key in shared:
                    inert_destinations[key] = destination
                derived[root.logical_id] = destination
        destinations.update(derived)
        destinations.update(
            {prefix + selector: path for selector, path in mapping.items()}
        )
    roots = {row.logical_id for row in doc.directories if row.parent_id is None
             and (selected_ids is None or row.logical_id in selected_ids)}
    credential_roots = {
        row.root_id for row in doc.files if row.owner_id == "recovery.credentials"
    }
    if roots - credential_roots - destinations.keys():
        raise ValueError("owner_relocation_unverified")
    plan = plan_restore(
        archive,
        mode=mode,
        destinations=destinations,
        target=target,
        profile_names=names,
        retained_configs=tuple(retained_configs),
        data_groups=data_groups,
        **review,
    )
    check_config_destinations(archive, plan)
    return plan


def hashlib_root(key):
    """Use an opaque archive identity solely as an inert local filename."""
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()[:16]
