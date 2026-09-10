"""Private bounded restore candidates; publication belongs to the journal executor."""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import archive_reader as reader
from .archive_models import SealedArchive
from .limits import ArchiveLimits
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
)
from .owner_registry import install_adapters
from .restore_plan import RestorePlan, _ancestor, _document, recheck_targets
from .space import require_capacity


def _mkdirs(path, root):
    if path == root:
        return
    _mkdirs(path.parent, root)
    if not path.exists():
        create_private_directory(path)


def _copy(source, destination, cancel):
    digest = hashlib.sha256()
    total = 0
    with (
        reader._regular(source) as stream,
        create_private_file(destination) as fd,
        os.fdopen(os.dup(fd), "wb") as output,
    ):
        before = reader._identity(os.fstat(stream.fileno()))
        while chunk := stream.read(64 * 1024):
            reader._check(cancel)
            require_capacity({destination.parent: len(chunk)})
            digest.update(chunk)
            total += len(chunk)
            output.write(chunk)
        if before != reader._identity(os.fstat(stream.fileno())):
            raise ValueError("candidate_changed")
    return total, digest.hexdigest()


def _config_targets(data, profile, config_target, doc, plan, owners):
    """Check installed inert selectors using the reviewed final config location."""
    from .config_adapter import _CONFIG_HISTORY
    from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext

    configured = dict(data)
    configured[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(config_target, profile)
    selected = dict(plan.restore)
    roots = dict(plan.destinations)
    directories = {row.logical_id: row for row in doc.directories}
    from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration
    from tldw_chatbook.MCP.recovery import _Store
    from tldw_chatbook.Model_Artifacts.recovery import _Artifacts, managed_artifact_root
    from tldw_chatbook.Persona_Visual.recovery import _Assets
    from tldw_chatbook.runtime_policy.recovery import _SourceState
    from tldw_chatbook.Workspaces.recovery import _ChangeTracking

    from .config_adapter import _Definition
    from .profile_paths import DATABASE_PATHS, database_path, user_data_dir
    from .recovered_media import _RecoveredAdapter

    databases = {owner: key for owner, key, _, _ in DATABASE_PATHS}
    for payload in doc.files:
        if (
            not payload.logical_id.startswith(f"profile:{profile}:")
            or payload.logical_id not in selected
        ):
            continue
        owner = owners[payload.owner_id]
        if "owner_setup_required:" + payload.owner_id in plan.issues:
            continue  # Retained inert bytes await explicit owner binding (Task20).
        if payload.owner_id == "external.files":
            continue
        if payload.owner_id == "config":
            expected = config_target
        elif payload.owner_id == "config.history":
            destination = selected[payload.logical_id]
            if destination.parent == config_target.parent and (
                destination == config_target.with_suffix(config_target.suffix + ".bak")
                or _CONFIG_HISTORY.fullmatch(destination.name)
            ):
                continue
            raise ValueError("owner_relocation_unverified:config.history")
        elif payload.owner_id in databases:
            expected = database_path(configured, databases[payload.owner_id])
        elif isinstance(owner, _SQLiteDeclaration):
            if owner.setting_name:
                expected = database_path(configured, owner.setting_name)
            elif owner.owner_id == "db.agent_runs":
                expected = (
                    database_path(configured, "chachanotes_db_path").parent / owner.leaf
                )
            else:
                expected = user_data_dir(configured) / owner.leaf
        elif type(owner) is _Store:
            expected = user_data_dir(configured) / owner.leaf
            suffix = (
                ".bak"
                if owner.owner_id == "mcp.permissions"
                else ".1"
                if owner.owner_id == "mcp.history"
                else None
            )
            if suffix and selected[payload.logical_id] == expected.with_name(
                expected.name + suffix
            ):
                continue
        elif type(owner) is _SourceState:
            expected = config_target.parent / "runtime_policy.json"
        elif type(owner) is _ChangeTracking:
            expected = user_data_dir(configured) / "change_review"
        elif type(owner) is _Assets:
            expected = owner._root(configured)
        elif isinstance(owner, _Definition):
            expected = owner._definition_path(configured)
        elif type(owner) is _Artifacts:
            expected = managed_artifact_root(user_data_dir(configured)).parent
        elif type(owner) is _RecoveredAdapter:
            expected = user_data_dir(configured) / "recovered_media"
        elif payload.owner_id in {
            "chat.attachments",
            "study.local",
            "quiz.local",
            "notes.sync_bindings",
        }:
            expected = database_path(configured, "chachanotes_db_path")
        else:
            raise ValueError("owner_relocation_unverified:" + payload.owner_id)
        destination = selected[payload.logical_id]
        root = directories[payload.root_id]
        if destination == expected or (
            not root.synthetic
            and roots[payload.root_id] == expected
            and expected in destination.parents
        ):
            continue
        raise ValueError("owner_relocation_unverified:" + payload.owner_id)


def _items(doc, plan):
    """Preserve archive topology while keeping local destinations explicit."""
    from .models import FileMetadata, StorageItem

    producer = {row.logical_id: row for row in doc.producer_inventory}
    targets = dict(plan.restore)
    directories = {row.logical_id for row in doc.directories}
    result = {}
    for record in (*doc.directories, *doc.files):
        if record.logical_id not in targets:
            continue
        source = producer.get(record.logical_id)
        owner_id = source.owner_id if source else getattr(record, "owner_id", None)
        if owner_id is None:
            continue  # Legacy directory-only extraction has no owner assertion.
        kind = "directory" if record.logical_id in directories else "file"
        metadata = record.metadata
        result[record.logical_id] = StorageItem(
            owner_id,
            record.logical_id,
            targets[record.logical_id],
            "included_directory" if kind == "directory" else "included",
            source.dependencies if source else (),
            source.shared_group if source else None,
            metadata=FileMetadata(
                1,
                record.root_id,
                record.relative_path,
                record.parent_id,
                kind,
                metadata.mode if metadata else 0o600,
                metadata.mtime_ns if metadata else 0,
                "private",
            ),
        )
    return result


def stage_restore(
    archive: SealedArchive,
    plan: RestorePlan,
    work_root: Path,
    cancel: Event,
    *,
    journal=None,
) -> Path:
    """Return a private candidate descriptor with explicit per-volume artifacts.

    All archive payloads first remain under their original inert payload names for
    credential-reference validation. Qualified copies are then placed on each
    destination volume. No destination file is opened for writing.
    """
    reader._check(cancel)
    if type(plan) is not RestorePlan or plan.archive_digest != archive.digest:
        raise ValueError("archive_plan_mismatch")
    doc = _document(archive)
    recheck_targets(plan)
    for _, live in (*plan.restore, *plan.retire, *plan.preserve):
        if work_root == live or live in work_root.parents or work_root in live.parents:
            raise ValueError("staging_target_alias")
    work_parent = _ancestor(work_root)
    targets = tuple(path for _, path in plan.destinations)
    if any(
        work_root == target
        or target in work_root.parents
        or work_root in target.parents
        for target in targets
    ):
        raise ValueError("staging_destination_alias")
    if work_root == archive.path.parent or work_root in archive.path.parents:
        raise ValueError("staging_archive_alias")
    if work_root.exists():
        with pinned_directory(work_root) as fd:
            if os.fstat(fd).st_mode & 0o077:
                raise ValueError("private_staging_required")
    elif work_root.parent != work_parent:
        raise ValueError("staging_parent_missing")
    from .qualification import qualified_for

    for _, destination in plan.destinations:
        parent = _ancestor(destination.parent)
        for operation in ("publish_file", "publish_directory"):
            allowed, reason = qualified_for(operation, parent)
            if not allowed:
                raise ValueError(reason)
    total = sum(payload.size for payload in doc.files)
    require_capacity({work_root: total * 2})
    if not work_root.exists():
        create_private_directory(work_root)
    stage = work_root / ("restore-" + uuid4().hex)
    create_private_directory(stage)
    volume_roots = []
    created_identities = {stage: (stage.stat().st_dev, stage.stat().st_ino)}
    successful = False
    try:
        limits = ArchiveLimits()
        extracted = {}
        with reader._regular(archive.path) as stream:
            central = reader._central_preflight(stream, limits)
            with zipfile.ZipFile(stream) as container:
                infos = container.infolist()
                offsets = reader._local_ranges(stream, infos, central)
                streamed = 0
                for payload in doc.files:
                    reader._check(cancel)
                    target = stage / payload.payload
                    _mkdirs(target.parent, stage)
                    digest = hashlib.sha256()
                    size = 0
                    info = container.getinfo(payload.payload)
                    with (
                        create_private_file(target) as fd,
                        os.fdopen(os.dup(fd), "wb") as output,
                    ):
                        for chunk in reader._member_chunks(
                            stream, info, offsets[info.filename], cancel
                        ):
                            size += len(chunk)
                            streamed += len(chunk)
                            if (
                                size > min(payload.size, limits.member_bytes)
                                or streamed > limits.expanded_bytes
                            ):
                                raise ValueError("expanded_limit")
                            require_capacity({stage: total - streamed})
                            output.write(chunk)
                            digest.update(chunk)
                    if size != payload.size or digest.hexdigest() != payload.sha256:
                        raise ValueError("payload_digest_mismatch")
                    extracted[payload.logical_id] = target
        reader.verify_sealed(archive, cancel)
        owners = {owner.owner_id: owner for owner in install_adapters()}
        selected = dict(plan.restore)
        items = _items(doc, plan)
        from .sqlite_validation import validate_candidate
        from .storage_admission import _preview_reads

        with _preview_reads():
            for payload in doc.files:
                if payload.logical_id not in selected:
                    continue
                owner = owners[payload.owner_id]
                candidate = extracted[payload.logical_id]
                item = items[payload.logical_id]
                policy = owner.schema_policy()
                role_check = getattr(owner, "restore_role", None)
                role = (
                    role_check(item)
                    if callable(role_check)
                    else (
                        "sqlite" if policy is not None and policy.schema_sql else "file"
                    )
                )
                if role == "sqlite":
                    issues = validate_candidate(owner, candidate, cancel, migrate=True)
                else:
                    validator = getattr(owner, "validate_restore", None)
                    issues = (
                        validator(item, candidate)
                        if callable(validator)
                        else owner.validate(candidate)
                    )
                if issues:
                    raise ValueError(issues[0])
                if payload.owner_id == "config":
                    import tomllib

                    import toml

                    from .config_adapter import remap_config_locations

                    profile = payload.logical_id.split(":")[1]
                    prefix = f"profile:{profile}:"
                    mapping = {
                        key[len(prefix) :]: value
                        for key, value in plan.selectors
                        if key.startswith(prefix)
                    }
                    data = tomllib.loads(candidate.read_text(encoding="utf-8"))
                    data = remap_config_locations(data, mapping)
                    data.setdefault("general", {})["users_name"] = dict(
                        plan.profile_names
                    )[profile]
                    _config_targets(
                        data, profile, selected[payload.logical_id], doc, plan, owners
                    )
                    # Replace only this exclusively created private working file.
                    candidate.write_text(toml.dumps(data), encoding="utf-8")
                relocate = getattr(owner, "relocate_restore", None)
                if callable(relocate):
                    relocate(item, candidate, selected)
                else:
                    owner.relocate(candidate, selected)
                validator = getattr(owner, "validate_restore", None)
                issues = (
                    validator(item, candidate)
                    if callable(validator)
                    else owner.validate(candidate)
                )
                if issues:
                    raise ValueError(issues[0])
        validated = {
            key: reader._hash(path, cancel)
            for key, path in extracted.items()
            if key in selected
        }
        shared = {}
        for item in doc.producer_inventory:
            if item.shared_group and item.logical_id in validated:
                previous = shared.setdefault(
                    item.shared_group, validated[item.logical_id]
                )
                if previous != validated[item.logical_id]:
                    raise ValueError("shared_candidate_mismatch")
        credentials = {}
        material = stage / "payload" / "credential-recovery.json"
        if doc.credential_policy != "exclude":
            from .credentials import _material, plan_credential_scopes

            if not material.exists():
                raise ValueError("credential_material_missing")
            _copy(material, stage / "credential-recovery.json", cancel)
            selected_payloads = {
                row.payload for row in doc.files if row.logical_id in selected
            }
            with _preview_reads():
                records = _material(stage)
                retained = [
                    record for record in records if record["file"] in selected_payloads
                ]
                if retained != records:
                    selected_material = {
                        "version": 1,
                        "mode": doc.credential_policy,
                        "records": retained,
                    }
                    (stage / "credential-recovery.json").write_bytes(
                        json.dumps(selected_material, sort_keys=True).encode()
                    )
                credentials = dict(plan_credential_scopes(stage))
        artifacts = []
        planned_metadata = {
            key: (desired, applied) for key, desired, applied in plan.metadata
        }
        candidates = {}
        by_volume = {}
        work_device = stage.stat().st_dev
        root_candidates = {}
        for root_id, destination in plan.destinations:
            parent = _ancestor(destination.parent)
            device = parent.stat().st_dev
            if device not in by_volume:
                require_capacity({parent: total})
                staging_parent = stage if device == work_device else parent
                private = staging_parent / (".chatbook-restore-" + uuid4().hex)
                create_private_directory(private)
                volume_roots.append(private)
                info = private.stat()
                created_identities[private] = (info.st_dev, info.st_ino)
                by_volume[device] = private
            if destination not in root_candidates:
                candidate = (
                    by_volume[device] / hashlib.sha256(root_id.encode()).hexdigest()
                )
                create_private_directory(candidate)
                root_candidates[destination] = candidate
            candidates[root_id] = root_candidates[destination]
        containers = []
        for key, destination in plan.containers:
            device = _ancestor(destination.parent).stat().st_dev
            candidate = by_volume[device] / key.replace(":", "-")
            create_private_directory(candidate)
            info = candidate.stat()
            containers.append(
                {
                    "logical_id": key,
                    "candidate": str(candidate),
                    "destination": str(destination),
                    "kind": "directory",
                    "identity": [
                        info.st_dev,
                        info.st_ino,
                        info.st_mode,
                        info.st_mtime_ns,
                    ],
                    "applied_metadata": {
                        "version": 1,
                        "mode": 0o700,
                        "mtime_ns": info.st_mtime_ns,
                    },
                }
            )
        for directory in sorted(
            doc.directories, key=lambda row: len(Path(row.relative_path).parts)
        ):
            if directory.logical_id not in selected:
                continue
            path = candidates[directory.root_id] / directory.relative_path
            _mkdirs(path, candidates[directory.root_id])
            artifacts.append(
                {
                    "logical_id": directory.logical_id,
                    "candidate": str(path),
                    "destination": str(selected[directory.logical_id]),
                    "kind": "directory",
                }
            )
        from types import MappingProxyType

        candidate_paths = {
            key: path for key, path in extracted.items() if key in selected
        }
        # Whole-file aliases have already passed exact post-relocation digest
        # equality. Their semantic validators must see the same private file
        # when the locally reviewed plan installs them at one destination.
        shared_candidates = {}
        for item in doc.producer_inventory:
            if item.shared_group and item.logical_id in candidate_paths:
                key = (item.shared_group, selected[item.logical_id])
                candidate_paths[item.logical_id] = shared_candidates.setdefault(
                    key, candidate_paths[item.logical_id]
                )
        candidate_paths.update(
            {row["logical_id"]: Path(row["candidate"]) for row in artifacts}
        )
        directory_ids = {row.logical_id for row in doc.directories}
        topology = MappingProxyType(
            {
                record.logical_id: (
                    record.root_id,
                    record.parent_id,
                    record.relative_path,
                    "directory" if record.logical_id in directory_ids else "file",
                )
                for record in (*doc.directories, *doc.files)
            }
        )
        from tldw_chatbook.Persona_Visual.recovery import _Assets

        with _preview_reads():
            synthetic = {row.logical_id for row in doc.directories if row.synthetic}
            for key, item in items.items():
                owner = owners[item.owner]
                if type(owner) is _Assets and item.metadata.root_id in synthetic:
                    raise ValueError("invalid_synthetic_asset_root")
                if key in synthetic:
                    continue  # Fabricated containers carry no source owner data.
                restore_check = getattr(owner, "validate_restore_dependencies", None)
                legacy_check = getattr(owner, "validate_dependencies", None)
                if callable(restore_check):
                    issues = restore_check(
                        item,
                        candidate_paths[key],
                        MappingProxyType(candidate_paths),
                        topology=topology,
                    )
                elif callable(legacy_check):
                    if not doc.producer_inventory:
                        raise ValueError("producer_inventory_required")
                    issues = legacy_check(
                        item, candidate_paths[key], MappingProxyType(candidate_paths)
                    )
                else:
                    continue
                if issues:
                    raise ValueError(issues[0])
        for payload in doc.files:
            if payload.logical_id not in selected:
                continue
            path = candidates[payload.root_id] / payload.relative_path
            if path.exists():
                size, digest = path.stat().st_size, reader._hash(path, cancel)
            else:
                size, digest = _copy(extracted[payload.logical_id], path, cancel)
            if digest != validated[payload.logical_id]:
                raise ValueError("candidate_changed")
            metadata = planned_metadata[payload.logical_id][1]
            os.utime(path, ns=(metadata.mtime_ns, metadata.mtime_ns))
            os.chmod(path, metadata.mode)
            artifacts.append(
                {
                    "logical_id": payload.logical_id,
                    "candidate": str(path),
                    "destination": str(selected[payload.logical_id]),
                    "kind": "file",
                    "size": size,
                    "sha256": digest,
                }
            )
        # Metadata is applied only after all children and owner validation settle.
        for directory in sorted(
            doc.directories,
            key=lambda row: len(Path(row.relative_path).parts),
            reverse=True,
        ):
            if directory.logical_id in selected:
                path = candidates[directory.root_id] / directory.relative_path
                metadata = planned_metadata[directory.logical_id][1]
                os.utime(path, ns=(metadata.mtime_ns, metadata.mtime_ns))
                os.chmod(path, metadata.mode)
        new_directories = {
            Path(row["destination"])
            for row in artifacts
            if row["kind"] == "directory" and not Path(row["destination"]).exists()
        }
        physical = {}
        for row in artifacts:
            destination = Path(row["destination"])
            row["publication_unit"] = (
                row["kind"] != "directory" or not destination.exists()
            ) and not any(parent in new_directories for parent in destination.parents)
            if row["candidate"] in physical:
                row["alias_of"] = physical[row["candidate"]]
                row["publication_unit"] = False
            else:
                physical[row["candidate"]] = row["logical_id"]
            desired, applied = planned_metadata[row["logical_id"]]
            row["desired_metadata"] = desired.model_dump() if desired else None
            row["applied_metadata"] = applied.model_dump()
            info = Path(row["candidate"]).lstat()
            row["identity"] = [info.st_dev, info.st_ino, info.st_mode, info.st_mtime_ns]
        recheck_targets(plan)
        reader.verify_sealed(archive, cancel)
        reader._check(cancel)
        descriptor = {
            "version": 1,
            "archive_digest": archive.digest,
            "target_fingerprint": plan.target_fingerprint,
            "profile_names": dict(plan.profile_names),
            "issues": plan.issues,
            "artifacts": artifacts,
            "containers": containers,
            "credential_scopes": credentials,
            "private_roots": [str(path) for path in volume_roots],
        }
        with (
            create_private_file(stage / "candidate.json") as fd,
            os.fdopen(os.dup(fd), "wb") as output,
        ):
            output.write(json.dumps(descriptor, sort_keys=True).encode())
        if journal is not None:
            journal.record_candidate(stage, plan, archive)
        successful = True
        return stage
    finally:
        if not successful:
            for root in (*volume_roots, stage):
                info = root.lstat()
                if (info.st_dev, info.st_ino) != created_identities[root]:
                    raise ValueError("staging_identity_changed")
                # Only this operation's pinned-identity private trees are removed.
                for directory, children, _ in os.walk(root, topdown=True):
                    os.chmod(directory, 0o700, follow_symlinks=False)
                    for child in children:
                        path = Path(directory) / child
                        if not path.is_symlink():
                            os.chmod(path, 0o700, follow_symlinks=False)
                shutil.rmtree(root)
