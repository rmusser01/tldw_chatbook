"""Confined, credential-excluded import of acquired portable bytes (ADR-256).

This is deliberately not a capture API. Historical paths are lexical input and
the resulting archive is partial even when every supported owner validates.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import struct
import tomllib
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event

import toml

from tldw_chatbook.Utils.platform_files import os

from .archive_reader import _regular
from .limits import ArchiveLimits
from .models import DiscoveryContext, Inventory, StorageItem, storage_logical_id


@dataclass(frozen=True)
class DetachedImportResult:
    """Verified archive locator and inert, non-secret review text."""

    archive: Path
    review_lines: tuple[str, ...]


def _check(cancel: Event) -> None:
    if cancel.is_set():
        raise InterruptedError("cancelled")


def _wal_checksum(data: bytes, order: str, checksum=(0, 0)):
    first, second = checksum
    words = struct.unpack(order + "I" * (len(data) // 4), data)
    for index in range(0, len(words), 2):
        first = (first + words[index] + second) & 0xFFFFFFFF
        second = (second + words[index + 1] + first) & 0xFFFFFFFF
    return first, second


def _validate_wal(path: Path, cancel: Event) -> None:
    """Refuse damaged or unfinished frames instead of SQLite silently ignoring them."""
    with _regular(path) as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise ValueError("detached_sqlite_wal_invalid")
        magic, version, page_size = struct.unpack(">III", header[:12])
        if (
            magic not in (0x377F0682, 0x377F0683)
            or version != 3007000
            or page_size < 512
            or page_size > 65536
            or page_size & (page_size - 1)
        ):
            raise ValueError("detached_sqlite_wal_invalid")
        order = "<" if magic == 0x377F0682 else ">"
        checksum = _wal_checksum(header[:24], order)
        if checksum != struct.unpack(">II", header[24:]):
            raise ValueError("detached_sqlite_wal_invalid")
        frames = 0
        committed = False
        while frame := stream.read(24 + page_size):
            _check(cancel)
            frames += 1
            if (
                len(frame) != 24 + page_size
                or frame[8:16] != header[16:24]
                or struct.unpack(">I", frame[:4])[0] == 0
            ):
                raise ValueError("detached_sqlite_wal_invalid")
            checksum = _wal_checksum(frame[:8] + frame[24:], order, checksum)
            if checksum != struct.unpack(">II", frame[16:24]):
                raise ValueError("detached_sqlite_wal_invalid")
            committed = struct.unpack(">I", frame[4:8])[0] != 0
        if frames and not committed:
            raise ValueError("detached_sqlite_wal_unfinished")


def _normalize_sqlite(owner, source: Path, destination: Path, cancel: Event) -> int:
    """Normalize only disposable copies and validate the complete committed image.

    The acquired main/WAL/SHM set remains unchanged. SHM is rebuilt locally, never
    trusted as evidence. All imported SQL is restricted before the first query.
    """
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    from .native_files import create_private_directory, create_private_file
    from .sqlite_validation import _check as validate_connection
    from .sqlite_validation import _installed_owner, validated_schema_version

    _check(cancel)
    if Path(str(source) + "-journal").exists():
        raise ValueError("detached_sqlite_rollback_unsupported")
    wal = Path(str(source) + "-wal")
    if wal.exists() and wal.stat().st_size:
        _validate_wal(wal, cancel)
    limits = ArchiveLimits()
    if any(
        path.exists() and path.stat().st_size > limits.member_bytes
        for path in (source, wal)
    ):
        raise ValueError("detached_sqlite_limit")
    scratch = destination.parent / (
        "normalize-" + hashlib.sha256(destination.name.encode()).hexdigest()
    )
    create_private_directory(scratch)
    main = scratch / "source.sqlite3"
    try:
        for original, target in ((source, main), (wal, Path(str(main) + "-wal"))):
            if not original.exists() or original == wal and not original.stat().st_size:
                continue
            with (
                _regular(original) as incoming,
                create_private_file(target) as fd,
                os.fdopen(os.dup(fd), "wb") as output,
            ):
                while chunk := incoming.read(1024**2):
                    _check(cancel)
                    output.write(chunk)
        installed = _installed_owner(owner.owner_id)
        with open_recovery_validation(
            installed.owner_id,
            main,
            writable=True,
            with_restrictions=True,
            cancel=cancel,
        ) as (connection, restrictions):
            issues, _ = validate_connection(
                connection, installed, installed.schema_policy(), restrictions
            )
            if issues:
                raise ValueError(issues[0])
            with create_private_file(destination):
                pass
            with closing(sqlite3.connect(destination)) as output:
                output.enable_load_extension(False)
                output.execute("PRAGMA trusted_schema=OFF")

                def progress(_status, _remaining, _total):
                    _check(cancel)
                    if restrictions.expired():
                        raise ValueError("detached_sqlite_limit")

                connection.backup(output, pages=64, progress=progress, sleep=0)
                output.execute("PRAGMA journal_mode=DELETE")
        return validated_schema_version(installed, destination, cancel)
    except sqlite3.Error:
        raise ValueError("detached_sqlite_invalid") from None
    # Keep operation-owned normalization evidence private. The outer transaction
    # owns retention/cleanup; ambiguous failures never recursively remove a tree.


def _json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("detached_duplicate_key")
            result[key] = value
        return result

    with _regular(path) as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("detached_definition_limit")
    return json.loads(raw, object_pairs_hook=unique)


def _mapped(value: str, old_data: Path, source: Path) -> Path:
    """Map an absolute historical suffix without observing any original target."""
    if not isinstance(value, str) or "\0" in value or ".." in Path(value).parts:
        raise ValueError("detached_selector_unsupported")
    old = (
        old_data / "home" / value[2:]
        if value.startswith(("~/", "~\\"))
        else old_data / "home"
        if value == "~"
        else Path(value)
    )
    if not old.is_absolute():
        raise ValueError("detached_selector_unsupported")
    try:
        relative = old.relative_to(old_data)
    except ValueError:
        raise ValueError("detached_external_storage_unsupported") from None
    return source / relative


def _builtin_support(owner, config):
    """Accept only exact references in the two installed bundled visual catalogs."""
    from .profile_paths import database_path

    references = owner._references(database_path(config, "chachanotes_db_path"))
    catalogs = {}
    for locator, size, digest in references:
        parts = Path(locator).parts
        if (
            len(parts) < 3
            or parts[0] != "characters"
            or parts[1] not in {"samira", "pixel_migu"}
        ):
            raise ValueError(
                "detached_owner_unavailable:persona.visual_identity_builtin"
            )
        name = parts[1]
        if name not in catalogs:
            document = _json(
                owner._root(config) / "characters" / name / "visual_identity_pack.json"
            )
            catalogs[name] = {
                asset["storage_relpath"]: (asset["bytes"], asset["sha256"])
                for asset in document["assets"]
            }
        if catalogs[name].get(locator) != (size, digest):
            raise ValueError(
                "detached_owner_unavailable:persona.visual_identity_builtin"
            )
    return owner.discover(config)


def _restored_cache_paths(config, root: Path) -> tuple[Path, ...]:
    """Recognize only the empty directory pair made by isolated cache remapping."""
    from .config_adapter import config_location
    from .native_files import pinned_directory

    cache = root / "embedding_config" / "model_cache_dir"
    value = config_location(config, ("embedding_config", "model_cache_dir"))
    if not value or Path(value) != cache:
        return ()
    reason = "detached_restored_cache_unsupported"
    try:
        with pinned_directory(root) as profile:
            try:
                os.stat(cache.parent.name, dir_fd=profile, follow_symlinks=False)
            except FileNotFoundError:
                return ()
        with pinned_directory(cache.parent) as parent:
            if set(os.listdir(parent)) != {cache.name}:
                raise ValueError(reason)
            descriptor = os.open(
                cache.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            try:
                held = os.fstat(descriptor)
                if not stat.S_ISDIR(held.st_mode) or os.listdir(descriptor):
                    raise ValueError(reason)
                named = os.stat(cache.name, dir_fd=parent, follow_symlinks=False)
                if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino) or set(
                    os.listdir(parent)
                ) != {cache.name}:
                    raise ValueError(reason)
            finally:
                os.close(descriptor)
        return cache.parent, cache
    except OSError:
        raise ValueError(reason) from None


def produce_detached_archive(
    source: Path, work: Path, cancel: Event
) -> DetachedImportResult:
    """Produce a strict partial archive from a complete supported acquired layout.

    Args:
        source: Private acquired Data tree, retained unchanged and inactive.
        work: New private staging directory outside the acquired tree.
        cancel: Cooperative cancellation signal for bounded copying/validation.

    Returns:
        Verified partial archive and non-secret local review lines.

    Raises:
        ValueError: Unsupported, unresolved, changed or unsafe source evidence.
        InterruptedError: Cancellation was requested before publication.
    """
    from .archive_writer import write_archive
    from .capture import CaptureResult, _manifest_for
    from .config_adapter import (
        CONFIG_LOCATION_KEYS,
        config_location,
        remap_config_locations,
    )
    from .credentials import process_credentials
    from .detached_evidence import validate_copied_evidence
    from .detached_source import recheck_source, review_source
    from .native_files import create_private_directory, create_private_file
    from .profile_paths import DATABASE_PATHS, user_data_dir
    from .publication import _validate_installed_copies
    from .storage_admission import _preview_reads

    _check(cancel)
    source, work = Path(source), Path(work)
    if (
        not source.is_absolute()
        or not work.is_absolute()
        or source in work.parents
        or work in source.parents
    ):
        raise ValueError("detached_work_overlap")
    observation = review_source(source)
    marker = _json(source / "portable-location.json")
    if (
        set(marker) != {"version", "root"}
        or type(marker["version"]) is not int
        or marker["version"] != 1
    ):
        raise ValueError("detached_marker_invalid")
    if not isinstance(marker["root"], str) or "\0" in marker["root"]:
        raise ValueError("detached_marker_invalid")
    old_root = Path(marker["root"])
    if not old_root.is_absolute() or ".." in old_root.parts:
        raise ValueError("detached_marker_invalid")
    selected = source / "home" / ".config" / "tldw_cli" / "config.toml"
    with _regular(selected) as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("detached_definition_limit")
    config = tomllib.loads(raw.decode("utf-8"))
    old_data = old_root / "Data"
    old_data / "home" / ".local" / "share" / "tldw_cli"
    base = source / "home" / ".local" / "share" / "tldw_cli"
    mapping = {}
    for location in CONFIG_LOCATION_KEYS:
        value = config_location(config, location)
        if not value:
            continue
        key = ".".join(location)
        if location in (
            ("console", "workspace_root"),
            ("notes", "sync_directory"),
            ("llm_management", "model_download_dir"),
        ):
            # External user references remain inert configuration, never source IO.
            continue
        if location[0] == "database":
            row = next(row for row in DATABASE_PATHS if row[1] == location[1])
            if row[3] and value == "~/.local/share/tldw_cli/" + row[3]:
                continue
        mapping[key] = _mapped(value, old_data, source)
    if "paths.data_dir" in mapping or "Paths.data_dir" in mapping:
        base = mapping.get("paths.data_dir", mapping.get("Paths.data_dir"))
    mapping.update({"paths.data_dir": base, "Paths.data_dir": base})
    config = remap_config_locations(config, mapping)
    rag = config.get("AppRAGSearchConfig", {}).get("rag", {})
    if any(
        rag.get(name, {}).get("persist_directory")
        for name in ("vector_store", "chroma")
    ):
        raise ValueError("detached_optional_layout_unsupported:rag.projections")
    root = user_data_dir(config)
    # Complete layout registration precedes every content validator.
    owners = _declared_layout(source, selected, root, config, observation)
    issues = validate_copied_evidence(source)
    if issues:
        raise ValueError(issues[0])
    create_private_directory(work)
    normal = work / "normalized"
    payload = work / "payload"
    directories = work / "directories"
    for path in (normal, payload, directories):
        create_private_directory(path)
    context = DiscoveryContext(
        selected, hashlib.sha256(str(selected).encode()).hexdigest()[:24]
    )
    from .models import DISCOVERY_CONTEXT_KEY

    config[DISCOVERY_CONTEXT_KEY] = context
    items, original_paths, versions = _owner_items(
        source, selected, root, config, context, owners, normal, cancel
    )
    # A required user-selected store never becomes an empty default on import.
    config_id = storage_logical_id(context, "config")
    staged_config = normal / "config.toml"
    serialized = {
        key: value for key, value in config.items() if key != DISCOVERY_CONTEXT_KEY
    }
    with create_private_file(staged_config) as fd:
        os.write(fd, toml.dumps(serialized).encode("utf-8"))
    items = tuple(
        replace(item, path=staged_config) if item.logical_id == config_id else item
        for item in items
    )
    _complete_coverage(observation, original_paths, source, selected)
    from .models import FileMetadata

    items = tuple(
        replace(
            item,
            metadata=FileMetadata(
                1,
                "root:" + hashlib.sha256(item.logical_id.encode()).hexdigest(),
                item.path.name,
                "root:" + hashlib.sha256(item.logical_id.encode()).hexdigest(),
                "file",
                0o600,
                os.stat(item.path).st_mtime_ns,
                "private",
            ),
        )
        if item.status == "included" and item.metadata is None
        else item
        for item in items
    )
    staged, candidates, topology = [], {}, {}
    for item in items:
        _check(cancel)
        if item.status not in {"included", "included_directory"}:
            continue
        if item.metadata is not None:
            meta = item.metadata
            topology[item.logical_id] = (
                meta.root_id,
                meta.parent_id,
                meta.relative_path,
                meta.kind,
            )
        if item.status == "included_directory":
            target = directories / hashlib.sha256(item.logical_id.encode()).hexdigest()
            create_private_directory(target)
            candidates[item.logical_id] = target
            continue
        target = payload / hashlib.sha256(item.logical_id.encode()).hexdigest()
        with (
            _regular(item.path) as incoming,
            create_private_file(target) as fd,
            os.fdopen(os.dup(fd), "wb") as output,
        ):
            while chunk := incoming.read(1024**2):
                _check(cancel)
                output.write(chunk)
        staged.append((item, target))
        candidates[item.logical_id] = target
    with _preview_reads():
        from .config_adapter import _ChatbookRegistry

        for item, target in staged:
            owner = owners[item.owner]
            if type(owner) is _ChatbookRegistry:
                owner.prepare_capture(item, target, items)
        rebound = Inventory(
            tuple(
                replace(item, path=candidates.get(item.logical_id, item.path))
                for item in items
            ),
            False,
            "",
            (),
        )
        credential_issues = process_credentials(
            work, rebound, mode="exclude", encrypted=False
        )
        if credential_issues:
            raise ValueError(credential_issues[0])
    selected_items = {
        item.logical_id: item for item in items if item.logical_id in candidates
    }
    validation_candidates = dict(candidates)
    shared_candidates = {}
    for item in selected_items.values():
        if item.status == "included" and item.shared_group:
            validation_candidates[item.logical_id] = shared_candidates.setdefault(
                item.shared_group, candidates[item.logical_id]
            )
    _validate_installed_copies(
        selected_items, validation_candidates, topology, set(), owners
    )
    from .inventory import classify_entries

    classified = classify_entries(items)
    if classified.issues:
        raise ValueError(classified.issues[0])
    aliases = {}
    for item, _ in staged:
        if item.shared_group:
            aliases.setdefault(item.shared_group, []).append(item.logical_id)
    inventory = Inventory(
        items, False, observation.digest, ("detached_copy_uncoordinated",)
    )
    limits = ArchiveLimits()
    report = (
        "Imported acquired portable profile bytes; original multi-store coherence is unknown.",
        "Managed credentials are excluded and need setup again.",
        "Copied recovery records and execution approvals remain inactive retained evidence.",
    )
    from tldw_chatbook.DB.recovery_core import core_adapters

    core_ids = {owner.owner_id for owner in core_adapters()}
    absent_defaults = sorted(
        item.owner
        for item in items
        if item.owner in core_ids and item.status == "unused"
    )
    if absent_defaults:
        report += (
            "Absent default stores were not imported: "
            + ", ".join(absent_defaults)
            + ".",
        )
    if any(
        item.owner == "persona.visual_identity_builtin" and item.status == "included"
        for item in items
    ):
        report += (
            "Exact installed bundled visual assets support the copied built-in references.",
        )
    if any(item.logical_id.endswith(":restored_embedding_cache_dir") for item in items):
        report += (
            "Empty canonical restored embedding cache scaffolding is intentionally excluded.",
        )
    manifest = _manifest_for(
        inventory,
        staged,
        aliases,
        {
            "root": work,
            "cancel": cancel,
            "mode": "exclude",
            "encrypted": False,
            "versions": versions,
            "limits": limits,
            "report_lines": report,
        },
        inventory.issues,
    )
    recheck_source(observation)
    destination = work.parent / (work.name + ".tldw-backup.zip")
    sealed = write_archive(
        CaptureResult(work, inventory, manifest),
        destination,
        password=None,
        cancel=cancel,
    )
    recheck_source(observation)
    return DetachedImportResult(sealed.path, report)


def _owner_items(source, selected, root, config, context, owners, normal, cancel):
    """Resolve installed owners from confined fixed paths, never saved authority."""
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_operations import (
        _sqlite_inventory_status,
        _SQLiteDeclaration,
    )

    from .config_adapter import _Definition, _excluded_root
    from .native_files import pinned_directory
    from .profile_paths import DATABASE_PATHS, database_path
    from .recovery_files import _RawDeclaration
    from .storage_admission import _preview_reads

    items, original, versions = [], set(), {}

    def add(owner_id, path, *, tree=False, suffix="", exclude=False):
        owner = owners[owner_id]
        if not path.exists():
            rows = (
                StorageItem(
                    owner_id,
                    storage_logical_id(context, owner_id, suffix),
                    path,
                    "unused",
                    (storage_logical_id(context, "config"),),
                ),
            )
        else:
            declaration = (
                owner if isinstance(owner, _Definition) else _RawDeclaration(owner_id)
            )
            rows = (
                declaration._tree(config, path)
                if tree
                else (declaration._item(config, path, suffix),)
            )
        for item in rows:
            if item.path is not None and item.path.is_relative_to(source):
                original.add(item.path)
            if item.status in {"unsupported", "unavailable", "missing_required"}:
                raise ValueError("detached_owner_unavailable:" + owner_id)
            items.append(
                replace(item, status="intentionally_excluded") if exclude else item
            )

    add("config", selected)
    items[-1] = replace(items[-1], dependencies=())
    for suffix, path in zip(
        ("restored_embedding_cache_parent", "restored_embedding_cache_dir"),
        _restored_cache_paths(config, root),
        strict=False,
    ):
        item = _excluded_root(config, "config", path, kind="directory", local_id=suffix)
        if item.status != "intentionally_excluded":
            raise ValueError("detached_restored_cache_unsupported")
        items.append(
            replace(item, dependencies=(storage_logical_id(context, "config"),))
        )
        original.add(path)
    # Every installed default SQLite declaration has one explicit confined path.
    database_rows = {row[0]: row for row in DATABASE_PATHS}
    sqlite_paths = {
        owner_id: database_path(config, row[1])
        for owner_id, row in database_rows.items()
    }
    for owner_id, owner in owners.items():
        if isinstance(owner, _SQLiteDeclaration) and owner_id not in sqlite_paths:
            sqlite_paths[owner_id] = (
                database_path(config, owner.setting_name)
                if owner.setting_name
                else root / owner.leaf
            )
    sqlite_paths["db.agent_runs"] = (
        sqlite_paths["db.chachanotes.primary"].parent / "agent_runs.db"
    )
    sqlite_paths["notes.file_notes"] = root / "file_notes.sqlite"
    for owner_id, row in database_rows.items():
        raw = config.get("database", {}).get(row[1])
        legacy = "~/.local/share/tldw_cli/" + row[3] if row[3] else None
        if raw and raw != legacy and not sqlite_paths[owner_id].exists():
            raise ValueError("detached_required_database_missing")
    normalized = {}
    normalized_owners = {}
    core_ids = {owner.owner_id for owner in core_adapters()}
    for owner_id, path in sqlite_paths.items():
        if owner_id not in owners:
            raise ValueError("detached_owner_unsupported:" + owner_id)
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                original.add(candidate)
        if not path.exists():
            if any(
                Path(str(path) + suffix).exists()
                for suffix in ("-wal", "-shm", "-journal")
            ):
                raise ValueError("detached_sqlite_orphan_sidecar")
            if owner_id in core_ids:
                # Detached import records observable default absence, never
                # historical never-use. Live capture keeps its required policy.
                with _preview_reads():
                    status = _sqlite_inventory_status(
                        config,
                        path,
                        owner=owner_id,
                        setting_name=owners[owner_id].setting_name,
                        optional_default=True,
                    )
                if status != "unused":
                    raise ValueError("detached_required_database_missing")
            items.append(
                StorageItem(
                    owner_id,
                    storage_logical_id(context, owner_id),
                    path,
                    "unused",
                    (storage_logical_id(context, "config"),),
                )
            )
            continue
        if path not in normalized:
            target_parent = normal / hashlib.sha256(str(path).encode()).hexdigest()
            from .native_files import create_private_directory

            create_private_directory(target_parent)
            target = target_parent / path.name
            versions[owner_id] = _normalize_sqlite(
                owners[owner_id], path, target, cancel
            )
            normalized[path] = target
            normalized_owners[path] = owner_id
        else:
            if {owner_id, normalized_owners[path]} != {
                "tts.profile_store",
                "tts.references",
            }:
                raise ValueError("detached_alias_unsupported")
            target = normalized[path]
            from .sqlite_validation import validated_schema_version

            versions[owner_id] = validated_schema_version(
                owners[owner_id], target, cancel
            )
        if owner_id in database_rows:
            config.setdefault("database", {})[database_rows[owner_id][1]] = str(target)
        dependencies = tuple(
            storage_logical_id(context, dependency)
            for dependency in ("config", *getattr(owners[owner_id], "dependencies", ()))
        )
        if owner_id == "tts.references":
            dependencies += (storage_logical_id(context, "tts.profile_store"),)
        if owner_id == "db.rag_indexing":
            dependencies += tuple(
                storage_logical_id(context, dependency)
                for dependency in (
                    "db.media.primary",
                    "db.chachanotes.primary",
                    "db.prompts.primary",
                )
            )
        items.append(
            StorageItem(
                owner_id,
                storage_logical_id(context, owner_id),
                target,
                "included",
                dependencies,
                shared_group="shared:tts:profile:" + context.profile_id
                if owner_id in {"tts.profile_store", "tts.references"}
                else None,
            )
        )
    # Core owns the alias/dependency policy; its queries use normalized images.
    with _preview_reads():
        for owner_id in (
            "db.chachanotes.primary",
            "db.media.primary",
            "db.prompts.primary",
            "db.library_collections",
            "db.library_ingest_jobs",
        ):
            if not any(
                item.owner == owner_id and item.status == "included" for item in items
            ):
                continue
            rows = owners[owner_id].discover(config)
            if any(
                item.status in {"unsupported", "unavailable", "missing_required"}
                for item in rows
            ):
                raise ValueError("detached_owner_unavailable:" + owner_id)
            items[:] = [item for item in items if item.owner != owner_id]
            items.extend(rows)
        core = next(item for item in items if item.owner == "db.chachanotes.primary")
        for alias in (
            "chat.attachments",
            "study.local",
            "quiz.local",
            "notes.sync_bindings",
        ):
            items.append(
                replace(
                    core,
                    owner=alias,
                    logical_id=storage_logical_id(context, alias),
                    dependencies=(core.logical_id,),
                )
            )
            if core.status == "included":
                versions[alias] = versions["db.chachanotes.primary"]
    for owner_id, owner in owners.items():
        if (
            not isinstance(owner, _Definition)
            or not owner.leaf
            or owner_id
            in {
                "runtime.instance_lock",
                "runtime.crash_forensics",
                "runtime.scheduler_heartbeat",
            }
        ):
            continue
        if owner.location == "package":
            if (
                owner_id == "persona.visual_identity_builtin"
                and core.status == "included"
            ):
                # Only this installed owner selects required relative paths from
                # the verified core and compares their exact installed digests.
                # It never accepts a user-supplied package root or walks the package.
                with _preview_reads():
                    rows = _builtin_support(owner, config)
                if any(
                    item.status in {"unsupported", "unavailable", "missing_required"}
                    for item in rows
                ):
                    raise ValueError("detached_owner_unavailable:" + owner_id)
                items.extend(rows)
                continue
            items.append(
                StorageItem(
                    owner_id, storage_logical_id(context, owner_id), None, "unused", ()
                )
            )
            continue
        if owner_id in {"persona.assets", "persona.visual_identity", "skills"}:
            if owner_id != "skills" and core.status != "included":
                path = (
                    root if owner.location == "data" else selected.parent
                ) / owner.leaf
                if path.exists():
                    raise ValueError("dependency_unavailable")
                add(owner_id, path, tree=owner.tree)
                continue
            with _preview_reads():
                rows = owner.discover(config)
            for item in rows:
                if item.path is not None:
                    if not item.path.is_relative_to(source):
                        raise ValueError("detached_selector_unsupported")
                    original.add(item.path)
                if item.status in {"unsupported", "unavailable", "missing_required"}:
                    raise ValueError("detached_owner_unavailable:" + owner_id)
                items.append(item)
            continue
        add(
            owner_id,
            (root if owner.location == "data" else selected.parent) / owner.leaf,
            tree=owner.tree,
        )
    add("chatbooks.registry", root / "tldw_chatbook_chatbooks.json")
    for owner_id, leaf in (
        ("runtime.source_state", "runtime_policy.json"),
        ("eval.definitions", "eval_overrides.yaml"),
    ):
        add(owner_id, selected.parent / leaf)
    for owner_id in (
        "mcp.local",
        "mcp.targets",
        "mcp.context",
        "mcp.permissions",
        "mcp.history",
    ):
        owner = owners[owner_id]
        for suffix in (
            ("", ".bak")
            if owner_id == "mcp.permissions"
            else ("", ".1")
            if owner_id == "mcp.history"
            else ("",)
        ):
            add(owner_id, root / (owner.leaf + suffix), suffix=suffix.lstrip("."))
    for owner_id, leaf in (
        ("workspaces.change_tracking", "change_review"),
        ("subscriptions.assets", "briefing_audio"),
        ("agents.history", "tool_sandbox"),
    ):
        if owner_id == "agents.history" and (root / leaf).exists():
            name = str(
                config.get("agents", {}).get("run_log_dir_name", "agent-runs")
            ).strip()
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or Path(name).is_absolute()
            ):
                name = "agent-runs"
            permitted = {
                name,
                name if name.startswith(".") else "." + name,
                "skill_script_output",
            }
            with pinned_directory(root / leaf) as descriptor:
                if set(os.listdir(descriptor)) - permitted:
                    raise ValueError(
                        "detached_optional_layout_unsupported:agents.history"
                    )
        add(owner_id, root / leaf, tree=True)
    with _preview_reads():
        model_owner = owners["models.artifacts"]
        model_ids = set()
        for item in model_owner._tree(config, root / "models"):
            if item.status == "included" and item.path.name == "manifest.json":
                model_ids.add(model_owner._descriptor(item.path).model_id)
        model_context = replace(
            context,
            selections=replace(context.selections, model_ids=tuple(sorted(model_ids))),
        )
        from .models import DISCOVERY_CONTEXT_KEY

        rows = model_owner.discover({**config, DISCOVERY_CONTEXT_KEY: model_context})
        for item in rows:
            if (
                item.path == root / "models" / "embeddings"
                and item.status == "unsupported"
                and item.metadata is not None
                and item.metadata.kind == "directory"
            ):
                # Config bootstrap creates this exact empty cache sibling. It
                # contains no model payload and grants no managed-store scope.
                with pinned_directory(item.path) as descriptor:
                    if not os.listdir(descriptor):
                        item = replace(item, status="intentionally_excluded")
            if item.path is not None:
                if not item.path.is_relative_to(source):
                    raise ValueError("detached_selector_unsupported")
                original.add(item.path)
            if item.status in {"unsupported", "unavailable", "missing_required"}:
                raise ValueError("detached_owner_unavailable:models.artifacts")
            items.append(item)
    with _preview_reads():
        # These exact installed discoveries inspect only fixed confined roots.
        for owner_id in (
            "config.history",
            "generation.assets",
            "diagnostics.logs",
            "cache.model_catalog",
            "runtime.instance_lock",
            "runtime.config_lock",
            "runtime.chatbook_scratch",
            "runtime.crash_forensics",
            "runtime.scheduler_heartbeat",
            "hooks.permissions",
            "notes.sync_state",
            "research.paste_staging",
            "actor_packs.import_staging",
            "db.workspace_project_index",
            "rag.definitions",
        ):
            rows = owners[owner_id].discover(config)
            for item in rows:
                if item.path is not None:
                    if not item.path.is_relative_to(source):
                        raise ValueError("detached_selector_unsupported")
                    original.add(item.path)
                if item.status in {"unsupported", "unavailable", "missing_required"}:
                    raise ValueError("detached_owner_unavailable:" + owner_id)
                items.append(item)
    # Unqualified optional layouts refuse instead of disappearing from coverage.
    for path in (
        root / "chromadb",
        root / "recovered_media",
        selected.parent / "chatterbox_voices",
        selected.parent / "higgs_voices",
        selected.parent / "kokoro_voice_blends",
    ):
        if path.exists() and any(path.iterdir()):
            raise ValueError("detached_optional_layout_unsupported")
        if path.exists():
            original.add(path)
    # Empty lifecycle scaffolding contains no user payload; retained bytes stay
    # in the acquired source, exactly as for live installed source declarations.
    scaffold = root / "collections_archives"
    if scaffold.exists():
        for path in (scaffold, *scaffold.rglob("*")):
            if path.is_file() and (
                path.name != ".lifecycle.lock" or path.read_bytes() != b"\0"
            ):
                raise ValueError("detached_optional_layout_unsupported")
            original.add(path)
    tts_lock = Path(str(sqlite_paths["tts.profile_store"]) + ".lock")
    if tts_lock.exists():
        with _regular(tts_lock) as stream:
            if stream.read(1):
                raise ValueError("detached_selector_lock_invalid")
        original.add(tts_lock)
    if any(item.status == "included" and item.path in normalized for item in items):
        # Raw ownership cannot adopt an original SQLite source through a second
        # filename role. Only the installed shared DB cohorts above may alias.
        raise ValueError("detached_alias_unsupported")
    return tuple(items), original, versions


def _complete_coverage(observation, paths, source, selected):
    """A recognized parent never authorizes silently dropping its durable children."""
    excluded = (
        source / "cache",
        source / "tmp",
        source / "wezterm",
        source / "home" / ".cache",
        source / "home" / "AppData" / "Local",
        selected.parent / "recovery-bootstrap",
        selected.parent / "recovery",
        source / "home" / ".local" / "share" / "wezterm",
    )
    from .profile_paths import DATA_ROOT_LOCK_NAME

    selector_lock = source / "home" / DATA_ROOT_LOCK_NAME
    if selector_lock.exists() and selector_lock.read_bytes() != b"":
        raise ValueError("detached_selector_lock_invalid")
    paths = {
        *paths,
        source / "portable-location.json",
        selector_lock,
        source / "home" / "AppData" / "Roaming",
    }
    ancestors = {parent for path in (*paths, *excluded) for parent in path.parents}
    for member in observation.members:
        path = source / member.relative
        if (
            path in paths
            or member.directory
            and path in ancestors
            or any(tree == path or tree in path.parents for tree in excluded)
        ):
            continue
        raise ValueError("detached_unknown_content")


def _declared_layout(source, selected, root, config, observation=None):
    """Reject unknown durable members before any owner opens application data."""
    from .config_adapter import _CONFIG_HISTORY, _Definition
    from .owner_registry import install_adapters
    from .profile_paths import DATABASE_PATHS

    owners = {owner.owner_id: owner for owner in install_adapters()}
    from .profile_paths import DATA_ROOT_LOCK_NAME

    known = {
        selected,
        selected.with_name("config.toml.lock"),
        source / "portable-location.json",
        source / "home" / DATA_ROOT_LOCK_NAME,
        root / ".generated_videos.capacity.lock",
    }
    known.update(_restored_cache_paths(config, root))
    known.update(
        Path(str(root / "tldw_chatbook_notes_sync_state.db") + suffix)
        for suffix in ("", "-wal", "-shm", "-journal")
    )
    diagnostic = owners["diagnostics.logs"]._log_path(config)
    if not diagnostic.is_absolute() or diagnostic.parent != root:
        raise ValueError("detached_selector_unsupported")
    known.add(diagnostic)
    for _, setting, _, _ in DATABASE_PATHS:
        value = config.get("database", {}).get(setting)
        if value and Path(value).is_absolute():
            known.update(
                Path(str(value) + suffix) for suffix in ("", "-wal", "-shm", "-journal")
            )
    trees = {
        source / "cache",
        source / "tmp",
        source / "wezterm",
        source / "home" / ".cache",
        source / "home" / "AppData" / "Local",
        selected.parent / "recovery-bootstrap",
        selected.parent / "recovery",
        source / "home" / ".local" / "share" / "wezterm",
    }
    for _, _, leaf, _ in DATABASE_PATHS:
        known.update(
            Path(str(root / leaf) + suffix)
            for suffix in ("", "-wal", "-shm", "-journal")
        )
    known.add(root / "tldw_chatbook_tts_profiles.db.lock")
    for owner in owners.values():
        if isinstance(owner, _Definition) and owner.location != "package":
            path = (root if owner.location == "data" else selected.parent) / owner.leaf
            (trees if owner.tree else known).add(path)
        leaf = getattr(owner, "leaf", None)
        if leaf and owner.schema_policy() and owner.schema_policy().schema_sql:
            known.update(
                Path(str(root / leaf) + suffix)
                for suffix in ("", "-wal", "-shm", "-journal")
            )
    known.update(
        root / name
        for name in (
            ".instance.lock",
            "config.toml.lock",
            "hook_permissions.json",
            "hook_permissions.json.lock",
            "scheduler_heartbeat.json",
            "faulthandler.log",
            "tldw_cli_app.log",
            "model_catalog_cache.json",
            "local_mcp_store.json",
            "mcp_server_targets.json",
            "unified_mcp_context.json",
            "mcp_permissions.json",
            "mcp_permissions.json.bak",
            "mcp_execution_log.jsonl",
            "mcp_execution_log.jsonl.1",
            "file_notes.sqlite",
        )
    )
    known.update(
        selected.parent / name
        for name in ("runtime_policy.json", "eval_overrides.yaml", "rag_pipelines.toml")
    )
    trees.update(
        root / name
        for name in (
            "generated_images",
            "generated_videos",
            "temp",
            "models",
            "recovered_media",
            "research_paste_staging",
            "actor_pack_imports",
            "collections_archives",
            "change_review",
            "project_index",
            "briefing_audio",
            "tool_sandbox",
            "rag_profiles",
            "chromadb",
        )
    )
    if observation is None:
        from .detached_source import review_source

        observation = review_source(source)
    for record in observation.members:
        member = source / record.relative
        log_suffix = member.name.removeprefix(diagnostic.name + ".")
        rotation = (
            member.parent == root
            and member.name.startswith(diagnostic.name + ".")
            and log_suffix.isascii()
            and log_suffix.isdigit()
        )
        history = member.parent == selected.parent and (
            member.name == "config.toml.bak" or _CONFIG_HISTORY.fullmatch(member.name)
        )
        if (
            not record.directory
            and not history
            and not rotation
            and member not in known
            and not any(tree == member or tree in member.parents for tree in trees)
        ):
            raise ValueError("detached_unknown_content")
    return owners
