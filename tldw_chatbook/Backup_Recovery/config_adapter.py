"""Pure configuration/definition recovery policy; no bootstrap or decryption."""

import hashlib
import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .models import OwnerAdapter, StorageItem, discovery_context, storage_logical_id
from .profile_paths import DATABASE_PATHS, default_config_path, setting, user_data_dir
from .recovery_files import _RawDeclaration

_CONFIG_HISTORY = re.compile(r"config_backup_[0-9]{8}_[0-9]{6}\.toml\Z")
_MAX_CONFIG_BYTES = 16 * 1024**2


def managed_secret_locations(
    config: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    """Expose the installed config encryption owner's key locations to task14.

    No values, environment contents or keychain reads escape. Arbitrary prose is
    never scanned or rewritten. Unknown encrypted blobs are not decrypted.
    """
    from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key

    locations = []

    def walk(value, prefix, depth):
        if depth > 64:
            raise ValueError("config_depth_limit")
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("invalid_config_key")
            if key.startswith("__chatbook_"):
                continue
            location = prefix + (key,)
            if is_sensitive_config_key(key):
                locations.append(location)
            elif isinstance(child, Mapping):
                walk(child, location, depth + 1)
            elif isinstance(child, list):
                for index, entry in enumerate(child):
                    if isinstance(entry, Mapping):
                        walk(entry, location + (str(index),), depth + 1)

    walk(config, (), 0)
    return tuple(sorted(locations))


# Exact installed managed location keys. Plain strings inside user content are
# not locators, and a mapping never authorizes a target or changes live config.
CONFIG_LOCATION_KEYS = tuple(("database", row[1]) for row in DATABASE_PATHS) + (
    ("paths", "data_dir"),
    ("Paths", "data_dir"),
    ("notes", "sync_directory"),
    ("console", "workspace_root"),
    ("llm_management", "model_download_dir"),
    ("embedding_config", "model_cache_dir"),
    ("app_tts", "CHATTERBOX_VOICE_DIR"),
    ("app_tts", "KOKORO_VOICE_BLENDS_DIR"),
    ("HiggsSettings", "voice_samples_dir"),
)


def remap_config_locations(
    config: Mapping[str, object], mapping: Mapping[str, Path]
) -> dict:
    """Map exact installed ``section.key`` selector names to new local paths.

    Unknown names and conflicting lower/uppercase data-dir aliases refuse.
    Original path strings are values, never mapping keys or substring targets.
    """
    accepted = {
        section + "." + key: (section, key) for section, key in CONFIG_LOCATION_KEYS
    }
    if any(
        key not in accepted or not isinstance(value, Path)
        for key, value in mapping.items()
    ):
        raise ValueError("invalid_relocation_mapping")
    if (
        "paths.data_dir" in mapping
        and "Paths.data_dir" in mapping
        and mapping["paths.data_dir"] != mapping["Paths.data_dir"]
    ):
        raise ValueError("ambiguous_relocation_mapping")
    result = deepcopy(dict(config))
    for selector, target in mapping.items():
        section, key = accepted[selector]
        values = result.setdefault(section, {})
        if type(values) is not dict:
            raise ValueError("invalid_config_selector")
        values[key] = str(target)
    return result


class _Config(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        item = self._item(config, context.config_path)
        return (
            replace(
                item,
                status="missing_required" if item.status == "unused" else item.status,
                dependencies=(),
            ),
        )

    def validate(self, candidate):
        from .storage_admission import _read_recovery_file

        try:
            data = _read_recovery_file(
                self.owner_id, candidate, max_bytes=_MAX_CONFIG_BYTES
            )
            tomllib.loads(data.decode("utf-8"))
            return ()
        except (OSError, ValueError, RuntimeError, UnicodeError, RecursionError):
            return ("config_validation_unavailable",)

    managed_secret_locations = staticmethod(managed_secret_locations)
    remap_locations = staticmethod(remap_config_locations)


class _History(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        current = context.config_path
        paths = [current.with_suffix(current.suffix + ".bak")]
        from tldw_chatbook.Utils.platform_files import os

        from .bootstrap import pinned_directory

        try:
            with pinned_directory(current.parent) as parent:
                with os.scandir(parent) as entries:
                    for entry in entries:
                        if _CONFIG_HISTORY.fullmatch(entry.name):
                            paths.append(current.parent / entry.name)
            return tuple(
                self._item(config, path, hashlib.sha256(str(path).encode()).hexdigest())
                for path in sorted(paths)
            )
        except (OSError, ValueError, RuntimeError):
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    current.parent,
                    "unavailable",
                    (storage_logical_id(context, "config"),),
                ),
            )


@dataclass(frozen=True)
class _Definition(_RawDeclaration):
    leaf: str = ""
    location: str = "data"
    tree: bool = False
    participant_pending: bool = False

    def _definition_path(self, config):
        context = discovery_context(config)
        base = {
            "data": user_data_dir(config),
            "config": context.config_path.parent,
            "package": Path(__file__).parents[1],
            "default_config": default_config_path().parent,
        }[self.location]
        return base / self.leaf

    def discover(self, config):
        context = discovery_context(config)
        path = self._definition_path(config)
        entries = self._tree(config, path) if self.tree else (self._item(config, path),)
        if self.participant_pending and any(
            item.status in {"included", "included_directory"} for item in entries
        ):
            # Explicit maintenance handoff: task10 cannot release startup admission
            # for these ordinary direct writers until their actual tokens drain.
            entries += (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "participant_pending"),
                    None,
                    "unsupported",
                    (storage_logical_id(context, "config"),),
                ),
            )
        return entries

    def _tree(self, config, root, **kwargs):
        context = discovery_context(config)
        entries = super()._tree(config, root, **kwargs)
        if entries:
            root_key = storage_logical_id(context, self.owner_id)
            old_key = entries[0].logical_id
            entries = tuple(
                replace(
                    item,
                    logical_id=root_key
                    if item.logical_id == old_key
                    else item.logical_id,
                    dependencies=tuple(
                        root_key if key == old_key else key for key in item.dependencies
                    ),
                    metadata=replace(
                        item.metadata,
                        root_id=root_key,
                        parent_id=root_key
                        if item.metadata.parent_id == old_key
                        else item.metadata.parent_id,
                    )
                    if item.metadata
                    else None,
                )
                for item in entries
            )
        return entries


class _ChatbookRegistry(_Definition):
    _reference_key = "__chatbook_archive_reference"

    @staticmethod
    def shared_dependencies(items):
        """Reconcile only ZIP dependencies of selected native registry aliases."""
        by_id = {item.logical_id: item for item in items}
        groups = {}
        for item in items:
            if (
                item.owner == "chatbooks.registry"
                and item.status == "included"
                and item.path is not None
                and item.shared_group
            ):
                groups.setdefault(item.shared_group, []).append(item)
        dependencies = {}
        for peers in groups.values():
            if len(peers) < 2 or len({peer.path for peer in peers}) != 1:
                continue
            archives = set()
            for peer in peers:
                prefix = peer.logical_id.removesuffix("chatbooks.registry")
                for key in peer.dependencies:
                    source = by_id.get(key)
                    if (
                        source is not None
                        and source.owner == "chatbooks.archives"
                        and source.status == "included"
                        and source.path is not None
                        and key.startswith(prefix + "chatbooks.archives:")
                    ):
                        archives.add(key)
            for peer in peers:
                dependencies[peer.logical_id] = tuple(
                    sorted(set(peer.dependencies) | archives)
                )
        return tuple(
            replace(item, dependencies=dependencies[item.logical_id])
            if item.logical_id in dependencies
            else item
            for item in items
        )

    @staticmethod
    def _reference_path(value, *, live=False):
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or (not live and not Path(value).is_absolute())
            or ".." in Path(value).parts
        ):
            raise ValueError("invalid_chatbook_reference")
        # Native create/update retain relative strings; native preview uses cwd.
        # Only live sources receive that interpretation, never imported records.
        return Path(value).absolute() if live else Path(value)

    @classmethod
    def _document(cls, data, *, live=False):
        import json

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_chatbook_key")
                result[key] = value
            return result

        document = json.loads(data, object_pairs_hook=unique)
        if not isinstance(document, dict) or not isinstance(
            document.get("records"), list
        ):
            raise TypeError("invalid_chatbook_records")
        for record in document["records"]:
            if not isinstance(record, dict):
                raise TypeError("invalid_chatbook_record")
            value = record.get("file_path")
            if cls._reference_key in record:
                marker = record[cls._reference_key]
                if value is not None or "file_path" not in record:
                    raise ValueError("invalid_chatbook_reference")
                if marker == {"status": "unresolved"}:
                    continue
                if (
                    not isinstance(marker, dict)
                    or set(marker) != {"logical_id"}
                    or not isinstance(marker["logical_id"], str)
                ):
                    raise ValueError("invalid_chatbook_reference")
            elif value is not None and value != "":
                cls._reference_path(value, live=live)
        return document

    @classmethod
    def _archive_reference(cls, item, record, sources=None):
        marker = record.get(cls._reference_key)
        if marker is None or marker == {"status": "unresolved"}:
            return None
        key = marker["logical_id"]
        profile = item.logical_id.split(":")
        prefix = ":".join(profile[:2]) + ":chatbooks.archives:"
        if (
            len(profile) != 3
            or profile[0] != "profile"
            or profile[2] != "chatbooks.registry"
            or key not in item.dependencies
        ):
            raise ValueError("invalid_chatbook_archive_dependency")
        if key.startswith(prefix) and key[len(prefix) :]:
            return key
        # Cross-profile references require this exact selected registry peer,
        # not merely a profile-shaped string or an imported shared-group label.
        referenced = key.split(":")
        sources = sources or {}
        peer = sources.get(":".join(referenced[:2]) + ":chatbooks.registry")
        archive = sources.get(key)
        if (
            len(referenced) != 4
            or referenced[0] != "profile"
            or referenced[2] != "chatbooks.archives"
            or not referenced[3]
            or sources.get(item.logical_id) != item
            or item.owner != "chatbooks.registry"
            or item.status != "included"
            or not item.shared_group
            or item.path is None
            or peer is None
            or peer.owner != "chatbooks.registry"
            or peer.status != "included"
            or peer.shared_group != item.shared_group
            or peer.path != item.path
            or key not in peer.dependencies
            or archive is None
            or archive.owner != "chatbooks.archives"
            or archive.status != "included"
            or archive.path is None
        ):
            raise ValueError("invalid_chatbook_archive_dependency")
        return key

    def prepare_capture(self, item, candidate, source_items):
        """Replace only staged locators using this capture's declared sources."""
        import json

        from .credentials import _read, _write

        document = self._document(_read(candidate), live=True)
        source_items = tuple(source_items)
        by_id = {source.logical_id: source for source in source_items}
        sources = {}
        for source in sorted(source_items, key=lambda row: row.logical_id):
            if (
                source.owner == "chatbooks.archives"
                and source.status == "included"
                and source.path is not None
                and source.logical_id in item.dependencies
            ):
                sources.setdefault(source.path, source.logical_id)
        for record in document["records"]:
            if self._reference_key in record:
                if record[self._reference_key] != {"status": "unresolved"}:
                    raise ValueError("live_chatbook_archive_marker")
                continue
            value = record.get("file_path")
            if value is None or value == "":
                continue
            key = sources.get(self._reference_path(value, live=True))
            record["file_path"] = None
            record[self._reference_key] = (
                {"logical_id": key} if key is not None else {"status": "unresolved"}
            )
            self._archive_reference(item, record, by_id)
        _write(candidate, json.dumps(document, ensure_ascii=False, indent=2))

    def validate_restore_reference_owners(
        self, item, candidate, payload_owners, source_items=()
    ):
        """Check authenticated payload ownership before any path relocation."""
        from .credentials import _read

        sources = {source.logical_id: source for source in source_items}
        for record in self._document(_read(candidate))["records"]:
            key = self._archive_reference(item, record, sources)
            if key is not None and payload_owners.get(key) != "chatbooks.archives":
                raise ValueError("invalid_chatbook_archive_owner")

    def relocate_restore(self, item, candidate, mapping, source_items=()):
        import json

        from .credentials import _read, _write

        document = self._document(_read(candidate))
        sources = {source.logical_id: source for source in source_items}
        for record in document["records"]:
            key = self._archive_reference(item, record, sources)
            if key is not None:
                if key not in mapping:
                    raise ValueError("chatbook_archive_mapping_required")
                if sources:
                    selected = sources.get(key)
                    if selected is None or selected.path != mapping[key]:
                        raise ValueError("chatbook_archive_mapping_required")
                record["file_path"] = str(self._reference_path(str(mapping[key])))
                del record[self._reference_key]
            elif record.get("file_path") not in (None, ""):
                raise ValueError("chatbook_archive_reference_required")
        _write(candidate, json.dumps(document, ensure_ascii=False, indent=2))

    def _definition_path(self, config):
        from .profile_paths import database_path

        # TldwCli._build_chatbook_db_paths always supplies Prompts; the installed
        # service selects its sibling first, ahead of ChaChaNotes and Media.
        return database_path(config, "prompts_db_path").with_name(
            "tldw_chatbook_chatbooks.json"
        )

    def discover(self, config):
        from .file_inventory import _inventory_root
        from .recovery_files import _tree_member_id
        from .storage_admission import _read_recovery_file

        context = discovery_context(config)
        entries = tuple(
            replace(
                item,
                dependencies=item.dependencies
                + (storage_logical_id(context, "db.prompts.primary"),),
            )
            for item in super().discover(config)
        )
        catalog = entries[0]
        if catalog.status != "included":
            return entries
        issues = []
        dependencies = set(catalog.dependencies)
        archive_root = user_data_dir(config) / "chatbooks"
        try:
            document = self._document(
                _read_recovery_file(
                    self.owner_id, catalog.path, max_bytes=16 * 1024**2
                ),
                live=True,
            )
            records = document["records"]
            if not isinstance(records, list):
                raise TypeError("invalid_chatbook_records")
            references = set()
            for record in records:
                if not isinstance(record, dict):
                    raise TypeError("invalid_chatbook_record")
                if self._reference_key in record:
                    if record[self._reference_key] != {"status": "unresolved"}:
                        raise ValueError("live_chatbook_archive_marker")
                    continue
                value = record.get("file_path")
                if value is None or value == "":
                    continue  # A metadata-only registry entry has no archive.
                if not isinstance(value, str):
                    raise TypeError("invalid_chatbook_reference")
                path = self._reference_path(value, live=True)
                # User-selected external destinations remain inert historical
                # locators. A registry string never adopts an external tree.
                if path != archive_root and archive_root not in path.parents:
                    continue
                if ".." in Path(value).parts:
                    raise ValueError("invalid_chatbook_reference")
                references.add(path)
            for path in sorted(references):
                item = _inventory_root(path, owner="chatbooks.archives", external=False)
                if item.status == "included":
                    dependencies.add(
                        _tree_member_id(
                            context, "chatbooks.archives", archive_root, path
                        )
                    )
                else:
                    key = (
                        "archive_reference_"
                        + hashlib.sha256(str(path).encode()).hexdigest()
                    )
                    issues.append(
                        StorageItem(
                            self.owner_id,
                            storage_logical_id(context, self.owner_id, key),
                            None,
                            "missing_required"
                            if item.status == "unused"
                            else "unsupported",
                            (catalog.logical_id,),
                        )
                    )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError):
            issues.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context, self.owner_id, "reference_catalog_unavailable"
                    ),
                    None,
                    "unavailable",
                    (catalog.logical_id,),
                )
            )
        return (
            replace(catalog, dependencies=tuple(sorted(dependencies))),
            *entries[1:],
            *issues,
        )


class _ChatbookArchives(_Definition):
    def discover(self, config):
        root = self._definition_path(config)
        return tuple(
            replace(item, status="unsupported")
            if item.path == root and item.status == "included"
            else item
            for item in super().discover(config)
        )


class _InstanceLock(_Definition):
    def discover(self, config):
        item = self._item(config, user_data_dir(config) / ".instance.lock")
        # Only the installed PID/portalocker file; a directory or linked alias
        # at that name is not evidence of this process-owned format.
        return (
            replace(item, status="intentionally_excluded")
            if item.status in {"included", "unused"}
            else item,
        )


class _ChatbookScratch(_Definition):
    def discover(self, config):
        root = user_data_dir(config) / "temp"
        result = []
        for item in self._tree(config, root):
            relative = item.path.relative_to(root) if item.path else Path()
            if relative.parts:
                # Creator and Importer each remove their per-run work directory
                # in finally. Neither root holds a recovery journal or catalog.
                if relative.parts[0] in {"chatbooks", "imports"}:
                    if len(relative.parts) == 1 and item.status == "included":
                        item = replace(item, status="unsupported")
                    elif item.status in {"included", "included_directory"}:
                        item = replace(item, status="intentionally_excluded")
                else:
                    item = replace(item, status="unsupported")
            result.append(item)
        return tuple(result)


def _excluded_root(config, owner, path, *, kind, local_id=""):
    """Apply installed exclusion policy only after checked exact-kind evidence."""
    from .file_inventory import _inventory_root

    context = discovery_context(config)
    item = _inventory_root(path, owner=owner, external=False)
    logical_id = storage_logical_id(context, owner, local_id)
    status = item.status
    if status == "unused":
        status = "intentionally_excluded"
    elif status in {"included", "included_directory"}:
        status = (
            "intentionally_excluded"
            if item.metadata is not None and item.metadata.kind == kind
            else "unsupported"
        )
    return replace(
        item,
        logical_id=logical_id,
        status=status,
        metadata=replace(item.metadata, root_id=logical_id, parent_id=None)
        if item.metadata
        else None,
    )


class _Generated(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        root = user_data_dir(config) / "generated_images"
        entries = self._tree(config, root)
        result = []
        for item in entries:
            relative = item.path.relative_to(root) if item.path else Path()
            if (
                relative.parts
                and relative.parts[0] == "temp"
                and not context.selections.temporary_media
            ):
                if len(relative.parts) == 1 and item.status == "included":
                    item = replace(item, status="unsupported")
                elif item.status in {"included", "included_directory"}:
                    item = replace(item, status="intentionally_excluded")
            elif relative.parts and (
                relative.parts[0] not in {"temp", "saved"}
                or context.selections.temporary_media
                and relative.parts[0] == "temp"
                and item.status == "included"
                and item.path.suffix.lower() != ".png"
            ):
                item = replace(item, status="unsupported")
            result.append(item)
        video_root = user_data_dir(config) / "generated_videos"
        # VideoStore._root_lease opens only this default sibling in append-binary
        # mode for portalocker. No payload is ever written into the lease file.
        lock = self._item(
            config,
            video_root.parent / ".generated_videos.capacity.lock",
            "video_capacity_lock",
        )
        if lock.status == "included":
            import stat

            from tldw_chatbook.Utils.platform_files import os

            from .bootstrap import pinned_directory

            try:
                with pinned_directory(lock.path.parent) as parent:
                    info = os.stat(lock.path.name, dir_fd=parent, follow_symlinks=False)
                    empty = (
                        stat.S_ISREG(info.st_mode)
                        and info.st_nlink == 1
                        and info.st_size == 0
                    )
                lock = replace(
                    lock, status="intentionally_excluded" if empty else "unsupported"
                )
            except (OSError, ValueError, RuntimeError):
                lock = replace(lock, status="unavailable")
        elif lock.status == "unused":
            lock = replace(lock, status="intentionally_excluded")
        result.append(lock)
        if context.selections.temporary_media:
            from tldw_chatbook.Video_Generation.video_metadata import (
                video_relative_path,
            )

            for item in self._tree(config, video_root):
                if item.status == "included":
                    relative = item.path.relative_to(video_root)
                    try:
                        valid = len(relative.parts) == 2 and video_relative_path(
                            relative.parts[0], item.path.stem, item.path.suffix[1:]
                        ) == relative
                    except ValueError:
                        valid = False
                    if not valid:
                        item = replace(item, status="unsupported")
                result.append(item)
        else:
            result.append(
                _excluded_root(
                    config,
                    self.owner_id,
                    video_root,
                    kind="directory",
                    local_id="temporary_videos",
                )
            )
        return tuple(result)


class _Diagnostics(_RawDeclaration):
    def _log_path(self, config):
        name = setting(config, "logging", "log_filename", "tldw_cli_app.log")
        if (
            type(name) is not str
            or not name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("invalid_diagnostic_selector")
        return user_data_dir(config) / name

    def _restore_path(self, config, relative_path):
        """Bind only the installed log basename and its decimal rotations."""
        path = self._log_path(config)
        if (
            type(relative_path) is str
            and relative_path
            and "/" not in relative_path
            and "\\" not in relative_path
        ):
            suffix = relative_path.removeprefix(path.name + ".")
            if relative_path == path.name or (
                relative_path.startswith(path.name + ".")
                and suffix.isascii()
                and suffix.isdigit()
            ):
                return path.with_name(relative_path)
        raise ValueError("owner_relocation_unverified:diagnostics.logs")

    def discover(self, config):
        from tldw_chatbook.Utils.platform_files import os

        from .bootstrap import pinned_directory

        context = discovery_context(config)
        path = self._log_path(config)
        root, name = path.parent, path.name
        paths = {path}
        try:
            with pinned_directory(root) as parent:
                with os.scandir(parent) as entries:
                    for entry in entries:
                        suffix = entry.name.removeprefix(name + ".")
                        if (
                            entry.name.startswith(name + ".")
                            and suffix.isascii()
                            and suffix.isdigit()
                        ):
                            paths.add(root / entry.name)
        except (OSError, ValueError, RuntimeError):
            # Failure to inspect the rotation namespace is not proof that its
            # installed artifacts are absent, even when diagnostics are off.
            return tuple(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context,
                        self.owner_id,
                        hashlib.sha256(str(path).encode()).hexdigest(),
                    ),
                    path,
                    "unavailable",
                    (),
                )
                for path in sorted(paths)
            )
        if not context.selections.diagnostics:
            return tuple(
                _excluded_root(
                    config,
                    self.owner_id,
                    path,
                    kind="file",
                    local_id=hashlib.sha256(str(path).encode()).hexdigest(),
                )
                for path in sorted(paths)
            )
        return tuple(
            self._item(config, path, hashlib.sha256(str(path).encode()).hexdigest())
            for path in sorted(paths)
        )


class _CatalogCache(_Definition):
    def discover(self, config):
        return (
            _excluded_root(
                config,
                self.owner_id,
                user_data_dir(config) / "model_catalog_cache.json",
                kind="file",
            ),
        )


@dataclass(frozen=True)
class _Attachments:
    """Semantic ownership of message attachment BLOBs in the shared core file."""

    owner_id: str = "chat.attachments"
    activation_required: bool = True

    @staticmethod
    def _core():
        from tldw_chatbook.DB.recovery_core import core_adapters

        return next(
            adapter
            for adapter in core_adapters()
            if adapter.owner_id == "db.chachanotes.primary"
        )

    def discover(self, config):
        context = discovery_context(config)
        from .profile_paths import database_path

        path = database_path(config, "chachanotes_db_path")
        return (
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id),
                path,
                "included" if path.is_file() else "missing_required",
                (storage_logical_id(context, "db.chachanotes.primary"),),
                shared_group="shared:chachanotes:profile:" + context.profile_id,
            ),
        )

    def schema_policy(self):
        return replace(self._core().schema_policy(), owner=self.owner_id)

    def validate(self, candidate):
        return self._core().validate(candidate)

    def capture(self, item, destination, cancel):
        if item.owner != self.owner_id:
            raise ValueError("invalid_capture_item")
        return self._core().capture(
            replace(item, owner="db.chachanotes.primary"), destination, cancel
        )

    def relocate(self, candidate, mapping):
        return self._core().relocate(candidate, mapping)


def config_adapter() -> OwnerAdapter:
    return _Config("config", format="toml", max_bytes=_MAX_CONFIG_BYTES)


class _ExternalFiles(_RawDeclaration):
    """Capture explicitly selected external entries declared by inventory."""

    def discover(self, config):
        # Inventory attaches the reviewed external selections to each profile.
        # This installed adapter supplies their raw capture/validation policy.
        return ()


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    """Installed config history and durable definition selectors only."""
    return (
        config_adapter(),
        _ExternalFiles("external.files"),
        _Attachments(),
        _Generated("generation.assets"),
        _Diagnostics("diagnostics.logs"),
        _CatalogCache("cache.model_catalog"),
        _InstanceLock("runtime.instance_lock"),
        _ChatbookScratch("runtime.chatbook_scratch"),
        _Definition("chat.prompt_history", leaf="prompt_history.jsonl"),
        _Definition(
            "ui.state",
            leaf="ui_state.toml",
            location="config",
        ),
        _Definition(
            "ui.emoji_recents",
            leaf="recent_emojis.json",
            location="config",
        ),
        _Definition(
            "ui.themes",
            leaf="themes",
            location="config",
            tree=True,
        ),
        _ChatbookRegistry("chatbooks.registry"),
        _ChatbookArchives("chatbooks.archives", leaf="chatbooks", tree=True),
        _Definition(
            "tokenizers.custom",
            leaf="tokenizers",
            location="default_config",
            tree=True,
        ),
        _History("config.history", max_bytes=_MAX_CONFIG_BYTES),
        _Definition("personas", leaf="tldw_chatbook_personas.json"),
        _Definition(
            "chat.dictionary_history",
            leaf="tldw_chatbook_chat_dictionary_history.json",
        ),
        _Definition(
            "chat.rag_context",
            leaf="tldw_chatbook_chat_rag_context.json",
        ),
        _Definition(
            "chat.grammars",
            leaf="tldw_chatbook_chat_grammars.json",
        ),
        _Definition("feedback", leaf="tldw_chatbook_feedback.json"),
        _Definition(
            "audio.history",
            leaf="tldw_chatbook_audio_history.json",
        ),
        _Definition("chat.dictionaries", leaf="chat_dicts", tree=True),
        _Definition(
            "chunking.templates",
            leaf="chunking_templates",
            tree=True,
        ),
        _Definition(
            "notes.templates",
            leaf="note_templates.json",
            location="config",
        ),
        _Definition(
            "chat.prompts", leaf="Chat/prompt_templates", location="package", tree=True
        ),
        _Definition(
            "generation.styles",
            leaf="image_generation_styles",
            tree=True,
        ),
    )
