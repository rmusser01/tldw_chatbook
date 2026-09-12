"""Inert RAG definitions and installed Chroma root discovery.

Never construct a profile manager or vector client here: both may migrate sources.
Original Task19 separately owns engine capture and restored retrieval readiness.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from tldw_chatbook.Backup_Recovery.models import (
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path, user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


def _absent(config, owner, path):
    """Distinguish never-created roots from unreadable or aliased parents."""
    try:
        path.lstat()
    except FileNotFoundError:
        if not any(parent.is_symlink() for parent in path.parents):
            context = discovery_context(config)
            return (
                StorageItem(
                    owner,
                    storage_logical_id(
                        context, owner, hashlib.sha256(os.fsencode(path)).hexdigest()
                    ),
                    path,
                    "unused",
                    (storage_logical_id(context, "config"),),
                ),
            )
    except OSError:
        pass
    return None


def _mapping(value):
    if not isinstance(value, dict):
        raise TypeError("invalid_rag_selector")
    return value


def _text(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("invalid_rag_selector")
    return value.strip() or None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_rag_key")
        result[key] = value
    return result


def _engine(value):
    """Match the installed selector: auto falls through to the next source."""
    selected = (_text(value) or "auto").lower()
    return None if selected == "auto" else selected


@dataclass(frozen=True)
class _Definitions(_RawDeclaration):
    max_bytes: int = 16 * 1024**2

    def _document(self, item, candidate):
        import tomllib

        from .credentials import (
            _rag_definition_kind,
            _rag_definition_valid,
        )
        from .storage_admission import _read_recovery_file

        kind = _rag_definition_kind(item.metadata.relative_path)
        encoded = _read_recovery_file(
            self.owner_id, candidate, max_bytes=self.max_bytes
        )
        data = (
            tomllib.loads(encoded.decode())
            if kind == "pipeline"
            else json.loads(encoded)
        )
        if not _rag_definition_valid(kind, data):
            raise ValueError("rag_definition_format_unsupported")
        return kind, data

    @staticmethod
    def _selectors(kind, data):
        from .credentials import _rag_config_sections

        # Descriptions and historical free text are never filesystem selectors.
        for parent, key, _ in _rag_config_sections(kind, data):
            section = parent[key]
            for config in (
                section,
                section.get("rag", {}),
                section.get("rag_config", {}),
            ):
                config = _mapping(config)
                for selected in (
                    config,
                    config.get("vector_store", {}),
                    config.get("chroma", {}),
                ):
                    selected = _mapping(selected)
                    value = _text(selected.get("persist_directory"))
                    if value:
                        historical = Path(value)
                        if not historical.is_absolute() or ".." in historical.parts:
                            raise ValueError("rag_definition_root_mapping_required")
                        yield selected, historical

    @staticmethod
    def _roots(
        item, mapping, source_items, synthetic, *, topology=None, candidates=None
    ):
        """Use authenticated selected root declarations, never receiver path probes."""
        parts = item.logical_id.split(":")
        if len(parts) != 4 or parts[0] != "profile" or parts[2] != "rag.definitions":
            raise ValueError("rag_definition_root_mapping_required")
        config_id = f"profile:{parts[1]}:config"
        if not mapping or config_id not in item.dependencies:
            raise ValueError("rag_definition_root_mapping_required")
        config_path = mapping.get(config_id)
        if not isinstance(config_path, Path) or not config_path.is_absolute():
            raise ValueError("rag_definition_root_mapping_required")
        from .models import DiscoveryContext

        context = DiscoveryContext(config_path, parts[1])
        prefix = f"profile:{parts[1]}:rag.projections:"
        roots = []
        for root in source_items:
            metadata = root.metadata
            destination = mapping.get(root.logical_id)
            if (
                root.owner != "rag.projections"
                or not root.logical_id.startswith(prefix)
                or len(root.logical_id.split(":")) != 4
                or root.logical_id in synthetic
                or root.status != "included_directory"
                or metadata is None
                or metadata.kind != "directory"
                or metadata.root_id != root.logical_id
                or metadata.parent_id is not None
                or metadata.relative_path != ""
                or item.logical_id not in root.dependencies
                or config_id not in root.dependencies
                or not isinstance(destination, Path)
                or not destination.is_absolute()
                or (candidates is not None and root.logical_id not in candidates)
                or (
                    topology is not None
                    and topology.get(root.logical_id)
                    != (root.logical_id, None, "", "directory")
                )
            ):
                continue
            roots.append((root.logical_id, destination))
        return context, roots

    def relocate_restore(
        self, item, candidate, mapping, source_items=(), *, synthetic=()
    ):
        """Relocate known absolute selectors through their exact captured root ID."""
        from .credentials import _write
        from .recovery_files import _tree_member_id

        kind, data = self._document(item, candidate)
        selectors = tuple(self._selectors(kind, data))
        if not selectors:
            return
        context, roots = self._roots(item, mapping, source_items, synthetic)
        changed = False
        for selected, historical in selectors:
            key = _tree_member_id(context, "rag.projections", historical, historical)
            matches = [destination for root_id, destination in roots if root_id == key]
            if (
                len(matches) != 1
                or sum(destination == matches[0] for _, destination in roots) != 1
            ):
                raise ValueError("rag_definition_root_mapping_required")
            destination = str(matches[0])
            if selected["persist_directory"] != destination:
                selected["persist_directory"] = destination
                changed = True
        if changed:
            import toml

            _write(
                candidate, toml.dumps(data) if kind == "pipeline" else json.dumps(data)
            )

    def validate_restore_dependencies(
        self,
        item,
        candidate,
        candidates,
        *,
        topology,
        mapping=None,
        source_items=(),
        synthetic=(),
    ):
        """Check final selector targets in both staging and installed private copies."""
        if item.metadata is None or item.metadata.kind != "file":
            return ()
        try:
            kind, data = self._document(item, candidate)
            selectors = tuple(self._selectors(kind, data))
            if not selectors:
                return ()
            _, roots = self._roots(
                item,
                mapping,
                source_items,
                synthetic,
                topology=topology,
                candidates=candidates,
            )
            for _, selected in selectors:
                if sum(destination == selected for _, destination in roots) != 1:
                    return ("rag_definition_root_mapping_required",)
        except ValueError as error:
            if str(error) == "rag_definition_format_unsupported":
                return (str(error),)
            return ("rag_definition_root_mapping_required",)
        return ()

    def discover(self, config):
        from .credentials import _rag_definition_kind

        root = user_data_dir(config) / "rag_profiles"
        profiles = _absent(config, self.owner_id, root) or self._tree(config, root)
        pipeline = discovery_context(config).config_path.parent / "rag_pipelines.toml"
        pipeline_entries = _absent(config, self.owner_id, pipeline)
        if pipeline_entries is None:
            item = self._item(config, pipeline, "pipelines")
            if item.metadata is not None:
                item = replace(
                    item,
                    metadata=replace(item.metadata, relative_path=pipeline.name),
                )
            pipeline_entries = (item,)
        entries = profiles + pipeline_entries
        return tuple(
            replace(item, status="unsupported")
            if item.status == "included"
            and (
                item.metadata is None
                or _rag_definition_kind(item.metadata.relative_path) is None
            )
            else item
            for item in entries
        )


class _Projections(_RawDeclaration):
    def validate_restore_dependencies(self, item, candidate, candidates, *, topology):
        """Validate one complete private root in staging and installed-copy checks."""
        if item.metadata is None or item.logical_id != item.metadata.root_id:
            return ()
        from threading import Event

        from .limits import ArchiveLimits
        from .models import FileMetadata
        from .rag_projection_validation import validate_groups

        group = []
        for key, (root, parent, relative, kind) in topology.items():
            if root != item.logical_id or key not in candidates:
                continue
            group.append(
                replace(
                    item,
                    logical_id=key,
                    path=candidates[key],
                    status="included_directory" if kind == "directory" else "included",
                    dependencies=item.dependencies if key == root else (parent,),
                    metadata=FileMetadata(
                        1,
                        root,
                        relative,
                        parent,
                        kind,
                        0o700 if kind == "directory" else 0o600,
                        0,
                        "private",
                    ),
                )
            )
        limits = ArchiveLimits()
        validate_groups(
            group, candidates, candidate.parent, Event(), limits, limits.expanded_bytes
        )
        return ()

    def discover(self, config):
        context = discovery_context(config)
        roots = {user_data_dir(config) / "chromadb"}
        issues = []
        try:
            rag = _mapping(
                _mapping(config.get("AppRAGSearchConfig", {})).get("rag", {})
            )
            vector = _mapping(rag.get("vector_store", {}))
            legacy = _mapping(rag.get("chroma", {}))
            selected = (
                _text(os.environ.get("RAG_PERSIST_DIR"))
                or _text(vector.get("persist_directory"))
                or _text(legacy.get("persist_directory"))
            )
            # Keep the default as retained evidence only when it exists. An
            # explicit selection must not invent a second absent active root.
            if selected:
                default = next(iter(roots))
                roots = {lexical_path(selected)} | (
                    {default} if default.exists() else set()
                )
            kind = (
                _engine(os.environ.get("RAG_VECTOR_STORE"))
                or _engine(vector.get("type"))
                or "auto"
            ).lower()
            if kind not in {"auto", "memory", "chroma"}:
                raise ValueError("unsupported_rag_engine")
        except (ValueError, TypeError, OSError):
            issues.append("active_selector")

        definitions = _Definitions("rag.definitions").discover(config)
        profile_root = user_data_dir(config) / "rag_profiles"
        for item in definitions:
            path = item.path
            if (
                path is None
                or path.parent != profile_root
                or (
                    path.suffix != ".json"
                    and path.name != "custom_profiles.json.migrated"
                )
            ):
                continue
            if item.status == "unused":
                continue
            try:
                from tldw_chatbook.Backup_Recovery.storage_admission import (
                    _read_recovery_file,
                )

                payload = json.loads(
                    _read_recovery_file(
                        "rag.definitions", path, max_bytes=16 * 1024**2
                    ),
                    object_pairs_hook=_unique,
                )
                payload = _mapping(payload)
                profiles = (
                    payload.get("profiles")
                    if path.name
                    in {"custom_profiles.json", "custom_profiles.json.migrated"}
                    else [payload]
                )
                if not isinstance(profiles, list):
                    raise TypeError("invalid_legacy_rag_profiles")
                for profile in profiles:
                    rag = _mapping(_mapping(profile)["rag_config"])
                    vector = _mapping(rag.get("vector_store", {}))
                    kind = (_text(vector.get("type")) or "auto").lower()
                    if kind not in {"auto", "memory", "chroma"}:
                        raise ValueError("unsupported_rag_engine")
                    selected = _text(vector.get("persist_directory"))
                    if selected:
                        roots.add(lexical_path(selected))
            except (
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                RecursionError,
            ):
                issues.append(item.logical_id)
        from .profile_paths import database_path
        from .rag_projection_validation import recognized_path

        dependencies = [storage_logical_id(context, "config")]
        dependencies.extend(
            item.logical_id
            for item in definitions
            if item.status in {"included", "included_directory"}
        )
        for owner, setting in (
            ("db.rag_indexing", "rag_indexing_db_path"),
            ("db.media.primary", "media_db_path"),
            ("db.chachanotes.primary", "chachanotes_db_path"),
            ("db.prompts.primary", "prompts_db_path"),
        ):
            if database_path(config, setting).exists():
                dependencies.append(storage_logical_id(context, owner))
        entries = []
        for root in sorted(roots):
            group = _absent(config, self.owner_id, root) or self._tree(config, root)
            members = tuple(item.logical_id for item in group)
            for item in group:
                metadata = item.metadata
                if metadata is not None:
                    item = replace(
                        item,
                        status=item.status
                        if recognized_path(metadata.relative_path, metadata.kind)
                        else "unsupported",
                        dependencies=tuple(
                            dict.fromkeys(
                                (
                                    *item.dependencies,
                                    *dependencies,
                                    *(members if metadata.relative_path == "" else ()),
                                )
                            )
                        ),
                    )
                entries.append(item)
        entries = tuple(entries)
        if issues:
            entries += (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "selector_pending"),
                    None,
                    "unsupported",
                    (storage_logical_id(context, "config"),),
                ),
            )
        return entries


def recovery_adapters():
    """Return declarations only; no engine imports, migrations, or writes."""
    from .rag_indexing import recovery_adapters as indexing_adapters

    return (
        _Definitions("rag.definitions"),
        _Projections("rag.projections"),
        *indexing_adapters(),
    )
