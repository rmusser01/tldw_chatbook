"""Inert RAG definitions and selector discovery; projections remain unqualified.

Never construct a profile manager or vector client here: both may migrate sources.
Original Task19 separately owns engine capture and restored retrieval readiness.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace

from tldw_chatbook.Backup_Recovery.models import (
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path, user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


def _pending(entries):
    return tuple(
        replace(
            item,
            status="unsupported" if item.status == "included" else item.status,
        )
        for item in entries
    )


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
    def validate(self, candidate):
        return ("rag_projection_capture_unqualified",)

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
        entries = tuple(
            item
            for root in sorted(roots)
            for item in _pending(
                _absent(config, self.owner_id, root) or self._tree(config, root),
            )
        )
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

    def capture(self, item, destination, cancel):
        raise ValueError("rag_projection_capture_unqualified")


def recovery_adapters():
    """Return declarations only; no engine imports, migrations, or writes."""
    from .rag_indexing import recovery_adapters as indexing_adapters

    return (
        _Definitions("rag.definitions"),
        _Projections("rag.projections"),
        *indexing_adapters(),
    )
