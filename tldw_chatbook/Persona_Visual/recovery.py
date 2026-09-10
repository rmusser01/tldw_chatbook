"""Installed artwork roots and exact retained/current core asset dependencies."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3

from tldw_chatbook.Backup_Recovery.config_adapter import _Definition
from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import database_path, user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id
from tldw_chatbook.Utils.path_validation import validate_recovery_relative_path


class _Assets(_Definition):
    def _references(self, candidate):
        from tldw_chatbook.DB.recovery_core import core_adapters
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        core = next(
            a for a in core_adapters() if a.owner_id == "db.chachanotes.primary"
        )
        if core.validate(candidate):
            raise ValueError("asset_core_unavailable")
        with closing(
            connect_private_sqlite("recovery.files.persona", candidate, read_only=True)
        ) as connection:
            if self.owner_id == "persona.assets":
                references = []
                for row in connection.execute(
                    "SELECT storage_relpath,bytes,sha256 FROM persona_visual_assets"
                ):
                    validate_recovery_relative_path(row[0])
                    references.append(row)
                    if len(references) > 100_000:
                        raise ValueError("asset_reference_limit")
                return tuple(references)
            source_kind = (
                "builtin"
                if self.owner_id == "persona.visual_identity_builtin"
                else "manual"
            )
            rows = connection.execute(
                "SELECT a.storage_relpath,a.bytes,a.sha256,a.preview_relpath,p.source_kind FROM visual_identity_assets a JOIN visual_identity_pack_versions v ON a.pack_version_id=v.id JOIN visual_identity_packs p ON v.pack_id=p.id"
            )
            references = []
            for count, (locator, size, digest, preview, kind) in enumerate(rows):
                if count >= 100_000:
                    raise ValueError("asset_reference_limit")
                if kind not in {"manual", "builtin"}:
                    raise ValueError("unsupported_visual_source_kind")
                if kind != source_kind:
                    continue
                validate_recovery_relative_path(locator)
                references.append((locator, size, digest))
                if preview is not None:
                    # No installed digest field for preview locators. Require
                    # their bytes; the capture manifest supplies their digest.
                    validate_recovery_relative_path(preview)
                    references.append((preview, None, None))
                if len(references) > 100_000:
                    raise ValueError("asset_reference_limit")
            return tuple(references)

    def _root(self, config):
        return (
            Path(__file__).parents[1] / "assets"
            if self.owner_id == "persona.visual_identity_builtin"
            else user_data_dir(config) / self.leaf
        )

    def _base_entries(self, config):
        if self.owner_id != "persona.visual_identity_builtin":
            return super().discover(config)
        context = discovery_context(config)
        root = self._root(config)
        references = self._references(database_path(config, "chachanotes_db_path"))
        selected = frozenset(locator for locator, _, _ in references)
        if not selected:
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    root,
                    "unused",
                    (),
                ),
            )
        return self._tree(config, root, selected_paths=selected)

    def _path(self, root, value):
        validate_recovery_relative_path(value)
        if not value:
            raise ValueError("invalid_asset_locator")
        if self.owner_id == "persona.assets":
            path = root.parent / value
        else:
            path = root / value
        path.relative_to(root)
        return path

    def discover(self, config):
        from tldw_chatbook.Backup_Recovery.storage_admission import (
            _digest_recovery_file,
        )

        context = discovery_context(config)
        root = self._root(config)
        entries = []
        try:
            entries = list(self._base_entries(config))
            if entries:
                entries[0] = replace(
                    entries[0],
                    dependencies=tuple(
                        dict.fromkeys(
                            (
                                *entries[0].dependencies,
                                storage_logical_id(context, "db.chachanotes.primary"),
                            )
                        )
                    ),
                )
            by_path = {item.path: index for index, item in enumerate(entries)}
            references = self._references(database_path(config, "chachanotes_db_path"))
            required = []
            for locator, size, digest in references:
                path = self._path(root, locator)
                key = _tree_member_id(context, self.owner_id, root, path)
                required.append(key)
                if path not in by_path:
                    entries.append(
                        StorageItem(
                            self.owner_id,
                            key,
                            path,
                            "missing_required",
                            (storage_logical_id(context, self.owner_id),),
                        )
                    )
                    continue
                index = by_path[path]
                item = entries[index]
                if item.status != "included":
                    continue
                actual = _digest_recovery_file(
                    self.owner_id, path, max_bytes=self.max_bytes
                )
                if size is not None and actual != (size, digest):
                    entries[index] = replace(item, status="unsupported")
            if entries and required:
                entries[0] = replace(
                    entries[0],
                    dependencies=tuple(
                        dict.fromkeys((*entries[0].dependencies, *required))
                    ),
                )
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            entries.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context, self.owner_id, "references_unavailable"
                    ),
                    None,
                    "unsupported",
                    (),
                )
            )
        return tuple(entries)

    def validate_dependencies(self, item, candidate, candidates):
        """Validate the semantic directory root's declared core and payload edges.

        Capture/restore consumers must validate included_directory items too;
        creating topology instead of copying a file does not bypass this check.
        """
        from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
        from tldw_chatbook.Backup_Recovery.storage_admission import (
            _digest_recovery_file,
        )

        parts = item.logical_id.split(":")
        if (
            len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != self.owner_id
            or item.path is None
        ):
            return ("invalid_dependency_context",)
        core_key = f"profile:{parts[1]}:db.chachanotes.primary"
        if core_key not in item.dependencies or core_key not in candidates:
            return ("dependency_unavailable",)
        context = DiscoveryContext(Path("/unused-config"), parts[1])
        try:
            for locator, size, digest in self._references(candidates[core_key]):
                source_path = self._path(item.path, locator)
                key = _tree_member_id(context, self.owner_id, item.path, source_path)
                if key not in item.dependencies or key not in candidates:
                    return ("dependency_unavailable",)
                actual = _digest_recovery_file(
                    self.owner_id, candidates[key], max_bytes=self.max_bytes
                )
                if size is not None and actual != (size, digest):
                    return ("asset_digest_mismatch",)
            return ()
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            return ("asset_dependency_unavailable",)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _Assets("persona.assets", leaf="persona_visual", tree=True),
        _Assets(
            "persona.visual_identity",
            leaf="visual_identities",
            tree=True,
        ),
        _Assets(
            "persona.visual_identity_builtin",
            leaf="assets",
            location="package",
            tree=True,
        ),
    )
