"""Installed recovery declaration for profile-owned subscriptions audio."""

from dataclasses import replace
import re
import stat

from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    OwnerAdapter,
    StorageItem,
)
from tldw_chatbook.Backup_Recovery.profile_paths import database_path, user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import (
    _ARTICLE_KEY,
    _RawDeclaration,
    article_id,
    article_tree,
)
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
from tldw_chatbook.Utils.platform_files import os


_OWNER = "subscriptions.assets"


def article_member(item: StorageItem | None) -> bool:
    """Recognize the installed flat article payload and its exact selecting edges."""
    if (
        item is None
        or item.owner != _OWNER
        or item.status != "included"
        or item.metadata is None
        or item.path is None
    ):
        return False
    parts = item.logical_id.split(":")
    meta = item.metadata
    if (
        len(parts) != 4
        or parts[0] != "profile"
        or parts[2] != _OWNER
        or not _ARTICLE_KEY.fullmatch(meta.relative_path)
    ):
        return False
    root = article_id(parts[1])
    return (
        meta.kind == "file"
        and meta.parent_id == meta.root_id == root
        and item.path.name == meta.relative_path
        and item.logical_id == article_id(parts[1], meta.relative_path)
        and set(item.dependencies)
        == {root, f"profile:{parts[1]}:config", f"profile:{parts[1]}:db.subscriptions"}
    )


class _BriefingAudio(_RawDeclaration):
    def absent_target_ids(
        self, source, declaration, producers, context, configured, target
    ):
        """Only a complete installed root observation can cover a source root absence."""
        from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id

        prefix = f"profile:{source}:{self.owner_id}"
        if (
            declaration.owner_id != self.owner_id
            or declaration.status != "unused"
            or declaration.shared_group
            or declaration.dependencies != (f"profile:{source}:config",)
        ):
            raise ValueError("selected_absence_mapping_required")
        configured = {**configured, DISCOVERY_CONTEXT_KEY: context}
        source_root = article_id(source)
        if declaration.logical_id == source_root:
            if any(row.logical_id.startswith(source_root + "-") for row in producers):
                raise ValueError("selected_absence_mapping_required")
            root = user_data_dir(configured) / "article_audio"
            observed = article_tree(configured, root)
        else:
            # All legacy briefing entries use opaque hashes, and detached
            # acquisition also has a bare absent root. A root absent on the
            # source cannot have any other legacy children in its inventory.
            legacy = [
                row
                for row in producers
                if row.owner_id == self.owner_id
                and (
                    row.logical_id == prefix
                    or re.fullmatch(re.escape(prefix) + ":[0-9a-f]{64}", row.logical_id)
                )
            ]
            if legacy != [declaration]:
                raise ValueError("selected_absence_mapping_required")
            root = user_data_dir(configured) / "briefing_audio"
            observed = self._tree(configured, root)
            same_location = _tree_member_id(
                DiscoveryContext(context.config_path, source), self.owner_id, root, root
            )
            if (
                observed[0].status != "unused"
                and declaration.logical_id != same_location
            ):
                raise ValueError("selected_absence_mapping_required")

        root_id = observed[0].logical_id
        local = {
            replace(row, shared_group=None)
            for row in target
            if row.owner == self.owner_id
            and row.logical_id.startswith(f"profile:{context.profile_id}:")
            and (
                row.logical_id == root_id
                or row.metadata is not None
                and row.metadata.root_id == root_id
                or row.path is not None
                and (row.path == root or root in row.path.parents)
            )
        }
        if (
            set(observed) != local
            or any(
                row.status not in {"included", "included_directory", "unused"}
                for row in observed
            )
            or observed[0].status not in {"unused", "included_directory"}
        ):
            raise ValueError("selected_absence_mapping_required")
        return {row.logical_id for row in observed if row.status != "unused"}

    def discover(self, config):
        from tldw_chatbook.Backup_Recovery.config_adapter import _excluded_root

        root = user_data_dir(config)
        database = database_path(config, "subscriptions_db_path")
        lock = database.with_name(database.name + ".article-listening.lock")
        lock_item = _excluded_root(
            config, self.owner_id, lock, kind="file", local_id="listening_lock"
        )
        if lock_item.status == "intentionally_excluded" and lock_item.metadata:
            try:
                with pinned_directory(lock.parent) as parent:
                    info = os.stat(lock.name, dir_fd=parent, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or info.st_uid != os.geteuid()
                        or info.st_mode & 0o077
                        or info.st_size
                    ):
                        lock_item = replace(lock_item, status="unsupported")
            except (OSError, ValueError, RuntimeError):
                lock_item = replace(lock_item, status="unavailable")
        return (
            *self._tree(config, root / "briefing_audio"),
            *article_tree(config, root / "article_audio"),
            lock_item,
        )

    def validate_restore_dependencies(self, item, candidate, candidates, *, topology):
        """Each restored article member needs its selecting subscriptions DB."""
        parts = item.logical_id.split(":")
        if len(parts) != 4 or not parts[3].startswith("article_audio"):
            return ()
        if (
            parts[:1] != ["profile"]
            or parts[2] != self.owner_id
            or item.metadata is None
        ):
            return ("dependency_unavailable",)
        profile = parts[1]
        root = article_id(profile)
        config_id = f"profile:{profile}:config"
        if (
            topology.get(root) != (root, None, "", "directory")
            or config_id not in item.dependencies
        ):
            return ("dependency_unavailable",)
        if item.logical_id == root:
            return ()
        name = item.metadata.relative_path
        db_id = f"profile:{profile}:db.subscriptions"
        if (
            not _ARTICLE_KEY.fullmatch(name)
            or item.logical_id != article_id(profile, name)
            or topology.get(item.logical_id) != (root, root, name, "file")
            or root not in item.dependencies
            or db_id not in item.dependencies
            or db_id not in candidates
        ):
            return ("dependency_unavailable",)
        return ()

    def restore_root(self, profile, payload, directories, producers, configured):
        """Resolve only a real same-owner root and its authenticated parent chain."""
        prefix = f"profile:{profile}:{self.owner_id}:"
        root = directories.get(payload.root_id)
        if (
            root is None
            or root.synthetic
            or root.logical_id != root.root_id
            or root.parent_id is not None
            or root.relative_path
            or root.logical_id not in producers
            or producers[root.logical_id].owner_id != self.owner_id
            or not root.logical_id.startswith(prefix)
        ):
            raise ValueError("owner_relocation_unverified:subscriptions.assets")
        relative = payload.relative_path
        article = root.logical_id == article_id(profile)
        if article:
            if (
                relative
                and not _ARTICLE_KEY.fullmatch(relative)
                or payload.logical_id != article_id(profile, relative)
                or payload.parent_id != (root.logical_id if relative else None)
                or relative
                and payload.logical_id in directories
            ):
                raise ValueError("owner_relocation_unverified:subscriptions.assets")
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", root.logical_id[len(prefix) :]):
                raise ValueError("owner_relocation_unverified:subscriptions.assets")
            node = payload
            while node.logical_id != root.logical_id:
                parent = directories.get(node.parent_id)
                if (
                    not node.logical_id.startswith(prefix)
                    or not re.fullmatch(r"[0-9a-f]{64}", node.logical_id[len(prefix) :])
                    or parent is None
                    or parent.synthetic
                    or parent.root_id != root.logical_id
                    or parent.logical_id not in producers
                    or producers[parent.logical_id].owner_id != self.owner_id
                    or not node.relative_path
                    or parent.relative_path != node.relative_path.rpartition("/")[0]
                ):
                    raise ValueError("owner_relocation_unverified:subscriptions.assets")
                node = parent
        return user_data_dir(configured) / (
            "article_audio" if article else "briefing_audio"
        )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    # SiteConfigManager stores definitions in the subscriptions DB owned by the
    # DB cohort; it is not a second physical capture of that file.
    return (_BriefingAudio("subscriptions.assets"),)
