"""Inert recovery of local MCP definitions, permissions and audit evidence."""

import re
from dataclasses import dataclass

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


@dataclass(frozen=True)
class _Store(_RawDeclaration):
    leaf: str = ""

    def validate_retained_destination(
        self, profile, payload, document, plan, canonical, config_target
    ):
        """Check explicit inert copies; their former bindings grant no authority."""
        prefix = f"profile:{profile}:"
        key = prefix + self.owner_id
        suffix = payload.logical_id.removeprefix(key + ":")
        leaves = {self.leaf}
        if self.owner_id == "mcp.permissions":
            leaves.add(self.leaf + ".bak")
        family = suffix == "fresh" or (
            self.owner_id == "mcp.permissions" and suffix == "fresh.bak"
        ) or re.fullmatch(r"history\.[0-9a-f]{64}", suffix)
        selected = dict(plan.restore)
        roots = dict(plan.destinations)
        producer = {row.logical_id: row for row in document.producer_inventory}
        files = {row.logical_id: row for row in document.files}
        directories = {row.logical_id: row for row in document.directories}
        item = producer.get(payload.logical_id)
        original = files.get(key)
        config = files.get(prefix + "config")
        root = directories.get(payload.root_id)
        target = selected.get(payload.logical_id)
        parent = roots.get(payload.root_id)
        if (
            self.owner_id not in {"mcp.local", "mcp.permissions", "mcp.context"}
            or not payload.logical_id.startswith(key + ":")
            or not family
            or payload.owner_id != self.owner_id
            or item is None or item.owner_id != self.owner_id
            or item.status != "included"
            or item.dependencies != (prefix + "config",)
            or original is None or original.owner_id != self.owner_id
            or selected.get(key) != canonical
            or config is None or config.owner_id != "config"
            or selected.get(prefix + "config") != config_target
            or root is None or not root.synthetic or root.parent_id is not None
            or payload.parent_id != root.logical_id
            or payload.relative_path not in leaves
            or suffix == "fresh" and payload.relative_path != self.leaf
            or suffix == "fresh.bak" and payload.relative_path != self.leaf + ".bak"
            or sum(row.root_id == payload.root_id for row in document.files) != 1
            or sum(row.root_id == payload.root_id for row in document.directories) != 1
            or parent is None or target != parent / payload.relative_path
            or target.parent == canonical.parent
        ):
            raise ValueError("owner_relocation_unverified:" + self.owner_id)

    def discover(self, config):
        path = user_data_dir(config) / self.leaf
        # Retained corruption backups and rotated logs are independent recovery
        # evidence. Never call load(), which can rename/reset corrupt stores.
        suffixes = (
            ("", ".bak")
            if self.owner_id == "mcp.permissions"
            else ("", ".1")
            if self.owner_id == "mcp.history"
            else ("",)
        )
        original = tuple(
            self._item(config, path.with_name(path.name + suffix), suffix.lstrip("."))
            for suffix in suffixes
        )
        from .recovery_activation import inventory_history, inventory_path

        context = discovery_context(config)
        try:
            fresh = inventory_path(context, self.owner_id, path)
            history = inventory_history(context, self.owner_id, path)
        except (OSError, ValueError, TypeError):
            return original + (StorageItem(self.owner_id, storage_logical_id(context, self.owner_id, "fresh"), None, "unavailable"),)
        return original + tuple(self._item(config, retained, suffix) for suffix, retained in history) + (
            tuple(self._item(config, fresh.with_name(fresh.name + suffix), "fresh" + suffix) for suffix in suffixes)
            if fresh is not None else ()
        )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _Store("mcp.local", "json", 16 * 1024**2, True, "local_mcp_store.json"),
        _Store("mcp.targets", "json", 16 * 1024**2, True, "mcp_server_targets.json"),
        _Store("mcp.context", "json", 16 * 1024**2, True, "unified_mcp_context.json"),
        # A corrupt retained backup remains useful bytes and is not valid live
        # permission input. This owner retains both generations as opaque data.
        _Store("mcp.permissions", "opaque", 16 * 1024**2, True, "mcp_permissions.json"),
        _Store("mcp.history", "opaque", 256 * 1024**3, True, "mcp_execution_log.jsonl"),
    )
