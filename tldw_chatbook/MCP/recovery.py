"""Inert recovery of local MCP definitions, permissions and audit evidence."""

from dataclasses import dataclass

from tldw_chatbook.Backup_Recovery.models import OwnerAdapter
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


@dataclass(frozen=True)
class _Store(_RawDeclaration):
    leaf: str = ""

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
        return tuple(
            self._item(config, path.with_name(path.name + suffix), suffix.lstrip("."))
            for suffix in suffixes
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
