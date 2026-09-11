"""Retained run-log bytes from exact sandbox and registered workspace roots."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    lexical_path,
    setting,
    user_data_dir,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class _RunLogs(_RawDeclaration):
    def discover(self, config):
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
        from tldw_chatbook.DB.recovery_operations import recovery_adapters

        context = discovery_context(config)
        configured = setting(config, "tools", "file_sandbox_root")
        default_root = user_data_dir(config) / "tool_sandbox"
        roots = {lexical_path(configured) if configured else default_root}
        issues = []
        unused_runs = None
        bound_root = bool(configured)
        # Current bindings and retained change history are exact owning references,
        # not authority to execute tools or enumerate arbitrary external contents.
        for owner, query in (
            (
                "db.workspaces",
                "SELECT locator FROM workspace_runtime_bindings WHERE binding_kind='local_filesystem'",
            ),
            ("db.agent_runs", "SELECT DISTINCT root FROM change_snapshots"),
        ):
            adapter = next(a for a in recovery_adapters() if a.owner_id == owner)
            item = adapter.discover(config)[0]
            if owner == "db.agent_runs" and item.status == "unused":
                unused_runs = item
                continue
            if item.status != "included":
                issues.append(
                    StorageItem(
                        self.owner_id,
                        storage_logical_id(context, self.owner_id, owner),
                        None,
                        "unavailable",
                        (item.logical_id,),
                    )
                )
                continue
            if adapter.validate(item.path):
                issues.append(
                    StorageItem(
                        self.owner_id,
                        storage_logical_id(context, self.owner_id, owner),
                        None,
                        "unsupported",
                        (item.logical_id,),
                    )
                )
                continue
            try:
                with closing(
                    connect_private_sqlite(
                        "recovery.operations.agent_logs", item.path, read_only=True
                    )
                ) as connection:
                    connection.execute("PRAGMA trusted_schema=OFF")
                    for row in connection.execute(query):
                        if (
                            not isinstance(row[0], str)
                            or not Path(row[0]).is_absolute()
                            or ".." in Path(row[0]).parts
                        ):
                            raise ValueError("invalid_log_root")
                        roots.add(Path(row[0]))
                        bound_root = True
            except (OSError, ValueError, sqlite3.Error):
                issues.append(
                    StorageItem(
                        self.owner_id,
                        storage_logical_id(context, self.owner_id, owner),
                        None,
                        "unavailable",
                        (item.logical_id,),
                    )
                )
        name = os.environ.get("TLDW_AGENTS_RUN_LOG_DIR_NAME") or setting(
            config, "agents", "run_log_dir_name", "agent-runs"
        )
        name = str(name).strip()
        if (
            not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or Path(name).is_absolute()
        ):
            name = "agent-runs"  # Exact RunLogWriter's safe component fallback.
        names = {name if name.startswith(".") else "." + name, name}
        result = list(issues)
        for root in sorted(roots):
            from tldw_chatbook.Backup_Recovery.file_inventory import _inventory_root
            from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id

            absent_default = (
                not configured
                and root == default_root
                and _inventory_root(root, owner=self.owner_id, external=False).status
                == "unused"
            )
            for leaf in sorted(names):
                path = root / leaf
                if absent_default:
                    result.append(
                        StorageItem(
                            self.owner_id,
                            _tree_member_id(context, self.owner_id, path, path),
                            path,
                            "unused",
                            (storage_logical_id(context, "config"),),
                        )
                    )
                else:
                    result.extend(self._tree(config, path))
        if unused_runs is not None and (
            bound_root
            or any(item.status in {"included", "included_directory"} for item in result)
        ):
            # Retained installed run logs are observable evidence of prior use.
            result.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "db.agent_runs"),
                    None,
                    "missing_required",
                    (unused_runs.logical_id,),
                )
            )
        return tuple(result)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_RunLogs("agents.history"),)
