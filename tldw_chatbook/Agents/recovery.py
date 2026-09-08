"""Retained run-log bytes from exact sandbox and registered workspace roots."""

from contextlib import closing
import os
from pathlib import Path
import sqlite3

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    user_data_dir,
    setting,
    lexical_path,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class _RunLogs(_RawDeclaration):
    def discover(self, config):
        from tldw_chatbook.DB.recovery_operations import recovery_adapters
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        context = discovery_context(config)
        configured = setting(config, "tools", "file_sandbox_root")
        roots = {
            lexical_path(configured)
            if configured
            else user_data_dir(config) / "tool_sandbox"
        }
        issues = []
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
            for leaf in sorted(names):
                result.extend(self._tree(config, root / leaf))
        return tuple(result)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_RunLogs("agents.history"),)
