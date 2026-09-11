"""Retained run-log bytes from exact sandbox and registered workspace roots."""

import os
import sqlite3
from contextlib import closing
from dataclasses import replace
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
        from tldw_chatbook.Skills_Interop.recovery import default_script_output_root

        context = discovery_context(config)
        script_output = default_script_output_root(config)
        configured = setting(config, "tools", "file_sandbox_root")
        default_root = user_data_dir(config) / "tool_sandbox"
        selected_root = lexical_path(configured) if configured else default_root
        roots = {selected_root}
        issues = []
        unused_runs = None
        bound_root = selected_root != default_root
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
        retained_logs = False
        for root in sorted(roots):
            from tldw_chatbook.Backup_Recovery.file_inventory import _inventory_root
            from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id

            container = None
            leaves = set(names)
            if root == default_root and script_output is not None:
                leaves.add(script_output.name)
            default_item = (
                _inventory_root(root, owner=self.owner_id, external=False)
                if root == default_root
                else None
            )
            absent_default = (
                default_item is not None and default_item.status == "unused"
            )
            if default_item is not None and not absent_default:
                key = _tree_member_id(context, self.owner_id, root, root)
                container = replace(
                    default_item,
                    logical_id=key,
                    dependencies=(storage_logical_id(context, "config"),),
                    metadata=replace(default_item.metadata, root_id=key, parent_id=None)
                    if default_item.metadata
                    else None,
                )
                if container.status == "included":
                    container = replace(container, status="unsupported")
                # Own the exact installed sandbox children, never arbitrary
                # content. Skills produces retained script output under this
                # same physical topology; bytes never grant script execution.
                if container.status == "included_directory":
                    try:
                        from tldw_chatbook.Backup_Recovery.native_files import (
                            pinned_directory,
                        )

                        with pinned_directory(root) as parent:
                            if set(os.listdir(parent)) - leaves:
                                container = replace(container, status="unsupported")
                    except (OSError, ValueError, RuntimeError):
                        container = replace(container, status="unavailable")
                result.append(container)
                if container.status != "included_directory":
                    continue
            for leaf in sorted(leaves):
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
                    group = self._tree(config, path)
                    if leaf in names and any(
                        item.status in {"included", "included_directory"}
                        for item in group
                    ):
                        retained_logs = True
                    if container is not None:
                        group = tuple(
                            replace(
                                item,
                                dependencies=item.dependencies + (container.logical_id,)
                                if item.path == path
                                else item.dependencies,
                                metadata=replace(
                                    item.metadata,
                                    root_id=container.logical_id,
                                    relative_path=str(item.path.relative_to(root)),
                                    parent_id=container.logical_id
                                    if item.path == path
                                    else item.metadata.parent_id,
                                )
                                if item.metadata
                                else None,
                            )
                            for item in group
                        )
                    result.extend(group)
        if unused_runs is not None and (bound_root or retained_logs):
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
