"""Retained run-log bytes from exact sandbox and registered workspace roots."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date, datetime, time
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
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit
from tldw_chatbook.Utils.platform_files import os


def _run_log_name_size(value, *, nested=False):
    """Bound str/repr of the closed native TOML value types without formatting."""
    if type(value) is str:
        # Container repr may escape each code point as \Uxxxxxxxx and add quotes.
        return 10 * len(value) + 2 if nested else len(value)
    if type(value) is bool:
        return 5
    if type(value) is int:
        # 30103/100000 is an upper bound on log10(2), plus sign/zero.
        return (value.bit_length() * 30103) // 100000 + 2
    if type(value) is float:
        return len("-1.7976931348623157e+308")
    if type(value) is datetime:
        return len(
            "datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=86399)))"
        )
    if type(value) is date:
        return len("datetime.date(9999, 12, 31)")
    if type(value) is time:
        return len("datetime.time(23, 59, 59, 999999)")
    if type(value) is list:
        return 2 + sum(_run_log_name_size(item, nested=True) + 2 for item in value)
    if type(value) is dict:
        return 2 + sum(
            _run_log_name_size(key, nested=True)
            + _run_log_name_size(item, nested=True)
            + 4
            for key, item in value.items()
        )
    raise ValueError("mobile_runs_selector_non_native_value")


def _run_log_names(config, *, _read_budget=None):
    """Installed RunLogWriter directory names, including its safe fallback."""
    name = os.environ.get("TLDW_AGENTS_RUN_LOG_DIR_NAME") or setting(
        config, "agents", "run_log_dir_name", "agent-runs"
    )
    if _read_budget is not None:
        size = max(_run_log_name_size(name), len("agent-runs"))
        # Native str, strip, Path component, dotted name and retained name facts.
        _read_budget.reserve(5 * (64 + 4 * (size + 1)))
        _read_budget.container(2)  # Native set, before its construction.
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
    return names


class _RunLogs(_RawDeclaration):
    def discover(self, config):
        return self.discover_for_stores(config)

    def discover_for_stores(
        self,
        config,
        stores=None,
        *,
        _runs_scope=None,
        _run_names=None,
        _read_budget=None,
    ):
        """Use the same installed log roots for explicitly reviewed legacy stores."""
        from tldw_chatbook.Backup_Recovery.mobile_preview_budget import (
            current,
            path_cost,
            reserve,
            table_cost,
        )
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
        from tldw_chatbook.DB.recovery_operations import recovery_adapters
        from tldw_chatbook.Skills_Interop.recovery import default_script_output_root

        context = discovery_context(config)
        native = current()
        if native is not None:
            if (
                _runs_scope is not None
                or _run_names is not None
                or _read_budget not in (None, native.read_budget)
            ):
                raise ValueError("mobile_runs_selector_scope_required")
            _read_budget = native.read_budget
            reserve(4096)
        if _runs_scope is not None:
            from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import (
                _CapturedSQLite,
            )

            if (
                type(_runs_scope) is not _CapturedSQLite
                or type(_run_names) is not frozenset
                or _read_budget is None
                or context.config_path != _runs_scope.captured.preview.source_selector
            ):
                raise ValueError("mobile_runs_selector_scope_required")
            _runs_scope.check()
        elif _run_names is not None or _read_budget is not None and native is None:
            raise ValueError("mobile_runs_selector_scope_required")
        script_output = default_script_output_root(config)
        configured = setting(config, "tools", "file_sandbox_root")
        default_root = (
            stores["db.agent_runs"].path.parent.parent
            if stores is not None
            else user_data_dir(config)
        ) / "tool_sandbox"
        selected_root = lexical_path(configured) if configured else default_root
        roots = {selected_root}
        retained_roots = set()
        issues = []
        unused_runs = None
        bound_root = selected_root != default_root
        # Current bindings and retained change history are exact owning references,
        # not authority to execute tools or enumerate arbitrary external contents.
        for owner, query in (
            (
                "db.workspaces",
                "SELECT locator FROM workspace_runtime_bindings WHERE binding_kind IN ('local_filesystem','local-filesystem')",
            ),
            ("db.agent_runs", "SELECT DISTINCT root FROM change_snapshots"),
        ):
            adapter = next(a for a in recovery_adapters() if a.owner_id == owner)
            item = stores[owner] if stores is not None else adapter.discover(config)[0]
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
                    table_cost(
                        connection,
                        "workspace_runtime_bindings"
                        if owner == "db.workspaces"
                        else "change_snapshots",
                        ("locator",) if owner == "db.workspaces" else ("root",),
                    )
                    for row in connection.execute(query):
                        reserve(
                            1024 + 32 * len(row[0]) if isinstance(row[0], str) else 1024
                        )
                        if (
                            not isinstance(row[0], str)
                            or not Path(row[0]).is_absolute()
                            or ".." in Path(row[0]).parts
                        ):
                            raise ValueError("invalid_log_root")
                        roots.add(Path(row[0]))
                        if stores is not None and owner == "db.agent_runs":
                            retained_roots.add(Path(row[0]))
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
        names = (
            _run_log_names(config, _read_budget=_read_budget)
            if _runs_scope is None
            else _run_names
        )
        reserve(1024 + 128 * (len(roots) + len(issues)))
        result = list(issues)
        retained_logs = False
        for root in sorted(roots):
            reserve(2048 + path_cost(root, copies=3))
            from tldw_chatbook.Backup_Recovery.file_inventory import _inventory_root
            from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id

            if root in retained_roots:
                retained = _inventory_root(root, owner=self.owner_id, external=False)
                if retained.status != "included_directory":
                    result.append(
                        replace(
                            retained,
                            logical_id=storage_logical_id(
                                context,
                                self.owner_id,
                                "legacy-root-"
                                + _tree_member_id(
                                    context, self.owner_id, root, root
                                ).split(":")[-1],
                            ),
                            status="missing_required"
                            if retained.status == "unused"
                            else "unsupported",
                        )
                    )
                    continue
            container = None
            if _read_budget is not None:
                _read_budget.container(len(names))
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

                        with pinned_directory(
                            root, _custody=native.custody if native else None
                        ) as parent:
                            if native is not None:
                                from tldw_chatbook.Backup_Recovery.bootstrap import (
                                    _bounded_names,
                                )

                                children = _bounded_names(
                                    parent, 100_000, _read_budget, native.custody
                                )
                                _read_budget.container(len(children))
                            else:
                                children = os.listdir(parent)
                            if set(children) - leaves:
                                container = replace(container, status="unsupported")
                    except _TraceReadLimit:
                        raise
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
                                    relative_path=item.path.relative_to(
                                        root
                                    ).as_posix(),
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
        # Canonical imports are not legacy input, even with coincident selectors.
        if stores is None:
            imported = self._tree(config, user_data_dir(config) / "agent_history")
            result.extend(imported)
            retained_logs |= any(
                item.status in {"included", "included_directory"} for item in imported
            )
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


class _HookPermissions(_RawDeclaration):
    """Local grant authority is recognized but never portable recovery input."""

    def discover(self, config):
        path = user_data_dir(config) / "hook_permissions.json"
        return tuple(
            replace(item, status="intentionally_excluded")
            if item.status in {"included", "unused"}
            else item
            for suffix in ("", ".lock")
            for item in (
                self._item(config, path.with_name(path.name + suffix), suffix),
            )
        )

    def validate(self, candidate: Path) -> tuple[str, ...]:
        return ("device_local_hook_permissions_not_restorable",)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_RunLogs("agents.history"), _HookPermissions("hooks.permissions"))
