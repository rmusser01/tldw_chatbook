"""Isolated fixed key worker; only framed private stdin/stdout."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import sys

from tldw_chatbook.MCP.permission_process import (
    ProtocolError,
    _Identity,
    _fields,
    _origins,
    read_message,
    write_message,
)
from .key_process import (
    _KeyResources,
    _facts,
    _unresolved_resources,
    _SQLITE_FAILURES,
    _stock_sources,
    _definitions_current,
    _DEFINITIONS,
)


def _failure(error):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    category = str(error)
    # Exception bodies may contain a private path or SQL; wire only named categories.
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", category):
        category = "value_error"
    if isinstance(error, RecoveryRequired):
        kind = "recovery"
    elif isinstance(error, sqlite3.Error):
        kind = "sqlite"
        category = next(
            (name for name, exception in _SQLITE_FAILURES if type(error) is exception),
            "Error",
        )
    elif isinstance(error, OSError):
        kind, category = "io", "io_error"
    elif isinstance(error, ValueError):
        kind = (
            "binding_unavailable"
            if category == "index_binding_unavailable"
            else "value"
        )
    else:
        raise error
    return {
        "kind": kind,
        "category": category,
        "errno": error.errno if isinstance(error, OSError) else None,
    }


def _empty(database, service):
    from tldw_chatbook.MCP.permission_worker import _empty as storage_empty

    participant = getattr(database, "_maintenance_participant", None)
    return (
        storage_empty()
        and not _unresolved_resources
        and participant is not None
        and not participant.connections
        and not participant.retiring_threads
        and not service._operations
        and not service._runs
        and not service._searches
    )


def serve(input_stream, output_stream):
    first = read_message(input_stream)
    identity = _Identity(first.get("epoch"), first.get("request"))
    _, expected = identity.accept(first, {"Initialize"})
    _fields(
        expected,
        {
            "protocol_version",
            "selected_config_path",
            "selected_data_path",
            "selected_workspace_db_path",
            "bootstrap_root",
            "expected_data_authority_id",
        },
    )
    if (
        type(expected["protocol_version"]) is not int
        or expected["protocol_version"] != 1
        or any(
            type(value) is not str
            for name, value in expected.items()
            if name != "protocol_version"
        )
    ):
        raise ProtocolError("invalid_key_initialization")
    # Independently select the installed profile only after the parent contains us.
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from . import runtime

    data_path = config.get_user_data_dir().absolute()
    workspace_path = config.get_workspaces_db_path().absolute()
    initialization = _KeyResources()
    with initialization:
        database = WorkspaceDB(workspace_path, initialize_schema=False, must_exist=True)
        with runtime.project_index_registry_connections(database):
            schema = database.get_schema_version()
            if schema != WorkspaceDB._CURRENT_SCHEMA_VERSION:
                raise ValueError("index_registry_schema_unavailable")
        registry = LocalWorkspaceRegistryService(database)
        service = runtime._compose_project_index_service(registry)
        initialization.service = service
        facts = _facts(database, service._factory)
        if facts != expected or service._factory.data_root != data_path:
            raise ProtocolError("key_worker_source_selection_changed")
    receipt = initialization.retirement()
    executable = str(Path(sys.executable).resolve())
    origins = _origins()
    write_message(
        output_stream,
        identity.frame(
            "Ready",
            {
                "facts": facts,
                "pid": os.getpid(),
                "executable": executable,
                "executable_digest": hashlib.sha256(
                    Path(executable).read_bytes()
                ).hexdigest(),
                "origins": origins,
                "schema_version": schema,
                "retirement": receipt,
            },
        ),
    )
    epoch, used = identity.epoch, {identity.request}
    while True:
        message = read_message(input_stream)
        identity = _Identity(epoch, message.get("request"))
        if identity.request in used:
            raise ProtocolError("reused_key_request")
        used.add(identity.request)
        kind, data = identity.accept(message, {"Command", "Shutdown"})
        if kind == "Shutdown":
            _fields(data, set())
            asyncio.run(service.aclose())
            service._factory.assert_retired()
            database.close()
            storage._shutdown()
            if not _empty(database, service):
                raise ProtocolError("key_worker_native_drain_unproven")
            write_message(output_stream, identity.frame("Drained", {"empty": True}))
            return
        _fields(data, {"operation", "workspace_id", "binding_id", "root"})
        if data["operation"] != "derive_project_index_key" or any(
            type(data[name]) is not str or not data[name] or "\0" in data[name]
            for name in ("workspace_id", "binding_id", "root")
        ):
            raise ProtocolError("invalid_key_command")
        root = Path(data["root"])
        if not root.is_absolute():
            raise ProtocolError("invalid_key_root")
        command = _KeyResources()
        command.service = service
        value = error = None
        status = "ok"
        try:
            with command:
                write_message(
                    output_stream,
                    identity.frame(
                        "Check",
                        {"site": "entry", "facts": _facts(database, service._factory)},
                    ),
                )
                reply, response = identity.accept(
                    read_message(input_stream), {"Continue", "Refuse"}
                )
                _fields(response, set())
                if reply == "Refuse":
                    raise bootstrap.RecoveryRequired("console_snapshot_owner_changed")
                if not _stock_sources(
                    service, registry, database
                ) or not _definitions_current(_DEFINITIONS):
                    raise ProtocolError("key_worker_definition_changed")
                value = runtime._project_index_key_owned(
                    service, data["workspace_id"], data["binding_id"], root
                )
        except (
            bootstrap.RecoveryRequired,
            OSError,
            ValueError,
            sqlite3.Error,
        ) as failure:
            if isinstance(failure, ProtocolError):
                _unresolved_resources.append(command)
                raise
            error = _failure(failure)
            status = "refused"
        except BaseException:
            _unresolved_resources.append(command)
            raise
        receipt = command.retirement()
        new_origins = _origins(origins)
        origins.update(new_origins)
        write_message(
            output_stream,
            identity.frame(
                "Settled",
                {
                    "status": status,
                    "value": asdict(value) if status == "ok" else None,
                    "error": error,
                    "facts": _facts(database, service._factory),
                    "origins": new_origins,
                    "retirement": receipt,
                },
            ),
        )


def main():
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        serve(input_stream, output_stream)
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
