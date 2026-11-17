"""Fixed contained stock permission worker; only private framed stdin/stdout."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

from .permission_process import (
    ProtocolError,
    _Command,
    _Identity,
    _command_arguments,
    _facts,
    _fields,
    _origins,
    _unresolved_commands,
    _wire_result,
    read_message,
    write_message,
)


def _empty():
    from ..Backup_Recovery import raw_participants as raw
    from ..Backup_Recovery import storage_admission as storage

    with storage._lock:
        return not any(
            (
                storage._startups,
                storage._pending_acquisitions,
                storage._live_leases,
                storage._holds,
                storage._retiring_holds,
                storage._raw_operations,
                storage._operations,
                raw._states,
                _unresolved_commands,
            )
        )


def serve(input_stream, output_stream):
    """Permanently stop intake before startup retirement and Drained."""
    first = read_message(input_stream)
    identity = _Identity(first.get("epoch"), first.get("request"))
    _, data = identity.accept(first, {"Initialize"})
    _fields(data, {"facts"})
    _fields(data["facts"], {"selected", "selector", "bootstrap"})
    # No config import, native enrollment or source selection before containment.
    from .. import config
    from ..Backup_Recovery import bootstrap, mcp_source_participants as sources
    from ..Backup_Recovery import storage_admission as storage
    from . import permission_store as store

    data_dir = config.get_user_data_dir()
    init = _Command(None, None)
    with init:
        source = store.MCPPermissionStore(data_dir / "mcp_permissions.json")
        facts = _facts(source)
        if facts != data["facts"] or sources.binding(source) != (
            "mcp.permissions",
            source.path,
            True,
        ):
            raise ProtocolError("worker_source_selection_changed")
    init.retirement()
    init.complete()
    commands = {
        "read_snapshot_strict": store.MCPPermissionStore.read_snapshot_strict,
        "read_profile_inventory_snapshot": store.MCPPermissionStore.read_profile_inventory_snapshot,
        "catalog_fingerprint": store.MCPPermissionStore.catalog_fingerprint,
        "list_profiles": store.MCPPermissionStore.list_profiles,
        "list_suggestion_dismissals": store.MCPPermissionStore.list_suggestion_dismissals,
        "get_global_default": store.MCPPermissionStore.get_global_default,
        "get_server_entry": store.MCPPermissionStore.get_server_entry,
        "get_tool_entry": store.MCPPermissionStore.get_tool_entry,
        "has_current_tool_arg_rule": store.MCPPermissionStore.has_current_tool_arg_rule,
        "list_tool_arg_rules": store.MCPPermissionStore.list_tool_arg_rules,
        "set_kill_switch": store.MCPPermissionStore.set_kill_switch,
        "save": store.MCPPermissionStore.save,
        "ensure_profile": store.MCPPermissionStore.ensure_profile,
        "set_global_default": store.MCPPermissionStore.set_global_default,
        "set_server_default": store.MCPPermissionStore.set_server_default,
        "set_tool_state": store.MCPPermissionStore.set_tool_state,
        "add_tool_arg_rule": store.MCPPermissionStore.add_tool_arg_rule,
        "remove_tool_arg_rule": store.MCPPermissionStore.remove_tool_arg_rule,
        "mark_config_changed": store.MCPPermissionStore.mark_config_changed,
        "dismiss_suggestion": store.MCPPermissionStore.dismiss_suggestion,
        "restore_suggestion": store.MCPPermissionStore.restore_suggestion,
        "clear_suggestion_dismissals": store.MCPPermissionStore.clear_suggestion_dismissals,
        "install_profile_if_absent": store.MCPPermissionStore.install_profile_if_absent,
        "update_imported_profile": store.MCPPermissionStore.update_imported_profile,
        "replace_profile_with_tombstone": store.MCPPermissionStore.replace_profile_with_tombstone,
    }
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
                "retirement": init.retirement(),
            },
        ),
    )
    epoch = identity.epoch
    used = {identity.request}
    while True:
        message = read_message(input_stream)
        identity = _Identity(epoch, message.get("request"))
        if identity.request in used:
            raise ProtocolError("reused_request")
        used.add(identity.request)
        kind, data = identity.accept(message, {"Command", "Shutdown"})
        if kind == "Shutdown":
            _fields(data, set())
            storage._shutdown()
            if not _empty():
                raise ProtocolError("worker_native_drain_unproven")
            write_message(output_stream, identity.frame("Drained", {"empty": True}))
            return
        _fields(data, {"operation", "arguments"})
        operation = data["operation"]
        arguments = _command_arguments(operation, data["arguments"])

        def exchange(kind, value, identity=identity):
            if kind == "Check":
                value = dict(value, facts=_facts(source))
            write_message(output_stream, identity.frame(kind, value))
            reply, response = identity.accept(
                read_message(input_stream),
                {"Continue", "Refuse"}
                if kind == "Check"
                else {"Acknowledge", "Refuse"},
            )
            _fields(response, set())
            if reply == "Refuse":
                raise ProtocolError("parent_source_refused")
            if not store._permission_process_definitions_current(source):
                raise ProtocolError("child_definition_changed")

        command = _Command(source, exchange)
        result = stamp = None
        status, error = "ok", None
        # Hold the same child source lock through wrapper exit and proof/completion.
        with source._path_lock:
            try:
                with command:
                    exchange("Check", {"site": "entry"})
                    if operation in {"load", "get_kill_switch"}:
                        captured = store._capture_permission_read_entry(
                            source, operation, publish=arguments["publish"]
                        )
                        if captured is None:
                            raise ProtocolError("child_source_ineligible")
                        result, _ = store._run_permission_read(source, captured)
                    else:
                        result = commands[operation](source, **arguments)
                receipt = command.retirement()
                command.complete()
                if operation == "save":
                    stamp = arguments["payload"].get("updated_at")
            except (
                store.ProfileMutationError,
                store.PermissionStoreSnapshotError,
                OSError,
                bootstrap.RecoveryRequired,
                ValueError,
                AttributeError,
                TypeError,
            ) as failure:
                if isinstance(failure, ProtocolError):
                    raise
                if any(
                    state.mcp_effects or state.uncertain for _, state in command.raw
                ):
                    source._permission_publication.fail()
                    source._mcp_persistence_error = "mcp_persistence_incomplete"
                    _unresolved_commands.append(command)
                    raise
                receipt = command.retirement()
                status = "refused"
                error = {
                    "kind": "mutation"
                    if isinstance(failure, store.ProfileMutationError)
                    else "snapshot"
                    if isinstance(failure, store.PermissionStoreSnapshotError)
                    else "io"
                    if isinstance(failure, OSError)
                    else "recovery"
                    if isinstance(failure, bootstrap.RecoveryRequired)
                    else "attribute"
                    if isinstance(failure, AttributeError)
                    else "type"
                    if isinstance(failure, TypeError)
                    else "value",
                    "category": "io_error"
                    if isinstance(failure, OSError)
                    else str(failure)
                    if isinstance(failure, bootstrap.RecoveryRequired)
                    else failure.category
                    if isinstance(
                        failure,
                        (
                            store.ProfileMutationError,
                            store.PermissionStoreSnapshotError,
                        ),
                    )
                    else str(failure),
                    "errno": failure.errno if isinstance(failure, OSError) else None,
                }
            except BaseException:
                source._permission_publication.fail()
                source._mcp_persistence_error = "mcp_persistence_incomplete"
                if command not in _unresolved_commands:
                    _unresolved_commands.append(command)
                raise
            publication = source._permission_publication
            new_origins = _origins(origins)
            origins.update(new_origins)
            write_message(
                output_stream,
                identity.frame(
                    "Settled",
                    {
                        "status": status,
                        "value": _wire_result(operation, result)
                        if status == "ok"
                        else None,
                        "error": error,
                        "stamp": stamp,
                        "facts": facts,
                        "origins": new_origins,
                        "retirement": receipt,
                        "publication": {
                            "digest": None
                            if publication.observed is None
                            else publication.observed.hex(),
                            "pending": publication.pending,
                            "uncertain": publication.uncertain,
                        },
                    },
                ),
            )


def main():
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    # Imported dependencies may print; protocol stdout is retained separately.
    sys.stdout = sys.stderr
    try:
        serve(input_stream, output_stream)
    except BaseException:
        # A failed command never exports a success/Drained, even on process exit.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
