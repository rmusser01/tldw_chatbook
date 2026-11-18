"""Closed domain commands for a Windows profile writer (ADR-126).

Only the decorated, stock AgentRuns methods are callable over the transport.
The client selects a profile, never a database path, SQL, or a transaction body.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
import sys
import uuid
from dataclasses import asdict
from functools import cache, wraps
from pathlib import Path
from typing import get_type_hints


_RUN_METHODS = {}


def encode_agent_definition(definition) -> dict:
    from tldw_chatbook.Agents.agent_models import AgentDefinition

    if type(definition) is not AgentDefinition:
        raise TypeError("An agent definition is required")
    return asdict(definition)


@cache
def _run_contract(method):
    """Use each closed method's declared request and result types at both ends."""
    from pydantic import ConfigDict, TypeAdapter, create_model

    signature = inspect.signature(method)
    hints = get_type_hints(method)
    fields = {
        name: (
            hints[name],
            ... if parameter.default is inspect.Parameter.empty else parameter.default,
        )
        for name, parameter in signature.parameters.items()
        if name != "self"
    }
    request = create_model(
        f"RunWriter_{method.__name__}",
        __config__=ConfigDict(extra="forbid", strict=True, allow_inf_nan=False),
        **fields,
    )
    return signature, request, TypeAdapter(hints["return"])


def profile_run_mutation(method):
    """Declare this complete repository operation as an allowed writer command."""
    name = method.__name__
    _RUN_METHODS[name] = method

    @wraps(method)
    def routed(database, *args, **kwargs):
        profile = database._writer_profile
        if profile is None:
            return method(database, *args, **kwargs)
        connection = getattr(database._thread_local, "conn", None)
        if connection is not None:
            try:
                nested = sqlite3.Connection.in_transaction.__get__(connection)
            except sqlite3.ProgrammingError:
                nested = False  # A previously closed read handle.
            if nested:
                # The original BEGIN on this connection must refuse a nested
                # AgentRuns transaction; an independent RPC could commit it.
                return method(database, *args, **kwargs)
        from tldw_chatbook.Backup_Recovery.participants import core_access
        from .profile_writer import (
            ProfileWriterOutcomeUnknown,
            call_bound_profile_writer,
        )

        core_access(database)
        if database.db_path != profile / "agent_runs.db" or database.db_path_str != str(
            database.db_path
        ):
            raise ValueError("AgentRuns writer selection changed")
        signature, _, result_type = _run_contract(method)
        bound = signature.bind(database, *args, **kwargs).arguments
        del bound["self"]
        if "defn" in bound:
            bound["defn"] = encode_agent_definition(bound["defn"])
        # Serialize before publishing run ownership or submitting a mutation.
        arguments = json.loads(json.dumps(bound, allow_nan=False))
        if name == "create_run":
            from .profile_writer_run_presence import claim_run, release_run

            run_id = arguments["run_id"] = arguments.get("run_id") or uuid.uuid4().hex
            claim_run(profile, run_id)
            try:
                result = call_bound_profile_writer(
                    database, profile, "agent_runs." + name, arguments
                )
            except BaseException:
                # The caller has no accepted live run. A committed row can
                # subsequently be reconciled, without replaying the request.
                release_run(profile, run_id)
                raise
        else:
            try:
                result = call_bound_profile_writer(
                    database, profile, "agent_runs." + name, arguments
                )
            except ProfileWriterOutcomeUnknown:
                # This executor cannot continue an unacknowledged mutation.
                # Once unwound, a later opener can repair a still-running row.
                if arguments.get("run_id"):
                    from .profile_writer_run_presence import release_run

                    release_run(profile, arguments["run_id"])
                raise
        try:
            result = result_type.validate_json(
                json.dumps(result, allow_nan=False), strict=True
            )
        except (ValueError, TypeError) as error:
            if arguments.get("run_id"):
                from .profile_writer_run_presence import release_run

                release_run(profile, arguments["run_id"])
            raise ProfileWriterOutcomeUnknown(
                "Invalid AgentRuns mutation receipt"
            ) from error
        if name in ("set_status", "set_terminal_with_step"):
            from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES
            from .profile_writer_run_presence import release_run

            if arguments["status"] in TERMINAL_RUN_STATUSES:
                release_run(profile, arguments["run_id"])
        return result

    return routed


_last_database = None


def _writer_database(path: Path, *, opening: bool):
    """Reuse the schema-initialized repository, with no handle held between calls."""
    global _last_database
    from .AgentRuns_DB import AgentRunsDB
    from .profile_writer import current_profile_writer_selection
    from .profile_writer_selection import writer_file_identity

    selection = current_profile_writer_selection()
    if not opening and _last_database is not None:
        database, identity, generation = _last_database
        if (
            database.db_path == path
            and generation == selection
            and writer_file_identity(path) == identity
        ):
            return database
    database = AgentRunsDB(path, reconcile_on_init=False)
    try:
        identity = writer_file_identity(path)
        if identity is None:
            raise RuntimeError("AgentRuns file disappeared during initialization")
        _last_database = database, identity, selection
        return database
    except BaseException:
        database.close()
        raise


def execute_profile_writer_command(profile: Path, operation: str, arguments: dict):
    """Validate the complete wire request before constructing a stock repository."""
    if type(operation) is not str or type(arguments) is not dict:
        raise ValueError("Invalid profile writer command")
    if operation.startswith("chacha.") or operation == "chatbook.import":
        from .profile_writer_chacha import (
            execute_profile_writer_command as execute_content,
        )

        return execute_content(profile, operation, arguments)
    # Loading the stock class registers its decorated methods, including on a
    # newly elected writer which has never constructed a repository.
    from . import AgentRuns_DB  # noqa: F401

    name = operation.removeprefix("agent_runs.")
    method = _RUN_METHODS.get(name)
    if not operation.startswith("agent_runs.") or (method is None and name != "open"):
        raise ValueError("Unknown profile writer command")
    if name == "open":
        if arguments:
            raise ValueError("Unexpected AgentRuns open arguments")
    else:
        _, request, result_type = _run_contract(method)
        try:
            arguments = vars(
                request.model_validate_json(json.dumps(arguments, allow_nan=False))
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid AgentRuns command arguments") from error

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from .profile_writer import ProfileWriterOutcomeUnknown, executing_writer_repository

    if not executing_writer_repository(profile, "db.agent_runs"):
        raise RecoveryRequired("agent_runs_writer_selection_changed")
    if name == "create_run" and sys.platform == "win32":
        from .profile_writer_run_presence import run_is_live

        if not arguments["run_id"] or not run_is_live(profile, arguments["run_id"]):
            raise ValueError("AgentRuns create requires a live executor")
    database = _writer_database(profile / "agent_runs.db", opening=name == "open")
    try:
        if name == "open":
            return database.receipt_capability_available
        if name == "reconcile_orphaned_runs":
            result = database._reconcile_orphaned_runs(profile)
        else:
            result = method(database, **arguments)
        # The transport accepts only JSON data. Preserve the public return
        # types (e.g. activity publication tuple) at the client boundary.
        try:
            return result_type.dump_python(
                result_type.validate_python(result, strict=True), mode="json"
            )
        except (TypeError, ValueError) as error:
            raise ProfileWriterOutcomeUnknown(
                "AgentRuns result could not be encoded"
            ) from error
    finally:
        try:
            database.close()
        except Exception as error:
            raise ProfileWriterOutcomeUnknown(
                "AgentRuns connection did not retire"
            ) from error
        if getattr(database._thread_local, "conn", None) is not None:
            raise ProfileWriterOutcomeUnknown("AgentRuns connection did not retire")


@cache
def _error_types():
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat.console_context_policy import ContextPolicyError
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDispatchCheckpointValidationError,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_models import (
        MixinPreferenceConflictError,
        MixinSnapshotUnavailableError,
        MixinSourceChangedError,
    )
    from tldw_chatbook.Chat.library_preparation import LibraryPreparationValidationError
    from tldw_chatbook.Notes.note_folder_models import NotesOrganizationRepositoryError
    from .AgentRuns_DB import AgentStepConflictError, ConsoleActivityReceiptsUnavailable
    from .ChaChaNotes_DB import (
        CharactersRAGDBError,
        ConflictError,
        InputError,
        SchemaError,
    )

    types = (
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        sqlite3.IntegrityError,
        sqlite3.OperationalError,
        sqlite3.DatabaseError,
        RecoveryRequired,
        AgentStepConflictError,
        ConsoleActivityReceiptsUnavailable,
        CharactersRAGDBError,
        ConflictError,
        InputError,
        SchemaError,
        NotesOrganizationRepositoryError,
        ConsoleDispatchCheckpointValidationError,
        ContextPolicyError,
        MixinPreferenceConflictError,
        MixinSnapshotUnavailableError,
        MixinSourceChangedError,
        LibraryPreparationValidationError,
    )
    return {kind.__name__: kind for kind in types}


def encode_profile_writer_error(error: Exception) -> dict:
    """Return only the fields required by known domain failures."""
    name = type(error).__name__
    if _error_types().get(name) is not type(error):
        return {"type": "RuntimeError", "message": "Profile writer operation failed"}
    message = error.args[0] if error.args and type(error.args[0]) is str else str(error)
    data = {"type": name, "message": message[:4096]}
    if name == "ConflictError":
        data.update(entity=error.entity, entity_id=error.entity_id)
    if name == "NotesOrganizationRepositoryError":
        data["reason_code"] = error.reason_code
    return data


def decode_profile_writer_error(data: dict) -> Exception:
    """Reconstruct a known domain exception without importing a wire-supplied type."""
    invalid = ValueError("Invalid profile writer error")
    if type(data) is not dict or type(data.get("type")) is not str:
        raise invalid
    kind = _error_types().get(data["type"])
    message = data.get("message")
    if kind is None or type(message) is not str or len(message) > 4096:
        raise invalid
    if data["type"] == "ConflictError":
        if (
            data.keys() != {"type", "message", "entity", "entity_id"}
            or type(data["entity"]) not in (str, type(None))
            or type(data["entity_id"]) not in (str, int, type(None))
        ):
            raise invalid
        return kind(message, entity=data["entity"], entity_id=data["entity_id"])
    if data["type"] == "NotesOrganizationRepositoryError":
        if (
            data.keys() != {"type", "message", "reason_code"}
            or type(data["reason_code"]) is not str
        ):
            raise invalid
        return kind(data["reason_code"], message)
    if data.keys() != {"type", "message"}:
        raise invalid
    return kind(message)
