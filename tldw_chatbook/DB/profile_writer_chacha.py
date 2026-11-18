"""Closed profile-writer operations for ChaChaNotes and Chatbook imports."""

from __future__ import annotations

import base64
import inspect
import sqlite3
import sys
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, fields
from datetime import date, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any


# A command is installed only by a decorator on a specific repository method.
# The request can never select a Python attribute, cursor, callback or SQL string.
_COMMANDS = {}
_LOCAL_ARGUMENTS = frozenset((
    "cursor", "transaction_callback", "_agent_lesson_context", "_agent_lesson_raw_arguments",
))
_TAG = "$chachanotes"


class _UnsupportedWriterValue(TypeError):
    """A client value needs its original in-process owner."""


def _pack(value, records=None):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {_TAG: "bytes", "value": base64.b64encode(value).decode("ascii")}
    if type(value) in (date, datetime):
        return {_TAG: type(value).__name__, "value": value.isoformat()}
    if type(value) in (tuple, set, frozenset):
        return {_TAG: type(value).__name__, "value": [_pack(item, records) for item in value]}
    if isinstance(value, list):
        return [_pack(item, records) for item in value]
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        entries = {key: _pack(item, records) for key, item in value.items()}
        return {_TAG: "mapping", "value": entries} if _TAG in entries else entries
    cls = type(value)
    if records and records.get(cls.__name__) is cls:
        payload = value.value if isinstance(value, Enum) else {
            field.name: getattr(value, field.name) for field in fields(cls) if field.init
        }
        return {_TAG: cls.__name__, "value": _pack(payload, records)}
    raise _UnsupportedWriterValue("Profile writer requires data-only arguments.")


def _unpack(value, records=None):
    if type(value) is list:
        return [_unpack(item, records) for item in value]
    if type(value) is not dict:
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("Invalid ChaChaNotes value.")
    if _TAG not in value:
        return {key: _unpack(item, records) for key, item in value.items()}
    if set(value) == {_TAG, "value"}:
        payload = value["value"]
        if value[_TAG] == "bytes" and type(payload) is str:
            return base64.b64decode(payload, validate=True)
        if value[_TAG] in ("date", "datetime") and type(payload) is str:
            return (date if value[_TAG] == "date" else datetime).fromisoformat(payload)
        if value[_TAG] in ("tuple", "set", "frozenset") and type(payload) is list:
            factory = {"tuple": tuple, "set": set, "frozenset": frozenset}[value[_TAG]]
            return factory(_unpack(item, records) for item in payload)
        if value[_TAG] == "mapping" and type(payload) is dict:
            return {key: _unpack(item, records) for key, item in payload.items()}
        if records and type(value[_TAG]) is str and value[_TAG] in records:
            cls = records[value[_TAG]]
            if issubclass(cls, Enum):
                return cls(payload)
            entries = _unpack(payload, records)
            if type(entries) is dict and entries.keys() == {field.name for field in fields(cls) if field.init}:
                return cls(**entries)
    raise ValueError("Invalid tagged ChaChaNotes value.")


def _client_profile(db) -> Path | None:
    from .ChaChaNotes_DB import CharactersRAGDB
    from ..Backup_Recovery.profile_paths import database_leaf

    if sys.platform != "win32" or type(db) is not CharactersRAGDB or db.is_memory_db:
        return None
    leaf = database_leaf("chachanotes_db_path")
    if db.db_path.name.casefold() != leaf.casefold() or db.managed_transaction_depth():
        return None
    from .profile_writer import in_profile_writer

    if in_profile_writer():
        return None
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        # Never split a transaction started through a borrowed native handle.
        try:
            if sqlite3.Connection.in_transaction.__get__(conn):
                return None
        except sqlite3.ProgrammingError:
            # A caller may have explicitly closed its previously cached handle.
            pass
    from ..config import get_user_data_dir

    profile = get_user_data_dir()
    return profile if db.db_path == profile / leaf else None


def _database_request(db, parameters, records=None):
    from ..Chat.console_library_policy import ConsoleLibraryMigrationSeed

    seed = db.console_library_migration_seed
    if seed is not None and type(seed) is not ConsoleLibraryMigrationSeed:
        raise TypeError("Invalid Console Library migration seed.")
    return {
        "client_id": db.client_id,
        "migration_seed": None if seed is None else seed.auto_retrieve_on_send,
        "parameters": _pack(parameters, records),
    }


def initialize_profile_schema(db) -> bool:
    """Run stock schema initialization/repair once on the elected writer."""
    profile = _client_profile(db)
    if profile is None:
        return False
    from .profile_writer import call_profile_writer

    db._schema_initial_version = call_profile_writer(
        profile, "chacha.initialize", _database_request(db, {}),
    )
    return True


def _domain_command(kind, function):
    operation = f"chacha.{kind}.{function.__name__}"
    signature = inspect.signature(function)
    _COMMANDS[operation] = (kind, function, signature)
    if sys.platform != "win32":
        return function

    @wraps(function)
    def routed(owner, *args, **kwargs):
        if any(kwargs.get(key) is not None for key in _LOCAL_ARGUMENTS):
            return function(owner, *args, **kwargs)
        if kind in ("console", "dispatch"):
            from .profile_writer_console import source_database

            db = source_database(kind, owner)
            if db is None:
                return function(owner, *args, **kwargs)
        elif kind == "notes":
            from ..Notes.Notes_Library import NotesInteropService

            if type(owner) is not NotesInteropService or owner._organization_failure_injector is not None:
                return function(owner, *args, **kwargs)
            bound = signature.bind(owner, *args, **kwargs)
            db = owner.notes_db(bound.arguments["user_id"])
        else:
            db = owner
        profile = _client_profile(db)
        if profile is None:
            return function(owner, *args, **kwargs)
        parameters = dict(signature.bind(owner, *args, **kwargs).arguments)
        del parameters["self"]
        if function.__name__ == "upsert_trajectory_rows":
            from .ChaChaNotes_DB import TrajectoryRowWrite

            rows = list(parameters["rows"])
            if not all(type(row) is TrajectoryRowWrite for row in rows):
                raise TypeError("Expected trajectory rows.")
            parameters["rows"] = [asdict(row) for row in rows]
        elif function.__name__ == "set_conversations_archived":
            parameters["conversation_ids"] = list(parameters["conversation_ids"])
        from .profile_writer import call_profile_writer

        records = None
        on_durable_commit = None
        if kind in ("console", "dispatch"):
            from .profile_writer_console import value_types, workspace_source

            workspace = workspace_source(kind, owner, parameters, profile)
            if workspace is False:
                return function(owner, *args, **kwargs)
            if function.__name__ == "replace_assistant_generation_projection_with_contributions":
                on_durable_commit = parameters.pop("on_durable_commit", None)
            records = value_types()
            parameters = {"call": parameters, "workspace_client_id": workspace}
        try:
            request = _database_request(db, parameters, records)
        except _UnsupportedWriterValue:
            # An extension's transaction contribution or process-local capability
            # stays with its original owner. No request has been transmitted.
            return function(owner, *args, **kwargs)
        result = _unpack(call_profile_writer(profile, operation, request), records)
        if on_durable_commit is not None:
            on_durable_commit()
        return result

    return routed


def chacha_mutation(function):
    """Route this repository-owned transaction when it has a stock profile."""
    return _domain_command("db", function)


def notes_mutation(function):
    """Route a complete stock Notes service operation, including its receipts."""
    return _domain_command("notes", function)


def console_mutation(function):
    """Route the stock persistence service's complete transaction."""
    return _domain_command("console", function)


def dispatch_mutation(function):
    """Route a dispatch checkpoint or terminal settlement transaction."""
    return _domain_command("dispatch", function)


def execute_profile_writer_command(profile: Path, operation: str, arguments: dict) -> Any:
    """Execute an installed mutation against the independently selected store."""
    from ..config import get_user_data_dir
    from .profile_writer import in_profile_writer

    if not in_profile_writer() or profile != get_user_data_dir():
        raise ValueError("ChaChaNotes writer profile mismatch.")
    if operation == "chatbook.import":
        from ..Chatbooks.chatbook_importer import _execute_profile_import

        return _execute_profile_import(profile, arguments)
    from ..app_services import open_chachanotes_db
    from ..Backup_Recovery.profile_paths import database_leaf
    from ..Chat.console_library_policy import ConsoleLibraryMigrationSeed
    from . import ChaChaNotes_DB

    if operation.startswith("chacha.notes."):
        from ..Notes.Notes_Library import NotesInteropService
    elif operation.startswith(("chacha.console.", "chacha.dispatch.")):
        from .profile_writer_console import execute_console_operation, value_types
    command = _COMMANDS.get(operation)
    if command is None and operation != "chacha.initialize":
        raise ValueError("Unknown ChaChaNotes writer command.")
    if set(arguments) != {"client_id", "migration_seed", "parameters"}:
        raise ValueError("Invalid ChaChaNotes writer arguments.")
    client_id = arguments["client_id"]
    seed = arguments["migration_seed"]
    if type(client_id) is not str or not client_id or (seed is not None and type(seed) is not bool):
        raise ValueError("Invalid ChaChaNotes writer identity.")
    console = command is not None and command[0] in ("console", "dispatch")
    records = value_types() if console else None
    parameters = _unpack(arguments["parameters"], records)
    if type(parameters) is not dict or any(parameters.get(key) is not None for key in _LOCAL_ARGUMENTS):
        raise ValueError("Invalid ChaChaNotes writer parameters.")
    if command is None:
        if parameters:
            raise ValueError("Schema initialization takes no parameters.")
    else:
        kind, function, signature = command
        if console:
            if set(parameters) != {"call", "workspace_client_id"} or type(parameters["call"]) is not dict:
                raise ValueError("Invalid Console writer request.")
            if "on_durable_commit" in parameters["call"]:
                raise ValueError("Publication belongs to the Console client.")
            signature.bind(None, **parameters["call"])
        else:
            signature.bind(None, **parameters)
        if kind == "notes" and parameters["user_id"].strip() != client_id:
            raise ValueError("Invalid Notes client identity.")
        if function.__name__ == "upsert_trajectory_rows":
            parameters["rows"] = [ChaChaNotes_DB.TrajectoryRowWrite(**row) for row in parameters["rows"]]
    path = profile / database_leaf("chachanotes_db_path")
    with closing(open_chachanotes_db(
        path, client_id=client_id,
        migration_seed=None if seed is None else ConsoleLibraryMigrationSeed(seed),
    )) as db:
        if command is None:
            return db._schema_initial_version
        if console:
            return _pack(execute_console_operation(kind, function, db, profile, parameters), records)
        if kind == "notes":
            service = NotesInteropService(path.parent, client_id, global_db_to_use=db)
            try:
                return _pack(function(service, **parameters))
            finally:
                service.close_all_user_connections()
        return _pack(function(db, **parameters))
