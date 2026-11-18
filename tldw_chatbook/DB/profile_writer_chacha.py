"""Closed profile-writer operations for ChaChaNotes and Chatbook imports."""

from __future__ import annotations

import base64
import inspect
import os
import sqlite3
import stat
import sys
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4


# A command is installed only by a decorator on a specific repository method.
# The request can never select a Python attribute, cursor, callback or SQL string.
_COMMANDS = {}
_LOCAL_ARGUMENTS = frozenset((
    "cursor", "transaction_callback", "_agent_lesson_context", "_agent_lesson_raw_arguments",
))
_TAG = "$chachanotes"


@dataclass(frozen=True)
class _OpenedRepository:
    db: Any
    generation: tuple
    schema: tuple[int, int]


# The elected interpreter dispatches serially. Entries contain a manager only
# after its native connection has closed. Startup/recovery enrollment is owned
# independently by the transport; a process restart begins with no receipts.
_OPENED: OrderedDict[tuple, _OpenedRepository] = OrderedDict()
_MAX_OPENED = 128


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
    profile = getattr(db, "_profile_writer_profile", None)
    if profile is None:
        return None
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        try:
            borrowed = sqlite3.Connection.in_transaction.__get__(conn)
        except sqlite3.ProgrammingError:
            # A caller may have explicitly closed its previously cached handle.
            pass
        else:
            from ..Backup_Recovery.bootstrap import RecoveryRequired
            from .profile_writer_selection import writer_file_identity

            if writer_file_identity(db.db_path) != db._local.profile_writer_file_identity:
                raise RecoveryRequired("profile_writer_database_changed")
            if borrowed:
                return None
    return None if db.managed_transaction_depth() else profile


def _database_request(db, parameters, records=None):
    from ..Chat.console_library_policy import ConsoleLibraryMigrationSeed

    seed = db.console_library_migration_seed
    if seed is not None and type(seed) is not ConsoleLibraryMigrationSeed:
        raise TypeError("Invalid Console Library migration seed.")
    return {
        "client_id": db.client_id,
        "opener": db._profile_writer_opener,
        "migration_seed": None if seed is None else seed.auto_retrieve_on_send,
        "parameters": _pack(parameters, records),
    }


def initialize_profile_schema(db) -> bool:
    """Run stock schema initialization/repair once on the elected writer."""
    from .ChaChaNotes_DB import CharactersRAGDB

    if sys.platform != "win32" or type(db) is not CharactersRAGDB or db.is_memory_db:
        return False
    from .profile_writer_selection import writer_repository_profile

    profile = writer_repository_profile(db.db_path, "db.chachanotes.primary")
    db._profile_writer_profile = profile
    if profile is None:
        return False
    from .profile_writer import call_profile_writer

    db._profile_writer_opener = uuid4().hex
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
        elif kind in ("file_import", "folder"):
            from .profile_writer_notes import source_database

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
        elif kind in ("file_import", "folder"):
            from .profile_writer_notes import prepare_iterables, value_types

            prepare_iterables(kind, parameters)
            records = value_types()
        try:
            request = _database_request(db, parameters, records)
        except _UnsupportedWriterValue:
            # An extension's transaction contribution or process-local capability
            # stays with its original owner. No request has been transmitted.
            if kind in ("file_import", "folder"):
                return function(owner, **parameters)
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


def note_import_mutation(function):
    """Route one reviewed effect, including its complete original transaction."""
    return _domain_command("file_import", function)


def note_folder_mutation(function):
    """Route an independently owned local folder or placement mutation."""
    return _domain_command("folder", function)


def console_mutation(function):
    """Route the stock persistence service's complete transaction."""
    return _domain_command("console", function)


def dispatch_mutation(function):
    """Route a dispatch checkpoint or terminal settlement transaction."""
    return _domain_command("dispatch", function)


def _file_identity(path):
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or not info.st_ino or info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return None
    return info.st_dev, info.st_ino, info.st_birthtime_ns


def _recovery_generation():
    """Observe the selected profile's original paired restore records under admission."""
    from ..Backup_Recovery import bootstrap
    from ..Backup_Recovery.profile_paths import effective_config_path

    root = bootstrap.default_bootstrap_root()
    key = bootstrap._key(str(effective_config_path()))
    observed = {}
    try:
        with bootstrap.pinned_directory(root) as parent:
            root_info = bootstrap.os.fstat(parent)
            for prefix in ("profile", "activation"):
                name = f"{prefix}-{key}.json"
                try:
                    bootstrap._read_record_bytes(parent, name, max_bytes=bootstrap.MAX_RECORD, _observed=observed)
                except FileNotFoundError:
                    observed[name] = None
            return root_info.st_dev, root_info.st_ino, tuple(observed.values())
    except FileNotFoundError:
        return None


def _schema_generation(db):
    connection = db.get_connection()
    with closing(connection.execute("PRAGMA main.schema_version")) as cursor:
        cookie = cursor.fetchone()[0]
    return cookie, db._get_db_version(connection)


@contextmanager
def _repository(path, opener, client_id, seed, *, initialize):
    from ..app_services import open_chachanotes_db
    from ..Backup_Recovery.bootstrap import RecoveryRequired
    from ..Chat.console_library_policy import ConsoleLibraryMigrationSeed
    from .profile_writer import current_profile_writer_selection

    key = (path, opener, client_id, seed)
    previous = _OPENED.pop(key, None)
    selection = (current_profile_writer_selection(), _recovery_generation())
    before = _file_identity(path)
    db = None
    receipt = None
    try:
        if not initialize and before and previous and previous.generation == (selection, before):
            db = previous.db
            if _schema_generation(db) == previous.schema:
                receipt = previous
            else:
                db.close_connection()
                db = None
        if db is None:
            db = open_chachanotes_db(
                path, client_id=client_id, migration_seed=None if seed is None else ConsoleLibraryMigrationSeed(seed),
            )
            identity = _file_identity(path)
            if identity is not None:
                receipt = _OpenedRepository(db, (selection, identity), _schema_generation(db))
        if before is not None and before != _file_identity(path):
            raise RecoveryRequired("profile_writer_database_changed")
        yield db
    finally:
        if db is not None:
            db.close_connection()
        if receipt is not None:
            _OPENED[key] = receipt
            if len(_OPENED) > _MAX_OPENED:
                _OPENED.popitem(last=False)


def execute_profile_writer_command(profile: Path, operation: str, arguments: dict) -> Any:
    """Execute an installed mutation against the independently selected store."""
    from .profile_writer import executing_writer_repository, in_profile_writer

    if not in_profile_writer():
        raise ValueError("ChaChaNotes writer profile mismatch.")
    if operation == "chatbook.import":
        from ..Chatbooks.chatbook_importer import _execute_profile_import
        from ..config import get_user_data_dir

        if profile != get_user_data_dir():
            raise ValueError("Chatbook writer profile mismatch.")
        return _execute_profile_import(profile, arguments)
    if not executing_writer_repository(profile, "db.chachanotes.primary"):
        raise ValueError("ChaChaNotes writer profile mismatch.")
    from ..Backup_Recovery.profile_paths import database_leaf
    from . import ChaChaNotes_DB

    if operation.startswith("chacha.notes."):
        from ..Notes.Notes_Library import NotesInteropService
    elif operation.startswith(("chacha.console.", "chacha.dispatch.")):
        from .profile_writer_console import execute_console_operation, value_types
    elif operation.startswith(("chacha.file_import.", "chacha.folder.")):
        from .profile_writer_notes import execute_notes_operation, value_types
    command = _COMMANDS.get(operation)
    if command is None and operation != "chacha.initialize":
        raise ValueError("Unknown ChaChaNotes writer command.")
    if set(arguments) != {"client_id", "opener", "migration_seed", "parameters"}:
        raise ValueError("Invalid ChaChaNotes writer arguments.")
    client_id = arguments["client_id"]
    opener = arguments["opener"]
    seed = arguments["migration_seed"]
    if (type(client_id) is not str or not client_id or (seed is not None and type(seed) is not bool)
        or type(opener) is not str or len(opener) != 32 or set(opener) - set("0123456789abcdef")):
        raise ValueError("Invalid ChaChaNotes writer identity.")
    console = command is not None and command[0] in ("console", "dispatch")
    local_notes = command is not None and command[0] in ("file_import", "folder")
    records = value_types() if console or local_notes else None
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
    with _repository(path, opener, client_id, seed, initialize=command is None) as db:
        if command is None:
            return db._schema_initial_version
        if console:
            return _pack(execute_console_operation(kind, function, db, profile, parameters), records)
        if local_notes:
            return _pack(execute_notes_operation(kind, function, db, parameters), records)
        if kind == "notes":
            service = NotesInteropService(path.parent, client_id, global_db_to_use=db)
            # This command implements the original user's existing Notes DB.
            # Its physical lifetime belongs to _repository for this call.
            service._db_instances[client_id] = db
            return _pack(function(service, **parameters))
        return _pack(function(db, **parameters))
