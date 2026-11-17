"""Default-disabled complete project-index key owner (ADR-126/230/353)."""

from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import weakref
from typing import TYPE_CHECKING

from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current as _definitions_current,
)
from tldw_chatbook.MCP.permission_process import (
    ProtocolError,
    PermissionProcess,
    _Command,
    _Identity,
    _DIGEST,
    _fields,
    read_message,
    write_message,
)
from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import ProducerLifetime

if TYPE_CHECKING:
    from .models import ProjectIndexKey


_resources = threading.local()
_owners_lock = threading.RLock()
_owners = set()


def _creator():
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    return os.getpid(), threading.current_thread(), storage._task_identity()


def _retain_pin(pin):
    receipt = getattr(_resources, "receipt", None)
    if receipt is not None:
        receipt.check_creator()
        receipt.pins.append(pin)


class _KeyResources:
    """Join existing exact storage retirement with the new SQLite/pin owners."""

    def __init__(self):
        self.creator = _creator()
        self.command = _Command(None, None)
        self.connections, self.pins = [], []
        self.close_failures = []
        self.service = None
        self.failure = None

    def check_creator(self):
        if self.creator != _creator():
            raise ProtocolError("foreign_key_resource_creator")

    def retain_connection(self, database, connection):
        self.check_creator()
        self.connections.append(
            (database, connection, getattr(database, "_maintenance_participant", None))
        )

    def retain_close_failure(self, database, connection, error):
        self.check_creator()
        self.close_failures.append((database, connection, error))

    def __enter__(self):
        from tldw_chatbook.DB.store_kernel import _connection_retirement

        self.check_creator()
        if (
            getattr(_resources, "receipt", None) is not None
            or getattr(_connection_retirement, "receiver", None) is not None
        ):
            raise ProtocolError("nested_key_command")
        self.command.__enter__()
        _resources.receipt = self
        _connection_retirement.receiver = self
        return self

    def __exit__(self, *args):
        from tldw_chatbook.DB.store_kernel import _connection_retirement

        self.check_creator()
        self.failure = args[1]
        _resources.receipt = None
        _connection_retirement.receiver = None
        self.command.__exit__(*args)

    def retirement(self):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Backup_Recovery.participants import retired_core_connections

        self.check_creator()
        valid = not self.close_failures and all(pin.closed for pin in self.pins)
        for database, connection, participant in self.connections:
            try:
                sqlite3.Connection.in_transaction.__get__(connection)
            except sqlite3.ProgrammingError:
                native_closed = True
            except BaseException:
                native_closed = False
            else:
                native_closed = False
            with storage._lock:
                valid &= (
                    native_closed
                    and participant is not None
                    and database._maintenance_participant is participant
                    and connection in retired_core_connections
                    and connection not in participant.connections
                    and not participant.retiring_threads
                )
        if self.service is not None:
            with self.service._lock:
                valid &= not self.service._operations and not self.service._runs
                valid &= not self.service._searches
            self.service._factory.assert_retired()
        if not valid:
            _unresolved_resources.append(self)
            raise ProtocolError("key_native_retirement_unproven") from self.failure
        try:
            result = self.command.retirement()
        except BaseException:
            _unresolved_resources.append(self)
            raise
        return dict(result, connections=len(self.connections), pins=len(self.pins))


_unresolved_resources = []


def _stock_sources(service, registry, database):
    from tldw_chatbook import config
    from tldw_chatbook.DB import Workspace_DB, base_db, store_kernel
    from tldw_chatbook.Workspaces import registry_service
    from . import runtime
    from . import models, service as service_module

    if (
        type(database) is not Workspace_DB.WorkspaceDB
        or database.is_memory_db
        or type(registry) is not registry_service.LocalWorkspaceRegistryService
        or registry.db is not database
        or not runtime._console_source_current(
            service, getattr(service, "_console_preparation_source", None)
        )
        or getattr(database, "_read_only", False)
    ):
        return False
    for module in (
        config,
        Workspace_DB,
        base_db,
        store_kernel,
        registry_service,
        runtime,
        models,
        service_module,
    ):
        if vars(module) is not module._KEY_PROCESS_GLOBALS or not _definitions_current(
            module._KEY_PROCESS_DEFINITIONS
        ):
            return False
        for owner, name, _outer, _records in module._KEY_PROCESS_DEFINITIONS:
            for instance in (database, registry, service, service._factory):
                if (
                    isinstance(owner, type)
                    and isinstance(instance, owner)
                    and name in vars(instance)
                ):
                    return False
    source = getattr(service, "_key_preparation_source", None)
    if source is None or source[-1][0] is not service._registry_connections:
        return False
    return all(
        function.__code__ is code
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is keywords
        and function.__closure__ is closure
        and all(
            cell.cell_contents is value
            for cell, value in zip(closure or (), values, strict=True)
        )
        for function, code, defaults, keywords, closure, values in source
    )


def _ambient_eligible(service, database):
    from tldw_chatbook.Backup_Recovery import (
        storage_admission as storage,
        raw_participants as raw,
    )
    from tldw_chatbook.Backup_Recovery.participants import core_cached_connection
    from tldw_chatbook.DB.base_db import _OPERATION_OWNERS
    from .service import _Operation

    if (
        getattr(storage._operation_local, "operation", None) is not None
        or raw._local.operation is not None
    ):
        return False
    operation = getattr(service._operation_local, "operation", None)
    if (
        type(operation) is not _Operation
        or operation not in service._operations
        or operation.retirement_unproven
    ):
        return False
    if getattr(_OPERATION_OWNERS, "ids", set()) != {id(database)}:
        return False
    connection = core_cached_connection(
        database, getattr(database._thread_local, "conn", None)
    )
    return connection is None or not sqlite3.Connection.in_transaction.__get__(
        connection
    )


def _facts(database, factory):
    from tldw_chatbook.Backup_Recovery.bootstrap import default_bootstrap_root
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

    return {
        "protocol_version": 1,
        "selected_config_path": str(effective_config_path().absolute()),
        "selected_data_path": str(factory.data_root),
        "selected_workspace_db_path": str(database.db_path),
        "bootstrap_root": str(default_bootstrap_root().absolute()),
        "expected_data_authority_id": factory.identity,
    }


class KeyRequestCapture:
    """Exact parent-only composition and worker sources, never serialized authority."""

    def __init__(self, source, service):
        self.creator = _creator()
        self.source, self.app, self.service = source, source.app, service
        self.registry, self.database = source.registry, source.database
        self.factory = service._factory
        self.facts = _facts(self.database, self.factory)
        self.reader_source = service._console_preparation_source
        self.closures = service._key_preparation_source
        self.owner = None

    def require_current(self) -> None:
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
        from tldw_chatbook.Chat.console_project_index_preparation import (
            _KEY_PREPARATION_READERS,
        )

        if (
            self.creator != _creator()
            or not _definitions_current(_KEY_PREPARATION_READERS)
            or (
                self.owner is not None
                and getattr(self.app, "_project_index_key_owner", None)
                is not self.owner
            )
            or self.source.service is not self.service
            or self.service._factory is not self.factory
            or self.service._console_preparation_source is not self.reader_source
            or self.service._key_preparation_source is not self.closures
            or not _stock_sources(self.service, self.registry, self.database)
            or not _definitions_current(_DEFINITIONS)
            or self.facts != _facts(self.database, self.factory)
            or not _ambient_eligible(self.service, self.database)
            or getattr(self.app, "_project_index_key_closed", False)
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        self.source.require_current()


def derive_for_console(controller, workspace_id, binding_id, root, *, ordinary):
    """Select once inside the original retained worker; never replay after selection."""
    from . import runtime

    if not runtime._KEY_PROCESS_ENABLED:
        return ordinary(controller.app, workspace_id, binding_id, root)
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_preparation_reads import preparation_source_for
    from tldw_chatbook.Chat.console_project_index_preparation import (
        _ProjectIndexPreparation,
    )

    source = preparation_source_for(controller)
    service = getattr(controller.app, "_project_index_service", None)
    if (
        os.name != "nt"
        or type(source) is not _ProjectIndexPreparation
        or service is None
        or source.service is not service
        or not _stock_sources(service, source.registry, source.database)
        or not _ambient_eligible(service, source.database)
        or not _definitions_current(_DEFINITIONS)
        or source.database.db_path != config.get_workspaces_db_path().absolute()
        or service._factory.data_root != config.get_user_data_dir().absolute()
    ):
        return ordinary(controller.app, workspace_id, binding_id, root)
    capture = KeyRequestCapture(source, service)
    capture.require_current()
    with _owners_lock:
        if participant.closed or getattr(
            controller.app, "_project_index_key_closed", False
        ):
            raise ProtocolError("key_intake_closed")
        owner = getattr(controller.app, "_project_index_key_owner", None)
        if owner is None:
            owner = KeyProcess(controller.app, capture.facts)
            controller.app._project_index_key_owner = owner
            _owners.add(owner)  # Visible before launch, including cold initialization.
        if type(owner) is not KeyProcess or owner.facts != capture.facts:
            raise ProtocolError("key_profile_changed")
        capture.owner = owner
    return owner.derive_key(workspace_id, binding_id, root, parent_capture=capture)


def _decode_key(value, facts, workspace_id, binding_id):
    from .models import ProjectIndexKey

    fields = {
        "data_authority_id",
        "workspace_id",
        "binding_id",
        "locator_fingerprint",
        "root_identity_fingerprint",
    }
    _fields(value, fields)
    if (
        any(type(value[name]) is not str for name in fields)
        or value["workspace_id"] != workspace_id
        or value["binding_id"] != binding_id
        or value["data_authority_id"] != facts["expected_data_authority_id"]
        or any(
            not _DIGEST.fullmatch(value[name])
            for name in (
                "data_authority_id",
                "locator_fingerprint",
                "root_identity_fingerprint",
            )
        )
    ):
        raise ProtocolError("invalid_key_result")
    try:
        return ProjectIndexKey(**value)
    except ValueError as error:
        raise ProtocolError("invalid_key_result") from error


_SQLITE_FAILURES = (
    ("OperationalError", sqlite3.OperationalError),
    ("IntegrityError", sqlite3.IntegrityError),
    ("ProgrammingError", sqlite3.ProgrammingError),
    ("InterfaceError", sqlite3.InterfaceError),
    ("DataError", sqlite3.DataError),
    ("InternalError", sqlite3.InternalError),
    ("NotSupportedError", sqlite3.NotSupportedError),
    ("DatabaseError", sqlite3.DatabaseError),
    ("Error", sqlite3.Error),
)


def _decode_failure(value):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    _fields(value, {"kind", "category", "errno"})
    kind, category, number = value["kind"], value["category"], value["errno"]
    if (
        type(category) is not str
        or len(category) > 128
        or (number is not None and type(number) is not int)
    ):
        raise ProtocolError("invalid_key_error")
    if kind == "recovery" and number is None:
        return RecoveryRequired(category)
    if kind in {"value", "binding_unavailable"} and number is None:
        return ValueError(category)
    if kind == "io" and category == "io_error":
        return OSError(number, "project_index_key_read_failed")
    if kind == "sqlite" and number is None:
        for name, exception in _SQLITE_FAILURES:
            if category == name:
                return exception("project_index_key_read_failed")
    raise ProtocolError("invalid_key_error")


class KeyProcess:
    """One serialized fixed command for an exact app/profile association."""

    # These two methods are transport/physical containment only; no permission state.
    _accept_origins = PermissionProcess._accept_origins
    _retire_process = PermissionProcess._retire_process

    def __init__(self, app, facts):
        self.app = weakref.ref(app) if hasattr(app, "__weakref__") else lambda: app
        self.facts = facts
        self.lifetime = ProducerLifetime()
        self.lock = threading.Lock()
        self.epoch = secrets.token_hex(16)
        self.process = self.tree = self.monitor = None
        self.finished = threading.Event()
        self.deadline = None
        self.retired = True
        self.closed = False
        self.unresolved = None
        self.containment_failure = None
        self.receipts = []
        self.origins, self.origin_roots = {}, ()
        self.selected = 0
        self.state = "new"

    def _receipt(self, value, *, key_succeeded=False):
        fields = {
            "retired",
            "acquisitions",
            "raw",
            "leases",
            "descriptors",
            "startup",
            "connections",
            "pins",
        }
        _fields(value, fields)
        if value["retired"] is not True or any(
            type(value[name]) is not int or value[name] < 0
            for name in fields - {"retired"}
        ):
            raise ProtocolError("key_retirement_unproven")
        if key_succeeded and (value["connections"] < 1 or value["pins"] != 1):
            raise ProtocolError("key_retirement_unproven")
        self.receipts.append(value)

    def _start(self):
        from tldw_chatbook.STT.executor_process_tree import (
            ExecutorProcessTree,
            WorkerContainmentIdentity,
        )
        from tldw_chatbook.Tools.workspace_tool_executor import _PopenProcessAdapter
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

        self.state = "initializing"
        root = str(Path(__file__).resolve().parents[3])
        packages = [
            Path(path).resolve()
            for path in sys.path
            if path and Path(path).name == "site-packages"
        ]
        dependencies = [
            str(Path(path).resolve())
            for path in sys.path
            if path
            and any(
                Path(path).resolve().is_relative_to(package) for package in packages
            )
        ]
        executable = str(
            Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        )
        digest = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        code = (
            "import runpy,sys;sys.path[:0]="
            + repr([root, *dependencies])
            + ";runpy.run_module('tldw_chatbook.Workspaces.project_index.key_worker',run_name='__main__')"
        )
        self.process = subprocess.Popen(
            [executable, "-I", "-S", "-B", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name == "posix",
        )
        self.retired = False
        self.tree = ExecutorProcessTree(
            _PopenProcessAdapter(self.process),
            threading.Event(),
            WorkerContainmentIdentity(
                self.process.pid, self.process.pid if os.name == "posix" else None
            ),
        )
        self.tree.admit()

        def monitor():
            while not self.finished.wait(0.05):
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.tree.terminate_tree()
                    return

        self.monitor = threading.Thread(
            target=monitor, name="project-index-key-retirement", daemon=True
        )
        self.monitor.start()
        identity = _Identity(self.epoch, secrets.token_hex(16))
        write_message(self.process.stdin, identity.frame("Initialize", self.facts))
        _, data = identity.accept(read_message(self.process.stdout), {"Ready"})
        _fields(
            data,
            {
                "facts",
                "pid",
                "executable",
                "executable_digest",
                "origins",
                "schema_version",
                "retirement",
            },
        )
        if (
            data["facts"] != self.facts
            or data["pid"] != self.process.pid
            or data["executable"] != executable
            or data["executable_digest"] != digest
            or type(data["schema_version"]) is not int
            or data["schema_version"] != WorkspaceDB._CURRENT_SCHEMA_VERSION
        ):
            raise ProtocolError("key_worker_identity_changed")
        self.origin_roots = (
            Path(root),
            Path(executable).parent,
            *(Path(path) for path in dependencies),
        )
        self._accept_origins(data["origins"])
        self._receipt(data["retirement"], key_succeeded=True)
        self.state = "ready"

    def _unresolve(self, record):
        if self.unresolved is None:
            self.unresolved = record
        self.state = "unresolved"
        self.lifetime.close()
        if self.process is not None:
            # Physical containment is retired, but lost native proof remains sticky.
            try:
                self._retire_process()
            except BaseException as error:
                # Preserve cleanup failures without replacing the original outcome.
                self.containment_failure = error

    def derive_key(
        self,
        workspace_id: str,
        binding_id: str,
        root: Path,
        *,
        parent_capture: KeyRequestCapture,
    ) -> ProjectIndexKey:
        """Derive one key after queue freshness and positive native settlement.

        Args:
            workspace_id: Exact requested workspace.
            binding_id: Exact requested local binding.
            root: Caller-selected local root, independently checked in the child.
            parent_capture: Original issuing worker's retained source references.

        Returns:
            The validated immutable key after all command resources retire.

        Raises:
            ProtocolError: The selected command has an uncertain or invalid outcome.
        """
        parent_capture.require_current()
        with self.lifetime.operation(), self.lock:
            if self.closed or self.unresolved is not None:
                raise ProtocolError("key_owner_unavailable")
            parent_capture.require_current()
            identity = _Identity(self.epoch, secrets.token_hex(16))
            received = None
            settled = False
            refusal = None
            self.deadline = time.monotonic() + 120
            try:
                if self.process is None:
                    self._start()
                # Ready proves idle initialization retired; revocation here is known.
                settled = True
                parent_capture.require_current()
                settled = False
                self.state = "active"
                self.selected += 1
                write_message(
                    self.process.stdin,
                    identity.frame(
                        "Command",
                        {
                            "operation": "derive_project_index_key",
                            "workspace_id": workspace_id,
                            "binding_id": binding_id,
                            "root": str(root),
                        },
                    ),
                )
                _, check = identity.accept(read_message(self.process.stdout), {"Check"})
                _fields(check, {"site", "facts"})
                if check != {"site": "entry", "facts": self.facts}:
                    raise ProtocolError("invalid_key_boundary")
                try:
                    parent_capture.require_current()
                except BaseException as error:
                    refusal = error
                write_message(
                    self.process.stdin,
                    identity.frame("Continue" if refusal is None else "Refuse", {}),
                )
                _, received = identity.accept(
                    read_message(self.process.stdout), {"Settled"}
                )
                _fields(
                    received,
                    {"status", "value", "error", "facts", "origins", "retirement"},
                )
                self._receipt(
                    received["retirement"], key_succeeded=received["status"] == "ok"
                )
                self._accept_origins(received["origins"])
                if received["facts"] != self.facts:
                    raise ProtocolError("key_worker_selection_changed")
                if (
                    received["status"] == "ok"
                    and received["error"] is None
                    and refusal is None
                ):
                    result = _decode_key(
                        received["value"], self.facts, workspace_id, binding_id
                    )
                    failure = None
                elif received["status"] == "refused" and received["value"] is None:
                    failure = _decode_failure(received["error"])
                    result = None
                else:
                    raise ProtocolError("invalid_key_outcome")
                settled = True
                self.state = "ready"
                if refusal is not None:
                    raise refusal
                parent_capture.require_current()
                if failure is not None:
                    raise failure
                return result
            except BaseException as error:
                if not settled and self.process is not None:
                    self._unresolve((identity, parent_capture, received, error))
                raise
            finally:
                self.deadline = None

    def close(self) -> None:
        """Call only after intake closure and accepted producer settlement."""
        with self.lock:
            if self.unresolved is not None:
                raise ProtocolError("key_outcome_unresolved")
            if self.lifetime.calls:
                raise ProtocolError("key_producers_not_settled")
            self.closed = True
            self.lifetime.close()
            self.state = "closing"
            if self.process is not None and not self.retired:
                identity = _Identity(self.epoch, secrets.token_hex(16))
                self.deadline = time.monotonic() + 120
                try:
                    write_message(self.process.stdin, identity.frame("Shutdown", {}))
                    _, data = identity.accept(
                        read_message(self.process.stdout), {"Drained"}
                    )
                    _fields(data, {"empty"})
                    if data["empty"] is not True:
                        raise ProtocolError("key_worker_not_drained")
                    self.process.wait(timeout=10)
                    if self.process.returncode != 0:
                        raise ProtocolError("key_worker_exit_failed")
                    self._retire_process()
                except BaseException as error:
                    self._unresolve(("shutdown", identity, error))
                    raise
                finally:
                    self.deadline = None
            self.state = "drained"


class KeyProcessLifetime:
    """Stable census exists before lazy app ownership or child launch."""

    def __init__(self):
        self.closed = False
        self.snapshot = ()

    def _maintenance_close_admission(self):
        with _owners_lock:
            self.closed = True
            self.snapshot = tuple(_owners)
            for owner in self.snapshot:
                owner.lifetime.close()

    async def _maintenance_drain(self, deadline):
        from tldw_chatbook.MCP.console_snapshot import _owned_worker

        if not self.closed:
            raise ProtocolError("key_intake_open")
        for owner in self.snapshot:
            if owner.unresolved is not None or not await owner.lifetime.drain(deadline):
                return False
            if time.monotonic() >= deadline:
                return False
            try:
                await _owned_worker(owner.close)
            except ProtocolError:
                return False
        return self.drained()

    def drained(self) -> bool:
        with _owners_lock:
            return (
                self.closed
                and _owners.issubset(self.snapshot)
                and all(
                    owner.closed
                    and owner.retired
                    and owner.unresolved is None
                    and not owner.lifetime.calls
                    for owner in self.snapshot
                )
            )

    def _maintenance_resume(self):
        with _owners_lock:
            if not self.drained():
                raise ProtocolError("key_process_not_drained")
            for owner in self.snapshot:
                app = owner.app()
                if app is None or getattr(app, "_project_index_key_closed", False):
                    _owners.discard(owner)
                    continue
                owner.process = owner.tree = owner.monitor = None
                owner.finished = threading.Event()
                owner.epoch = secrets.token_hex(16)
                owner.closed = False
                owner.state = "new"
                owner.lifetime.resume()
            self.closed = False
            self.snapshot = ()


participant = KeyProcessLifetime()


async def close_app(app: object) -> None:
    """Close key intake before waiting for the whole original parent callback."""
    from tldw_chatbook.Chat.console_preparation_reads import drain_preparation_reads
    from tldw_chatbook.MCP.console_snapshot import _owned_worker

    with _owners_lock:
        app._project_index_key_closed = True
        owner = getattr(app, "_project_index_key_owner", None)
        if owner is not None:
            owner.lifetime.close()
    runtime = getattr(app, "console_runtime", None)
    if runtime is not None:
        await drain_preparation_reads(runtime._preparation_reads)
    if owner is not None:
        if not await owner.lifetime.drain(time.monotonic() + 120):
            raise ProtocolError("key_producers_not_settled")
        await _owned_worker(owner.close)
        with _owners_lock:
            _owners.discard(owner)


def _shutdown():
    for owner in tuple(_owners):
        if owner.unresolved is not None:
            if owner.process is not None:
                owner._retire_process()
        else:
            owner.close()


atexit.register(_shutdown)

# Explicit namespace slots retain imported aliases independently of __name__.
_DEFINITIONS = capture_definitions(
    tuple(
        (globals(), name, globals()[name])
        for name in (
            "_creator",
            "_stock_sources",
            "_ambient_eligible",
            "_facts",
            "derive_for_console",
            "_decode_key",
            "_decode_failure",
            "read_message",
            "write_message",
            "_fields",
            "_definitions_current",
            "_retain_pin",
        )
    )
)
_DEFINITIONS += capture_definitions(
    tuple(
        (owner, name, getattr(owner, name))
        for owner, names in (
            (
                _KeyResources,
                (
                    "retain_connection",
                    "retain_close_failure",
                    "__enter__",
                    "__exit__",
                    "retirement",
                ),
            ),
            (KeyRequestCapture, ("__init__", "require_current")),
            (
                KeyProcess,
                (
                    "__init__",
                    "_start",
                    "derive_key",
                    "_receipt",
                    "_accept_origins",
                    "_retire_process",
                    "_unresolve",
                    "close",
                ),
            ),
            (_Identity, ("__init__", "frame", "accept")),
        )
        for name in names
    )
)
