"""Exclusive physical-file grants for installed, caller-bound transactions."""

from __future__ import annotations

import ctypes
import hashlib
import os
import sqlite3
import sys
import threading
import time
from contextlib import ExitStack, closing, contextmanager, nullcontext
from uuid import uuid4

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from . import profile_writer as writer
from .profile_writer_agent_sources import AGENT_LOCAL_GRANT_KINDS
from .profile_writer_chacha_sources import CHACHA_LOCAL_GRANT_KINDS


_KINDS = AGENT_LOCAL_GRANT_KINDS | {"goal.checkpoint", "workflow.phase_control", "workflow.child_commit"}
_LIMIT = 1024
# A failed fence is owned by the process, independently of any Python thread.
# Neither a disconnect nor unwinding a stack establishes native retirement.
_uncertain = {}
_pending = {}  # Service-side native process and positive-retirement event.
_pending_lock = threading.Lock()
_local = threading.local()


def require_unparked_writer(profile):
    """A callback cannot issue another command to its own parked writer."""
    grant = getattr(_local, "grant", None)
    if grant is not None and grant.profile == profile:
        raise RecoveryRequired("profile_writer_local_transaction_active")


def _guard_name(identity):
    from tldw_chatbook.Utils.windows_files import _native

    digest = hashlib.sha256(f"{_native().user_sid}/{identity[0]}/{identity[1]}".encode()).hexdigest()
    return r"\\.\pipe\tldw-agent-file-" + digest


def _completion(identity, *, create):
    """Open the owner's private retirement event, independently of a Python thread."""
    import _winapi
    from .profile_writer_run_presence import _event_api

    api = _event_api()
    name = "Global\\tldw-agent-retired-" + _guard_name(identity).rsplit("-", 1)[1]
    if create:
        with writer.pipe.private_security_attributes() as attributes:
            deadline = time.monotonic() + 1
            while True:
                ctypes.set_last_error(0)
                handle = api.CreateEventW(ctypes.byref(attributes), True, False, name)
                error = ctypes.get_last_error()
                if not handle or error != 183:
                    break
                # A positively finished earlier caller may still be consuming
                # its final receipt. Never adopt its signalled event as our own.
                signalled = _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_OBJECT_0
                _winapi.CloseHandle(handle)
                if not signalled or time.monotonic() >= deadline:
                    raise RecoveryRequired("profile_writer_local_transaction_active")
                time.sleep(.01)
    else:
        handle = api.OpenEventW(_winapi.SYNCHRONIZE | 0x20000, False, name)
        error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    try:
        writer.pipe.require_owner(handle)
    except BaseException:
        _winapi.CloseHandle(handle)
        raise
    return handle


def _signal(handle):
    from tldw_chatbook.Utils.windows_files import _native

    native = _native()
    native.check(native.kernel.SetEvent(ctypes.c_void_p(handle)))


def _check_pending(identity):
    import _winapi

    with _pending_lock:
        previous = _pending.get(identity)
        if previous is not None:
            if _winapi.WaitForMultipleObjects(previous, False, 0) == _winapi.WAIT_TIMEOUT:
                raise RecoveryRequired("profile_writer_local_transaction_active")
            for handle in _pending.pop(identity):
                _winapi.CloseHandle(handle)


def _remember_original(identity):
    """A successor independently pins the client behind an outstanding file grant."""
    import _winapi

    with ExitStack() as proof:
        try:
            completed = _completion(identity, create=False)
            proof.callback(_winapi.CloseHandle, completed)
            with writer.pipe.client(_guard_name(identity)) as peer:
                process = writer.pipe.server_process(peer)
                proof.callback(_winapi.CloseHandle, process)
        except OSError:
            return  # The live exclusive pipe still denies this attempted mutation.
        with _pending_lock:
            if identity not in _pending:
                _pending[identity] = process, completed
                proof.pop_all()


def _observe_guard(handle, stop):
    """Accept process identity observations from successive elected generations."""
    while not stop.is_set():
        try:
            writer.pipe.accept(handle, .1)
        except TimeoutError:
            continue
        except OSError as error:
            if getattr(error, "winerror", None) != 232:  # ERROR_NO_DATA: an observer already closed.
                return
        try:
            # Peers query the kernel's server PID and close without sending
            # application bytes. No application operation exists on this pipe.
            writer.pipe.read_frame(handle, .5, maximum=1)
        except (OSError, EOFError, ValueError):
            pass
        writer.pipe.disconnect(handle)


@contextmanager
def reserve_writer_file(identity, *, observable=False):
    """Reserve the actual NTFS file until the physical mutation has finished."""
    if identity is None:
        # Until the elected opener creates this file there can be no local
        # transaction grant: every grant requires its pre-existing physical ID.
        yield
        return
    _check_pending(identity)
    with ExitStack() as scope:
        try:
            handle = scope.enter_context(writer.pipe.server(_guard_name(identity)))
        except OSError as error:
            if error.winerror not in (5, 231):
                raise
            _remember_original(identity)
            raise RecoveryRequired("profile_writer_local_transaction_active") from error
        if observable:
            stop = threading.Event()
            observer = threading.Thread(target=_observe_guard, args=(handle, stop), daemon=True)
            observer.start()
            scope.callback(observer.join)
            scope.callback(stop.set)
        yield


def _send(handle, token, state):
    writer.pipe.write_frame(handle, writer._encode({"id": token, "state": state}), writer._IO_SECONDS)


def _receive(handle, token, *, waiting=False):
    value = writer._decode(writer.pipe.read_frame(handle, None if waiting else writer._IO_SECONDS, maximum=_LIMIT))
    if set(value) != {"id", "state"} or value["id"] != token or type(value["state"]) is not str:
        raise ValueError("invalid profile writer transaction receipt")
    return value["state"]


def _expect(handle, token, expected, *, waiting=False):
    actual = _receive(handle, token, waiting=waiting)
    if actual.startswith("refused:"):
        raise RecoveryRequired(actual.removeprefix("refused:"))
    if actual != expected:
        raise ValueError("unexpected profile writer transaction receipt")


def _grant_target(kind):
    from tldw_chatbook.Backup_Recovery.profile_paths import database_leaf

    if type(kind) is str and kind in _KINDS:
        return "db.agent_runs", "agent_runs.db"
    if type(kind) is str and kind in CHACHA_LOCAL_GRANT_KINDS:
        return "db.chachanotes.primary", database_leaf("chachanotes_db_path")
    raise RecoveryRequired("profile_writer_invalid_transaction")


def _prepare(profile, facts, identity, kind, enrollment):
    from .profile_writer_repository import _artifacts
    from .profile_writer_selection import require_writer_repository, writer_file_identity

    owner, leaf = _grant_target(kind)
    path = profile / leaf
    with enrollment.command(profile), reserve_writer_file(identity):
        require_writer_repository(profile, owner, facts=facts)
        if writer_file_identity(path) != identity:
            raise RecoveryRequired("profile_writer_database_changed")
        holder = enrollment.agent_runs if owner == "db.agent_runs" else enrollment.chacha
        holder.retire()
        observed = _artifacts(path)[0]
        if observed is None or observed[:2] != identity:
            raise RecoveryRequired("profile_writer_database_changed")


def serve_grant(profile, facts, payload, token, handle, enrollment):
    """Park the elected intake while the authenticated original process runs."""
    import _winapi

    try:
        if (type(payload) is not dict or set(payload) != {"kind", "database"}
            or type(payload["database"]) is not list
            or len(payload["database"]) != 2 or any(type(n) is not int for n in payload["database"])):
            raise RecoveryRequired("profile_writer_invalid_transaction")
        identity = tuple(payload["database"])
        enrollment.executor.submit(_prepare, profile, facts, identity, payload["kind"], enrollment).result()
    except (RecoveryRequired, SystemExit) as error:
        reason = str(error)
        if isinstance(error, SystemExit):
            if not reason.startswith("Recovery required: "):
                raise
            reason = reason.removeprefix("Recovery required: ")
        _send(handle, token, "refused:" + reason)
        _expect(handle, token, "received")
        return
    pid = writer.pipe.client_pid(handle)
    process = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid)
    try:
        _send(handle, token, "offer")
        _expect(handle, token, "claimed")
        # The OS identifies both sides independently. The primary peer's pinned
        # process must still live when it publishes the new exclusive file pipe.
        with writer.pipe.client(_guard_name(identity)) as guard:
            if writer.pipe.server_pid(guard) != pid or not writer._process_running(process):
                raise ValueError("profile writer grant owner changed")
            with _pending_lock:
                _pending[identity] = process, _completion(identity, create=False)
                process = None  # The generation keeps this exact kernel object on EOF.
        _send(handle, token, "granted")
        try:
            _expect(handle, token, "retired", waiting=True)
        except (ValueError, EOFError, OSError):
            # A cooperating live client retains its distinct kernel guard on an
            # uncertain native outcome, including through writer re-election.
            return
        # A JSON retirement claim alone cannot hand custody back. Require a
        # signalled native outcome (or process death) and the exclusive file.
        try:
            with reserve_writer_file(identity):
                _send(handle, token, "complete")
        except RecoveryRequired:
            _send(handle, token, "refused:profile_writer_local_transaction_active")
        _expect(handle, token, "received")
    finally:
        if process is not None:
            _winapi.CloseHandle(process)


class _LocalOnly:
    def bind(self, connection):
        pass

    def retire(self, connection):
        pass

    def uncertain(self):
        pass


class _Grant:
    owner_id = "db.agent_runs"
    private_policy = "db.base"

    @staticmethod
    def profile(database):
        return database._writer_profile

    @staticmethod
    def connection(database):
        return getattr(database._thread_local, "conn", None)

    @staticmethod
    def source_identity(database, expected):
        return database._writer_file == expected

    @staticmethod
    def schema(database, connection):
        from .profile_writer_repository import _schema

        return _schema(connection)

    @staticmethod
    def close(database):
        database.close()

    def __init__(self, database, identity, facts, admission):
        from tldw_chatbook.Backup_Recovery import bootstrap
        from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
        from .private_sqlite import _ordinary_connections

        self.database = database
        self.profile, self.path = type(self).profile(database), database.db_path
        self.identity = identity
        self.facts = facts
        self.admission = admission
        self.scope = admission.execution_context(self.path)
        if self.scope[1] is None:
            raise RecoveryRequired("profile_writer_grant_unqualified")
        self.epoch = bootstrap._admission_epoch
        self.thread = threading.current_thread()
        self.participant = participants.repository_participant(database)
        self.prior = self.connection(database)
        self.prior_lease = _ordinary_connections.get(self.prior) if self.prior is not None else None
        self.conn = self.resource = self.synchronous = None
        self.artifacts = self.schema = None
        self.ambiguous = self.retired = False
        # Set by retire() when an enclosing owned call keeps the idle fresh handle.
        self.owner_retains = False
        with storage._lock:
            self.original_scopes = self._scopes(storage)
            self.original_connections = self._connections(storage)
            if self.participant.read_only or self.participant.closed:
                raise RecoveryRequired("profile_writer_grant_source_changed")
        if self.prior is not None:
            try:
                live = sqlite3.Connection.in_transaction.__get__(self.prior)
            except sqlite3.ProgrammingError:
                if self.prior_lease is not None:
                    raise RecoveryRequired("profile_writer_grant_source_changed") from None
                self.prior = None
            else:
                if live or self.prior_lease is None:
                    raise RecoveryRequired("profile_writer_grant_source_changed")

    def _scopes(self, storage):
        return {operation for operation in storage._operations
                if operation.thread is self.thread and operation.participant is self.participant}

    def _connections(self, storage):
        """Include a failed native acquisition even if no cache entry was published."""
        return {lease for lease in storage._live_leases
                if lease.resource_thread is self.thread and lease.resource_path == self.path}

    def _connection(self, conn):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from .private_sqlite import _ordinary_connections, _validated_owner_policy

        lease = _ordinary_connections.get(conn)
        if (self.thread is not threading.current_thread() or self.connection(self.database) is not conn
            or lease is None or self.participant.connections.get(conn) is not lease
            or lease not in storage._live_leases or lease._key != self.admission._key
            or lease.resource_policy is not _validated_owner_policy(self.private_policy) or lease.resource_path != self.path
            or lease.resource_participant is not self.participant or self.participant.repository() is not self.database
            or lease.resource_thread is not self.thread or lease.resource_close_failed
            or self.prior is not None and (self.prior is not conn or self.prior_lease is not lease)
            or sqlite3.Connection.in_transaction.__get__(conn)):
            raise RecoveryRequired("profile_writer_grant_connection_changed")
        return lease

    def _current(self):
        from tldw_chatbook.Backup_Recovery import bootstrap
        from .profile_writer_repository import _artifacts
        from .profile_writer_selection import require_writer_repository

        require_writer_repository(self.profile, self.owner_id, facts=self.facts)
        if (type(self).profile(self.database) != self.profile or self.database.db_path != self.path
            or self.database.db_path_str != str(self.path) or not self.source_identity(self.database, self.identity)
            or self.admission.execution_context(self.path) != self.scope or bootstrap._admission_epoch != self.epoch):
            raise RecoveryRequired("profile_writer_grant_source_changed")
        observed = _artifacts(self.path)
        if observed[0] is None or observed[0][:2] != self.identity or self.artifacts is not None and any(
            before is not None and after is not None and before != after
            for before, after in zip(self.artifacts, observed, strict=True)
        ):
            raise RecoveryRequired("profile_writer_database_changed")
        return observed

    def bind(self, connection):
        """Witness the exact original handle and synchronous policy before BEGIN."""
        if self.conn is not None:
            raise RecoveryRequired("profile_writer_grant_already_bound")
        self.artifacts = self._current()
        self.resource = self._connection(connection)
        self.schema = type(self).schema(self.database, connection)
        with closing(connection.execute("PRAGMA main.synchronous")) as cursor:
            self.synchronous = cursor.fetchone()[0]
        self.conn = connection

    def uncertain(self):
        """Retain process custody after an unproved commit, rollback or restore."""
        self.ambiguous = True

    def _owner_closes_later(self):
        """Whether an enclosing owned call, not this grant, must close the fresh handle.

        ``core_closing`` refuses to close a handle while a counted operation on
        its thread uses the repository. ``run_owned_db_call`` holds one around
        every callback. The grant's operation scopes equal ``original_scopes``
        at retirement, so a non-empty set means the grant's close would be
        refused. The outermost ``operation_owned_connection`` closes the handle
        opened during its operation when that operation ends.
        """
        from . import base_db

        return bool(self.original_scopes) and id(self.database) in getattr(
            base_db._OPERATION_OWNERS, "ids", ()
        )

    def retire(self, connection):
        """Verify the original transaction and all introduced resources retired.

        A fresh handle is closed here unless an enclosing owned call keeps it
        (see ``_owner_closes_later``). That idle handle then stays with its owner,
        like a borrowed one, and ``owner_retains`` is set. The transaction,
        lease, schema and synchronous policy are verified either way before the
        file is released.
        """
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        if (self.ambiguous or self.retired or self.conn is None or self.conn is not connection
            or self.thread is not threading.current_thread()):
            raise writer.ProfileWriterOutcomeUnknown("unconfirmed local transaction")
        self._current()
        with storage._lock:
            if self._scopes(storage) != self.original_scopes:
                raise writer.ProfileWriterOutcomeUnknown("local transaction scope still active")
        owned = False
        try:
            active = sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            if self.prior is not None:
                raise writer.ProfileWriterOutcomeUnknown("borrowed connection was closed") from None
        else:
            if active or self._connection(connection) is not self.resource or type(self).schema(self.database, connection) != self.schema:
                raise writer.ProfileWriterOutcomeUnknown("local transaction did not retire")
            with closing(connection.execute("PRAGMA main.synchronous")) as cursor:
                if cursor.fetchone()[0] != self.synchronous:
                    raise writer.ProfileWriterOutcomeUnknown("SQLite policy did not retire")
            if self.prior is None:
                if self._owner_closes_later():
                    owned = True
                else:
                    self.close(self.database)
        if owned:
            # _connection() above proved the handle is still this thread's cache.
            with storage._lock:
                if (self.resource not in storage._live_leases
                        or self.participant.connections.get(connection) is not self.resource):
                    raise writer.ProfileWriterOutcomeUnknown("owned grant lease changed")
        elif self.prior is None:
            try:
                sqlite3.Connection.in_transaction.__get__(connection)
            except sqlite3.ProgrammingError:
                pass
            else:
                raise writer.ProfileWriterOutcomeUnknown("local grant connection still live")
            with storage._lock:
                if self.resource in storage._live_leases or connection in self.participant.connections:
                    raise writer.ProfileWriterOutcomeUnknown("local grant lease still live")
        expected = self.original_connections | ({self.resource} if owned else set())
        with storage._lock:
            if self._connections(storage) != expected:
                raise writer.ProfileWriterOutcomeUnknown("local grant connection leases did not retire")
        self.owner_retains = owned
        self.retired = True


def local_agent_transaction(database, kind):
    """Issue a grant only at the two installed caller-bound FULL transaction sites."""
    if os.name != "nt" or getattr(database, "_writer_profile", None) is None:
        return nullcontext(_LocalOnly())
    frame = sys._getframe(1)
    caller = frame.f_locals.get("self")
    if kind == "goal.checkpoint":
        from . import goal_runs as module

        cls = module.GoalRunsStore
        source, code = module._PROFILE_CHECKPOINT_SOURCE
        allowed = (type(caller) is cls and caller.db is database and "checkpoint" not in vars(caller)
                   and vars(cls).get("checkpoint") is source)
    else:
        from tldw_chatbook.Agents import workflow_phase as module

        cls = module.WorkflowPhaseAuthorization
        wrapper, source, code = module._PROFILE_FENCE_SOURCE
        allowed = (type(caller) is cls and caller._ledger._db is database and "_fence" not in vars(caller)
                   and vars(cls).get("_fence") is wrapper and wrapper.__wrapped__ is source
                   and (kind, frame.f_locals.get("operation")) in (
                       ("workflow.phase_control", "control"), ("workflow.child_commit", "child_commit"),
                   ))
    if (not allowed or source.__code__ is not code or frame.f_code is not code
        or frame.f_globals is not source.__globals__ or source.__globals__ is not vars(module)):
        raise RecoveryRequired("profile_writer_grant_source_changed")
    return _local_agent_transaction(database, kind)


def _local_agent_transaction(database, kind):
    from .AgentRuns_DB import AgentRunsDB

    if type(database) is not AgentRunsDB or type(kind) is not str or kind not in _KINDS:
        raise RecoveryRequired("profile_writer_grant_source_changed")
    return _local_transaction(database, kind, _Grant)


def _local_chacha_transaction(database, kind):
    from .ChaChaNotes_DB import CharactersRAGDB
    from .profile_writer_chacha_grant import ChaChaTransactionGrant

    if type(database) is not CharactersRAGDB or type(kind) is not str or kind not in CHACHA_LOCAL_GRANT_KINDS:
        raise RecoveryRequired("profile_writer_grant_source_changed")
    return _local_transaction(database, kind, ChaChaTransactionGrant)


@contextmanager
def _local_transaction(database, kind, grant_type):
    import _winapi
    from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
    from .profile_writer_selection import require_writer_repository, writer_file_identity

    owner, leaf = _grant_target(kind)
    path, profile = database.db_path, grant_type.profile(database)
    if (getattr(_local, "grant", None) is not None or profile is None or path != profile / leaf
        or database.db_path_str != str(path) or owner != grant_type.owner_id):
        raise RecoveryRequired("profile_writer_grant_source_changed")
    participants.core_access(database)
    token = uuid4().hex
    resources = ExitStack()
    try:
        # Hold this process's actual external recovery group from before the
        # offer through positive original transaction/handle retirement.
        admission = resources.enter_context(storage.acquire_storage(path))
        facts = require_writer_repository(profile, owner)
        identity = writer_file_identity(path)
        if identity is None or not grant_type.source_identity(database, identity):
            raise RecoveryRequired("profile_writer_database_changed")
        grant = grant_type(database, identity, facts, admission)
        body = writer._encode({"v": writer._VERSION, "id": token, "intent": "grant", "profile": facts,
                               "data": {"kind": kind, "database": list(identity)}})
        with ExitStack() as connection:
            handle = writer._connect(connection, profile, writer._named_address(profile, *facts[:2]), missing_ok=False)
            writer.pipe.write_frame(handle, body, writer._IO_SECONDS)
            try:
                _expect(handle, token, "offer", waiting=True)
            except RecoveryRequired:
                _send(handle, token, "received")
                raise
            guard = ExitStack()
            resources.callback(guard.close)
            guard.enter_context(reserve_writer_file(identity, observable=True))
            completed = _completion(identity, create=True)
            resources.callback(_winapi.CloseHandle, completed)
            _send(handle, token, "claimed")
            try:
                _expect(handle, token, "granted")
            except BaseException:
                _signal(completed)  # The application never began its transaction.
                raise
            try:
                _local.grant = grant
                try:
                    yield grant
                finally:
                    del _local.grant
            finally:
                if grant.ambiguous or not grant.retired:
                    # A pre-BEGIN refusal may release only a never-bound grant
                    # with no new native resource and its original scopes intact.
                    with storage._lock:
                        safe = (grant.conn is None and not grant.ambiguous
                                and grant.prior is grant.connection(database)
                                and grant.original_scopes == grant._scopes(storage)
                                and grant.original_connections == grant._connections(storage))
                    if not safe:
                        _uncertain[token] = resources.pop_all(), grant
                        raise writer.ProfileWriterOutcomeUnknown("local transaction retirement uncertain")
                try:
                    guard.close()
                    _signal(completed)
                except BaseException as error:
                    _uncertain[token] = resources.pop_all(), grant
                    raise writer.ProfileWriterOutcomeUnknown("local file guard retirement uncertain") from error
                try:
                    _send(handle, token, "retired")
                    _expect(handle, token, "complete")
                    _send(handle, token, "received")
                except (RecoveryRequired, ValueError, OSError, EOFError) as error:
                    raise writer.ProfileWriterOutcomeUnknown("local transaction receipt unavailable") from error
    finally:
        resources.close()
