"""One elected Windows writer for cooperating clients of a private profile."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from .store_errors import StoreMutationOutcomeUnknown

if os.name == "nt":
    import _winapi

    from . import _profile_writer_windows as pipe

_CONNECTION_SECONDS = 30
_IO_SECONDS = 10
_IDLE_SECONDS = 5
_VERSION = 2
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_OPERATION = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")
_execution = threading.local()
_enrollment_generation = 0
_host_elects = False


class ProfileWriterOutcomeUnknown(StoreMutationOutcomeUnknown):
    """The mutation may have committed. The caller must observe original state."""


class _StaleWriterResource(RuntimeError):
    """Refuse the unstarted command and retire a no-longer-qualified writer."""


def in_profile_writer() -> bool:
    """Return whether this thread is executing the elected writer's domain handler."""
    return getattr(_execution, "active", False)


def executing_writer_repository(profile: Path, owner: str) -> bool:
    """Whether the elected command has checked this exact installed store."""
    return in_profile_writer() and getattr(_execution, "repository", None) == (profile, owner)


def current_profile_writer_selection() -> tuple:
    """Return the current command's verified directory and configuration generation."""
    return _execution.selection


def use_host_profile_writer() -> None:
    """Make a contained interpreter depend on its application host for election."""
    global _host_elects
    _host_elects = True


def _selection(profile: Path) -> tuple[str, list]:
    """Identify the private, pinned directory and the current recovery selector."""
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
    from tldw_chatbook.Utils.platform_files import os as files
    from tldw_chatbook.Utils.windows_files import _native

    if not profile.is_absolute() or ".." in profile.parts:
        raise ValueError("profile writer needs an absolute profile directory")
    with pinned_directory(profile) as directory:
        info = files.fstat(directory)
        if info.st_uid != files.geteuid() or info.st_mode & 0o077:
            raise PermissionError("profile writer directory is not private")
        # The normalized volume path unifies drive-letter, case and short-name
        # aliases. Keep the name stable across replacement; bind the live file
        # identity independently in every command, under recovery admission.
        key = _native().user_sid + "\0" + pipe.final_directory_name(directory)
        address = r"\\.\pipe\tldw-profile-writer-" + hashlib.sha256(key.encode()).hexdigest()
        selector = effective_config_path()
        stamp = files.stat(selector, follow_symlinks=False)
        facts = [
            info.st_dev, info.st_ino, os.path.normcase(selector),
            os.path.normcase(bootstrap.default_bootstrap_root()),
            [stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns],
        ]
        return address, facts


def _require_selected(profile: Path) -> None:
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    if config.get_user_data_dir() != profile:
        raise RecoveryRequired("profile_writer_selection_changed")


@lru_cache(maxsize=32)
def _named_address(profile: Path, device: int, inode: int) -> str:
    """Cache the routing name only; admission always uses fresh repository facts."""
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    address, facts = _selection(profile)
    if facts[:2] != [device, inode]:
        raise RecoveryRequired("profile_writer_selection_changed")
    return address


def _repository(profile: Path, operation: str):
    from tldw_chatbook.Backup_Recovery.profile_paths import database_leaf

    if operation.startswith("agent_runs."):
        return profile / "agent_runs.db", "db.agent_runs"
    if operation.startswith("chacha."):
        return profile / database_leaf("chachanotes_db_path"), "db.chachanotes.primary"
    return profile, None


def _encode(value: dict) -> bytes:
    from tldw_chatbook.Utils.input_validation import (
        STRICT_JSON_MAX_DEPTH, STRICT_JSON_MAX_NODES, _strict_json_within_limits,
    )

    if not _strict_json_within_limits(
        value, max_depth=STRICT_JSON_MAX_DEPTH, max_nodes=STRICT_JSON_MAX_NODES
    ):
        raise ValueError("invalid profile writer data")
    output = io.BytesIO()
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for part in encoder.iterencode(value):
        for offset in range(0, len(part), 65536):
            chunk = part[offset:offset + 65536].encode()
            if output.tell() + len(chunk) > pipe.MAX_MESSAGE_BYTES:
                raise ValueError("profile writer message length out of bounds")
            output.write(chunk)
    return output.getvalue()


def _decode(body: bytes) -> dict:
    from tldw_chatbook.Utils.input_validation import strict_json_loads

    value = strict_json_loads(body.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("invalid profile writer frame")
    return value


def _launch(profile: Path) -> subprocess.Popen:
    """Launch the installed writer code, excluding cwd and PYTHONPATH injection."""
    root = Path(__file__).resolve().parents[2]
    roots = [root]
    core = root / "packages" / "tldw_profile_core" / "src"
    if (core / "tldw_profile_core" / "__init__.py").is_file():
        roots.append(core)
    installed = [Path(p).resolve() for p in sys.path if p and Path(p).name == "site-packages"]
    for entry in sys.path:
        if entry:
            path = Path(entry).resolve()
            if any(path.is_relative_to(site) for site in installed):
                roots.append(path)
    code = (
        "import sys;sys.path[:0]=" + repr([str(path) for path in roots])
        + ";from tldw_chatbook.DB.profile_writer import _serve;"
        "from pathlib import Path;raise SystemExit(_serve(Path(sys.argv[1])))"
    )
    executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    return subprocess.Popen(
        [str(executable), "-I", "-S", "-B", "-c", code, str(profile)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True,
    )


def _connect(stack, profile, address, *, missing_ok):
    """Retry election/connect only. Application bytes have not yet been issued."""
    deadline = time.monotonic() + _CONNECTION_SECONDS
    candidate = None
    while time.monotonic() < deadline:
        try:
            return stack.enter_context(pipe.client(address))
        except FileNotFoundError:
            if missing_ok:
                return None
            if candidate is not None:
                status = candidate.poll()
                if status == 0:
                    candidate = None  # A losing candidate already retired.
                elif status is not None:
                    raise RuntimeError("profile writer failed to start") from None
            if candidate is None and not _host_elects:
                candidate = _launch(profile)
            time.sleep(.025)
        except OSError as error:
            if error.winerror != _winapi.ERROR_PIPE_BUSY:
                raise
            try:
                pipe.wait_available(address, min(1, max(0, deadline - time.monotonic())))
            except OSError as wait_error:
                if wait_error.winerror not in (2, 121):  # vanished / wait timed out
                    raise
    raise TimeoutError("profile writer unavailable before submission")


def _exchange(profile, intent, argument, *, missing_ok=False, pin=False, target=None):
    address, facts = target if target is not None else _selection(profile)
    token = uuid4().hex
    body = _encode({"v": _VERSION, "id": token, "profile": facts, "intent": intent, "data": argument})
    with ExitStack() as stack:
        handle = _connect(stack, profile, address, missing_ok=missing_ok)
        if handle is None:
            return None
        process = pipe.server_process(handle) if pin else None
        try:
            # Once WriteFile is attempted, no path back to _connect exists.
            pipe.write_frame(handle, body, _IO_SECONDS)
            reply = _decode(pipe.read_frame(handle, None if intent == "call" else _IO_SECONDS))
            if (
                set(reply) != {"v", "id", "status", "data"}
                or type(reply["v"]) is not int or reply["v"] != _VERSION or reply["id"] != token
                or reply["status"] not in ("ok", "error", "recovery", "unknown")
            ):
                raise ValueError("invalid writer receipt")
            # DisconnectNamedPipe discards unread outbound bytes. The server
            # waits for this acknowledgment or EOF before reusing its handle.
            try:
                pipe.write_frame(handle, b"received", _IO_SECONDS)
            except OSError:
                pass  # A complete, validated receipt was already received.
        except BaseException as error:
            if process is not None:
                _winapi.CloseHandle(process)
            if intent == "call" and isinstance(error, Exception):
                raise ProfileWriterOutcomeUnknown("profile writer receipt unavailable") from error
            raise
    status, data = reply["status"], reply["data"]
    if status == "ok":
        return (data, process) if pin else data
    if process is not None:
        _winapi.CloseHandle(process)
    if status == "recovery":
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        raise RecoveryRequired(data)
    if status == "error":
        from .profile_writer_commands import decode_profile_writer_error

        raise decode_profile_writer_error(data)
    raise ProfileWriterOutcomeUnknown("profile writer could not prove retirement")


def call_profile_writer(profile: Path, operation: str, arguments: dict):
    """Execute one whitelisted domain mutation and return its committed result.

    Args:
        profile: The current private user data directory.
        operation: Installed dotted domain operation name.
        arguments: Data-only JSON arguments for this operation.

    Raises:
        ProfileWriterOutcomeUnknown: Delivery/commit or physical retirement is uncertain.
    """
    if operation == "agent_runs.open":
        return open_profile_agent_runs(profile, arguments)[0]
    return call_bound_profile_writer(None, profile, operation, arguments)


def open_profile_agent_runs(profile: Path, arguments: dict) -> tuple[bool, tuple[int, int]]:
    """Bind the opener to the file the writer actually checked and initialized."""
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from .profile_writer_selection import writer_file_identity

    receipt = call_bound_profile_writer(None, profile, "agent_runs.open", arguments)
    if (type(receipt) is not dict or set(receipt) != {"capability", "database"}
        or type(receipt["capability"]) is not bool or type(receipt["database"]) is not list
        or len(receipt["database"]) != 2 or any(type(n) is not int for n in receipt["database"])):
        raise ProfileWriterOutcomeUnknown("invalid AgentRuns opening receipt")
    identity = tuple(receipt["database"])
    if identity != writer_file_identity(Path(profile) / "agent_runs.db"):
        raise RecoveryRequired("profile_writer_database_changed")
    return receipt["capability"], identity


def call_bound_profile_writer(database, profile: Path, operation: str, arguments: dict):
    """Send a stock operation against the same physical file its opener enrolled."""
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import repository_participant
    from .profile_writer_selection import require_writer_repository, writer_file_identity

    if os.name != "nt" or in_profile_writer():
        raise RuntimeError("a Windows profile writer client is required")
    if type(operation) is not str or not _OPERATION.fullmatch(operation) or type(arguments) is not dict:
        raise ValueError("invalid profile writer operation")
    profile = Path(profile)
    target, participant = _repository(profile, operation)
    core = getattr(storage._operation_local, "operation", None)
    proof = None
    if core is not None:
        # The producer retained its original file operation. It confers no
        # authority over a sibling repository or the enclosing directory.
        proof = storage._check_operation(core, target)
        if core.participant.owner_id != participant or core.participant.read_only:
            raise RecoveryRequired("profile_writer_repository_scope_changed")
    elif storage.local_pause_active():
        raise RecoveryRequired("storage_locally_paused")
    if participant is not None:
        # This caller has no SQLite work. Validate the selected DB and any
        # retained producer, then let the writer acquire its own native custody.
        facts = require_writer_repository(profile, participant)
        endpoint = _named_address(profile, *facts[:2]), facts
        identity = writer_file_identity(target)
    else:
        endpoint = _selection(profile)
        with storage.acquire_storage(target):
            _require_selected(profile)
        identity = None
    if database is not None:
        installed = repository_participant(database)
        if participant == "db.chachanotes.primary":
            expected = database._local.profile_writer_file_identity
        else:
            expected = database._writer_file
        if (installed.path != target or installed.owner_id != participant or installed.read_only
            or (core is not None and core.participant is not installed)
            or identity is None or expected != identity):
            raise RecoveryRequired("profile_writer_database_changed")
    if proof is not None:
        with storage._lock:
            proof.require_current()
    return _exchange(profile, "call", {
        "operation": operation, "arguments": arguments, "database": list(identity) if identity else None,
    }, target=endpoint)


def _require_retired(startup, idle=None) -> None:
    """Only the exact startup and independently owned idle SQLite lease may survive."""
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    with storage._lock:
        startups = {startup} if startup is not None else set()
        leases = startups | ({idle} if idle is not None else set())
        keys = {lease._key for lease in leases if lease._key is not None}
        if (
            (idle is not None and (startup is None or idle._key is None or idle._key != startup._key))
            or set(storage._startups.values()) != startups or storage._live_leases != leases
            or storage._holds.keys() != keys or storage._pending_acquisitions or storage._operations
            or storage._raw_operations or raw._states or storage._retiring_holds
            or any(hold.count != sum(lease._key == hold.key for lease in leases)
                   or not hold.ready.is_set() or hold.stop.is_set()
                   or hold.error is not None or not hold.thread.is_alive() for hold in storage._holds.values())
        ):
            raise ProfileWriterOutcomeUnknown("profile writer resources did not retire")


class _WriterEnrollment:
    """A serial command and the idle maintenance observer share one startup lease."""

    def __init__(self):
        from .profile_writer_repository import AgentRunsWriterDatabase

        self.startup = None
        self.agent_runs = AgentRunsWriterDatabase()
        self.created = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="profile-writer-storage")
        self.stop = threading.Event()
        self.watcher = threading.Thread(target=self._monitor, name="profile-writer-maintenance", daemon=True)
        self.watcher.start()

    def _release(self):
        global _enrollment_generation
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        _require_retired(self.startup, self.agent_runs.idle_lease())
        self.agent_runs.retire()
        _require_retired(self.startup)
        if self.startup is not None:
            storage._shutdown()
            self.startup = None
            _enrollment_generation += 1
            self.created.clear()
        _require_retired(None)

    def _yield_if_requested(self):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        if self.startup is not None and storage._local_pause_requested():
            self._release()
            return True
        return False

    def _monitor(self):
        from tldw_chatbook.Backup_Recovery.runtime_maintenance import MAINTENANCE_PROBE_INTERVAL_SECONDS

        while not self.stop.wait(MAINTENANCE_PROBE_INTERVAL_SECONDS):
            try:
                self.executor.submit(self._yield_if_requested).result()
            except BaseException:
                # Retain native leases and the election until process exit.
                os._exit(1)

    @contextmanager
    def command(self, profile):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        if self._yield_if_requested():
            raise RecoveryRequired("profile_writer_maintenance_pause")
        try:
            if self.startup is None:
                # Acquire the current native group, refusing a closed gate
                # before startup can wait behind its own caller's native work.
                self.startup = storage.admit_profile_writer_startup()
            yield
        finally:
            _require_retired(self.startup, self.agent_runs.idle_lease())

    def drain(self):
        self.executor.submit(self._release).result()

    def close(self):
        self.stop.set()
        self.watcher.join()
        try:
            self.drain()
            self.executor.shutdown()
        except BaseException:
            os._exit(1)


def _command(profile, facts, payload, previous_stamp, enrollment):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from .profile_writer_commands import encode_profile_writer_error, execute_profile_writer_command
    from .profile_writer_selection import require_writer_repository, writer_file_identity

    if (
        type(payload) is not dict or set(payload) != {"operation", "arguments", "database"}
        or type(payload["operation"]) is not str or not _OPERATION.fullmatch(payload["operation"])
        or type(payload["arguments"]) is not dict
    ):
        return "error", encode_profile_writer_error(ValueError("invalid writer command"))
    try:
        with enrollment.command(profile), ExitStack() as order:
            from tldw_chatbook import config

            operation = payload["operation"]
            path, owner = _repository(profile, operation)
            if owner is not None:
                require_writer_repository(profile, owner, facts=facts)
                current = writer_file_identity(path)
                expected = payload["database"]
                if expected is None:
                    if operation not in ("agent_runs.open", "chacha.initialize") or (
                        current is not None and enrollment.created.get(owner) != current
                    ):
                        raise RecoveryRequired("profile_writer_database_changed")
                elif (type(expected) is not list or len(expected) != 2
                      or any(type(item) is not int for item in expected) or tuple(expected) != current):
                    raise RecoveryRequired("profile_writer_database_changed")
                if owner == "db.agent_runs":
                    from .profile_writer_grant import reserve_agent_file

                    order.enter_context(reserve_agent_file(current))
            else:
                if _selection(profile)[1] != facts:
                    raise RecoveryRequired("profile_writer_selection_changed")
                if payload["database"] is not None:
                    raise RecoveryRequired("profile_writer_database_changed")
                if previous_stamp != facts[-1] and not config._load_cli_config_bootstrap(force_reload=True).succeeded:
                    raise RecoveryRequired("profile_writer_configuration_unavailable")
                _require_selected(profile)
            _execution.active = True
            _execution.repository = (profile, owner)
            _execution.agent_runs = enrollment.agent_runs
            _execution.selection = (*facts[:-1], tuple(facts[-1]), _enrollment_generation)
            try:
                result = execute_profile_writer_command(profile, payload["operation"], payload["arguments"])
                status = "ok"
            except (ProfileWriterOutcomeUnknown, RecoveryRequired, _StaleWriterResource):
                raise
            except Exception as error:
                status, result = "error", encode_profile_writer_error(error)
            finally:
                del _execution.repository
                del _execution.agent_runs
                del _execution.selection
                del _execution.active
            if _selection(profile)[1] != facts:
                raise ProfileWriterOutcomeUnknown("profile changed during mutation")
            if owner is not None:
                after = writer_file_identity(path)
                if current is None and after is not None:
                    enrollment.created[owner] = after
                elif current != after:
                    raise ProfileWriterOutcomeUnknown("database changed during mutation")
                if operation == "agent_runs.open" and status == "ok":
                    if after is None:
                        raise ProfileWriterOutcomeUnknown("AgentRuns opening identity unavailable")
                    result = {"capability": result, "database": list(after)}
            return status, result
    except RecoveryRequired as error:
        return "recovery", str(error)
    except SystemExit as error:
        # config imports run admit_startup(), whose public startup refusal is
        # SystemExit("Recovery required: <reason>") before runtime code loads.
        prefix = "Recovery required: "
        if not isinstance(error.code, str) or not error.code.startswith(prefix):
            raise
        return "recovery", error.code[len(prefix):]


def _process_running(process: int) -> bool:
    return _winapi.WaitForSingleObject(process, 0) == _winapi.WAIT_TIMEOUT


def _prune_hosts(hosts: dict) -> None:
    for ticket, (_, process) in list(hosts.items()):
        if not _process_running(process):
            _winapi.CloseHandle(process)
            del hosts[ticket]


def _wait_for_drain(receipt) -> None:
    if receipt is None:
        return
    last_host, process = receipt
    try:
        if last_host and _winapi.WaitForSingleObject(process, _IO_SECONDS * 1000) != _winapi.WAIT_OBJECT_0:
            raise TimeoutError("profile writer retirement unconfirmed")
    finally:
        _winapi.CloseHandle(process)


def drain_profile_writer(profile: Path) -> None:
    """Wait for previously accepted work and retire an owner with no live UI hosts."""
    _wait_for_drain(_exchange(Path(profile), "drain", None, missing_ok=True, pin=True))


class ProfileWriterHost:
    """Keep the writer outside a contained runtime and replace a dead writer."""

    def __init__(self, profile: Path):
        self._profile = profile
        self._initial_target = _selection(profile)
        self._ticket = uuid4().hex
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._retired = False
        _, process = _exchange(profile, "join", self._ticket, pin=True, target=self._initial_target)
        self._watcher = threading.Thread(
            target=self._watch, args=(process,), name="profile-writer-host", daemon=True,
        )
        self._watcher.start()

    def _watch(self, process: int) -> None:
        try:
            while not self._closed.wait(.1):
                if process is not None and _process_running(process):
                    continue
                if process is not None:
                    _winapi.CloseHandle(process)
                    process = None
                try:
                    target = _selection(self._profile)
                    if target[0] != self._initial_target[0]:
                        raise ValueError("writer host profile moved")
                    _require_selected(self._profile)
                    _, process = _exchange(self._profile, "join", self._ticket, pin=True, target=target)
                except (OSError, RuntimeError, ValueError):
                    # Retirement/restore can temporarily deny the profile. Only
                    # the idempotent host registration is retried after refusal.
                    self._closed.wait(.5)
        finally:
            if process is not None:
                _winapi.CloseHandle(process)

    def close(self) -> None:
        """Retire this host. The last host observes the actual writer process exit."""
        with self._close_lock:
            if self._retired:
                return
            self._closed.set()
            self._watcher.join()
            _wait_for_drain(_exchange(
                self._profile, "leave", self._ticket, missing_ok=True, pin=True, target=self._initial_target,
            ))
            self._retired = True


def start_profile_writer(profile: Path) -> ProfileWriterHost:
    """Register this normal application process as a lifetime and re-election owner."""
    if os.name != "nt" or _host_elects or in_profile_writer():
        raise RuntimeError("a Windows profile writer host is required")
    profile = Path(profile)
    _require_selected(profile)
    return ProfileWriterHost(profile)


def _serve(profile: Path) -> int:
    """Hold one kernel election across intake, dispatch, retirement and reply."""
    address, original = _selection(profile)
    hosts = {}
    previous_stamp = None
    with ExitStack() as owner:
        try:
            handle = owner.enter_context(pipe.server(address))
        except OSError as error:
            if error.winerror in (5, 231):  # Another process already owns the one instance.
                return 0
            raise
        enrollment = _WriterEnrollment()
        owner.callback(enrollment.close)
        while True:
            try:
                pipe.accept(handle, _IDLE_SECONDS)
            except TimeoutError:
                _prune_hosts(hosts)
                if not hosts:
                    return 0
                continue
            except OSError as error:
                if error.winerror != _winapi.ERROR_NO_DATA:
                    raise
                pipe.disconnect(handle)
                continue
            stop = fatal = False
            try:
                request = _decode(pipe.read_frame(handle, _IO_SECONDS))
                if (
                    set(request) != {"v", "id", "profile", "intent", "data"}
                    or type(request["v"]) is not int or request["v"] != _VERSION
                    or type(request["id"]) is not str or not _REQUEST_ID.fullmatch(request["id"])
                ):
                    raise ValueError("invalid writer request")
                facts, intent, payload = request["profile"], request["intent"], request["data"]
                status, data = "ok", None
                if intent == "leave" and type(payload) is str and _REQUEST_ID.fullmatch(payload):
                    # Unregistration is bound to this endpoint, ticket and actual
                    # process. It must work even if recovery moved the directory.
                    _prune_hosts(hosts)
                    if payload in hosts:
                        if hosts[payload][0] != pipe.client_pid(handle):
                            raise ValueError("foreign writer host registration")
                        _winapi.CloseHandle(hosts.pop(payload)[1])
                        stop = not hosts
                    data = stop
                elif intent == "grant" and type(facts) is list and facts[:4] == original[:4]:
                    from .profile_writer_grant import serve_grant

                    try:
                        serve_grant(profile, facts, payload, request["id"], handle, enrollment)
                    except ProfileWriterOutcomeUnknown:
                        # An uncertain physical close must keep the election
                        # and all native custody until the process dies.
                        os._exit(1)
                    continue
                elif intent == "call" and type(facts) is list and facts[:4] == original[:4]:
                    try:
                        status, data = enrollment.executor.submit(
                            _command, profile, facts, payload, previous_stamp, enrollment,
                        ).result()
                        if status == "ok":
                            previous_stamp = facts[-1]
                    except _StaleWriterResource as error:
                        status, data, fatal = "recovery", str(error), True
                    except BaseException:
                        # Retain the kernel election until the entire process
                        # exits; no successor may execute alongside an uncertain owner.
                        status, data, fatal = "unknown", None, True
                elif facts != _selection(profile)[1]:
                    status, data = "recovery", "profile_writer_selection_changed"
                elif facts[:4] != original[:4]:
                    status, data, stop = "recovery", "profile_writer_selection_changed", True
                elif intent == "join" and type(payload) is str and _REQUEST_ID.fullmatch(payload):
                    _prune_hosts(hosts)
                    pid = pipe.client_pid(handle)
                    if payload not in hosts:
                        hosts[payload] = pid, _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid)
                    elif hosts[payload][0] != pid:
                        raise ValueError("foreign writer host registration")
                    data = False
                elif intent == "drain" and payload is None:
                    _prune_hosts(hosts)
                    enrollment.drain()
                    data = stop = not hosts
                else:
                    raise ValueError("invalid writer intent")
                reply = {"v": _VERSION, "id": request["id"], "status": status, "data": data}
                try:
                    body = _encode(reply)
                except (ValueError, TypeError, OverflowError):
                    # A result which cannot cross the boundary may have committed.
                    reply.update(status="unknown", data=None)
                    body = _encode(reply)
                pipe.write_frame(handle, body, _IO_SECONDS)
                if pipe.read_frame(handle, _IO_SECONDS, maximum=len(b"received")) != b"received":
                    raise ValueError("invalid writer receipt acknowledgment")
            except (ValueError, EOFError, OSError):
                pass  # Dropped, late or malformed peer; never replay its command.
            finally:
                if fatal:
                    os._exit(1)
                pipe.disconnect(handle)
            if stop:
                return 0
