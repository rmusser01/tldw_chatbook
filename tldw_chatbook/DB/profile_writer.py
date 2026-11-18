"""One elected Windows writer for cooperating clients of a private profile."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from .store_errors import StoreMutationOutcomeUnknown

if os.name == "nt":
    import _winapi

    from . import _profile_writer_windows as pipe

_CONNECTION_SECONDS = 30
_IO_SECONDS = 10
_IDLE_SECONDS = 5
_VERSION = 1
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_OPERATION = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")
_execution = threading.local()
_host_elects = False


class ProfileWriterOutcomeUnknown(StoreMutationOutcomeUnknown):
    """The mutation may have committed. The caller must observe original state."""


def in_profile_writer() -> bool:
    """Return whether this thread is executing the elected writer's domain handler."""
    return getattr(_execution, "active", False)


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


def _encode(value: dict) -> bytes:
    from tldw_chatbook.Utils.input_validation import (
        STRICT_JSON_MAX_DEPTH, STRICT_JSON_MAX_NODES, _strict_json_within_limits,
    )

    if not _strict_json_within_limits(
        value, max_depth=STRICT_JSON_MAX_DEPTH, max_nodes=STRICT_JSON_MAX_NODES
    ):
        raise ValueError("invalid profile writer data")
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > pipe.MAX_FRAME_BYTES:
        raise ValueError("profile writer frame length out of bounds")
    return body


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
            if candidate is None and not _host_elects:
                candidate = _launch(profile)
            if candidate is not None and candidate.poll() not in (None, 0):
                raise RuntimeError("profile writer failed to start") from None
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
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    if os.name != "nt" or in_profile_writer():
        raise RuntimeError("a Windows profile writer client is required")
    if type(operation) is not str or not _OPERATION.fullmatch(operation) or type(arguments) is not dict:
        raise ValueError("invalid profile writer operation")
    profile = Path(profile)
    with acquire_storage(profile):
        _require_selected(profile)
        return _exchange(profile, "call", {"operation": operation, "arguments": arguments})


def _retire_command() -> None:
    """Release this isolated command's startup enrollment, then verify physical drain."""
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    storage._shutdown()
    with storage._lock:
        if any((
            storage._startups, storage._pending_acquisitions, storage._operations,
            storage._raw_operations, raw._states, storage._live_leases, storage._holds,
            storage._retiring_holds,
        )):
            raise ProfileWriterOutcomeUnknown("profile writer resources did not retire")


def _command(profile, facts, payload, previous_stamp):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    if (
        type(payload) is not dict or set(payload) != {"operation", "arguments"}
        or type(payload["operation"]) is not str or not _OPERATION.fullmatch(payload["operation"])
        or type(payload["arguments"]) is not dict
    ):
        raise ValueError("invalid writer command")
    try:
        with acquire_storage(profile):
            from tldw_chatbook import config

            if _selection(profile)[1] != facts:
                raise RecoveryRequired("profile_writer_selection_changed")
            # The normal CLI cache throttles external file stat calls. A changed
            # native generation must refresh before selecting a write destination.
            if previous_stamp != facts[-1] and not config._load_cli_config_bootstrap(force_reload=True).succeeded:
                raise RecoveryRequired("profile_writer_configuration_unavailable")
            _require_selected(profile)
            if _selection(profile)[1] != facts:
                raise RecoveryRequired("profile_writer_selection_changed")
            from .profile_writer_commands import encode_profile_writer_error, execute_profile_writer_command

            _execution.active = True
            try:
                result = execute_profile_writer_command(profile, payload["operation"], payload["arguments"])
            except (ProfileWriterOutcomeUnknown, RecoveryRequired):
                raise
            except Exception as error:
                return "error", encode_profile_writer_error(error)
            finally:
                del _execution.active
            if _selection(profile)[1] != facts:
                raise ProfileWriterOutcomeUnknown("profile changed during mutation")
            return "ok", result
    except RecoveryRequired as error:
        return "recovery", str(error)
    except SystemExit as error:
        # config imports run admit_startup(), whose public startup refusal is
        # SystemExit("Recovery required: <reason>") before runtime code loads.
        prefix = "Recovery required: "
        if not isinstance(error.code, str) or not error.code.startswith(prefix):
            raise
        return "recovery", error.code[len(prefix):]
    finally:
        _retire_command()


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
                elif facts != _selection(profile)[1]:
                    status, data = "recovery", "profile_writer_selection_changed"
                elif facts[:4] != original[:4]:
                    status, data, stop = "recovery", "profile_writer_selection_changed", True
                elif intent == "call":
                    try:
                        status, data = _command(profile, facts, payload, previous_stamp)
                        if status == "ok":
                            previous_stamp = facts[-1]
                    except BaseException:
                        # Retain the kernel election until the entire process
                        # exits; no successor may execute alongside an uncertain owner.
                        status, data, fatal = "unknown", None, True
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
                pipe.read_frame(handle, _IO_SECONDS)
            except (ValueError, EOFError, OSError):
                pass  # Dropped, late or malformed peer; never replay its command.
            finally:
                if fatal:
                    os._exit(1)
                pipe.disconnect(handle)
            if stop:
                return 0
