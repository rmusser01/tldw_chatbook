"""Contained stock Windows permission persistence owner (ADR-126/230)."""

from __future__ import annotations

import io
import inspect
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
import threading
import time
import weakref
from contextlib import contextmanager

_CHUNK = 65536
_HEX = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ProtocolError(ValueError):
    """The selected permission request cannot be adopted or replayed."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate_field")
        result[key] = value
    return result


def _constant(value):
    raise ProtocolError("nonfinite_number")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError("nonfinite_number")
    return result


def write_message(stream, value):
    """Write strict JSON in bounded chunks without a payload-size ceiling."""
    encoder = json.JSONEncoder(
        ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    buffer = bytearray()
    for piece in encoder.iterencode(value):
        encoded = piece.encode("ascii")
        for offset in range(0, len(encoded), _CHUNK):
            buffer.extend(encoded[offset : offset + _CHUNK])
            while len(buffer) >= _CHUNK:
                stream.write(struct.pack("!I", _CHUNK))
                stream.write(buffer[:_CHUNK])
                del buffer[:_CHUNK]
    if buffer:
        stream.write(struct.pack("!I", len(buffer)))
        stream.write(buffer)
    stream.write(b"\0\0\0\0")
    stream.flush()


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        part = stream.read(count - len(result))
        if not part:
            raise ProtocolError("transport_lost")
        result.extend(part)
    return bytes(result)


def read_message(stream):
    """Read one chunked message, rejecting ambiguous and excessive nesting."""
    body = io.BytesIO()
    depth = 0
    quoted = escaped = False
    while True:
        size = struct.unpack("!I", _read_exact(stream, 4))[0]
        if size == 0:
            break
        if size > _CHUNK:
            raise ProtocolError("oversize_chunk")
        chunk = _read_exact(stream, size)
        for char in chunk:
            if escaped:
                escaped = False
            elif quoted and char == 92:
                escaped = True
            elif char == 34:
                quoted = not quoted
            elif not quoted:
                if char in (91, 123):
                    depth += 1
                    if depth > 128:
                        raise ProtocolError("excessive_nesting")
                elif char in (93, 125):
                    depth -= 1
        body.write(chunk)
    try:
        value = json.loads(
            body.getvalue(),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_finite_float,
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("invalid_json") from error
    if type(value) is not dict:
        raise ProtocolError("invalid_message")
    return value


class _Identity:
    def __init__(self, epoch, request):
        if not all(
            type(value) is str and _HEX.fullmatch(value) for value in (epoch, request)
        ):
            raise ProtocolError("invalid_identity")
        self.epoch, self.request, self.ordinal = epoch, request, 0

    def frame(self, kind, data):
        self.ordinal += 1
        return {
            "epoch": self.epoch,
            "request": self.request,
            "ordinal": self.ordinal,
            "type": kind,
            "data": data,
        }

    def accept(self, message, kinds):
        if (
            set(message) != {"epoch", "request", "ordinal", "type", "data"}
            or message["epoch"] != self.epoch
            or message["request"] != self.request
            or type(message["ordinal"]) is not int
            or message["ordinal"] != self.ordinal + 1
            or message["type"] not in kinds
            or type(message["data"]) is not dict
        ):
            raise ProtocolError("unexpected_frame")
        self.ordinal += 1
        return message["type"], message["data"]


class _PublicationMirror:
    def __init__(self, fence):
        self.fence = fence
        self.candidate = None
        self.touched = False

    def observe(self, digest):
        fence = self.fence
        if (
            type(digest) is not str
            or not _DIGEST.fullmatch(digest)
            or fence.pending
            or fence.uncertain
        ):
            raise ProtocolError("contradictory_observation")
        observed = bytes.fromhex(digest)
        if observed != fence.observed:
            fence.token = None
            fence.observed = observed
        self.candidate = observed
        self.touched = True

    def begin(self):
        self.candidate = None
        self.touched = True
        self.fence.token = self.fence.observed = None
        self.fence.pending = True

    def settle(self, state):
        fence = self.fence
        if (
            set(state) != {"digest", "pending", "uncertain"}
            or state["pending"] is not False
            or state["uncertain"] is not False
        ):
            raise ProtocolError("unsettled_publication")
        expected = None if fence.observed is None else fence.observed.hex()
        if self.touched and state["digest"] != expected or fence.uncertain:
            raise ProtocolError("publication_changed")
        if not self.touched:
            return
        fence.pending = False
        if self.candidate is not None and fence.token is None:
            fence.token = object()


_local = threading.local()
_unresolved_commands = []


def _active(source=None):
    command = getattr(_local, "command", None)
    if command is None:
        return None
    from ..Backup_Recovery import storage_admission as storage

    if (
        type(command) is not _Command
        or command.creator
        != (os.getpid(), threading.current_thread(), storage._task_identity())
        or source is not None
        and source is not command.source
    ):
        raise ProtocolError("foreign_command_creator")
    return command


def _enroll(kind, value):
    retirement = sys.modules.get("tldw_chatbook.Chat.console_execution_retirement")
    if retirement is not None:
        retirement.observe_enrollment(kind, value)
    local = sys.modules.get("tldw_chatbook.MCP.local_permission_operation")
    if local is not None:
        local.enroll(kind, value)
    command = _active()
    if command is None:
        return
    from ..Backup_Recovery import storage_admission as storage

    if kind == "acquisition":
        command.acquisitions.append(value)
    elif kind == "lease":
        command.leases.append((value, storage._holds.get(value._key)))
    elif kind == "raw":
        command.raw.append(value)
    elif kind == "descriptor":
        command.descriptors.append(value)
    elif kind == "file":
        command.files.append(value)
    else:
        raise ProtocolError("unknown_enrollment")


def _boundary(source, site):
    local = sys.modules.get("tldw_chatbook.MCP.local_permission_operation")
    if local is not None:
        local.selected(source)
    command = _active()
    if command is not None and command.source is source:
        command.exchange("Check", {"site": site})


class _Command:
    """Exact references survive registry removal until physical settlement."""

    def __init__(self, source, exchange):
        from ..Backup_Recovery import storage_admission as storage

        self.creator = (
            os.getpid(),
            threading.current_thread(),
            storage._task_identity(),
        )
        self.source, self.exchange = source, exchange
        self.acquisitions, self.leases, self.raw, self.descriptors = [], [], [], []
        self.completions = []
        self.files = []
        self.validated = False
        with storage._lock:
            self.startups = tuple(
                (lease, storage._holds.get(lease._key))
                for lease in storage._startups.values()
            )

    def __enter__(self):
        if _active() is not None:
            raise ProtocolError("nested_permission_command")
        _local.command = self
        return self

    def __exit__(self, *args):
        if _active() is not self:
            raise ProtocolError("command_context_changed")
        _local.command = None

    def retirement(self):
        from ..Backup_Recovery import raw_participants as raw
        from ..Backup_Recovery import storage_admission as storage

        with storage._lock:
            startup_leases = {lease for lease, _ in self.startups}
            startup_holds = {hold for _, hold in self.startups if hold is not None}
            valid = (
                set(storage._startups.values()) == startup_leases
                and storage._live_leases == startup_leases
                and not storage._pending_acquisitions
                and not storage._raw_operations
                and not raw._states
                and not storage._operations
                and not storage._retiring_holds
                and set(storage._holds.values()) == startup_holds
            )
            for acquisition in self.acquisitions:
                valid &= acquisition not in storage._pending_acquisitions
            valid &= all(
                handle.closed and native.closed for handle, native in self.files
            )
            for operation, state in self.raw:
                valid &= (
                    operation not in raw._states
                    and operation not in storage._raw_operations
                    and not state.active
                    and not state.uncertain
                    and not state.files
                    and not state.descriptors
                    and not state.pins
                    and state.companion_guard is None
                )
            for lease, hold in self.leases:
                valid &= lease not in storage._live_leases and lease._key is None
                if hold is not None and hold not in startup_holds:
                    valid &= (
                        not hold.thread.is_alive()
                        and hold not in storage._retiring_holds
                        and hold not in storage._holds.values()
                        and hold.count == 0
                        and (
                            hold.error is None
                            or isinstance(hold.error, storage.AdmissionCancelled)
                        )
                    )
            for hold in startup_holds:
                valid &= (
                    hold.error is None
                    and not hold.stop.is_set()
                    and hold.ready.is_set()
                    and hold.thread.is_alive()
                    and hold.count == sum(saved is hold for _, saved in self.startups)
                )
            if not valid:
                if self not in _unresolved_commands:
                    _unresolved_commands.append(self)
                if self.source is not None:
                    self.source._permission_publication.fail()
                    self.source._mcp_persistence_error = "mcp_persistence_incomplete"
                raise ProtocolError("native_retirement_unproven")
            self.validated = True
        return {
            "retired": True,
            "acquisitions": len(self.acquisitions),
            "raw": len(self.raw),
            "leases": len(self.leases),
            "descriptors": len(self.descriptors),
            "startup": len(self.startups),
        }

    def complete(self):
        from ..Backup_Recovery import mcp_source_participants as sources

        if not self.validated:
            raise ProtocolError("native_retirement_unproven")
        for state in self.completions:
            sources.complete(state)
        self.completions.clear()


def _fields(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ProtocolError("invalid_fields")


_TEXT = (str,)
_OPTIONAL_TEXT = (str, type(None))
_OPTIONAL_INT = (int, type(None))
_PROFILE = {"profile_id": _TEXT}
_CAS = {"expected_profile_digest": _OPTIONAL_TEXT, "expected_revision": _OPTIONAL_INT}
_TOOL = {"server_key": _TEXT, "tool_name": _TEXT}
_REPLACEMENT = {
    "profile_id": _TEXT,
    "expected_revision": (int,),
    "max_store_bytes": (int,),
    "expected_generation": _OPTIONAL_TEXT,
    "expected_profile_digest": _OPTIONAL_TEXT,
}
_SCHEMAS = {
    "load": {"publish": (bool,)},
    "get_kill_switch": {"publish": (bool,)},
    "read_snapshot_strict": {},
    "read_profile_inventory_snapshot": {},
    "catalog_fingerprint": {},
    "list_profiles": {},
    "list_suggestion_dismissals": {},
    "get_global_default": _PROFILE,
    "get_server_entry": {"server_key": _TEXT, **_PROFILE},
    "get_tool_entry": {**_TOOL, **_PROFILE},
    "has_current_tool_arg_rule": {
        **_TOOL,
        **_PROFILE,
        "args": (dict,),
        "definition_hash": _OPTIONAL_TEXT,
    },
    "list_tool_arg_rules": {**_TOOL, **_PROFILE},
    "save": {"payload": (dict,), "expected_generation": _OPTIONAL_TEXT},
    "set_kill_switch": {"value": (bool,)},
    "ensure_profile": _PROFILE,
    "set_global_default": {"state": _TEXT, **_PROFILE, **_CAS},
    "set_server_default": {
        "server_key": _TEXT,
        "state": _OPTIONAL_TEXT,
        **_PROFILE,
        **_CAS,
    },
    "set_tool_state": {
        **_TOOL,
        "state": _OPTIONAL_TEXT,
        "definition_hash": _OPTIONAL_TEXT,
        **_PROFILE,
        **_CAS,
        "expected_policy_chain_digest": _OPTIONAL_TEXT,
    },
    "add_tool_arg_rule": {
        **_TOOL,
        **_PROFILE,
        "args": (dict,),
        "definition_hash": _OPTIONAL_TEXT,
    },
    "remove_tool_arg_rule": {**_TOOL, **_PROFILE, "rule_id": _TEXT},
    "mark_config_changed": {**_TOOL, **_PROFILE, **_CAS},
    "dismiss_suggestion": _TOOL,
    "restore_suggestion": _TOOL,
    "clear_suggestion_dismissals": {},
    "install_profile_if_absent": {
        "profile_id": _TEXT,
        "imported_profile": (dict,),
        "expected_generation": _TEXT,
        "max_profiles": (int,),
        "max_store_bytes": (int,),
    },
    "update_imported_profile": {**_REPLACEMENT, "replacement_profile": (dict,)},
    "replace_profile_with_tombstone": {**_REPLACEMENT, "tombstone_profile": (dict,)},
}


def _command_arguments(operation, arguments):
    schema = _SCHEMAS.get(operation)
    if schema is None:
        raise ProtocolError("unknown_operation")
    _fields(arguments, schema)
    if any(type(arguments[name]) not in kinds for name, kinds in schema.items()):
        raise ProtocolError("unsupported_argument")
    if operation == "get_kill_switch" and not arguments["publish"]:
        raise ProtocolError("invalid_publish")
    pending, seen = [arguments], set()
    while pending:
        item = pending.pop()
        if type(item) in {dict, list}:
            if id(item) in seen:
                continue
            seen.add(id(item))
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise ProtocolError("unsupported_argument")
                pending.extend(item.values())
            else:
                pending.extend(item)
        elif type(item) not in {str, int, float, bool, type(None)} or (
            type(item) is float and not math.isfinite(item)
        ):
            raise ProtocolError("unsupported_argument")
    # Detach before selection; non-JSON custom protocols are not coerced.
    stream = io.BytesIO()
    try:
        write_message(stream, arguments)
    except (TypeError, ValueError, RecursionError) as error:
        raise ProtocolError("unsupported_argument") from error
    stream.seek(0)
    return read_message(stream)


def _origins(known=()):
    result = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and name not in known:
            path = Path(filename).resolve()
            if path.is_file():
                result[name] = [
                    str(path),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ]
    return result


def _wire_result(operation, value):
    from . import permission_store as store

    if operation in {"read_snapshot_strict", "read_profile_inventory_snapshot"}:
        return {
            "payload": store._thaw_snapshot(value.payload),
            "generation": value.generation,
            "file_exists": value.file_exists,
        }
    if operation in {
        "install_profile_if_absent",
        "update_imported_profile",
        "replace_profile_with_tombstone",
    }:
        return {
            "profile_id": value.profile_id,
            "revision": value.revision,
            "policy_digest": value.policy_digest,
            "store_generation": value.store_generation,
        }
    return value


def _result(operation, value):
    from . import permission_store as store

    if operation in {"read_snapshot_strict", "read_profile_inventory_snapshot"}:
        _fields(value, {"payload", "generation", "file_exists"})
        if (
            type(value["payload"]) is not dict
            or type(value["generation"]) is not str
            or type(value["file_exists"]) is not bool
        ):
            raise ProtocolError("invalid_snapshot")
        return store.PermissionStoreSnapshot(
            store._freeze_snapshot(value["payload"]),
            value["generation"],
            value["file_exists"],
        )
    if operation in {
        "install_profile_if_absent",
        "update_imported_profile",
        "replace_profile_with_tombstone",
    }:
        _fields(value, {"profile_id", "revision", "policy_digest", "store_generation"})
        if type(value["revision"]) is not int or any(
            type(value[key]) is not str
            for key in ("profile_id", "policy_digest", "store_generation")
        ):
            raise ProtocolError("invalid_profile_result")
        return store.ProfileMutationResult(**value)
    if operation == "catalog_fingerprint":
        if value == ["fresh"]:
            return ("fresh",)
        if (
            type(value) is not list
            or len(value) != 3
            or value[0] != "file"
            or type(value[1]) is not str
            or not _DIGEST.fullmatch(value[1])
            or type(value[2]) is not list
            or len(value[2]) != 5
            or any(type(item) is not int for item in value[2])
        ):
            raise ProtocolError("invalid_fingerprint")
        return (value[0], value[1], tuple(value[2]))
    if operation in {
        "get_kill_switch",
        "has_current_tool_arg_rule",
        "remove_tool_arg_rule",
        "mark_config_changed",
    }:
        valid = type(value) is bool
    elif operation in {"load", "list_suggestion_dismissals"}:
        valid = type(value) is dict
    elif operation in {"list_profiles", "list_tool_arg_rules"}:
        valid = type(value) is list
    elif operation in {"get_global_default", "get_server_entry", "get_tool_entry"}:
        # These legacy raw getters deliberately expose malformed stored JSON too.
        valid = True
    else:
        valid = value is None
    if not valid:
        raise ProtocolError("invalid_result")
    return value


def _facts(source):
    from ..Backup_Recovery import bootstrap, mcp_source_participants as sources

    bound = sources._BINDINGS[source]
    return {
        "selected": str(source.path),
        "selector": str(bound.profile),
        "bootstrap": str(bootstrap.default_bootstrap_root()),
    }


def _config_selection(source):
    """Retain only original memory inputs; default-root IO stays in the child."""
    from ..Backup_Recovery import mcp_source_participants as sources

    data = sources._BINDINGS[source].config._CONFIG_CACHE
    if type(data) is not dict:
        raise ProtocolError("permission_config_ineligible")
    sections = tuple(data.get(name) for name in ("paths", "Paths", "general"))
    if any(section is not None and type(section) is not dict for section in sections):
        raise ProtocolError("permission_config_ineligible")
    values = tuple(
        section.get(key) if type(section) is dict else None
        for section, key in zip(
            sections, ("data_dir", "data_dir", "users_name"), strict=True
        )
    )
    if any(value is not None and type(value) is not str for value in values):
        raise ProtocolError("permission_config_ineligible")
    environment = tuple(
        os.environ.get(key) for key in ("HOME", "USERPROFILE", "TLDW_CONFIG_PATH")
    )
    return (data, *sections), (values, environment)


_brokers = set()
_brokers_lock = threading.RLock()
_intake_closed = False


class PermissionProcess:
    """One path owner; each command retains its actual parent's source capture."""

    def __init__(self, source):
        from ..Backup_Recovery.runtime_producer_lifetime import ProducerLifetime
        from . import permission_store as store

        captured = store._capture_permission_read_entry(source, "load", publish=True)
        if captured is None:
            raise ProtocolError("permission_source_ineligible")
        self.path_lock = source._path_lock
        self.publication = source._permission_publication
        self.sources = weakref.WeakSet()
        self._source_ref = weakref.ref(source)
        self._captures = weakref.WeakKeyDictionary()
        self._remember(source, captured)
        self._request = None
        self.facts = _facts(source)
        self.config_selection = _config_selection(source)
        self.pid = os.getpid()
        self.epoch = secrets.token_hex(16)
        self.lifetime = ProducerLifetime()
        self.process = self.tree = self.monitor = None
        self.finished = threading.Event()
        self.deadline = None
        self.unresolved = None
        self.receipts = []
        self.origins = None
        self.origin_roots = ()
        self.retired = True
        self.closed = False
        with _brokers_lock:
            if _intake_closed:
                raise ProtocolError("permission_intake_closed")
            if any(
                item.path_lock is source._path_lock and not item.closed
                for item in _brokers
            ):
                raise ProtocolError("permission_path_already_owned")
            _brokers.add(self)

    @property
    def source(self):
        if self._request is not None:
            return self._request[0]
        return self._source_ref()

    @property
    def capture(self):
        from . import permission_store as store

        source = self.source
        if source is None:
            raise ProtocolError("permission_source_retired")
        if self._request is not None:
            return self._request[1]
        load, publication, method, publish, receipt = self._captures[source]
        return store._PermissionReadEntry(
            (source, *load), publication, method, publish, (source, *receipt)
        )

    def _remember(self, source, captured):
        if source in self._captures:
            return
        self.sources.add(source)
        self._captures[source] = (
            captured.load[1:],
            captured.publication,
            captured.method,
            captured.publish,
            captured.receipt[1:],
        )
        weakref.finalize(source, self._source_retired)

    def _source_retired(self):
        if not tuple(self.sources) and not self.closed and self.unresolved is None:
            try:
                self.close()
            except BaseException:
                # Global ownership census retains the failed close; never replace it.
                self._poison({"source_retirement": True})

    def _current(self):
        from . import permission_store as store

        containers, values = _config_selection(self.source)
        if (
            self.pid != os.getpid()
            or self.unresolved is not None
            or self.source._mcp_persistence_error is not None
            or not store._permission_read_current(self.source, self.capture)
            or not store._permission_process_definitions_current(self.source)
            or any(
                current is not original
                for current, original in zip(
                    containers, self.config_selection[0], strict=True
                )
            )
            or values != self.config_selection[1]
            or _facts(self.source) != self.facts
        ):
            raise ProtocolError("permission_source_changed")

    def _poison(self, request):
        from ..Backup_Recovery import mcp_source_participants as sources

        self.unresolved = request
        self.publication.fail()
        for source in tuple(sources._BINDINGS):
            if getattr(source, "_path_lock", None) is self.path_lock:
                source._mcp_persistence_error = "mcp_persistence_incomplete"

    def _start(self):
        from ..STT.executor_process_tree import (
            ExecutorProcessTree,
            WorkerContainmentIdentity,
        )
        from ..Tools.workspace_tool_executor import _PopenProcessAdapter

        self._current()
        root = str(Path(__file__).resolve().parents[2])
        package_roots = [
            Path(path).resolve()
            for path in sys.path
            if path and Path(path).name == "site-packages"
        ]
        # Preserve selected package subdirectories (e.g. pywin32's win32/lib)
        # without executing .pth files or adding unrelated parent search paths.
        dependencies = [
            str(Path(path).resolve())
            for path in sys.path
            if path
            and any(Path(path).resolve().is_relative_to(root) for root in package_roots)
        ]
        executable = str(
            Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        )
        executable_digest = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        code = (
            "import runpy,sys;sys.path[:0]="
            + repr([root, *dependencies])
            + ";runpy.run_module('tldw_chatbook.MCP.permission_worker',run_name='__main__')"
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
            target=monitor, name="permission-process-retirement", daemon=True
        )
        self.monitor.start()
        identity = _Identity(self.epoch, secrets.token_hex(16))
        write_message(
            self.process.stdin, identity.frame("Initialize", {"facts": self.facts})
        )
        _, data = identity.accept(read_message(self.process.stdout), {"Ready"})
        _fields(
            data,
            {
                "facts",
                "pid",
                "executable",
                "executable_digest",
                "origins",
                "retirement",
            },
        )
        if (
            data["facts"] != self.facts
            or data["pid"] != self.process.pid
            or data["executable"] != executable
            or data["executable_digest"] != executable_digest
        ):
            raise ProtocolError("worker_identity_changed")
        self.origin_roots = (
            Path(root),
            Path(executable).parent,
            *(Path(path) for path in dependencies),
        )
        self.origins = {}
        self._accept_origins(data["origins"])
        self._receipt(data["retirement"])

    def _accept_origins(self, origins):
        if type(origins) is not dict:
            raise ProtocolError("invalid_origins")
        for name, record in origins.items():
            if (
                type(name) is not str
                or type(record) is not list
                or len(record) != 2
                or not all(type(value) is str for value in record)
            ):
                raise ProtocolError("invalid_origin")
            filename, digest = record
            path = Path(filename).resolve()
            if not any(
                path.is_relative_to(directory) for directory in self.origin_roots
            ):
                raise ProtocolError("worker_origin_outside_selection")
            loaded = getattr(sys.modules.get(name), "__file__", None)
            if (
                loaded is not None
                and Path(loaded).resolve() != path
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ProtocolError("worker_origin_changed")
            if name in self.origins and self.origins[name] != record:
                raise ProtocolError("worker_origin_changed")
            self.origins[name] = record

    def _receipt(self, receipt):
        _fields(
            receipt,
            {"retired", "acquisitions", "raw", "leases", "descriptors", "startup"},
        )
        if receipt["retired"] is not True or any(
            type(receipt[key]) is not int or receipt[key] < 0
            for key in set(receipt) - {"retired"}
        ):
            raise ProtocolError("native_retirement_unproven")
        self.receipts.append(receipt)

    def call(
        self,
        operation,
        *,
        publish=True,
        value=None,
        payload=None,
        expected_generation=None,
        require_current=None,
        _source=None,
        _arguments=None,
    ):
        """Run one original body while retaining the caller's path reservation."""
        from . import permission_store as store
        from ..Backup_Recovery import storage_admission as storage

        arguments = (
            _arguments
            if _arguments is not None
            else (
                {"publish": publish}
                if operation in {"load", "get_kill_switch"}
                else {"value": value}
                if operation == "set_kill_switch"
                else {"payload": payload, "expected_generation": expected_generation}
                if operation == "save"
                else {}
            )
        )
        if operation == "save":
            payload = arguments["payload"]
        arguments = _command_arguments(operation, arguments)
        creator = (os.getpid(), threading.current_thread(), storage._task_identity())
        callback = (
            None
            if require_current is None
            else (
                require_current,
                require_current.__code__,
                store._capture_controller_inputs(require_current),
            )
        )

        def current():
            if creator != (
                os.getpid(),
                threading.current_thread(),
                storage._task_identity(),
            ):
                raise ProtocolError("foreign_request_creator")
            self._current()
            if callback is not None:
                function, code, inputs = callback
                if (
                    function.__code__ is not code
                    or store._capture_controller_inputs(function) != inputs
                ):
                    raise ProtocolError("source_callback_changed")
                function()

        with self.lifetime.operation(), self.path_lock:
            if self.closed:
                raise ProtocolError("permission_process_closed")
            source = self.source if _source is None else _source
            captured = store._capture_permission_read_entry(
                source, "load", publish=True
            )
            if captured is None or source._path_lock is not self.path_lock:
                raise ProtocolError("permission_source_changed")
            self._remember(source, captured)
            saved_source = self._source_ref
            self._source_ref = weakref.ref(source)
            self._request = (source, self.capture)
            identity = _Identity(self.epoch, secrets.token_hex(16))
            mirror = _PublicationMirror(self.source._permission_publication)
            selected = False
            settled_failure = None
            data = None
            self.deadline = time.monotonic() + 120
            try:
                current()
                if self.process is None:
                    self._start()
                current()
                selected = True
                write_message(
                    self.process.stdin,
                    identity.frame(
                        "Command", {"operation": operation, "arguments": arguments}
                    ),
                )
                while True:
                    kind, data = identity.accept(
                        read_message(self.process.stdout),
                        {"Check", "MutationBegin", "Observe", "Settled"},
                    )
                    if kind == "Settled":
                        _fields(
                            data,
                            {
                                "status",
                                "value",
                                "error",
                                "stamp",
                                "publication",
                                "retirement",
                                "facts",
                                "origins",
                            },
                        )
                        self._receipt(data["retirement"])
                        self._accept_origins(data["origins"])
                        current()
                        if data["facts"] != self.facts:
                            raise ProtocolError("worker_selection_changed")
                        if data["status"] != "ok":
                            if data["status"] != "refused":
                                raise ProtocolError("invalid_outcome")
                            failure = data["error"]
                            _fields(failure, {"kind", "category", "errno"})
                            category = failure["category"]
                            if type(category) is not str:
                                raise ProtocolError("invalid_error")
                            if failure["kind"] == "mutation":
                                exception = store.ProfileMutationError(category)
                            elif failure["kind"] == "snapshot" and category in {
                                "invalid_json",
                                "invalid_shape",
                                "duplicate_key",
                                "io_error",
                            }:
                                exception = store.PermissionStoreSnapshotError(category)
                            elif failure["kind"] == "io" and category == "io_error":
                                code = failure["errno"]
                                if code is not None and type(code) is not int:
                                    raise ProtocolError("invalid_errno")
                                exception = OSError(
                                    code, "permission_store_read_failed"
                                )
                            elif failure["kind"] == "recovery":
                                from ..Backup_Recovery.bootstrap import RecoveryRequired

                                exception = RecoveryRequired(category)
                            elif failure["kind"] == "value":
                                exception = ValueError(category)
                            elif failure["kind"] == "attribute":
                                exception = AttributeError(category)
                            elif failure["kind"] == "type":
                                exception = TypeError(category)
                            else:
                                raise ProtocolError("invalid_error")
                            mirror.settle(data["publication"])
                            settled_failure = exception
                            raise exception
                        if data["error"] is not None:
                            raise ProtocolError("contradictory_result")
                        result = _result(operation, data["value"])
                        if (
                            operation == "load"
                            and publish
                            and data["publication"]["digest"] is not None
                        ):
                            digest = hashlib.sha256(
                                json.dumps(
                                    result, sort_keys=True, allow_nan=False
                                ).encode()
                            ).hexdigest()
                            if digest != data["publication"]["digest"]:
                                raise ProtocolError("payload_digest_changed")
                        if operation == "save":
                            if type(data["stamp"]) is not str:
                                raise ProtocolError("missing_caller_stamp")
                        elif data["stamp"] is not None:
                            raise ProtocolError("unexpected_caller_stamp")
                        mirror.settle(data["publication"])
                        if operation == "save":
                            payload["updated_at"] = data["stamp"]
                        return result
                    current()
                    if kind == "Check":
                        _fields(data, {"site", "facts"})
                        if data["facts"] != self.facts or data["site"] not in {
                            "entry",
                            "protected_open",
                            "permission_read_custody",
                            "source_custody_entry",
                            "source_custody_exit",
                        }:
                            raise ProtocolError("invalid_boundary")
                        reply = "Continue"
                    elif kind == "Observe":
                        _fields(data, {"digest"})
                        mirror.observe(data["digest"])
                        reply = "Acknowledge"
                    else:
                        _fields(data, set())
                        mirror.begin()
                        reply = "Acknowledge"
                    write_message(self.process.stdin, identity.frame(reply, {}))
            except BaseException as failure:
                if failure is settled_failure:
                    raise
                if selected or self.process is not None:
                    self._poison(
                        {
                            "identity": identity,
                            "creator": creator,
                            "source": source,
                            "capture": self.capture,
                            "arguments": arguments,
                            "payload": payload,
                            "mirror": mirror,
                            "received": data,
                        }
                    )
                    # Close input to release a child parked at the rejected rendezvous.
                    if self.process is not None:
                        self.process.stdin.close()
                raise
            finally:
                self._request = None
                self._source_ref = saved_source
                self.deadline = (
                    None if self.unresolved is None else time.monotonic() + 5
                )

    def close(self):
        """Require Drained, actual process exit, pipes and containment retirement."""
        with self.path_lock:
            if self.unresolved is not None:
                raise ProtocolError("permission_outcome_unresolved")
            self.closed = True
            self.lifetime.close()
            if self.process is None or self.retired:
                with _brokers_lock:
                    _brokers.discard(self)
                return
            identity = _Identity(self.epoch, secrets.token_hex(16))
            self.deadline = time.monotonic() + 120
            try:
                write_message(self.process.stdin, identity.frame("Shutdown", {}))
                _, data = identity.accept(
                    read_message(self.process.stdout), {"Drained"}
                )
                _fields(data, {"empty"})
                if data["empty"] is not True:
                    raise ProtocolError("worker_not_drained")
                self.process.wait(timeout=10)
                if self.process.returncode != 0:
                    raise ProtocolError("worker_exit_failed")
                self._retire_process()
                with _brokers_lock:
                    _brokers.discard(self)
            except BaseException:
                self._poison({"shutdown": identity})
                raise
            finally:
                self.deadline = (
                    None if self.unresolved is None else time.monotonic() + 5
                )

    def _retire_process(self):
        self.finished.set()
        proven = False
        errors = []
        try:
            if self.tree is not None:
                proven = self.tree.terminate_tree()
            else:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=5)
                proven = self.process.poll() is not None
        except BaseException as error:
            errors.append((self.tree or self.process, error))
        finally:
            if self.monitor is not None:
                try:
                    self.monitor.join(timeout=5)
                except BaseException as error:
                    errors.append((self.monitor, error))
            for pipe in (self.process.stdin, self.process.stdout):
                try:
                    pipe.close()
                except BaseException as error:
                    errors.append((pipe, error))
        self.retired = (
            proven
            and self.process.poll() is not None
            and (self.monitor is None or not self.monitor.is_alive())
            and self.process.stdin.closed
            and self.process.stdout.closed
        )
        if errors:
            self._retirement_errors = [
                *getattr(self, "_retirement_errors", ()),
                *errors,
            ]
            raise BaseExceptionGroup(
                "process_retirement_failed", [error for _resource, error in errors]
            )
        if not self.retired:
            raise ProtocolError("process_retirement_unproven")

    def contain_unresolved(self):
        """Retire physical containment without claiming persistence settlement."""
        if self.unresolved is None:
            raise ProtocolError("not_unresolved")
        if self.process is not None:
            self._retire_process()

    def captured_load(self, require_current):
        """Return the original parent receipt, never child identity substitutes."""
        with self.source._path_lock:
            payload = self.call("load", publish=False, require_current=require_current)
            self._current()
            return payload, self.capture.receipt


def select(source):
    """Choose compatibility before issue; never replay a selected failure."""
    from . import permission_store as store
    from ..Backup_Recovery import raw_participants as raw

    local = sys.modules.get("tldw_chatbook.MCP.local_permission_operation")
    if local is not None and local.selected(source):
        return None
    if _active() is not None or getattr(raw._local, "operation", None) is not None:
        return None
    captured = store._capture_permission_read_entry(source, "load", publish=True)
    if captured is None or not store._permission_process_definitions_current(source):
        return None
    try:
        config_selection = _config_selection(source)
    except ProtocolError:
        return None
    facts = _facts(source)
    with source._path_lock:
        with _brokers_lock:
            if _intake_closed:
                raise ProtocolError("permission_intake_closed")
            owners = tuple(_brokers)
        for broker in owners:
            if broker.unresolved is not None:
                if broker.path_lock is source._path_lock:
                    raise ProtocolError("permission_outcome_unresolved")
                if not broker.retired:
                    raise ProtocolError("permission_context_busy")
                continue
            same_context = (
                broker.facts == facts
                and all(
                    a is b
                    for a, b in zip(
                        broker.config_selection[0], config_selection[0], strict=True
                    )
                )
                and broker.config_selection[1] == config_selection[1]
                and (
                    source not in broker._captures
                    or broker._captures[source][4] == captured.receipt[1:]
                )
            )
            if (
                broker.path_lock is source._path_lock
                and same_context
                and not broker.closed
            ):
                broker._remember(source, captured)
                return broker
            if not same_context:
                # A previous healthy helper must release startup before re-selection.
                # A caller may already own another mutation_fence. Never invert
                # two path reservations while retiring a changed configuration.
                if not broker.path_lock.acquire(blocking=False):
                    raise ProtocolError("permission_context_busy")
                try:
                    broker.close()
                finally:
                    broker.path_lock.release()
        return PermissionProcess(source)


def forward(source, operation, args, kwargs):
    from . import permission_store as store

    if operation not in _SCHEMAS or _active() is not None:
        return False, None
    # Bind the actual frozen stock signature before wire conversion. Custom values
    # and unusual calling protocols keep their original Python behavior.
    callback = store._PERMISSION_PROCESS_COMMANDS[operation]
    try:
        bound = inspect.signature(callback).bind(source, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        if operation in {"load", "get_kill_switch"}:
            arguments = {"publish": True}
        _command_arguments(operation, arguments)
    except (TypeError, ProtocolError):
        return False, None
    broker = select(source)
    if broker is None:
        return False, None
    context = getattr(_local, "parent_boundary", None)
    checker = None
    if context is not None:
        if context[0] is not source:
            raise ProtocolError("permission_callback_source_changed")
        checker = context[1]
    return True, broker.call(
        operation, _source=source, _arguments=arguments, require_current=checker
    )


@contextmanager
def reservation(source, require_current):
    """Keep the parent callback local while its individual store calls route."""
    with source._path_lock:
        previous = getattr(_local, "parent_boundary", None)
        _local.parent_boundary = (source, require_current)
        try:
            require_current()
            yield
            require_current()
        finally:
            _local.parent_boundary = previous


class PermissionProcessLifetime:
    """Census includes helpers created without an installed MCP service."""

    def __init__(self):
        self.closed = None

    def _maintenance_close_admission(self):
        global _intake_closed
        with _brokers_lock:
            _intake_closed = True
            if self.closed is None:
                self.closed = tuple(_brokers)
            for broker in self.closed:
                broker.lifetime.close()

    async def _maintenance_drain(self, deadline):
        from .console_snapshot import _owned_worker

        if self.closed is None:
            raise ProtocolError("permission_intake_open")
        for broker in self.closed:
            if broker.unresolved is not None or not await broker.lifetime.drain(
                deadline
            ):
                return False
            if time.monotonic() >= deadline:
                return False
            try:
                await _owned_worker(broker.close)
            except ProtocolError:
                return False
        return self.drained()

    def drained(self):
        with _brokers_lock:
            return (
                self.closed is not None
                and not _brokers
                and all(
                    broker.retired and broker.unresolved is None
                    for broker in self.closed
                )
            )

    def _maintenance_resume(self):
        global _intake_closed
        with _brokers_lock:
            if not self.drained():
                raise ProtocolError("permission_process_not_drained")
            for broker in self.closed:
                if not tuple(broker.sources):
                    continue
                broker.process = broker.tree = broker.monitor = None
                broker.finished = threading.Event()
                broker.epoch = secrets.token_hex(16)
                broker.closed = False
                broker.lifetime.resume()
                _brokers.add(broker)
            self.closed = None
            _intake_closed = False


participant = PermissionProcessLifetime()


def _shutdown():
    """Exit healthy direct-store helpers before the parent's startup teardown."""
    for broker in tuple(_brokers):
        if broker.unresolved is not None:
            broker.contain_unresolved()
        else:
            broker.close()


atexit.register(_shutdown)
