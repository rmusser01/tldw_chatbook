"""Pre-import containment child for the TASK-23019 closeout runner."""

from __future__ import annotations

import argparse
import gettext
import importlib.util
import json
import os
import stat
import sys
import sysconfig
import threading
from pathlib import Path

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None


CONTAINMENT_EXIT_STATUS = 86
FILESYSTEM_READ_DENIED = b'{"category":"filesystem_read_denied"}\n'
FILESYSTEM_WRITE_DENIED = b'{"category":"filesystem_write_denied"}\n'
NETWORK_DENIED = b'{"category":"network_denied"}\n'
PROCESS_DENIED = b'{"category":"process_denied"}\n'
RAW_RESULT_BYTE_LIMIT = 1024 * 1024

_ATTEMPT_FD = -1
_SCRATCH = ""
_READ_ROOTS: tuple[str, ...] = ()
_READ_FILES: frozenset[str] = frozenset()
_METADATA_ONLY_PATHS: frozenset[str] = frozenset()
_ENUMERATION_DIRS: frozenset[str] = frozenset()
_OPEN_FDS: dict[int, tuple[str, bool, bool, tuple[int, int, int, tuple[str, int]]]] = {}
_RESOLVING_PATH = threading.local()
_OPEN_AUDIT_PATH = threading.local()
THREAD_BASELINE: tuple[int | None, ...] = ()
TASK_BASELINE: tuple[object, ...] = ()
DENIED_ROOTS: tuple[str, ...] = ()

_REAL_OS_OPEN = os.open
_REAL_OS_CLOSE = os.close
_REAL_OS_DUP = os.dup
_REAL_OS_REPLACE_FD_APIS = {
    name: getattr(os, name) for name in ("dup2", "dup3") if hasattr(os, name)
}
_REAL_OS_WRITE = os.write
_REAL_OS_FSYNC = os.fsync
_REAL_OS_FSTAT = os.fstat
_REAL_FCNTL = _fcntl.fcntl if _fcntl is not None else None
_F_GETFL = _fcntl.F_GETFL if _fcntl is not None else None
_REAL_GET_OSFHANDLE = _msvcrt.get_osfhandle if _msvcrt is not None else None
_ACCESS_MODE_MASK = getattr(os, "O_ACCMODE", os.O_WRONLY | os.O_RDWR)
_REAL_OS_EXIT = os._exit
_REAL_OS_REALPATH = os.path.realpath
_REAL_METADATA_READ_APIS = {
    name: getattr(os, name)
    for name in ("stat", "lstat", "access", "readlink", "statvfs", "pathconf")
    if hasattr(os, name)
}
_REAL_UNAUDITED_MUTATORS = {
    name: getattr(os, name) for name in ("mkfifo", "mknod") if hasattr(os, name)
}
_REAL_FD_READ_APIS = {
    name: getattr(os, name) for name in ("listdir", "scandir", "chdir")
}
_OS_SUPPORT_SETS = (
    "supports_dir_fd",
    "supports_effective_ids",
    "supports_fd",
    "supports_follow_symlinks",
)


def _contain(record: bytes) -> None:
    _REAL_OS_WRITE(_ATTEMPT_FD, record)
    _REAL_OS_FSYNC(_ATTEMPT_FD)
    _REAL_OS_EXIT(CONTAINMENT_EXIT_STATUS)


def _resolved(path: object) -> str | None:
    prior_active = getattr(_RESOLVING_PATH, "active", False)
    try:
        if path is None:
            return _REAL_OS_REALPATH(os.getcwd())
        if isinstance(path, int):
            return None
        lexical = os.path.abspath(os.fsdecode(os.fspath(path)))
        _RESOLVING_PATH.active = True
        return _REAL_OS_REALPATH(lexical)
    except (TypeError, ValueError):
        return None
    finally:
        _RESOLVING_PATH.active = prior_active


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _strict_ancestor(path: str) -> bool:
    return any(path != root and _under(root, path) for root in _READ_ROOTS)


def _fd_platform_token(descriptor: int) -> tuple[str, int] | None:
    if _REAL_FCNTL is not None and _F_GETFL is not None:
        return "access_mode", _REAL_FCNTL(descriptor, _F_GETFL) & _ACCESS_MODE_MASK
    if _REAL_GET_OSFHANDLE is not None:
        return "os_handle", int(_REAL_GET_OSFHANDLE(descriptor))
    return None


def _fd_identity(
    descriptor: int, *, require_write: bool = False
) -> tuple[int, int, int, tuple[str, int]] | None:
    try:
        status = _REAL_OS_FSTAT(descriptor)
        platform_token = _fd_platform_token(descriptor)
        if require_write and _REAL_FCNTL is None and stat.S_ISREG(status.st_mode):
            _REAL_OS_WRITE(descriptor, b"")
    except (OSError, ValueError):
        return None
    if platform_token is None:
        return None
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode), platform_token


def _fd_authority(
    descriptor: int,
) -> tuple[str, bool, bool, tuple[int, int, int, tuple[str, int]]] | None:
    authority = _OPEN_FDS.get(descriptor)
    if (
        authority is not None
        and _fd_identity(descriptor, require_write=authority[2] and not authority[1])
        == authority[3]
    ):
        return authority
    _OPEN_FDS.pop(descriptor, None)
    return None


def _resolved_at(path: object, dir_fd: object = None) -> str | None:
    if isinstance(dir_fd, int) and dir_fd < 0:
        dir_fd = None
    if isinstance(path, int):
        opened = _fd_authority(path)
        return opened[0] if opened is not None else None
    try:
        frozen = os.fsdecode(os.fspath(path))
    except (TypeError, ValueError):
        return None
    if dir_fd is None or os.path.isabs(frozen):
        return _resolved(frozen)
    if not isinstance(dir_fd, int) or os.pardir in frozen.split(os.sep):
        return None
    opened = _fd_authority(dir_fd)
    if opened is None or not opened[1]:
        return None
    return _resolved(os.path.join(opened[0], frozen))


def _read_allowed(path: object) -> bool:
    resolved = _resolved_at(path)
    return resolved is not None and (
        resolved in _READ_FILES or any(_under(resolved, root) for root in _READ_ROOTS)
    )


def _read_event_allowed(event: str, path: object) -> bool:
    resolved = _resolved_at(path)
    return _read_allowed(resolved) or (
        event in {"os.listdir", "os.scandir"} and resolved in _ENUMERATION_DIRS
    )


def _write_allowed(path: object) -> bool:
    resolved = _resolved_at(path)
    return resolved is not None and _under(resolved, _SCRATCH)


def _fd_write_allowed(descriptor: int) -> bool:
    opened = _fd_authority(descriptor)
    return bool(opened is not None and opened[2] and _under(opened[0], _SCRATCH))


def _traversal_allowed(path: str, flags: int) -> bool:
    directory_only = bool(getattr(os, "O_DIRECTORY", 0) & flags)
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    no_follow = not no_follow_flag or bool(no_follow_flag & flags)
    return directory_only and no_follow and _strict_ancestor(path)


def _write_requested(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(character in mode for character in "wax+"):
        return True
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    return isinstance(flags, int) and bool(flags & write_flags)


_MUTATION_SINGLE_PATH_EVENTS = frozenset(
    {
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.chmod",
        "os.chown",
        "os.utime",
        "os.truncate",
        "os.chflags",
    }
)
_READ_PATH_EVENTS = frozenset({"os.listdir", "os.scandir", "os.chdir"})
_MUTATION_DIR_FD_INDEX = {
    "os.mkdir": 2,
    "os.remove": 1,
    "os.rmdir": 1,
    "os.chmod": 2,
    "os.chown": 3,
    "os.utime": 3,
}
_NETWORK_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.sendto",
        "socket.sendmsg",
        "socket.bind",
        "socket.listen",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
    }
)


def _audit(event: str, arguments: tuple[object, ...]) -> None:
    if event == "open":
        path = getattr(_OPEN_AUDIT_PATH, "path", arguments[0])
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        if _write_requested(mode, flags):
            if not _write_allowed(path):
                _contain(FILESYSTEM_WRITE_DENIED)
        elif not (_read_allowed(path) or _traversal_allowed(path, flags)):
            _contain(FILESYSTEM_READ_DENIED)
        return
    if event in _READ_PATH_EVENTS:
        path = arguments[0] if arguments else None
        if not _read_event_allowed(event, path):
            _contain(FILESYSTEM_READ_DENIED)
        return
    if event in _MUTATION_SINGLE_PATH_EVENTS:
        if arguments and isinstance(arguments[0], int):
            if not _fd_write_allowed(arguments[0]):
                _contain(FILESYSTEM_WRITE_DENIED)
            return
        dir_fd_index = _MUTATION_DIR_FD_INDEX.get(event)
        dir_fd = (
            arguments[dir_fd_index]
            if dir_fd_index is not None and len(arguments) > dir_fd_index
            else None
        )
        path = _resolved_at(arguments[0], dir_fd) if arguments else None
        if not _write_allowed(path):
            _contain(FILESYSTEM_WRITE_DENIED)
        return
    if event == "os.rename":
        source_fd = arguments[2] if len(arguments) > 2 else None
        destination_fd = arguments[3] if len(arguments) > 3 else None
        if len(arguments) < 2 or not all(
            _write_allowed(path)
            for path in (
                _resolved_at(arguments[0], source_fd),
                _resolved_at(arguments[1], destination_fd),
            )
        ):
            _contain(FILESYSTEM_WRITE_DENIED)
        return
    if event == "os.link":
        source_fd = arguments[2] if len(arguments) > 2 else None
        destination_fd = arguments[3] if len(arguments) > 3 else None
        if (
            len(arguments) < 2
            or not _write_allowed(_resolved_at(arguments[0], source_fd))
            or not _write_allowed(_resolved_at(arguments[1], destination_fd))
        ):
            _contain(FILESYSTEM_WRITE_DENIED)
        return
    if event == "os.symlink":
        destination_fd = arguments[2] if len(arguments) > 2 else None
        if len(arguments) < 2 or not _write_allowed(
            _resolved_at(arguments[1], destination_fd)
        ):
            _contain(FILESYSTEM_WRITE_DENIED)
        return
    if event == "sqlite3.connect":
        database = arguments[0] if arguments else None
        if database != ":memory:" and not _write_allowed(database):
            _contain(FILESYSTEM_WRITE_DENIED)
        return
    if event in _NETWORK_EVENTS:
        _contain(NETWORK_DENIED)
    if event in {"os.system", "os.posix_spawn", "os.fork", "os.forkpty"} or event in {
        "subprocess.Popen",
        "os.exec",
    }:
        _contain(PROCESS_DENIED)


def _deny_network(*_arguments: object, **_keywords: object) -> None:
    _contain(NETWORK_DENIED)


def _deny_process(*_arguments: object, **_keywords: object) -> None:
    _contain(PROCESS_DENIED)


def _process_guard():
    def guarded(*_arguments: object, **_keywords: object) -> None:
        _contain(PROCESS_DENIED)

    return guarded


def _contained_architecture(
    executable: str = sys.executable, bits: str = "", linkage: str = ""
) -> tuple[str, str]:
    """Report interpreter width without platform's subprocess fallback."""
    del executable
    return bits or ("64bit" if sys.maxsize > 2**32 else "32bit"), linkage


def _guarded_os_open(
    path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
) -> int:
    resolved = _resolved_at(path, dir_fd)
    write_requested = _write_requested(None, flags)
    if resolved is None or (
        not _write_allowed(resolved)
        if write_requested
        else not (_read_allowed(resolved) or _traversal_allowed(resolved, flags))
    ):
        _contain(FILESYSTEM_WRITE_DENIED if write_requested else FILESYSTEM_READ_DENIED)
    had_prior_audit_path = hasattr(_OPEN_AUDIT_PATH, "path")
    prior_audit_path = getattr(_OPEN_AUDIT_PATH, "path", None)
    try:
        _OPEN_AUDIT_PATH.path = resolved
        descriptor = _REAL_OS_OPEN(path, flags, mode, dir_fd=dir_fd)
    finally:
        if had_prior_audit_path:
            _OPEN_AUDIT_PATH.path = prior_audit_path
        else:
            del _OPEN_AUDIT_PATH.path
    write_capable = flags & _ACCESS_MODE_MASK != os.O_RDONLY
    identity = _fd_identity(descriptor, require_write=write_capable)
    if identity is not None:
        platform_kind, platform_value = identity[3]
        if platform_kind == "access_mode":
            write_capable = platform_value != os.O_RDONLY
        _OPEN_FDS[descriptor] = (
            resolved,
            identity[2] == stat.S_IFDIR,
            write_capable,
            identity,
        )
    return descriptor


def _guarded_os_close(descriptor: int) -> None:
    # Revoke before close: another thread may reuse the numeric descriptor as
    # soon as the kernel close succeeds, and a post-close pop could erase the
    # newly opened descriptor's independent authority.
    _OPEN_FDS.pop(descriptor, None)
    _REAL_OS_CLOSE(descriptor)


def _inherit_fd_authority(source: int, destination: int) -> None:
    authority = _fd_authority(source)
    identity = _fd_identity(
        destination,
        require_write=(authority is not None and authority[2] and not authority[1]),
    )
    same_authority = (
        authority is not None
        and identity is not None
        and identity[:3] == authority[3][:3]
        and (authority[3][3][0] == "os_handle" or identity[3] == authority[3][3])
    )
    if not same_authority:
        _OPEN_FDS.pop(destination, None)
    else:
        _OPEN_FDS[destination] = (*authority[:3], identity)


def _guarded_os_dup(descriptor: int) -> int:
    duplicated = _REAL_OS_DUP(descriptor)
    _inherit_fd_authority(descriptor, duplicated)
    return duplicated


def _guarded_os_replace_fd(
    name: str, source: int, destination: int, *args: object, **kwargs: object
) -> int:
    replaced = _REAL_OS_REPLACE_FD_APIS[name](source, destination, *args, **kwargs)
    _inherit_fd_authority(source, destination)
    return replaced


def _replace_fd_guard(name: str):
    def guarded(source: int, destination: int, *args: object, **kwargs: object) -> int:
        return _guarded_os_replace_fd(name, source, destination, *args, **kwargs)

    return guarded


def _guarded_metadata_read(name: str, path: object, *args: object, **kwargs: object):
    if getattr(_RESOLVING_PATH, "active", False):
        return _REAL_METADATA_READ_APIS[name](path, *args, **kwargs)
    frozen_path = path if isinstance(path, int) else os.fspath(path)
    dir_fd = kwargs.get("dir_fd")
    resolved = _resolved_at(frozen_path, dir_fd)
    traversal_component = (
        os.fsdecode(frozen_path) if isinstance(frozen_path, (str, bytes)) else ""
    )
    traversal_stat = (
        name == "stat"
        and isinstance(dir_fd, int)
        and kwargs.get("follow_symlinks") is False
        and traversal_component not in {"", os.curdir, os.pardir}
        and not os.path.isabs(traversal_component)
        and os.sep not in traversal_component
        and (os.altsep is None or os.altsep not in traversal_component)
        and (opened := _fd_authority(dir_fd)) is not None
        and opened[1]
        and resolved is not None
        and _strict_ancestor(resolved)
    )
    named_nofollow_ancestor = resolved in _METADATA_ONLY_PATHS and (
        name == "lstat" or (name == "stat" and kwargs.get("follow_symlinks") is False)
    )
    if resolved is None or not (
        _read_allowed(resolved) or traversal_stat or named_nofollow_ancestor
    ):
        _contain(FILESYSTEM_READ_DENIED)
    return _REAL_METADATA_READ_APIS[name](frozen_path, *args, **kwargs)


def _guarded_realpath(path: object, *args: object, **kwargs: object):
    prior_active = getattr(_RESOLVING_PATH, "active", False)
    try:
        frozen_path = os.fspath(path)
        if not _read_allowed(frozen_path):
            _contain(FILESYSTEM_READ_DENIED)
        _RESOLVING_PATH.active = True
        return _REAL_OS_REALPATH(frozen_path, *args, **kwargs)
    finally:
        _RESOLVING_PATH.active = prior_active


def _guarded_unaudited_mutator(
    name: str, path: object, *args: object, **kwargs: object
):
    frozen_path = os.fspath(path)
    resolved = _resolved_at(frozen_path, kwargs.get("dir_fd"))
    if resolved is None or not _write_allowed(resolved):
        _contain(FILESYSTEM_WRITE_DENIED)
    return _REAL_UNAUDITED_MUTATORS[name](frozen_path, *args, **kwargs)


def _metadata_read_guard(name: str):
    def guarded(path: object, *args: object, **kwargs: object):
        return _guarded_metadata_read(name, path, *args, **kwargs)

    return guarded


def _mutator_guard(name: str):
    def guarded(path: object, *args: object, **kwargs: object):
        return _guarded_unaudited_mutator(name, path, *args, **kwargs)

    return guarded


def _fd_read_guard(name: str):
    def guarded(path: object = "."):
        if not _read_event_allowed("os." + name, path):
            _contain(FILESYSTEM_READ_DENIED)
        return _REAL_FD_READ_APIS[name](path)

    return guarded


def _replace_os_api(name: str, replacement: object) -> None:
    original = getattr(os, name)
    for set_name in _OS_SUPPORT_SETS:
        supported = getattr(os, set_name, None)
        if supported is not None and original in supported:
            supported.remove(original)
            supported.add(replacement)
    setattr(os, name, replacement)


def _runtime_authority(
    checkout: str,
    scratch: str,
    *,
    configured_paths: tuple[str, ...] | None = None,
    locale_root: str | None = None,
    language: str | None = None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    roots = {checkout, scratch}
    files = {_REAL_OS_REALPATH(sys.executable), _REAL_OS_REALPATH(os.devnull)}
    if os.name == "posix":
        files.add(_REAL_OS_REALPATH("/proc/stat"))
    locale = _REAL_OS_REALPATH(
        locale_root or os.path.join(sys.base_prefix, "share", "locale")
    )
    if os.path.isdir(locale):
        roots.add(locale)
    else:
        files.add(locale)
        if language is None:
            language = next(
                (
                    os.environ[name]
                    for name in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG")
                    if os.environ.get(name)
                ),
                "C",
            )
        variants = {
            variant
            for configured in language.split(":")
            for variant in (configured, *gettext._expand_lang(configured))
            if variant
            and variant != "C"
            and variant not in {os.curdir, os.pardir}
            and os.sep not in variant
            and (os.altsep is None or os.altsep not in variant)
        }
        for variant in variants:
            expected = (variant, "LC_MESSAGES", "messages.mo")
            probe = _REAL_OS_REALPATH(os.path.join(locale, *expected))
            if (
                probe != locale
                and _under(probe, locale)
                and tuple(os.path.relpath(probe, locale).split(os.sep)) == expected
            ):
                files.add(probe)
    runtime_paths = sysconfig.get_paths()
    for name in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = runtime_paths.get(name)
        if value:
            roots.add(_REAL_OS_REALPATH(value))
    configured_paths = (
        tuple(sys.path[1:]) if configured_paths is None else configured_paths
    )
    for value in configured_paths:
        if not value or value.startswith("__editable__"):
            continue
        resolved = _REAL_OS_REALPATH(value)
        if os.path.isdir(resolved):
            roots.add(resolved)
        else:
            files.add(resolved)
    return tuple(sorted(roots)), frozenset(files)


def _runtime_metadata_ancestors(scratch: str) -> frozenset[str]:
    """Return exact no-follow metadata probes needed to reach macOS temp roots."""
    if sys.platform != "darwin":
        return frozenset()
    parts = Path(scratch).parts
    if parts[:3] == (os.sep, "private", "tmp"):
        return frozenset({"/private", "/private/tmp"})
    if (
        len(parts) >= 7
        and parts[:4] == (os.sep, "private", "var", "folders")
        and parts[6] == "T"
    ):
        return frozenset(str(Path(*parts[:part_count])) for part_count in range(2, 8))
    return frozenset()


def _install_boundary(checkout: str, scratch: str) -> None:
    global _ATTEMPT_FD, _ENUMERATION_DIRS, _METADATA_ONLY_PATHS
    global _READ_FILES, _READ_ROOTS, _SCRATCH
    global TASK_BASELINE, THREAD_BASELINE

    _SCRATCH = _REAL_OS_REALPATH(scratch)
    checkout_root = _REAL_OS_REALPATH(checkout)
    sys.path[:] = [
        checkout_root,
        *(
            value
            for value in sys.path
            if value
            and not value.startswith("__editable__")
            and _REAL_OS_REALPATH(value) != checkout_root
        ),
    ]
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not getattr(finder, "__module__", type(finder).__module__).startswith(
            "__editable__"
        )
    ]
    attempts_path = _REAL_OS_REALPATH(os.path.join(_SCRATCH, "attempts.jsonl"))
    if not _under(attempts_path, _SCRATCH):
        raise SystemExit(2)
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    _ATTEMPT_FD = _REAL_OS_OPEN(attempts_path, open_flags, 0o600)
    _READ_ROOTS, _READ_FILES = _runtime_authority(checkout_root, _SCRATCH)
    _METADATA_ONLY_PATHS = _runtime_metadata_ancestors(_SCRATCH)
    _ENUMERATION_DIRS = frozenset({_REAL_OS_REALPATH("/dev/fd")})
    sys.addaudithook(_audit)

    import platform
    import socket
    import subprocess

    platform.architecture = _contained_architecture
    socket.has_ipv6 = False
    for name in (
        "connect",
        "sendto",
        "sendmsg",
        "bind",
        "listen",
    ):
        if hasattr(socket.socket, name):
            setattr(socket.socket, name, _deny_network)
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        if hasattr(socket, name):
            setattr(socket, name, _deny_network)

    subprocess.Popen = _deny_process
    for name in dir(os):
        value = getattr(os, name, None)
        if callable(value) and (
            name == "system"
            or name in {"posix_spawn", "posix_spawnp", "fork", "forkpty"}
            or name.startswith("spawn")
            or name.startswith("exec")
        ):
            _replace_os_api(name, _process_guard())
    _replace_os_api("open", _guarded_os_open)
    _replace_os_api("close", _guarded_os_close)
    _replace_os_api("dup", _guarded_os_dup)
    for name in _REAL_OS_REPLACE_FD_APIS:
        _replace_os_api(name, _replace_fd_guard(name))
    os.path.realpath = _guarded_realpath
    for name in _REAL_METADATA_READ_APIS:
        _replace_os_api(name, _metadata_read_guard(name))
    for name in _REAL_UNAUDITED_MUTATORS:
        _replace_os_api(name, _mutator_guard(name))
    for name in _REAL_FD_READ_APIS:
        _replace_os_api(name, _fd_read_guard(name))

    THREAD_BASELINE = tuple(thread.ident for thread in threading.enumerate())
    TASK_BASELINE = ()


class ResultRecorder:
    def __init__(self) -> None:
        self.results: dict[str, str] = {}

    def pytest_collection_finish(self, session: object) -> None:
        for item in session.items:
            self.results[item.nodeid] = "NOT_SETTLED"

    def pytest_runtest_logreport(self, report: object) -> None:
        if report.failed:
            self.results[report.nodeid] = "FAIL"
        elif report.when == "call" or (report.when == "setup" and report.skipped):
            self.results[report.nodeid] = {
                "passed": "PASS",
                "failed": "FAIL",
                "skipped": "SKIP",
            }[report.outcome]


def _write_json(path: Path, payload: object) -> bool:
    """Atomically write a bounded result or a small fail-closed marker."""
    temporary = path.with_name(path.name + ".tmp")
    oversized = False
    try:
        with temporary.open("wb") as handle:
            size = 0
            for text in json.JSONEncoder(indent=2, sort_keys=True).iterencode(payload):
                chunk = text.encode("utf-8")
                if size + len(chunk) > RAW_RESULT_BYTE_LIMIT:
                    oversized = True
                    break
                handle.write(chunk)
                size += len(chunk)
            if not oversized and size < RAW_RESULT_BYTE_LIMIT:
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if oversized:
            with temporary.open("wb") as handle:
                handle.write(b'{\n  "error": "result_too_large"\n}\n')
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return not oversized


def _run_pytest(target: Path, scratch: Path) -> int:
    import pytest

    pytest_config = scratch / "pytest.ini"
    pytest_config.write_text("[pytest]\n", encoding="utf-8")
    target_path, separator, selector = str(target).partition("::")
    module_target = ".".join(
        Path(target_path).relative_to(Path.cwd()).with_suffix("").parts
    )
    module_target += separator + selector
    recorder = ResultRecorder()
    returncode = int(
        pytest.main(
            [
                "--pyargs",
                module_target,
                "-c",
                str(pytest_config),
                "--confcutdir",
                str(Path.cwd()),
                "--rootdir",
                str(Path.cwd()),
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(scratch / "pytest-tmp"),
                "--junitxml",
                str(scratch / "results" / "pytest-results.xml"),
                "-o",
                f"log_file={scratch / 'pytest.log'}",
                "-q",
            ],
            plugins=[recorder],
        )
    )
    results_written = _write_json(scratch / "automated-results.json", recorder.results)
    return returncode if results_written else 2


def _run_live(target: Path, scenario: str, scratch: Path) -> int:
    import asyncio

    spec = importlib.util.spec_from_file_location(
        "task23019_supplied_scenarios", target
    )
    if spec is None or spec.loader is None:
        _write_json(scratch / "live-results.json", {"error": "scenario_not_defined"})
        return 2
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scenarios = getattr(module, "SCENARIOS", None)
    selected = scenarios.get(scenario) if isinstance(scenarios, dict) else None
    if selected is None or not asyncio.iscoroutinefunction(selected):
        _write_json(scratch / "live-results.json", {"error": "scenario_not_defined"})
        return 2
    raw_root = scratch / "raw-evidence"
    raw_root.mkdir(parents=True, exist_ok=True)
    os.environ["TASK23019_RAW_ROOT"] = str(raw_root)
    result = asyncio.run(selected())
    if not isinstance(result, dict):
        _write_json(scratch / "live-results.json", {"error": "scenario_result_invalid"})
        return 2
    payload = {scenario: result} if "status" in result else result
    return 0 if _write_json(scratch / "live-results.json", payload) else 2


def _parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--mode", choices=("pytest", "live"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--denied-root", action="append", default=[])
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    global DENIED_ROOTS

    parsed = _parse_arguments(arguments)
    checkout = Path(parsed.checkout).resolve()
    scratch = Path(parsed.scratch).resolve()
    target = Path(parsed.target).resolve()
    DENIED_ROOTS = tuple(_REAL_OS_REALPATH(path) for path in parsed.denied_root)
    _install_boundary(str(checkout), str(scratch))
    if parsed.mode == "pytest":
        return _run_pytest(target, scratch)
    if parsed.scenario is None:
        _write_json(scratch / "live-results.json", {"error": "scenario_not_defined"})
        return 2
    return _run_live(target, parsed.scenario, scratch)


if __name__ == "__main__":
    raise SystemExit(main())
