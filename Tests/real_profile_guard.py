"""Refuse every write to the real user's tldw profile while pytest runs.

TASK-33665. On 2026-10-02 a test-built app ran while the real HOME was in
effect. It rewrote the owner's ``~/.config/tldw_cli/ui_state.toml`` and opened
their library-collections database. The suite's isolation was environment
redirection alone (HOME, TLDW_CONFIG_PATH), so anything that ran with the real
values (a session-end restore, an early import that froze the config path, a
late thread) reached the real profile and nothing stopped it.

The first line of defence is now that ``Tests/conftest.py`` never points the
environment back at the real profile, not even after the session. This guard is
the backstop, and it checks the PATH, not the environment. A
``sys.addaudithook`` hook sees write-mode opens, SQLite connects and the path
events in ``_PATH_EVENTS`` (delete, rename, link, mkdir, rmtree, chmod and the
other metadata writes) in this process. It refuses any whose path falls under
the real user's ``~/.config/tldw_cli`` or ``~/.local/share/tldw_cli``, by
abspath or by realpath, and any destructive tree operation on a directory that
holds them (``shutil.rmtree(~/.config)``). Paths relative to a directory fd are
resolved through the fd. The real home comes from ``TLDW_TEST_REAL_HOME``,
which the first process to install the guard exports so xdist workers and
pytest children inherit it, then from the password database, never from the
sandboxed ``HOME``.

Each refusal raises at the call site and is recorded with the writing thread
and stack, so a writer that swallows the exception still fails its test (the
autouse fixture in ``Tests/conftest.py``) or the run (``session_end_check``,
which carries an xdist worker's refusals to the controller). A refusal after
the session-end check is printed to stderr at exit. Audit hooks cannot be
removed, so the guard also covers threads that outlive the session.

Known limits:

- An ``open`` relative to a directory fd (``os.open(name, dir_fd=fd)``) raises
  an audit event that carries only the leaf name, so the guard cannot place
  it; the stray file it creates is the leftover. Not restoring the
  environment is what keeps such writers out of the real profile.
- Only processes that load these conftests are guarded. A Python subprocess
  that is not pytest is not.
- Audit hooks see only the audited Python calls. Writes made in C (SQLite's
  own ``-wal``/``-journal`` files, extension modules) and SQLite
  ``ATTACH``/``VACUUM INTO`` targets, which never pass ``sqlite3.connect``,
  are invisible.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

REAL_HOME_ENV = "TLDW_TEST_REAL_HOME"
WORKEROUTPUT_KEY = "tldw_real_profile_refusals"
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# event -> ((path arg, dir-fd arg or None, destructive tree op), ...). The
# positions were checked against CPython 3.12 with a live probe; os.replace
# raises "os.rename", os.unlink raises "os.remove".
_PATH_EVENTS = {
    "os.remove": ((0, 1, True),),
    "os.rmdir": ((0, 1, True),),
    "os.mkdir": ((0, 2, False),),
    "os.rename": ((0, 2, True), (1, 3, False)),
    "os.link": ((0, 2, False), (1, 3, False)),
    "os.symlink": ((1, 2, False),),
    "os.truncate": ((0, None, False),),
    "os.chmod": ((0, 2, False),),
    "os.utime": ((0, 3, False),),
    "os.chown": ((0, 3, False),),
    "os.chflags": ((0, None, False),),
    "os.setxattr": ((0, None, False),),
    "os.removexattr": ((0, None, False),),
    "shutil.rmtree": ((0, 1, True),),
    "shutil.move": ((0, None, True), (1, None, False)),
    "shutil.copyfile": ((1, None, False),),
    "sqlite3.connect": ((0, None, False),),
}
_violations: list[str] = []
_worker_violations: list[str] = []
_lock = threading.Lock()
_roots: tuple[str, ...] = ()


class RealProfileWriteError(PermissionError):
    """A test tried to write the real user's tldw profile."""


def _real_home() -> Path | None:
    exported = os.environ.get(REAL_HOME_ENV)
    if exported:
        return Path(exported)
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):  # Windows: no pwd module.
        home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
        return Path(home) if home else None


def _norm(path: str) -> str:
    # normcase folds case only on Windows; APFS is case-insensitive as well.
    path = os.path.normcase(path)
    return path.lower() if sys.platform == "darwin" else path


def protected_roots() -> tuple[str, ...]:
    """Return the real-profile directories the guard protects.

    Returns:
        Absolute, case-normalised paths of the real user's tldw config and
        data directories, or an empty tuple when no real home is known.
    """
    home = _real_home()
    if home is None:
        return ()
    roots = {home / ".config" / "tldw_cli", home / ".local" / "share" / "tldw_cli"}
    expanded = roots | {Path(os.path.realpath(root)) for root in roots}
    return tuple(sorted(_norm(os.path.abspath(root)) for root in expanded))


def _fd_path(fd: int) -> str | None:
    try:
        if sys.platform == "darwin":
            import fcntl

            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
            return os.fsdecode(raw.split(b"\0", 1)[0])
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, ValueError):
        return None


def _as_path(arg: object, dir_fd: object) -> str | None:
    if arg is None or isinstance(arg, int):
        return None
    try:
        raw = os.fsdecode(arg)
    except (TypeError, ValueError):
        return None
    if raw.startswith("file:"):  # SQLite URI: check the path it names.
        raw = unquote(urlparse(raw).path)
    if not raw or raw.startswith(":memory:"):
        return None
    if isinstance(dir_fd, int) and dir_fd >= 0 and not os.path.isabs(raw):
        base = _fd_path(dir_fd)
        if base is None:
            return None
        raw = os.path.join(base, raw)
    return raw


def _protected(arg: object, dir_fd: object = None, tree: bool = False) -> str | None:
    raw = _as_path(arg, dir_fd)
    if raw is None:
        return None
    try:  # a deleted cwd or a NUL byte fails the call itself, not this hook
        candidates = (os.path.abspath(raw), os.path.realpath(raw))
    except (OSError, ValueError):
        return None
    for candidate in candidates:
        text = _norm(candidate)
        for root in _roots:
            if text == root or text.startswith(root + os.sep):
                return candidate
            if tree and root.startswith(text.rstrip(os.sep) + os.sep):
                return candidate
    return None


def _is_write_open(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _refuse(event: str, path: str) -> None:
    thread = threading.current_thread().name
    writer = "".join(traceback.format_stack(limit=12)[:-2])
    message = f"{event} on the real profile path {path!r} in thread {thread!r}"
    with _lock:
        _violations.append(f"{message}\n{writer}")
    raise RealProfileWriteError(f"TASK-33665 guard: {message} during pytest")


def _hook(event: str, args: tuple) -> None:
    if event == "open":
        if len(args) >= 3 and _is_write_open(args[1], args[2]):
            path = _protected(args[0])
            if path is not None:
                _refuse(event, path)
        return
    specs = _PATH_EVENTS.get(event)
    if specs is None:
        return
    for position, fd_position, tree in specs:
        if position >= len(args):
            continue
        dir_fd = None
        if fd_position is not None and fd_position < len(args):
            dir_fd = args[fd_position]
        path = _protected(args[position], dir_fd, tree)
        if path is not None:
            _refuse(event, path)


def _report_late_refusals() -> None:
    late = take_violations()
    if late:
        print(
            "\nTASK-33665: writes to the real tldw profile were refused after "
            "the session-end check:\n" + "\n".join(late),
            file=sys.__stderr__,
        )


def install() -> None:
    """Install the guard once per process; later calls do nothing."""
    global _roots
    if _roots:
        return
    home = _real_home()
    if home is None:
        return
    os.environ.setdefault(REAL_HOME_ENV, str(home))
    _roots = protected_roots()
    sys.addaudithook(_hook)
    atexit.register(_report_late_refusals)


def take_violations() -> list[str]:
    """Return and clear the refusals recorded since the last call.

    Returns:
        One message per refused operation, each with the writer's stack.
    """
    with _lock:
        taken = list(_violations)
        _violations.clear()
    return taken


def session_end_check(session) -> None:
    """Fail the run on any refusal that no test claimed.

    An xdist worker's exit status and stdout are discarded, so a worker hands
    its refusals to the controller through ``workeroutput`` instead; the
    controller's ``pytest_testnodedown`` passes them to
    ``collect_worker_refusals``.

    Args:
        session: The ending ``pytest.Session``.
    """
    refused = take_violations()
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput.setdefault(WORKEROUTPUT_KEY, []).extend(refused)
        return
    with _lock:
        refused += _worker_violations
        _worker_violations.clear()
    if refused:
        print("\nTASK-33665: writes to the real tldw profile were refused:\n")
        print("\n".join(refused))
        if not session.exitstatus:
            session.exitstatus = 1


def collect_worker_refusals(node) -> None:
    """Keep the refusals an xdist worker reported at its session end.

    Args:
        node: The finished worker's controller node.
    """
    refused = getattr(node, "workeroutput", None) or {}
    with _lock:
        _worker_violations.extend(refused.get(WORKEROUTPUT_KEY, ()))
