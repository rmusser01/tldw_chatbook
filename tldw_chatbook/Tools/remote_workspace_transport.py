"""SSH transport for remote workspace bindings: ControlMaster lifecycle
plus the per-call client path.

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``,
"Transport & executor" (call path, deadlines, failure taxonomy) and
"ControlMaster lifecycle — explicit, and the executor owns its health".
The rules this module exists to enforce:

- **Explicit masters.** The first need for a host starts one
  ``ssh -MNf`` process (no remote command) under a per-host-key
  ``threading.Lock``; per-call clients ride the shared connection with
  ``ControlMaster=no`` and never become masters themselves.
- **The master is a separate session.** ``start_new_session=True`` puts it
  outside any call's process group, so a deadline kill of one call cannot
  take the shared connection — or other in-flight calls — with it.
- **Keepalives and persistence belong to the master only.**
  ``ControlPersist`` rides the ``-MNf`` command itself, so a master
  orphaned by an app crash self-expires instead of living until the
  network drops; ``ServerAlive*`` runs on the master, where it holds the
  connection up. Per-call clients never run their own keepalives — they
  multiplex a connection the master already keeps alive, where
  ``ServerAlive`` does nothing.
- **Health is failure-triggered.** With ``ControlMaster=no`` a client
  cannot transparently replace a dead master, so the executor detects
  death (a failed mux connection, Task 11) and calls
  :meth:`SshMasterManager.restart_if_dead` — never a proactive
  ``ssh -O check`` per send batch, which would cost a subprocess every
  batch.
- **Shutdown is explicit.** ``close_all()`` sends ``ssh -O exit`` for every
  key this process touched (registered from app quit, ``app.py``), bounded
  per host and failure-tolerant; ``ControlPersist`` remains the crash
  backstop.

The mux directory is a short ``0700`` dir: ``<state>/cs/`` by default,
or — when the state dir is so long that the rendered socket path would
reach ssh's ``sun_path`` budget (a worktree scratch profile does) — a
short ``/tmp/tldw-cs-<uid>`` fallback (INFO-logged once). Real ssh
refuses an oversized ControlPath ("ControlPath too long ... >= 104
bytes") for BOTH the master and the per-call direct fallback, which
would degrade the whole binding to a misleading "unreachable or auth
failed"; the fallback keeps the rendered path small regardless of data
dir length. If even the fallback cannot fit (pathological),
multiplexing is off for the manager's lifetime: per-call direct
connections, exactly as ``enabled=False``. The ControlPath handed to
ssh is ``<dir>/%C`` — OpenSSH expands ``%C`` itself to a 40-character
hash of the connection endpoints, giving every host key its own socket
file in the shared dir while keeping every argv free of
component-derived filenames (a validated host charset still admits
``..``, so building filenames from locator parts would be a path
traversal). The rendered socket path is kept under macOS's 104-byte
``sun_path`` limit: ``%C`` alone is 40 characters, so the primary
budget is ``len(state_dir) + len("/cs/") + 40 < 100`` (margin under
104).

argv construction is always :func:`build_ssh_argv` from the parsed
:class:`RemoteLocator` parts — ``[-p port] [-l user] -- host`` — never from
the raw locator string.
"""

from __future__ import annotations

import enum
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from loguru import logger

from tldw_chatbook.Tools.build_remote_worker_bundle import expected_bundle_stamp
from tldw_chatbook.Tools.remote_binding_locator import (
    RemoteLocator,
    build_ssh_argv,
)
from tldw_chatbook.Tools.remote_worker_bundle import RESPONSE_MAGIC
from tldw_chatbook.Tools.remote_workspace_executor import _bundle_payload

__all__ = [
    "PERCENT_C_EXPANSION_LENGTH",
    "SSH_UNAVAILABLE_MESSAGE",
    "SUN_PATH_LIMIT",
    "RemoteCallResult",
    "RemoteWorkspaceTransport",
    "SshMasterManager",
    "TransportFailure",
    "TransportFailureKind",
    "get_master_manager",
    "ssh_available",
]

#: macOS's ``struct sockaddr_un.sun_path`` budget (including NUL); Linux
#: allows 108 but the shared code must fit the smaller of the two.
SUN_PATH_LIMIT = 104

#: The one feature-off message every user entry point (Settings add-form
#: submit, Console project-instruction picker) surfaces when no ``ssh``
#: client is on PATH — a clear refusal, never a crash (spec
#: "User-facing surfaces", Task 21).
SSH_UNAVAILABLE_MESSAGE = (
    "OpenSSH client (ssh) not found — SSH workspace bindings unavailable"
)


def ssh_available() -> bool:
    """True when an OpenSSH client is on PATH (the feature floor).

    The cheap ``shutil.which`` gate the user entry points consult BEFORE
    any binding is created or offered: without an ``ssh`` binary the
    feature surfaces as disabled with :data:`SSH_UNAVAILABLE_MESSAGE`
    instead of failing later inside a spawn. Availability is checked per
    entry-point use, not cached, so installing/removing the client takes
    effect without a restart.

    Returns:
        True when ``shutil.which("ssh")`` resolves a binary.
    """
    return shutil.which("ssh") is not None

#: OpenSSH's ``%C`` expands to a SHA-1 hex digest of the connection
#: endpoints — always 40 characters. Used only for budget arithmetic; the
#: expansion itself is ssh's job.
PERCENT_C_EXPANSION_LENGTH = 40

#: Two-character leaf under the app-state dir. Every character here is a
#: character subtracted from the state-dir budget above; see module
#: docstring. Do not lengthen without re-checking the budget.
_CONTROL_DIR_NAME = "cs"

#: Rendered socket paths at or beyond this many bytes step down to the
#: fallback (or to no mux at all): macOS rejects at 104 including the
#: NUL, and the margin keeps ssh's own handling of the expansion safely
#: inside the budget. UAT: a worktree-length data dir renders ``>= 104``
#: and real ssh refuses the socket for both master and direct fallback.
_CONTROL_PATH_RENDER_LIMIT = SUN_PATH_LIMIT - 4  # 100

#: Short fallback control dir when the app-state dir cannot fit the
#: budget. Hardcoded short — the whole point is bytes. POSIX-only by
#: construction (mux never runs on Windows; ``enable_multiplexing`` is
#: off there, so this is never formatted).
_FALLBACK_CONTROL_DIR = "/tmp/tldw-cs-{uid}"


def _rendered_socket_length(directory: Path) -> int:
    """Bytes in the socket path ssh renders for ``<directory>/%C``.

    ``%C`` always expands to :data:`PERCENT_C_EXPANSION_LENGTH` hex
    characters; ssh does the expanding, this does the budgeting.
    """
    return len(directory.as_posix()) + 1 + PERCENT_C_EXPANSION_LENGTH

# Keepalive cadence for the master only (spec values).
_SERVER_ALIVE_INTERVAL = 15
_SERVER_ALIVE_COUNT_MAX = 2

# Timeouts (seconds). ``-O check`` / ``-O exit`` talk to a local socket or
# a missing peer: 5s is far beyond any legitimate runtime. The ``-MNf``
# spawn backgrounds after authentication, so connect_timeout plus a small
# grace bounds it.
_CHECK_TIMEOUT_SECONDS = 5.0
_EXIT_TIMEOUT_SECONDS = 5.0
_MASTER_SPAWN_GRACE_SECONDS = 5.0

# After a successful ``-MNf`` (rc 0) the backgrounded process creates the
# socket microseconds later; poll this long for it to appear so the socket
# path can be learned for later staleness removal.
_SOCKET_LEARN_TIMEOUT_SECONDS = 2.0
_SOCKET_LEARN_POLL_SECONDS = 0.025

#: The process-wide manager (see :func:`get_master_manager`).
_MASTER_MANAGER: SshMasterManager | None = None
_MASTER_MANAGER_LOCK = threading.Lock()


def _host_key(loc: RemoteLocator) -> tuple[str | None, str, int | None]:
    """The per-master identity: canonical ``(user, host, port)``."""
    return (loc.user, loc.host, loc.port)


def _ensure_private_dir(directory: Path) -> None:
    """Create ``directory`` (and any missing parents) with mode 0700.

    ``Path.mkdir(mode=...)`` is masked by the process umask, so every
    created level is chmod'ed explicitly — the app-state dir's privacy
    posture must not depend on whatever umask the app was launched under.
    Levels that already exist are left untouched (the state dir owns its
    own permissions).
    """
    missing: list[Path] = []
    probe = directory
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:  # filesystem root cannot be missing
            break
        probe = probe.parent
    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            pass  # concurrent creator; chmod below re-asserts the mode
        os.chmod(path, 0o700)
    if missing:
        os.chmod(directory, 0o700)


def _unlink_quietly(path: Path) -> None:
    """Best-effort removal; a stale socket must never crash its janitor."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - platform-specific races
        logger.debug(f"Could not remove control socket {path}: {exc}")


class SshMasterManager:
    """Owns the ControlMaster lifecycle for remote workspace bindings.

    One instance per process (see :func:`get_master_manager`). Thread-safe:
    every per-host mutation runs under that host key's lock, and the shared
    bookkeeping maps run under a single registry lock. Subprocess calls
    hold the per-host lock only for spawn/check/remove — never across a
    client call.

    Args:
        ssh_bin: Path to (or name of) the ``ssh`` binary.
        control_persist: OpenSSH ``ControlPersist`` duration, passed
            verbatim on the master command (the crash backstop).
        enabled: False disables the whole machinery — no masters, no
            ControlPath; clients connect directly per call.
        connect_timeout_s: Ceiling on the master handshake and per-call
            connects (also sizes the master-spawn subprocess timeout).
        state_dir: App-state directory the control dir is placed under.
            ``None`` resolves lazily to the app's user-data dir, keeping
            the config import off this module's import time.
    """

    def __init__(
        self,
        ssh_bin: str = "ssh",
        *,
        control_persist: str = "10m",
        enabled: bool = True,
        connect_timeout_s: int = 3,
        state_dir: Path | None = None,
    ) -> None:
        self._ssh_bin = ssh_bin
        self._control_persist = control_persist
        self._enabled = enabled
        self._connect_timeout_s = connect_timeout_s
        self._state_dir = state_dir
        self._registry_lock = threading.Lock()
        self._locks: dict[tuple[str | None, str, int | None], threading.Lock] = {}
        #: Host key -> the concrete socket file learned at spawn time.
        self._sockets: dict[
            tuple[str | None, str, int | None], Path
        ] = {}
        #: Host key -> locator, for close_all's ``-O exit`` argv.
        self._registrations: dict[
            tuple[str | None, str, int | None], RemoteLocator
        ] = {}
        self._control_dir: Path | None = None
        #: Terminal degradation: no usable control dir fits sun_path, so
        #: no masters and no ControlPath options for this manager's
        #: lifetime (set once, inside the registry lock).
        self._mux_unusable = False

    @property
    def ssh_bin(self) -> str:
        """The ssh binary every manager op and per-call client executes.

        ``RemoteWorkspaceTransport.call`` spawns through this value so a
        manager (and its tests) inject exactly one binary for the whole
        transport — master lifecycle and calls alike.
        """
        return self._ssh_bin

    # -- control dir -----------------------------------------------------

    def control_path_for(self, loc: RemoteLocator) -> Path | None:
        """Return the 0700 control-socket directory for ``loc`` (created).

        The directory is shared by every host key: per-host discrimination
        is ssh's job (the ``%C`` token in the option value expands per
        connection identity), which also keeps locator components out of
        filesystem paths.

        The primary location is ``<state>/cs/``. When that directory
        renders a socket path at or beyond the sun_path budget (a long
        data dir — e.g. a worktree scratch profile — makes real ssh
        refuse the socket for both the master and the per-call direct
        fallback), a short ``/tmp/tldw-cs-<uid>`` dir is used instead
        (INFO, once). When even the fallback cannot fit (pathological),
        the manager degrades to no-ControlPath behaviour for its
        lifetime (WARNING, once) — per-call direct connections, exactly
        as ``enabled=False``.

        Args:
            loc: A validated locator (the option value handed to ssh is
                the same for every key; ``loc`` is accepted because the
                directory is per-binding-state, and to keep the API honest
                about who calls it).

        Returns:
            The created directory, mode 0700 — or ``None`` when no
            usable directory exists (no mux: callers must omit
            ControlPath options entirely).
        """
        with self._registry_lock:
            if self._control_dir is None and not self._mux_unusable:
                self._control_dir = self._resolve_control_dir()
            return self._control_dir

    def _resolve_control_dir(self) -> Path | None:
        """Pick (and create) the control dir once; ``None`` when none fits.

        Runs under the registry lock. ``/tmp`` on POSIX only — Windows
        never multiplexes, so this path is unreachable there.
        """
        base = self._state_dir
        if base is None:
            from tldw_chatbook.config import get_user_data_dir

            base = get_user_data_dir()
        primary = base / _CONTROL_DIR_NAME
        if _rendered_socket_length(primary) < _CONTROL_PATH_RENDER_LIMIT:
            _ensure_private_dir(primary)
            return primary
        fallback = Path(_FALLBACK_CONTROL_DIR.format(uid=os.getuid()))
        if _rendered_socket_length(fallback) < _CONTROL_PATH_RENDER_LIMIT:
            logger.info(
                f"mux control dir {primary} exceeds the {SUN_PATH_LIMIT}-byte "
                f"sun_path limit once ssh expands %C; "
                f"using short fallback {fallback}"
            )
            _ensure_private_dir(fallback)
            return fallback
        logger.warning(
            f"no usable SSH control dir: both {primary} and {fallback} "
            f"render >= {_CONTROL_PATH_RENDER_LIMIT} bytes against the "
            f"{SUN_PATH_LIMIT}-byte sun_path limit; multiplexing "
            "disabled for this session (per-call direct connections)"
        )
        self._mux_unusable = True
        return None

    def _control_path_option(self, loc: RemoteLocator) -> str | None:
        """The ``ControlPath=<dir>/%C`` option value; ssh expands ``%C``.

        ``None`` when there is no shared connection to name (mux
        disabled, or no usable control dir): callers must omit the
        option entirely rather than hand ssh a socket path it will
        refuse.
        """
        if not self._enabled:
            return None
        directory = self.control_path_for(loc)
        if directory is None:
            return None
        return f"ControlPath={directory.as_posix()}/%C"

    def _lock_for(
        self, key: tuple[str | None, str, int | None]
    ) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    # -- option sets -----------------------------------------------------

    def client_options(self, loc: RemoteLocator) -> list[str]:
        """ssh options for one per-call client of the shared connection.

        Deliberately NO ``ServerAlive*`` and NO ``ControlPersist``: the
        master owns the connection's keepalive and lifetime; a client only
        multiplexes it (see module docstring). When multiplexing is
        disabled — or no usable control dir fits sun_path — there is no
        shared connection to name, so no ``ControlPath``/``ControlMaster``
        options at all: per-call direct connections.

        Args:
            loc: A validated locator.

        Returns:
            Option tokens for :func:`build_ssh_argv`.
        """
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self._connect_timeout_s}",
        ]
        control_path = self._control_path_option(loc)
        if control_path is not None:
            options += [
                "-o",
                "ControlMaster=no",
                "-o",
                control_path,
            ]
        return options

    # -- lifecycle ---------------------------------------------------------

    def ensure_master(self, loc: RemoteLocator) -> None:
        """Guarantee a live master for ``loc``'s host key, or step aside.

        Under the per-host lock: an in-process-tracked socket that still
        exists is assumed healthy (health is failure-triggered — see
        :meth:`restart_if_dead`); an untracked key asks ssh once whether
        an out-of-process master (a previous app run's ControlPersist
        master) already owns the socket and rides it if so; otherwise one
        ``-MNf`` master is spawned in its own session. Spawn failures are
        logged, not raised — the transport surfaces them as typed call
        errors and the next ``ensure_master`` retries.

        No-op when multiplexing is disabled.

        Args:
            loc: A validated locator.
        """
        if not self._enabled:
            return
        key = _host_key(loc)
        with self._lock_for(key):
            try:
                control_dir = self.control_path_for(loc)
            except OSError as exc:
                # An unusable state dir must degrade to direct connections
                # (logged), not raise through the send path.
                logger.warning(
                    f"control dir for {loc.host} is unavailable: {exc!r}"
                )
                return
            if control_dir is None:
                # No dir fits sun_path (logged at resolution): no masters,
                # no ControlPath — per-call direct connections.
                return
            self._registrations.setdefault(key, loc)
            known = self._sockets.get(key)
            if known is not None:
                if known.exists():
                    return
                # Tracked socket vanished (crashed master cleaned up after
                # itself, or a janitor): forget it and spawn fresh.
                with self._registry_lock:
                    self._sockets.pop(key, None)
            elif self._master_alive(loc):
                # A master from outside this process still owns the socket;
                # starting ours would race for the same %C path.
                return
            self._spawn_master(loc, key, control_dir)

    def restart_if_dead(self, loc: RemoteLocator) -> bool:
        """Failure-triggered health path: replace a dead master.

        Runs ``ssh -O check`` (bounded); if the master is gone, removes the
        stale socket file (so the fresh master cannot collide with it) and
        calls :meth:`ensure_master`. Called ONLY after a mux failure —
        never proactively per batch.

        Args:
            loc: A validated locator.

        Returns:
            True iff a restart was attempted (the check said dead).
        """
        if not self._enabled:
            return False
        if self.control_path_for(loc) is None:
            return False  # sun_path-degraded: no mux to restart
        key = _host_key(loc)
        with self._lock_for(key):
            if self._master_alive(loc):
                return False
            known = self._sockets.get(key)
            if known is not None:
                _unlink_quietly(known)
                with self._registry_lock:
                    self._sockets.pop(key, None)
        # Outside the lock: ensure_master re-acquires it and re-derives
        # all state, so a racing thread cannot double-spawn.
        self.ensure_master(loc)
        return True

    def close_all(self) -> None:
        """Send ``ssh -O exit`` for every registered host key; idempotent.

        Registrations are snapshotted and cleared first, so a second call
        (or a concurrent one) issues nothing. Each exit is bounded and its
        failure is logged-and-swallowed: app quit must never hang or fail
        here, and ControlPersist bounds any master this fails to reach.
        """
        with self._registry_lock:
            keys = list(self._registrations)
            self._registrations.clear()
            sockets = {key: self._sockets.pop(key) for key in keys if key in self._sockets}
        for key in keys:
            loc = self._snapshot_locator(key)
            if loc is None:
                continue
            option = self._control_path_option(loc)
            if option is not None:
                try:
                    argv = [
                        self._ssh_bin,
                        *build_ssh_argv(loc, ["-O", "exit", "-o", option], []),
                    ]
                    subprocess.run(
                        argv,
                        capture_output=True,
                        timeout=_EXIT_TIMEOUT_SECONDS,
                        check=False,
                    )
                except (subprocess.TimeoutExpired, OSError) as exc:
                    logger.warning(
                        f"ssh -O exit for {key!r} failed: {exc!r}; "
                        "ControlPersist will expire the master"
                    )
            socket = sockets.get(key)
            if socket is not None:
                _unlink_quietly(socket)

    def _snapshot_locator(
        self, key: tuple[str | None, str, int | None]
    ) -> RemoteLocator | None:
        """Rebuild a minimal locator for ``key`` (close_all's argv source).

        ``build_ssh_argv`` consumes only user/host/port; the path is a
        placeholder that never reaches argv.
        """
        user, host, port = key
        return RemoteLocator(
            user=user, host=host, port=port, ipv6=False, path=PurePosixPath("/")
        )

    # -- subprocess helpers -------------------------------------------------

    def _master_alive(self, loc: RemoteLocator) -> bool:
        """``ssh -O check`` against the locator's control socket.

        Any inability to run ssh, timeout included, reads as dead: the
        caller's next act is a fresh spawn, and a wedged binary must not
        wedge the send path with it.
        """
        option = self._control_path_option(loc)
        if option is None:
            return False  # no shared connection to check
        argv = [
            self._ssh_bin,
            *build_ssh_argv(loc, ["-O", "check", "-o", option], []),
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=_CHECK_TIMEOUT_SECONDS,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug(f"ssh -O check could not run for {loc.host}: {exc!r}")
            return False
        return completed.returncode == 0

    def _spawn_master(
        self,
        loc: RemoteLocator,
        key: tuple[str | None, str, int | None],
        control_dir: Path,
    ) -> None:
        """Run the one ``ssh -MNf`` master and learn its socket path.

        ``-f`` backgrounds ssh after authentication, so this call returns
        once the foreground process exits; ``start_new_session=True``
        detaches the master from this process group (a call's deadline
        kill must not reach it). The concrete socket file is learned by
        diffing the control dir across the spawn — ``%C`` is ssh's
        expansion and is not recomputed locally — and the result feeds
        stale-socket removal later. A spawn that fails or whose socket
        never appears simply leaves the key unlearned: the next
        ``ensure_master`` re-checks and retries.
        """
        option = self._control_path_option(loc)
        if option is None:  # pragma: no cover - caller resolved a live dir
            return
        before = self._dir_names(control_dir)
        options = [
            "-MNf",
            "-o",
            option,
            "-o",
            f"ControlPersist={self._control_persist}",
            "-o",
            f"ServerAliveInterval={_SERVER_ALIVE_INTERVAL}",
            "-o",
            f"ServerAliveCountMax={_SERVER_ALIVE_COUNT_MAX}",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self._connect_timeout_s}",
        ]
        argv = [self._ssh_bin, *build_ssh_argv(loc, options, [])]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=self._connect_timeout_s + _MASTER_SPAWN_GRACE_SECONDS,
                check=False,
                start_new_session=True,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(
                f"ssh -MNf master for {loc.host} could not start: {exc!r}"
            )
            return
        if completed.returncode != 0:
            logger.warning(
                f"ssh -MNf master for {loc.host} exited with status "
                f"{completed.returncode}; calls will fall back to direct "
                "connections until the next ensure"
            )
            return
        deadline = time.monotonic() + _SOCKET_LEARN_TIMEOUT_SECONDS
        learned: Path | None = None
        while True:
            new = self._dir_names(control_dir) - before
            if new:
                # Exactly one new entry is ours; more means another host
                # key spawned concurrently — leave the key unlearned and
                # let the next -O check arbitrate instead of guessing.
                if len(new) == 1:
                    learned = control_dir / next(iter(new))
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(_SOCKET_LEARN_POLL_SECONDS)
        if learned is not None:
            with self._registry_lock:
                self._sockets[key] = learned

    @staticmethod
    def _dir_names(directory: Path) -> set[str]:
        try:
            with os.scandir(directory) as entries:
                return {entry.name for entry in entries}
        except FileNotFoundError:
            return set()


def get_master_manager() -> SshMasterManager:
    """The process-wide manager, constructed lazily from ``[console_ssh]``.

    Lazy construction keeps the config import off the module import path
    (and off any worker-import closure that touches this package) while
    giving ``app.py``'s shutdown hook a one-line reach: close_all on the
    singleton this accessor returns.

    Returns:
        The shared :class:`SshMasterManager`.
    """
    global _MASTER_MANAGER
    with _MASTER_MANAGER_LOCK:
        if _MASTER_MANAGER is None:
            from tldw_chatbook.config import get_console_ssh_settings

            settings = get_console_ssh_settings()
            _MASTER_MANAGER = SshMasterManager(
                control_persist=settings.control_persist,
                enabled=settings.enable_multiplexing,
                connect_timeout_s=settings.connect_timeout_s,
            )
        return _MASTER_MANAGER


# ---------------------------------------------------------------------------
# Per-call client path (Phase 2b)
# ---------------------------------------------------------------------------

#: Leading stdout garbage tolerated before the first response magic
#: (spec: ~4KB; a magicless channel beyond this is unusable). The same
#: bound the loopback harness enforces
#: (``remote_workspace_executor._NOISE_GARBAGE_CAP``).
_NOISE_GARBAGE_CAP = 4 * 1024

#: Hard ceiling on captured stderr per call — enough for the python
#: version line, the watchdog marker, and a traceback; bounded so a
#: chatty remote cannot grow memory (bytes past the cap are drained and
#: discarded, never blocking the child).
_STDERR_CAPTURE_CAP = 64 * 1024

#: Grace added to both laptop-side deadlines, absorbing channel latency
#: and clock drift (spec value; a constructor knob so tests can run
#: deadlines in milliseconds).
_DEFAULT_GRACE_SECONDS = 5.0

#: How long the exchange waits for the ssh process itself to exit after
#: stdout reached a conclusion (final frame or EOF) before killing the
#: group; ssh exits with the channel, so this is a backstop only.
_POST_READ_SETTLE_SECONDS = 2.0

#: Bound on joining the stdin-writer / stderr-drainer threads once the
#: process is settled; both see EOF/EPIPE at process death.
_THREAD_JOIN_SECONDS = 2.0

_READ_CHUNK_BYTES = 65536

#: Cap on a stderr-derived failure reason (the line is remote output).
_REASON_MAX_CHARS = 200

#: stderr fragments that identify a ControlMaster/mux failure (OpenSSH
#: wording variants). A mux failure types the call MUX_ERROR and
#: triggers the failure-triggered master restart — never a retry.
_MUX_ERROR_MARKERS: tuple[bytes, ...] = (
    b"ControlSocket",
    b"mux_client",
    b"mux_protocol",
)

#: stderr marker the worker's graceful watchdog tier writes before
#: ``os._exit(75)``; its presence types a post-admission failure
#: OP_TIMEOUT (one of the classification triggers, Task 12 ruling).
_WATCHDOG_STDERR_MARKER: tuple[bytes, ...] = (b"tldw-worker-watchdog",)

#: The bootstrap's version-gate stderr line: the found version rides the
#: line the taxonomy parses into the PYTHON_TOO_OLD reason.
_PY_VERSION_RE = re.compile(
    rb"tldw-worker:python3\.10\+:found:(\d+(?:\.\d+)+)"
)


class TransportFailureKind(enum.Enum):
    """The transport failure taxonomy (spec: "Failure taxonomy").

    The worker's admitted marker — not exit-code guessing — is the
    primary classifier: kinds above :attr:`OP_TIMEOUT` mean the worker
    never accepted the root (transport/setup class, BLOCKED-eligible for
    the status cache); :attr:`OP_TIMEOUT`/:attr:`REMOTE_OP_FAILED` mean
    the marker arrived and the operation itself failed (typed tool
    error, status unchanged). Task 13/14 consume these names verbatim.
    """

    #: ssh exit 255 with no remote output, or a marker-arrival deadline
    #: kill: unreachable, auth failure, or a stalled handshake.
    UNREACHABLE = "unreachable"
    #: Remote exit 127: the configured interpreter is not on the host.
    INTERPRETER_MISSING = "interpreter_missing"
    #: Remote exit 76: the bootstrap's version gate rejected the python.
    PYTHON_TOO_OLD = "python_too_old"
    #: Any other no-marker exit: the worker never reached framing.
    WORKER_FAILED_TO_START = "worker_failed_to_start"
    #: Admitted, then watchdog exit 75, the watchdog stderr marker, the
    #: completion-deadline kill, or any post-admission death after the
    #: op ran its full budget (the worker's own signal-killing tiers).
    OP_TIMEOUT = "op_timeout"
    #: Admitted, then any other death (signal, lost connection, EOF
    #: without a final frame).
    REMOTE_OP_FAILED = "remote_op_failed"
    #: >4KB of magicless stdout: the channel is unusable.
    STDOUT_NOISE = "stdout_noise"
    #: Mux connection/protocol failure; triggers a master restart for
    #: the NEXT call, never a retry of this one.
    MUX_ERROR = "mux_error"


@dataclass(frozen=True)
class TransportFailure:
    """One typed transport-layer failure.

    Attributes:
        kind: The taxonomy row (see :class:`TransportFailureKind`).
        exit_code: The reaped ssh process's returncode — negative for
            signal deaths (e.g. ``-9`` after the deadline group kill),
            ``None`` only when ssh could not be spawned at all.
        reason: Static, human-readable reason (Task 13 surfaces it);
            may quote one bounded stderr line when a marker line exists
            (python version, watchdog, mux error).
    """

    kind: TransportFailureKind
    exit_code: int | None
    reason: str


@dataclass(frozen=True)
class RemoteCallResult:
    """The outcome of one ``RemoteWorkspaceTransport.call``.

    Attributes:
        admitted: Whether the admitted marker (``root_pinned``) was
            seen — the status cache's transport-vs-operation bit.
        response: The FINAL response frame bytes (magic stripped,
            newline stripped), or ``None`` on transport failure. A
            single worker-refusal frame is a response, not a transport
            failure: the worker framed its own refusal.
        failure: ``None`` when a frame was delivered; otherwise the
            typed transport failure.
    """

    admitted: bool
    response: bytes | None
    failure: TransportFailure | None


class _BoundedCapture:
    """Byte sink capped at ``cap``; extra chunks are drained, not kept."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._chunks: list[bytes] = []
        self._size = 0

    def append(self, chunk: bytes) -> None:
        if self._size >= self._cap:
            return
        kept = chunk[: self._cap - self._size]
        self._chunks.append(kept)
        self._size += len(kept)

    def value(self) -> bytes:
        return b"".join(self._chunks)


@dataclass
class _ExchangeOutcome:
    """What the stdout watcher concluded about one call.

    Attributes:
        terminal_frame: The final response frame, or ``None`` when the
            exchange died before one arrived.
        admitted: Whether the admitted marker was seen.
        admitted_at: ``time.monotonic()`` when the marker parsed — the
            classification clock's anchor (``None`` when no marker).
        killed: Whether the laptop-side deadline group kill fired.
        noise_capped: Whether leading stdout garbage exceeded the cap.
    """

    terminal_frame: bytes | None
    admitted: bool
    admitted_at: float | None
    killed: bool
    noise_capped: bool


def _frame_is_admitted_marker(frame: bytes) -> bool:
    """Whether one parsed frame is the admitted marker.

    Mirrors the two-frame contract ``_parse_worker_output`` enforces:
    the admitted frame is ``outcome="admitted"`` with
    ``code="root_pinned"``. Full frame validation stays with the
    executor layer; the transport only needs the marker bit.
    """
    try:
        payload = json.loads(frame)
    except ValueError:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("outcome") == "admitted"
        and payload.get("code") == "root_pinned"
    )


#: Hex characters of the bundle stamp carried on the per-call audit log
#: line — enough to correlate a call with the committed artifact; the
#: full 64-char stamp rides the wire contract's echo check.
_BUNDLE_STAMP_PREFIX_CHARS = 12


@lru_cache(maxsize=1)
def _bundle_stamp_prefix() -> str:
    """The committed bundle's stamp, prefixed for the audit log line.

    Single source, no recomputation: derived through the same
    :func:`~tldw_chatbook.Tools.build_remote_worker_bundle.expected_bundle_stamp`
    rule every verification site (the loopback harness, the executor's
    ping echo checks) uses, from the ``lru_cache``-held artifact bytes,
    so the audit line can never disagree with what the wire verifies.
    Cached because the committed artifact is fixed for the process's
    life.
    """
    bundle, _compressed, _bootstrap = _bundle_payload()
    return expected_bundle_stamp(bundle)[:_BUNDLE_STAMP_PREFIX_CHARS]


def _request_operation_label(request_bytes: bytes) -> str:
    """Best-effort operation name from one wire request (audit label).

    The transport treats ``request_bytes`` as opaque; this tolerant
    peek exists ONLY for the per-call audit log line and never gates
    behavior — anything malformed or shapeless labels as ``"unknown"``.
    """
    try:
        payload = json.loads(request_bytes)
    except ValueError:
        return "unknown"
    if isinstance(payload, dict):
        operation = payload.get("operation")
        if isinstance(operation, str) and operation:
            return operation
    return "unknown"


def _stderr_marker_line(
    stderr: bytes, markers: tuple[bytes, ...]
) -> str | None:
    """The first bounded stderr line containing any marker, or ``None``."""
    for line in stderr.splitlines():
        if any(marker in line for marker in markers):
            text = line.decode("utf-8", errors="replace").strip()
            return text[:_REASON_MAX_CHARS]
    return None


def _kill_group(proc: subprocess.Popen[bytes]) -> bool:
    """SIGKILL the call's whole process group; True iff we killed it.

    The call ssh was spawned with ``start_new_session=True``, so its
    group is exactly itself (plus its children) — the shared master,
    in its own session, is unreachable from here. Returns False when
    the process already exited, so classification can prefer the real
    exit code over the kill.
    """
    if proc.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return False
    proc.wait()
    return True


def _settle_process(proc: subprocess.Popen[bytes]) -> int:
    """Reap the call ssh, killing the group if it outlives the read."""
    try:
        return proc.wait(timeout=_POST_READ_SETTLE_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return proc.wait()


def _feed_stdin(stream: BinaryIO, payload: bytes) -> None:
    """Write the bundle + request to the call's stdin, then close it.

    The compressed bundle (~64KB) exceeds a pipe buffer, so this runs on
    its own thread: a remote that dies without reading must not
    deadlock the exchange against the stdout read loop.
    """
    try:
        stream.write(payload)
        stream.flush()
    except (BrokenPipeError, OSError):
        pass  # ssh died before reading; classification handles it
    finally:
        try:
            stream.close()
        except (BrokenPipeError, OSError):
            pass


def _drain_stderr(stream: BinaryIO, capture: _BoundedCapture) -> None:
    """Read stderr to EOF, keeping at most ``cap`` bytes.

    Draining continues past the cap so a chatty remote can never block
    its own stderr pipe (which would wedge the call until the deadline
    kill for a reason that has nothing to do with the operation).
    """
    while True:
        chunk = stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            return
        capture.append(chunk)


def _watch_exchange(
    proc: subprocess.Popen[bytes], *, budget: float, grace: float
) -> _ExchangeOutcome:
    """Read the call's stdout to a conclusion under two anchored deadlines.

    Byte-level magic scan first (never line-based): pre-magic bytes are
    garbage under the 4KB cap —STDOUT_NOISE, and a magic straddling a
    chunk boundary is preserved. Frames complete at their newline; the
    FIRST frame is either the admitted marker (``admitted_at`` recorded
    the moment it parses) or a terminal worker refusal; any frame after
    the marker is terminal.

    Deadlines: marker-arrival at spawn + budget + grace; once the
    marker arrives, completion at admitted_at + budget + grace — the
    anchoring that keeps the laptop kill strictly behind the worker's
    own watchdog tiers (spec: "Worker-side watchdog, two tiers").
    """
    magic_len = len(RESPONSE_MAGIC)
    selector = selectors.DefaultSelector()
    stdout_fd = proc.stdout.fileno()  # type: ignore[union-attr]
    selector.register(stdout_fd, selectors.EVENT_READ)
    try:
        buffered = b""
        garbage = bytearray()
        expect_magic = True
        admitted_at: float | None = None
        terminal: bytes | None = None
        killed = False
        noise_capped = False
        marker_deadline = time.monotonic() + budget + grace
        while True:
            progressed = True
            while progressed:
                progressed = False
                if expect_magic:
                    index = buffered.find(RESPONSE_MAGIC)
                    if index != -1:
                        garbage += buffered[:index]
                        buffered = buffered[index + magic_len :]
                        expect_magic = False
                        progressed = True
                    elif len(buffered) > magic_len - 1:
                        # No magic yet: everything except a possible
                        # magic prefix straddling the boundary is noise.
                        cut = len(buffered) - (magic_len - 1)
                        garbage += buffered[:cut]
                        buffered = buffered[cut:]
                else:
                    newline = buffered.find(b"\n")
                    if newline != -1:
                        frame = buffered[:newline]
                        buffered = buffered[newline + 1 :]
                        expect_magic = True
                        progressed = True
                        if admitted_at is None and _frame_is_admitted_marker(
                            frame
                        ):
                            admitted_at = time.monotonic()
                        else:
                            terminal = frame
            # The garbage cap outranks a parsed terminal frame, matching
            # the loopback harness: >4KB of leading noise makes the
            # channel unusable even if frames eventually followed.
            if len(garbage) > _NOISE_GARBAGE_CAP:
                noise_capped = True
                killed = _kill_group(proc)
                break
            if terminal is not None:
                break
            deadline = (
                admitted_at + budget + grace
                if admitted_at is not None
                else marker_deadline
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                killed = _kill_group(proc)
                break
            if not selector.select(remaining):
                continue  # deadline re-checked at the top of the loop
            chunk = os.read(stdout_fd, _READ_CHUNK_BYTES)
            if not chunk:
                break  # EOF: the settle/classify phase takes over
            buffered += chunk
    finally:
        selector.close()
    return _ExchangeOutcome(
        terminal_frame=terminal,
        admitted=admitted_at is not None,
        admitted_at=admitted_at,
        killed=killed,
        noise_capped=noise_capped,
    )


class RemoteWorkspaceTransport:
    """Executes one worker-bundle exchange per ssh call.

    Each :meth:`call` spawns its own ssh client through the manager's
    ``ssh_bin`` and ``client_options`` (BatchMode, ConnectTimeout,
    ``ControlMaster=no`` + shared ControlPath when multiplexing), in its
    OWN session so the deadline group kill can never touch the shared
    master. stdin carries the zlib-compressed bundle (the bootstrap's N
    is exactly that byte count) followed by the request JSON.

    One call, one result — recovery and retries are the executor
    layer's job (Task 13/14), never this class.
    """

    def __init__(
        self,
        master_manager: SshMasterManager,
        *,
        grace_seconds: float = _DEFAULT_GRACE_SECONDS,
    ) -> None:
        """Configure the transport around one master manager.

        Args:
            master_manager: The ControlMaster lifecycle owner; also the
                source of the ssh binary and per-call client options,
                and the target of the failure-triggered mux restart.
            grace_seconds: Slack added to both deadlines (spec default
                5.0); a knob so tests can tighten it.
        """
        self._manager = master_manager
        if grace_seconds <= 0:
            raise ValueError("grace_seconds must be positive")
        self._grace_seconds = float(grace_seconds)

    @property
    def grace_seconds(self) -> float:
        """Deadline slack (seconds) on both anchored deadlines."""
        return self._grace_seconds

    @property
    def master_manager(self) -> SshMasterManager:
        """The master lifecycle this transport's calls multiplex."""
        return self._manager

    # -- public surface ----------------------------------------------------

    def call(
        self,
        loc: RemoteLocator,
        request_bytes: bytes,
        *,
        budget: float,
        python: str = "python3",
    ) -> RemoteCallResult:
        """Run one bundle exchange over ssh; return its typed outcome.

        Args:
            loc: A validated locator; argv is rebuilt from its parsed
                parts.
            request_bytes: The wire request JSON appended to the
                compressed bundle on the call's stdin.
            budget: Operation budget (seconds) anchoring both
                deadlines; the same value the executor sends as the
                request's remaining budget.
            python: Remote interpreter invoked as
                ``<python> -I -c <bootstrap>`` (binding-configured).

        Returns:
            A :class:`RemoteCallResult` — never an exception for a
            failed exchange (unexpected local errors type as
            WORKER_FAILED_TO_START with the exception's class name in
            the reason, after killing and reaping the ssh process);
            spawn/typing problems raise ``TypeError``/``ValueError``
            before any subprocess exists.
        """
        if not isinstance(request_bytes, bytes):
            raise TypeError(
                f"request_bytes must be bytes, got {type(request_bytes).__name__}"
            )
        if budget <= 0:
            raise ValueError("budget must be positive")
        if python.startswith("-") or any(char.isspace() for char in python):
            raise ValueError(
                f"python must be a bare interpreter name: {python!r}"
            )
        _bundle, compressed, bootstrap = _bundle_payload()
        # Spec ("Transport & executor"): each call logs the bundle hash
        # for audit — one debug line at spawn, endpoint label + stamp
        # prefix + op name only, never request content. Placed before
        # the spawn so an OSError from Popen is still accounted for.
        logger.debug(
            f"remote workspace call host_key={_host_key(loc)!r} "
            f"bundle={_bundle_stamp_prefix()} "
            f"op={_request_operation_label(request_bytes)}"
        )
        argv = [
            self._manager.ssh_bin,
            *self._manager.client_options(loc),
            *build_ssh_argv(loc, [], [python, "-I", "-c", bootstrap]),
        ]
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return RemoteCallResult(
                admitted=False,
                response=None,
                failure=TransportFailure(
                    TransportFailureKind.UNREACHABLE,
                    None,
                    f"ssh could not be run: {exc}",
                ),
            )
        stderr_capture = _BoundedCapture(_STDERR_CAPTURE_CAP)
        stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(proc.stderr, stderr_capture),
            daemon=True,
            name=f"ssh-call-stderr-{loc.host}",
        )
        stdin_thread = threading.Thread(
            target=_feed_stdin,
            args=(proc.stdin, compressed + request_bytes),
            daemon=True,
            name=f"ssh-call-stdin-{loc.host}",
        )
        stderr_thread.start()
        stdin_thread.start()

        outcome: _ExchangeOutcome | None = None
        exit_code: int | None = None
        local_error: Exception | None = None
        try:
            outcome = _watch_exchange(
                proc, budget=budget, grace=self._grace_seconds
            )
            exit_code = _settle_process(proc)
        except Exception as exc:  # noqa: BLE001 - the call contract is typed
            # Any local failure (selectors/os.read under fd pressure, a
            # parse error ValueError does not cover) is a typed result,
            # never an escaping exception.
            local_error = exc
        finally:
            # Every path reaps: a local error must never orphan the ssh
            # process past this call (its group is its own session).
            if proc.poll() is None:
                _kill_group(proc)
            if exit_code is None:
                exit_code = proc.wait()
            stdin_thread.join(timeout=_THREAD_JOIN_SECONDS)
            stderr_thread.join(timeout=_THREAD_JOIN_SECONDS)
        stderr_bytes = stderr_capture.value()

        if local_error is not None:
            return RemoteCallResult(
                admitted=False,
                response=None,
                failure=TransportFailure(
                    TransportFailureKind.WORKER_FAILED_TO_START,
                    exit_code,
                    f"local transport error: {type(local_error).__name__}",
                ),
            )
        assert outcome is not None  # only reachable when no local error
        if outcome.terminal_frame is not None:
            return RemoteCallResult(
                admitted=outcome.admitted,
                response=outcome.terminal_frame,
                failure=None,
            )
        failure = self._classify_failure(
            loc,
            exit_code=exit_code,
            admitted=outcome.admitted,
            admitted_at=outcome.admitted_at,
            budget=budget,
            killed=outcome.killed,
            noise_capped=outcome.noise_capped,
            stderr=stderr_bytes,
        )
        return RemoteCallResult(
            admitted=outcome.admitted, response=None, failure=failure
        )

    # -- taxonomy ----------------------------------------------------------

    def _classify_failure(
        self,
        loc: RemoteLocator,
        *,
        exit_code: int,
        admitted: bool,
        admitted_at: float | None,
        budget: float,
        killed: bool,
        noise_capped: bool,
        stderr: bytes,
    ) -> TransportFailure:
        """Bucket one failed exchange (marker primary, exit code secondary).

        Order mirrors the spec's taxonomy: the garbage cap and the
        admitted marker outrank exit codes; within no-marker exits the
        specific codes (255 mux-checked, 127, 76) precede the
        WORKER_FAILED_TO_START catch-all.

        Within admitted failures, OP_TIMEOUT fires on any of: the
        reserved watchdog exit 75; the watchdog stderr marker; our
        completion-deadline kill; or the CLOCK rule — the op ran its
        full budget (``monotonic_now - admitted_at >= budget``). The
        clock rule exists because the worker's own alarm/RLIMIT_CPU
        tiers (Task 12) kill the remote python mid-op by SIGNAL, and
        ssh deliberately does not decode remote signal deaths (they
        surface as a bare non-zero exit): an admitted op that dies on
        its own only after consuming its whole budget is a timeout by
        definition, whichever tier fired. Fast post-admission deaths
        (elapsed < budget, no marker) stay REMOTE_OP_FAILED — a mid-op
        process/network death is an operation failure, not a timeout.
        """
        if noise_capped:
            return TransportFailure(
                TransportFailureKind.STDOUT_NOISE,
                exit_code,
                "remote shell emits stdout noise before the response magic",
            )
        if admitted:
            watchdog_marker = _stderr_marker_line(stderr, _WATCHDOG_STDERR_MARKER)
            ran_full_budget = (
                admitted_at is not None
                and time.monotonic() - admitted_at >= budget
            )
            if killed or exit_code == 75 or watchdog_marker is not None or ran_full_budget:
                return TransportFailure(
                    TransportFailureKind.OP_TIMEOUT,
                    exit_code,
                    "operation timed out",
                )
            return TransportFailure(
                TransportFailureKind.REMOTE_OP_FAILED, exit_code, "remote op failed"
            )
        if killed:
            # No marker + our deadline kill: a stalled handshake is
            # transport-class, never OP_TIMEOUT (the worker never
            # accepted the root, so no operation ever started).
            return TransportFailure(
                TransportFailureKind.UNREACHABLE,
                exit_code,
                "handshake stalled",
            )
        if exit_code == 255:
            mux_line = _stderr_marker_line(stderr, _MUX_ERROR_MARKERS)
            if mux_line is not None:
                self._restart_master_after_mux_failure(loc)
                return TransportFailure(
                    TransportFailureKind.MUX_ERROR, exit_code, mux_line
                )
            return TransportFailure(
                TransportFailureKind.UNREACHABLE,
                exit_code,
                "unreachable or auth failed",
            )
        if exit_code == 127:
            return TransportFailure(
                TransportFailureKind.INTERPRETER_MISSING,
                exit_code,
                "host lacks python3",
            )
        if exit_code == 76:
            match = _PY_VERSION_RE.search(stderr)
            reason = (
                f"python ≥ 3.10 required (found {match.group(1).decode()})"
                if match
                else "python ≥ 3.10 required"
            )
            return TransportFailure(
                TransportFailureKind.PYTHON_TOO_OLD, exit_code, reason
            )
        return TransportFailure(
            TransportFailureKind.WORKER_FAILED_TO_START,
            exit_code,
            "remote worker failed to start",
        )

    def _restart_master_after_mux_failure(self, loc: RemoteLocator) -> None:
        """Replace a dead master for the NEXT call; never retry this one.

        Fire-and-forget by contract (spec: "ControlMaster lifecycle"):
        the failed call already returned its typed error — this runs the
        failure-triggered ``restart_if_dead`` so subsequent calls find a
        live master. Bounded inside the manager; any failure is logged,
        never raised into the call path.
        """
        try:
            restarted = self._manager.restart_if_dead(loc)
            logger.info(
                f"mux failure on {loc.host}: master restart attempted={restarted}"
            )
        except Exception as exc:  # noqa: BLE001 - janitor must not break the call
            logger.warning(f"mux master restart for {loc.host} failed: {exc!r}")
