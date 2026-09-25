"""SSH transport for remote workspace bindings: the ControlMaster lifecycle.

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``,
"ControlMaster lifecycle — explicit, and the executor owns its health". The
rules this module exists to enforce:

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

The mux directory is a short ``0700`` dir under the app-state dir (default
``<state>/cs/``). The ControlPath handed to ssh is ``<dir>/%C`` — OpenSSH
expands ``%C`` itself to a 40-character hash of the connection endpoints,
giving every host key its own socket file in the shared dir while keeping
every argv free of component-derived filenames (a validated host charset
still admits ``..``, so building filenames from locator parts would be a
path traversal). The layout keeps the FULL rendered path under macOS's
104-byte ``sun_path`` limit for realistic state dirs: ``%C`` alone is 40
characters, so the budget is ``len(state_dir) + len("/cs/") + 40 < 104``.

argv construction is always :func:`build_ssh_argv` from the parsed
:class:`RemoteLocator` parts — ``[-p port] [-l user] -- host`` — never from
the raw locator string.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

from loguru import logger

from tldw_chatbook.Tools.remote_binding_locator import (
    RemoteLocator,
    build_ssh_argv,
)

__all__ = [
    "PERCENT_C_EXPANSION_LENGTH",
    "SUN_PATH_LIMIT",
    "SshMasterManager",
    "get_master_manager",
]

#: macOS's ``struct sockaddr_un.sun_path`` budget (including NUL); Linux
#: allows 108 but the shared code must fit the smaller of the two.
SUN_PATH_LIMIT = 104

#: OpenSSH's ``%C`` expands to a SHA-1 hex digest of the connection
#: endpoints — always 40 characters. Used only for budget arithmetic; the
#: expansion itself is ssh's job.
PERCENT_C_EXPANSION_LENGTH = 40

#: Two-character leaf under the app-state dir. Every character here is a
#: character subtracted from the state-dir budget above; see module
#: docstring. Do not lengthen without re-checking the budget.
_CONTROL_DIR_NAME = "cs"

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
_MASTER_MANAGER: "SshMasterManager | None" = None
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

    # -- control dir -----------------------------------------------------

    def control_path_for(self, loc: RemoteLocator) -> Path:
        """Return the 0700 control-socket directory for ``loc`` (created).

        The directory is shared by every host key: per-host discrimination
        is ssh's job (the ``%C`` token in the option value expands per
        connection identity), which also keeps locator components out of
        filesystem paths.

        Args:
            loc: A validated locator (the option value handed to ssh is
                the same for every key; ``loc`` is accepted because the
                directory is per-binding-state, and to keep the API honest
                about who calls it).

        Returns:
            The created directory, mode 0700, under the app-state dir.
        """
        with self._registry_lock:
            if self._control_dir is None:
                base = self._state_dir
                if base is None:
                    from tldw_chatbook.config import get_user_data_dir

                    base = get_user_data_dir()
                directory = base / _CONTROL_DIR_NAME
                _ensure_private_dir(directory)
                self._control_dir = directory
            return self._control_dir

    def _control_path_option(self, loc: RemoteLocator) -> str:
        """The ``ControlPath=<dir>/%C`` option value; ssh expands ``%C``."""
        return f"ControlPath={self.control_path_for(loc).as_posix()}/%C"

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
        disabled there is no shared connection to name, so no
        ``ControlPath``/``ControlMaster`` options at all — per-call direct
        connections.

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
        if self._enabled:
            options += [
                "-o",
                "ControlMaster=no",
                "-o",
                self._control_path_option(loc),
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
            try:
                argv = [
                    self._ssh_bin,
                    *build_ssh_argv(
                        loc,
                        ["-O", "exit", "-o", self._control_path_option(loc)],
                        [],
                    ),
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
        argv = [
            self._ssh_bin,
            *build_ssh_argv(
                loc, ["-O", "check", "-o", self._control_path_option(loc)], []
            ),
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
        before = self._dir_names(control_dir)
        options = [
            "-MNf",
            "-o",
            self._control_path_option(loc),
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
