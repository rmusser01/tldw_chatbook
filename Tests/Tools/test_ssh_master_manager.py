"""SshMasterManager: ControlMaster lifecycle tests (Phase 2a, Task 10).

Every subprocess run by these tests is a fake ``ssh`` (a Python script the
test installs) that logs one line per argv token and simulates the three
behaviours the manager depends on:

* ``-MNf`` spawn — creates the control-socket file at the ControlPath it
  was handed (``%C`` substituted with a fixed 40-char token) and exits 0;
* ``-O check`` — exit 0 iff the substituted socket path exists, unless
  ``FAKE_SSH_CHECK_RC`` forces a specific status;
* ``-O exit`` — removes the socket file and exits 0.

Pinned here because the spec makes them load-bearing
("ControlMaster lifecycle — explicit, and the executor owns its health"):
exactly one master under contention, keepalives on the master only,
failure-triggered restart, ``-O exit`` on quit, and the sun_path budget.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_chatbook.Tools import remote_workspace_transport as transport
from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator
from tldw_chatbook.Tools.remote_workspace_transport import (
    PERCENT_C_EXPANSION_LENGTH,
    SUN_PATH_LIMIT,
    SshMasterManager,
)


def _short_state_dir() -> Path:
    """A unique state dir whose primary ``<dir>/cs/%C`` fits sun_path.

    pytest's ``tmp_path`` does NOT qualify (on macOS it is already longer
    than the whole 104-byte budget), so a fixture state dir under it
    would silently exercise the sun_path fallback instead of the primary
    layout these tests pin. Short and unique, removed by the fixture.
    """
    return Path(f"/tmp/tldw-cs-test-{os.getpid()}-{uuid.uuid4().hex[:8]}")


def _fallback_control_dir() -> Path:
    """The production short-dir fallback (mirrors the module template)."""
    return Path(transport._FALLBACK_CONTROL_DIR.format(uid=os.getuid()))

# The fake ssh, verbatim. Written without a module docstring so the outer
# Python triple-quoted string does not need escaping gymnastics.
_FAKE_SSH = """\
#!/usr/bin/env python3
# Fake ssh for SshMasterManager tests; see this test module's docstring.
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
log = os.environ.get("FAKE_SSH_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("=== argv ===\\n")
        for arg in argv:
            fh.write(arg + "\\n")


def control_path():
    for arg in argv:
        if arg.startswith("ControlPath="):
            return arg.split("=", 1)[1].replace("%C", "f" * 40)
    return None


if "-O" in argv:
    index = argv.index("-O")
    action = argv[index + 1] if index + 1 < len(argv) else ""
    if action == "check":
        forced = os.environ.get("FAKE_SSH_CHECK_RC")
        if forced is not None:
            sys.exit(int(forced))
        path = control_path()
        sys.exit(0 if path and Path(path).exists() else 1)
    if action == "exit":
        path = control_path()
        if path and Path(path).exists():
            Path(path).unlink()
        sys.exit(0)

if "-MNf" in argv:
    if os.environ.get("FAKE_SSH_SPAWN_FAIL"):
        sys.exit(255)
    path = control_path()
    if path:
        Path(path).touch()
    sys.exit(0)

sys.exit(0)
"""


class FakeSsh:
    """Installs the fake binary and parses its invocation log."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bin = tmp_path / "fake-ssh"
        self.bin.write_text(_FAKE_SSH, encoding="utf-8")
        self.bin.chmod(0o755)
        self.log_path = tmp_path / "ssh-invocations.log"
        monkeypatch.setenv("FAKE_SSH_LOG", str(self.log_path))
        for stale in ("FAKE_SSH_CHECK_RC", "FAKE_SSH_SPAWN_FAIL"):
            monkeypatch.delenv(stale, raising=False)
        self.state_dir = _short_state_dir()
        # Guard the fixture itself: if this dir ever renders at/past the
        # budget, every "primary path" test below silently tests the
        # fallback instead.
        rendered = (
            len((self.state_dir / "cs").as_posix())
            + 1
            + PERCENT_C_EXPANSION_LENGTH
        )
        assert rendered < SUN_PATH_LIMIT, self.state_dir

    def invocations(self) -> list[list[str]]:
        """One argv list per logged invocation."""
        if not self.log_path.exists():
            return []
        argvs: list[list[str]] = []
        current: list[str] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line == "=== argv ===":
                if current:
                    argvs.append(current)
                current = []
            else:
                current.append(line)
        if current:
            argvs.append(current)
        return argvs

    def count(self, token: str) -> int:
        return sum(argv.count(token) for argv in self.invocations())

    def first_with(self, token: str) -> list[str]:
        for argv in self.invocations():
            if token in argv:
                return argv
        raise AssertionError(f"no invocation carrying {token!r}")

    def socket_path(self, manager: SshMasterManager, loc: Any) -> Path:
        """Where the fake's fixed ``%C`` substitution puts the socket."""
        return Path(manager.control_path_for(loc).as_posix() + "/" + "f" * 40)


_LOCATOR = "ssh://ops@build-box.internal:2222/srv/work/workspace"
_LOC = parse_remote_locator(_LOCATOR)


@pytest.fixture()
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeSsh(tmp_path, monkeypatch)
    yield fake
    shutil.rmtree(fake.state_dir, ignore_errors=True)


def _manager(fake: FakeSsh, **kwargs: Any) -> SshMasterManager:
    defaults: dict[str, Any] = {
        "ssh_bin": str(fake.bin),
        "control_persist": "10m",
        "enabled": True,
        "connect_timeout_s": 3,
        "state_dir": fake.state_dir,
    }
    defaults.update(kwargs)
    return SshMasterManager(**defaults)


def _control_path_value(options: list[str]) -> str:
    values = [v for v in options if v.startswith("ControlPath=")]
    assert len(values) == 1, options
    return values[0]


# ---------------------------------------------------------------------------
# control dir: short, private
# ---------------------------------------------------------------------------


def test_control_dir_is_short_private_and_under_state_dir(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)

    control_dir = manager.control_path_for(_LOC)

    assert control_dir == fake_ssh.state_dir / "cs"
    assert control_dir.exists()
    mode = control_dir.stat().st_mode & 0o777
    assert mode == 0o700, f"control dir mode is {oct(mode)}, expected 0o700"
    # The option value the manager hands ssh carries the %C token; ssh does
    # the expansion itself, so only the rendered LENGTH is ours to budget —
    # pinned separately against realistic state dirs below, because a pytest
    # tmp_path is far longer than any real one.
    rendered = _control_path_value(manager.client_options(_LOC))
    assert rendered.endswith("/%C")


def test_control_dir_parents_are_created_private(fake_ssh: FakeSsh) -> None:
    deep_state = fake_ssh.state_dir / "nested" / "deeper"
    manager = _manager(fake_ssh, state_dir=deep_state)

    manager.control_path_for(_LOC)

    for created in (deep_state, deep_state.parent):
        assert created.exists()
        assert created.stat().st_mode & 0o777 == 0o700


def test_rendered_control_path_fits_sun_path_for_realistic_state_dirs() -> None:
    """The layout budget: <state>/cs/ + 40-char %C must stay under 104 bytes.

    Realistic app-state dirs for this app (profile_paths: ~/.local/share/
    tldw_cli/<user>, fallback ~/.tldw_cli-data/<user>) with plausible
    usernames. This pins the two layout choices the budget depends on: the
    two-character subdir name and the 40-character %C expansion. The
    arithmetic leaves 60 bytes for the whole state dir — anything longer
    (a worktree scratch profile, a deep home) overflows and takes the
    short-dir fallback pinned below, never a broken argv.
    """
    from tldw_chatbook.Tools.remote_workspace_transport import _CONTROL_DIR_NAME

    assert len(_CONTROL_DIR_NAME) == 2
    budget = SUN_PATH_LIMIT - len(f"/{_CONTROL_DIR_NAME}/") - (
        PERCENT_C_EXPANSION_LENGTH + 1  # + NUL
    )
    assert budget == 59

    realistic_states = [
        "/Users/macbook-pro/.local/share/tldw_cli/default_user",
        "/Users/christopher/.local/share/tldw_cli/default_user",
        "/home/christopher/.local/share/tldw_cli/default_user",
        "/Users/macbook-pro/.tldw_cli-data/default_user",
    ]
    for state in realistic_states:
        rendered = f"{state}/{_CONTROL_DIR_NAME}/" + "c" * PERCENT_C_EXPANSION_LENGTH
        assert len(rendered) < SUN_PATH_LIMIT, (
            f"rendered ControlPath {rendered!r} is {len(rendered)} bytes"
        )


# ---------------------------------------------------------------------------
# control dir: sun_path fallback (UAT defect)
# ---------------------------------------------------------------------------


def test_long_state_dir_falls_back_to_short_tmp_dir(
    fake_ssh: FakeSsh, tmp_path: Path
) -> None:
    """UAT defect: a data dir whose ``<dir>/cs/%C`` renders >= 104 bytes
    must never reach ssh — real ssh refuses the socket ("ControlPath too
    long ... >= 104 bytes") for BOTH the master and the per-call direct
    fallback, degrading the whole binding to "unreachable or auth
    failed". The manager steps down to the short ``/tmp/tldw-cs-<uid>``
    dir, visibly (one INFO naming sun_path).
    """
    long_state = tmp_path / ("x" * 64) / ("y" * 24)
    primary = long_state / "cs"
    assert (
        len(primary.as_posix()) + 1 + PERCENT_C_EXPANSION_LENGTH
    ) >= SUN_PATH_LIMIT, "fixture must actually overflow the budget"
    infos: list[str] = []
    handler = logger.add(lambda message: infos.append(str(message)), level="INFO")

    try:
        manager = _manager(fake_ssh, state_dir=long_state)
        control_dir = manager.control_path_for(_LOC)
        manager.control_path_for(_LOC)  # resolution happens once
        # Capture on-disk facts before the cleanup below removes them.
        assert control_dir == _fallback_control_dir()
        assert control_dir.exists()
        mode = control_dir.stat().st_mode & 0o777
        rendered = f"{control_dir.as_posix()}/{'f' * PERCENT_C_EXPANSION_LENGTH}"
    finally:
        logger.remove(handler)
        shutil.rmtree(_fallback_control_dir(), ignore_errors=True)

    assert mode == 0o700, f"fallback dir mode is {oct(mode)}, expected 0o700"
    # The overflowing primary was never created.
    assert not primary.exists()
    # Visible degradation: exactly one INFO naming sun_path.
    assert len([m for m in infos if "sun_path" in m]) == 1, infos
    # And the fallback's rendered socket fits the real %C expansion.
    assert len(rendered) < SUN_PATH_LIMIT


def test_fallback_dir_fits_sun_path_for_realistic_uids() -> None:
    """The fallback template's whole job is bytes: pin the module's
    template under the limit with the real 40-char %C expansion."""
    for uid in (0, 501, 12345, 2**31 - 1, 2**32 - 2):
        fallback = Path(transport._FALLBACK_CONTROL_DIR.format(uid=uid))
        rendered = f"{fallback.as_posix()}/{'c' * PERCENT_C_EXPANSION_LENGTH}"
        assert len(rendered) < SUN_PATH_LIMIT, rendered


def test_pathological_fallback_disables_mux(
    fake_ssh: FakeSsh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When even the fallback cannot fit (pathological), the manager
    behaves exactly as ``enabled=False``: no ControlPath options, no
    spawns, one warning — never an argv ssh would refuse.
    """
    monkeypatch.setattr(
        transport,
        "_FALLBACK_CONTROL_DIR",
        "/tmp/" + "p" * 100 + "-{uid}",
    )
    long_state = tmp_path / ("x" * 64)
    warnings: list[str] = []
    handler = logger.add(
        lambda message: warnings.append(str(message)), level="WARNING"
    )

    try:
        manager = _manager(fake_ssh, state_dir=long_state)
        assert manager.control_path_for(_LOC) is None
        assert manager.control_path_for(_LOC) is None  # warned once, stayed off
        manager.ensure_master(_LOC)
        assert manager.restart_if_dead(_LOC) is False
    finally:
        logger.remove(handler)

    assert fake_ssh.invocations() == []
    assert manager.client_options(_LOC) == [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
    ]
    assert len([w for w in warnings if "sun_path" in w]) == 1, warnings


# ---------------------------------------------------------------------------
# ensure_master: one master under contention
# ---------------------------------------------------------------------------


def test_concurrent_ensure_master_spawns_exactly_once(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)

    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        manager.ensure_master(_LOC)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)

    assert fake_ssh.count("-MNf") == 1, fake_ssh.invocations()
    assert fake_ssh.socket_path(manager, _LOC).exists()


def test_ensure_master_is_idempotent_when_socket_lives(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)

    manager.ensure_master(_LOC)
    manager.ensure_master(_LOC)
    manager.ensure_master(_LOC)

    assert fake_ssh.count("-MNf") == 1


def test_ensure_master_rides_an_out_of_process_master(fake_ssh: FakeSsh) -> None:
    """A ControlPersist master from a previous process is reused, not raced.

    A fresh manager instance has no in-process memory; its one ``-O check``
    finds the socket alive and no ``-MNf`` is spawned.
    """
    first = _manager(fake_ssh)
    first.ensure_master(_LOC)
    assert fake_ssh.count("-MNf") == 1

    fresh = _manager(fake_ssh)
    fresh.ensure_master(_LOC)

    assert fake_ssh.count("-MNf") == 1
    # Exactly two liveness checks: the first manager's initial check (dead,
    # pre-spawn) and the fresh manager's (alive). None after that — health
    # is failure-triggered, never per batch.
    assert fake_ssh.count("-O") == 2, fake_ssh.invocations()


def test_ensure_master_respawns_when_tracked_socket_vanishes(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)
    manager.ensure_master(_LOC)
    fake_ssh.socket_path(manager, _LOC).unlink()

    manager.ensure_master(_LOC)

    assert fake_ssh.count("-MNf") == 2


def test_disabled_manager_never_spawns_and_has_no_control_options(
    fake_ssh: FakeSsh,
) -> None:
    manager = _manager(fake_ssh, enabled=False)

    manager.ensure_master(_LOC)

    assert fake_ssh.invocations() == []
    assert manager.client_options(_LOC) == [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
    ]


# ---------------------------------------------------------------------------
# master argv vs client options
# ---------------------------------------------------------------------------


def test_master_argv_carries_persist_keepalives_and_parsed_parts(
    fake_ssh: FakeSsh,
) -> None:
    manager = _manager(fake_ssh, control_persist="4h")

    manager.ensure_master(_LOC)

    argv = fake_ssh.first_with("-MNf")
    # The fake IS the binary, so the manager's argv[0] never reaches the
    # log; the spy test below pins it instead.
    assert argv[0] == "-MNf"
    assert "-MNf" in argv
    for expected in (
        "ControlPersist=4h",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=2",
        "BatchMode=yes",
        "ConnectTimeout=3",
    ):
        assert expected in argv, argv
    assert any(
        value.startswith("ControlPath=") and value.endswith("/%C")
        for value in argv
    ), argv
    # argv rebuilt from parsed parts: port/user flags, then -- before host.
    assert argv[argv.index("--") + 1:] == ["build-box.internal"], argv
    assert argv[argv.index("-p") + 1] == "2222"
    assert argv[argv.index("-l") + 1] == "ops"
    # Keepalives belong to the master only: exactly one of each.
    assert argv.count("ServerAliveInterval=15") == 1
    assert argv.count("ServerAliveCountMax=2") == 1


def test_client_options_have_no_keepalives_or_persist(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)

    options = manager.client_options(_LOC)

    assert options == [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
        "-o",
        "ControlMaster=no",
        "-o",
        f"ControlPath={manager.control_path_for(_LOC).as_posix()}/%C",
    ]
    assert not any("ServerAlive" in value for value in options)
    assert not any("ControlPersist" in value for value in options)


def test_master_spawn_uses_own_session_and_bounded_timeout(
    fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    real_run = subprocess.run

    def spying_run(argv: list[str], **kwargs: Any) -> Any:
        if "-MNf" in argv:
            captured.update(kwargs)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Tools.remote_workspace_transport.subprocess.run", spying_run
    )
    manager = _manager(fake_ssh)

    manager.ensure_master(_LOC)

    assert captured.get("start_new_session") is True
    assert captured.get("timeout") is not None and captured["timeout"] > 0
    assert captured.get("shell") in (None, False)
    assert captured.get("check") is False


# ---------------------------------------------------------------------------
# restart_if_dead: failure-triggered only
# ---------------------------------------------------------------------------


def test_restart_if_dead_respawns_exactly_once(
    fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_ssh)
    manager.ensure_master(_LOC)
    assert fake_ssh.count("-MNf") == 1

    # Simulate a dead master: -O check reports 1 while the socket file sits
    # stale on disk (forced rc, so the file's presence cannot mask it).
    monkeypatch.setenv("FAKE_SSH_CHECK_RC", "1")
    restarted = manager.restart_if_dead(_LOC)

    assert restarted is True
    assert fake_ssh.count("-MNf") == 2
    assert fake_ssh.socket_path(manager, _LOC).exists()


def test_restart_if_dead_removes_the_stale_socket_file(
    fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_ssh)
    manager.ensure_master(_LOC)
    socket = fake_ssh.socket_path(manager, _LOC)
    assert socket.exists()

    monkeypatch.setenv("FAKE_SSH_CHECK_RC", "1")
    # A failing respawn proves the removal happened independently of the
    # new master recreating the path.
    monkeypatch.setenv("FAKE_SSH_SPAWN_FAIL", "1")

    assert manager.restart_if_dead(_LOC) is True

    assert not socket.exists(), "stale control socket survived the restart"


def test_restart_if_dead_returns_false_when_alive(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)
    manager.ensure_master(_LOC)

    assert manager.restart_if_dead(_LOC) is False
    assert fake_ssh.count("-MNf") == 1


def test_restart_if_dead_is_a_noop_when_disabled(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh, enabled=False)

    assert manager.restart_if_dead(_LOC) is False
    assert fake_ssh.invocations() == []


# ---------------------------------------------------------------------------
# close_all
# ---------------------------------------------------------------------------


def test_close_all_issues_o_exit_and_is_idempotent(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)
    manager.ensure_master(_LOC)
    assert fake_ssh.count("-MNf") == 1

    manager.close_all()
    manager.close_all()

    exits = [argv for argv in fake_ssh.invocations() if "exit" in argv]
    assert len(exits) == 1, fake_ssh.invocations()
    assert exits[0][exits[0].index("-O") + 1] == "exit"
    assert exits[0][exits[0].index("--") + 1:] == ["build-box.internal"]
    assert not fake_ssh.socket_path(manager, _LOC).exists()


def test_close_all_without_masters_is_a_noop(fake_ssh: FakeSsh) -> None:
    manager = _manager(fake_ssh)

    manager.close_all()

    assert fake_ssh.invocations() == []


def test_close_all_tolerates_exit_failures(
    fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(fake_ssh)
    manager.ensure_master(_LOC)
    real_run = subprocess.run

    def failing_exit_run(argv: list[str], **kwargs: Any) -> Any:
        if "-O" in argv and "exit" in argv:
            return subprocess.CompletedProcess(argv, 255)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Tools.remote_workspace_transport.subprocess.run",
        failing_exit_run,
    )

    manager.close_all()  # must not raise


# ---------------------------------------------------------------------------
# module singleton
# ---------------------------------------------------------------------------


def test_get_master_manager_is_a_lazy_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Tools import remote_workspace_transport as transport

    # Stub the config accessor: the singleton behaviour under test is the
    # lazy construction and caching, not the config read (covered by the
    # console_ssh config tests), and a full guarded config load is not
    # reliable inside an arbitrary test process.
    monkeypatch.setattr(
        config_module,
        "get_console_ssh_settings",
        lambda: config_module.ConsoleSshSettings(
            control_persist="31m",
            enable_multiplexing=False,
            connect_timeout_s=9,
        ),
    )
    monkeypatch.setattr(transport, "_MASTER_MANAGER", None)

    first = transport.get_master_manager()

    assert first is transport.get_master_manager()
    assert first._control_persist == "31m"
    assert first._enabled is False
    assert first._connect_timeout_s == 9

    monkeypatch.setattr(transport, "_MASTER_MANAGER", None)
