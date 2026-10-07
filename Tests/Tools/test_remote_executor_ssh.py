"""SSH-mode executor wiring: the fake-ssh integration suite (Phase 2e, Task 14).

Every test drives ``RemoteWorkspaceToolExecutor.for_ssh`` against a fake
``ssh`` (a bash script) whose per-call path runs the REAL committed
bundle — the fake reconstructs the remote command the way real ssh
delivers it: the post-``--`` argv after the host is joined with spaces
into ONE string and executed via ``sh -c`` with stdin passthrough (the
UAT flattening — an unquoted bootstrap is a shell syntax error before
any interpreter starts). The "remote worker" is therefore the true
artifact AND the true argument delivery: bootstrap, decompress,
exec, magic-prefixed frames, two-frame contract, ping identity capture,
and the two-tier watchdog all execute exactly as they would on a host.

Scenario knobs (``FAKE_SSH_MODE``): ``worker`` (default) plays a healthy
remote — it answers the master-lifecycle ops the manager drives and runs
the bundle; ``down`` exits 255 immediately on EVERY invocation, the way
an unreachable host does. Failure variety beyond that (127/76/noise/
mux) is transport-level and stays in Task 11's suite; this suite proves
the EXECUTOR wiring around the transport:

* per-call client guard (BLOCKED/MISSING fail fast without spawning);
* cold-root two-call flow (ping captures identity, then the op — one
  call thereafter);
* result mapping into the status cache (BLOCKED on transport failures,
  STALE_IDENTITY — never BLOCKED — on framed pin refusals, READY on
  success, state untouched on operation timeouts);
* the catastrophic-regex starvation case: the worker's own watchdog
  tiers kill it, the executor buckets ``op_timeout`` with the admitted
  bit, the cache stays READY, and the remote child is gone;
* BLOCKED → direct probe success → next call re-admitted (the probe is
  driven synchronously for determinism; the debounced background
  dispatch is pinned separately);
* master restart after the tracked socket dies;
* the per-host ``max_concurrent_calls`` semaphore (shared across
  executors on one host, not per binding);
* recovery-probe dispatch: fire-and-forget, error-swallowing, debounced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Tools.build_remote_worker_bundle import expected_bundle_stamp
from tldw_chatbook.Tools.remote_binding_locator import (
    CanonicalTarget,
    canonical_fingerprint,
    parse_remote_locator,
)
from tldw_chatbook.Tools.remote_binding_status import (
    BindingState,
    RemoteBindingStatusCache,
)
from tldw_chatbook.Tools.remote_workspace_executor import (
    RemoteWorkspaceExecutionError,
    RemoteWorkspaceToolExecutor,
    _bundle_payload,
    parse_fs_read_stamps,
)
from tldw_chatbook.Tools.remote_workspace_transport import SshMasterManager, TransportFailureKind

#: The fake ssh, generated per test. The per-call path reconstructs
#: ssh's remote-command flattening from its OWN argv (join the
#: post-``--`` elements after the host with spaces, execute via
#: ``sh -c``), so the worker starts only when the transport shell-quoted
#: the bootstrap for the remote login shell — the baked-in
#: ``@@PYTHON@@ -I -c '@@BOOTSTRAP@@'`` exec this script used to perform
#: bypassed argument delivery entirely, which is exactly why every
#: suite stayed green while real servers broke (UAT finding).
_FAKE_SSH_TEMPLATE = r"""#!/bin/bash
# Fake ssh for the Task 14 executor suite: the remote host is this laptop.
log="${FAKE_SSH_LOG:-}"
if [ -n "$log" ]; then
  {
    echo "=== argv ==="
    for arg in "$@"; do
      echo "$arg"
    done
  } >> "$log"
fi

mode="${FAKE_SSH_MODE:-worker}"

if [ "$mode" != "worker" ]; then
  # "down": every invocation (lifecycle or call) dies unreachable.
  exit 255
fi

# -- ssh -G: print the resolved destination, connect nowhere -----------------
for arg in "$@"; do
  if [ "$arg" = "-G" ]; then
    printf 'hostname %s\nport 22\nuser tester\n' "${FAKE_SSH_G_HOST:-fake-host}"
    exit 0
  fi
done

# -- master lifecycle --------------------------------------------------------
prev=""
for arg in "$@"; do
  if [ "$prev" = "-O" ]; then
    case "$arg" in
      check) exit 1 ;;
      exit)  exit 0 ;;
    esac
  fi
  prev="$arg"
done
for arg in "$@"; do
  if [ "$arg" = "-MNf" ]; then
    for opt in "$@"; do
      case "$opt" in
        ControlPath=*)
          sock="${opt#ControlPath=}"
          sock="${sock//%C/cccccccccccccccccccccccccccccccccccc}"
          mkdir -p "$(dirname "$sock")"
          touch "$sock"
          ;;
      esac
    done
    exit 0
  fi
done

# -- per-call: flatten + run the command ssh would have delivered -------------
# Real ssh joins the post-`--` argv after the host with spaces into ONE
# string the remote user's login shell re-parses; this fake joins the
# same elements and hands the string to sh, with stdin passthrough.
after_dd=0
skipped_host=0
remote_args=()
for arg in "$@"; do
  if [ "$after_dd" -eq 0 ]; then
    if [ "$arg" = "--" ]; then
      after_dd=1
    fi
    continue
  fi
  if [ "$skipped_host" -eq 0 ]; then
    skipped_host=1
    continue
  fi
  remote_args+=("$arg")
done
if [ "${#remote_args[@]}" -eq 0 ]; then
  exit 0
fi
remote_cmd="${remote_args[*]}"

if [ -n "${FAKE_SSH_PID_FILE:-}" ]; then
  echo $$ > "$FAKE_SSH_PID_FILE"
fi

bump() {
  delta="$1"
  until mkdir "$counter.lock" 2>/dev/null; do sleep 0.01; done
  cur=$(cat "$counter" 2>/dev/null || echo 0)
  cur=$((cur + delta))
  echo "$cur" > "$counter"
  if [ "$delta" -gt 0 ]; then
    peak=$(cat "$counter.max" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$peak" ]; then
      echo "$cur" > "$counter.max"
    fi
  fi
  rmdir "$counter.lock"
}

counter="${FAKE_SSH_CONCURRENCY_COUNTER:-}"
if [ -n "$counter" ]; then
  bump 1
  if [ -n "${FAKE_SSH_STDIN_CAPTURE:-}" ]; then
    tee "${FAKE_SSH_STDIN_CAPTURE}" | sh -c "$remote_cmd"
  else
    sh -c "$remote_cmd"
  fi
  rc=$?
  bump -1
  exit "$rc"
fi

if [ -n "${FAKE_SSH_STDIN_CAPTURE:-}" ]; then
  tee "${FAKE_SSH_STDIN_CAPTURE}" | sh -c "$remote_cmd"
  exit $?
fi
exec sh -c "$remote_cmd"
"""

_READ_ARGS = {"path": "alpha.txt", "sensitive_exclusions": []}


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha body\n", encoding="utf-8")
    (root / "beta.bin").write_bytes(b"\x00binary\x00")
    return root


def _short_state_dir() -> Path:
    """Unique SHORT state dir: pytest's ``tmp_path`` alone exceeds the
    ControlPath sun_path budget, which would route these managers onto
    the manager's /tmp fallback instead of the primary layout."""
    return Path(f"/tmp/tldw-cs-test-{os.getpid()}-{uuid.uuid4().hex[:8]}")


class FakeSsh:
    """Writes and installs the fake binary; parses its invocation log."""

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _FAKE_SSH_TEMPLATE
        self.bin = tmp_path / "fake-ssh"
        self.bin.write_text(script, encoding="utf-8")
        self.bin.chmod(0o755)
        self.log_path = tmp_path / "ssh-invocations.log"
        self.stdin_capture = tmp_path / "call-stdin.bin"
        self.pid_file = tmp_path / "fake-ssh.pid"
        self.state_dir = _short_state_dir()
        self.counter = tmp_path / "concurrency.count"
        monkeypatch.setenv("FAKE_SSH_LOG", str(self.log_path))
        monkeypatch.setenv("FAKE_SSH_PID_FILE", str(self.pid_file))
        monkeypatch.delenv("FAKE_SSH_MODE", raising=False)
        monkeypatch.delenv("FAKE_SSH_STDIN_CAPTURE", raising=False)
        monkeypatch.delenv("FAKE_SSH_CONCURRENCY_COUNTER", raising=False)

    # -- log parsing -------------------------------------------------------

    def invocations(self) -> list[list[str]]:
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

    def call_invocations(self) -> list[list[str]]:
        """Per-call client invocations (the ones carrying ``-I``)."""
        return [argv for argv in self.invocations() if "-I" in argv]

    # -- scenario knobs ------------------------------------------------------

    def go_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAKE_SSH_MODE", "down")

    def go_alive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAKE_SSH_MODE", "worker")

    def enable_concurrency_counting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAKE_SSH_CONCURRENCY_COUNTER", str(self.counter))

    def max_concurrency(self) -> int:
        peak = self.counter.with_suffix(".count.max")
        if not peak.exists():
            return 0
        return int(peak.read_text(encoding="utf-8").strip() or "0")

    def enable_stdin_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAKE_SSH_STDIN_CAPTURE", str(self.stdin_capture))

    def assert_pid_gone(self) -> None:
        pid = int(self.pid_file.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.fixture()
def env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    # addfinalizer (not yield) so the short /tmp state dir is removed
    # even on failure without disturbing the fixture's return shape.
    _bundle, compressed, _bootstrap = _bundle_payload()
    fake = FakeSsh(tmp_path, monkeypatch)
    workspace = _workspace(tmp_path)
    loc = parse_remote_locator(f"ssh://fake-host{workspace}")
    cache = RemoteBindingStatusCache()
    masters = SshMasterManager(
        ssh_bin=str(fake.bin), state_dir=fake.state_dir
    )

    def make(
        *, binding_id: str = "binding-1", **kwargs: Any
    ) -> RemoteWorkspaceToolExecutor:
        # Explicit cap: the lazy [console_ssh] read drags a full guarded
        # config load, which is not reliable inside an arbitrary test
        # process (same posture as the master-manager suite); the one
        # test that exercises the read stubs the accessor and passes
        # ``max_concurrent_calls=None`` explicitly. The interpreter is
        # the venv python: the fake runs whatever the transport sends,
        # and macOS's /usr/bin/python3 (3.9) would trip the bootstrap's
        # own version gate before any wiring under test executes.
        merged = {
            "cache": cache,
            "masters": masters,
            "recovery_probes": False,
            "max_concurrent_calls": 8,
            "python": sys.executable,
        }
        merged.update(kwargs)
        return RemoteWorkspaceToolExecutor.for_ssh(loc, binding_id, **merged)

    def read(executor: RemoteWorkspaceToolExecutor) -> dict[str, Any]:
        return executor.execute("fs_read", dict(_READ_ARGS), intent="read")

    request.addfinalizer(lambda: shutil.rmtree(fake.state_dir, ignore_errors=True))
    return SimpleNamespace(
        fake=fake,
        workspace=workspace,
        loc=loc,
        cache=cache,
        masters=masters,
        make=make,
        read=read,
        compressed=compressed,
    )


# ---------------------------------------------------------------------------
# happy path: the real bundle through the fake ssh
# ---------------------------------------------------------------------------


def test_for_ssh_fs_read_runs_the_real_bundle_end_to_end(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.fake.enable_stdin_capture(monkeypatch)
    executor = env.make()

    result = env.read(executor)

    assert result["outcome"] == "success"
    body = b"alpha body\n"
    stamps = parse_fs_read_stamps(result["result"] or "")
    assert stamps == (hashlib.sha256(body).hexdigest(), len(body))
    assert "alpha body" in (result["result"] or "")

    # Cold root: exactly one master spawn and TWO transport calls (the
    # identity-capturing ping, then the op).
    assert env.fake.count("-MNf") == 1, env.fake.invocations()
    calls = env.fake.call_invocations()
    assert len(calls) == 2, env.fake.invocations()

    # The request the worker actually consumed: compressed bundle then
    # wire JSON — composed by the executor through the transport.
    captured = env.fake.stdin_capture.read_bytes()
    n = len(env.compressed)
    assert zlib.decompress(captured[:n]) == _bundle_payload()[0]
    request = json.loads(captured[n:])
    assert request["operation"] == "fs_read"
    assert request["timeout_seconds"] > 0

    # The ping captured the identity chain and flipped the cache READY.
    chain = env.cache.identity_for("binding-1")
    assert chain, "ping must capture the identity chain"
    assert chain[0][0] == str(env.workspace.resolve())
    assert env.cache.status("binding-1").state == BindingState.READY


def test_second_execute_is_a_single_transport_call(env: SimpleNamespace) -> None:
    executor = env.make()

    assert env.read(executor)["outcome"] == "success"
    after_first = len(env.fake.call_invocations())
    assert after_first == 2  # ping + op on the cold root

    assert env.read(executor)["outcome"] == "success"
    assert len(env.fake.call_invocations()) == after_first + 1


# ---------------------------------------------------------------------------
# catastrophic regex: the worker's own tiers kill it; cache untouched
# ---------------------------------------------------------------------------


def test_catastrophic_regex_op_timeout_status_unchanged_child_gone(
    env: SimpleNamespace,
) -> None:
    (env.workspace / "poison.txt").write_text("a" * 60000 + "b\n", encoding="utf-8")
    executor = env.make(budget_seconds=1.0, grace=4.0)

    started = time.monotonic()
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.execute(
            "fs_grep",
            {
                "pattern": "(a+)+$",
                "sensitive_exclusions": [],
                "content_exclusions": [],
            },
            intent="read",
        )
    elapsed = time.monotonic() - started

    assert raised.value.code == "op_timeout"
    assert raised.value.admitted is True
    # Either watchdog tier (or the laptop backstop) must have killed the
    # worker long before the outer budget could matter.
    assert elapsed < 10.0, f"regex outran every tier ({elapsed:.2f}s)"

    # Status unchanged: an admitted-marker failure never flips the cache.
    status = env.cache.status("binding-1")
    assert status.state == BindingState.READY
    assert status.identity_chain, "the ping's capture must survive the timeout"
    assert env.cache.transient_failure_count("binding-1") == 1

    # The remote child is gone — no orphaned worker.
    env.fake.assert_pid_gone()


# ---------------------------------------------------------------------------
# host down: BLOCKED, fail-fast guard, probe recovery
# ---------------------------------------------------------------------------


def test_host_down_blocks_guard_fails_fast_and_probe_re_admits(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = env.make()
    env.fake.go_down(monkeypatch)

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(executor)
    assert raised.value.code == "unreachable"
    assert raised.value.admitted is False

    status = env.cache.status("binding-1")
    assert status.state == BindingState.BLOCKED
    assert status.reason == "unreachable or auth failed"

    # Fail fast: the guard refuses WITHOUT spawning anything at all.
    invoked = len(env.fake.invocations())
    with pytest.raises(RemoteWorkspaceExecutionError) as guard:
        env.read(executor)
    assert guard.value.code == "binding_blocked"
    assert guard.value.admitted is False
    assert len(env.fake.invocations()) == invoked

    # The recovery probe (driven synchronously for determinism) flips the
    # binding READY and re-captures identity; the next call is admitted.
    env.fake.go_alive(monkeypatch)
    payload = executor.ping()
    assert payload["bundle_sha256"] == expected_bundle_stamp(
        _bundle_payload()[0]
    )
    assert env.cache.status("binding-1").state == BindingState.READY
    assert env.cache.identity_for("binding-1")

    assert env.read(executor)["outcome"] == "success"


def test_guard_fails_fast_when_missing(env: SimpleNamespace) -> None:
    executor = env.make()
    env.cache.record_missing("binding-1", "probe found the root absent")

    invoked = len(env.fake.invocations())
    with pytest.raises(RemoteWorkspaceExecutionError) as guard:
        env.read(executor)
    assert guard.value.code == "binding_missing"
    assert guard.value.admitted is False
    assert len(env.fake.invocations()) == invoked


# ---------------------------------------------------------------------------
# stale identity: a framed pin refusal is STALE_IDENTITY, never BLOCKED
# ---------------------------------------------------------------------------


def test_stale_identity_pin_failure_marks_stale_not_blocked(
    env: SimpleNamespace,
) -> None:
    executor = env.make()
    assert env.read(executor)["outcome"] == "success"

    # Recreate the root: same path, new inode — the cached chain is stale.
    shutil.rmtree(env.workspace)
    env.workspace.mkdir()
    (env.workspace / "alpha.txt").write_text("replaced body\n", encoding="utf-8")

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(executor)
    assert raised.value.code == "root_pin_failed"
    assert raised.value.admitted is False

    status = env.cache.status("binding-1")
    assert status.state == BindingState.STALE_IDENTITY
    assert status.identity_chain, "stale chain is retained for diagnosis"


# ---------------------------------------------------------------------------
# master restart after the tracked socket dies
# ---------------------------------------------------------------------------


def test_master_restart_after_dead_socket(env: SimpleNamespace) -> None:
    executor = env.make()
    assert env.read(executor)["outcome"] == "success"
    assert env.fake.count("-MNf") == 1

    control_dir = env.fake.state_dir / "cs"
    sockets = list(control_dir.iterdir())
    assert len(sockets) == 1
    sockets[0].unlink()  # the master died and cleaned up after itself

    assert env.read(executor)["outcome"] == "success"
    assert env.fake.count("-MNf") == 2, env.fake.invocations()


# ---------------------------------------------------------------------------
# per-host concurrency cap, shared across executors
# ---------------------------------------------------------------------------


def test_concurrency_capped_per_host_across_executors(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.fake.enable_concurrency_counting(monkeypatch)
    executor_one = env.make(binding_id="binding-1", max_concurrent_calls=2)
    executor_two = env.make(binding_id="binding-2", max_concurrent_calls=2)

    assert env.read(executor_one)["outcome"] == "success"  # warm identity+master

    outcomes: list[str] = []
    lock = threading.Lock()

    def call(executor: RemoteWorkspaceToolExecutor) -> None:
        try:
            result = executor.execute("fs_read", dict(_READ_ARGS), intent="read")
            with lock:
                outcomes.append(result["outcome"])
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            with lock:
                outcomes.append(f"error: {exc!r}")

    threads = [
        threading.Thread(
            target=call, args=(executor_one if index % 2 else executor_two,)
        )
        for index in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)

    assert outcomes == ["success"] * 10, outcomes
    peak = env.fake.max_concurrency()
    assert peak <= 2, f"{peak} concurrent ssh invocations exceeded the cap"
    assert peak >= 2, "the semaphore never actually admitted two calls"


# ---------------------------------------------------------------------------
# recovery-probe dispatch: fire-and-forget, error-swallowing, debounced
# ---------------------------------------------------------------------------


def test_recovery_probe_dispatch_is_fire_and_forget_and_debounced(
    env: SimpleNamespace,
) -> None:
    executor = env.make(recovery_probes=True)
    fired = threading.Event()
    calls: list[int] = []

    def probe() -> dict[str, Any]:
        calls.append(1)
        fired.set()
        raise RemoteWorkspaceExecutionError("unreachable", admitted=False)

    executor.ping = probe  # type: ignore[method-assign]
    executor.maybe_schedule_recovery_probe()
    assert fired.wait(2.0), "the probe never dispatched"

    time.sleep(0.2)
    assert len(calls) == 1, "immediate re-dispatch must be debounced"
    executor.maybe_schedule_recovery_probe()
    time.sleep(0.2)
    assert len(calls) == 1, "the debounce window was not honored"


def test_recovery_probe_rearms_after_the_debounce_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSsh(tmp_path, monkeypatch)
    workspace = _workspace(tmp_path)
    loc = parse_remote_locator(f"ssh://fake-host{workspace}")
    cache = RemoteBindingStatusCache(probe_debounce_s=0.05)
    masters = SshMasterManager(ssh_bin=str(fake.bin), state_dir=fake.state_dir)
    executor = RemoteWorkspaceToolExecutor.for_ssh(
        loc,
        "binding-1",
        cache=cache,
        masters=masters,
        max_concurrent_calls=8,
        recovery_probes=True,
    )

    calls: list[int] = []
    fired = threading.Event()

    def probe() -> dict[str, Any]:
        calls.append(1)
        fired.set()
        return {}

    executor.ping = probe  # type: ignore[method-assign]

    try:
        executor.maybe_schedule_recovery_probe()
        assert fired.wait(2.0)
        time.sleep(0.3)  # past the 0.05s window
        fired.clear()
        executor.maybe_schedule_recovery_probe()
        assert fired.wait(2.0)
        assert len(calls) == 2
    finally:
        shutil.rmtree(fake.state_dir, ignore_errors=True)


def test_for_ssh_reads_max_concurrent_calls_from_config(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook import config as config_module

    monkeypatch.setattr(
        config_module,
        "get_console_ssh_settings",
        lambda: config_module.ConsoleSshSettings(max_concurrent_calls=5),
    )
    executor = env.make(max_concurrent_calls=None)
    assert executor.max_concurrent_calls == 5


# ---------------------------------------------------------------------------
# destination check: a retargeted ssh alias never has its identity adopted
# ---------------------------------------------------------------------------


def _recorded_fingerprint(env: SimpleNamespace) -> str:
    return canonical_fingerprint(
        CanonicalTarget(hostname="fake-host", port=22, user="tester"), env.loc.path
    )


def test_matching_destination_captures_identity(env: SimpleNamespace) -> None:
    executor = env.make(expected_fingerprint=_recorded_fingerprint(env))
    assert env.read(executor)["outcome"] == "success"
    assert env.cache.status("binding-1").state == BindingState.READY


def test_retargeted_alias_blocks_recapture_and_heals_on_revert(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin catches the retarget (new root identity); the re-capture
    that would adopt the new host's identity is refused because ssh -G
    now resolves elsewhere. Reverting the config lets the next probe heal."""
    executor = env.make(expected_fingerprint=_recorded_fingerprint(env))
    assert env.read(executor)["outcome"] == "success"
    calls_before = len(env.fake.call_invocations())

    monkeypatch.setenv("FAKE_SSH_G_HOST", "other-host.example")
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.ping()
    assert raised.value.code == "destination_changed"
    status = env.cache.status("binding-1")
    assert status.state == BindingState.BLOCKED
    assert "different destination" in str(status.reason)
    assert len(env.fake.call_invocations()) == calls_before, (
        "no worker runs against an unverified destination"
    )

    monkeypatch.setenv("FAKE_SSH_G_HOST", "fake-host")
    executor.ping()
    assert env.cache.status("binding-1").state == BindingState.READY


def test_missing_recorded_destination_fails_closed(env: SimpleNamespace) -> None:
    executor = env.make(expected_fingerprint="")
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(executor)
    assert raised.value.code == "destination_changed"
    assert env.fake.call_invocations() == []


# ---------------------------------------------------------------------------
# session worker routing (SSH session worker spec 2026-09-27, Task 7)
# ---------------------------------------------------------------------------
#
# The ``session_spawn`` seam ignores the ssh argv and runs the real
# bootstrap locally (same approach as test_remote_session_worker.py), so
# the real loader, bundle and fork-server answer every session frame. The
# fake ssh still serves ``ssh -G``, the master lifecycle and the one-shot
# path, so ``call_invocations()`` counts one-shot calls only.


@pytest.fixture()
def sessions(monkeypatch: pytest.MonkeyPatch):
    """Fresh session registry, stubbed ``[console_ssh]`` settings, spawn seam."""
    import subprocess

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Tools import remote_session_registry as registry_module
    from tldw_chatbook.Tools.build_remote_worker_bundle import loader_payload
    from tldw_chatbook.Tools.remote_workspace_executor import bootstrap_source

    monkeypatch.setattr(registry_module, "_REGISTRY", None)
    state = SimpleNamespace(
        worker=True, idle=60, spawns=[], key=f"run-{uuid.uuid4().hex[:8]}"
    )
    monkeypatch.setattr(
        config_module,
        "get_console_ssh_settings",
        lambda: config_module.ConsoleSshSettings(
            session_worker=state.worker, session_idle_s=state.idle, bundle_cache=False
        ),
    )

    def spawn(ssh_argv: list[str]) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", bootstrap_source(len(loader_payload()))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        state.spawns.append(proc)
        return proc

    state.spawn = spawn
    yield state
    registry_module.close_all_remote_sessions()


def _session_executor(env: SimpleNamespace, sessions: SimpleNamespace, **kwargs: Any):
    return env.make(session_key=sessions.key, session_spawn=sessions.spawn, **kwargs)


def test_calls_route_through_one_session_per_run_key(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    executor = _session_executor(env, sessions)
    for _ in range(10):
        assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 1
    assert env.fake.call_invocations() == [], "one-shot transport never called"
    assert env.cache.status("binding-1").state == BindingState.READY
    # A second executor for the same run key and binding reuses the session.
    assert env.read(_session_executor(env, sessions))["outcome"] == "success"
    assert len(sessions.spawns) == 1


def test_call_after_run_end_goes_one_shot_without_a_new_session(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    """R12: a closed run key is tombstoned; stragglers never reopen a session."""
    from tldw_chatbook.Tools.remote_session_registry import close_remote_sessions

    executor = _session_executor(env, sessions)
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 1
    close_remote_sessions(sessions.key)
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 1, "no new session after run end"
    assert len(env.fake.call_invocations()) >= 1, "served by one-shot"


def test_kill_switch_uses_one_shot(env: SimpleNamespace, sessions: SimpleNamespace) -> None:
    sessions.worker = False
    executor = _session_executor(env, sessions)
    assert env.read(executor)["outcome"] == "success"
    assert sessions.spawns == []
    assert len(env.fake.call_invocations()) == 2  # ping + op, as before


def test_blocked_binding_probe_uses_one_shot(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind

    executor = _session_executor(env, sessions)
    env.cache.record_transport_failure(
        "binding-1", TransportFailureKind.UNREACHABLE, "down earlier"
    )
    assert env.cache.status("binding-1").state == BindingState.BLOCKED

    executor.ping()

    assert sessions.spawns == []
    assert len(env.fake.call_invocations()) == 1
    assert env.cache.status("binding-1").state == BindingState.READY


def test_retargeted_destination_starts_no_session(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint = _recorded_fingerprint(env)
    # Warm the identity one-shot so the session executor goes straight to
    # a session start (no ping): the check inside the start must refuse.
    assert env.read(env.make(expected_fingerprint=fingerprint))["outcome"] == "success"
    monkeypatch.setenv("FAKE_SSH_G_HOST", "other-host.example")
    executor = _session_executor(env, sessions, expected_fingerprint=fingerprint)

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(executor)

    assert raised.value.code == "destination_changed"
    assert sessions.spawns == []
    assert env.cache.status("binding-1").state == BindingState.BLOCKED


def test_protocol_start_failure_falls_back_for_rest_of_run(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Tools import remote_session_worker as worker_module

    monkeypatch.setattr(worker_module, "expected_bundle_stamp", lambda _data: "0" * 64)
    executor = _session_executor(env, sessions)

    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 1
    assert len(env.fake.call_invocations()) == 2  # ping + op went one-shot

    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 1, "a disabled key never restarts a session"
    assert len(env.fake.call_invocations()) == 3


def _live_session(sessions: SimpleNamespace):
    from tldw_chatbook.Tools.remote_session_registry import get_session_registry

    return get_session_registry()._sessions[(sessions.key, "binding-1")]


def _wait_dead(worker, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while worker.alive and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not worker.alive, "session never ended"


def test_host_idle_exit_gets_a_fresh_session_never_blocked(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R10: the host idling out (exit 0) is a benign end; the next call
    starts a fresh session without spending the run's single restart."""
    from tldw_chatbook.Tools import remote_session_registry as registry_module

    # Simulate the laptop missing its reap so the HOST idle-exit is what ends it.
    monkeypatch.setattr(registry_module.RemoteSessionRegistry, "reap_idle", lambda *a, **k: None)
    sessions.idle = 1  # host idle = 1 + grace
    executor = _session_executor(env, sessions, grace=0.2)
    assert env.read(executor)["outcome"] == "success"
    for expected_spawns in (2, 3):  # twice in one run
        _wait_dead(_live_session(sessions))
        assert env.read(executor)["outcome"] == "success"
        assert len(sessions.spawns) == expected_spawns
        assert env.cache.status("binding-1").state == BindingState.READY
    assert env.fake.call_invocations() == [], "never fell back to one-shot"
    assert env.cache.transient_failure_count("binding-1") == 0


def test_laptop_tells_host_to_idle_out_after_the_laptop_reap(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Tools import remote_session_worker as worker_module

    seen = []
    real_init = worker_module.RemoteSessionWorker.__init__

    def spy(self, *args, **kwargs):
        seen.append(kwargs["idle_s"])
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(worker_module.RemoteSessionWorker, "__init__", spy)
    assert env.read(_session_executor(env, sessions, grace=0.5))["outcome"] == "success"
    assert seen == [60.5]


def test_dead_session_restarts_once_then_falls_back_to_one_shot(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    executor = _session_executor(env, sessions)
    assert env.read(executor)["outcome"] == "success"

    sessions.spawns[-1].kill()  # a genuine death: nonzero exit
    _wait_dead(_live_session(sessions))
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 2  # the one restart
    assert env.fake.call_invocations() == []

    sessions.spawns[-1].kill()
    _wait_dead(_live_session(sessions))
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 2, "no second restart"
    assert len(env.fake.call_invocations()) == 1  # one-shot from here on


def test_transport_start_failure_is_the_calls_failure_no_one_shot(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    def refuse(_argv):
        raise OSError("no ssh")

    sessions.spawn = refuse
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(_session_executor(env, sessions))
    assert raised.value.code == "unreachable"
    assert env.cache.status("binding-1").state == BindingState.BLOCKED
    assert env.fake.call_invocations() == []


def test_transport_start_failure_without_a_typed_failure_is_unreachable(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Tools import remote_session_worker as worker_module

    def start(self):
        raise worker_module.SessionStartError(True, None, "connect timed out")

    monkeypatch.setattr(worker_module.RemoteSessionWorker, "start", start)
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(_session_executor(env, sessions))
    assert raised.value.code == "unreachable"
    status = env.cache.status("binding-1")
    assert status.state == BindingState.BLOCKED
    assert env.fake.call_invocations() == []
    assert sessions.spawns == []


def test_session_start_against_a_silent_host_is_bounded_by_the_call_budget(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    """TASK-33400: the handshake gives up at budget + grace, not after 30 s."""
    import subprocess

    def silent(ssh_argv: list[str]) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            ["sh", "-c", "exec sleep 30"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sessions.spawns.append(proc)
        return proc

    sessions.spawn = silent
    executor = _session_executor(env, sessions, budget_seconds=1.0, grace=0.5)
    started = time.monotonic()
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(executor)
    assert time.monotonic() - started < 8
    assert raised.value.code == TransportFailureKind.UNREACHABLE.value


def test_session_reaped_between_acquire_and_call_is_not_a_tool_error(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-33401: the reaper closing a just-handed-out session costs a new session, not an error."""
    from tldw_chatbook.Tools import remote_session_registry as registry_module

    executor = _session_executor(env, sessions)
    assert env.read(executor)["outcome"] == "success"
    real_acquire = registry_module.RemoteSessionRegistry.acquire
    raced = {"left": 1}

    def racing_acquire(self, key, create):
        worker = real_acquire(self, key, create)
        if worker is not None and raced["left"]:
            raced["left"] -= 1
            self.reap_idle(time.monotonic() + 1e6, 0)  # reaper wins the race
            deadline = time.monotonic() + 5
            while worker.alive and time.monotonic() < deadline:
                time.sleep(0.01)
        return worker

    monkeypatch.setattr(registry_module.RemoteSessionRegistry, "acquire", racing_acquire)
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 2


def test_stale_control_socket_at_start_runs_one_shot_then_session(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    """TASK-33402: no tool error, and the warm path is back on the next call."""
    import subprocess

    real_spawn = sessions.spawn
    stale = {"left": 1}

    def spawn(ssh_argv: list[str]) -> subprocess.Popen[bytes]:
        if stale["left"]:
            stale["left"] -= 1
            proc = subprocess.Popen(
                ["sh", "-c", "echo 'mux_client_hello_exchange: write packet: Broken pipe' >&2; exit 255"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            sessions.spawns.append(proc)
            return proc
        return real_spawn(ssh_argv)

    sessions.spawn = spawn
    executor = _session_executor(env, sessions)
    one_shot_before = len(env.fake.call_invocations())
    assert env.read(executor)["outcome"] == "success"  # one-shot, no error
    assert len(env.fake.call_invocations()) == one_shot_before + 1
    assert env.read(executor)["outcome"] == "success"  # a session again
    assert len(sessions.spawns) == 2
    assert len(env.fake.call_invocations()) == one_shot_before + 1


def test_budget_expiry_mid_upload_is_op_timeout_and_keeps_the_warm_path(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    """TASK-33420: slow cache-miss upload + short budget -> OP_TIMEOUT, not BLOCKED, session next call."""
    import subprocess

    real_spawn = sessions.spawn
    _, compressed, _ = _bundle_payload()
    stall = {"left": 1}
    need_no_read = (
        "import os, sys, time\n"
        "os.read(0, 1 << 20)\n"
        "os.write(1, b'TLDW-REMOTE-0001NEED ' + sys.argv[1].encode() + b'\\n')\n"
        "time.sleep(30)\n"
    )

    def spawn(ssh_argv: list[str]) -> subprocess.Popen[bytes]:
        if stall["left"]:
            stall["left"] -= 1
            proc = subprocess.Popen(
                [sys.executable, "-c", need_no_read, hashlib.sha256(compressed).hexdigest()],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            sessions.spawns.append(proc)
            return proc
        return real_spawn(ssh_argv)

    sessions.spawn = spawn
    executor = _session_executor(env, sessions, budget_seconds=1.0, grace=0.5)
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        env.read(executor)
    assert raised.value.code == TransportFailureKind.OP_TIMEOUT.value
    assert env.cache.status("binding-1").state != BindingState.BLOCKED
    executor = _session_executor(env, sessions)  # a normal budget
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 2  # the key was not disabled: a session again


def test_close_between_register_and_write_reacquires(
    env: SimpleNamespace, sessions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-33421: no tool error; the call gets a fresh session."""
    from tldw_chatbook.Tools import remote_session_registry as registry_module
    from tldw_chatbook.Tools.remote_session_worker import RemoteSessionWorker

    executor = _session_executor(env, sessions)
    assert env.read(executor)["outcome"] == "success"
    real_write = RemoteSessionWorker._write
    raced = {"left": 1}

    def reap_first(self, data, deadline):
        if raced["left"] and self._alive:
            raced["left"] -= 1
            # Exactly what the idle reaper does, made synchronous: pop the
            # session from the registry, then close it -- after this call
            # registered its request and before any REQUEST byte is written.
            registry_module.get_session_registry()._sessions.pop((sessions.key, "binding-1"), None)
            self.close()
        return real_write(self, data, deadline)

    monkeypatch.setattr(RemoteSessionWorker, "_write", reap_first)
    assert env.read(executor)["outcome"] == "success"
    assert len(sessions.spawns) == 2
    registry = registry_module.get_session_registry()
    assert (sessions.key, "binding-1") not in registry._restarted  # a reap costs no restart
