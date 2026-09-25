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
from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator
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
from tldw_chatbook.Tools.remote_workspace_transport import SshMasterManager

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
