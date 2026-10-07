# SSH Session Worker Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Burn down the nine follow-ups PR #2879's final review deferred (ruling R14), TASK-33400–33408, in one PR against `dev`.

**Architecture:** Each fix is local to the session layer PR #2879 added. The registry (`remote_session_registry.py`) gets failure fan-out, an off-thread parallel close and a mux-failure policy. The laptop worker (`remote_session_worker.py`) gets a budget-bounded handshake, clean start cleanup, a tail stderr capture and a "closed before send" signal. The host fork-server (`remote_session_serve.py`, bundled) answers a failed fork per request and, only if measurement pays, coalesces a fast op's writes. The master manager gets a shutdown latch, and ADR-181 gets the policy rows and the inode-reuse note.

**Tech Stack:** Python ≥3.12 laptop, stdlib-only bundle for the host (Python ≥3.10), pytest, the real stage-1 loader/bundle driven locally through the `spawn` seam, opt-in live UAT against `ml-user@192.168.5.84`.

**Spec:** `Docs/superpowers/specs/2026-09-27-ssh-session-worker-and-bundle-cache-design.md` and `backlog/decisions/181-ssh-remote-workspace-bindings.md` ("Amendment 2026-09-27"). Each task's backlog file, `backlog/tasks/task-3340N - *.md`, holds its acceptance criteria.

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/ssh-session-followups`, branch `feat/ssh-session-followups`. Work only there: check `pwd` before every edit. Do not push. The controller pushes.
- Python: `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (the worktree has no venv). Run tests as `$PY -m pytest …`.
- **Never write the user's real config.** No test or probe may call an unmocked config writer. When a probe needs config, set `TLDW_CONFIG_PATH` to a scratch file. Never bypass `Tests/private_profile.py` or any isolation shim.
- **Bundled modules** (`remote_session_serve.py`, `remote_session_frames.py`, and everything in `BUNDLE_MODULES`) must stay stdlib-only and Python-3.10 compatible. After editing any of them, run `$PY -m tldw_chatbook.Tools.build_remote_worker_bundle` and `$PY -m tldw_chatbook.Tools.build_remote_worker_bundle --check`, and commit the regenerated `remote_worker_bundle.py`.
- **Status-cache rules (ADR-181):**
  - Only BLOCKING transport kinds flip a binding to BLOCKED.
  - A live session proves reachability, so no failure on a live session may flip to BLOCKED.
  - `MUX_ERROR`, `OP_TIMEOUT` and `REMOTE_OP_FAILED` never flip state.
- **Logs:** static text only. Never a path, ControlPath or stderr line. Every new `logger.*` line must appear in the production diagnostic inventory (Task 7 regenerates it after reading the drift rows).
- **Boot ratchet (ADR-097):** add no module-level imports to modules loaded at UI-ready. The session modules are imported lazily from the SSH path; keep it that way.
- **Known local noise, do not chase:**
  - `Tests/Chat/test_console_chat_controller.py` fails locally with `RecoveryRequired: raw_source_selection_changed`. It fails on clean `origin/dev` too.
  - `Tests/Architecture` needs `-p no:xdist`.
- One commit per task, with a conventional message ending in the `Co-Authored-By` trailer the controller gives you.

## Review Focus

- **A dead or blackholed host hit by several concurrent calls** (parallel sub-agents). Every call should fail within about one connect or handshake timeout, not k of them in series. Pinned in Task 1 (`test_waiters_share_one_transport_start_failure`).
- **A short tool budget on a slow link.** The session start must give up at budget + grace and classify like a one-shot call (UNREACHABLE), never hang 30 s. Pinned in Task 1 (`test_session_start_against_a_silent_host_is_bounded_by_the_call_budget`).
- **First call after laptop sleep/resume** (stale ControlMaster socket). No tool error, and the warm path is back on the next call. Pinned in Task 2 (`test_stale_control_socket_at_start_runs_one_shot_then_session`).
- **Quitting with a wedged host.** App exit waits a bounded time, and no new master outlives the app. Pinned in Task 3 (`test_close_all_closes_in_parallel_and_is_bounded`) and Task 4 (`test_no_master_is_spawned_after_close_all`).
- **A host at its process limit.** One request fails with a status-preserving error; the other requests and the session carry on. Pinned in Task 5 (`test_failed_spawn_fails_only_that_request`).

## File Structure

| File | Change | Tasks |
|---|---|---|
| `tldw_chatbook/Tools/remote_session_registry.py` | failure fan-out, mux policy, off-thread parallel close | 1, 2, 3 |
| `tldw_chatbook/Tools/remote_session_worker.py` | `handshake_timeout`, start cleanup, `_TailCapture`, `SessionClosed`, spawn-failure mapping | 1, 2, 3, 5 |
| `tldw_chatbook/Tools/remote_workspace_executor.py` | pass `handshake_timeout`; re-acquire on `SessionClosed` | 1, 3 |
| `tldw_chatbook/Tools/remote_workspace_transport.py` | `SshMasterManager` shutdown latch | 4 |
| `tldw_chatbook/Tools/remote_session_frames.py` | `HOST_SPAWN_FAILED = 71` (bundled) | 5 |
| `tldw_chatbook/Tools/remote_session_serve.py` | per-request spawn failure; optional coalescing (bundled) | 5, 6 |
| `tldw_chatbook/Tools/remote_worker_bundle.py` | regenerated | 5, 6 |
| `Tests/conftest.py` | reset the master-manager singleton too | 4 |
| `Tests/Tools/test_remote_session_{registry,worker,serve}.py`, `test_remote_executor_ssh.py`, `test_ssh_master_manager.py` | new pins | 1–6 |
| `backlog/decisions/181-ssh-remote-workspace-bindings.md`, `CHANGELOG.md`, `Docs/security/production-diagnostic-inventory.json`, `backlog/tasks/task-3340*.md` | docs, inventory, notes | 7 |

---

### Task 1: Dead host fails fast; handshake bounded by the call budget (TASK-33400)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_session_registry.py` (`acquire`, `close_key`, `close_all`, `__init__`)
- Modify: `tldw_chatbook/Tools/remote_session_worker.py` (`__init__`, `start`, `_fail_start`, new `_abandon_start`)
- Modify: `tldw_chatbook/Tools/remote_workspace_executor.py` (`_ssh_session_call`'s `create`)
- Test: `Tests/Tools/test_remote_session_registry.py`, `Tests/Tools/test_remote_session_worker.py`, `Tests/Tools/test_remote_executor_ssh.py`

**Interfaces:**
- Produces: `RemoteSessionWorker(..., handshake_timeout: float | None = None)`. `None` means the module's `_HANDSHAKE_TIMEOUT_S` (30 s); a number is capped at it.
- Produces: `RemoteSessionRegistry._start_failures: dict[tuple[str, str], tuple[float, SessionStartError]]`. It is private, and Task 2 adds its mux branch next to it.

- [ ] **Step 1: Write the failing registry test**

Append to `Tests/Tools/test_remote_session_registry.py`:

```python
def test_waiters_share_one_transport_start_failure():
    """A dead host costs one start, not one per queued caller (TASK-33400)."""
    reg = RemoteSessionRegistry()
    creates = []
    release = threading.Event()
    entered = threading.Barrier(5)

    def create():
        creates.append(1)
        return FakeWorker(
            start_error=SessionStartError(True, None, "255"),
            start_hook=lambda: release.wait(5),
        )

    errors = []

    def caller():
        entered.wait(5)
        try:
            reg.acquire(("run-1", "b1"), create)
        except SessionStartError as error:
            errors.append(error)

    threads = [threading.Thread(target=caller) for _ in range(4)]
    for thread in threads:
        thread.start()
    entered.wait(5)
    time.sleep(0.3)  # all four are inside acquire: one starting, three queued
    release.set()
    for thread in threads:
        thread.join(10)
    assert len(creates) == 1
    assert len(errors) == 4 and all(error.transport for error in errors)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py::test_waiters_share_one_transport_start_failure -v -p no:randomly`
Expected: FAIL. `len(creates) == 4`, because each queued caller starts again.

- [ ] **Step 3: Implement the fan-out in `remote_session_registry.py`**

Add `import time` to the imports. In `__init__`, after `self._restarted`:

```python
        #: Last transport-class start failure per key, with the monotonic
        #: time it happened: callers that queued behind that start share it
        #: instead of each paying another connect timeout (TASK-33400).
        self._start_failures: dict[tuple[str, str], tuple[float, SessionStartError]] = {}
```

In `acquire`, capture the entry time first and check the failure once the key lock is held. The full new body, from the first line to `worker = create()`:

```python
        entered = time.monotonic()
        with self._lock:
            if self._shutdown:
                return None
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                # A closed run key is one-shot for good: surviving
                # sub-agents and Stop stragglers never reopen an unowned
                # session.
                if self._shutdown or key in self._disabled or key[0] in self._closed_keys:
                    return None
                worker = self._sessions.get(key)
                if worker is not None and worker.alive:
                    return worker
                failed = self._start_failures.get(key)
            if failed is not None and failed[0] >= entered:
                # This caller queued behind a start that failed
                # transport-class: share that failure (a fresh exception per
                # thread) rather than start again against the same dead host.
                raise SessionStartError(failed[1].transport, failed[1].failure, str(failed[1]))
            # ended_cleanly() may wait for the reap: outside the global lock.
```

Keep the existing restart block that follows. In the `except SessionStartError as error:` handler, record the failure just before the final `raise`:

```python
                kind = error.failure.kind.value if error.failure else "unknown"
                logger.debug(f"ssh session worker start failed (transport): {kind}")
                with self._lock:
                    self._start_failures[key] = (time.monotonic(), error)
                raise
```

After a successful start, in the `with self._lock:` block that computes `closed_meanwhile`, add `self._start_failures.pop(key, None)` as its first line.

In `close_key`, next to the `_restarted` prune, add:
`self._start_failures = {k: v for k, v in self._start_failures.items() if k[0] != session_key}`.
In `close_all`, add `self._start_failures.clear()` to the clear line.

- [ ] **Step 4: Run the registry tests**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py -v -p no:randomly`
Expected: all PASS. The new test passes, and `test_transport_failure_raises_and_is_retried_next_call` still passes, because a call made after the failure retries.

- [ ] **Step 5: Write the failing worker tests**

In `Tests/Tools/test_remote_session_worker.py`, give `worker_factory`'s `make(...)` one more keyword, `handshake_timeout: float | None = None`, and pass `handshake_timeout=handshake_timeout` to `RemoteSessionWorker(...)`. Then append:

```python
def test_handshake_is_bounded_by_the_call_budget(worker_factory):
    """TASK-33400: a silent host fails at the budget, classified like one-shot."""
    worker, _ = worker_factory(spawn_argv=["sh", "-c", "exec sleep 30"], handshake_timeout=1.0)
    started = time.monotonic()
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert time.monotonic() - started < 6
    assert err.value.transport is True
    assert err.value.failure.kind is TransportFailureKind.UNREACHABLE


def test_failed_start_reaps_ssh_and_closes_its_pipes(worker_factory):
    worker, spawns = worker_factory(spawn_argv=["sh", "-c", "exit 255"])
    with pytest.raises(SessionStartError):
        worker.start()
    proc = spawns[0]
    assert proc.returncode is not None
    assert proc.stdin.closed and proc.stdout.closed and proc.stderr.closed


def test_unexpected_pipe_error_during_start_is_a_protocol_start_error(worker_factory, monkeypatch):
    worker, spawns = worker_factory(spawn_argv=["sh", "-c", "exec sleep 30"])

    def broken_read(self, deadline):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(RemoteSessionWorker, "_read_handshake_line", broken_read)
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is False
    proc = spawns[0]
    assert proc.returncode is not None and proc.stdout.closed
```

- [ ] **Step 6: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -k "bounded_by_the_call_budget or reaps_ssh or unexpected_pipe" -v -p no:randomly`
Expected:
- the first test fails with a TypeError (unexpected `handshake_timeout`);
- the second fails because its pipes are still open;
- the third fails because the OSError propagates and the process is left running.

- [ ] **Step 7: Implement in `remote_session_worker.py`**

`__init__`: add the parameter `handshake_timeout: float | None = None`, document it in the class docstring (Args) and store it as `self._handshake_timeout = handshake_timeout`.

`start()`: replace everything from `self._proc = proc` to the end of the method with the following:

```python
        self._proc = proc
        try:
            self._handshake(proc, loader=loader, compressed=compressed, artifact=artifact)
        except SessionStartError:
            self._abandon_start()
            raise
        except OSError:
            # A local pipe error outside the classified paths: nothing can
            # be said about the host, so fall back to one-shot for the run.
            self._abandon_start()
            raise SessionStartError(False, None, "session start failed: local pipe error") from None
        except BaseException:
            self._abandon_start()
            raise
```

Move the old body, from `os.set_blocking(...)` through the final `logger.debug(... started ...)` line, unchanged into a new method `def _handshake(self, proc: subprocess.Popen[bytes], *, loader: bytes, compressed: bytes, artifact) -> None:`. Type `artifact` exactly as `_bundle_payload()` returns it; it and `loader`/`compressed` are the values `start()` already computes before spawning. Make one edit: the deadline becomes

```python
        limit = _HANDSHAKE_TIMEOUT_S
        if self._handshake_timeout is not None:
            limit = min(self._handshake_timeout, limit)
        deadline = time.monotonic() + limit
        self._handshake_limit = limit
```

and `_fail_start` passes `budget=self._handshake_limit` instead of `budget=_HANDSHAKE_TIMEOUT_S`. Initialise `self._handshake_limit = _HANDSHAKE_TIMEOUT_S` in `__init__`.

Add:

```python
    def _abandon_start(self) -> None:
        """Leave nothing behind after a failed start: reap ssh, close its pipes."""
        self._kill_and_reap()
        if self._stderr_thread is not None:
            self._stderr_thread.join(_CLOSE_WAIT_S)
        proc = self._proc
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is None:
                continue
            if stream is proc.stderr and self._stderr_thread is not None and self._stderr_thread.is_alive():
                continue  # still read by the drain thread: left to GC, never closed under it
            try:
                stream.close()
            except OSError:
                pass
```

`_kill_and_reap` already tolerates a reaped process, so the double reap after `_fail_start` is harmless.

- [ ] **Step 8: Run the worker tests**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -v -p no:randomly`
Expected: all PASS, including `test_bundle_write_is_bounded_by_the_handshake_deadline`. That test monkeypatches `_HANDSHAKE_TIMEOUT_S`, which is still read at call time.

- [ ] **Step 9: Write the failing executor test**

Append to `Tests/Tools/test_remote_executor_ssh.py` after the session tests:

```python
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
```

If `time` or `TransportFailureKind` is not imported yet at the top of that module, import it there, next to its siblings.

- [ ] **Step 10: Run it to verify it fails**

Run: `$PY -m pytest Tests/Tools/test_remote_executor_ssh.py::test_session_start_against_a_silent_host_is_bounded_by_the_call_budget -v -p no:randomly`
Expected: FAIL. The elapsed time is about 30 s (the fixed handshake).

- [ ] **Step 11: Pass the budget from the executor**

In `remote_workspace_executor.py`, inside `_ssh_session_call`'s `create()`, add this argument to `RemoteSessionWorker(...)`:

```python
                # The start is part of this call: it may not outlive the
                # call's own budget (same deadline as a one-shot handshake).
                handshake_timeout=budget + cfg.transport.grace_seconds,
```

- [ ] **Step 12: Run the session-related suites**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py Tests/Tools/test_remote_session_loopback.py -p no:randomly -q`
Expected: all PASS.

- [ ] **Step 13: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_registry.py tldw_chatbook/Tools/remote_session_worker.py tldw_chatbook/Tools/remote_workspace_executor.py Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py
git commit -m "fix: SSH session start fails fast against a dead host and within the call budget (TASK-33400)"
```

---

### Task 2: Mux failure at start keeps the warm path; stderr keeps its tail (TASK-33402, TASK-33403)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_session_registry.py` (`acquire` except branch, `__init__`, `close_key`, `close_all`)
- Modify: `tldw_chatbook/Tools/remote_session_worker.py` (new `_TailCapture`, `_stderr` construction, imports)
- Test: `Tests/Tools/test_remote_session_registry.py`, `Tests/Tools/test_remote_session_worker.py`, `Tests/Tools/test_remote_executor_ssh.py`

**Interfaces:**
- Consumes: Task 1's `_start_failures`. The mux branch must run **before** the transport recording, and it never records one.
- Produces: `remote_session_worker._TailCapture(cap: int)` with `append(chunk: bytes) -> None` and `value() -> bytes`.

- [ ] **Step 1: Write the failing registry tests**

Append to `Tests/Tools/test_remote_session_registry.py`:

```python
from tldw_chatbook.Tools.remote_workspace_transport import TransportFailure, TransportFailureKind


def _mux_error():
    failure = TransportFailure(TransportFailureKind.MUX_ERROR, 255, "mux_client_hello_exchange")
    return SessionStartError(False, failure, "session start failed: mux")


def test_mux_start_failure_goes_one_shot_for_that_call_only():
    """TASK-33402: a stale control socket costs one one-shot call, not the run's warm path."""
    reg = RemoteSessionRegistry()
    assert reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=_mux_error())) is None
    assert reg.acquire(("run-1", "b1"), FakeWorker) is not None


def test_repeated_mux_start_failure_disables_the_key_for_the_run():
    reg = RemoteSessionRegistry()
    assert reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=_mux_error())) is None
    assert reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=_mux_error())) is None
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("mux-disabled key restarted")) is None
```

Put the import next to the file's other imports, not mid-file.

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py -k mux -v -p no:randomly`
Expected: the first test fails. Its second acquire returns `None`, because the key is disabled on the first mux failure.

- [ ] **Step 3: Implement the mux policy**

In `remote_session_registry.py`, add
`from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind`
next to the worker import. In `__init__`, add:

```python
        #: Keys whose session start already hit a mux failure this run: the
        #: first costs one one-shot call, a second disables the key (TASK-33402).
        self._mux_failed: set[tuple[str, str]] = set()
```

At the top of `acquire`'s `except SessionStartError as error:` handler, before `if not error.transport:`, add:

```python
                if error.failure is not None and error.failure.kind is TransportFailureKind.MUX_ERROR:
                    # A stale control socket: classifying it already restarted
                    # the master, and nothing was sent, so this call runs
                    # one-shot and the next call tries a session again. A
                    # second mux start failure in the run disables the key.
                    with self._lock:
                        repeat = key in self._mux_failed
                        self._mux_failed.add(key)
                        if repeat:
                            self._disabled.add(key)
                    logger.info(
                        "ssh session start hit a stale control socket; using one-shot calls "
                        + ("for this run" if repeat else "for this call")
                    )
                    return None
```

Prune `_mux_failed` by run key in `close_key`, the same way as `_restarted`, and clear it in `close_all`.

- [ ] **Step 4: Run the registry tests**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Write the failing worker tests**

Append to `Tests/Tools/test_remote_session_worker.py`:

```python
def test_tail_capture_keeps_the_last_bytes():
    capture = worker_module._TailCapture(8)
    capture.append(b"0123456789")
    capture.append(b"ab")
    assert capture.value() == b"456789ab"


def test_mux_marker_after_64k_of_stderr_still_classifies_mux(worker_factory, workspace):
    """TASK-33403: a long session's death reason is at the END of its stderr."""
    worker, _ = worker_factory()
    worker.start()
    worker.close()
    worker._stderr.append(b"x" * (70 * 1024) + b"\n")
    worker._stderr.append(b"mux_client_request_session: read from master failed: Broken pipe\n")
    worker._death_natural, worker._death_code = True, 255  # as if ssh died on its own
    worker._retired = False  # Task 3 adds this flag; harmless before it exists
    result = worker._dead_session_result()
    assert result.failure.kind is TransportFailureKind.MUX_ERROR


def test_mux_error_at_start_is_classified_mux(worker_factory):
    worker, _ = worker_factory(
        spawn_argv=[
            "sh", "-c",
            "echo 'mux_client_hello_exchange: write packet: Broken pipe' >&2; exit 255",
        ]
    )
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.failure.kind is TransportFailureKind.MUX_ERROR
```

- [ ] **Step 6: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -k "tail_capture or mux" -v -p no:randomly`
Expected:
- `test_tail_capture_keeps_the_last_bytes` fails with an AttributeError;
- the 64K test fails because it returns UNREACHABLE (the head capture dropped the marker);
- `test_mux_error_at_start_is_classified_mux` should already PASS, pinning the classification the registry relies on.

- [ ] **Step 7: Implement `_TailCapture`**

In `remote_session_worker.py`, below `_wait_fd`, add:

```python
class _TailCapture:
    """Byte sink keeping the LAST ``cap`` bytes (TASK-33403).

    A long-lived session's death reason (a mux marker, ssh's final error)
    is at the end of its stderr; a head capture drops it once earlier
    output fills the cap, turning a status-preserving MUX_ERROR into
    UNREACHABLE (BLOCKED).
    """

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._buf += chunk
            if len(self._buf) > self._cap:
                del self._buf[: len(self._buf) - self._cap]

    def value(self) -> bytes:
        with self._lock:
            return bytes(self._buf)
```

Change `self._stderr = _BoundedCapture(_STDERR_CAP)` to `self._stderr = _TailCapture(_STDERR_CAP)`. Remove `_BoundedCapture` from the transport import if nothing else in the module uses it (`grep -n _BoundedCapture` first).

- [ ] **Step 8: Write the failing executor test**

Append to `Tests/Tools/test_remote_executor_ssh.py`:

```python
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
```

If the executor's first call is a one-shot identity ping, `one_shot_before` is read **after** building the executor, so the assertions count only the read. If the implementer finds that the first `env.read` also issues a ping, warm it first with `env.read(executor)` on a separate executor instance, and say so in the report.

- [ ] **Step 9: Run everything touched**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py -p no:randomly -q`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_registry.py tldw_chatbook/Tools/remote_session_worker.py Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py
git commit -m "fix: stale control socket at SSH session start keeps the warm path; session stderr keeps its tail (TASK-33402, TASK-33403)"
```

---

### Task 3: Close sessions off the caller's thread, in parallel; never fail a just-acquired call (TASK-33401, TASK-33405)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_session_registry.py` (new `_close_workers`, `_CLOSE_JOIN_S`; `reap_idle`, `close_key`, `close_all`)
- Modify: `tldw_chatbook/Tools/remote_session_worker.py` (new `SessionClosed`; `close`, `call`, `__init__`)
- Modify: `tldw_chatbook/Tools/remote_workspace_executor.py` (`_ssh_session_call` retry loop)
- Test: `Tests/Tools/test_remote_session_registry.py`, `Tests/Tools/test_remote_session_worker.py`, `Tests/Tools/test_remote_executor_ssh.py`

**Interfaces:**
- Produces: `remote_session_worker.SessionClosed(Exception)`, raised by `RemoteSessionWorker.call` only when the laptop closed a healthy session **before** this call's request was registered, so nothing was sent.
- Produces: `RemoteSessionWorker._retired: bool`. It is True once `close()` closed a session that was alive at that moment.

- [ ] **Step 1: Write the failing registry tests**

Append to `Tests/Tools/test_remote_session_registry.py`:

```python
class SlowCloseWorker(FakeWorker):
    def __init__(self, delay=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delay, self.closed_on = delay, None

    def close(self):
        self.closed_on = threading.current_thread()
        time.sleep(self.delay)
        super().close()


def test_reap_idle_never_closes_on_the_calling_thread():
    """TASK-33401: a wedged session's close never lands on an unrelated call."""
    reg = RemoteSessionRegistry()
    worker = reg.acquire(("run-1", "b1"), lambda: SlowCloseWorker(delay=2.0, idle_since=0.0))
    started = time.monotonic()
    reg.reap_idle(now=100.0, idle_s=1.0)
    assert time.monotonic() - started < 0.5
    deadline = time.monotonic() + 5
    while worker.closed_on is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.closed_on is not threading.current_thread()


def test_close_all_closes_in_parallel_and_is_bounded(monkeypatch):
    """TASK-33405: app exit costs about the slowest close, never their sum."""
    from tldw_chatbook.Tools import remote_session_registry as registry_module

    reg = RemoteSessionRegistry()
    workers = [
        reg.acquire((f"run-{i}", "b1"), lambda: SlowCloseWorker(delay=1.0)) for i in range(3)
    ]
    started = time.monotonic()
    reg.close_all()
    assert time.monotonic() - started < 2.5  # serial would be ~3 s
    assert all(w.closed for w in workers)

    monkeypatch.setattr(registry_module, "_CLOSE_JOIN_S", 0.5)
    reg = RemoteSessionRegistry()
    reg.acquire(("run-x", "b1"), lambda: SlowCloseWorker(delay=5.0))
    started = time.monotonic()
    reg.close_all()
    assert time.monotonic() - started < 2.0  # a wedged close never holds app exit
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py -k "reap_idle_never or close_all_closes" -v -p no:randomly`
Expected: both FAIL. The reap blocks for about 2 s, and close_all takes about 3 s.

- [ ] **Step 3: Implement the parallel close**

In `remote_session_registry.py`, below `_CLOSED_KEYS_MAX`:

```python
#: How long run end / app exit waits for session closes (all in parallel).
#: A close still running after this finishes on its daemon thread; at app
#: exit the master's ``-O exit`` ends its channel anyway.
_CLOSE_JOIN_S = 5.0


def _close_workers(workers: list[RemoteSessionWorker], *, wait: float | None) -> None:
    """Close ``workers`` concurrently, one daemon thread each.

    Args:
        workers: Sessions already removed from the registry.
        wait: Total seconds to wait for all closes; ``None`` returns at once
            (the idle reaper: a close never lands on the calling tool call).
    """
    threads = [
        threading.Thread(target=worker.close, name="ssh-session-close", daemon=True)
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    if wait is None:
        return
    deadline = time.monotonic() + wait
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
```

(`_CLOSE_JOIN_S` is read at call time, so tests can monkeypatch it.) Then replace the trailing `for worker in workers: worker.close()` loops:
- `close_key` → `_close_workers(workers, wait=_CLOSE_JOIN_S)`
- `close_all` → `_close_workers(workers, wait=_CLOSE_JOIN_S)`
- `reap_idle` → `_close_workers(workers, wait=None)`

- [ ] **Step 4: Run the registry tests**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py -v -p no:randomly`
Expected: all PASS. If `test_close_key_and_idle_reap` asserted that a reaped fake is `closed` right after `reap_idle` returns, make it wait for `worker.closed`, with a bounded poll like the one in Step 1. The close is now asynchronous by design.

- [ ] **Step 5: Write the failing worker test (contract change)**

In `Tests/Tools/test_remote_session_worker.py`, replace `test_call_after_close_is_remote_op_failed` with:

```python
def test_call_after_close_raises_session_closed(worker_factory, workspace):
    """TASK-33401: a session the laptop retired sent nothing; the caller re-acquires."""
    worker, _ = worker_factory()
    worker.start()
    worker.close()
    with pytest.raises(worker_module.SessionClosed):
        worker.call(read_request(workspace, "a.txt"), budget=10)


def test_call_after_a_stuck_kill_is_still_remote_op_failed(worker_factory, workspace):
    """Only a laptop close of a HEALTHY session is retryable; a killed one is not."""
    worker, _ = worker_factory()
    worker.start()
    worker._die(natural=False)
    result = worker.call(read_request(workspace, "a.txt"), budget=10)
    assert result.failure.kind is TransportFailureKind.REMOTE_OP_FAILED
```

- [ ] **Step 6: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -k "session_closed or stuck_kill" -v -p no:randomly`
Expected: the first test fails with an AttributeError (no `SessionClosed`). The second passes and pins existing behaviour.

- [ ] **Step 7: Implement `SessionClosed`**

In `remote_session_worker.py`, next to `SessionStartError`:

```python
class SessionClosed(Exception):
    """The laptop closed this healthy session (idle reap, run end, app
    exit) before the call's request was registered: nothing was sent, so
    the caller may ask the registry again (TASK-33401)."""
```

In `__init__`, add `self._retired = False` next to `_death_natural`. In `close()`, inside the first `with self._lock:` block, after `was_alive, self._alive = self._alive, False`, add `self._retired = self._retired or was_alive`.

In `call()`, replace

```python
        if not alive:
            return self._dead_session_result()
```

with

```python
        if not alive:
            if self._retired:
                raise SessionClosed()
            return self._dead_session_result()
```

Also read `self._retired` inside the same `with self._lock:` block as `alive`: capture `retired = self._retired` there and test `retired`, so the pair is consistent.

- [ ] **Step 8: Write the failing executor test**

Append to `Tests/Tools/test_remote_executor_ssh.py`:

```python
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
```

- [ ] **Step 9: Run it to verify it fails**

Run: `$PY -m pytest Tests/Tools/test_remote_executor_ssh.py::test_session_reaped_between_acquire_and_call_is_not_a_tool_error -v -p no:randomly`
Expected: FAIL. `SessionClosed` propagates, because the executor doesn't handle it yet.

- [ ] **Step 10: Re-acquire in the executor**

In `_ssh_session_call`, import `SessionClosed` next to `SessionStartError` (lazy import block) and replace the body under `with _host_semaphore(...)` with:

```python
            registry.reap_idle(time.monotonic(), settings.session_idle_s)
            for _attempt in range(2):
                try:
                    session = registry.acquire((cfg.session_key, cfg.binding_id), create)
                except SessionStartError as error:
                    return RemoteCallResult(
                        False,
                        None,
                        error.failure
                        or TransportFailure(TransportFailureKind.UNREACHABLE, None, str(error)),
                    )
                if session is None:
                    return None
                try:
                    return session.call(request_bytes, budget=budget)
                except SessionClosed:
                    # Retired (idle reap, run end, app exit) after acquire
                    # handed it out and before this request was sent: nothing
                    # ran, so ask again -- a fresh session, or one-shot once
                    # the run or app has ended.
                    continue
            return None
```

Update `_ssh_session_call`'s docstring with one sentence on the re-acquire.

- [ ] **Step 11: Run all session suites**

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py Tests/Tools/test_remote_session_loopback.py -p no:randomly -q`
Expected: all PASS. `test_call_after_run_end_goes_one_shot_without_a_new_session` must still pass: a straggler now takes `SessionClosed` → re-acquire → `None` → one-shot.

- [ ] **Step 12: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_registry.py tldw_chatbook/Tools/remote_session_worker.py tldw_chatbook/Tools/remote_workspace_executor.py Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py
git commit -m "fix: SSH sessions close off the caller's thread and in parallel; a just-reaped session re-acquires (TASK-33401, TASK-33405)"
```

---

### Task 4: No new ControlMaster after app exit (TASK-33406)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_workspace_transport.py` (`SshMasterManager.__init__`, `ensure_master`, `restart_if_dead`, `close_all`)
- Modify: `Tests/conftest.py` (`reset_remote_session_registry`)
- Test: `Tests/Tools/test_ssh_master_manager.py`

**Interfaces:**
- Produces: `SshMasterManager._closed: bool`. It is set by `close_all()`, and after that `ensure_master` never spawns and `restart_if_dead` returns False.

- [ ] **Step 1: Write the failing tests**

Append to the `close_all` section of `Tests/Tools/test_ssh_master_manager.py`:

```python
def test_no_master_is_spawned_after_close_all(fake_ssh: FakeSsh) -> None:
    """TASK-33406: a straggler call after app exit starts no detached master."""
    manager = _manager(fake_ssh)
    manager.close_all()
    manager.ensure_master(_LOC)
    assert fake_ssh.count("-MNf") == 0
    assert manager.restart_if_dead(_LOC) is False
    assert not any("check" in argv for argv in fake_ssh.invocations())


def test_closed_singleton_part_1_app_exit() -> None:
    from tldw_chatbook.Tools.remote_workspace_transport import get_master_manager

    get_master_manager().close_all()
    assert get_master_manager()._closed


def test_closed_singleton_part_2_next_test_gets_a_working_manager() -> None:
    from tldw_chatbook.Tools.remote_workspace_transport import get_master_manager

    assert not get_master_manager()._closed
```

Part 2 depends on part 1 running first in the same process. Copy exactly how `Tests/Tools/test_remote_session_registry.py` orders and guards its `test_shutdown_singleton_part_1/2` pair (same markers or `xdist_group`, if it uses any). If it has none, add none.

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_ssh_master_manager.py -k "after_close_all or closed_singleton" -v -p no:randomly -p no:xdist`
Expected:
- `test_no_master_is_spawned_after_close_all` fails (one `-MNf`);
- part 1 fails with an AttributeError on `_closed`.

- [ ] **Step 3: Implement the latch**

In `SshMasterManager.__init__`, after `self._mux_unusable = False`, add:

```python
        #: Set by close_all (app exit): no new masters or health checks
        #: afterwards, so a straggler call connects directly instead of
        #: leaving a detached master behind (TASK-33406).
        self._closed = False
```

In `close_all`, make `self._closed = True` the first line inside `with self._registry_lock:`.

In `ensure_master`, change the first guard to `if not self._enabled or self._closed: return`. Inside the `with self._lock_for(key):` block, immediately before `self._spawn_master(loc, key, control_dir)`, add:

```python
            if self._closed:
                # ponytail: a start that passed the entry check just before
                # close_all can still spawn here; ControlPersist bounds it.
                return
```

In `restart_if_dead`, return `False` at its top when `self._closed` (next to its existing disabled early-return). Document both in the docstrings: "After :meth:`close_all` this is a no-op".

- [ ] **Step 4: Reset the singleton between tests**

In `Tests/conftest.py`, extend `reset_remote_session_registry`'s teardown, and its docstring with one sentence:

```python
    transport = sys.modules.get("tldw_chatbook.Tools.remote_workspace_transport")
    if transport is not None:
        transport._MASTER_MANAGER = None
```

Leave it without a lock. The fixture runs between tests, the same as the registry reset.

- [ ] **Step 5: Run the manager tests and the app-exit test**

Run: `$PY -m pytest Tests/Tools/test_ssh_master_manager.py Tests/Tools/test_remote_session_registry.py -p no:randomly -p no:xdist -q`
Expected: all PASS, including `test_app_exit_closes_sessions_before_masters_in_one_best_effort_try`.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Tools/remote_workspace_transport.py Tests/conftest.py Tests/Tools/test_ssh_master_manager.py
git commit -m "fix: no SSH ControlMaster is started after app exit (TASK-33406)"
```

---

### Task 5: A failed host fork fails one request, not the session (TASK-33404)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_session_frames.py` (`HOST_SPAWN_FAILED`), bundled
- Modify: `tldw_chatbook/Tools/remote_session_serve.py` (`_spawn`, `start`, docstring), bundled
- Modify: `tldw_chatbook/Tools/remote_session_worker.py` (`_result`)
- Regenerate: `tldw_chatbook/Tools/remote_worker_bundle.py`
- Test: `Tests/Tools/test_remote_session_serve.py`, `Tests/Tools/test_remote_session_worker.py`

**Interfaces:**
- Produces: `remote_session_frames.HOST_SPAWN_FAILED = 71` (sysexits `EX_OSERR`), the STATUS exit code for a request the host could not start.

- [ ] **Step 1: Write the failing serve test**

In `Tests/Tools/test_remote_session_serve.py`:
- add `HOST_SPAWN_FAILED` to the frames import;
- give `_session` a `prelude=""` parameter that is spliced in before the serve import;
- add the test.

```python
def _session(max_children=4, idle_s=30.0, prelude=""):
    script = HANDLER.replace(
        "from tldw_chatbook.Tools.remote_session_serve import serve",
        prelude + "\nfrom tldw_chatbook.Tools.remote_session_serve import serve",
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    os.write(proc.stdin.fileno(), encode_frame(HELLO, 0, json.dumps({"max_children": max_children, "idle_s": idle_s}).encode()))
    return proc, FrameReader(max_body=1 << 20)


_FAIL_ONCE = textwrap.dedent('''
    import errno
    _real, _left = os.{name}, [1]
    def _once(*args):
        if _left[0]:
            _left[0] -= 1
            raise OSError(errno.{errno}, "refused")
        return _real(*args)
    os.{name} = _once
''')


@pytest.mark.parametrize("name,errno_name", [("fork", "EAGAIN"), ("pipe", "EMFILE")])
def test_failed_spawn_fails_only_that_request(name, errno_name):
    """TASK-33404: the host at its process/fd limit fails one request, not the session."""
    proc, reader = _session(prelude=_FAIL_ONCE.format(name=name, errno=errno_name))
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"first") + encode_frame(REQUEST, 2, b"second"))
        frames = _collect(proc, reader, 2)
        statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
        assert statuses[1] == (HOST_SPAWN_FAILED, None)
        assert statuses[2] == (0, None)
        assert not any(kind == LINE and rid == 1 for kind, rid, _ in frames)
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 3, b"third"))
        assert any(kind == STATUS and rid == 3 for kind, rid, _ in _collect(proc, reader, 1))
    finally:
        _terminate(proc)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest Tests/Tools/test_remote_session_serve.py -k failed_spawn -v -p no:randomly`
Expected: FAIL at collection with an ImportError (`HOST_SPAWN_FAILED`). Once the constant exists it fails again, because the session exits and no STATUS arrives.

- [ ] **Step 3: Implement**

In `remote_session_frames.py`, after the kind constants, add:

```python
#: STATUS exit code for a request the host could not start (fork or pipe
#: refused, e.g. EAGAIN at the process limit): sysexits' EX_OSERR. The
#: request never ran; the session and every other request carry on.
HOST_SPAWN_FAILED = 71
```

In `remote_session_serve.py`:
- import `HOST_SPAWN_FAILED` from frames;
- in `_spawn`, guard the fork;
- in `start()`, turn a spawn `OSError` into a per-request STATUS.

```python
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        raise
```

```python
    def start(request_id: int, raw: bytes) -> None:
        try:
            child = _spawn(raw, request_id, run_request)
        except OSError:
            # fork/pipe refused (process or fd limit): only this request
            # fails; the loop, its children and its queue carry on.
            outbox.extend(encode_frame(STATUS, request_id, encode_status(HOST_SPAWN_FAILED, None)))
            return
        children[child.fd] = child
        by_request[request_id] = child
        selector.register(child.fd, selectors.EVENT_READ, child)
```

Add one sentence to `serve`'s docstring: a request whose fork or pipe fails gets `STATUS(HOST_SPAWN_FAILED, None)` and the session continues. Also remove the fork case from the `Raises: OSError` wording.

In `remote_session_worker.py`:
- import `HOST_SPAWN_FAILED` from frames;
- in `_result`, right after `exit_code, signal_no = pending.status`, add:

```python
            if not admitted and exit_code == HOST_SPAWN_FAILED:
                # The host could not fork for this request (process limit):
                # the live session proves reachability, so status-preserving.
                return RemoteCallResult(
                    False,
                    None,
                    TransportFailure(
                        TransportFailureKind.REMOTE_OP_FAILED,
                        exit_code,
                        "host could not start the operation",
                    ),
                )
```

- [ ] **Step 4: Write the worker mapping test**

Append to `Tests/Tools/test_remote_session_worker.py`:

```python
def test_host_spawn_failure_is_status_preserving(worker_factory):
    from tldw_chatbook.Tools.remote_session_frames import HOST_SPAWN_FAILED

    worker, _ = worker_factory()
    pending = worker_module._Pending(status=(HOST_SPAWN_FAILED, None))
    result = worker._result(pending, 5.0, killed=False)
    assert not result.admitted
    assert result.failure.kind is TransportFailureKind.REMOTE_OP_FAILED
```

- [ ] **Step 5: Regenerate the bundle and run**

```bash
$PY -m tldw_chatbook.Tools.build_remote_worker_bundle
$PY -m tldw_chatbook.Tools.build_remote_worker_bundle --check
$PY -m pytest Tests/Tools/test_remote_session_serve.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_session_loopback.py Tests/Tools/test_remote_session_loader.py Tests/Tools/test_remote_watchdog.py -p no:randomly -q
```

Expected: `--check` prints `ok`, and all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_frames.py tldw_chatbook/Tools/remote_session_serve.py tldw_chatbook/Tools/remote_session_worker.py tldw_chatbook/Tools/remote_worker_bundle.py Tests/Tools/test_remote_session_serve.py Tests/Tools/test_remote_session_worker.py
git commit -m "fix: a failed host fork fails one SSH session request, not the session (TASK-33404)"
```

---

### Task 6: Measure, then coalesce a fast op's writes only if it pays (TASK-33407)

**Files:**
- Modify (only if the measurement pays): `tldw_chatbook/Tools/remote_session_serve.py`, `tldw_chatbook/Tools/remote_worker_bundle.py`
- Test: `Tests/Tools/test_remote_session_serve.py`
- Record: `backlog/tasks/task-33407 - *.md` (Implementation Notes: every measurement, either way)

**Decision rule, fixed before measuring:** ship coalescing only if the warm median minus the same-window ping median falls by **≥ 2 ms** in **every** one of three interleaved A/B pairs. Otherwise revert the serve change, keep no code, and record the numbers.

- [ ] **Step 1: Write the coalescing tests (they define "coalesced")**

In `Tests/Tools/test_remote_session_serve.py`, add two commands to `HANDLER`'s `run_request`, before the final `echo` line:

```python
        if cmd in ("twostep", "twostep-slow"):
            out.write(b"one\\n"); out.flush()
            time.sleep(0.002 if cmd == "twostep" else 0.6)
            out.write(b"two\\n"); out.flush()
            return 0
```

Add a raw-chunk reader and two tests:

```python
def _chunks_until_status(proc, reader, timeout=10.0):
    """Frames grouped by the os.read chunk they arrived in."""
    chunks, end, fd = [], time.monotonic() + timeout, proc.stdout.fileno()
    while time.monotonic() < end:
        ready, _, _ = select.select([fd], [], [], end - time.monotonic())
        if not ready:
            break
        data = os.read(fd, 65536)
        if not data:
            break
        frames = reader.feed(data)
        chunks.append(frames)
        if any(kind == STATUS for kind, _, _ in frames):
            break
    return chunks


# A generous window for the tests (the shipped value is _COALESCE_S): a
# loaded CI runner must not turn "child exited within the window" flaky.
_HOLD_200MS = "import tldw_chatbook.Tools.remote_session_serve as _s; _s._COALESCE_S = 0.2"


def test_fast_op_output_and_status_leave_in_one_write():
    proc, reader = _session(prelude=_HOLD_200MS)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"twostep"))
        chunks = _chunks_until_status(proc, reader)
        assert len(chunks) == 1, chunks
        assert [kind for kind, _, _ in chunks[0]] == [LINE, LINE, STATUS]
    finally:
        _terminate(proc)


def test_slow_op_first_line_is_not_held_until_the_end():
    proc, reader = _session(prelude=_HOLD_200MS)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"twostep-slow"))
        chunks = _chunks_until_status(proc, reader)
        assert len(chunks) >= 2
        assert [kind for kind, _, _ in chunks[0]] == [LINE]
    finally:
        _terminate(proc)
```

Run: `$PY -m pytest Tests/Tools/test_remote_session_serve.py -k "one_write or not_held" -v -p no:randomly`
Expected: `test_fast_op_output_and_status_leave_in_one_write` FAILS, because the first LINE is flushed alone. The slow test PASSES.

- [ ] **Step 2: Implement the hold in `remote_session_serve.py`**

Below `_QUEUE_BYTES_FACTOR`:

```python
#: How long the parent holds a still-running child's output before writing
#: it, so a fast operation's admitted marker, result and STATUS leave in one
#: write (one ssh packet) instead of two small ones. A slow operation's
#: marker goes out this much later; nothing else waits. 0 disables it.
_COALESCE_S = 0.010
```

In `serve`, next to `outbox = bytearray()`, add `hold_until: float | None = None` and `urgent = False`. Set `urgent = True` wherever a STATUS or BUSY frame is appended:
- in `finish()` (add `nonlocal urgent`);
- in `refuse()` (add `nonlocal urgent`);
- on the BUSY append;
- on the queued-CANCEL STATUS append;
- in Task 5's spawn-failure STATUS inside `start()` (add `nonlocal urgent`).

Where the child branch appends LINE frames (the `for line in lines[:-1]:` loop), set the hold once, right after the loop:

```python
                    if lines[:-1] and hold_until is None:
                        hold_until = clock() + _COALESCE_S
```

(Declare nothing here: this code is in `serve`'s own scope, not a nested function.)

Replace the select-timeout computation and the idle check:

```python
            busy = bool(children or queue)
            if hold_until is not None:
                timeout = max(0.0, hold_until - clock())
            elif busy:
                timeout = None
            else:
                timeout = max(0.0, idle_s - (clock() - last_activity))
            events = selector.select(timeout)
            if not events and not busy and hold_until is None and clock() - last_activity >= idle_s:
                return 0
```

Replace the final `if outbox: flush()` with:

```python
            if outbox and (urgent or hold_until is None or clock() >= hold_until):
                flush()
                urgent, hold_until = False, None
```

Run: `$PY -m pytest Tests/Tools/test_remote_session_serve.py -p no:randomly -q`
Expected: all PASS, the new two included.

- [ ] **Step 3: Regenerate the bundle, then measure A/B on the live host**

```bash
$PY -m tldw_chatbook.Tools.build_remote_worker_bundle && $PY -m tldw_chatbook.Tools.build_remote_worker_bundle --check
```

Run three interleaved pairs, B (coalesce, `_COALESCE_S = 0.010`) and A (`_COALESCE_S = 0`), in the order **B A B A B A**. Rebuild the bundle after each constant flip, because the host runs the bundle. For each run:

```bash
TLDW_LIVE_SSH_HOST=ml-user@192.168.5.84 $PY -m pytest Tests/Tools/test_remote_session_live.py -k warm_call_latency -q -s -p no:randomly
```

Record `warm_median_ms`, `warm_p90_ms` and `ping_median_ms` from the printed RESULTS for each run, in a table in your report. Compute `warm_median − ping_median` per run.

The live test only creates and removes its own `~/tldw-live-*` scratch folder and this feature's cache entries on the host. It never writes laptop config. **Do not change the live test.**

- [ ] **Step 4: Apply the decision rule**

- **Pays (≥ 2 ms better in all three pairs):** set `_COALESCE_S = 0.010`, rebuild, run `--check` and the serve, loopback and worker suites, then commit:

```bash
git add tldw_chatbook/Tools/remote_session_serve.py tldw_chatbook/Tools/remote_worker_bundle.py Tests/Tools/test_remote_session_serve.py
git commit -m "perf: SSH session sends a fast op's marker, result and status in one write (TASK-33407)"
```

- **Does not pay:** revert every change to `remote_session_serve.py`, `remote_worker_bundle.py` and the two coalescing tests (`git checkout -- <files>`). Keep the `twostep` handler commands out too. Commit nothing for this task except the measurement table. Write the table into `backlog/tasks/task-33407 - *.md` under `## Implementation Notes`, then commit:

```bash
git add "backlog/tasks/task-33407 - Measure-and-if-it-pays-coalesce-the-SSH-session-s-admitted-marker-and-result.md"
git commit -m "docs: record the SSH write-coalescing measurement; not shipped (TASK-33407)"
```

Either way, report the full table and the decision.

---

### Task 7: ADR-181, CHANGELOG, inventory, task close-out (TASK-33408 and the notes for all)

**Files:**
- Modify: `backlog/decisions/181-ssh-remote-workspace-bindings.md`
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `Docs/security/production-diagnostic-inventory.json` (regenerated after reading the drift)
- Modify: `backlog/tasks/task-33400 … task-33408 - *.md`

- [ ] **Step 1: ADR-181: inode-reuse note (TASK-33408)**

In `## Decision`, append this paragraph to the end of the first bullet (the `SSH_FILESYSTEM` bullet that names `(st_dev, st_ino)`):

```markdown
  **Limit, accepted:** `(st_dev, st_ino)` names a directory, not its
  history. A filesystem that hands a freed inode number straight back
  (observed on ext4 on the 2026-09-27 live host: `rm -rf root && mkdir
  root` recreated the same pair) makes a deleted-and-recreated root
  indistinguishable from the original, so it is served without a
  `STALE_IDENTITY` stop. The pin still defeats what it exists for — a
  symlink swap, a root replaced by a *different* live directory, and
  escapes out of the root — and this is the same limit local bindings have.
  A stronger identity (ext4's `i_generation` via an ioctl) is
  filesystem-specific and not stdlib-portable, so it is not used.
```

- [ ] **Step 2: ADR-181: follow-up rows in the amendment**

In the `## Amendment 2026-09-27` section, first edit the failure-classification row "A session the laptop ended (… or a call arriving on an already-closed session)". Remove "or a call arriving on an already-closed session" from that list, and add after that row:

```markdown
  - A call whose session the laptop closed while it was healthy (idle reap,
    run end, app exit) **before the call's request was sent** is not failed:
    nothing ran, so it asks the registry again and gets a fresh session, or
    the one-shot path once the run or the app has ended.
```

Then append a new sub-list at the end of the amendment:

```markdown
- **Follow-ups (2026-09-28, TASK-33400–33408).**
  - Callers queued behind a session start that fails transport-class all
    get that failure; nobody starts again against the same dead host. A
    later call still tries a fresh session.
  - A session handshake is part of its call: it gives up at the call's
    budget + grace (capped at 30 s) and is classified exactly like a
    one-shot handshake.
  - A mux failure (`MUX_ERROR`) at session start runs that call one-shot
    over the restarted master and fails nothing; the next call tries a
    session again. A second one in the same run switches the binding to
    one-shot for the run.
  - A session's death is classified from the **last** 64 KiB of its ssh
    stderr, where the reason is.
  - A request the host cannot fork or pipe for gets `STATUS(71)`
    (`EX_OSERR`) and fails alone as `REMOTE_OP_FAILED` (status-preserving:
    the live session proves reachability); the session carries on.
  - Idle reaping closes sessions on a background thread, never on the
    calling tool call; run end and app exit close a run's (or every)
    session in parallel and wait at most 5 s.
  - After app exit the master manager starts no new ControlMaster and runs
    no health check; a straggler call connects directly.
```

If Task 6 shipped, also add:

```markdown
  - A fast operation's admitted marker, result and STATUS leave the host
    in one write; the parent holds a running child's output at most 10 ms
    before writing it. Measured: <paste Task 6's pairs table summary>.
```

If Task 6 did not ship, add instead:

```markdown
  - Coalescing the admitted marker and result into one write was measured
    and not adopted: <paste Task 6's summary line>.
```

- [ ] **Step 3: CHANGELOG**

Under `## [Unreleased]`, in `### Fixed` (create it if absent, in the file's existing style), add one bullet per user-visible fix:

```markdown
- SSH workspaces: several tool calls against a host that just went down now fail together within one connect timeout instead of one after another, and a session start never outlives the call's time budget.
- SSH workspaces: the first call after the laptop wakes from sleep (stale ControlMaster socket) no longer errors or loses the fast session path for the rest of the run.
- SSH workspaces: a host at its process limit fails only the one request it could not start; the binding is not marked unreachable.
- SSH workspaces: quitting with several remote bindings closes their sessions in parallel within 5 s, and no ControlMaster is left running after exit.
```

If Task 6 shipped, add a `### Changed` bullet: "SSH workspaces: fast remote tool calls return about one network round trip sooner."

- [ ] **Step 4: Diagnostic inventory**

Run `PATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin:$PATH ./scripts/preflight.sh`. If the diagnostic inventory reports drift:
- read every row it names;
- confirm each is one of this branch's new static-text `logger` lines (the mux line from Task 2, plus any others you can point to in the diff);
- regenerate with the command preflight prints (`--write`).

Then re-run preflight until it prints `preflight: all derived-artifact checks passed.`

- [ ] **Step 5: Close out the nine tasks**

For each of `task-33400` … `task-33408`:
- tick every AC it met (`- [ ]` → `- [x]`);
- add `## Implementation Notes`: 2–5 lines on the approach, files and tests, plus deviations;
- set `status: Done` and `updated_date: '2026-09-28 HH:MM'`;
- set `assignee:` to `- '@claude'`.

TASK-33407's notes carry the measurement table either way; its third AC is ticked if it did not ship, its second if it did.

- [ ] **Step 6: Full gate and commit**

```bash
$PY -m tldw_chatbook.Tools.build_remote_worker_bundle --check
$PY -m pytest Tests/Tools/ -p no:randomly -q -k "remote or ssh or session"
PATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin:$PATH ./scripts/preflight.sh
git add backlog/decisions/181-ssh-remote-workspace-bindings.md CHANGELOG.md Docs/security/production-diagnostic-inventory.json backlog/tasks/task-3340*.md
git commit -m "docs: ADR-181 session follow-ups and inode-reuse limit; close TASK-33400..33408"
```

Expected: `--check` ok, tests PASS, and preflight passes.
