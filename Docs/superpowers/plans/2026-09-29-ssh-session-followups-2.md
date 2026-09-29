# SSH Session Follow-ups 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two known limits PR #2899 left open, TASK-33420 and TASK-33421, in one PR against `dev`.

**Architecture:** Both fixes stay in the laptop session worker (`remote_session_worker.py`), plus one registry line. TASK-33420 adds a notion of "the host has answered the handshake": once the loader has sent its first line, the channel is proven live. After that point:
- running out of the call's budget fails only that call, as a status-preserving `OP_TIMEOUT`;
- hitting the 30 s cap is protocol-class (one-shot for the run);
- neither is ever `UNREACHABLE`.

TASK-33421 makes `_write`'s "stdin already closed" pre-check (nothing written) a distinct error. When the REQUEST write hits it on a session the laptop retired, `call()` raises `SessionClosed`, and the executor already re-acquires on that. No bundled module changes.

**Tech Stack:** Python ≥3.12, pytest, the real stage-1 loader driven locally through the `spawn` seam, and small fake-host scripts.

**Spec:** `backlog/decisions/181-ssh-remote-workspace-bindings.md` (amendment + "Follow-ups (2026-09-28)"), `Docs/superpowers/specs/2026-09-27-ssh-session-worker-and-bundle-cache-design.md`, and the two task files `backlog/tasks/task-33420 - *.md` and `task-33421 - *.md`.

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/ssh-session-followups-2`, branch `feat/ssh-session-followups-2`. Check `pwd` before every edit. Do not push.
- Python: `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Run tests as `$PY -m pytest … -p no:randomly`.
- **Never write the user's real config.** No unmocked config writers, and never bypass `Tests/private_profile.py` or any isolation shim. Never run the live SSH test.
- **Status-cache rules (ADR-181):**
  - Only BLOCKING kinds flip a binding to BLOCKED.
  - A live channel proves reachability, so no failure after the host has answered may be transport-class.
  - `OP_TIMEOUT`, `REMOTE_OP_FAILED` and `MUX_ERROR` never flip state.
- **Never retry a request that may have run.** `SessionClosed` is raised only when no REQUEST byte was written.
- **Logs:** static text only. No new module-level imports in UI-ready modules.
- **Known local noise, do not chase:** `Tests/Chat/test_console_chat_controller.py` fails locally with `RecoveryRequired`, and it does on clean dev too.
- One commit per task, with a conventional message ending in a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

- **Slow link, short tool budget, cache miss.** The call fails `OP_TIMEOUT` and the binding is not BLOCKED; the next call tries a session again. Pinned in Task 1 (`test_budget_expiry_mid_upload_fails_only_that_call`).
- **Host answered NEED, took the bundle, never sent READY** (a stuck loader). This must never be UNREACHABLE: budget → `OP_TIMEOUT`, cap → protocol. Pinned in Task 1 (`test_ready_stall_after_answer_is_never_unreachable`).
- **Several callers queued behind a budget-expired start.** Each gets its own attempt, because their budgets differ and the failure is not shared. Pinned in Task 1 (`test_budget_expired_start_is_not_shared_with_waiters`).
- **Idle reap, run end or app exit landing between a call's registration and its REQUEST write.** No tool error: a fresh session, or one-shot after the run ends. Pinned in Task 2 (`test_close_between_register_and_write_reacquires`).
- **A request that was written, then the session closed.** Never retried. Pinned by the existing `test_session_death_fails_inflight_and_marks_dead`, plus Task 2's `test_cancel_write_on_closed_stdin_is_not_session_closed`.

---

### Task 1: After the host answers, a start deadline is never transport-class (TASK-33420)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_session_worker.py`: `_handshake`, `_handshake_write`, and a new `_stalled_after_answer`
- Modify: `tldw_chatbook/Tools/remote_session_registry.py`: `acquire`'s transport branch
- Test: `Tests/Tools/test_remote_session_worker.py`, `Tests/Tools/test_remote_session_registry.py`, `Tests/Tools/test_remote_executor_ssh.py`

**Interfaces:**
- Produces: a start that runs out of the call's budget after the host answered raises `SessionStartError(True, TransportFailure(OP_TIMEOUT, None, "operation timed out"), "session start ran out of the call's budget")`. A deadline at the 30 s cap after the host answered raises `SessionStartError(False, None, …)`.

- [ ] **Step 1: Write the failing worker tests**

In `Tests/Tools/test_remote_session_worker.py`, next to `_NEED_NO_READ_HOST`, add a fake host that takes the bundle and never says READY:

```python
_NEED_THEN_SILENT_HOST = textwrap.dedent(
    """
    import os, sys, time
    loader_len, bundle_hash = int(sys.argv[1]), sys.argv[2]
    def read_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = os.read(0, n - len(buf))
            if not chunk:
                sys.exit(3)
            buf += chunk
        return buf
    read_exact(loader_len)
    header = b""
    while not header.endswith(b"\\n"):
        header += read_exact(1)
    os.write(1, b"TLDW-REMOTE-0001NEED " + bundle_hash.encode() + b"\\n")
    size = int.from_bytes(read_exact(4), "big")
    read_exact(size)
    time.sleep(30)
    """
)


def _need_then_silent_argv() -> list[str]:
    _, compressed, _ = _bundle_payload()
    return [
        sys.executable, "-c", _NEED_THEN_SILENT_HOST,
        str(len(loader_payload())), hashlib.sha256(compressed).hexdigest(),
    ]


def _need_no_read_argv() -> list[str]:
    _, compressed, _ = _bundle_payload()
    return [sys.executable, "-c", _NEED_NO_READ_HOST, hashlib.sha256(compressed).hexdigest()]


def test_budget_expiry_mid_upload_fails_only_that_call(worker_factory):
    """TASK-33420: the host answered NEED; the call's budget ran out mid-upload."""
    worker, _ = worker_factory(spawn_argv=_need_no_read_argv(), handshake_timeout=1.0)
    started = time.monotonic()
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert time.monotonic() - started < 8
    assert err.value.transport is True  # the call's result, no one-shot retry
    assert err.value.failure.kind is TransportFailureKind.OP_TIMEOUT


@pytest.mark.parametrize("budget_bounded", [True, False])
def test_ready_stall_after_answer_is_never_unreachable(worker_factory, monkeypatch, budget_bounded):
    """TASK-33420 AC #3: a READY that never comes after NEED is never UNREACHABLE."""
    if budget_bounded:
        worker, _ = worker_factory(spawn_argv=_need_then_silent_argv(), handshake_timeout=1.0)
    else:
        monkeypatch.setattr(worker_module, "_HANDSHAKE_TIMEOUT_S", 1.0)
        worker, _ = worker_factory(spawn_argv=_need_then_silent_argv())
    with pytest.raises(SessionStartError) as err:
        worker.start()
    if budget_bounded:
        assert err.value.transport is True
        assert err.value.failure.kind is TransportFailureKind.OP_TIMEOUT
    else:
        assert err.value.transport is False  # protocol-class: one-shot for the run
        assert err.value.failure is None or err.value.failure.kind is not TransportFailureKind.UNREACHABLE
```

`loader_payload` is already imported in that module. If `hashlib` or `textwrap` are not imported there yet, import them at the top with their siblings.

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -k "budget_expiry_mid_upload or ready_stall_after_answer" -v -p no:randomly`
Expected:
- `test_budget_expiry_mid_upload…` fails, because `transport is False` today (protocol "write stalled").
- Both `ready_stall` cases fail, because today the start is classified `UNREACHABLE` with `transport True`.
- The existing `test_bundle_write_is_bounded_by_the_handshake_deadline`, which exercises the cap and expects protocol "write stalled", still passes.

- [ ] **Step 3: Implement in `remote_session_worker.py`**

Add below `_fail_start`:

```python
    def _stalled_after_answer(self) -> NoReturn:
        """The host answered the handshake, then the deadline passed.

        A live channel proves reachability, so this is never transport-class
        (ADR-181, R8). When the deadline was the call's own budget (a slow
        cache-miss upload, a short tool timeout), this call fails as
        ``OP_TIMEOUT`` and the next call tries a session again. When it was
        the 30 s cap, the loader is stuck: protocol-class, one-shot for the
        run.
        """
        self._kill_and_reap()
        if self._handshake_limit < _HANDSHAKE_TIMEOUT_S:
            failure = TransportFailure(TransportFailureKind.OP_TIMEOUT, None, "operation timed out")
            raise SessionStartError(True, failure, "session start ran out of the call's budget")
        raise SessionStartError(False, None, "handshake stalled after the host answered")
```

Add `NoReturn` to the `typing` import.

Give `_handshake_write` an `answered` flag:

```python
    def _handshake_write(self, data: bytes, deadline: float, *, answered: bool = False) -> None:
        try:
            self._write(data, deadline)
        except OSError:
            pass  # the process died; the read below sees EOF and classifies it
        except (_WriteStalled, _NotSent):
            if answered:
                self._stalled_after_answer()
            self._kill_and_reap()
            raise SessionStartError(False, None, "handshake write stalled") from None
```

In `_handshake`, replace the cache-miss branch:

```python
            if not cache_hit:
                # The host answered NEED: from here a deadline is never
                # transport-class (TASK-33420).
                self._handshake_write(
                    len(compressed).to_bytes(4, "big") + compressed, deadline, answered=True
                )
                try:
                    line = self._read_handshake_line(deadline)
                except _HandshakeFailed as ended:
                    if ended.stalled:
                        self._stalled_after_answer()
                    raise
```

The outer `except _HandshakeFailed as ended: self._fail_start(ended)` still handles EOF and noise after NEED. Those are classified by exit code as before, because the process ended.

In the cap case the budget-bounded check reads `self._handshake_limit`, which `_handshake` sets before any write. The existing test that monkeypatches `_HANDSHAKE_TIMEOUT_S = 2.0` with no `handshake_timeout` gives `limit == _HANDSHAKE_TIMEOUT_S`, so it takes the cap branch and still gets "write stalled".

- [ ] **Step 4: Run the worker tests**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -p no:randomly -q`
Expected: all PASS.

- [ ] **Step 5: Write the failing registry test**

Append to `Tests/Tools/test_remote_session_registry.py`:

```python
def test_budget_expired_start_is_not_shared_with_waiters():
    """TASK-33420: OP_TIMEOUT is the first caller's budget, not a dead host."""
    reg = RemoteSessionRegistry()
    timeout = TransportFailure(TransportFailureKind.OP_TIMEOUT, None, "operation timed out")
    with pytest.raises(SessionStartError):
        reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=SessionStartError(True, timeout, "budget")))
    assert ("run-1", "b1") not in reg._start_failures
    assert ("run-1", "b1") not in reg._disabled
    assert reg.acquire(("run-1", "b1"), FakeWorker) is not None  # next call gets a session
```

Run it: it FAILS on the first `not in reg._start_failures`, because today every transport start failure is recorded for fan-out.

- [ ] **Step 6: Implement in `remote_session_registry.py`**

In `acquire`'s transport branch, record the fan-out entry only for real transport failures:

```python
                kind = error.failure.kind.value if error.failure else "unknown"
                logger.debug(f"ssh session worker start failed (transport): {kind}")
                budget_expired = (
                    error.failure is not None
                    and error.failure.kind is TransportFailureKind.OP_TIMEOUT
                )
                with self._lock:
                    # OP_TIMEOUT is this caller's budget running out on a live
                    # host: waiters have their own budgets, so it is not shared.
                    if not budget_expired and not (self._shutdown or key[0] in self._closed_keys):
                        self._start_failures[key] = (time.monotonic(), error)
                raise
```

Run: `$PY -m pytest Tests/Tools/test_remote_session_registry.py -p no:randomly -q`. Expected: all PASS.

- [ ] **Step 7: Write the failing executor test**

Append to `Tests/Tools/test_remote_executor_ssh.py` after the session tests:

```python
def test_budget_expiry_mid_upload_is_op_timeout_and_keeps_the_warm_path(
    env: SimpleNamespace, sessions: SimpleNamespace
) -> None:
    """TASK-33420: slow cache-miss upload + short budget -> OP_TIMEOUT, not BLOCKED, session next call."""
    import hashlib
    import subprocess

    from tldw_chatbook.Tools.remote_workspace_executor import _bundle_payload

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
```

If the first `env.read` is preceded by a one-shot identity ping, it goes through the session path too, which is fine. The assertions hold either way. If you find they don't, adapt only the counting and explain in your report.

Run it: it FAILS today at `pytest.raises`. The stalled upload is protocol-class, so the key is disabled and the call is retried one-shot, which the fake ssh serves successfully.

- [ ] **Step 8: Run all touched suites**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_executor_ssh.py Tests/Tools/test_remote_session_loopback.py -p no:randomly -q`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_worker.py tldw_chatbook/Tools/remote_session_registry.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_executor_ssh.py
git commit -m "fix: after the host answers, an SSH session start deadline is never transport-class (TASK-33420)"
```

---

### Task 2: A close before the REQUEST is written re-acquires (TASK-33421)

**Files:**
- Modify: `tldw_chatbook/Tools/remote_session_worker.py`: a new `_StdinClosed`, `_write`'s pre-check, and `call()`
- Test: `Tests/Tools/test_remote_session_worker.py`, `Tests/Tools/test_remote_executor_ssh.py`

**Interfaces:**
- Consumes: `SessionClosed` (existing, TASK-33401) and the executor's two-attempt re-acquire loop (existing).
- Produces: `_StdinClosed(BrokenPipeError)`, raised by `_write` only when stdin was already closed before any byte was written.

- [ ] **Step 1: Write the failing worker tests**

Append to `Tests/Tools/test_remote_session_worker.py`:

```python
def test_close_between_register_and_write_raises_session_closed(worker_factory, workspace, monkeypatch):
    """TASK-33421: registered, then retired before any REQUEST byte: nothing was sent."""
    worker, _ = worker_factory()
    worker.start()
    real_write = worker._write

    def close_first(data, deadline):
        worker.close()  # lands after registration, before the REQUEST write
        return real_write(data, deadline)

    monkeypatch.setattr(worker, "_write", close_first)
    with pytest.raises(worker_module.SessionClosed):
        worker.call(read_request(workspace, "a.txt"), budget=10)
    assert worker._pending == {}


def test_cancel_write_on_closed_stdin_is_not_session_closed(worker_factory, workspace, monkeypatch):
    """A REQUEST that was written is never retried, even if a later CANCEL finds stdin closed."""
    worker, _ = worker_factory()
    worker.start()
    real_write = worker._write
    calls = {"n": 0}

    def close_before_cancel(data, deadline):
        calls["n"] += 1
        if calls["n"] == 2:  # the CANCEL: the REQUEST already went out
            worker.close()
        return real_write(data, deadline)

    monkeypatch.setattr(worker, "_write", close_before_cancel)
    # The request's own budget (30 s) outlives the call's deadline
    # (0.2 s + grace 1.0 s), so the laptop CANCELs before the watchdog ends it.
    result = worker.call(grep_request(workspace, "(a+)+$", budget=30), budget=0.2)
    assert calls["n"] == 2, "the CANCEL write was never reached"
    assert result.failure is not None  # a classified failure, never SessionClosed
```

Confirm this against the real code: `grep_request`'s `budget` is the request's own timeout, and the call's deadline is `(admitted_at or sent_at) + budget + grace`. If the CANCEL write is still not reached, pick the smallest change that makes it deterministic and say so in your report. Do not drop the `calls["n"] == 2` assertion.

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py -k "between_register_and_write or cancel_write_on_closed" -v -p no:randomly`
Expected: the first test FAILS, because today it returns `REMOTE_OP_FAILED` rather than raising. The second test PASSES already and pins the rule that must not break.

- [ ] **Step 3: Implement in `remote_session_worker.py`**

Next to `_WriteStalled`/`_NotSent`:

```python
class _StdinClosed(BrokenPipeError):
    """Internal: stdin was already closed (by ``close()``) before any byte
    of this write went out — distinct from a mid-write EPIPE."""
```

In `_write`, change the pre-check line `raise BrokenPipeError("session stdin closed")` to `raise _StdinClosed("session stdin closed")`. Its docstring gains `_StdinClosed: stdin was closed before anything was written.`

In `call()`, wrap only the REQUEST write:

```python
        try:
            try:
                self._write(encode_frame(REQUEST, request_id, request_bytes), sent_at + budget + grace)
            except _StdinClosed:
                if self._retired:
                    # close() retired this healthy session after the request
                    # registered but before any REQUEST byte was written:
                    # nothing ran, so the caller may ask again (TASK-33421).
                    raise SessionClosed() from None
                raise
            while not pending.done.is_set():
                ...  (unchanged)
```

`SessionClosed` is not an `OSError`, so it propagates past the existing `except` clauses. The `finally` still pops the pending entry. A `_StdinClosed` on the CANCEL write is still a `BrokenPipeError`, so it goes down the existing `(OSError, ValueError)` path unchanged. Update `call()`'s docstring with one sentence on this.

- [ ] **Step 4: Write the failing executor test**

Append to `Tests/Tools/test_remote_executor_ssh.py`:

```python
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
```

The session is popped before the close, as in a real reap, so the re-acquire starts a fresh session without spending the run's single restart. If the executor's first call on a fresh executor is a one-shot identity ping, `spawns` still counts only session starts. If the real code contradicts this expectation, report NEEDS_CONTEXT rather than bending the assertion.

- [ ] **Step 5: Run all touched suites and commit**

Run: `$PY -m pytest Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_session_loopback.py -p no:randomly -q`
Expected: all PASS.

```bash
git add tldw_chatbook/Tools/remote_session_worker.py Tests/Tools/test_remote_session_worker.py Tests/Tools/test_remote_executor_ssh.py
git commit -m "fix: a call whose SSH session closes before its request is written re-acquires (TASK-33421)"
```

---

### Task 3: ADR-181, CHANGELOG, close-out

**Files:** `backlog/decisions/181-ssh-remote-workspace-bindings.md`, `CHANGELOG.md`, `Docs/security/production-diagnostic-inventory.json` (only if preflight reports drift), `backlog/tasks/task-33420 - *.md`, `backlog/tasks/task-33421 - *.md`

- [ ] **Step 1: ADR-181**

In `## Amendment 2026-09-27` → "Follow-ups (2026-09-28, TASK-33400–33408)":
- Rewrite the handshake bullet so it states the new rule. The handshake gives up at the call's budget + grace (capped at 30 s). Before the host answers, a stall is classified like a one-shot handshake (transport). Once it has answered (NEED or READY), the deadline is never transport-class:
  - running out of the call's budget fails only that call as `OP_TIMEOUT` (status-preserving) and the next call tries a session again;
  - the 30 s cap is protocol-class (one-shot for the run).
  - Drop the old "except when the budget runs out mid bundle-upload" qualification.
- Remove the "Known limits (follow-ups TASK-33420, TASK-33421)" bullet. Replace it with one sentence under the failure-classification row about calls closed before being sent: that row now covers a close landing after the call registered but before any REQUEST byte was written (TASK-33421).

- [ ] **Step 2: CHANGELOG**

Under `## [Unreleased]` → `### Fixed`, add two bullets in the file's existing style:
- "SSH workspaces: when a slow link or a short tool timeout runs out during a session's first upload, only that call times out; the binding is no longer marked unreachable or switched to the slower path for the rest of the run."
- "SSH workspaces: a tool call whose session was closed (idle, run end or app exit) just before it was sent now gets a fresh session instead of an error."

- [ ] **Step 3: Close out TASK-33420 and TASK-33421**

For each task:
- tick every AC;
- add `## Implementation Plan` pointing to this plan's task (`See Docs/superpowers/plans/2026-09-29-ssh-session-followups-2.md, Task 1` / `Task 2`);
- add `## Implementation Notes` (2–5 lines: approach, files, tests, deviations);
- set `status: Done`, `updated_date: '2026-09-29 HH:MM'` (current time) and `assignee:` to `- '@claude'`.

Keep the section markers exact, and make sure `backlog task 33420 --plain` and `backlog task 33421 --plain` both parse and show Done.

- [ ] **Step 4: Gates and commit**

```bash
$PY -m pytest Tests/Tools/ -p no:randomly -q -k "remote or ssh or session"
PATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin:$PATH ./scripts/preflight.sh
git add backlog/decisions/181-ssh-remote-workspace-bindings.md CHANGELOG.md backlog/tasks/task-3342*.md Docs/security/production-diagnostic-inventory.json
git commit -m "docs: ADR-181 handshake deadline and close-before-send rules; close TASK-33420, TASK-33421"
```

If preflight reports inventory drift, read the rows first. Only this branch's own static logger lines may be regenerated.
