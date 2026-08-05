# Skills Script Execution (trust-gated) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a trusted skill's bundled scripts actually execute — via a new `run_skill_script` runtime tool gated by fail-closed policy, per-run trust re-verification, and an in-chat human confirm with a digest-pinned "always allow this skill" grant — returning capped stdout/stderr to the agent.

**Architecture:** Six seams, mirroring the five merged skills layers. A new pure `skill_script_runner.py` owns the sandboxed subprocess (no skills/trust/policy knowledge). `LocalSkillsService` gains `describe_skill_script` (read-only resolve for the confirm card) and `run_skill_script` (authoritative enforce → trust → resolve → run). `SkillsScopeService` adds local-only passthroughs plus a public `enforce_run_script()`. `SkillTrustService` gains a digest-pinned grant store. The Agents runtime gains a 6th runtime tool wired unconditionally (caller identity is NOT a gate). The bridge closure sequences enforce → describe → grant-check → confirm → run, and the controller/UI clone the install-confirm HITL card.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest, `subprocess` + `resource` (stdlib), `pydantic` (existing skills schemas).

**Spec:** `Docs/superpowers/specs/2026-07-24-skills-script-execution-design.md` (read it before Task 1).

## Global Constraints

- **Worktree/cwd (STEP ZERO for every task):** work ONLY in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/skills-script-execution` on branch `worktree-skills-script-execution`. Subagents start in the MAIN checkout — `cd` there and verify `git rev-parse --abbrev-ref HEAD` prints `worktree-skills-script-execution` BEFORE any edit. Never `git checkout` this plan file or `.superpowers/sdd/progress.md`.
- **Tests run venv-only:** `source .venv/bin/activate` first. The `timeout` command does not exist in this environment. Never leave a test run in the background and stop — run it in the foreground and wait.
- **TDD, non-vacuously.** Write the failing test, RUN it and see it fail for the stated reason, then implement. A test that passes before the implementation is a broken test — fix the test, not the assertion.
- **Docstrings:** Google style with Args/Returns/Raises on every new public callable (CLAUDE.md-mandated; Qodo flags this on every PR in this program).
- **No `preexec_fn`.** The Agents runtime runs sync on a worker thread bridged by `asyncio.run` — arbitrary-Python `preexec_fn` across fork/exec can deadlock there. Use `start_new_session=True` + the Python trampoline (Task 1).
- **Never `communicate()`/`capture_output=True`** for script output — it buffers to EOF (OOM). Bounded streaming reader only.
- **Never set `RLIMIT_NPROC`** — it is per-real-UID, not per-tree; an absolute cap breaks `fork()` on any busy desktop.
- **Never `shell=True`;** argv is always a list.
- **Scratch cwd is never the skill directory.**
- Interpreter lookup uses `shutil.which(name, path=_SCRUBBED_PATH)` — never `os.environ`.
- **Fail closed everywhere:** any confirm exception, missing UI, unknown policy id, or trust doubt ⇒ deny.
- Commit after every task. Conventional-commit subjects; end bodies with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Skills_Interop/skill_script_runner.py` (**new**) | `ScriptRunLimits`, `ScriptRunResult`, `run_script_subprocess()` — the sandboxed subprocess. Knows nothing about skills/trust/policy. |
| `tldw_chatbook/Skills_Interop/local_skills_service.py` | `describe_skill_script()` (read-only resolve+classify), `run_skill_script()` (authoritative). |
| `tldw_chatbook/Skills_Interop/skills_scope_service.py` | Local-only passthroughs + public `enforce_run_script()`. |
| `tldw_chatbook/Skills_Interop/skill_trust_service.py` | Digest-pinned script-grant store. |
| `tldw_chatbook/runtime_policy/registry.py` | `skills.run_script` resource row. |
| `tldw_chatbook/Agents/agent_models.py`, `tool_catalog.py`, `agent_runtime.py`, `agent_service.py` | 6th runtime tool: name, schema, LoopDeps field + dispatch, unconditional wiring. |
| `tldw_chatbook/Library/library_skills_state.py` | Drift-guard name. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | The closure (enforce → describe → grant → confirm → run). |
| `tldw_chatbook/Chat/console_chat_controller.py` | HITL: request/marshal/resolve/deny-on-context-change. |
| `tldw_chatbook/Widgets/Chat_Widgets/skill_script_confirm_card.py` (**new**), `chat_task_cards.py`, `UI/Screens/chat_screen_state.py`, `UI/Screens/chat_screen.py` | Confirm card + state + wiring. |
| `tldw_chatbook/Widgets/Library/library_skills_canvas.py`, `UI/Screens/library_screen.py` | Grant visibility + revoke in the skills trust panel. |

---

### Task 1: Sandboxed script runner

**Files:**
- Create: `tldw_chatbook/Skills_Interop/skill_script_runner.py`
- Test: `Tests/Skills/test_skill_script_runner.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `SCRUBBED_PATH: str = "/usr/bin:/bin"`
  - `@dataclass(frozen=True) ScriptRunLimits(cpu_seconds: int = 10, address_space_bytes: int = 512*1024*1024, open_files: int = 128, file_size_bytes: int = 8*1024*1024, wall_clock_seconds: float = 60.0, output_cap_bytes: int = 65536)`
  - `@dataclass(frozen=True) ScriptRunResult(exit_code: int | None, stdout: str, stderr: str, timed_out: bool, output_capped: bool, duration_seconds: float, truncated_stdout: bool, truncated_stderr: bool, sandbox_warnings: tuple[str, ...])`
  - `def run_script_subprocess(target_argv: list[str], *, cwd: Path, limits: ScriptRunLimits) -> ScriptRunResult`
  - `def resolve_interpreter(name: str) -> str | None`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Skills/test_skill_script_runner.py`:

```python
"""Runner unit tests: real subprocesses, real limits, no mocks."""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop.skill_script_runner import (
    ScriptRunLimits,
    resolve_interpreter,
    run_script_subprocess,
)


def _script(tmp_path: Path, body: str, name: str = "s.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_captures_stdout_and_exit_code(tmp_path):
    script = _script(tmp_path, "print('hello from script')")
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 0
    assert "hello from script" in result.stdout
    assert result.timed_out is False


def test_nonzero_exit_is_reported_not_raised(tmp_path):
    script = _script(tmp_path, "import sys; sys.stderr.write('boom'); sys.exit(3)")
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 3
    assert "boom" in result.stderr


def test_wall_clock_timeout_kills_whole_process_group(tmp_path):
    """A grandchild must die too: proves start_new_session + killpg."""
    marker = tmp_path / "grandchild.pid"
    script = _script(
        tmp_path,
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        open({str(marker)!r}, 'w').write(str(child.pid))
        time.sleep(120)
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(wall_clock_seconds=2.0),
    )
    assert result.timed_out is True
    grandchild_pid = int(marker.read_text())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("grandchild survived the process-group kill")


def test_runaway_output_is_capped_and_does_not_buffer_to_eof(tmp_path):
    """OOM safety: an infinite writer returns BOUNDED output and still terminates.

    The child writes far more than the cap; the runner must retain only ~cap
    bytes (never buffering to EOF) and must come back at the wall clock rather
    than hanging on a full pipe.
    """
    script = _script(
        tmp_path,
        """
        import sys
        chunk = 'x' * 65536
        while True:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        """,
    )
    started = time.monotonic()
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(output_cap_bytes=4096, wall_clock_seconds=5.0),
    )
    elapsed = time.monotonic() - started
    assert result.truncated_stdout is True
    assert result.output_capped is True
    assert len(result.stdout) <= 4096, "retained output must not exceed the cap"
    assert result.timed_out is True
    assert elapsed < 20.0, "must return at the wall clock, not hang on a full pipe"


def test_chatty_but_finite_script_still_completes(tmp_path):
    """Past-the-cap output is discarded, not deadlocked: exit code survives."""
    script = _script(
        tmp_path,
        """
        import sys
        for _ in range(50):
            sys.stdout.write('y' * 4096)
        sys.stdout.flush()
        sys.exit(0)
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(output_cap_bytes=4096, wall_clock_seconds=30.0),
    )
    assert result.exit_code == 0, "a chatty but finite script must not be killed"
    assert result.timed_out is False
    assert result.truncated_stdout is True
    assert len(result.stdout) <= 4096


def test_environment_is_scrubbed(tmp_path):
    os.environ["TLDW_FAKE_API_KEY"] = "super-secret"
    try:
        script = _script(
            tmp_path,
            "import os; print(os.environ.get('TLDW_FAKE_API_KEY', 'ABSENT')); print(os.environ['PATH'])",
        )
        result = run_script_subprocess(
            [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
        )
    finally:
        os.environ.pop("TLDW_FAKE_API_KEY", None)
    assert "super-secret" not in result.stdout
    assert "ABSENT" in result.stdout
    assert "/usr/bin:/bin" in result.stdout


def test_cwd_is_the_supplied_scratch_dir(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script = _script(tmp_path, "import os; print(os.path.realpath(os.getcwd()))")
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=scratch, limits=ScriptRunLimits()
    )
    assert str(scratch.resolve()) in result.stdout


def test_file_size_limit_is_enforced(tmp_path):
    script = _script(
        tmp_path,
        """
        try:
            with open('big.bin', 'wb') as handle:
                handle.write(b'x' * (5 * 1024 * 1024))
                handle.flush()
            print('WROTE')
        except Exception:
            print('BLOCKED')
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(file_size_bytes=64 * 1024),
    )
    assert "WROTE" not in result.stdout


def test_moderate_forking_is_not_blocked(tmp_path):
    """RLIMIT_NPROC must NOT be set (it is per-UID and breaks busy desktops)."""
    script = _script(
        tmp_path,
        """
        import subprocess, sys
        for _ in range(5):
            subprocess.run([sys.executable, '-c', 'pass'], check=True)
        print('FORKED OK')
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert "FORKED OK" in result.stdout


def test_limits_apply_to_a_non_python_target(tmp_path):
    """The trampoline must bound direct-exec/shell targets too, not just Python."""
    script = _script(tmp_path, "echo shell-ran\n", name="s.sh")
    result = run_script_subprocess(
        ["/bin/sh", str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 0
    assert "shell-ran" in result.stdout


def test_binary_output_does_not_crash_the_runner(tmp_path):
    script = _script(
        tmp_path,
        "import sys; sys.stdout.buffer.write(b'\\xff\\xfe\\x00binary'); sys.stdout.flush()",
    )
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 0
    assert isinstance(result.stdout, str)


def test_resolve_interpreter_uses_scrubbed_path_only():
    assert resolve_interpreter("sh") == "/bin/sh"
    assert resolve_interpreter("definitely-not-a-real-interpreter-xyz") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_skill_script_runner.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Skills_Interop.skill_script_runner'`

- [ ] **Step 3: Implement the runner**

Create `tldw_chatbook/Skills_Interop/skill_script_runner.py`:

```python
"""Sandboxed subprocess execution for skill-bundled scripts.

The single place this app spawns a skill's own code. It knows nothing about
skills, trust, or policy — callers resolve and authorize a script, then hand
this module an argv to run under best-effort containment.

Three deliberate divergences from ``Evals/specialized_runners.py`` (whose
limit VALUES this borrows):

1. No ``preexec_fn``. The Agents runtime runs synchronously on a worker
   thread and bridges tool calls through ``asyncio.run``; running arbitrary
   Python between fork and exec in a multi-threaded process can deadlock.
   ``start_new_session=True`` does the session/process-group setup inside
   CPython's own C fork helper, and the resource limits are applied by a
   Python *trampoline* that ``setrlimit``s in a fresh single-threaded
   process and then ``os.execv``s the real target.
2. No ``communicate()``/``capture_output``. Those read to EOF into memory,
   so a script that spews output OOMs the app before any cap applies. A
   bounded reader thread per stream stops at ``output_cap_bytes``.
3. No ``RLIMIT_NPROC``. It is enforced per real-UID across the whole
   session, not per process tree, so an absolute cap makes the child's
   first fork fail on any desktop that already exceeds it.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

SCRUBBED_PATH = "/usr/bin:/bin"

_TRAMPOLINE = """
import os, resource, sys
cpu, addr_space, nofile, fsize = (int(v) for v in sys.argv[1:5])
target = sys.argv[5:]
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
try:
    resource.setrlimit(resource.RLIMIT_AS, (addr_space, addr_space))
except (ValueError, OSError):
    pass  # Darwin/BSD alias RLIMIT_AS to RSS and refuse to lower it.
os.execv(target[0], target)
"""


@dataclass(frozen=True)
class ScriptRunLimits:
    """Best-effort containment budget for one script run."""

    cpu_seconds: int = 10
    address_space_bytes: int = 512 * 1024 * 1024
    open_files: int = 128
    file_size_bytes: int = 8 * 1024 * 1024
    wall_clock_seconds: float = 60.0
    output_cap_bytes: int = 65536


@dataclass(frozen=True)
class ScriptRunResult:
    """Outcome of one sandboxed script run."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_capped: bool
    duration_seconds: float
    truncated_stdout: bool
    truncated_stderr: bool
    sandbox_warnings: tuple[str, ...] = field(default=())


def memory_limit_enforced() -> bool:
    """Return whether RLIMIT_AS can actually cap memory on this platform.

    Returns:
        False on macOS/BSD, where ``setrlimit(RLIMIT_AS, ...)`` raises and the
        memory cap silently does not apply.
    """
    return platform.system() != "Darwin"


def resolve_interpreter(name: str) -> str | None:
    """Resolve an interpreter against the SCRUBBED PATH, never ``os.environ``.

    Args:
        name: Interpreter name (``python3``) or absolute path (``/bin/sh``).

    Returns:
        The absolute path, or None when it does not resolve on the scrubbed
        PATH (the caller surfaces that as an unavailable mechanism rather
        than falling back to the user's environment).
    """
    if os.path.isabs(name):
        return name if os.path.exists(name) else None
    return shutil.which(name, path=SCRUBBED_PATH)


def _scrubbed_env(cwd: Path) -> dict[str, str]:
    env = {
        "PATH": SCRUBBED_PATH,
        "HOME": str(cwd),
        "TMPDIR": str(cwd),
    }
    for passthrough in ("LANG", "LC_ALL"):
        value = os.environ.get(passthrough)
        if value:
            env[passthrough] = value
    return env


def _read_capped(stream, cap: int, sink: dict) -> None:
    """Read a stream to EOF while KEEPING at most ``cap`` bytes.

    Reading deliberately continues past the cap, discarding the excess, so the
    child never blocks writing into a pipe nobody drains. Memory stays bounded
    at ``cap`` (the OOM property) while a chatty but well-behaved script can
    still run to completion; a script that never stops is bounded by the wall
    clock instead.

    Args:
        stream: The child's stdout/stderr pipe, in binary mode.
        cap: Maximum bytes to retain.
        sink: Mutable dict receiving ``{"data": bytes, "capped": bool}``.
    """
    chunks: list[bytes] = []
    kept = 0
    total = 0
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            total += len(chunk)
            if kept < cap:
                room = cap - kept
                chunks.append(chunk[:room])
                kept += min(len(chunk), room)
    except (OSError, ValueError):
        pass
    finally:
        sink["data"] = b"".join(chunks)
        sink["capped"] = total > cap


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def run_script_subprocess(
    target_argv: list[str],
    *,
    cwd: Path,
    limits: ScriptRunLimits,
) -> ScriptRunResult:
    """Run ``target_argv`` under best-effort containment and capped output.

    Args:
        target_argv: Full argv of the real target (interpreter + script +
            args, or an executable + args). Never passed to a shell.
        cwd: Scratch working directory (the caller guarantees this is not the
            skill directory).
        limits: Resource/time/output budget.

    Returns:
        A ScriptRunResult. A non-zero exit or a timeout is a normal result,
        not an exception.

    Raises:
        OSError: The target could not be spawned at all.
    """
    warnings: list[str] = []
    if not memory_limit_enforced():
        warnings.append(
            "memory (RLIMIT_AS) is not enforced on macOS/BSD; this script is "
            "bounded by CPU and wall-clock time but not by peak memory"
        )

    argv = [
        sys.executable,
        "-c",
        _TRAMPOLINE,
        str(limits.cpu_seconds),
        str(limits.address_space_bytes),
        str(limits.open_files),
        str(limits.file_size_bytes),
        *target_argv,
    ]
    started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 — argv list, shell=False, scrubbed env
        argv,
        cwd=str(cwd),
        env=_scrubbed_env(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    out_sink: dict = {"data": b"", "capped": False}
    err_sink: dict = {"data": b"", "capped": False}
    readers = [
        threading.Thread(
            target=_read_capped,
            args=(process.stdout, limits.output_cap_bytes, out_sink),
            daemon=True,
        ),
        threading.Thread(
            target=_read_capped,
            args=(process.stderr, limits.output_cap_bytes, err_sink),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    deadline = started + limits.wall_clock_seconds
    while True:
        if process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.02)

    if process.poll() is None:
        _kill_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("skill script did not reap after SIGKILL")
    for reader in readers:
        reader.join(timeout=2)
    for stream in (process.stdout, process.stderr):
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    return ScriptRunResult(
        exit_code=process.returncode,
        stdout=out_sink["data"].decode("utf-8", errors="replace"),
        stderr=err_sink["data"].decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_capped=bool(out_sink["capped"] or err_sink["capped"]),
        duration_seconds=time.monotonic() - started,
        truncated_stdout=bool(out_sink["capped"]),
        truncated_stderr=bool(err_sink["capped"]),
        sandbox_warnings=tuple(warnings),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_skill_script_runner.py -q`
Expected: PASS (12 tests). If the process-group test is flaky on a loaded machine, raise its wait deadline — do NOT weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_script_runner.py Tests/Skills/test_skill_script_runner.py
git commit -m "feat(skills): add sandboxed script runner

Trampoline-applied RLIMITs (no preexec_fn from the worker thread),
start_new_session + process-group kill, bounded streaming output reader
(no communicate() OOM), scrubbed env, scratch cwd. RLIMIT_NPROC
deliberately omitted (per-UID, breaks fork on busy desktops).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Policy row + trust grant store

**Files:**
- Modify: `tldw_chatbook/runtime_policy/registry.py` (skills capability block, ~:1015-1022)
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_service.py`
- Test: `Tests/Skills/test_skill_script_grants.py`

**Interfaces:**
- Consumes: `SkillTrustService._fingerprints_digest`, `_scan_skill`, `_normalize_skill_name` (existing private helpers).
- Produces (on `SkillTrustService`):
  - `current_fingerprint_digest(skill_name: str) -> str`
  - `script_grant_digest(skill_name: str) -> str | None`
  - `grant_script_execution(skill_name: str) -> None`
  - `revoke_script_execution(skill_name: str) -> None`
  - `script_execution_granted(skill_name: str) -> bool`
- Produces (policy): action id `skills.run_script.launch.local`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Skills/test_skill_script_grants.py`:

```python
"""Digest-pinned 'always allow scripts' grants + the run_script policy row."""

import pytest

from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY, get_capability_entry
from tldw_chatbook.runtime_policy.types import PolicyDeniedError


def test_run_script_policy_row_exists():
    assert "skills.run_script.launch.local" in CAPABILITY_REGISTRY
    entry = get_capability_entry("skills.run_script.launch.local")
    assert entry is not None


def test_unknown_action_id_still_fails_closed():
    with pytest.raises(PolicyDeniedError):
        get_capability_entry("skills.run_script.launch.nonsense")


def test_grant_records_current_digest(trust_service_with_skill):
    service, name = trust_service_with_skill
    assert service.script_execution_granted(name) is False
    service.grant_script_execution(name)
    assert service.script_execution_granted(name) is True
    assert service.script_grant_digest(name) == service.current_fingerprint_digest(name)


def test_grant_is_invalidated_when_content_changes(trust_service_with_skill, tmp_path):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    assert service.script_execution_granted(name) is True
    (service.skills_dir / name / "scripts" / "hello.py").write_text(
        "print('mutated')", encoding="utf-8"
    )
    assert service.script_execution_granted(name) is False, (
        "a content change must drop the standing grant back to per-run confirm"
    )


def test_revoke_clears_the_grant(trust_service_with_skill):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    service.revoke_script_execution(name)
    assert service.script_execution_granted(name) is False
    assert service.script_grant_digest(name) is None


def test_grant_persists_across_a_fresh_service_instance(
    trust_service_with_skill, make_trust_service
):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    reloaded = make_trust_service()
    assert reloaded.script_execution_granted(name) is True
```

Add the fixtures to `Tests/Skills/conftest.py` (create the file if absent; if it exists, append and reuse its existing trust-store construction helpers instead of duplicating them):

```python
import pytest

from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore


@pytest.fixture
def make_trust_service(tmp_path):
    """Build SkillTrustService instances sharing one on-disk store."""
    skills_dir = tmp_path / "skills"
    trust_dir = tmp_path / "trust"
    skills_dir.mkdir(exist_ok=True)
    trust_dir.mkdir(exist_ok=True)

    def _make():
        return SkillTrustService(
            skills_dir=skills_dir,
            trust_store=SkillTrustStore(store_dir=trust_dir),
        )

    return _make


@pytest.fixture
def trust_service_with_skill(make_trust_service):
    service = make_trust_service()
    name = "demo-skill"
    skill_dir = service.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text(
        "print('hello')", encoding="utf-8"
    )
    return service, name
```

> **Implementer note:** `SkillTrustStore` may require a `marker_store=` argument. Check its dataclass signature (`skill_trust_store.py:302`) and pass whatever the existing `Tests/Skills` suite passes — reuse the established test construction pattern rather than inventing one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_skill_script_grants.py -x -q`
Expected: FAIL — the policy id is missing from `CAPABILITY_REGISTRY`, and `SkillTrustService` has no `grant_script_execution`.

- [ ] **Step 3a: Add the policy row**

In `tldw_chatbook/runtime_policy/registry.py`, inside the `server_skills` capability's `resources=(...)` tuple, immediately after the `skills.install_remote` line:

```python
            _resource("skills.install_remote", actions=(LAUNCH,)),
            _resource("skills.run_script", actions=(LAUNCH,)),
```

- [ ] **Step 3b: Add the grant store to `SkillTrustService`**

Add the filename constant near the module's other constants:

```python
_SCRIPT_GRANTS_FILENAME = "skill_script_grants.json"
```

Add these methods to `SkillTrustService` (place them after `trust_current_skill`):

```python
    def _script_grants_path(self) -> Path:
        """Return the local-only script-grant sidecar path.

        Deliberately a sibling of the trust manifest rather than a field
        inside it: granting a run must never perturb the MAC'd fingerprint
        material that trust review depends on.

        Returns:
            Path to the grant sidecar (may not exist yet).
        """
        return self.trust_store.store_dir / _SCRIPT_GRANTS_FILENAME

    def _load_script_grants(self) -> dict[str, str]:
        path = self._script_grants_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in data.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def _save_script_grants(self, grants: dict[str, str]) -> None:
        path = self._script_grants_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(grants, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def current_fingerprint_digest(self, skill_name: str) -> str:
        """Return the digest of a skill's live on-disk fingerprints.

        Args:
            skill_name: Skill name (normalized internally).

        Returns:
            The same digest value that invalidates a captured trust review,
            so any content change also invalidates a script grant.
        """
        normalized = self._normalize_skill_name(skill_name)
        return self._fingerprints_digest(self._scan_skill(normalized))

    def script_grant_digest(self, skill_name: str) -> str | None:
        """Return the fingerprint digest a script grant was pinned to.

        Args:
            skill_name: Skill name (normalized internally).

        Returns:
            The pinned digest, or None when no grant is recorded.
        """
        normalized = self._normalize_skill_name(skill_name)
        return self._load_script_grants().get(normalized)

    def grant_script_execution(self, skill_name: str) -> None:
        """Record an 'always allow scripts' grant pinned to current content.

        Args:
            skill_name: Skill name (normalized internally).
        """
        normalized = self._normalize_skill_name(skill_name)
        grants = self._load_script_grants()
        grants[normalized] = self.current_fingerprint_digest(normalized)
        self._save_script_grants(grants)

    def revoke_script_execution(self, skill_name: str) -> None:
        """Drop any standing script grant for a skill.

        Args:
            skill_name: Skill name (normalized internally).
        """
        normalized = self._normalize_skill_name(skill_name)
        grants = self._load_script_grants()
        if grants.pop(normalized, None) is not None:
            self._save_script_grants(grants)

    def script_execution_granted(self, skill_name: str) -> bool:
        """Return whether scripts may run for this skill without a prompt.

        The grant is honoured only while the skill's content still matches the
        digest it was pinned to; any change (which already forces a trust
        re-review) drops it back to per-run confirmation.

        Args:
            skill_name: Skill name (normalized internally).

        Returns:
            True only when a grant exists AND still matches live content.
        """
        granted = self.script_grant_digest(skill_name)
        if not granted:
            return False
        try:
            return granted == self.current_fingerprint_digest(skill_name)
        except Exception:  # noqa: BLE001 — unreadable content ⇒ no grant
            return False
```

Ensure `json` and `Path` are imported at module scope (add if missing).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_skill_script_grants.py -q && pytest Tests/ -q -k "policy_registry or registry_completeness"`
Expected: PASS. The registry-validation tests must still pass — the new resource must satisfy `validate_registry_completeness()` (it runs at import, so an import error here means the row needs whatever fields its siblings have).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy/registry.py tldw_chatbook/Skills_Interop/skill_trust_service.py Tests/Skills/test_skill_script_grants.py Tests/Skills/conftest.py
git commit -m "feat(skills): add run_script policy row and digest-pinned script grants

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Service seams — describe + run

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Test: `Tests/Skills/test_skill_script_service.py`

**Interfaces:**
- Consumes: Task 1's `ScriptRunLimits`/`ScriptRunResult`/`run_script_subprocess`/`resolve_interpreter`; existing `_enforce`, `_require_trusted_skill`, `_skill_dir`, `_read_bundle_manifest`, `get_safe_relative_path`, `validate_supporting_file_path`.
- Produces:
  - `@dataclass(frozen=True) ScriptPlan(skill_name: str, script_path: str, mechanism: str, interpreter_display: str, is_binary: bool)` — `mechanism` is `"direct-exec"` or `"interpreter"`.
  - `LocalSkillsService.describe_skill_script(skill_name, script_path) -> ScriptPlan` (async)
  - `LocalSkillsService.run_skill_script(skill_name, script_path, args, *, limits=None) -> ScriptRunResult` (async)
  - `SkillsScopeService.describe_skill_script(...)`, `.run_skill_script(...)` (local-only), `.enforce_run_script()`
  - Constant `_INTERPRETER_MAP = {".py": "python3", ".sh": "sh", ".bash": "bash", ".js": "node"}`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Skills/test_skill_script_service.py`:

```python
"""Trust + resolution discipline for the script-execution seams."""

import os
import stat

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


@pytest.mark.asyncio
async def test_runs_a_text_script_via_the_interpreter_map(script_service):
    service, name = script_service
    result = await service.run_skill_script(name, "scripts/hello.py", [])
    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_passes_args_through(script_service):
    service, name = script_service
    result = await service.run_skill_script(name, "scripts/echo_args.py", ["a", "b"])
    assert "a|b" in result.stdout


@pytest.mark.asyncio
async def test_exec_bit_file_runs_direct(script_service):
    service, name = script_service
    path = service._skill_dir(name) / "scripts" / "direct.sh"
    path.write_text("#!/bin/sh\necho direct-ran\n", encoding="utf-8")
    os.chmod(path, path.stat().st_mode | stat.S_IXUSR)
    plan = await service.describe_skill_script(name, "scripts/direct.sh")
    assert plan.mechanism == "direct-exec"
    result = await service.run_skill_script(name, "scripts/direct.sh", [])
    assert "direct-ran" in result.stdout


@pytest.mark.asyncio
async def test_untrusted_skill_refuses_without_spawning(script_service_untrusted):
    service, name = script_service_untrusted
    with pytest.raises(SkillTrustBlockedError):
        await service.run_skill_script(name, "scripts/hello.py", [])


@pytest.mark.asyncio
async def test_describe_also_refuses_when_untrusted(script_service_untrusted):
    service, name = script_service_untrusted
    with pytest.raises(SkillTrustBlockedError):
        await service.describe_skill_script(name, "scripts/hello.py")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_path",
    ["../outside.py", "/etc/passwd", "SKILL.md", "scripts/missing.py"],
)
async def test_bad_paths_are_rejected(script_service, bad_path):
    service, name = script_service
    with pytest.raises(ValueError):
        await service.run_skill_script(name, bad_path, [])


@pytest.mark.asyncio
async def test_symlink_escape_is_indistinguishable_from_missing(script_service, tmp_path):
    """Symlink-oracle safety: existing vs missing targets give the SAME error."""
    service, name = script_service
    scripts = service._skill_dir(name) / "scripts"
    real_target = tmp_path / "real_outside.py"
    real_target.write_text("print('outside')", encoding="utf-8")
    (scripts / "to_existing.py").symlink_to(real_target)
    (scripts / "to_missing.py").symlink_to(tmp_path / "nope.py")

    errors = []
    for link in ("scripts/to_existing.py", "scripts/to_missing.py"):
        with pytest.raises(ValueError) as excinfo:
            await service.run_skill_script(name, link, [])
        errors.append(str(excinfo.value))
    assert errors[0] == errors[1], "symlink target existence must not leak"


@pytest.mark.asyncio
async def test_unrunnable_type_errors_clearly(script_service):
    service, name = script_service
    (service._skill_dir(name) / "notes.txt").write_text("just text", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        await service.run_skill_script(name, "notes.txt", [])
    assert "unrunnable_script_type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_script_cannot_write_into_its_own_bundle(script_service):
    """Scratch cwd, not the skill dir — a script must not tamper its fingerprints."""
    service, name = script_service
    skill_dir = service._skill_dir(name)
    (skill_dir / "scripts" / "writer.py").write_text(
        "open('tampered.txt', 'w').write('x'); print('wrote')", encoding="utf-8"
    )
    result = await service.run_skill_script(name, "scripts/writer.py", [])
    assert "wrote" in result.stdout
    assert not (skill_dir / "tampered.txt").exists()


@pytest.mark.asyncio
async def test_scratch_root_config_knob_is_reachable(script_service, tmp_path, monkeypatch):
    """The 3-arg get_cli_setting form must actually reach [skills].

    Patches ``tldw_chatbook.config`` (not the skills module) because the
    helper imports get_cli_setting lazily, at call time.
    """
    import tldw_chatbook.config as config_module

    custom_root = tmp_path / "custom-scratch"
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(custom_root)
            if (section, key) == ("skills", "script_scratch_root")
            else default
        ),
    )
    service, name = script_service
    (service._skill_dir(name) / "scripts" / "cwd.py").write_text(
        "import os; print(os.path.realpath(os.getcwd()))", encoding="utf-8"
    )
    result = await service.run_skill_script(name, "scripts/cwd.py", [])
    assert str(custom_root.resolve()) in result.stdout


@pytest.mark.asyncio
async def test_scope_service_rejects_server_mode(script_scope_service):
    scope, name = script_scope_service
    with pytest.raises(ValueError, match="local-only"):
        await scope.run_skill_script(name, "scripts/hello.py", [], mode="server")


@pytest.mark.asyncio
async def test_scope_enforce_run_script_denies_when_policy_off(script_scope_service_denied):
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    scope, _name = script_scope_service_denied
    with pytest.raises(PolicyDeniedError):
        scope.enforce_run_script()
```

Add to `Tests/Skills/conftest.py`:

```python
@pytest.fixture
def script_service(make_trust_service, tmp_path):
    """A LocalSkillsService with one TRUSTED skill carrying scripts."""
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    trust = make_trust_service()
    name = "demo-skill"
    skill_dir = trust.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text("print('hello')", encoding="utf-8")
    (skill_dir / "scripts" / "echo_args.py").write_text(
        "import sys; print('|'.join(sys.argv[1:]))", encoding="utf-8"
    )
    service = LocalSkillsService(
        store_dir=trust.skills_dir.parent,
        trust_service=trust,
        allow_untrusted_without_trust_service=False,
    )
    # Bootstrap + trust so _require_trusted_skill passes. Reuse whatever the
    # existing Tests/Skills suite does to reach a trusted state.
    _bootstrap_and_trust(trust, name)
    return service, name
```

> **Implementer note:** `_bootstrap_and_trust` is NOT a real helper — find how the existing `Tests/Skills` tests reach a trusted skill (search for `trust_current_skill` / `bootstrap_trust` in `Tests/Skills/`) and reuse that exact pattern, factoring it into the conftest. Same for the `script_service_untrusted`, `script_scope_service`, and `script_scope_service_denied` fixtures: build them from the same pieces, with `script_scope_service_denied` wiring a REAL `ServicePolicyEnforcer` whose registry row for `skills.run_script.launch.local` is disabled (mirror `test_e2e_install_skill_from_github_tree_url_real_services`). An enforcer-less scope service silently no-ops — a policy test without a real enforcer is vacuous.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_skill_script_service.py -x -q`
Expected: FAIL — `AttributeError: 'LocalSkillsService' object has no attribute 'run_skill_script'`

- [ ] **Step 3a: Implement the local service seams**

In `local_skills_service.py`, add near the other module constants:

```python
_INTERPRETER_MAP = {
    ".py": "python3",
    ".sh": "sh",
    ".bash": "bash",
    ".js": "node",
}
```

Add the dataclass (module scope, after the constants):

```python
@dataclass(frozen=True)
class ScriptPlan:
    """How a bundled script would be run, for display and dispatch."""

    skill_name: str
    script_path: str
    mechanism: str  # "direct-exec" | "interpreter"
    interpreter_display: str
    is_binary: bool
```

Add these methods to `LocalSkillsService`:

```python
    def _resolve_script(self, skill_name: str, script_path: str) -> tuple[Path, Path]:
        """Resolve a bundle-relative script path, containment-first.

        Args:
            skill_name: Canonical skill name.
            script_path: POSIX relative path within the bundle.

        Returns:
            ``(skill_dir, absolute_script_path)``.

        Raises:
            ValueError: Unknown skill, or a path that is unsafe, missing, a
                symlink, or the canonical body (all surfaced as the SAME
                ``local_skill_script_not_found`` error so an escape can never
                be distinguished from a genuinely missing file).
        """
        from ..tldw_api.skills_schemas import validate_supporting_file_path

        if script_path == _SKILL_FILENAME:
            raise ValueError(f"local_skill_script_not_found:{script_path}")
        validate_supporting_file_path(script_path)
        skill_dir = self._skill_dir(skill_name)
        if not skill_dir.is_dir():
            raise ValueError(f"local_skill_not_found:{skill_name}")
        path = skill_dir / PurePosixPath(script_path)
        # Containment BEFORE any stat (PR#814 symlink-oracle hardening).
        if get_safe_relative_path(path, skill_dir) is None:
            raise ValueError(f"local_skill_script_not_found:{script_path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"local_skill_script_not_found:{script_path}")
        return skill_dir, path

    def _plan_for_script(self, skill_name: str, script_path: str, path: Path) -> ScriptPlan:
        """Classify how a resolved script should be invoked.

        Args:
            skill_name: Canonical skill name.
            script_path: POSIX relative path within the bundle.
            path: The resolved absolute path.

        Returns:
            A ScriptPlan naming the mechanism and interpreter.

        Raises:
            ValueError: ``unrunnable_script_type`` when the file is neither
                executable nor a known text-script extension, or when a
                mapped interpreter does not resolve on the scrubbed PATH.
        """
        import stat as _stat

        from .skill_script_runner import resolve_interpreter

        raw = path.read_bytes()[:8192]
        is_binary = b"\x00" in raw
        if path.stat().st_mode & _stat.S_IXUSR:
            return ScriptPlan(
                skill_name=skill_name,
                script_path=script_path,
                mechanism="direct-exec",
                interpreter_display="direct-exec",
                is_binary=is_binary,
            )
        if is_binary:
            raise ValueError(f"unrunnable_script_type:{script_path}")
        interpreter_name = _INTERPRETER_MAP.get(PurePosixPath(script_path).suffix)
        if interpreter_name is None:
            raise ValueError(f"unrunnable_script_type:{script_path}")
        resolved = resolve_interpreter(interpreter_name)
        if resolved is None:
            raise ValueError(
                f"unrunnable_script_type:{script_path} "
                f"(interpreter '{interpreter_name}' is not available)"
            )
        return ScriptPlan(
            skill_name=skill_name,
            script_path=script_path,
            mechanism="interpreter",
            interpreter_display=resolved,
            is_binary=False,
        )

    @staticmethod
    def _script_scratch_root() -> str | None:
        """Resolve the optional ``[skills] script_scratch_root`` config root.

        Uses the THREE-argument ``get_cli_setting`` form on purpose: the
        section-dict form (``get_cli_setting("skills", {})``) silently returns
        ``{}`` for any section without a dot in its name (config.py:3965), so
        it would make this knob permanently unreachable.

        Returns:
            The configured scratch root, or None to use the OS temp dir.
        """
        try:
            from ..config import get_cli_setting

            configured = get_cli_setting("skills", "script_scratch_root", "")
        except Exception:  # noqa: BLE001 — config problems fall back to temp
            return None
        if not configured or not isinstance(configured, str):
            return None
        root = Path(configured).expanduser()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return str(root)

    async def describe_skill_script(self, skill_name: str, script_path: str) -> ScriptPlan:
        """Resolve a script for display WITHOUT running it.

        Lets a caller build a confirm prompt and fail early — with no prompt —
        on policy, trust, path, or type errors. Read-only and side-effect-free;
        ``run_skill_script`` re-runs every one of these checks authoritatively,
        so a plan that goes stale before the user decides can never widen what
        actually executes.

        Args:
            skill_name: Canonical skill name.
            script_path: POSIX relative path within the bundle.

        Returns:
            A ScriptPlan describing the mechanism and interpreter.

        Raises:
            SkillTrustBlockedError: Skill not currently trusted.
            ValueError: Unsafe/missing path or unrunnable file type.
        """
        self._enforce("skills.run_script.launch.local")
        self._require_trusted_skill(skill_name)
        _skill_dir, path = self._resolve_script(skill_name, script_path)
        return self._plan_for_script(skill_name, script_path, path)

    async def run_skill_script(
        self,
        skill_name: str,
        script_path: str,
        args: list[str],
        *,
        limits: "ScriptRunLimits | None" = None,
    ) -> "ScriptRunResult":
        """Run a bundled script of a trusted skill under best-effort containment.

        Order is load-bearing and re-verified here even if the caller already
        called ``describe_skill_script``: policy gate, per-RUN trust
        re-verification (a skill revoked or mutated mid-run stops being
        runnable immediately), containment-first path resolution, then
        classification, then the sandboxed subprocess in a fresh scratch
        directory that is never the skill directory.

        Args:
            skill_name: Canonical skill name.
            script_path: POSIX relative path within the bundle.
            args: Arguments appended after the script path. Never shell-parsed.
            limits: Optional containment budget; defaults to ScriptRunLimits().

        Returns:
            A ScriptRunResult; a non-zero exit or timeout is a normal result.

        Raises:
            SkillTrustBlockedError: Skill not currently trusted.
            ValueError: Unsafe/missing path or unrunnable file type.
        """
        import shutil as _shutil
        import tempfile

        from .skill_script_runner import ScriptRunLimits, run_script_subprocess

        self._enforce("skills.run_script.launch.local")
        self._require_trusted_skill(skill_name)
        _skill_dir, path = self._resolve_script(skill_name, script_path)
        plan = self._plan_for_script(skill_name, script_path, path)
        effective_limits = limits or ScriptRunLimits()
        target_argv = (
            [str(path), *[str(a) for a in args]]
            if plan.mechanism == "direct-exec"
            else [plan.interpreter_display, str(path), *[str(a) for a in args]]
        )
        scratch = Path(
            tempfile.mkdtemp(
                prefix="tldw-skill-script-", dir=self._script_scratch_root()
            )
        )
        try:
            return run_script_subprocess(
                target_argv, cwd=scratch, limits=effective_limits
            )
        finally:
            _shutil.rmtree(scratch, ignore_errors=True)
```

Ensure `dataclass` and `PurePosixPath` are imported at module scope (`PurePosixPath` already is — it is used by `read_skill_file`).

- [ ] **Step 3b: Implement the scope-service passthroughs**

In `skills_scope_service.py`, after `enforce_install_remote`:

```python
    def enforce_run_script(self) -> None:
        """Gate a skill-script run (public seam for the agent bridge closure).

        Public by design so the bridge closure can deny on policy BEFORE
        prompting the user, mirroring ``enforce_install_remote``.

        Raises:
            PolicyDeniedError: When a wired policy enforcer denies the action.
        """
        self._enforce_policy("skills.run_script.launch.local")

    async def describe_skill_script(
        self,
        skill_name: str,
        script_path: str,
        *,
        mode: SkillsBackend | str | None = None,
    ):
        """Resolve a LOCAL skill's script for display, without running it.

        Args:
            skill_name: Canonical skill name.
            script_path: POSIX relative path within the bundle.
            mode: Backend selector; only local is accepted.

        Returns:
            The local service's ScriptPlan.

        Raises:
            ValueError: Server mode, unavailable local backend, or a bad path.
            SkillTrustBlockedError: Skill not currently trusted.
        """
        normalized_mode = (
            self._normalize_mode(mode) if mode is not None else SkillsBackend.LOCAL
        )
        if normalized_mode is not SkillsBackend.LOCAL:
            raise ValueError("skill scripts run local-only")
        service = self._require_service(SkillsBackend.LOCAL)
        self._enforce_policy("skills.run_script.launch.local")
        return await self._maybe_await(
            service.describe_skill_script(skill_name, script_path)
        )

    async def run_skill_script(
        self,
        skill_name: str,
        script_path: str,
        args: list[str],
        *,
        mode: SkillsBackend | str | None = None,
    ):
        """Run a LOCAL trusted skill's bundled script (runtime run_skill_script seam).

        Args:
            skill_name: Canonical skill name.
            script_path: POSIX relative path within the bundle.
            args: Arguments appended after the script path.
            mode: Backend selector; only local is accepted.

        Returns:
            The local service's ScriptRunResult.

        Raises:
            ValueError: Server mode, unavailable local backend, bad path, or
                an unrunnable file type.
            SkillTrustBlockedError: Skill not currently trusted.
        """
        normalized_mode = (
            self._normalize_mode(mode) if mode is not None else SkillsBackend.LOCAL
        )
        if normalized_mode is not SkillsBackend.LOCAL:
            raise ValueError("skill scripts run local-only")
        service = self._require_service(SkillsBackend.LOCAL)
        self._enforce_policy("skills.run_script.launch.local")
        return await self._maybe_await(
            service.run_skill_script(skill_name, script_path, args)
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_skill_script_service.py -q && pytest Tests/Skills -q`
Expected: PASS, and no regression in the existing `Tests/Skills` suite.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/Skills_Interop/skills_scope_service.py Tests/Skills/
git commit -m "feat(skills): add describe_skill_script and run_skill_script seams

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Sixth runtime tool

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (~:31-44)
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (after `INSTALL_SKILL_TOOL_SCHEMA`)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (LoopDeps ~:246, dispatch ~:544-552)
- Modify: `tldw_chatbook/Agents/agent_service.py` (ctor, pin ~:371, deps ~:635)
- Modify: `tldw_chatbook/Library/library_skills_state.py` (~:39-59)
- Test: `Tests/Agents/test_run_skill_script_runtime_tool.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3 (the callable is injected).
- Produces:
  - `RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"` (joins `RUNTIME_TOOL_NAMES`)
  - `RUN_SKILL_SCRIPT_TOOL_SCHEMA`
  - `LoopDeps.run_skill_script: Callable[[str, str, list[str]], ToolResult] | None = None`
  - `AgentService(..., run_skill_script_tool: Callable[[str, str, list[str]], ToolResult] | None = None)`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Agents/test_run_skill_script_runtime_tool.py`. Mirror the construction style of `Tests/Agents/test_skill_file_runtime_tool.py` — read that file first and reuse its fixtures/fakes rather than inventing new ones.

```python
"""The run_skill_script runtime tool: pinning, dispatch, and reach."""

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    RUNTIME_TOOL_NAMES,
    RUN_SKILL_SCRIPT_TOOL_NAME,
)
from tldw_chatbook.Agents.tool_catalog import RUN_SKILL_SCRIPT_TOOL_SCHEMA


def test_name_is_a_runtime_tool():
    assert RUN_SKILL_SCRIPT_TOOL_NAME == "run_skill_script"
    assert RUN_SKILL_SCRIPT_TOOL_NAME in RUNTIME_TOOL_NAMES


def test_schema_has_a_description_and_the_three_params():
    assert RUN_SKILL_SCRIPT_TOOL_SCHEMA.description.strip()
    props = RUN_SKILL_SCRIPT_TOOL_SCHEMA.parameters["properties"]
    assert set(props) == {"skill_name", "script_path", "args"}
    assert RUN_SKILL_SCRIPT_TOOL_SCHEMA.parameters["required"] == [
        "skill_name",
        "script_path",
    ]


def test_shadow_guard_lists_the_new_name():
    from tldw_chatbook.Library.library_skills_state import _SHADOWED_BUILTIN_NAMES

    assert "run_skill_script" in _SHADOWED_BUILTIN_NAMES


def test_dispatch_routes_to_the_wired_callable():
    from tldw_chatbook.Agents.agent_models import ToolCall, ToolResult
    from tldw_chatbook.Agents.agent_runtime import LoopDeps

    seen = {}

    def fake_run(skill_name, script_path, args):
        seen["call"] = (skill_name, script_path, args)
        return ToolResult(ok=True, content="ran")

    deps = LoopDeps(
        call_model=lambda *a, **k: None,
        invoke_tool=lambda call: ToolResult(ok=False, error="wrong path"),
        spawn=lambda *a, **k: None,
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        on_step=lambda s: None,
        run_skill_script=fake_run,
    )
    assert deps.run_skill_script is not None
    result = deps.run_skill_script("demo", "scripts/hello.py", ["x"])
    assert result.ok is True
    assert seen["call"] == ("demo", "scripts/hello.py", ["x"])


def test_schema_is_pinned_for_primary_and_for_subagents(make_agent_service):
    """All-agents scope: unlike install_skill, this is NOT primary-gated."""
    service = make_agent_service(run_skill_script_tool=lambda *a: None)
    primary = service._runtime_schema_names(agent_kind=AGENT_KIND_PRIMARY)
    child = service._runtime_schema_names(agent_kind=AGENT_KIND_SUBAGENT)
    assert "run_skill_script" in primary
    assert "run_skill_script" in child


def test_tool_is_absent_when_not_wired(make_agent_service):
    service = make_agent_service(run_skill_script_tool=None)
    assert "run_skill_script" not in service._runtime_schema_names(
        agent_kind=AGENT_KIND_PRIMARY
    )
```

> **Implementer note:** `_runtime_schema_names` does not exist. Either (a) assert against the real pinned schemas by driving `_run_one` with the suite's existing fake model (preferred — mirror how `test_skill_file_runtime_tool.py` asserts pinning), or (b) extract a tiny private helper on `AgentService` that returns the runtime schema list for an `agent_kind`, and have `_run_one` call it. Do NOT leave a test calling a method that doesn't exist. If you add the helper, keep `_run_one`'s behavior byte-identical.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_run_skill_script_runtime_tool.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'RUN_SKILL_SCRIPT_TOOL_NAME'`

- [ ] **Step 3a: Name constant** — in `agent_models.py`:

```python
SKILL_FILE_TOOL_NAME = "skill_file"
INSTALL_SKILL_TOOL_NAME = "install_skill"
RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"
RUNTIME_TOOL_NAMES = frozenset(
    {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
        RUN_SKILL_SCRIPT_TOOL_NAME,
    }
)
```

- [ ] **Step 3b: Schema** — in `tool_catalog.py`, after `INSTALL_SKILL_TOOL_SCHEMA` (import the new name alongside the others):

```python
RUN_SKILL_SCRIPT_TOOL_SCHEMA = ToolSchema(
    id="runtime:run_skill_script",
    name=RUN_SKILL_SCRIPT_TOOL_NAME,
    description=(
        "Run a script bundled with a trusted skill. The user is asked to "
        "confirm each run unless they have granted this skill standing "
        "permission. The script runs with a scrubbed environment in a "
        "temporary working directory (not the skill's own folder), under CPU "
        "and time limits; only its stdout/stderr and exit code come back, and "
        "any files it writes are discarded. Args: skill_name (the skill that "
        "owns the script), script_path (relative POSIX path, e.g. "
        "scripts/extract.py), args (optional list of string arguments)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The skill whose bundled script to run.",
            },
            "script_path": {
                "type": "string",
                "description": (
                    "Relative POSIX path of the script inside the skill's "
                    "bundle, e.g. scripts/extract.py."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional string arguments passed to the script.",
            },
        },
        "required": ["skill_name", "script_path"],
    },
)
```

- [ ] **Step 3c: LoopDeps field + dispatch** — in `agent_runtime.py`, after the `install_skill` field:

```python
    # run_skill_script: the sixth runtime tool (trust-gated script execution).
    # Unlike install_skill this is NOT agent_kind-gated -- the user chose an
    # all-agents caller scope, because the per-run confirm card and the
    # per-skill grant (not the caller's identity) are what gate each run.
    # `None` (the default) means the run is not wired for it and a call by
    # that name falls through to the generic deps.invoke_tool path.
    run_skill_script: Callable[[str, str, list[str]], ToolResult] | None = None
```

And in the dispatch elif chain, immediately after the `INSTALL_SKILL_TOOL_NAME` branch:

```python
                elif (
                    call.name == RUN_SKILL_SCRIPT_TOOL_NAME
                    and deps.run_skill_script is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    raw_args = call.args.get("args") or []
                    if not isinstance(raw_args, (list, tuple)):
                        raw_args = [raw_args]
                    result = deps.run_skill_script(
                        str(call.args.get("skill_name", "")),
                        str(call.args.get("script_path", "")),
                        [str(item) for item in raw_args],
                    )
```

Import `RUN_SKILL_SCRIPT_TOOL_NAME` alongside the other tool names at the top of `agent_runtime.py`.

- [ ] **Step 3d: AgentService wiring** — add the ctor kwarg (keyword-only, defaulting to `None`) and store it as `self._run_skill_script_tool`. Then the schema pin, after the install pin:

```python
        # All-agents scope (spec §4.3): NO agent_kind gate. _run_one recurses
        # on this same service instance, so this intentionally reaches every
        # depth -- primary, skill forks, and spawned subagents alike. The gate
        # for each run is policy + trust + the confirm card / per-skill grant,
        # applied in the bridge closure and the service, not here.
        if self._run_skill_script_tool is not None:
            runtime_schemas.append(RUN_SKILL_SCRIPT_TOOL_SCHEMA)
```

And the LoopDeps wiring, after `install_skill=`:

```python
            run_skill_script=self._run_skill_script_tool,
```

- [ ] **Step 3e: Drift guard** — in `library_skills_state.py`, inside `_SHADOWED_BUILTIN_NAMES`:

```python
        # The run_skill_script runtime tool (same drift-guard rationale as
        # skill_file/install_skill above).
        "run_skill_script",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_run_skill_script_runtime_tool.py -q && pytest Tests/Agents Tests/Library -q`
Expected: PASS. The `_SHADOWED_BUILTIN_NAMES` drift-guard sync test must pass — it fires whenever `RUNTIME_TOOL_NAMES` gains a member.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/ tldw_chatbook/Library/library_skills_state.py Tests/Agents/test_run_skill_script_runtime_tool.py
git commit -m "feat(agents): add run_skill_script as the sixth runtime tool

Wired unconditionally (all-agents caller scope): unlike install_skill it
is not agent_kind-gated, because the confirm card and per-skill grant are
the real gate.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Controller HITL + bridge closure

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (constants ~:79, state ~:423-428, methods after `_deny_pending_skill_install_on_context_change` ~:1191, `switch_session` ~:724, `run_reply` call site ~:3283)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (closure after the install closure ~:1043, `run_reply` signature ~:902, AgentService construction ~:1154-1165)
- Test: `Tests/Chat/test_console_skill_script_confirm.py`

**Interfaces:**
- Consumes: Task 3's scope-service methods; Task 4's `run_skill_script_tool` kwarg; Task 2's grant methods.
- Produces:
  - `ConsoleChatController.request_skill_script_confirm(payload: dict) -> dict` returning `{"allow": bool, "remember": bool}`
  - `.resolve_pending_skill_script(allow: bool, remember: bool) -> None`
  - `._deny_pending_skill_script_on_context_change() -> None`
  - `.set_pending_skill_script: Callable[[dict | None], None] | None`
  - `ConsoleAgentBridge.run_reply(..., request_skill_script_confirm: Callable[[dict], dict] | None = None)`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_console_skill_script_confirm.py`. Read `Tests/UI/test_console_mcp_approval.py` and the existing install-confirm tests first and reuse their `_FakeApp` (with `call_from_thread`) rather than writing a new one.

```python
"""HITL confirm + bridge closure for run_skill_script."""

import threading

import pytest


def test_no_ui_bridge_denies_immediately(make_controller):
    """Headless must fail closed at once, not block for the full timeout."""
    controller = make_controller()
    controller.app = None
    controller.set_pending_skill_script = None
    decision = controller.request_skill_script_confirm({"skill_name": "demo"})
    assert decision == {"allow": False, "remember": False}


def test_allow_round_trip(make_controller):
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm(
            {"skill_name": "demo", "script_path": "scripts/hello.py"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller.resolve_pending_skill_script(True, False)
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": False}


def test_always_allow_round_trip(make_controller):
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm({"skill_name": "demo"})

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller.resolve_pending_skill_script(True, True)
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": True}


def test_context_change_denies_a_pending_confirm(make_controller):
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm({"skill_name": "demo"})

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller._deny_pending_skill_script_on_context_change()
    thread.join(timeout=5)
    assert result["decision"]["allow"] is False


def test_closure_denies_on_policy_without_prompting(bridge_closure_env):
    """Policy denial must not show a card."""
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    env = bridge_closure_env(
        enforce_side_effect=PolicyDeniedError(
            action_id="skills.run_script.launch.local",
            reason_code="authority_denied",
            user_message="Script execution is disabled by policy.",
            effective_source="local",
            authority_owner="local",
        )
    )
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is False
    assert "policy" in result.error.lower() or "disabled" in result.error.lower()
    assert env.confirm_calls == []


def test_closure_denies_on_bad_path_without_prompting(bridge_closure_env):
    env = bridge_closure_env(
        describe_side_effect=ValueError("local_skill_script_not_found:../x.py")
    )
    result = env.closure("demo", "../x.py", [])
    assert result.ok is False
    assert env.confirm_calls == []


def test_closure_skips_the_prompt_when_the_skill_is_granted(bridge_closure_env):
    env = bridge_closure_env(granted=True)
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is True
    assert env.confirm_calls == [], "a standing grant must not re-prompt"
    assert env.run_calls, "the script must still actually run"


def test_closure_records_the_grant_on_always_allow(bridge_closure_env):
    env = bridge_closure_env(confirm_result={"allow": True, "remember": True})
    env.closure("demo", "scripts/hello.py", [])
    assert env.granted_names == ["demo"]


def test_closure_denies_when_the_user_declines(bridge_closure_env):
    env = bridge_closure_env(confirm_result={"allow": False, "remember": False})
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is False
    assert "declined" in result.error.lower()
    assert env.run_calls == []


def test_closure_fails_closed_when_confirm_raises(bridge_closure_env):
    env = bridge_closure_env(confirm_side_effect=RuntimeError("ui exploded"))
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is False
    assert env.run_calls == []


def test_nonzero_exit_is_ok_true_with_the_failure_described(bridge_closure_env):
    """A failed SCRIPT is a successful TOOL CALL — the agent must see it."""
    env = bridge_closure_env(run_result_exit_code=3, run_result_stderr="boom")
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is True
    assert "3" in result.content
    assert "boom" in result.content


def test_tool_is_absent_without_a_confirm_callback(bridge_without_confirm):
    """Advertised must equal usable (the #847 lesson)."""
    assert bridge_without_confirm.run_skill_script_tool is None
```

> **Implementer note:** `_wait_until`, `make_controller`, `bridge_closure_env`, and `bridge_without_confirm` are fixtures YOU must write in this test module (or `Tests/Chat/conftest.py`), built from the existing install-confirm test's fakes. `bridge_closure_env` should construct the real closure from `console_agent_bridge` with a fake scope service whose `enforce_run_script`/`describe_skill_script`/`run_skill_script` are controllable, and a fake trust service for grants — then expose `.closure`, `.confirm_calls`, `.run_calls`, `.granted_names`. Do not test a reimplementation of the closure; test the real one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/Chat/test_console_skill_script_confirm.py -x -q`
Expected: FAIL — `AttributeError: ... has no attribute 'request_skill_script_confirm'`

- [ ] **Step 3a: Controller HITL** — in `console_chat_controller.py`, add the timeout constant next to `_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS`:

```python
_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = 120.0
```

Add the state fields next to their install analogues (`__init__`):

```python
        self.set_pending_skill_script: Callable[[dict | None], None] | None = None
        self._pending_skill_script_event: threading.Event | None = None
        self._pending_skill_script_decision: dict[str, bool] | None = None
```

Add the methods after `_deny_pending_skill_install_on_context_change`:

```python
    # -- Skill-script confirm bridge -----------------------------------------

    def request_skill_script_confirm(self, payload: dict[str, Any]) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm running a skill's script.

        Mirrors request_skill_install_confirm, but carries a two-part decision:
        allow this run, and whether to remember the choice for this skill.

        Args:
            payload: Confirm details to render ({"skill_name", "script_path",
                "mechanism", "args", ...}); a "timeout_seconds" key is added.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path (deny,
            cancel, stop, timeout, context change, no wired UI) returns
            ``allow=False``.
        """
        if self.app is None or self.set_pending_skill_script is None:
            return {"allow": False, "remember": False}

        event = threading.Event()
        decision: dict[str, bool] = {}
        self._pending_skill_script_event = event
        self._pending_skill_script_decision = decision

        timeout_seconds = _DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
        deadline = time.monotonic() + timeout_seconds
        card_payload = dict(payload)
        card_payload["timeout_seconds"] = timeout_seconds
        try:
            self._marshal_pending_skill_script(card_payload)
            while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                if self._stop_requested or (
                    self._active_cancel_event is not None
                    and self._active_cancel_event.is_set()
                ):
                    break
                if time.monotonic() >= deadline:
                    break
            return {
                "allow": bool(decision.get("allow", False)),
                "remember": bool(decision.get("remember", False)),
            }
        finally:
            self._pending_skill_script_event = None
            self._pending_skill_script_decision = None
            try:
                self._marshal_pending_skill_script(None)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Failed to clear skill-script confirm during teardown"
                )

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-script confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        if self.app is not None and self.set_pending_skill_script is not None:
            self.app.call_from_thread(self.set_pending_skill_script, payload)

    def resolve_pending_skill_script(self, allow: bool, remember: bool) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        Args:
            allow: True to run the script this once.
            remember: True to also grant this skill standing permission.
        """
        decision = self._pending_skill_script_decision
        event = self._pending_skill_script_event
        if decision is None or event is None:
            return
        decision["allow"] = bool(allow)
        decision["remember"] = bool(remember)
        event.set()

    def _deny_pending_skill_script_on_context_change(self) -> None:
        """Force-deny a pending script confirm (Event set, decision left False)."""
        event = self._pending_skill_script_event
        if event is not None:
            event.set()
```

In `switch_session`, next to the existing `_deny_pending_skill_install_on_context_change()` call, add:

```python
        self._deny_pending_skill_script_on_context_change()
```

At the `run_reply` call site (next to `request_skill_install_confirm=`), add:

```python
            request_skill_script_confirm=self.request_skill_script_confirm,
```

- [ ] **Step 3b: Bridge closure** — in `console_agent_bridge.py`, add the `run_reply` keyword parameter next to `request_skill_install_confirm`:

```python
        request_skill_script_confirm: Callable[[dict], dict] | None = None,
```

Add the closure immediately after the install closure block:

```python
        # Trust-gated skill script execution (6th runtime tool). Built only
        # when BOTH a skills service AND a confirm callback exist -- without a
        # callback the tool is absent (never advertised) rather than
        # auto-denying every call. Order (load-bearing): enforce policy (no
        # prompt on denial) -> describe/resolve (no prompt on a bad path or an
        # unrunnable type) -> grant check (no prompt when the user already
        # granted this skill) -> confirm (plain blocking call, OUTSIDE any
        # asyncio.run) -> run -> broad-catch wrap. run_skill_script re-verifies
        # policy/trust/path authoritatively, so a stale plan can never widen
        # what actually executes.
        run_skill_script_tool = None
        if self._skills_service is not None and request_skill_script_confirm is not None:
            scope = self._skills_service
            trust_service = getattr(
                getattr(scope, "local_service", None), "trust_service", None
            )

            def run_skill_script_tool(
                skill_name: str, script_path: str, args: list[str]
            ) -> ToolResult:
                from tldw_chatbook.runtime_policy.types import PolicyDeniedError

                try:
                    scope.enforce_run_script()
                except PolicyDeniedError as exc:
                    return ToolResult(ok=False, error=exc.user_message)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=str(exc))
                try:
                    plan = asyncio.run(
                        scope.describe_skill_script(skill_name, script_path)
                    )
                except Exception as exc:  # noqa: BLE001 (trust/path/type)
                    return ToolResult(ok=False, error=f"run_skill_script: {exc}")

                granted = False
                if trust_service is not None:
                    try:
                        granted = bool(
                            trust_service.script_execution_granted(skill_name)
                        )
                    except Exception:  # noqa: BLE001 — doubt ⇒ prompt
                        granted = False
                if not granted:
                    try:
                        decision = request_skill_script_confirm(
                            {
                                "skill_name": skill_name,
                                "script_path": script_path,
                                "mechanism": plan.mechanism,
                                "interpreter": plan.interpreter_display,
                                "is_binary": plan.is_binary,
                                "args": [str(a) for a in args],
                            }
                        )
                    except Exception:  # noqa: BLE001 — a UI error fails closed
                        decision = {"allow": False, "remember": False}
                    if not isinstance(decision, Mapping):
                        decision = {"allow": False, "remember": False}
                    if not decision.get("allow", False):
                        return ToolResult(
                            ok=False, error="The user declined to run this script."
                        )
                    if decision.get("remember", False) and trust_service is not None:
                        try:
                            trust_service.grant_script_execution(skill_name)
                        except Exception:  # noqa: BLE001 — grant is best-effort
                            logger.opt(exception=True).debug(
                                "Failed to persist skill script grant"
                            )
                try:
                    outcome = asyncio.run(
                        scope.run_skill_script(skill_name, script_path, list(args))
                    )
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=f"run_skill_script: {exc}")

                lines = [f"exit_code: {outcome.exit_code}"]
                if outcome.timed_out:
                    lines.append("timed out — the script was killed")
                if outcome.output_capped:
                    lines.append("output was truncated at the size cap")
                for warning in outcome.sandbox_warnings:
                    lines.append(f"note: {warning}")
                if outcome.stdout:
                    lines.append(f"stdout:\n{outcome.stdout}")
                if outcome.stderr:
                    lines.append(f"stderr:\n{outcome.stderr}")
                return ToolResult(ok=True, content="\n".join(lines))
```

Pass it into the `AgentService(...)` construction next to `install_skill_tool=`:

```python
            run_skill_script_tool=run_skill_script_tool,
```

Ensure `Mapping` is imported in this module (it is used by the runtime; add `from collections.abc import Mapping` if absent).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Chat/test_console_skill_script_confirm.py -q && pytest Tests/Chat -q`
Expected: PASS with no regression in `Tests/Chat`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_skill_script_confirm.py
git commit -m "feat(chat): add skill-script confirm HITL and the bridge closure

enforce -> describe -> grant-check -> confirm -> run, fail-closed at every
step; a standing per-skill grant skips the prompt.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Confirm card + screen wiring

**Files:**
- Create: `tldw_chatbook/Widgets/Chat_Widgets/skill_script_confirm_card.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py` (~:205-262)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (controller binding ~:3348, setter ~:15593, `@on` ~:15616)
- Test: `Tests/UI/test_skill_script_confirm_card.py`

**Interfaces:**
- Consumes: Task 5's `resolve_pending_skill_script(allow, remember)` and `set_pending_skill_script`.
- Produces: `SkillScriptConfirmCard` with `ScriptDecided(allow: bool, remember: bool)` and `set_script(payload)`; `TaskResumeState.pending_skill_script` + `has_pending_skill_script()`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_skill_script_confirm_card.py`:

```python
"""The skill-script confirm card and its task-state plumbing."""

import pytest

from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.skill_script_confirm_card import (
    SkillScriptConfirmCard,
)


def test_state_carries_and_serializes_a_pending_script():
    state = TaskResumeState(pending_skill_script={"skill_name": "demo"})
    assert state.has_pending_skill_script() is True
    assert TaskResumeState.from_dict(state.to_dict()).pending_skill_script == {
        "skill_name": "demo"
    }


def test_state_without_a_pending_script():
    assert TaskResumeState().has_pending_skill_script() is False


def test_card_statics_are_markup_free():
    """Agent-supplied paths/args must never render as Rich markup."""
    card = SkillScriptConfirmCard()
    for widget in card.compose():
        if hasattr(widget, "_render_markup"):
            assert widget._render_markup is False


@pytest.mark.asyncio
async def test_card_shows_details_and_emits_three_decisions(card_app):
    """Allow / Always allow / Deny each post the right ScriptDecided."""
    async with card_app.run_test() as pilot:
        card = card_app.query_one(SkillScriptConfirmCard)
        card.set_script(
            {
                "skill_name": "demo",
                "script_path": "scripts/extract.py",
                "mechanism": "interpreter",
                "interpreter": "/usr/bin/python3",
                "args": ["--in", "x.pdf"],
                "timeout_seconds": 120.0,
            }
        )
        await pilot.pause()
        assert card.display is True

        for button_id, expected in (
            ("#skill-script-allow", (True, False)),
            ("#skill-script-always", (True, True)),
            ("#skill-script-deny", (False, False)),
        ):
            card.set_script({"skill_name": "demo", "script_path": "s.py"})
            await pilot.pause()
            card_app.decisions.clear()
            await pilot.click(button_id)
            await pilot.pause()
            assert card_app.decisions == [expected]


@pytest.mark.asyncio
async def test_task_cards_container_becomes_visible_for_a_pending_script(cards_app):
    """Without extending the display gate the card is invisible (spec §4.4)."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

    async with cards_app.run_test() as pilot:
        cards = cards_app.query_one(ChatTaskCards)
        cards.sync_state(TaskResumeState(pending_skill_script={"skill_name": "demo"}))
        await pilot.pause()
        assert cards.display is True
```

> **Implementer note:** `card_app` / `cards_app` are small Textual test apps you write in this module — mirror the existing card tests under `Tests/UI/` (search for `SkillInstallConfirmCard` tests and copy their harness). `card_app` records posted `ScriptDecided` messages into `card_app.decisions` as `(allow, remember)` tuples.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/UI/test_skill_script_confirm_card.py -x -q`
Expected: FAIL — `ModuleNotFoundError: ... skill_script_confirm_card`

- [ ] **Step 3a: The card** — create `tldw_chatbook/Widgets/Chat_Widgets/skill_script_confirm_card.py`:

```python
"""Allow / Always-allow / Deny card for running a skill's bundled script.

The skill name, script path, and args are agent-influenced, so every Static
renders with markup=False.
"""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class SkillScriptConfirmCard(Container):
    """Prompts the user to allow or deny running a skill's script."""

    class ScriptDecided(Message):
        """Posted when the user allows, always-allows, or denies the run."""

        def __init__(self, allow: bool, remember: bool) -> None:
            self.allow = allow
            self.remember = remember
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(
            "An agent wants to run a script from a skill:",
            id="skill-script-prompt",
            markup=False,
        )
        yield Static("", id="skill-script-target", markup=False)
        yield Static("", id="skill-script-args", markup=False)
        yield Static(
            "It runs with a scrubbed environment in a temporary folder (not "
            "the skill's own folder); only its output comes back.",
            id="skill-script-note",
            markup=False,
        )
        yield Horizontal(
            Button("Allow once", id="skill-script-allow", variant="primary"),
            Button("Always allow this skill", id="skill-script-always"),
            Button("Deny", id="skill-script-deny", variant="error"),
            id="skill-script-buttons",
        )

    def on_mount(self) -> None:
        self.display = False

    def set_script(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload``, or hide it if None.

        Args:
            payload: The pending confirm's dict ({"skill_name", "script_path",
                "mechanism", "interpreter", "is_binary", "args"}), or None.
        """
        if not payload:
            self.display = False
            return
        skill_name = str(payload.get("skill_name", ""))
        script_path = str(payload.get("script_path", ""))
        mechanism = str(payload.get("mechanism", ""))
        interpreter = str(payload.get("interpreter", ""))
        if mechanism == "direct-exec":
            how = "runs directly"
            if payload.get("is_binary"):
                how = "runs directly (a binary you cannot review as text)"
        else:
            how = f"runs with {interpreter}"
        self.query_one("#skill-script-target", Static).update(
            f"{skill_name} — {script_path} ({how})"
        )
        args = payload.get("args") or []
        self.query_one("#skill-script-args", Static).update(
            ("arguments: " + " ".join(str(a) for a in args)) if args else "no arguments"
        )
        self.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions = {
            "skill-script-allow": (True, False),
            "skill-script-always": (True, True),
            "skill-script-deny": (False, False),
        }
        decision = decisions.get(event.button.id or "")
        if decision is None:
            return
        event.stop()
        self.display = False
        self.post_message(self.ScriptDecided(*decision))
```

- [ ] **Step 3b: Task state** — in `chat_screen_state.py`, add the field after `pending_skill_install`:

```python
    pending_skill_script: Optional[Dict[str, Any]] = None
```

the predicate after `has_pending_skill_install`:

```python
    def has_pending_skill_script(self) -> bool:
        """Return True when a skill-script confirm should be shown.

        Returns:
            True when a skill-script confirm should be shown.
        """
        return bool(self.pending_skill_script)
```

and `pending_skill_script` to BOTH `to_dict()` and `from_dict()` alongside `pending_skill_install`.

- [ ] **Step 3c: Cards container** — in `chat_task_cards.py`: import the card, yield it in `compose()` after the install card:

```python
        yield SkillScriptConfirmCard(id="chat-skill-script-card")
```

in `sync_state`, query it and sync it:

```python
        script_card = self.query_one(SkillScriptConfirmCard)
        ...
        script_card.set_script(task_state.pending_skill_script)
```

and EXTEND the display gate (without this the card is invisible — a hidden parent hides descendants):

```python
        self.display = (
            task_state.has_pending_approval()
            or task_state.has_pending_skill_install()
            or task_state.has_pending_skill_script()
            or task_state.has_resume_content()
        )
```

- [ ] **Step 3d: Screen wiring** — in `chat_screen.py`, next to the install analogues: bind `controller.set_pending_skill_script = self._set_console_pending_skill_script`; add that UI-thread setter (mirror `_set_console_pending_skill_install`, using `replace(current, pending_skill_script=payload)`); and add the handler:

```python
    @on(SkillScriptConfirmCard.ScriptDecided)
    def _on_skill_script_decided(self, event: SkillScriptConfirmCard.ScriptDecided) -> None:
        event.stop()
        controller = self._console_controller
        if controller is not None:
            controller.resolve_pending_skill_script(event.allow, event.remember)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/UI/test_skill_script_confirm_card.py -q && pytest Tests/UI -q`
Expected: PASS with no `Tests/UI` regression.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/skill_script_confirm_card.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_skill_script_confirm_card.py
git commit -m "feat(ui): add the skill-script confirm card and wire it into chat

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Grant visibility + revoke in the trust panel

A standing "always allow" grant the user cannot see or withdraw is a hole. This closes it in the place the user already governs a skill.

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (trust panel compose, `#library-skill-trust-panel` ~:1031)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_render_library_skill_trust_panel` ~:7850, plus a button handler beside the existing trust-action handlers)
- Test: `Tests/Library/test_skill_script_grant_panel.py`

**Interfaces:**
- Consumes: Task 2's `script_execution_granted(skill_name)` and `revoke_script_execution(skill_name)`.
- Produces: `skill_script_grant_line(granted: bool) -> str` in `library_skills_canvas.py`; panel widgets `#library-skill-script-grant` (Static) and `#library-skill-script-grant-revoke` (Button).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_skill_script_grant_panel.py`:

```python
"""Grant visibility + revoke in the Library skills trust panel."""

from tldw_chatbook.Widgets.Library.library_skills_canvas import skill_script_grant_line


def test_line_states_when_scripts_may_run_without_asking():
    line = skill_script_grant_line(True)
    assert "without asking" in line.lower()


def test_line_states_when_every_run_is_confirmed():
    line = skill_script_grant_line(False)
    assert "confirm" in line.lower() or "asked" in line.lower()


def test_revoking_clears_the_grant(trust_service_with_skill):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    assert service.script_execution_granted(name) is True
    service.revoke_script_execution(name)
    assert service.script_execution_granted(name) is False
```

> **Implementer note:** reuse the `trust_service_with_skill` fixture from Task 2 (`Tests/Skills/conftest.py`); if `Tests/Library` cannot see it, promote that fixture to a shared conftest rather than duplicating it. Add a panel-render test only if the existing `Tests/Library` suite already has a harness that mounts the skills canvas — mirror it; do not invent a new app harness for this.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && pytest Tests/Library/test_skill_script_grant_panel.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'skill_script_grant_line'`

- [ ] **Step 3a: Copy helper** — in `library_skills_canvas.py`, beside the other trust copy helpers (e.g. near `skill_trust_panel_remediation_copy` ~:212):

```python
def skill_script_grant_line(granted: bool) -> str:
    """Return the trust-panel line describing this skill's script permission.

    Args:
        granted: Whether a standing script-execution grant is in effect.

    Returns:
        A single plain-text line for the trust panel.
    """
    if granted:
        return (
            "Scripts: this skill may run its bundled scripts without asking. "
            "Any change to its files cancels this automatically."
        )
    return "Scripts: you are asked to confirm each time this skill runs a script."
```

- [ ] **Step 3b: Panel widgets** — inside the `#library-skill-trust-panel` Vertical in `library_skills_canvas.py`, after the remediation Static:

```python
            yield Static(
                skill_script_grant_line(False),
                id="library-skill-script-grant",
                markup=False,
            )
            yield Button(
                "Revoke script access",
                id="library-skill-script-grant-revoke",
                disabled=True,
            )
```

- [ ] **Step 3c: Render + handler** — in `library_screen.py`'s `_render_library_skill_trust_panel`, add another guarded `try` block matching the existing ones:

```python
        try:
            granted = False
            trust_service = getattr(self, "local_skill_trust_service", None) or getattr(
                self.app, "local_skill_trust_service", None
            )
            if trust_service is not None and state.name:
                granted = bool(trust_service.script_execution_granted(state.name))
            self.query_one("#library-skill-script-grant", Static).update(
                skill_script_grant_line(granted)
            )
            self.query_one(
                "#library-skill-script-grant-revoke", Button
            ).disabled = not granted
        except (NoMatches, QueryError, AttributeError):
            pass
```

Add `skill_script_grant_line` to the existing `library_skills_canvas` import block at `library_screen.py:214-216`. Then add a button handler beside the other trust-action handlers, following whatever dispatch style they already use (`@on(Button.Pressed, "#...")` or an `if event.button.id == ...` chain — match the file, do not introduce a second style):

```python
        # Revoke a standing script-execution grant, then re-render the panel.
        trust_service = getattr(self, "local_skill_trust_service", None) or getattr(
            self.app, "local_skill_trust_service", None
        )
        state = self._library_skill_editor_state
        if trust_service is not None and state is not None and state.name:
            trust_service.revoke_script_execution(state.name)
            self._render_library_skill_trust_panel()
```

> **Implementer note:** verify the editor state's skill-name attribute is really `state.name` in this file before using it (it may be `state.skill_name`); use whatever the neighbouring trust code uses.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Library/test_skill_script_grant_panel.py -q && pytest Tests/Library -q`
Expected: PASS with no `Tests/Library` regression (note: `test_library_shell` has 4 known-failing baselines — confirm they are unchanged, not newly broken).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_skills_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_skill_script_grant_panel.py
git commit -m "feat(library): show and allow revoking skill script grants

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: End-to-end with real services

**Files:**
- Test: `Tests/Skills/test_e2e_run_skill_script.py`
- Modify (only if the e2e exposes a defect): any file from Tasks 1-7.

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: no new production interfaces.

- [ ] **Step 1: Write the failing end-to-end tests**

Create `Tests/Skills/test_e2e_run_skill_script.py`. Model it on `test_e2e_install_skill_from_github_tree_url_real_services` — real `ServicePolicyEnforcer` + `RuntimeSourceState` on BOTH services, real `LocalSkillsService`/`SkillsScopeService`/`SkillTrustService`, a real trusted skill on disk, and the real bridge closure. Only the confirm callback is faked.

```python
"""End-to-end: agent tool call -> real services -> real subprocess."""

import pytest


def test_agent_call_runs_a_real_script_and_returns_its_stdout(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": False})
    result = env.tool("demo-skill", "scripts/hello.py", [])
    assert result.ok is True
    assert "hello from a real skill script" in result.content
    assert "exit_code: 0" in result.content


def test_denied_confirm_never_runs_the_script(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": False, "remember": False})
    result = env.tool("demo-skill", "scripts/marker.py", [])
    assert result.ok is False
    assert not env.marker_path.exists(), "the script must never have executed"


def test_policy_disabled_denies_before_any_prompt(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": False}, policy_enabled=False)
    result = env.tool("demo-skill", "scripts/hello.py", [])
    assert result.ok is False
    assert env.confirm_calls == []


def test_untrusted_skill_is_refused_end_to_end(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": False}, trusted=False)
    result = env.tool("demo-skill", "scripts/hello.py", [])
    assert result.ok is False
    assert env.confirm_calls == []


def test_always_allow_persists_and_the_second_run_does_not_prompt(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": True})
    first = env.tool("demo-skill", "scripts/hello.py", [])
    second = env.tool("demo-skill", "scripts/hello.py", [])
    assert first.ok is True and second.ok is True
    assert len(env.confirm_calls) == 1, "the grant must suppress the second prompt"


def test_mutating_the_skill_after_a_grant_re_prompts(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": True})
    env.tool("demo-skill", "scripts/hello.py", [])
    env.mutate_script("print('changed')")
    env.retrust()
    env.tool("demo-skill", "scripts/hello.py", [])
    assert len(env.confirm_calls) == 2, (
        "a content change must invalidate the standing grant"
    )
```

- [ ] **Step 2: Run to verify they fail (or reveal real defects)**

Run: `source .venv/bin/activate && pytest Tests/Skills/test_e2e_run_skill_script.py -x -q`
Expected: FAIL initially (fixtures/wiring). Any failure that is a genuine defect in Tasks 1-7 gets FIXED in the owning file — do not weaken the e2e to make it pass.

- [ ] **Step 3: Build the fixture and fix anything it exposes**

Write `e2e_bridge_env` in the test module. It MUST:
- construct a real `ServicePolicyEnforcer` + `RuntimeSourceState` and pass it to BOTH the local and scope services (an enforcer-less scope service silently no-ops — the Task-6 non-vacuity lesson from the install layer);
- write a real skill dir with `scripts/hello.py` (`print('hello from a real skill script')`) and `scripts/marker.py` (writes `env.marker_path`), bootstrap trust, and approve it;
- build the REAL closure from `console_agent_bridge` (not a copy) with the fake confirm callback, exposing it as `env.tool`;
- record confirm invocations in `env.confirm_calls`;
- provide `mutate_script` / `retrust` helpers for the grant-invalidation test;
- prove non-vacuity for `policy_enabled=False` by MUTATION: temporarily disable the `skills.run_script.launch.local` row and confirm the test fails if the enforcer is not actually wired.

- [ ] **Step 4: Run the full affected suite**

Run:
```bash
source .venv/bin/activate && pytest Tests/Skills Tests/Agents Tests/Chat Tests/UI Tests/Library -q
```
Expected: PASS. Record any pre-existing baseline failures explicitly (compare against `git stash` + same command on the merge-base if unsure) — do not silently absorb them.

- [ ] **Step 5: Commit**

```bash
git add Tests/Skills/test_e2e_run_skill_script.py
git commit -m "test(skills): end-to-end run_skill_script through real services

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Post-Plan: Documentation

After Task 8, before requesting review:

- [ ] Document the `[skills]` config knobs (`script_scratch_root` and any `ScriptRunLimits` overrides) wherever the repo documents skills settings, noting the 3-arg `get_cli_setting("skills", "<key>", default)` form — the section-dict form silently returns `{}` (`config.py:3965`).
- [ ] Add the residual-risk note (network egress, user-level reads, macOS memory, opaque binaries, silent runs under a standing grant from any agent) to the skills documentation, matching spec §9.
