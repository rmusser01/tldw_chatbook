# Console Run Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code–style lifecycle hooks for Console chat sessions — user-configured external commands fired at six session/run events, with deny-only guardrails.

**Architecture:** A pure engine module (`Agents/run_hooks.py`) parses `[hooks]` config and executes argv-list subprocesses with a JSON stdin protocol; fire points ride existing seams — the review-hook chain (PreToolUse), a new optional runtime dep mirroring `run_skill_script` (PostToolUse), the submit path (UserPromptSubmit), the approval bridges and run-settle paths (ApprovalRequested, Stop, SubagentStop). The engine is a `ConsoleRuntime`-owned singleton so headless wake runs share it.

**Tech Stack:** Python ≥3.11 stdlib only (`subprocess`, `asyncio.to_thread`, `concurrent.futures`, `threading`) — no new dependencies.

**Spec:** `Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md` (ADR-148: `backlog/decisions/148-console-run-hooks.md`). The plan argues from the spec — read both.

## Global Constraints

- No new dependencies; stdlib only.
- Commands are argv lists — never a shell string, never `shell=True`.
- Deny-only: no hook output can produce an "allow"/"proceed" verdict; hooks only add refusals.
- Shared budget constant `HOOK_IO_BUDGET_CHARS = 4000` governs every truncation site (payload fields, stdout, stderr); default timeout `HOOK_DEFAULT_TIMEOUT_S = 10.0`.
- Blocking-event semantics: `PreToolUse` fails CLOSED on any unclean outcome; `UserPromptSubmit` fails OPEN (only an explicit block rejects the send).
- Event loop safety: nothing on the event loop may call blocking `engine.fire` — `submit_draft` is async on the loop, so that path uses `await engine.fire_async(...)`; the run/review chain runs in threads and uses `engine.fire(...)`.
- Timeouts kill the whole process group (`start_new_session=True` + `os.killpg`; `taskkill /T` on Windows).
- Targeted test runs only (AGENTS.md rule): never run the full suite unless the user asks.
- Commits add ONLY the files named in the step (`git add <paths>` then `git commit`) — this working tree carries unrelated WIP; do not sweep it (repo lesson, see `git log` fix commits).
- Settings sub-screen is OUT OF SCOPE (next PR); the config schema here is that PR's contract.

---

### Task 1: Engine config model + validation

**Files:**
- Create: `tldw_chatbook/Agents/run_hooks.py`
- Test: `Tests/Agents/test_run_hooks.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `HOOK_EVENTS`, `TOOL_NAME_EVENTS`, `HookSpec(event, command: tuple[str, ...], matcher: str | None, timeout_s: float)`, `RunHooksConfig(enabled: bool, hooks: tuple[HookSpec, ...])`, `load_hooks_config(config: Mapping) -> RunHooksConfig`, `HOOK_IO_BUDGET_CHARS`, `HOOK_DEFAULT_TIMEOUT_S`. Later tasks import these exact names.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the Console run-hooks engine (spec: 2026-09-11-console-run-hooks-design)."""

import sys

import pytest

from tldw_chatbook.Agents.run_hooks import (
    HOOK_DEFAULT_TIMEOUT_S,
    HOOK_EVENTS,
    HookSpec,
    RunHooksConfig,
    load_hooks_config,
)


def _cfg(enabled=True, **hook_kwargs):
    base = {"event": "PreToolUse", "command": ["/bin/true"]}
    base.update(hook_kwargs)
    return {"hooks": {"enabled": enabled, "hook": [base]} if hook_kwargs or True else None}


class TestLoadHooksConfig:
    def test_empty_section_is_no_op(self):
        result = load_hooks_config({})
        assert result == RunHooksConfig(enabled=True, hooks=())

    def test_master_switch_off(self):
        result = load_hooks_config({"hooks": {"enabled": False}})
        assert result.enabled is False

    def test_valid_hook_parsed(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": "PreToolUse", "matcher": "fs_*",
                                  "command": ["/bin/guard", "--strict"], "timeout_s": 5}]}}
        )
        assert result.hooks == (HookSpec("PreToolUse", ("/bin/guard", "--strict"), "fs_*", 5.0),)

    def test_unknown_event_disables_that_hook(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Nope", "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_matcher_rejected_on_non_tool_event(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "matcher": "fs_*",
                                                         "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_empty_command_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": []}]}})
        assert result.hooks == ()

    def test_non_positive_timeout_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"],
                                                         "timeout_s": 0}]}})
        assert result.hooks == ()

    def test_string_command_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": "/bin/true"}]}})
        assert result.hooks == ()

    def test_default_timeout_applied(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"]}]}})
        assert result.hooks[0].timeout_s == HOOK_DEFAULT_TIMEOUT_S
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_run_hooks.py -v`
Expected: collection error — `ModuleNotFoundError`/`ImportError` for `tldw_chatbook.Agents.run_hooks`.

- [ ] **Step 3: Implement the config model**

```python
# tldw_chatbook/Agents/run_hooks.py
"""Console run hooks: user-configured external commands on session/run lifecycle events.

Spec: Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md (ADR-148).
Deny-only guardrails, argv-list commands, JSON-on-stdin protocol, per-purpose
fail direction (PreToolUse fails closed, UserPromptSubmit fails open).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Mapping

from loguru import logger

HOOK_EVENTS: frozenset[str] = frozenset(
    {"UserPromptSubmit", "PreToolUse", "PostToolUse", "ApprovalRequested", "Stop", "SubagentStop"}
)
TOOL_NAME_EVENTS: frozenset[str] = frozenset({"PreToolUse", "PostToolUse"})
HOOK_IO_BUDGET_CHARS: int = 4000
HOOK_DEFAULT_TIMEOUT_S: float = 10.0
BLOCKING_EVENTS: frozenset[str] = frozenset({"UserPromptSubmit", "PreToolUse"})


@dataclass(frozen=True)
class HookSpec:
    event: str
    command: tuple[str, ...]
    matcher: str | None = None
    timeout_s: float = HOOK_DEFAULT_TIMEOUT_S

    def matches_tool(self, tool_name: str) -> bool:
        if self.matcher is None:
            return True
        return fnmatch.fnmatchcase(tool_name, self.matcher)


@dataclass(frozen=True)
class RunHooksConfig:
    enabled: bool = True
    hooks: tuple[HookSpec, ...] = ()


def _parse_hook(raw: object) -> HookSpec | None:
    if not isinstance(raw, dict):
        logger.warning("run-hooks: hook entry is not a table; disabled: {!r}", raw)
        return None
    event = raw.get("event")
    if event not in HOOK_EVENTS:
        logger.warning("run-hooks: unknown event {!r}; hook disabled", event)
        return None
    command = raw.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
        logger.warning("run-hooks: command must be a non-empty list of strings; hook disabled")
        return None
    matcher = raw.get("matcher")
    if matcher is not None:
        if event not in TOOL_NAME_EVENTS:
            logger.warning("run-hooks: matcher is only valid on {}, got {}; hook disabled",
                           sorted(TOOL_NAME_EVENTS), event)
            return None
        if not isinstance(matcher, str) or not matcher:
            logger.warning("run-hooks: matcher must be a non-empty string; hook disabled")
            return None
    timeout = raw.get("timeout_s", HOOK_DEFAULT_TIMEOUT_S)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        logger.warning("run-hooks: timeout_s must be positive; hook disabled")
        return None
    return HookSpec(event=event, command=tuple(command), matcher=matcher, timeout_s=float(timeout))


def load_hooks_config(config: Mapping) -> RunHooksConfig:
    section = config.get("hooks") if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        return RunHooksConfig(enabled=True, hooks=())
    enabled = section.get("enabled", True)
    hooks = tuple(h for h in (_parse_hook(r) for r in section.get("hook", [])) if h is not None)
    return RunHooksConfig(enabled=bool(enabled), hooks=hooks)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_run_hooks.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/run_hooks.py Tests/Agents/test_run_hooks.py
git commit -m "feat(run-hooks): config model + fail-loud validation for [hooks]"
```

---

### Task 2: Engine execution core

**Files:**
- Modify: `tldw_chatbook/Agents/run_hooks.py`
- Test: `Tests/Agents/test_run_hooks.py`

**Interfaces:**
- Consumes: Task 1's `HookSpec`, `RunHooksConfig`, `HOOK_IO_BUDGET_CHARS`, `BLOCKING_EVENTS`.
- Produces: `HookOutcome(blocked: bool, reason: str, context: str)`, `RunHooksEngine(config_provider, cwd_provider)` with `.fire(event, *, session_id, run_id, data) -> HookOutcome`, `async .fire_async(...) -> HookOutcome`, `.notify(event, **same) -> None`. Tasks 5–8 consume these.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Agents/test_run_hooks.py`)

```python
import asyncio
import json
import os
import signal
import time

from tldw_chatbook.Agents.run_hooks import HookOutcome, RunHooksEngine


def _engine(*hooks, enabled=True):
    cfg = RunHooksConfig(enabled=enabled, hooks=tuple(hooks))
    return RunHooksEngine(lambda: cfg, lambda: os.getcwd())


class TestFire:
    def test_exit0_clean_pass_logs_stdout(self):
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", "print('done')")))
        out = eng.fire("Stop", session_id="s", run_id="r", data={})
        assert out == HookOutcome(blocked=False, reason="", context="")

    def test_json_decision_beats_exit_code(self):
        code = "import sys; print(json.dumps({'decision':'deny','reason':'nope'}))" \
               .replace("json.dumps", "__import__('json').dumps")
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "fs_write"})
        assert out.blocked is True and out.reason == "nope"

    def test_exit2_is_deny_shorthand(self):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c",
            "import sys; sys.stderr.write('blocked'); sys.exit(2)")))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and "blocked" in out.reason

    def test_allow_decision_ignored(self):
        code = "print(__import__('json').dumps({'decision':'allow'}))"
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is False

    def test_pretooluse_crash_fails_closed(self):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", "sys.exit(1)")))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and "failed" in out.reason

    def test_userpromptsubmit_crash_fails_open(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "sys.exit(1)")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False

    def test_userpromptsubmit_stdout_is_context(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('extra ctx')")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False and out.context == "extra ctx"

    def test_userpromptsubmit_block(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c",
            "print(__import__('json').dumps({'decision':'block','reason':'no'}))")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is True and out.reason == "no"

    def test_payload_envelope_on_stdin(self):
        seen = {}
        code = ("import sys, json; p=json.load(sys.stdin); "
                "open(%r,'w').write(json.dumps(p))" % "/tmp/hook_payload.json")
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", code)))
        eng.fire("Stop", session_id="s1", run_id="r1", data={"status": "completed"})
        with open("/tmp/hook_payload.json") as fh:
            seen = json.load(fh)
        assert seen["hook_event"] == "Stop" and seen["session_id"] == "s1"
        assert seen["run_id"] == "r1" and seen["data"] == {"status": "completed"}
        assert "timestamp" in seen and "cwd" in seen

    def test_truncation_to_budget(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('x'*999999)")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert len(out.context) <= HOOK_IO_BUDGET_CHARS

    def test_first_deny_wins_across_concurrent_hooks(self):
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c",
                "import time; time.sleep(0.5); print('late')")),
            HookSpec("PreToolUse", (sys.executable, "-c", "sys.exit(2)")),
        )
        start = time.monotonic()
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and time.monotonic() - start < 0.45

    def test_master_switch_off_fires_nothing(self):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", "sys.exit(2)")), enabled=False)
        assert eng.fire("PreToolUse", session_id="s", data={}).blocked is False

    def test_engine_never_raises(self):
        eng = _engine(HookSpec("Stop", ("/nonexistent/hook-binary", "--x")))
        assert eng.fire("Stop", session_id="s", data={}) == HookOutcome()


class TestTimeoutKill:
    def test_timeout_kills_process_group(self):
        if sys.platform == "win32":
            pytest.skip("POSIX process-group test")
        child_marker = "/tmp/hook_child_alive.txt"
        code = (
            "import subprocess, sys, time;"
            "subprocess.Popen([sys.executable, '-c', "
            "\"open('%s','w').write('alive'); import time; time.sleep(30)\"]);"
            "time.sleep(30)" % child_marker
        )
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code), timeout_s=1.0))
        out = eng.fire("PreToolUse", session_id="s", data={})
        assert out.blocked is True  # fail-closed on timeout
        time.sleep(0.4)  # give the orphan-check a moment
        try:
            os.remove(child_marker)
            pytest.fail("hook's child survived the process-group kill")
        except FileNotFoundError:
            pass  # marker never written OR child died before writing: group died


class TestFireAsync:
    def test_loop_stays_responsive_while_hook_sleeps(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c",
            "import time; time.sleep(1)")))

        async def main():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.05)
                    ticks += 1

            task = asyncio.create_task(ticker())
            await eng.fire_async("UserPromptSubmit", session_id="s", data={"prompt": "x"})
            task.cancel()
            return ticks

        ticks = asyncio.run(main())
        assert ticks >= 8  # loop kept ticking through the 1s hook


class TestNotify:
    def test_notify_returns_immediately_and_eventually_runs(self):
        marker = "/tmp/hook_notify_marker.txt"
        try:
            os.remove(marker)
        except FileNotFoundError:
            pass
        eng = _engine(HookSpec("PostToolUse", (sys.executable, "-c",
            "open('%s','w').write('x')" % marker)))
        eng.notify("PostToolUse", session_id="s", data={"tool_name": "t"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not os.path.exists(marker):
            time.sleep(0.05)
        assert os.path.exists(marker)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_run_hooks.py -v`
Expected: FAIL — `HookOutcome`/`RunHooksEngine` not importable.

- [ ] **Step 3: Implement the execution core** (append to `run_hooks.py`)

```python
import asyncio
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HookOutcome:
    blocked: bool = False
    reason: str = ""
    context: str = ""


def _truncate(text: str) -> str:
    return text if len(text) <= HOOK_IO_BUDGET_CHARS else text[:HOOK_IO_BUDGET_CHARS] + "…[truncated]"


@dataclass(frozen=True)
class _Decision:
    denied: bool = False
    reason: str = ""
    context: str = ""


def _decide(event: str, proc: subprocess.CompletedProcess) -> _Decision:
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(stdout)
        parsed = candidate if isinstance(candidate, dict) else None
    except (ValueError, TypeError):
        parsed = None
    decision = (parsed or {}).get("decision")
    if event == "UserPromptSubmit":
        if decision == "block" or proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "blocked by hook")
            return _Decision(denied=True, reason=reason)
        return _Decision(context=_truncate(stdout.strip()))
    if event == "PreToolUse":
        if decision == "deny" or proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "denied by hook")
            return _Decision(denied=True, reason=reason)
        if proc.returncode != 0:
            return _Decision(denied=True, reason=f"hook failed (exit {proc.returncode}); failing closed")
        return _Decision()
    return _Decision()


def _kill_process_group(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _run_hook(spec: HookSpec, payload: dict[str, Any]) -> tuple[HookSpec, _Decision | None]:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            spec.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=payload.get("cwd") or None,
            start_new_session=(sys.platform != "win32"),
        )
    except OSError as exc:
        logger.warning("run-hooks: failed to start {}: {}", spec.command, exc)
        return spec, (_Decision(denied=True, reason="hook failed to start; failing closed")
                      if spec.event == "PreToolUse" else None)
    try:
        stdout, stderr = proc.communicate(json.dumps(payload), timeout=spec.timeout_s)
        completed = subprocess.CompletedProcess(spec.command, proc.returncode,
                                                stdout=stdout, stderr=stderr)
        decision = _decide(spec.event, completed)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.communicate()
        decision = (_Decision(denied=True, reason=f"hook timed out after {spec.timeout_s}s; failing closed")
                    if spec.event == "PreToolUse" else None)
    logger.info("run-hooks: event={} hook={} exit={} took_ms={}",
                spec.event, spec.command[0], proc.returncode,
                int((time.monotonic() - started) * 1000))
    return spec, decision


class RunHooksEngine:
    """Executes configured hooks for lifecycle events. Never raises to callers."""

    def __init__(self, config_provider: Callable[[], RunHooksConfig],
                 cwd_provider: Callable[[], str]) -> None:
        self._config_provider = config_provider
        self._cwd_provider = cwd_provider
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="run-hook")
        self._notify_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-hook-notify")

    def _matching(self, event: str, tool_name: str | None) -> list[HookSpec]:
        cfg = self._config_provider()
        if not cfg.enabled:
            return []
        return [h for h in cfg.hooks if h.event == event and tool_name is not None
                and h.matches_tool(tool_name)] or \
               [h for h in cfg.hooks if h.event == event and tool_name is None]

    def _payload(self, event: str, session_id: str, run_id: str | None, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "hook_event": event,
            "session_id": session_id,
            "run_id": run_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cwd": self._cwd_provider(),
            "data": data,
        }

    def fire(self, event: str, *, session_id: str, run_id: str | None = None,
             data: dict[str, Any] | None = None) -> HookOutcome:
        tool_name = (data or {}).get("tool_name")
        specs = self._matching(event, tool_name)
        if not specs:
            return HookOutcome()
        payload = self._payload(event, session_id, run_id, data or {})
        futures = [self._pool.submit(_run_hook, s, payload) for s in specs]
        outcome = HookOutcome()
        for fut, spec in zip(futures, specs):
            try:
                _, decision = fut.result()
            except Exception as exc:  # engine never raises to callers
                logger.warning("run-hooks: hook {} raised {}", spec.command, exc)
                decision = _Decision(denied=True, reason="hook raised; failing closed") \
                    if spec.event == "PreToolUse" else None
            if decision is None:
                continue
            if decision.denied and not outcome.blocked:
                outcome = HookOutcome(blocked=True, reason=decision.reason)
            if decision.context and not outcome.context:
                outcome = HookOutcome(blocked=outcome.blocked, reason=outcome.reason,
                                      context=decision.context)
        return outcome

    async def fire_async(self, event: str, **kwargs: Any) -> HookOutcome:
        return await asyncio.to_thread(self.fire, event, **kwargs)

    def notify(self, event: str, **kwargs: Any) -> None:
        try:
            self._notify_worker.submit(self.fire, event, **kwargs)
        except RuntimeError:  # executor shutdown/queue full — drop, never stall chat
            logger.warning("run-hooks: notify executor unavailable; {} dropped", event)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_run_hooks.py -v`
Expected: all PASS. If `test_timeout_kills_process_group` flakes on the marker race, strengthen the child to write the marker immediately before the parent's sleep, not after.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/run_hooks.py Tests/Agents/test_run_hooks.py
git commit -m "feat(run-hooks): subprocess execution core — JSON protocol, deny-only, fail direction, process-group timeout"
```

---

### Task 3: Review-hook integration (`wrap_review`)

**Files:**
- Modify: `tldw_chatbook/Agents/run_hooks.py`
- Test: `Tests/Agents/test_run_hooks.py`

**Interfaces:**
- Consumes: Task 2's `RunHooksEngine.fire`; `ToolCall(name, args: dict, call_id, raw_arguments)` from `Agents/agent_models.py`.
- Produces: `RunHooksEngine.wrap_review(inner, *, session_id) -> Callable[[list[ToolCall], str], dict[str, str]]`. Task 6 consumes it. Verdict maps are **name-keyed** (agent_runtime.py:352: "Returns a name -> verdict map"); `"proceed"` or a refusal string.

- [ ] **Step 1: Write the failing tests** (append)

```python
from tldw_chatbook.Agents.agent_models import ToolCall


class TestWrapReview:
    def _engine_with(self, spec):
        cfg = RunHooksConfig(enabled=True, hooks=(spec,))
        return RunHooksEngine(lambda: cfg, lambda: os.getcwd())

    def test_deny_short_circuits_before_inner(self):
        eng = self._engine_with(HookSpec("PreToolUse", (sys.executable, "-c", "sys.exit(2)"),
                                         matcher="fs_*"))
        called = []

        def inner(calls, run_id):
            called.append([c.name for c in calls])
            return {c.name: "proceed" for c in calls}

        wrapped = eng.wrap_review(inner, session_id="s")
        verdicts = wrapped([ToolCall(name="fs_write", args={"path": "x"}),
                            ToolCall(name="calculator", args={})], "run-1")
        assert verdicts["fs_write"] != "proceed" and "hook" in verdicts["fs_write"].lower() \
            or verdicts["fs_write"] == ""  # refusal string per protocol
        assert verdicts["fs_write"] not in ("proceed",)
        assert called == [["calculator"]]  # denied call never reaches the permission store

    def test_clean_pass_delegates_to_inner(self):
        eng = self._engine_with(HookSpec("PreToolUse", (sys.executable, "-c", "pass")))

        def inner(calls, run_id):
            return {c.name: "proceed" for c in calls}

        wrapped = eng.wrap_review(inner, session_id="s")
        assert wrapped([ToolCall(name="fs_write", args={})], "run-1") == {"fs_write": "proceed"}

    def test_no_hooks_configured_is_identity(self):
        eng = _engine()

        def inner(calls, run_id):
            return {c.name: "proceed" for c in calls}

        assert eng.wrap_review(inner, session_id="s") is inner
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_run_hooks.py::TestWrapReview -v`
Expected: FAIL — `wrap_review` not defined.

- [ ] **Step 3: Implement `wrap_review`** (method on `RunHooksEngine`)

```python
    def wrap_review(self, inner: Callable[[list, str], dict[str, str]], *,
                    session_id: str) -> Callable[[list, str], dict[str, str]]:
        """Wrap a review_tool_calls callable so PreToolUse hooks deny first.

        Denied names are EXCLUDED from the inner review (no approval card for
        a call a hook already refused) and merged back as refusal strings.
        Hooks can never produce "proceed" — deny-only, enforced here.
        """
        if not any(h.event == "PreToolUse" for h in self._config_provider().hooks):
            return inner

        def review(calls: list, run_id: str) -> dict[str, str]:
            refusals: dict[str, str] = {}
            surviving = []
            for call in calls:
                outcome = self.fire(
                    "PreToolUse", session_id=session_id, run_id=run_id,
                    data={"tool_name": call.name, "tool_args": call.args},
                )
                if outcome.blocked:
                    refusals[call.name] = outcome.reason
                else:
                    surviving.append(call)
            verdicts = dict(inner(surviving, run_id)) if surviving else {}
            verdicts.update(refusals)
            return verdicts

        return review
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_run_hooks.py -v`
Expected: all PASS. Clean up the first test's sloppy assertion to a single strict line: `assert verdicts["fs_write"] not in ("proceed",)` and `assert called == [["calculator"]]`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/run_hooks.py Tests/Agents/test_run_hooks.py
git commit -m "feat(run-hooks): deny-only PreToolUse wrapper around the review verdict chain"
```

---

### Task 4: Runtime ownership + `[hooks]` config section

**Files:**
- Modify: `tldw_chatbook/Chat/console_runtime.py` (beside `ensure_chat_store`, ~line 550; follow its idempotent `ensure_*` pattern)
- Modify: `tldw_chatbook/config.py` (follow the `coerce_bool_setting` pattern at ~line 1549)
- Modify: `tldw_chatbook/config.toml.example` (if the example template lives elsewhere, `rg -l "local_tools_enabled" --glob "*.toml*"` finds it)
- Test: `Tests/Agents/test_run_hooks.py`, `Tests/Chat/test_console_viewless_hooks.py` (append ownership test there — it is the runtime-ownership test home)

**Interfaces:**
- Consumes: `RunHooksEngine`, `load_hooks_config` (Tasks 1–2); `ensure_console_runtime` (`Chat/console_runtime.py:1087`).
- Produces: `ConsoleRuntime.ensure_run_hooks() -> RunHooksEngine | None` — returns `None` when no hooks are configured (so every fire site can `if engine is not None`). Tasks 5–8 consume this.

- [ ] **Step 1: Write the failing test** (append to `Tests/Chat/test_console_viewless_hooks.py`)

```python
def test_ensure_run_hooks_is_idempotent_and_none_without_config():
    from tldw_chatbook.Chat.console_runtime import ensure_console_runtime
    runtime = ensure_console_runtime(app=None)  # follow this file's existing fixture style
    engine1 = runtime.ensure_run_hooks()
    engine2 = runtime.ensure_run_hooks()
    assert engine1 is engine2
    # with an empty config the engine is absent: fire sites skip entirely
```

(Adapt the `ensure_console_runtime(...)` invocation to however this test module already builds its runtime — read its existing fixtures first and reuse them.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_viewless_hooks.py -k run_hooks -v`
Expected: FAIL — `ensure_run_hooks` missing.

- [ ] **Step 3: Implement**

In `console_runtime.py`, inside `class ConsoleRuntime` (beside `ensure_chat_store`):

```python
    def ensure_run_hooks(self) -> "RunHooksEngine | None":
        """App-owned hooks engine (spec 2026-09-11). None = no [hooks] configured.

        Idempotent like every ensure_* here: the config provider reads the
        app's loaded config fresh each call, so config edits apply to the
        next fire without rebuilding the engine (mtime-cached upstream).
        """
        existing = getattr(self, "_run_hooks_engine", _UNSET)
        if existing is not _UNSET:
            return existing
        from tldw_chatbook.Agents.run_hooks import RunHooksEngine, load_hooks_config
        app_config = getattr(self._app, "config", {}) or {}
        engine: RunHooksEngine | None = None
        if load_hooks_config(app_config).hooks:
            engine = RunHooksEngine(lambda: load_hooks_config(
                getattr(self._app, "config", {}) or {}),
                lambda: str(getattr(self._app, "workspace_root", "") or os.getcwd()))
        self._run_hooks_engine = engine
        return engine
```

Add `_UNSET = object()` at module level and `_run_hooks_engine: Any = _UNSET` handling consistent with how this class stores lazily-built objects (mirror `ensure_chat_store`'s attribute pattern; adapt `self._app` to whatever the runtime actually holds — read the class fields first).

In `config.py`, beside the console settings assembly (~line 1549):

```python
final_console_settings_cli["hooks_enabled"] = coerce_bool_setting(
    final_console_settings_cli.get("hooks_enabled", True),
)
```

— only if hooks need a UI-visible mirror flag; otherwise the engine reads the raw `[hooks]` table directly and `config.py` needs nothing beyond defaults documentation. Prefer the latter (no `config.py` change) if the loaded config object already exposes raw sections to the runtime; verify with `rg -n "def get_console_settings|raw_sections|self\\._config" tldw_chatbook/config.py | head` and pick the existing raw-section accessor.

Append to the example config template (beside `local_tools_enabled`, ~line 2953):

```toml
[hooks]
enabled = true
# [[hooks.hook]] entries: event / matcher / command / timeout_s
# event: UserPromptSubmit | PreToolUse | PostToolUse | ApprovalRequested | Stop | SubagentStop
# matcher: tool-name glob, valid only on PreToolUse/PostToolUse
# command: argv list, no shell — e.g. ["/usr/local/bin/guard.sh", "--strict"]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_viewless_hooks.py -k run_hooks Tests/Agents/test_run_hooks.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/config.py tldw_chatbook/config.toml.example Tests/Chat/test_console_viewless_hooks.py
git commit -m "feat(run-hooks): ConsoleRuntime-owned engine singleton + [hooks] config surface"
```

---

### Task 5: `post_tool_call` runtime dep (PostToolUse fire point)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (dep field beside `run_skill_script` ~line 391; fire point in the dispatch loop immediately after BOTH `_emit_record(... "tool_result" ...)` calls, ~lines 1724–1742, using the still-full `content`, guarded by `verdict == "proceed"`)
- Modify: `tldw_chatbook/Agents/agent_service.py` (ctor param beside `run_skill_script_tool` ~line 1045; storage ~line 1151; deps wiring beside `run_skill_script=self._run_skill_script_tool` ~line 4701)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (pass `post_tool_call=` at the `AgentService(` construction — `rg -n "AgentService\(" tldw_chatbook/Chat/console_agent_bridge.py`)
- Test: `Tests/Agents/test_run_hooks.py` (dep contract), `Tests/Agents/test_agent_runtime_review_hook.py` (append fire-point test — it is the runtime dispatch test home)

**Interfaces:**
- Consumes: `ensure_run_hooks()` (Task 4); `ToolCall(name, args, call_id)`.
- Produces: `AgentDeps.post_tool_call: Callable[[str, str, dict, str, bool], None] | None` — `(tool_name, call_id, args, result_content, ok)`; fires only on dispatched calls; engine truncates content. `RunHooksEngine.post_tool_dep(*, session_id) -> Callable[...]` helper.

- [ ] **Step 1: Write the failing tests**

In `Tests/Agents/test_run_hooks.py` (append):

```python
class TestPostToolDep:
    def test_dep_fires_notify_with_full_payload(self):
        fired = []
        eng = _engine(HookSpec("PostToolUse", (sys.executable, "-c", "pass")))
        dep = eng.post_tool_dep(session_id="s")
        # intercept notify to observe without racing the executor
        eng.notify = lambda event, **kw: fired.append((event, kw))
        dep("fs_write", "call-1", {"path": "x"}, "full result " * 1000, True)
        event, kw = fired[0]
        assert event == "PostToolUse"
        assert kw["data"]["tool_name"] == "fs_write"
        assert len(kw["data"]["tool_result"]) <= HOOK_IO_BUDGET_CHARS
        assert kw["data"]["is_error"] is False
```

In `Tests/Agents/test_agent_runtime_review_hook.py` (append, reusing this module's existing dispatch-loop test fixtures — read them first):

```python
def test_post_tool_call_dep_fires_only_on_dispatched_calls():
    """Refused calls fire nothing; dispatched calls carry name/call_id/args/content."""
    fired = []
    # Build the module's existing minimal run harness; set deps.post_tool_call = fired.append
    # and deps.review_tool_calls to refuse one call and proceed another.
    # Assert: exactly one firing, for the proceeded call, content == full result.
```

— then concretize it against the harness: if this module's fixtures drive `_run_one` directly, wire `post_tool_call` into its `LoopDeps`; if they go through `AgentService`, add the ctor param first and assert through one level. Do not leave the test as a sketch — write the real body using the fixtures you find.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_run_hooks.py::TestPostToolDep Tests/Agents/test_agent_runtime_review_hook.py -v`
Expected: FAIL — `post_tool_dep` missing.

- [ ] **Step 3: Implement**

`run_hooks.py` (method on `RunHooksEngine`):

```python
    def post_tool_dep(self, *, session_id: str) -> Callable[[str, str, dict, str, bool], None]:
        def post_tool_call(tool_name: str, call_id: str, args: dict,
                           content: str, ok: bool) -> None:
            self.notify("PostToolUse", session_id=session_id,
                        data={"tool_name": tool_name, "tool_args": args,
                              "tool_result": _truncate(content),
                              "is_error": not ok})
        return post_tool_call
```

`agent_runtime.py` — in the `LoopDeps`/deps dataclass beside `run_skill_script`:

```python
    # run-hooks PostToolUse: fired at the dispatch capture point, ONLY for
    # calls that actually dispatched (verdict == "proceed") — refusals fire
    # nothing. Receives the still-UNCAPPED content; the engine truncates.
    post_tool_call: Callable[[str, str, dict, str, bool], None] | None = None
```

In the dispatch loop, immediately after each `_emit_record(..., "tool_result", ...)` call (both the continuation-checkpoint branch — where the full text is `full_content` — and the plain branch — where `content` is still full), add:

```python
            if deps.post_tool_call is not None and verdict == "proceed":
                try:
                    deps.post_tool_call(call.name, call.call_id, call.args,
                                        full_content, result.ok)
                except Exception as exc:
                    logger.warning("post_tool_call consumer raised (exception_type={})", type(exc).__name__)
```

(use `content` instead of `full_content` in the plain branch).

`agent_service.py` — ctor param beside `run_skill_script_tool`:

```python
        post_tool_call: Callable[[str, str, dict, str, bool], None] | None = None,
```

storage beside `self._run_skill_script_tool = run_skill_script_tool`:

```python
        self._post_tool_call = post_tool_call
```

deps wiring beside `run_skill_script=self._run_skill_script_tool`:

```python
            post_tool_call=self._post_tool_call,
```

`console_agent_bridge.py` — at the `AgentService(` construction, add:

```python
            post_tool_call=(
                engine.post_tool_dep(session_id=session_id)
                if (engine := runtime.ensure_run_hooks()) is not None
                else None
            ),
```

adapting `runtime`/`session_id` to the names in scope at that construction site.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_run_hooks.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_fleet_continuation.py -v`
Expected: all PASS (`test_fleet_continuation.py` guards the deps shape changes).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Agents/test_run_hooks.py Tests/Agents/test_agent_runtime_review_hook.py
git commit -m "feat(run-hooks): post_tool_call runtime dep fires PostToolUse at the capture point"
```

---

### Task 6: PreToolUse wiring in the bridge

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (where the run's review callable is built from `build_tool_review_hook` — `rg -n "build_tool_review_hook|review_tool_calls=" tldw_chatbook/Chat/console_agent_bridge.py`)
- Test: `Tests/Chat/test_console_local_review_hook.py` (append — it is the review-hook integration home)

**Interfaces:**
- Consumes: `ensure_run_hooks()` (Task 4), `wrap_review` (Task 3).
- Produces: every run's `review_tool_calls` passes through `engine.wrap_review(...)` when hooks exist.

- [ ] **Step 1: Write the failing test** (append to `Tests/Chat/test_console_local_review_hook.py`, reusing its `ConsoleChatController` fixtures)

```python
def test_pretooluse_hook_denies_before_permission_store(tmp_path):
    """A configured exit-2 PreToolUse hook denies the matched call and the
    approval round never carries it (deny-only, spec §5)."""
    import sys as _sys
    from tldw_chatbook.Agents.run_hooks import HookSpec, RunHooksConfig, RunHooksEngine

    engine = RunHooksEngine(
        lambda: RunHooksConfig(enabled=True, hooks=(
            HookSpec("PreToolUse", (_sys.executable, "-c", "sys.exit(2)"), matcher="fs_*"),)),
        lambda: str(tmp_path))
    # Follow this module's existing pattern for building a controller + calls;
    # wrap its review callable: wrapped = engine.wrap_review(inner, session_id=SESSION)
    # Assert: fs_* call's verdict != "proceed"; non-matching call proceeds.
```

Concretize using the module's existing `build_combined_review_hook`/controller fixture pattern — write the real body; the assertion set is `verdicts["<fs tool>"] != "proceed"` and `verdicts["<other tool>"] == "proceed"`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_local_review_hook.py -k pretooluse -v`
Expected: FAIL (test body asserts on wiring that does not exist yet — once concretized it fails because the bridge does not wrap).

Note: this test exercises `wrap_review` directly (unit-level); the bridge wiring is asserted in Task 8's sweep.

- [ ] **Step 3: Implement the bridge wiring**

At the review-callable construction site in `console_agent_bridge.py`:

```python
        review_callable = build_tool_review_hook(...)  # existing call, unchanged
        engine = runtime.ensure_run_hooks()
        if engine is not None:
            review_callable = engine.wrap_review(review_callable, session_id=session_id)
```

adapting variable names to the site (it may be an inline kwarg — in that case bind it to a local first, wrap, then pass).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_local_review_hook.py Tests/Chat/test_console_provider_gateway.py Tests/Agents/test_agent_runtime_review_hook.py -v`
Expected: all PASS (the gateway tests guard the review chain end to end).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_local_review_hook.py
git commit -m "feat(run-hooks): bridge wraps the run review chain with PreToolUse hooks"
```

---

### Task 7: UserPromptSubmit in the submit path

**Files:**
- Modify: `tldw_chatbook/Chat/message_metadata.py` (beside `MESSAGE_ORIGIN_AGENT_WAKE` line 58 and `MESSAGE_ORIGINS` line 74)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (submit path; anchor: the wake-guarded `_record_prompt_history` call at ~line 3813–3817)
- Test: `Tests/Chat/test_console_chat_controller.py` (append — the submit-path test home)

**Interfaces:**
- Consumes: `ensure_run_hooks()` (Task 4 — reach it via the controller's runtime reference; check how the controller accesses `ConsoleRuntime`), `fire_async` (Task 2), `ConsoleSubmitResult(accepted, should_clear_draft, visible_copy)`.
- Produces: `MESSAGE_ORIGIN_HOOK = "hook"`; hooks block sends (refusal result + SYSTEM row) or inject context (SYSTEM row carrying hook origin, persisted, visible — never merged into the user's message).

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_chat_controller.py`, reusing its submit fixtures — read them first and reuse the existing draft/controller harness)

```python
class TestUserPromptSubmitHooks:
    def test_block_rejects_send_without_assistant_row(self):
        # engine configured with an exit-2 UserPromptSubmit hook; submit_draft
        # returns accepted=False, should_clear_draft=False, reason in visible_copy;
        # no ASSISTANT row appended.
        ...

    def test_stdout_injected_as_hook_origin_system_row(self):
        # engine whose hook prints "ctx"; submit proceeds; a SYSTEM row with
        # metadata origin == MESSAGE_ORIGIN_HOOK and content "ctx" exists,
        # persisted, after the user row and before the assistant row.
        ...

    def test_wake_notice_fires_nothing(self):
        # origin=AGENT_WAKE submit never invokes the engine (wake invariant).
        ...

    def test_hook_crash_send_proceeds(self):
        # exit-1 hook: send proceeds, no SYSTEM row, warning logged.
        ...
```

Write the four bodies against the module's existing submit harness (`rg -n "async def test.*submit" Tests/Chat/test_console_chat_controller.py | head` shows the pattern to copy); the engine is injected by patching the controller's runtime `ensure_run_hooks` to return a `RunHooksEngine` built over a temp config.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_chat_controller.py -k UserPromptSubmitHooks -v`
Expected: FAIL — constant missing / no firing.

- [ ] **Step 3: Implement**

`message_metadata.py`:

```python
MESSAGE_ORIGIN_HOOK = "hook"
```

and extend `MESSAGE_ORIGINS` to `frozenset({"", MESSAGE_ORIGIN_AGENT_WAKE, MESSAGE_ORIGIN_HOOK})` (grep for tests asserting that set's contents — `rg -n "MESSAGE_ORIGINS" Tests/` — and update them in the same commit).

`console_chat_controller.py`, in `_submit_draft_inner` immediately AFTER the wake-guarded `_record_prompt_history` block (both share the manual-origin condition):

```python
        # Run hooks (spec 2026-09-11): manual-origin sends only — a wake
        # notice is machine text and never fires UserPromptSubmit.
        if origin is not ConsoleSubmissionOrigin.AGENT_WAKE:
            hooks_engine = self._run_hooks_engine()  # returns None when unconfigured
            if hooks_engine is not None:
                outcome = await hooks_engine.fire_async(
                    "UserPromptSubmit", session_id=session.id,
                    data={"prompt": _truncate(clean_draft)})
                if outcome.blocked:
                    self.store.append_message(
                        session.id, role=ConsoleMessageRole.SYSTEM,
                        content=f"Send blocked by hook: {outcome.reason}",
                        persist=self.store.persistence is not None,
                        metadata=MessageMetadata(origin=MESSAGE_ORIGIN_HOOK))
                    return ConsoleSubmitResult(accepted=False, should_clear_draft=False,
                                               visible_copy=f"Blocked by hook: {outcome.reason}")
                if outcome.context:
                    self.store.append_message(
                        session.id, role=ConsoleMessageRole.SYSTEM,
                        content=outcome.context,
                        persist=self.store.persistence is not None,
                        metadata=MessageMetadata(origin=MESSAGE_ORIGIN_HOOK))
```

Add the small accessor on the controller (it must work viewless — headless wake shares this code path):

```python
    def _run_hooks_engine(self):
        runtime = getattr(self, "_runtime_ref", None)
        return runtime.ensure_run_hooks() if runtime is not None else None
```

— adapt `_runtime_ref` to how the controller actually reaches `ConsoleRuntime` (`rg -n "console_runtime|ensure_console_runtime" tldw_chatbook/Chat/console_chat_controller.py | head`); if it holds no reference today, thread one through its constructor the way other app-owned services reach it, keeping the parameter optional (default `None`) so existing tests construct unchanged.

Note `MESSAGE_ORIGIN_HOOK` must be imported where `MESSAGE_ORIGIN_AGENT_WAKE` already is.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_headless_wake_invariants.py -v`
Expected: all PASS (the headless-wake invariants guard the wake-never-fires rule).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/message_metadata.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(run-hooks): UserPromptSubmit — block/inject on manual sends, wake exempt"
```

---

### Task 8: Stop, SubagentStop, ApprovalRequested fire points

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (approval round site — anchor: `request_mcp_approvals`, where `add_pending_round` registers the round before consulting the bridge slots, ~lines 4100–4140)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (run terminal state — anchor: the once-guarded branches that invoke `notify_run_outcome`/`notify_run_failure`, `rg -n "notify_run_outcome|notify_run_failure" tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py`; child settle — the `on_child_settled=functools.partial(...)` at ~line 4211)
- Test: `Tests/Agents/test_fleet_send_to_agent.py` (settle path), `Tests/Chat/test_console_fleet_wake.py` (headless), `Tests/Chat/test_console_local_review_hook.py` (approval path)

**Interfaces:**
- Consumes: `ensure_run_hooks()` (Task 4), `engine.notify` (Task 2).
- Produces: complete six-event firing per the spec's event table.

- [ ] **Step 1: Write the failing tests**

In `Tests/Chat/test_console_local_review_hook.py` (approval round):

```python
def test_approval_requested_fires_with_round_payload():
    # Patch the controller's engine accessor with a recording fake;
    # drive one ask-state call through request_mcp_approvals;
    # assert engine.notify was called once with event="ApprovalRequested",
    # data carrying the call name and session_active=True.
```

In `Tests/Agents/test_fleet_send_to_agent.py` (child settle):

```python
def test_subagent_stop_fires_on_child_settle():
    # Recording fake engine on the bridge; drive a child run to settle;
    # assert notify("SubagentStop", data={"child_run_id": ..., "status": ...}).
```

In `Tests/Chat/test_console_fleet_wake.py` (headless Stop):

```python
def test_stop_fires_for_wake_run_terminal_state():
    # Recording fake engine; headless wake run reaches terminal state;
    # assert notify("Stop", data={"status": ...}) fired despite no view.
```

Concretize all three against each module's existing fixtures (they exist precisely for these paths); each asserts a single `notify` call with the exact event name and payload keys.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_local_review_hook.py Tests/Agents/test_fleet_send_to_agent.py Tests/Chat/test_console_fleet_wake.py -v`
Expected: new tests FAIL (no firing yet).

- [ ] **Step 3: Implement the three fire sites**

Approval (controller, in `request_mcp_approvals`, immediately after the round is registered — after `add_pending_round`, before the bridge slots are consulted):

```python
            if (engine := self._run_hooks_engine()) is not None:
                engine.notify("ApprovalRequested", session_id=session_id,
                              data={"calls": [{"name": c.name} for c in pending_calls],
                                    "session_active": True})
```

(adapt `pending_calls` to the local variable holding the round's `MCPPendingCall` list; the parked/background round site — the `park_pending_approval` consumer, ~line 4637's territory — fires the same event with `session_active=False`. Find it with `rg -n "park_pending_approval\\(" tldw_chatbook/Chat/console_chat_controller.py` and mirror.)

Stop (bridge, beside BOTH once-guarded terminal branches that invoke `notify_run_outcome` / `notify_run_failure`):

```python
            if (engine := runtime.ensure_run_hooks()) is not None:
                engine.notify("Stop", session_id=session_id,
                              run_id=run_id,
                              data={"status": "error" if failed else "completed"})
```

SubagentStop (bridge, in the `on_child_settled` wiring at ~4211 — wrap, don't replace, the existing partial):

```python
            _settled = functools.partial(
                self._on_fleet_child_settled,
                conversation_id, session_id, assistant_message_id,
            )

            def on_child_settled(child_run_id, status, *, _inner=_settled, _sid=session_id):
                if (engine := runtime.ensure_run_hooks()) is not None:
                    engine.notify("SubagentStop", session_id=_sid, run_id=child_run_id,
                                  data={"child_run_id": child_run_id, "status": status})
                _inner(child_run_id, status)
```

and pass `on_child_settled=on_child_settled` at the original site (adapt `runtime` to the bridge's runtime accessor).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_local_review_hook.py Tests/Agents/test_fleet_send_to_agent.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_fleet_settle_fanout.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_local_review_hook.py Tests/Agents/test_fleet_send_to_agent.py Tests/Chat/test_console_fleet_wake.py
git commit -m "feat(run-hooks): Stop / SubagentStop / ApprovalRequested fire points"
```

---

### Task 9: Docs + final integration sweep

**Files:**
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md` (new "Run hooks" section)
- Modify: `CLAUDE.md` / `AGENTS.md` special-systems note (one paragraph beside the Tool Calling section)
- Modify: `Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md` (status → Implemented, note the loguru execution-log mapping)
- Test: no new tests — this task runs the accumulated targeted sweep

**Interfaces:**
- Consumes: everything above.
- Produces: documentation; the ADR-check already exists (ADR-148, linked from spec and plan).

- [ ] **Step 1: Write the user-guide section**

In `Docs/User_Guide/console/agent-runs-and-tools.md`, add a section covering: the six events with one-line payload descriptions; the `[hooks]` config example copied verbatim from the spec's §6; the verdict rules table (deny-only, fail directions); the security notes (user-scope only, argv-only, timeouts, run-log visibility); and a pointer that a settings sub-screen lands in the next PR.

- [ ] **Step 2: Add the one-paragraph special-systems note to `CLAUDE.md`/`AGENTS.md`**

Beside the existing Tool Calling subsection:

```markdown
### Console Run Hooks
- User-configured external commands at six lifecycle events (UserPromptSubmit, PreToolUse, PostToolUse, ApprovalRequested, Stop, SubagentStop); config: `[hooks]` in config.toml (ADR-148).
- Deny-only: hooks can refuse tool calls but never bypass the permission store; PreToolUse fails closed, UserPromptSubmit fails open; argv-list commands only, process-group timeout kill.
```

- [ ] **Step 3: Run the accumulated targeted sweep**

Run: `pytest Tests/Agents/test_run_hooks.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_fleet_continuation.py Tests/Agents/test_fleet_send_to_agent.py Tests/Chat/test_console_viewless_hooks.py Tests/Chat/test_console_local_review_hook.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_headless_wake_invariants.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_fleet_settle_fanout.py Tests/Chat/test_console_provider_gateway.py -v`
Expected: all PASS. (Full suite only if the user asks — AGENTS.md rule.)

- [ ] **Step 4: Update the spec status and Implementation Notes**

Set the spec's Status line to `Implemented`. Then per AGENTS.md backlog flow: create the backlog task (`backlog task create "Console run hooks" ...`), mark ACs from the spec, add Implementation Notes, link ADR-148, set Done only when the DoD checklist is complete.

- [ ] **Step 5: Commit**

```bash
git add Docs/User_Guide/console/agent-runs-and-tools.md CLAUDE.md AGENTS.md Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md
git commit -m "docs(run-hooks): user guide + special-systems notes for Console run hooks"
```

---

## Self-Review (performed after writing)

1. **Spec coverage:** §3 events → Tasks 5–8 (all six wired); §4 execution model → Task 2; §5 verdict semantics → Tasks 2–3; §6 config → Tasks 1+4; §7 payload hygiene → Task 2 (envelope, truncation) with `_truncate` applied at payload build in fire sites; §8 wiring → Tasks 4–8; §9 error handling → Task 2 (`fire` never raises; fail directions); §10 testing → per-task tests + Task 9 sweep; §2 non-goals honored (no settings UI, no project scope, no rewriting).
2. **Placeholder scan:** Tasks 4, 5, 6, 7, 8 contain "concretize against the module's fixtures" instructions — these name the exact file, anchor, and assertion set where the harness must be reused rather than reinvented; the engine-module code (the new logic) is fully specified. This is deliberate: reinventing those harnesses blind would drift from their fixture discipline.
3. **Type consistency:** `HookSpec(event, command, matcher, timeout_s)`, `HookOutcome(blocked, reason, context)`, `wrap_review(inner, *, session_id)`, `post_tool_dep(*, session_id)`, `ensure_run_hooks() -> RunHooksEngine | None`, `fire(event, *, session_id, run_id=None, data=None)` used consistently across tasks.
