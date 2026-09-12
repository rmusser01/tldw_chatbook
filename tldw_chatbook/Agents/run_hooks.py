# tldw_chatbook/Agents/run_hooks.py
"""Console run hooks: user-configured external commands on session/run lifecycle events.

Spec: Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md (ADR-148).
Deny-only guardrails, argv-list commands, JSON-on-stdin protocol, per-purpose
fail direction (PreToolUse fails closed, UserPromptSubmit fails open).
"""

from __future__ import annotations

import asyncio
import datetime
import fnmatch
import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping

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
    if not isinstance(event, str) or event not in HOOK_EVENTS:
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
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        logger.warning("run-hooks: timeout_s must be a positive finite number; hook disabled")
        return None
    return HookSpec(event=event, command=tuple(command), matcher=matcher, timeout_s=float(timeout))


def load_hooks_config(config: Mapping) -> RunHooksConfig:
    section = config.get("hooks") if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        return RunHooksConfig(enabled=True, hooks=())
    enabled = section.get("enabled", True)
    raw_hooks = section.get("hook", [])
    if not isinstance(raw_hooks, list):
        logger.warning("run-hooks: [hooks] hook must be a list of tables; hooks disabled: {!r}", raw_hooks)
        raw_hooks = []
    hooks = tuple(h for h in (_parse_hook(r) for r in raw_hooks) if h is not None)
    return RunHooksConfig(enabled=bool(enabled), hooks=hooks)


# ---------------------------------------------------------------------------
# Execution core (task 2)
# ---------------------------------------------------------------------------

_TRUNCATION_MARKER = "…[truncated]"


@dataclass(frozen=True)
class HookOutcome:
    """Result of firing all hooks for one lifecycle event."""

    blocked: bool = False
    reason: str = ""
    context: str = ""


def _truncate(text: str) -> str:
    """Cap hook-produced text at HOOK_IO_BUDGET_CHARS *total* (marker included)."""
    if len(text) <= HOOK_IO_BUDGET_CHARS:
        return text
    return text[: HOOK_IO_BUDGET_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


# Public alias (ruling R4): downstream tasks use this name, never the private helper.
truncate_hook_text = _truncate


@dataclass(frozen=True)
class _Decision:
    denied: bool = False
    reason: str = ""
    context: str = ""


def _decide(event: str, proc: subprocess.CompletedProcess) -> _Decision:
    """Map a finished hook process to a deny/pass/context decision.

    JSON stdout decisions beat exit codes; exit 2 is the deny/block shorthand.
    Only UserPromptSubmit (fail-open, stdout-as-context) and PreToolUse
    (fail-closed) have decision semantics; every other event is pass-through.
    """
    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(stdout)
    except (ValueError, TypeError):
        candidate = None
    if isinstance(candidate, dict):
        parsed = candidate
    decision = (parsed or {}).get("decision")
    if event == "UserPromptSubmit":
        if decision == "block" or proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "blocked by hook")
            return _Decision(denied=True, reason=_truncate(reason))
        return _Decision(context=_truncate(stdout.strip()))
    if event == "PreToolUse":
        if decision == "deny" or proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "denied by hook")
            return _Decision(denied=True, reason=_truncate(reason))
        if proc.returncode != 0:
            return _Decision(denied=True, reason=f"hook failed (exit {proc.returncode}); failing closed")
        return _Decision()
    return _Decision()


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Best-effort kill of the hook's whole process group. Never raises."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    except ProcessLookupError:
        return  # process/group already gone
    except PermissionError:
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_hook(spec: HookSpec, payload: dict[str, Any]) -> tuple[HookSpec, _Decision | None]:
    """Run one hook to completion (or timeout-kill). Never raises to the pool.

    Returns None as the decision for non-blocking events whose hook crashed or
    timed out: those outcomes are logged and dropped (fail-open), while
    PreToolUse fails closed per spec.
    """
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
        if spec.event == "PreToolUse":
            return spec, _Decision(denied=True, reason="hook failed to start; failing closed")
        return spec, None
    decision: _Decision | None
    try:
        stdout, stderr = proc.communicate(json.dumps(payload), timeout=spec.timeout_s)
        completed = subprocess.CompletedProcess(spec.command, proc.returncode,
                                                stdout=stdout, stderr=stderr)
        decision = _decide(spec.event, completed)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            proc.communicate()
        except Exception:  # noqa: BLE001 - post-kill reaping must never mask the timeout
            pass
        if spec.event == "PreToolUse":
            decision = _Decision(denied=True,
                                 reason=f"hook timed out after {spec.timeout_s}s; failing closed")
        else:
            decision = None
    logger.info("run-hooks: event={} hook={} exit={} took_ms={}",
                spec.event, spec.command[0] if spec.command else "<empty>", proc.returncode,
                int((time.monotonic() - started) * 1000))
    return spec, decision


class RunHooksEngine:
    """Executes configured hooks for lifecycle events. Never raises to callers.

    fire()/fire_async() run all matching hooks concurrently and block until a
    decision is needed: the first deny wins and remaining results are discarded
    (their processes still run to completion in the pool). notify() is
    fire-and-forget on a single-worker pool.
    """

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
        matched: list[HookSpec] = []
        for hook in cfg.hooks:
            if hook.event != event:
                continue
            # Matcher-less hooks match everything; matcher hooks need a tool
            # name to test against and are skipped when none was provided.
            if hook.matcher is None or (tool_name is not None and hook.matches_tool(tool_name)):
                matched.append(hook)
        return matched

    def _payload(self, event: str, session_id: str, run_id: str | None,
                 data: dict[str, Any]) -> dict[str, Any]:
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
        """Run matching hooks and return the combined outcome. Never raises.

        Engine-level failures degrade per spec: PreToolUse fails closed, every
        other event is logged and dropped (fail-open).
        """
        try:
            return self._fire(event, session_id=session_id, run_id=run_id, data=data)
        except Exception as exc:  # noqa: BLE001 - engine must never raise to callers
            logger.warning("run-hooks: fire({}) engine error: {}", event, exc)
            if event == "PreToolUse":
                return HookOutcome(blocked=True, reason="hooks engine error; failing closed")
            return HookOutcome()

    def _fire(self, event: str, *, session_id: str, run_id: str | None,
              data: dict[str, Any] | None) -> HookOutcome:
        tool_name: str | None = None
        if isinstance(data, dict):
            candidate = data.get("tool_name")
            if isinstance(candidate, str):
                tool_name = candidate
        specs = self._matching(event, tool_name)
        if not specs:
            return HookOutcome()
        payload = self._payload(event, session_id, run_id, data if isinstance(data, dict) else {})
        futures = {self._pool.submit(_run_hook, spec, payload): spec for spec in specs}
        context = ""
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                _, decision = fut.result()
            except Exception as exc:  # noqa: BLE001 - a hook crash never propagates
                logger.warning("run-hooks: hook {} raised: {}", spec.command, exc)
                decision = (_Decision(denied=True, reason="hook raised; failing closed")
                            if spec.event == "PreToolUse" else None)
            if decision is None:
                continue
            if decision.denied:
                # First deny wins (ruling R1): stop waiting now; the remaining
                # futures run to completion in the pool and are discarded.
                return HookOutcome(blocked=True, reason=decision.reason)
            if decision.context and not context:
                context = decision.context
        return HookOutcome(context=context)

    async def fire_async(self, event: str, **kwargs: Any) -> HookOutcome:
        try:
            return await asyncio.to_thread(self.fire, event, **kwargs)
        except Exception as exc:  # noqa: BLE001 - engine must never raise to callers
            # fire() already never raises; this guards to_thread machinery and
            # argument-binding errors from the in-thread call site. Degrade with
            # fire()'s semantics: PreToolUse fails closed, others dropped.
            logger.warning("run-hooks: fire_async({}) engine error: {}", event, exc)
            if event == "PreToolUse":
                return HookOutcome(blocked=True, reason="hooks engine error; failing closed")
            return HookOutcome()

    def notify(self, event: str, **kwargs: Any) -> None:
        """Fire-and-forget: returns immediately, outcome is logged and dropped."""
        def _run() -> None:
            try:
                self.fire(event, **kwargs)
            except Exception as exc:  # noqa: BLE001 - defensive; fire already never raises
                logger.warning("run-hooks: notify {} failed: {}", event, exc)

        try:
            self._notify_worker.submit(_run)
        except Exception as exc:  # noqa: BLE001 - executor unavailable — drop, never stall chat
            logger.warning("run-hooks: notify dropped for {}: {}", event, exc)
