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
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from loguru import logger

from tldw_chatbook.Agents.agent_models import ToolCall

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

    Ruling R12 (refined): when stdout parses as a JSON object containing a
    "decision" key, that literal decision wins over the exit-2 shorthand —
    deny/block map to their denial; allow or any other value is a no-opinion
    (ignored + logged). A decision key suppresses only the shorthand: crash
    exits (non-zero, non-2) still fail closed on PreToolUse regardless of
    parsed stdout.

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
    has_decision = parsed is not None and "decision" in parsed
    decision_value = parsed.get("decision") if has_decision else None
    if event == "UserPromptSubmit":
        if has_decision:
            if decision_value == "block":
                reason = str(parsed.get("reason") or stderr or "blocked by hook")
                return _Decision(denied=True, reason=_truncate(reason))
            logger.warning("run-hooks: UserPromptSubmit hook decision {!r} ignored; "
                           "exit code {} suppressed", decision_value, proc.returncode)
            return _Decision(context=_truncate(stdout.strip()))
        if proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "blocked by hook")
            return _Decision(denied=True, reason=_truncate(reason))
        if proc.returncode != 0:
            logger.warning("run-hooks: UserPromptSubmit hook exited {} — failing open",
                           proc.returncode)
        return _Decision(context=_truncate(stdout.strip()))
    if event == "PreToolUse":
        if has_decision and decision_value == "deny":
            reason = str(parsed.get("reason") or stderr or "denied by hook")
            return _Decision(denied=True, reason=_truncate(reason))
        if not has_decision and proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "denied by hook")
            return _Decision(denied=True, reason=_truncate(reason))
        if proc.returncode not in (0, 2):
            # Crash exits fail closed even when stdout parsed as an allow/other
            # decision (ruling R12 refined): a decision key suppresses only the
            # exit-2 shorthand, never the fail-closed crash path.
            return _Decision(denied=True, reason=f"hook failed (exit {proc.returncode}); failing closed")
        if has_decision:
            # allow/other decision with a clean or shorthand-suppressed exit:
            # no-opinion, logged.
            logger.warning("run-hooks: PreToolUse hook decision {!r} ignored (deny-only; "
                           "exit {})", decision_value, proc.returncode)
            return _Decision()
        if stdout.strip():
            # Clean pass, but the hook tried to speak the decision protocol and
            # we could not parse it — surface that instead of silently ignoring.
            logger.warning("run-hooks: PreToolUse hook exited 0 with unrecognized stdout "
                           "(expected empty or JSON decision): {!r}", _truncate(stdout.strip()))
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
    PreToolUse fails closed per spec. UserPromptSubmit fail-open paths
    (start-failure, timeout) log at WARNING (spec §5).
    """
    started = time.monotonic()
    cmd0 = spec.command[0] if spec.command else "<empty>"
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
        if spec.event == "UserPromptSubmit":
            logger.warning("run-hooks: UserPromptSubmit hook failed to start — failing open")
        return spec, None
    stdout = ""
    stderr = ""
    decision: _Decision | None
    timed_out = False
    try:
        stdout, stderr = proc.communicate(json.dumps(payload), timeout=spec.timeout_s)
        completed = subprocess.CompletedProcess(spec.command, proc.returncode,
                                                stdout=stdout, stderr=stderr)
        decision = _decide(spec.event, completed)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        try:
            stdout, stderr = proc.communicate()
        except Exception:  # noqa: BLE001 - post-kill reaping must never mask the timeout
            stdout, stderr = "", ""
        stdout, stderr = stdout or "", stderr or ""
        if spec.event == "PreToolUse":
            decision = _Decision(denied=True,
                                 reason=f"hook timed out after {spec.timeout_s}s; failing closed")
        else:
            decision = None
            if spec.event == "UserPromptSubmit":
                logger.warning("run-hooks: UserPromptSubmit hook timed out after {}s — failing open",
                               spec.timeout_s)
    took_ms = int((time.monotonic() - started) * 1000)
    if spec.event in BLOCKING_EVENTS:
        logger.info("run-hooks: event={} hook={} exit={} timed_out={} took_ms={}",
                    spec.event, cmd0, proc.returncode, timed_out, took_ms)
    else:
        # Ruling R13: non-blocking events log captured, truncated output at
        # INFO carrying session_id/run_id/hook event for observability.
        logger.info("run-hooks: event={} session_id={} run_id={} hook={} exit={} timed_out={} "
                    "stdout={!r} stderr={!r} took_ms={}",
                    spec.event, payload.get("session_id"), payload.get("run_id"), cmd0,
                    proc.returncode, timed_out,
                    _truncate((stdout or "").strip()), _truncate((stderr or "").strip()), took_ms)
    return spec, decision


def _hook_raised_decision(spec: HookSpec) -> _Decision | None:
    """Decision substituted when a hook invocation raises instead of returning."""
    return (_Decision(denied=True, reason="hook raised; failing closed")
            if spec.event == "PreToolUse" else None)


class RunHooksEngine:
    """Executes configured hooks for lifecycle events. Never raises to callers.

    fire()/fire_async() run all matching hooks and block until a decision is
    needed: the first deny wins and remaining results are discarded (their
    processes still run to completion in the pool).

    Pool isolation (ruling R14): blocking events (UserPromptSubmit,
    PreToolUse) execute on a dedicated 4-worker pool that nothing else uses.
    Non-blocking events execute inline on the calling thread — for notify()
    that is the dedicated single notify worker — so blocking fires can never
    queue behind notify-driven hooks. notify() is fire-and-forget.
    """

    def __init__(self, config_provider: Callable[[], RunHooksConfig],
                 cwd_provider: Callable[[], str]) -> None:
        self._config_provider = config_provider
        self._cwd_provider = cwd_provider
        # Blocking-event pool only (ruling R14); never shared with notify work.
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
                 data: dict[str, Any], cwd: str | None = None) -> dict[str, Any]:
        return {
            "hook_event": event,
            "session_id": session_id,
            "run_id": run_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            # Ruling R18: a fire-site-supplied cwd (the session's bound
            # workspace root, when one exists) overrides the provider's
            # app-level fallback; None keeps the provider value.
            "cwd": cwd if cwd is not None else self._cwd_provider(),
            "data": data,
        }

    def fire(self, event: str, *, session_id: str, run_id: str | None = None,
             data: dict[str, Any] | None = None,
             cwd: str | None = None) -> HookOutcome:
        """Run matching hooks and return the combined outcome. Never raises.

        Engine-level failures degrade per spec: PreToolUse fails closed, every
        other event is logged and dropped (fail-open).

        Args:
            event: Lifecycle event name (a HOOK_EVENTS member).
            session_id: The Console session the fire belongs to.
            run_id: The agent run id, when one is in flight.
            data: Event payload data (``tool_name``/``tool_args`` for tool
                events, ``prompt`` for UserPromptSubmit, ...).
            cwd: Fire-site override for the payload's ``cwd`` (Ruling R18) —
                the session's bound workspace root when one is resolvable.
                ``None`` falls back to the engine's cwd_provider.
        """
        try:
            return self._fire(event, session_id=session_id, run_id=run_id,
                              data=data, cwd=cwd)
        except Exception as exc:  # noqa: BLE001 - engine must never raise to callers
            logger.warning("run-hooks: fire({}) engine error: {}", event, exc)
            if event == "PreToolUse":
                return HookOutcome(blocked=True, reason="hooks engine error; failing closed")
            return HookOutcome()

    def _fire(self, event: str, *, session_id: str, run_id: str | None,
              data: dict[str, Any] | None, cwd: str | None = None) -> HookOutcome:
        tool_name: str | None = None
        if isinstance(data, dict):
            candidate = data.get("tool_name")
            if isinstance(candidate, str):
                tool_name = candidate
        specs = self._matching(event, tool_name)
        if not specs:
            return HookOutcome()
        payload = self._payload(event, session_id, run_id,
                                data if isinstance(data, dict) else {}, cwd)
        if event in BLOCKING_EVENTS:
            futures = {self._pool.submit(_run_hook, spec, payload): spec for spec in specs}
            results = self._results_from_futures(futures)
        else:
            # Ruling R14: non-blocking events run inline on the calling thread
            # (the notify worker for notify()-driven fires) and never consume
            # blocking-pool capacity.
            results = self._results_inline(specs, payload)
        return self._reduce(results)

    @staticmethod
    def _results_from_futures(
            futures: dict[Future[tuple[HookSpec, _Decision | None]], HookSpec],
    ) -> Iterator[tuple[HookSpec, _Decision | None]]:
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                yield fut.result()
            except Exception as exc:  # noqa: BLE001 - a hook crash never propagates
                logger.warning("run-hooks: hook {} raised: {}", spec.command, exc)
                yield spec, _hook_raised_decision(spec)

    @staticmethod
    def _results_inline(specs: list[HookSpec],
                        payload: dict[str, Any]) -> Iterator[tuple[HookSpec, _Decision | None]]:
        for spec in specs:
            try:
                yield _run_hook(spec, payload)
            except Exception as exc:  # noqa: BLE001 - a hook crash never propagates
                logger.warning("run-hooks: hook {} raised: {}", spec.command, exc)
                yield spec, _hook_raised_decision(spec)

    @staticmethod
    def _reduce(results: Iterable[tuple[HookSpec, _Decision | None]]) -> HookOutcome:
        context = ""
        for _spec, decision in results:
            if decision is None:
                continue
            if decision.denied:
                # First deny wins (ruling R1): stop waiting now; the remaining
                # futures run to completion in the pool and are discarded.
                return HookOutcome(blocked=True, reason=decision.reason)
            if decision.context and not context:
                context = decision.context
        return HookOutcome(context=context)

    async def fire_async(self, event: str, *, cwd: str | None = None,
                         **kwargs: Any) -> HookOutcome:
        """Async face of `fire` (same args plus the R18 `cwd` override)."""
        try:
            return await asyncio.to_thread(self.fire, event, cwd=cwd, **kwargs)
        except Exception as exc:  # noqa: BLE001 - engine must never raise to callers
            # fire() already never raises; this guards to_thread machinery and
            # argument-binding errors from the in-thread call site. Degrade with
            # fire()'s semantics: PreToolUse fails closed, others dropped.
            logger.warning("run-hooks: fire_async({}) engine error: {}", event, exc)
            if event == "PreToolUse":
                return HookOutcome(blocked=True, reason="hooks engine error; failing closed")
            return HookOutcome()

    def notify(self, event: str, *, cwd: str | None = None,
               **kwargs: Any) -> None:
        """Fire-and-forget: returns immediately, outcome is logged and dropped.

        `cwd` is the R18 fire-site override, passed through to the payload.
        """
        def _run() -> None:
            try:
                self.fire(event, cwd=cwd, **kwargs)
            except Exception as exc:  # noqa: BLE001 - defensive; fire already never raises
                logger.warning("run-hooks: notify {} failed: {}", event, exc)

        try:
            self._notify_worker.submit(_run)
        except Exception as exc:  # noqa: BLE001 - executor unavailable — drop, never stall chat
            logger.warning("run-hooks: notify dropped for {}: {}", event, exc)

    def wrap_review(self, inner: Callable[[list[ToolCall], str], dict[str, str]], *,
                    session_id: str) -> Callable[[list[ToolCall], str], dict[str, str]]:
        """Wrap a review_tool_calls callable so PreToolUse hooks deny first.

        Denied names are EXCLUDED from the inner review (no approval card for
        a call a hook already refused) and merged back as refusal strings,
        namespaced with a "hook: " prefix (ruling R16): hook-produced text can
        never equal the "proceed" dispatch sentinel, so a deny stays a deny.
        Hooks can never produce "proceed" — deny-only, enforced here.
        """
        if not any(h.event == "PreToolUse" for h in self._config_provider().hooks):
            return inner

        def review(calls: list[ToolCall], run_id: str) -> dict[str, str]:
            refusals: dict[str, str] = {}
            surviving: list[ToolCall] = []
            for call in calls:
                outcome = self.fire(
                    "PreToolUse", session_id=session_id, run_id=run_id,
                    data={"tool_name": call.name, "tool_args": call.args},
                )
                if outcome.blocked:
                    refusals[call.name] = f"hook: {outcome.reason}"
                else:
                    surviving.append(call)
            verdicts = dict(inner(surviving, run_id)) if surviving else {}
            verdicts.update(refusals)
            return verdicts

        return review

    def post_tool_dep(self, *, session_id: str) -> Callable[..., None]:
        """Build the runtime's ``post_tool_call`` dep for one Console session.

        The dep is fired by the dispatch loop (``LoopDeps.post_tool_call``,
        threaded through ``AgentService``) at its run-log capture point,
        ONLY for calls that actually dispatched (verdict ``"proceed"``) —
        a review refusal fires nothing. It receives the STILL-UNCAPPED
        result content; truncation to the payload budget happens here, so
        the runtime layer stays free of budget knowledge. Built on
        ``notify``, so the dep returns immediately and a slow hook never
        stalls a dispatch.

        Args:
            session_id: The Console session the fired events belong to
                (closed over; the runtime call site has no session identity).

        Returns:
            The ``(tool_name, call_id, args, content, ok, run_id) -> None``
            callable to install as the service's ``post_tool_call`` dep.
            The trailing ``run_id`` (R20) is bound per run by the service's
            ``_run_one`` -- the same per-run lambda binding the review hook
            gets -- so the envelope's ``run_id`` names the FIRING run (a
            fleet child's tool use is attributable to the child, not the
            session's primary). It defaults to ``None`` for direct callers.
        """

        def post_tool_call(tool_name: str, call_id: str, args: dict,
                           content: str, ok: bool,
                           run_id: str | None = None) -> None:
            self.notify("PostToolUse", session_id=session_id, run_id=run_id,
                        data={"tool_name": tool_name, "tool_args": args,
                              "tool_result": _truncate(content),
                              "is_error": not ok})

        return post_tool_call
