"""Console run hooks: user-configured external commands on session/run lifecycle events.

Spec: Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md (ADR-148).
Deny-only guardrails, argv-list commands, JSON-on-stdin protocol, per-purpose
fail direction (PreToolUse fails closed, UserPromptSubmit fails open).
"""

from __future__ import annotations

import asyncio
import datetime
import fnmatch
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_args

from loguru import logger
from pydantic import ValidationError

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Utils.input_validation import (
    CONSOLE_HOOK_CONTAINER_INPUT,
    CONSOLE_HOOK_DEFAULT_TIMEOUT_S,
    CONSOLE_HOOK_ID_INPUT,
    CONSOLE_HOOK_ROWS_INPUT,
    CONSOLE_HOOK_SWITCH_INPUT,
    CONSOLE_HOOK_TOOL_EVENTS,
    ConsoleHookEvent,
    ConsoleHookInput,
)
from tldw_chatbook.Utils.path_validation import validate_existing_absolute_directory

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hooks_v2.models import HookHandler

HOOK_EVENTS: frozenset[str] = frozenset(get_args(ConsoleHookEvent))
TOOL_NAME_EVENTS: frozenset[str] = CONSOLE_HOOK_TOOL_EVENTS
HOOK_IO_BUDGET_CHARS: int = 4000
HOOK_DEFAULT_TIMEOUT_S: float = CONSOLE_HOOK_DEFAULT_TIMEOUT_S
# Bound retained stream bytes, including four-byte UTF-8 characters.
HOOK_IO_BUDGET_BYTES = HOOK_IO_BUDGET_CHARS * 4
HOOK_NOTIFY_CAPACITY = 64
# Oversized notifications are dropped whole; guard arguments are never cut.
HOOK_NOTIFY_PAYLOAD_BYTES = 1024 * 1024
HOOK_NOTIFY_MAX_NODES = 16384
HOOK_NOTIFY_MAX_DEPTH = 64
# Ceiling for asynchronous process cleanup after explicitly closing pipes.
# Escaped descendants can retain inherited write ends, so EOF cannot be the
# prerequisite for closing readers and reaping the owned subprocess.
HOOK_POST_KILL_REAP_TIMEOUT_S: float = 5.0
BLOCKING_EVENTS: frozenset[str] = frozenset({"UserPromptSubmit", "PreToolUse"})


@dataclass(frozen=True)
class HookSpec:
    """Validated command for one lifecycle event.

    Attributes:
        event: Supported lifecycle event name.
        command: Executable and arguments, without shell interpretation.
        matcher: Optional case-sensitive tool-name glob.
        timeout_s: Positive maximum execution duration.
    """

    event: str
    command: tuple[str, ...]
    matcher: str | None = None
    timeout_s: float = HOOK_DEFAULT_TIMEOUT_S

    def matches_tool(self, tool_name: str) -> bool:
        """Match a tool against the configured glob.

        Args:
            tool_name: Provider-visible tool name.

        Returns:
            Whether the name matches, or True when no matcher is configured.
        """
        if self.matcher is None:
            return True
        return fnmatch.fnmatchcase(tool_name, self.matcher)


@dataclass(frozen=True, slots=True)
class HookTarget:
    """An exact definition and grant epoch captured for one firing."""

    config_scope: str
    key: str
    fingerprint: str
    spec: HookSpec | HookHandler = field(repr=False)
    approval_token: str | None


class HookLaunchRefused(Exception):
    """Consent refusal, distinct from an execution error that may fail open."""

    def __init__(self, reason: str, skip: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.skip = skip


@dataclass(frozen=True)
class V2InvalidAdmission:
    """Bounded metadata for a rejected v2 batch entry, never its raw body.

    `policy` is explicit_required, event_control, optional or unresolved.
    `event` is None when no supported event can be trusted. H2 must refuse
    activation of the rejected v2 set; it uses these records to retain each
    possible controlling event scope, not as a global run-denial boolean.
    """

    index: int | None
    event: str | None
    policy: str


_V2_EVENT_CONTROL_EFFECTS = {
    "PreToolUse": frozenset({"deny", "updated_input"}),
    "SubagentStart": frozenset({"deny", "child_limits"}),
}


@dataclass(frozen=True)
class RunHooksConfig:
    """Validated user configuration.

    Attributes:
        enabled: Master switch controlling every firing.
        hooks: Immutable validated hook entries.
    """

    enabled: bool = True
    hooks: tuple[HookSpec, ...] = ()
    v2_handlers: tuple[HookHandler, ...] = ()
    v2_invalid: bool = False
    v2_invalid_required: bool = False
    v2_invalid_admissions: tuple[V2InvalidAdmission, ...] = ()

    @property
    def v2_requirements_unsatisfied(self) -> bool:
        """Whether configuration alone prevents required v2 satisfaction.

        Execution/success remains the later runtime owner's responsibility.
        """
        return self.v2_invalid_required or (
            not self.enabled and any(handler.required for handler in self.v2_handlers)
        )


def _invalid_v2_admissions(raw: object) -> tuple[V2InvalidAdmission, ...]:
    """Retain bounded, body-free scope hints when a whole batch is rejected."""
    from tldw_chatbook.Agents.hooks_v2.validation import EFFECTS

    if not isinstance(raw, list):
        return (V2InvalidAdmission(None, None, "unresolved"),)
    records: list[V2InvalidAdmission] = []
    for index, item in enumerate(raw[:256]):
        if type(item) is not dict:
            records.append(V2InvalidAdmission(index, None, "unresolved"))
            continue
        proposed_event = item.get("event")
        event = (
            proposed_event
            if type(proposed_event) is str and proposed_event in EFFECTS
            else None
        )
        required = item.get("required", False)
        effects = item.get("effects")
        if required is True:
            policy = "explicit_required"
        elif (
            required is not False
            or event is None
            or not isinstance(effects, list)
            or len(effects) > 6
        ):
            policy = "unresolved"
        elif any(
            effect in _V2_EVENT_CONTROL_EFFECTS.get(event, ())
            for effect in effects
            if type(effect) is str
        ):
            policy = "event_control"
        elif (
            any(type(effect) is not str for effect in effects)
            or len(set(effects)) != len(effects)
            or not set(effects) <= EFFECTS[event]
        ):
            policy = "unresolved"
        else:
            policy = "optional"
        records.append(V2InvalidAdmission(index, event, policy))
    if len(raw) > 256:
        records.append(V2InvalidAdmission(None, None, "unresolved"))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class HookInventoryRow:
    """One source row; invalid definitions remain visible for repair."""

    index: int
    key: str
    spec: HookSpec | HookHandler | None
    enabled: bool | None
    error: str | None
    source: str = "hook"


@dataclass(frozen=True, slots=True)
class HookInventory:
    """Lossless inventory, independent of the executable projection."""

    master_enabled: bool | None
    container_error: str | None
    rows: tuple[HookInventoryRow, ...] = ()
    v2_rows: tuple[HookInventoryRow, ...] = ()

    @property
    def review_rows(self) -> tuple[HookInventoryRow, ...]:
        """Both schemas share consent while legacy editor indices stay stable."""
        return self.rows + self.v2_rows

    @property
    def requires_authority(self) -> bool:
        """Require review unless absence/disable is explicitly verified."""
        if self.master_enabled is False:
            return False
        return bool(
            self.container_error
            or any(row.enabled is not False for row in self.review_rows)
        )


def fingerprint_hook(spec: HookSpec | HookHandler) -> str:
    """Hash an exact normalized execution definition, without enable switches.

    Args:
        spec: Validated lifecycle event, argv, matcher, and timeout.

    Returns:
        Versioned SHA-256 identity of the execution definition.
    """
    definition = (
        {
            "version": 1,
            "event": spec.event,
            "command": list(spec.command),
            "matcher": spec.matcher,
            "timeout_s": float(spec.timeout_s),
        }
        if isinstance(spec, HookSpec)
        else {"version": 2, **spec.model_dump(mode="json")}
    )
    encoded = json.dumps(
        definition,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_hook_definition(raw: object) -> tuple[HookSpec | None, str | None]:
    """Validate execution fields and return a content-free repair message.

    Args:
        raw: Saved hook table, including any unknown fields to preserve.

    Returns:
        Validated execution specification or a bounded repair message.
    """
    if not isinstance(raw, Mapping):
        return None, "Hook entry must be a table."
    try:
        validated = ConsoleHookInput.model_validate(dict(raw))
    except ValidationError as error:
        field = error.errors(include_input=False, include_context=False)[0]["loc"][0]
        messages = {
            "event": "Choose a supported hook event.",
            "command": "Command must be a nonempty argv list of NUL-free strings.",
            "matcher": (
                "Matcher is only valid for PreToolUse or PostToolUse."
                if raw.get("event") not in ("PreToolUse", "PostToolUse")
                else "Matcher must be a nonempty glob string."
            ),
            "timeout_s": "Timeout must be a positive finite number.",
        }
        return None, messages[field]
    return HookSpec(
        validated.event,
        tuple(validated.command),
        validated.matcher,
        validated.timeout_s,
    ), None


def inspect_hooks_config(config: Mapping[str, object]) -> HookInventory:
    """Inspect every raw row and retain malformed containers for recovery.

    Args:
        config: Raw configuration mapping with an optional Hooks section.

    Returns:
        Lossless inventory of switches, identities, and repair states.
    """
    if not isinstance(config, Mapping):
        return HookInventory(None, "Configuration must be a table.")
    if "hooks" not in config:
        return HookInventory(True, None)
    section = config["hooks"]
    try:
        section = CONSOLE_HOOK_CONTAINER_INPUT.validate_python(
            dict(section) if isinstance(section, Mapping) else section
        )
    except ValidationError:
        return HookInventory(
            None, "Hooks section must be a table; repair in Advanced Config."
        )
    container_error = None
    try:
        master = CONSOLE_HOOK_SWITCH_INPUT.validate_python(section.get("enabled", True))
    except ValidationError:
        master = None
        container_error = "Hooks enabled must be a boolean."
    try:
        raw_hooks = CONSOLE_HOOK_ROWS_INPUT.validate_python(section.get("hook", []))
    except ValidationError:
        return HookInventory(master, "Hooks list must be an array of tables.")
    ids: dict[str, int] = {}
    identities = []
    for raw in raw_hooks:
        hook_id = None
        if isinstance(raw, Mapping) and "id" in raw:
            try:
                hook_id = CONSOLE_HOOK_ID_INPUT.validate_python(raw["id"])
            except ValidationError:
                pass  # The raw row remains visible with a repair error below.
        identities.append(hook_id)
        if hook_id is not None:
            ids[hook_id] = ids.get(hook_id, 0) + 1
    occurrences: dict[str, int] = {}
    rows = []
    for index, raw in enumerate(raw_hooks):
        spec, error = validate_hook_definition(raw)
        try:
            enabled = CONSOLE_HOOK_SWITCH_INPUT.validate_python(
                raw.get("enabled", True) if isinstance(raw, Mapping) else None
            )
        except ValidationError:
            enabled = None
            error = "Hook enabled must be a boolean."
        hook_id = identities[index]
        if isinstance(raw, Mapping) and "id" in raw and hook_id is None:
            error = "Hook ID must be a nonempty string."
        if hook_id is not None and ids.get(hook_id, 0) > 1:
            error = "Duplicate hook ID; assign distinct IDs."
        if error:
            spec = None
        if spec is None:
            key = f"invalid:{index}"
        elif hook_id is not None:
            key = f"id:{hook_id}"
        else:
            fingerprint = fingerprint_hook(spec)
            occurrence = occurrences.get(fingerprint, 0)
            occurrences[fingerprint] = occurrence + 1
            key = f"legacy:{fingerprint}:{occurrence}"
        rows.append(HookInventoryRow(index, key, spec, enabled, error))
    raw_v2 = section.get("handler", [])
    v2_rows = ()
    if raw_v2 or not isinstance(raw_v2, list):
        from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

        try:
            handlers = parse_handlers(raw_v2)
        except ValueError:
            # The schema rejects the whole batch. Never approve a valid-looking
            # prefix or retain an unbounded malformed body in review metadata.
            indices = range(min(len(raw_v2), 256)) if isinstance(raw_v2, list) else (0,)
            v2_rows = tuple(
                HookInventoryRow(
                    index,
                    f"v2:invalid:{index}",
                    None,
                    True,
                    "Invalid v2 handler batch; repair in Advanced Config.",
                    "handler",
                )
                for index in indices
            )
            if not v2_rows:
                v2_rows = (
                    HookInventoryRow(
                        0,
                        "v2:invalid:0",
                        None,
                        True,
                        "Invalid v2 handler batch; repair in Advanced Config.",
                        "handler",
                    ),
                )
        else:
            v2_rows = tuple(
                HookInventoryRow(
                    index, "v2:id:" + handler.id, handler, True, None, "handler"
                )
                for index, handler in enumerate(handlers)
            )
    return HookInventory(master, container_error, tuple(rows), v2_rows)


def load_hooks_config(config: Mapping) -> RunHooksConfig:
    """Project valid enabled rows; inventory remains the consent authority.

    Args:
        config: Saved configuration mapping with an optional Hooks section.

    Returns:
        Master switch and valid enabled execution definitions. Invalid raw
        rows remain in the separate consent inventory for repair. V2 batches
        are independently validated; rejected batches retain body-free
        requirement metadata for the later v2 execution owner.
    """
    inventory = inspect_hooks_config(config)
    if inventory.container_error:
        logger.warning("run-hooks: {}", inventory.container_error)
    for row in inventory.rows:
        if row.error:
            logger.warning("run-hooks: {}; hook disabled", row.error)
    enabled = inventory.master_enabled is True
    hooks = tuple(
        row.spec
        for row in inventory.rows
        if row.spec is not None and row.enabled is True
    )
    section = config.get("hooks", {}) if isinstance(config, Mapping) else {}
    raw_v2 = section.get("handler", []) if isinstance(section, Mapping) else []
    try:
        # Keep the v2 import out of default startup and all legacy-only loads.
        if raw_v2:
            from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

            v2_handlers = parse_handlers(raw_v2)
        else:
            v2_handlers = ()
            if not isinstance(raw_v2, list):
                raise ValueError("handlers must be a list")
    except ValueError:
        logger.warning("run-hooks: invalid v2 handler batch; activation refused")
        invalid_admissions = _invalid_v2_admissions(raw_v2)
        return RunHooksConfig(
            enabled=enabled,
            hooks=hooks,
            v2_invalid=True,
            v2_invalid_required=any(
                item.policy == "explicit_required" for item in invalid_admissions
            ),
            v2_invalid_admissions=invalid_admissions,
        )
    return RunHooksConfig(enabled=enabled, hooks=hooks, v2_handlers=v2_handlers)


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
    """Cap hook-produced text at HOOK_IO_BUDGET_CHARS *total* (marker included).

    TASK-32808.3: delegates to the shared `truncate` with this module's own
    `"…[truncated]"` marker — byte-identical to the previous local body.
    """
    from tldw_chatbook.Utils.Utils import truncate

    return truncate(text, HOOK_IO_BUDGET_CHARS, marker=_TRUNCATION_MARKER)


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

    Ruling R31: UserPromptSubmit stdout injection is gated to exactly
    "exit 0, no parsed decision key" (spec §5). Block decisions, exit 2,
    crashes, timeouts, and ignored decision keys all inject nothing.

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
            logger.warning(
                "run-hooks: UserPromptSubmit hook decision {!r} ignored; "
                "exit code {} suppressed",
                decision_value
                if decision_value in ("allow", "deny", "block")
                else type(decision_value).__name__,
                proc.returncode,
            )
            return _Decision()
        if proc.returncode == 2:
            reason = str((parsed or {}).get("reason") or stderr or "blocked by hook")
            return _Decision(denied=True, reason=_truncate(reason))
        if proc.returncode != 0:
            logger.warning(
                "run-hooks: UserPromptSubmit hook exited {} — failing open",
                proc.returncode,
            )
            return _Decision()
        # Ruling R31: the ONLY injection path — exit 0, no decision key.
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
            return _Decision(
                denied=True,
                reason=f"hook failed (exit {proc.returncode}); failing closed",
            )
        if has_decision:
            # allow/other decision with a clean or shorthand-suppressed exit:
            # no-opinion, logged.
            logger.warning(
                "run-hooks: PreToolUse hook decision {!r} ignored (deny-only; exit {})",
                decision_value
                if decision_value in ("allow", "deny", "block")
                else type(decision_value).__name__,
                proc.returncode,
            )
            return _Decision()
        if stdout.strip():
            # Clean pass, but the hook tried to speak the decision protocol and
            # we could not parse it — surface that instead of silently ignoring.
            logger.warning(
                "run-hooks: PreToolUse hook exited 0 with unrecognized stdout "
                "(expected empty or JSON decision)"
            )
        return _Decision()
    return _Decision()


class _HookProtocol(asyncio.SubprocessProtocol):
    """Drain both pipes continuously while retaining only bounded prefixes."""

    def __init__(self) -> None:
        self.done = asyncio.get_running_loop().create_future()
        self.output = {1: bytearray(), 2: bytearray()}
        self.overflow = {1: False, 2: False}

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        retained = self.output[fd]
        remaining = HOOK_IO_BUDGET_BYTES - len(retained)
        retained.extend(data[:remaining])
        self.overflow[fd] |= len(data) > remaining

    def connection_lost(self, exc: Exception | None) -> None:
        if not self.done.done():
            self.done.set_result(None)

    def text(self, fd: int) -> str:
        value = self.output[fd].decode("utf-8", errors="replace")
        if self.overflow[fd]:
            value += _TRUNCATION_MARKER
        return _truncate(value)


def _kill_process_group(pid: int) -> None:
    """Kill the original group even after its leader has been reaped."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=1,
            )
        else:
            # start_new_session makes the pid the group id. getpgid(pid)
            # cannot recover it once a short-lived leader has exited.
            os.killpg(pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass


async def _capture_hook(
    target: HookTarget,
    payload: dict[str, Any],
    closed: threading.Event | None,
    launch_guard: Callable[..., AbstractContextManager[None]],
) -> tuple[int, _HookProtocol, bool]:
    spec = target.spec
    protocol = _HookProtocol()
    transport = None
    timed_out = False
    try:
        if closed is not None and closed.is_set():
            raise RuntimeError("engine closed")
        cwd = str(validate_existing_absolute_directory(payload["cwd"]))
        payload = dict(payload, cwd=cwd)
        stdin = json.dumps(payload).encode("utf-8")
        data = payload.get("data", {})
        tool_name = data.get("tool_name") if isinstance(data, dict) else None
        with ExitStack() as authority:
            try:
                authority.enter_context(launch_guard(target, tool_name=tool_name))
            except HookLaunchRefused:
                raise
            except Exception:  # noqa: BLE001 - authority failure cannot fail open
                raise HookLaunchRefused(
                    "Hook authority unavailable at launch."
                ) from None
            transport, _ = await asyncio.get_running_loop().subprocess_exec(
                lambda: protocol,
                *spec.command,
                cwd=cwd,
                start_new_session=(sys.platform != "win32"),
            )
        writer = transport.get_pipe_transport(0)
        writer.write(stdin)
        writer.close()
        deadline = time.monotonic() + spec.timeout_s
        while not protocol.done.done():
            if closed is not None and closed.is_set():
                raise RuntimeError("engine closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            await asyncio.wait({protocol.done}, timeout=min(0.05, remaining))
        if timed_out:
            _kill_process_group(transport.get_pid())
            # Close readers explicitly: an escaped descendant may retain
            # the write ends forever. Process reaping is independent of EOF.
            transport.close()
            await asyncio.wait_for(
                asyncio.shield(protocol.done), HOOK_POST_KILL_REAP_TIMEOUT_S
            )
        return transport.get_returncode(), protocol, timed_out
    finally:
        if transport is not None:
            if not protocol.done.done():
                _kill_process_group(transport.get_pid())
            transport.close()
            if not protocol.done.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(protocol.done), HOOK_POST_KILL_REAP_TIMEOUT_S
                    )
                except TimeoutError:
                    logger.warning("run-hooks: bounded process reap expired")


def _run_hook(
    target: HookTarget,
    payload: dict[str, Any],
    closed: threading.Event | None,
    launch_guard: Callable[..., AbstractContextManager[None]],
) -> tuple[HookSpec, _Decision | None]:
    """Execute one hook with bounded output and content-free log attribution."""
    spec = target.spec
    fingerprint = hashlib.sha256(json.dumps(spec.command).encode("utf-8")).hexdigest()[
        :12
    ]
    identity = (
        f"event={spec.event} session_id={payload.get('session_id')} "
        f"run_id={payload.get('run_id')} hook={fingerprint}"
    )
    with logger.contextualize(
        hook_event=spec.event,
        session_id=payload.get("session_id"),
        run_id=payload.get("run_id"),
        hook_id=fingerprint,
    ):
        try:
            return _execute_hook(target, payload, closed, identity, launch_guard)
        except HookLaunchRefused as refusal:
            logger.warning("run-hooks: {} consent refused", identity)
            return spec, (
                _Decision(denied=True, reason=refusal.reason)
                if spec.event in BLOCKING_EVENTS and not refusal.skip
                else None
            )
        except Exception as exc:  # noqa: BLE001 - process owner cleans up first
            logger.warning(
                "run-hooks: {} hook failed (exception_type={})",
                identity,
                type(exc).__name__,
            )
            if spec.event == "UserPromptSubmit":
                logger.warning("run-hooks: {} hook failed — failing open", identity)
            return spec, _hook_raised_decision(spec)


def _execute_hook(
    target: HookTarget,
    payload: dict[str, Any],
    closed: threading.Event | None,
    identity: str,
    launch_guard: Callable[..., AbstractContextManager[None]],
) -> tuple[HookSpec, _Decision | None]:
    spec = target.spec
    started = time.monotonic()
    returncode, capture, timed_out = asyncio.run(
        _capture_hook(target, payload, closed, launch_guard)
    )
    stdout, stderr = capture.text(1), capture.text(2)
    if timed_out:
        logger.warning("run-hooks: {} timeout; process reaped", identity)
        decision = (
            _Decision(
                denied=True,
                reason=f"hook timed out after {spec.timeout_s}s; failing closed",
            )
            if spec.event == "PreToolUse"
            else None
        )
        if spec.event == "UserPromptSubmit":
            logger.warning("run-hooks: {} hook timed out — failing open", identity)
    elif capture.overflow[1]:
        # Never turn incomplete decision JSON into a clean guard pass.
        logger.warning("run-hooks: {} stdout exceeded capture budget", identity)
        decision = (
            _Decision(denied=True, reason="hook output exceeded budget; failing closed")
            if spec.event == "PreToolUse"
            else None
        )
    else:
        # Decision parsing needs the complete bounded bytes, before display
        # truncation (a valid JSON reason may exceed the character budget).
        decision = _decide(
            spec.event,
            subprocess.CompletedProcess(
                spec.command,
                returncode,
                stdout=capture.output[1].decode("utf-8", errors="replace"),
                stderr=stderr,
            ),
        )
    logger.info(
        "run-hooks: {} exit={} timed_out={} stdout={!r} stderr={!r} took_ms={}",
        identity,
        returncode,
        timed_out,
        stdout if spec.event not in BLOCKING_EVENTS else "",
        stderr if spec.event not in BLOCKING_EVENTS else "",
        int((time.monotonic() - started) * 1000),
    )
    return spec, decision


def _hook_raised_decision(spec: HookSpec) -> _Decision | None:
    """Decision substituted when a hook invocation raises instead of returning."""
    return (
        _Decision(denied=True, reason="hook raised; failing closed")
        if spec.event == "PreToolUse"
        else None
    )


def _notification_structure_fits(
    value: Any, *, size_budget: int = HOOK_NOTIFY_PAYLOAD_BYTES
) -> bool:
    """Bound traversal and scalar sizes before JSON allocates encoded chunks.

    The size estimate is a lower bound; the encoder still enforces the exact
    escaped size. Scalars are individually bounded, and node/depth limits also
    bound inspection of tiny values, deeply nested inputs, and cycles.
    """
    remaining = size_budget
    nodes = 0

    def visit(item: Any, depth: int) -> bool:
        nonlocal remaining, nodes
        nodes += 1
        if nodes > HOOK_NOTIFY_MAX_NODES or depth > HOOK_NOTIFY_MAX_DEPTH:
            return False
        if isinstance(item, str):
            remaining -= len(item) + 2
        elif item is None or isinstance(item, (bool, float)):
            remaining -= 1
        elif isinstance(item, int):
            # Decimal digits need at least one character per four bits.
            # Avoid converting an arbitrarily large integer just to size it.
            remaining -= max(1, item.bit_length() // 4)
        elif isinstance(item, (dict, list, tuple)):
            width = len(item) * (2 if isinstance(item, dict) else 1)
            if width > HOOK_NOTIFY_MAX_NODES - nodes:
                return False
            remaining -= 2 + max(0, width - 1)
            if remaining < 0:
                return False
            if isinstance(item, dict):
                for key, child in item.items():
                    if not visit(key, depth + 1) or not visit(child, depth + 1):
                        return False
            else:
                for child in item:
                    if not visit(child, depth + 1):
                        return False
        else:
            return False
        return remaining >= 0

    return visit(value, 0)


def summarize_hook_arguments(arguments: Any) -> str:
    """Serialize small approval arguments without allocating oversized summaries.

    Args:
        arguments: JSON-compatible approval arguments, never coerced with str().

    Returns:
        Complete JSON within the hook text budget, or an explicit omission label.
    """
    omitted = "Arguments omitted (too large or unsupported)."
    try:
        if not _notification_structure_fits(
            arguments, size_budget=HOOK_IO_BUDGET_CHARS
        ):
            return omitted
        encoded = json.dumps(arguments)
        return encoded if len(encoded) <= HOOK_IO_BUDGET_CHARS else omitted
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return omitted


class RunHooksEngine:
    """Executes configured hooks for lifecycle events. Never raises to callers.

    fire()/fire_async() run all matching hooks and block until a decision is
    needed: the first deny wins and remaining results are discarded (their
    processes still run to completion in the pool).

    Pool isolation (ruling R14): blocking events (UserPromptSubmit,
    PreToolUse) execute on a dedicated 4-worker pool that nothing else uses.
    Notification firings remain ordered on one coordinator, with their
    matching hooks concurrent on a separate four-worker pool. Admission is
    bounded; blocking fires never queue behind notification hooks.
    """

    def __init__(
        self,
        target_provider: Callable[[str, str | None], tuple[HookTarget, ...]],
        cwd_provider: Callable[[], str],
        *,
        notification_targets: Callable[[str, str | None], tuple[HookTarget, ...]],
        launch_guard: Callable[..., AbstractContextManager[None]],
    ) -> None:
        self._closed = threading.Event()
        self._admission_lock = threading.Lock()
        self._notify_slots = threading.BoundedSemaphore(HOOK_NOTIFY_CAPACITY)
        self._target_provider = target_provider
        self._notification_targets = notification_targets
        self._launch_guard = launch_guard
        self._cwd_provider = cwd_provider
        # Blocking-event pool only (ruling R14); never shared with notify work.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="run-hook")
        self._notify_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="run-hook-notify"
        )
        self._notification_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="run-hook-observe"
        )

    def _attempt_targets(
        self, authority_read: Any, event: str, tool_name: str | None
    ) -> tuple[()] | None:
        """Answer "no hook to launch" from the firing attempt's earlier read.

        ADR-225 decision 3: a Send attempt that already performed one full
        consent read passes it here instead of having ``_target_provider``
        repeat it -- but only to learn that nothing matches. Used only when
        that read came from the very owner whose ``targets`` and
        ``launch_guard`` this engine was built with, and only when its answer
        is the empty selection. A non-empty selection from an earlier read
        could be stale: a hook another process disabled since would still be
        launched into its fresh ``launch_guard`` and refused, which blocks a
        blocking event where a fresh selection would just have omitted it.
        ``None`` (another owner, a stale read, any target or refusal) means
        the provider's fresh read selects, exactly as without sharing.
        """
        owner = getattr(authority_read, "owner", None)
        select = getattr(owner, "attempt_targets", None)
        if (
            select is None
            or self._target_provider != getattr(owner, "targets", None)
            or self._launch_guard != getattr(owner, "launch_guard", None)
        ):
            return None
        # The owner already answers only "none"; never launch from a read
        # even if a future owner answered more.
        return () if select(authority_read, event, tool_name) == () else None

    def _matching(
        self, event: str, tool_name: str | None, authority_read: Any = None
    ) -> tuple[HookTarget, ...]:
        try:
            targets = (
                self._attempt_targets(authority_read, event, tool_name)
                if authority_read is not None
                else None
            )
            if targets is None:
                targets = self._target_provider(event, tool_name)
            if not isinstance(targets, tuple) or not all(
                isinstance(target, HookTarget) for target in targets
            ):
                raise HookLaunchRefused("Hook authority returned invalid targets.")
            return targets
        except HookLaunchRefused:
            raise
        except Exception:  # noqa: BLE001 - authority cannot inherit fail-open
            raise HookLaunchRefused(
                "Hook authority unavailable; review before continuing."
            ) from None

    def _payload(
        self,
        event: str,
        session_id: str,
        run_id: str | None,
        data: dict[str, Any],
        cwd: str | None = None,
    ) -> dict[str, Any]:
        return {
            "hook_event": event,
            "session_id": session_id,
            "run_id": run_id,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            # Ruling R18: a fire-site-supplied cwd (the session's bound
            # workspace root, when one exists) overrides the provider's
            # app-level fallback; None keeps the provider value.
            "cwd": cwd if cwd is not None else self._cwd_provider(),
            "data": data,
        }

    def fire(
        self,
        event: str,
        *,
        session_id: str,
        run_id: str | None = None,
        data: dict[str, Any] | None = None,
        cwd: str | None = None,
        authority_read: Any = None,
    ) -> HookOutcome:
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
            authority_read: The firing Send attempt's own earlier full consent
                read (``HookAuthorityRead``). While it stands it may answer
                only that no hook matches; any matching hook is selected by a
                fresh read, as with ``None``. Launches re-check authority
                fresh either way.
        """
        try:
            if self._closed.is_set():
                raise RuntimeError("engine closed")
            return self._fire(
                event,
                session_id=session_id,
                run_id=run_id,
                data=data,
                cwd=cwd,
                authority_read=authority_read,
            )
        except HookLaunchRefused as refusal:
            logger.warning("run-hooks: event={} consent refused", event)
            return HookOutcome(
                blocked=event in BLOCKING_EVENTS and not refusal.skip,
                reason=refusal.reason
                if event in BLOCKING_EVENTS and not refusal.skip
                else "",
            )
        except Exception as exc:  # noqa: BLE001 - engine must never raise to callers
            logger.warning(
                "run-hooks: event={} session_id={} run_id={} hook=unavailable "
                "engine error (exception_type={})",
                event,
                session_id,
                run_id,
                type(exc).__name__,
            )
            if event == "PreToolUse":
                return HookOutcome(
                    blocked=True, reason="hooks engine error; failing closed"
                )
            return HookOutcome()

    def _fire(
        self,
        event: str,
        *,
        session_id: str,
        run_id: str | None,
        data: dict[str, Any] | None,
        cwd: str | None = None,
        targets: tuple[HookTarget, ...] | None = None,
        authority_read: Any = None,
    ) -> HookOutcome:
        tool_name: str | None = None
        if isinstance(data, dict):
            candidate = data.get("tool_name")
            if isinstance(candidate, str):
                tool_name = candidate
        targets = (
            self._matching(event, tool_name, authority_read)
            if targets is None
            else targets
        )
        if not targets:
            return HookOutcome()
        payload = self._payload(
            event, session_id, run_id, data if isinstance(data, dict) else {}, cwd
        )
        with self._admission_lock:
            if self._closed.is_set():
                raise RuntimeError("engine closed")
            pool = self._pool if event in BLOCKING_EVENTS else self._notification_pool
            futures = {
                pool.submit(
                    _run_hook, target, payload, self._closed, self._launch_guard
                ): target.spec
                for target in targets
            }
        results = self._results_from_futures(futures)
        return self._reduce(results)

    @staticmethod
    def _results_from_futures(
        futures: dict[Future[tuple[HookSpec, _Decision | None]], HookSpec],
    ) -> Iterator[tuple[HookSpec, _Decision | None]]:
        pending = set(futures)
        while pending:
            # shutdown(cancel_futures=True) leaves some futures CANCELLED,
            # not CANCELLED_AND_NOTIFIED: as_completed can wait forever.
            ready = {future for future in pending if future.done()}
            if not ready:
                ready, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            for fut in ready:
                pending.remove(fut)
                spec = futures[fut]
                try:
                    yield fut.result()
                except Exception as exc:  # noqa: BLE001 - event failure policy
                    logger.warning(
                        "run-hooks: hook raised (exception_type={})", type(exc).__name__
                    )
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

    async def fire_async(
        self, event: str, *, cwd: str | None = None, **kwargs: Any
    ) -> HookOutcome:
        """Async face of `fire` (same args plus the R18 `cwd` override)."""
        try:
            return await asyncio.to_thread(self.fire, event, cwd=cwd, **kwargs)
        except Exception as exc:  # noqa: BLE001 - engine must never raise to callers
            # fire() already never raises; this guards to_thread machinery and
            # argument-binding errors from the in-thread call site. Degrade with
            # fire()'s semantics: PreToolUse fails closed, others dropped.
            logger.warning(
                "run-hooks: fire_async({}) engine error (exception_type={})",
                event,
                type(exc).__name__,
            )
            if event == "PreToolUse":
                return HookOutcome(
                    blocked=True, reason="hooks engine error; failing closed"
                )
            return HookOutcome()

    def notify(self, event: str, *, cwd: str | None = None, **kwargs: Any) -> None:
        """Admit a bounded immutable notification or drop it whole with a diagnostic.

        Args:
            event: Nonblocking lifecycle event.
            cwd: Optional working-directory override.
            **kwargs: Arguments to fire, serialized within the notification budget.
        """
        if self._closed.is_set() or not self._notify_slots.acquire(blocking=False):
            logger.warning("run-hooks: notify dropped (closed or queue full)")
            return
        submitted = False
        try:
            # iterencode emits each scalar as one complete chunk. Reject
            # oversized structures before allocating those escaped strings.
            payload = dict(kwargs, cwd=cwd)
            if not _notification_structure_fits(payload):
                logger.warning(
                    "run-hooks: notify dropped (payload too large or complex)"
                )
                return
            # Freeze ownership and enforce the exact escaped encoding budget.
            chunks: list[str] = []
            size = 0
            for chunk in json.JSONEncoder().iterencode(payload):
                size += len(chunk)
                if size > HOOK_NOTIFY_PAYLOAD_BYTES:
                    logger.warning("run-hooks: notify dropped (payload too large)")
                    return
                chunks.append(chunk)
            frozen = json.loads("".join(chunks))
            data = frozen.get("data")
            tool_name = data.get("tool_name") if isinstance(data, dict) else None
            targets = self._notification_targets(event, tool_name)
            if not targets:
                return
            with self._admission_lock:
                if self._closed.is_set():
                    logger.warning("run-hooks: notify dropped (closed)")
                    return
                future = self._notify_worker.submit(
                    self._fire,
                    event,
                    session_id=frozen["session_id"],
                    run_id=frozen.get("run_id"),
                    data=frozen.get("data"),
                    cwd=frozen.get("cwd"),
                    targets=targets,
                )
                future.add_done_callback(self._notification_done)
                submitted = True
        except Exception as exc:  # noqa: BLE001 - notifications never propagate
            logger.warning(
                "run-hooks: notify dropped (exception_type={})", type(exc).__name__
            )
        finally:
            if not submitted:
                self._notify_slots.release()

    def _notification_done(self, future: Any) -> None:
        self._notify_slots.release()
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                logger.warning(
                    "run-hooks: notification omitted (exception_type={})",
                    type(error).__name__,
                )

    def close(self) -> None:
        """Seal admission and cancel queued work without blocking the application.

        Live process owners observe cancellation within 50 ms, kill their process
        groups, close pipes, and reap before their worker exits. Repeated calls
        are harmless; PreToolUse fires against a closed engine fail closed.
        """
        with self._admission_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            for pool in (self._notify_worker, self._notification_pool, self._pool):
                pool.shutdown(wait=False, cancel_futures=True)

    def wrap_review(
        self, inner: Callable[[list[ToolCall], str], dict[str, str]], *, session_id: str
    ) -> Callable[[list[ToolCall], str], dict[str, str]]:
        """Wrap a review_tool_calls callable so PreToolUse hooks deny first.

        Denied names are EXCLUDED from the inner review (no approval card for
        a call a hook already refused) and merged back as refusal strings,
        namespaced with a "hook: " prefix (ruling R16): hook-produced text can
        never equal the "proceed" dispatch sentinel, so a deny stays a deny.
        Hooks can never produce "proceed" — deny-only, enforced here.
        """

        def review(calls: list[ToolCall], run_id: str) -> dict[str, str]:
            refusals: dict[str, str] = {}
            surviving: list[ToolCall] = []
            for call in calls:
                outcome = self.fire(
                    "PreToolUse",
                    session_id=session_id,
                    run_id=run_id,
                    data={"tool_name": call.name, "tool_args": call.args},
                )
                if outcome.blocked:
                    refusals[call.call_id or call.name] = f"hook: {outcome.reason}"
                else:
                    surviving.append(call)
            try:
                verdicts = dict(inner(surviving, run_id)) if surviving else {}
            except Exception as exc:  # noqa: BLE001 - preserve guard refusals
                logger.warning(
                    "run-hooks: inner review failed (exception_type={})",
                    type(exc).__name__,
                )
                verdicts = {
                    call.call_id
                    or call.name: "hook: permission review failed; failing closed"
                    for call in surviving
                }
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

        def post_tool_call(
            tool_name: str,
            call_id: str,
            args: dict,
            content: str,
            ok: bool,
            run_id: str | None = None,
        ) -> None:
            self.notify(
                "PostToolUse",
                session_id=session_id,
                run_id=run_id,
                data={
                    "tool_name": tool_name,
                    "tool_args": args,
                    "tool_result": _truncate(content),
                    "is_error": not ok,
                },
            )

        return post_tool_call
