"""Console run hooks: user-configured external commands on session/run lifecycle events.

Spec: Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md (ADR-345).
Deny-only guardrails, argv-list commands, JSON-on-stdin protocol, per-purpose
fail direction (PreToolUse fails closed, UserPromptSubmit fails open).
"""

from __future__ import annotations

# Capture the original tuple before other module initialization.
# isort: split
from tldw_chatbook.Agents import legacy_hook_ownership as _cleanup_owner
from tldw_chatbook.Agents.legacy_hook_ownership import (
    _CLEANUP_ORIGINALS as _ORIGINAL_CLEANUP_SOURCE,
)
from tldw_chatbook.Chat.console_ordinary_tree import (
    _NATIVE_CURRENTNESS_SOURCE as _ORIGINAL_CURRENTNESS_SOURCE,
)

# isort: split

import asyncio
import datetime
import fnmatch
import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from types import FunctionType
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


def _native_popen_record(function):
    if type(function) is not FunctionType:
        return None
    return (
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
    )


_ORIGINAL_NATIVE_POPEN = subprocess.Popen
_NATIVE_POPEN_DEFINITIONS = tuple(
    (name, _native_popen_record(vars(_ORIGINAL_NATIVE_POPEN).get(name)))
    for name in ("__init__", "__del__", "_internal_poll", "_execute_child")
)
_ORIGINAL_SUBPROCESS_CLEANUP = subprocess._cleanup
_ORIGINAL_SUBPROCESS_CLEANUP_RECORD = _native_popen_record(_ORIGINAL_SUBPROCESS_CLEANUP)


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
        self.exited = asyncio.get_running_loop().create_future()
        self.transport = None
        self.output = {1: bytearray(), 2: bytearray()}
        self.overflow = {1: False, 2: False}

    def connection_made(self, transport) -> None:
        self.transport = transport

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)

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


def _native_darwin_abi():
    return sys.platform == "darwin" and platform.machine() in ("arm64", "x86_64")


def _native_popen_current():
    unchanged = _ORIGINAL_CURRENTNESS_SOURCE[5][0]
    return bool(
        type(_ORIGINAL_NATIVE_POPEN) is type
        and subprocess.Popen is _ORIGINAL_NATIVE_POPEN
        and subprocess._cleanup is _ORIGINAL_SUBPROCESS_CLEANUP
        and _ORIGINAL_SUBPROCESS_CLEANUP_RECORD is not None
        and unchanged(_ORIGINAL_SUBPROCESS_CLEANUP_RECORD)
        and vars(_ORIGINAL_NATIVE_POPEN).get("__new__", object.__new__)
        is object.__new__
        and vars(_ORIGINAL_NATIVE_POPEN).get(
            "__getattribute__", object.__getattribute__
        )
        is object.__getattribute__
        and vars(_ORIGINAL_NATIVE_POPEN).get("__setattr__", object.__setattr__)
        is object.__setattr__
        and all(
            record is not None
            and vars(_ORIGINAL_NATIVE_POPEN).get(name) is record[0]
            and unchanged(record)
            for name, record in _NATIVE_POPEN_DEFINITIONS
        )
    )


def _native_cleanup_definitions_current():
    (
        source_type,
        target_type,
        event_type,
        event_source_type,
        issuer_type,
        guard,
        definitions,
        reader,
        registry_type,
        condition_type,
        condition_lock_type,
        event_lock_type,
    ) = _ORIGINAL_CLEANUP_SOURCE
    checker = _ORIGINAL_CURRENTNESS_SOURCE[5]
    unchanged = checker[0]
    namespace = vars(_cleanup_owner)
    activation = next(owner for owner, name, _ in definitions if name == "_sources")
    operation_type = next(
        owner for owner, name, _ in definitions if name == "mark_stopping"
    )
    owner_type = next(
        owner for owner, name, _ in definitions if name == "mark_cleanup_unproven"
    )
    if (
        namespace.get("_CLEANUP_ORIGINALS") is not _ORIGINAL_CLEANUP_SOURCE
        or namespace.get("_CleanupSource") is not source_type
        or namespace.get("_TargetCustody") is not target_type
        or namespace.get("_EventCustody") is not event_type
        or namespace.get("_EventSource") is not event_source_type
        or namespace.get("_NativeLegacyHookSource") is not issuer_type
        or namespace.get("_EventCustodyRegistry") is not registry_type
        or namespace.get("Condition") is not condition_type
        or namespace.get("_CONDITION_LOCK_TYPE") is not condition_lock_type
        or namespace.get("_EVENT_LOCK_TYPE") is not event_lock_type
        or namespace.get("OwnedOperation") is not operation_type
        or namespace.get("ExecutionOwner") is not owner_type
        or namespace.get("_guard") is not guard
        or namespace.get("_CLEANUP_DEFINITIONS") is not definitions
        or namespace.get("_cleanup_getattr_static") is not reader
        or namespace.get("_cleanup_activation") is not activation
        or vars(activation).get("_guard") is not guard
        or vars(guard).get("sources") is not vars(activation).get("_sources")
        or any(
            vars(cls).get("__getattribute__", object.__getattribute__)
            is not object.__getattribute__
            for cls in (
                source_type,
                target_type,
                event_type,
                event_source_type,
                issuer_type,
            )
        )
        or any(
            vars(cls).get("__setattr__", object.__setattr__) is not object.__setattr__
            for cls in (
                source_type,
                target_type,
                event_type,
                event_source_type,
                issuer_type,
            )
        )
        or unchanged.__code__ is not checker[1]
        or unchanged.__defaults__ is not checker[2]
        or unchanged.__kwdefaults__ is not checker[3]
        or unchanged.__closure__ is not checker[5]
    ):
        return False
    return all(
        vars(owner).get(name) is record[0] and unchanged(record)
        for owner, name, record in definitions
    )


def _native_linux_abi():
    if sys.platform != "linux" or platform.machine() not in ("aarch64", "x86_64"):
        return False
    try:
        return os.confstr("CS_GNU_LIBC_VERSION").startswith("glibc ")
    except (AttributeError, OSError, ValueError):
        return False


def _native_host_preflight():
    """Refuse configurations that cannot preserve the original waiter."""
    import ctypes
    from asyncio import unix_events

    if sys.implementation.name != "cpython" or sys.version_info[:2] not in (
        (3, 12),
        (3, 13),
        (3, 14),
    ):
        raise RuntimeError("native hook Python watcher is unqualified")
    policy = asyncio.get_event_loop_policy()
    if type(policy) is not unix_events._UnixDefaultEventLoopPolicy:
        raise RuntimeError("native hook child watcher is unqualified")
    loop = asyncio.get_running_loop()
    if type(loop) is not unix_events._UnixSelectorEventLoop:
        raise RuntimeError("native hook event loop is unqualified")
    if sys.version_info[:2] == (3, 14):
        watcher = vars(loop).get("_watcher")
        supported = (unix_events._ThreadedChildWatcher, unix_events._PidfdChildWatcher)
    else:
        watcher = vars(policy).get("_watcher")
        supported = (unix_events.ThreadedChildWatcher, unix_events.PidfdChildWatcher)
    if watcher is not None and type(watcher) not in supported:
        raise RuntimeError("native hook child watcher is unqualified")
    if sys.version_info[:2] == (3, 14) and watcher is None:
        raise RuntimeError("native hook child watcher is unqualified")
    if _native_darwin_abi():
        fields = [
            ("handler", ctypes.c_void_p),
            ("mask", ctypes.c_uint32),
            ("flags", ctypes.c_int),
        ]
        expected, no_cld_wait = (16, 8, 12), 32
    elif _native_linux_abi() and hasattr(os, "waitid"):
        fields = [
            ("handler", ctypes.c_void_p),
            ("mask", ctypes.c_ulong * 16),
            ("flags", ctypes.c_int),
            ("restorer", ctypes.c_void_p),
        ]
        expected, no_cld_wait = (152, 8, 136), 2
    else:
        raise RuntimeError("native hook wait ABI is unqualified")

    class Sigaction(ctypes.Structure):
        _fields_ = fields

    if (
        ctypes.sizeof(Sigaction),
        ctypes.alignment(Sigaction),
        Sigaction.flags.offset,
    ) != expected:
        raise RuntimeError("native hook sigaction ABI mismatch")
    action = Sigaction()
    libc = ctypes.CDLL(None, use_errno=True)
    sigaction = libc.sigaction
    sigaction.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Sigaction))
    sigaction.restype = ctypes.c_int
    if sigaction(signal.SIGCHLD, None, ctypes.byref(action)) != 0:
        raise OSError(ctypes.get_errno(), "sigaction failed")
    if action.handler is not None or action.flags & no_cld_wait:
        raise RuntimeError("native hook SIGCHLD disposition is unqualified")


def _native_observe(pid):
    """Observe this child's terminal status without consuming it."""
    if hasattr(os, "waitid"):
        info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        if info is None:
            return False
        child, code = info.si_pid, info.si_code
        terminal = (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)
    elif _native_darwin_abi():
        import ctypes

        class Info(ctypes.Union):
            _fields_ = [("alignment", ctypes.c_void_p), ("data", ctypes.c_ubyte * 104)]

        if (ctypes.sizeof(Info), ctypes.alignment(Info)) != (104, 8):
            raise RuntimeError("native hook wait ABI mismatch")
        info = Info()
        waitid = ctypes.CDLL(None, use_errno=True).waitid
        waitid.argtypes = (
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.POINTER(Info),
            ctypes.c_int,
        )
        waitid.restype = ctypes.c_int
        if waitid(1, pid, ctypes.byref(info), 4 | 32 | 1) == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        child = ctypes.c_int.from_buffer(info, 12).value
        code = ctypes.c_int.from_buffer(info, 8).value
        terminal = (1, 2, 3)
        if child == 0:
            return False
    else:
        raise RuntimeError("native hook wait ABI is unqualified")
    if child != pid or code not in terminal:
        raise RuntimeError("native hook child status is unqualified")
    return True


def _native_lost(native, process):
    # The object remains attached to the unresolved target; CPython must not
    # later poll a PID whose wait ownership was lost.
    object.__setattr__(process, "_child_created", False)
    object.__setattr__(native, "_retained_process", process)


def _native_late_reap(native, process):
    def reap():
        try:
            while True:
                try:
                    pid, status = os.waitpid(process.pid, 0)
                    break
                except InterruptedError:
                    continue
            if pid == process.pid:
                object.__setattr__(
                    process, "returncode", os.waitstatus_to_exitcode(status)
                )
            else:
                _native_lost(native, process)
        except OSError:
            _native_lost(native, process)

    object.__setattr__(native, "_retained_process", process)
    worker = threading.Thread(target=reap, name="legacy-hook-exact-reap", daemon=True)
    object.__setattr__(native, "_late_reaper", worker)
    worker.start()


async def _native_cleanup(
    native, process, deadline, current, report, owned=True, io_returned=True
):
    """Signal once before exact wait, then separately prove group absence."""
    if owned and not current():
        owned = False
    if owned:
        try:
            while True:
                if not current():
                    owned = False
                    break
                try:
                    _native_observe(process.pid)
                    break
                except InterruptedError:
                    if time.monotonic() >= deadline:
                        owned = False
                        break
        except (OSError, RuntimeError):
            owned = False
    if not owned:
        _native_lost(native, process)
    elif current():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    else:
        _native_lost(native, process)
        owned = False

    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                io_returned = False

    status_returned = False
    if owned:
        while True:
            if not current():
                _native_lost(native, process)
                owned = False
                break
            try:
                pid, status = os.waitpid(process.pid, os.WNOHANG)  # noqa: ASYNC222 -- nonblocking exact-child observation
            except InterruptedError:
                if time.monotonic() >= deadline:
                    _native_lost(native, process)
                    break
                continue
            except OSError:
                _native_lost(native, process)
                owned = False
                break
            if pid == process.pid:
                object.__setattr__(
                    process, "returncode", os.waitstatus_to_exitcode(status)
                )
                status_returned = True
                break
            if pid != 0:
                _native_lost(native, process)
                owned = False
                break
            if time.monotonic() >= deadline:
                try:
                    _native_late_reap(native, process)
                except (OSError, RuntimeError):
                    _native_lost(native, process)
                break
            await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))

    absent = False
    if status_returned:
        while True:
            if not current() or time.monotonic() >= deadline:
                break
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                absent = time.monotonic() < deadline
                break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))
    proven = status_returned and io_returned and absent and current()
    report(proven)
    return proven


async def _native_finish(
    native, process, current, report, owned=True, io_returned=True
):
    """Complete the original finalizer despite repeated caller cancellation."""
    cleanup = asyncio.create_task(
        _native_cleanup(
            native,
            process,
            time.monotonic() + HOOK_POST_KILL_REAP_TIMEOUT_S,
            current,
            report,
            owned,
            io_returned,
        )
    )
    cancelled = None
    while True:
        try:
            return await asyncio.shield(cleanup), cancelled
        except asyncio.CancelledError as exc:
            if cleanup.cancelled():
                if process.returncode is None:
                    _native_lost(native, process)
                report(False)
                raise
            cancelled = cancelled or exc
        except BaseException:
            if process.returncode is None:
                _native_lost(native, process)
            report(False)
            raise


async def _native_capture(
    native, process, protocol, stdin, closed, timeout, current, report
):
    """Drain all three nonblocking pipes and retain custody through cleanup."""
    timed_out = False
    failure = None
    terminal = False
    wait_owned = True
    io_returned = True
    eof = set()
    offset = 0
    streams = {0: process.stdin, 1: process.stdout, 2: process.stderr}
    selector = None
    try:
        selector = selectors.DefaultSelector()
        for stream in streams.values():
            os.set_blocking(stream.fileno(), False)
        for fd in (1, 2):
            selector.register(streams[fd], selectors.EVENT_READ, fd)
        selector.register(streams[0], selectors.EVENT_WRITE, 0)
        deadline = time.monotonic() + timeout
        while True:
            if closed is not None and closed.is_set():
                raise RuntimeError("engine closed")
            if time.monotonic() >= deadline:
                timed_out = True
                break
            if not current():
                wait_owned = False
                raise RuntimeError("native hook cleanup source changed")
            try:
                terminal = _native_observe(process.pid)
            except InterruptedError:
                continue
            except BaseException:
                wait_owned = False
                raise
            if terminal and eof == {1, 2}:
                break
            ready = selector.select(0)
            for key, _ in ready:
                fd = key.data
                if fd == 0:
                    try:
                        offset += os.write(
                            streams[0].fileno(), stdin[offset : offset + 65536]
                        )
                    except BrokenPipeError:
                        offset = len(stdin)
                    except BlockingIOError:
                        continue
                    if offset == len(stdin):
                        selector.unregister(streams[0])
                        try:
                            streams[0].close()
                        except OSError:
                            io_returned = False
                            raise
                else:
                    try:
                        data = os.read(streams[fd].fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if data:
                        protocol.pipe_data_received(fd, data)
                    else:
                        eof.add(fd)
                        selector.unregister(streams[fd])
                        try:
                            streams[fd].close()
                        except OSError:
                            io_returned = False
                            raise
            await asyncio.sleep(0 if ready else 0.01)
    except BaseException as exc:  # noqa: BLE001 -- cancellation must still drain the original child
        failure = exc
    finally:
        if selector is not None:
            try:
                selector.close()
            except OSError as exc:
                io_returned = False
                failure = failure or exc

    if failure is not None or timed_out:
        protocol.output = {1: bytearray(), 2: bytearray()}
    proven, cancelled = await _native_finish(
        native, process, current, report, wait_owned, io_returned
    )
    failure = failure or cancelled
    if not proven:
        raise RuntimeError("native hook cleanup unproven") from failure
    if failure is not None:
        raise failure
    return process.returncode, protocol, timed_out


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
    native_process = None
    native_capture_started = False
    native_current = lambda: False
    native_report = None
    timed_out = False
    from .legacy_hook_ownership import current_target

    native = current_target()
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
            if native is not None:
                from types import ModuleType as _currentness_module_type

                from tldw_chatbook.Chat import (
                    console_chat_controller as _currentness_controller,
                )
                from tldw_chatbook.Chat import console_runtime as _currentness_runtime

                _currentness_source = _ORIGINAL_CURRENTNESS_SOURCE
                if (
                    type(_currentness_runtime) is not _currentness_module_type
                    or type(_currentness_controller) is not _currentness_module_type
                    or _currentness_source is not vars(_currentness_runtime).get("_ORIGINAL_CURRENTNESS_SOURCE")
                    or _currentness_source is not vars(_currentness_controller).get("_ORIGINAL_CURRENTNESS_SOURCE")
                    or _currentness_source[6].modules.get("tldw_chatbook.Chat.console_runtime") is not _currentness_runtime
                    or _currentness_source[6].modules.get("tldw_chatbook.Chat.console_chat_controller") is not _currentness_controller
                ):
                    raise RuntimeError("original native currentness consumer changed")
                _currentness_record = _currentness_source[4][-1][1]
                _currentness_reader, _currentness_code, _currentness_defaults, _currentness_kwdefaults, _currentness_entries, _currentness_closure, _currentness_cells = _currentness_record
                if (
                    _currentness_reader.__code__ is not _currentness_code
                    or _currentness_reader.__defaults__ is not _currentness_defaults
                    or _currentness_reader.__kwdefaults__ is not _currentness_kwdefaults
                    or tuple((_currentness_kwdefaults or {}).items()) != _currentness_entries
                    or _currentness_reader.__closure__ is not _currentness_closure
                    or any(cell.cell_contents is not value for cell, value in _currentness_cells)
                ):
                    raise RuntimeError("original native currentness dispatcher changed")

                if not _native_cleanup_definitions_current():
                    raise RuntimeError("native hook cleanup definitions changed")
                if not _native_popen_current():
                    raise RuntimeError("native hook process owner changed")
                if (
                    type(native) is not _ORIGINAL_CLEANUP_SOURCE[1]
                    or type(native.event) is not _ORIGINAL_CLEANUP_SOURCE[2]
                    or type(native.event.source) is not _ORIGINAL_CLEANUP_SOURCE[3]
                    or type(native.event.source.issuer)
                    is not _ORIGINAL_CLEANUP_SOURCE[4]
                ):
                    raise RuntimeError("native hook cleanup receiver changed")
                native.event.source.check()
                if not _currentness_reader(_currentness_source, 'platform'):
                    raise RuntimeError("native hook group proof unavailable")
                _native_host_preflight()
                cleanup_originals = _ORIGINAL_CLEANUP_SOURCE
                snapshot = cleanup_originals[0](native, target)
                current_method = vars(_ORIGINAL_CLEANUP_SOURCE[0])["current"]
                terminal_method = vars(_ORIGINAL_CLEANUP_SOURCE[0])["terminal"]
                terminal_code = terminal_method.__code__
                definition_check = _native_cleanup_definitions_current
                definition_code = definition_check.__code__
                popen_check = _native_popen_current
                popen_check_code = popen_check.__code__

                def native_report(proven):
                    if terminal_method.__code__ is not terminal_code:
                        object.__setattr__(native, "proven", False)
                        return
                    terminal_method(snapshot, proven)

                def native_current():
                    try:
                        return bool(
                            _native_cleanup_definitions_current is definition_check
                            and definition_check.__code__ is definition_code
                            and _ORIGINAL_CLEANUP_SOURCE is cleanup_originals
                            and definition_check()
                            and _native_popen_current is popen_check
                            and popen_check.__code__ is popen_check_code
                            and popen_check()
                            and (
                                native_process is None
                                or type(native_process) is _ORIGINAL_NATIVE_POPEN
                            )
                            and _currentness_reader.__code__ is _currentness_code
                            and _currentness_reader.__defaults__
                            is _currentness_defaults
                            and _currentness_reader.__kwdefaults__
                            is _currentness_kwdefaults
                            and _currentness_reader.__closure__ is _currentness_closure
                            and tuple((_currentness_kwdefaults or {}).items())
                            == _currentness_entries
                            and all(
                                cell.cell_contents is value
                                for cell, value in _currentness_cells
                            )
                            and _currentness_reader(_currentness_source, "platform")
                            and current_method(snapshot, native_process)
                        )
                    except Exception:  # noqa: BLE001 -- changed original source fails closed
                        return False

                if not native_current():
                    raise RuntimeError("native hook cleanup source changed")
                native.process_attempted()
                try:
                    native_process = _ORIGINAL_NATIVE_POPEN(
                        spec.command,
                        cwd=cwd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                        start_new_session=True,
                    )
                    object.__setattr__(native, "_cleanup_process", native_process)
                except (OSError, ValueError):
                    native_report(True)
                    raise
            else:
                transport, _ = await asyncio.get_running_loop().subprocess_exec(
                    lambda: protocol,
                    *spec.command,
                    cwd=cwd,
                    start_new_session=(sys.platform != "win32"),
                )
        if native_process is not None:
            native_capture_started = True
            return await _native_capture(
                native,
                native_process,
                protocol,
                stdin,
                closed,
                spec.timeout_s,
                native_current,
                native_report,
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
        transport = transport or protocol.transport
        if native_process is not None and not native_capture_started:
            proven, cancelled = await _native_finish(
                native, native_process, native_current, native_report
            )
            if not proven:
                logger.warning(
                    "run-hooks: native process/group cleanup remains unproven"
                )
                raise RuntimeError("native hook cleanup unproven")
            if cancelled is not None:
                raise cancelled
        elif transport is not None:
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
        logger.warning("run-hooks: {} timeout; bounded reap attempted", identity)
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
        self._native_source = None
        self._native_attached = False
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

        ADR-230 decision 3: a Send attempt that already performed one full
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
        _custody=None,
    ) -> HookOutcome:
        from contextlib import nullcontext

        tool_name = data.get("tool_name") if isinstance(data, dict) else None
        tool_name = tool_name if isinstance(tool_name, str) else None
        if self._native_attached and self._native_source is None:
            raise RuntimeError("native hook source unavailable")
        capture = (
            self._native_source.capture(session_id)
            if self._native_attached and _custody is None
            else nullcontext(None)
        )
        custody = _custody
        futures = {}
        try:
            with capture as source:
                targets = (
                    self._matching(event, tool_name, authority_read) if targets is None else targets
                )
                if not targets:
                    return HookOutcome()
                payload = self._payload(
                    event,
                    session_id,
                    run_id,
                    data if isinstance(data, dict) else {},
                    cwd,
                )
                if source is not None:
                    if event != "UserPromptSubmit":
                        source.handoff = None
                    custody = source.admit(targets)
                if custody is not None:
                    custody.source.check()
                with self._admission_lock:
                    if self._closed.is_set() or (
                        custody is not None and not custody.source.current_memory()
                    ):
                        raise RuntimeError("hook source closed or changed")
                    pool = (
                        self._pool
                        if event in BLOCKING_EVENTS
                        else self._notification_pool
                    )
                    for index, target in enumerate(targets):
                        args = (target, payload, self._closed, self._launch_guard)
                        future = (
                            pool.submit(custody.targets[index].invoke, _run_hook, *args)
                            if custody is not None
                            else pool.submit(_run_hook, *args)
                        )
                        futures[future] = target.spec
                        if custody is not None:
                            custody.targets[index].future = future
                if custody is not None:
                    custody.cancel_queued_if_requested()
                return self._reduce(self._results_from_futures(futures))
        finally:
            if custody is not None:
                for future, target in zip(futures, custody.targets):
                    future.add_done_callback(
                        lambda done, target=target: (
                            target.cancel_before_start() if done.cancelled() else None
                        )
                    )
                # A submit failure may leave some targets accepted by the pool.
                # Only positively cancelled/never-submitted targets can unwind.
                for target in custody.targets[len(futures) :]:
                    target.cancel_before_start()
                if custody.source.handoff is None and _custody is None:
                    custody.host_done()

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

    def prepare_notification(self, event: str, session_id: str, tool_name=None):
        """Prehold this original ordinary producer's optional legacy notification."""
        if not self._native_attached or self._native_source is None:
            return None
        try:
            with self._native_source.capture(session_id) as source:
                targets = self._notification_targets(event, tool_name)
                if not targets:
                    return None
                source.handoff = None
                custody = source.admit(targets)
                custody.notification = (event, session_id, tool_name)
                custody.notification_claimed = False
                return custody
        except Exception as exc:  # noqa: BLE001 - optional notification stays fail-open
            logger.warning(
                "run-hooks: future notification omitted ({})", type(exc).__name__
            )
            return None

    def notify(
        self, event: str, *, cwd: str | None = None, _prepared=None, **kwargs: Any
    ) -> None:
        """Admit a bounded immutable notification or drop it whole with a diagnostic.

        Args:
            event: Nonblocking lifecycle event.
            cwd: Optional working-directory override.
            **kwargs: Arguments to fire, serialized within the notification budget.
        """
        custody = _prepared
        if custody is not None:
            from .legacy_hook_ownership import _EventCustody

            data = kwargs.get("data")
            tool_name = data.get("tool_name") if isinstance(data, dict) else None
            if (
                type(custody) is not _EventCustody
                or custody.source.issuer.engine is not self
                or getattr(custody, "notification", None)
                != (event, kwargs.get("session_id"), tool_name)
            ):
                raise ValueError("original notification custody required")
            with custody._lock:
                if custody.notification_claimed:
                    raise ValueError("notification already consumed")
                custody.notification_claimed = True
        if self._closed.is_set() or not self._notify_slots.acquire(blocking=False):
            logger.warning("run-hooks: notify dropped (closed or queue full)")
            if custody is not None:
                custody.abandon_unstarted()
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
            from contextlib import nullcontext

            if self._native_attached and self._native_source is None:
                raise RuntimeError("native hook source unavailable")
            capture = (
                self._native_source.capture(frozen["session_id"])
                if self._native_attached and custody is None
                else nullcontext(None)
            )
            with capture as source:
                targets = (
                    custody.source.targets
                    if custody is not None
                    else self._notification_targets(event, tool_name)
                )
                if not targets:
                    return
                if source is not None:
                    # Notifications have their own conclusive discard consumer.
                    source.handoff = None
                    custody = source.admit(targets)
                with self._admission_lock:
                    if self._closed.is_set() or (
                        custody is not None and not custody.source.current_memory()
                    ):
                        logger.warning("run-hooks: notify dropped (closed or changed)")
                        return
                    future = self._notify_worker.submit(
                        self._fire,
                        event,
                        session_id=frozen["session_id"],
                        run_id=frozen.get("run_id"),
                        data=frozen.get("data"),
                        cwd=frozen.get("cwd"),
                        targets=targets,
                        _custody=custody,
                    )
                    submitted = True
                original_discard = self._notification_done

                def discarded(done):
                    try:
                        original_discard(done)
                    finally:
                        if custody is not None:
                            if done.cancelled():
                                custody.abandon_unstarted()
                            else:
                                custody.host_done()

                future.add_done_callback(discarded)
        except Exception as exc:  # noqa: BLE001 - notifications never propagate
            logger.warning(
                "run-hooks: notify dropped (exception_type={})", type(exc).__name__
            )
        finally:
            if not submitted:
                self._notify_slots.release()
                if custody is not None:
                    custody.abandon_unstarted()

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


_NATIVE_LEGACY_CLOSE_HOOK_SOURCE = (
    sys.modules[__name__],
    RunHooksEngine,
    _ORIGINAL_CLEANUP_SOURCE,
    tuple(
        (name, _native_popen_record(globals()[name]))
        for name in (
            "_capture_hook",
            "_native_cleanup_definitions_current",
        )
    ),
)
