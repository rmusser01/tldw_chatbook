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
