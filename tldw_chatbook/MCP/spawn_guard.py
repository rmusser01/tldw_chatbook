"""TASK-26013: MCP server-spawn configuration guard.

Any command saved as an MCP server is a code-execution primitive, and imported
configs (Claude Desktop JSON via ``mcp_import.py``) are an untrusted-input path.
Nothing previously inspected the *command* -- only its environment. This module
screens the command line for a small set of well-known dangerous shapes and is
applied identically at save time, spawn time, and import time so a config
edited on disk cannot bypass the save-time check.

MCP servers are spawned with ``create_subprocess_exec`` (no shell), so the real
danger is a shell/interpreter ``-c``/``-e`` payload; the whole command line is
scanned defensively regardless, since an ordinary stdio server (npx/uvx/
``python -m``/a plain binary) contains none of these shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence


class SpawnGuardError(RuntimeError):
    """Raised (only when requested) when a spawn config matches a dangerous rule."""

    def __init__(self, rule: str, reason: str):
        self.rule = rule
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class SpawnGuardVerdict:
    """A matched dangerous-shape rule and its human-readable reason."""

    rule: str
    reason: str


# Interpreters whose inline-code flags execute an arbitrary payload string.
_INTERPRETERS = r"(?:sh|bash|zsh|dash|ksh|python[0-9.]*|perl|ruby|php|node|nodejs|pwsh|powershell)"
# Tools that fetch a remote resource.
_FETCHERS = r"(?:curl|wget|fetch|iwr|invoke-webrequest|invoke-restmethod|httpie|http)"
# Sensitive files whose modification is a persistence primitive.
_SENSITIVE_PATHS = (
    r"(?:\.bashrc|\.bash_profile|\.bash_login|\.zshrc|\.zprofile|\.profile"
    r"|\.zshenv|\.bash_logout|authorized_keys|\.ssh/|/etc/cron|crontab"
    r"|\.config/autostart|LaunchAgents|\.bin/|/etc/rc\.local)"
)
# Markers that a payload is encoded/obfuscated rather than a plain module load.
_ENCODED_MARKERS = (
    r"(?:base64|b64decode|b64encode|fromCharCode|atob|"
    r"frombase64string|\-enc\b|\-encodedcommand\b|codecs\.decode|"
    r"eval\b|exec\b|__import__|Buffer\.from)"
)


def _rules(cmdline: str):
    """Yield (rule, reason) for each dangerous shape present in ``cmdline``."""
    low = cmdline.lower()

    # 1. Remote fetch piped/chained into an interpreter.
    if re.search(_FETCHERS, low) and re.search(
        _FETCHERS + r".*[|;&`]|\$\(.*" + _FETCHERS, low
    ) and re.search(r"[|;&]\s*" + _INTERPRETERS + r"\b|" + _INTERPRETERS + r"\s*<", low):
        yield (
            "remote-fetch-piped-to-interpreter",
            "command fetches a remote resource and pipes it into an interpreter",
        )
    elif re.search(_FETCHERS, low) and re.search(r"\|\s*" + _INTERPRETERS + r"\b", low):
        yield (
            "remote-fetch-piped-to-interpreter",
            "command fetches a remote resource and pipes it into an interpreter",
        )

    # 2. Writing to shell startup files or authorized_keys (persistence).
    if re.search(_SENSITIVE_PATHS, low) and re.search(
        r"(?:>>|>|\btee\b|\bdd\b\s+of=|\bcp\b|\bmv\b|\bcat\b.*>>?)", low
    ):
        yield (
            "shell-startup-or-authorized_keys-write",
            "command writes to a shell startup file or authorized_keys",
        )

    # 3. Inline interpreter invocation of an encoded/obfuscated payload.
    if re.search(
        _INTERPRETERS + r"\b.*\s-(?:c|e|enc|encodedcommand)\b", low
    ) and re.search(_ENCODED_MARKERS, low):
        yield (
            "inline-interpreter-encoded-payload",
            "command invokes an interpreter with an inline encoded/obfuscated payload",
        )
    # PowerShell's -enc always carries a base64 payload by definition.
    elif re.search(r"(?:pwsh|powershell)\b.*\s-(?:enc|encodedcommand)\b", low):
        yield (
            "inline-interpreter-encoded-payload",
            "command invokes PowerShell with an encoded command payload",
        )


def screen_spawn_command(
    command: str,
    args: Optional[Sequence[str]] = None,
    *,
    raise_on_match: bool = False,
) -> Optional[SpawnGuardVerdict]:
    """Screen an MCP server spawn config for dangerous command shapes.

    Returns the first matching :class:`SpawnGuardVerdict`, or ``None`` when the
    config is ordinary. With ``raise_on_match=True`` a match raises
    :class:`SpawnGuardError` instead (for call sites that gate on an exception).
    """
    parts = [command or ""]
    parts.extend(str(a) for a in (args or ()))
    cmdline = " ".join(p for p in parts if p)

    for rule, reason in _rules(cmdline):
        if raise_on_match:
            raise SpawnGuardError(rule, reason)
        return SpawnGuardVerdict(rule, reason)
    return None
