# tldw_chatbook/MCP/mcp_import.py
"""Parse Claude-Desktop-style {"mcpServers": ...} JSON into local profile payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

_PLACEHOLDER_RE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$")


@dataclass
class ImportCandidate:
    profile_id: str
    command: str
    args: list[str] = field(default_factory=list)
    env_placeholders: dict[str, str] = field(default_factory=dict)
    env_literals: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Return the exact save-payload keys the local store accepts."""
        return {
            "profile_id": self.profile_id,
            "command": self.command,
            "args": list(self.args),
            "env_placeholders": dict(self.env_placeholders),
            "env_literals": dict(self.env_literals),
        }


def _literal_is_storable(profile_id: str, command: str, key: str, value: str) -> bool:
    """Authoritative check: round-trip the store's own validation."""
    try:
        LocalExternalMCPProfile.from_input_dict({
            "profile_id": profile_id or "candidate", "command": command or "cmd",
            "env_literals": {key: value},
        })
    except ValueError:
        return False
    return True


def parse_mcp_servers_json(
    text: str, *, existing_ids: set[str] = frozenset()
) -> list[ImportCandidate]:
    """Parse mcpServers JSON into import candidates with per-entry warnings.

    Args:
        text: Raw JSON text ({"mcpServers": {name: {command, args?, env?}}}).
        existing_ids: Profile ids already present (overwrite warnings).

    Returns:
        One candidate per server entry, in file order.

    Raises:
        ValueError: Invalid JSON, missing/empty "mcpServers", or a non-dict entry.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON: {exc}") from None
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise ValueError('Expected a top-level "mcpServers" object with at least one entry.')
    candidates: list[ImportCandidate] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise ValueError(f'Entry "{name}" must be an object.')
        candidate = ImportCandidate(
            profile_id=str(name).strip(),
            command=str(entry.get("command") or "").strip(),
            args=[str(a) for a in entry.get("args") or []],
        )
        for key, raw_value in (entry.get("env") or {}).items():
            value = str(raw_value)
            if _PLACEHOLDER_RE.match(value.strip()):
                candidate.env_placeholders[str(key)] = value.strip()
            elif _literal_is_storable(candidate.profile_id, candidate.command, str(key), value):
                candidate.env_literals[str(key)] = value
            else:
                candidate.env_placeholders[str(key)] = f"${key}"
                candidate.warnings.append(
                    f"{key}: value can't be stored — will reference ${key}; "
                    "export it before connecting."
                )
        if candidate.profile_id in existing_ids:
            candidate.warnings.append(
                f"{candidate.profile_id}: will overwrite the existing profile.")
        candidates.append(candidate)
    return candidates
