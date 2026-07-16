"""Schema-versioned permission store for the MCP Hub (Permissions mode, Phase 4).

Persists chatbook's client-side tool-permission gate: a global kill switch, a
global default state, and per-server / per-tool overrides, keyed
``<source>:<server_id>`` for servers and ``<tool_name>`` within a server for
tools. See ``Docs/superpowers/specs/2026-07-13-mcp-hub-redesign-design.md``
§9 for the product model (precedence: tool override -> server default ->
global default; absence of a key = "Inherit").

Store shape (spec-verbatim)::

    {
      "schema_version": 1,
      "kill_switch": false,
      "profiles": {
        "default": {
          "global_default": "ask",
          "servers": {
            "<source>:<server_id>": {
              "default": "ask",
              "tools": {
                "<tool_name>": {"state": "allow|ask|deny", "definition_hash": "..."}
              }
            }
          }
        }
      }
    }

Atomic writes mirror ``LocalMCPStore.save()`` (``local_store.py``): a
``.tmp`` sibling file is written first (``json.dump(..., indent=2,
sort_keys=True)``) and then atomically renamed onto the real path via
``Path.replace()``; ``updated_at`` is stamped with an ISO-UTC timestamp on
every save.

Corruption policy — deliberate divergence from ``LocalMCPStore``: local_store
raises ``LocalMCPStoreLoadError`` on an unreadable/corrupt file, forcing the
caller to handle it. Per spec §9 ("unknown schema version -> back up file and
start fresh, never crash"), this store never raises out of ``load()``: a
missing file returns a fresh default payload; a corrupt/non-JSON file, a
JSON payload that is not a dict, or one whose ``schema_version`` does not
match ``SCHEMA_VERSION`` is renamed to ``<name>.bak`` (replacing any prior
backup), a warning is logged, and a fresh default payload is returned. The
original path stays absent until the next ``save()``.

This module intentionally implements only the storage primitive (Task 1 of
the Phase 4 plan). Effective-state resolution (precedence walk, rug-pull
definition-hash comparison, cycle-safety) is Task 2 and lives in this same
module as pure functions layered on top of ``MCPPermissionStore`` — nothing
here should need to change to accommodate them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

SCHEMA_VERSION = 1
STORE_STATES: tuple[str, ...] = ("allow", "ask", "deny")
DEFAULT_GLOBAL = "ask"

_DEFAULT_PROFILE_ID = "default"


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fresh_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": False,
        "profiles": {
            _DEFAULT_PROFILE_ID: {
                "global_default": DEFAULT_GLOBAL,
                "servers": {},
            }
        },
    }


def _validate_state(state: str) -> None:
    if state not in STORE_STATES:
        raise ValueError(f"Invalid permission state {state!r}; expected one of {STORE_STATES}")


def _entry_is_empty(entry: dict[str, Any]) -> bool:
    return not entry.get("default") and not entry.get("tools")


class MCPPermissionStore:
    """Read-modify-write accessor over the on-disk permission-store JSON file.

    Single-instance usage is assumed (the Hub UI); across concurrent
    instances, last write wins — every mutator reloads the full payload,
    applies its change, and saves the full payload back.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # -- raw load/save -----------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Return the full store payload, always valid.

        Missing file -> fresh default payload. Corrupt JSON, JSON that does
        not decode to a dict, or a dict whose ``schema_version`` is not
        ``SCHEMA_VERSION`` -> the existing file is backed up to
        ``<name>.bak`` (replacing any prior backup), a warning is logged,
        and a fresh default payload is returned. Never raises.
        """
        if not self.path.exists():
            return _fresh_payload()

        try:
            raw_text = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                f"MCP permission store at '{self.path}' is unreadable/corrupt ({exc}); "
                "backing it up and resetting to defaults."
            )
            self._backup_corrupt_file()
            return _fresh_payload()

        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                f"MCP permission store at '{self.path}' has an unrecognized shape or "
                f"schema_version (expected {SCHEMA_VERSION}); backing it up and resetting to defaults."
            )
            self._backup_corrupt_file()
            return _fresh_payload()

        return payload

    def save(self, payload: dict[str, Any]) -> None:
        """Atomically write ``payload`` to disk, stamping ``updated_at``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload["updated_at"] = _iso_utc_now()

        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        temp_path.replace(self.path)

    def _backup_corrupt_file(self) -> None:
        backup_path = self.path.with_suffix(f"{self.path.suffix}.bak")
        try:
            self.path.replace(backup_path)
        except OSError as exc:
            logger.warning(f"Failed to back up corrupt MCP permission store at '{self.path}': {exc}")

    # -- profile helpers -----------------------------------------------------

    @staticmethod
    def _profile(payload: dict[str, Any]) -> dict[str, Any]:
        profiles = payload.setdefault("profiles", {})
        profile = profiles.setdefault(_DEFAULT_PROFILE_ID, {})
        profile.setdefault("global_default", DEFAULT_GLOBAL)
        profile.setdefault("servers", {})
        return profile

    # -- kill switch -----------------------------------------------------

    def get_kill_switch(self) -> bool:
        return bool(self.load().get("kill_switch", False))

    def set_kill_switch(self, value: bool) -> None:
        payload = self.load()
        payload["kill_switch"] = bool(value)
        self.save(payload)

    # -- global default -----------------------------------------------------

    def get_global_default(self) -> str:
        return self._profile(self.load()).get("global_default", DEFAULT_GLOBAL)

    def set_global_default(self, state: str) -> None:
        _validate_state(state)
        payload = self.load()
        profile = self._profile(payload)
        profile["global_default"] = state
        self.save(payload)

    # -- server default -----------------------------------------------------

    def get_server_entry(self, server_key: str) -> dict[str, Any] | None:
        servers = self._profile(self.load()).get("servers", {})
        return servers.get(server_key)

    def set_server_default(self, server_key: str, state: str | None) -> None:
        if state is not None:
            _validate_state(state)

        payload = self.load()
        profile = self._profile(payload)
        servers = profile.setdefault("servers", {})

        if state is None:
            entry = servers.get(server_key)
            if entry is not None:
                entry.pop("default", None)
                if _entry_is_empty(entry):
                    servers.pop(server_key, None)
        else:
            entry = servers.setdefault(server_key, {})
            entry["default"] = state

        self.save(payload)

    # -- tool state -----------------------------------------------------

    def get_tool_entry(self, server_key: str, tool_name: str) -> dict[str, Any] | None:
        servers = self._profile(self.load()).get("servers", {})
        entry = servers.get(server_key, {})
        tools = entry.get("tools", {})
        return tools.get(tool_name)

    def set_tool_state(
        self,
        server_key: str,
        tool_name: str,
        state: str | None,
        *,
        definition_hash: str | None = None,
    ) -> None:
        if state is not None:
            _validate_state(state)
            if state == "allow" and not definition_hash:
                raise ValueError("definition_hash is required when state is 'allow'")

        payload = self.load()
        profile = self._profile(payload)
        servers = profile.setdefault("servers", {})

        if state is None:
            entry = servers.get(server_key)
            if entry is not None:
                tools = entry.get("tools", {})
                tools.pop(tool_name, None)
                if not tools:
                    entry.pop("tools", None)
                if _entry_is_empty(entry):
                    servers.pop(server_key, None)
        else:
            entry = servers.setdefault(server_key, {})
            tools = entry.setdefault("tools", {})
            # Replacing the entry wholesale (rather than mutating in place)
            # is what clears any persisted `config_changed` marker.
            tool_entry: dict[str, Any] = {"state": state}
            if state == "allow":
                tool_entry["definition_hash"] = definition_hash
            tools[tool_name] = tool_entry

        self.save(payload)

    def mark_config_changed(self, server_key: str, tool_name: str) -> bool:
        """Set ``config_changed: true`` on a tool entry.

        Returns True only on the not-already-set -> set transition (the
        emit-once signal Task 4 uses to append a single audit entry).
        """
        payload = self.load()
        profile = self._profile(payload)
        servers = profile.setdefault("servers", {})
        entry = servers.setdefault(server_key, {})
        tools = entry.setdefault("tools", {})
        tool_entry = tools.setdefault(tool_name, {})

        already_set = bool(tool_entry.get("config_changed"))
        tool_entry["config_changed"] = True
        self.save(payload)
        return not already_set
