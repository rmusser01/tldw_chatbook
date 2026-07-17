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

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from tldw_chatbook.MCP.hub_tool_catalog import HubTool

SCHEMA_VERSION = 1
STORE_STATES: tuple[str, ...] = ("allow", "ask", "deny")
DEFAULT_GLOBAL = "ask"
HIGH_RISK_TAGS = frozenset({"mutates", "process"})

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


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a payload value to a dict, tolerating hand-edited junk.

    Args:
        value: Any value pulled from a permission-store payload -- expected
            to be a dict, but a hand-edited ``mcp_permissions.json`` can
            carry ``null`` or some other JSON type at any nesting level
            instead.

    Returns:
        ``value`` unchanged when it is already a ``Mapping``; an empty
        dict otherwise. Applied at every traversal step in
        ``resolve_effective_state()``/``resolve_effective_state_by_key()``
        (and in ``MCPPermissionStore.load()``'s own normalization) so a
        malformed intermediate degrades to "nothing configured here"
        instead of raising ``AttributeError``.
    """
    return value if isinstance(value, Mapping) else {}


def _normalize_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce ``profiles`` / the default profile / its ``servers`` to dicts.

    A payload can pass ``load()``'s dict + ``schema_version`` check yet
    still carry ``null`` (or another non-mapping value) for one of these
    nested containers -- e.g. a user hand-editing ``mcp_permissions.json``.
    Every ``MCPPermissionStore`` method assumes they are dicts
    (``_profile()`` in particular ``.setdefault()``s into them), so
    normalizing once here means no store method has to guard against it
    separately. Mutates ``payload`` in place.

    Args:
        payload: A payload dict that has already passed the dict +
            ``schema_version`` check in ``load()``.

    Returns:
        ``payload``, with ``profiles``, the default profile, and that
        profile's ``servers`` coerced to dicts in place.
    """
    profiles = _as_mapping(payload.get("profiles"))
    payload["profiles"] = profiles
    profile = _as_mapping(profiles.get(_DEFAULT_PROFILE_ID))
    profiles[_DEFAULT_PROFILE_ID] = profile
    profile["servers"] = _as_mapping(profile.get("servers"))
    return payload


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
        and a fresh default payload is returned. A schema-valid payload
        with a non-mapping ``profiles`` / default profile / ``servers``
        (e.g. hand-edited to ``null``) has those coerced to dicts in place
        rather than being treated as corrupt.

        Returns:
            The payload dict, always shaped so ``profiles["default"]`` and
            its ``servers`` key are dicts. Never raises.
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

        return _normalize_payload_shape(payload)

    def save(self, payload: dict[str, Any]) -> None:
        """Atomically write ``payload`` to disk, stamping ``updated_at``.

        Args:
            payload: Full store payload to persist. Mutated in place to
                add/overwrite ``updated_at`` before it is written.
        """
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
        """Return whether the global kill switch is enabled.

        Returns:
            True when all tool execution should be blocked regardless of
            any other setting; False otherwise (including when the store
            file is missing).
        """
        return bool(self.load().get("kill_switch", False))

    def set_kill_switch(self, value: bool) -> None:
        """Persist the global kill switch.

        Args:
            value: True to block all tool execution; False to re-enable
                normal precedence-based resolution.
        """
        payload = self.load()
        payload["kill_switch"] = bool(value)
        self.save(payload)

    # -- global default -----------------------------------------------------

    def get_global_default(self) -> str:
        """Return the profile's global default permission state.

        Returns:
            One of ``STORE_STATES``, or ``DEFAULT_GLOBAL`` when unset.
        """
        return self._profile(self.load()).get("global_default", DEFAULT_GLOBAL)

    def set_global_default(self, state: str) -> None:
        """Persist the profile's global default permission state.

        Args:
            state: One of ``STORE_STATES``.

        Raises:
            ValueError: If ``state`` is not one of ``STORE_STATES``.
        """
        _validate_state(state)
        payload = self.load()
        profile = self._profile(payload)
        profile["global_default"] = state
        self.save(payload)

    # -- server default -----------------------------------------------------

    def get_server_entry(self, server_key: str) -> dict[str, Any] | None:
        """Return the raw stored entry for a server, if any.

        Args:
            server_key: Server's stable key (``<source>:<server_id>``).

        Returns:
            The server's entry dict (an optional ``"default"`` and/or
            ``"tools"`` key), or None when the server has no entry at all
            (fully "Inherit").
        """
        servers = self._profile(self.load()).get("servers", {})
        return servers.get(server_key)

    def set_server_default(self, server_key: str, state: str | None) -> None:
        """Set or clear a server-level default permission state.

        Args:
            server_key: Server's stable key (``<source>:<server_id>``).
            state: One of ``STORE_STATES`` to set an explicit default, or
                None to clear it (inherit from the global default). The
                server's entry is pruned entirely once it has neither a
                default nor any tool overrides left.

        Raises:
            ValueError: If ``state`` is not None and not one of
                ``STORE_STATES``.
        """
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
        """Return the raw stored entry for one tool, if any.

        Args:
            server_key: Owning server's stable key.
            tool_name: Tool name within that server.

        Returns:
            The tool's entry dict (``"state"`` guaranteed; ``"definition_hash"``
            and ``"config_changed"`` optional), or None when the tool has
            no explicit entry (inherits from the server/global default).
        """
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
        """Set or clear a tool-level permission override.

        Args:
            server_key: Owning server's stable key.
            tool_name: Tool name within that server.
            state: One of ``STORE_STATES`` to set an explicit override, or
                None to clear it (inherit from the server/global default).
                Setting replaces any existing entry wholesale, which is
                what clears a persisted ``config_changed`` marker.
            definition_hash: Required when ``state`` is ``"allow"`` -- the
                tool's current fingerprint (see ``definition_hash()``),
                stored alongside the allow for the rug-pull guard to
                compare against later.

        Raises:
            ValueError: If ``state`` is not None and not one of
                ``STORE_STATES``, or if ``state`` is ``"allow"`` without a
                ``definition_hash``.
        """
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

        Args:
            server_key: Owning server's stable key.
            tool_name: Tool name within that server.

        Returns:
            True only on the not-already-set -> set transition (the
            emit-once signal Task 4 uses to append a single audit entry).
            Returns False -- without writing to disk or creating any
            entry -- when the marker is already set, so a resolution pass
            over a tool that is already downgraded does not rewrite the
            store file on every call.
        """
        payload = self.load()
        profile = self._profile(payload)
        servers = profile.get("servers", {})
        entry = servers.get(server_key, {})
        tools = entry.get("tools", {})
        tool_entry = tools.get(tool_name, {})
        if bool(tool_entry.get("config_changed")):
            return False

        servers = profile.setdefault("servers", {})
        entry = servers.setdefault(server_key, {})
        tools = entry.setdefault("tools", {})
        tool_entry = tools.setdefault(tool_name, {})
        tool_entry["config_changed"] = True
        self.save(payload)
        return True


# -- effective-state resolution (pure; no store I/O) -------------------------
#
# Everything below operates on a plain payload dict (the shape `load()`
# returns) and a `HubTool`. Nothing here reads or writes disk -- callers
# fetch the payload once (e.g. via `MCPPermissionStore.load()`) and resolve
# as many tools against it as they like.


def definition_hash(description: str | None, input_schema: dict | None) -> str:
    """Fingerprint a tool's advertised shape for the rug-pull guard.

    Mirrors ``LocalControlService._approval_fingerprint``'s canonicalization
    (``local_control_service.py``): sorted-key, compact-separator JSON,
    sha256 hex digest.

    Args:
        description: The tool's advertised description, if any.
        input_schema: The tool's advertised JSON input schema, if any.

    Returns:
        A sha256 hex digest fingerprinting ``description``/``input_schema``.
    """
    canonical = json.dumps(
        {"description": description or "", "inputSchema": input_schema or {}},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EffectiveToolState:
    """The resolved allow/ask/deny verdict for one tool, plus why.

    Attributes:
        state: One of ``STORE_STATES``.
        origin: Which precedence level produced ``state`` before any
            downgrade -- ``tool_override``, ``server_default``, or
            ``global_default``.
        config_changed: True when an explicit tool-level ``allow`` was
            downgraded to ``ask`` by the rug-pull guard (hash mismatch
            and/or a persisted ``config_changed`` marker).
        risk_floored: True when an *inherited* ``allow`` was downgraded to
            ``ask`` by the high-risk floor.
    """

    state: str
    origin: str
    config_changed: bool = False
    risk_floored: bool = False

    @property
    def ui_label(self) -> str:
        # I2: second layer of defense against a hand-edited/corrupted
        # `mcp_permissions.json` whose `global_default` is some non-store
        # value (e.g. "banana") -- `resolve_effective_state()` itself now
        # falls back to `DEFAULT_GLOBAL` for that case, but this property
        # must never `KeyError` on an unrecognized `state` regardless of
        # how it got here (a future caller constructing `EffectiveToolState`
        # directly, a store format this code hasn't seen yet, ...): render
        # the raw state capitalized, or "Ask" for a falsy/empty one, rather
        # than raising and panicking whatever worker/render pass called
        # this (mcp_workbench.py's `_sync_children`, in particular).
        known = {"allow": "Allow", "ask": "Ask", "deny": "Off"}
        if self.state in known:
            return known[self.state]
        return self.state.capitalize() if self.state else "Ask"


def resolve_effective_state(payload: dict[str, Any], tool: HubTool) -> EffectiveToolState:
    """Resolve ``tool``'s effective permission state from ``payload``.

    Precedence: an explicit tool-level entry (``tool_override``) beats the
    owning server's ``default`` (``server_default``), which beats the
    profile's ``global_default`` (``global_default``); absence at each level
    means "inherit from the next level down".

    Two downgrades apply on top of precedence, in order:

    1. Rug-pull guard: an explicit tool-level ``allow`` is downgraded to
       ``ask`` (``config_changed=True``) when the live tool's current
       ``definition_hash`` no longer matches the one stored alongside the
       ``allow``, or when the entry carries a persisted ``config_changed``
       marker -- regardless of whether the hash happens to match again.
       Only a fresh ``set_tool_state`` (Task 1) clears the marker.
    2. High-risk floor: an *inherited* ``allow`` (origin ``server_default``
       or ``global_default``) is downgraded to ``ask``
       (``risk_floored=True``) when the tool's tags intersect
       ``HIGH_RISK_TAGS``. Explicit tool-level ``allow`` is never floored --
       the operator opted in with full knowledge of the specific tool.

    Every nested container this walks (``profiles``, the default profile,
    ``servers``, a server entry, its ``tools``, a tool entry) is coerced
    via ``_as_mapping()`` before being read, so a hand-edited
    ``mcp_permissions.json`` with ``null`` (or other non-mapping junk) at
    any of those levels resolves the same as an absent one instead of
    raising ``AttributeError`` -- this function takes a raw payload and
    must be safe standalone, independent of ``MCPPermissionStore.load()``'s
    own normalization.

    Args:
        payload: A permission-store payload dict (the shape ``load()``
            returns, or any raw dict -- this function does not assume it
            has already been normalized).
        tool: The live tool to resolve, used for its ``server_key``,
            ``name``, ``description``/``input_schema`` (hash comparison),
            and ``tags`` (high-risk floor).

    Returns:
        The resolved ``EffectiveToolState``.
    """
    profile = _as_mapping(_as_mapping(payload.get("profiles")).get(_DEFAULT_PROFILE_ID))
    servers = _as_mapping(profile.get("servers"))
    server_entry = _as_mapping(servers.get(tool.server_key))
    tools = _as_mapping(server_entry.get("tools"))
    tool_entry = tools.get(tool.name)
    if not isinstance(tool_entry, Mapping):
        tool_entry = None

    config_changed = False

    if tool_entry is not None and tool_entry.get("state") in STORE_STATES:
        origin = "tool_override"
        state = tool_entry["state"]
        if state == "allow":
            current_hash = definition_hash(tool.description, tool.input_schema)
            stale_hash = tool_entry.get("definition_hash") != current_hash
            marked_changed = bool(tool_entry.get("config_changed"))
            if stale_hash or marked_changed:
                state = "ask"
                config_changed = True
    else:
        server_default = server_entry.get("default")
        if server_default in STORE_STATES:
            origin = "server_default"
            state = server_default
        else:
            origin = "global_default"
            state = profile.get("global_default", DEFAULT_GLOBAL)
            if state not in STORE_STATES:
                # I2: a hand-edited `mcp_permissions.json` can carry a
                # `schema_version` that still passes `load()`'s corruption
                # check (so the file is never backed up/reset) yet an
                # invalid `global_default` value (e.g. "banana"). Passing
                # that straight through used to reach `ui_label`/
                # `format_tool_state_label` as an unrecognized dict key,
                # `KeyError`-ing out of `_sync_children` and panicking the
                # app. Fail safe to the same default a missing key already
                # gets, exactly like the server-level `server_default in
                # STORE_STATES` check just above.
                state = DEFAULT_GLOBAL

    risk_floored = False
    if origin != "tool_override" and state == "allow" and set(tool.tags) & HIGH_RISK_TAGS:
        state = "ask"
        risk_floored = True

    return EffectiveToolState(
        state=state,
        origin=origin,
        config_changed=config_changed,
        risk_floored=risk_floored,
    )


def resolve_effective_state_by_key(
    payload: dict[str, Any], server_key: str, tool_name: str
) -> EffectiveToolState:
    """Resolve ``(server_key, tool_name)``'s effective permission state
    from ``payload`` alone -- no live ``HubTool`` to fingerprint.

    I1: the Test Tool gate (``UnifiedMCPControlPlaneService.gate_tool_test()``)
    needs a live ``HubTool`` to hash-compare against a stored
    ``definition_hash`` (the rug-pull guard). When a tool has dropped out
    of the workbench's catalog snapshot since it was selected -- a stale
    selection, or a resync racing a rug-pull refresh -- there is no
    ``HubTool`` left to gate with, and ``test_hub_tool()``/
    ``execute_external_tool()`` need no ``HubTool`` either (they dispatch
    by ``server_key``/``tool_name`` alone against the live server). Falling
    through ungated in that gap would let a DENIED tool run just because
    it briefly vanished from the snapshot. This resolves the same
    precedence walk as ``resolve_effective_state`` (tool override -> server
    default -> global default, with the same global-default validation)
    but skips the hash comparison entirely:

    - ``deny``/``ask`` verdicts (explicit or inherited) are trustworthy
      without a hash check -- there is nothing to downgrade a deny or ask
      to that would be safer -- so they resolve at full fidelity.
    - Any verdict that resolves to ``allow`` -- an explicit tool-level
      override, or an inherited server/global default -- cannot be
      confirmed fresh without the tool's current description/input_schema
      to hash-compare, so it resolves to ``ask`` instead
      (``config_changed=True``, reusing the rug-pull marker's "review
      before trusting this" UI treatment): safer than silently trusting a
      stale ``allow``.

    High-risk-tag flooring is skipped too (no ``HubTool.tags`` to check);
    that's covered by the "any allow downgrades to ask" rule above, which
    is strictly more conservative.

    Like ``resolve_effective_state()``, every nested container this walks
    is coerced via ``_as_mapping()`` before being read, so a hand-edited
    ``mcp_permissions.json`` with ``null`` (or other non-mapping junk) at
    any level resolves the same as an absent one instead of raising
    ``AttributeError``.

    Args:
        payload: A permission-store payload dict (the shape ``load()``
            returns, or any raw dict -- this function does not assume it
            has already been normalized).
        server_key: Owning server's stable key.
        tool_name: Tool name within that server.

    Returns:
        The resolved ``EffectiveToolState``.
    """
    profile = _as_mapping(_as_mapping(payload.get("profiles")).get(_DEFAULT_PROFILE_ID))
    servers = _as_mapping(profile.get("servers"))
    server_entry = _as_mapping(servers.get(server_key))
    tools = _as_mapping(server_entry.get("tools"))
    tool_entry = tools.get(tool_name)
    if not isinstance(tool_entry, Mapping):
        tool_entry = None

    if tool_entry is not None and tool_entry.get("state") in STORE_STATES:
        origin = "tool_override"
        state = tool_entry["state"]
    else:
        server_default = server_entry.get("default")
        if server_default in STORE_STATES:
            origin = "server_default"
            state = server_default
        else:
            origin = "global_default"
            state = profile.get("global_default", DEFAULT_GLOBAL)
            if state not in STORE_STATES:
                state = DEFAULT_GLOBAL

    if state == "allow":
        return EffectiveToolState(state="ask", origin=origin, config_changed=True)
    return EffectiveToolState(state=state, origin=origin)


_CYCLE_UI_STATES: dict[str | None, str | None] = {
    None: "allow",
    "allow": "ask",
    "ask": "deny",
    "deny": None,
}

_CYCLE_GLOBAL_STATES: dict[str, str] = {
    "allow": "ask",
    "ask": "deny",
    "deny": "allow",
}


def cycle_ui_state(current: str | None) -> str | None:
    """Advance a per-server/per-tool state one Space-press.

    Args:
        current: The current stored state, or None for "Inherit".

    Returns:
        The next state in the cycle: Inherit -> Allow -> Ask -> Off ->
        Inherit (None).
    """
    return _CYCLE_UI_STATES[current]


def cycle_global(current: str) -> str:
    """Advance the global default one Space-press (no Inherit option).

    Args:
        current: The current global default state, one of ``STORE_STATES``.

    Returns:
        The next state in the cycle: Allow -> Ask -> Off -> Allow.
    """
    return _CYCLE_GLOBAL_STATES[current]
