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

Named permission profiles (workspace assistant defaults): the ``profiles``
dict carries any number of additional profiles alongside ``default``.
Every mutator and resolver takes a keyword ``profile_id`` (default
``"default"`` -- byte-identical to the single-profile behavior). Resolvers
walk the named profile's precedence levels first and fall through to the
default profile only for levels the named profile leaves unset, so a fresh
empty named profile (``{"servers": {}}``, no ``global_default`` key)
resolves exactly like the default profile alone. Kill switch and
``schema_version`` stay global; the shape change is additive, so
``SCHEMA_VERSION`` stays 1.

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
#: Risk tags that floor an INHERITED ``allow`` to ``ask`` for in-process
#: built-ins. A superset of ``HIGH_RISK_TAGS``: built-ins additionally
#: treat filesystem reads as prompt-worthy, because an agent reading
#: arbitrary sandbox files is a disclosure risk even though it mutates
#: nothing, and treat network egress as prompt-worthy too, because egress
#: is the exfiltration leg of a prompt-injection chain. MCP deliberately
#: keeps ``HIGH_RISK_TAGS``. TASK-845 asked whether ``network`` should move
#: to the shared set and resolved NO, on evidence rather than preference:
#: an MCP tool's tags are not ours, they are derived from the remote
#: server's own payload -- ``risk_class`` plus a free-form ``capabilities``
#: list, lowercased (``MCP.hub_tool_catalog._extra_tags``). "network" is an
#: ordinary word for a server to list among its capabilities, so widening
#: the shared set would not be the no-op it looks like: it would start
#: prompting on real servers because of a string they chose for unrelated
#: reasons. The built-in set is a vocabulary WE control and can reason
#: about; the shared set is partly server-supplied and should stay narrow.
BUILTIN_HIGH_RISK_TAGS = HIGH_RISK_TAGS | frozenset({"reads", "network"})

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
        raise ValueError(
            f"Invalid permission state {state!r}; expected one of {STORE_STATES}"
        )


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
    """Coerce ``profiles`` / every profile / every profile's ``servers`` to dicts.

    A payload can pass ``load()``'s dict + ``schema_version`` check yet
    still carry ``null`` (or another non-mapping value) for one of these
    nested containers -- e.g. a user hand-editing ``mcp_permissions.json``.
    Every ``MCPPermissionStore`` method assumes they are dicts
    (``_profile()`` in particular ``.setdefault()``s into them), so
    normalizing once here means no store method has to guard against it
    separately. Mutates ``payload`` in place.

    Since named permission profiles (workspace assistant defaults), the
    every-profile ``servers`` coercion covers the named profiles too: the
    default profile keeps its exact legacy treatment (coerced as a whole,
    then its ``servers`` coerced), and each additional profile under
    ``profiles`` that is itself a mapping has its ``servers`` coerced the
    same way. A profile VALUE that is not a mapping is left untouched
    (junk stays junk; the resolvers' ``_as_mapping()`` traversal already
    tolerates it, and no store method reaches it without ``_profile()``'s
    own coercion).

    Args:
        payload: A payload dict that has already passed the dict +
            ``schema_version`` check in ``load()``.

    Returns:
        ``payload``, with ``profiles``, the default profile, and every
        mapping profile's ``servers`` coerced to dicts in place.
    """
    profiles = _as_mapping(payload.get("profiles"))
    payload["profiles"] = profiles
    profile = _as_mapping(profiles.get(_DEFAULT_PROFILE_ID))
    profiles[_DEFAULT_PROFILE_ID] = profile
    profile["servers"] = _as_mapping(profile.get("servers"))
    for named in profiles.values():
        if isinstance(named, Mapping):
            named["servers"] = _as_mapping(named.get("servers"))
    return payload


def _profile_chain(payload: dict[str, Any], profile_id: str) -> list[dict[str, Any]]:
    """Return the profile-resolution chain for ``profile_id``.

    The chain is ``[named, default]`` when ``profile_id`` names a profile
    other than ``"default"`` and that profile exists (a non-mapping or
    absent named profile contributes nothing -- an unknown ``profile_id``
    resolves exactly like the default profile alone, which is what a
    deleted-but-still-referenced workspace profile must degrade to), and
    ``[default]`` otherwise. The default profile is always last: it is
    the inheritance fallback every named profile falls through to.

    Args:
        payload: A permission-store payload dict (raw is fine; every step
            is ``_as_mapping()``-coerced).
        profile_id: The profile being resolved.

    Returns:
        Non-empty list of profile dicts, most specific first.
    """
    profiles = _as_mapping(payload.get("profiles"))
    chain: list[dict[str, Any]] = []
    if profile_id != _DEFAULT_PROFILE_ID:
        named = _as_mapping(profiles.get(profile_id))
        if named:
            chain.append(named)
    chain.append(_as_mapping(profiles.get(_DEFAULT_PROFILE_ID)))
    return chain


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

        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
        ):
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
            logger.warning(
                f"Failed to back up corrupt MCP permission store at '{self.path}': {exc}"
            )

    # -- profile helpers -----------------------------------------------------

    @staticmethod
    def _profile(
        payload: dict[str, Any], profile_id: str = _DEFAULT_PROFILE_ID
    ) -> dict[str, Any]:
        """Return (seeding as needed) the writable dict for one profile.

        For ``"default"`` the seeding is exactly the legacy behavior:
        ``global_default`` and ``servers`` are ``.setdefault()``ed. For a
        named profile only ``servers`` is seeded -- the named profile
        deliberately carries no ``global_default`` key, because at that
        precedence level *absence means inherit* from the default profile
        (see the resolvers' chain walk).

        Args:
            payload: A loaded (hence normalized) payload to mutate.
            profile_id: Profile to fetch; seeded as above when absent.

        Returns:
            The profile's dict, coerced to a dict even when a hand-edited
            payload carried junk at that key.
        """
        profiles = payload.setdefault("profiles", {})
        profile = profiles.setdefault(profile_id, {})
        if not isinstance(profile, dict):
            profile = {}
            profiles[profile_id] = profile
        if profile_id == _DEFAULT_PROFILE_ID:
            profile.setdefault("global_default", DEFAULT_GLOBAL)
        profile.setdefault("servers", {})
        return profile

    def ensure_profile(self, profile_id: str) -> None:
        """Create the named profile if it does not exist.

        Seeded ``{"servers": {}}`` only -- no ``global_default`` key, so
        every unset level inherits from the ``default`` profile. A no-op
        for the default profile (it always exists) and for empty ids
        (not a valid profile reference).

        Args:
            profile_id: Profile to create; existing profiles are left
                untouched.
        """
        if profile_id == _DEFAULT_PROFILE_ID or not profile_id:
            return
        payload = self.load()
        profiles = payload.setdefault("profiles", {})
        profiles.setdefault(profile_id, {"servers": {}})
        self.save(payload)

    def list_profiles(self) -> list[str]:
        """Return every stored profile id, sorted.

        Returns:
            Sorted list of the keys under ``profiles`` (always includes
            ``"default"`` after any ``load()``).
        """
        return sorted(_as_mapping(self.load().get("profiles")).keys())

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

    def set_global_default(
        self, state: str, *, profile_id: str = _DEFAULT_PROFILE_ID
    ) -> None:
        """Persist a profile's global default permission state.

        Args:
            state: One of ``STORE_STATES``.
            profile_id: Profile to write; defaults to the ``default``
                profile (byte-identical to the pre-profiles behavior).

        Raises:
            ValueError: If ``state`` is not one of ``STORE_STATES``.
        """
        _validate_state(state)
        payload = self.load()
        profile = self._profile(payload, profile_id)
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

    def set_server_default(
        self,
        server_key: str,
        state: str | None,
        *,
        profile_id: str = _DEFAULT_PROFILE_ID,
    ) -> None:
        """Set or clear a server-level default permission state.

        Args:
            server_key: Server's stable key (``<source>:<server_id>``).
            state: One of ``STORE_STATES`` to set an explicit default, or
                None to clear it (inherit from the global default). The
                server's entry is pruned entirely once it has neither a
                default nor any tool overrides left.
            profile_id: Profile to write; defaults to the ``default``
                profile (byte-identical to the pre-profiles behavior).

        Raises:
            ValueError: If ``state`` is not None and not one of
                ``STORE_STATES``.
        """
        if state is not None:
            _validate_state(state)

        payload = self.load()
        profile = self._profile(payload, profile_id)
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
        profile_id: str = _DEFAULT_PROFILE_ID,
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
                compare against later. Not required for ``server_key``
                values in ``HASH_FREE_SERVER_KEYS``.
            profile_id: Profile to write; defaults to the ``default``
                profile (byte-identical to the pre-profiles behavior).

        Raises:
            ValueError: If ``state`` is not None and not one of
                ``STORE_STATES``, or if ``state`` is ``"allow"`` without a
                ``definition_hash`` and ``server_key`` is not in
                ``HASH_FREE_SERVER_KEYS``.
        """
        if state is not None:
            _validate_state(state)
            if (
                state == "allow"
                and not definition_hash
                and server_key not in HASH_FREE_SERVER_KEYS
            ):
                raise ValueError("definition_hash is required when state is 'allow'")

        payload = self.load()
        profile = self._profile(payload, profile_id)
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

    def mark_config_changed(
        self,
        server_key: str,
        tool_name: str,
        *,
        profile_id: str = _DEFAULT_PROFILE_ID,
    ) -> bool:
        """Set ``config_changed: true`` on a tool entry.

        Args:
            server_key: Owning server's stable key.
            tool_name: Tool name within that server.
            profile_id: Profile to write; defaults to the ``default``
                profile (byte-identical to the pre-profiles behavior).

        Returns:
            True only on the not-already-set -> set transition (the
            emit-once signal Task 4 uses to append a single audit entry).
            Returns False -- without writing to disk or creating any
            entry -- when the marker is already set, so a resolution pass
            over a tool that is already downgraded does not rewrite the
            store file on every call.
        """
        payload = self.load()
        profile = self._profile(payload, profile_id)
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


#: Permission namespace for the agent runtime's in-process built-in tools.
#: Deliberately NOT ``builtin:tldw_chatbook`` -- that key belongs to the
#: built-in MCP *server* (see ``readiness.BUILTIN_SERVER_KEY``), and sharing
#: it would let one decision govern two different execution paths. No MCP
#: routing label (``local:``/``builtin:``/``server:``) claims ``agent:``.
BUILTIN_TOOL_SERVER_KEY = "agent:builtin"

#: Server keys whose tools carry no meaningful ``definition_hash``, so
#: ``set_tool_state(..., "allow")`` does not require one, and
#: ``resolve_effective_state()`` skips its hash-staleness comparison for
#: them entirely (see the ``tool.server_key not in HASH_FREE_SERVER_KEYS``
#: guard in that function).
#:
#: The hash is a RUG-PULL guard: it detects a *remote* server changing a
#: tool's description/schema after the user trusted it. ``agent:builtin``
#: tools are in-process code shipped with the app -- an attacker who can
#: change them already has code execution, so the check protects nothing,
#: while a stored hash would force a re-prompt on every release that edits
#: a docstring. ``resolve_builtin_state`` correspondingly never reads one.
#:
#: ``builtin:tldw_chatbook`` (RAG-48 part 2) is exempted for the identical
#: reason: it is the built-in MCP *server*'s namespace (see
#: ``readiness.BUILTIN_SERVER_KEY``) -- also in-process code that ships
#: with the app and changes only via an app update, not a live remote
#: connection. Without this exemption, RAG-48 part 1 synthesizing a real
#: ``inputSchema`` for these tools (previously always ``None``) would make
#: every already-stored "allow" decision's definition_hash go stale on the
#: very next resolve, silently downgrading it to "ask" with "Definition
#: changed since you allowed it" -- a rug-pull false positive against the
#: app's own code, not an attacker.
#:
#: Adding a REMOTE namespace here would silently disable the rug-pull guard
#: for it; the contents are pinned by test.
HASH_FREE_SERVER_KEYS = frozenset({BUILTIN_TOOL_SERVER_KEY, "builtin:tldw_chatbook"})

#: Fix Round C (PR-T3 review), Item 2. The subset of ``HASH_FREE_SERVER_KEYS``
#: for which ``resolve_effective_state_by_key()``'s by-key ``allow``
#: exemption (see that function) is actually SAFE. Deliberately narrower
#: than ``HASH_FREE_SERVER_KEYS`` -- ``BUILTIN_TOOL_SERVER_KEY``
#: (``"agent:builtin"``) is a member of that wider set (it is hash-free too,
#: for the rug-pull reasons documented above) but is EXCLUDED here:
#:
#: * It never reaches ``resolve_effective_state_by_key()`` in production.
#:   Agent-runtime built-ins resolve exclusively through
#:   ``resolve_builtin_state()`` (``Agents/builtin_tool_gate.py``); the
#:   Hub's by-key seam (``UnifiedMCPControlPlaneService.gate_tool_test_by_key()``)
#:   is only ever called with a key drawn from the Hub tool catalog
#:   (``mcp_workbench.py``'s ``_last_hub_tools``), which never contains an
#:   ``agent:builtin`` row -- built-in-tool rows are namespaced under
#:   ``BUILTIN_TOOL_SERVER_KEY`` in a SIBLING section the matrix never
#:   threads into ``_last_hub_tools`` (see ``_builtin_permission_matrix_rows()``'s
#:   own docstring, "Constraint 1/5").
#: * If it ever did reach this resolver -- a future caller bug, not a
#:   config a user can produce -- exempting it here would be actively
#:   UNSAFE, not merely redundant. This function has no ``HubTool.tags`` to
#:   floor on, so its only defense against an inherited-allow high-risk
#:   tool is the "any allow collapses to ask" rule it narrows for hash-free
#:   keys. Real ``agent:builtin`` tools ARE floored -- by
#:   ``resolve_builtin_state()``'s own ``BUILTIN_HIGH_RISK_TAGS`` check --
#:   so a stray call landing on this seam for that key and getting the
#:   exemption would silently skip a floor its production resolver
#:   enforces.
#:
#: ``builtin:tldw_chatbook`` stays exempt here because every ``HubTool``
#: the Hub catalog can ever produce for it carries ``tags=()``
#: unconditionally (``hub_tool_catalog.builtin_tools_from_inventory`` never
#: reads a tags field off the manifest) -- there is nothing this seam's
#: collapse could ever floor for that key regardless. That invariant is
#: guarded by a tripwire test
#: (``test_builtin_tools_never_carry_risk_tags_even_when_offered_them`` in
#: ``Tests/MCP/test_hub_tool_catalog.py``) that fails the day it stops
#: being true -- at which point this exemption would need re-examining,
#: not just this comment.
BY_KEY_HASH_FREE_SERVER_KEYS = frozenset({"builtin:tldw_chatbook"})

#: Precedence floor for built-in tools: they inherit ``allow`` rather than
#: the MCP ``global_default``, so changing MCP's posture never starts
#: prompting for calculator/datetime. High-risk tags still floor it to ask.
BUILTIN_DEFAULT_STATE = "allow"


@dataclass(frozen=True)
class GatedToolRef:
    """The minimum a resolver needs to gate one in-process tool.

    Deliberately not ``HubTool``: that type models a *hub* tool (its
    ``source`` enum is ``local|builtin|server``, and its ``stale``/
    ``executable``/tag-cap fields are meaningless here), and borrowing it
    would import MCP's hub model into the tools layer.
    """

    server_key: str
    name: str
    description: str
    input_schema: dict | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class EffectiveToolState:
    """The resolved allow/ask/deny verdict for one tool, plus why.

    Attributes:
        state: One of ``STORE_STATES``.
        origin: Which precedence level produced ``state`` before any
            downgrade -- ``tool_override``, ``server_default``,
            ``global_default``, or (built-in tools only, via
            ``resolve_builtin_state``) ``builtin_default``, the allow
            floor applied when nothing more specific overrides it.
            One SYNTHETIC value exists outside the precedence walk:
            ``gate_error``, constructed by ``MCPWorkbench`` (paired
            unconditionally with ``state="deny"``) when per-tool
            resolution RAISES -- a fail-closed verdict, not a configured
            state, which is why ``ui_label`` renders it "Unknown"
            (task-2870) and every copy surface derives its explanation
            from ``PERMISSION_STATE_UNRESOLVED_CLAUSE`` rather than
            claiming "Off".
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
        # task-2870: `origin == "gate_error"` is the synthesized
        # fail-closed verdict -- the permission RESOLVER raised, not a
        # configured Off (`MCPWorkbench._resolve_test_gate()`/
        # `_effective_for_display()` pair it unconditionally with
        # `state="deny"`). Mapping it to "Off" made every renderer of this
        # property (Permissions matrix State cells, Tools-mode State
        # column, the inspector's permission block) print a confident
        # configuration claim about a state that could not be read -- the
        # contradiction PR #1385's round J removed from the inspector one
        # surface at a time. Owned HERE so every renderer tells the same
        # truth; severity/color is deliberately NOT this property's job
        # (`tool_state_kind()` keeps the fail-closed deny in the "error"
        # bucket -- the blocked EFFECT is real, only the causal label
        # lied).
        if self.origin == "gate_error":
            return "Unknown"
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


def resolve_effective_state(
    payload: dict[str, Any],
    tool: HubTool,
    *,
    profile_id: str = _DEFAULT_PROFILE_ID,
) -> EffectiveToolState:
    """Resolve ``tool``'s effective permission state from ``payload``.

    Precedence: an explicit tool-level entry (``tool_override``) beats the
    owning server's ``default`` (``server_default``), which beats the
    profile's ``global_default`` (``global_default``); absence at each level
    means "inherit from the next level down".

    Named profiles: with ``profile_id`` other than ``"default"``, that
    full precedence walk runs against the named profile FIRST, and the
    default profile is consulted only for levels the named profile leaves
    unset -- i.e. the chain is per-level across ``[named, default]``:
    tool override in named, then server default in named, then
    global_default in named, then the same three levels in default, then
    ``DEFAULT_GLOBAL``. A named profile that sets a server default
    therefore shadows a default-profile tool override for that server,
    and a fresh empty named profile (seeded with no ``global_default``)
    resolves exactly like the default profile alone.

    Two downgrades apply on top of precedence, in order:

    1. Rug-pull guard: an explicit tool-level ``allow`` is downgraded to
       ``ask`` (``config_changed=True``) when the live tool's current
       ``definition_hash`` no longer matches the one stored alongside the
       ``allow``, or when the entry carries a persisted ``config_changed``
       marker -- regardless of whether the hash happens to match again.
       Only a fresh ``set_tool_state`` (Task 1) clears the marker. Skipped
       entirely (never downgrades, ``config_changed`` stays False) when
       ``tool.server_key`` is in ``HASH_FREE_SERVER_KEYS`` -- those
       namespaces store no ``definition_hash`` to begin with (RAG-48 part
       2), so comparing would always "mismatch" a live tool's real schema
       against the stored ``None`` and rug-pull every stored allow the
       first time a schema is attached.
    2. High-risk floor: an *inherited* ``allow`` (origin ``server_default``
       or ``global_default``) is downgraded to ``ask``
       (``risk_floored=True``) when the tool's tags intersect
       ``HIGH_RISK_TAGS``. Explicit tool-level ``allow`` is never floored --
       the operator opted in with full knowledge of the specific tool.

    Both downgrades run after the profile walk, regardless of which
    profile supplied the verdict.

    Every nested container this walks (``profiles``, each profile in the
    chain, ``servers``, a server entry, its ``tools``, a tool entry) is
    coerced via ``_as_mapping()`` before being read, so a hand-edited
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
        profile_id: Profile to resolve against; ``"default"`` (the
            default) is byte-identical to the pre-profiles behavior.

    Returns:
        The resolved ``EffectiveToolState``.
    """
    config_changed = False
    state: str | None = None
    origin = ""

    for profile in _profile_chain(payload, profile_id):
        servers = _as_mapping(profile.get("servers"))
        server_entry = _as_mapping(servers.get(tool.server_key))
        tools = _as_mapping(server_entry.get("tools"))
        tool_entry = tools.get(tool.name)
        if not isinstance(tool_entry, Mapping):
            tool_entry = None

        if tool_entry is not None and tool_entry.get("state") in STORE_STATES:
            origin = "tool_override"
            state = tool_entry["state"]
            if state == "allow" and tool.server_key not in HASH_FREE_SERVER_KEYS:
                current_hash = definition_hash(tool.description, tool.input_schema)
                stale_hash = tool_entry.get("definition_hash") != current_hash
                marked_changed = bool(tool_entry.get("config_changed"))
                if stale_hash or marked_changed:
                    state = "ask"
                    config_changed = True
            break

        server_default = server_entry.get("default")
        if server_default in STORE_STATES:
            origin = "server_default"
            state = server_default
            break

        global_default = profile.get("global_default")
        if global_default in STORE_STATES:
            origin = "global_default"
            state = global_default
            break
        # Nothing at any precedence level in this profile: inherit from
        # the next profile in the chain (named -> default).

    if state is None:
        # Chain exhausted with no valid verdict anywhere -- including the
        # I2 case of a hand-edited `mcp_permissions.json` carrying an
        # invalid `global_default` (e.g. "banana") in every profile that
        # passes `load()`'s corruption check. Fail safe to the same
        # default a missing key already gets, never the raw junk value
        # (which used to `KeyError` out of `ui_label` and panic the app).
        origin = "global_default"
        state = DEFAULT_GLOBAL

    risk_floored = False
    if (
        origin != "tool_override"
        and state == "allow"
        and set(tool.tags) & HIGH_RISK_TAGS
    ):
        state = "ask"
        risk_floored = True

    return EffectiveToolState(
        state=state,
        origin=origin,
        config_changed=config_changed,
        risk_floored=risk_floored,
    )


def resolve_builtin_state(
    payload: dict[str, Any],
    tool: GatedToolRef,
    *,
    profile_id: str = _DEFAULT_PROFILE_ID,
) -> EffectiveToolState:
    """Resolve a built-in tool's effective permission state.

    Mirrors ``resolve_effective_state``'s precedence walk (including its
    named-profile chain: the named profile's tool/server levels shadow the
    default profile's, level by level) with two deliberate differences:

    * The final fallback is ``BUILTIN_DEFAULT_STATE`` (``allow``), not any
      profile's ``global_default`` -- the global level is skipped entirely
      in every profile. Built-ins are in-process code the user already
      installed; inheriting MCP's ``ask`` would prompt on every calculator
      call, and changing MCP's global posture (in any profile) must not
      silently change built-in behavior.
    * No ``definition_hash`` comparison. That guard exists for a REMOTE
      server mutating a tool after you trusted it; for in-process code an
      attacker who can change the tool already has code execution, so it
      buys nothing -- while any release editing a description or schema
      would flip ``config_changed`` and re-prompt every user at upgrade
      time. ``config_changed`` is therefore always False here.

    The high-risk floor: an INHERITED ``allow`` (not an explicit tool
    override) whose tags intersect ``BUILTIN_HIGH_RISK_TAGS`` is
    downgraded to ``ask`` with ``risk_floored=True``. That set is a
    superset of MCP's ``HIGH_RISK_TAGS`` -- built-ins additionally floor
    on ``"reads"`` and ``"network"``.

    Args:
        payload: A loaded permission-store payload (``{}`` is valid and
            resolves everything to the floor).
        tool: The built-in tool reference to resolve.
        profile_id: Profile to resolve against; ``"default"`` (the
            default) is byte-identical to the pre-profiles behavior.

    Returns:
        The resolved ``EffectiveToolState``.
    """
    state: str | None = None
    origin = ""

    for profile in _profile_chain(payload, profile_id):
        servers = _as_mapping(profile.get("servers"))
        server_entry = _as_mapping(servers.get(tool.server_key))
        tools = _as_mapping(server_entry.get("tools"))
        tool_entry = tools.get(tool.name)
        if not isinstance(tool_entry, Mapping):
            tool_entry = None

        if tool_entry is not None and tool_entry.get("state") in STORE_STATES:
            origin = "tool_override"
            state = tool_entry["state"]
            break

        server_default = server_entry.get("default")
        if server_default in STORE_STATES:
            origin = "server_default"
            state = server_default
            break
        # Built-ins never read any profile's global_default (see the
        # docstring): this profile contributes nothing, inherit from the
        # next profile in the chain (named -> default).

    if state is None:
        origin = "builtin_default"
        state = BUILTIN_DEFAULT_STATE

    risk_floored = False
    if (
        origin != "tool_override"
        and state == "allow"
        and set(tool.tags) & BUILTIN_HIGH_RISK_TAGS
    ):
        state = "ask"
        risk_floored = True

    return EffectiveToolState(
        state=state,
        origin=origin,
        config_changed=False,
        risk_floored=risk_floored,
    )


def resolve_effective_state_by_key(
    payload: dict[str, Any],
    server_key: str,
    tool_name: str,
    *,
    profile_id: str = _DEFAULT_PROFILE_ID,
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
    default -> global default, with the same global-default validation
    and the same named-profile chain: the named profile's levels shadow
    the default profile's, level by level) but skips the hash comparison
    entirely:

    - ``deny``/``ask`` verdicts (explicit or inherited) are trustworthy
      without a hash check -- there is nothing to downgrade a deny or ask
      to that would be safer -- so they resolve at full fidelity.
    - Any verdict that resolves to ``allow`` -- an explicit tool-level
      override, or an inherited server/global default -- cannot be
      confirmed fresh without the tool's current description/input_schema
      to hash-compare, so it resolves to ``ask`` instead
      (``config_changed=True``, reusing the rug-pull marker's "review
      before trusting this" UI treatment): safer than silently trusting a
      stale ``allow``. **Except** for ``server_key in
      BY_KEY_HASH_FREE_SERVER_KEYS`` -- today, just
      ``"builtin:tldw_chatbook"``, the built-in MCP server. This is
      DELIBERATELY NARROWER than ``HASH_FREE_SERVER_KEYS``: see that
      constant's own docstring for why ``BUILTIN_TOOL_SERVER_KEY``
      (``"agent:builtin"``, the OTHER hash-free key) is excluded from the
      exemption on THIS path even though it is hash-free too. For the keys
      that remain exempt: they're already exempt from the hash comparison
      in ``resolve_effective_state()`` for the reason documented on
      ``HASH_FREE_SERVER_KEYS`` -- they're app code, not a remote server,
      so there is no staleness to guard against -- and the rationale for
      THIS collapse ("cannot be confirmed fresh without a hash to compare")
      is void when there was never going to be a hash comparison in the
      first place. An explicit or inherited ``Allow`` for one of those keys
      resolves at full fidelity, same as ``resolve_effective_state()``
      would with a live ``HubTool``. Without this exemption, the SAME
      hash-free tool set to Allow would record ``decision="approved"``
      (asked-and-approved) when resolved here and ``decision="allowed"``
      when resolved via ``resolve_effective_state()`` -- a cross-surface
      split in the audit trail.

    High-risk-tag flooring has NO coverage on this path for keys in
    ``BY_KEY_HASH_FREE_SERVER_KEYS``. This function has no ``HubTool.tags``
    to check, and -- unlike every other server key -- the "any allow
    downgrades to ask" rule above does not apply to them either, so an
    inherited allow for one of those keys returns at full fidelity with
    NOTHING between it and the caller: no hash check, no tag floor, no
    ask-collapse. This is safe TODAY only because every ``HubTool`` the Hub
    catalog can ever produce for ``"builtin:tldw_chatbook"`` carries
    ``tags=()`` unconditionally (``hub_tool_catalog.builtin_tools_from_inventory``
    never reads a tags field off the manifest) -- there is nothing for a
    floor to ever catch, so the absence of one is a non-event rather than a
    gap. If a future release ever gives a ``builtin:tldw_chatbook`` tool a
    real risk tag, this collapse would start silently returning an
    un-floored ``allow`` where a live-``HubTool`` resolve would ask; a
    tripwire test in ``Tests/MCP/test_hub_tool_catalog.py`` fails the day
    that happens, specifically to surface the regression here rather than
    let it decay behind this comment.

    For every server key NOT in ``BY_KEY_HASH_FREE_SERVER_KEYS``, flooring
    genuinely is redundant to check separately: the "any allow downgrades
    to ask" rule above already collapses every such inherited allow before
    a tag comparison would ever matter, which is the strictly-more-
    conservative relationship the phrase originally described here --
    it just no longer describes ALL server keys.

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
        profile_id: Profile to resolve against; ``"default"`` (the
            default) is byte-identical to the pre-profiles behavior.

    Returns:
        The resolved ``EffectiveToolState``.
    """
    state: str | None = None
    origin = ""

    for profile in _profile_chain(payload, profile_id):
        servers = _as_mapping(profile.get("servers"))
        server_entry = _as_mapping(servers.get(server_key))
        tools = _as_mapping(server_entry.get("tools"))
        tool_entry = tools.get(tool_name)
        if not isinstance(tool_entry, Mapping):
            tool_entry = None

        if tool_entry is not None and tool_entry.get("state") in STORE_STATES:
            origin = "tool_override"
            state = tool_entry["state"]
            break

        server_default = server_entry.get("default")
        if server_default in STORE_STATES:
            origin = "server_default"
            state = server_default
            break

        global_default = profile.get("global_default")
        if global_default in STORE_STATES:
            origin = "global_default"
            state = global_default
            break
        # Nothing at any precedence level in this profile: inherit from
        # the next profile in the chain (named -> default).

    if state is None:
        # Chain exhausted with no valid verdict anywhere -- same I2
        # hand-edited-junk fallback as resolve_effective_state().
        origin = "global_default"
        state = DEFAULT_GLOBAL

    if state == "allow" and server_key not in BY_KEY_HASH_FREE_SERVER_KEYS:
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
