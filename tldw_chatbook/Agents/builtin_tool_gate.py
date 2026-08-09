"""Permission gate for the agent runtime's in-process built-in tools.

The impure seam between ``BuiltinToolProvider`` (which must stay
dependency-light) and the MCP permission store. Owns per-turn payload
caching and this turn's approval stamps.

See ``Docs/superpowers/specs/2026-07-25-builtin-tool-permission-gate-design.md``.
"""

from __future__ import annotations

import contextlib

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    EffectiveToolState,
    GatedToolRef,
    _DEFAULT_PROFILE_ID,
    _as_mapping,
    resolve_builtin_state,
)
from tldw_chatbook.Tools.tool_executor import Tool

#: Stamp values that permit execution for this turn.
_PERMITTING = {"approve_once", "approve_session", "always_allow"}


def tool_ref(tool: Tool) -> GatedToolRef:
    """Adapt a built-in ``Tool`` into the resolver's reference type.

    Args:
        tool: The built-in tool to adapt.

    Returns:
        A ``GatedToolRef`` carrying ``tool``'s name, description, input
        schema, and risk tags under the built-in server key, suitable for
        ``resolve_builtin_state``.
    """
    return GatedToolRef(
        server_key=BUILTIN_TOOL_SERVER_KEY,
        name=tool.name,
        description=tool.description,
        input_schema=tool.parameters,
        tags=tuple(tool.risk_tags),
    )


class BuiltinToolGate:
    """Resolves and enforces allow/ask/deny for built-in tools.

    One instance per run. ``begin_turn()`` clears the previous turn's
    stamps and cached payload; ``resolve()`` reports a state (used by the
    review hook to build the approval card); ``check()`` is the
    execution-time verdict used by ``BuiltinToolProvider.invoke``.
    """

    def __init__(self, service: Any | None) -> None:
        self._service = service
        self._payload: dict | None = None
        self._stamps: dict[str, str] = {}

    def begin_turn(self) -> None:
        """Drop this run's cached payload and the previous turn's stamps.

        Clearing stamps at turn start (not after a round trip) means a
        raising approval path can never leave a stale prior-turn stamp
        live for the next turn to consume -- the same discipline
        ``build_mcp_review_hook`` applies to its own stamps.
        """
        self._payload = None
        self._stamps.clear()

    @contextlib.contextmanager
    def stamp_scope(self) -> Iterator[None]:
        """Snapshot this turn's stamps on enter; RESTORE (not merge) on exit.

        task-628. Wired as part of ``AgentService``'s ``review_state_scope``
        so it wraps every NESTED sub-agent run, mirroring
        ``MCPToolProvider.stamp_scope``.

        ``spawn_subagent`` runs the child's entire loop INLINE and
        synchronously on the parent's own call stack, before the parent's
        remaining same-batch tool calls are dispatched, and the child
        invokes the SAME shared review hook — whose first act is
        ``begin_turn()``, which clears ``_stamps``. Without this scope a
        child wipes the verdicts the parent's user just gave for this turn.

        That is worse for built-ins than the MCP case it mirrors: MCP's
        ``invoke`` has a per-call ``_approval_callback`` fallback, so a lost
        stamp merely re-prompts. ``BuiltinToolGate.check`` has no such
        fallback — its only approval sources are ``_stamps`` (set solely by
        the batch review hook) and a live session approval — so a clobbered
        stamp fails CLOSED outright, making an approved tool unusable from
        inside any sub-agent.

        ``_payload`` is restored too: it is a per-turn cache the child's
        ``begin_turn()`` also drops, and restoring it keeps the parent's
        "one permission-store load per turn" property intact.

        Yields:
            None. On exit the parent's stamps and cached payload are put
            back exactly as they were, discarding whatever the nested run
            recorded — a restore, never a merge.
        """
        stamps = dict(self._stamps)
        payload = self._payload
        try:
            yield
        finally:
            self._stamps = stamps
            self._payload = payload

    def stamp(self, tool_name: str, decision: str) -> None:
        """Record this turn's decision for ``tool_name``.

        ``"always_allow"`` is accepted as a permitting stamp for THIS
        call only -- Constraint 3 (P1 is session-scoped only) means it is
        never persisted via ``set_tool_state``; the built-in approval
        card does not offer that option in the first place, but the gate
        does not trust the caller to have enforced that and simply never
        makes the call that would write it.
        """
        self._stamps[tool_name] = decision
        if decision == "approve_session" and self._service is not None:
            approve = getattr(self._service, "approve_for_session", None)
            if approve is not None:
                try:
                    approve(BUILTIN_TOOL_SERVER_KEY, tool_name)
                except Exception as exc:  # noqa: BLE001 — best effort
                    logger.warning(f"builtin session approval failed: {exc}")

    def _load_payload(self) -> dict:
        # Constraint 8: one load per turn. A missing service, or any
        # failure reaching the store, resolves against {} -- the allow
        # floor, with risk flooring intact.
        #
        # The real accessor: `UnifiedMCPControlPlaneService.permission_store`
        # is a property returning `MCPPermissionStore | None` (there is no
        # `_load_payload`-named method on the service); its `.load()` is
        # what returns the raw payload dict (see
        # `unified_control_plane_service.py`'s `effective_tool_states`,
        # `gate_tool_test`, etc., which all follow this same
        # `store = self.permission_store; ...; store.load()` shape).
        if self._payload is None:
            self._payload = {}
            if self._service is not None:
                try:
                    store = getattr(self._service, "permission_store", None)
                    if store is not None:
                        loaded = store.load()
                        if isinstance(loaded, dict):
                            self._payload = loaded
                except Exception as exc:  # noqa: BLE001 — fail to the floor
                    logger.warning(f"builtin permission load failed: {exc}")
        return self._payload

    def resolve(self, tool: Tool) -> EffectiveToolState:
        """Resolve ``tool``'s effective state (no stamps, no kill switch)."""
        return resolve_builtin_state(self._load_payload(), tool_ref(tool))

    def _kill_switch(self) -> bool:
        if self._service is None:
            return False
        getter = getattr(self._service, "get_kill_switch", None)
        if getter is None:
            return False
        try:
            return bool(getter())
        except Exception as exc:  # noqa: BLE001 — a failed read must not
            # brick tools; the per-tool state still gates them.
            logger.warning(f"kill switch read failed: {exc}")
            return False

    def _session_approved(self, tool_name: str) -> bool:
        if self._service is None:
            return False
        checker = getattr(self._service, "is_session_approved", None)
        if checker is None:
            return False
        try:
            return bool(checker(BUILTIN_TOOL_SERVER_KEY, tool_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"session approval read failed: {exc}")
            return False

    def is_session_approved(self, tool_name: str) -> bool:
        """Public read of whether ``tool_name`` has a live session approval.

        Review finding 1 (task-545/T6): exposed for
        ``console_chat_controller.build_tool_review_hook``, which must
        skip adding a pending row for a tool that is already
        session-approved -- ``resolve()``/``resolve_builtin_state`` read
        the permission store ONLY and never consult session approvals (by
        design, mirroring ``resolve_effective_state``'s own store-only
        contract), so without this read a user who picks "Approve for
        session" would be re-prompted on the very next turn even though
        ``check()`` (via ``_session_approved``, below) already honors it
        at execution time. Thin wrapper over the existing private
        ``_session_approved`` -- kept as two names so ``check()``'s own
        call site never has to change, and so this public one reads as
        the intentional, stable seam a caller outside this class should
        use.
        """
        return self._session_approved(tool_name)

    def check(self, tool: Tool) -> str | None:
        """Execution-time verdict.

        Returns:
            ``None`` when the call may proceed, else a human-readable
            refusal reason for a failed ``ToolResult``. Never raises for
            a well-formed ``Tool`` (a subclass whose ``name``/
            ``description``/``parameters``/``risk_tags`` property itself
            raises would still propagate here); callers still wrap this
            in a try/except and fail closed (see
            ``BuiltinToolProvider.invoke``).
        """
        if self._kill_switch():
            return "tool execution is disabled by the kill switch"

        # An effective `deny` (the user set the tool -- or its server
        # default -- to "Off") is absolute: it must be consulted BEFORE
        # any permitting stamp or session approval, never after. Built-in
        # tools have no catalog filtering (unlike MCP's
        # `compose_catalog`, which never discloses a denied tool to the
        # model in the first place) -- this `check()` is the ONLY thing
        # standing between the model and execution, so a stamp or a live
        # session approval must never be allowed to shadow a resolved
        # deny. `resolve()` is cheap here even on the stamped path:
        # `_load_payload` caches per turn (`begin_turn()` clears it), so
        # this adds no extra I/O within a turn.
        state = self.resolve(tool)
        if state.state == "deny":
            return f"tool is set to Off: {tool.name}"

        stamp = self._stamps.get(tool.name)
        if stamp == "deny":
            return f"tool call denied by the user: {tool.name}"
        if stamp in _PERMITTING:
            return None

        if state.state == "allow":
            return None
        if self._session_approved(tool.name):
            return None
        # "ask" with no stamp and no session approval: fail closed. In P1
        # this is unreachable (nothing is tagged high-risk yet); P2's
        # mutating tools make it live.
        return f"tool requires approval and none was granted: {tool.name}"


def build_builtin_gate(service: Any | None = None) -> BuiltinToolGate:
    """Construct the real gate.

    No app-discovery sketch here: `self.app.unified_mcp_service` (the
    established pattern -- see `console_chat_controller.py`'s
    `_compose_mcp_provider`/`_record_cancelled_approval_decisions`, and
    `mcp_workbench.py`) is only reachable from a Textual widget/screen's
    `.app` property, not from a bare module-level function. There is no
    equivalent global/class-level accessor to fall back on. The caller
    that already has app access (the agent bridge / console controller,
    Task 6) is responsible for passing `service` explicitly; `service=None`
    yields a service-less gate, which is fail-closed-correct per
    Constraint 7 (untagged tools still run, `"mutates"` tools still fail
    closed) rather than "ungated".

    Args:
        service: The control-plane service (typically
            `UnifiedMCPControlPlaneService`) whose `permission_store`,
            `get_kill_switch`, `is_session_approved`, and
            `approve_for_session` the gate reads and calls. `None` builds
            a service-less gate that still gates (fail-closed), not an
            ungated one.

    Returns:
        A `BuiltinToolGate` wired to `service`.
    """
    return BuiltinToolGate(service)


# --- task-627 (P2 Task 2): settings-time enumeration ------------------------
#
# A settings-time enumerator lives beside the runtime gate deliberately, to
# keep one definition of how a built-in tool maps to a `GatedToolRef`
# (`tool_ref`, above) shared between the execution-time gate and the
# permissions UI.


def _stored_builtin_tool_names(payload: dict) -> set[str]:
    """Names with a persisted decision under the built-in tool namespace.

    Args:
        payload: A loaded permission-store payload; tolerates a malformed
            shape (e.g. a hand-edited file) at every nesting level below the
            top-level mapping via ``_as_mapping``.

    Returns:
        The set of tool names with a ``tools`` entry under
        ``profiles.default.servers[BUILTIN_TOOL_SERVER_KEY]``.
    """
    profile = _as_mapping(_as_mapping(payload.get("profiles")).get(_DEFAULT_PROFILE_ID))
    servers = _as_mapping(profile.get("servers"))
    server_entry = _as_mapping(servers.get(BUILTIN_TOOL_SERVER_KEY))
    tools = _as_mapping(server_entry.get("tools"))
    return set(tools.keys())


@dataclass(frozen=True)
class BuiltinPermRow:
    """One built-in tool's row for the permissions UI.

    Attributes:
        name: The tool's LLM-facing name.
        description: One-line description, empty for an orphaned entry.
        effective: State resolved by ``resolve_builtin_state`` -- NEVER by
            the MCP resolver (see the design doc's spike findings).
        orphaned: True when a stored decision exists for a name no live
            built-in tool provides. Such rows must stay listed so the user
            can clear a decision for a tool a later release removed.
    """

    name: str
    description: str
    effective: EffectiveToolState
    orphaned: bool = False


def builtin_permission_rows(payload: dict) -> list[BuiltinPermRow]:
    """Enumerate built-in tools with their effective permission state.

    Settings-time enumeration: constructs a throwaway ``BuiltinToolProvider``
    (cheap -- it builds two Tool objects and its gate is lazy, built only on
    ``invoke()``), so no agent run is started and no gate is created.

    Args:
        payload: A loaded permission-store payload; ``{}`` is valid and
            resolves everything to the built-in allow floor.

    Returns:
        One row per live built-in tool, plus one per stored ``agent:builtin``
        tool entry with no matching live tool (``orphaned=True``), sorted by
        name.
    """
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    provider = BuiltinToolProvider()
    rows: list[BuiltinPermRow] = []
    live: set[str] = set()
    for entry in provider.list_catalog():
        tool = provider.tool_for(entry.name)
        if tool is None:            # defensive: catalog/registry disagree
            continue
        live.add(entry.name)
        rows.append(
            BuiltinPermRow(
                name=entry.name,
                description=entry.one_line_description,
                effective=resolve_builtin_state(payload, tool_ref(tool)),
            )
        )

    for name in _stored_builtin_tool_names(payload) - live:
        rows.append(
            BuiltinPermRow(
                name=name,
                description="",
                effective=resolve_builtin_state(
                    payload,
                    GatedToolRef(
                        server_key=BUILTIN_TOOL_SERVER_KEY,
                        name=name,
                        description="",
                        input_schema=None,
                        tags=(),
                    ),
                ),
                orphaned=True,
            )
        )
    return sorted(rows, key=lambda row: row.name)


# --- task-3240: unified [tools]/[console] REGISTRATION gate enumerator -----
#
# `builtin_permission_rows()` (above) enumerates the PERMISSION layer -- one
# row per LIVE tool a gate-enabled `BuiltinToolProvider` already registered.
# It cannot represent a gate-OFF tool at all (no instance to build a row
# from), which is exactly the layer conflation task-3240's design doc
# rejects: registration and permission are different concerns, and the
# affordance for "is this tool even switched on" belongs beside the other
# [mcp]-source config toggles (Servers mode's built-in detail pane), not
# folded into the Permissions matrix. `ToolGate`/`all_tool_gates()` are that
# enumerator's single source of truth.


@dataclass(frozen=True)
class ToolGate:
    """One ``[tools]``/``[console]`` registration gate, on or off.

    Unlike ``BuiltinPermRow``, this enumerates every KNOWN gate regardless
    of its current state -- the UI needs rows for gates that are OFF, which
    is exactly why it cannot ask a provider (a provider only lists what its
    own gates already permit; ``_GATEABLE_BUILTINS``'s own docstring in
    ``tool_catalog.py`` makes the identical argument for that table).

    Attributes:
        section: The config table the gate's ``key`` lives under -- ``
            "tools"`` for the ``[tools] <x>_enabled`` convention
            (``_GATEABLE_BUILTINS`` and ``web_deep_search``) or ``
            "console"`` for the ``[console] local_tools_enabled`` master
            switch.
        key: The config key within ``section``.
        tool_name: The LLM-facing tool name -- or, for the local group's
            master switch (which gates a GROUP, not a single tool), the
            config key itself, doubling as its label.
        description: One-line description for display.
        enabled: The gate's current state, read through
            ``coerce_bool_setting`` -- never raw truthiness (task-3240's
            Critical prerequisite fixed the identical bug one layer down,
            in ``tool_catalog.py``'s registration read; this enumerator
            must never reintroduce it one layer up).
        group: ``"builtin"`` (the 7 ``_GATEABLE_BUILTINS`` rows) or ``
            "local"`` (the local-workspace-tool group: its master switch
            plus ``web_deep_search``, which shares the group it masters).
    """

    section: str
    key: str
    tool_name: str
    description: str
    enabled: bool
    group: str


#: The local group's master-switch config key, named so it isn't re-typed
#: as a literal at every call/comparison site (the enumerator itself, and
#: the Servers-mode UI's master-vs-dependent branching).
LOCAL_TOOLS_MASTER_KEY = "local_tools_enabled"

#: Hand-written description for the local group's master switch -- it has
#: no corresponding Tool instance to read a description off of (it gates a
#: GROUP of tools, not one), unlike every _GATEABLE_BUILTINS row.
#:
#: task-3240 fix round 1 (Important 2): scoped to the Console/agent path
#: ONLY -- `MCP/local_server_tools.py`'s `build_server_local_provider()`
#: (external MCP-client serving, gated by `[mcp] expose_local_tools`)
#: never reads this key at all, so an enabled `web_deep_search` with this
#: master OFF is still live to external MCP clients. Wording-only fix;
#: making this switch also govern MCP exposure is a separate, out-of-scope
#: behavior change.
_LOCAL_TOOLS_MASTER_DESCRIPTION = (
    "Master switch for the local workspace tool group in the Console/agent "
    "path (fs_*/web_*/todo_* tools plus web_deep_search). Off by default; "
    "the group's individual gates still apply once this is on. Does NOT "
    "govern exposure to external MCP clients -- that's the separate "
    "[mcp] expose_local_tools switch."
)

#: Hand-written description for web_deep_search -- also has no Tool
#: instance (it's a LocalToolSpec, not a Tool ABC subclass; see
#: WEB_DEEP_SEARCH_GATE_KEY's own docstring in local_tool_provider.py).
_WEB_DEEP_SEARCH_DESCRIPTION = (
    "Multi-query web research; costs real money on paid providers. "
    "Requires an app restart to take effect."
)


def all_tool_gates() -> list[ToolGate]:
    """Every ``[tools]``/``[console]`` registration gate, on or off.

    THE single source of truth for task-3240's MCP-hub gate affordance
    (Servers mode's built-in-source detail pane): the 7
    ``_GATEABLE_BUILTINS`` rows (group ``"builtin"``, registration order),
    then the local group (group ``"local"``) -- its master switch
    (``[console] local_tools_enabled``) listed FIRST, since it masters the
    gate listed right after it, then ``web_deep_search``'s own gate.
    ``Tools_Settings_Window``'s separate hand-wiring is untouched and NOT
    extended from here -- that surface stays deprecated (see its module
    docstring); this enumerator is a new, independent consumer of the same
    underlying config keys.

    A builtin row's ``description`` is read off a real, constructed Tool
    instance (mirrors ``Tools_Settings_Window._compose_tool_settings``'s
    own precedent) -- construction failure degrades that one row's
    description rather than the whole enumeration (mirrors
    ``BuiltinToolProvider.__init__``'s own per-entry try/except).

    Returns:
        Nine gates in the order described above.
    """
    from ..config import coerce_bool_setting, get_cli_setting
    from .local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY
    from .tool_catalog import _GATEABLE_BUILTINS, build_gateable_tool

    gates: list[ToolGate] = []
    for entry in _GATEABLE_BUILTINS:
        try:
            description = build_gateable_tool(entry).description
        except Exception as exc:  # noqa: BLE001 — degrade the row, not the enumerator
            logger.opt(exception=True).warning(
                f"Could not describe gateable tool {entry.factory_name}: {exc}"
            )
            description = "Unavailable on this system."
        gates.append(
            ToolGate(
                section="tools",
                key=entry.gate_key,
                tool_name=entry.tool_name,
                description=description,
                enabled=coerce_bool_setting(
                    get_cli_setting("tools", entry.gate_key, False), False
                ),
                group="builtin",
            )
        )

    gates.append(
        ToolGate(
            section="console",
            key=LOCAL_TOOLS_MASTER_KEY,
            tool_name=LOCAL_TOOLS_MASTER_KEY,
            description=_LOCAL_TOOLS_MASTER_DESCRIPTION,
            enabled=coerce_bool_setting(
                get_cli_setting("console", LOCAL_TOOLS_MASTER_KEY, False), False
            ),
            group="local",
        )
    )
    gates.append(
        ToolGate(
            section="tools",
            key=WEB_DEEP_SEARCH_GATE_KEY,
            tool_name="web_deep_search",
            description=_WEB_DEEP_SEARCH_DESCRIPTION,
            enabled=coerce_bool_setting(
                get_cli_setting("tools", WEB_DEEP_SEARCH_GATE_KEY, False), False
            ),
            group="local",
        )
    )
    return gates


def _gate_key_pairs() -> list[tuple[str, str]]:
    """Every gate's (section, key), in enumeration order, WITHOUT
    constructing any Tool — the cheap skeleton `all_tool_gates()` and the
    count path share so the key set can never drift between them."""
    from .local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY
    from .tool_catalog import _GATEABLE_BUILTINS

    pairs: list[tuple[str, str]] = [
        ("tools", entry.gate_key) for entry in _GATEABLE_BUILTINS
    ]
    pairs.append(("console", LOCAL_TOOLS_MASTER_KEY))
    pairs.append(("tools", WEB_DEEP_SEARCH_GATE_KEY))
    return pairs


def count_off_tool_gates() -> int:
    """How many registration gates are currently OFF — coerced config reads
    only, NO Tool construction (Qodo PR #1453: the breadcrumb runs on every
    Permissions-mode resync, and `all_tool_gates()` builds every gateable
    Tool just to read descriptions — needless work plus repeated warning
    logs for optional tools that cannot construct on this system)."""
    from ..config import coerce_bool_setting, get_cli_setting

    return sum(
        1
        for section, key in _gate_key_pairs()
        if not coerce_bool_setting(get_cli_setting(section, key, False), False)
    )


def tool_gate_breadcrumb(gates: list[ToolGate] | None = None) -> str | None:
    """A one-line "N tool gate(s) are off" sentence, or ``None`` if none are.

    Shared text for task-3240's two discoverability breadcrumbs: the
    Permissions matrix's always-visible legend (primary) and
    ``MCPWorkbench._empty_tools_diagnosis()`` (secondary/partial, blind
    whenever any local tool source is non-empty -- see that method's own
    docstring). Both compute this fresh each render; ``gates`` lets a
    caller that already has a batch (e.g. one about to also render
    per-gate rows) reuse it instead of re-enumerating.

    Args:
        gates: A pre-fetched batch, or ``None`` to call ``all_tool_gates()``.

    Returns:
        The sentence, or ``None`` when every gate is on.
    """
    if gates is None:
        # Count-only path: no Tool construction (see count_off_tool_gates).
        off = count_off_tool_gates()
    else:
        off = sum(1 for gate in gates if not gate.enabled)
    if off == 0:
        return None
    return (
        f"{off} tool gate(s) are off — enable them in the built-in "
        "server's detail (Servers mode)."
    )
