# Built-in Tool Permission Gate Implementation Plan (P1 of TASK-545)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate the agent runtime's built-in tools through the existing MCP permission machinery — risk tags, an allow-floor resolver, dual enforcement, session-scoped approvals, and a global kill switch — without porting any tools or touching config.

**Architecture:** A new pure resolver (`resolve_builtin_state`) lives beside MCP's in `permission_store.py`. A new impure seam (`Agents/builtin_tool_gate.py`) adapts a built-in `Tool` into that resolver and owns per-turn caching/stamping. `BuiltinToolProvider` consults the gate on every `invoke` (defense-in-depth); a run-level review hook resolves the turn's calls once and stamps verdicts (primary path, good UX). The pure `agent_runtime` is untouched.

**Tech Stack:** Python 3.11, dataclasses, pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-25-builtin-tool-permission-gate-design.md` (commits `ff351e0ce`, `e324f9375`). Read it for rationale; THIS plan carries the code.

## Global Constraints

1. **Namespace:** built-in permissions use `server_key = "agent:builtin"`. **Never** `builtin:tldw_chatbook` — that is a live key (`MCP/readiness.py:263` `BUILTIN_SERVER_KEY`) for the built-in MCP *server*, and sharing it would put two execution paths under one permission identity.
2. **Do not modify `resolve_effective_state`.** Add `resolve_builtin_state` as a sibling. MCP's existing behavior and tests must be untouched.
3. **No definition-hash comparison** for built-ins; **no persistent** allow/deny written under `agent:builtin` (session scope only).
4. **Precedence:** tool override → server default → built-in floor `"allow"`, then the existing risk-flooring pass (inherited allow + `HIGH_RISK_TAGS` → `ask`, `risk_floored=True`).
5. **Fail closed, never raise.** Every refusal returns `ToolResult(ok=False, error=...)`. `run_agent_loop` is pure and must never see an exception from tool invocation.
6. **`gate=None` means "build the real gate lazily", never "ungated".** A bare `BuiltinToolProvider()` must be gated.
7. **Missing service** (`getattr(app, "unified_mcp_service", None)` is `None`) → resolve against an **empty payload**: untagged tools run, `"mutates"` tools reach `ask` and fail closed. Never allow-everything.
8. **One store load per turn.** `MCPPermissionStore.load()` re-reads JSON with no cache (`permission_store.py:162-183`).
9. `risk_tags` is a **concrete** property defaulting to `()` — abstract would break `DateTimeTool`, `CalculatorTool`, `RAGSearchTool`, `CodeAuditTool`, and the file/note tools.
10. Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-builtin-gate` (branch `feat/builtin-tool-permission-gate`); tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` FROM the worktree. Never touch the main checkout. `git add` only each task's listed files, never `-A`.
11. **Line numbers are as-of `origin/dev` `d8364963b` — re-verify with `grep -n` before editing. The target TEXT is authoritative.**

**Baseline:** before Task 1, run `pytest Tests/Agents/ Tests/MCP/ -q` and record pre-existing failures — report, don't fix.

---

### Task 1: `risk_tags` on the `Tool` ABC

**Files:** Modify `tldw_chatbook/Tools/tool_executor.py`; Test `Tests/Tools/test_tool_executor.py`

**Interfaces:** Produces `Tool.risk_tags -> tuple[str, ...]`, default `()`.

- [ ] **Step 1: Write the failing test**

```python
def test_tool_risk_tags_defaults_empty_and_is_concrete():
    """Every existing Tool subclass must keep working without declaring tags."""
    from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

    assert CalculatorTool().risk_tags == ()
    assert DateTimeTool().risk_tags == ()


def test_tool_subclass_may_declare_risk_tags():
    from tldw_chatbook.Tools.tool_executor import Tool

    class Mutating(Tool):
        @property
        def name(self) -> str:
            return "mutating"

        @property
        def description(self) -> str:
            return "d"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        @property
        def risk_tags(self) -> tuple[str, ...]:
            return ("mutates",)

        async def execute(self, **kwargs):
            return {}

    assert Mutating().risk_tags == ("mutates",)
```

- [ ] **Step 2: Run — verify fail.** `pytest Tests/Tools/test_tool_executor.py -q -k risk_tags` → AttributeError.

- [ ] **Step 3: Implement.** In `Tools/tool_executor.py`, add to `class Tool(ABC)` after the `parameters` property (keep it **concrete**, not `@abstractmethod`):

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Risk classes for the permission gate, e.g. ``("mutates",)``.

        Concrete with an empty default so every existing subclass keeps
        working unchanged. The vocabulary is the permission store's
        ``HIGH_RISK_TAGS`` (``mutates``/``process``) -- a tool tagged with
        one of those has an INHERITED ``allow`` floored to ``ask`` by
        ``resolve_builtin_state``. Read-only tools leave this empty.
        """
        return ()
```

- [ ] **Step 4: Run — verify pass.** `pytest Tests/Tools/ -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Tools/tool_executor.py Tests/Tools/test_tool_executor.py
git commit -m "feat(tools): add concrete risk_tags to the Tool ABC [TASK-545]"
```

---

### Task 2: `GatedToolRef` + `resolve_builtin_state`

**Files:** Modify `tldw_chatbook/MCP/permission_store.py`; Test `Tests/MCP/test_permission_store.py`

**Interfaces:**
- Produces `GatedToolRef(server_key, name, description, input_schema, tags)` (frozen dataclass) and `resolve_builtin_state(payload: dict, tool: GatedToolRef) -> EffectiveToolState`.
- Consumes existing `STORE_STATES`, `HIGH_RISK_TAGS`, `_as_mapping`, `_DEFAULT_PROFILE_ID`, `EffectiveToolState`.

- [ ] **Step 1: Write the failing tests** — append to `Tests/MCP/test_permission_store.py` (read the file first for its existing payload-building helpers and reuse them):

```python
from tldw_chatbook.MCP.permission_store import (
    GatedToolRef,
    resolve_builtin_state,
)

BUILTIN_KEY = "agent:builtin"


def _ref(name="calculator", tags=()):
    return GatedToolRef(
        server_key=BUILTIN_KEY,
        name=name,
        description="d",
        input_schema={"type": "object"},
        tags=tuple(tags),
    )


def _payload(*, global_default=None, server_default=None, tool_state=None):
    server: dict = {}
    if server_default is not None:
        server["default"] = server_default
    if tool_state is not None:
        server["tools"] = {"calculator": {"state": tool_state}}
    profile: dict = {"servers": {BUILTIN_KEY: server} if server else {}}
    if global_default is not None:
        profile["global_default"] = global_default
    return {"profiles": {"default": profile}}


def test_builtin_floor_is_allow_not_the_mcp_global_default():
    # MCP's global default is "ask"; built-ins must NOT inherit it, or
    # calculator would prompt on every use.
    eff = resolve_builtin_state(_payload(global_default="ask"), _ref())
    assert eff.state == "allow"
    assert eff.risk_floored is False


def test_empty_payload_resolves_to_allow_floor():
    eff = resolve_builtin_state({}, _ref())
    assert eff.state == "allow"


def test_high_risk_tag_floors_inherited_allow_to_ask():
    eff = resolve_builtin_state({}, _ref(tags=("mutates",)))
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_explicit_tool_override_allow_is_not_floored():
    eff = resolve_builtin_state(
        _payload(tool_state="allow"), _ref(tags=("mutates",))
    )
    assert eff.state == "allow"
    assert eff.origin == "tool_override"
    assert eff.risk_floored is False


def test_server_default_beats_the_builtin_floor():
    eff = resolve_builtin_state(_payload(server_default="deny"), _ref())
    assert eff.state == "deny"
    assert eff.origin == "server_default"


def test_no_hash_comparison_for_builtins():
    # An allow override with a STALE/absent definition_hash must stay
    # allow -- the rug-pull guard is deliberately not applied to
    # in-process code (it would re-prompt on every app upgrade).
    payload = _payload(tool_state="allow")
    entry = payload["profiles"]["default"]["servers"][BUILTIN_KEY]["tools"]
    entry["calculator"]["definition_hash"] = "stale-and-wrong"
    eff = resolve_builtin_state(payload, _ref())
    assert eff.state == "allow"
    assert eff.config_changed is False


def test_builtin_and_mcp_builtin_server_namespaces_are_disjoint():
    """A decision for the built-in MCP SERVER must not govern the
    agent-runtime tool of the same name, or vice versa."""
    from tldw_chatbook.MCP.readiness import BUILTIN_SERVER_KEY

    assert BUILTIN_SERVER_KEY != BUILTIN_TOOL_SERVER_KEY
    # A deny recorded against the MCP built-in server leaves the
    # agent-runtime tool of the same name on its own floor.
    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_SERVER_KEY: {"tools": {"calculator": {"state": "deny"}}}
                }
            }
        }
    }
    assert resolve_builtin_state(payload, _ref("calculator")).state == "allow"


def test_mcp_resolver_is_unaffected_by_the_builtin_floor():
    # Guard the "do not modify resolve_effective_state" constraint: an
    # MCP tool with no entries still inherits the MCP global default.
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=(),
        stale=False, executable=True,
    )
    assert resolve_effective_state({}, tool).state == "ask"
```

- [ ] **Step 2: Run — verify fail.** `pytest Tests/MCP/test_permission_store.py -q -k "builtin or floor or hash"` → ImportError.

- [ ] **Step 3: Implement.** In `MCP/permission_store.py`, add near `EffectiveToolState`:

```python
#: Permission namespace for the agent runtime's in-process built-in tools.
#: Deliberately NOT ``builtin:tldw_chatbook`` -- that key belongs to the
#: built-in MCP *server* (see ``readiness.BUILTIN_SERVER_KEY``), and sharing
#: it would let one decision govern two different execution paths. No MCP
#: routing label (``local:``/``builtin:``/``server:``) claims ``agent:``.
BUILTIN_TOOL_SERVER_KEY = "agent:builtin"

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
```

Then add the resolver (place it immediately after `resolve_effective_state` so the two walks read side by side):

```python
def resolve_builtin_state(
    payload: dict[str, Any], tool: GatedToolRef
) -> EffectiveToolState:
    """Resolve a built-in tool's effective permission state.

    Mirrors ``resolve_effective_state``'s precedence walk with two
    deliberate differences:

    * The final fallback is ``BUILTIN_DEFAULT_STATE`` (``allow``), not the
      MCP ``global_default``. Built-ins are in-process code the user
      already installed; inheriting MCP's ``ask`` would prompt on every
      calculator call, and changing MCP's global posture must not silently
      change built-in behavior.
    * No ``definition_hash`` comparison. That guard exists for a REMOTE
      server mutating a tool after you trusted it; for in-process code an
      attacker who can change the tool already has code execution, so it
      buys nothing -- while any release editing a description or schema
      would flip ``config_changed`` and re-prompt every user at upgrade
      time. ``config_changed`` is therefore always False here.

    The high-risk floor is unchanged: an INHERITED ``allow`` (not an
    explicit tool override) whose tags intersect ``HIGH_RISK_TAGS`` is
    downgraded to ``ask`` with ``risk_floored=True``.

    Args:
        payload: A loaded permission-store payload (``{}`` is valid and
            resolves everything to the floor).
        tool: The built-in tool reference to resolve.

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

    if tool_entry is not None and tool_entry.get("state") in STORE_STATES:
        origin = "tool_override"
        state = tool_entry["state"]
    else:
        server_default = server_entry.get("default")
        if server_default in STORE_STATES:
            origin = "server_default"
            state = server_default
        else:
            origin = "builtin_default"
            state = BUILTIN_DEFAULT_STATE

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
        config_changed=False,
        risk_floored=risk_floored,
    )
```

- [ ] **Step 3b: Give the new origin a sentence.** In `UI/MCP_Modules/mcp_inspector.py`, add a `"builtin_default"` entry to `_ORIGIN_SENTENCES` (find it with `grep -n "_ORIGIN_SENTENCES" `), matching the existing entries' phrasing — e.g. `"Built-in tools default to allow."`. Existing consumers already fall back safely (`.get(..., _UNKNOWN_ORIGIN_SENTENCE)`), so this is polish, not a fix.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/MCP/ -q` (the whole MCP suite — Constraint 2 requires MCP behavior to be unchanged).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/MCP/permission_store.py tldw_chatbook/UI/MCP_Modules/mcp_inspector.py Tests/MCP/test_permission_store.py
git commit -m "feat(mcp): add resolve_builtin_state + GatedToolRef for in-process tools [TASK-545]"
```

---

### Task 3: The built-in gate seam

**Files:** Create `tldw_chatbook/Agents/builtin_tool_gate.py`; Test `Tests/Agents/test_builtin_tool_gate.py`

**Interfaces:**
- Consumes Task 1's `Tool.risk_tags`, Task 2's `GatedToolRef`/`resolve_builtin_state`/`BUILTIN_TOOL_SERVER_KEY`.
- Produces `BuiltinToolGate` with `begin_turn()`, `resolve(tool) -> EffectiveToolState`, `stamp(name, decision)`, `check(tool) -> str | None` (None = permitted, str = refusal reason), and `build_builtin_gate(service=None) -> BuiltinToolGate`.

- [ ] **Step 1: Write the failing tests** in a new `Tests/Agents/test_builtin_tool_gate.py`:

```python
import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Tools.tool_executor import CalculatorTool, Tool


class _Mutating(Tool):
    @property
    def name(self) -> str:
        return "write_thing"

    @property
    def description(self) -> str:
        return "d"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return ("mutates",)

    async def execute(self, **kwargs):
        return {}


class _FakeService:
    """Stands in for UnifiedMCPControlPlaneService."""

    def __init__(self, payload=None, kill=False, session=()):
        self._payload = payload or {}
        self._kill = kill
        self._session = set(session)
        self.loads = 0
        self.session_approved = []

    def get_kill_switch(self):
        return self._kill

    def _load_payload(self):
        self.loads += 1
        return self._payload

    def is_session_approved(self, server_key, tool_name):
        return tool_name in self._session

    def approve_for_session(self, server_key, tool_name):
        self.session_approved.append(tool_name)
        self._session.add(tool_name)


def test_untagged_tool_is_permitted():
    gate = BuiltinToolGate(_FakeService())
    assert gate.check(CalculatorTool()) is None


def test_mutating_tool_without_approval_fails_closed():
    gate = BuiltinToolGate(_FakeService())
    reason = gate.check(_Mutating())
    assert reason is not None and "approval" in reason.lower()


def test_kill_switch_blocks_even_untagged_tools():
    gate = BuiltinToolGate(_FakeService(kill=True))
    reason = gate.check(CalculatorTool())
    assert reason is not None and "kill switch" in reason.lower()


def test_stamped_approval_permits_a_mutating_tool():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")
    assert gate.check(_Mutating()) is None


def test_begin_turn_clears_prior_stamps():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")
    gate.begin_turn()
    assert gate.check(_Mutating()) is not None


def test_session_approval_permits_without_a_stamp():
    gate = BuiltinToolGate(_FakeService(session={"write_thing"}))
    assert gate.check(_Mutating()) is None


def test_approve_session_records_a_session_approval():
    svc = _FakeService()
    gate = BuiltinToolGate(svc)
    gate.begin_turn()
    gate.stamp("write_thing", "approve_session")
    assert gate.check(_Mutating()) is None
    assert svc.session_approved == ["write_thing"]


def test_no_service_still_gates_mutating_tools():
    # Constraint 7: a missing service is never allow-everything.
    gate = BuiltinToolGate(None)
    assert gate.check(CalculatorTool()) is None
    assert gate.check(_Mutating()) is not None


def test_payload_is_loaded_once_per_turn():
    # Constraint 8: one store load per turn, not per call.
    svc = _FakeService()
    gate = BuiltinToolGate(svc)
    gate.begin_turn()
    for _ in range(5):
        gate.check(CalculatorTool())
    assert svc.loads == 1


def test_no_persistent_state_is_ever_written_for_builtins():
    """Constraint 3: P1 is session-scoped only. `always_allow` must not
    reach `set_tool_state` -- the card does not offer it for built-in
    rows, and the gate must not honor it as a persistent write either."""
    svc = _FakeService()
    svc.set_tool_state = lambda *a, **k: pytest.fail("persistent write")
    gate = BuiltinToolGate(svc)
    gate.begin_turn()
    gate.stamp("write_thing", "always_allow")
    gate.check(_Mutating())          # must not raise via set_tool_state


def test_deny_state_blocks():
    payload = {
        "profiles": {
            "default": {
                "servers": {"agent:builtin": {"default": "deny"}},
            }
        }
    }
    gate = BuiltinToolGate(_FakeService(payload=payload))
    assert gate.check(CalculatorTool()) is not None
```

- [ ] **Step 2: Run — verify fail.** `pytest Tests/Agents/test_builtin_tool_gate.py -q` → ImportError.

- [ ] **Step 3: Implement** `tldw_chatbook/Agents/builtin_tool_gate.py`:

```python
"""Permission gate for the agent runtime's in-process built-in tools.

The impure seam between ``BuiltinToolProvider`` (which must stay
dependency-light) and the MCP permission store. Owns per-turn payload
caching and this turn's approval stamps.

See ``Docs/superpowers/specs/2026-07-25-builtin-tool-permission-gate-design.md``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    EffectiveToolState,
    GatedToolRef,
    resolve_builtin_state,
)
from tldw_chatbook.Tools.tool_executor import Tool

#: Stamp values that permit execution for this turn.
_PERMITTING = {"approve_once", "approve_session", "always_allow"}


def tool_ref(tool: Tool) -> GatedToolRef:
    """Adapt a built-in ``Tool`` into the resolver's reference type."""
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

    def stamp(self, tool_name: str, decision: str) -> None:
        """Record this turn's decision for ``tool_name``."""
        self._stamps[tool_name] = decision
        if decision == "approve_session" and self._service is not None:
            approve = getattr(self._service, "approve_for_session", None)
            if approve is not None:
                try:
                    approve(BUILTIN_TOOL_SERVER_KEY, tool_name)
                except Exception as exc:  # noqa: BLE001 — best effort
                    logger.warning(f"builtin session approval failed: {exc}")

    def _load_payload(self) -> dict:
        # Constraint 8: one load per turn. A missing service resolves
        # against {} -> the allow floor, with risk flooring intact.
        if self._payload is None:
            self._payload = {}
            if self._service is not None:
                loader = getattr(self._service, "_load_payload", None)
                if loader is not None:
                    try:
                        loaded = loader()
                        if isinstance(loaded, dict):
                            self._payload = loaded
                    except Exception as exc:  # noqa: BLE001 — fail to floor
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

    def check(self, tool: Tool) -> str | None:
        """Execution-time verdict.

        Returns:
            ``None`` when the call may proceed, else a human-readable
            refusal reason for a failed ``ToolResult``. Never raises.
        """
        if self._kill_switch():
            return "tool execution is disabled by the kill switch"

        stamp = self._stamps.get(tool.name)
        if stamp == "deny":
            return f"tool call denied by the user: {tool.name}"
        if stamp in _PERMITTING:
            return None

        state = self.resolve(tool)
        if state.state == "allow":
            return None
        if state.state == "deny":
            return f"tool is set to Off: {tool.name}"
        if self._session_approved(tool.name):
            return None
        # "ask" with no stamp and no session approval: fail closed. In P1
        # this is unreachable (nothing is tagged high-risk yet); P2's
        # mutating tools make it live.
        return f"tool requires approval and none was granted: {tool.name}"


def build_builtin_gate(service: Any | None = None) -> BuiltinToolGate:
    """Construct the real gate, discovering the service when not given."""
    if service is None:
        try:
            from textual.app import App

            app = App.app if hasattr(App, "app") else None
            service = getattr(app, "unified_mcp_service", None)
        except Exception:  # noqa: BLE001 — no app context (tests, headless)
            service = None
    return BuiltinToolGate(service)
```

**Note for the implementer:** `_load_payload` on the service may not exist under that name — check `UnifiedMCPControlPlaneService` for how it reaches `MCPPermissionStore.load()` (look for a `store`/`_store` attribute or a `load`-ish method) and use the real accessor, keeping the `getattr` + try/except shape. If the cleanest access is `service._store.load()`, use that and say so in your report. The `_FakeService` in the tests must be updated to match whatever you use.

Likewise `build_builtin_gate`'s app discovery is a sketch — if the codebase has an established way to reach the running app (grep for `unified_mcp_service` consumers), use that instead and simplify. If no clean accessor exists, prefer requiring the caller to pass the service (Task 4 wires it from the bridge, which has app access) and make `build_builtin_gate()` with no service simply return `BuiltinToolGate(None)` — that is fail-closed-correct per Constraint 7.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/Agents/test_builtin_tool_gate.py -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Agents/builtin_tool_gate.py Tests/Agents/test_builtin_tool_gate.py
git commit -m "feat(agents): add BuiltinToolGate seam over the permission store [TASK-545]"
```

---

### Task 4: Enforce the gate in `BuiltinToolProvider`

**Files:** Modify `tldw_chatbook/Agents/tool_catalog.py`; Test `Tests/Agents/test_tool_catalog.py`

**Interfaces:** Consumes Task 3's `BuiltinToolGate`/`build_builtin_gate`. Produces `BuiltinToolProvider(gate=None)` whose `invoke` refuses gated calls, plus `BuiltinToolProvider.tool_for(name: str) -> Tool | None` (Task 6's hook uses it to map a call name to the `Tool` it must resolve).

- [ ] **Step 1: Write the failing tests** — append to `Tests/Agents/test_tool_catalog.py`:

```python
def test_builtin_provider_refuses_when_gate_denies():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class DenyGate:
        def check(self, tool):
            return "nope"

    out = BuiltinToolProvider(gate=DenyGate()).invoke(
        "builtin:calculator", {"expression": "1+1"}
    )
    assert out.ok is False
    assert "nope" in out.error


def test_builtin_provider_runs_when_gate_permits():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class AllowGate:
        def check(self, tool):
            return None

    out = BuiltinToolProvider(gate=AllowGate()).invoke(
        "builtin:calculator", {"expression": "6*7"}
    )
    assert out.ok is True


def test_gate_none_is_not_ungated(monkeypatch):
    # Constraint 6: a bare provider must be gated, not open.
    import tldw_chatbook.Agents.tool_catalog as tc

    class DenyGate:
        def check(self, tool):
            return "denied by default gate"

    monkeypatch.setattr(tc, "build_builtin_gate", lambda: DenyGate())
    out = tc.BuiltinToolProvider().invoke("builtin:calculator", {"expression": "1+1"})
    assert out.ok is False
    assert "denied by default gate" in out.error


def test_gate_failure_does_not_raise_into_the_loop():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class BoomGate:
        def check(self, tool):
            raise RuntimeError("gate exploded")

    out = BuiltinToolProvider(gate=BoomGate()).invoke(
        "builtin:calculator", {"expression": "1+1"}
    )
    assert out.ok is False  # fail closed, never raise
```

- [ ] **Step 2: Run — verify fail.** `pytest Tests/Agents/test_tool_catalog.py -q -k gate` → `TypeError: unexpected keyword 'gate'`.

- [ ] **Step 3: Implement.** In `Agents/tool_catalog.py`, change `BuiltinToolProvider.__init__` and `invoke`. Keep the lazy import inside the function (Constraint: module stays dependency-light):

```python
    def __init__(self, gate: Any | None = None) -> None:
        self._tools = {t.name: t for t in (CalculatorTool(), DateTimeTool())}
        # `None` means "build the real gate on first use" -- NOT "ungated".
        # Every construction site (console_agent_bridge's default registry
        # and its per-run registry) passes nothing today, so an ungated
        # default would silently leave the shipping path unprotected.
        self._gate = gate
```

Add a public name→tool lookup (Task 6's hook needs it to resolve which calls are built-in and to hand the `Tool` to the gate):

```python
    def tool_for(self, name: str) -> Any | None:
        """Return the built-in ``Tool`` registered under ``name``, if any."""
        return self._tools.get(name)
```

Add a resolver for the lazy gate:

```python
    def _resolve_gate(self) -> Any:
        if self._gate is None:
            from tldw_chatbook.Agents.builtin_tool_gate import build_builtin_gate

            self._gate = build_builtin_gate()
        return self._gate
```

And gate `invoke`, immediately after the unknown-tool check and **before** `asyncio.run`:

```python
    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1]
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown builtin tool: {name}")
        # Defense in depth: the run-level review hook is the primary gate
        # (it batches approvals into one card per turn), but a caller that
        # reaches invoke() without going through it must still not execute
        # ungated. A gate that raises fails CLOSED -- never into the pure
        # loop, which must not see exceptions from tool invocation.
        try:
            refusal = self._resolve_gate().check(tool)
        except Exception as exc:  # noqa: BLE001 — fail closed
            return ToolResult(ok=False, error=f"permission check failed: {exc}")
        if refusal is not None:
            return ToolResult(ok=False, error=refusal)
        try:
            ...  # unchanged from here
```

- [ ] **Step 4: Run — verify pass.** `pytest Tests/Agents/ -q` (the whole agents suite — this changes a shared construction site).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_tool_catalog.py
git commit -m "feat(agents): enforce the permission gate in BuiltinToolProvider.invoke [TASK-545]"
```

---

### Task 5: Per-row decision options on the approval card

**Files:** Modify `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, `tldw_chatbook/Agents/mcp_tool_provider.py`, `tldw_chatbook/Chat/console_chat_controller.py`; Test `Tests/UI/test_chat_approval_card.py` (create if absent — check with `ls Tests/UI/ | grep -i approval`)

**Interfaces:** Produces support for an optional `options` key on each call dict passed to `set_batch`. Rows omitting it keep all four options (MCP behavior unchanged).

**The `options` value has to travel three hops**, so all three are in this task:
1. `MCPPendingCall` (`Agents/mcp_tool_provider.py:79-87`) gains `options: tuple[str, ...] = ()` — a defaulted field, so every existing construction site is untouched and MCP rows stay empty (= "all options").
2. `ConsoleChatController.request_mcp_approvals`'s dict conversion (`console_chat_controller.py:822-835`) passes it through: add `"options": list(call.options)` to the per-call dict.
3. `ChatApprovalCard` honors it per row via `_options_for_row` below.

- [ ] **Step 1: Write the failing tests.**

```python
def test_row_without_options_offers_all_four():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _DECISION_OPTIONS,
        _options_for_row,
    )

    assert _options_for_row({}) == _DECISION_OPTIONS


def test_row_options_filter_to_the_requested_subset():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import _options_for_row

    got = _options_for_row({"options": ["approve_once", "approve_session"]})
    assert [value for _label, value in got] == ["approve_once", "approve_session"]


def test_unknown_option_values_are_ignored_not_rendered():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _DECISION_OPTIONS,
        _options_for_row,
    )

    got = _options_for_row({"options": ["approve_once", "teleport"]})
    assert [value for _label, value in got] == ["approve_once"]


def test_empty_options_list_falls_back_to_all():
    # An empty subset would render a Select with no choices -- unusable.
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _DECISION_OPTIONS,
        _options_for_row,
    )

    assert _options_for_row({"options": []}) == _DECISION_OPTIONS
```

- [ ] **Step 2: Run — verify fail.** ImportError on `_options_for_row`.

- [ ] **Step 3: Implement.** Add beside `_DECISION_OPTIONS` in `chat_approval_card.py`:

```python
def _options_for_row(call: "Mapping[str, Any] | dict") -> list[tuple[str, str]]:
    """Decision options for one row, honoring an optional ``options`` key.

    Rows that omit ``options`` (every MCP row) get the full set, so MCP
    behavior is unchanged. A row may narrow it -- built-in tools offer
    only the session-scoped choices in P1, because persistent decisions
    for them cannot yet be undone in the UI. Unknown values are dropped,
    and an empty result falls back to the full set rather than rendering
    an unusable empty ``Select``.
    """
    requested = call.get("options") if isinstance(call, Mapping) else None
    if not isinstance(requested, (list, tuple)) or not requested:
        return _DECISION_OPTIONS
    wanted = set(requested)
    narrowed = [pair for pair in _DECISION_OPTIONS if pair[1] in wanted]
    return narrowed or _DECISION_OPTIONS
```

Then use it where the per-row `Select` is built (find it with `grep -n "_DECISION_OPTIONS" chat_approval_card.py` — it is passed as the `Select`'s options). Replace the module-constant reference with `_options_for_row(call)` for that row. Also ensure the row's default value stays valid: if `_DEFAULT_DECISION` is not among the narrowed options, use the first narrowed option instead.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/UI/test_chat_approval_card.py -q` plus any existing approval-card tests (`grep -rln "ChatApprovalCard" Tests/`).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py Tests/UI/test_chat_approval_card.py
git commit -m "feat(ui): allow per-row decision options on the approval card [TASK-545]"
```

---

### Task 6: Run-level review hook + wiring + kill-switch relabel

**Files:** Modify `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`; Test `Tests/Chat/test_console_agent_bridge.py` or the file that tests `build_mcp_review_hook` (find with `grep -rln "build_mcp_review_hook" Tests/`)

**Interfaces:** Consumes Task 3's gate, Task 4's `tool_for`, and Task 5's `options` key. Produces:

```python
def build_tool_review_hook(
    builtin_gate,          # BuiltinToolGate
    builtin_provider,      # BuiltinToolProvider (for .tool_for)
    mcp_provider,          # MCPToolProvider | None
    request_approvals,     # Callable[[list[dict]], dict[str, str]]
) -> Callable[[list[ToolCall]], dict[str, str]]
```

wired unconditionally.

**This is the riskiest task — it changes `_compose_mcp_provider`'s contract.** Read the spec's §4 before starting.

- [ ] **Step 1: Write the failing tests.** `ToolCall` has `.name`, `.args`, and `.llm_name` (see `Agents/agent_models.py`); construct them the way the existing `build_mcp_review_hook` tests do.

```python
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.MCP.permission_store import EffectiveToolState


class _Gate:
    def __init__(self, state="ask", risk_floored=True):
        self._state = state
        self._floored = risk_floored
        self.turns = 0
        self.stamped = []

    def begin_turn(self):
        self.turns += 1

    def resolve(self, tool):
        return EffectiveToolState(
            state=self._state,
            origin="builtin_default",
            risk_floored=self._floored,
        )

    def stamp(self, name, decision):
        self.stamped.append((name, decision))


class _Provider:
    """Minimal stand-in for BuiltinToolProvider."""

    def __init__(self, tool):
        self._tool = tool

    def tool_for(self, name):
        return self._tool if name == self._tool.name else None


def _call(name):
    # ToolCall is (name, args, call_id) -- there is NO llm_name on it.
    # `llm_name` belongs to MCPPendingCall, the approval-row type; the
    # verdict map the runtime consumes is keyed by the LLM-facing name,
    # which equals ToolCall.name.
    return ToolCall(name=name, args={})


def test_review_hook_gates_builtins_with_no_mcp_provider():
    """The whole point: a user with no MCP servers must still be gated."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    gate = _Gate()
    asked = {}

    def request_approvals(pending):
        asked["pending"] = pending
        return {p["llm_name"]: "approve_once" for p in pending}

    hook = build_tool_review_hook(gate, _Provider(_Mutating()), None,
                                  request_approvals)
    verdicts = hook([_call("write_thing")])

    assert gate.turns == 1                       # begin_turn ran first
    assert gate.stamped == [("write_thing", "approve_once")]
    # Rows are MCPPendingCall dataclasses (what request_mcp_approvals
    # takes), NOT dicts -- the dict conversion happens inside it.
    row = asked["pending"][0]
    assert row.server_key == "agent:builtin"
    assert row.server_label == "Built-in"
    assert row.reason == "risk_floored"
    assert row.options == ("approve_once", "approve_session")
    assert verdicts == {"write_thing": "proceed"}


def test_allow_resolved_builtin_never_prompts():
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    calls = []
    hook = build_tool_review_hook(
        _Gate(state="allow", risk_floored=False),
        _Provider(_Mutating()),
        None,
        lambda pending: calls.append(pending) or {},
    )
    assert hook([_call("write_thing")]) == {}
    assert calls == []                            # no card shown


def test_deny_resolved_builtin_is_not_offered_to_the_user():
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    calls = []
    hook = build_tool_review_hook(
        _Gate(state="deny", risk_floored=False),
        _Provider(_Mutating()),
        None,
        lambda pending: calls.append(pending) or {},
    )
    hook([_call("write_thing")])
    assert calls == []      # a tool that is Off gets no approval card


def test_begin_turn_runs_even_when_approvals_raise():
    """A raising approval path must not leave stale stamps for next turn."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    gate = _Gate()

    def boom(pending):
        raise RuntimeError("ui gone")

    hook = build_tool_review_hook(gate, _Provider(_Mutating()), None, boom)
    with pytest.raises(RuntimeError):
        hook([_call("write_thing")])
    assert gate.turns == 1


def test_unknown_names_are_returned_unreviewed():
    """Skill tools and native spawn are owned by neither gate."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    hook = build_tool_review_hook(_Gate(), _Provider(_Mutating()), None,
                                  lambda pending: {})
    assert hook([_call("some_skill")]) == {}
```

Additionally write `test_mcp_and_builtin_share_one_round_trip`: a turn with one MCP call and one built-in call must invoke `request_approvals` **exactly once** with both rows present. Build the MCP half using the existing `build_mcp_review_hook` tests' fake provider (find it with `grep -rln "build_mcp_review_hook" Tests/`) — reuse that fixture rather than inventing a second one.

- [ ] **Step 2: Run — verify fail.** ImportError on `build_tool_review_hook`.

- [ ] **Step 3a: Add the run-level hook** in `console_chat_controller.py`, beside `build_mcp_review_hook` (keep that function — the new one delegates to its logic for the MCP half rather than duplicating it; if that means extracting the MCP loop into a helper both call, do that instead of copying the body).

The hook must:
1. Call `builtin_gate.begin_turn()` **first**, unconditionally, mirroring `build_mcp_review_hook`'s clear-at-entry discipline (a raising approval path must not leave stale stamps live).
2. Clear MCP stamps as today (`provider.apply_batch_decisions({})`) when a provider exists.
3. For each call, route by owner: MCP-claimed names → the existing `provider.pending_gate_for` path; otherwise, names the run's registry resolves to the built-in provider → `builtin_gate.resolve(tool)`, collecting a pending row when the state is not `allow`; anything else (skills, native spawn) → unreviewed.
4. Built-in pending rows use `server_key="agent:builtin"`, `server_label="Built-in"`, `reason="risk_floored"` when `state.risk_floored`, and `options=["approve_once", "approve_session"]` (Task 5).
5. Make **one** `request_approvals` call with the merged MCP + built-in rows.
6. Apply decisions to both sides: `provider.apply_batch_decisions(...)` for MCP rows, `builtin_gate.stamp(name, decision)` for built-in rows.
7. Return the `{llm_name: "proceed"}` map exactly as today.

A built-in row resolving to `deny` must **not** be offered to the user — it is refused outright (the `invoke` gate will reject it; do not put a card in front of the user for a tool that is Off).

- [ ] **Step 3b: Wire it unconditionally.** In `console_chat_controller.py` around line 3291, `_compose_mcp_provider()` currently returns `(provider, hook)` and the hook is `None` whenever MCP is absent. Change so the hook is always built:

```python
        mcp_provider, _unused = await self._compose_mcp_provider()
        self._mcp_provider = mcp_provider
        builtin_gate = build_builtin_gate(
            getattr(self.app, "unified_mcp_service", None)
        )
        review_hook = build_tool_review_hook(
            builtin_gate, mcp_provider, self.request_mcp_approvals, ...
        )
```

and pass `review_tool_calls=review_hook` at the `run_reply` call (~line 3318). Prefer changing `_compose_mcp_provider` to return only the provider (updating its docstring and its `(None, None)` return paths) over leaving a dead second element — but if that ripples too far, returning the provider alone from a thin wrapper is acceptable; say which you did in your report.

The same `builtin_gate` instance must reach `BuiltinToolProvider` so `invoke`'s stamps match the hook's. Thread it through `run_reply` into `_compose_run_registry_and_allowed`'s `BuiltinToolProvider(...)` construction (`console_agent_bridge.py:756-757`), and the bridge's default registry (`:865`). If threading proves invasive, an acceptable alternative is for the bridge to construct the gate and expose it for the controller to hand to the hook — but the hook and the provider MUST share one instance; two gates would mean stamps written to one and read from the other, silently re-prompting or failing closed.

- [ ] **Step 3b-bis: Pin the approval-timeout ordering.** Built-in approvals reuse `request_mcp_approvals`, so they inherit its `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS` (120s) — already comfortably under `RunBudget.max_tool_call_seconds` (300s). Add a comment at that constant recording **why** the ordering matters: the approval wait happens inside task-327's `_call_with_timeout` wrapper, so an approval timeout at or above the tool-call ceiling would let the wrapper fire first, tell the agent the call failed, and still execute the tool for real on the abandoned thread when the user approves late. Any future change to either value must preserve `approval_timeout < max_tool_call_seconds`.

- [ ] **Step 3c: Relabel the kill switch** in `UI/MCP_Modules/mcp_workbench.py`. It is presented as MCP-only (`:1278` `"MCP kill switch read failed"`, `:1684` save-failed log, `:1689` `echo = f"kill switch → ..."`). Update the user-visible label/echo and any nearby help text so it reads as a global tool kill switch covering built-in tools as well as MCP. Find the switch's label with `grep -n "kill" mcp_workbench.py` and change the display string, not the storage key.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/Chat/ Tests/Agents/ Tests/UI/ -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py <test files>
git commit -m "feat(chat): gate built-in tools via a run-level review hook, wired without MCP [TASK-545]"
```

---

### Task 7: Backlog hygiene

**Files:** Modify `backlog/tasks/task-545 - *.md`; create follow-up task files.

- [ ] **Step 1: Rewrite TASK-545's description and ACs** to match what was built. Its current text describes wiring `ToolExecutor` (System A, the deprecated legacy path) and names `write_file`/`create_note`/`update_note` — none of which exist on the gated path in P1. CLAUDE.md requires updating the AC when deviating, not just the notes. Set the ACs to the spec's list, mark P1's as `- [x]`, and state that P2 (tool porting) and P3 (config/legacy decision) remain.

- [ ] **Step 2: File the follow-ups** named in the spec. **First run an ID sweep** to avoid a collision (this repo has had ten+ collision events):

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-builtin-gate
git fetch -q origin dev
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "
import re, subprocess
out = subprocess.run(['git','ls-tree','-r','--name-only','origin/dev','backlog/'],
                     capture_output=True, text=True).stdout
ids = sorted({int(m.group(1)) for m in re.finditer(r'task-(\d+)', out)})
print('max id on origin/dev:', ids[-1])
"
```
Also scan the working tree (`os.listdir` on `backlog/tasks` + `backlog/drafts`) — `git ls-tree | uniq` misses em-dash filenames. Assign IDs above the max of BOTH.

Three follow-ups:
1. **Built-in tool permissions UI** — surface `agent:builtin` in the workbench (or a Tools settings pane) so persistent allow/deny becomes safe to offer. Blocks P2 offering persistent decisions.
2. **Child-run approval routing** — a sub-agent inherits the parent's allow-list and can call built-in tools, but the review hook is per-run; a child's `ask` currently fails closed. Prerequisite for P2 shipping gated mutating tools.
3. **`get_cli_setting("database", {})` at `Local_Ingestion/local_file_ingestion.py:1148`** — the second instance of TASK-547's flat-section/non-string-default bug.

- [ ] **Step 3: Commit**
```bash
git add backlog/
git commit -m "chore(backlog): rescope TASK-545 to the agent-runtime gate; file P1 follow-ups"
```

---

## Post-Implementation

Run `pytest Tests/Agents/ Tests/MCP/ Tests/Chat/ Tests/UI/ Tests/Tools/ -q` from the worktree, then hand off to the final whole-branch review (opus) and superpowers:finishing-a-development-branch.

**Do not mark TASK-545 Done** — P1 satisfies only its gate ACs. P2 (port tools) and P3 (config + legacy decision) remain.
