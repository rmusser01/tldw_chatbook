# MCP Hub Redesign — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MCP screen's compact-workbench view with a rail + canvas + inspector workbench with a real 4-mode strip (Servers | Tools | Permissions | Audit), driven by a new readiness model, with zero loss of existing capability (legacy action runner survives as the Advanced escape hatch).

**Architecture:** New pure-logic modules (`MCP/readiness.py`, `MCP/redaction.py`) feed new view widgets under `UI/MCP_Modules/` (rail, servers-mode canvas, inspector, workbench assembly). `MCPScreen` hosts mode chips + the new `MCPWorkbench`. The existing `UnifiedMCPControlPlaneService` is consumed read-only through its verified surface: `load_context()`, `load_section()`, `available_actions()`, `runtime_state_override()`, `run_action()`, `select_source()`, `select_server_target()`, `select_scope()`, `select_section()`, and `service.target_store`. `UnifiedMCPPanel` is NOT modified (it and its tests stay green; it is retired in Phase 6).

**Tech Stack:** Python ≥3.11, Textual ≥3.3.0, pytest + pytest-asyncio + Hypothesis. Spec: `Docs/superpowers/specs/2026-07-13-mcp-hub-redesign-design.md`.

## Global Constraints

- Run tests with the venv binary: `.venv/bin/pytest` from the repo root (plain `pytest` may resolve outside the venv; the `timeout` shell command does not exist in this environment).
- ~33 pre-existing UI test failures exist at baseline. Only the test files named in this plan must pass; never "fix" unrelated failures.
- CSS: shared styles are edited in `tldw_chatbook/css/components/_agentic_terminal.tcss`, then rebuilt into `tldw_cli_modular.tcss` via `python3 tldw_chatbook/css/build_css.py`. Widgets also carry minimal `DEFAULT_CSS` geometry so test harnesses (which lack the app stylesheet) render correctly — see `library_screen.py:374-409` for the precedent.
- Design-system contract (`Docs/Design/master-shell-design-system-contract.md`): use `ds-*` classes (`ds-inspector`, `ds-status-badge`, `ds-recovery-callout`, `ds-field-row`, `ds-toolbar`), flat-button vocabulary (`console-action-primary/-secondary/-subdued`), state classes (`.is-active`, `.is-blocked`, `.is-disabled`). No new hex colors in widget `DEFAULT_CSS` beyond neutral geometry.
- Any user/remote-supplied text rendered into `Static`/`Button` labels must use `markup=False` or `rich.markup.escape` (established repo lesson).
- Work in a git worktree off `origin/dev` (concurrent sessions mutate this checkout). Conventional commits.
- Textual widget `id` values must match `[a-zA-Z_-][a-zA-Z0-9_-]*` — server/profile ids are user-supplied, so rail/table rows use index-based ids with a key lookup list, never raw profile ids.

## Verified Facts (planning-time verifications, spec §17 — all resolved)

1. **Chat tool loop** (Phase 5 input): detection, execution, and tool-message persistence EXIST (`worker_events.py:394-436`, `chat_streaming_events.py:197-244`, DB schema v7+ tool role). Gaps: schemas only reach providers via manual JSON paste (`ToolExecutor.get_schemas()` has zero chat-path callers), and continuation is text-injection into `#chat-input` + programmatic send click (`chat_streaming_events.py:470-519`), not a structured tool-result turn.
2. **Async boundary** (Phase 5 input): chat sends are THREAD workers (`chat_events.py:1272-1275`); `MCPClient` methods are async-only; no `run_coroutine_threadsafe` exists anywhere in the repo yet.
3. **Server readiness payload**: per-external-server readiness fields are NOT modeled client-side — `get_external_servers` passes raw dicts through. Client-side normalization is required (Task 2 does it). Server *target* status IS persisted: `ConfiguredServerTarget.last_known_reachability` ∈ {unknown, reachable, unreachable}, `last_known_auth_state` ∈ {unknown, authenticated, auth_required, session_invalid}, `last_connected_at`, `updated_at`.
4. **Built-in server**: started ONLY via `python -m tldw_chatbook.MCP`, stdio-only (HTTP branch raises `NotImplementedError`, no port binding). SPEC CORRECTION: the built-in row is stdio-only; no transport/port controls until HTTP exists.
5. **`DestinationModeStrip`**: static one-row `Horizontal` — mode chips follow the `LibraryScreen` precedent (`LIBRARY_MODES` dict + `Button.library-mode-chip` + `.is-active`, `library_screen.py:168-217, 6295-6357`).

**Local readiness signals available today** (from `LocalMCPControlService.get_external_servers()` items = `profile.to_dict()` + `discovery_snapshot` + `is_connected`): durable = `discovery_snapshot` presence and `env_placeholders` vs `os.environ`; ephemeral = `is_connected`. `LocalExternalMCPProfile` has NO name/transport/enabled/last_error fields; `profile_id` is the stable identifier and display name. A saved snapshot always has ≥1 capability (`_has_capabilities` gating at `local_control_service.py:597`). `load_section("external_servers")` returns a **bare list** for local source and an **envelope dict** (`{"external_servers": [...], ...}`) for server source.

## File Structure

| File | Responsibility |
|---|---|
| Create `tldw_chatbook/MCP/readiness.py` | Readiness enums, priority table, reason→action table, `ReadinessSnapshot`, derivation functions for local profiles / server targets / server external records / built-in server, aggregate summary |
| Create `tldw_chatbook/MCP/redaction.py` | Secret redaction for mappings, CLI arg lists, URLs |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_rail.py` | Left rail: source select, server rows with readiness badges, scope selects (server source) |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py` | Servers-mode canvas: aggregate line + servers table + recovery callouts + server detail |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` | Inspector: readiness block, action buttons, Advanced escape hatch (legacy section browser + action runner) |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` | Assembly: rail + mode canvases (ContentSwitcher) + inspector; service wiring, workers, view state |
| Modify `tldw_chatbook/UI/Screens/mcp_screen.py` | Mode registry + chips in the strip, number-key bindings, hosts `MCPWorkbench`, tolerant save/restore |
| Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ rebuild `tldw_cli_modular.tcss`) | Shared styles for chips, rail, badges, canvas/inspector geometry |
| Tests | `Tests/MCP/test_readiness_model.py`, `Tests/MCP/test_readiness_derivation.py`, `Tests/MCP/test_redaction.py`, `Tests/UI/test_mcp_rail.py`, `Tests/UI/test_mcp_servers_mode.py`, `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py` |

NOT touched in Phase 1: `unified_mcp_panel.py`, `unified_mcp_sections.py` (imported, not modified), `unified_control_plane_service.py`, any store/service. `Tests/UI/test_unified_mcp_panel.py` must remain green.

---

### Task 1: Readiness model (`MCP/readiness.py` — enums, tables, snapshot)

**Files:**
- Create: `tldw_chatbook/MCP/readiness.py`
- Test: `Tests/MCP/test_readiness_model.py`

**Interfaces:**
- Consumes: nothing (pure logic).
- Produces: `ReadinessState`, `ReasonCode`, `HubAction` (str Enums); `REASON_PRIORITY: tuple[ReasonCode, ...]`; `REASON_TO_STATE: dict`; `REASON_TO_ACTIONS: dict`; `READY_ACTIONS: tuple`; `STATE_GLYPHS: dict[ReadinessState, str]`; `STATE_LABELS: dict[ReadinessState, str]`; `ReadinessSnapshot` frozen dataclass with `.primary_reason`, `.allowed_actions`, `.badge_text()`; `resolve_state(reasons) -> ReadinessState`; `aggregate_summary(snapshots) -> str`. Tasks 2-7 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_readiness_model.py
from __future__ import annotations

from tldw_chatbook.MCP.readiness import (
    READY_ACTIONS,
    REASON_PRIORITY,
    REASON_TO_ACTIONS,
    REASON_TO_STATE,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    aggregate_summary,
    resolve_state,
)


def _snap(state: ReadinessState, reasons: tuple[ReasonCode, ...] = ()) -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:demo",
        label="demo",
        source="local",
        state=state,
        reasons=reasons,
        message="",
    )


def test_every_reason_code_has_state_actions_and_priority():
    for code in ReasonCode:
        assert code in REASON_TO_STATE, f"{code} missing display state"
        assert code in REASON_TO_ACTIONS, f"{code} missing action set"
        assert code in REASON_PRIORITY, f"{code} missing from priority order"
    assert len(REASON_PRIORITY) == len(set(REASON_PRIORITY)) == len(list(ReasonCode))


def test_resolve_state_uses_priority_order_not_input_order():
    # discovery_not_run alone -> needs_setup
    assert resolve_state((ReasonCode.DISCOVERY_NOT_RUN,)) is ReadinessState.NEEDS_SETUP
    # auth_missing outranks discovery_not_run regardless of input order
    assert (
        resolve_state((ReasonCode.DISCOVERY_NOT_RUN, ReasonCode.AUTH_MISSING))
        is ReadinessState.NEEDS_SETUP
    )
    assert (
        resolve_state((ReasonCode.NO_TOOLS_RETURNED, ReasonCode.UNREACHABLE))
        is ReadinessState.NEEDS_ATTENTION
    )
    assert resolve_state(()) is ReadinessState.READY


def test_primary_reason_and_allowed_actions_follow_priority():
    snap = _snap(
        ReadinessState.NEEDS_SETUP,
        (ReasonCode.DISCOVERY_NOT_RUN, ReasonCode.AUTH_MISSING),
    )
    assert snap.primary_reason is ReasonCode.AUTH_MISSING
    assert snap.allowed_actions == REASON_TO_ACTIONS[ReasonCode.AUTH_MISSING]


def test_ready_snapshot_gets_ready_actions_and_badge():
    snap = _snap(ReadinessState.READY)
    assert snap.primary_reason is None
    assert snap.allowed_actions == READY_ACTIONS
    assert HubAction.REFRESH_DISCOVERY in snap.allowed_actions
    assert "Ready" in snap.badge_text()


def test_aggregate_summary_counts_states():
    snaps = [
        _snap(ReadinessState.READY),
        _snap(ReadinessState.READY),
        _snap(ReadinessState.NEEDS_SETUP, (ReasonCode.AUTH_MISSING,)),
        _snap(ReadinessState.STALE, (ReasonCode.RUNTIME_UNAVAILABLE,)),
    ]
    summary = aggregate_summary(snaps)
    assert "2 of 4" in summary
    assert "needs setup" in summary
    assert aggregate_summary([]) == "No MCP servers configured yet."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/MCP/test_readiness_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.MCP.readiness'`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/MCP/readiness.py
"""Readiness model for the MCP Hub.

Single source of truth mapping control-plane state onto display states,
priority-ordered reason codes, and the allowed-action set the Hub UI renders.
Adapted from the tldw_server webui readiness model (mcpHubReadiness.ts) to
chatbook's control-plane data shapes. Pure logic: no Textual, no I/O beyond
an injectable environment mapping (added in the derivation half of this
module, Task 2).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReadinessState(str, Enum):
    NEEDS_SETUP = "needs_setup"
    CHECKING = "checking"
    READY = "ready"
    NEEDS_ATTENTION = "needs_attention"
    NO_TOOLS = "no_tools"
    STALE = "stale"


class ReasonCode(str, Enum):
    NOT_CONFIGURED = "not_configured"
    AUTH_MISSING = "auth_missing"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    PREFLIGHT_FAILED = "preflight_failed"
    UNREACHABLE = "unreachable"
    DISCOVERY_FAILED = "discovery_failed"
    CONFIG_CHANGED = "config_changed"
    DISCOVERY_NOT_RUN = "discovery_not_run"
    NO_TOOLS_RETURNED = "no_tools_returned"
    CATALOG_EXPIRED = "catalog_expired"
    PARTIAL_CAPABILITY = "partial_capability"


class HubAction(str, Enum):
    ADD_SERVER = "add_server"
    EDIT_CONFIG = "edit_config"
    OPEN_CREDENTIALS = "open_credentials"
    CONNECT = "connect"
    REFRESH_DISCOVERY = "refresh_discovery"
    VALIDATE = "validate"
    VIEW_DETAILS = "view_details"
    OPEN_TOOL_CATALOG = "open_tool_catalog"
    OPEN_AUDIT = "open_audit"


# First matching code wins the display state, primary message, and action set.
REASON_PRIORITY: tuple[ReasonCode, ...] = (
    ReasonCode.NOT_CONFIGURED,
    ReasonCode.AUTH_MISSING,
    ReasonCode.RUNTIME_UNAVAILABLE,
    ReasonCode.PREFLIGHT_FAILED,
    ReasonCode.UNREACHABLE,
    ReasonCode.DISCOVERY_FAILED,
    ReasonCode.CONFIG_CHANGED,
    ReasonCode.DISCOVERY_NOT_RUN,
    ReasonCode.NO_TOOLS_RETURNED,
    ReasonCode.CATALOG_EXPIRED,
    ReasonCode.PARTIAL_CAPABILITY,
)

REASON_TO_STATE: dict[ReasonCode, ReadinessState] = {
    ReasonCode.NOT_CONFIGURED: ReadinessState.NEEDS_SETUP,
    ReasonCode.AUTH_MISSING: ReadinessState.NEEDS_SETUP,
    # Discovered but not currently connected: usable knowledge, inactive
    # runtime — stale, not broken (disabled != broken contract).
    ReasonCode.RUNTIME_UNAVAILABLE: ReadinessState.STALE,
    ReasonCode.PREFLIGHT_FAILED: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.UNREACHABLE: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.DISCOVERY_FAILED: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.CONFIG_CHANGED: ReadinessState.NEEDS_ATTENTION,
    ReasonCode.DISCOVERY_NOT_RUN: ReadinessState.NEEDS_SETUP,
    ReasonCode.NO_TOOLS_RETURNED: ReadinessState.NO_TOOLS,
    ReasonCode.CATALOG_EXPIRED: ReadinessState.STALE,
    ReasonCode.PARTIAL_CAPABILITY: ReadinessState.STALE,
}

REASON_TO_ACTIONS: dict[ReasonCode, tuple[HubAction, ...]] = {
    ReasonCode.NOT_CONFIGURED: (HubAction.ADD_SERVER,),
    ReasonCode.AUTH_MISSING: (HubAction.OPEN_CREDENTIALS, HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.RUNTIME_UNAVAILABLE: (HubAction.CONNECT, HubAction.VIEW_DETAILS),
    ReasonCode.PREFLIGHT_FAILED: (HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.UNREACHABLE: (HubAction.VALIDATE, HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.DISCOVERY_FAILED: (HubAction.REFRESH_DISCOVERY, HubAction.VIEW_DETAILS),
    ReasonCode.CONFIG_CHANGED: (HubAction.REFRESH_DISCOVERY, HubAction.VIEW_DETAILS),
    ReasonCode.DISCOVERY_NOT_RUN: (HubAction.CONNECT, HubAction.VALIDATE, HubAction.VIEW_DETAILS),
    ReasonCode.NO_TOOLS_RETURNED: (HubAction.REFRESH_DISCOVERY, HubAction.EDIT_CONFIG, HubAction.VIEW_DETAILS),
    ReasonCode.CATALOG_EXPIRED: (HubAction.REFRESH_DISCOVERY, HubAction.VIEW_DETAILS),
    ReasonCode.PARTIAL_CAPABILITY: (HubAction.VIEW_DETAILS, HubAction.OPEN_AUDIT),
}

READY_ACTIONS: tuple[HubAction, ...] = (
    HubAction.OPEN_TOOL_CATALOG,
    HubAction.REFRESH_DISCOVERY,
    HubAction.VIEW_DETAILS,
)

STATE_GLYPHS: dict[ReadinessState, str] = {
    ReadinessState.READY: "●",
    ReadinessState.CHECKING: "◐",
    ReadinessState.NEEDS_SETUP: "○",
    ReadinessState.NEEDS_ATTENTION: "!",
    ReadinessState.NO_TOOLS: "∅",
    ReadinessState.STALE: "◌",
}

STATE_LABELS: dict[ReadinessState, str] = {
    ReadinessState.READY: "Ready",
    ReadinessState.CHECKING: "Checking",
    ReadinessState.NEEDS_SETUP: "Needs setup",
    ReadinessState.NEEDS_ATTENTION: "Needs attention",
    ReadinessState.NO_TOOLS: "No tools",
    ReadinessState.STALE: "Stale",
}


def resolve_state(reasons: tuple[ReasonCode, ...]) -> ReadinessState:
    """Return the display state for a reason set via the priority order."""
    for code in REASON_PRIORITY:
        if code in reasons:
            return REASON_TO_STATE[code]
    return ReadinessState.READY


@dataclass(frozen=True)
class ReadinessSnapshot:
    """One server's readiness as rendered by rail, table, and inspector."""

    server_key: str  # "<source>:<stable_id>", e.g. "local:docs", "server:main"
    label: str
    source: str  # "local" | "server" | "builtin"
    state: ReadinessState
    reasons: tuple[ReasonCode, ...]
    message: str
    tool_count: int | None = None
    resource_count: int | None = None
    prompt_count: int | None = None
    transport: str = "stdio"
    auth_display: str = "none"
    scope_display: str = "—"
    is_connected: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_reason(self) -> ReasonCode | None:
        for code in REASON_PRIORITY:
            if code in self.reasons:
                return code
        return None

    @property
    def allowed_actions(self) -> tuple[HubAction, ...]:
        primary = self.primary_reason
        if primary is None:
            return READY_ACTIONS
        return REASON_TO_ACTIONS[primary]

    def badge_text(self) -> str:
        return f"{STATE_GLYPHS[self.state]} {STATE_LABELS[self.state]}"


def aggregate_summary(snapshots: list[ReadinessSnapshot]) -> str:
    """One-line hub summary, e.g. '2 of 4 servers ready — 1 needs setup, 1 stale.'"""
    if not snapshots:
        return "No MCP servers configured yet."
    total = len(snapshots)
    counts = Counter(snap.state for snap in snapshots)
    ready = counts.get(ReadinessState.READY, 0)
    problems = [
        f"{count} {STATE_LABELS[state].lower()}"
        for state, count in counts.items()
        if state is not ReadinessState.READY and count
    ]
    suffix = f" — {', '.join(problems)}" if problems else ""
    return f"{ready} of {total} servers ready{suffix}."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/MCP/test_readiness_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/readiness.py Tests/MCP/test_readiness_model.py
git commit -m "feat(mcp): add readiness model (states, reason codes, action table)"
```

---

### Task 2: Readiness derivation for local / server / built-in sources

**Files:**
- Modify: `tldw_chatbook/MCP/readiness.py` (append derivation functions)
- Test: `Tests/MCP/test_readiness_derivation.py`

**Interfaces:**
- Consumes: Task 1 names; record shapes verified in "Verified Facts".
- Produces (exact signatures, used by Task 7's workbench):
  - `local_profile_readiness(record: dict[str, Any], *, environ: Mapping[str, str] | None = None) -> ReadinessSnapshot`
  - `server_target_readiness(target: Any) -> ReadinessSnapshot` (accepts `ConfiguredServerTarget` or any object with `server_id`, `label`, `last_known_reachability`, `last_known_auth_state`)
  - `server_external_record_readiness(record: dict[str, Any], *, server_id: str) -> ReadinessSnapshot`
  - `builtin_readiness(*, enabled: bool, expose_tools: bool = True, expose_resources: bool = True, expose_prompts: bool = True) -> ReadinessSnapshot`
  - `env_placeholder_names(env_placeholders: Mapping[str, str]) -> list[str]`
  - `BUILTIN_SERVER_KEY = "builtin:tldw_chatbook"`, `BUILTIN_CLIENT_SNIPPET` (str constant with the `python -m tldw_chatbook.MCP` client-config JSON)

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_readiness_derivation.py
from __future__ import annotations

from tldw_chatbook.MCP.readiness import (
    BUILTIN_SERVER_KEY,
    HubAction,
    ReadinessState,
    ReasonCode,
    builtin_readiness,
    env_placeholder_names,
    local_profile_readiness,
    server_external_record_readiness,
    server_target_readiness,
)


def _local_record(**overrides):
    record = {
        "profile_id": "docs",
        "command": "python",
        "args": ["-m", "demo.server"],
        "env_placeholders": {},
        "env_literals": {},
        "discovery_snapshot": None,
        "is_connected": False,
    }
    record.update(overrides)
    return record


def test_env_placeholder_names_strips_dollar_forms():
    assert env_placeholder_names({"API_KEY": "$MY_KEY", "TOKEN": "${OTHER}"}) == [
        "MY_KEY",
        "OTHER",
    ]


def test_local_profile_never_validated_is_needs_setup_discovery_not_run():
    snap = local_profile_readiness(_local_record(), environ={})
    assert snap.server_key == "local:docs"
    assert snap.state is ReadinessState.NEEDS_SETUP
    assert snap.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert snap.tool_count is None


def test_local_profile_missing_env_var_is_auth_missing():
    record = _local_record(env_placeholders={"API_KEY": "$MISSING_VAR"})
    snap = local_profile_readiness(record, environ={})
    assert snap.primary_reason is ReasonCode.AUTH_MISSING
    assert "MISSING_VAR" in snap.message
    present = local_profile_readiness(record, environ={"MISSING_VAR": "x"})
    assert ReasonCode.AUTH_MISSING not in present.reasons


def test_local_profile_discovered_but_disconnected_is_stale_runtime_unavailable():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}, {"name": "b"}], "resources": [], "prompts": []},
        is_connected=False,
    )
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.STALE
    assert snap.primary_reason is ReasonCode.RUNTIME_UNAVAILABLE
    assert snap.tool_count == 2
    assert HubAction.CONNECT in snap.allowed_actions


def test_local_profile_connected_with_snapshot_is_ready():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=True,
    )
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.READY
    assert snap.reasons == ()
    assert snap.auth_display == "none"


class _Target:
    def __init__(self, reachability, auth_state):
        self.server_id = "main"
        self.label = "Main Server"
        self.auth_mode = "api_key"
        self.last_known_reachability = reachability
        self.last_known_auth_state = auth_state


def test_server_target_states():
    assert (
        server_target_readiness(_Target("reachable", "authenticated")).state
        is ReadinessState.READY
    )
    assert (
        server_target_readiness(_Target("unreachable", "unknown")).primary_reason
        is ReasonCode.UNREACHABLE
    )
    assert (
        server_target_readiness(_Target("reachable", "auth_required")).primary_reason
        is ReasonCode.AUTH_MISSING
    )
    never_probed = server_target_readiness(_Target(None, None))
    assert never_probed.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert never_probed.server_key == "server:main"


def test_server_external_record_passthrough_and_fallback():
    reported = server_external_record_readiness(
        {
            "server_id": "web-search",
            "name": "Web Search",
            "display_state": "needs_attention",
            "reason_codes": ["auth_missing"],
            "tool_count": 3,
            "transport": "http",
        },
        server_id="main",
    )
    assert reported.state is ReadinessState.NEEDS_SETUP  # auth_missing outranks via table
    assert reported.primary_reason is ReasonCode.AUTH_MISSING
    assert reported.tool_count == 3
    assert reported.server_key == "server:main/web-search"

    bare = server_external_record_readiness({"name": "Mystery"}, server_id="main")
    assert bare.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert "not reported" in bare.message.lower()


def test_builtin_readiness():
    on = builtin_readiness(enabled=True)
    assert on.server_key == BUILTIN_SERVER_KEY
    assert on.state is ReadinessState.READY
    assert on.transport == "stdio"
    off = builtin_readiness(enabled=False)
    assert off.state is ReadinessState.NEEDS_SETUP
    assert off.primary_reason is ReasonCode.NOT_CONFIGURED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/MCP/test_readiness_derivation.py -v`
Expected: FAIL with `ImportError: cannot import name 'local_profile_readiness'`

- [ ] **Step 3: Append the derivation functions to `MCP/readiness.py`**

```python
# append to tldw_chatbook/MCP/readiness.py
import os
import re
from collections.abc import Mapping

BUILTIN_SERVER_KEY = "builtin:tldw_chatbook"
BUILTIN_CLIENT_SNIPPET = (
    '{\n'
    '  "mcpServers": {\n'
    '    "tldw_chatbook": {\n'
    '      "command": "python3",\n'
    '      "args": ["-m", "tldw_chatbook.MCP"]\n'
    '    }\n'
    '  }\n'
    '}'
)

_PLACEHOLDER_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def env_placeholder_names(env_placeholders: Mapping[str, str]) -> list[str]:
    """Extract the environment-variable names referenced by $NAME/${NAME} values."""
    names: list[str] = []
    for raw in env_placeholders.values():
        match = _PLACEHOLDER_RE.match(str(raw).strip())
        if match:
            names.append(match.group(1))
    return names


def _snapshot_counts(snapshot: Mapping[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    if not snapshot:
        return None, None, None
    return (
        len(snapshot.get("tools") or []),
        len(snapshot.get("resources") or []),
        len(snapshot.get("prompts") or []),
    )


def local_profile_readiness(
    record: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> ReadinessSnapshot:
    """Derive readiness for one LocalMCPControlService.get_external_servers() item.

    Durable signals: discovery_snapshot presence and env placeholders vs the
    environment. Ephemeral: is_connected. needs_attention/checking/stale-by-age
    require persisted attempt tracking that does not exist yet (Phase 2+).
    """
    env = os.environ if environ is None else environ
    profile_id = str(record.get("profile_id") or "unknown")
    placeholders = dict(record.get("env_placeholders") or {})
    snapshot = record.get("discovery_snapshot")
    is_connected = bool(record.get("is_connected"))

    reasons: list[ReasonCode] = []
    missing = [name for name in env_placeholder_names(placeholders) if name not in env]
    if missing:
        reasons.append(ReasonCode.AUTH_MISSING)
    tool_count, resource_count, prompt_count = _snapshot_counts(snapshot)
    if snapshot is None:
        reasons.append(ReasonCode.DISCOVERY_NOT_RUN)
    else:
        if (tool_count or 0) == 0 and (resource_count or 0) == 0 and (prompt_count or 0) == 0:
            reasons.append(ReasonCode.NO_TOOLS_RETURNED)
        if not is_connected:
            reasons.append(ReasonCode.RUNTIME_UNAVAILABLE)

    reason_tuple = tuple(reasons)
    state = resolve_state(reason_tuple)
    if missing:
        message = f"Missing environment variables: {', '.join(missing)}."
    elif snapshot is None:
        message = "Not validated yet — connect or test to discover tools."
    elif not is_connected:
        message = f"{tool_count or 0} tools discovered; not currently connected."
    else:
        message = f"Connected — {tool_count or 0} tools available."

    return ReadinessSnapshot(
        server_key=f"local:{profile_id}",
        label=profile_id,
        source="local",
        state=state,
        reasons=reason_tuple,
        message=message,
        tool_count=tool_count,
        resource_count=resource_count,
        prompt_count=prompt_count,
        transport="stdio",
        auth_display=f"env ({len(placeholders)})" if placeholders else "none",
        scope_display="Personal",
        is_connected=is_connected,
        detail={
            "command": record.get("command"),
            "args": list(record.get("args") or []),
            "env_placeholders": placeholders,
            "missing_env": missing,
            "discovery_snapshot": snapshot,
        },
    )


def server_target_readiness(target: Any) -> ReadinessSnapshot:
    """Derive readiness for a configured tldw_server target (connection level)."""
    server_id = str(getattr(target, "server_id", "unknown"))
    label = str(getattr(target, "label", server_id))
    reachability = getattr(target, "last_known_reachability", None)
    auth_state = getattr(target, "last_known_auth_state", None)

    reasons: list[ReasonCode] = []
    if reachability == "unreachable":
        reasons.append(ReasonCode.UNREACHABLE)
    if auth_state in ("auth_required", "session_invalid"):
        reasons.append(ReasonCode.AUTH_MISSING)
    if reachability in (None, "unknown") and not reasons:
        reasons.append(ReasonCode.DISCOVERY_NOT_RUN)

    reason_tuple = tuple(reasons)
    state = resolve_state(reason_tuple)
    if state is ReadinessState.READY:
        message = "Reachable and authenticated."
    elif ReasonCode.UNREACHABLE in reason_tuple:
        message = "Server unreachable at last check."
    elif ReasonCode.AUTH_MISSING in reason_tuple:
        message = "Authentication required — sign in to this server."
    else:
        message = "Not checked yet — open the server to probe it."

    return ReadinessSnapshot(
        server_key=f"server:{server_id}",
        label=label,
        source="server",
        state=state,
        reasons=reason_tuple,
        message=message,
        transport="http",
        auth_display=str(getattr(target, "auth_mode", "api_key")),
        scope_display="—",
        detail={"base_url": getattr(target, "base_url", None)},
    )


_VALID_REASON_VALUES = {code.value for code in ReasonCode}


def server_external_record_readiness(record: dict[str, Any], *, server_id: str) -> ReadinessSnapshot:
    """Normalize a raw /mcp/hub external-server record (pass-through dict).

    Readiness fields are backend-owned and not guaranteed; unknown or missing
    vocabulary degrades to discovery_not_run with an honest message rather
    than inventing a state.
    """
    external_id = str(record.get("server_id") or record.get("id") or record.get("name") or "unknown")
    label = str(record.get("name") or external_id)
    raw_reasons = record.get("reason_codes") or []
    reasons = tuple(
        ReasonCode(value) for value in raw_reasons if isinstance(value, str) and value in _VALID_REASON_VALUES
    )
    reported = bool(reasons) or isinstance(record.get("display_state"), str)
    if not reasons and not reported:
        reasons = (ReasonCode.DISCOVERY_NOT_RUN,)
        message = "Readiness not reported by the server — validate to check."
    else:
        message = str(record.get("status_message") or "Reported by server.")

    tool_count = record.get("tool_count")
    if tool_count is None and isinstance(record.get("tools"), list):
        tool_count = len(record["tools"])

    return ReadinessSnapshot(
        server_key=f"server:{server_id}/{external_id}",
        label=label,
        source="server",
        state=resolve_state(reasons),
        reasons=reasons,
        message=message,
        tool_count=tool_count if isinstance(tool_count, int) else None,
        transport=str(record.get("transport") or "stdio"),
        auth_display=str(record.get("credential_state") or "—"),
        scope_display=str(record.get("owner_scope_type") or "—"),
        detail={"raw": record},
    )


def builtin_readiness(
    *,
    enabled: bool,
    expose_tools: bool = True,
    expose_resources: bool = True,
    expose_prompts: bool = True,
) -> ReadinessSnapshot:
    """Readiness for chatbook's own MCP server (stdio-only; started by clients
    via `python -m tldw_chatbook.MCP`, never in-process)."""
    if enabled:
        reasons: tuple[ReasonCode, ...] = ()
        message = "Served over stdio when an MCP client launches chatbook."
    else:
        reasons = (ReasonCode.NOT_CONFIGURED,)
        message = "Disabled in config ([mcp].enabled = false)."
    return ReadinessSnapshot(
        server_key=BUILTIN_SERVER_KEY,
        label="tldw_chatbook (built-in)",
        source="builtin",
        state=resolve_state(reasons),
        reasons=reasons,
        message=message,
        transport="stdio",
        auth_display="none",
        scope_display="—",
        detail={
            "expose_tools": expose_tools,
            "expose_resources": expose_resources,
            "expose_prompts": expose_prompts,
            "client_snippet": BUILTIN_CLIENT_SNIPPET,
        },
    )
```

- [ ] **Step 4: Run both readiness test files**

Run: `.venv/bin/pytest Tests/MCP/test_readiness_model.py Tests/MCP/test_readiness_derivation.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/readiness.py Tests/MCP/test_readiness_derivation.py
git commit -m "feat(mcp): derive readiness from local profiles, server targets, and built-in server"
```

---

### Task 3: Redaction utility

**Files:**
- Create: `tldw_chatbook/MCP/redaction.py`
- Test: `Tests/MCP/test_redaction.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `REDACTED = "***"`; `is_secret_key(key: str) -> bool`; `redact_mapping(data: Mapping) -> dict` (recursive); `redact_args(args: Sequence[str]) -> list[str]`; `redact_url(url: str) -> str`. Used by Tasks 4-6 anywhere config/env/args render.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_redaction.py
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tldw_chatbook.MCP.redaction import (
    REDACTED,
    is_secret_key,
    redact_args,
    redact_mapping,
    redact_url,
)


def test_is_secret_key_matches_common_forms():
    for key in ("api_key", "API-KEY", "Authorization", "token", "client_secret", "PASSWORD"):
        assert is_secret_key(key), key
    for key in ("command", "name", "url", "working_dir"):
        assert not is_secret_key(key), key


def test_redact_mapping_recurses_and_preserves_safe_values():
    data = {"command": "python", "env": {"API_KEY": "sk-123", "PATH": "/usr/bin"}}
    redacted = redact_mapping(data)
    assert redacted["command"] == "python"
    assert redacted["env"]["API_KEY"] == REDACTED
    assert redacted["env"]["PATH"] == "/usr/bin"
    assert data["env"]["API_KEY"] == "sk-123"  # input not mutated


def test_redact_args_handles_flag_and_inline_forms():
    args = ["--api-key", "sk-123", "--verbose", "token=abc", "plain"]
    assert redact_args(args) == ["--api-key", REDACTED, "--verbose", f"token={REDACTED}", "plain"]


def test_redact_url_strips_secret_query_values():
    url = "https://api.example.com/v1?api_key=sk-123&page=2"
    redacted = redact_url(url)
    assert "sk-123" not in redacted
    assert "page=2" in redacted


@given(st.dictionaries(st.text(min_size=1, max_size=20), st.text(max_size=40), max_size=8))
def test_redact_mapping_never_leaks_values_under_secret_keys(data):
    redacted = redact_mapping(data)
    for key, value in data.items():
        if is_secret_key(key) and value:
            assert redacted[key] == REDACTED
        else:
            assert redacted[key] == value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/MCP/test_redaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.MCP.redaction'`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/MCP/redaction.py
"""Secret redaction applied at every MCP display and log boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "***"

_SECRET_KEY_RE = re.compile(
    r"(?i)(token|secret|passwd|password|api[-_]?key|authorization|bearer|credential)"
)
_INLINE_ARG_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+)=(?P<value>.+)$")


def is_secret_key(key: str) -> bool:
    """Whether a mapping key / arg name looks like it holds a secret."""
    return bool(_SECRET_KEY_RE.search(str(key)))


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with values under secret-looking keys replaced."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            result[key] = redact_mapping(value)
        elif is_secret_key(key) and value not in (None, ""):
            result[key] = REDACTED
        else:
            result[key] = value
    return result


def redact_args(args: Sequence[str]) -> list[str]:
    """Redact CLI arg values: `--api-key VALUE` pairs and `key=value` forms."""
    redacted: list[str] = []
    previous_was_secret_flag = False
    for arg in args:
        text = str(arg)
        if previous_was_secret_flag:
            redacted.append(REDACTED)
            previous_was_secret_flag = False
            continue
        inline = _INLINE_ARG_RE.match(text)
        if inline and is_secret_key(inline.group("key")):
            redacted.append(f"{inline.group('key')}={REDACTED}")
            continue
        redacted.append(text)
        if text.startswith("-") and is_secret_key(text.lstrip("-")):
            previous_was_secret_flag = True
    return redacted


def redact_url(url: str) -> str:
    """Redact secret-named query parameter values in a URL."""
    parts = urlsplit(str(url))
    if not parts.query:
        return str(url)
    query = [
        (key, REDACTED if is_secret_key(key) else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/MCP/test_redaction.py -v`
Expected: all PASS (including the Hypothesis property)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/redaction.py Tests/MCP/test_redaction.py
git commit -m "feat(mcp): add shared secret-redaction utility"
```

---

### Task 4: MCP rail widget

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_rail.py`
- Test: `Tests/UI/test_mcp_rail.py`

**Interfaces:**
- Consumes: `ReadinessSnapshot`, `STATE_GLYPHS` from Task 1.
- Produces: `MCPRail(Vertical)` with:
  - `__init__(self, *, source: str, snapshots: list[ReadinessSnapshot], selected_server_key: str | None, scope_options: list[tuple[str, str]], scope_value: str, scope_ref_options: list[tuple[str, str]], scope_ref_value: str | None, **kwargs)`
  - `sync_state(...)` same keyword args → `refresh(recompose=True)`
  - Messages: `MCPRail.SourceChanged(source: str)`, `MCPRail.ServerSelected(server_key: str | None)` (None = "All servers"), `MCPRail.ScopeChanged(scope: str, scope_ref: str | None)`
  - Constant `MCP_RAIL_ROW_PREFIX = "mcp-rail-row-"` (index-based ids; `server_key` carried on a lookup list, never in widget ids)

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_mcp_rail.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

from tldw_chatbook.MCP.readiness import ReadinessSnapshot, ReadinessState
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCP_RAIL_ROW_PREFIX, MCPRail


def _snap(key: str, label: str, state: ReadinessState = ReadinessState.READY) -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key=key, label=label, source=key.split(":", 1)[0],
        state=state, reasons=(), message="",
    )


class RailApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPRail(
            source="local",
            snapshots=[
                _snap("builtin:tldw_chatbook", "tldw_chatbook (built-in)"),
                _snap("local:docs", "docs", ReadinessState.NEEDS_SETUP),
            ],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
            id="mcp-rail",
        )

    def on_mcp_rail_server_selected(self, event: MCPRail.ServerSelected) -> None:
        self.events.append(event)

    def on_mcp_rail_source_changed(self, event: MCPRail.SourceChanged) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_rail_renders_all_servers_row_plus_one_row_per_snapshot():
    app = RailApp()
    async with app.run_test() as pilot:
        rows = list(app.query(f"Button.mcp-rail-row"))
        # "All servers" + builtin + docs
        assert len(rows) == 3
        labels = [str(row.label) for row in rows]
        assert any("All servers" in label for label in labels)
        assert any("docs" in label for label in labels)


@pytest.mark.asyncio
async def test_rail_row_click_posts_server_selected_with_key():
    app = RailApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # index 2 = "local:docs"
        await pilot.pause()
        selected = [e for e in app.events if isinstance(e, MCPRail.ServerSelected)]
        assert selected and selected[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_rail_source_select_posts_source_changed():
    app = RailApp()
    async with app.run_test() as pilot:
        select = app.query_one("#mcp-rail-source", Select)
        select.value = "server"
        await pilot.pause()
        changed = [e for e in app.events if isinstance(e, MCPRail.SourceChanged)]
        assert changed and changed[-1].source == "server"


@pytest.mark.asyncio
async def test_rail_hides_scope_section_for_local_source():
    app = RailApp()
    async with app.run_test() as pilot:
        assert not list(app.query("#mcp-rail-scope"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/UI/test_mcp_rail.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.UI.MCP_Modules.mcp_rail'`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/UI/MCP_Modules/mcp_rail.py
"""MCP Hub left rail: source switch, server rows with readiness badges, scope."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Label, Select, Static

from tldw_chatbook.MCP.readiness import STATE_GLYPHS, ReadinessSnapshot

MCP_RAIL_ROW_PREFIX = "mcp-rail-row-"
_MAX_ROW_LABEL = 22


def _row_label(snapshot: ReadinessSnapshot) -> str:
    label = snapshot.label
    if len(label) > _MAX_ROW_LABEL:
        label = f"{label[: _MAX_ROW_LABEL - 3].rstrip()}..."
    prefix = "⌂ " if snapshot.source == "builtin" else ""
    suffix = f" · {snapshot.tool_count}" if snapshot.tool_count is not None else ""
    return f"{STATE_GLYPHS[snapshot.state]} {prefix}{label}{suffix}"


class MCPRail(Vertical):
    """Left rail for the MCP workbench. Index-based row ids; keys in a list."""

    DEFAULT_CSS = """
    MCPRail {
        width: 3fr;
        min-width: 24;
        height: 100%;
        min-height: 0;
    }
    Button.mcp-rail-row {
        width: 100%;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
    """

    class SourceChanged(Message):
        def __init__(self, source: str) -> None:
            super().__init__()
            self.source = source

    class ServerSelected(Message):
        def __init__(self, server_key: str | None) -> None:
            super().__init__()
            self.server_key = server_key

    class ScopeChanged(Message):
        def __init__(self, scope: str, scope_ref: str | None) -> None:
            super().__init__()
            self.scope = scope
            self.scope_ref = scope_ref

    def __init__(
        self,
        *,
        source: str,
        snapshots: list[ReadinessSnapshot],
        selected_server_key: str | None,
        scope_options: list[tuple[str, str]],
        scope_value: str,
        scope_ref_options: list[tuple[str, str]],
        scope_ref_value: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.snapshots = snapshots
        self.selected_server_key = selected_server_key
        self.scope_options = scope_options
        self.scope_value = scope_value
        self.scope_ref_options = scope_ref_options
        self.scope_ref_value = scope_ref_value
        self._row_keys: list[str | None] = []

    def sync_state(
        self,
        *,
        source: str,
        snapshots: list[ReadinessSnapshot],
        selected_server_key: str | None,
        scope_options: list[tuple[str, str]],
        scope_value: str,
        scope_ref_options: list[tuple[str, str]],
        scope_ref_value: str | None,
    ) -> None:
        self.source = source
        self.snapshots = snapshots
        self.selected_server_key = selected_server_key
        self.scope_options = scope_options
        self.scope_value = scope_value
        self.scope_ref_options = scope_ref_options
        self.scope_ref_value = scope_ref_value
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        yield Static("Source", classes="destination-section mcp-rail-heading")
        yield Select(
            [("Local", "local"), ("Server", "server")],
            id="mcp-rail-source",
            allow_blank=False,
            value=self.source if self.source in ("local", "server") else "local",
        )
        yield Static("Servers", classes="destination-section mcp-rail-heading")
        self._row_keys = [None] + [snap.server_key for snap in self.snapshots]
        all_row = Button(
            "All servers",
            id=f"{MCP_RAIL_ROW_PREFIX}0",
            classes="mcp-rail-row console-action-subdued",
            compact=True,
        )
        all_row.set_class(self.selected_server_key is None, "is-active")
        yield all_row
        for index, snap in enumerate(self.snapshots, start=1):
            row = Button(
                _row_label(snap),
                id=f"{MCP_RAIL_ROW_PREFIX}{index}",
                classes="mcp-rail-row console-action-subdued",
                compact=True,
            )
            row.tooltip = snap.message or snap.label
            row.set_class(snap.server_key == self.selected_server_key, "is-active")
            yield row
        if self.source == "server":
            with Vertical(id="mcp-rail-scope"):
                yield Label("Scope", classes="form-label")
                yield Select(
                    self.scope_options or [("Personal", "personal")],
                    id="mcp-rail-scope-select",
                    allow_blank=False,
                    value=self.scope_value,
                )
                yield Label("Scope Entity", classes="form-label")
                ref_options = self.scope_ref_options or [("No scope entities", Select.BLANK)]
                yield Select(
                    ref_options,
                    id="mcp-rail-scope-ref",
                    value=self.scope_ref_value if self.scope_ref_value else Select.BLANK,
                    disabled=not self.scope_ref_options,
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith(MCP_RAIL_ROW_PREFIX):
            return
        event.stop()
        index = int(button_id.removeprefix(MCP_RAIL_ROW_PREFIX))
        if 0 <= index < len(self._row_keys):
            self.post_message(self.ServerSelected(self._row_keys[index]))

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""
        if select_id == "mcp-rail-source":
            event.stop()
            if event.value in ("local", "server") and event.value != self.source:
                self.post_message(self.SourceChanged(str(event.value)))
        elif select_id == "mcp-rail-scope-select":
            event.stop()
            self.post_message(self.ScopeChanged(str(event.value), None))
        elif select_id == "mcp-rail-scope-ref":
            event.stop()
            ref = None if event.value is Select.BLANK else str(event.value)
            self.post_message(self.ScopeChanged(self.scope_value, ref))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/UI/test_mcp_rail.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_rail.py Tests/UI/test_mcp_rail.py
git commit -m "feat(mcp-hub): add rail widget with readiness badges and source/scope controls"
```

---

### Task 5: Servers-mode canvas (overview table + server detail)

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`
- Test: `Tests/UI/test_mcp_servers_mode.py`

**Interfaces:**
- Consumes: `ReadinessSnapshot`, `aggregate_summary`, `ReadinessState`, `STATE_LABELS` (Task 1); `redact_args`, `redact_mapping` (Task 3); `BUILTIN_CLIENT_SNIPPET` via snapshot `detail["client_snippet"]`.
- Produces: `MCPServersMode(Vertical)` with:
  - `update_overview(snapshots: list[ReadinessSnapshot]) -> None` — aggregate line + DataTable + recovery callouts
  - `show_detail(snapshot: ReadinessSnapshot | None) -> None` — None returns to overview
  - Message `MCPServersMode.ServerRowSelected(server_key: str)` on table row selection
  - Copy button (`#mcp-detail-copy-snippet`) for the built-in server calling `self.app.copy_to_clipboard(...)`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_mcp_servers_mode.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.MCP.readiness import (
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    builtin_readiness,
)
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode


def _snap(key: str, label: str, state=ReadinessState.READY, reasons=(), message="", **kw):
    return ReadinessSnapshot(
        server_key=key, label=label, source=key.split(":", 1)[0],
        state=state, reasons=reasons, message=message, **kw,
    )


class CanvasApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPServersMode(id="mcp-mode-canvas-servers")

    def on_mcp_servers_mode_server_row_selected(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_overview_renders_aggregate_table_and_callouts():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        canvas.update_overview(
            [
                _snap("local:docs", "docs", tool_count=4),
                _snap(
                    "local:web", "web",
                    state=ReadinessState.NEEDS_SETUP,
                    reasons=(ReasonCode.AUTH_MISSING,),
                    message="Missing environment variables: KEY.",
                ),
            ]
        )
        await pilot.pause()
        summary = app.query_one("#mcp-overview-summary", Static)
        assert "1 of 2" in str(summary.renderable)
        table = app.query_one("#mcp-servers-table", DataTable)
        assert table.row_count == 2
        callouts = list(app.query(".ds-recovery-callout"))
        assert len(callouts) == 1  # one problem row -> one callout


@pytest.mark.asyncio
async def test_table_row_selection_posts_server_key():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        canvas.update_overview([_snap("local:docs", "docs")])
        await pilot.pause()
        table = app.query_one("#mcp-servers-table", DataTable)
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.events and app.events[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_detail_renders_redacted_config_and_builtin_snippet():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        local = _snap(
            "local:docs", "docs",
            detail={
                "command": "python",
                "args": ["--api-key", "sk-123"],
                "env_placeholders": {"API_KEY": "$MY_KEY"},
                "missing_env": [],
                "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
            },
        )
        canvas.show_detail(local)
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "sk-123" not in body
        assert "python" in body

        canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        assert list(app.query("#mcp-detail-copy-snippet"))

        canvas.show_detail(None)
        await pilot.pause()
        assert app.query_one("#mcp-servers-overview").display
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/UI/test_mcp_servers_mode.py -v`
Expected: FAIL with `ModuleNotFoundError` for `mcp_servers_mode`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py
"""Servers-mode canvas: readiness overview table and per-server detail."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.MCP.readiness import (
    ReadinessSnapshot,
    ReadinessState,
    aggregate_summary,
)
from tldw_chatbook.MCP.redaction import redact_args

_TABLE_COLUMNS = ("Name", "Transport", "Status", "Tools", "Auth", "Scope")


class MCPServersMode(Vertical):
    """Canvas for the Servers mode. Read-only in Phase 1."""

    DEFAULT_CSS = """
    MCPServersMode {
        width: 1fr;
        height: 100%;
        min-height: 0;
    }
    #mcp-servers-table {
        height: 1fr;
        min-height: 4;
    }
    #mcp-detail-scroll {
        height: 1fr;
        min-height: 0;
    }
    """

    class ServerRowSelected(Message):
        def __init__(self, server_key: str) -> None:
            super().__init__()
            self.server_key = server_key

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshots: list[ReadinessSnapshot] = []
        self._detail_snapshot: ReadinessSnapshot | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-servers-overview"):
            yield Static("", id="mcp-overview-summary", classes="ds-status-badge", markup=False)
            table = DataTable(id="mcp-servers-table")
            table.cursor_type = "row"
            yield table
            yield Vertical(id="mcp-overview-callouts")
        with Vertical(id="mcp-servers-detail"):
            yield Static("", id="mcp-detail-title", classes="destination-section", markup=False)
            with VerticalScroll(id="mcp-detail-scroll"):
                yield Static("", id="mcp-detail-body", classes="ds-field-row", markup=False)
                yield Button(
                    "Copy client config",
                    id="mcp-detail-copy-snippet",
                    classes="console-action-secondary",
                    compact=True,
                )

    def on_mount(self) -> None:
        table = self.query_one("#mcp-servers-table", DataTable)
        table.add_columns(*_TABLE_COLUMNS)
        self._show_overview_container(True)

    def _show_overview_container(self, show_overview: bool) -> None:
        self.query_one("#mcp-servers-overview").display = show_overview
        self.query_one("#mcp-servers-detail").display = not show_overview

    def update_overview(self, snapshots: list[ReadinessSnapshot]) -> None:
        self._snapshots = list(snapshots)
        summary = self.query_one("#mcp-overview-summary", Static)
        summary.update(aggregate_summary(self._snapshots))
        table = self.query_one("#mcp-servers-table", DataTable)
        table.clear()
        for snap in self._snapshots:
            table.add_row(
                snap.label,
                snap.transport,
                snap.badge_text(),
                "—" if snap.tool_count is None else str(snap.tool_count),
                snap.auth_display,
                snap.scope_display,
                key=snap.server_key,
            )
        callouts = self.query_one("#mcp-overview-callouts", Vertical)
        callouts.remove_children()
        for snap in self._snapshots:
            if snap.state in (ReadinessState.READY, ReadinessState.CHECKING):
                continue
            callouts.mount(
                Static(
                    f"{snap.label}: {snap.message}",
                    classes="ds-recovery-callout",
                    markup=False,
                )
            )
        if self._detail_snapshot is None:
            self._show_overview_container(True)

    def show_detail(self, snapshot: ReadinessSnapshot | None) -> None:
        self._detail_snapshot = snapshot
        if snapshot is None:
            self._show_overview_container(True)
            return
        self._show_overview_container(False)
        self.query_one("#mcp-detail-title", Static).update(
            f"{snapshot.badge_text()}  {snapshot.label}"
        )
        self.query_one("#mcp-detail-body", Static).update(self._detail_text(snapshot))
        self.query_one("#mcp-detail-copy-snippet", Button).display = (
            snapshot.source == "builtin"
        )

    def _detail_text(self, snapshot: ReadinessSnapshot) -> str:
        detail = snapshot.detail or {}
        lines: list[str] = [snapshot.message, ""]
        if snapshot.source == "local":
            args = redact_args([str(a) for a in detail.get("args") or []])
            lines.append(f"Command · {detail.get('command') or '—'} {' '.join(args)}".rstrip())
            placeholders = detail.get("env_placeholders") or {}
            missing = set(detail.get("missing_env") or [])
            for env_key, raw in placeholders.items():
                marker = "missing" if str(raw).strip("${}") in missing else "set"
                lines.append(f"Env · {env_key} ({marker})")
            discovery = detail.get("discovery_snapshot") or {}
            for kind in ("tools", "resources", "prompts"):
                items = discovery.get(kind) or []
                names = ", ".join(str(item.get("name") or item.get("uri") or "?") for item in items[:8])
                suffix = f": {names}" if names else ""
                lines.append(f"{kind.title()} · {len(items)}{suffix}")
        elif snapshot.source == "server":
            lines.append(f"Base URL · {detail.get('base_url') or '—'}")
            lines.append(f"Auth · {snapshot.auth_display}")
            lines.append("External server records: see Advanced ▸ External Servers.")
        else:  # builtin
            lines.append("Runs over stdio when an MCP client launches it:")
            lines.append("  python3 -m tldw_chatbook.MCP")
            for flag in ("expose_tools", "expose_resources", "expose_prompts"):
                lines.append(f"{flag} · {detail.get(flag, True)}")
        return "\n".join(lines)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self.post_message(self.ServerRowSelected(str(event.row_key.value)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "mcp-detail-copy-snippet":
            return
        event.stop()
        snippet = ""
        if self._detail_snapshot is not None:
            snippet = str((self._detail_snapshot.detail or {}).get("client_snippet") or "")
        if snippet:
            self.app.copy_to_clipboard(snippet)
            self.app.notify("Client config copied to clipboard.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/UI/test_mcp_servers_mode.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py Tests/UI/test_mcp_servers_mode.py
git commit -m "feat(mcp-hub): add servers-mode canvas with readiness table and redacted detail"
```

---

### Task 6: Inspector with readiness actions + Advanced escape hatch

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`
- Test: `Tests/UI/test_mcp_inspector.py`

**Interfaces:**
- Consumes: `ReadinessSnapshot`, `HubAction` (Task 1); service surface `available_actions()`, `runtime_state_override()`, `run_action(name, payload)`, `load_section(section)`; `render_unified_mcp_section` from `tldw_chatbook.UI.MCP_Modules.unified_mcp_sections` (existing, unmodified); app policy gate `require_ui_action_allowed(action_id=..., runtime_state_override=...)` (same seam the legacy panel uses).
- Produces: `MCPInspector(Vertical)`:
  - `update_readiness(snapshot: ReadinessSnapshot | None) -> None`
  - `set_service_context(service: Any, sections: list[tuple[str, str]]) -> None` (Advanced browser options)
  - Message `MCPInspector.HubActionRequested(action: HubAction, server_key: str | None)` for the wired actions (view_details / open_tool_catalog / open_audit); all other action buttons render disabled with tooltip "Available in a later phase — use Advanced below."
  - Advanced section ids: `#mcp-adv-section-select`, `#mcp-adv-content`, `#mcp-adv-action-select`, `#mcp-adv-payload`, `#mcp-adv-run`, `#mcp-adv-result` (fresh ids — never reuse legacy `#unified-mcp-*` ids, both widgets can exist in one app during transition).

Behavior of the Advanced runner (capability preserved from the legacy panel): pick section → `load_section(section)` → `render_unified_mcp_section` text; pick action from `available_actions()` (options gated through `app.require_ui_action_allowed` with `service.runtime_state_override()` when both exist, mirroring `unified_mcp_panel.py:677-748`); payload TextArea seeds from the descriptor's `payload_template`; Run parses JSON (`json.JSONDecodeError` → inline error in `#mcp-adv-result`, no crash) and awaits `service.run_action(name, payload)` in a worker; result (or error string) rendered redacted via `redact_mapping` when the result is a dict.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_mcp_inspector.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static, TextArea

from tldw_chatbook.MCP.readiness import (
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector


class FakeAdvService:
    def __init__(self) -> None:
        self.action_calls: list[tuple[str, dict]] = []

    async def load_section(self, section=None):
        return {"source": "local", "section": section or "overview"}

    def available_actions(self):
        return [
            {
                "name": "profile.connect",
                "label": "Connect Profile",
                "action_id": "mcp.external_profiles.configure.local",
                "payload_template": '{"profile_id":"demo"}',
            }
        ]

    async def run_action(self, action_name, payload):
        self.action_calls.append((action_name, dict(payload or {})))
        return {"ok": True}


class InspectorApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.service = FakeAdvService()
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield MCPInspector(id="mcp-inspector")

    def on_mount(self) -> None:
        inspector = self.query_one(MCPInspector)
        inspector.set_service_context(self.service, [("Overview", "overview"), ("Inventory", "inventory")])

    def on_mcp_inspector_hub_action_requested(self, event) -> None:
        self.events.append(event)


def _stale_snap() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:docs", label="docs", source="local",
        state=ReadinessState.STALE, reasons=(ReasonCode.RUNTIME_UNAVAILABLE,),
        message="2 tools discovered; not currently connected.",
    )


@pytest.mark.asyncio
async def test_readiness_block_shows_state_message_and_action_buttons():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.update_readiness(_stale_snap())
        await pilot.pause()
        badge = str(app.query_one("#mcp-inspector-state", Static).renderable)
        assert "Stale" in badge
        buttons = {b.id: b for b in app.query("Button.mcp-inspector-action")}
        # connect: not wired in Phase 1 -> disabled; view_details: wired -> enabled
        assert buttons["mcp-inspector-action-connect"].disabled
        assert not buttons["mcp-inspector-action-view_details"].disabled


@pytest.mark.asyncio
async def test_wired_action_posts_hub_action_requested():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.update_readiness(_stale_snap())
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-view_details")
        await pilot.pause()
        assert app.events
        assert app.events[-1].action is HubAction.VIEW_DETAILS
        assert app.events[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_advanced_runner_runs_action_with_template_payload():
    app = InspectorApp()
    async with app.run_test() as pilot:
        select = app.query_one("#mcp-adv-action-select", Select)
        assert select.value == "profile.connect"
        payload = app.query_one("#mcp-adv-payload", TextArea)
        assert "demo" in payload.text
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert app.service.action_calls == [("profile.connect", {"profile_id": "demo"})]
        assert "ok" in str(app.query_one("#mcp-adv-result", Static).renderable)


@pytest.mark.asyncio
async def test_advanced_runner_reports_invalid_json_without_crashing():
    app = InspectorApp()
    async with app.run_test() as pilot:
        payload = app.query_one("#mcp-adv-payload", TextArea)
        payload.text = "{not json"
        await pilot.click("#mcp-adv-run")
        await pilot.pause()
        assert app.service.action_calls == []
        assert "Invalid JSON" in str(app.query_one("#mcp-adv-result", Static).renderable)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/UI/test_mcp_inspector.py -v`
Expected: FAIL with `ModuleNotFoundError` for `mcp_inspector`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/UI/MCP_Modules/mcp_inspector.py
"""MCP Hub inspector: readiness explanation, actions, Advanced escape hatch."""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Label, Select, Static, TextArea

from tldw_chatbook.MCP.readiness import HubAction, ReadinessSnapshot
from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.UI.MCP_Modules.unified_mcp_sections import render_unified_mcp_section

# Actions that have first-class UI in Phase 1. Everything else renders
# disabled and points at the Advanced runner below (capability preserved).
_WIRED_ACTIONS = {HubAction.VIEW_DETAILS, HubAction.OPEN_TOOL_CATALOG, HubAction.OPEN_AUDIT}

_ACTION_LABELS: dict[HubAction, str] = {
    HubAction.ADD_SERVER: "Add server",
    HubAction.EDIT_CONFIG: "Edit config",
    HubAction.OPEN_CREDENTIALS: "Open credentials",
    HubAction.CONNECT: "Connect",
    HubAction.REFRESH_DISCOVERY: "Refresh tools",
    HubAction.VALIDATE: "Check readiness",
    HubAction.VIEW_DETAILS: "View details",
    HubAction.OPEN_TOOL_CATALOG: "Open tool catalog",
    HubAction.OPEN_AUDIT: "Open audit",
}


class MCPInspector(Vertical):
    """Right-pane inspector: what is selected, why, what can I do."""

    DEFAULT_CSS = """
    MCPInspector {
        width: 3fr;
        min-width: 28;
        height: 100%;
        min-height: 0;
    }
    #mcp-adv-scroll {
        height: 1fr;
        min-height: 0;
    }
    #mcp-adv-payload {
        height: 6;
        min-height: 3;
    }
    Button.mcp-inspector-action {
        width: 100%;
        height: 1;
        min-height: 1;
        border: none;
    }
    """

    class HubActionRequested(Message):
        def __init__(self, action: HubAction, server_key: str | None) -> None:
            super().__init__()
            self.action = action
            self.server_key = server_key

    def __init__(self, **kwargs: Any) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"ds-inspector {classes}".strip(), **kwargs)
        self._snapshot: ReadinessSnapshot | None = None
        self._service: Any = None
        self._sections: list[tuple[str, str]] = [("Overview", "overview")]
        self._action_templates: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("Inspector", classes="destination-section")
        yield Static("Select a server to see its readiness.", id="mcp-inspector-state",
                     classes="ds-status-badge", markup=False)
        yield Static("", id="mcp-inspector-message", classes="ds-field-row", markup=False)
        yield Vertical(id="mcp-inspector-actions")
        yield Static("Advanced (legacy control plane)", classes="destination-section")
        with VerticalScroll(id="mcp-adv-scroll"):
            yield Label("Section", classes="form-label")
            yield Select(self._sections, id="mcp-adv-section-select", allow_blank=False,
                         value="overview")
            yield Static("", id="mcp-adv-content", classes="ds-field-row", markup=False)
            yield Label("Action", classes="form-label")
            yield Select([("No actions available", Select.BLANK)], id="mcp-adv-action-select",
                         value=Select.BLANK)
            yield Label("Payload (JSON)", classes="form-label")
            yield TextArea("{}", id="mcp-adv-payload")
            yield Button("Run Action", id="mcp-adv-run", classes="console-action-primary",
                         compact=True)
            yield Static("", id="mcp-adv-result", classes="ds-field-row", markup=False)

    # -- readiness block -----------------------------------------------------

    def update_readiness(self, snapshot: ReadinessSnapshot | None) -> None:
        self._snapshot = snapshot
        state = self.query_one("#mcp-inspector-state", Static)
        message = self.query_one("#mcp-inspector-message", Static)
        actions = self.query_one("#mcp-inspector-actions", Vertical)
        actions.remove_children()
        if snapshot is None:
            state.update("Select a server to see its readiness.")
            message.update("")
            return
        state.update(f"{snapshot.badge_text()}  {snapshot.label}")
        reason = snapshot.primary_reason
        reason_suffix = f" [{reason.value}]" if reason else ""
        message.update(f"{snapshot.message}{reason_suffix}")
        for action in snapshot.allowed_actions:
            button = Button(
                _ACTION_LABELS[action],
                id=f"mcp-inspector-action-{action.value}",
                classes="mcp-inspector-action console-action-secondary",
                compact=True,
            )
            if action not in _WIRED_ACTIONS:
                button.disabled = True
                button.tooltip = "Available in a later phase — use Advanced below."
            actions.mount(button)

    # -- advanced escape hatch -----------------------------------------------

    def set_service_context(self, service: Any, sections: list[tuple[str, str]]) -> None:
        self._service = service
        self._sections = sections or [("Overview", "overview")]
        section_select = self.query_one("#mcp-adv-section-select", Select)
        with section_select.prevent(Select.Changed):
            section_select.set_options(self._sections)
            section_select.value = self._sections[0][1]
        self._refresh_advanced_actions()
        self.run_worker(self._load_advanced_section(self._sections[0][1]),
                        group="mcp-adv-section", exclusive=True)

    def _refresh_advanced_actions(self) -> None:
        action_select = self.query_one("#mcp-adv-action-select", Select)
        payload = self.query_one("#mcp-adv-payload", TextArea)
        run_button = self.query_one("#mcp-adv-run", Button)
        descriptors = []
        if self._service is not None:
            loader = getattr(self._service, "available_actions", None)
            if callable(loader):
                descriptors = [d for d in (loader() or []) if self._action_allowed(d)]
        self._action_templates = {
            str(d["name"]): str(d.get("payload_template") or "{}") for d in descriptors
        }
        with action_select.prevent(Select.Changed):
            if not descriptors:
                action_select.set_options([("No actions available", Select.BLANK)])
                action_select.value = Select.BLANK
                action_select.disabled = True
                run_button.disabled = True
                return
            options = [(str(d["label"]), str(d["name"])) for d in descriptors]
            action_select.set_options(options)
            action_select.value = options[0][1]
            action_select.disabled = False
            run_button.disabled = False
            payload.text = self._action_templates.get(options[0][1], "{}")

    def _action_allowed(self, descriptor: dict[str, Any]) -> bool:
        """Mirror the legacy panel's policy gate; permissive when seams absent."""
        gate = getattr(self.app, "require_ui_action_allowed", None)
        override = getattr(self._service, "runtime_state_override", None)
        if not callable(gate) or not callable(override):
            return True
        try:
            decision = gate(action_id=str(descriptor.get("action_id") or ""),
                            runtime_state_override=override())
        except Exception:
            return True
        return bool(getattr(decision, "allowed", True))

    async def _load_advanced_section(self, section: str) -> None:
        if self._service is None:
            return
        payload = await self._service.load_section(section)
        self.query_one("#mcp-adv-content", Static).update(
            render_unified_mcp_section(section, payload)
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""
        if select_id == "mcp-adv-section-select":
            event.stop()
            self.run_worker(self._load_advanced_section(str(event.value)),
                            group="mcp-adv-section", exclusive=True)
        elif select_id == "mcp-adv-action-select":
            event.stop()
            if event.value is not Select.BLANK:
                self.query_one("#mcp-adv-payload", TextArea).text = (
                    self._action_templates.get(str(event.value), "{}")
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-adv-run":
            event.stop()
            self.run_worker(self._run_advanced_action(), group="mcp-adv-run", exclusive=True)
            return
        if button_id.startswith("mcp-inspector-action-"):
            event.stop()
            action = HubAction(button_id.removeprefix("mcp-inspector-action-"))
            server_key = self._snapshot.server_key if self._snapshot else None
            self.post_message(self.HubActionRequested(action, server_key))

    async def _run_advanced_action(self) -> None:
        result_widget = self.query_one("#mcp-adv-result", Static)
        action_select = self.query_one("#mcp-adv-action-select", Select)
        if self._service is None or action_select.value is Select.BLANK:
            return
        raw = self.query_one("#mcp-adv-payload", TextArea).text or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            result_widget.update(f"Invalid JSON payload: {exc}")
            return
        try:
            result = await self._service.run_action(str(action_select.value), payload)
        except Exception as exc:  # surface, never crash the inspector
            result_widget.update(f"Action failed: {exc}")
            return
        if isinstance(result, dict):
            result = redact_mapping(result)
        result_widget.update(json.dumps(result, default=str, indent=1)[:2000])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/UI/test_mcp_inspector.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_inspector.py Tests/UI/test_mcp_inspector.py
git commit -m "feat(mcp-hub): add inspector with readiness actions and Advanced escape hatch"
```

---

### Task 7: Workbench assembly

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Test: `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: Tasks 1-6 exports; service surface (`load_context`, `load_section`, `select_source`, `select_server_target`, `select_scope`, `target_store.list_targets()`); `get_cli_setting` from `tldw_chatbook.config` for `[mcp]` built-in flags; `UnifiedMCPContext` fields (`selected_source`, `selected_active_server_id`, `selected_scope`, `selected_scope_ref`, `per_server_state`).
- Produces: `MCPWorkbench(Container)`:
  - `__init__(self, app_instance: Any = None, **kwargs)` — same `app_instance` fallback idiom as the legacy panel
  - `MCP_HUB_MODES: dict[str, dict]` registry (keys `servers|tools|permissions|audit`; values `label`, `button_id`, `placeholder`)
  - `set_mode(mode: str) -> None`; `active_mode: str` property
  - `get_view_state() -> dict` → `{"mode", "source", "selected_server_key", "scope", "scope_ref"}`
  - `set_initial_view_state(state: dict | None) -> None` — tolerant: unknown keys ignored, unknown mode → "servers"; ALSO accepts the legacy panel shape (reads `selected_source` when `source` is absent)
  - `async reload() -> None` — used by the screen's runtime-backend hook
- Phase 1 scoping (recorded deviation): the rail lists the ACTIVE source's servers (built-in row always shown under Local) because the control-plane service exposes local profiles only through source-scoped `load_section`; cross-source listing arrives when the service grows typed methods (Phase 2+, spec §12).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_mcp_workbench.py
from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ContentSwitcher, Static

from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCP_RAIL_ROW_PREFIX, MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench


class FakeTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class FakeTargetStore:
    def list_targets(self):
        return [FakeTarget()]


class FakeHubService:
    def __init__(self) -> None:
        self.target_store = FakeTargetStore()
        self.context = UnifiedMCPContext(selected_source="local", selected_section="overview")

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        self.context = replace(self.context, selected_active_server_id=server_id)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        if self.context.selected_source == "local":
            return [
                {
                    "profile_id": "docs",
                    "command": "python",
                    "args": [],
                    "env_placeholders": {},
                    "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
                    "is_connected": True,
                }
            ]
        return {"external_servers": [], "source": "server", "section": "external_servers"}

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class WorkbenchApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = FakeHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_workbench_mounts_rail_canvas_inspector_and_loads_local_servers():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"
        rail = app.query_one(MCPRail)
        # builtin + docs rows (+ "All servers")
        assert len(list(app.query("Button.mcp-rail-row"))) == 3
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-overview").display


@pytest.mark.asyncio
async def test_mode_switch_shows_placeholder_canvases():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        switcher = app.query_one(ContentSwitcher)
        assert switcher.current == "mcp-mode-canvas-permissions"
        placeholder = str(
            app.query_one("#mcp-mode-canvas-permissions Static", Static).renderable
        )
        assert "later phase" in placeholder.lower()


@pytest.mark.asyncio
async def test_rail_selection_drives_detail_and_view_state():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-detail").display
        state = app.query_one(MCPWorkbench).get_view_state()
        assert state["selected_server_key"] == "local:docs"
        assert state["mode"] == "servers"


@pytest.mark.asyncio
async def test_restore_tolerates_legacy_and_garbage_state():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # legacy panel shape
        workbench.set_initial_view_state(
            {"selected_source": "local", "selected_section": "inventory"}
        )
        await pilot.pause()
        assert workbench.active_mode == "servers"
        # garbage
        workbench.set_initial_view_state({"mode": "nonsense", "bogus": 1})
        await pilot.pause()
        assert workbench.active_mode == "servers"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/UI/test_mcp_workbench.py -v`
Expected: FAIL with `ModuleNotFoundError` for `mcp_workbench`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
"""MCP Hub workbench: rail + mode canvases + inspector assembly."""

from __future__ import annotations

from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import ContentSwitcher, Static

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.MCP.readiness import (
    BUILTIN_SERVER_KEY,
    HubAction,
    ReadinessSnapshot,
    builtin_readiness,
    local_profile_readiness,
    server_target_readiness,
)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode

MCP_HUB_MODES: dict[str, dict[str, str]] = {
    "servers": {"label": "Servers", "button_id": "mcp-mode-servers", "placeholder": ""},
    "tools": {
        "label": "Tools",
        "button_id": "mcp-mode-tools",
        "placeholder": (
            "Tools mode arrives in a later phase. Until then, a server's tools are "
            "listed in its Server detail, and tool actions run via Advanced in the inspector."
        ),
    },
    "permissions": {
        "label": "Permissions",
        "button_id": "mcp-mode-permissions",
        "placeholder": (
            "Permissions mode arrives in a later phase. MCP tools are not yet callable "
            "from chat, so there is nothing to permit yet."
        ),
    },
    "audit": {
        "label": "Audit",
        "button_id": "mcp-mode-audit",
        "placeholder": (
            "Audit mode arrives in a later phase. Action results appear inline in the "
            "inspector's Advanced section for now."
        ),
    },
}

_LEGACY_SECTIONS = [
    ("Overview", "overview"),
    ("Inventory", "inventory"),
    ("External Servers", "external_servers"),
    ("Governance", "governance"),
    ("Advanced", "advanced"),
]


class MCPWorkbench(Container):
    """Assembles the Phase 1 MCP Hub. Read-only over the control-plane service."""

    DEFAULT_CSS = """
    MCPWorkbench {
        width: 100%;
        height: 1fr;
        min-height: 0;
    }
    #mcp-hub-grid {
        width: 100%;
        height: 100%;
        min-height: 0;
    }
    #mcp-hub-canvas {
        width: 5fr;
        min-width: 38;
        height: 100%;
        min-height: 0;
    }
    """

    def __init__(self, app_instance: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self.active_mode = "servers"
        self._source = "local"
        self._selected_server_key: str | None = None
        self._snapshots: list[ReadinessSnapshot] = []
        self._pending_view_state: dict[str, Any] | None = None

    @property
    def app_instance(self) -> Any:
        if self._app_instance is not None:
            return self._app_instance
        try:
            return self.app
        except Exception:
            return None

    def _service(self) -> Any:
        return getattr(self.app_instance, "unified_mcp_service", None)

    def compose(self) -> ComposeResult:
        with Horizontal(id="mcp-hub-grid", classes="destination-workbench"):
            yield MCPRail(
                source=self._source,
                snapshots=[],
                selected_server_key=None,
                scope_options=[("Personal", "personal")],
                scope_value="personal",
                scope_ref_options=[],
                scope_ref_value=None,
                id="mcp-hub-rail",
                classes="destination-workbench-pane",
            )
            with ContentSwitcher(
                initial="mcp-mode-canvas-servers",
                id="mcp-hub-canvas",
                classes="destination-workbench-pane",
            ):
                yield MCPServersMode(id="mcp-mode-canvas-servers")
                for mode, spec in MCP_HUB_MODES.items():
                    if mode == "servers":
                        continue
                    with Vertical(id=f"mcp-mode-canvas-{mode}"):
                        yield Static(
                            spec["placeholder"],
                            classes="ds-recovery-callout",
                            markup=False,
                        )
            yield MCPInspector(id="mcp-hub-inspector", classes="destination-workbench-pane")

    async def on_mount(self) -> None:
        await self.reload()
        if self._pending_view_state:
            await self._apply_view_state(self._pending_view_state)
            self._pending_view_state = None

    # -- data loading ---------------------------------------------------------

    async def reload(self) -> None:
        service = self._service()
        if service is not None:
            try:
                context = await service.load_context()
                self._source = context.selected_source or "local"
            except Exception as exc:
                logger.warning(f"MCP workbench context load failed: {exc}")
        self._snapshots = await self._collect_snapshots()
        self._sync_children()
        inspector = self.query_one(MCPInspector)
        inspector.set_service_context(self._service(), _LEGACY_SECTIONS)

    async def _collect_snapshots(self) -> list[ReadinessSnapshot]:
        snapshots: list[ReadinessSnapshot] = []
        service = self._service()
        if self._source == "local":
            snapshots.append(
                builtin_readiness(
                    enabled=bool(get_cli_setting("mcp", "enabled", False)),
                    expose_tools=bool(get_cli_setting("mcp", "expose_tools", True)),
                    expose_resources=bool(get_cli_setting("mcp", "expose_resources", True)),
                    expose_prompts=bool(get_cli_setting("mcp", "expose_prompts", True)),
                )
            )
            if service is not None:
                try:
                    records = await service.load_section("external_servers")
                except Exception as exc:
                    logger.warning(f"MCP local profile listing failed: {exc}")
                    records = []
                if isinstance(records, list):  # local source returns a bare list
                    snapshots.extend(local_profile_readiness(r) for r in records)
        else:
            target_store = getattr(service, "target_store", None)
            if target_store is not None:
                snapshots.extend(
                    server_target_readiness(t) for t in target_store.list_targets()
                )
        return snapshots

    def _snapshot_for(self, server_key: str | None) -> ReadinessSnapshot | None:
        if server_key is None:
            return None
        for snap in self._snapshots:
            if snap.server_key == server_key:
                return snap
        return None

    def _sync_children(self) -> None:
        rail = self.query_one(MCPRail)
        rail.sync_state(
            source=self._source,
            snapshots=self._snapshots,
            selected_server_key=self._selected_server_key,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
        )
        canvas = self.query_one(MCPServersMode)
        canvas.update_overview(self._snapshots)
        selected = self._snapshot_for(self._selected_server_key)
        canvas.show_detail(selected)
        self.query_one(MCPInspector).update_readiness(selected)

    # -- modes & view state ---------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in MCP_HUB_MODES:
            mode = "servers"
        self.active_mode = mode
        self.query_one(ContentSwitcher).current = f"mcp-mode-canvas-{mode}"

    def get_view_state(self) -> dict[str, Any]:
        return {
            "mode": self.active_mode,
            "source": self._source,
            "selected_server_key": self._selected_server_key,
            "scope": "personal",
            "scope_ref": None,
        }

    def set_initial_view_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if self.is_mounted:
            self.run_worker(self._apply_view_state(dict(state)),
                            group="mcp-workbench-restore", exclusive=True)
        else:
            self._pending_view_state = dict(state)

    async def _apply_view_state(self, state: dict[str, Any]) -> None:
        # Tolerant restore: unknown keys ignored; legacy panel shape accepted.
        source = state.get("source") or state.get("selected_source")
        if source in ("local", "server") and source != self._source:
            await self._switch_source(str(source))
        self.set_mode(str(state.get("mode") or "servers"))
        server_key = state.get("selected_server_key")
        if isinstance(server_key, str) and self._snapshot_for(server_key) is not None:
            self._selected_server_key = server_key
        self._sync_children()

    # -- event wiring -----------------------------------------------------------

    async def _switch_source(self, source: str) -> None:
        service = self._service()
        if service is not None:
            try:
                await service.select_source(source)
            except Exception as exc:
                logger.warning(f"MCP source switch failed: {exc}")
        self._source = source
        self._selected_server_key = None
        self._snapshots = await self._collect_snapshots()
        self._sync_children()

    async def on_mcp_rail_source_changed(self, event: MCPRail.SourceChanged) -> None:
        event.stop()
        await self._switch_source(event.source)

    async def on_mcp_rail_server_selected(self, event: MCPRail.ServerSelected) -> None:
        event.stop()
        self._selected_server_key = event.server_key
        service = self._service()
        if (
            service is not None
            and event.server_key is not None
            and event.server_key.startswith("server:")
            and "/" not in event.server_key
        ):
            try:
                await service.select_server_target(event.server_key.split(":", 1)[1])
            except Exception as exc:
                logger.warning(f"MCP server target selection failed: {exc}")
        self._sync_children()

    async def on_mcp_rail_scope_changed(self, event: MCPRail.ScopeChanged) -> None:
        event.stop()
        service = self._service()
        if service is not None:
            try:
                await service.select_scope(event.scope, event.scope_ref)
            except Exception as exc:
                logger.warning(f"MCP scope selection failed: {exc}")

    def on_mcp_servers_mode_server_row_selected(
        self, event: MCPServersMode.ServerRowSelected
    ) -> None:
        event.stop()
        self._selected_server_key = event.server_key
        self._sync_children()

    def on_mcp_inspector_hub_action_requested(
        self, event: MCPInspector.HubActionRequested
    ) -> None:
        event.stop()
        if event.action is HubAction.VIEW_DETAILS and event.server_key:
            self._selected_server_key = event.server_key
            self.set_mode("servers")
            self._sync_children()
        elif event.action is HubAction.OPEN_TOOL_CATALOG:
            self.set_mode("tools")
        elif event.action is HubAction.OPEN_AUDIT:
            self.set_mode("audit")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/UI/test_mcp_workbench.py -v`
Expected: all PASS

- [ ] **Step 5: Run all new + neighboring tests**

Run: `.venv/bin/pytest Tests/MCP/test_readiness_model.py Tests/MCP/test_readiness_derivation.py Tests/MCP/test_redaction.py Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_unified_mcp_panel.py -v`
Expected: all PASS (legacy panel tests untouched and green)

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/test_mcp_workbench.py
git commit -m "feat(mcp-hub): assemble rail+canvas+inspector workbench with mode switching"
```

---

### Task 8: Rewire `MCPScreen` (mode chips, bindings, tolerant state)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py` (full replacement of the 77-line file)
- Test: append to `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: `MCPWorkbench`, `MCP_HUB_MODES` (Task 7); `BaseAppScreen`, `DestinationModeStrip` (existing).
- Produces: `MCPScreen(BaseAppScreen)` with mode-chip buttons (`Button.mcp-mode-chip`, ids from `MCP_HUB_MODES[mode]["button_id"]`), number-key bindings 1-4, state persisted under `"mcp_hub_view_state"` (legacy `"unified_mcp_view_state"` read once for source carry-over), `handle_runtime_backend_changed` triggering `workbench.reload()`.

- [ ] **Step 1: Write the failing tests (append to `Tests/UI/test_mcp_workbench.py`)**

```python
# append to Tests/UI/test_mcp_workbench.py
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen


class _StubApp:
    unified_mcp_service = None


def test_screen_hosts_workbench_with_mode_action_and_tolerant_restore():
    screen = MCPScreen(_StubApp())
    # New surface: workbench host + mode action (old screen has mcp_panel, no workbench).
    assert hasattr(screen, "workbench")
    assert not hasattr(screen, "mcp_panel")
    assert callable(getattr(screen, "action_mcp_mode", None))
    # Never crashes on legacy shape, garbage, or empty state pre-mount.
    screen.restore_state({"unified_mcp_view_state": {"selected_source": "server"}})
    screen.restore_state({"mcp_hub_view_state": {"mode": "tools"}})
    screen.restore_state({})
    state = screen.save_state()
    assert isinstance(state, dict)


def test_mcp_hub_modes_registry_is_complete():
    from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCP_HUB_MODES

    assert list(MCP_HUB_MODES) == ["servers", "tools", "permissions", "audit"]
    for spec in MCP_HUB_MODES.values():
        assert spec["label"] and spec["button_id"].startswith("mcp-mode-")
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `.venv/bin/pytest Tests/UI/test_mcp_workbench.py -v -k "screen or registry"`
Expected: the screen test FAILS (`assert not hasattr(screen, "mcp_panel")` — the old screen still builds `UnifiedMCPPanel` and has no `workbench`/`action_mcp_mode`); the registry test passes (Task 7 defined `MCP_HUB_MODES`)

- [ ] **Step 3: Replace `mcp_screen.py`**

```python
# tldw_chatbook/UI/Screens/mcp_screen.py
"""MCP destination shell: mode strip + rail/canvas/inspector workbench."""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench
from ..Navigation.base_app_screen import BaseAppScreen

_MODE_BY_BUTTON_ID = {spec["button_id"]: mode for mode, spec in MCP_HUB_MODES.items()}


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, and audit surface."""

    BINDINGS = [
        Binding("1", "mcp_mode('servers')", "Servers", show=False),
        Binding("2", "mcp_mode('tools')", "Tools", show=False),
        Binding("3", "mcp_mode('permissions')", "Permissions", show=False),
        Binding("4", "mcp_mode('audit')", "Audit", show=False),
    ]

    DEFAULT_CSS = """
    Button.mcp-mode-chip {
        width: auto;
        min-width: 8;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
    .mcp-mode-chip.is-active {
        border: none;
        text-style: bold underline;
    }
    """

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "mcp", **kwargs)
        self.workbench: MCPWorkbench | None = None

    def compose_content(self) -> ComposeResult:
        with Vertical(id="mcp-shell"):
            yield Static("MCP", id="mcp-title", classes="ds-destination-header")
            yield Static(
                "Manage MCP servers, scoped tools, permissions, and audit readiness.",
                id="mcp-purpose",
                classes="destination-purpose",
            )
            with DestinationModeStrip(id="mcp-mode-strip", classes="destination-mode-strip"):
                for mode, spec in MCP_HUB_MODES.items():
                    chip = Button(
                        spec["label"],
                        id=spec["button_id"],
                        classes="mcp-mode-chip console-action-subdued",
                        compact=True,
                    )
                    chip.set_class(mode == "servers", "is-active")
                    yield chip
            self.workbench = MCPWorkbench(self.app_instance, id="mcp-hub-workbench")
            self.workbench.set_initial_view_state(self._initial_view_state())
            yield self.workbench

    def _initial_view_state(self) -> dict[str, Any] | None:
        state = self.state_data.get("mcp_hub_view_state")
        if isinstance(state, dict):
            return state
        legacy = self.state_data.get("unified_mcp_view_state")
        return legacy if isinstance(legacy, dict) else None

    def _activate_mode(self, mode: str) -> None:
        if self.workbench is None:
            return
        self.workbench.set_mode(mode)
        for candidate, spec in MCP_HUB_MODES.items():
            chips = list(self.query(f"#{spec['button_id']}"))
            if chips:
                chips[0].set_class(candidate == self.workbench.active_mode, "is-active")

    def action_mcp_mode(self, mode: str) -> None:
        self._activate_mode(mode)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode = _MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode is None:
            return
        event.stop()
        self._activate_mode(mode)

    def save_state(self) -> dict[str, Any]:
        state = super().save_state()
        if self.workbench:
            state["mcp_hub_view_state"] = self.workbench.get_view_state()
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        if self.workbench:
            self.workbench.set_initial_view_state(self._initial_view_state())

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        _ = runtime_backend
        if self.workbench:
            self.run_worker(
                self.workbench.reload(),
                name="mcp-screen-runtime-refresh",
                group="mcp-screen-runtime-refresh",
                exclusive=True,
            )
```

- [ ] **Step 4: Run the appended tests + full new-file set**

Run: `.venv/bin/pytest Tests/UI/test_mcp_workbench.py Tests/UI/test_unified_mcp_panel.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/mcp_screen.py Tests/UI/test_mcp_workbench.py
git commit -m "feat(mcp-hub): rewire MCP screen to mode chips + workbench with tolerant state restore"
```

---

### Task 9: Shared CSS, build, geometry/focus verification, QA captures

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (append), then rebuild `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: appended geometry test in `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: widget ids/classes from Tasks 4-8.
- Produces: app-bundle styling; verified geometry; QA screenshots for user approval.

- [ ] **Step 1: Append shared styles to `_agentic_terminal.tcss`**

```css
/* --- MCP Hub (Phase 1 workbench) ------------------------------------- */

#mcp-mode-strip Button.mcp-mode-chip {
    width: auto;
    min-width: 10;
    height: 1;
    padding: 0 1;
    border: none;
}

#mcp-mode-strip .mcp-mode-chip.is-active {
    text-style: bold underline;
}

#mcp-hub-rail {
    width: 3fr;
    min-width: 24;
}

#mcp-hub-rail .mcp-rail-heading {
    height: 1;
    min-height: 1;
    margin-top: 1;
}

#mcp-hub-rail Button.mcp-rail-row.is-active {
    text-style: bold;
}

#mcp-hub-canvas {
    width: 5fr;
    min-width: 38;
}

#mcp-hub-inspector {
    width: 3fr;
    min-width: 28;
}

#mcp-overview-summary,
#mcp-detail-title {
    height: 1;
    min-height: 1;
}

#mcp-overview-callouts .ds-recovery-callout {
    margin-top: 1;
}
```

- [ ] **Step 2: Rebuild the bundle and verify**

Run: `python3 tldw_chatbook/css/build_css.py && grep -c "mcp-hub-rail" tldw_chatbook/css/tldw_cli_modular.tcss`
Expected: build succeeds; grep prints ≥ 1

- [ ] **Step 3: Append the geometry pilot test (plain-Vertical clipping lesson)**

```python
# append to Tests/UI/test_mcp_workbench.py
@pytest.mark.asyncio
async def test_workbench_panes_have_nonzero_geometry():
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for selector in ("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"):
            widget = app.query_one(selector)
            assert widget.size.width > 0, f"{selector} has zero width"
            assert widget.size.height > 0, f"{selector} has zero height"
        table = app.query_one("#mcp-servers-table")
        assert table.size.height > 0, "servers table clipped to zero height"
```

- [ ] **Step 4: Run the full Phase 1 test set one final time**

Run: `.venv/bin/pytest Tests/MCP/test_readiness_model.py Tests/MCP/test_readiness_derivation.py Tests/MCP/test_redaction.py Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_non_obscuring_focus_contract.py -v`
Expected: all PASS (focus-contract test confirms the new flat buttons comply)

- [ ] **Step 5: Screenshot QA (requires user approval before merge)**

Follow the established textual-serve capture recipe (see memory: Console rail IA / Notes rebuild — live capture at 2050×1240 with the real app stylesheet, not harness CSS). Capture: (a) Servers mode overview with ≥2 local profiles in mixed readiness states, (b) server detail for a local profile, (c) built-in server detail, (d) each placeholder mode, (e) Advanced runner open. Present captures to the user for explicit approval — the per-screen approval gate applies.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_mcp_workbench.py
git commit -m "style(mcp-hub): shared workbench styles, geometry test, CSS bundle rebuild"
```

---

## Out of scope for Phase 1 (spec §16 phases 2-6)

Add-server wizard and any mutation UX; credential slot editing; connect/test/refresh buttons wired outside Advanced; permission store; chat bridge; execution log; server audit findings; retiring `unified_mcp_panel.py`/`unified_mcp_sections.py` (Phase 6). The spec correction from Verified Fact 4 (built-in row is stdio-only) must be applied to spec §7 wording during Phase 2 planning.
