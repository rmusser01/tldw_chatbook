# MCP Hub Redesign — Phase 3 Implementation Plan (Tools Mode)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Tools mode placeholder into a working cross-server tool catalog with a schema-driven Test Tool runner, diagnostic empty states, an execution log (pulled forward from spec §10), and the Phase 3 UX batch (canvas layout, focus outlines, CHECKING time expectation, catalog batch reads).

**Architecture:** A new pure derivation module (`MCP/hub_tool_catalog.py`) folds local discovery snapshots (which carry real `inputSchema` — verified), built-in inventory (schema-less — verified), and server-source inventory/registry passthrough dicts (untyped extras read defensively — verified) into one `HubTool` list. New typed control-plane execute seams (`test_external_tool`, `test_builtin_tool`) wrap the verified-but-unwired `MCPClient.call_tool` and the existing built-in `execute_tool`, with wall-clock timeouts and execution-log recording — these are the exact seams the agent runtime's `MCPToolProvider` (task-201, seam reserved in `Agents/tool_catalog.py:4`) will consume later. UI: a Tools canvas (grouped catalog + filters + diagnostic empty state) and an inspector tool view hosting the schema-generated parameter form with raw-JSON fallback.

**Tech Stack:** Python ≥3.11, Textual 8.2.7 (pinned), pytest + pytest-asyncio. Spec: `Docs/superpowers/specs/2026-07-13-mcp-hub-redesign-design.md` §8/§10/§16.3. UX inputs: `Docs/Design/mcp-hub-phase3-ux-inputs.md`.

## Global Constraints

- Tests: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest <files>` from `.worktrees/mcp-hub-phase3`. No `timeout` shell command. Only plan-named test files must pass; the 2 pre-existing Library snapshot-timeout failures in `Tests/UI/test_destination_visual_parity_correction.py` stay untouched.
- CSS: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, rebuild bundle via `python3 tldw_chatbook/css/build_css.py`, commit both; never hand-merge the bundle. App CSS beats widget DEFAULT_CSS; widget DEFAULT_CSS is harness-only geometry baseline.
- Textual 8.2.7 verified rules (all bitten before): Message classes in `MCP*`-named widgets need explicit `namespace=`; `Select.NULL` is the no-selection sentinel (`Select.BLANK` is not an API; placeholder `(label, False)` self-consistent; handlers treat both as no-selection); `Select` posts `Changed` for its constructor value at mount (guard echoes one-shot); `Collapsible` fires `Toggled` at mount when constructed expanded; Buttons need BOTH `text-align: left` AND `content-align: left middle`; round borders need ≥2 lines; AWAIT `remove_children()` before re-mounting id-bearing children; `run_worker` has `exit_on_error=True` — uncaught worker exceptions are app panics; user/remote text into Statics/labels: `markup=False` or `escape_markup`; user-supplied strings never in widget ids (index-based ids + lookup lists).
- Tool identity everywhere: `(server_key, tool_name)` where `server_key` is the Phase 1 format (`local:<id>`, `builtin:tldw_chatbook`, `server:<t>/<ext>` — server-source external tools are NOT in Phase 3 catalog scope; server-source tools come from the target's inventory keyed `server:<target>`).
- Governance: never re-gate in the control plane — `LocalMCPControlService` methods self-gate (`_require_allowed`); the new external-execute method gates with the EXISTING `mcp.external_profiles.trigger.local` (same trust class as `test_external_profile` — verified no external tool-execute id exists).
- Redaction (`MCP/redaction.py`, frozen): applied to logged args/results and any displayed result payloads.
- Conventional commits ending with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Work in `.worktrees/mcp-hub-phase3` on branch `claude/mcp-hub-phase3` only. Frozen: `unified_mcp_panel.py`, `unified_mcp_sections.py`, `MCP/redaction.py`, `Agents/*` (read-only reference).

## Verified Facts (planning-time, 2026-07-14)

1. **`inputSchema` survives end-to-end for external local tools**: `_tool_from_payload` keeps it (`client.py:25-30`), `get_server_tools` emits `{name, description, inputSchema}` (`client.py:682-701`), `describe_server` snapshots store it verbatim to disk (`local_store.py:678`). Test fixtures confirm.
2. **External tool execution does not exist**: `MCPClient.call_tool(server_id, tool_name, arguments) -> dict` (`client.py:565`) has ZERO callers; requires a connected session (`self.sessions`); returns `{"result": ...}` or `{"error": "Server X not connected"}`. `run_action("tool.execute")` executes BUILT-IN runtime tools only (`unified_control_plane_service.py:917` → `local_service.execute_tool` → `runtime_delegate`). Connect-if-needed pattern precedent: `_describe_profile` (`local_control_service.py:615-618`). Client attribute: `local_control_service.py:77` `self.client`, lazy `_get_client()` (`:573`).
3. **Built-in tools have NO parameter schema**: inventory entries are AST-derived `{name, description}` only (`server.py:40-69`); the raw-JSON fallback form is the only honest input path for them.
4. **Server-source**: catalog listing reachable via `load_section("inventory")` (raw dict passthrough — `{name, description}` guaranteed, extras like `risk_class`/`capabilities`/`inputSchema` flow through UNTYPED only if the backend sends them; fixtures show none). Registry entries via advanced section: `{tool_name, display_name, module}`. **Server-source tool EXECUTION is net-new** (no service wrapper, no action id — `mcp.runtime.trigger.server` doesn't exist) → **deferred to Phase 4** (recorded, with the per-tool `blocked` annotation which also has no unified-path field; `ToolInfo.canExecute` exists only on the unwired `/api/v1/tools` surface).
5. **No JSON-schema→form generator exists** (Evals dialogs are hardcoded); `Widgets/form_components.py:14` `create_form_field(label, field_id, field_type=...)` is the per-widget building block. **No reusable JSONL record writer exists.**
6. **Agent runtime (PR #623 + Plan B now on dev)**: consumes tools via `ToolCatalogRegistry`/`ToolProvider` (`Agents/tool_catalog.py`), NOT `ToolExecutor`. T3's typed seams (`test_external_tool`/`test_builtin_tool` + the execution log) are designed as the shared gate/execute/record path a future `MCPToolProvider` calls — keep their signatures UI-free.
7. Phase 2 seams available: `local_external_catalog()` (records incl. `discovery_snapshot` + `runtime_state`), `_run_local_lifecycle` timeout/recording pattern, `as_checking`, `STATE_CSS_CLASSES`, in-flight guard conventions, `hub_lifecycle_timeout_seconds` config key.

## File Structure

| File | Responsibility |
|---|---|
| Create `tldw_chatbook/MCP/execution_log.py` | Bounded JSONL execution log: append + rotation + read; redaction; arg-capture setting |
| Create `tldw_chatbook/MCP/hub_tool_catalog.py` | Pure derivation: `HubTool` dataclass + builders from local/builtin/server payloads + filters |
| Modify `tldw_chatbook/MCP/local_control_service.py` | `execute_external_tool(profile_id, tool_name, arguments)` (self-gated, connect-if-needed) |
| Modify `tldw_chatbook/MCP/unified_control_plane_service.py` | Typed `test_external_tool` / `test_builtin_tool` (timeout + log recording); `local_catalog_bundle` consumption |
| Modify `tldw_chatbook/MCP/local_store.py` | `get_catalog_bundle()` batch accessor (one load for profiles+snapshots+runtime state) |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py` | JSON-schema→fields parse + `MCPSchemaForm` widget with raw-JSON fallback |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py` | Tools canvas: grouped catalog table, filter bar, diagnostic empty state |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` | Tool detail view + hosted schema form + result rendering + busy/cancel |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` | Tools canvas registration (placeholder out), tool selection state, execute orchestration, CHECKING "(up to Ns)" |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py` + `mcp_tools_mode.py` | Canvas layout fix (auto-height capped table; callouts hug content) |
| Modify `tldw_chatbook/UI/Screens/mcp_screen.py` | `t` binding (test selected tool), footer hints update |
| Modify CSS source + bundle | Tools-mode styles, quiet content-pane focus outline, layout-fix rules |
| Tests | `Tests/MCP/test_execution_log.py`, `Tests/MCP/test_hub_tool_catalog.py`, `Tests/MCP/test_control_plane_tool_execute.py`, extend `Tests/MCP/test_local_store_runtime_state.py` (bundle accessor), `Tests/UI/test_mcp_schema_form.py`, `Tests/UI/test_mcp_tools_mode.py`, extend `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py` |

Execution order: T1 (log) → T2 (catalog derivation + bundle accessor) → T3 (execute seams) → T4 (schema form) → T5 (tools canvas) → T6 (inspector runner) → T7 (UX batch) → T8 (bindings/CSS/gate).

---

### Task 1: Execution log (`MCP/execution_log.py`)

**Files:**
- Create: `tldw_chatbook/MCP/execution_log.py`
- Test: `Tests/MCP/test_execution_log.py`

**Interfaces:**
- Consumes: `MCP/redaction.py` (`redact_mapping`, frozen); `get_cli_setting` (arg-capture: `get_cli_setting("mcp", "log_tool_arguments", True)`).
- Produces (exact — T3 records through this; Phase 5's Audit mode reads it):
  - `@dataclass(frozen=True) ExecutionRecord: ts: str; server_key: str; tool_name: str; initiator: str  # "test" now, "chat"/"agent" later; decision: str  # "allowed" fixed for tests now; ok: bool; duration_ms: int; error: str | None; arguments: dict | None; result_excerpt: str | None`
  - `class MCPExecutionLog:` `__init__(self, path: Path, *, max_records_per_file: int = 500)`; `append(self, record: ExecutionRecord) -> None` (JSONL append; when the active file reaches max records, rotate: `<path>` → `<path>.1` (replacing any existing `.1`), start fresh — two generations total); `read_recent(self, limit: int = 200) -> list[dict]` (newest first, spanning both generations); arguments are stored ALREADY-REDACTED by the caller — the log also defensively applies `redact_mapping` to any dict `arguments` on append; `result_excerpt` is caller-truncated (≤500 chars).
  - Module helper: `build_record(*, server_key, tool_name, initiator, ok, duration_ms, error=None, arguments=None, result_excerpt=None, capture_args=True) -> ExecutionRecord` — stamps `ts` (`datetime.now(timezone.utc).isoformat()`), drops `arguments` to None when `capture_args` is False, redacts + copies when kept.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_execution_log.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.MCP.execution_log import ExecutionRecord, MCPExecutionLog, build_record


@pytest.fixture()
def log(tmp_path: Path) -> MCPExecutionLog:
    return MCPExecutionLog(tmp_path / "mcp_execution_log.jsonl", max_records_per_file=3)


def _record(tool: str = "search_docs", **kw) -> ExecutionRecord:
    defaults = dict(server_key="local:docs", tool_name=tool, initiator="test",
                    ok=True, duration_ms=12, arguments={"query": "x"})
    defaults.update(kw)
    return build_record(**defaults)


def test_append_and_read_recent_roundtrip(log, tmp_path):
    log.append(_record("a"))
    log.append(_record("b", ok=False, error="boom"))
    rows = log.read_recent()
    assert [r["tool_name"] for r in rows] == ["b", "a"]  # newest first
    assert rows[0]["error"] == "boom" and rows[0]["ok"] is False
    raw = (tmp_path / "mcp_execution_log.jsonl").read_text().strip().splitlines()
    assert all(json.loads(line) for line in raw)  # valid JSONL


def test_rotation_keeps_two_generations(log, tmp_path):
    for i in range(7):  # cap 3 per file -> rotations
        log.append(_record(f"t{i}"))
    active = tmp_path / "mcp_execution_log.jsonl"
    rotated = tmp_path / "mcp_execution_log.jsonl.1"
    assert active.exists() and rotated.exists()
    names = [r["tool_name"] for r in log.read_recent(limit=100)]
    assert names[0] == "t6"          # newest first
    assert len(names) <= 6           # bounded: at most two generations
    assert "t0" not in names         # oldest generation dropped


def test_arguments_redacted_and_capture_off_drops_them():
    kept = build_record(server_key="local:docs", tool_name="t", initiator="test",
                        ok=True, duration_ms=1,
                        arguments={"api_key": "sk-123", "query": "ok"})
    assert kept.arguments["api_key"] == "***"
    assert kept.arguments["query"] == "ok"
    dropped = build_record(server_key="local:docs", tool_name="t", initiator="test",
                           ok=True, duration_ms=1,
                           arguments={"api_key": "sk-123"}, capture_args=False)
    assert dropped.arguments is None


def test_read_recent_survives_corrupt_line(log, tmp_path):
    log.append(_record("good"))
    with (tmp_path / "mcp_execution_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    log.append(_record("after"))
    names = [r["tool_name"] for r in log.read_recent()]
    assert "good" in names and "after" in names  # corrupt line skipped, no crash
```

- [ ] **Step 2: Run to verify failure** — `PYTHONPATH=. .../pytest Tests/MCP/test_execution_log.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/MCP/execution_log.py
"""Bounded JSONL log of MCP tool executions (Hub tests now; chat/agents later).

Append-only with two-generation size rotation (crash-safe: a torn final line
is skipped on read). Arguments are redacted before they ever reach disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_chatbook.MCP.redaction import redact_mapping

_RESULT_EXCERPT_LIMIT = 500


@dataclass(frozen=True)
class ExecutionRecord:
    ts: str
    server_key: str
    tool_name: str
    initiator: str
    decision: str
    ok: bool
    duration_ms: int
    error: str | None = None
    arguments: dict[str, Any] | None = None
    result_excerpt: str | None = None


def build_record(*, server_key: str, tool_name: str, initiator: str, ok: bool,
                 duration_ms: int, error: str | None = None,
                 arguments: dict[str, Any] | None = None,
                 result_excerpt: str | None = None,
                 decision: str = "allowed",
                 capture_args: bool = True) -> ExecutionRecord:
    """Build a redacted, timestamped execution record.

    Args:
        server_key: Hub server key ("local:<id>" / "builtin:tldw_chatbook").
        tool_name: Tool invoked.
        initiator: "test" (Hub-initiated) — "chat"/"agent" in later phases.
        ok: Whether the execution succeeded.
        duration_ms: Wall-clock duration.
        error: Error summary on failure (caller-truncated).
        arguments: Call arguments; redacted here, dropped when capture_args
            is False.
        result_excerpt: Caller-provided excerpt; truncated to 500 chars.
        decision: Permission decision ("allowed" for user-initiated tests).
        capture_args: The [mcp] log_tool_arguments setting value.

    Returns:
        A frozen ExecutionRecord safe to persist.
    """
    kept_arguments: dict[str, Any] | None = None
    if capture_args and isinstance(arguments, dict):
        kept_arguments = redact_mapping(arguments)
    excerpt = None
    if result_excerpt is not None:
        excerpt = str(result_excerpt)[:_RESULT_EXCERPT_LIMIT]
    return ExecutionRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        server_key=server_key, tool_name=tool_name, initiator=initiator,
        decision=decision, ok=ok, duration_ms=int(duration_ms),
        error=(str(error)[:300] if error else None),
        arguments=kept_arguments, result_excerpt=excerpt,
    )


class MCPExecutionLog:
    """Two-generation bounded JSONL store for ExecutionRecords."""

    def __init__(self, path: Path, *, max_records_per_file: int = 500) -> None:
        self.path = Path(path)
        self.max_records_per_file = max_records_per_file

    def append(self, record: ExecutionRecord) -> None:
        """Append one record, rotating generations at the size cap."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._count_lines(self.path) >= self.max_records_per_file:
            rotated = self.path.with_name(self.path.name + ".1")
            self.path.replace(rotated)
        payload = asdict(record)
        if isinstance(payload.get("arguments"), dict):
            payload["arguments"] = redact_mapping(payload["arguments"])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def read_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return up to `limit` records, newest first, across generations."""
        rows: list[dict[str, Any]] = []
        rotated = self.path.with_name(self.path.name + ".1")
        for source in (rotated, self.path):  # oldest generation first
            if not source.exists():
                continue
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn/corrupt line — skip, never crash
        rows.reverse()
        return rows[:limit]

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
```

- [ ] **Step 4: Run** — file green + `Tests/MCP/ -q` no regressions.
- [ ] **Step 5: Commit** — `feat(mcp): bounded JSONL execution log with redaction and rotation`

---

### Task 2: Catalog derivation (`MCP/hub_tool_catalog.py`) + store batch accessor

**Files:**
- Create: `tldw_chatbook/MCP/hub_tool_catalog.py`
- Modify: `tldw_chatbook/MCP/local_store.py` (batch accessor), `tldw_chatbook/MCP/unified_control_plane_service.py` (`local_external_catalog` uses it)
- Test: `Tests/MCP/test_hub_tool_catalog.py`, extend `Tests/MCP/test_local_store_runtime_state.py`

**Interfaces:**
- Consumes: local catalog record shape (Phase 2 `local_external_catalog()` items: profile fields + `discovery_snapshot` (tools with `inputSchema`) + `is_connected` + `runtime_state`); builtin inventory `{name, description}` from `local_service.get_inventory()`; server inventory payload `{tools: [raw dicts]}`.
- Produces (exact — T4/T5/T6 depend on these):
  - `@dataclass(frozen=True) HubTool: server_key: str; server_label: str; source: str  # local|builtin|server; name: str; description: str; input_schema: dict | None; tags: tuple[str, ...]; stale: bool; executable: bool` — `tool_id` property → `f"{server_key}::{name}"`.
  - `local_tools_from_record(record: dict) -> list[HubTool]` — from a catalog record: tools from `discovery_snapshot["tools"]` (schema kept when a non-empty dict); `stale = not record.get("is_connected")`; `executable=True`; no snapshot → `[]`.
  - `builtin_tools_from_inventory(inventory: dict) -> list[HubTool]` — `{name, description}` entries; `input_schema=None`; `executable=True`; `stale=False`.
  - `server_tools_from_inventory(payload: dict, *, target_id: str, target_label: str) -> list[HubTool]` — defensive raw-dict reads: `name` (skip nameless), `description`, `inputSchema` when a dict, tags from `risk_class`/`capabilities` extras WHEN present (strings only, lowercased, ≤5); `executable=False` (server-source execution is Phase 4 — recorded); `stale=False`.
  - `filter_tools(tools, *, server_key=None, text=None) -> list[HubTool]` — server filter exact; text filter case-insensitive over name+description.
  - `LocalMCPStore.get_catalog_bundle() -> dict` — ONE `load()` returning `{"profiles": [profile.to_dict()...], "discovery_snapshots": {id: dict}, "profile_runtime_state": {id: dict}}` (UX-inputs batching item: replaces the 2N+1 loads).
  - `local_external_catalog()` on the control plane switches to the bundle (same output shape as today — records with `discovery_snapshot`/`is_connected`/`runtime_state`; `is_connected` still read from the client sessions).

- [ ] **Step 1: Failing tests**

```python
# Tests/MCP/test_hub_tool_catalog.py
from __future__ import annotations

from tldw_chatbook.MCP.hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    filter_tools,
    local_tools_from_record,
    server_tools_from_inventory,
)


def _local_record(connected=True, tools=None):
    return {
        "profile_id": "docs", "command": "npx",
        "is_connected": connected,
        "discovery_snapshot": {"tools": tools if tools is not None else [
            {"name": "search", "description": "Search docs.",
             "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            {"name": "bare", "description": "", "inputSchema": {}},
        ]},
    }


def test_local_tools_carry_schema_and_stale_flag():
    tools = local_tools_from_record(_local_record(connected=False))
    assert [t.name for t in tools] == ["search", "bare"]
    assert tools[0].input_schema["properties"]["q"]["type"] == "string"
    assert tools[1].input_schema is None  # empty schema dict -> None
    assert all(t.stale and t.executable and t.source == "local" for t in tools)
    assert tools[0].server_key == "local:docs"
    assert tools[0].tool_id == "local:docs::search"


def test_local_record_without_snapshot_yields_nothing():
    assert local_tools_from_record({"profile_id": "x", "discovery_snapshot": None}) == []


def test_builtin_tools_have_no_schema_but_execute():
    tools = builtin_tools_from_inventory({"tools": [
        {"name": "chat_with_llm", "description": "Chat."}]})
    assert tools[0].input_schema is None and tools[0].executable
    assert tools[0].server_key == "builtin:tldw_chatbook"


def test_server_tools_read_extras_defensively():
    payload = {"tools": [
        {"name": "web_search", "description": "Search.",
         "risk_class": "High", "capabilities": ["Network", 7, "mutates"],
         "inputSchema": {"type": "object"}},
        {"description": "nameless — skipped"},
        "not-a-dict",
    ]}
    tools = server_tools_from_inventory(payload, target_id="main", target_label="Main")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.server_key == "server:main" and tool.server_label == "Main"
    assert tool.tags == ("high", "network", "mutates")
    assert tool.input_schema == {"type": "object"}
    assert tool.executable is False  # server-source execution is Phase 4


def test_filter_by_server_and_text():
    tools = local_tools_from_record(_local_record()) + builtin_tools_from_inventory(
        {"tools": [{"name": "create_note", "description": "Notes."}]})
    assert [t.name for t in filter_tools(tools, server_key="builtin:tldw_chatbook")] == ["create_note"]
    assert [t.name for t in filter_tools(tools, text="SEARCH")] == ["search"]
```

```python
# append to Tests/MCP/test_local_store_runtime_state.py
def test_get_catalog_bundle_single_shape(store):
    _save_profile(store)
    store.save_discovery_snapshot("docs", {"tools": [{"name": "a"}], "resources": [], "prompts": []})
    store.save_profile_runtime_state("docs", {"ok": True})
    bundle = store.get_catalog_bundle()
    assert [p["profile_id"] for p in bundle["profiles"]] == ["docs"]
    assert bundle["discovery_snapshots"]["docs"]["tools"][0]["name"] == "a"
    assert bundle["profile_runtime_state"]["docs"]["ok"] is True
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — `hub_tool_catalog.py` (~120 lines, pure; tags builder: `risk_class` string lowercased first, then string entries of `capabilities` lowercased, cap 5, tuple); `get_catalog_bundle` on the store (one `self.load()`, mirror the accessor docstring style); rewire `local_external_catalog()` to one bundle call + per-record merge (keep `is_connected` from `client.sessions` — read the existing implementation and preserve its output shape byte-for-byte; the existing Phase 2 tests must stay green unmodified).
- [ ] **Step 4: Run** — new file + `Tests/MCP/ -q` green (Phase 2 lifecycle/catalog tests prove the rewire preserved shape).
- [ ] **Step 5: Commit** — `feat(mcp): hub tool catalog derivation and single-load store bundle`

---

### Task 3: Typed execute seams (external + builtin) with log recording

**Files:**
- Modify: `tldw_chatbook/MCP/local_control_service.py`, `tldw_chatbook/MCP/unified_control_plane_service.py`
- Test: `Tests/MCP/test_control_plane_tool_execute.py`

**Interfaces:**
- Consumes: `MCPClient.call_tool(server_id, tool_name, arguments)` (session-gated; returns `{"result": ...}` or `{"error": ...}`); connect-if-needed precedent (`_describe_profile`); T1 log; `hub_lifecycle_timeout_seconds`.
- Produces (exact — the future agent `MCPToolProvider` calls these; keep them UI-free):
  - `LocalMCPControlService.execute_external_tool(self, profile_id: str, tool_name: str, arguments: dict | None = None) -> dict` — self-gated with `self._require_allowed(action_id="mcp.external_profiles.trigger.local", ...)` exactly as `test_external_profile` does (copy its gating call verbatim); connect-if-needed (`profile_id not in getattr(client, "sessions", {})` → `await self.connect_profile(profile_id)`); then `await client.call_tool(profile_id, tool_name, arguments or {})`; an `{"error": ...}` response raises `RuntimeError(payload["error"])`; returns the raw result dict.
  - Control plane: `async test_hub_tool(self, server_key: str, tool_name: str, arguments: dict | None = None) -> dict` — routes by key prefix: `local:` → `execute_external_tool`; `builtin:` → `self.local_service.execute_tool(tool_name, arguments)` (existing built-in path, self-gated); other prefixes → `ValueError("Tool testing for server-source tools arrives in Phase 4.")`. Wall-clock `asyncio.wait_for` with the shared `_lifecycle_timeout()`; timeout → `RuntimeError(f"Timed out after {timeout:.0f}s")`.
  - Recording: every call appends to the execution log via a new lazy property `self.execution_log` (path: same directory as the local store file — derive `Path(store.path).with_name("mcp_execution_log.jsonl")` via `getattr(self.local_service, "store", None)`; when no store, recording silently skips) using `build_record(server_key=..., tool_name=..., initiator="test", ok=..., duration_ms=..., error=..., arguments=..., result_excerpt=str(result)[:500], capture_args=get_cli_setting("mcp", "log_tool_arguments", True))`. Recording is best-effort (try/except + `logger.warning` — the Phase 2 masking lesson, apply from the start).

- [ ] **Step 1: Failing tests** — mirror `Tests/MCP/test_control_plane_lifecycle.py`'s harness style (FakeLocalService + real `LocalMCPStore` in tmp_path): (1) `test_hub_tool` on `local:docs` connects-if-needed then calls the fake client's call_tool and returns the result; (2) an `{"error": "..."}` client response raises RuntimeError AND a log record with `ok=False` lands in `mcp_execution_log.jsonl` next to the store; (3) `builtin:` routes to `execute_tool`; (4) unknown prefix raises ValueError mentioning Phase 4; (5) timeout path (gated fake, monkeypatched timeout 0.05) raises + records; (6) log-write failure does not mask the tool result (masking regression, RED-first: fake log path unwritable → result still returned). Write all six concretely in the file (use the lifecycle test file as the template for fakes; the fake local service gets `client` with `sessions` dict + `async call_tool`, `connect_profile` recording calls, `execute_tool` for builtin, and a real store).
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** per Interfaces (read `test_external_profile`'s gating lines and mirror them exactly in `execute_external_tool`).
- [ ] **Step 4: Run** — new file + `Tests/MCP/ -q` green.
- [ ] **Step 5: Commit** — `feat(mcp): typed hub tool execution seams with timeout and execution-log recording`

---

### Task 4: Schema-driven form (`UI/MCP_Modules/mcp_schema_form.py`)

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py`
- Test: `Tests/UI/test_mcp_schema_form.py`

**Interfaces:**
- Consumes: nothing MCP-specific (pure + Textual).
- Produces (exact — T6 hosts this):
  - `@dataclass(frozen=True) SchemaField: name: str; kind: str  # string|number|integer|boolean|enum; required: bool; description: str; default: object | None; choices: tuple[str, ...] = ()`
  - `parse_schema(schema: dict | None) -> list[SchemaField] | None` — PURE. None/empty/non-dict schema → None. `type` must be "object" (else None). Each `properties` entry maps: `enum` list → kind "enum" (string choices); `type` string/number/integer/boolean → same kind; anything else (nested objects, arrays, oneOf, missing type) → **the whole parse returns None** (raw-fallback trigger — partial forms lie). `required` from the schema's `required` list; `default`/`description` passthrough.
  - `class MCPSchemaForm(Vertical):` `__init__(self, *, schema: dict | None, **kwargs)` — renders parsed fields (string/number/integer → `Input` (number/integer get `type="number"` hint via placeholder), boolean → `Checkbox` (default honored), enum → `Select` of choices, required fields' labels suffixed " *"); when `parse_schema` returns None → a raw-mode `TextArea` (`#mcp-schema-raw`) seeded `"{}"` + a note Static "This tool's parameters can't be rendered as a form — edit raw JSON." (markup=False). Field widgets get index-based ids `#mcp-schema-field-<i>`.
  - `collect_arguments(self) -> dict` — form mode: coerce per kind (number → float, integer → int, boolean → checkbox value, empty optional strings omitted); coercion failure or missing required → `ValueError("<field>: must be a number." / "<field>: required.")`. Raw mode: `json.loads` (JSONDecodeError → `ValueError("Not valid JSON: ...")`); non-dict → ValueError.
  - `is_raw_mode: bool` property.

- [ ] **Step 1: Failing tests** — concrete file with: parse happy path (string+number+bool+enum+required), parse rejects nested object schema entirely (returns None), widget render (Input/Checkbox/Select present per field, required star in label), collect coercion (number "3.5"→3.5, integer "2"→2, bool, enum, omitted optional), collect errors (bad number ValueError names the field; missing required), raw fallback (unrenderable schema → `#mcp-schema-raw` exists, `collect_arguments` parses JSON, invalid JSON ValueError). ~8 tests, written out fully in the plan-executor's TDD pass (this brief carries the exact behaviors; write assertions to match the Interfaces text verbatim).
- [ ] **Step 2: RED.** — module absent.
- [ ] **Step 3: Implement** (~180 lines; `parse_schema` first as pure functions with its own tests passing before the widget).
- [ ] **Step 4: Run** — file + neighbors green.
- [ ] **Step 5: Commit** — `feat(mcp-hub): JSON-schema-driven parameter form with honest raw fallback`

---

### Task 5: Tools mode canvas (`UI/MCP_Modules/mcp_tools_mode.py`)

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (register real canvas; collect tools)
- Test: `Tests/UI/test_mcp_tools_mode.py`, extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: T2 `HubTool`/`filter_tools`; workbench snapshots + catalog records; readiness for empty-state diagnosis (`aggregate` of current snapshots).
- Produces:
  - `MCPToolsMode(Vertical)` with: filter bar (`#mcp-tools-filter-text` Input, `#mcp-tools-filter-server` Select of current servers + "All servers" via `Select.NULL` handling per the sentinel rules), catalog `DataTable` (`#mcp-tools-table`, columns **Tool | Server | Tags | Schema**, row key = `tool_id`, rows sorted by (server_label, name) so servers group visually; Tags cell = ", ".join(tags) or "—"; Schema cell = "form" / "raw"; stale tools get "(stale)" appended to the Server cell — plain `Text` cells, markup-safe), and a diagnostic empty state `#mcp-tools-empty` (`ds-recovery-callout`) shown when zero tools: message + ONE primary Button derived from the workbench-provided diagnosis (see below).
  - `async update_tools(self, tools: list[HubTool], *, empty_diagnosis: tuple[str, str] | None) -> None` — rebuilds rows (awaited remove/mount discipline for the empty-state container); `empty_diagnosis` = (message, action_key) where action_key ∈ {"add_server", "connect", "refresh"}; the Button posts `MCPToolsMode.EmptyActionRequested(action_key)` (namespace `mcp_tools_mode`).
  - `MCPToolsMode.ToolSelected(tool_id: str)` (namespace `mcp_tools_mode`) on row selection; filter changes re-filter client-side (the widget caches the last full list).
  - Workbench: `_collect_hub_tools() -> list[HubTool]` — local records (already loaded for snapshots — reuse, don't re-fetch) through `local_tools_from_record`, builtin inventory via `local_service.get_inventory()` (guarded getattr) through `builtin_tools_from_inventory`, and in server source the inventory payload through `server_tools_from_inventory`. Empty-state diagnosis mirrors the spec: no servers → ("No servers configured — add one to see its tools.", "add_server"); servers but none connected/discovered → ("No tools discovered yet — connect or refresh a server.", "connect"); otherwise refresh. Tools recomputed inside `_sync_children` (only when Tools mode is active OR cheaply always — compute from already-loaded data, no extra I/O). `EmptyActionRequested` routes: add_server → the existing add-form path; connect/refresh → switch to servers mode with a notify hint.
- Tests: rows render grouped/sorted with tags and schema column; stale annotation; filter by text and server; selection posts tool_id; empty diagnosis renders message + button and the button posts; workbench-level: tools appear when the fake catalog has snapshots, placeholder canvas is GONE (`#mcp-mode-canvas-tools` now hosts `MCPToolsMode`).

- [ ] **Steps 1-5**: failing tests → implement (canvas replaces the `MCP_HUB_MODES["tools"]` placeholder Vertical in the workbench's ContentSwitcher; keep the placeholder text for permissions/audit) → run (tools-mode file + workbench + servers-mode suites) → commit `feat(mcp-hub): tools-mode catalog with filters and diagnostic empty state`.

---

### Task 6: Inspector tool view + Test Tool runner

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Test: extend `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: T3 `test_hub_tool`; T4 `MCPSchemaForm`; T5 `ToolSelected`; in-flight guard + notify conventions.
- Produces:
  - `MCPInspector.show_tool(self, tool: HubTool | None) -> None` (async, under the existing `_refresh_lock`): renders into a dedicated `#mcp-inspector-tool` container (above the Advanced collapsible; hidden when None): tool name + server label (escaped), description (markup=False), tags line, schema availability ("Parameters: form" / "Parameters: raw JSON"), stale note when stale, and — when `tool.executable` — a **Test Tool** Button (`#mcp-inspector-test-tool`, tooltip "Run this tool with test arguments."); non-executable (server-source) tools show "Testing server-source tools arrives in Phase 4." (markup=False, `ds-field-row`).
  - Pressing Test Tool mounts an `MCPSchemaForm(schema=tool.input_schema)` + Run (`#mcp-inspector-test-run`) / Close (`#mcp-inspector-test-close`) buttons + result Static (`#mcp-inspector-test-result`, markup=False) inside `#mcp-inspector-tool`. Run: `collect_arguments()` (ValueError → result Static shows the message, no call), then posts `MCPInspector.ToolTestRequested(tool_id, arguments)` (namespace `mcp_inspector`).
  - Workbench: handler runs a guarded worker (`self._tool_test_in_flight: set[str]`, double-run → warning toast) calling `service.test_hub_tool(server_key, tool_name, arguments)` (split `tool_id` on `"::"`); result → `inspector.show_tool_result(ok=True, text=<redacted excerpt ≤500 chars via redact_mapping when dict → json.dumps>, duration_ms=...)`; failure → `show_tool_result(ok=False, text=str(exc))`. Result line format: `"OK · 123ms"` / `"Failed · 45ms"` + excerpt below. While in flight the Run button disables (re-enabled in `show_tool_result`).
  - Selection wiring: `on_mcp_tools_mode_tool_selected` → workbench resolves the `HubTool` from its cached list → `inspector.show_tool(tool)`; switching modes or servers clears the tool view (`show_tool(None)`).
- Tests: show_tool renders name/tags/schema-availability + Test button for executable, Phase-4 note for server-source; test flow end-to-end with a fake service (form collect → ToolTestRequested → fake `test_hub_tool` records call → result "OK ·" rendered; error path renders "Failed ·"; double-run guarded; ValueError from collect shows message without calling the service); raw-mode tool drives the raw TextArea path.

- [ ] **Steps 1-5**: failing tests → implement → run (inspector + workbench + tools-mode + schema-form suites) → commit `feat(mcp-hub): inspector tool detail and schema-driven Test Tool runner`.

---

### Task 7: UX batch — canvas layout, focus outline, CHECKING bound

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`, `mcp_tools_mode.py`, `mcp_workbench.py` (CHECKING copy), CSS source (+ bundle in T8)
- Test: extend `Tests/UI/test_mcp_servers_mode.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces (UX-inputs items, acceptance below):**
  - **Canvas layout**: servers-mode `#mcp-servers-table` and tools-mode `#mcp-tools-table` switch to `height: auto; max-height: 70%;` (both CSS layers) so callouts/empty-states hug the last row; geometry tests assert the callouts container's y-origin is within 3 rows of the table's last row for a 4-row table at 120×40 (the Phase 1 geometry-test pattern).
  - **Focus outline**: content panes (`#mcp-detail-scroll`, `#mcp-adv-scroll`, tools canvas scroll if any) get a quiet focus treatment — CSS `&:focus { border: none; background: $boost; }`-style rule replacing the dashed outline (match how Console content panes style focus — find one precedent with grep `:focus` in `_agentic_terminal.tcss` and reuse its idiom; document which). Verified via the bundled-CSS test family (rule presence + no `dashed` for these ids).
  - **CHECKING bound**: `as_checking` message gains the bound — workbench passes it: `as_checking(snap, f"{action}… (up to {int(self._service()._lifecycle_timeout() if hasattr(...) else 45)}s)")` — cleaner: workbench reads `float(get_cli_setting("mcp", "hub_lifecycle_timeout_seconds", 45))` directly and formats `"connect (up to 45s)"` as the action string (no readiness change needed). Test: in-flight snapshot message contains "(up to".
- [ ] **Steps 1-5**: failing tests → implement → run → commit `feat(mcp-hub): canvas hugging layout, quiet pane focus, CHECKING time bound`.

---

### Task 8: `t` binding, footer hints, CSS bundle, full gate

**Files:**
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py`, CSS source + rebuilt bundle
- Test: extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
  - `MCP_SHORTCUTS` gains `("t", "test tool")`; `Binding("t", "mcp_test_tool", "Test tool", show=False)` → switches to tools mode and, when a tool is selected, opens the test form (workbench method `open_test_for_selected_tool()`; no selection → notify "Select a tool first."). Footer registration unchanged otherwise.
  - CSS block appended (tools-mode geometry: filter bar heights, table max-height rules from T7, `#mcp-inspector-tool` spacing, quiet-focus rules) + `python3 tldw_chatbook/css/build_css.py`, both committed, fidelity check (regenerate → timestamp-only).
  - Full gate:
`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/ Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_tools_mode.py Tests/UI/test_mcp_schema_form.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_profile_form.py Tests/UI/test_mcp_server_mutations.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py -q`
Expected: all green except the 2 documented pre-existing Library failures.
- [ ] **Steps 1-5**: failing tests (binding + footer) → implement → bundle rebuild → full gate → commit `feat(mcp-hub): test-tool binding, footer hints, Phase 3 styles + bundle`.

**Post-task (controller-owned):** live screenshot QA (Phase 2 recipe; seeded isolated HOME needs a profile with a real schema-bearing snapshot — reuse `/private/tmp/tldw-qa-mcp-hub-p2-20260714` and extend the docs-server snapshot with an `inputSchema`) covering: tools catalog grouped with tags/schema columns, filters, diagnostic empty state, tool inspector + schema form (form AND raw modes), test run result (OK + failure), CHECKING with "(up to Ns)", hugging layout. User screenshot approval gates the PR.

---

## Out of scope (recorded)

Server-source tool EXECUTION + per-tool blocked/denied annotation (no unified-path seam or capability id — Phase 4, alongside the permission store whose UI shares the tool rows); permission state column on tool rows (Phase 4); Audit-mode UI over the execution log (Phase 5 — the log module ships now); chat/agent initiators in the log (Phase 5 / task-201); selective-import checkboxes and inspector-as-form-help (Phase 3 UX-inputs items deferred to Phase 4 planning — recorded); the remaining engineering backlog items not named in T2/T7 (mutations-panel remount, slot-delete confirm, cancel-isn't-failure, import regex alignment) — file as backlog tasks post-merge.
