# MCP Hub Redesign — Phase 4 (Permissions Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Permissions mode of the MCP Hub — schema-versioned permission store with kill switch, effective-state resolution with rug-pull definition-hash guard and high-risk inherit floor, the Space-cycling matrix canvas with policy preview, inspector rule explanation, and gate-aware Test Tool — plus the task-233 tuple-identity refactor of the execute path and the Tools-mode permission-state column deferred from Phase 3.

**Architecture:** A new pure-ish module `MCP/permission_store.py` owns the `mcp_permissions.json` store (spec §9 schema, v1) AND the pure effective-state resolution functions (precedence, hash guard, risk floor). The control plane grows typed methods (`effective_tool_states`, `set_tool_state`, `set_server_default`, `set_global_default`, `set_kill_switch`, `gate_tool_test`) that enforce the hash check and emit downgrade audit entries into the existing execution log. UI: new `UI/MCP_Modules/mcp_permissions_mode.py` canvas replaces the placeholder; Tools mode gains a State column; the inspector explains which rule produced an effective state; the Test Tool runner consults the gate (deny blocks, ask arms a confirm, allow runs).

**Tech Stack:** Python ≥3.11, Textual 8.2.7, JSON stores under the local-store directory, pytest (+ existing harness patterns).

## Global Constraints

- Store schema is spec §9 VERBATIM: `{"schema_version": 1, "kill_switch": false, "profiles": {"default": {"global_default": "ask", "servers": {"<source>:<server_id>": {"default": "ask", "tools": {"<tool_name>": {"state": "allow|ask|deny", "definition_hash": "…"}}}}}}`. File name `mcp_permissions.json`, sited next to the local store (`Path(store.path).with_name(...)`, same pattern as the execution log).
- Store I/O mirrors `LocalMCPStore`: `.tmp` temp file + `Path.replace()` atomic write, `json.dump(..., indent=2, sort_keys=True)`, `updated_at` stamp. Corruption/unknown `schema_version` → **back up the file (rename to `<name>.bak`) and start fresh with defaults — never crash** (spec §9 mandate; deliberately diverges from `LocalMCPStore`'s raise-on-corrupt policy — document this in the module docstring).
- UI states are **Inherit / Allow / Ask / Off**; store states are `allow|ask|deny` (`Off` ⇄ `deny`); **Inherit = absence of the key** (tool entry removed / server `default` absent / server entry absent). Global default has no Inherit (cycles Allow → Ask → Off); factory default `ask`.
- Precedence: tool override → server default → global default. Rug-pull: explicit Allow stores `definition_hash` = sha256 of canonical JSON `{"description": …, "inputSchema": …}` (mirror `_approval_fingerprint`'s `sort_keys=True, default=str, separators=(",", ":")` pattern at `local_control_service.py:743-754`); mismatch at resolution downgrades effective state to **ask**, flags `config_changed`, and emits ONE audit entry per transition (idempotent — the store persists a `config_changed: true` marker; re-allow clears it and stores the new hash). Audit copy: `f"{tool_name} definition changed since you allowed it — review and re-allow"`.
- High-risk inherit floor: a tool whose `tags` intersect `{"mutates", "process"}` with NO explicit tool-level state never inherits an Allow — effective state floors to **ask** with `risk_floored=True`.
- Kill switch lives in the store only (`kill_switch` field; runtime-mutable, NO config knob — spec §12). It gates **chat send-time assembly (Phase 5)**; the Hub's Test Tool is deliberately NOT kill-switch-gated (operator diagnostics) — document this in the toggle's tooltip and the plan-of-record comment.
- Tuple identity (task-233): after this phase, NOTHING in the execute path parses `"::"`. `tool_id` (`f"{server_key}::{name}"`) survives ONLY as a DataTable row key / display string. Messages and routing carry `server_key` + `tool_name` as separate fields. The execution log already stores them separately — no log change.
- Textual 8.2.7 discipline (verified lessons): Message classes in MCP*-named widgets need explicit `namespace=`; AWAIT `remove_children()` before mounting id-bearing children; no uncaught worker exceptions (app panic); `markup=False` on every Static rendering tool/store-derived text; every new Button carries a tooltip (`test_destination_shells` audit); **every new interactive widget gets a real-bundle-CSS harness assertion** (Phase 3's 0×0 Select lesson) and geometry-critical rules go dual-layer (DEFAULT_CSS + bundle source, regenerated via `build_css.py`, never hand-merged).
- Test commands (FOREGROUND; no `timeout` command; worktree has no venv): `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest <paths> -q` from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase4`.
- Known dev baseline: 2 pre-existing Library snapshot-timeout failures in `test_destination_visual_parity_correction.py` — do NOT fix, do NOT count as regressions.
- CONFIG_CHANGED readiness reason code exists fully wired in `readiness.py:32-139` but is emitted by no client derivation. Phase 4 does NOT change `readiness.py`: config-changed surfaces at tool level (matrix, Tools column, inspector, audit entry). Server-level readiness emission stays reserved for backend passthrough. Document this decision in `permission_store.py`'s docstring.

## File Structure

- Create `tldw_chatbook/MCP/permission_store.py` — store + pure resolution (T1, T2).
- Modify `tldw_chatbook/MCP/unified_control_plane_service.py` — typed permission methods + gate + downgrade audit (T4).
- Modify `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`, `mcp_workbench.py` — tuple identity (T3); gate-aware Test Tool (T5); permission explanation (T7).
- Create `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py` — matrix canvas (T6).
- Modify `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py` — State column (T8).
- Modify CSS source + bundle (T9).
- Tests: `Tests/MCP/test_permission_store.py`, `test_permission_resolution.py`, extend `test_control_plane_tool_execute.py`; `Tests/UI/test_mcp_permissions_mode.py`, extend `test_mcp_tools_mode.py`, `test_mcp_inspector.py`, `test_mcp_workbench.py`.

---

### Task 1: Permission store (`MCP/permission_store.py`) — store half

**Files:**
- Create: `tldw_chatbook/MCP/permission_store.py`
- Test: `Tests/MCP/test_permission_store.py`

**Interfaces:**
- Consumes: nothing MCP-specific (stdlib only). Sited by callers next to the local store.
- Produces (exact — T2/T4/T6 depend on these):
  - Module constants: `SCHEMA_VERSION = 1`, `STORE_STATES = ("allow", "ask", "deny")`, `DEFAULT_GLOBAL = "ask"`.
  - `class MCPPermissionStore:` `__init__(self, path: Path)`.
    - `load(self) -> dict` — returns the full payload dict (always valid: missing file → fresh default payload; corrupt JSON / non-dict / `schema_version != 1` → **rename existing file to `<name>.bak` (replace any prior .bak), log a warning, return fresh default**). Fresh default = the Global-Constraints schema with empty `servers`.
    - `save(self, payload: dict) -> None` — atomic (`.tmp` + `replace`), `sort_keys=True, indent=2`, stamps `payload["updated_at"]` ISO-UTC.
    - `get_kill_switch(self) -> bool` / `set_kill_switch(self, value: bool) -> None`.
    - `get_global_default(self) -> str` / `set_global_default(self, state: str) -> None` (validates against `STORE_STATES`, `ValueError` otherwise).
    - `get_server_entry(self, server_key: str) -> dict | None` — `{"default": str|ABSENT, "tools": {...}}` or None.
    - `set_server_default(self, server_key: str, state: str | None) -> None` — `None` = Inherit (remove the `default` key; prune the server entry when it becomes empty).
    - `get_tool_entry(self, server_key: str, tool_name: str) -> dict | None`.
    - `set_tool_state(self, server_key: str, tool_name: str, state: str | None, *, definition_hash: str | None = None) -> None` — `None` state = Inherit (remove the tool entry, prune empties). `state == "allow"` REQUIRES `definition_hash` (ValueError if missing). Setting any state clears a persisted `config_changed` marker; `allow` stores the new hash.
    - `mark_config_changed(self, server_key: str, tool_name: str) -> bool` — sets `config_changed: true` on the tool entry; returns True only when it was not already set (the emit-once transition signal for T4's audit entry).
    - All mutators are read-modify-write through `load()`/`save()` (single-instance UI; last-write-wins across instances per spec).

- [ ] **Step 1: Write the failing tests** — `Tests/MCP/test_permission_store.py`, concrete cases: fresh default payload shape on missing file; corrupt JSON → `.bak` created with the corrupt bytes + fresh default returned + original path gone until next save; `schema_version: 2` → same backup path; kill switch round-trip; global default validate + round-trip; server default set/inherit-prune; tool state set/inherit-prune; allow-without-hash ValueError; allow stores hash + clears config_changed; `mark_config_changed` returns True then False (idempotence signal); atomic write leaves no `.tmp` behind.
- [ ] **Step 2: RED** — module absent. `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/test_permission_store.py -v` → ImportError.
- [ ] **Step 3: Implement** (~170 lines). Mirror `LocalMCPStore.save()`'s atomic pattern (`local_store.py:622-631`); the backup-and-fresh policy is THIS store's deliberate divergence — module docstring must state both the spec mandate and the divergence from `LocalMCPStoreLoadError`.
- [ ] **Step 4: GREEN** — file + `Tests/MCP/ -q`.
- [ ] **Step 5: Commit** — `feat(mcp): schema-versioned permission store with kill switch and backup-on-corrupt`

---

### Task 2: Effective-state resolution (same module) — pure half

**Files:**
- Modify: `tldw_chatbook/MCP/permission_store.py`
- Test: `Tests/MCP/test_permission_resolution.py`

**Interfaces:**
- Consumes: T1 payload shape; `HubTool` (`tldw_chatbook/MCP/hub_tool_catalog.py:20`, fields `server_key/name/description/input_schema/tags`).
- Produces (exact):
  - `HIGH_RISK_TAGS = frozenset({"mutates", "process"})`
  - `def definition_hash(description: str, input_schema: dict | None) -> str` — sha256 hexdigest of `json.dumps({"description": description or "", "inputSchema": input_schema or {}}, sort_keys=True, default=str, separators=(",", ":"))`.
  - `@dataclass(frozen=True) class EffectiveToolState: state: str  # allow|ask|deny; origin: str  # tool_override|server_default|global_default; config_changed: bool = False; risk_floored: bool = False; ui_label property` → `"Allow"/"Ask"/"Off"` from state.
  - `def resolve_effective_state(payload: dict, tool: HubTool) -> EffectiveToolState` — PURE, no I/O:
    1. Tool entry exists → origin `tool_override`. If its state is `allow` and its stored `definition_hash != definition_hash(tool.description, tool.input_schema)` (or entry carries `config_changed: true`) → state `ask`, `config_changed=True`.
    2. Else server `default` → origin `server_default`; else global default → origin `global_default`.
    3. Floor: when origin is NOT `tool_override`, resolved state is `allow`, and `set(tool.tags) & HIGH_RISK_TAGS` → state `ask`, `risk_floored=True`.
  - `def cycle_ui_state(current: str | None) -> str | None` — the Space-cycle helper over store values: `None → "allow" → "ask" → "deny" → None` (Inherit → Allow → Ask → Off → Inherit). Global default variant `cycle_global(current: str) -> str`: `allow → ask → deny → allow`.

- [ ] **Step 1: Failing tests** — precedence (tool over server over global, each origin asserted); hash mismatch downgrades allow→ask with config_changed (and matching hash does NOT); persisted `config_changed` marker downgrades even when hashes re-match until re-allow clears it — wait, re-allow clears the marker in T1's `set_tool_state`, so: marker present → downgrade regardless of hash (test both); floor applies to inherited allow with `mutates` tag, NOT to explicit tool-level allow, NOT to inherited ask; both cycle helpers full-loop.
- [ ] **Step 2: RED.** **Step 3: Implement** (~90 lines). **Step 4: GREEN** — both permission test files + `Tests/MCP/ -q`.
- [ ] **Step 5: Commit** — `feat(mcp): effective-state resolution with rug-pull hash guard and high-risk floor`

---

### Task 3: Tuple identity through the execute path (task-233)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (`ToolTestRequested`, `_handle_test_run` post site ~line 627), `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (`on_mcp_inspector_tool_test_requested` ~1277, `_run_tool_test` ~1302-1357, `_tool_test_in_flight`, `_tool_for` ~1225)
- Test: extend `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Produces: `MCPInspector.ToolTestRequested(server_key: str, tool_name: str, arguments: dict)` (namespace `mcp_inspector` — three fields, `tool_id` REMOVED); workbench `_run_tool_test(server_key: str, tool_name: str, arguments: dict)` with NO `partition("::")` anywhere (delete the malformed-id guard at ~1330-1332); `_tool_test_in_flight: set[tuple[str, str]]` keyed `(server_key, tool_name)`; `_tool_for(server_key: str, tool_name: str)` compares fields, not packed strings; `show_tool_result(..., server_key=..., tool_name=...)` — the stale-drop compare at `mcp_inspector.py:645` matches on both fields. Row keys in `mcp_tools_mode.py` KEEP `tool.tool_id` (display/dedup only — unchanged this task).
- `ToolSelected(tool_id)` from the tools table also stays (row-key echo); the workbench resolves it via a new `_tool_for_row_key(tool_id)` that scans `_last_hub_tools` comparing `tool.tool_id == tool_id` — packing REMAINS legal for row keys, only PARSING is banned. Grep-gate in the test: assert `"partition(\"::\")" not in Path("tldw_chatbook/UI/MCP_Modules/mcp_workbench.py").read_text()` (and no `split("::")`).

- [ ] **Steps 1-5**: failing tests (message carries fields; grep-gate; in-flight double-run still guarded, now on tuple; stale-result drop still works across two tools; same-tool re-run still renders) → implement → run inspector+workbench+tools suites → commit `refactor(mcp-hub): carry (server_key, tool_name) through the execute path — no :: parsing (task-233)`.

---

### Task 4: Control-plane typed permission methods + gate + downgrade audit

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Test: `Tests/MCP/test_control_plane_permissions.py`

**Interfaces:**
- Consumes: T1 store, T2 resolution, existing lazy-property pattern (`execution_log` at ~1884) and `build_record`.
- Produces (exact — T5/T6/T7/T8 call these):
  - Lazy property `permission_store -> MCPPermissionStore | None` — `Path(store.path).with_name("mcp_permissions.json")` via `getattr(self.local_service, "store", None)`; None when no store.
  - `def effective_tool_states(self, tools: list[HubTool]) -> dict[tuple[str, str], EffectiveToolState]` — one `load()`, resolve each; for each tool whose resolution flags a FRESH hash mismatch (tool entry state `allow`, marker not yet set): call `store.mark_config_changed(...)`, and when it returns True emit ONE audit record via the existing `_record_tool_execution`-style path: `build_record(server_key=…, tool_name=…, initiator="system", decision="downgraded", ok=False, duration_ms=0, error=f"{tool_name} definition changed since you allowed it — review and re-allow")` appended best-effort (same try/except contract). No store → every tool resolves `EffectiveToolState(state="ask", origin="global_default")`.
  - `def set_tool_state(self, server_key: str, tool_name: str, ui_state: str | None, *, tool: HubTool | None = None) -> None` — ui_state in `{None, "allow", "ask", "deny"}`; `allow` computes `definition_hash(tool.description, tool.input_schema)` (ValueError if `tool is None`).
  - `def set_server_default(self, server_key: str, state: str | None) -> None`, `def set_global_default(self, state: str) -> None`, `def get_kill_switch(self) -> bool`, `def set_kill_switch(self, value: bool) -> None` — thin store passthroughs (no store → no-op/False).
  - `def gate_tool_test(self, tool: HubTool) -> EffectiveToolState` — single-tool resolution for the Test Tool (fresh `load()`; NO audit emission here — the sync pass owns that; kill switch deliberately ignored, comment why).
- All sync methods (store is file I/O but small; matches the store's own sync API — callers already run inside workers).

- [ ] **Steps 1-5**: failing tests (states resolve per precedence with a real store in tmp_path; fresh mismatch emits exactly one JSONL `decision="downgraded"` record across two consecutive `effective_tool_states` calls; `set_tool_state` allow stores hash + clears marker; no-store fallbacks; gate returns deny/ask/allow per store) → RED → implement → `Tests/MCP/ -q` GREEN → commit `feat(mcp): typed permission methods, tool-test gate, and rug-pull downgrade audit`.

---

### Task 5: Gate-aware Test Tool (deny blocks, ask arms confirm)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (`on_mcp_inspector_tool_test_requested`), `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (test panel Run flow)
- Test: extend `Tests/UI/test_mcp_workbench.py`, `Tests/UI/test_mcp_inspector.py`

**Interfaces:**
- Consumes: T3 message fields, T4 `gate_tool_test`.
- Produces: in `on_mcp_inspector_tool_test_requested`, BEFORE the in-flight check: resolve `tool = self._tool_for(server_key, tool_name)`; `gate = service.gate_tool_test(tool)`. Routing:
  - `deny` → NO worker; `inspector.show_tool_result(ok=False, text='Blocked — this tool is set to Off in Permissions.', duration_ms=0, server_key=…, tool_name=…)`.
  - `ask` (incl. `config_changed`/`risk_floored`) → the inspector's Run button arms instead of running: first press relabels Run to `Confirm run` (`variant="primary"`) with tooltip `"Ask is set for this tool — press again to run once."`; any other interaction (Close, tool switch, mode switch) disarms (mirror the delete-arm contract, `mcp_servers_mode.py:613-685`). Second press posts the run. When `config_changed`, the arm notice Static (markup=False) reads `"Definition changed since you allowed it — review in Permissions."`.
  - `allow` → run immediately (today's behavior).
  - Implementation seam: gate resolution lives in the workbench handler; the ARM state lives in the inspector (`_test_run_armed: bool` + a `require_confirm(notice: str | None)` method the workbench calls; inspector re-posts `ToolTestRequested` on the confirming press). Keep the existing in-flight tuple guard AFTER the gate.
- [ ] **Steps 1-5**: failing tests (deny → no service call + Blocked result line; ask → first Run press does NOT call service, arms Confirm run with tooltip, second press calls it; disarm on close; allow unchanged; config_changed notice text) → implement → inspector+workbench suites → commit `feat(mcp-hub): permission-gated Test Tool with arm-to-confirm on Ask`.

---

### Task 6: Permissions mode canvas (`mcp_permissions_mode.py`)

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (replace the `permissions` placeholder in `MCP_HUB_MODES`/compose; feed it from `_sync_children`)
- Test: `Tests/UI/test_mcp_permissions_mode.py`, extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: T2 `EffectiveToolState`/`cycle_ui_state`/`cycle_global`; T4 typed methods; workbench `_last_hub_tools`.
- Produces:
  - `MCPPermissionsMode(Vertical)`: kill-switch row (`#mcp-perm-kill-switch` Checkbox, label `"MCP tools in chat"`, tooltip `"Master switch for chat tool calls (arrives with the chat bridge). Does not affect Hub tool tests."`); matrix `DataTable` (`#mcp-perm-table`, `cursor_type="row"`, columns **Tool | State | Tags**); pinned rows: first row key `__global__` label `"Global default"`, then per server group a `__server__::<server_key>` row `"Server default — <label>"` followed by its tools (row key = `tool.tool_id`); State cell shows the UI label with origin marker — explicit override renders `"Allow •"` (bullet = override), inherited renders plain `"Ask"`, config_changed renders `"Ask ⚠"` , risk-floored `"Ask ⚑"` (plain `Text` cells, markup-safe); policy preview strip `#mcp-perm-preview` (Static, markup=False) — plain-language sentences scoped to the workbench rail selection, e.g. `"docs-server: 2 allowed, 1 asks, 1 off. Global default: Ask."`.
  - `Binding("space", "cycle_state", "Cycle permission", show=False)` on the widget: cycles the CURSOR row — tool rows via `cycle_ui_state`, server-default rows likewise (None allowed), global row via `cycle_global` — and posts `MCPPermissionsMode.StateCycleRequested(row_kind: str, server_key: str, tool_name: str | None, new_state: str | None)` (namespace `mcp_permissions_mode`; `row_kind` in `{"global", "server", "tool"}`). The WORKBENCH mutates via T4 methods then resyncs (single writer; the widget never touches the store).
  - `async update_matrix(self, rows: list[PermRow], *, kill_switch: bool, preview: str) -> None` where `PermRow` is a small frozen dataclass in this module: `(kind, server_key, server_label, tool_name, state_label, tags_label, cycle_current)` — the workbench builds rows from `_last_hub_tools` + `effective_tool_states`; the widget is render-only.
  - Kill-switch `Checkbox.Changed` → `KillSwitchToggled(value: bool)` message → workbench `set_kill_switch` + resync (guard the mount echo).
- [ ] **Steps 1-5**: failing tests (pinned row order global→server→tools, grouped/sorted; override bullet vs inherited plain; ⚠/⚑ markers; Space on a tool row posts StateCycleRequested with the next state per cycle helper; Space on global row never posts None; kill-switch toggle posts once — no mount echo; preview text renders; workbench round-trip: cycle message → store mutated → matrix re-rendered with the override marker; real-bundle-CSS harness assertion: the matrix table and kill-switch row render non-zero — Phase 3 lesson) → implement → commit `feat(mcp-hub): permissions matrix with Space cycling, kill switch, and policy preview`.

---

### Task 7: Inspector rule explanation + re-allow

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (selection wiring from the matrix)
- Test: extend `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Produces: `MCPInspector.show_permission(self, tool: HubTool, effective: EffectiveToolState) -> None` — renders into `#mcp-inspector-permission` (own container, same await-remove/mount discipline as `#mcp-inspector-tool`): effective UI label; origin sentence (exact copy) `"From this tool's override."` / `"Inherited from the server default."` / `"Inherited from the global default."`; when `config_changed`: notice `"Definition changed since you allowed it."` + a `Re-allow` Button (`#mcp-inspector-reallow`, tooltip `"Store the new definition hash and allow again."`) that posts `MCPInspector.ReallowRequested(server_key, tool_name)` (namespace `mcp_inspector`) → workbench calls `set_tool_state(..., "allow", tool=tool)` + resync; when `risk_floored`: notice `"High-risk tool — asks even though the inherited default is Allow."`. Matrix `RowHighlighted`/selection → workbench routes `show_permission` for tool rows, `show_tool(None)`+summary for pinned rows. Tools-mode selection ALSO appends the permission block below the existing tool detail (one call site: workbench passes `effective` into the existing `show_tool` flow via a new keyword `effective: EffectiveToolState | None = None`).
- [ ] **Steps 1-5**: failing tests (origin sentences exact; re-allow button only when config_changed; ReallowRequested round-trip clears ⚠ in the matrix; pinned-row selection doesn't crash; DuplicateIds on repeated selection) → implement → commit `feat(mcp-hub): inspector permission explanation with re-allow`.

---

### Task 8: Tools-mode State column + server-source governance read-only

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py` (`_TABLE_COLUMNS` line 26, `_apply_filter`), `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (`_sync_tools_mode` passes states), `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py` (server-source section)
- Test: extend `Tests/UI/test_mcp_tools_mode.py`, `Tests/UI/test_mcp_permissions_mode.py`

**Interfaces:**
- Produces: Tools table columns become **Tool | State | Server | Tags | Schema** (State cell = the same label+marker rendering as the matrix; `update_tools` gains `states: dict[tuple[str, str], EffectiveToolState] | None = None`, absent → `"—"`). Permissions mode on the SERVER source additionally mounts `#mcp-perm-server-profiles` (below the matrix): read-only listing of `get_governance()["permission_profiles"]` names/ids (plain Text, defensive raw-dict reads like `server_tools_from_inventory`) + pointer Static `"Server-side profiles are managed in the tldw_server webui. The matrix above is chatbook's client-side gate and still applies."` (markup=False). Local/builtin sources: section absent.
- [ ] **Steps 1-5**: failing tests (column present + rendering parity with matrix labels; states=None renders em-dash; server-source section renders profile names from a fake governance payload and is absent on local source) → implement → commit `feat(mcp-hub): permission state column in tools catalog; read-only server governance listing`.

---

### Task 9: CSS bundle, footer hints, full gate

**Files:**
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py` (`MCP_SHORTCUTS` gains `("space", "cycle permission")` hint — display only, the binding lives on the matrix widget), CSS source component file (the `_agentic_terminal.tcss` MCP block) + rebuilt bundle
- Test: extend `Tests/UI/test_mcp_workbench.py` (bundle-parity family)

- [ ] **Step 1**: dual-layer rules for the new geometry-critical ids (`#mcp-perm-table` height discipline mirroring `#mcp-tools-table`'s `height: auto; max-height: 70%;`; `#mcp-perm-preview` and the kill-switch row `height: auto`), adjacent to the existing lockstep block, same comment style; regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`; bundle-parity tests assert source+bundle carry the rules; fidelity check (regenerate → timestamp-only).
- [ ] **Step 2**: full gate:
`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/ Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_tools_mode.py Tests/UI/test_mcp_schema_form.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_profile_form.py Tests/UI/test_mcp_server_mutations.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py -q`
Expected: all green except the 2 documented pre-existing Library snapshot failures.
- [ ] **Step 3**: Commit — `feat(mcp-hub): permissions-mode styles, shortcut hint, Phase 4 gate`

**Post-task (controller-owned):** live screenshot QA (Phase 3 recipe: textual-serve port 9191, clone `/private/tmp/tldw-qa-mcp-hub-p3-20260714` → `-p4-`, seed `mcp_permissions.json` with one explicit allow + one stale-hash allow + one deny) covering: matrix with pinned rows and override/⚠/⚑ markers, Space cycle live, kill-switch toggle, policy preview, inspector explanation + Re-allow, Test Tool deny-block and ask-arm flows, Tools State column, server-source read-only section. User screenshot approval gates the PR.

## Out of scope (recorded)

Chat send-time assembly honoring the kill switch + batch approval UX (Phase 5); Audit-mode UI over the execution log (Phase 5); server-level CONFIG_CHANGED readiness emission (reserved for backend passthrough); permission profiles beyond `default` (store is profile-ready by schema); task-234's backend shape verification (tracked separately — server-source tools remain display-only and non-executable this phase; their matrix states are stored client-side and take effect when execution arrives); risk filter + needs-auth empty-state bucket + transport/last-used inspector lines (task-242, Phase 5/6 planning).
