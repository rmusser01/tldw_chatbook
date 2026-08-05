# MCP Hub Redesign — Phase 6 (Finale: Remediation, Polish, Legacy Retirement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 6-phase MCP Hub program: colored state words, Findings inline remediation, the deferred UX B-batch structural items, matrix scale posture, Advanced-out-of-default with full legacy retirement, and the §14 resources/prompts read-only listing.

**Architecture:** No new service seams — this phase is UI/polish/cleanup over the Phases 1-5 foundation. **Server-source tool execution is DEFERRED ENTIRELY (user decision 2026-07-17): the program ends with local+builtin execution; server-source tools stay display-only** (task-234 closes as deferred-by-decision). State-word coloring applies `$ds-status-*` tokens (user decision — the New_UI mockup direction; glyphs stay the colorblind-safe channel). Legacy retirement is FULL (user decision): `UnifiedMCPPanel`, its Tools_Settings pane, `unified_mcp_sections.py`, and `LAYOUT_MODE_*` go; Advanced becomes opt-in with the six spec-protected actions preserved.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, existing readiness/permission/log seams.

## Global Constraints

- Server-source execution/access-context/task-234: OUT OF SCOPE by user decision. Do not add `server:` routing to `execute_hub_tool`; the "arrives in Phase 4" rejection copy becomes `"Server-source tools are display-only."` (fix the stale copy where it appears; tests pin it).
- Spec-protected Advanced actions that must stay reachable after retirement: `governance_rule.save/preview/delete`, `runtime.access.preview`, `resource.read`, `prompt.get` (they exist ONLY in the Advanced panel — verified `unified_control_plane_service.py:252-327`).
- **Coordination hazard (verified): unmerged branch `ux/shell-header-remediation` (c7140fbb) DELETES `AppFooterStatus` + `register_footer_shortcuts` (replaced by `AppStatusLine`).** Phase 6 must NOT modify footer-hint mechanics; keep `MCP_SHORTCUTS` data + the `register_footer_shortcuts` call as-is (guarded if cheap: `getattr(self, "register_footer_shortcuts", None)`), and expect to port at rebase if that branch merges first.
- Colored state words use the existing tokens (`tldw_cli_modular.tcss:35-44`) via the existing class wrappers (`.mcp-status-ready/-warning/-error/-info`, plus add `.mcp-status-muted` → `$ds-text-muted` if needed). DataTable cells color via `rich.text.Text(..., style=...)` — Textual DataTables don't take CSS classes per cell; use `Text` style with the theme-resolved color where feasible or the established glyph+styled-Text pattern; VERIFY the approach against how `mcp_rail.py:265` styles glyphs today and keep markup-safety (no markup parsing of tool-derived text).
- Textual 8.2.7 discipline (all verified lessons apply): namespace= on MCP* messages; await remove_children; markup=False on derived text; every Button tooltipped; dual-layer geometry CSS via build_css.py only; real-bundle harness assertions for new interactive widgets; row-key cursor re-seat on rebuilds; the exclusive-group cancel-before-body trap; no dead scaffolding — every UI reader needs a producer (bit 3× in Phase 5).
- Test commands (FOREGROUND; no `timeout`; worktree has no venv): `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest <paths> -q` from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase6`.
- Known baseline: 2 pre-existing Library snapshot failures in `test_destination_visual_parity_correction.py` (outside the phase gate command); 1 pre-existing skip in `test_destination_shells.py`.
- Verified seam facts (2026-07-17 session): `HubAction` enum = ADD_SERVER/EDIT_CONFIG/OPEN_CREDENTIALS/CONNECT/REFRESH_DISCOVERY/VALIDATE/VIEW_DETAILS/OPEN_TOOL_CATALOG/OPEN_AUDIT (`readiness.py:46-56`); workbench HubAction routing at `mcp_workbench.py:1839-1867` (lifecycle verbs local-only; `server:` keys silently dropped; ADD_SERVER/OPEN_CREDENTIALS unrouted); findings carry NO machine-actionable code (free-text `remediation`/`suggested_remediation`); Advanced collapsible composes unconditionally with a global `mcp.hub_state.advanced_open` persistence (`mcp_inspector.py:541-567`); `UnifiedMCPPanel` mounted at `Tools_Settings_Window.py:2821` (pane `ts-view-unified-mcp`, nav `ts-nav-unified-mcp`); `LAYOUT_MODE_FULL/COMPACT` effectively dead (`unified_mcp_panel.py:19-20`, never passed by callers); `tools_settings` alias → MCPScreen locked by `test_screen_navigation.py:477-484`; `unified_mcp_sections.py` live importers = `unified_mcp_panel.py` + `mcp_inspector.py:1399` (dispatcher only) + two test files; NavigateToScreen cross-screen precedent (`main_navigation.py:18`, used `personas_screen.py:500`); STATE_CSS_CLASSES glyph coloring exists (`readiness.py:537-544`).

## File Structure

- Modify: `mcp_servers_mode.py`, `mcp_tools_mode.py`, `mcp_permissions_mode.py`, `mcp_audit_mode.py`, `mcp_inspector.py`, `mcp_workbench.py`, `mcp_rail.py` (T1 colors; T2 remediation; T3 provenance/jump/echo; T4 filter); `Tools_Settings_Window.py` + delete `unified_mcp_panel.py`/`unified_mcp_sections.py` (T5); CSS source + bundle (T6).
- Tests: extend the existing per-widget suites; delete `test_unified_mcp_panel.py` with its widget (T5).

---

### Task 1: Colored state words

**Files:** Modify `mcp_rail.py`, `mcp_servers_mode.py`, `mcp_tools_mode.py`, `mcp_permissions_mode.py`, `mcp_audit_mode.py`, `mcp_inspector.py`; Test: extend each suite

**Interfaces:**
- A shared module-level helper in `mcp_permissions_mode.py` (beside `format_tool_state_label`): `state_text(label: str, state_class: str) -> Text` returning a `rich.text.Text` styled with the resolved status color — resolve theme variables the way the repo already styles Text in tables (INVESTIGATE `mcp_rail.py:265`'s glyph approach first; if theme-variable resolution inside Text styles is impractical in DataTable cells, fall back to Rich named colors matching the ds tokens' resolved values and document). Apply to: Servers-mode state column + rail state glyph+word, Tools/Permissions State cells (allow→ready/green, ask→warning/amber, deny/off→error-ish dim red, config_changed/risk_floored keep ⚠/⚑ + warning color), Audit Decision/Outcome cells (approved/allowed→ready, denied*/blocked→error, downgraded→warning), Findings Severity (critical/high→error, medium→warning, low/info→info), inspector permission/readiness state words. Glyphs unchanged (colorblind channel).
- [ ] **Steps 1-5**: failing tests (assert cell `Text.style`/spans carry the expected style per state — pick the assertion form that works against the chosen mechanism; cover one representative per surface) → implement → run the six suites → commit `feat(mcp-hub): semantic state colors across all mode surfaces`.

---

### Task 2: Findings inline remediation + per-source action routing

**Files:** Modify `mcp_inspector.py` (finding detail actions), `mcp_workbench.py` (routing), `mcp_audit_mode.py` (if needed); Test: extend `test_mcp_audit_mode.py`, `test_mcp_workbench.py`, `test_mcp_inspector.py`

**Interfaces:**
- Heuristic mapping (findings carry no machine code — free-text only): module-level `remediation_actions(finding: Mapping) -> tuple[HubAction, ...]` in `mcp_audit_mode.py`: lowercase `finding_type` + `message` scanned for keywords → `{"discovery"|"stale"|"catalog": (REFRESH_DISCOVERY, VIEW_DETAILS), "auth"|"credential"|"token": (OPEN_CREDENTIALS, VIEW_DETAILS), "permission"|"policy": (VIEW_DETAILS, OPEN_AUDIT)}`, default `(VIEW_DETAILS,)`. Pure, defensive, tested.
- `show_finding` gains action Buttons for the mapped HubActions (reuse `_ACTION_LABELS`; ids `#mcp-finding-action-<action>`; tooltipped), posting the EXISTING `HubActionRequested` message with the finding's owning server key when derivable (target-level `server:<target_id>`; else the selected rail server).
- Workbench routing extended for `server:` keys — the navigational + refresh subset only: VIEW_DETAILS→servers mode+select; OPEN_TOOL_CATALOG→tools; OPEN_AUDIT→audit; REFRESH_DISCOVERY→for server source, invalidate the governance/findings cache + full resync (the Phase-4 cached-by-(source,target) seam — clear it) with a toast; OPEN_CREDENTIALS→servers mode + select + notify "Credentials are managed in the server's config." (no credentials editor exists for server source — honest copy); CONNECT/VALIDATE/EDIT_CONFIG stay local-only (unchanged). NO MORE silent drops: any unrouted action for a server key → info toast "Managed on the server."
- UX-item-10 fix (same seam): the Tools-mode empty-state diagnosis on the SERVER source must not route to connect/refresh actions that are disabled — server-source empty diagnosis becomes ("No tools visible from this server — refresh or check the server.", "refresh") where refresh = the cache-invalidating resync above.
- [ ] **Steps 1-5**: failing tests (mapping table incl. default; finding detail renders mapped buttons; each routing branch incl. cache invalidation + honest toasts; server empty-state routing) → implement → commit `feat(mcp-hub): findings inline remediation and per-source hub-action routing`.

---

### Task 3: Cascade provenance, cross-mode jump, mutation echo

**Files:** Modify `mcp_inspector.py`, `mcp_workbench.py`, `mcp_permissions_mode.py`; Test: extend the three suites

**Interfaces:**
- **Cascade provenance** (replaces the single origin sentence in `show_permission`): three Statics (markup=False) — `Tool override: Allow •` / `Server default: —` / `Global default: Ask`, the winning rung prefixed `▸ ` and colored (T1 helper), non-contributing rungs dimmed. Data: the workbench already loads the store payload per sync — pass the raw tool-entry/server-default/global values through `show_permission(..., cascade: tuple[str|None, str|None, str] | None = None)`; None → fall back to the current origin sentence.
- **Cross-mode jump**: Button `Change in Permissions` (`#mcp-inspector-goto-permission`, tooltipped) rendered in (a) the Test Tool blocked/ask result area and (b) the Tools-mode permission block — posts the existing mode-switch + matrix row selection (reuse the audit drill's "Adjust permission" routing — extract a shared workbench helper `_goto_permission_row(server_key, tool_name)` so audit drill + these two callers share one implementation; no duplication).
- **Mutation echo**: after a Space-cycle/kill-switch/re-allow mutation, the permissions preview strip's first line temporarily prefixes the echo `"{tool_name} → {new_ui_label} · "` (or for kill switch `"kill switch → on/off · "`) — plain state, cleared on the next full sync that isn't a standalone mutation resync (keep it simple: the workbench passes `echo: str | None` into `update_matrix`/preview build; standalone handlers set it, full syncs pass None). No undo this phase (recorded).
- [ ] **Steps 1-5**: failing tests (cascade rungs + winner marking incl. fallback; jump buttons route to the matrix row from both sites via the shared helper; echo appears after cycle and clears on full resync) → implement → commit `feat(mcp-hub): cascade provenance, change-in-permissions jump, mutation echo`.

---

### Task 4: Permissions matrix filter

**Files:** Modify `mcp_permissions_mode.py`, `mcp_workbench.py`; Test: extend `test_mcp_permissions_mode.py`

**Interfaces:**
- `#mcp-perm-filter-text` Input above the matrix (placeholder "filter tools"): client-side filter over TOOL rows (name + server_label, case-insensitive) — pinned rows (global + server defaults of servers with visible tools) always shown; empty filter → all. Cached rows re-filtered widget-side (mirror `MCPToolsMode._apply_filter`, incl. cursor re-seat + dedupe guard). Preview strip unchanged (summarizes the UNFILTERED state — note in a comment). Collapsible groups: deferred (recorded).
- [ ] **Steps 1-5**: failing tests (narrows tool rows, keeps relevant pinned rows, cursor re-seat across re-filter, Space still cycles the visible cursor row correctly) → implement → commit `feat(mcp-hub): permissions matrix text filter`.

---

### Task 5: Advanced opt-in + full legacy retirement + resources/prompts listing

**Files:** Modify `mcp_inspector.py`, `mcp_workbench.py`, `mcp_servers_mode.py`, `Tools_Settings_Window.py`, `UI/MCP_Modules/__init__.py`; Delete `UI/MCP_Modules/unified_mcp_panel.py`, `UI/MCP_Modules/unified_mcp_sections.py`, `Tests/UI/test_unified_mcp_panel.py`; Test: extend `test_mcp_inspector.py`, `test_mcp_servers_mode.py`, `Tests/UI/test_tools_settings_window.py` (check it exists), keep `test_screen_navigation.py` green

**Interfaces:**
- **Advanced opt-in**: the collapsible is NO LONGER composed by default. A new setting `mcp.hub_state.advanced_visible` (default False) gates composition; reveal affordance = a small `Advanced…` Button (`#mcp-inspector-advanced-reveal`, tooltip "Show the legacy control-plane action runner.") at the inspector's bottom that flips the setting + remounts (await discipline). When visible, the existing collapsible + persistence behave as today. The six protected actions remain available inside it (no action-template changes). The JSON section renderer the collapsible uses (`render_unified_mcp_section` import at `mcp_inspector.py:1399`) must survive the module deletion: MOVE the minimal needed rendering (plain JSON dump path) inline into `mcp_inspector.py` (do not keep the whole legacy module for one dispatcher — inline a small `_render_section_payload(section, payload) -> str` that json.dumps sensibly; the old per-section text renderers die with the module).
- **Retirement**: delete `unified_mcp_panel.py` + `unified_mcp_sections.py` + their test file; remove the `Unified MCP` nav button + `ts-view-unified-mcp` pane + imports/queries from `Tools_Settings_Window.py` (verified sites :35, :2789, :2818-2823, :5280-5288); purge `LAYOUT_MODE_*` from `UI/MCP_Modules/__init__.py` re-exports; repo-wide grep proves zero remaining importers. `tools_settings` alias routing untouched (`test_screen_navigation.py:477-484` must stay green).
- **Resources & prompts read-only listing** (§14 compensation for Advanced leaving default view): the servers-mode DETAIL panel gains two compact sections listing resource names/URIs and prompt names from the local discovery snapshot (`discovery_snapshot["resources"|"prompts"]`, defensive reads, markup=False, "none" empty copy); server source shows counts only (from the existing payloads). Read-only, no interactions (test-read/test-get remain via opt-in Advanced).
- [ ] **Steps 1-5**: failing tests (reveal button composes Advanced + persists; hidden by default; protected actions listed once revealed; deletion leaves zero importers [grep-gate test]; Tools_Settings pane gone + window still composes; resources/prompts sections render from a seeded snapshot) → implement → run inspector/servers/navigation/tools-settings suites → commit `feat(mcp-hub): opt-in Advanced, retire legacy MCP panel and section renderers, resources/prompts listing`.

---

### Task 6: Copy fixes, CSS bundle, full gate

**Files:** Modify `unified_control_plane_service.py` (stale copy), CSS source + bundle; Test: bundle-parity + the full gate

- [ ] **Step 1**: stale copy — `execute_hub_tool`'s server-key rejection message becomes `"Server-source tools are display-only."` (update its docstrings' "until Phase 4" lines too); grep for any remaining "arrives in Phase 4"/"in a later phase" copy on shipped surfaces and fix with tests.
- [ ] **Step 2**: dual-layer CSS for anything T1-T5 added that is geometry-critical (`#mcp-perm-filter-text` width/height; finding action row; reveal button; resources/prompts sections) adjacent to the lockstep block; regenerate bundle via build_css.py; parity tests; fidelity check.
- [ ] **Step 3**: full gate:
`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/ Tests/Agents/ Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_tools_mode.py Tests/UI/test_mcp_schema_form.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_audit_mode.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_mcp_profile_form.py Tests/UI/test_mcp_server_mutations.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py Tests/UI/test_chat_approvals_and_resume.py Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_chat_controller.py -q`
Expected: all green (the pre-existing skip allowed; `test_unified_mcp_panel.py` deleted in T5 so absent from the command).
- [ ] **Step 4**: Commit `feat(mcp-hub): Phase 6 copy, styles, and gate`

**Post-task (controller-owned):** live screenshot QA (established recipe; clone the P5 HOME) covering: colored state words on all four modes, finding remediation buttons + refresh routing, cascade provenance, Change-in-Permissions jump, mutation echo, matrix filter, opt-in Advanced (hidden → revealed), resources/prompts listing, Tools_Settings without the MCP pane. User screenshot approval gates the PR. Post-merge: program close-out (task-234 dispositioned deferred-by-decision; tasks 233/242 remnants re-triaged; memory finale).

## Out of scope (recorded)

Server-source tool execution + access-context + task-234 verification (DEFERRED BY USER DECISION — the program ships with local+builtin execution; server tools remain display-only); single-step undo for permission mutations (echo only); collapsible matrix server groups; footer-hint mechanics (unmerged `ux/shell-header-remediation` owns that surface — MCP_SHORTCUTS data kept, port at rebase if it lands); full resources/prompts browsing; task-88's Settings ▸ MCP Defaults category (separate task, unstarted — the NavigateToScreen precedent is recorded for it).
