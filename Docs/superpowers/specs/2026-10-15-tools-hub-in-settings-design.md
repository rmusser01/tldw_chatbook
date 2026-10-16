# Tools Hub in Settings — Design

Date: 2026-10-15
Status: Approved 2026-10-16 — user confirmed all four open decisions:
full absorption (Approach A); category title "Tools & MCP"; Tool Profiles
stays separate with a cross-link; `mcp-defaults` deleted outright.
Proposed ADR: ADR-233 (retire the MCP destination; Settings hosts the Tools hub)

## Problem

The tool-configuration surface lives behind a top-level shell destination
labeled **MCP** (route `mcp`, legacy alias `tools_settings`, ctrl+9). The label
is protocol jargon, the screen's own copy apologizes for existing ("most
people never need to change anything here"), and users looking for "tools" in
Settings find only two fragments: a **Tool Profiles** category (profile
lifecycle) and a read-only **MCP Defaults** pointer card that says "server
operations stay in MCP." There is no single obvious place to manage tools.

Goal: one obvious, plain-language home for tools inside the Settings screen,
without rewriting the (healthy, recently redesigned) MCP workbench.

## Non-goals

- No changes to `MCPWorkbench` internals, `UI/MCP_Modules/*` widgets, the
  permission store, ToolPackService, or any backend.
- No changes to ADR-148's rail IA, responsive triad, recovery review, or audit
  drill-down behavior.
- No new persistence; all existing stores and config sections stay as-is.

## Approaches considered

**A. Full absorption (recommended).** Retire the MCP destination; Settings
gains one **Tools & MCP** category that hosts the existing four-mode
`MCPWorkbench` intact. The 347-line screen shell's responsibilities (mode
chips, footer hints, reload worker, deep-link context) port into a new
sibling pane module. Reuses the ADR-172 retirement template (Artifacts →
Library) exactly: compatibility hotkey, route aliases, palette aliases.
- Pros: delivers the ask; one less jargon destination; the 7k-line workbench
  is re-hosted, not rewritten — engineering risk is contained to navigation
  seams and the new pane; every mechanical piece is precedented in-repo.
- Cons: ~16 test files touch `MCPScreen` and need porting; ADR-015's
  destination set changes (13 → 12); Settings hosts one large interactive
  pane (a departure from form-style categories, though Workspaces/Hooks
  already do immediate interactive actions).

**B. Config/operations split.** Settings absorbs only tool *configuration*
(permissions matrix, tool gates, `[mcp]` defaults); Servers runtime
operations, the test bench, and Audit stay in a (renamed) destination.
- Rejected: splits "where do I manage tools?" across two surfaces — the exact
  confusion this change exists to fix — and requires dissecting the workbench,
  which is riskier than re-hosting it whole.

**C. Hub-and-pointer (lightest).** Keep the workbench; redesign Settings'
  tool categories into a linked hub.
- Rejected: this is essentially what the domain-contract system already does,
  and the user has judged it not easy enough.

The in-code `mcp-defaults` domain contract warns against "flattening tools
into Settings." That objection was about flattening tool *config into form
categories*; Approach A does not flatten — it re-homes the workbench intact
and deletes the pointer card. ADR-233 records this supersession explicitly.

## Design

### 1. Information architecture and navigation

- Remove the `mcp` entry from `SHELL_DESTINATION_ORDER`
  (`UI/Navigation/shell_destinations.py`); the strip drops from 13 to 12
  destinations. Remove `mcp`'s ctrl+9 from `SHELL_DESTINATION_SHORTCUTS`
  (lockstep rule, ADR-031/ADR-152).
- Add `MCP_COMPATIBILITY_SHORTCUT = "ctrl+9"` and an app-level hidden
  `Binding` that navigates to Settings with `{"category": "tools"}` — the
  exact `ARTIFACTS_COMPATIBILITY_SHORTCUT` pattern (ADR-172).
- Route compatibility, mirroring `_ROUTE_MAP["artifacts"]`:
  `mcp` and `tools_settings` resolve to
  `ResolvedShellRoute("settings", "settings", …)`, and both stay in
  `_ROUTABLE_LEGACY_ROUTES`.
- `screen_registry.py`: delete the `mcp`/`tools_settings` `ScreenRoute`s; add
  aliases `"mcp": "settings"` and `"tools_settings": "settings"` (the
  `"artifacts": "library"` pattern).
- Default navigation context for the bare legacy routes (the
  `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` mechanism, `"customize"` precedent):
  `mcp` / `tools_settings` → `{"category": "tools"}`.
- Palette: the Settings destination command gains alias terms ("tools",
  "mcp", "tool settings"); typing "mcp" finds Settings, activating lands on
  the Tools & MCP category.

### 2. The Settings category

- `SettingsCategoryId.TOOLS = "tools"`; retire `MCP_DEFAULTS`
  ("mcp-defaults") — its domain contract and ownership record are deleted,
  subsumed by the real surface. Unknown-category contexts (e.g. a stale
  `{"category": "mcp-defaults"}`) already no-op safely via
  `apply_navigation_context`'s validation.
- Sidebar entry: **Tools & MCP**, seated with the agent/tool categories
  (near Tool Profiles / Console Behavior / Hooks). Description in
  plain language: "Agent tools, MCP servers, permissions, and audit."
- New sibling module `UI/Screens/settings_tools_hub.py` — the pane composes:
  the plain-language purpose line, a `DestinationModeStrip` (Servers / Tools /
    Permissions / Audit chips), and one `MCPWorkbench` instance. The
  35k-line `settings_screen.py` grows only by category registration and
  dispatch glue, per the existing `settings_*` module pattern.
- Port from `mcp_screen.py`: mode-chip sync, per-mode footer hints, the
  exclusive reload worker (`_RELOAD_WORKER_GROUP`, including
  `handle_runtime_backend_changed` forwarding), and view-state
  save/restore under the existing `mcp_hub_view_state` key. Then delete
  `UI/Screens/mcp_screen.py`.

### 3. Keybindings (the one real collision)

SettingsScreen binds `s/r/t/a/c/b/e/d/x` screen-wide today; the MCP shell
uses `1-4/a/r/t`. Resolution follows the screen's own established precedent
(screen-wide keys, per-category dispatch/self-guard — see the RAG
accelerator comment in `settings_screen.py` and the s/r/t guided verbs):

- `1-4` (mode switch): new screen-level bindings, self-guarded to the TOOLS
  category (no collision exists today).
- `a` (add server), `r` (refresh), `t` (test tool): the existing actions gain
  a TOOLS-category branch — when Tools & MCP is active they dispatch to the
  pane's ported actions (add-server form, forced workbench reload,
  test-tool panel); otherwise today's behavior is unchanged.
- Footer hints: `_category_footer_shortcuts(TOOLS)` advertises the per-mode
  hint set (F-055: only keys that work in the active mode), replacing the
  screen-level `register_footer_shortcuts(source="mcp")` registration. The
  pane re-invokes the screen's category-hint path on every mode change so
  the footer stays per-mode honest, as `on_mcp_workbench_mode_changed` does
  today.
- Per project gotcha #11, no terminal-convention keys are touched and no
  hint is advertised that isn't implemented.

### 4. Deep links

- `apply_navigation_context` on Settings already selects a category; extend
  its TOOLS branch to accept the bounded MCP context keys that
  `MCPScreen.apply_navigation_context` validates today
  (`mode`, `tool_policy_profile_id`, `profile_revision`,
  `profile_policy_digest`) and forward them to the pane → workbench
  `set_initial_view_state`. Same validation, new home. A `mode` key, when
  present, must be one of the four workbench mode ids (`servers`, `tools`,
  `permissions`, `audit`) or the context is ignored.
- The Tool Profiles category's "edit policy in MCP Permissions" handoff
  (covered by `Tests/UI/test_mcp_permissions_handoff.py`) becomes an
  internal category switch + context — same user flow, one screen.
- A `mode` context key (e.g. `{"category": "tools", "mode": "audit"}`) makes
  the compatibility hotkey and palette land on Servers mode by default while
  allowing direct deep links to any mode.

### 5. Naming and copy

- Category title: **Tools & MCP**; keeps "MCP" searchable for users who know
  the protocol, leads with the plain word. The purpose line keeps today's
  plain-language copy ("MCP (Model Context Protocol) lets chatbook use
  external tools — most people never need to change anything here.").
- The nav label "MCP" disappears from the shell; nothing else in the app
  renames.

### 6. What does not change

- `UI/MCP_Modules/*` (workbench + 11 mode/inspector modules): untouched.
- Permission store, ToolPackService, workspace assistant defaults, ADR-148's
  rail IA, ADR-167/168/169 save-lifetime semantics: untouched.
- Console's links to the MCP surface keep working via the route alias.

### 7. ADR plan

ADR required: yes — `backlog/decisions/233-mcp-destination-retirement-tools-hub-in-settings.md`.

- Decision: retire the MCP shell destination; Settings hosts the Tools & MCP
  workbench intact.
- Amends ADR-015 (destination set 13 → 12; fold map gains `mcp`/`tools_settings` → Settings).
- Supersedes the `mcp-defaults` domain contract's "server operations stay in
  MCP" stance (the contract object itself is deleted).
- Reuses ADR-172's retirement template; ADR-148's workbench IA is preserved
  unchanged; shortcut removal follows ADR-031/152 lockstep.

### 8. Testing

- Port/extend: `Tests/UI/test_destination_shells.py` (mcp shell cases →
  Settings pane mount), `test_screen_navigation.py` (alias resolution),
  `test_screen_footer_hints.py` (per-mode hints via category), 
  `test_mcp_permissions_handoff.py` (internal switch),
  `Tests/ProductionApp/test_retired_destination_root_state.py` and
  `test_reactive_ownership_maturity.py` (retired-destination invariants),
  `Tests/Performance/test_ui_latency_guardrails.py`,
  `Tests/UI/test_latest_dev_core_app_usability_smoke.py`,
  `test_destination_visual_parity_correction.py`,
  `test_mount_io_off_pump.py`, `test_widget_css_consolidation.py`,
  `Tests/Backup_Recovery/test_mcp_recovery_*.py` (mount path changes only).
- New tests: route alias resolution for `mcp`/`tools_settings`; ctrl+9
  compatibility binding lands on Settings → Tools; TOOLS branch of
  `apply_navigation_context` accepts the bounded profile deep-link and
  rejects malformed ones; `1-4/a/r/t` dispatch guards (active in TOOLS,
  inert elsewhere — mirror the RAG accelerator self-guard tests).
- Targeted runs only, per repo testing guideline (no full sweep unless
  requested).
- Live verification per `backlog/docs/lessons-live-verification.md`: run the
  app; F4 → Tools & MCP; each mode; ctrl+9; palette search "tools" and
  "mcp"; Tool Profiles → edit-policy handoff.

### 9. Rollout slicing (for the implementation plan)

1. Add the TOOLS category + pane module; MCP destination still present
   (both surfaces live, no navigation change yet).
2. Port deep links and the Tool Profiles handoff to the Settings pane.
3. Retire the destination: shell order/shortcuts, route aliases, default
   context, palette aliases, compatibility binding; delete `mcp_screen.py`.
4. Test migration + ADR-233 + docs (AGENTS.md tool-calling section,
   Architecture index).

## Risks

| Risk | Mitigation |
| --- | --- |
| Settings' 35k-line file grows further | All pane logic lives in `settings_tools_hub.py`; `settings_screen.py` gains only registration + dispatch glue |
| Keybinding regressions in other categories | `a/r/t` gain branches, existing branches untouched; self-guard tests mirror the RAG accelerator precedent |
| Test blast radius (~16 files) | Workbench tests are untouched by construction; shell tests port mechanically to the pane mount |
| Muscle-memory loss (ctrl+9) | Compatibility binding preserved (ADR-172 pattern) |
| Unknown-category stale contexts | Existing `apply_navigation_context` validation no-ops safely |

## Open decisions (resolved 2026-10-16)

1. **Scope confirm:** full absorption (Approach A). ✓
2. **Category title:** "Tools & MCP". ✓
3. **Tool Profiles:** stays a separate category with a cross-link. ✓
4. **`mcp-defaults` handling:** deleted outright. ✓
