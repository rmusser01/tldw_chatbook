# MCP Hub Rail IA Split — Design

Date: 2026-09-11
Status: Draft (awaiting review)
Review basis: MCP screen UX review 2026-09-11 (Finding F4b), evidence screenshots `01/03/08`
ADR: backlog/decisions/148-mcp-hub-rail-ia-and-responsive-triad.md
Related: ADR-032 (local:__local__), ADR-053 (built-in stdio server), task-2240 (single-problem preselection), F-054

## Problem

One rail row — `⌂ tldw_chatbook (built-in)` — currently fronts three distinct
subsystems:

1. **The app-as-MCP-server** readiness (the stdio process `python3 -m
   tldw_chatbook.MCP` that external clients launch; `[mcp]` enable/expose
   toggles).
2. **The in-process agent tool catalog's registration gates** (`[tools]` /
   `[console]` checkboxes rendered in the same detail pane — the code itself
   documents this as an accepted trade-off: "a different subsystem sharing
   the same detail pane for discoverability").
3. **The Permissions preview scope**: selecting the row scopes the policy
   preview sentence to the agent built-in group and reports it under the
   plain label `tldw_chatbook` ("tldw_chatbook: 0 allow · 30 ask"), which
   reads as the MCP server it is not.

The conflation forces explanatory copy everywhere (subheadings, notes,
doc paragraphs) and still leaks the mismatch through the preview label.

## Goals

- The rail presents **two honest sections**: `Servers` (built-in MCP server
  + local profiles + server-source records) and `Agent tools` (the
  in-process catalog the Console uses).
- Each section's detail pane renders only the controls that govern it.
- The Permissions preview label matches the group it counts.
- The Wave-A stopgap (F4a preview label suffix) is superseded and removed.

## Non-goals

- No change to permission resolution, profiles, the kill switch, or any
  store semantics (ADR-081 posture unchanged).
- No change to Tools mode's local-config panel (it already owns the
  master switch + workspace root).
- No new keybindings beyond the rail navigation already shipped in the
  bounded waves.

## Approaches considered

**(a) Second rail section `Agent tools` with one pseudo-row (recommended).**
The rail grows a section heading plus a single row keyed by the existing
hub server key `builtin:tldw_chatbook` (no new keys — Tools/Permissions
already group under it; only the rail presentation changes). The built-in
row keeps `[mcp]` enable/expose toggles + `Copy client config`; the
`Tool gates` group moves under the new row's detail view. The preview
label comes from `server_label`, which becomes `Agent tools (built-in)`.
Trade-off: one more rail row and a detail-routing branch; cheapest honest
fix because every downstream surface already keys on the same identity.

**(b) Rebadge + relocate.** Rename the built-in row "App MCP server" and
move `Tool gates` into the Tools-mode local-config panel. Trade-off: no
new rail row, but buries registration gates inside a mode whose table
they don't belong to, and Tools mode's panel is Console-scoped already —
mixing `[tools]` built-ins there repeats the conflation one pane over.

**(c) Labels only** (status quo + F4a). Trade-off: zero risk, but the
structural confusion persists; rejected as the terminal state.

## Design (approach a)

**Rail** (`mcp_rail.py`): `compose()` renders `Servers` heading + rows,
then an `Agent tools` heading + one row. The pseudo-row's readiness glyph
derives from a new lightweight snapshot: ready when the local master
switch is on, `off (opt-in)` when off (no connection semantics). Row
ids/`_row_keys` machinery is unchanged; the section heading is a Static,
not a row.

**Selection routing** (`mcp_workbench.py`): `_collect_snapshots()`
appends the agent-tools snapshot; `_show_selected_detail()` routes
`source == "agent_builtin"` to a new `MCPServersMode.show_agent_detail()`
that renders the `Tool gates` group (the existing
`_tool_gate_widgets()` builder moves with it verbatim, including the
master-off dependency logic). The built-in server's detail drops the
gates group and keeps a one-line pointer ("Agent tool gates live under
Agent tools in the rail").

**Inspector** (`mcp_inspector.py`): readiness actions for the pseudo-row
are the base set only (View details / Open tool catalog / Open audit) —
no Connect/Check (nothing to connect). `OPEN_TOOL_CATALOG` already
switches to Tools mode.

**Permissions** (`mcp_workbench._build_permission_preview`): no special
case needed once `server_label` is honest; F4a's suffix hack is deleted.

**View state**: saved selections of `builtin:tldw_chatbook` restore onto
the agent-tools row when the saved mode was permissions/tools (the key is
identical; only Servers-mode detail routing branches). No migration file
needed — document the semantic in `get_view_state()`.

**Readiness model** (`MCP/readiness.py`): new snapshot builder mirrors
`builtin_readiness()`'s shape; no new `ReadinessState` values.

## Testing

- `Tests/UI/test_mcp_rail.py`: two-section compose; agent row glyph logic.
- `Tests/UI/test_mcp_servers_mode.py`: gates group renders under agent
  detail, absent from built-in detail; built-in pointer line present.
- `Tests/UI/test_mcp_workbench.py`: selection routing; preview label;
  view-state restore across the split.
- Update the task-2240 preselection test: on a fresh install the lone
  *problem* row is still the built-in server (off/opt-in) — preselection
  semantics unchanged, but the asserted detail contents change.
- Doc-contract test (`test_mcp_documentation_contract.py`) updated with
  the user-guide rewrite.

## Open questions (for reviewer)

1. Should the agent-tools row sort above or below local profiles within
   its own section ordering? (Design says: it is its own section; order
   within it is trivial while there is one row.)
2. Do we keep the ⌂ glyph on the built-in server row only (recommended)
   and give agent tools a distinct marker (e.g. `•`)?
