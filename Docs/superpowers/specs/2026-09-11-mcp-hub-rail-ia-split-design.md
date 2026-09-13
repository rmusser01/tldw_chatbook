# MCP Hub Rail IA Split — Design

Date: 2026-09-11
Status: Draft (awaiting review)
Review basis: MCP screen UX review 2026-09-11 (Finding F4b); evidence PNGs in Docs/superpowers/specs/assets/2026-09-11-mcp-hub-ux/ (01, 03, 08)
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

## Identity map (verified against code, 2026-09-11)

The conflation involves **three distinct store identities**, which any
split must keep honest:

| Identity | What it is | Where it appears today |
| --- | --- | --- |
| `builtin:tldw_chatbook` (`BUILTIN_SERVER_KEY`) | The standalone MCP stdio server **and** its published inventory group (30 HubTools via `builtin_tools_from_inventory`, label `tldw_chatbook`) | Rail row; Servers detail; a full Permissions/Tools section |
| `local:__local__` (ADR-032) | Console/agent workspace-web-Watchlists tools (`fs_*`, `git_*`, `shell_exec`, `watchlists_*`) | Tools table + Permissions section "Local workspace, web, and Watchlists" |
| `agent:builtin` (`BUILTIN_TOOL_SERVER_KEY`, permission_store.py:1618) | In-process agent built-ins (`read_file`, `list_directory`, …) | Permissions-only extra section (`_builtin_permission_matrix_rows`); **never in `_last_hub_tools`** |

**Additionally discovered during this verification pass:** the same tool
NAMES can render under BOTH `local:__local__` and `builtin:tldw_chatbook`
(whether they do depends on the local-tools master switch and the
inventory projection). They are two separate store entries, so an Allow
on one does not affect the other, and nothing in the matrix distinguishes
the surfaces. [Qodo #2622 #7: the literal `fs_edit` example overstated
the guaranteed overlap — the shipped labels cover the general case, and
the pinned two-row test constructs the overlap explicitly.] The split
below fixes the labels.

## Goals

- The rail presents **two honest sections**: `Servers` (built-in MCP server
  + local profiles + server-source records) and `Agent tools` (the
  in-process catalog the Console uses).
- Each section's detail pane renders only the controls that govern it.
- The Permissions matrix labels say which surface each section governs
  (`tldw_chatbook (external MCP)` vs the Console agent groups); selecting
  the agent-tools row renders the **unscoped** preview summary (it never
  miscounts a group it doesn't represent).

## Non-goals

- No change to permission resolution, profiles, the kill switch, or any
  store semantics (ADR-081 posture unchanged).
- No change to Tools mode's local-config panel (it already owns the
  master switch + workspace root).
- No new keybindings beyond the rail navigation already shipped in the
  bounded waves.
- No deduplication of the underlying store entries (the two surfaces are
  genuinely separate permission domains); this spec only labels them so
  the duplication reads as intentional.

## Approaches considered

**(a) Second rail section `Agent tools` with one pseudo-row (recommended).**
The rail grows a section heading plus a single row keyed `agent:builtin`
—a store identity that already exists (`BUILTIN_TOOL_SERVER_KEY`) but
has no rail presence, so no rail row ever shares a `server_key` (two
rows on one key would break `MCPRail`'s `_row_keys`/`is-active` selection
identity). The built-in row keeps `[mcp]` enable/expose toggles +
`Copy client config`; the `Tool gates` group moves under the new row's
detail view. The `builtin:tldw_chatbook` inventory group's display label
becomes `tldw_chatbook (external MCP)` (label only — the key, and every
store entry under it, is untouched). Selecting the agent row scopes the
Permissions preview to nothing it can count: `_build_permission_preview`
finds no `agent:builtin` group in `_last_hub_tools` (verified — agent
built-ins are never in the hub catalog) and falls to the unscoped
"global default · N overrides" summary, which is the honest answer.
Trade-off: one more rail row, a new snapshot source value, and a
detail-routing branch.

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

**Permissions** (`hub_tool_catalog.builtin_tools_from_inventory` +
`mcp_workbench`): the inventory group's `server_label` becomes
`tldw_chatbook (external MCP)` — this is the display fix for the
duplicate-row confusion (the Console-side groups keep their existing
labels, with the local group's label gaining `(Console agents)` at the
same time so the pairing reads as deliberate). Preview behavior needs no
special case: selecting the agent row finds no hub group and renders the
unscoped summary; selecting the built-in server row scopes to the
external-MCP group under its new honest label. The Wave-A preview-suffix
stopgap (F4a) is dropped from Wave A and never lands — this spec
replaces it.

**View state**: the agent row's key `agent:builtin` is new to the rail;
saved selections of `builtin:tldw_chatbook` keep restoring onto the
built-in *server* row exactly as today. No migration, no ambiguity.

**Readiness model** (`MCP/readiness.py`): new snapshot builder mirrors
`builtin_readiness()`'s shape; no new `ReadinessState` values.

## Verification notes (Qodo #2622 #8)

The built-in row REMAINS the lone rail row the fresh-install preselect
lands on — the Agent tools snapshot is deliberately not in `_snapshots`,
so task-2240's heuristic is unchanged by design.

## Testing

- `Tests/UI/test_mcp_rail.py`: two-section compose; agent row glyph logic.
- `Tests/UI/test_mcp_servers_mode.py`: gates group renders under agent
  detail, absent from built-in detail; built-in pointer line present.
- `Tests/UI/test_mcp_workbench.py`: selection routing; preview label;
  view-state restore across the split; a new test pinning the
  duplicate-row disambiguation (a tool name present under both
  `local:__local__` and `builtin:tldw_chatbook` renders two rows whose
  section labels name the two surfaces).
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
