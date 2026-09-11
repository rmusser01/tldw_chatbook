# ADR-148: MCP Hub Rail Information Architecture and Responsive Triad

Status: Proposed
Date: 2026-09-11
Related Tasks: MCP Hub UX program 2026-09-11 (Waves D and E); supersedes the "accepted trade-off" recorded in `mcp_servers_mode._tool_gate_widgets()`'s docstring
Specs: Docs/superpowers/specs/2026-09-11-mcp-hub-rail-ia-split-design.md; Docs/superpowers/specs/2026-09-11-mcp-hub-narrow-width-triad-design.md
Related: ADR-032 (local:__local__ synthetic hub key), ADR-053 (built-in stdio server boundary)

## Decision

1. The MCP Hub rail presents **two sections**: `Servers` (the built-in
   MCP stdio server, local stdio profiles, server-source records) and
   `Agent tools` (one row, keyed `agent:builtin` — a store identity that
   already exists as `BUILTIN_TOOL_SERVER_KEY` but has no rail presence;
   no two rail rows ever share a `server_key`). The `Tool gates`
   checkbox group moves from the built-in server's detail pane to the
   Agent-tools row's detail pane. The built-in server row keeps only the
   `[mcp]` enable/expose controls that actually govern it. The
   `builtin:tldw_chatbook` inventory group's display label becomes
   `tldw_chatbook (external MCP)` and the Console-side local group's
   label gains `(Console agents)`, so the same tool names legitimately
   appearing under both surfaces (two separate permission domains: the
   agent path and the external MCP server) read as intentional rather
   than duplicated by accident. Selecting the agent-tools row renders
   the unscoped Permissions summary — it never counts a hub group it
   does not represent. These are presentation and label changes only:
   no store key, store entry, or permission resolution semantic changes.
2. Below `_COMPACT_WIDTH` (120 columns), the triad **stacks the
   inspector below the canvas** instead of squeezing three columns: a
   bounded, internally scrolling inspector band under a rail+canvas
   row, via a `layout: vertical` toggle on the grid container plus a
   `Horizontal` wrapper for the rail+canvas row (the `layout:` override
   has in-repo precedent, e.g. `css/screen_feature_watchlists.tcss`).
   Mid-word CSS breaks and unpinned Select widths are eliminated
   regardless of width; the Advanced collapsible renders collapsed
   inside the band at compact widths.

## Why

The single built-in rail row conflated three subsystems (the app-as-MCP-
server, the agent tool-catalog registration gates, and the Permissions
preview scope for the agent built-ins), which no amount of explanatory
copy fully hid — most visibly when selecting the "MCP server" row made
the permission preview count 30 agent tools under the label
`tldw_chatbook`. Separately, at ≤120 columns the three-column triad
rendered the inspector unreadable (mid-token word breaks, truncated
JSON), and the F-057 narrow-width work had covered only the tables.

## Alternatives considered

- **Labels-only fixes** (rename rows, suffix the preview label): zero
  risk, but leaves the structural confusion; kept only as the Wave-A
  stopgap until this ADR's split lands.
- **Relocating Tool gates into Tools mode's local-config panel**: buries
  registration gates in a Console-scoped panel and repeats the conflation
  one pane over.
- **Hiding the inspector behind a toggle at compact widths**: preserves
  canvas height but hides the action surface (Connect/Edit/permission
  explanations) behind a conditional control — the same discoverability
  failure class as the review's first-run finding.
- **Free-form narrow-column squeezing with wrap fixes only**: does not
  make a ~20-column inspector pane usable.

## Consequences

- The rail, workbench selection routing, one detail-pane builder, and
  the narrow-width grid CSS change; permission resolution, store shape,
  keybindings policy (ADR-031), and all ADR-081/090 safety semantics are
  untouched.
- Saved view state keyed on `builtin:tldw_chatbook` keeps working; only
  Servers-mode detail routing branches on the new snapshot source.
- The Wave-A preview-label stopgap (F4a) is deleted when this lands.
- Tests pinning the built-in detail contents, rail row enumeration, and
  task-2240 preselection assertions are updated with the split.
