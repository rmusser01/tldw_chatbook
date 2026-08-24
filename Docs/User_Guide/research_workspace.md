# Research — authority-explicit workspace foundation

## What this screen is for

Research is the destination for workspace-scoped research. In the current
foundation phase, it lets you choose the Local or Server workspace catalog,
inspect the separate Sources, Grounded Chat, and Studio regions, and arrange
their panes. It does not yet ingest sources, send grounded-chat messages, or
generate Studio outputs.

Research has two separate screens:

- **Workspace** is the new source-to-answer-to-output workbench.
- **Runs** is the existing durable Research Runs operator for run lifecycle,
  checkpoints, events, bundles, and artifacts.

The mode buttons navigate between real screens. Opening **Runs** does not embed
or replace Workspace, and direct links to the existing `research` route still
open Runs.

## Getting there

- Press **F10** from anywhere to open Research Workspace.
- Select **F10 Research** in the top navigation bar.
- Open **Ctrl+P** and choose "Tab Navigation: Switch to Research". Search
  aliases include "research workspace", "research runs", "research sessions",
  "deep research", and "notebook"; the destination command opens Workspace.

Research sits after Library and before Artifacts. Inside the Research header,
use **Workspace** and **Runs** to move between its two screens.

## Layout tour

The pinned header keeps three different facts separate:

- **Workspace data: Local | Server** chooses the complete workspace catalog
  and identity authority for this screen.
- **Processing: not configured** is a separate processing-route status. Local
  data does not promise on-device inference.
- **Sources: 0 ready** reports source readiness. Source attachment arrives in
  a later phase.

Below the header are the Sources, Grounded Chat, and Studio regions. The status
row names the selected catalog, whether recovery is required, and that the
current surface is foundation-only.

### Workspace data authority

**Local** reads only this installation's local research-workspace catalog.
**Server** reads only the selected server profile and principal's catalog.
Workspace identities remain qualified by Local/Server authority and, for
Server, by profile and principal. Identically named or numbered Local and
Server workspaces are not combined.

Switching authority is not a Copy, sync, or fallback operation. If Server is
missing, unreachable, or needs authentication, **Server remains selected** and
the screen shows the problem and recovery. It never substitutes Local data.

### Responsive panes

At wide widths, Sources and Studio can be collapsed independently around the
dominant Grounded Chat pane. The visible controls are exact ASCII labels:

| Pane | Expanded: collapse | Collapsed: reveal |
|---|---|---|
| Sources | **`<---`** | **`--->`** |
| Studio | **`--->`** | **`<---`** |

The full control names are "Collapse Sources pane", "Expand Sources pane",
"Collapse Studio pane", and "Expand Studio pane". Collapsing a focused pane
moves focus to its reveal control; expanding it moves focus into the revealed
pane. When both side panes are collapsed, Grounded Chat uses the available
width.

From 100 through 149 columns, Chat appears with at most one companion pane and
the **Sources (0)**, **Chat**, and **Studio (0)** pane controls choose what is
visible. Below 100 columns, exactly one pane is visible and those same controls
switch it. Responsive collapse is temporary: widening the terminal restores
your stored wide-layout choices.

## Features & controls

| Control | What it does now |
|---|---|
| **Workspace** | Opens the Research Workspace screen. |
| **Runs** | Opens the separate durable Research Runs screen. |
| **Local** | Selects and loads only the Local workspace catalog. |
| **Server** | Selects and loads only the active Server catalog, or keeps Server selected with recovery. |
| **Manage Workspaces...** | Opens Settings workspace management. |
| **`<---` / `--->`** | Collapses or reveals the corresponding wide/medium side pane with focus relocation. |
| **Sources (0) / Chat / Studio (0)** | Chooses the visible pane arrangement at medium and narrow widths. |

Pane preferences are a **device-only overlay** stored privately on this
installation and keyed by the qualified workspace identity. They are not
uploaded, shared, or available on another device. Width-forced collapse is not
saved over those choices.

## Common tasks

1. **Inspect Local workspaces.** Open Research with **F10**, choose **Local**,
   and read the selected Local workspace name in the workspace row.
2. **Check a Server catalog without fallback.** Choose **Server**. If the
   configured server is ready, its qualified catalog loads. Otherwise, leave
   Server selected and follow the displayed retry, configuration, or
   authentication recovery.
3. **Give Chat more room.** At a wide terminal, use Sources **`<---`** and
   Studio **`--->`**. Use Sources **`--->`** or Studio **`<---`** to restore a
   pane.
4. **Move between Workspace and Runs.** Use the header's **Workspace** and
   **Runs** buttons. Each is a separate screen with its own state.
5. **Manage Local workspace records.** Choose **Manage Workspaces...** to open
   Settings. Full local workspace management and destructive deletion remain
   Settings-owned.

## Keyboard & commands

Research Workspace adds no screen-local letter shortcuts in this foundation
phase. Use Tab/Shift+Tab to move through visible controls. Global keys and
pane-cycle behavior are described in the [guide index](index.md).

## Related settings & docs

- **Settings ▸ Workspaces** owns full Local workspace management.
- [Library](library.md) owns global source browsing, ingestion, and editing.
- [Console](console.md) remains the live agent/tool surface; Research Workspace
  is not a second Console.
- The approved architecture is recorded in
  [ADR-078](../../backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md).

## Quirks & troubleshooting

- This is the TASK-21507 foundation only. Sources has no Add/ingest action,
  Grounded Chat has no Send action, and Studio has no Generate or Quick Notes
  editor yet. The empty copy is intentional; future controls are not advertised
  as working.
- **Processing: not configured** is honest Phase 1 status. Choosing Local says
  where workspace data is owned, not where a future model request will run.
- If Local storage could not initialize, Local remains selected with a recovery
  message. If Server is unavailable, Server remains selected; choose another
  authority only by explicitly pressing its selector.
- Pane choices persist only after a qualified research workspace is available.
  They are presentation preferences, not canonical workspace data.

—
*Verified against codex/research-workspace @ 5370de4aa + TASK-21507 Task 5 — 2026-08-24*
