# Library UX Fixes + Notes Workbench Design

Date: 2026-06-09
Status: Approved (user-validated via brainstorming session)
ADR required: no
ADR path: N/A
Reason: Direct implementation of the existing master-shell design-system contract
(`Docs/Design/master-shell-design-system-contract.md`). No storage, navigation-contract,
or cross-module boundary changes.

## Problem

1. The Library screen — the most mature destination in the master-shell redesign — has
   accumulated UX debt: redundant header chrome, duplicated pane headings, two navigation
   paths to the same surfaces, inline style assignments that bypass the design system,
   and verbose non-actionable empty states.
2. Notes has no redesigned surface. `UI/Screens/notes_screen.py` runs inside the new
   shell (BaseAppScreen, scope-aware state, Console handoff) but still mounts the legacy
   `NotesSidebarLeft`/`NotesSidebarRight` widgets. Several controls on it are dead today
   (create-from-template, import, sort-order, load/edit-selected) because the screen never
   dispatches the legacy `NOTES_BUTTON_HANDLERS` map — only the old `UI/Notes_Window.py`
   tab does.

## Decisions

- **Notes stays a Library sub-surface.** No change to `SHELL_DESTINATION_ORDER`; the
  `"notes"` route keeps resolving to the Library destination. Entry points: Library's
  "Open Notes" button and the command palette. `Docs/Design/New_UI/Notes.png` is
  directional inspiration, not a literal contract.
- **Notes v1 scope = feature parity** with the old screen, rebuilt on the destination
  workbench pattern. Review mode, folder tree, and the agent-handoff strip from the
  mockup are explicitly deferred.
- **Combined workbench, not Browse/Edit tabs.** List, editor, and inspector are visible
  simultaneously; a small mode strip (Notes | Sync | Templates) swaps the workbench body.
  Browse/Edit as separate modes (as drawn in the mockup) was rejected because the old
  screen's core strength is list+editor side by side.
- **Mode switching by display-toggling pre-composed regions**, not remount, so unsaved
  editor content survives mode flips. (Library's remount approach is fine for copy panes,
  wrong for a live editor.)
- **Console-style collapsible rails.** Per user direction, Notes follows the Console
  screen's UI principles: the navigator and inspector panes are collapsible rails with
  Console rail headers (title + compact collapse button) that collapse to
  `ConsoleRailHandle` handles, reusing the `console-rail-*` design-system classes. The
  editor is the primary surface, expanding when rails are collapsed — mirroring Console's
  Context/Inspector rails. Rail state persists via the existing
  `left/right_sidebar_collapsed` fields.
- **Library issues get fixed, not just documented** (user choice), as an independent PR
  that lands first; the Notes screen reuses its slimmed header pattern and a generalized
  `.destination-mode-chip` style.

## Notes screen layout (Notes mode)

```
┌ Notes ─────────────────────────────────────────────┐
│ [Notes] [Sync] [Templates]           status | Local│
├───────────┬──────────────────────────┬─────────────┤
│ Search…   │ Title: ____________      │ Details     │
│ ┌───────┐ │ ┌──────────────────────┐ │ Keywords    │
│ │note 1 │ │ │ editor               │ │ Sync status │
│ │note 2*│ │ │                      │ │ Export ▾    │
│ └───────┘ │ └──────────────────────┘ │ Delete      │
│ + New ▾   │ saved · 234 words        │ Use in      │
│ Import    │                          │ Console     │
└───────────┴──────────────────────────┴─────────────┘
```

- **Navigator pane (2fr):** debounced FTS search, keyword filter, sort select + order
  toggle, New / Create-from-template / Import, Local/Server/Workspaces scope sections.
- **Editor pane (5fr):** title input, TextArea editor, workspace context panel,
  controls row (save, preview, sync, auto-save switch, unsaved indicator, word count).
- **Inspector pane (2fr):** note metadata (created/modified/version), keywords editor,
  per-note sync status, export/copy actions, delete with confirmation, Use in Console.
- **Sync mode:** embeds the section containers from `notes_sync_widget_improved.py`
  (status card, quick sync, progress, recent activity). Sync profiles/history is a
  documented follow-up.
- **Templates mode:** template list + content preview + create, returning to Notes mode
  with the new note loaded.

## Library fixes (summary)

Slim the status row to `{status} | Local`; remove inline `styles.*` in favor of TCSS;
de-duplicate the Inspector heading; drop the mode next-action prose; remove the
`#library-open-search` / `#library-open-collections` buttons (pure duplicates of mode
chips — the other source-browser buttons do real navigation and stay); make empty states
actionable (primary "Import Sources" button navigating to ingest).

## Constraints

- Legacy `UI/Notes_Window.py`, `Event_Handlers/notes_events.py`, `NOTES_BUTTON_HANDLERS`,
  and `app.py` are untouched; the only sidebar-file edit is a behavior-neutral extraction
  of list-population methods into a shared mixin.
- All functional widget IDs are preserved (67 tests in `Tests/UI/test_notes_screen.py`
  and ~8 Library test files pin structure); expected test diffs are enumerated in the
  implementation plan.
- TCSS edits require rebuilding `css/tldw_cli_modular.tcss` via `css/build_css.py`.

## Delivery

Three PRs: (A) Library UX fixes → (B1) Notes workbench, Notes mode with feature parity →
(B2) Sync + Templates modes. Implementation plan with file-level steps lives in the
session plan (`help-me-review-the-shiny-origami.md`) and the Backlog tasks for PR A/B1/B2.
