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

Revised 2026-06-10 after PR #503 (Library content hub / Search-RAG evidence workflow)
landed on dev and superseded several originally planned fixes. Surviving scope:

- Single-row mode strip: all chip/strip/pane sizing moves from inline `styles.*`
  assignments to design-system TCSS tokens (`$ds-library-mode-*`, bar/chip height 1).
  Chip rules use `Button.library-mode-chip` selectors to out-specify Textual's
  `Button.-style-default` tall borders, and the active chip reads via background +
  bold underline (a border would consume the single content row).
- Remove `_frame_library_region`'s hardcoded `#6f7782` border in favor of
  `$ds-grid-line` TCSS; add `LibraryScreen.DEFAULT_CSS` baseline geometry so
  stylesheet-less harness tests render correctly.

Superseded by #503 (kept as dev has them): the dynamic per-mode status row, the
Library Modules buttons (now load-bearing hub navigation with active-state sync,
including `#library-open-search`/`#library-open-collections`), the per-mode
next-action copy, the hub empty-state copy, and the mode-specific inspector
headings ("Hub inspector" etc.).

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
