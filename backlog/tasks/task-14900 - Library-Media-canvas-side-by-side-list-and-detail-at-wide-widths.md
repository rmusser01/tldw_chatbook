---
id: TASK-14900
title: Library Media canvas side-by-side list and detail at wide widths
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 17:20'
updated_date: '2026-08-11 14:07'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from task-4023 AC#7 (re-critique 2026-08-09, layout heuristic #8). The bounded
half shipped there: canvas rows no longer inherit the rail's 20-cell title cap, so
titles render in full at wide widths. The structural half remains: the Media canvas
stacks its preview/detail BELOW the list, so on a 170-column terminal the right
~half of the canvas is blank while the user scrolls vertically between list and
preview. A side-by-side (list | detail) split above a width breakpoint — the shape
Collections' workbench already uses (`#library-collections-workbench`) — is a
layout redesign with focus-order, compact-mode, and select-mode implications, too
large to ride a copy/grammar batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At wide widths the Media list and its preview/detail render side by side; below the breakpoint the current stacked layout is preserved
- [x] #2 Keyboard traversal (rows, preview actions, viewer entry) works in both layouts and is advertised honestly by the footer
- [x] #3 Select mode and the bulk-action toolbar remain fully usable in both layouts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. **Precedent read (done)**. Two in-house pieces, reused rather than forked:
   - *Container grammar* — Collections' `#library-collections-workbench`: a plain
     `Horizontal` wrapping a `-list` `Vertical` and a `-detail` `Vertical`, with a
     persistent "No Collection selected." placeholder in the detail half.
     Collections has NO breakpoint of its own (always split).
   - *Breakpoint mechanism* — the screen's ONE measured width regime:
     `LIBRARY_NOTES_COMPACT_BREAKPOINT = 120` measured on `#library-shell-grid`
     (`on_resize` + `_update_library_notes_responsive_state`, transition only on
     crossing), applied as the `library-notes-compact` class on `#library-canvas`
     both at compose time and in `_apply_library_notes_stage_visibility`. That
     class is already applied while the MEDIA canvas is mounted (the host is
     tagged regardless of canvas kind), so the media split can be pure CSS keyed
     off it — zero new breakpoint Python, one width vocabulary on the screen,
     no second split mechanism.
2. **Which views split**: the media LIST view only.
   - List: `Horizontal(id="library-media-workbench")` wraps the existing
     `#library-media-list` and `#library-media-preview` (ids stable), plus a
     wide-only `#library-media-detail-empty` placeholder (Collections' detail
     grammar) shown when the preview is hidden (Select mode / empty list).
   - Trash: NO split — it is a list-only surface; no detail half exists in its
     state (no preview data), and building a restore preview is a new feature
     outside these ACs.
   - Viewer: NO split — it IS the detail, entered deliberately; halving its
     reading width to re-show the list would be a redesign, not this layout fix.
3. **CSS tier**: rules in the bundle source `css/components/_agentic_terminal.tcss`
   (the tier that wins live and under `LibraryHarness`, which loads
   `tldw_cli_modular.tcss`), mirrored in `LibraryScreen.DEFAULT_CSS` as the
   no-bundle harness fallback — the exact two-tier pattern the notes-compact
   geometry already uses. Wide (default) = `layout: horizontal`, panes
   `width: 1fr; height: 100%; overflow-y: auto` (each half becomes its own
   scroll owner); `.library-notes-compact` override = `layout: vertical`,
   panes `width: 100%; height: auto` (byte-identical geometry to today's
   stacked flow). Inline `styles.height = "auto"` on list/preview moves into
   CSS so the class flip can restyle them (inline styles outrank stylesheets).
   Rebuild via `build_css.py`, verify `check_bundle_sync.py`.
4. **Focus order**: DOM order unchanged (toolbar → status → rows → preview
   actions), so Tab order and `_move_library_list_row_focus` (rows share the
   same `#library-media-list` parent) are identical in both layouts; wide panes
   scroll focused children into view via `overflow-y: auto`. Footer stays on the
   shared seam (`_library_footer_shortcuts_for_current_state`) unchanged —
   honest in both layouts because the key set is layout-invariant.
5. **TDD**: new `Tests/UI/test_library_media_side_by_side.py` on the real
   `LibraryScreen` inside `LibraryHarness` (real bundle CSS): RED geometry
   pins at 170x48 (preview strictly right of the list, same row band) and
   100x30 (preview strictly below, full width); select-mode bulk toolbar
   usable at both sizes (buttons on-screen, within terminal width); keyboard
   viewer entry at both sizes (focus row → Enter); placeholder shown only
   wide; footer honesty pin.
6. **Live-verify** at ≥3 widths (wide ~170 / breakpoint edge ~119-121 /
   compact ~100): layouts, select mode + bulk toolbar, keyboard-only
   traversal, footer copy; ANSI captures. Then backlog hygiene, docs stamp,
   commit, self-review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Takeover: a predecessor session implemented the full diff (workbench
split, CSS tiers, 9 tests, the width-contract fix, task-15140 filing) and
died before live verification. The successor re-derived and KEPT the
approach after verifying every claim independently — nothing was redone.

**Approach.** The Media LIST view's list and preview now share a
`Horizontal(id="library-media-workbench")` — Collections'
`#library-collections-workbench` container grammar — that lays them out
1fr | 1fr above the screen's ONE measured width regime and flips back to
the stacked flow below it via pure CSS keyed off the existing
`library-notes-compact` class on `#library-canvas`
(`LIBRARY_NOTES_COMPACT_BREAKPOINT = 120` on `#library-shell-grid`,
maintained at compose time and on every measured `on_resize` crossing).
Zero new breakpoint Python; no second split mechanism. Rules live in the
bundle source (`css/components/_agentic_terminal.tcss`, rebuilt via
`build_css.py`, `check_bundle_sync.py` green) mirrored in
`LibraryScreen.DEFAULT_CSS` as the no-bundle harness fallback. In the
wide split each pane owns its own vertical scroll; a wide-only
`#library-media-detail-empty` placeholder (Collections' detail-pane
grammar) explains the right half when Select mode / an empty selection
hides the preview ("No preview in Select mode." / "No media item
selected."), CSS-visibility only so the compact tier can hide it.

**Which views split (decision):** the media LIST view only. Trash: no
split — a list-only surface with no detail half in its state; a restore
preview would be a new feature outside these ACs. Viewer: no split — it
IS the detail, entered deliberately; halving its reading width to re-show
the list would be a redesign. Conversations canvas: untouched (out of
scope, stacked at every width).

**Width-contract fix ridden along:** `LibraryMediaCanvas` and
`LibraryMediaTrashCanvas` set `styles.width = "13fr"`, which resolves
per-fraction against the HOST's content width (≈13x wider than visible;
`LibraryMediaViewer` documented and fixed the same trap first) — both now
`1fr` so the split panes divide the REAL width. 67 media
`test_library_shell` tests + 65 trash/multiselect tests green after.

**Recompose discipline:** every conditional the workbench compose owns
(`has-preview` class, placeholder text, preview display) re-derives
through `_sync_library_canvas("media")` → canvas-scoped `sync_state`
recompose on every select-mode change; the one in-place patcher
(`_apply_library_row_toggle`) touches only marker/count/disabled, and the
new keyboard-toggle tests pin marker preservation in both layouts.

**Tests:** `Tests/UI/test_library_media_side_by_side.py`, 11 tests on the
real `LibraryScreen` in `LibraryHarness` (real bundle CSS, geometry from
rendered regions): AC#1 side-by-side at 170x48 / stacked at 100x30 /
trash single-column wide; AC#2 traversal + viewer entry + footer honesty
both sizes; AC#3 bulk toolbar usable both sizes (the pre-existing narrow
overflow pinned as-is, tracked in task-15140, independently re-verified
and A/B'd at base 345da0422: right edge fixed at 111 cells, overflow at
<=110-col terminals, byte-identical at base); select-mode keyboard toggle
+ armed-confirm footer honesty both sizes (successor additions); wide
placeholder shown / narrow hidden.

**Live verification** (tmux `lqT5lib*`, scratch TLDW_CONFIG_PATH,
users_name sdd_lq5, 3-item seeded media DB via sqlite3 CLI, clicks by
character index, ANSI captures for focus/marker claims): 170x48 split
(list left, preview right, same row band); armed bulk delete "Delete 2
selected items?…" + footer "esc cancel delete"; keyboard-only Down/Up row
focus (ANSI focus tint moved), Tab → "┃ Open in viewer" → Enter opened
the PREVIEWED item (Alpha, not focused Gamma); Select mode showed "No
preview in Select mode." in the right half with the full toolbar
on-screen. Breakpoint edge: split at 121 cols, stacked at 119. 100x30:
stacked flow, keyboard toggle ("2 selected", both ☑, ○ markers dropped),
armed confirm + honest footer, task-15140's clipped "○ Delet" observed.
Footer flips viewer "esc back to list" ↔ list "esc focus rail" live.
Cleanup: C-q, kill-server, profile removed, live config grepped for probe
inputs — zero matches. (A leftover app instance + tmux server from the
predecessor's interrupted session was found holding the sdd_lq5 profile
and killed before verification.)

**Files:** `tldw_chatbook/Widgets/Library/library_media_canvas.py`,
`library_media_trash_canvas.py`, `tldw_chatbook/UI/Screens/
library_screen.py` (DEFAULT_CSS tier), `tldw_chatbook/css/components/
_agentic_terminal.tcss` + regenerated bundle,
`Tests/UI/test_library_media_side_by_side.py` (new),
`Docs/User_Guide/library/media-and-conversations.md` (layout tour +
stamp), `backlog/tasks/task-15140*` (side-find, id collision-swept clean
across origin/* and all worktrees).
<!-- SECTION:NOTES:END -->
