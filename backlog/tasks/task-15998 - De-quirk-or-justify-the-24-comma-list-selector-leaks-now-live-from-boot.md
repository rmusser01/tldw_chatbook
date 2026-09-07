---
id: TASK-15998
title: 'De-quirk or justify the 24 comma-list selector leaks now live from boot'
status: Done
assignee: ['@claude']
created_date: '2026-08-14 01:10'
labels:
  - css
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual scopes only the LAST selector of a comma list in scoped DEFAULT_CSS (parser-level, confirmed in the 15450 review); the consolidation de-quirks the screen sheets (`build_css.py` `scope_every_selector=True`) precisely because they are live from boot — but the identical argument applies to the 50 consolidated widget classes, whose leaked selectors used to go live at first mount and are now live from app start. Enumerated exposure: 24 leaked selectors across 6 classes (`MCPAuditMode`, `MCPToolsMode`, `MainNavigationBar`, `LibraryScreen`, `MCPScreen`, `SyncStatusWidget`) — all ID selectors or feature-specific class chains, inert in practice, and the boot-stop computed-style diff (dev vs branch) was identical. Either extend de-quirking to the widget sheets or record the asymmetry as a decision with the enumeration pinned by a test, so the leak set cannot grow silently. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Widget-sheet comma-list selectors are either fully scoped like the screen sheets, or the current 24-selector leak set is pinned by a test that fails on growth
- [x] #2 Computed-style parity evidence for whichever path is taken
- [x] #3 The decision and rationale are recorded next to the builder code
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the upstream quirk against the installed Textual 8.2.8 directly: parse `A, .b {…}` as scoped DEFAULT_CSS and confirm only the last selector gains the scope prefix.
2. Enumerate the current leak set (self-stream selectors that do not start with their class name, quirk mode) and confirm the review-time figure of 24 selectors / 6 classes.
3. Build the candidate: pass `scope_every_selector=True` to the widget-defaults stream in `build_css.py`, regenerate all five sheets.
4. Computed-style parity diff, current build vs de-quirked build, over the full 13-destination tour (every node, every stop) using the shared test-app factory — the 15450 method. The specificity shift (0,0,0)→(0,0,1) moves leaked selectors from the self stream (tie-breaker 0) to the scoped stream (tie-breaker -1,000,000); the parity diff is the arbiter of whether any tie flips.
5. If parity is clean, ship (a): builder change + regenerated sheets + a guard test asserting the widget-defaults leak enumeration is 0 (born-red against the quirked build) + decision recorded next to the builder code; update the quirk-documenting comments/tests. If parity is not clean, revert to (b): pin the 24-selector enumeration with a growth-failing test and record the asymmetry with the parity evidence.
6. Run `Tests/UI/test_widget_css_consolidation.py`, `check_bundle_sync.py` (5/5), a tour-adjacent suite, and ruff on touched files; regenerated bundle must differ only by timestamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Path (a) shipped: the widget-defaults sheets are now fully scoped like the screen sheets** — `build_widget_defaults` passes `scope_every_selector=True` (`tldw_chatbook/css/build_css.py`), eliminating the leak class instead of pinning it. The parity diff decided it, per the owner ruling.

**Upstream quirk re-verified against installed Textual 8.2.8 first** (not taken from the 15450 review): parsing `.leaked, #also-leaked, .scoped { … }` as `MyWidget`-scoped DEFAULT_CSS leaves `.leaked` (0,1,0) and `#also-leaked` (1,0,0) unscoped; only `.scoped` gains the prefix, and the injected prefix carries specificity (0,0,0) (`MyWidget .scoped` stayed (0,1,0)).

**Leak enumeration (parser-based, over the generated self sheet's non-class-named selectors): 56 selectors across the same 6 classes at this HEAD, not the review-time 24** — the set had already grown silently in the two days since the 15450 review (LibraryScreen 44, MainNavigationBar 5, MCPScreen 2, MCPToolsMode 2, SyncStatusWidget 2, MCPAuditMode 1; LibraryScreen's Notes-workbench compact rules are the growth). After the change: **0** (all 56 moved to the scoped stream with their written prefix; spot-checked `LibraryScreen #library-notes-filter-row` ×1 and `MainNavigationBar .nav-button.nav-button-clip-ghost` ×6 present in the scoped sheet, 0 unscoped remnants in the self sheet).

**Computed-style parity evidence (the 15450 method, extended).** Probe: full 13-destination tour (ctrl+1..0, F7/F8/F9) on the shared `_build_test_app` factory at 235x52, capturing every node's cascade-applied styles (`node._css_styles.css`) per stop, plus forced `:hover`/`:focus`/`:disabled` sweeps (via `Widget._PSEUDO_CLASSES` patching + re-`stylesheet.apply`) on the Chat, Library and MCP screens. Because 44 of the 56 leaked selectors target Library Notes/Media-canvas nodes that never mount at rest, the probe drove the Library screen through 8 extra sub-stops: notes browse, forced `.library-notes-compact` (breakpoint class forced on `#library-shell-grid`/`#library-canvas`), select mode, sort strip, in-canvas note editor, sync panel, note-create viewport, media canvas; plus a nav-clip-ghost sub-stop on Chat (class forced onto a nav button, then swept). 43 of the 44 distinct leak-target anchors had mounted matching nodes in the capture; the one exception (`#library-note-loading-title`, a transient load surface) is covered by the containment census below. **Result: 22 stops, 9,449 node-states per build, dev(quirked)-vs-branch(de-quirked) diff = 0 differences.** Probe determinism was proven first (A-vs-A rerun = 0 after normalizing the Console session-tab UUID embedded in node ids; before normalization the only diffs were that one widget pair's random ids). Probe was a temporary uncommitted test file; method recorded here.

**Static containment census (corroboration, covers the unmounted residue):** every leaked selector's anchor id/class has its only compose sites inside the declaring widget's subtree — `#library-*` ids only in `library_screen.py`/`Widgets/Library/library_notes_canvas.py`/`library_media_canvas.py`, the compact chains all anchored on `#library-canvas` (composed only in `library_screen.py`), `mcp-*` ids/`.mcp-mode-chip` only in their MCP modules, `.nav-button-clip-ghost` only in `main_navigation.py`, `#scheduling-owner-local`/`#scheduling-last-pull` composed only in `sync_status_widget.py` (`schedules_workbench.py` only handles/queries them). So the written scope prefix cannot un-style anything the leaked selectors were reaching, and the +1 specificity shift is absorbed by the scoped stream's `SCOPED_DEFAULTS_TIE_BREAKER` exactly as the screen sheets' design already handles (LOSE→TIE compensation; no TIE→WIN flip demonstrated — the 0-diff parity is the measurement).

**Born-red pin:** new guard `test_generated_sheets_scope_every_selector` (Tests/UI/test_widget_css_consolidation.py, parametrized over all four generated sheets — it also pins the screen sheets' previously-unpinned de-quirked property) asserts every top-level selector chain in every generated block starts with its declaring class. Red against the quirked build (restored HEAD sheets: `1 failed — 56 selector(s) not scoped to their declaring class`), green after regeneration (4 passed). Vacuity-guarded (asserts banners found and selectors parsed).

**Regeneration discipline:** no generated sheet hand-edited; all five regenerated via `build_css.py`; `check_bundle_sync.py` 5/5; `tldw_cli_modular.tcss` differs only by its `Generated:` timestamp line.

**Tests:** `Tests/UI/test_widget_css_consolidation.py` 21 passed (includes the 13-destination tour/parse-cache integration test and both mounted selection-dialog pilots); + `test_css_build_integrity.py` (31 passed combined); `Tests/UI/test_screen_navigation.py` 129 passed; `ruff check` + `ruff format` clean on the three touched files.

**Modified files:** `tldw_chatbook/css/build_css.py` (the flag + the decision/rationale comment, AC#3), `tldw_chatbook/css/widget_css.py` (`_scope_one_selector` note updated — "filed as a follow-up" resolved), `Tests/UI/test_widget_css_consolidation.py` (new guard + quirk-test docstring), regenerated `widget_defaults_self.tcss`/`widget_defaults_scoped.tcss`/`tldw_cli_modular.tcss` (timestamp only).
<!-- SECTION:NOTES:END -->
