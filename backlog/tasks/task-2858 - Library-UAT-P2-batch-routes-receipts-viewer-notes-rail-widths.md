---
id: TASK-2858
title: 'Library UAT P2 batch: routes, receipts, viewer, notes, rail, widths'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 13:48'
labels:
  - library
  - ux
  - uat-2026-08-06
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 P2 findings (LIB-03/09/11/12/13/14/15/17/18/19), critique snapshot
`.impeccable/critique/2026-08-07T01-01-42Z__tldw-chatbook-ui-screens-library-screen-py.md`,
observed at dev `6ffa56516`. Grouped for one pass; split if any item grows.

1. LIB-03 — Entry-route-dependent landing: palette "Library" lands the Import canvas while
   "Switch to Library" lands the hub; re-entering Library resets the previously visited canvas.
2. LIB-09 — Help/advertisement contract: F1 on the Media canvas lists skills/evidence keys that
   do nothing there; media-viewer footer advertises `u` which is inert in the viewer.
3. LIB-11 — Empty "Export chatbook" click is a silent no-op (no toast/disabled styling/reason).
4. LIB-12 — Successful export leaves no durable receipt (zip written; canvas pixel-identical).
5. LIB-13 — Media viewer renders raw markdown while Notes Preview renders it properly.
6. LIB-14 — Note lifecycle: "Blank note" commits a DB row before typing; literal "Untitled" text
   must be hand-deleted; version bumps from clicking Preview.
7. LIB-15 — Rail gloss/count lifecycle is nondeterministic ("Collections — item sets" → "(0)" →
   bare → "(1)"; some rows keep glosses, some lose them).
8. LIB-17 — Click into a prefilled query lands the cursor at position 0 (typed text prepends);
   the rail search box retains stale queries across screen switches.
9. LIB-18 — Width degradation: row labels truncate mid-word at ≤120 ("Conversa... (0)",
   "Flash... due: 0"); at ≤100 the footer's screen-specific keys hide behind a leading "…"; at 80
   the nav hard-cuts a tab label mid-word.
10. LIB-19 — Three folder-notes concepts (Database mode, Files mode, Sync) are never related to
    each other anywhere; at minimum one sentence on each should place it relative to the others.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library landing is route-independent (one canonical landing) or intentionally routed with the difference stated on-screen; revisits restore the last canvas or intentionally reset (decision recorded)
- [x] #2 F1 and footers advertise only keys that work on the current surface
- [x] #3 The Export button is never a silent no-op, and a successful export leaves a durable on-canvas receipt with the output path
- [x] #4 The media viewer renders markdown (with a raw toggle) for markdown media
- [x] #5 Blank notes no longer commit literal "Untitled" rows that require hand-deletion; version stamps change only on content saves
- [ ] #6 Rail glosses/counts follow one deterministic rule across all rows
- [ ] #7 Prefilled search inputs are editable without cursor traps, and stale rail queries do not survive screen switches
- [ ] #8 At 120/100/80 columns no rail row label truncates mid-word, and each finding's surface is re-verified live
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Four-group SDD arc on fix/library-uat-p2-batch (plan: docs/superpowers/plans/library-uat-p2-batch.md):\n1. Entry routing + canvas restoration (LIB-03, AC1) — generic entries land one canonical surface, deep links keep their labeled destinations, revisits restore.\n2. Honest advertisement + export feedback (LIB-09/11/12, AC2-3) — full BINDINGS gate audit, disabled-with-reason export button, durable last-export receipt.\n3. Content surfaces (LIB-13/14, AC4-5) — viewer markdown via the existing Notes renderer + Raw toggle; note lifecycle (placeholder title, no premature commit, version on save only).\n4. Rail determinism + input traps + widths + folder-notes copy (LIB-15/17/18/19, AC6-8) + close-out.\nEvery item re-verified at HEAD first (P1 arc + task-1993 may have moved several). Same process as the P1 arc: TDD, task review + scoped re-reviews per fix round, live tmux verification per task, final whole-branch review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 (LIB-03 -> AC#1) -- Entry routing + canvas restoration:

HEAD re-verification (live tmux, socket p2T1lib22744, scratch profile sdd_p2t1) found all
three originally-observed behaviors ALREADY CORRECT at branch-base 6b38a13b8 -- no product
code change was needed:
  (a) Palette query "Library" now top-ranks "Tab Navigation: Switch to Library" (generic) and
      selecting it lands the restored/hub canvas, not Import. This differs from the original
      6ffa56516 observation, where the palette's "Library: Import..."/"Library: Add content..."
      entry apparently outranked the generic command for that query.
  (b) Palette "Switch to Library" lands the hub on a first visit -- unchanged/correct.
  (c) Visit Search/RAG -> Home -> generic Library re-entry (nav-bar tab click) RESTORES
      Search/RAG -- it does NOT reset to Import. This is the direct opposite of the original
      finding.
  Also verified: explicit deep links ("Library: Import...", "Library -- Skills") still land
  their own labeled canvas even when a different canvas was left behind by a prior visit --
  a real Import visit after Search/RAG round-tripped correctly overrode the restore, then
  itself became the new "last visited" canvas for the next generic entry.

Root cause of the original finding (recorded, not fixed here): the cross-visit restore
mechanism (ScreenStateStore, ADR-033/task-644 + LibraryScreen.save_state/restore_state,
landed by fed3662f6 "real cross-visit state persistence for Library, Media, and Search
screens") already predates 6ffa56516 and was already correct. The most likely explanation
for the "always resets to Import" observation is (a): the palette's ambiguous "Library"
query kept landing the tester on the Import deep link, which -- correctly, by design --
also becomes the next "last visited" canvas, producing the appearance of a persistent
reset. Whatever changed the palette ranking between 6ffa56516 and this branch (candidate:
task-2857's Import/Export naming unification) is not chased further here since (a) is
independently confirmed fixed live.

Decision (recorded per the task's own direction): REVISITS RESTORE THE LAST CANVAS
(restore-over-reset), not an intentional reset. No collision with an ingest-in-progress
reclaim contract was found -- the ingest registry listener only recomposes the current
canvas on job-state changes; it never reassigns `_library_selected_row_id`.

No `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` / `LIBRARY_NAV_MODE_TO_ROW_ID` / ScreenStateStore
changes were made; the existing seams already implement the direction exactly (generic
entry -> hub-or-restore via ScreenStateStore; explicit deep link -> its own canvas via
`message.screen_context` or the legacy-alias context table, applied AFTER restore so it
overrides).

Tests added (TDD, Tests/UI/test_screen_navigation.py, real TldwCli app + real LibraryScreen
via `_build_test_app()` + `app.handle_screen_navigation`, mirroring
`test_rapid_tab_switch_storm_leaves_no_zombie_widgets`'s harness):
  - test_generic_library_entry_lands_hub_on_first_visit
  - test_deep_link_library_route_lands_its_canvas_over_restored_state
  - test_generic_reentry_restores_last_visited_library_canvas
RED evidence: an initial harness bug (polling `app.screen`'s type instead of
`_initial_screen_pushed`) made all three fail for the wrong reason -- fixed, then all
three passed against real HEAD. To prove non-vacuous coverage of the actual routing
contract (not just the harness), the restore-state application and the nav-context
application in `app.py`'s `_complete_screen_navigation` were each temporarily neutered
(`if False and ...`) and reverted via Edit (never git checkout): disabling restore-state
application failed exactly `test_generic_reentry_restores_last_visited_library_canvas`;
disabling nav-context application failed the deep-link and restore tests but not the
first-visit test (which needs neither). `git diff tldw_chatbook/app.py` is empty --
product code is unchanged.

Verification: `Tests/UI/test_screen_navigation.py` full file (109 passed, includes the 3
new tests) and `Tests/Library --collect-only -q` (1079 collected, 0 errors) both green.
Live tmux covered all three flows plus the two deep-link checks above; no CSS touched
(no code changes at all), so no build_css.py run was needed.

Task 2 (LIB-09/11/12 -> AC#2/AC#3) -- Honest advertisement + export feedback:

HEAD re-verification (live tmux, socket p2T2lib9953, scratch profile sdd_p2t2) at
471dc47ee found LIB-09 SPLIT into two independently-true halves:
  (a) The FOOTER half was already fixed pre-branch (task-420's row-scoped
      registration: LIBRARY_SHORTCUTS, the only set containing "u", is selected
      only for LIBRARY_ROW_BROWSE_SEARCH) -- confirmed live: Media canvas footer
      never showed "u", Search/RAG's did.
  (b) The F1 half was NOT fixed and was NOT touched by the P1 arc's check_action
      gates: LibraryScreen has no `action_show_workbench_help` override, so
      app.py's generic fallback (`_show_generic_screen_help`/`_bindings_to_
      shortcuts`) flattened the raw static BINDINGS list UNCONDITIONALLY --
      it never calls check_action at all. Reproduced live exactly as the
      original finding: F1 on the Media canvas showed "LibraryScreen Shortcuts"
      with "ctrl+s: Save skill" / "escape: Back to skills list" while browsing
      media. Root cause: check_action gating governs Textual's live key
      resolution and footer registration is its own separate seam, but this
      app's F1 help bypasses check_action entirely via the generic fallback.

Fix:
  - Added check_action gates for the three previously-ungated BINDINGS actions:
    `library_rag_use_in_console` ("u", gated to LIBRARY_ROW_BROWSE_SEARCH,
    mirroring the action body's own guard) and `library_rag_result_card_select`/
    `_open` (Enter/"o", gated to `_focused_library_rag_result_card_index()
    is not None`). Every action on LibraryScreen.BINDINGS now has an explicit
    check_action branch -- none fall through to the default `return True`.
  - Added `LibraryScreen.action_show_workbench_help()` (the same delegation
    seam SettingsScreen already uses for its own per-category F1 filtering) +
    a `_active_library_binding_shortcuts()` helper that filters BINDINGS
    through check_action with the identical keep/drop rule Textual's own
    `Screen.active_bindings` uses. This is the actual fix -- check_action
    gates alone do not change F1's output without this override.
  - LIB-11: `LibraryExportFormState` already had `export_enabled` (single
    predicate covering running/counts-loading/empty-scope/no-destination) and
    the submit Button's `disabled=not state.export_enabled` was already wired
    at compose time AND patched in place on both mutation paths (counts
    landing, destination chosen via recompose) -- Textual's own
    `Button.press()` already refuses to post `Pressed` while `disabled`, so a
    "silent no-op" click was never actually reachable through a real click.
    What was missing was the "why": no tooltip existed, so a correctly-
    disabled button gave no explanation, reading as a dead/broken control.
    Added `export_button_tooltip()` (library_export_state.py) mirroring
    export_enabled's own predicate order, reusing EMPTY_SCOPE_COPY verbatim
    for the empty-scope case per the task's "same predicate" requirement, and
    wired it everywhere `disabled` is set (compose, counts-landing patch,
    run-completion patch) so it never goes stale.
  - LIB-12: added `_library_export_last_path`/`_library_export_last_at`
    screen fields (deliberately NOT touched by
    `_reset_library_export_transient_state`, which resets every other export
    field on every canvas entry), a `format_last_export_line()` pure
    formatter (library_export_state.py, "just now"/"Nm ago"/"Nh ago"/"Nd ago"
    style matching the two existing local precedents in
    activity_log.py/ChatbookExportManagementWindow.py), a new always-mounted
    `#library-export-last-line` Static (display-toggled, same discipline as
    the empty-scope/status/error lines), and set the two fields in
    `_apply_library_export_success` (before the staleness guard, alongside
    the notifications, since the zip genuinely landed regardless of which
    canvas is displayed). Round-tripped through save_state/restore_state too
    (an "obvious existing seam" already used for the screen's other
    selection/view state) so the receipt survives a full navigate-away-and-
    back to Library, not just an in-session canvas switch.

Tests added (TDD): 4 new tests in Tests/UI/test_screen_navigation.py
(check_action gates for the 3 new actions; a BINDINGS audit test enumerating
the full static list and asserting every action is gated-or-declared-
universal on a bare landing instance; a behavioral test of
action_show_workbench_help pinning the exact LIB-09 regression -- Media list
canvas must not advertise ctrl+s/skill-editor/RAG keys). Updated 2 pre-
existing check_action tests whose "unrelated action" control used
`library_rag_use_in_console` (now gated) -- switched to a genuinely
nonexistent action name. 20 new tests in Tests/Library/test_library_export_
state.py (export_button_tooltip predicate order incl. verbatim EMPTY_SCOPE_
COPY reuse; format_last_export_line's just-now/m/h/d boundaries;
last_export_line pass-through). New Tests/UI/test_library_export_receipt.py
(13 tests): compose-time disabled+tooltip rendering via real Pilot mounts;
a REAL Pilot click-dispatch test proving a disabled Export button posts NO
Button.Pressed (with an enabled-button control case proving the harness
itself dispatches clicks correctly); in-place patcher tests for both
_apply_library_export_counts and _update_library_export_canvas_after_run
confirming tooltip/receipt stay in sync without recompose; receipt
survival across `_reset_library_export_transient_state` and across
save_state/restore_state round-trips. One REAL (non-mocked) export test
added to Tests/Library/test_library_export_roundtrip.py, seeding real DBs,
driving the real LocalChatbookService, and asserting the receipt fields
`_apply_library_export_success` sets name the exact real on-disk zip path.

RED evidence: reverse-applied the product diff (git apply -R, files under
tldw_chatbook/ only) and reran the new/changed tests -- 2 ImportErrors
(EXPORT_BUTTON_COUNTING_TOOLTIP etc. didn't exist) and 4 AttributeError/
assertion failures (action_show_workbench_help missing; check_action
returning True where a gate was expected) confirmed RED for exactly the
right reasons, then git apply restored the implementation and every test
went GREEN.

Verification: Tests/Library/test_library_export_state.py +
Tests/UI/test_library_export_receipt.py (48 passed);
Tests/UI/test_screen_navigation.py full file (113 passed, incl. 4 new);
Tests/Library/test_library_export_roundtrip.py (5 passed, incl. 1 new);
Tests/Library --collect-only -q (1092 collected, 0 errors). A combined run
of test_library_export_cancel.py + test_library_shell.py +
test_library_export_execution.py + test_skills_library_flow.py showed 2
failures (388 passed); A/B isolation confirmed both pre-existing:
test_landing_footer_advertises_the_landing_keyboard_story is task-3022's
known ambient debt (named in the plan's Global Constraints), and
test_library_shell_ingest_canvas_live_updates_without_manual_recompose is
order-dependent flakiness unrelated to this change -- it passed both alone
and in a standalone full run of test_library_shell.py (352 passed, only the
known-debt test failed). Neither test touches export/binding code this task
changed.

Live tmux (socket p2T2lib9953, scratch profile sdd_p2t2, cleaned up after):
F1 on the Media canvas now shows only "escape: Focus rail" (no skill-editor/
RAG contamination); F1 on Search/RAG shows "u: Use Library context in
Console"; F1 in the Notes editor shows only "escape: Back to notes list".
Footer never showed "u" outside Search/RAG. The disabled Export button
rendered visibly dim/non-bold (ANSI-confirmed) vs. the enabled state's bold
accent styling, and a real click on it produced zero visible change (no
crash, no notification, canvas pixel-identical) -- proving the click-
swallow-prevention claim live, not just in the test harness. A real export
(1 note, no media/conversations) wrote a genuine zip to
/private/tmp/p2T2/receipt_test.zip (1120 bytes, confirmed a valid zip via
`file`), and "Last export: /private/tmp/p2T2/receipt_test.zip · just now"
appeared above the submit button; switching to Media and back to Export
(fresh form, destination reset to "No destination chosen" as expected) still
showed the same receipt line, confirming LIB-12's persistence contract.

Docs: Docs/User_Guide/library/import-and-export.md updated (layout tour +
features table describe the tooltip and the "Last export: …" receipt row) with
a new "Verified against branch-base `6b38a13b8`" stamp for this change.

Files changed: tldw_chatbook/Library/library_export_state.py,
tldw_chatbook/Widgets/Library/library_export_canvas.py,
tldw_chatbook/UI/Screens/library_screen.py,
Tests/UI/test_screen_navigation.py, Tests/Library/test_library_export_state.py,
Tests/Library/test_library_export_roundtrip.py,
Tests/UI/test_library_export_receipt.py (new),
Docs/User_Guide/library/import-and-export.md.

Task 3 (LIB-13/14 -> AC#4/AC#5) -- Content surfaces: media viewer markdown +
note lifecycle:

HEAD re-verification (live tmux, socket p2T3lib23968 -- an earlier socket
p2T3lib13583 was reused mid-task for the first walkthrough, cleaned up, and
restarted after the CSS fix landed -- scratch profile sdd_p2t3, cleaned up
after) at 6b38a13b8:
  LIB-13: STILL PRESENT. `LibraryMediaViewer._content_renderable()` only
    ever built a plain/highlighted-``Text`` ``Static`` from the raw string
    -- no ``Markdown`` widget existed anywhere in the media viewer. Live
    import of a real `.md` fixture (heading + GFM table) confirmed literal
    `#`/`##`/`|` shown raw, exactly as originally reported.
  LIB-14: STILL PRESENT on all three sub-points. `handle_library_notes_
    create_blank` unconditionally called the create seam with
    `title="Untitled", content=""` the instant the button is pressed (a
    real DB row, confirmed live and via a real-DB test: Notes (0)->(1) with
    zero typing). The title `Input` was constructed with the literal
    editable `value="Untitled"`, so typing right after opening landed
    after the existing text (live-reproduced: typing "Atlas follow-ups"
    without first clearing the field produced "UntitledAtlas follow-ups").
    Preview toggling, by contrast, was ALREADY CORRECT at HEAD: the
    pre-existing `_library_note_editor_armed` mount-time-Changed-event
    guard (added for a different, earlier reason -- its own docstring
    already names this exact hazard) also covers the toggle's recompose;
    a dedicated pilot test (monkeypatched autosave interval, toggled
    Preview on and back off, polled ~1.2s across many cycles) found no
    save/autosave call and no version change. Recorded as "already
    correct, verified" -- no fix needed for that sub-point, matching how
    Task 1 handled an already-fixed LIB-03 sub-finding.

Decisions:
  LIB-13 rendering: reused the Notes Preview render path VERBATIM --
    `Markdown` widget + `front_matter_parser_factory()` from
    `Utils/markdown_parsing.py` (task-1993), the same two lines
    `library_notes_canvas.py`'s `_compose_editor` already uses for Preview.
    No new markdown dependency, no second pipeline; `Markdown` parses and
    escapes internally exactly like the existing Notes Preview path (the
    RAG arc's "escaping must be the terminal step" lesson is respected by
    reuse, not by adding a new escaper).
  "Markdown-typed media" detection: local ingestion (`local_file_
    ingestion.py`) maps BOTH `.md`/`.markdown` and `.txt`/`.rst`/`.csv`/
    `.log` to the single `media_type="plaintext"`, so `media_type` alone
    cannot distinguish a real markdown file from a plain one. Added a pure
    content sniff (`looks_like_markdown_content` in
    `library_media_viewer_state.py`: an ATX heading, a fenced code block,
    or a GFM table separator row) gated behind a small type allowlist
    (`plaintext`/`markdown`/`obsidian_note`) -- "Rendered" only defaults
    for an item that is BOTH a plausible type AND actually contains
    markdown syntax, so a `.txt`/`.csv` item with `media_type="plaintext"`
    still defaults to Raw (own test coverage).
  Toggle placement/idiom: mirrored `LibraryScreen`'s own "Database
    (selected) | Files" notes-source strip exactly -- a plain `Horizontal`
    of two compact, unstyled `Button`s (label suffixed "(selected)" for
    the active mode -- state-in-text, per the task's own naming of this
    idiom) separated by a `Static("|")`. Scoped to the Content section
    (inside `LibraryMediaViewer`, not the screen header) since it only
    concerns that one section of one item, unlike Database/Files which
    switches the whole canvas. Rendered defaults for markdown items, Raw
    for everything else (unchanged for every pre-existing item).
  Content search: always matches the RAW stored text regardless of the
    visible mode (simplest, and the only text the existing `find_content_
    matches`/highlighting machinery can operate on) -- the search box's
    placeholder becomes "Search content (raw text)…" whenever a toggle is
    offered, stating which text it searches per the task's own
    requirement. Highlighting inside the body only applies in the Raw
    view (the Markdown widget has no Rich-Text-span injection point);
    Prev/Next's `scroll_to` still runs in Rendered mode too (imprecise but
    harmless, matching its own pre-existing "not pixel-perfect" contract).
  LIB-14(a) title fix: empty `value` + `placeholder="Untitled"` (the
    brief's placeholder option), not select-all-on-focus -- side-steps
    Textual's focus/selection timing entirely rather than depending on it,
    and doubles as the SAME flag driving (b) below (see next). An empty
    save now falls back to "Untitled" at the seam (`_save_library_note`)
    so the DB never stores a blank title.
  LIB-14(b) lifecycle: GC-untouched-blank-on-exit, not create-on-first-
    edit -- picked as the smaller diff. Create-on-first-edit would have
    required `_save_library_note`'s early guard
    (`if ... not self._selected_note_id: return`) to grow a whole create
    branch (new id, `_local_source_records` append-not-patch, etc.) for a
    path that today no-ops by design. GC instead reuses
    `_flush_library_note_save` -- already the single choke point every
    editor-exit path (Back, note-row switch, rail-row switch, screen
    navigation, the Delete confirm flow) already awaits before tearing the
    editor down -- adding one `_library_note_pending_blank_gc_id` flag
    (set on a Blank-note create, cleared by any real edit or an explicit
    Save) and one `_gc_pending_blank_note` helper. No new call sites
    needed at any of those seven+ exit points. The one collision handled
    explicitly: pressing the editor's own "Delete" button now clears the
    pending-GC flag BEFORE its own `_flush_library_note_save()` call, so
    GC never races the user's own explicit delete-confirm flow.
  LIB-14(c): no code change -- already correct (see re-verification
    above); the existing `_library_note_editor_armed` guard is the
    mechanism, reused unmodified.

A defect found and fixed along the way (not in the original LIB-13/14
findings): the Rendered|Raw toggle's `Static("|")` separator initially
rendered off-screen entirely in a 170-column terminal -- a bare `Static`
has no width rule of its own (only `height: auto` in its own
`DEFAULT_CSS`), so it silently inherited Textual's base `1fr` default and
consumed the Horizontal's remaining width, pushing the "Raw" button's
region to `x=184` against a 170-column screen. This passed an
existence/label-only `query_one` check while being completely invisible
live -- exactly the P1 arc's documented "headless DOM queries miss
off-screen/oversized widgets" trap named in this plan's Global
Constraints. Found live (screenshot showed "Rendered (selected) |" with
no "Raw" after it), reproduced via a dedicated rendered-geometry pytest
(`region.width`/`region.x` assertions, RED confirmed before the fix), and
fixed with a `LibraryMediaViewer.DEFAULT_CSS` rule pinning the
separator's width to 1 -- mirroring `LibraryScreen`'s own
`#library-notes-source-separator` rule for the strip this one was modeled
on (which already had this exact pin; it was simply missed when copying
the pattern into a new widget's DEFAULT_CSS instead of the screen's).

Tests (TDD, RED evidence captured via a temporary Edit-based revert of the
specific line under test, run, then restored -- never `git checkout`):
Tests/Library/test_library_media_viewer_state.py gained 12 pure-function
tests for `looks_like_markdown_content` and `build_library_media_viewer_
state`'s `media_type`/`is_markdown` fields (heading/table/fenced-code
detection, the plaintext-without-markdown-syntax negative case, the
non-markdown-type-never-flagged case even with `#`-prefixed content, empty/
None content). Tests/UI/test_library_shell.py gained: 7 media-viewer tests
(defaults-to-rendered with a real `MarkdownH1` widget query, raw toggle
round-trip both directions, non-markdown item never shows the toggle,
plaintext-without-syntax defaults raw, search placeholder states "raw
text", content_mode resets on Back and on opening a different item, and
the rendered-geometry regression test for the off-screen-button defect);
3 note-lifecycle tests (the real-DB GC test -- builds a real
`NotesScopeService` over a real temp-file `CharactersRAGDB`, asserts
`count_notes` 0->1 on create and back to 0 after an untouched Back-exit;
the contrast real-DB test asserting an edited-then-Back note survives
with its typed content; the Preview-toggle-never-bumps-version test with
a monkeypatched short autosave interval polled across ~60 cycles); 1
pre-existing test (`test_library_shell_create_blank_note_lands_in_editor`)
updated for the new title-Input contract (empty value + "Untitled"
placeholder, plus asserting `_library_note_pending_blank_gc_id`).

RED evidence (each captured via a scoped Edit-revert, run, then restored):
disabling the Markdown-widget branch failed both media-viewer-defaults and
raw-toggle-round-trip tests; reverting the title Input's `value=""` swap
failed `test_library_shell_create_blank_note_lands_in_editor`; disabling
the `_flush_library_note_save` GC hook failed the real-DB GC test (poll
loop timeout, DB count never reached 0); the off-screen-separator
rendered-geometry test failed before the `DEFAULT_CSS` fix was added.

Verification: Tests/UI/test_library_shell.py full file (362 passed, 1
failed -- `test_landing_footer_advertises_the_landing_keyboard_story`, the
plan's own named ambient debt task-3022; unrelated to notes/media-viewer
code, not touched by this task) and Tests/Library --collect-only -q (1104
collected, 0 errors) both green. Tests/Library/test_library_media_viewer_
state.py (54 passed) and Tests/Notes/test_notes_scope_service_library_
canvas.py (9 passed, unaffected) both green.

Live tmux (socket p2T3lib23968, scratch profile sdd_p2t3, cleaned up
after): imported a real `.md` fixture (heading + GFM table) via the Import
canvas's path input -- opened straight into "Rendered (selected) | Raw"
with the heading rendered as a real bordered heading and the table as a
real bordered table (no literal `#`/`|`); clicked "Raw" and confirmed the
literal `# Setup Guide` / `## Steps` / `| --- | --- |` source; toggled
back to "Rendered (selected)". Created a Blank note, confirmed the title
field showed "Untitled" in placeholder-gray (same color as the rail's
"Search Library…" placeholder, distinct from real typed-text white), typed
"Atlas follow-ups" and confirmed the value was exactly that (no
"UntitledAtlas" prefix) -- Notes (0)->(1) on create. Toggled Preview twice
and confirmed the meta line's version stayed at "v2" both times (v2 from
the one real autosaved edit; no bump from either toggle). Created a SECOND
blank note (Notes (1)->(2)), pressed "Back to list" without typing
anything, and confirmed Notes dropped back to (1) with only the edited
"Atlas follow-ups" note remaining -- the untouched blank row was GC'd.

Docs: Docs/User_Guide/library/notes.md ("New note view" section rewritten
for the placeholder title + GC-on-exit contract) and Docs/User_Guide/
library/media-and-conversations.md ("Content" bullet rewritten for the
Rendered|Raw toggle and the raw-text search placeholder), both with a new
"Verified against dev @ 6b38a13b8" stamp for this change.

Files changed: tldw_chatbook/Library/library_media_viewer_state.py,
tldw_chatbook/Widgets/Library/library_media_viewer.py,
tldw_chatbook/Widgets/Library/library_notes_canvas.py,
tldw_chatbook/UI/Screens/library_screen.py,
Tests/Library/test_library_media_viewer_state.py,
Tests/UI/test_library_shell.py,
Docs/User_Guide/library/notes.md,
Docs/User_Guide/library/media-and-conversations.md.

Task 3 fix round 1 (review) -- AC#5 gap: "type then delete everything"
still committed a stray row:

Review finding: ``_mark_library_note_dirty`` cleared
``_library_note_pending_blank_gc_id`` on the FIRST edit regardless of
final content, and ``_flush_library_note_save`` branched on
``_library_note_dirty`` (not final emptiness) -- so typing into a blank
note then deleting everything back to empty exited through the SAVE
branch, persisting `title="Untitled"` + empty content: exactly the row
AC#5 forbids. No test covered the sequence.

Fix (matches the directed approach): added a second, edit-surviving flag
``_library_note_session_blank_id`` (set only by the "Blank note" create
path; NOT cleared by ``_mark_library_note_dirty``, unlike the older
``_library_note_pending_blank_gc_id`` which still exists but now drives
ONLY the title-placeholder UI). ``_flush_library_note_save`` now decides
GC-vs-save by reading the LIVE editor fields (title/body/keywords, never
the stale detail) and checking "all three blank" -- gated on session-blank
id equality, not on the dirty flag -- which subsumes both "never touched"
and "typed then emptied" with one check. Scope guard is structural: the
flag is only ever set by the Blank-note path, so a pre-existing note that
gets emptied out always falls through to the normal save branch (own
test, real DB: row survives with title falling back to "Untitled" and
content=""). Keywords count toward "empty" (documented decision: a
user-provided tag on an otherwise-blank note is still content worth
keeping). Autosave-interaction decision (explicitly asked for): an
intermediate autosave does NOT clear the session-blank flag -- only an
explicit Save press does (mirroring the older flag's existing exemption)
-- so a session blank that got autosaved with real text and was then
emptied out again before exit still GCs, since AC#5 cares about the FINAL
state, not what happened mid-session (own test, real DB, with
`LIBRARY_NOTES_AUTOSAVE_SECONDS` monkeypatched short and a poll loop that
waits for the autosave to actually land before emptying).

Two real bugs surfaced by writing the directed tests (not by inspection --
both were caught by a failing assertion, fixed, re-run to green):
 1. GC'ing the row left ``_library_note_dirty`` at its pre-GC ``True``
    value, so ``_exit_library_note_editor_guarded``'s own dirty-veto
    (`if self._library_note_dirty: return False`) blocked the exit right
    after a SUCCESSFUL GC -- the DB row was gone but the editor stayed
    open on "editor" instead of returning to "list". Fixed by clearing
    `_library_note_dirty` unconditionally inside `_gc_pending_blank_note`
    (GC must never block the exit it runs inside, matching the function's
    own existing best-effort philosophy).
 2. `_gc_pending_blank_note` hardcoded `version=1` for the delete call --
    correct for "never touched," wrong for "autosaved then emptied": a
    mid-session autosave (which the design above deliberately still
    allows before GC) bumps the row's real DB version, so the delete
    silently failed on a version mismatch (`NotesScopeService.delete_note`
    normalizes a stale-version `ConflictError` to a quiet `False`) and the
    row survived. Fixed by sending `self._library_note_version` (the
    screen's own up-to-date tracking, bumped by every successful save)
    instead of the hardcoded literal.
Both were found by the `_typed_then_deleted_all_is_gc_from_real_db` and
`_autosaved_then_emptied_still_gcs_on_back` tests respectively, RED before
each fix, GREEN after -- exactly the kind of thing a shallower
(assertion-only, no real-DB-round-trip) test would have missed, and
exactly why the review asked for the real-sequence tests rather than
accepting the design on inspection alone.

Minor finding addressed: `_gc_pending_blank_note()` and
`_exit_library_note_editor_guarded()` both call
`_refresh_local_source_snapshot()` on the same Back/Escape exit path --
documented (both docstrings now say so explicitly) that this is
intentional and harmless: both target the same
`@work(exclusive=True, group="library_source_snapshot")` worker, so the
second call supersedes (cancels-and-restarts) the first rather than
running both; no skip-if-queued logic was added since the existing
`exclusive=True` already provides it for free.

Also added explicit clearing of both flags at two more "switching to a
different note" sites that had an inline reset block rather than routing
through the shared `_reset_library_note_editor_state()` helper
(`handle_library_notes_row`'s row-press switch,
`_apply_navigation_context_state`'s note_id deep-link branch) -- belt and
suspenders: the id-equality guard in `_flush_library_note_save` already
prevents any wrong behavior even without this (a stale flag value can
never coincidentally equal a DIFFERENT future note's real id), but
clearing eagerly avoids a harmless-but-untidy stale string sitting in
memory between visits.

New tests (TDD, real DB via the same `_real_notes_scope_service` fixture
the earlier round used): `test_library_shell_blank_note_typed_then_
deleted_all_is_gc_from_real_db` (the exact reviewer-directed sequence:
blank -> type -> delete all -> Back -> real-DB count asserted back to 0,
never sticking at 1), `test_library_shell_pre_existing_note_emptied_out_
still_saves_in_real_db` (the scope-guard counter-case: a pre-seeded real
note emptied out still saves, exactly 1 row survives with the emptied
content), `test_library_shell_blank_note_autosaved_then_emptied_still_
gcs_on_back` (the autosave-interaction case, and the test that caught bug
#2 above).

Verification: Tests/UI/test_library_shell.py full file -- 365 passed, 1
failed (same pre-existing `test_landing_footer_advertises_the_landing_
keyboard_story`, task-3022's named ambient debt, unrelated). Targeted
note/blank-note sweep (95 tests) green.

Files changed (fix round 1): tldw_chatbook/UI/Screens/library_screen.py,
Tests/UI/test_library_shell.py.
<!-- SECTION:NOTES:END -->
