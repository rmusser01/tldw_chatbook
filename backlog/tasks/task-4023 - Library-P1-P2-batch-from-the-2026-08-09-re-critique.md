---
id: TASK-4023
title: Library P1/P2 batch from the 2026-08-09 re-critique
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 20:30'
updated_date: '2026-08-11 00:21'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09, snapshot
`.impeccable/critique/2026-08-09T20-15-07Z__tldw-chatbook-ui-screens-library-screen-py.md`
(22/40; trend 23 → 21 → 27 → 22 → 22). The P0 (Escape crash) shipped in PR #1464; the three
highest-value findings are tasks 4020/4021/4022. This is the remainder — grouped for one pass,
split if any item grows.

**Accessibility / honesty (highest value in this batch)**
1. RC-07 — disabled state is colour-only, measured: Select-mode bulk buttons **1.08:1** when 0
   selected (the very buttons task-2853 added — present, focusable, meaningful, unreadable);
   Media `Select` when empty 1.45:1 and silent on click; Export ~1.4–1.51:1; Collections' three
   buttons **2.30:1 even when enabled**. Floor disabled contrast at 3:1, add a non-colour marker,
   and attach the reason to the control. The product's non-colour vocabulary already exists
   (`☐/☑`, `▸`, `┃…┃`, `(selected)`, `✓/○`) — disabled state simply never joined it.
2. RC-06 — the Notes canvas copy says "switch to Files" but the `Database | Files` strip
   (`library_screen.py:7333`) does not render on first paint: the fast rail-click path
   (`_replace_library_browse_canvas`) swaps only the inner canvas and never composes the strip;
   only a full recompose renders it. Named `task-3317` in an unmerged sibling branch's test
   docstrings — reconcile rather than duplicate.
3. RC-09 — DB sizes in the Details disclosure are computed once and never refreshed: UI showed
   `Prompts 148.0KB / Media 476.0KB` while disk incl. sidecars was 180.0KB/508.0KB; a recompose
   with no disk change corrected both. task-2859's WAL-inclusive helper is correct and stale.
4. RC-10 — F1 lists Escape 2–3× with contradictory labels (`- esc: focus rail` / `- escape: Back`
   / `- escape: Focus rail`), omits F6 on Search/RAG though the footer advertises it, says nothing
   about Collections on the Collections panel, and does not close on a second F1.

**Interaction grammar (the score's current ceiling)**
5. Four footer dialects across seven canvases (different separators, different key names:
   `F6 panes` vs `F6 next pane`, `/ Find` vs `/ focus search`); the hub's `i`/`n` shortcuts vanish
   elsewhere with no statement of whether they still work.
6. Three active-state markers (`▸` prefix, `┃…┃` bars, `(selected)` text) and three toolbar
   layouts (Media vertical, Notes 3×2 grid, Prompts/Skills single row).
7. `▸` carries two incompatible meanings: disclosure (`Details ▸`) vs silent cycler
   (`type: All ▸`, `mode: Search ▸`) that advances with no menu — the option set is undiscoverable.
8. Escape still inert on Export, Collections, and the Study staging canvas; the staging canvas has
   no back path at all; `Export…` from within Media navigates away with no return.

**Search**
9. RC-08 — results land ~30 rows below the fold behind the configuration panel; clicking `Run`
   leaves the visible half of the canvas pixel-identical. Enter in the rail search navigates and
   pre-fills but does not run. Two search inputs are live with different values and navigation
   silently overwrites one with the other; never-executed strings still enter `Recent searches`.

**Layout**
10. Media list renders in a ~30-char column on a 170-col terminal with the detail below it, and
    truncates titles at 17 chars while ~115 columns sit blank; the media viewer gives a 33-line
    document a 7-row viewport while spending 2 lines on a `file://` temp path.
11. At ≤100 cols the landing canvas vanishes entirely while the rail still reads "pick a section
    on the left".

**Copy**
12. "opens staging canvas" printed three times in the primary nav (a truthfulness fix that traded
    a lie for internal jargon); Collections stacks four "nothing here" sentences and still offers
    no "Add to collection" anywhere; export scope "Everything" excludes Prompts/Skills/Collections;
    `Type: plaintext` for a `.md` the viewer renders as markdown, with extensions stripped from
    every list title.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabled controls meet a 3:1 floor, carry a non-colour marker, and state their reason at the control
- [x] #2 The Files strip renders on first paint; the Notes copy and the rendered controls agree
- [x] #3 Details DB sizes refresh rather than reporting a stale first reading
- [x] #4 F1 lists each binding once with one label per key, includes the keys its own footer advertises, and closes on a second press
- [x] #5 One footer grammar, one active-state marker vocabulary, and one meaning per glyph across the Library's canvases
- [x] #6 Search results are visible at the point of action, Enter runs the search, and one query model backs both inputs
- [x] #7 Each remaining copy/layout item is fixed or declined with a one-line reason in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 2 of the library-honesty-accessibility batch owns AC#1-#4 ONLY (AC#5-#7 stay for Task 3).
1. Re-verify all four findings at HEAD (dev has moved): AC1 ratios via live tmux + ANSI decode with computed WCAG ratios; AC2 against dev d1df7d0a7 (TASK-13213), which gates the fast canvas swap on matching chrome — reconcile with task-3317 (AC#1 there already closed dev-side); AC3 against DBStatusManager's 120s cache vs the rail's render-once Static; AC4 against task-3312's canonical-key dedupe (landed AFTER the critique snapshot).
2. AC1 (RC-07): extend the Legible Disabled recipe (TASK-1801/DESIGN.md, app-tier CSS — source tcss, build_css.py, check_bundle_sync.py) to the Library action-button surfaces: select-mode bulk buttons, Media/Notes/Conversations Select toggle, Export canvas actions, Collections form actions (incl. their ENABLED 2.30:1). Non-colour marker from the existing vocabulary + reason attached at the control (F-018 tooltip idiom + visible reason where a static already anchors the toolbar). Before/after ratio tables from live ANSI decode.
3. AC2 (RC-06): expected dissolved at HEAD via TASK-13213; prove the strip on FIRST paint of the fast rail-press path live, keep/extend the pinned test, and record the task-3317 reconciliation (adopt/adapt/supersede) in the notes.
4. AC3 (RC-09): refresh the DB-sizes reading at a sensible trigger — Details disclosure open and Library canvas entry — patching the rail Static in place (recompose discipline: updater owns the same conditional compose owns). No new polling.
5. AC4 (RC-10): add F6 to LIBRARY_SHORTCUTS (Search/RAG parity with every sibling set); intra-extras key dedupe in action_show_workbench_help (first gate-True binding wins, matching Textual resolution order); f1->dismiss Binding on WorkbenchHelpPanel so a second F1 closes instead of stacking a panel-about-the-panel. Fix at the shared _library_footer_shortcuts_for_current_state seam, never fork it.
6. TDD throughout (RED reproducing the observed effect), targeted pytest + collect-only sweep, live tmux verification per constraint (socket haT2lib$RANDOM, scratch /tmp/haT2, users_name sdd_hat2), User Guide stamp updates, single commit with specific paths.

Task 3 of the batch owns AC#5-#7 + close-out. Plan (after live re-verification at HEAD, socket haT3lib5390, scratch /tmp/haT3, sdd_hat3):
1. Inherited Task-2 review Medium: patcher-level pins for the in-place disabled-marker patchers (`_apply_library_row_toggle` across the 0-vs-1 boundary; collections patcher label rebuild) — mutations C/D must go RED.
2. AC#5 grammar convergence, one vocabulary per concept: (a) footers — rewrite the five LIBRARY_NOTES_* run-on sets + six inline notes-workflow sets into the standard `(key, action)` grammar (lowercase keys, verb-phrase labels, F6 included) so one dialect remains; (b) markers — leading `▸ ` = selected list row (extended to Collections rows), `(selected)` = active inline-strip option, `┃…┃` = focus (unchanged), section headers keep trailing `▸/▾` disclosure; (c) cyclers move OFF `▸` to a dedicated ` ⇄` suffix with an option-enumerating tooltip (9 sites); (d) Media toolbar converges from a vertical button stack to the shared ds-toolbar row grammar. Written convention recorded in Implementation Notes.
3. AC#6: Run/results — scroll the Evidence region into view at run start and on results arrival (results land below the fold behind the config region); one query truth at the WIDGET level — typing in either search input patches the other mounted input in place (state was already single-source; the sibling widget went visibly stale, proven live). Enter-runs and executed-only Recents re-verified DISSOLVED at HEAD (both proven live) — pinned, not re-fixed.
4. AC#7 bounded fixes: canvas row-title budget decoupled from the rail's 20-cell cap; landing copy loses "on the left" (canvas hidden at compact widths); "opens staging canvas" ×3 → user-language meta; Collections empty state 4 sentences → 2; export scope "Everything" → "All media, conversations, and notes"; viewer "Type: plaintext" → names the rendered markdown honestly; Escape wired on Export (back to origin canvas or hub — also fixes Export…-from-Media having no return), Collections (focus rail), and the Study staging canvas (back to hub), each advertised via the shared footer seam. Declines with reasons in the notes: file:// URL line, 7-row viewport (not reproduced at HEAD), extension-stripped titles.
5. Close-out: verify all seven ACs, consolidate notes (hand-edit), file follow-up tasks for the deferred remainder (media list/detail side-by-side layout; one reversibility story; cycler option-set menus), flip Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cross-task observation (2026-08-09, task-4022 review round 2): Library now ships three different reversibility stories on one screen -- notes are silently GC'd with no undo, bulk media delete gets an in-place Undo receipt, and single media delete gets nothing at all. This is the same divergence the re-critique scored heuristic #4 (Consistency and Standards) a 1 for; worth reconciling as part of AC#5 (one active-state/interaction grammar) rather than as a separate item.

---

**Task 2 of the library-honesty-accessibility batch: AC#1–#4 ONLY. AC#5–#7 are deliberately untouched — Task 3 owns them and the task must NOT be flipped Done until it lands.**

Re-verified all four findings at HEAD before implementing (dev had moved since the 2026-08-09T20:15Z critique snapshot):

- **AC#1 (RC-07) — alive, fixed.** Measured live via ANSI decode BEFORE: select-mode bulk buttons 1.39:1 (#333333/#191919), empty-list Select 1.45:1, Export submit 1.44:1, Collections' three form buttons 2.30:1 disabled. (The critique's "2.30:1 even when enabled" did NOT reproduce at HEAD — enabled Create measured 10.63:1.) AFTER: bulk/Select/Export 7.25:1, Collections 5.91:1, all with a leading "○" disabled marker (extending the existing ✓/○ vocabulary, never a new glyph) and F-018 reason tooltips (the empty-list Select toggle, previously reason-less, gets "Nothing here to select yet."). CSS at the app tier (source tcss → build_css.py → check_bundle_sync.py): `Button.library-canvas-action:disabled` / `Button.library-source-action:disabled` Legible-Disabled escapes, a fixed `#library-export-submit:disabled` (its `$ds-text-disabled` colour WAS the 1.44:1), and a `(1,1,1)` `#library-collection-actions Button:disabled` variant because that container's ID-scoped Button rule outranks the class escape — found by re-measuring live after the first build, not by reading specificity. Marker labels are rebuilt by every in-place patcher that flips `disabled` (`_apply_library_row_toggle`, the two export-submit gate patchers via the new shared `apply_library_export_submit_gate`, `_refresh_collections_panel_action_state_widgets`) — the recompose-discipline rule. Scoped to Library surfaces only; no app-wide restyle.
- **AC#2 (RC-06) — DISSOLVED at HEAD; task-3317 reconciliation: ADOPT.** Dev commit d1df7d0a7 (TASK-13213, ancestor of this branch) already gates `_replace_library_browse_canvas` on chrome parity (`notes_source_strip_mounted != (canvas_kind == "notes")` → full recompose), exactly the direction task-3317 AC#1 asked for; the sibling branch's test docstrings and task-3317's own file record that closure (bae0f2fc1). Live-proven at HEAD twice this task: the `Database (selected) | Files` strip renders on the FIRST paint (0.4s capture) of the fast rail-press path, and the copy/controls agree. Regression coverage already exists (`test_library_notes_source_choices_render_and_switch_by_keyboard`, re-run green). No product change; nothing to duplicate. task-3317 AC#2/#3 (LIB-19 compact cost, test-pin unification) remain open THERE and are out of this task's scope.
- **AC#3 (RC-09) — alive, fixed.** The DBStatusManager cache refreshes on a 120s app timer but the rail Static rendered once: live at HEAD the disclosure showed Prompts 180.0KB against 4,879.9KB on disk (incl. sidecars) and close/reopen did not correct it. Fix: opening the Details disclosure triggers `_refresh_library_details_db_sizes` (exclusive worker) — recompute via the manager's WAL-inclusive path, then patch `#library-details-db-sizes` in place, MOUNTING it when the compose-time cache was empty (the updater owns the compose branch's conditional; formatting extracted to `_library_db_sizes_line`, one source for both paths). No polling added. Live AFTER: 1.1MB shown → grew the DB to 12.32MB on disk → close/reopen → 12.3MB shown.
- **AC#4 (RC-10) — partially dissolved, remainder fixed.** The "Escape listed 2–3×" half was already closed by task-3312's canonical-key dedupe (f73a5f65b, landed ~3h AFTER the critique snapshot; verified live: Notes-list F1 shows exactly one "esc: focus rail"). Still alive and now fixed: (a) Search/RAG omitted F6 — `LIBRARY_SHORTCUTS` gains ("F6", "next pane"), and per the task-2860 merge rule the footer now renders "F6 next pane" verbatim with the generic "F6 panes" dropped from the globals (pins updated); (b) second F1 did nothing (the app-level delegate finds no handler on the panel screen) — `WorkbenchHelpPanel` now binds f1→dismiss, making F1 a true toggle for EVERY screen using the shared panel (deliberate cross-screen improvement, behavior-only); (c) same-key BINDINGS extras had no intra-set dedupe — the seen-set now grows while extras accumulate, first active entry wins (Textual's own resolution order); (d) "Collections' panel says nothing about Collections" — the panel title now names its surface ("Library Shortcuts — Collections"), derived from the selected rail row (surface identity), while the shortcut SET still has exactly one source (`_library_footer_shortcuts_for_current_state`, unforked).

Tests: 13 new (Tests/UI/test_library_honesty_accessibility.py + a second-F1 test in test_workbench_focus_help.py) written RED against the observed effects, plus 3 updated pins in test_screen_footer_hints.py. Touched-suite gate: 115 passed; test_library_screen/ingest_keyboard/skills 150 passed; nav-suite library subset 36 passed with 2 failures A/B-proven ambient at base 117620b76 (`_library_note_dirty` is a read-only property; the dirty-guard tests assign to it — pre-existing, filed nowhere by this task). Live verification in an isolated profile (sdd_hat2, scratch /tmp/haT2) with computed WCAG ratios from `capture-pane -e` decode. Docs: library.md (+ Details refresh sentence + stamp), media-and-conversations.md, import-and-export.md, collections.md. Full evidence: .superpowers/sdd/library-honesty-accessibility/task-2-report.md.

---

**Task 3 of the batch: AC#5–#7 + close-out** (commits fa3d9295f, 97700fa77, 8c0dc5af4, 57ad075de; live verification in isolated profile sdd_hat3, scratch /tmp/haT3, tmux socket haT3lib5390, 170x50 / 170x24 / 100x50 / 60x20).

**The Library interaction-grammar convention (AC#5), for the next screen to conform to:**

- *Footer hints*: every set is a tuple of `(key, action)` pairs; keys lowercase (`ctrl+s`, `esc`, `enter`, `/`, `pgup/pgdn`) except F-keys (`F6`); actions are short verb phrases naming what the key does here, reusing the shared vocabulary (`focus rail`, `back to notes`, `next pane`); the footer joins pairs with `" | "` and appends the global cluster. A locked state advertises nothing (globals only), never a dead key. Sets may carry a `_COMPACT` label tier (same keys, same destinations, shorter labels) for ≤100-col presentation — mirroring the footer's own FULL/COMPACT global tiers, not a new dialect. The global F6 label is "F6 next pane" everywhere.
- *Markers, one meaning per glyph*: leading `▸ ` = the selected row of a list (rail rows, media/conversations rows, Collections rows); `(selected)` suffix = the active option of an inline choice strip (`Database (selected) | Files`, `Rendered (selected) | Raw`); `┃…┃` = keyboard focus (CSS, unchanged); trailing `▸`/`▾` on a section HEADER = disclosure state (collapsed/open). Trailing ` ⇄` = a value-cycle control (press advances; the tooltip enumerates the cycle) — built only via `library_cycle_label`/`library_cycle_tooltip` (library_shell_state.py). Deliberate residue: `▸` still appears as the breadcrumb separator in prose ("Library ▸ Study") — path notation, positionally distinct from all of the above.
- *Toolbars*: canvas actions live in horizontal `ds-toolbar` rows beneath the canvas title (Media converged from a vertical stack); a second row only for a distinct action family (Notes' transfer row).

**AC#5 re-verified alive at HEAD, then fixed.** Live before-captures showed: notes list/editor footers in the run-on dialect (`Ctrl+N New · / Find · Esc Library` / `Ctrl+S Save · Esc Notes`) beside the standard dialect on Media, with the globals rendering `F6 panes` beside Library's own `F6 next pane`; `▸` carrying selected-row + disclosure + cycler meanings in one screenful; Media's vertical action stack vs Notes' 2×3 grid vs Prompts/Skills' single row. Fixed as above (9 cycler sites incl. the prompts-collection in-place patcher; Collections rows gained the `▸ ` marker; Media toolbar converged). The notes-workflow footer runs through one tier helper; running create/sync/conflict states advertise nothing instead of "Esc Locked". Live after-captures across Media/Notes/editor/Prompts/Search/Export at 170 cols and the compact editor at 60x20 (the compact tier genuinely DISPLAYS — the full labels would be dropped whole by the responsive ladder, which is why the run-ons existed; the tier system replaces that trade with the footer's own established compress-don't-fork idiom).

**AC#6 re-verified: two of four sub-findings had DISSOLVED at HEAD** (standing rule): Enter in the rail search RUNS the query (proven live: "1 result for 'terminals'"; behavior dates to July, pinned by test_library_shell's rail-submit tests) and Recents records only executed queries (recording sits after the run gate; proven live — an edited-but-never-run string stayed out). Closed on evidence, plus a new blocked-run-records-nothing pin. Still alive, fixed: (a) results below the fold — Run start and results arrival now scroll the Evidence region into view; root-caused the counter-scroll while stabilizing the pin: Textual's `Collapsible._watch_collapsed` schedules an ANIMATED `scroll_visible` on every programmatic collapse, so the results-arrival force-collapse of Recents scrolled the panel to the BOTTOM past the evidence — the force-collapse now bypasses the watcher (user clicks keep the normal path); (b) one query truth at the WIDGET level — state was already single-source, but the sibling mounted input went visibly stale (live: canvas `terminals render` beside rail `terminals`); typing in either now patches the other in place, loop-guarded by the existing value-equality no-op. Both fixes mutation-proven RED.

**AC#7 dispositions (fixed unless noted):** canvas row titles no longer inherit the rail's 20-cell cap (17-char cut with ~115 blank columns → full titles; CSS ellipsis owns overflow); landing copy drops "on the left" (at ≤100 cols the shell is one-pane and the canvas holding the copy is hidden); "opens staging canvas" ×3 → "see what carries over"; Collections empty state 4 sentences → 2; export scope "Everything: …" → "All media, conversations & notes: …"; viewer `Type: plaintext` → `Type: markdown (stored as plaintext)` when rendered as markdown; Escape wired on Export (back to the canvas whose Export… opened it — closing the no-return-path finding — or the hub from the rail), Collections (focus rail, the shared list-canvas contract), and the Study staging canvases (back to hub), each advertised via the unforked footer seam. **Declined, with reasons:** the viewer's `file://` URL line stays (it is the item's only provenance; one line at working widths; hiding real metadata to save a line trades honesty for tidiness); the "7-row viewport" did NOT reproduce at HEAD (the content region shows up to 18 rows at 170x50 — measured live with a 33-line document showing 16+ rows; partially dissolved, the `max-height: 18` region is a deliberate bound against nested-scroll friction); extensions stripped from list titles stays (titles are derived at import by design; retro-renaming stored titles would rewrite user data, and the viewer's Type line now names the format honestly). **Deferred to filed follow-ups:** task-14900 (Media list/detail side-by-side at wide widths — the structural half of the 170-col finding), task-14901 (one reversibility story across Library destructive actions — the cross-task observation above), task-14902 (cycle controls get real option menus; the ⇄ + tooltip is the bounded mitigation), task-14903 (HIGH: Textual text-selection MouseDown crash during relayout killed the app once during live verification — full traceback in the task).

**Inherited Task-2 review M-1 closed:** patcher-level pins now drive `_apply_library_row_toggle` across the 0↔1 boundary and the collections patcher through the real Input.Changed path, asserting marker labels both directions; both verified RED under the review's exact surviving mutations (C and D), then restored.

Tests: 13 new in Tests/UI/test_library_honesty_accessibility.py (2 patcher pins, 5 AC#5, 3 AC#6, 3 AC#7) — file total 26 passed; pins updated across test_screen_footer_hints, test_library_shell, test_library_prompt_collections, test_library_skills_canvas, test_chrome_ux_fixes, test_app_footer_shortcut_context, phase1/phase39/gate16, test_library_content_hub, test_destination_shells, visual-parity, phase6, Tests/Library (export scope/state, collections state, shell state), Tests/Widgets/Library. Targeted gates all green (~500 tests across the touched suites). Known-ambient, A/B-proven at base and untouched: gate16 `test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional` (ordering pin), `test_shadow_name_set_stays_in_sync_with_real_sources` (dev's stream-video/generate-video console commands), and the two `test_screen_navigation` dirty-guard property-assignment failures Task 2 already documented. Docs: library.md + media-and-conversations/import-and-export/collections/search-and-rag/notes stamps and copy updates. Full evidence: .superpowers/sdd/library-honesty-accessibility/task-3-report.md.
<!-- SECTION:NOTES:END -->
