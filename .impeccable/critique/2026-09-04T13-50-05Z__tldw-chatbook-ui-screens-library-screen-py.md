---
target: Library ▸ Media (post fix wave 3)
total_score: 25
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 4
timestamp: 2026-09-04T13-50-05Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated; parent traced A's P0 and P1 mechanics in source before scoring)

Target: Library ▸ Browse ▸ Media (Operate mode) at dev tip 8b6e7501d6, after fix wave 3 (#2366, #2367, #2369). Live at 235×52 and 100×30 against six seeded CRIT4 items (removed after the run).

## Design Health Score — 25/40 (Acceptable, upper edge)

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Banner, footer progress, "Loaded ·" prefix and receipts are exemplary; but after Escape closes Find the footer keeps saying `esc close find` (A cap 08/23, B cap_21), and both undo receipts clip their Undo button to `Un`/`Und` in the Items pane (B cap_83, cap_99) |
| 2 | Match System / Real World | 3 | "Page boundary is unknown.", "workspace workspace-local-1", "Local Media item" above every title, More ▸ "Open manager" that routes back to this screen (library_screen.py:41720-41740) |
| 3 | User Control and Freedom | 2 | Walk keys typed into a search field (P0); three Escapes to leave the Reader with the second landing inside the rail's Search input (A cap 20); Escape does not close the More menu (B cap_106); "‹ Back" changes the key map without changing the screen |
| 4 | Consistency and Standards | 2 | Read collapses Find until asked, Analysis mounts it permanently (viewer.py:626-631); receipts never got the multi-row grammar the toolbars got; eight distinct `esc …` labels; `]`/`[`/`m`/`R` advertised, `l`/`c`/`t` hidden (`show=False`, :1016-1018); doc says "Open in Library ▸ Media" and "no permanent delete", UI ships "Open manager" and "Delete permanently" |
| 5 | Error Prevention | 3 | Destructive choreography is textbook (armed confirm naming both recovery paths, receipt, durable Trash); but `t` arms delete from the Reader unadvertised (B cap_97), Space with the footer's blessing collapsed the Library pane (A cap 31→32), picker Dismiss sits one cell from the open button |
| 6 | Recognition Rather Than Recall | 2 | Rows show `type · age` and nothing else: no analysis marker, no reviewed marker mid-walk (B cap_02); "Analyze after import" off by default inside a collapsed "▶ Import behavior" group with no state on its header (ingest_capabilities.py:1013-1020); keyword `day2` on four rows filters to "No media matched" (B cap_15) |
| 7 | Flexibility and Efficiency | 2 | The `]` walk is dead in Analysis mode (P0); no bulk Analyze; no key for "Review these" or "Sets"; Delete key unbound; F6's content stop is invisible (B cap_57) |
| 8 | Aesthetic and Minimalist Design | 2 | Reader now fills its pane (held), but eight rows of chrome precede content (A cap 03), the Find bar jumps above the header on Enter (B cap_20), a `┐─────` join artifact appears after Find closes (14 captures), Trash shows ~36 blank rows above a pinned Restore (B cap_102), prose runs ~150 cells against DESIGN.md's 65–75 |
| 9 | Error Recovery | 3 | Undo now covers delete, bulk delete and Dismiss; Retry, honest empty states; minus Generate learns the provider is unready only after the click (library_screen.py:41613-41618), and the Undo control is unreadable where it renders |
| 10 | Help and Documentation | 3 | Footer teaching and F1 are strong; the guide drifts from the UI in two places; the completion gesture (final `]` on the last item) is not labelled anywhere on screen (B cap_50) |
| **Total** | | **25/40** | **Acceptable (upper edge) — down from 28; a fix-wave regression of my own owns the drop** |

## Design Specificity Verdict

**LLM assessment (A, unanchored):** authored in its vocabulary, generic in its layout. The state language is unmistakably Chatbook's: the footer that relabels the same key by context (`] next item` → `] next in set … 3 of 6 · 2 reviewed`), `○` disabled markers with reasons, "Loaded ·" row prefixes, the receipt grammar naming Trash, the banner "Reviewing: All media — 1 of 6 · 0 reviewed · not yet reviewed". The shell those words sit in (rail | filter + choosers + list | identity line / Back / title / toolbar / tabs / section header / bordered box) is interchangeable list-detail scaffolding, and the one product-specific interaction, the `]` walk over a pinned set, is exactly where the seams now show.

**Deterministic scan (B):** `detect.mjs --json` over the five media files returned `[]`, exit 0. No-signal: the engine scans web extensions only, `.py`/`.tcss` are skipped silently, and even an HTML probe ran degraded without parser modules. Mechanical greps: 0 hex colors in the five files and in the library-media TCSS section; buttons vs tooltips are canvas 24/5, viewer 25/0, content 2/0, picker 4/1; every Unicode marker sits next to words (none glyph-only); the one message sent at two severities is documented as deliberate ambient-vs-gesture; `#library-media-viewer-content` is `height: 1fr` with no `max-height` (the 75vh cap is gone).

**Visual overlays:** not applicable, terminal UI.

**Where A and B agree:** all six wave-3 fixes held under an independent live probe. Review-selected exits select mode and opens the Reader with `1 of 2` (B cap_73); auto-resume lands on the cursor item and the first `]` advances (cap_77/78); the sort chooser shows all four options with ✓ and works by keyboard (cap_03-08); Dismiss leaves an undo receipt (cap_83); the Reader box reaches row 49 of 52 with no Find bar on a fresh open and no dead pager (cap_17); the "Paused '2 selected items' at 1 of 2 · 0 reviewed. Resume from Sets." notice fires and picker rows carry dates (cap_74, cap_79). B's `]]]]]` in the search field (cap_32b) and A's `▊ ]` (cap 22) are the same defect seen from two sides.

## Overall Impression

The instrument-panel discipline is intact and the review-set model is right. But the wave-3 Find fix (#2367, mine) shipped with a focus hook that fires on every item load, and the Analysis tab (task-28026, mine) mounts that bar unconditionally. So the product's core sequential-review gesture now silently types into a text box. That regression, plus undo receipts that clip their own Undo button, is the whole story of the three-point drop. Fix those two and this surface is back above 28 with the rest as polish.

## What's Working

1. **The footer is a status contract, not decoration.** It drops `[` on the first item and `]` on the last, swaps to `enter choose sort | esc cancel` while a chooser is open, and carries set progress as a chip (`_review_footer_entries`, `_library_footer_shortcuts_for_current_state`). Verified in nine distinct states.
2. **One undo grammar for every destructive act.** Single delete, bulk delete and set Dismiss all leave "✓ … · Undo / Dismiss" in place, name the durable path, and never open a modal (canvas.py:857-940; viewer.py:238-256).
3. **The review-set model survives its chrome.** Pinned snapshot with tombstones, progress over live items only, forward-marks / back-never-marks, one-active with a displacement notice, resume on entry (review_set_state.py:198-280; library_screen.py:38804-38877). B's walk matched the model exactly.

## Priority Issues

1. **[P0] Walking with any search bar mounted types your keys into it.** Two faces of one mechanism. In Analysis mode the search bar is composed whenever an analysis exists (viewer.py:626-631) and its `on_mount` focuses the input whenever the query is empty (content.py:351-362), which is every fresh item load: `]` loads the next item, focus jumps into the box, the next `]` is text (A cap 22/24/36). In Read mode the same happens whenever Find is open across a walk: B's field read `]]]]]` (cap_32b) and, inside a review, `]`, `m`, `m`, `[` were swallowed with the banner frozen at `2 of 6 · 1 reviewed` (cap_41-44). The Find button does not toggle the bar off (cap_38); Escape from inside the input only blurs on the first press. **Why:** workflow 3, "review every analysis of a set", is precisely Analysis mode plus `]`; it now costs Escape + `]` per item and fails silently. **Origin:** my #2367 focus-on-mount hook meeting my task-28026 always-rendered Analysis bar. **Fix:** pass an explicit focus flag only from the Find gesture (`handle_library_media_reader_find`, library_screen.py:41064-41083) instead of inferring it from an empty query; gate the Analysis bar behind `find_open` exactly as Read does (viewer.py:311-322); on a `]`/`[` walk keep the query but never re-take focus; pin with a test that walks two items in Analysis mode and asserts `focused` is not the Input. **Command:** /impeccable harden.

2. **[P1] Undo receipts clip their Undo button in the Items pane.** "✓ dismissed · 2 selected items  Un" and "✓ deleted · 1 item · in Trash  Und" (B cap_83, cap_99). Undo is unfindable by label, the trailing Dismiss button is off-pane, and B could only recover by a raw click on the `U` cell. Receipts are single-line content-width rows in the ~38-col pane; the toolbars were converted to multi-row grammar in #2350 but the receipts were not. **Why:** the receipt is the product's signature safety net; an unreadable Undo is a hidden recovery state, the product's own anti-reference. **Fix:** two-row receipt (message line, then `Undo   Dismiss` on its own row) with `width: 100%` and the toolbar min-width lift; shorten the message; pin with a painted-text test at 38 cols. **Command:** /impeccable adapt.

3. **[P1] The footer lies at four seams.** (a) After Escape closes Find the footer still says `esc close find` because `_library_media_escape_label` (39761-39797) reads the DOM before the recompose lands. (b) Right after `s` the footer promises `space toggle selection` while Space is a no-op unless a row is focused (B cap_69; gate at 27812-27827) and, with focus on the pane grip, collapses the Library pane instead (A cap 31→32), since `_toggle_library_media_select_mode` never moves focus to a row. (c) At `6 of 6` the footer still reads `] next in set` although the next `]` is the completion gesture (B cap_50; 3685). (d) `l`/`c`/`t` are real Reader keys (:1016-1018, `show=False`) and `t` arms delete, yet nothing advertises them. **Fix:** recompute footer shortcuts after `_close_library_media_find` settles; focus the first row on `s` and compute the Space chip from `self.focused`; label the last-item key `] finish review`; add `l`/`c`/`t` (at least `t`) to the Reader footer set. **Command:** /impeccable audit.

4. **[P1] Leaving and moving focus are three-key rituals that end in text fields.** "‹ Back" sets `_library_media_view = "list"` (38493-38522) but the Reader keeps showing the document (A cap 25/31/39), `]`/`[` go dead, and the footer switches to `s select` on identical pixels. Escape from the Reader goes focus Items → focus Library, and the second Escape lands inside `#library-search-input` (A cap 20); B's `s` at the F6 rail stop typed into "Search Library…" (cap_79). F6 cycles reader → rail search → filter and B could not see it land on the content box (cap_57) although the code lists content first (`_MEDIA_WORKBENCH_FOCUS_TARGETS`): the stop is real but invisible after the 31221 outline suppression. Escape does not close the More menu (B cap_106) though the label promises `close more`. Eight distinct `esc …` labels seen live. **Fix:** Escape from the Reader focuses the loaded list row, never an Input; give the content stop a visible focus treatment that does not paint over text (border tint); make Escape close More; in the three-pane shell either retire "‹ Back" or render list mode visibly. **Command:** /impeccable clarify.

5. **[P1] Workflow 1 still has no set-level analysis path, and rows carry no state.** (carried from critique #1; both halves are work I filed and deferred: 28007 batch analyze, 28008/28009 row markers behind the 5-key summary contract.) Rows show `type · 5m` and nothing about analysis or reviewed state (B cap_02); "Analyze after import" is off by default inside a collapsed group with no state on its header; Select mode's bulk row is Export / Review / Delete (canvas.py:359-374); Generate learns "no provider" after the click via toast. **Fix:** "Analyze" in the bulk row reusing `_generate_library_media_analysis` per id in one worker group with an in-list receipt; a state summary on the collapsed header; disable Generate with the resolver's reason in the label; bump the summary contract so rows carry an analysis glyph and a reviewed glyph. **Command:** /impeccable shape.

**P2 batch:** the filter matches titles only, so keyword `day2` on four rows yields "No media matched 'day2'" (B cap_15; cause suspected, not traced), undercutting "Review these" over a tag scope; returning from Trash after Restore leaves the list stale ("Media changed; retry to load a current page.", every row and action `○`, "Page boundary is unknown.") until a manual Retry (B cap_104-110) although the app itself made the change; the Find bar relocates above the header on Enter, pushing the header down six rows (B cap_20), and a `┐─────Local Media item` join artifact appears after Find closes (14 captures); eight rows of Reader chrome before content, an empty byline row, a section header repeating the tab, and `##` headings rendered literally for video transcripts (library_media_viewer_state.py:201-208); Trash header clipped to "Local Trash · 1 i" with ~36 blank rows above a pinned Restore (28015, deferred).

## Persona Red Flags

**Alex (impatient power user):** Analysis mode plus `]` is two keys per item with a silent miss when he forgets. Coming to Media to open a different file while a set is active pulls him to the cursor item on every entry (39683-39692) — user ruling at this close: an explicit open must bypass auto-resume. No key for "Review these" or "Sets"; nine controls above the rows to Tab through. No "next unreviewed" after un-marking with `m`. Delete key does nothing; `t` does, unlabelled. Generate tells him the provider is unready only after the click.

**Sam (keyboard-only, text over color):** Space collapsed the rail while the footer said `space toggle selection`. The F6 content stop exists but shows nothing. The dead-key state is a two-word chip; the field that now owns `]` says nothing. At 100×30 the `esc` hint drops out of the footer entirely (`F1 · Ctrl+P · Ctrl+Q`, B cap_108). Pane grips are bare `--->` / `<---` with words only in a mouse tooltip. Works for Sam: `☑/☐/○` always beside words, `(selected)` on tabs, "✓ reviewed / not yet reviewed" spelled out.

**Conference researcher (40 talks, must miss none):** the Items list never shows which rows are reviewed; the only ✓ is the banner for the loaded item, so finding the one she skipped means re-walking (my set-local design; row markers deferred). The gesture that completes the set (a final `]` on the last item) is labelled `] next in set`. A talk deleted mid-walk silently drops the total. Completion is a five-second toast and a footer chip, no landing state. "Review these" pins the current filter and names the set after it ("pdf items") without warning her the filter is on. Her tag filter says no media matched.

## Minor Observations

- After picking a sort the button clips to `sort: Title A` (B cap_04).
- In the picker `✓` marks the active set and moves as you open sets; one line below, ✓ means completed (B cap_82).
- "Loaded · " eats nine cells of the row that matters most; the ✓ glyph in a title is ellipsized away.
- Viewer has 25 buttons and 0 tooltips (B).
- "Restored outside the current filter." and "Selection discarded (N items)" exist; "1 item in this set was deleted" does not.
- Sets cannot be named; auto-names plus dates are the only handle.
- Doc/UI drift: "Open in Library ▸ Media" vs "Open manager"; "no permanent delete" vs "Delete permanently".
- The Sets picker's empty state points at "Review these", which is behind the modal you are in.

## Decisions I own (corrected at the close; originally mis-framed as questions to the user)

1. Reviewed marks live only on the loaded item because I designed review sets with set-local done marks and twice parked the row-marker work (28008/28009). Proposal: bump the 5-key media summary contract so rows carry a reviewed glyph and an analysis glyph, as its own PR.
2. Batch analysis is missing because I filed it (28007) and never shipped it. Proposal: an Analyze action on the select-mode bulk row, reusing the per-item generator in one worker group with an in-list receipt.
3. I recommended resume-on-every-entry and the user approved it; it now overrides explicit opens. User ruling at this close: an explicit open (deep link, open-by-id, Enter on a row) bypasses auto-resume.
