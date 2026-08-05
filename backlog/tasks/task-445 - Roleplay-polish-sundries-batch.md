---
id: TASK-445
title: Roleplay polish sundries batch
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 09:11'
labels:
  - roleplay
  - ux
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Small items from the review, batched: (1) footer hint noise - persistent "ctrl+s save unavailable | esc back unavailable" text; (2) "1 characters" count grammar; (3) model discovery exists in Settings but its results do not offer themselves into the Model field (users hand-type 50-char gguf names); (4) transient rendering artifact - a tall empty selection frame appeared under the selected library row after first selection; (5) import success toast is easy to miss - consider inline confirmation near the list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Footer shows only currently available actions (or renders unavailable ones dimmed without the word 'unavailable')
- [x] #2 Count line uses correct singular/plural
- [x] #3 Model discovery results can be applied to the Model field with one action
- [x] #4 Library rail no longer renders the empty selection-frame artifact
- [x] #5 Import success is confirmed visibly at normal reading pace
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify each of the 5 items against current HEAD (drift check) before touching code.
2. Item 1 (footer 'unavailable' noise): root-caused to ShortcutContext.render() in
   UI/Navigation/shortcut_context.py rendering every ShortcutAction unconditionally with
   a literal ' unavailable' suffix; personas_screen.py's dynamic footer context is the
   only caller that sets available=False. Fix: render() drops unavailable actions
   entirely (the AC's first option: show only currently-available actions). Update the
   two existing tests in test_personas_workbench.py that pinned the old text.
3. Item 2 ('1 characters' grammar): root-caused to personas_library_pane.py's count line
   (f"{total} {noun}" / f"{len(rows)} of {total} {noun}") never adjusting for count==1.
   Add _noun_for_count() using the file's existing _singular_noun() helper; apply at both
   sites keyed off total (the number the noun describes). RED tests added first.
4. Item 3 (model discovery -> Model field): investigate only; if already covered by a
   prior task, document drift disposition instead of re-implementing.
5. Item 4 (library-rail selection-frame artifact): reproduce live via tmux + SGR mouse
   clicks (verify skill) against the real app before guessing; root-cause via CSS/DOM
   inspection once reproduced; fix in the component .tcss and rebuild the bundle.
6. Item 5 (import success toast easy to miss): compare the notify severity/timeout used
   at the import-success call sites against comparable confirmations elsewhere in the
   codebase; adjust minimally to linger longer if it is drifting short.
7. Run the full personas/footer/settings-discovery test files + `python -c "import
   tldw_chatbook.app"` in the foreground; fix any fallout from the render()/count
   changes surfaced by existing tests.
8. Update task file (AC checkboxes, Implementation Notes with per-item disposition) and
   commit only the touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Per-item disposition (all 5 verified live/via tests before and after; evidence below):

1. Footer 'unavailable' noise -- REAL BUG, FIXED. Root cause: ShortcutContext.render()
   in UI/Navigation/shortcut_context.py rendered every ShortcutAction unconditionally,
   appending a literal " unavailable" suffix when available=False. personas_screen.py's
   dynamic _shortcut_context() is the only caller that ever sets available=False (ctrl+s
   save / esc back / ctrl+enter attach), so a fresh/unedited Personas screen showed
   "ctrl+s save unavailable | esc back unavailable" persistently, exactly as the review
   saw. Fix: render() now drops unavailable actions entirely (AC's first option). Live
   tmux capture after the fix: footer reads "ctrl+n new | ctrl+f search | [ ] mode" with
   no "unavailable" anywhere. Updated 3 existing tests in test_personas_workbench.py that
   had pinned the old text (test_footer_shortcut_context_set_and_cleared,
   test_footer_save_hint_flips_with_edit_mode, test_import_refreshes_attach_action) --
   they now assert the hint's ABSENCE when unavailable instead of the literal suffix.
   Other AppFooterStatus.set_workbench_shortcuts() callers (Console, Library, Settings,
   MCP) always pass available=True (no availability concept there), so this is
   zero-blast-radius outside Personas.

2. "1 characters" grammar -- REAL BUG, FIXED. Root cause: personas_library_pane.py's
   count line (`f"{total} {noun}"` / `f"{len(rows)} of {total} {noun}"`) never adjusted
   for count==1, even though the file already had a `_singular_noun()` helper (used only
   for the "Showing N x matches" phrasing). Added `_noun_for_count(count, noun)` and
   applied it at both remaining sites, keyed off `total` (the number the noun
   grammatically describes). RED tests added first (test_singular_count_uses_singular_noun,
   test_singular_filtered_count_uses_singular_noun) confirmed the bug, then went green.
   Live tmux capture: Characters library rail now reads "1 character" (was "1 characters").
   Fixed 3 stale assertions elsewhere in the suite that had pinned the old plural-only
   text at count==1 (test_search_filters_loaded_characters_locally,
   test_search_filters_profiles_in_personas_mode,
   test_update_rows_without_page_kwargs_keeps_plain_count).

3. Model discovery -> Model field apply -- ALREADY SATISFIED on this branch (TASK-369,
   already committed history, part of the merged Console-UX-expert-review batch).
   Settings' "Save selected" button (#settings-save-discovered-provider-models) both
   persists the selected discovered models AND auto-populates an EMPTY Model field with
   the first saved model id (_activate_saved_model_if_field_empty /
   _model_to_activate_after_save) -- a single action. The Model field also carries a
   live typeahead suggester over discovered model ids
   (_model_field_suggester/_refresh_model_field_suggester), and the Model field + the
   "Model discovery" section live in the SAME Settings category/screen, right below each
   other, so there is no cross-screen apply problem. Verified via existing tests
   (test_model_to_activate_after_save_prefers_first_saved_when_field_empty,
   test_model_field_suggester_completes_discovered_ids) -- both pass on current HEAD, no
   code change needed. No split-out required.

4. Library-rail empty selection-frame artifact -- REAL BUG, FIXED. Reproduced live via
   tmux + injected SGR mouse clicks against the real running app (Characters mode,
   select "Default Assistant"): a bordered box (`| |` pairs closed by a bottom
   `L______J`) rendered from directly under the 2-row character item all the way down to
   the count line -- ~24 empty rows. Root cause: the global fallback
   `*:focus { outline: solid $ds-focus-accent; }` (core/_reset.tcss, "visible
   non-obscuring keyboard focus fallback") draws its outline around the
   `#personas-library-rows` ListView's own box once it takes keyboard focus after a row
   is selected; that box fills the rail's remaining height (taller than its 1-2 actual
   rows), so the outline rendered as the reported "tall empty selection frame ... under
   the selected library row". The row already shows selection via
   .is-active/.-highlight backgrounds, so the fallback outline was both redundant and
   visually broken here. Fix (mirrors the existing TASK-383 Console-chip precedent):
   added `#personas-library-rows:focus { outline: none; }` to
   css/components/_agentic_terminal.tcss and rebuilt the bundle via
   `python3 tldw_chatbook/css/build_css.py`. Re-verified live: the stray inner box is
   gone after the fix; only the outer library-pane frame remains. Added
   test_personas_library_rail_focus_outline.py (CSS-presence contract test, same pattern
   as test_console_chip_focus_contract.py) covering both the component tcss and the
   built bundle.

5. Import success confirmation easy to miss -- REAL ISSUE, FIXED. The two
   "Character imported." / "Character already existed; selected it." notifies at the end
   of _import_character_from_path() had no explicit timeout, so they used Textual's
   plain 5s App.NOTIFICATION_TIMEOUT default -- and that toast appears at the exact same
   moment the card view + inspector swap in (a big simultaneous visual change), so it
   reads as a flash even at 5s. Extended PersonasScreen._notify() with an optional
   keyword-only `timeout` param (defaults to None, preserving all ~144 other call sites'
   behavior unchanged) and passed timeout=6.0 at the two import-success call sites,
   matching the codebase's existing convention for confirmations that need a deliberate
   beat (e.g. import-conflict warnings elsewhere use timeout=6). Added
   test_import_success_notification_lingers_past_the_app_default asserting the notify
   call's timeout kwarg is present and > 5.

Modified: UI/Navigation/shortcut_context.py, UI/Screens/personas_screen.py,
Widgets/Persona_Widgets/personas_library_pane.py, css/components/_agentic_terminal.tcss
(+ rebuilt tldw_cli_modular.tcss). Tests: test_personas_workbench.py,
test_personas_library_pane.py, test_personas_library_pane_paging.py, new
test_personas_library_rail_focus_outline.py.

Verification: full personas/footer/settings-discovery/chip-focus test set (249 tests)
green; requested sanity subset
(test_personas_workbench.py -k "readiness or Console or gate or duplicate", 29 tests)
green; `python -c "import tldw_chatbook.app"` clean. One PRE-EXISTING baseline failure
found and confirmed unrelated via git-stash bisection:
test_screen_footer_hints.py::test_library_registration_updates_the_screens_own_footer
fails identically with and without this diff (a LibraryScreen background worker raises
"ValueError: Local prompt backend is unavailable." during PromptScopeService in this
harness, breaking footer registration before it runs) -- left untouched, out of scope
for task-445.
<!-- SECTION:NOTES:END -->
