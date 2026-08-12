---
id: TASK-15477
title: Media viewer prompt search is dead code that raises per keystroke
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - bug
  - media
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: `Widgets/Media/media_viewer_panel.py:1606` imports `get_prompts_db` from `DB/Prompts_DB` — a symbol that does not exist anywhere in the repo — so every keystroke in the prompt-search box raises ImportError, swallowed at `:1643`; the feature has silently never worked. The handler also calls `call_from_thread` from the UI thread (`:1641`, illegal in Textual) and, as designed, would run its sqlite search inline per keystroke with no debounce/worker (`:1624-1631`).

Decide: wire it to the real Prompts DB seam (threaded + debounced per the task-15476 shape) or remove the affordance. Either way the per-keystroke exception churn stops. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt search either works against the live prompts store, off-loop and debounced, or the affordance is removed
- [x] #2 No exception per keystroke (log evidence)
- [x] #3 A regression test covers the chosen path
<!-- AC:END -->

## Implementation Plan

Investigated reachability before deciding:

1. `MediaViewerPanel` (`Widgets/Media/media_viewer_panel.py`) is only ever
   constructed by `UI/MediaWindow_v2.py`, which backs the standalone
   `MediaScreen` route (`UI/Screens/media_screen.py`).
2. `UI/Navigation/screen_registry.py`'s `_SCREEN_ALIASES["media"] = "library"`
   (task-2851, Library UAT 2026-08-06) permanently redirects the "media"
   route — including any startup-config `default_screen` — to Library's own
   media canvas instead. `app.py:1481`/`:7473-7481` confirm the same fold at
   the shell-destination layer. There is no other call site that reaches
   `MediaScreen`/`MediaWindow_v2`/`MediaViewerPanel` from live navigation.
3. Library's *replacement* media viewer (`Widgets/Library/library_media_viewer.py`,
   the one actually reachable today) never grew an equivalent
   provider/model/prompt-search "Generate Analysis" panel -- it only has a
   read-only analysis text + a manual edit `TextArea` (`_compose_analysis`).
   So the broken prompt-search affordance is not a regression of a feature
   users currently rely on; it never had a live audience post-fold.
4. Decision: **remove the affordance**, not wire it to a DB seam. Building a
   threaded+debounced `PromptScopeService`/`PromptsDatabase` integration
   (the task-15476/console-picker shape) for a screen with zero reachable
   entry points would be net-new maintenance surface for code nothing can
   ever mount from the running app -- the opposite of long-term stability.
   The manually-editable System Prompt / User Prompt `TextArea`s and the
   "Generate Analysis" button stay: they don't depend on the search/select
   widgets (confirmed via `prepare_analysis_messages`/
   `handle_generate_analysis`, which only read the two `TextArea`s) and
   remain the only way to drive analysis today, matching the Library
   replacement's own "no prompt library" scope.
5. Empirically reproduced the bug pre-fix with a throwaway pilot-mounted
   probe: typing into `#prompt-search-input` logs
   `ERROR ... search_prompts:1644 - Error searching prompts: cannot import
   name 'get_prompts_db' from 'tldw_chatbook.DB.Prompts_DB'` on every
   keystroke, swallowed, UI unaffected -- matches the audit exactly.
6. Implementation steps:
   - Write regression tests first (red against current code): assert the
     panel's `compose()` no longer yields `#prompt-search-input`,
     `#prompt-keyword-input`, `#prompt-select` (and their "Search Prompts:"/
     "Filter by Keywords:" labels), and that `MediaViewerPanel` no longer
     defines `search_prompts`/`load_prompt_details`/`_update_prompt_select`/
     `handle_prompt_search`/`handle_prompt_keyword_change`/
     `handle_prompt_selection`. Add a companion test asserting the System/
     User Prompt `TextArea`s and Generate button are untouched (guard
     against over-deleting).
   - Remove those five widgets, three id-scoped `@on` handlers, and the
     three backing methods (`search_prompts` incl. the illegal
     `call_from_thread`, `_update_prompt_select`, `load_prompt_details`)
     from `Widgets/Media/media_viewer_panel.py`.
   - Confirm no CSS/`save_state`/`restore_state` reference to the removed
     ids (checked: none found -- `.prompt-label` is a shared class still
     used by the surviving System/User Prompt labels).
   - Run the new tests plus the existing `media_viewer_panel`-touching
     suites (`test_media_handoffs.py`, `test_reader_scroll_keys_1994.py`,
     `test_media_window_v2_parity.py`, `test_markdown_hygiene_1995.py`).

## Implementation Notes

**Chosen resolution: removed the affordance** (not wired to a live DB
seam). Reachability investigation (see plan) found `MediaViewerPanel` --
the only place these widgets existed -- is exclusively mounted by
`MediaWindow_v2`, which backs the standalone `MediaScreen` route.
`UI/Navigation/screen_registry.py`'s `_SCREEN_ALIASES["media"] = "library"`
(task-2851) permanently folds that route onto Library's own media canvas,
which never grew an equivalent LLM-analysis/prompt-search panel of its own
(`Widgets/Library/library_media_viewer.py` only has a read-only analysis
text + manual edit `TextArea`). So the search box had no reachable
audience: wiring it to `PromptScopeService`/`PromptsDatabase` (the
task-15476/console-picker shape) would have added real maintenance surface
for a screen nothing in the running app can mount. Confirmed pre-fix via a
throwaway pilot-mounted probe that typing into `#prompt-search-input`
logged `Error searching prompts: cannot import name 'get_prompts_db' from
'tldw_chatbook.DB.Prompts_DB'` on every keystroke (matches the audit
exactly) with no crash surfacing.

**What changed** (`tldw_chatbook/Widgets/Media/media_viewer_panel.py`):
removed the "Search Prompts" / "Filter by Keywords" `Input`s and the
prompt-selection `Select` from `compose()`; removed the three backing
methods (`search_prompts` -- the `@work(thread=True)` method with the
nonexistent import and the illegal `self.app.call_from_thread` off a
worker thread, `_update_prompt_select`, `load_prompt_details`); removed
the three `@on` handlers wired to those widget ids
(`handle_prompt_search`, `handle_prompt_keyword_change`,
`handle_prompt_selection`). The System Prompt / User Prompt `TextArea`s
and the "Generate Analysis" button are untouched -- `handle_generate_analysis`
and `prepare_analysis_messages` only ever read those two `TextArea`s, never
the removed widgets, so manual prompt entry + analysis generation keeps
working exactly as before. No CSS or `save_state`/`restore_state` code
referenced the removed ids (verified by grep before removing).

**Tests** (`Tests/UI/test_media_viewer_prompt_search_15477.py`, new file):
5 tests, mounting the real `MediaViewerPanel` via a Textual pilot harness
(mirrors `test_media_handoffs.py`'s `MediaViewerTestApp` pattern). Two are
literal red-then-green regression tests for AC1/AC2 (confirmed red against
pre-fix code, then green after the edit):
`test_prompt_search_widgets_are_gone` (the three removed ids raise
`NoMatches`) and `test_prompt_search_methods_are_removed` (`hasattr`
checks against `MediaViewerPanel` for the five removed method/handler
names). Three supporting tests: `test_surviving_prompt_widgets_still_compose`
(guards against over-deletion of the System/User Prompt `TextArea`s and
Generate button), `test_prompts_db_module_still_has_no_get_prompts_db`
(documents the root cause and guards against a future PR re-adding the
same broken import without verifying the symbol first), and
`test_mounting_and_settling_the_panel_logs_no_prompt_search_error` (loguru
sink capture over a full pilot mount+settle, AC2's "log evidence" --
combined with the other two it closes the loop that no code path can ever
produce that log line again).

**Verification**: `pytest Tests/UI/test_media_handoffs.py
Tests/UI/test_reader_scroll_keys_1994.py
Tests/UI/test_media_window_v2_parity.py Tests/UI/test_markdown_hygiene_1995.py
Tests/UI/test_media_viewer_prompt_search_15477.py` -- 51 passed, 0 failed
(the four pre-existing `media_viewer_panel`-touching suites found via
`grep -rl media_viewer_panel Tests/`, plus the new file). `pyflakes` clean
on both changed/added files. `ast.parse` sanity check on the edited module.

**Files changed**: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
(124 lines removed, 0 added); `Tests/UI/test_media_viewer_prompt_search_15477.py`
(new).
