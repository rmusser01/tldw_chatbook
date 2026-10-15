---
id: TASK-34987
title: The Watchlists screen file is over its module-size budget on dev
status: Done
created_date: 2026-10-14 16:21
labels:
- architecture
- tests
- watchlists
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-15 16:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Architecture/test_module_size_ratchet.py pins a line budget per large module that may only go down. tldw_chatbook/UI/Screens/watchlists_collections_screen.py was 14,378 lines on dev on 2026-10-14 against a budget of 14,324, so the ratchet fails on dev and every change to the file inherits the red (PR #3107, which fixes a Watchlists race for TASK-34951, brings it to 14,374, still 50 over). The ratchet is not in the required check, which is why it went unnoticed. The owner's rule is that a budget is never raised to fit growth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The module size ratchet passes on dev for watchlists_collections_screen.py with its budget unchanged or lowered
- [x] #2 The lines removed move into Watchlists modules that tests already cover, with no behaviour change
- [x] #3 Code that nothing references is deleted rather than moved, each method confirmed by an untruncated repo-wide search
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Measured on dev c12611946c, which includes #3107: watchlists_collections_screen.py is 14,354 lines against a 14,324 budget, 30 over. The slack tolerance is 50, so a result between 14,274 and 14,324 passes with the row unchanged; ending lower means lowering the row to the exact measurement.

1. Baseline: run Tests/Architecture serially on the unchanged branch and record every failure with its value.
2. Dead code first. Check for duplicate definitions with ruff F811, with dummy-variable-rgx narrowed to ^_$ so _-prefixed names are covered, and with an AST pass over method names. Delete methods nothing references, each confirmed by an untruncated repo-wide search that covers monkeypatch strings, getattr and hasattr strings, Textual handler naming, @on, and BINDINGS action strings. Candidates: _rail_unread_suffix (caller removed by 3cd3f5bd72) and _watchlists_are_empty (orphaned since TASK-1020).
3. Move the pure OPML import-summary formatter _opml_import_summary_text into Watchlists_Modules/opml_dialogs.py, the screen's existing OPML module. The screen re-exports it, because Tests/Watchlists imports it from the screen module.
4. Leave #3107's restructured code alone: _seed_*, _rail_inputs/_rail_view, _reseed_*.
5. Measure. Rerun Tests/Architecture serially and compare failures by value. Run the Watchlists tests on this branch and on dev-base-1011 and report only new failures. Run preflight under the scratch env. Report any diagnostic-inventory drift without regenerating it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The screen is back under its unchanged budget: 14,354 lines on dev c12611946c, now 14,287, against a 14,324 budget. That is 37 lines under, within the 50-line slack, so the row is not lowered. Dev a37e4bcc01 adds a 6-line comment elsewhere in the file. After the rebase the file will measure 14,293, still 31 under.

- Deleted two methods nothing references (24 lines): `_rail_unread_suffix`, whose last caller was removed by 3cd3f5bd72, and `_watchlists_are_empty`, orphaned since TASK-1020. Each was checked with an untruncated search across code, tests, scripts and Docs/security: calls, monkeypatch, getattr and hasattr strings, @on, Textual handler names and BINDINGS action strings. F811, run with `dummy-variable-rgx = ^_$`, and an AST scan for duplicate method names find nothing duplicated. AC #3 was added for these deletions.
- Moved the pure formatter `_opml_import_summary_text` (43 lines net) verbatim into `UI/Watchlists_Modules/opml_dialogs.py`, the screen's existing OPML module. The screen imports it back, so `Tests/Watchlists/test_watchlists_collections_screen.py` still finds it on the screen module, and the notify call site is unchanged. opml_dialogs is already in the screen's preimport payload, so no import census changes.
- #3107's code (`_seed_*`, `_rail_inputs`/`_rail_view`, `_reseed_*`) is untouched.

**Evidence**
- Tests/Architecture, run serially on c12611946c and on this branch, goes from 56 failures to 55. Only the Watchlists module-ratchet failure is gone; every other failure is the same by name and value. The one wording change is the `test_persona_visual_runtime_boundary` message, which names a different forbidden import; it scans files this task does not touch.
- Tests/Watchlists (all files) plus every Tests/UI file that names the screen or OPML, 59 files in all, give identical failure names at the base and on this branch. Most fail on both with `RecoveryRequired`, and `test_opml_import_summary_text_tells_the_whole_story` passes.
- Preflight passes everything, and no diagnostic-inventory change (no logger moved). Repo-config ruff finds nothing new in either file. /verify (an app launch) was not run: this session forbids launching the app.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
