---
id: TASK-41653
title: "Tests/ChaChaNotesDB historical bootstrap cases 82..86 are red and run on no CI lane"
status: In Progress
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- database
priority: medium
references:
- Tests/ChaChaNotesDB/test_historical_bootstrap.py
- Tests/ChaChaNotesDB/historical_bootstrap.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41310 A (schema v86→v87) found the historical bootstrap cases for versions 82..86 red: the bootstrapped schemas have tables the historical declarations do not list. No CI lane runs the file, so the failures went unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The historical bootstrap cases 82..86 pass
- [ ] #2 The file runs on a CI lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test literal only; the schema and its version are unchanged).

1. Run the cases on dev d0486978dd and read the failing assertion.
2. Find the migrations whose tables/columns the hand-declared literal misses, and the commits that added them.
3. Declare them in `Tests/ChaChaNotesDB/expected_table_columns.py`, transcribed from the migration SQL (never from a live schema), in the file's name order.
4. Re-run the whole sweep serially and record its time for the coordinator's lane decision (AC #2).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (test drift).** Every case, 4..86, failed, not only 82..86: the assertion that fires is on the *fresh template*, before any replay comparison. `EXPECTED_TABLE_COLUMNS` never declared 14 tables and 2 columns that four migrations added:
- v80->v81 `chachanotes_v80_to_v81_response_rules.sql` (d24c4c04b2): the six `console_response_rule_*` tables and `console_machine_followup_receipts`.
- v82->v83 / v83->v84 (d2b5eeb7ca, 05fde3443f): `character_portrait_performances` (plus v84's `composition_recipe_json`), `character_portrait_performance_frames`, `character_portrait_regions`.
- v84->v85 (dd8d60c9e8): `conversations.archived_at`.
- v85->v86 `chachanotes_prompt_mixins.sql` (7970fbcada): the four `console_prompt_mixin_*` tables and `console_dispatch_checkpoints.mixin_snapshot_schema_version`.

**Fix.** Declared them, transcribed from the CREATE/ALTER TABLE text of those files. No schema or version change.

**Evidence (2026-11-18, load 45-90).**
- Base d0486978dd, `-n 4`: 83 of 83 sweep cases failed ("[fresh template] live tables not declared ...").
- Head, serial, with `test_trigger_census.py`: 102 passed in 235 s wall, 50.6 s user CPU at load 80-90.

**AC #2 (coordinator).** The file is not `bootstrap_profile`; it bootstraps 83 historical databases (one per version) plus one template, each 2-3 s at load 80. Serial: about 230 s wall / about 50 s CPU here, so expect about 1 min on an unloaded runner, or about 20 s at `-n 4`. Recommended lane: `recovered-suites` in `.github/workflows/derived-artifacts.yml`, beside `Tests/ChaChaNotesDB/test_trigger_census.py`, which has the same drift class and no lane either.

**Seen, not fixed (other ChaChaNotesDB reds on base, same dir, not this AC):** `test_index_census.py` (5 unpinned indexes from v81..v86, and `idx_canvas_revisions_canvas_sequence` lost its UNIQUE flag in v81->v82), `test_sargable_timestamps.py` (pins version 80), the v82/v83 portrait migration tests, and cascade-delete tests refused by "semantic mutation authorization required". These are reported to the coordinator as follow-ups.

**Why it drifted.** The checklist in `tldw_chatbook/DB/migrations/README.md` named `VALID_TABLES` and the index census, but not this literal or the trigger census. It now has steps 7 and 8 for them.

**Files.** `Tests/ChaChaNotesDB/expected_table_columns.py`, `tldw_chatbook/DB/migrations/README.md`.
<!-- SECTION:NOTES:END -->
