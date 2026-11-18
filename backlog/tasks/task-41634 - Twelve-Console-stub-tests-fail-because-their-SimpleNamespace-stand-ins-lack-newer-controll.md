---
id: TASK-41634
title: "Twelve Console stub tests fail because their SimpleNamespace stand-ins lack newer controller attributes"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_turn_attention.py
- Tests/UI/test_console_transcript_context_projection.py
dependencies:
- TASK-41313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41313 found 12 tests red on dev: 3 in `Tests/UI/test_console_turn_attention.py` and 9 in `Tests/UI/test_console_transcript_context_projection.py`. They build the screen or controller from a `SimpleNamespace`, which lacks attributes the product now reads (`_current_console_conversation_id`, `sync_unproven_cleanup`). This is test drift, not a product bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 12 tests pass, built from the real collaborator or a stub that carries what the code under test reads
- [x] #2 No product code changes only to satisfy a stub
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test-only drift repair)

1. Run both files and read each failure; name the product commit that added each read.
2. Give each stub exactly the attribute the code now reads, with the commit named beside it.
3. Where a stub cannot satisfy a changed contract by an attribute, change the harness, not the product, and mutation-check that the tests still see the product behaviour they cover.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Causes (all test drift; no product change).**
- `ab92bbb9d0` (TASK-40716, 2026-11-14) made the transcript sync call `screen._agent.sync_unproven_cleanup(transcript, messages)`. Neither harness's `_agent` stub had it: all 12 tests failed there.
- `a699492c7b` (TASK-33620.15.3, 2026-10-10) made the sync read `screen._current_console_conversation_id()` for re-freeze bookkeeping. `test_console_turn_attention.py`'s `_sync_harness` lacked it (the projection file's stub already had it).
- `95ca4bf47b` (PR #3536 replay, 2026-11-14) moved the transcript tick-gate commit from the gate check to the end of a published pass (`acknowledge_console_transcript`). The projection file's harness stopped every pass by raising `BannerPainted` from the banner paint, so no pass ever committed the gate, and three gate tests (`test_context_publication_reopens_unchanged_transcript_gate`, `test_custom_live_reader_keeps_original_post_gate_cadence_and_thread`, `test_custom_presentation_reader_is_not_newly_used_by_transcript`) saw every later render refetch. This third cause was hidden behind the first.

**Fix.** Added the two attributes to the stubs (`sync_unproven_cleanup` returns 0 held slots). In the projection file the banner paint now records instead of raising, `refresh_messages` is an `AsyncMock`, and the stub screen gets inert doubles for the rest of the pipeline (image, video, speech, fingerprint, receipts, guidance), so each pass runs to publication as in the app. `BannerPainted` is gone.

**Evidence** (worktree fu2-redui at d0486978dd, scratch-profile wrapper, load ~35):
- Before: both files at `-n 4`: 12 failed, 40 passed (the 12 listed in the task).
- After: both files at `-n 4`: 52 passed. The projection file alone, serial: 9 passed.
- Mutation: with the gate commit in `acknowledge_console_transcript` removed, the three gate tests fail again (3 failed, 6 passed), so the reworked harness still sees the gate.

Files: `Tests/UI/test_console_turn_attention.py`, `Tests/UI/test_console_transcript_context_projection.py`.
<!-- SECTION:NOTES:END -->
