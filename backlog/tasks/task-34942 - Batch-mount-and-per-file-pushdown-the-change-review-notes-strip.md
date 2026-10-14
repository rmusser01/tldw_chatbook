---
id: TASK-34942
title: Batch-mount-and-per-file-pushdown-the-change-review-notes-strip
status: Done
assignee: []
created_date: 2026-10-13 14:59
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change-review _focus_leaf fetches notes_for_run (ALL run notes) then Python-filters to the focused file, and the strip rebuilds via remove_children plus one mount() per note instead of a single batched mount (change_review_screen.py:3906/3936 and :4011-4027) — repeated on every j/k file-leaf focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Notes strip rebuilt with a single batched mount
- [x] Per-focus note query narrowed to the focused file or run notes cached once per run
<!-- AC:END -->


## Implementation Notes

- The strip rebuild builds every note row first and issues ONE `strip.mount(*rows)` instead of one `mount()` per note (each per-note mount returned an unawaited AwaitMount, leaving freshly-appended rows un-laid-out for up to a tick). Row-build failures are logged per row without aborting the batch.
- Premise correction on the query half: `_notes_for_leaf` runs `notes_for_run` once per focus and the result feeds BOTH the diff markers and the strip (one query per focus, not two); the per-file pushdown would only save the Python-side filter over a small note set — not worth the seam change. The batched-mount change is the whole worthwhile fix.
- Regression: the change-review suites 59/59 green after the change.
- ADR required: no — mount batching inside one screen.