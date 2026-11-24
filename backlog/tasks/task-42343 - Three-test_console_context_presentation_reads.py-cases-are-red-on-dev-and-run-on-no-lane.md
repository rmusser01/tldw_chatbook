---
id: TASK-42343
title: >-
  Three test_console_context_presentation_reads.py cases are red on dev and run
  on no lane
status: Done
assignee: []
created_date: '2026-11-22T16:37:00Z'
updated_date: '2026-11-24T20:57:43Z'
labels:
  - follow-up
  - tests
  - console
dependencies: []
references:
  - Tests/Chat/test_console_context_presentation_reads.py
  - tldw_chatbook/Chat/console_context_repository.py
  - .github/workflows/derived-artifacts.yml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-34975 port (round 20, fu/r20-ctxclose) and its review. The port's notes record the same three nodes failing with dev's source swapped in. Re-run at 59cc99e61b (dev 8a55555798 plus the round-20 stack) through the scratch wrapper at `-n 4`; all three are red. The file is in no PR lane (`scripts/ui_pr_gate_census.txt` and `.github/workflows/` do not name it), so nothing stops it going red.

- `test_cold_and_expired_presentation_never_fall_back_to_live_ui_reads` patches `controller.context_control_inputs`. That method is a `_CONTEXT_DISPLAY_CALLBACKS` member, so `callbacks_current()` (`console_context_repository.py:1343`) returns False and the read takes the live-UI fallback the test forbids ("presentation fell back to live UI reads", `:107`).
- `test_actual_compaction_resume_invalidates_context_echo[False]` and `[True]` seed no memory. The lineage skip therefore reads no versions: `[False]` fails at `:393` (`assert (0 == 1)`, no version read) and `[True]` at `:385` (the held version read is never entered).

Until these pass, the file cannot guard the context-presentation reads that TASK-34975 changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of each red node is named, and either the test or the product is fixed (the product only where the test's expectation is right)
- [x] #2 All three nodes pass through the scratch wrapper at `-n 4` on a branch carrying the fix
- [x] #3 The file runs on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the three cases on current dev. Observe live readers without changing callback identity and persist memory before taking the compaction case snapshot.
2. Run the three cases and the full file through the same private profile at four workers. Measure the serial file and add it to a required admission-sensitive lane.

ADR required: no
ADR path: N/A
Reason: Test fixtures and CI coverage only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
On dev 9fa2bf1b1d the three reported cases failed at four workers. The fixed exact cases passed 3/3, and the full module passed 15/15 at four workers (43.33 seconds) and 15/15 serially (74.12 seconds). The cold/expired check observes the original callback without changing its identity. The compaction test persists memory before capturing the initial revision. Required PR Fast Lane (admission-sensitive 3) now selects the whole module. Hosted execution is pending.

The final Ubuntu 24.04 qualification passed all 15 cases at four workers in 15.37 seconds. Its scheduled PR Fast Lane job was still queued when this evidence was recorded. [Linux job](https://github.com/rmusser01/tldw_chatbook/actions/runs/45475265787/job/138733698816).

The official PR Fast Lane (admission-sensitive 3) job on PR #3763 passed on reviewed commit 0df0b348f6, including this task's test suite. [GitHub Actions job](https://github.com/rmusser01/tldw_chatbook/actions/runs/45480195611/job/138754609078).
<!-- SECTION:NOTES:END -->
