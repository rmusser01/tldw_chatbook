---
id: TASK-42343
title: "Three test_console_context_presentation_reads.py cases are red on dev and run on no lane"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_context_presentation_reads.py
- tldw_chatbook/Chat/console_context_repository.py
- .github/workflows/derived-artifacts.yml
dependencies: []
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
- [ ] #1 The cause of each red node is named, and either the test or the product is fixed (the product only where the test's expectation is right)
- [ ] #2 All three nodes pass through the scratch wrapper at `-n 4` on a branch carrying the fix
- [ ] #3 The file runs on a PR lane
<!-- AC:END -->
