---
id: TASK-38061
title: Isolate deferred Console acceptance sync evidence from unrelated callbacks
status: Done
created_date: 2026-10-30 05:26
assignee:
- '@codex'
updated_date: 2026-10-30 05:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3308 required UI CI counts two empty-draft action-state syncs in the acceptance projection test. Determine the extra callback owner and keep the regression evidence specific to the accepted clear without weakening its before-and-after behavior checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The acceptance projection test identifies its own deferred callback and still rejects inline or missing synchronization.
- [x] #2 Relevant Console and local SSE regression checks pass, with hosted failure attribution and evidence limits recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the hosted assertion and trace the mounted composer sync callers in an unchanged targeted run.
2. If unrelated layout or startup work contaminates the count, isolate the projection callback while retaining exact count, draft, restyle and final Send-state assertions; otherwise repair the proven production defect.
3. Exercise missing and inline callback negative controls, run targeted Console and local provider checks, lint, and request independent review before re-arming PR3308.
ADR required: no
ADR path: N/A
Reason: anticipated test-evidence isolation repair; reassess if the traced cause needs a production boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Required CI run 40977116425 failed UI Fast Lane 6 because the acceptance-projection spy recorded two empty-draft syncs; the aggregate check propagated that failure. A targeted local diagnostic reproduced the same count and traced one callback through Textual on_callback and one through _sync_native_console_chat_ui -> _sync_console_session_draft -> load_draft. This identifies a local background caller; the hosted extra invocation was not stack-traced.
Restore the composer's original _sync_current_action_state after the projection's synchronous assertions, before yielding. call_later already captured the recorder, so its count remains specific to that projection. No production changes, waits, or count/outcome assertions were relaxed. Ruff normalized the adjacent pre-existing assertion formatting.
GREEN: all five tests in test_console_composer_state_restyle.py passed (68.94s); all 104 local SSE/provider/cleanup cases passed on the rebased branch (22.34s). Isolated temporary negative controls each failed at the intended assertion: missing callback recorded zero, inline callback restyled Send in the projection, duplicate callback recorded two. Existing dependency warnings only. No full suite or live server qualification claimed.
Scoped Ruff check, Ruff format check and git diff --check pass. Independent review found no actionable issues and confirmed captured callback identity and all three negative-control semantics. Updated the acceptance-projection test and incident lesson. ADR required: no; test-observation isolation only. Latest dev is ff815f8559, user Qodo waiver for PR3308 remains limited to Qodo.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
