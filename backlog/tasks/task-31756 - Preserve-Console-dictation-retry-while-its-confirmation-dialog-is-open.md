---
id: TASK-31756
title: Preserve Console dictation retry while its confirmation dialog is open
status: Done
created_date: 2026-09-05 22:04
references:
- backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
- Tests/UI/test_console_dictation.py
updated_date: 2026-09-05 22:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Choosing Retry or Keep draft in the real Parakeet retry dialog leaves the composer stuck on Dictate… instead of completing the captured-audio recovery. Restore the intended retry and decline behavior while preserving cleanup on navigation and teardown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirming the real retry dialog inserts the recovered transcript once and returns dictation to idle.
- [x] #2 Declining the real retry dialog preserves the draft, clears retained audio and returns dictation to idle.
- [x] #3 Navigation, teardown and stale retry completion still release retained capture state and Buddy ownership without affecting newer captures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: Routine retry lifecycle correction preserving existing capture and request-owned Buddy boundaries.

1. Use the two baseline-confirmed failing mounted retry-dialog tests to trace the lost completion.
2. Make the smallest correction preserving retry/decline behavior and full navigation/unmount cleanup.
3. Run focused retry plus Buddy lifecycle tests and compare scoped Ruff/Bandit against baseline.
4. Independently review immediate stop/cancel/start Buddy event timing and record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed the real retry dialog's suspend race: opening the owned confirmation dialog was invoking unconditional dictation teardown, which discarded retained audio and invalidated the completion handler. Console now calls dictation.suspend(); only the exact active retry dialog preserves its retained recording. Explicit teardown and unrelated suspension remain unconditional. A post-dialog session identity check prevents a late affirmative answer from replaying audio after teardown. ADR-074 remains the existing ownership boundary; no new ADR.

Files: tldw_chatbook/UI/Console_Modules/dictation.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_dictation.py, Tests/UI/test_console_dictation_buddy.py, backlog/docs/lessons-testing-evidence.md. The prior Buddy change (renumbered TASK-31812 from TASK-31741) remains intact.

Verification: prior unchanged-code failures were the two mounted Retry/Keep draft cases. After correction, pytest Tests/UI/test_console_dictation.py Tests/UI/test_console_dictation_buddy.py -q --tb=short: 23 passed, one existing requests warning (47.64s). Streaming selection (-k 'retry or abandon or start_worker or preparing or start_returns'): 5 passed, 83 deselected (5.53s). New teardown-before-confirm regression fails with retry_calls 1 instead of 0 when the identity fence is removed, and passes with the fence. Scoped Ruff across both production files and tests: 204 existing findings before/after, none added. Scoped Bandit on both production files: 9 before/after, none added. New Buddy tests Ruff check/format clean; git diff --check passes. Existing production/test formatting debt was preserved.

Independent Buddy timing review: listening is published only after successful startup and the existing mounted/session identity fence. Cancelling during startup nulls the session first, so late successful startup discards rather than acquiring. State exits release the captured exact sink/UUID, so delayed old callbacks cannot clear newer ownership. Preparation cancellation and same-session restart regressions pass. No microphone, server, or git mutation performed. Main task owns final integration.
Root verification repeated the focused mounted dictation/Buddy suites: 23 passed (42.86s), /private/tmp/migu-dictation-root-verification.log. Fix is committed as 14cd81b2247acde771180c3af95a8d6cd5f03ecb and independently reviewed. No new lint or security findings versus baseline; task documentation, regression evidence, ADR check and incident lesson are complete. Human listening-state acceptance remains separately tracked in TASK-31812.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
