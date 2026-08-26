---
id: TASK-2371
title: Console first send on a fresh session could vanish silently (critique D2)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04 20:07'
labels:
  - console
  - reliability
dependencies: []
priority: high
---

## Description

The 2026-08-04 RAG re-score critique (D2) reported that on a fresh profile, the first send in a new Console session could vanish entirely: the composer cleared, no transcript row was appended, and no DB row was written — with nothing shown to the user to indicate the message was dropped.

## Acceptance Criteria

- [x] Silent-refusal paths in the send pipeline surface a visible signal instead of dropping a message with no trace
- [x] The no-op `Button.press()` path identified as a plausible live contributor is hardened against silently swallowing a send
- [x] Where the original hypothesis (an empty-string session-id sentinel) could not be confirmed as the true mechanism, that is recorded honestly rather than claimed as fixed, and a recurrence-detection mechanism is in place to identify the true cause if it recurs

## Implementation Notes

Fixed/hardened in PR-T1 Task 4, commits `487f07296` (initial) and `863c8ab71` (review round 1 fixes).

**Honesty note (load-bearing):** the `""` session-id sentinel premise that motivated D2 did **not** reproduce at HEAD — the blocked-reason call path already creates the session before the id is read, a long-standing behavior, not new to this PR. The alternative hypothesis (a no-op `Button.press()` swallowing the click) was also fixed, but review determined the pure no-op-press hypothesis **cannot alone explain** the reported "the second send worked via keyboard" detail — a duplicate-send guard would have latched shut in that case. **The true mechanism behind the original live observation remains unproven.**

What shipped is deliberate hardening, not a proven root-cause fix:
- Silent refusal paths in the send pipeline (`_active_run_rejection`) now append a visible row, scoped specifically to `submit_draft` (the one caller that lacked any signal; six call sites enumerated to confirm no other caller needed it).
- The no-op `Button.press()` path is guarded with a stash-restore watchdog plus a duplicate-guard unsticking fix.
- A recurrence logger is now in place specifically so that if this defect resurfaces, its actual mechanism can be identified from evidence rather than re-guessed.

Review (opus) confirmed the refutation via an all-entry-paths trace (single funnel, four entries, all through the blocked-reason check) and required three fixes in round 1: the new "session closed" toast was scoped from an over-wide ~19/20 mid-stream close sites down to the actual dispatch gap; the SYSTEM refusal row was scoped from six callers down to the one (`submit_draft`) that lacked a signal; and the watchdog's latch was hardened (identity-checked, consumption-decoupled) against a check-vs-delivery race that could otherwise permanently swallow it.
