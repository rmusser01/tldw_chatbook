---
id: TASK-2372
title: Staged evidence and sent-notice were destroyed by Console navigation (critique D3)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04 20:07'
labels:
  - console
  - navigation
  - rag
dependencies: []
priority: medium
---

## Description

The 2026-08-04 RAG re-score critique (D3) found that navigating away from Console and back destroyed any staged evidence and the "Evidence sent with this message" notice. The critique's working theory blamed Library's "Run" action as the destroyer, but Run is a pure function with no side effects on Console state — the actual cause is that screens in this app are never reused (each navigation creates a fresh screen instance), so screen teardown discarded in-memory staged state that had nowhere durable to live.

## Acceptance Criteria

- [x] Staged evidence and the "Evidence sent" notice survive a navigate-away-and-back round trip in Console
- [x] Library's Run action is confirmed innocent of destroying Console state (the critique's original theory)
- [x] A real navigation round-trip test covers this (not only a unit test of the state object in isolation)

## Implementation Notes

Fixed in PR-T1 Task 3, commit `b3114dd88`. Launch context and the sent-notice are now serialized into native console state (rather than living only in transient in-memory fields), so both are restored when Console is re-entered. Verified with a real pilot navigation test (chat → home → chat) plus three handoff-kind tests, all RED before the fix and GREEN after.

Review (sonnet) traced four specific risks and found none reachable as a problem:
- The "zero active sessions" serializer-skip path is non-reachable in practice — compose-time claim of the launch context precedes the async session-bootstrap tick, so save_state never lands in that window from a navigation trigger.
- No `NATIVE_CONSOLE_STATE_VERSION` bump was needed (verified by exhaustive grep — it's a write-only constant here).
- Restore ordering was traced through `app.py` and confirmed to run before `compose()`.
- A collision between a resident restored launch and a newer handoff store entry resolves correctly both ways: the resident restored launch wins, and a newer store entry defers rather than silently dropping.

Deferred (documented, not fixed): the precedence rule above (resident-wins) is currently explained only in test docstrings, not in a code comment at the early-return in `_consume_pending_console_launch`; and the "zero active sessions" path, while non-reachable via this PR's own state, is a pre-existing latent risk (a persistent bootstrap failure would drop ALL native console state) noted in the review report but not filed as a separate task.
