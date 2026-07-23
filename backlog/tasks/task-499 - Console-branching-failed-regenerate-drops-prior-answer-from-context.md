---
id: TASK-499
title: >-
  Console branching: a failed regenerate drops the prior good answer from
  provider context until swipe-back
status: To Do
assignee: []
created_date: '2026-07-23'
labels:
  - console
  - chat
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With Console branching (Phase A, PR #799), regenerate forks a new empty sibling assistant node and moves the active leaf onto it. If that regenerate stream fails or returns empty, the new sibling ends `failed` (there is no variant base to restore, since `variant_mode=False`), so `_provider_messages_for_session(skip_failed=True)` excludes it — and the original good answer (the anchor) is now off the active path, so it is excluded too. The model therefore loses the previously-good answer from context until the user swipes back to the anchor or retries the failed node. This is a deliberate consequence of the node model and is recoverable/visible (the failed node is shown, and Task 7's swipe + retry-on-failed-sibling are the recovery paths), but the UX footgun is worth smoothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed/empty regenerate does not silently strip the anchor's prior good answer from the next send's context without a clear, discoverable recovery affordance
- [ ] #2 Chosen approach (e.g. auto-restore the anchor as active on failed regenerate, or surface a one-key swipe-back/retry hint on the failed sibling) is documented and unit-covered
- [ ] #3 Verified in the live TUI: regenerate → force a failure → confirm the good answer is recoverable in one obvious step
<!-- AC:END -->
