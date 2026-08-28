---
id: TASK-571
title: >-
  Console branching: a failed regenerate drops the prior good answer from
  provider context until swipe-back
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-25'
updated_date: '2026-08-28 05:19'
labels:
  - console
  - chat
  - ux
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-27-console-failed-regenerate-auto-restore-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(Re-filed from task-499 in the PR #803 rework, unchanged in substance.)

With Console branching (Phase A, PR #799), regenerate forks a new empty sibling assistant node and moves the active leaf onto it. If that regenerate stream fails or returns empty, the new sibling ends `failed` (there is no variant base to restore, since `variant_mode=False`), so `_provider_messages_for_session(skip_failed=True)` excludes it — and the original good answer (the anchor) is now off the active path, so it is excluded too. The model therefore loses the previously-good answer from context until the user swipes back to the anchor or retries the failed node. This is a deliberate consequence of the node model and is recoverable/visible (the failed node is shown, and swipe + retry-on-failed-sibling are the recovery paths), but the UX footgun is worth smoothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed/empty regenerate does not silently strip the anchor's prior good answer from the next send's context without a clear, discoverable recovery affordance
- [ ] #2 Chosen approach (e.g. auto-restore the anchor as active on failed regenerate, or surface a one-key swipe-back/retry hint on the failed sibling) is documented and unit-covered
- [ ] #3 Verified in the live TUI: regenerate → force a failure → confirm the good answer is recoverable in one obvious step
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

ADR required: no

ADR path: N/A

Reason: this is a routine recovery correction within the existing active-leaf and sibling-branch contracts. It changes no schema, persistence boundary, service contract, security policy, dependency, or long-lived application structure.

1. Pin transport-failure and empty-stream recovery with red controller tests that assert the original answer returns to the active path and provider context while the failed sibling remains stored.
2. Add the minimal `regenerate_message` postcondition using the existing `set_active_leaf` contract; retain current successful and stopped-regenerate behavior.
3. Exercise the real mounted Console regenerate action with a controlled post-validation provider failure and assert automatic recovery.
4. Update the branching guide, run focused tests and static checks, self-review the diff, then record implementation evidence and close the task.

Detailed TDD steps: `Docs/superpowers/plans/2026-08-27-console-failed-regenerate-auto-restore.md`

<!-- SECTION:PLAN:END -->
