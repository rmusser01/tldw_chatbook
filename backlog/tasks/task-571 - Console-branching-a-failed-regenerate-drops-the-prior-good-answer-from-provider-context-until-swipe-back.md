---
id: TASK-571
title: >-
  Console branching: a failed regenerate drops the prior good answer from
  provider context until swipe-back
status: Done
assignee:
  - '@codex'
created_date: '2026-07-25'
updated_date: '2026-08-28 06:25'
labels:
  - console
  - chat
  - ux
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-27-console-failed-regenerate-auto-restore-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(Re-filed from task-499 in the PR #803 rework, unchanged in substance.)

With Console branching (Phase A, PR #799), regenerate forks a new empty sibling assistant node and moves the active leaf onto it. If that regenerate stream fails or returns empty, the new sibling ends `failed` (there is no variant base to restore, since `variant_mode=False`), so `_provider_messages_for_session(skip_failed=True)` excludes it — and the original good answer (the anchor) is now off the active path, so it is excluded too. The model therefore loses the previously-good answer from context until the user swipes back to the anchor or retries the failed node. This is a deliberate consequence of the node model and is recoverable/visible (the failed node is shown, and swipe + retry-on-failed-sibling are the recovery paths), but the UX footgun is worth smoothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed/empty regenerate does not silently strip the anchor's prior good answer from the next send's context without a clear, discoverable recovery affordance
- [x] #2 Chosen approach (e.g. auto-restore the anchor as active on failed regenerate, or surface a one-key swipe-back/retry hint on the failed sibling) is documented and unit-covered
- [x] #3 Verified in the live TUI: regenerate → force a failure → confirm the good answer is recoverable in one obvious step
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

- After stream settlement, `regenerate_message` now restores the selected original assistant anchor only when the replacement sibling settled as `failed`, using the existing `set_active_leaf` contract. The failed sibling remains stored and retryable; successful and intentionally stopped branches remain active.
- Restoring an older mid-conversation anchor makes that anchor the active leaf. Its former descendant tail remains stored off-path, while provider context ends at and includes the restored good answer.
- Focused coverage exercises transport failure, empty output, the mid-conversation boundary, success and stop controls, provider-context projection, and the mounted real regenerate-button action. The mounted test gates the failure, observes the pending replacement, waits for the exact session worker, and relies on the production final repaint before asserting the restored row.
- Updated the Console branching user guide to explain automatic failed/empty recovery, retained failed siblings, the older-anchor boundary, and stopped-branch behavior.
- ADR required: no. ADR path: N/A. This is a bounded recovery correction within existing controller, active-leaf, and sibling-branch contracts; it changes no schema, persistence boundary, service contract, security policy, dependency, or long-lived application structure.
- Fresh post-formatting verification: focused pytest completed with `29 passed, 2 warnings in 10.03s`; warnings were the existing `RequestsDependencyWarning` for the installed urllib3/chardet/charset-normalizer versions and Python 3.12's `audioop` deprecation from pydub. Ruff lint reported `All checks passed!`, and `git diff --check origin/dev...HEAD` exited 0.
- Scoped Ruff formatter checks exited 0 for every TASK-571-owned region: controller lines `11111-11290`, branching failure tests `143-214`, and variant recovery/stop controls `365-467`. All TASK-571-owned hunks in `Tests/UI/test_console_regenerate_feedback.py` (module docstring/support additions, failing gateway, bounded wait helper, and mounted recovery test) are formatter-clean. The full four-file formatter check remains read-only exit 1 (`test_console_variant_stream.py` is already formatted; the controller, branching, and mounted files report drift), but every remaining hunk is outside these owned regions and is present in the `origin/dev` version: eleven controller hunks, the branching persistence assertion at current line 274, and the named pre-existing mounted helper/retry hunks at current lines 58 and 218. Formatting each `origin/dev` file through Ruff on stdin reproduced those baseline hunks, so unrelated code was not reformatted.

<!-- SECTION:NOTES:END -->
