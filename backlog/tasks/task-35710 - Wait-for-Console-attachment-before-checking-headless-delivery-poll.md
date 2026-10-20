---
id: TASK-35710
title: Wait for Console attachment before checking headless delivery poll
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20'
labels:
  - ci
  - tests
priority: high
dependencies: []
---

## Description

Qualify the headless-delivery poll check against a completed Console visit so asynchronous attachment does not produce an early false failure. Keep the idle control and active-delivery regression meaningful.

PR #3176's admission-sensitive job failed on unchanged Console paths with `reconciled=False`, an active delivery and the same controller: [job log](https://github.com/rmusser01/tldw_chatbook/actions/runs/39375455061/job/118472787475). The test had waited for the composer and a message-pump pause, which do not witness completion of the full attachment reconciliation. The exact unchanged node passed locally in 66.24 s; this is not a reproduced local red.

[TASK-34789](./task-34789%20-%20Flaky-tests-in-the-required-check-evict-good-PRs-from-the-merge-queue.md) is completed historical flake measurement, including earlier failures of this test with different signatures. This follow-up owns only the new early-assertion signature. [TASK-32873](./task-32873%20-%20Fix-five-red-test_console_runtime_ownership-tests-on-dev-fleet-wake-rename-tombstone-order.md) covers older fixture/API drift. No production polling redesign is included; that remains under [ADR-231](../decisions/231-console-polling-and-full-state-reconciliation.md). This qualifies CI for [TASK-35680](./task-35680%20-%20Keep-parallel-run-form-refreshes-out-of-Settings-drafts.md).

## Acceptance Criteria

- [x] #1 The test observes successful current-visit attachment before checking both the idle no-poll control and active-delivery poll, including the initial fixture visit.
- [x] #2 Attachment that never completes and a completed active-delivery visit that never arms its poll still fail; the original poll assertions, navigation, wake precondition, profile guard and timeout limits remain intact.
- [ ] #3 Targeted Console attachment/poll checks and static checks pass, independent review assesses the correction, and the original CI failure plus qualification limits are recorded.

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: test-only completion barrier for an existing attachment contract; no runtime, polling, ownership or timing policy change. Existing [ADR-231](../decisions/231-console-polling-and-full-state-reconciliation.md) and [ADR-103](../decisions/103-fast-pr-lane-and-required-gate-aggregation.md) remain applicable.

1. Read the failed job to its assertion, compare the paths against dev and inspect the existing successful attachment waits. Keep the observed hosted red separate from the passing local rerun and earlier timeout signatures.
2. Use a bounded condition wait for the actual attachment-complete flag at the three visits, matching the neighboring production-startup test's existing 50 polls at 0.1-second intervals. Preserve the idle and active poll assertions and leave product code unchanged.
3. Qualify the corrected real-navigation test and neighboring attachment/poll controls. Verify negative controls for incomplete attachment and missing poll without weakening assertions or increasing existing timeouts. Run targeted lint/format and Backlog guards, request independent delta review, and update PR #3176 before checking its new-head CI.

## Implementation Notes

The navigation test now waits for the existing attachment-complete flag and current screen identity before inspecting post-attachment polling on the initial, idle-return and delivery-return visits. It follows the neighboring real-startup test's bounded polling pattern and adds attachment-phase diagnostics. Both original poll assertions, the real navigation and wake precondition, and the 180-second test limit remain unchanged. No product code or runtime/polling policy changed; no new ADR is required.

The original hosted failure is recorded in [TASK-35680 qualification](../../Docs/superpowers/qa/2026-10-19-task-35680-qualification.md). Eight targeted ownership checks passed in 93.02 s. Disposable negative controls failed at the intended assertions: incomplete attachment in 56.52 s, and completed attachment with poll arming disabled in 53.19 s (the latter reported reconciled=True). These controls were not committed. Ruff lint and formatting of the modified function pass; inherited whole-file formatter differences are unchanged. Both Backlog guards and diff whitespace checks pass. The missing Backlog CLI required direct task Markdown updates, as in the preceding PR tasks.

Modified the ownership test, this task, the qualification note and the testing-evidence lesson. Review and new-head CI are recorded separately before publication/completion; no local reproduction of the hosted early-assertion red or particular native deferral cause is claimed.
