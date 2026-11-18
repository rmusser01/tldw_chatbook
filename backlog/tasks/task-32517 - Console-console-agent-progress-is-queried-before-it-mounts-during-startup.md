---
id: TASK-32517
title: 'Console: ''#console-agent-progress'' is queried before it mounts during startup'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 00:13'
updated_date: '2026-09-29 19:28'
labels:
  - console
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During CI's warm-up for
`test_library_opens_within_budget_on_a_seeded_profile`, the Console left
rail's `_sync_progress_count` (`tldw_chatbook/UI/Console_Modules/left_rail.py`)
raised a Textual `NoMatches` for `#console-agent-progress`. Cause INFERRED,
not proven: either the 0.5 s count sync fires before the progress widget has
mounted on a cold start, or it keeps firing after the widget is torn down —
`left_rail.py` guards both the compose and the sync with the same
`_open_agent_progress` flag, so a query after unmount is at least as likely
as one before mount; the fix has to establish which. The test recovered on
its own, so today this is a startup-log error rather than a failure, but a
sync that queries a widget that is not there is an ordering bug and will
surface as a hang or a crash the day the handler stops swallowing it.

Evidence: the non-required "UI latency" CI job of PR #2654's run (wave-3
backlinks-table, 2026-09-13), red on
`test_library_opens_within_budget_on_a_seeded_profile` with this NoMatches
during the test's Chat warm-up — not a budget overrun (branch 7.15 s vs dev
8.90 s, both under 10 s). Recorded by the T11 landing pass; carried here by
the wave-3 docs sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh startup and descendant replacement survive an absent progress button without a progress-query NoMatches
- [x] #2 The label reflects queued progress for the active native session and navigation counts refresh while the label is absent
- [x] #3 The current seeded Library warm-up regression completes without the historical progress-query exception
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile the historical startup hypothesis against the actual mount, descendant replacement and timer-unmount flow.
2. Verify the shared missing-label defect through the mounted rail regression while retaining cold full-app and CI warm-up evidence as separate pending checks.
3. Record the queued-progress semantics and evidence without claiming historical CI qualification.
ADR required: no
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: routine bug reconciliation; existing queued-progress ownership and UI contracts apply.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the historical ordering hypothesis with the shared rail timer flow. on_mount performs an initial sync and arms the 0.5s timer; a mounted rail can lose its progress button during descendant replacement; on_unmount already stops the timer. The incoming NoMatches guard avoided the crash but skipped the progress snapshot, count comparison and navigation refresh in that window. The shared callback now skips only the absent label write.

Fresh minimal-host evidence: startup displays Progress: 3 queued; a callback with its button removed updates cached/navigation counts to 5; recompose paints Progress: 5 queued; rail removal stops further navigation updates. Current-guard RED and historical unguarded NoMatches RED were both observed before the fix. Targeted rail and recovery timer checks: 2 passed in 18.87s. The exact seeded Library warm-up node passes locally: 1 passed in 49.48s, using its private_profile_test harness and /private/tmp/tldw-progress-timer-library-green.

The original AC2 says active agent runs, whereas ConsoleAgentController.progress_state and ADR-136 expose queued progress for the active native session. This minimal-host check does not qualify that original cold full-app count criterion; the local warm-up is not a CI run. All criteria remain pending independent review/reconciliation.
ADR required: no; existing ADR-136 and ADR-150 apply. Shared modified files: tldw_chatbook/UI/Console_Modules/left_rail.py and Tests/UI/test_console_rail_progress_timer.py. Test lint/format and source fatal-rule checks pass; full source lint/format retain confirmed HEAD debt.

Acceptance criteria reconciled on 2026-09-29: ADR-136 defines queued progress rather than active-run count. The exact 2026-09-13 CI execution cannot be rerun retroactively; current mounted startup/recomposition and exact seeded Library warm-up regressions are the reproducible evidence. Original incident retained in Description.

Final disposition 2026-09-29: independent read-only implementation review approved the scoped repair with no actionable findings. The targeted acceptance checks and changed-line static checks recorded above pass; inherited whole-file lint/format debt remains outside this correctness task. All acceptance criteria are checked and this task is Done. No full-suite or live-provider qualification is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
