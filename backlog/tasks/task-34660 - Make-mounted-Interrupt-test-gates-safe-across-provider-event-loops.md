---
id: TASK-34660
title: Make-mounted-Interrupt-test-gates-safe-across-provider-event-loops
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-07 14:30'
updated_date: '2026-10-07 18:27'
labels: []
dependencies: []
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the retained mounted Interrupt teardown finding with causal evidence for its controlled provider gate and proof of physical worker retirement. Preserve the real mounted Stop entry, hooks, authority and existing deadlines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original mounted Interrupt teardown failure and its causal call path are reproduced and retained.
- [x] #2 The mounted test completes body and executor teardown with real provider-loop work settled, preserving all original Interrupt and seal assertions and deadlines.
- [x] #3 Affected viewless and mounted Interrupt controls remain passing without skips, worker cancellation substitutes or warning suppression.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the unchanged mounted node and trace its provider loop and stranded bridge future.
2. Replace only a proven unsafe test gate with the existing thread-safe asynchronous event-wait pattern; keep real runtime and hook execution.
3. Verify the original node including physical executor exit and affected module, then static/Backlog guards and scoped review.

ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
Reason: a test-only synchronization repair preserves existing production turn lifetime; a discovered production ownership change requires its own reviewed design before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented only thread-safe scheduling of the mounted test provider gates on their owning event loops. Original body passed but executor teardown failed at the unchanged 300-second bound (1 passed/1 error in 300.63s). Transparent original-set loop-wakeup control observed both wrong-loop calls and passed in 10.27s; its late logs remain retained. Final uninstrumented node passed in 8.52s including process teardown; all 13 affected module tests passed in 83.83s without pytest warnings. Whole-file Ruff/format, whitespace and both Backlog guards pass; self-review confirms original assertions/deadlines and production untouched. ADR094 applies; no new ADR. Evidence: Docs/QA/task-34660/README.md. Integration verification remains pending; no push or merge performed.

Later integration selection preserved all 13 hook teardown controls, including mounted Interrupt, without failures; changed test still passes whole-file Ruff/format. The wider aggregate remains non-green on four other nodes; this causal gate repair does not establish aggregate SQLite/FD retirement. Source/evidence preserved locally; no fresh-context review, push or merge.

Final original hook teardown module passes with strict AgentRuns ownership repair: 13 passed under first-acquisition native census in 92.21s and 13 uninstrumented in 101.00s without pytest warnings. Thread-safe gate test source, assertions and deadlines unchanged; whole-file Ruff/format pass. Later real worker-retirement failures and separately approved production fixes are retained under TASK34661, not erased or credited solely to test gates. QA reconciled. Status remains In Progress for review/integration and retained static-gate disposition; no source commit, PR, push, merge or other-session control.
<!-- SECTION:NOTES:END -->
