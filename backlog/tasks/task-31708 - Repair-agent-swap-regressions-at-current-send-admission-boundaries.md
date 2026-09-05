---
id: TASK-31708
title: Repair agent swap regressions at current send admission boundaries
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:50'
updated_date: '2026-09-05 19:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful citation repair and tool authority identity evidence through the current controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Citation repair fallback, regeneration and exact shared authority assertions execute their intended boundaries;Complete agent swap file and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update SignalGateway to the explicit current route/capture signature and gate doubles to profile_id, retaining exact route/profile and shared-identity assertions. 2. Strengthen the reproduced agent-regenerate failure test for empty and partial replies, one visible error notice, restored provider context and retained failed sibling. 3. Repair the existing TASK-571 rehoming guard to use the owning session failure state, not the agent result partial-answer text; keep the result contract unchanged. 4. Run complete affected files, scoped static checks, and self-review. ADR required: no. ADR path: N/A. Reason: routine recovery correction within the existing TASK-571 active-leaf contract, no new runtime/storage/security boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the real agent-regenerate notice loss: rehome the matching SYSTEM failure row using the owning session terminal failure copy, not ConsoleSubmitResult.visible_copy (which intentionally retains the partial assistant answer). Kept existing active-leaf restoration and result contracts. Gateway and builtin-gate fixtures now accept and assert current route/capture/profile seams; empty and partial failed siblings retain the original answer/model context and exactly one transcript-only error notice. Existing empty-node persistence deferral remains unchanged. Fresh combined six complete files:376 passed in48.70s, XML /private/tmp/tldw-review-agent-controller-regeneration-verified.xml. Includes full agent-swap, controller, branching, variant-stream, provider-failure and generation-action files. Test Ruff lint/full formatting pass; controller changed region format and whitespace pass. Full controller lint has the same27 pre-existing findings in HEAD and current file, none in the repaired region. Resource warning248 above limit200 is separately attributed to test-owned SQLite lifecycle and remains under investigation, not hidden. ADR required:no; N/A, routine existing TASK571 recovery correction. Independent scoped review pending before commit.

Independent scoped review found no actionable issue in the owning-session rehome guard, current route/profile fixtures, or restored-branch assertions. Reviewed runtime guard preserves the existing SYSTEM-role and exact-copy checks. Functional repair complete; fixture resource lifecycle remains separately tracked.
<!-- SECTION:NOTES:END -->
