---
id: TASK-32194
title: Start native goal setup from the Console composer
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 01:06'
updated_date: '2026-09-10 01:22'
labels:
  - console
  - agents
dependencies:
  - TASK-32120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users can enter /goal followed by a task description in the composer to start the existing bounded native goal workflow, with the same launch review and authority controls as the palette entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing /goal with a task description opens native goal setup with the complete description prefilled, including multiline text; bare /goal opens the ordinary setup form.
- [x] #2 The slash popup discovers and completes /goal, and command dispatch consumes it without sending the command as an ordinary model message; existing literal/paste behavior is preserved.
- [x] #3 Goal enablement, selected workspace/provider resources, finite limits and explicit launch review remain authoritative; no goal work dispatches before Start confirmation.
- [x] #4 Cancellation and setup refusal preserve the composer draft, and confirmed launch persists the described objective through the existing goal service.
- [x] #5 Targeted composer/setup regressions and user documentation cover the new entry point.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/141-native-console-goal-runs.md (existing accepted decision)
Reason: A composer entry point reuses the accepted native goal setup, runtime, permissions and storage; no new execution or persistence boundary.

1. Extend the existing command registry and popup description for /goal, retaining ordinary command and paste gating.
2. Pass the complete command description to existing goal setup as its initial objective; retain launch review, finite policy, refusal behavior and the ordinary composer draft.
3. Verify mounted composer-to-form and confirmed service behavior, disabled/cancel paths, and parser/popup regressions with targeted tests. No live provider call or full-suite sweep is needed for this entry point.
4. Update the Console guide, perform scoped lint/format and review, close this task, and include the change in the requested draft PR against dev. The preserved branch has divergent prerequisites that remain an integration dependency.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added /goal [task description] to the existing Console command registry, popup and dispatcher. The complete description prepopulates existing native goal setup; bare /goal retains ordinary setup. Existing provider/resource selection, enablement, finite policy and explicit Review/Start remain authoritative. Cancel and refusal preserve the draft, and recognized /goal commands never become ordinary chat sends. No new storage or runtime ownership.

Verification: 118 affected composer/parser/popup/goal setup/navigation tests passed, followed by 38 overlapping final tests after mechanical lint cleanup. Six-file Ruff and format pass; the existing ChatScreen owner has 150 baseline/current diagnostics with no additions. Whitespace checks pass. Independent scoped review approved with no actionable findings. The initial two mounted RED failures reproduced missing popup/dispatch support; an intermediate test-fixture error was corrected by inspecting the saved goal through service.get rather than its body-free history entry. Existing RequestsDependencyWarning remains. No full-suite sweep or live model call.

ADR required: no new ADR. Existing backlog/decisions/141-native-console-goal-runs.md governs the reused launch boundary. User documentation is updated in Docs/User_Guide/console/agent-runs-and-tools.md. Exact gate output, independent review and limits are retained in Docs/superpowers/qa/native-goals/composer/README.md and linked from the original qualification report. The requested single draft PR against dev includes all session changes; full-branch prerequisite integration remains separate from this bounded entry-point approval.

Identifier reconciliation: initially filed as TASK-32193; the pre-PR scan found a concurrent committed shared-search task using that number, so this new task was renumbered to TASK-32194 before its first commit. The independent review retains the former ID as history; code and test evidence are unchanged. ADR-141 also conflicts with a later Library retirement ADR; our goal ADR was introduced first in 77bc58dc17 (2026-09-08), while the Library ADR first appears in 5caa634c2d (2026-09-09). Preserve the earlier goal ADR and disclose the remaining cross-branch conflict for integration.
<!-- SECTION:NOTES:END -->
