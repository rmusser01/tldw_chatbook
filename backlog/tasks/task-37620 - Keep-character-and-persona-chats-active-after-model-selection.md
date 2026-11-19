---
id: TASK-37620
title: Keep character and persona chats active after model selection
status: Done
assignee:
  - '@codex'
created_date: '2026-10-29T20:04:00Z'
updated_date: '2026-10-29T20:41:00Z'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Starting a character or persona chat from a named workspace can leave the active workspace registry pointing at the previous chat. Selecting a model resumes Console and switches away from the originating conversation. Keep the selected conversation and its workspace authority aligned across model-picker Apply and Cancel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starting or reusing a character or persona conversation aligns the active workspace with that conversation without changing its scope.
- [x] #2 Model-picker Apply and Cancel preserve the originating conversation, transcript, assistant identity, and session count after starting from a named workspace.
- [x] #3 Model Apply updates only the originating conversation and existing cross-screen workspace changes still reconcile normally.
- [x] #4 Targeted regressions, relevant static checks, architecture documentation, and review are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for roleplay activation from a named workspace and model-picker return. 2. Reuse the existing workspace-alignment helper in all affected character/persona activation paths through named callbacks. 3. Run targeted tests and static checks, update architecture and task evidence, review, and open a PR against dev. ADR required: no. ADR path: backlog/decisions/033-application-session-state-ownership.md and backlog/decisions/095-conversation-owned-console-generation-settings.md. Reason: repair existing activation ownership without changing storage, scope, or reconciliation policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the existing workspace registry at character-picker and character/persona handoff activation before publishing the selected chat. Kept existing session switches and notification epochs; pristine named-workspace reuse and genuine cross-screen reconciliation retain their existing behavior. Wired one required named character-controller callback to the existing workspace owner.

Added eight activation/model-picker regressions (six real mouse Apply/Cancel paths and two pristine handoffs), registered them in the PR census, updated direct constructor builders and repaired stale persona visual store fixtures. Updated Console architecture and incident documentation. ADR required: no; applies existing ADR-033 and ADR-095.

Validation: all eight new regressions passed together; focused run passed40 and exposed one final stale fixture, then the complete persona visual file plus isolated avatar sync passed12 after its correction. Related session/model-apply checks passed44; both module-size guards passed at6990 lines. Ruff (excluding unchanged baseline E721), formatter, UI census1152, Backlog ID/file guards and diff checks pass. Independent whole-diff review approved with no remaining findings.

Broader diagnostics: two bare-screen architecture failures reproduce on unchanged dev; a wider avatar run had order/load-sensitive portrait/native-resource failures, while the portrait test passes in isolation on base and branch. No full-suite claim. PR integration and final merge are tracked by the accompanying implementation plan.
<!-- SECTION:NOTES:END -->
