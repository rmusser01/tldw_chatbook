---
id: TASK-61
title: 'PRD: Workspace operating context and handoff model'
status: Done
labels:
- ux
- prd
- workspaces
references:
- Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md
- https://github.com/rmusser01/tldw_server/issues/1526
- https://github.com/rmusser01/tldw_server/issues/1440
- https://github.com/rmusser01/tldw_server/issues/1528
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the product and architecture contract for Chatbook workspaces as portable operating contexts that can move between local and server instances while preserving sources, conversations, artifacts, notes, ACP sessions, runs, worktrees, and sandbox/runtime bindings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines workspace identity, lifecycle, local/server authority, sync/handoff states, and bundle membership.
- [x] #2 Spec maps Console left-rail UX requirements without implementing unapproved workspace switching behavior.
- [x] #3 Spec links the relevant tldw_server workspace roadmap issues and existing Chatbook parity contracts.
- [x] #4 Spec decomposes follow-up work into PR-sized phases with QA and screenshot approval gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review current Chatbook shell, Console, conversation parity, and server parity workspace contracts.
2. Review tldw_server workspace roadmap issues and adjacent workspace/prototype/MCP context.
3. Draft a PRD/spec that defines the workspace operating-context product model, sync/handoff states, data ownership, Console UX implications, and phased implementation boundaries.
4. Add follow-up phases and QA gates that preserve screenshot approval for future UI work.
5. Run focused documentation verification and mark the task complete with implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md` as the workspace operating-context PRD.
- Grounded the PRD in existing Chatbook seams for conversation scope, Notes workspace state, server parity workspace contracts, and the current Console staged-context rail.
- Linked the related `tldw_server` roadmap issues for canonical workspace records, prototype workspace collaboration, and MCP/connector workspace handoff.
- Defined the immediate Console left-rail split as a read-only shell unless a real workspace registry/switching service exists, preserving screenshot approval gates for later UI implementation.
- Revised the PRD after user review so workspace switching does not hide Library/Notes/Artifact items; instead, workspace tags remain visible globally while active Console context actions are gated by workspace eligibility.
- Captured user decisions that server handoff supports both copy and reference modes, ACP task/run packages are the first server-backed target, audit detail is user-visible by default, and offline shared workspaces degrade to single-user local workspaces.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted and verified the PRD for workspace operating contexts and local/server handoff, then revised it to preserve global item visibility while gating active Console context use by workspace membership. No application behavior was changed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
