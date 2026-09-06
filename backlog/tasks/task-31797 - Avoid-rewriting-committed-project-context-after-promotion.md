---
id: TASK-31797
title: Avoid rewriting committed project context after promotion
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 00:53'
labels:
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Temporary-chat promotion already commits project controls inside its atomic
bundle, then unnecessarily writes the same local state again after publication.
Keep a single authoritative transaction and preserve rollback on bundle failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Promotion persists project context exactly once inside the atomic bundle.
- [x] #2 Durable state survives reopen and bundle failure retains complete rollback.
- [x] #3 Complete project-context and atomic-promotion tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced transaction trace [True, False] and inspect both write owners.
2. Remove only the duplicate postcommit project-state write; retain ordinary first
   persistence and explicit later state changes, and staged context-policy flushing.
3. Verify the original transaction assertion, durable reopen and rollback tests;
   run full affected files and independent review before saving the checkpoint.

ADR required: no
ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md; backlog/decisions/079-console-library-conversation-authority.md
Reason: Routine removal of a duplicate write; existing local-only control state
and atomic publication contracts remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed only the duplicate _persist_project_instruction_state after atomic temporary-chat promotion. The bundle already writes encoded controls before commit; ordinary first persistence, explicit setters, scope callback handling and staged context-policy flushing remain unchanged. Preserved original exact [True] transaction assertion (baseline was [True,False]) and added actual database reopen evidence plus failure-safe test cleanup. Existing complete-bundle rollback test remains. Five complete affected files:143 passed,2 existing dependency warnings in28.90s, XML:/private/tmp/tldw-offloop-promotion-final.xml. Whole changed-file Ruff, test formatting, changed Store range formatting and diff checks pass; independent review clear. ADR required:no; existing ADR-069 local controls and ADR-079 atomic publication contracts retained. The separate fork census still has3 reproduced failures and was not relaxed.
<!-- SECTION:NOTES:END -->
