---
id: TASK-31767
title: Align scripted Console benchmark with admitted-root authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:57'
updated_date: '2026-09-05 17:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the scripted mounted benchmark after the Console moved structured filesystem tools to admitted Workspace bindings and invalidated older permission fingerprints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The mounted sample renews only its fixture-owned fs_write grant against the admitted-root schema and retains fail-closed stale-grant behavior.
- [x] #2 The real composer and queue execute fs_write in the explicitly bound Workspace folder, with no scratch mutation.
- [x] #3 Targeted benchmark and permission regression checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce stale permission fingerprint against the real admitted-root schema and preserve a regression proving it asks before renewal.
2. Align only the scripted mounted sample with its explicitly bound Workspace root; renew the fixture-owned fs_write grant using the exact admitted-root HubTool through the real control plane.
3. Verify real composer/queue execution, Workspace mutation bytes, absent scratch mutation, targeted tests, scoped static checks, and independent review.
ADR required: no
ADR path: backlog/decisions/102-console-run-admitted-local-path-authority.md (existing)
Reason: the fixture follows existing ADR-102 admitted-root authority and ADR-069 selected-root rules without changing runtime permission policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned only the scripted mounted sample with ADR-102 using canonical run-admitted roots for its explicit writable Workspace binding. The fixture renews its exact fs_write descriptor through the real control plane, verifies the Workspace mutation, and rejects a scratch fallback. Tests prove the old schema grant becomes Ask/config_changed, renewal leaves fs_edit unapproved, and unrelated sessions are rejected. Historical cross-revision runtime fixtures remain unchanged.

Verification: the two new authority tests reproduced RED before implementation, then passed. In an isolated installed Python 3.12 environment, 10 targeted benchmark/workspace/authority tests passed, including real composer entry, queued third send, subprocess fs_write, mutation bytes, and absence of scratch mutation. Full-file Ruff checks, changed-region formatting checks, and git diff --check passed; existing whole-file formatting drift was preserved. The shared environment cannot import this checkout under python -I, so subprocess evidence uses an isolated editable installation with the existing dependencies; shared environment unchanged. Self-review completed. Recorded the actual schema/isolated-worker incidents in lessons-testing-evidence.md.

ADR required: no; existing backlog/decisions/102-console-run-admitted-local-path-authority.md governs the fixture correction. Changed files: Performance runner and its tests, this task, and testing-evidence lessons.
<!-- SECTION:NOTES:END -->
