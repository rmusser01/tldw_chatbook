---
id: TASK-33014
title: Persist archive chronology and workspace lifecycle revisions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 15:19'
updated_date: '2026-09-27 16:00'
labels: []
dependencies:
  - TASK-32772
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make archive dates accurate and stale workspace recovery detectable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both stores record real archive times and preserve unknown historical dates.
- [x] #2 Workspace lifecycle changes use atomic archive-revision checks, including existing UI and Undo paths.
- [x] #3 Migration and real SQLite tests preserve conversation version and sync invariants.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md. Reason: implement approved local chronology and workspace CAS contracts. Follow Task 1 in Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md: real-store red regression, next schema migrations, all lifecycle writer updates, fresh/upgraded/rollback and sync qualification, scoped static checks and commit. Interpreter: /tmp/archive-recovery-venv/bin/python. Full suite is not authorized.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Persisted local archive chronology with new transactional ChaChaNotes v74 and Workspace v9 migrations; historical archives retain NULL time. Workspace archive/restore and Default repair now increment a monotonic revision and condition writes on captured state/revision/name. Console archive confirmation, Restore as, Undo and Settings archive confirmation/Undo propagate snapshots at the actual storage write boundary (controller-approved correction to Task 1 file scope). Conversation metadata/page projections expose archived_at; archive-only and timestamp-only changes retain existing shared sync events and payloads. IDs, branches, messages, active workspace behavior and refusal semantics remain covered. ADR: backlog/decisions/166-agent-assisted-archive-recovery.md; no additional ADR required. Verification: six targeted files, 164 passed in 27.04s; changed-range Ruff formatting, no introduced Ruff diagnostics against 811 baseline diagnostics, and git diff --check pass. Tests/ChaChaNotesDB/historical_bootstrap.py was audited: current ordinary seed writers do not touch archived_at, so no forward scaffold is needed. Task remains In Progress for controller review. Detailed RED/GREEN evidence: .superpowers/sdd/2026-09-27-agent-archive-recovery/task-1-report.md. The chronology index query plan is qualified on 500 mixed archive/deletion/date records without a temporary sort.

Independent Task 1 review approved spec compliance and code quality with no actionable findings. Controller resolved unchanged-code checks: conversation_archive_actions.py delegates captured versions through ChatConversationService to the updated DB writer; Chatbooks/chatbook_creator.py uses explicit conv_data export fields excluding local archive chronology. Fresh evidence: 164 targeted tests passed, changed-range format/diff checks pass, no introduced Ruff diagnostics. Existing unrelated lint debt is recorded above. ADR-166 governs implementation; Task 1 is complete.
<!-- SECTION:NOTES:END -->
