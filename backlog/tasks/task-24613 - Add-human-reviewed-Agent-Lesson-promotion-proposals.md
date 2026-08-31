---
id: TASK-24613
title: Add human-reviewed Agent Lesson promotion proposals
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 01:17'
updated_date: '2026-08-31 02:45'
labels:
  - notes
  - agents
  - security
dependencies:
  - TASK-24309
documentation:
  - >-
    Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-agent-lesson-promotion.md
  - backlog/decisions/106-human-reviewed-agent-lesson-promotion.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn verified Agent Lessons into small reviewable proposals for authorized user-owned instructions while keeping lesson content untrusted and every application human-controlled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A foreground primary can present one exact promotion proposal from independently verified lesson evidence
- [x] #2 A repository-instruction proposal identifies the writable binding, target path, current effective instruction chain, exact resulting content, and current target-state precondition before asking for approval
- [x] #3 Repository instruction application uses one existing file-mutation seam with an atomic expected-digest or expected-absent check at the write boundary, so stale proposals fail without mutation and unrelated user edits are preserved
- [x] #4 Chatbook-managed local skills remain proposal-only in Console; the user manually applies an approved proposal through the existing Library skill editor/service and its version plus re-trust boundary
- [x] #5 Subagents can return evidence and candidate text but cannot present an approval request or apply a promotion change
- [x] #6 Ineligible targets, missing authority, changed bindings, changed effective instruction chains, and stale content fail without mutation and report a non-sensitive reason
- [x] #7 Promotion outcomes are recorded only through separately approved ordinary Agent Lesson Note updates and never authorize later writes
- [x] #8 Targeted deterministic tests and scripted behavioral evaluations cover proposal quality, exact review, stale-state refusal, role enforcement, and the untrusted-data boundary
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/106-human-reviewed-agent-lesson-promotion.md
Reason: ADR-106 already fixes the untrusted Notes, file-authority, approval, project-instruction, and managed-skill trust boundaries; this task directly implements it.

1. Define immutable promotion evidence, eligibility, proposal, and digest contracts with focused tests.
2. Add target snapshot/revalidation and generic atomic fs_write compare-and-swap with deterministic race tests.
3. Compose primary-only preparation and application approval gates with exact run-bound proposals.
4. Keep managed local skills proposal-only through the existing Library editor/service and re-trust flow.
5. Add capability-aware guidance, end-to-end/outcome coverage, documentation, targeted verification, and live disposable-binding verification.

Detailed execution checklist: Docs/superpowers/plans/2026-08-29-agent-lesson-promotion.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented immutable evidence/proposal contracts, capability-aware trusted
guidance, primary-only exact preparation/application reviews, run-bound
single-use stamps, project-instruction snapshot revalidation, and generic
atomic `fs_write` dry-run/CAS behavior. Repository targets remain limited to
the accepted writable binding; admitted-root `root_alias` routing is included
in exact review while routing-only data is stripped before filesystem
execution. Managed local skills expose a reviewed read-only proposal action
only to the primary and still require manual Library editing, version checks,
and re-trust.

Integrated the latest `origin/dev` before closure. The merge preserved dev's
shipped ChaChaNotes v55→v57 semantic-trace migrations and appended the Notes
organization/Agent Lessons chain at v57→v61; it also preserved token-disclosure
ADR-104 and moved the unshipped promotion ADR to ADR-106. Test fixtures were
updated to use latest dev's model-relative progressive-disclosure plan.

Verification used real disposable filesystem and SQLite fixtures plus mounted
Textual approval/editor tests; no external server was required. Targeted
verification passed 1,218 tests across promotion, file authority, project
instructions, AgentService/runtime, catalog/skills/fleet, UI, migrations, and
index census. `compileall`, feature-scoped Ruff, and `git diff --check` passed.
The full suite was not run, per repository policy. A broad branch-diff Ruff
sweep still reports inherited baseline findings in legacy semicolon-formatted
tests and `app.py` import ordering; none are in TASK-24613's scoped files.

ADR required: yes

ADR path: `backlog/decisions/106-human-reviewed-agent-lesson-promotion.md`

Files: Agent proposal/runtime/provider/project-instruction and generic local
write modules; Console review/bridge composition; focused Agent, Chat, Tools,
Skills, UI, and migration tests; Agent/Notes/Skills user guides and ADR-106.
<!-- SECTION:NOTES:END -->
