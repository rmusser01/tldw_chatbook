---
id: TASK-32705
title: Verify Library exact conversation Resume journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 00:24'
updated_date: '2026-09-17 00:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qualify the Library-to-Console exact Resume journey against persisted identity, branch selection and unrelated draft preservation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cold Resume opens the original persisted conversation and older active branch, retaining off-path messages.
- [x] #2 Warm Resume reuses the existing Console session and preserves an unrelated populated draft.
- [x] #3 Both themes and compact/wide sizes receive native private-profile qualification with honest evidence limits.
- [x] #4 Targeted tests, static checks, source preservation and normal process shutdown are recorded.
- [x] #5 QA evidence and the review ledger record findings, scope and the next bounded journey.
- [x] #6 Stale Resume regression fixtures are updated to current production entry points without weakening identity, history or draft assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md (existing); backlog/decisions/086-library-adaptive-reader-shell.md
Reason: verification of the existing recovery and adaptive reader contracts; no new architecture.
1. Inspect Resume owners and existing regression coverage, then run targeted tests.
2. Drive real Library Resume in a guarded native profile using four independent branched fixtures and an unrelated populated draft; cover cold hydration and warm session reuse in both themes and sizes.
3. Check original durable records, process exit, private database integrity and default-profile isolation; inspect captures in one batch.
4. Review the bounded evidence, repair verified defects if needed within updated acceptance criteria, and update the QA ledger/task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified active local Library Resume through eight native actions across dark/light and 170x48/80x24. Cold hydration retains the original conversation, older active branch and newer sibling; warm Resume reuses the same session. An unrelated pasted draft remains exact and painted after tab return. No production code/CSS change.
Repaired stale tests: mounted restore/resume sends through the actual Send button; the warm-consumer census includes exact Resume. Final targeted selection: 83 passed (baseline 80 passed/3 failed); all four changed/helper Python files lint/format clean.
Native run005 returned normally, exited 0 and its PID was independently absent. Read-only checks confirm four unchanged conversations, twelve unchanged messages, no copies, ten healthy private databases, unchanged default-profile files and clean logs. Twelve captures inspected; independent code/evidence review has no outstanding findings. Early harness failures and final scope limits are retained in Docs/superpowers/qa/2026-09-16-conversation-resume/README.md; review ledger updated.
ADR required: no new ADR; existing backlog/decisions/147-conversation-archive-and-exact-resume.md and backlog/decisions/086-library-adaptive-reader-shell.md apply. No full suite, real provider send, push or merge. Native branch switching, archived workspace restore and draft durability after process exit are outside this fixture.
<!-- SECTION:NOTES:END -->
