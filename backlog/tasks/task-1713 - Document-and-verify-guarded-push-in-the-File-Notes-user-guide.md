---
id: TASK-1713
title: Document and verify guarded push in the File Notes user guide
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 09:05'
updated_date: '2026-08-01 09:20'
labels:
  - file-notes
  - docs
  - git
  - uat
dependencies:
  - TASK-1711
documentation:
  - Docs/User_Guide/library/file-notes.md
  - Docs/superpowers/specs/2026-07-30-file-notes-guarded-session-push-design.md
  - backlog/decisions/039-file-notes-guarded-session-push.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the File Notes user guide into line with the shipped guarded exact-session push workflow so users can safely review and push a prepared session and understand typed outcomes and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The File Notes overview and Session Git section describe guarded push as part of the shipped workflow
- [x] #2 The common task gives the exact review authorization and push sequence without implying unrelated changes are included
- [x] #3 Success refusal uncertain-outcome and query-only recovery states are accurately documented
- [x] #4 The guide preserves the explicit boundary that Chatbook does not provide pull fetch merge rebase remote management or credential setup
- [x] #5 A focused verification confirms the documented labels and flow match the shipped implementation and the guide records the verified dev revision
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit the current user guide against TASK-1711, ADR-039, the guarded-push implementation, and focused tests.
2. Update only Docs/User_Guide/library/file-notes.md with the shipped review, authorization, push, outcome, recovery, and scope-boundary guidance.
3. Run focused documentation and guarded-push verification; do not run the full application suite for this documentation-only change.
4. Self-review the diff, check every acceptance criterion, record concise implementation notes, and mark the task Done.

ADR required: no
ADR path: N/A
Reason: This task documents and verifies the existing ADR-039 decision without changing architecture, storage, sync, Git policy, or application behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the File Notes overview, layout, Session Git workflow, common task, keyboard behavior, and troubleshooting guidance for the shipped guarded exact-commit push. The guide now covers pre-contact destination authorization, local and remote check phases, immutable destination/lease review, safe focus and cancellation behavior, same-operation reattachment, typed results, query-only uncertain recovery, process-only attribution, secure transport/platform limits, and the explicit external-Git boundary.

Modified files: Docs/User_Guide/library/file-notes.md and this Backlog task record. The existing overview SVG remains accurate because it depicts the unchanged pre-link File Notes workspace rather than the Session Git workflow.

Verification against origin/dev 949e2ef73: Tests/UI/test_library_file_notes_git_push.py passed 59 tests (one pre-existing requests dependency warning); the focused guide check found 21/21 shipped labels, zero broken local links, 11/11 scope/recovery boundaries, a matching dev stamp, no placeholders, and one unique TASK-1713 ID; git diff --check passed. Independent guarded-push documentation review returned APPROVED.

ADR required: no. This documentation-only closeout implements no architecture or behavior change and remains governed by ADR-039.
<!-- SECTION:NOTES:END -->
