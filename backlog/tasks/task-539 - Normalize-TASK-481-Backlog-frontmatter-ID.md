---
id: TASK-539
title: Normalize TASK-481 Backlog frontmatter ID
status: Done
assignee: []
created_date: '2026-07-24 21:28'
updated_date: '2026-07-24 21:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore TASK-481's canonical uppercase frontmatter identifier so Backlog discovery and repository task-hygiene checks can recognize the existing task without altering its completed work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TASK-481 frontmatter contains exactly `id: TASK-481` and is discoverable by Backlog and the task-ID harness
- [x] #2 No duplicate task ID is introduced and TASK-481's status, content, and acceptance criteria remain otherwise unchanged
- [x] #3 Targeted frontmatter-parser verification, Backlog inspection, and diff validation pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the lowercase TASK-481 frontmatter identifier causes its Backlog-discovery and task-ID parsing failure.
2. Normalize only that identifier to the canonical uppercase TASK-481 form, preserving all other task content.
3. Verify Backlog inspection, targeted task-ID parsing, duplicate-ID safety for TASK-481, and diff hygiene.
4. Request independent review before documenting and closing the task.

ADR required: no
ADR path: N/A
Reason: This is a one-line correction to existing Backlog metadata and does not introduce an architectural decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Normalized TASK-481’s sole malformed frontmatter field from `id: task-481` to `id: TASK-481`; all other TASK-481 bytes, status, and acceptance criteria are unchanged.
- Verification: `backlog task 481 --plain` resolves the completed task; the product-maturity harness parser maps `TASK-481` to exactly one file; `git diff --check` passes.
- The repository-wide uniqueness test now proceeds past TASK-481 and exposes a separate pre-existing collision between the citation-provenance epic and the completed response-prefill task, both claiming TASK-401. That collision is reserved for an independent Backlog-identity remediation and is not caused by this change.
- Independent review reproduced the targeted checks, verified byte-for-byte preservation outside the identifier, and approved atomic closeout with no actionable findings.
- ADR required: no; ADR path: N/A; reason: correction of existing task metadata only, with no architecture or runtime change.
<!-- SECTION:NOTES:END -->
