---
id: TASK-28001
title: Fix saved Console conversations prompting on close
status: Done
assignee: []
created_date: '2026-09-02 04:06'
updated_date: '2026-09-02 04:45'
labels:
  - console
  - ux
  - bug
dependencies: []
references:
  - backlog/decisions/046-visible-bounded-console-prompt-queue.md
  - backlog/decisions/033-application-session-state-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the Console close-session confirmation from warning about transcript loss when the conversation is already durable and no agent, tool, approval, or queued-prompt work remains active.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An idle saved Console conversation closes without a confirmation because its durable transcript is not at risk.
- [x] #2 An unsaved Console conversation with transcript messages still requires confirmation before close.
- [x] #3 A saved Console conversation with live agent/tool activity or queued prompts still requires confirmation and preserves the revision recheck.
- [x] #4 Focused Console close regressions and static checks pass.
- [x] #5 Unsaved messages on inactive conversation-tree branches still require confirmation before close.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted regression for closing an idle saved conversation without confirmation and verify it fails against the current transcript-count predicate.
2. Make session close impact count unpersisted messages across every branch owned by the session, while retaining lifecycle loss counts and revision fencing.
3. Run the focused close tests, reached lifecycle tests, ruff, and diff checks; self-review the scoped change.

ADR required: no
ADR path: backlog/decisions/046-visible-bounded-console-prompt-queue.md and backlog/decisions/033-application-session-state-ownership.md
Reason: Existing ADRs already define combined Console loss confirmation, transient lifecycle ownership, and revision-pinned revalidation; this bug fix corrects the existing predicate without changing those boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Changed the close-impact transcript count from every visible message to only
messages without a durable `persisted_message_id`. This is more precise than
the planned conversation-id check: saved rows add zero loss risk while any
partially persisted row remains protected. Unsaved rows still require
confirmation, and live-run, approval, queued-prompt, and revision-revalidation
behavior is unchanged.

Qodo review identified that the active transcript is only one projection of the
conversation tree. Added a read-only store query for all session-owned tree
nodes and based close impact on that complete set, so an unpersisted inactive
branch cannot be deleted silently. Added direct unit cases for fully persisted,
fully unpersisted, and mixed durability plus a mounted hidden-branch close
regression.

Added a mounted real-store regression for a restored persisted message alongside
the existing unsaved-message close coverage. Before the fix, the restored
session remained open behind the confirmation; after the fix it closes without
a modal. After rebasing onto current `origin/dev` and addressing Qodo review,
22 focused tests passed across the direct close-impact unit cases and the full
mounted Console button-routing file, including saved, unsaved, hidden-branch,
queue, and revision-recheck close flows. Ruff, focused formatting, Python
compilation, the backlog-ID guard, and `git diff --check` pass. The
repository-wide suite was not run under the project's targeted-test policy.

ADR required: no. ADR-046 and ADR-033 remain the governing decisions; no state
owner, lifecycle boundary, or cross-module contract changed. No new
generalizable lesson was produced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console close confirmation now counts only transcript rows that are not durably
saved. Idle saved conversations close silently, while unsaved content and live
or queued agent work remain protected by the existing revision-pinned warning.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] All acceptance criteria are complete.
- [x] The approved plan was followed; the more precise per-message durability check is documented.
- [x] Unit and mounted regression coverage protects saved, unsaved, mixed, and inactive-branch transcript impact.
- [x] Reached mounted, lifecycle, cleanup, and queue tests pass.
- [x] Ruff, Python compilation, and whitespace checks pass.
- [x] Implementation Notes and Final Summary document the completed change.
- [x] Self-review found no regression to lifecycle, privacy, persistence, or revision-fence ownership.
- [x] No performance, security, or licence behavior changed.
- [x] No new generalizable lesson was produced.
- [x] ADR check completed against ADR-046 and ADR-033; no new ADR is required.
- [x] Task status is Done.
<!-- DOD:END -->
