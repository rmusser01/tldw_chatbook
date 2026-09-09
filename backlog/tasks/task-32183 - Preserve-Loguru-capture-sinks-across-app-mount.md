---
id: TASK-32183
title: Preserve Loguru capture sinks across app mount
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 09:26'
updated_date: '2026-09-05 09:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep worker-warning tests attached to the Loguru lifecycle they observe so app logging initialization cannot invalidate their owned sink.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Worker-warning capture tests install their sink after app logging initialization.
- [x] #2 Known navigation workers remain silent while unknown worker groups still warn.
- [x] #3 The complete `Tests/App` suite passes.
- [x] #4 Scoped Ruff and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both worker-warning capture failures independently.
2. Install each test-owned Loguru sink only after the mounted app finishes logging setup.
3. Run the focused worker event file and complete App suite.
4. Run scoped static and diff checks.

ADR required: no

ADR path: N/A

Reason: this repairs test resource ownership without changing logging architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved both warning-capture sinks inside the mounted-app context, after `_setup_logging` has removed and replaced process-wide Loguru handlers. Each test now owns its sink for exactly the observation window and removes it before app teardown. Verification: the focused file passed 4 tests, `Tests/App` passed 254 tests, Ruff passed, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31714 was renumbered to TASK-31901 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.

On 2026-09-07 the review record moved again, from TASK-31901 to TASK-32014,
to preserve upstream Console note-follow-up ownership on dev 3090013cfe.
The earlier mapping above remains historical provenance.

On 2026-09-08 the user explicitly approved moving this review-only record from
TASK-32014 to TASK-32108, preserving upstream Meetings ServerDiarizer TASK-32014.
The fresh all-ref/worktree census found maximum TASK-32107. Original dates,
completed implementation notes, and historical evidence remain unchanged.

On 2026-09-09, landed Buddy qualification TASK-32108 on dev9faf96d9f8
required this unmerged review record to move to TASK-32136. The fresh census
across 1,027 local/remote refs and all registered worktrees found maximum
TASK-32135. Only the review identity and active references move; landed IDs,
original dates, implementation notes and historical evidence stay unchanged.

On 2026-09-09, the Library Notes Folder-files task landed on dev 88b07c4f6f
with TASK-32136. That claimant was created at 2026-09-08 21:39 and added by
f054f35ae before this review adopted ID 32136 at 2026-09-09 00:31 in
5f840d365. Following the user's review-only renumbering policy, this unmerged
record moves to TASK-32183 while the landed Library identity stays unchanged.
The fresh all-ref history and registered-worktree census found maximum 32182
and no candidate 32183 path or content reference. Only this filename,
frontmatter and active inbound pointers change; earlier IDs, original dates,
completed implementation notes and literal evidence paths remain provenance.
