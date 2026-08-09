---
id: TASK-196
title: Prompt version-history UI in the Library editor
status: In Progress
assignee: []
created_date: '2026-07-12 13:16'
updated_date: '2026-08-09 01:40'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the 2026-07-12 Library Prompts spec: LocalPromptService already supports list_prompt_versions/restore_prompt_version (rebuilt from sync_log); the v1 editor shows only the vN meta line. Add a history disclosure with per-version preview and Restore.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Library editor exposes a collapsed `Retained history (N)` disclosure for the selected local Prompt or Recipe; opening it lazily loads one bounded page, older pages can be loaded explicitly, and stale results cannot replace another prompt's history.
- [ ] #2 Each retained row shows its version, timestamp, artifact type, and a truthful changed-field summary, with literal read-only metadata and System/User preview; version 1 shows `Created`, and a pruning/version gap shows `Earlier baseline unavailable`.
- [ ] #3 The app-wired `PromptScopeService` exposes create/update-only local retained history through an indexed, cursor-paged query that reads at most `page_size + 1` matching snapshots and remains independent of unrelated sync-log volume.
- [ ] #4 Future create/update snapshots capture effective ordered keywords after membership settles, while older snapshots remain readable as `keywords_captured = false`; Prompt fields, keyword links, and snapshot persistence commit or roll back together.
- [ ] #5 Valid legacy text and structured-v2 Prompt/Recipe snapshots are restorable under current capabilities; malformed, mismatched, unknown, future, and foreign structured-v1 snapshots remain preview-only with the exact compatibility reason, and a dirty working copy disables restore without disabling viewing.
- [ ] #6 Restore rechecks the expected current version, uses the ordinary conditional update path, appends a new current version without changing retained history in place, reports both source and new versions, and enters the existing conflict/Reload state when stale.
- [ ] #7 A byte-identical restore returns `no_change` without a new sync row; a modern snapshot restores captured keywords, an older snapshot retains and discloses current keywords, and duplicate-name or keyword failures leave the Prompt, keywords, and history unchanged.
- [ ] #8 Automated DB, service, state, and Textual UI tests cover migration/index use, bounded paging and pruning gaps, compatibility and keyword semantics, conditional restore outcomes and rollback, lazy disclosure/paging, dirty-state gating, stale-result guards, and exact user-facing outcomes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.

ADR path: `backlog/decisions/049-local-prompt-retained-version-history.md`

Reason: TASK-196 makes retained sync payloads a user-visible source and fixes durable storage/index, pruning, keyword, compatibility, conditional-restore, concurrency, and rollback semantics.

1. Add a Prompt-specific schema migration and composite sync-log index, then prove fresh/upgrade behavior, bounded `page_size + 1` cursor paging, create/update filtering, malformed-row visibility, and index-backed query plans.
2. Capture effective ordered keywords in future create/update payloads after membership settles, preserving additive compatibility and atomic rollback of Prompt fields, keyword links, and snapshots.
3. Normalize one retained-version envelope with exact preview text, keyword-capture state, immediate-predecessor summaries, pruning-gap disclosure, and ADR-040 compatibility/restore eligibility.
4. Route app-wired local history through `PromptScopeService` and the bounded local database API while preserving the existing server route for other callers.
5. Implement expected-version conditional restore through the ordinary update transaction, including modern/legacy keyword handling, `no_change`, conflict, validation, and rollback outcomes.
6. Add immutable history UI state and a lazy Library disclosure with read-only preview, paging/retry, dirty and compatibility gating, type-change confirmation, stale-worker guards, and explicit success/conflict/error copy.
7. Update the Library Prompt user guide, run focused and full verification, self-review the change, and record exact evidence and ADR links before moving TASK-196 to Done.
<!-- SECTION:PLAN:END -->
