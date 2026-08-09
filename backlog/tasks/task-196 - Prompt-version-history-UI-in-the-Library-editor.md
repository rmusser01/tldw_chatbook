---
id: TASK-196
title: Prompt version-history UI in the Library editor
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 13:16'
updated_date: '2026-08-09 09:22'
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
- [x] #1 The Library editor exposes a collapsed retained-history disclosure for the selected local Prompt or Recipe; an index-only exact count changes its loading label from `Retained history (…)` to `Retained history (N)`, opening it lazily loads one bounded page, older pages can be loaded explicitly, load failures offer Retry, and stale count/page results cannot replace another prompt's history.
- [x] #2 Each retained row shows its version, timestamp, artifact type, and a truthful changed-field summary, with literal read-only metadata and System/User preview; version 1 shows `Created`, and a pruning/version gap shows `Earlier baseline unavailable`.
- [x] #3 The app-wired `PromptScopeService` exposes create/update-only local retained history through an indexed, cursor-paged query that reads at most `page_size + 1` matching snapshots and remains independent of unrelated sync-log volume.
- [x] #4 Future create/update snapshots capture effective ordered keywords after membership settles, while older snapshots remain readable as `keywords_captured = false`; Prompt fields, keyword links, and snapshot persistence commit or roll back together.
- [x] #5 Valid legacy text and structured-v2 Prompt/Recipe snapshots are restorable under current capabilities; malformed, mismatched, unknown, future, and foreign structured-v1 snapshots remain preview-only with the exact compatibility reason, a dirty working copy disables restore without disabling viewing, and confirmation always states that restore creates a new current version plus any Prompt↔Recipe type change.
- [x] #6 Restore re-resolves the exact retained change/version and expected current version in one conditional transaction, refuses missing/pruned snapshots and deleted current records, uses the ordinary update path, appends a new current version without changing retained history in place, reports both source and new versions, and enters the existing conflict/Reload state when stale.
- [x] #7 A byte-identical restore returns `no_change` without a new sync row; canonical keyword membership avoids order-only changes, a modern snapshot restores captured keywords, an older snapshot retains and discloses current keywords, and artifact-validation, duplicate-name, or keyword failures leave the Prompt, keywords, and history unchanged while preserving the selected row for retry.
- [x] #8 Automated DB, service, state, and Textual UI tests cover migration/index use, bounded paging and pruning gaps, compatibility and keyword semantics, conditional restore outcomes and rollback, lazy disclosure/paging, dirty-state gating, stale-result guards, and exact user-facing outcomes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.

ADR path: `backlog/decisions/049-local-prompt-retained-version-history.md`

Reason: TASK-196 makes retained sync payloads a user-visible source and fixes durable storage/index, pruning, keyword, compatibility, conditional-restore, concurrency, and rollback semantics.

1. Add a Prompt-specific schema migration and composite sync-log index, then prove fresh/upgrade behavior, an index-only exact count, bounded `page_size + 1` cursor paging, create/update filtering, malformed-row visibility, and index-backed query plans.
2. Capture effective ordered keywords in future create/update payloads after membership settles, preserving additive compatibility and atomic rollback of Prompt fields, keyword links, and snapshots.
3. Normalize one retained-version envelope with exact preview text, keyword-capture state, immediate-predecessor summaries, pruning-gap disclosure, and ADR-040 compatibility/restore eligibility.
4. Route app-wired local history through `PromptScopeService` and the bounded local database API while preserving the existing server route for other callers.
5. Implement exact retained-row plus expected-current-version conditional restore through the ordinary update transaction, including pruning/deleted-record refusal, modern/legacy keyword handling, `no_change`, conflict, validation, and rollback outcomes.
6. Add immutable history UI state and a lazy Library disclosure with read-only preview, paging/retry, dirty and compatibility gating, type-change confirmation, stale-worker guards, and explicit success/conflict/error copy.
7. Update the Library Prompt user guide, run focused and full verification, self-review the change, and record exact evidence and ADR links before moving TASK-196 to Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded, indexed retained Prompt history and conditional restore end to end.

- ADR: backlog/decisions/049-local-prompt-retained-version-history.md. No new ADR was needed beyond ADR-049 during implementation.
- Storage and service: migrated Prompts DB to schema v4 with a validated partial covering index, exact count and bounded cursor paging; captured effective keywords atomically; routed both app-wired local service seams through bounded history APIs; restored exact snapshots in one immediate expected-version transaction with no_change and rollback semantics.
- Normalization and compatibility: preserved literal legacy and structured previews, pruning gaps, raw future types, stable compatibility reasons, legacy Recipe preview-only handling, and current local capability/size admission before confirmation with transaction-time revalidation.
- UI: added immutable history state, a screen-owned Textual disclosure/controller, lazy pages, read-only previews, Retry/Reload, confirmation, exact outcomes, dirty and compatibility gates, fixed action-strip geometry, readable disabled actions, and UUID/scope/request guards for navigation, delayed workers, old-DOM events, restore refresh, collapse, and same-ID ABA races.
- Documentation: updated the Library Prompt guide and production diagnostic inventory; existing lessons-testing-evidence guidance was applied and no new generalized lesson was required.
- Main files: tldw_chatbook/DB/Prompts_DB.py; tldw_chatbook/Prompt_Management/{local_prompt_service,prompt_chatbook_scope_service,prompt_normalizers,prompt_restore_errors,prompt_scope_service}.py; tldw_chatbook/Library/library_prompts_state.py; tldw_chatbook/UI/Library_Modules/prompt_history*.py; tldw_chatbook/UI/Screens/library_screen.py; tldw_chatbook/Widgets/Library/library_prompts_canvas.py; focused DB/service/state/UI tests and docs.
- Fresh rebased verification: 240 affected DB/state/normalizer/service/controller tests passed; 179 full Prompt canvas tests passed; 2 Prompt shell tests passed; 5 server-parity tests passed; 8 diagnostic-inventory tests passed; changed-file Ruff lint/format, py_compile, CSS build/sync, and diff checks passed.
- Repository-wide maxfail comparison: branch reached 1101 passed and 1 skipped before three inherited failures (chat_screen size ratchet and two dictation shutdown tests). Exact current dev reproduces the ratchet and both dictation failures; current dev also had the stale diagnostic inventory repaired by this branch. Remote CI is non-gating per user direction.
- Review: independent Task 7 spec and code/security/UX reviews plus final whole-branch review all approved with no remaining Critical, Important, or Minor findings.
<!-- SECTION:NOTES:END -->
