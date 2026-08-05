---
id: TASK-517
title: Refresh server-client migration audit semantic keys
status: Done
assignee: []
created_date: '2026-07-24 18:31'
updated_date: '2026-07-24 18:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the provider-migration static guard after intentional formatting and type-signature changes altered line-level semantic keys without adding or removing legacy builder seams.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The migration audit semantic snippets exactly match all current legacy builder lines
- [x] #2 The four drifted files retain their existing reason categories and audited match counts
- [x] #3 No production client construction or runtime policy code changes are made
- [x] #4 The full provider-migration audit guard passes
- [x] #5 The full RuntimePolicy suite passes together with the watchlist and skill-oracle corrections
- [x] #6 Task documentation records the merge-base failure, semantic-key deltas, ADR decision, and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the four-file semantic drift on feature branch and merge base and inventory every missing/extra normalized line.
2. Update only the audit document snippets and informational line hints to the current line-level forms for the study handlers, app assignments, and multiline bootstrap signature.
3. Run the full provider-migration audit guard and RuntimePolicy suite with TASK-515/516 changes present.
4. Run Markdown/diff checks and independently review before completing all three tasks.

ADR required: no
ADR path: N/A
Reason: This refreshes documentation keys for unchanged, already-classified compatibility seams; it changes no service contract, provider boundary, ownership, or runtime architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Refreshed the provider-migration audit's normalized semantic keys and matching line hints after multiline formatting changed the matched source lines without changing the audited compatibility seams.

Semantic-key deltas:
- `flashcards_handler.py` and `quizzes_handler.py`: replaced each former one-line call with its current first line, `server_service = Server{Study,Quiz}Service.from_config(`, at lines 199 and 221.
- `app.py`: replaced seven assignment-prefixed keys with the current continuation-line keys for `ServerResearchSearchService` (two matches), `ServerVoiceAssistantService`, `ServerPersonalizationService`, `ServerCollectionsFeedsService`, `ServerLLMProviderCatalogService`, and `ServerUserGovernanceService`; updated their hints to lines 4209, 4508, 4417, 4447, 4563, 4777, and 4831.
- `runtime_policy/bootstrap.py`: replaced the former one-line annotated signature with the current first line, `def build_runtime_api_client_from_config(`, at line 87.
- Preserved every row's reason category and per-file audited match count; no builder seam was added, removed, or reclassified.

RED evidence:
- Feature branch before the fix: the full migration-audit file reported 1 failed and 9 passed, with exactly four semantic-drift entries (`flashcards_handler.py`, `quizzes_handler.py`, `app.py`, and `runtime_policy/bootstrap.py`).
- Merge base `ba6b45cdf4dd548796e072f5933cdcf44c8c0344`: the same full file reported 1 failed and 9 passed with exactly the same four drifted files.

Verification:
- `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`: 10 passed.
- Full `Tests/RuntimePolicy/` with TASK-515 and TASK-516 corrections present: 248 passed; one pre-existing `requests` dependency-version warning.
- Both owned Markdown files parsed successfully with the installed Python Markdown tables extension.
- `git diff --check` passed for the owned files.
- Scope review confirmed only this audit document and TASK-517 were changed; no production client-construction or runtime-policy code was modified.

ADR required: no
ADR path: N/A
Reason: This documentation-only semantic-key refresh preserves existing compatibility seams and changes no service contract, provider boundary, ownership, or runtime architecture.

Files modified:
- `Docs/Development/server-client-provider-migration-audit.md`
- `backlog/tasks/task-517 - Refresh-server-client-migration-audit-semantic-keys.md`
<!-- SECTION:NOTES:END -->
