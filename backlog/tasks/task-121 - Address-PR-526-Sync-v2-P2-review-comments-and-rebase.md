---
id: TASK-121
title: Address PR 526 Sync v2 P2 review comments and rebase
status: Done
assignee: []
created_date: '2026-06-16 13:37'
updated_date: '2026-06-16 13:37'
labels:
  - pr-review
  - sync-v2
  - bugfix
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #526 onto the latest dev branch and resolve the outstanding Sync v2 P2 review comments around server-trusted notes application safety, docs secret handling, mirror persistence, idempotent acknowledgements, and envelope validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto current origin/dev without losing Sync v2 P2 notes.note behavior
- [x] #2 Docs no longer contain hardcoded or key-shaped API key literals in the P2 plan examples
- [x] #3 Notes M1 apply path validates required mirror/dataset/id inputs and sanitizes persisted note payload fields
- [x] #4 NotesMirror validates database paths and uses transaction context for schema and write operations
- [x] #5 NotesM1SyncFlow updates mirror metadata for accepted and idempotent acknowledgements without persisting empty or downgraded hash/revision metadata
- [x] #6 Legacy encrypted apply paths fail deterministically when dataset_key is missing instead of crashing
- [x] #7 SyncV2Envelope rejects payloads that omit both entity_id and object_id
- [x] #8 Focused regression tests, whitespace checks, and touched-scope Bandit are run and documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase PR #526 worktree branch onto current origin/dev, preserving PR #525 fixes already on dev.
2. Audit all unresolved Gemini and Qodo review threads and map each to a scoped fix.
3. Use TDD for missing guard/validation/mirror/idempotency/envelope-identifier behaviors.
4. Remove hardcoded/key-shaped API key examples from the P2 plan document.
5. Run focused Sync v2 pytest targets, git diff --check, and Bandit on touched production paths.
6. Update Backlog notes, commit, push back to PR #526, and resolve addressed review threads.

ADR required: no
ADR path: backlog/decisions/008-sync-v2-client-m1-contract-alignment.md
Reason: ADR 008 already records the Sync v2 M1 contract; this is bounded PR review remediation and rebase work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased PR #526 worktree branch onto origin/dev at 6744b466. Rebase skipped PR #525 commits already on dev and required one conflict resolution in tldw_chatbook/tldw_api/sync_schemas.py; resolution preserved PR #525 flexible domain parsing/defaults and P2 operation/encryption superset behavior. origin/dev is now an ancestor of HEAD.

Addressed PR #526 review comments:
- Replaced key-shaped SINGLE_USER_API_KEY examples in the P2 plan with environment-variable based shell/Python snippets.
- Added NotesM1SyncAdapter guards for missing notes_mirror/dataset_id and missing object identity; invalid notes.note payloads now record conflicts instead of persisting malformed data.
- Sanitized server-provided note title/content before local persistence.
- Validated NotesMirror db_path with path_validation utilities, rejected parent traversal, and wrapped schema/write operations in sqlite transaction contexts.
- Updated NotesM1SyncFlow.push to process accepted and idempotent acknowledgements, skip mirror writes when a returned client_envelope_id cannot be matched to a submitted payload hash, and preserve existing mirror metadata instead of downgrading on incomplete acknowledgement metadata.
- Added a legacy encrypted apply guard so missing dataset_key records a missing_dataset_key conflict instead of crashing in decryption.
- Added SyncV2Envelope validation requiring entity_id or object_id.
- Added Google-style Args/Returns details to canonical_payload_hash.

Verification run:
- Focused red tests initially failed for the expected review-comment reasons: missing adapter guard, invalid payload persistence, unsanitized content, unsafe mirror path, idempotent mirror omission, missing mirror_errors, metadata downgrade, missing dataset_key crash, and missing envelope identifier acceptance.
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest [9 focused regression targets] -v (9 passed after fixes).
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Sync_Interop/test_envelope_m1_superset.py Tests/Sync_Interop/test_push_pull_m1_responses.py Tests/Sync_Interop/test_canonical_hash.py Tests/Sync_Interop/test_notes_mirror.py Tests/Sync_Interop/test_notes_m1_adapters.py Tests/Sync_Interop/test_notes_m1_flow.py Tests/Sync_Interop/test_envelope_applier.py Tests/tldw_api/test_sync_schemas_m1.py Tests/tldw_api/test_sync_client.py Tests/Sync_Interop/test_server_sync_service.py -v (85 passed).
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api -q (609 passed, 2 existing Pydantic serialization warnings).
- git diff --check (passed).
- key-shaped literal search in Docs/superpowers/plans/2026-06-15-sync-v2-m1-p2-notes-vertical.md (no matches).
- conflict marker search for ^<<<<<<</^>>>>>>> in touched scopes (no matches).
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m bandit -r tldw_chatbook/Sync_Interop tldw_chatbook/tldw_api -f json -o /tmp/bandit_pr526.json (reported pre-existing findings in sync_state_repository.py and tldw_api/utils.py only; none were on changed lines).

CI check state is intentionally ignored per user direction because checks are being actively canceled.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #526 was rebased onto current origin/dev and all actionable review comments were addressed locally. The branch now hardens the P2 notes.note apply path, mirror persistence, idempotent acknowledgement handling, legacy encrypted apply behavior, envelope identity validation, and P2 plan secret examples. Focused and broad Sync/API pytest suites, whitespace checks, key-literal search, conflict-marker search, and touched-scope Bandit were run and documented; CI status is intentionally excluded per user instruction.
<!-- SECTION:FINAL_SUMMARY:END -->
