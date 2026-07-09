---
id: TASK-124
title: Address PR 525 Sync v2 M1 review comments and rebase
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-16 13:13'
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
Rebase PR #525 onto the latest dev branch and resolve the outstanding Sync v2 M1 review comments around legacy capabilities parsing, dry-run domain matching, unsafe API-key examples, and nullable capability fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto current origin/dev without losing Sync v2 M1 client behavior.
- [x] #2 Legacy capabilities payloads with flat supported_operations parse without validation errors and preserve back-compat accessors.
- [x] #3 Dry-run domain selection handles M1 dotted capability domains and nullable raw capability fields.
- [x] #4 Plan documentation no longer contains hardcoded or key-shaped API key literals.
- [x] #5 Focused regression tests and touched-scope security checks are run and documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase `feat/sync-v2-m1-client-conformance` onto current `origin/dev` and resolve any conflicts without losing scoped Sync v2 M1 behavior.
2. Audit PR #525 Qodo and Gemini review comments against the rebased code.
3. Use TDD for remaining behavior gaps: legacy flat `supported_operations`, coarse-to-dotted dry-run domain matching, and nullable capability domain handling.
4. Remove hardcoded/key-shaped API key literals from the implementation plan document examples.
5. Run focused Sync tests, `git diff --check`, and Bandit on touched production paths.
6. Update Backlog implementation notes, commit, push, and resolve addressed GitHub review threads.

ADR required: no new ADR
ADR path: backlog/decisions/008-sync-v2-client-m1-contract-alignment.md
Reason: ADR 008 already records the M1 contract decision; this task is bounded PR review remediation and rebase work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased feat/sync-v2-m1-client-conformance onto origin/dev at 59c52e03 with no conflicts; origin/dev is an ancestor of HEAD.

Addressed PR #525 review comments:
- Normalized legacy flat supported_operations list payloads into operations={"*": [...]} before Pydantic field validation.
- Updated Sync v2 domain request typing/defaults to allow M1 dotted domains while preserving legacy coarse strings used by existing local sync paths.
- Added dry-run advertised-domain selection so coarse requests like notes/chat match server-advertised dotted domains such as notes.note and chat.message.
- Made raw capability domain extraction tolerate domains=None by falling back to supported_domains or an empty list.
- Removed key-shaped SINGLE_USER_API_KEY literals from the M1 plan examples and switched them to environment-variable based snippets.

Verification run:
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py::test_capabilities_parses_legacy_flat_supported_operations -v (passed)
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Sync_Interop/test_server_sync_service.py::test_server_sync_service_dry_run_maps_coarse_requests_to_m1_dotted_domains Tests/Sync_Interop/test_server_sync_service.py::test_server_sync_service_dry_run_falls_back_when_domains_value_is_none -v (passed after fixes; failed before fixes as expected)
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/tldw_api/test_sync_schemas_m1.py Tests/tldw_api/test_sync_client.py Tests/Sync_Interop/test_server_sync_service.py -v (47 passed)
- git diff --check (passed)
- rg for the reviewed key-shaped literal in Docs/superpowers/plans/2026-06-14-sync-v2-m1-p1-schema-transport.md (no matches)
- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m bandit -r tldw_chatbook/tldw_api tldw_chatbook/Sync_Interop -f json -o /tmp/bandit_pr525.json (reported pre-existing findings in sync_state_repository.py and tldw_api/utils.py only; none were on changed lines).

CI check state is intentionally ignored per user direction because checks are being actively canceled.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #525 was rebased onto current origin/dev and review comments were addressed locally. The branch now handles legacy flat capabilities payloads, M1 dotted dry-run domains, nullable capability domain fields, and removes hardcoded/key-shaped API key examples from the plan documentation. Focused pytest, diff whitespace checks, key-literal search, and touched-scope Bandit were run and documented; CI status is intentionally excluded per user instruction.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
