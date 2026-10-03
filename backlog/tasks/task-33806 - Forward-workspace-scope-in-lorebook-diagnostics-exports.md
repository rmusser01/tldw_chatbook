---
id: TASK-33806
title: Forward workspace scope in lorebook diagnostics exports
status: Done
created_date: 2026-10-03 01:02
assignee:
- '@codex'
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2968
modified_files:
- tldw_chatbook/tldw_api/client.py
- Tests/tldw_api/test_character_persona_client.py
- Tests/tldw_api/test_lorebook_diagnostics_transport.py
- Tests/Character_Chat/test_character_persona_scope_service.py
updated_date: 2026-10-03 03:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace-chat lorebook diagnostics exports through the Chatbook API client fail because the client cannot supply the server-required conversation scope. Repair the client contract without weakening server authorization or changing legacy global calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace-scoped diagnostics requests include the matching scope and workspace identifier.
- [x] #2 Calls without scope retain their existing pagination-only request and response behavior.
- [x] #3 Invalid scope combinations fail before network dispatch through existing validation.
- [x] #4 Targeted regressions and read-only live-server compatibility checks pass, and a companion PR is created.
- [x] #5 The owning service-wiring regression reaches its existing assertions using an isolated real SQLite database and closes the database after the test.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Rebase the scoped client fix onto latest dev and verify the original commits are range-diff equivalent. Status: Complete. Stage 2: Repair the owning wiring-test fixture without changing production guards, then verify targeted tests, security/static checks, performance ratchets, and read-only authenticated real-server compatibility. Status: Complete. Stage 3: Publish the companion PR, address verified Qodo findings, preserve the requester-authored Change summary, and complete independent review. Status: Complete. ADR required: no; ADR path: N/A; routine existing-contract compatibility fix. This task tracks the scoped implementation and review; operational protected-merge completion is tracked on PR2968 and is not claimed here.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added optional scope_type/workspace_id to export_lorebook_diagnostics, reusing the existing ConversationScopeParams normalizer. Existing positional parameters and omitted-scope pagination-only behavior are preserved; wrappers already forward kwargs and observe-policy checks remain intact. No server, UI, dependency, or ownership changes.
TDD: 7 new cases fail on dev2612fc56 for unsupported scope arguments; the owning client file passes all22 cases after the fix. Final client/conversation/schema run:46passed. Broader run:99passed and1pre-existing app-wiring failure; complete untouched-base service run:53passed and the same Mock missing is_memory_db failure. This is not claimed as an all-green/full-suite result.
Read-only authenticated real HTTP checks (no mocks, no sends): before fix, omitted-scope client request404 while raw correct scope200; after fix, explicit and inferred workspace scope match the actual200 response. Omitted/wrong scope still404. Conversation history SHA256 remains fca0d271c84eee53b1c7e7c9f0bb2568e384715a0c8023f05ca1f9e81ccbbf6f. API runtime imported server940fd21b on base9958110d; no TEST_MODE. This qualifies the API client, not native TUI/browser export UI.
Changed blocks pass Ruff formatting; filewide Ruff debt decreases755to754 with zero added-line findings. Bandit client scope:0findings,0errors. Compilation, diff whitespace, task-ID uniqueness, and4787task-file readability checks pass. Independent reviewer: no actionable findings.
ADR required:no; routine existing-contract bug fix. Evidence retained at /private/tmp/chatbook-lorebook-workspace-20261002-mQ7iVP, including failed initial TDD and premature baseline-export invocations; the latter ran no tests and were repeated only after complete baseline export. Full-suite sweep not requested. PR publication/CI/review and requester-authored Change summary remain pending; no companion merge authorized.
Published companion PR2968 against dev: https://github.com/rmusser01/tldw_chatbook/pull/2968 . Fix commit33d57c3c8611dac4546c027fc9298772c3244fcc. All scoped acceptance criteria are satisfied; task remains In Progress for hosted CI/Qodo review and requester-authored merge summary. No companion merge performed. This follow-up task-only update does not change qualified application or test source.
Qodo posted two verified review comments on PR2968: missing Google-style public docstring sections (4171087200) and no committed HTTP transport integration test (4171087202). Addressing both within existing scope before handoff.
Addressed Qodo docstring4171087200 with Google-style Args/Returns/Raises, including scope inference, ignored global workspace IDs, and validation/network/server errors. Addressed transport coverage4171087202 with two committed tests through the unmodified public client, real HTTPX socket transport, and an owned stdlib loopback fixture endpoint. The receiving endpoint records exact scope and pagination query fields. Both tests fail against the exact original client and pass against the fix. This fixture-based integration coverage is explicitly not live-server UAT; independent authenticated no-mock real-server checks pass again with unchanged history.
Final five-file run:101passed and the same one pre-existing app-wiring Mock/is_memory_db failure. Client/schema/transport subset:48passing cases. New transport file has zero Ruff findings and passes formatting; changed production block formatting passes. Application behavior is unchanged from the qualified scope fix. Full-suite sweep not requested; CI/latest-head review and human merge-summary gate remain pending.
Requester supplied the human-written Change summary; installed verbatim in PR2968 and read back exactly. Requester now explicitly authorizes rebasing, addressing Qodo feedback, and merging this companion PR. Clean rebase from base2612fc56 onto dev2d34cbf80d1d7569abf0490e5c9821101d892661 produced7f8b9e18c89080119dd121e142c37c11cd957a0f; all three commits are range-diff equivalent. Preserved pre-rebase head in codex/backup-lorebook-pre-rebase-3a95247b. No source conflicts or unrelated edits.
Fresh rebase run retained the known class-spec DB mock failure: participants._core_operation requires the instance-owned is_memory_db attribute, which Mock(spec=CharactersRAGDB) omits. Replace only the wiring-test DB fixture with real in-memory CharactersRAGDB and guaranteed finalizer cleanup; keep all production recovery checks and wiring assertions unchanged. Two additional transport setup errors are sandbox bind PermissionError, not application failures; repeat with approved real numeric-loopback access. Failed attempt retained in rebase-tests.log/.xml.
Rebased qualification:102/102 targeted tests pass after replacing the wiring-test DB mock with isolated in-memory SQLite and a pytest finalizer that closes it. No production recovery guards or existing assertions changed. Read-only no-mock real authenticated API checks execute this exact checkout: explicit/inferred scope200 matching raw endpoint, missing/wrong scope404, protected history byte-equivalent with prior SHA256. Added-line Ruff findings0; changed blocks/new file formatting and compilation pass; production client Bandit0findings/0errors. Existing filewide Ruff debt remains and is not claimed clean. Stage3 is now awaiting exact-lease publication, current-head hosted CI and Qodo, and requested merge.
Additional rebased verification:37/37 Perf Guard cases pass locally, with3budget-headroom warnings (no failing guards). CSS reproduction, profile-owned-path and persistent diagnostic inventories, schema table allowlist, index-plan pins, worker/timestamp contracts, and UI gate census pass. Independent exact-head review found no actionable issues. Qodo footer identifies c387faa084102a4c486a700f266d87d9ab583a7b and reports0active bugs/rules/cross-repo/skills, with no unresolved threads. Hosted Perf Guard and Derived Artifacts are waiting for runner capacity; strict dev branch protection requires Derived artifacts reproduce from their sources. No admin bypass or merge performed. Only superseded runs of this same PR were cancelled, including force-cancel of the initial-head run that held the new workflow concurrency group after normal cancellation. Current-head runs are preserved.
Scoped implementation and review are complete against dev2d34cbf8. All139local cases pass (102owning and37PerfGuard); real authenticated HTTP positive/negative controls pass without mocks and preserve history. Hosted Perf Guard, PR Fast Lane, and UI Fast Lane succeeded for c387faa084102a4c486a700f266d87d9ab583a7b. Required derived-artifact job remains queued; no protected merge has occurred. Closing the implementation task adds only tracking metadata, with production/test bytes preserved; final-head CI and exact-head merge remain mandatory before reporting the PR merged. No full-suite or native TUI/browser UAT claim is made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workspace lorebook diagnostics exports now forward scope through the existing validated client contract, with legacy calls unchanged. Scope unit and real HTTP transport regressions cover the fix; the owning app-wiring test uses isolated real SQLite with cleanup.139local tests pass, touched-scope static/security checks pass, live authenticated read-only API checks pass without mocks, and exact-head independent/Qodo reviews have no actionable findings. The requester-authored Change summary is preserved verbatim. Implementation complete; protected PR2968 merge remains pending hosted CI and is not claimed complete.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
