---
id: TASK-33806
title: Forward workspace scope in lorebook diagnostics exports
status: In Progress
created_date: 2026-10-03 01:02
assignee:
- '@codex'
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2968
modified_files:
- tldw_chatbook/tldw_api/client.py
- Tests/tldw_api/test_character_persona_client.py
updated_date: 2026-10-03 01:20
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace diagnostics export callers and compare existing conversation-scope client methods on dev2612fc56.
2. Add failing client regressions, then reuse existing scope normalization in export_lorebook_diagnostics without changing positional arguments.
3. Run targeted client/service regressions, static checks, and authenticated read-only live-server positive/negative controls without mocks.
4. Commit the atomic fix and task record, publish a companion PR against dev, and cross-link server PR3071.
ADR required: no
ADR path: N/A
Reason: Routine compatibility fix using the existing ConversationScopeParams contract; no server authorization, storage, or ownership change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added optional scope_type/workspace_id to export_lorebook_diagnostics, reusing the existing ConversationScopeParams normalizer. Existing positional parameters and omitted-scope pagination-only behavior are preserved; wrappers already forward kwargs and observe-policy checks remain intact. No server, UI, dependency, or ownership changes.
TDD: 7 new cases fail on dev2612fc56 for unsupported scope arguments; the owning client file passes all22 cases after the fix. Final client/conversation/schema run:46passed. Broader run:99passed and1pre-existing app-wiring failure; complete untouched-base service run:53passed and the same Mock missing is_memory_db failure. This is not claimed as an all-green/full-suite result.
Read-only authenticated real HTTP checks (no mocks, no sends): before fix, omitted-scope client request404 while raw correct scope200; after fix, explicit and inferred workspace scope match the actual200 response. Omitted/wrong scope still404. Conversation history SHA256 remains fca0d271c84eee53b1c7e7c9f0bb2568e384715a0c8023f05ca1f9e81ccbbf6f. API runtime imported server940fd21b on base9958110d; no TEST_MODE. This qualifies the API client, not native TUI/browser export UI.
Changed blocks pass Ruff formatting; filewide Ruff debt decreases755to754 with zero added-line findings. Bandit client scope:0findings,0errors. Compilation, diff whitespace, task-ID uniqueness, and4787task-file readability checks pass. Independent reviewer: no actionable findings.
ADR required:no; routine existing-contract bug fix. Evidence retained at /private/tmp/chatbook-lorebook-workspace-20261002-mQ7iVP, including failed initial TDD and premature baseline-export invocations; the latter ran no tests and were repeated only after complete baseline export. Full-suite sweep not requested. PR publication/CI/review and requester-authored Change summary remain pending; no companion merge authorized.
Published companion PR2968 against dev: https://github.com/rmusser01/tldw_chatbook/pull/2968 . Fix commit33d57c3c8611dac4546c027fc9298772c3244fcc. All scoped acceptance criteria are satisfied; task remains In Progress for hosted CI/Qodo review and requester-authored merge summary. No companion merge performed. This follow-up task-only update does not change qualified application or test source.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
