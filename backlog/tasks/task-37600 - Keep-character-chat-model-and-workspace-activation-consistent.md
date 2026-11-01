---
id: TASK-37600
title: Keep character chat model and workspace activation consistent
status: Done
assignee:
- '@codex'
created_date: 2026-10-29 20:04
updated_date: 2026-11-01 08:35
labels:
- console
- roleplay
- bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Starting a character conversation can revive an older provider selection, and closing its model picker can return the user to a different workspace chat. Preserve the current chat model selection and the newly opened character conversation through modal resume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character launches use the current non-pristine chat provider and model even when older controls disagree, while untouched chats still use current defaults.
- [x] #2 Character chats opened from a named workspace remain active after model Apply or Cancel, with the registry aligned to the character chat.
- [x] #3 Existing control-derived fallback, prompt seeding, pristine-session reuse, and source conversation behavior remain covered.
- [x] #4 Targeted regressions and UAT exercise launch, Apply, Cancel and a real local HTTP reply, with evidence recorded.
- [x] #5 The five adjacent app-backed character prompt tests run with a valid isolated config lifetime and pass their original assertions.
- [x] #6 The character launch regressions run in required PR CI, and the session controller stays within its existing size budget.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md; existing TASK-15120 workspace ownership ruling. Reason: restores existing conversation settings and workspace activation ownership without a new storage, security or service boundary. 1. Add failing character launch and modal resume regressions. 2. Prefer the active worked chat pair over stale control mirrors through one shared character-default helper. 3. Align the registry through the existing workspace seam at character activation. 4. Run targeted tests, static checks, actual-app UAT and independent review; record evidence and open a PR against dev.

Merge follow-up: repair the five reproduced app-factory config lifetime failures using the existing private-profile test helper, without bypassing config admission; rerun affected tests, obtain a fresh independent review and merge through the configured queue once required checks/review policy permit.

Reuse the existing active-session lookup in character/persona handoffs to meet the unchanged module budget; register bootstrap-marked launch cases in admission-sensitive CI and repaired prompt cases in the UI gate. User authorized independent review in place of unavailable Qodo for PR 3306.
2026-11-01: Follow the integration section in Docs/superpowers/plans/2026-10-29-character-chat-selection.md. Rebase on621b8f1550, preserve1253census entries plus5prompt nodes, correct younger SSE task identity to38328, qualify focused behavior, independent review, merge under user-authorized hosted-CI waiver. ADR required:no; ADR006/095.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Character launch resolves the current worked chat provider/model through one shared helper and the existing defaults chain. Personas handoffs and Console picker New Chat align the target workspace before activation, preserving selection after modal resume and leaving source content intact. Reused the existing active-session lookup in character/persona handoffs to retain the unchanged 7000-line controller budget. Repaired five adjacent app-backed prompt fixtures using the existing private-profile helper and resource-owner fixture, preserving their assertions. Added six launch variants to required admission-sensitive CI and five prompt nodes to the UI census (1150 entries). Changed session.py, character.py, wiring.py, focused tests, CI enrollment and architecture/QA docs. Six regressions failed before the fix; final verification includes 13 prompt cases, 32 combined character/workspace cases, 19 post-cleanup character/persona cases, and 400 CI-contract/size checks, all passing. Actual served-app UAT passed Pixel Migu launch, model Apply, backdrop Cancel and HTTP reply with selected model and character prompt; source messages stayed unchanged. Independent code/evidence and final followup reviews found no issues. Scoped Ruff, task-ID/census, worker-contract and whitespace guards pass with no introduced lint diagnostics; local Canvas preflight requires unavailable Python3.12.11, which hosted CI pins. Existing CI-contract formatting debt is unchanged. Full suite not run per targeted-only policy. No new ADR: existing ADR-006/095 and TASK-15120 apply. QA evidence and limits: Docs/superpowers/qa/2026-10-29-character-chat-selection/qualification.md. PR3306 targets dev and is current at d36affec9b; required CI and the merge queue enforce integration. Qodo lacked workspace credits; the user explicitly authorized independent review to replace Qodo for this PR.

Final dev rebase: prior head a43cc847e8 passed every hosted required CI check. Dev then merged PR3307 / TASK37620 at ed7481a30c, supplying workspace activation for character/persona entry paths. Rebase retains that implementation and removes our redundant optional callback/wiring; remaining production diff fixes stale provider defaults. Both CI enrollments retained (1157 entries); controller6989 <= dev budget6990. Post-rebase31 combined character/prompt/real-model-picker tests and400CI/census/size tests pass. Independent conflict-resolution review of ed7481a30c..51de16779e found no findings. New hosted CI is required before queue merge.

Second dev rebase onto ff815f8559 (PR3302): d0ae6d68be also passed every hosted check. Resolved additive lesson/census conflicts by retaining both sets (1161 entries); session production logic and all character tests are unchanged from the prior green head, and the defaults callback remains the only wiring diff from dev. All31 combined character/prompt/real-model-picker tests and all400CI/census/size contracts pass again; independent rebase review has no findings. Scoped lint, task IDs across5928files and whitespace checks pass. Recorded the queue FIFO trap in lessons-backlog-hygiene: rearming resets position by enabledAt, so leave the PR armed for ordinary queue updates and repair a queue eviction when needed.
2026-11-01 integration on dev621b8f1550: retained current dev ownership/recovery code, all1253census entries plus5prompt nodes and both admission selectors. Corrected the younger SSE task to38328 under the older-arrival rule, retaining characterTASK37600. Fresh private-profile qualification:48character/prompt/model-picker tests pass including all6Apply/Cancel combinations;336CI/census/size contracts pass with3unchanged-dev size overruns covered by the user waiver. Scoped Ruff has no new diagnostics;4changedPython files format clean,3retain existing base formatter debt. Census1258 and taskIDs6041 pass. Independent review found no findings in production/tests and identity delta; finalhead verification is retained beside the worktree. Existing ADR006/095; no new ADR. Full evidence/limits and superseding CI waiver:Docs/superpowers/qa/2026-11-01-pr3306-backlog-integration.md. Historical UAT was not rerun; fresh mounted behavior passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->