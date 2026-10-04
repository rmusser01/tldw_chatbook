---
id: TASK-34362
title: Qualify integrated Console response rules and document evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:56'
updated_date: '2026-10-04 15:57'
labels: []
dependencies:
  - TASK-34361
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 9. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Integrated route tests cover learning, checking, cancellation, shared repair and inert reopen.
- [ ] #2 Provider behavior and native UI evidence are qualified separately from scripted tests.
- [x] #3 Targeted regression and static checks pass and independent whole-branch review findings are addressed or recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: integrated response-rule qualification and private lifecycle boundaries. Test actual composer learning/repair/reopen/exclusion and preserve original actions, run targeted feature and touched-owner/static gates, attempt separately-labelled provider/native-terminal qualification, perform one fresh whole-branch review and document evidence and limitations before task closure.

Latest-dev PR integration (2026-10-04): retain ADR-219; merge origin/dev 1b12df2757, preserve incoming Console model/settings and quit behavior, move unpublished response-rule schema to v77 after the installed v76 notes FTS migration, qualify exact v75/v76 recovery plus the targeted feature/owner/static gates, record remaining qualification limits, then push and open the requested PR against dev. ADR required: no new ADR; ADR path: backlog/decisions/219-console-learned-response-rules.md; reason: integration preserves the approved ownership and runtime contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the integrated Console response-rule journeys and resolved all seven Important whole-branch review findings in one native TDD pass. Initial implementation evidence: 233 feature cases and 62 touched-owner cases passed. Latest-dev publication integrates origin/dev 1b12df2757, retains incoming model picker/Chat Settings/quit behavior and installs response rules as v77 after dev's v76 notes FTS migration; exact v75/v76 restricted recovery is covered. ADR-219 remains authoritative.

Fresh integrated evidence: the 17-file feature run passed 232 cases with three initial-send UI timeouts; the selected owner run passed 222 cases with two UI timeouts and two documented exclusions. Correcting test predicate polling and settling one frame before the next click preserved all assertions and 30-second domain deadlines; the final six-case gate passed all four learn/repair/reopen/exclude journeys and both literal-send cases in 501.32s. Ruff on 70 changed Python files, Black on 34 authored files plus changed legacy ranges, mypy on 12 domain files, final test formatting/lint, diff and generated-bundle checks passed. Governance passed 29 cases with two ratchets identifying unchanged dev sources. The sparse-context-policy Save assertion reproduces with exact current-dev ConsoleChatStore after real profile bootstrap; symlink privilege is unavailable. Failed combined attempts and exclusions remain in Docs/superpowers/reviews/2026-10-03-console-response-rules.md; no whole-suite pass is claimed. Updated source/UI/tests, migration/recovery owners, ADR-219, user guide, implementation evidence and concrete scratch-ACL/UI-synchronization lessons. Repairs retain completed work and original answers, prompt recall excludes machine feedback, promotion preserves body-free calibration and capacity refuses before publication. One independent review, no re-review, no deferred minors. Actual configured provider probe returned ConnectError, visible native-terminal tooling is unavailable and mixed native/external-hook transport qualification remains open. Acceptance criterion 2 stays unchecked and this task remains In Progress. The subsequent user request authorizes push and a draft PR against dev; retain the managed worktree for review feedback.
<!-- SECTION:NOTES:END -->
