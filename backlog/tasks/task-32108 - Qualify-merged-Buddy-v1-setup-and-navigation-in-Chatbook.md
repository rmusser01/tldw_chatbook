---
id: TASK-32108
title: Qualify merged Buddy v1 setup and navigation in Chatbook
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:32'
updated_date: '2026-09-09 05:41'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the merged independent Buddy and Persona journeys using disposable fresh and upgraded profiles, retaining exact conversation ownership and navigation continuity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh and upgraded disposable profiles exercise Buddy selection without a Persona, conversation and workspace attachment, Persona defaults, and Static/Dynamic presentation.
- [ ] #2 Cold entry and cross-screen replies preserve exact conversation identity, drafts, running work, and workspace inbox semantics.
- [x] #3 Source-bound rendered evidence, profile isolation, targeted checks, and any remaining native or human-voice verification gaps are recorded without overstating coverage.
- [ ] #4 Any reproduced defect has a focused regression and scoped static verification before a completion claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md. Reason: qualify existing merged Buddy behavior. Pin current dev and isolate config/data/keyring before launch. Exercise fresh and upgraded database fixtures, cold Buddy entry, management and cross-screen scoped replies with targeted production-app tests and rendered captures. Record source hashes, exact checks and real versus simulated UI coverage. No physical microphone, paid provider, full test suite, or normal-profile writes. Qualification implementer owns Chatbook notes and Docs/Reviews/2026-09-09-buddy-v1-qualification.md; root owns native integration and publication. Keep the task In Progress until native integration coverage is resolved.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified unchanged merged source5655c4820733d24754f872c21cc6196d83bf1504 under ADR139 with177 targeted tests passing:142 Buddy/database/management/cold-entry/draft/workspace/layout tests,33 atomic Persona/default assignment tests and2 new headless rendered fresh/schema69→70 upgrade journeys. Both real-app capture profiles preserve independent Buddy ownership and Persona list, exact conversation binding, explicitNone workspace default and settled inbox; Static stays on one frame and Dynamic completes0→1→2→3→0 using real bundled artwork. Config/data/HOME and null keyring are isolated before launch. Tests/UI/test_buddy_v1_qualification_capture.py is the new harness; source hashes,4 representative original SVGs and detailed limits are in Docs/Reviews/2026-09-09-buddy-v1-qualification.md and adjacent artifacts. Ruff/formatter/compile/whitespace checks pass; no production code changed. Initial harness default-screen query and premature loading-state capture were corrected without production changes. No physical native-terminal walkthrough, microphone, audible playback, real provider, normal-profile access or full suite claimed. Keep In Progress and AC uncompleted pending root native integration coverage.

Root reconciliation: source-bound evidence and explicit coverage limits are recorded, so AC3 is verified. Native Terminal selection was explicitly rejected by Computer Use policy; native Chrome separately lacks Computer Use permissions. No workaround was attempted. The permitted in-app browser qualified the separate server fresh WebUI journey, which does not replace native Textual evidence. Keep the task In Progress and remaining native acceptance open. ADR-139 continues to govern the unchanged production implementation.

Rebased qualification commit onto updated dev 2e3389e694e93592a1c66e5c3416bf29a1057d6c after TTS recovery changes landed. Refreshed two rendered profile journeys plus eleven guarded-speech cases:13 passed,3 warnings in39.04s; exact command/source hashes/log retained in after-rebase-verification.json. Prior177-test evidence stays attributed to its original snapshot, with repeated cases not counted twice. Normalized whitespace-only lines in four retained SVG copies, retaining both raw/export and repository hashes; raw exports remain in scratch storage. No production change and no native/audio completion claim.

Published draft PR https://github.com/rmusser01/tldw_chatbook/pull/2536 against dev. Production remains unchanged; refreshed automated evidence is included and native qualification remains open. No merge performed.
<!-- SECTION:NOTES:END -->
