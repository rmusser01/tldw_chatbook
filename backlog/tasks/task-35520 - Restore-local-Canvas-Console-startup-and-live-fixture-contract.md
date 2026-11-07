---
id: TASK-35520
title: Restore local Canvas Console startup and live fixture contract
status: Done
created_date: 2026-10-18 15:28
assignee:
- '@codex'
labels:
- canvas
- console
- bug
- uat
priority: high
references:
- backlog/tasks/task-25907.23 - Integrate-Personal-Context-memory-improvements-onto-current-dev.md
documentation:
- Docs/Canvas/UAT_2026-10-18.md
updated_date: 2026-10-18 16:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unblock the native and served Canvas user journeys authorized on 2026-10-18 without resuming synchronization. The existing Console resize handler crashes when its grid is present but rails are absent; qualify that lifecycle boundary and restore any demonstrated stale live-provider fixture contract instead of weakening production startup or metadata validation. This is the Canvas-scoped qualification of the lifecycle defect already recorded by TASK-25907.23, not completion of that broader integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A partially composed or torn-down Console subtree cannot crash on Resize, mutate rail preferences or consume the width-band transition needed by the next complete layout; fully mounted responsive behavior remains intact.
- [x] #2 The Canvas scripted provider fixture satisfies the actual offline metadata contract needed by the real Console; startup reaches the usable composer without suppressing composition failures or weakening provider readiness.
- [x] #3 The native create/automatic-browser-open journey and all four actual served Console create/update/history-reopen variants pass against disposable data, preserving exact origins, revisions and no-provider reopen behavior.
- [x] #4 Focused regressions demonstrate failure before repair and successful complete-layout/retry behavior afterward, existing source-admission guards and runtime authority remain intact, documentation records the final UAT result and server/client sync tasks stay paused.
- [x] #5 The new partial-layout regression is selected exactly once by an admission-sensitive required PR lane, and the existing CI contract guard prevents silent removal.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; existing ADR-121 Canvas ownership/sandbox and ADR-150/161 UI lifecycle/component contracts apply. Reason: routine bug and fixture-contract correction with no new policy, storage, interface or capability. 1. Trace the real compose fallback and Resize callers; preserve prior native/served RED receipts and add an actual partial-subtree regression before source edits. 2. Make the minimal complete lifecycle guard and width-band retry correction. 3. Diagnose and test the scripted provider offline metadata contract; correct only a demonstrated fixture omission. 4. Run the exact native/served owning journeys, responsive regressions and scoped static checks; independently review and record evidence. No dependency install, full-suite, normal-profile launch, server change, sync outbox/inbox work, push or merge.
Independent review found the new short mounted regression had no PR gate. Before integration, register it in the existing admission-sensitive PR Fast Lane and pin it through the existing Console regression-gate contract; demonstrate missing-selection RED, then run Tests/CI plus the original focused module. No new CI job or timeout change.
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md (existing); no new ADR required.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Guarded the complete Console grid/rails/handles before consuming a resize transition or scheduling header sync, preserving retry and preferences. Reused the existing ready-provider offline cached_context_window contract in the Canvas scripted gateway; retained the admitted private profile only for exact real-config consumers. Added five genuine partial-subtree RED/GREEN regressions and pinned their existing admission-sensitive required PR selection. Final evidence: original41 UAT passed; combined121 affected regressions passed;994 CI contracts passed with one intentional nested-child skip; independent review has no remaining findings. The supplemental Inspector initial-focus failure reproduces with the committed handler and stays with the existing broader Console owners; no whole-UI pass claim. Existing ADR121 applies: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md. See Docs/Canvas/UAT_2026-10-18.md and its preserved initial checkpoint. Server/client sync stays paused; no push/merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
