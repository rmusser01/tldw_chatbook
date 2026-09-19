---
id: TASK-32756
title: Qualify Speech and TTS Settings keyboard and compact workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 20:37'
updated_date: '2026-09-17 21:13'
labels:
  - ui
  - design-system
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the existing global Speech and TTS configuration workflow with current profile ownership and production styling, repairing confirmed keyboard, layout or draft-recovery gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Speech defaults and provider controls remain keyboard reachable and fully painted in dark/light at wide and compact sizes.
- [x] #2 Local validation, save/revert and guarded navigation retain the intended draft and keep runtime work in Speech Lab.
- [x] #3 Targeted tests and private native journeys qualify the reviewed Settings scope with explicit synthesis and provider limits.
- [x] #4 The Voice value selector and Browse in Speech Lab action remain fully painted together after reflow; no one-row action container clips the selector.
- [x] #5 Horizontal Speech form controls share the available width with their labels, keeping the complete defaults, provider selector and endpoint controls inside the form.
- [x] #6 All three unsaved-changes dialog actions display their complete labels, and Save and continue persists the draft before changing category.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md; backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: existing Settings configuration, credential and runtime ownership remain unchanged.
1. Read the existing Speech task, critique, ownership ADR and production panel; retain current fixture failures.
2. Restore only affected reviewed cases with the existing lifetime-bound private-profile helper, preserving their assertions.
3. Exercise real Settings CSS and keyboard journeys for visible controls, validation/revert, guarded navigation, and compact modal actions. Extend AC before repairing confirmed gaps.
4. Verify native dark/light wide/compact journeys using a private HOME/config/data profile and real local saves, without model downloads or audio generation.
5. Run targeted checks, scoped static analysis and review; record exact evidence and remaining service limits in the completion ledger and draft PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed clipped Voice/Browse controls, horizontal field containers and all unsaved-dialog actions using existing tokens and generated CSS. Restored affected lifetime-bound profile tests and added production-CSS keyboard, resize and Browse handoff coverage. 68 distinct targeted checks, all preflight guards and scoped no-new-debt static checks pass. Four private native dark/light wide/compact journeys verify exact local Save and Save-and-continue writes; 16 captures inspected, 12 databases healthy, default-profile hashes unchanged, normal exit and PID absence verified. QA: Docs/superpowers/qa/2026-09-17-settings-speech/README.md. Existing ADR-039/012/150/161 apply; no new ADR or runtime/provider boundary change. Guide, surface brief, testing lesson and completion ledger updated. Broader provider-specific/realtime workflows remain listed separately.
<!-- SECTION:NOTES:END -->
