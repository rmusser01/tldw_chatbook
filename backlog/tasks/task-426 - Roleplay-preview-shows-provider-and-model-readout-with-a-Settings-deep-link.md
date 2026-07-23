---
id: TASK-426
title: Roleplay preview shows provider and model readout with a Settings deep-link
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 14:08'
labels:
  - roleplay
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The preview pane gives no indication of which provider/model will answer (personas_preview_controller.py:270-276 hardwires character_defaults) and offers no way to inspect or change it. Console solves this with a Model section + Configure link; the preview should at least show the resolved provider/model and link to Settings > Providers & Models. Complements task-425.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview pane displays the resolved provider and model that character replies will use
- [x] #2 A visible affordance navigates to the provider configuration surface
- [x] #3 Readout updates when the resolution changes (settings saved, config reloaded)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/004-personas-destination-native-workbench.md; backlog/decisions/007-personas-workbench-route-consolidation.md; backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/012-provider-credential-settings-boundary.md
Reason: This is corrective normalization, documentation, and UI verification within the existing Personas preview, provider-default, and Settings navigation boundaries; it introduces no new storage, runtime, security, dependency, or long-lived UX decision.

1. Replay only the task-426 commit onto current origin/dev, preserving the merged task-425 implementation and resolving conflicts against its reviewed final shape.
2. Retarget PR #746 from the retired stacked base branch to dev and verify the diff contains only task-426 work plus review remediation.
3. Add a focused failing regression proving whitespace-only character provider/model values are treated as unset and the readout plus Configure target match actual send-path normalization.
4. Implement the smallest controller normalization fix and add the required Google-style Args section to set_provider_readout.
5. Run focused Personas preview/workbench tests, lint, type checks, diff checks, and task/backlog validation.
6. Commit and push the rebased remediation, reply to and resolve every review thread, then request an independent code review.
7. Monitor final GitHub checks, compare any inherited failures with the exact dev baseline, verify mergeability and unresolved-thread state, then merge through the normal GitHub path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased the task-426 feature directly onto current dev after task-425 merged, so PR #746 contains only the provider/model preview readout, Settings deep-link, and review remediation. Preserved the final task-425 readiness gateway shape while resolving the stacked-branch conflict.

Review remediation: added Google-style Args documentation to PersonasPreviewPane.set_provider_readout(). Provider and model normalization now happens in the shared _selection_from_defaults send path, and provider_readout derives both character and Console-default labels from those effective selections. This keeps display text, Settings navigation, logs, and actual sends aligned, including whitespace-only defaults and models inherited from api_settings.

TDD evidence: the whitespace end-to-end and inherited-provider-model cases both failed before the shared-path fix, then passed after it. The tests submit real preview requests through ReadinessMapPreviewGateway and assert the normalized/effective ConsoleProviderSelection values, fallback behavior, rendered readout, and Configure target.

Verification: Tests/UI/test_personas_workbench.py + Tests/UI/test_personas_preview.py = 197 passed; Ruff passed on all changed test/controller/widget files; mypy passed for personas_preview_controller.py; compileall and git diff --check passed; Backlog Guard found no duplicate IDs across 593 task files. Independent pre-merge review found no Critical issues; its two Important alignment findings were fixed and covered by the new end-to-end assertions.

ADR required: no. Existing boundaries remain governed by backlog/decisions/004-personas-destination-native-workbench.md, 006-provider-aware-generation-settings.md, 007-personas-workbench-route-consolidation.md, and 012-provider-credential-settings-boundary.md.

Modified files: personas_preview_controller.py, personas_preview_pane.py, personas_pane_messages.py, personas_screen.py, Tests/UI/test_personas_workbench.py, and this task record.
<!-- SECTION:NOTES:END -->
