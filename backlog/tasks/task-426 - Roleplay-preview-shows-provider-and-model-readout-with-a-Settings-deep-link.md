---
id: TASK-426
title: Roleplay preview shows provider and model readout with a Settings deep-link
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 16:08'
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
Implemented the Personas preview provider/model readout and Configure deep-link within the existing provider-aware Settings boundary. Centralized provider/model normalization so display, send resolution, navigation targeting, and logs share effective selections; added fallback/resolved-provider copy; documented the pane API; and covered whitespace-only defaults plus inherited provider models. After rebasing onto dev 0f5904e6, final integration review found TASK-434 could restore preview state without a character-load event, so restore_conversation now refreshes the readout directly and a no-load navigation-restore regression verifies both visible text and the Configure provider. Verification: 206 focused Personas workbench/preview/persistence tests passed; Ruff passed on all task code (the sole full-screen F821 is inherited unchanged from dev at personas_screen.py:4215); mypy passed for the controller; production files compile; git diff --check passed; duplicate guard passed across 595 task files. Independent re-review found no Critical or Important issues. ADR required: no. Existing ADR-004, ADR-006, ADR-007, and ADR-012 govern this corrective slice. Modified files: Personas preview controller/pane/messages/screen, focused UI tests including restore coverage, and TASK-426.
<!-- SECTION:NOTES:END -->
