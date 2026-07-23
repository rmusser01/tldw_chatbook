---
id: TASK-426
title: Roleplay preview shows provider and model readout with a Settings deep-link
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 13:46'
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
Stacked on task-425. Added a provider/model readout line at the top of the preview pane body (#personas-preview-provider) plus a Configure button in the toolbar.

Readout (PersonasPreviewController.provider_readout): computed synchronously from config — no readiness probe — mirroring 425's resolution order. Shows the character_defaults provider/model that replies try first and, when chat_defaults names a distinct provider, the Console-default fallback target ('- Console default if unavailable: <provider>'). With no character provider it shows the chat default as '(Console default)'. Recomputed on every seed (reset + handle_character_loaded), so config changes reflect on the next selection (AC #3). After a real reply resolves in _run_reply, the readout is repainted with the provider/model that actually answered (free — we already resolved), so it reflects the fallback rather than intent (AC #1). Provider keys → display names via PROVIDER_DISPLAY_NAMES.

Configure (AC #2): pane posts PreviewConfigureProviderRequested → screen → controller.open_provider_settings() posts NavigateToScreen('settings', {category: PROVIDERS_MODELS, provider: <char provider>}). Settings' own unsaved-changes guard may suppress the provider preselect (pre-existing), but navigation to Providers & Models is reliable.

Tests: 5 new cases in test_personas_workbench.py (readout content for char+fallback / same-provider / no-char-provider; Configure navigation context; post-send resolved readout). Full test_personas_workbench + test_personas_preview = 195 passed. Live-verified in the real TUI: readout renders 'Provider: Anthropic / claude-3-haiku - Console default if unavailable: llama.cpp', flips post-send to 'Provider: llama.cpp / local-gemma.gguf (Console default)', and Configure lands on Settings > Providers & Models.

Files: personas_preview_controller.py, personas_preview_pane.py, personas_pane_messages.py, personas_screen.py, Tests/UI/test_personas_workbench.py.
<!-- SECTION:NOTES:END -->
