---
id: TASK-426
title: Roleplay preview shows provider and model readout with a Settings deep-link
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-21 21:18'
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
Stacked on task-425 branch (needs _resolve_selection_with_fallback).
1. TDD in Tests/UI/test_personas_workbench.py: (a) provider_readout() sync formatting — char+distinct-chat shows fallback note; char-only no note; no-char-provider falls to chat/none; (b) readout Static populated after selecting a character; (c) Configure button posts NavigateToScreen settings/providers-models with char provider; (d) post-send readout reflects resolved (fallback) provider.
2. Pane (personas_preview_pane.py): add #personas-preview-provider Static at top of body; set_provider_readout(text); 'Configure' Button in toolbar posting PreviewConfigureProviderRequested.
3. Message: add PreviewConfigureProviderRequested to personas_pane_messages.py.
4. Controller: provider_readout()->(text,nav_provider) sync from config using PROVIDER_DISPLAY_NAMES; refresh_provider_readout() sets pane; call from handle_character_loaded/reset_for_character/reset; open_provider_settings() posts NavigateToScreen; in _run_reply update readout to resolution.provider/model (+Console default when fallback).
5. Screen: @on(PreviewConfigureProviderRequested)->preview.open_provider_settings().
6. Scoped pytest + live-verify readout + Configure nav in real TUI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stacked on task-425. Added a provider/model readout line at the top of the preview pane body (#personas-preview-provider) plus a Configure button in the toolbar.

Readout (PersonasPreviewController.provider_readout): computed synchronously from config — no readiness probe — mirroring 425's resolution order. Shows the character_defaults provider/model that replies try first and, when chat_defaults names a distinct provider, the Console-default fallback target ('- Console default if unavailable: <provider>'). With no character provider it shows the chat default as '(Console default)'. Recomputed on every seed (reset + handle_character_loaded), so config changes reflect on the next selection (AC #3). After a real reply resolves in _run_reply, the readout is repainted with the provider/model that actually answered (free — we already resolved), so it reflects the fallback rather than intent (AC #1). Provider keys → display names via PROVIDER_DISPLAY_NAMES.

Configure (AC #2): pane posts PreviewConfigureProviderRequested → screen → controller.open_provider_settings() posts NavigateToScreen('settings', {category: PROVIDERS_MODELS, provider: <char provider>}). Settings' own unsaved-changes guard may suppress the provider preselect (pre-existing), but navigation to Providers & Models is reliable.

Tests: 5 new cases in test_personas_workbench.py (readout content for char+fallback / same-provider / no-char-provider; Configure navigation context; post-send resolved readout). Full test_personas_workbench + test_personas_preview = 195 passed. Live-verified in the real TUI: readout renders 'Provider: Anthropic / claude-3-haiku - Console default if unavailable: llama.cpp', flips post-send to 'Provider: llama.cpp / local-gemma.gguf (Console default)', and Configure lands on Settings > Providers & Models.

Files: personas_preview_controller.py, personas_preview_pane.py, personas_pane_messages.py, personas_screen.py, Tests/UI/test_personas_workbench.py.
<!-- SECTION:NOTES:END -->
