---
id: TASK-425
title: >-
  Character chat provider falls back to chat_defaults when character_defaults is
  unset
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-21 20:26'
labels:
  - roleplay
  - ux
  - config
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). P0. The Roleplay preview chat resolves its provider solely from [character_defaults], which ships as Anthropic/claude-3-haiku (config.py:2762). The guided first-run setup writes only [chat_defaults] and [api_settings.*], so a fully onboarded new user (provider tested green, readiness green) gets "anthropic is not ready: Missing API key" on their first character message, and the error steers them toward configuring Anthropic instead of the provider they already set up. No UI can change the character provider today (Settings > Domain Defaults > Personas is a read-only contract page; only the Expert raw-TOML editor works, and no message names the section). Character-flavored generation should inherit the user's working chat defaults when character defaults are absent, and the failure copy should name the real remedy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh profile that configures a provider only via the guided Get-started/Settings flow gets a successful Roleplay preview reply with that provider, with no config file edits
- [x] #2 An explicit [character_defaults] section still wins over chat_defaults when present
- [x] #3 When character-chat provider resolution fails, the error names the resolved provider source and points at an in-app remedy (not a raw TOML section for a different provider)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD in Tests/UI/test_personas_workbench.py: fallback-used / explicit-wins / both-unready cases via _FakePreviewGateway with per-provider readiness
2. Extract selection resolution from PersonasPreviewController._run_reply into a fallback-aware helper: resolve character_defaults selection; if not ready and chat_defaults names a different provider/model, resolve that; use whichever is ready (character wins when both ready)
3. Surface honesty: status shows 'via Console default: <provider>' when falling back; both-unready status keeps gateway copy but appends Settings > Providers & Models remedy
4. Root cause context: first-run writes the full config template including [character_defaults] provider=Anthropic, so presence-in-file cannot distinguish user intent — fallback must be readiness-based
5. Run scoped pytest (venv), live-verify with scratch TLDW_CONFIG_PATH profile reproducing the review's failing state
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Character-flavored preview generation now resolves character_defaults first and, when that provider is not READY, falls back to the user's chat_defaults provider (PersonasPreviewController._resolve_selection_with_fallback). A ready character provider always wins; the fallback keys on readiness, not section presence, because first-run writes the full [character_defaults] template (Anthropic/claude-3-haiku) verbatim so presence-in-file cannot signal user intent (the root cause of the P0). When neither provider is ready the status appends 'Configure a provider in Settings: Providers & Models.'; when the fallback is used the status honestly reads 'via Console default: <provider>' (Running and Ready).

AC mapping: #1 guided-flow user (only chat_defaults written) now gets an in-character reply — live-verified against the exact repro config; #2 explicit ready character_defaults wins (test_ready_character_provider_wins_over_chat_defaults); #3 both-unready status names the resolved blocker copy + the in-app remedy (test_both_providers_unready_names_settings_remedy).

Tests: 3 new cases + _ReadinessMapPreviewGateway double in Tests/UI/test_personas_workbench.py; full test_personas_workbench + test_personas_preview = 189 passed. Live verification: reproduced the review's failing state (character_defaults=Anthropic shipped default, chat_defaults=llama_cpp, no key) in the real TUI; before=silent 'anthropic is not ready' dead end, after=fallback reply 'via Console default: llama_cpp' (drove with a local OpenAI-compatible mock since the :9099 server was down).

Files: tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py, Tests/UI/test_personas_workbench.py. Follow-up task-426 (preview provider/model readout + Settings deep-link) will make the fallback visible before send, not just in status.
<!-- SECTION:NOTES:END -->
