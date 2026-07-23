---
id: TASK-425
title: Character chat provider falls back to chat_defaults when character_defaults is unset
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
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
- [ ] #1 A fresh profile that configures a provider only via the guided Get-started/Settings flow gets a successful Roleplay preview reply with that provider, with no config file edits
- [ ] #2 An explicit [character_defaults] section still wins over chat_defaults when present
- [ ] #3 When character-chat provider resolution fails, the error names the resolved provider source and points at an in-app remedy (not a raw TOML section for a different provider)
<!-- AC:END -->
