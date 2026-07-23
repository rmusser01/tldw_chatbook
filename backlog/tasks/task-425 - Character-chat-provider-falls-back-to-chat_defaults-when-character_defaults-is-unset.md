---
id: TASK-425
title: >-
  Character chat provider falls back to chat_defaults when character_defaults is
  unset
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 05:04'
labels:
  - roleplay
  - ux
  - config
dependencies: []
references:
  - backlog/decisions/006-provider-aware-generation-settings.md
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
- [x] #4 Fallback and primary preview selections preserve configured endpoint, streaming, and generation defaults.
- [x] #5 Resolution failures retain safe structured provider/model/selection context, and changed test helpers satisfy the repository class-naming rule.
- [x] #6 The default CI test environment installs the existing Markdown dependency required to collect subscription prompt tests on every matrix job.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase the PR branch onto current origin/dev and verify the pre-change targeted baseline.
2. TDD: extend Roleplay preview tests so configured character and chat defaults, non-default llama.cpp endpoints, streaming, and sampling fields fail against the current provider/model-only selections.
3. TDD: add a provider-resolution exception regression that requires the existing safe structured preview log context.
4. Reuse build_default_console_session_settings to derive each defaults section, map the effective snapshot into ConsoleProviderSelection, and log failures at the selection that raised.
5. Rename the private test gateway helper to satisfy the PascalCase review rule.
6. Run focused tests, the full affected Roleplay test pair, static checks, and git diff hygiene; update implementation notes and acceptance criteria.
7. Reproduce the rebased dev matrix collection failure, align requirements-test.txt with the already-declared Markdown dependency imported by subscription prompt tests, and rerun the full GitHub matrix.

ADR required: no
ADR path: backlog/decisions/006-provider-aware-generation-settings.md (existing for the Roleplay settings change); N/A for the test-manifest correction
Reason: The Roleplay fix applies an existing provider-settings boundary. The CI correction adds no dependency or runtime choice; it exposes the already-declared subscriptions Markdown dependency to the test environment that imports it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Character-flavored preview generation still resolves character_defaults first and falls back to the user's chat_defaults provider only when the character provider is not ready. Review remediation now derives both selections through build_default_console_session_settings and maps the resulting endpoint, streaming, sampling, token, and reasoning fields into ConsoleProviderSelection. Character defaults are presented through the same established settings boundary so their own generation values remain authoritative; a ready character provider still wins.

Resolution exceptions are logged at the exact primary or fallback selection that raised, using _reply_log_context for the same safe operation/provider/model/entity/generation/streaming fields as later provider failures. The new test gateway helper was renamed to satisfy the PascalCase review rule.

ADR required: no. Existing ADR: backlog/decisions/006-provider-aware-generation-settings.md. The change applies its existing Settings-persistence / Console-effective-resolution boundary and adds no new storage, provider, security, or runtime contract.

AC mapping: #1 and #4 are covered by test_unready_character_provider_falls_back_to_chat_defaults, including a non-default llama.cpp endpoint plus streaming/sampling/token defaults; #2 and #4 are covered by test_ready_character_provider_wins_over_chat_defaults, including character-specific generation values; #3 remains covered by test_both_providers_unready_names_settings_remedy; #5 is covered by test_resolution_failure_logs_safe_preview_context and the PascalCase helper rename.

Verification: TDD red run produced the three expected failures (missing structured context, endpoint None, streaming True); the same focused set then passed 3/3. Tests/UI/test_personas_workbench.py plus Tests/UI/test_personas_preview.py passed 190/190 on rebased dev. Ruff passed both changed Python files; mypy passed personas_preview_controller.py; git diff --check and the 593-task duplicate-ID guard passed. The repository-wide command successfully collected 13,372 tests after installing the declared subscriptions extra locally, then the redundant serial run was stopped at 5%; the exact pushed SHA is gated by the repository's parallel GitHub unit/integration/UI jobs before merge.

Files: tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py, Tests/UI/test_personas_workbench.py, and this TASK-425 record.

Rebase CI follow-up: the refreshed matrix reproduced the current dev baseline failure on every job during collection: Tests/Internal_Prompts/test_subscriptions_migration.py imports Subscriptions.briefing_generator, whose existing top-level markdown import was not installed by requirements-test.txt. Recent dev run 29952753854 failed for the same reason across unit/integration platforms, and no overlapping PR existed. Added markdown to requirements-test.txt, matching the already-declared subscriptions extra without changing runtime ownership. The focused subscription prompt suite passes 14/14 locally; the next pushed SHA reruns the full matrix. Local pip check still reports pre-existing textual-web/textual and uvloop version conflicts in the shared developer venv; those packages and constraints are unrelated to this PR and are not changed.
<!-- SECTION:NOTES:END -->
