---
id: TASK-33005
title: 'Phase 5: Shared readiness evidence and an explicit cloud key check'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-5
  - readiness
  - console
  - settings
  - ux
dependencies:
  - TASK-33001
  - TASK-33002
  - TASK-33004
references:
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/backlog-adr-check.md'
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'tldw_chatbook/Chat/provider_test_evidence.py'
  - 'tldw_chatbook/Chat/provider_readiness.py'
  - 'tldw_chatbook/Chat/console_session_settings.py'
  - 'tldw_chatbook/UI/Screens/chat_screen.py'
  - 'tldw_chatbook/UI/Screens/settings_screen.py'
  - 'tldw_chatbook/Widgets/Console/console_settings_modal.py'
  - 'tldw_chatbook/Widgets/Console/console_settings_summary.py'
  - 'tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py'
  - 'tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py'
  - 'backlog/decisions/012-provider-credential-settings-boundary.md'
  - 'backlog/decisions/033-application-session-state-ownership.md'
  - 'backlog/decisions/114-llamacpp-lab-console-connection-authority.md'
  - 'Docs/User_Guide/console.md'
  - 'Docs/User_Guide/settings.md'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 5 of the model-configuration redesign "Switchboard with field truth" (qa/model-config-ux-review-2026-09-26/judge-synthesis.md §3 rule 3 and §4 P5; spec §5). It ships as ONE PR. It closes verified findings C2 and C3 (verified-claims.md). The C3 label question is settled by owner decision D2, which the ADR-012 amendment of 2026-09-26 (already drafted) records.

Why. Console readiness is config-only by design: TASK-30011 AC#2 and ADR-114 say "Ready" means no known blocker, so Ready for an untested endpoint is intended. The defect is that connection-test evidence never leaves the surface that produced it.
- There are three stores: the Chat settings modal's (console_settings_modal.py:1304) and Settings' (settings_screen.py:2949, recreated at :13313-13318). Settings also copies its store into a cross-visit snapshot (:12854-12855, restored at :12950-12952).
- The Console's active readiness (chat_screen.py:9853-9869) and its future-chat readiness (chat_screen.py:2846-2859) pass no evidence. So ProviderReadiness.snapshot always yields endpoint "not_tested" (provider_readiness.py:316-319).
- As a result, a known refused llama.cpp keeps reading Ready in the Console while the modal says Not ready. That is a regression against TASK-30011 AC#6.
- build_console_settings_readiness already turns refused evidence into the endpoint_unreachable / retry_connection blocker (console_session_settings.py:1644, pinned by Tests/Chat/test_console_session_settings.py:1169-1182). Only the evidence is missing.

The Settings 't' half. 'Test Provider' never contacts a cloud provider: _provider_live_probe_base_url returns "" outside URL_BASED_PROVIDER_KEYS (settings_screen.py:15343-15363). So a fake OpenAI key reads "configuration is complete" (:15246-15251). ADR-012:33 excluded provider-specific secret validation until the D2 amendment.
- D2: 't' runs an explicit, non-generating, authenticated model listing for cloud providers that reports "key accepted (models listed); generation not tested". Cloud providers are never auto-probed.
- The listing already exists. LocalLLMProviderCatalogService.discover_models accepts staged draft settings (local_llm_provider_catalog_service.py:481-613). The discovery client sends Anthropic's x-api-key header (openai_compatible_model_discovery.py:165-186) and maps 401/403 (:726-737).
- Local providers have a related dead end: the live probe is gated on a model (settings_screen.py:15238, used at :30802), so a URL provider cannot be probed until a model is typed, although listing models is how a first-timer finds one (report persona Jordan).

Constraints.
- Respect the performance limits of task-24454 (readiness is recomputed on the composer keystroke path) and task-32804.3 (the idle 0.25 s credential poll).
- Coordinate with task-32806.1 (placeholder keys, in progress) without absorbing it.
- chat_screen.py has 32 lines of headroom under its screen ratchet (Tests/Architecture/test_screen_size_ratchet.py:85). console_chat_controller.py is already 183 lines over its module ratchet row (Tests/Architecture/test_module_size_ratchet.py:64) at c4225b5d38.
- Target sizes: 211x44 first, then 235x52.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The phase ships as one PR containing every subtask below
- [ ] #2 A refused or timed-out connection test in Chat settings or Settings turns the Console's readiness for the same connection to Not ready, with that reason and a retry action, without a restart (C2 closed)
- [ ] #3 A connection with no evidence still reads Ready (qualified 'not tested'), and reading readiness never starts a network request (TASK-30011 AC#2 preserved)
- [ ] #4 Settings 't' checks a cloud provider's key with an authenticated model listing and reports accepted, rejected or unavailable without ever generating (C3 closed per D2)
- [ ] #5 Settings 't' lists a URL-based provider's models even before a model is chosen
- [ ] #6 Behaviour matches the ADR-012 amendment of 2026-09-26, including its outcome table and its rule that a public listing never reads as a key check
- [ ] #7 The Console status row, rail Model section, setup card, model switcher, Chat settings and Settings show the same readiness words for the same connection at the same moment (TASK-30011 AC#6 restored)
- [ ] #8 No cloud provider is contacted except by an explicit 't'
- [ ] #9 Composer keystroke cost and the idle credential poll's per-tick cost do not rise, measured by the methods recorded in task-24454 and task-32804.3
- [ ] #10 chat_screen.py stays within its screen-size ratchet row, console_chat_controller.py gains no lines, and the ADR-097 boot ratchets (boot CSS bytes, _ui_ready module census) do not rise
- [ ] #11 Live evidence at 211x44 and 235x52 comes from a scratch profile (TLDW_CONFIG_PATH; the real ~/.config/tldw_cli is never touched). It shows three cases: a stopped llama.cpp reads 'Not ready · refused' in the switcher and status row; 't' on a valid cloud key shows 'Ready · verified HH:MM' in Settings and Console; 't' on a rejected key shows 'Not ready · key rejected'
- [ ] #12 Docs/User_Guide pages updated: console.md (readiness words), and settings.md (the Test Provider section near :228-232, the first-run steps near :816-817 and the 't' key row near :869)
- [ ] #13 ./scripts/preflight.sh passes
<!-- AC:END -->
