---
id: TASK-33005
title: 'Phase 5: Shared readiness evidence and an explicit cloud key check'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-26 11:47'
updated_date: '2026-10-02 08:30'
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
- [x] #1 The phase ships as one PR containing every subtask below
- [x] #2 A refused or timed-out connection test in Chat settings or Settings turns the Console's readiness for the same connection to Not ready, with that reason and a retry action, without a restart (C2 closed)
- [x] #3 A connection with no evidence still reads Ready (qualified 'not tested'), and reading readiness never starts a network request (TASK-30011 AC#2 preserved)
- [x] #4 Settings 't' checks a cloud provider's key with an authenticated model listing and reports accepted, rejected or unavailable without ever generating (C3 closed per D2)
- [x] #5 Settings 't' lists a URL-based provider's models even before a model is chosen
- [x] #6 Behaviour matches the ADR-012 amendment of 2026-09-26, including its outcome table and its rule that a public listing never reads as a key check
- [x] #7 The Console status row, rail Model section, setup card, model switcher, Chat settings and Settings show the same readiness words for the same connection at the same moment (TASK-30011 AC#6 restored)
- [x] #8 No cloud provider is contacted except by an explicit 't'
- [x] #9 Composer keystroke cost and the idle credential poll's per-tick cost do not rise, measured by the methods recorded in task-24454 and task-32804.3
- [x] #10 chat_screen.py stays within its screen-size ratchet row, console_chat_controller.py gains no lines, and the ADR-097 boot ratchets (boot CSS bytes, _ui_ready module census) do not rise
- [x] #11 (owner ruling 2026-10-02, TASK-33005.6 AC#1: the documented local stand-in is accepted for the valid-cloud-key case) Live evidence at 211x44 and 235x52 comes from a scratch profile (TLDW_CONFIG_PATH; the real ~/.config/tldw_cli is never touched). It shows three cases: a stopped llama.cpp reads 'Not ready · refused' in the switcher and status row; 't' on a valid cloud key shows 'Ready · verified HH:MM' in Settings and Console; 't' on a rejected key shows 'Not ready · key rejected'
- [x] #12 Docs/User_Guide pages updated: console.md (readiness words), and settings.md (the Test Provider section near :228-232, the first-run steps near :816-817 and the 't' key row near :869)
- [x] #13 ./scripts/preflight.sh passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** Connection-test evidence now has one process-memory owner on the app (ADR-033, TTS precedent), keyed by provider, endpoint, credential source and a salted digest of the key a send uses. Every model surface reads it through one readiness mapping, so a refused or timed-out test anywhere turns the Console Not ready with a retry, and Settings 't' checks a cloud key with one authenticated, non-generating model listing (D2, ADR-012 amendment of 2026-09-26). Branch `feat/model-config-p5-readiness`, rebased onto dev `92a95170a5` (the P4 commits, merged as #2947, were dropped from the branch).

**Subtasks and commits** (hashes after the rebase; each subtask's notes hold its rewritten tests and evidence):
- Plan: `a958655d86`.
- TASK-33005.1, one owner of connection evidence: `8fec4e1827`, `93faf78397`.
- TASK-33005.2, Console readiness honours evidence and Retry connection re-tests in place: `162e237d45`, `d2ba760d75`.
- TASK-33005.3, one readiness vocabulary: `031953729c`, `87f662edd2`.
- TASK-33005.4, Settings 't' key check and listing with no model: `f6692498c1`, `c8a7fc3123`, `fccbf1cc40`.
- TASK-33005.5, Switch model probes local servers: `3501724906`, `90c43708f7`.
- Final fix wave (commit "fix(model-config): Phase 5 final review fixes (TASK-33005)"): the final review's I-1..I-6 plus its cheap minors, below.

**How two ACs were read.** AC#1: the five subtasks are on this one branch, for one PR; riders .6-.10 are follow-ups, not part of it. AC#10: the chat_screen.py row was already red on dev (25,311 against 25,218) before this branch, so it was read as the plan's ruling, nets <= 0 lines and methods; it nets -16 lines and 0 methods.

**Delegations.** AC#11's valid-cloud-key case has no real key behind it (a local stand-in, documented in qa/model-config-p5-2026-10-01/task-4/README.md); that, keyless "verified" and shipped-default probing were owner rulings, delegated to TASK-33005.6. The owner ruled on 2026-10-02: the stand-in is accepted (so AC#11 is ticked), a keyless successful paid test reads "verified" (already so), and shipped localhost defaults stay probed but an untouched one that refuses is listed quietly as not running. TASK-33005.6 is Done and records the rulings. The rejected-key case is live at 211x44 (task-4 captures 01-02) and 235x52 (final-fix captures 6-7); the stopped llama.cpp case at both sizes (task-5, final-fix).

**Riders filed** (not in this PR): TASK-33005.6 owner rulings; .7 the four remaining readiness copies (Settings "Readiness: provider / model" rows, Inspector "Provider configuration required", composer copy after a rejected key, the empty-transcript Ready line); .8 Settings rows keep "(draft)" after Save until a return visit; .9 Home, Library lock pre-check and the first-run wizard stay config-only; .10 unmeasured warm-switch cost, `_BEGIN_ORDER` without a lock, Save rebuilding the saved identity, three test gaps.

**Final fix wave.**
- Rebase (I-2): `git rebase --onto origin/dev 5b5db496cf`. Conflicts: chat_screen keeps dev's TASK-33620.4 call with no `active_run` and adds `connection_evidence=`; console_settings_summary keeps dev's active-run detail under the one word; settings_screen imports both names; the modal active-run test keeps both sets of asserts. CSS bundle regenerated, not merged.
- Boot CSS (I-1): the rail readiness line reuses `.conversation-attention-error` (the same readable error token) instead of its own `#console-model-section-recovery.-blocked` rule; its unused `console-model-section-recovery` class is gone. Bundle 74 B smaller than base, boot CSS about 608,014 B against the 608,090 B limit.
- Keystroke census (I-3): green after the rebase with no change for it. dev's `405643f836` (TASK-33374) had already proved that row timing-coupled and bounded it at <= 1; measured empty 0 / 400 messages 0.
- Cloud retry dead end (I-4): a failed cloud key check's Retry connection opens Settings ▸ Providers & Models at that provider and says "Press t to test <provider> again"; the switcher row reads "Enter: open Settings" and Enter opens it, instead of "start it; rechecked on open". A 429, 5xx or other non-auth HTTP answer to 't' is "model listing unavailable" and blocks nothing.
- Surfaces under the switcher (I-5): each switcher probe result also refreshes the Console summary and control bar beneath it (the idle poll skips a covered screen), so the header and the switcher agree while it is open.
- Vacuous pin (I-6): the census counts readiness builds at every import binding (home module, ChatScreen, defaults), holds the 0.25 s credential poll still during every burst, and fails on zero: 25 builds per 24 keys on a quiet run. The scale test bounds that count per key (0 < n/key <= 3) on both sides instead of requiring equality, because the trailing draft repaint builds readiness too and, with six runs in parallel, the 400-message run read 28-34. The task-33005.2 claim is corrected in place.
- Cheap minors: a debug log in the Settings key-check catch-all and in each switcher probe (diagnostic inventory reviewed and regenerated: the two new calls log the provider key and the exception class only, no URL, key or user text); `_recently_tested` rejects a future timestamp; NEEDS SETUP heading "Enter opens the fix or explains it"; keyless qwencloud pinned never-probed; the AC#15 docstring says its prober is stubbed; the lessons paragraph no longer names deleted code.

**Tests rewritten on purpose in the final wave:** Tests/UI/test_console_active_run_readiness_surfaces.py (`_model_recovery_line`: a healthy run reads "Ready ·" and never red, a missing key reads "Not ready ·" and red; negative control: putting `active_run` back turns it red), Tests/UI/test_console_session_settings.py (active-run modal test, dev and P5 asserts merged), Tests/UI/test_settings_endpoint_probe.py (request_failed + http_status now model_listing_unavailable), Tests/UI/test_console_rail_color_grammar.py, test_console_endpoint_discovery.py and test_console_resize_reflow.py (the error class name), Tests/UI/test_console_switcher_local_probe.py (heading; header asserted while the switcher is open, negative control: without the refresh it fails), Tests/Performance/test_console_keystroke_work_census.py (I-6). New: `test_a_cloud_key_check_failure_retries_in_settings_not_chat_settings`, `test_a_cloud_row_whose_key_check_timed_out_opens_settings`.

**Behaviour changes for the PR description.**
- The Console honours any connection test this session: refused or timed out reads "Not ready · refused :PORT" / "timed out" everywhere, with Retry connection re-testing a local or URL server in place.
- One word on every model surface: "Ready · not tested", "Ready · reachable HH:MM", "Ready · verified HH:MM", "Not ready · <reason>"; the rail Model line always shows it (red only when Not ready).
- Settings 't' checks a cloud key with one model listing (accepted, rejected, unavailable; OpenRouter/NVIDIA public lists never count), and lists a URL provider's models before a model is chosen.
- Switch model probes keyless local servers on loopback or private addresses (3 at a time, 10 s reuse); nothing cloud or public is contacted automatically.
- A failed cloud key check's Retry connection opens Settings at that provider; a non-auth HTTP answer to 't' never blocks sending.

**Capture checkpoint fix wave (2026-10-02).** The full-screen captures (qa/model-config-33005-captures) found two surfaces contradicting the Readiness row; both are fixed, with the AC#3 ruling, in "fix(model-config): one readiness truth on Settings rows; quiet refusals for untouched defaults (TASK-33005)":
- Settings' Key row read "present, not verified" under "Not ready · key rejected"; it now says "key rejected" from the same 401/403 evidence (`test_a_rejected_key_reads_rejected_on_the_key_row_too`).
- The Overview "Status:" was config-only ("Ready" above a refused "Last connection test"). It now reads the Console's future-chat readiness through `build_console_settings_readiness` with the shared evidence and `readiness_words` (`test_settings_overview_status_reads_a_refused_test_as_the_console_does`). Rewritten on purpose to the one vocabulary: the three TASK-31805 Overview pins in Tests/UI/test_settings_configuration_hub.py ("Not ready · no key", "Ready · not tested", "Not ready · no model"), three in Tests/UI/test_settings_custom_endpoint_default.py ("unsupported", "check settings") and the "Checking Claude" waits in Tests/UI/test_settings_subscription_readiness.py ("Not ready · checking login").
- Switch model lists a refused, untouched shipped default once under NOT RUNNING instead of NEEDS SETUP (TASK-33005.6 AC#3).
Riders filed from the checkpoint: TASK-33005.11 ("verified" for a model the key check did not list), .12 (switcher rows ignore the models a probe listed), .13 ("not saved this session" counts per visit), .14 (Test Provider looks like a heading, provider names cut at 20 columns, "~4k" vs "Context unknown"). Live: qa/model-config-33005-captures/fix/.

**Pre-existing reds (not this branch).** ADR-126 `RecoveryRequired: raw_source_selection_changed` locally in clean worktrees (failure-name sets compared instead). Size rows red at base `92a95170a5` and unchanged: chat_screen.py and library_screen.py screen rows; console_chat_controller, console_chat_store, mcp_workbench, llm_screen, personas_screen, watchlists_collections_screen, FirstRunSetupWizard, console_transcript, tldw_api/client module rows (11 names, identical at base and wave). `test_css_class_coverage_contract` red on dev with the same message.

**Verification (final wave).** See the final-review "Final fix wave" section for the commands. Ratchets: chat_screen.py 25,295 lines (dev 25,311, nets -16, no new method), console_settings_modal.py 7,751 (<= 7,761), console_chat_controller.py untouched, `_ui_ready` census and boot CSS green. Covering suites: failure names at the wave compared with dev. Live (scratch profile, real profile hash unchanged): qa/model-config-p5-2026-10-01/final-fix/. `./scripts/preflight.sh` rc 0.
<!-- SECTION:NOTES:END -->
