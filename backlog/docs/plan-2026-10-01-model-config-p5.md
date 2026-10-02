# Plan: Phase 5: Shared readiness evidence and an explicit cloud key check (TASK-33005)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33005 - Phase-5-Shared-readiness-evidence-and-an-explicit-cloud-key-check.md

## Phase goal (parent)

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

### Parent acceptance criteria

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

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p5. Start EVERY shell command with `cd <that path> &&`. Never touch the main checkout (another session's uncommitted work).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental: compare failure-NAME sets against origin/dev.
- Size ratchets never rise (ADR-097); console_settings_modal.py net lines <= 0 against its current row.
- Geometry only through tokens in css/core/_variables.tcss (ADR-150/161); rebuild the CSS bundle with the repo script.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer hints must match working bindings.
- TDD; real-implementation tests for config/provider surfaces; rewrite pinned tests on purpose and name them.
- Commit per task `fix|feat(model-config): <summary> (TASK-33005.N)` + `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in each subtask file. Update Docs/User_Guide CONTENT; never add "Verified against" paragraphs (record verification in task notes).
- Lessons (backlog/docs/lessons-*.md): insert new entries MID-FILE near a related section, never appended at the end of the file (dev appends there constantly and every rebase conflicts).
- Evidence: commit captures (.txt/.ansi.txt) under qa/, but NEVER commit one-off driver or probe scripts (.sh/.py with machine-specific paths); describe the capture procedure in a short qa README instead. Captures are reviewed by AI reviewers, so they must not contradict the PR's claims.
- Before DONE: covering tests + `PYTHON=<venv> ./scripts/preflight.sh` (rc 0; never piped through tail).

Phase 5 builds on Phases 1-4 (P1-P3 merged to dev; P4 is PR #2947). This worktree is cut from the Phase 4 branch and will be rebased onto dev after #2947 merges. Owner decision D2 (ADR-012 amendment of 2026-09-26): Settings 't' runs an explicit, non-generating, authenticated model listing for cloud providers ('key accepted (models listed); generation not tested'); never auto-probe cloud providers; an OpenRouter listing never reads as key proof. Readiness words (spec section 5): 'Ready · not tested', 'Ready · verified HH:MM', 'Ready · reachable HH:MM', 'Not ready · <reason>' -- TASK-30011 AC#2 (Ready = no known blocker) and AC#6 (surfaces must not overclaim one another). Performance constraints: task-24454 (no readiness recompute per keystroke) and task-32804.3. Coordinate with task-32806.1 (placeholder keys), do not absorb it. Automatic local probes only for loopback/private hosts (KEYLESS custom slots can point at hosted services). Switch model (P4) row readiness comes from an injected config-only resolver; P5 adds evidence to it. Ratchets: the bare-type CSS rule ratchet is at its limit and boot CSS headroom is ~340 bytes; console_settings_modal.py <= 7,761 lines; chat_screen.py nets <= 0.

## Task 1: Hold provider connection evidence in one process-memory owner (TASK-33005.1)

Task file: backlog/tasks/task-33005.1 - Hold-provider-connection-evidence-in-one-process-memory-owner.md
Depends on: TASK-33001

### Why

Foundation for the phase. Today connection-test evidence cannot cross surfaces, for three reasons:
- Each surface owns a private store: the Chat settings modal (console_settings_modal.py:1304) and Settings (settings_screen.py:2949 and :13313-13318). Settings also copies its store into a cross-visit snapshot (:12854-12855, restored at :12950-12952), so every visit forks it.
- ProviderTestEvidenceStore is single-slot. It keeps one identity's evidence, clears it when another identity begins, and rejects older draft generations (provider_test_evidence.py:564-620, :996-1019).
- Each surface stamps its own counters into ProviderDraftIdentity: the modal uses _model_discovery_generation and its entry credential revision (console_settings_modal.py:5807-5814), and Settings uses _provider_draft_generation and _provider_credential_revision (settings_screen.py:13401-13436). So identical saved configuration never compares equal across surfaces.

The evidence record has no observation time (ProviderTestEvidence, provider_test_evidence.py:461), yet the readiness words need one.

ADR-033 permits a narrow process-memory owner but no new root application-state object. The TTS provider-test evidence store is the in-repo precedent: it is process-scoped and keyed by a fingerprint that includes the saved revision (ProcessProviderTestEvidenceStore and process_provider_test_evidence_store, settings_speech_tts.py:604 and :731-742).

### Acceptance criteria

- [ ] #1 Evidence recorded by a test in Chat settings, Settings or the Console is returned to every other surface that asks about the same provider connection (provider, endpoint, credential source and credential revision), and to no surface that asks about a different connection
- [ ] #2 Evidence for one provider connection survives a test of another: testing llama.cpp and then Anthropic keeps both results
- [ ] #3 Evidence from an unsaved draft (a typed key or endpoint) applies only to that draft. After Save it carries to the saved connection only when the saved values equal the tested ones, as today's save rebase does
- [ ] #4 Changing the saved endpoint or credential marks older evidence 'changed since test' instead of reusing it
- [ ] #5 Each evidence record carries the local time it was observed
- [ ] #6 Leaving Settings and returning shows the same evidence the Console sees, because the cross-visit snapshot no longer holds a private copy
- [ ] #7 Evidence lives in process memory only: nothing is written to config or any database, and after a restart every connection reads as not tested
- [ ] #8 ADR-033 holds: no new root application-state object is introduced
- [ ] #9 Existing guarantees hold per connection: stale and duplicate settlements are rejected, and an older draft generation cannot overwrite newer evidence (Tests/Chat/test_provider_test_evidence.py, Tests/Chat/test_provider_readiness.py, Tests/UI/test_settings_provider_test_draft.py). Any test that pins cross-provider eviction by the single slot is rewritten on purpose and named in the PR
- [ ] #10 A test covers concurrent settlements from worker threads for two different connections

### References

- tldw_chatbook/Chat/provider_test_evidence.py
- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_speech_tts.py
- backlog/decisions/033-application-session-state-ownership.md
- Tests/Chat/test_provider_test_evidence.py
- Tests/Chat/test_provider_readiness.py
- Tests/UI/test_settings_provider_test_draft.py

## Task 2: Make Console readiness honour known connection evidence (TASK-33005.2)

Task file: backlog/tasks/task-33005.2 - Make-Console-readiness-honour-known-connection-evidence.md
Depends on: TASK-33005.1

### Why

This is the C2 root cause.
- The Console's active readiness, _active_console_settings_readiness_uncached (chat_screen.py:9853-9869), calls build_console_settings_readiness with no evidence and no current identity.
- The future-chat path, _console_default_readiness (chat_screen.py:2846-2859), does the same.
- So the endpoint facet is always 'not_tested' (provider_readiness.py:316-319), and the unreachable blocker at console_session_settings.py:1644 can never fire in the Console.
- llama.cpp is keyless (provider_readiness.py:89-111) and defaults to 127.0.0.1:9099 (console_provider_gateway.py:195). A known refused server therefore still reads Ready in the Console, while the modal that ran the test says Not ready. That violates TASK-30011 AC#6.

This readiness feeds the status row, rail Model section, setup card, Inspector and send gate (the consumer list is in _console_provider_blocker_copy, chat_screen.py:15067-15107). A known failure therefore becomes a real send blocker, so recovering from it must take one action.

The readiness derivation runs inside the per-pass memo (_console_derivation_scope) and on hot paths:
- task-24454: readiness is recomputed on every composer keystroke.
- task-32804.3: the 0.25 s idle credential poll rebuilds readiness each tick.

Evidence reads must not add cost there.

### Acceptance criteria

- [ ] #1 After a refused or timed-out test of the active chat's connection, the Console status row, rail Model section, setup card and Inspector show Not ready with the failure reason and a retry recovery action, without a restart
- [ ] #2 Running the retry after the server has started restores Ready and unblocks send in one action, without a restart
- [ ] #3 Future-chat readiness (the default-readiness path used for new-chat and switcher decisions) reads the same evidence
- [ ] #4 A connection with no evidence still reads Ready, and reading readiness never starts a probe (TASK-30011 AC#2)
- [ ] #5 Evidence for a different endpoint or credential never changes this chat's readiness
- [ ] #6 An evidence change refreshes Console readiness once. Ordinary composer keystrokes cause no additional provider-config reads (the task-24454 probe method), and the idle credential poll's per-tick cost does not rise (the task-32804.3 method)
- [ ] #7 An integration test drives the real Chat settings modal and ChatScreen, stubbing only the network probe, and asserts that the status row reads Not ready after a refused test. The existing unit pin at Tests/Chat/test_console_session_settings.py:1169-1182 still passes
- [ ] #8 chat_screen.py stays within its ratchet row (32 lines of headroom at c4225b5d38), and console_chat_controller.py gains no lines

### References

- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/provider_readiness.py
- tldw_chatbook/Chat/console_provider_gateway.py
- backlog/tasks/task-24454 - Provider-readiness-is-recomputed-on-the-composer-keystroke-path.md
- backlog/tasks/task-32804.3 - Stop-the-idle-Console-credential-poll-rebuilding-readiness-every-tick.md
- backlog/tasks/task-30011 - Separate-Conversation-Settings-operability-from-verification-evidence.md
- Tests/Chat/test_console_session_settings.py

## Task 3: Show one readiness vocabulary on every model surface (TASK-33005.3)

Task file: backlog/tasks/task-33005.3 - Show-one-readiness-vocabulary-on-every-model-surface.md
Depends on: TASK-33005.1, TASK-33002

### Why

The judge's rule 3 and spec §5 call for one readiness vocabulary from one evidence owner:
- 'Ready · not tested'
- 'Ready · verified HH:MM'
- 'Ready · reachable HH:MM'
- 'Not ready · <reason>'

This keeps TASK-30011 AC#2 (Ready means no known blocker) while ending the overclaiming that AC#6 forbids. Spec §5 limits 'reachable' to local or URL endpoints, and says a public listing (OpenRouter's catalog needs no key, ADR-020) proves nothing about a key, so that provider stays 'Ready · not tested'.

Today each surface uses different words:
- Console: 'Ready to send' and 'Ready to send — credential not verified' (build_console_readiness_presentation, console_settings_summary.py:137-158).
- Settings: prose that leads with 'configuration is complete' (settings_screen.py:15246-15251).
- Setup verdicts: their own codes and copy (provider_readiness_verdict, provider_test_evidence.py:311-366).

There is also no way to record 'key accepted by a listing'. The credential facet becomes 'authenticated' only after a successful paid generation (_normalize_generation_credential, provider_test_evidence.py:1135-1155).

### Acceptance criteria

- [ ] #1 Readiness reads as exactly one of 'Ready · not tested', 'Ready · reachable HH:MM', 'Ready · verified HH:MM' or 'Not ready · <reason>'
- [ ] #2 'Ready · verified HH:MM' appears only when the provider accepted the credential (an authenticated model listing or a successful paid test)
- [ ] #3 'Ready · reachable HH:MM' appears only for a local or URL-based endpoint that answered its model listing
- [ ] #4 A cloud provider whose model listing needs no key (OpenRouter) stays 'Ready · not tested' after that listing answers
- [ ] #5 Neither 'verified' nor 'reachable' is ever worded as generation success
- [ ] #6 A key accepted by a listing is recorded distinctly from a successful generation, so the paid-test result is never implied
- [ ] #7 One mapping owns the words. A table test maps every readiness blocker and verdict code to exactly one word, and a new code without a word fails that test
- [ ] #8 The Console status row, rail Model section, setup card, switcher rows, Chat settings readiness and the Settings test result show the same word for the same connection at the same moment (TASK-30011 AC#6)
- [ ] #9 HH:MM is the local time the evidence was observed, never a time taken from the provider's response
- [ ] #10 Readiness is always conveyed in words, never by colour alone
- [ ] #11 Tests that pin today's 'Ready to send' wording are rewritten on purpose: Tests/UI/test_console_session_settings.py, Tests/UI/test_console_subscription_readiness.py and Tests/UI/test_console_settings_geometry.py

### References

- tldw_chatbook/Widgets/Console/console_settings_summary.py
- tldw_chatbook/Chat/provider_test_evidence.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Chat/console_session_settings.py
- backlog/docs/spec-2026-09-26-model-config-redesign.md
- backlog/tasks/task-30011 - Separate-Conversation-Settings-operability-from-verification-evidence.md

## Task 4: Check cloud keys, and endpoints with no model yet, on Settings 't' without generating (TASK-33005.4)

Task file: backlog/tasks/task-33005.4 - Check-cloud-keys-and-endpoints-with-no-model-yet-on-Settings-t-without-generating.md
Depends on: TASK-33005.1, TASK-33005.3, TASK-33001, TASK-33002, TASK-32806.1

### Why

This implements owner decision D2, which settles C3's labelling question. The ADR-012 amendment of 2026-09-26 (drafted in this worktree with the spec) already states the rules and the outcome table; this task makes the code match it.

Today 'Test Provider' ('t') never contacts a cloud provider. _provider_live_probe_base_url returns "" outside URL_BASED_PROVIDER_KEYS (settings_screen.py:15343-15363; the action is at :30793-30830). So a fake OpenAI key reads 'OpenAI configuration is complete. Credential is present; provider acceptance has not been tested.' (:15246-15251).

That behaviour was deliberate and is pinned:
- TASK-191, and TASK-386's visible guidance (settings_screen.py:16926-16941).
- Tests/UI/test_settings_configuration_hub.py:4622 test_settings_provider_test_skips_probe_for_cloud_providers.
- The toast copy pins at :3778 and :4653.

The local half: `passed = bool(readiness.ready and model)` (settings_screen.py:15238) gates the live probe (:30802), so a URL-based provider cannot be probed until a model is typed. Listing the server's models is exactly how a first-timer would find one (report persona Jordan: 'Test is blocked until a model ID is typed').

The seam already exists:
- LocalLLMProviderCatalogService.discover_models resolves saved or staged draft credentials and endpoints (local_llm_provider_catalog_service.py:481-613).
- The discovery client sends the provider's auth header, including Anthropic's x-api-key (openai_compatible_model_discovery.py:165-186), and maps 401/403 to rejected credentials (:726-737).

Eligibility is decided by supports_openai_compatible_model_discovery (:392-422). Anthropic is not in the base-URL-inferable set (:52-69), so whether its shipped endpoint qualifies must be proven, not assumed.

The amendment requires the check to use the endpoint and credential the spend path would use. Two open tasks bear on that and are not absorbed: TASK-2117 (the send path drops api_base_url for most providers) and TASK-2523/TASK-2524 (readiness and spend credential lookups diverge).

Other constraints:
- ADR-002: discovery persistence stays manual.
- ADR-020: consent must not be recorded through a Settings action.
- TASK-30014 AC#1: providers with no non-billable check must say so.
- TASK-32806.1 owns the placeholder-key validity rule; it is not absorbed here.

### Acceptance criteria

- [ ] #1 Settings 't' follows the ADR-012 amendment of 2026-09-26: it runs one explicit, non-generating, authenticated model listing for cloud providers, nothing else triggers it, and the paid 1-token test stays opt-in
- [ ] #2 For a cloud provider whose key is saved, comes from an env var, or is typed but unsaved, 't' lists models with that key and reports 'Key accepted (N models listed) · generation not tested' with the time. No generation request is made and nothing is saved
- [ ] #3 A 401 or 403 reports 'Not ready · key rejected' with the next step
- [ ] #4 Timeouts and connection errors report their own reasons
- [ ] #5 No key, header, raw response body or credential-bearing URL reaches the UI or logs
- [ ] #6 For a provider whose listing needs no key (OpenRouter's public catalog, ADR-020), 't' reports 'models listed; key not checked' and never says the key was accepted
- [ ] #7 A missing, placeholder or whitespace-padded key is reported as missing and no request is sent, using the same validity rule as TASK-32806.1
- [ ] #8 A cloud provider with no supported authenticated listing states that no non-billable key check is available and sends nothing (TASK-30014 AC#1 wording)
- [ ] #9 OpenAI and Anthropic on their shipped default endpoints both get a real key check, and Anthropic's eligibility is proven by a test
- [ ] #10 The request goes to the endpoint, with the credential, that the spend path would use for the draft under test (ADR-012 2026-09-19 precedence)
- [ ] #11 For a URL-based provider with an endpoint but no model chosen, 't' still runs the endpoint's model listing and reports what it found, while still saying a model must be chosen
- [ ] #12 The result enters the shared evidence owner. After Save, the Console shows 'Ready · verified HH:MM' for the same key, or 'Not ready · key rejected' after a 401
- [ ] #13 A key check whose draft changed before the reply arrived is discarded, and a repeated check shows only the latest result
- [ ] #14 't' never records ADR-020 refresh consent and never writes [providers] (ADR-002 manual persistence unchanged)
- [ ] #15 A counting transport shows zero requests to any cloud provider while opening Settings, switching categories, typing, saving, opening Console and opening the switcher
- [ ] #16 An integration test with a scratch TLDW_CONFIG_PATH drives 't' through the real LocalLLMProviderCatalogService.discover_models and discovery client against an httpx mock transport returning 200, 401, 403 and a timeout. It asserts the request carried the draft key in the provider's auth header and that no config file changed
- [ ] #17 The Test button tooltip, its visible guidance (settings_screen.py:16926-16941), the footer verb and the F1 entry each describe what 't' checks for cloud and for local providers, and none claims generation
- [ ] #18 Tests pinning the old behaviour are rewritten on purpose: test_settings_provider_test_skips_probe_for_cloud_providers (Tests/UI/test_settings_configuration_hub.py:4622) and the toast-copy pins at :3778 and :4653. The footer-verb pins (Tests/UI/test_screen_footer_hints.py:485; Tests/UI/test_settings_configuration_hub.py:11593 and :12474) change only if the verb changes

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
- tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py
- backlog/decisions/012-provider-credential-settings-boundary.md
- backlog/decisions/002-openai-compatible-model-discovery.md
- backlog/decisions/020-automatic-model-catalog-refresh.md
- backlog/tasks/task-32806.1 - Stop-handing-placeholder-and-unvalidated-credentials-to-providers.md
- backlog/tasks/task-30014 - Harden-Conversation-Settings-verification-accessibility-and-geometry.md
- backlog/tasks/task-2117 - api_base_url-is-dropped-for-every-non-llamacpp-provider-on-the-primary-send-path.md
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/test_screen_footer_hints.py

## Task 5: Probe local servers when the model switcher opens (TASK-33005.5)

Task file: backlog/tasks/task-33005.5 - Probe-local-servers-when-the-model-switcher-opens.md
Depends on: TASK-33005.1, TASK-33005.3, TASK-33004

### Why

This covers the local half of C2. Keyless local servers read Ready once a model is set, whether or not they are running (llama.cpp defaults to 127.0.0.1:9099, console_provider_gateway.py:195), and nothing probes them in the background.

The switcher from phase P4 is where a user chooses between them, so that is where a bounded, cached reachability probe belongs (judge-synthesis §4 P5; spec §5 'local endpoints'). A bounded, non-generating probe already exists: _test_console_connection (chat_screen.py:3207-3247) calls probe_settings_endpoint with a 2.5 s timeout (SETTINGS_ENDPOINT_PROBE_TIMEOUT_SECONDS, settings_endpoint_probe.py:49).

The provider set needs care:
- URL_BASED_PROVIDER_KEYS includes qwencloud (provider_endpoint_contract.py:24-40), which is a cloud provider. D2 forbids auto-probing it, so that set cannot be the probe set.
- KEYLESS_PROVIDER_KEYS (provider_readiness.py:89-111) excludes qwencloud but includes custom, custom_2 and custom_openai_api, which can point at a remote hosted service. 'Keyless' alone is therefore not 'local'.
- A custom endpoint that resolves a credential would send it, so it is excluded too.

The mockup's NEEDS SETUP row for a refused server reads 'start it; rechecked on open': the fix is outside the app, so Enter must not send the user to Settings for it.

The Fedora UI-lag program found network and keyring calls on the UI loop causing stalls. Probes must never delay opening the switcher, and ADR-011 keeps provider calls out of shared widgets.

### Acceptance criteria

- [ ] #1 Opening the switcher probes each configured keyless endpoint it lists whose host is loopback or a private-network address, off the UI thread. The switcher paints and accepts keys without waiting for any probe
- [ ] #2 Each probe uses the existing short bounded timeout, and one open never has more than a small, fixed number of probes in flight
- [ ] #3 A result younger than a named cache window is reused: opening the switcher twice inside the window sends at most one probe per endpoint
- [ ] #4 No credential is ever sent by an automatic probe. qwencloud, every key-requiring provider, any keyless endpoint on a public host, and any custom endpoint that resolves a credential are never contacted automatically
- [ ] #5 Probes run through a seam the screen or a Console module owns; the switcher widget makes no network call itself (ADR-011)
- [ ] #6 Results enter the shared evidence owner. A refused llama.cpp shows 'Not ready · refused :9099' in the switcher and, when it is the active chat's connection, in the Console status row. A running server shows 'Ready · reachable HH:MM'
- [ ] #7 A row whose failure lies outside the app (a refused or timed-out local server) shows in-place guidance such as 'start it; rechecked on open', and Enter on it does not navigate
- [ ] #8 Closing the switcher mid-probe raises nothing, and a late result cannot overwrite newer evidence
- [ ] #9 A test proves the probe runs off the main thread and that the switcher's first paint does not wait for it
- [ ] #10 A live capture at 211x44, with llama.cpp stopped and Ollama running, shows both words in the switcher

### References

- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/settings_endpoint_probe.py
- tldw_chatbook/Chat/provider_endpoint_contract.py
- tldw_chatbook/Chat/provider_readiness.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- backlog/decisions/011-chatbook-workbench-ui-system.md
- backlog/decisions/012-provider-credential-settings-boundary.md
- qa/model-config-ux-review-2026-09-26/judge-synthesis.md
