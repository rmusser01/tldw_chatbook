---
id: TASK-42478
title: "The Console, Settings and Personas credential polls read the Claude Code Keychain credential every 5 s even when Anthropic subscription auth is not selected"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 11:13
updated_date: 2026-11-23 18:52
labels:
- follow-up
- console
- security
- privacy
priority: high
references:
- tldw_chatbook/LLM_Calls/anthropic_subscription.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/personas_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42477's review. Checked on `fu/r22-keychain` (the TASK-42477 fix on 62531d037a); paths are under `tldw_chatbook/`.

**Why.** Three UI polls ask for the subscription credential status, whichever provider and auth source are selected:
- `UI/Screens/chat_screen.py:10621`, the Console's credential poll (`_poll_console_credential_readiness`, 4 Hz);
- `UI/Screens/settings_screen.py:5310`, the Settings provider status gate;
- `UI/Screens/personas_screen.py:7356`, the Personas handoff readiness gate.

Each calls `subscription_credential_status(background=True)` unconditionally, as one element of its change-detection tuple. `_SubscriptionReadinessCache.status()` (`LLM_Calls/anthropic_subscription.py`) starts a background `read_claude_code_credential()` whenever its 5 s snapshot has lapsed. That read opens `~/.claude/.credentials.json`. On macOS, when the file is absent, it spawns `/usr/bin/security find-generic-password -s "Claude Code-credentials" -w`. TASK-26022's live verification (AC#7, 2026-09-02) found Claude Code's credential in exactly that Keychain item, and this path returned it.

So a macOS user logged in to Claude Code, with no credential file, has Chatbook spawn `security` and take their Claude Code OAuth token into process memory about every 5 s while any of those screens is mounted. That holds even with `auth_source = "api_key"` or another provider selected. TASK-26022 framed the borrow as explicit opt-in: the module docstring says only `auth_source = "claude_subscription"` activates it. TASK-42477's moved-HOME check does not change this for a real user, whose HOME is the account's own. In TASK-42477's measurement, every mounted Console test spawned the read with no Anthropic subscription configured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While the provider a screen reports on is not Anthropic, or is Anthropic with an auth source other than `claude_subscription`, keeping the Console, Settings or Personas screen mounted causes no credential read: no read of the credential file and no `security` spawn. A test per screen pins it
- [x] #2 With Anthropic and `auth_source = "claude_subscription"`, each screen still shows the background readiness states (checking, ready, expired, missing) and refreshes when a read completes, as today
- [x] #3 Switching to Anthropic subscription auth starts the read without a restart; switching away stops further reads
- [x] #4 The send-time and provider-readiness paths for subscription users behave as before
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace each poll's cheap pre-gate and what decides its readiness: Console (the active chat's session settings and the fresh config, read behind the config locks), Settings (the panel's provider and the Sign in with choice it shows), Personas (`chat_defaults` in the app config).
2. Gate the `subscription_credential_status(background=True)` element of each pre-gate on that screen's own "Anthropic with `claude_subscription`" answer, from inputs the pre-gate can read without config I/O. Share one predicate between `get_provider_readiness` and the Personas poll so the two cannot drift.
3. Keep TASK-42474's quiet wait and the other tests that copy the Console pre-gate in step with its new shape.
4. Tests, in the per-screen subscription readiness files, for AC#1 and AC#3 on each mounted screen (red at base, green after); AC#2/AC#4 by the existing subscription tests, with a negative control.
5. Ratchets, preflight, ruff, Tests/CI, Architecture; notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** Each poll's cheap pre-gate now holds `None` instead of `subscription_credential_status(background=True)` unless the screen it belongs to is on Anthropic with `claude_subscription`. A `None` element reads nothing; the readiness paths behind the gates were already gated, because `get_provider_readiness` asks for the credential only on that branch.

**Why three gates and not one seam.** No single place sees each screen's selection cheaply, so each poll uses what its own readiness reads:
- **Console** (`chat_screen.py`): the pre-gate runs ahead of the config locks and must not read config (TASK-34956), and the active chat's provider and fresh config are only known behind them. So the readiness derivation (`_active_console_settings_readiness_uncached`) records whether its result was on the subscription (`readiness.subscription_status is not None`). The pre-gate reads that flag. Provider changes, Console settings saves and a return from Settings already re-derive readiness, so switching to the subscription starts reads and switching away stops them, with no restart.
- **Settings** (`settings_screen.py`, `category_panels.py`): the gate already held the Sign in with value. The caller now passes the choice its category shows, the saved one for Overview and the draft for Providers & Models, and the credential element reads only when that choice is `claude_subscription`. (`_provider_auth_source_value` returns `api_key` for any other provider.) The category comparison stays in the callers: `test_settings_screen_category_comparisons_do_not_rise` forbids a new one in the screen.
- **Personas** (`personas_screen.py`): new `uses_claude_subscription(provider, app_config)` in `Chat/provider_readiness.py`, applied to the same `chat_defaults` provider and app config that `console_handoff_readiness` uses. `get_provider_readiness` now calls it too, in place of its inline check (same test: provider key `anthropic`, then `anthropic_auth_source` of `provider_settings_for_key(..., "anthropic")`), so the poll and readiness cannot drift.

**The pre-gate's shape.** TASK-42474's quiet wait (`Tests/UI/test_console_rail_reconciliation.py::_wait_for_inspector_quiet`) copies the Console pre-gate, so its copy now applies the same condition. With the old copy, the two tests that use it failed in all 6 cases with `Inspector still busy: ['credential poll pass']`, and the copy itself made 18 credential reads. `test_console_roleplay_resume_navigation._setup_card_settled` reads `pre_gate[0] != "pending"`, which `None` satisfies. `test_console_credential_poll_gate._screen` stands for a subscription chat, so it sets the flag.

**Tests** (each with a `credential_reads` fixture in `Tests/UI/test_console_subscription_readiness.py`: darwin forced, the credential file absent, a fake `run`, so a read counts one file read and one Keychain spawn, and nothing real runs):
- `test_console_reads_the_credential_only_while_its_chat_uses_the_subscription[other-provider|api-key]`;
- `Tests/UI/test_settings_anthropic_auth_source.py::test_settings_reads_the_credential_only_while_the_subscription_is_chosen[...]`, which covers Overview, then Providers;
- `Tests/UI/test_personas_subscription_readiness.py::test_handoff_poll_reads_the_credential_only_while_the_subscription_is_chosen[...]`.

Each test mounts the screen and lets its poll run, lapses the snapshot, and requires zero reads (AC#1). It then switches to the subscription and requires a read, and a second read after a lapse (AC#3, start). It then switches away, by provider or by "API key", lapses again and requires no new read (AC#3, stop).

**Red/green.**
- At base 82055f4750, all 6 cases failed with `assert (2, 2) == (0, 0)`: two reads before any subscription was chosen. After the fix all 6 pass, and the refusing probe counted 12 faked reads and 0 spawns.
- AC#2, negative control (a temporary edit, reverted): with every gate's element forced to `None`, `test_console_send_refreshes_expiry_and_renews_stale_snapshot_without_edits` and Settings' two refresh tests failed. Personas' refresh test still passed: its header inputs call `console_handoff_readiness` every tick, which reads for a subscription user through readiness. With the fix, all the existing subscription refresh tests pass (AC#2), as do the send/readiness tests in `Tests/Chat/test_subscription_background_readiness.py`, `Tests/Chat/test_provider_readiness.py` and `Tests/LLM_Calls` (AC#4).

**Runs** (scratch wrapper, refusing probe on, load 7-15; 0 `security` spawns in every run):
- Touched and related files at `-n 4` (16 UI/Chat/Wizards files plus `test_settings_provider_test_draft.py` and `test_settings_configuration_hub.py`): 1094 passed, 5 failed. All 5 fail at base with the same command: the 4 mounted-resume tests in `test_console_roleplay_resume_navigation.py`, and `test_t_hotkey_does_not_run_test_while_input_focused` (`RecoveryRequired: raw_source_selection_changed`). In an earlier run, under higher load, `test_credential_poll_defers_native_config_contention[_config_file_lock]` (its 10 s wait) and `test_a_routed_providers_key_is_typed_under_its_own_name[chat-settings]` also failed. Both passed when re-run alone, at base and on this branch, and in this run.
- `Tests/LLM_Calls` plus `Tests/test_network_guard.py` at `-n 4`: 2307 passed, 48 skipped, 7 failed. A Console/Settings/Personas sample of 8 more files: 757 passed, 3 failed, and the probe saw 0 calls to the Keychain reader (base: 1). All 10 failures fail at base with the same command, and none is related to this task or to TASK-42477. The 7 in `Tests/LLM_Calls` are the `RecoveryRequired: raw_source_selection_changed` trap (the Anthropic redirect test, the two `test_realtime_protocol` connect tests and the keyring-isolation test), the fireworks/deepseek preset set (`{'deepseek','fireworks'} == {'fireworks'}`) and the three summarization manifest-inventory tests. The other 3 are the two `test_personas_workbench.py` cases (`RecoveryRequired`) and the hotkey test.
- `Tests/Architecture` serially: 1491 passed, 5 skipped, 1 failed. The failure was `test_chat_screen_remains_within_reviewed_projection_without_ratchet_raise`, which pins a copy of the ChatScreen row; it is re-pinned, and the three ratchet files pass (241 passed).
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped.
- Preflight passes. `ruff check tldw_chatbook --select F` is clean, `Tests/lint/ruff_gate.py` exits 0, and the touched test files are F-clean.

**Ratchets** (exact growth, dated):
- ChatScreen 24,537 -> 24,543 lines, methods unchanged, in `test_screen_size_ratchet.py` and its copy in `test_console_review_selection_controller_boundary.py`;
- `personas_screen.py` 16,892 -> 16,896;
- `settings_screen.py` stays under its row (34,988 of 34,993).

**Lane.** The Settings test is in a whole-file census entry. The Console and Personas files are laned by node id only, so two node ids were appended at the census tail and `MINIMUM_FILES` was raised to the measured entry count. Their local serial times, UI-lane flags, load 12-15: Console entry 21.5 s (2 cases), Personas entry 10.6 s (2 cases).

**User Guide.** Not changed: what a user sees is unchanged. The only difference is that a credential nobody had chosen is no longer pre-read, so the first switch to the subscription shows the documented "checking the Claude Code login…" state first.

**Files.**
- Product: `tldw_chatbook/Chat/provider_readiness.py`, `UI/Screens/chat_screen.py`, `UI/Screens/settings_screen.py`, `UI/Settings_Modules/category_panels.py`, `UI/Screens/personas_screen.py`.
- Tests: `Tests/UI/test_console_subscription_readiness.py`, `test_settings_anthropic_auth_source.py`, `test_personas_subscription_readiness.py`, `test_console_credential_poll_gate.py`, `test_console_rail_reconciliation.py`, and the three Architecture files.
- Census: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
### Review round (2026-11-23)

The review found the change "ready with fixes"; each is fixed in a commit on this branch.

1. **Census floor in these notes.** The Lane paragraph above no longer pins the floor.
2. **The `credential_reads` fixture's fake `run`** (`64a813deda`). It patched stdlib `subprocess.run` process-wide and counted and faked every call as a Keychain spawn. It now fakes only `/usr/bin/security`. Any other command fails with `pytest.fail` naming it, and the teardown checks again, in case the fail was raised on a worker thread. Evidence, from a scratch test (deleted): `subprocess.run(["/usr/bin/true"])` on the test thread was faked and counted under the old fixture (`DID NOT RAISE`), and fails by name under the new one. From a worker thread it passed silently under the old fixture, and now errors at teardown naming the command.
3. **One predicate** (`e93789ebc9`, `05ad5534f2`). `uses_claude_subscription` now also decides:
   - first-run's `_signs_in_with_subscription`, which compared `provider != "anthropic"` without normalising;
   - Settings' `_provider_saved_auth_source`, which had the same semantics in its own code (provider key normalised, then `anthropic_auth_source` of the first normalised table). Its callers still get the `api_key` or `claude_subscription` string, and `test_settings_screen_category_comparisons_do_not_rise` stays green. It reads the config through Settings' existing `_app_config_mapping()`: a first version added a literal `getattr` on the app object, which `test_app_string_lookups_match_their_pin[UI]` failed (1316 > 1315);
   - the Anthropic send (item 6).

   `provider_setup_persistence.py` is unchanged. The laned test `test_first_run_setup_state.py::TestProviderTrustChain::test_a_subscription_table_spelled_otherwise_reads_as_readiness_reads_it` now runs with the provider spelled `anthropic` and `Anthropic`, against `[api_settings.Anthropic]`. It asks the predicate, readiness, first-run's listing copy, Settings' saved choice and the send. Red with first-run's and the send's code from 3989dadde6: `[Anthropic]` failed on the listing copy (`subscription_cannot_list` was False), and `[anthropic]` failed on the send (`'Anthropic API Key is required.'`). Settings' check already passed there.
4. **Double resolution** (`e93789ebc9`). `uses_claude_subscription` takes a keyword-only `provider_settings`, and `get_provider_readiness` passes the table it already resolved for the same key, so behaviour is unchanged.
5. **The pre-gate was committed after a deferred control-bar sync** (`05bf5c59a0`).
   - **Fix.** A pass whose `_sync_console_control_bar()` returns False now returns before committing the snapshot or the pre-gate, so the next 0.25 s tick retries, as the TASK-36511.4 comment says. A pass whose snapshot is unchanged still commits (TASK-36511.4's fix for idle rebuilds).
   - **No hot loop.** The poll runs on its timer, so it retries at most 4 times a second, and the deferral's own replay is coalesced (`request_control_bar_sync`). With a storage pause held in the mounted test (a temporary edit, reverted), each refused poll took 1.7 ms (mean of 12 polls, load 5).
   - **The mounted test.** `test_shared_evidence_change_refreshes_the_console_once` failed with `assert 1 == 2` at :1136 at 3989dadde6, and also with base 82055f4750's product code. So its red was already on dev; by that point in the test the base's credential status had settled too. With the fix, the test went on to fail in a later section, "A cold lookup cannot adopt a different selected configuration source". That section (90bf1daeee) expected an idle poll to read config, which the poll has not done since TASK-34918's pre-gate (4b0104e8e1, the next day). The section now settles a result first and lets the episode's first cold read defer (TASK-36511.4), then expects `RecoveryRequired`. Without the deferring poll it fails with `DID NOT RAISE`.
   - **Unit test.** `test_console_credential_poll_gate.py::test_a_deferred_control_refresh_leaves_the_gate_open` fails with the old poll ("a deferred refresh must be retried") and passes now. The `_screen` stub's `_sync_console_control_bar` now returns True, as the real method does after rendering.
   - **Quiet wait.** The copy in `test_console_rail_reconciliation.py` still matches: it compares the committed pre-gate with the one the poll would read now, and the gate is now committed only after a completed pass.
6. **The send's auth source** (`e93789ebc9`). Config load does not normalise table keys. A scratch config holding only `[api_settings.Anthropic] auth_source = "claude_subscription"` loads as two tables: the shipped defaults' `anthropic` first, then `Anthropic`. With the defaults merged, readiness and the send both read `anthropic`, so a loaded config agreed. An `api_settings` without the canonical table first, such as an app config built in code, split them. `chat_with_anthropic` now asks `uses_claude_subscription("anthropic", loaded_config)`. Its fallback to the legacy `anthropic_api` dict is dropped: the loader never puts `auth_source` there, and readiness never read it.

**Lane.** Neither `Tests/UI/test_console_credential_poll_gate.py` nor the mounted test was in a PR lane, which is how the mounted test's red went unseen. The file is appended at the census tail as a whole-file entry, and the floor is raised to the measured entry count. Local serial time with the UI-lane flags: 1.8 s for 4 tests (load 13).

**Ratchets.**
- ChatScreen grew 1 line (24,543 -> 24,544; methods unchanged), raised in `test_screen_size_ratchet.py` and in the copy in `test_console_review_selection_controller_boundary.py`.
- `settings_screen.py` is at 34,989 lines, under its row (34,993), and `LLM_API_Calls.py` at 5,199, under its row (5,202).

**Runs** (scratch wrapper, refusing probe on, load 5-14; no `security` spawn in any run):
- **Covering set** at `-n 4`: the touched files plus the Console poll, Settings, Personas and Home subscription, Anthropic send, `Tests/Chat/test_provider_readiness.py` and first-run files, and all of `Tests/Wizards`. Result: 3264 passed, 95 failed. All 95 also fail with 82055f4750's product code (same node ids), with the same causes in the failure sections:
  - the `raw_source_selection_changed` trap, in most of them: the live-contract tests in `test_first_run_wizard_live_contract.py`, the Anthropic send tests in `Tests/Chat`, `Tests/LLM_Calls` and `Tests/Agents`, and the hotkey test;
  - the 5 mounted-resume tests;
  - three wizard live-contract tests with `assert False`;
  - an agent-budget test with `assert 100200 < 100200`.

  This run came before `05ad5534f2`; after it, the six Settings files that cover the saved choice re-ran with 185 passed and 1 failed (the hotkey test, the trap). The Anthropic send tests hit the trap at `load_settings()`, before the changed code. Under a temporary `@private_profile_test` marker (reverted), `test_chat_with_anthropic_refuses_cross_origin_redirect` passes, and `Tests/LLM_Calls/test_anthropic_subscription.py`, which patches `load_settings`, passes in full.
- **`Tests/LLM_Calls` plus `Tests/test_network_guard.py`** at `-n 4`: 2307 passed, 48 skipped, 7 failed. All 7 fail with 82055f4750's product code:
  - the trap: the Anthropic redirect test and the two `test_realtime_protocol` connect tests (the keyring-isolation test passed in this run);
  - the fireworks/deepseek preset set;
  - the three summarization manifest-inventory tests.
- **`Tests/CI`** at `-n 4`: 1823 passed, 2 skipped.
- **The three ratchet files**, serially: 241 passed. **`Tests/Architecture`**, serially: 1492 passed, 5 skipped.
- **Preflight** passes. `ruff check tldw_chatbook --select F` and the touched test files are clean, and `Tests/lint/ruff_gate.py` exits 0.

**Files.**
- Product: `Chat/provider_readiness.py`, `UI/Wizards/first_run_model_discovery.py`, `UI/Screens/settings_screen.py`, `LLM_Calls/LLM_API_Calls.py`, `UI/Screens/chat_screen.py`.
- Tests: `Tests/UI/test_console_subscription_readiness.py`, `test_console_credential_poll_gate.py`, `test_console_endpoint_discovery.py`, `Tests/Wizards/test_first_run_setup_state.py`, and the two Architecture files.
- Census: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
