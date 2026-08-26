---
id: TASK-15210
title: Five pre-existing Console contract failures surfaced by the network guard
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 07:00'
updated_date: '2026-08-11 15:23'
labels:
  - console
  - tests
  - dev-baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while implementing task-15111. Running the nine Console modules the socket shim had caught reaching the network gave `147 passed, 4 failed, 0 blocked-egress hits` — the four failures are **not** caused by the guard: they were re-run with the guard and both mechanism fixtures disabled and failed identically. A fifth was found in the same sweep.

1. `section:starred` collapse preference.
2. `ConsoleChatController._turn_context_provider` — `AttributeError`.
3. and 4. Two auto-RAG ordering contracts.
5. `test_console_command_popup::test_slash_opens_popup_and_typing_filters` pins a **6-item** slash list; `/generate-video` and `/stream-video` (task-3401.5, already on dev) grew it to **8**.

The fifth is the clearest and sets the pattern: a list-length pin that a legitimate feature addition broke, sitting red until something happened to run the file whole. That is the same stale-contract class as task-14920, where twenty such failures had accumulated unnoticed and one of them was hiding a real product bug that shipped.

Triage before repair, as in task-14920: a pinned count or attribute that a deliberate change moved is a test fix, but `_turn_context_provider` raising `AttributeError` and auto-RAG *ordering* changing are both shapes that can be real regressions. Do not rewrite to green without naming the commit that changed each behaviour and reading its intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the five is classified as a stale pin or a real regression, with the causing commit named and its intent read
- [x] #2 Real regressions are fixed in the product; stale pins are updated while preserving the original claim (assert the behaviour, not a count or a class string that the next honest change breaks again)
- [x] #3 The affected Console test modules run WHOLE with READ pass counts and no unexpected failures
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the nine sweep modules + command popup WHOLE; record the real node ids and exact errors (inventory before repair).
2. For each failure, name the causing commit via git log -S over the product files it exercises, read that commit's intent, and confirm the test fails for the reason the symptom suggests (instrument with a throwaway probe rather than assuming).
3. Classify each independently: stale pin (update the test, preserving the ORIGINAL claim and preferring the property over a magic number/polarity), real regression (fix the product, TDD + mutation-check), or deliberately-removed behaviour (state the new contract and where it is enforced).
4. Implement, mutation-checking every fix (revert the fix, confirm RED returns).
5. Run each touched module WHOLE with READ pass counts, plus the coordinator keep-green set (test_console_moved_seam_guard, test_background_signal_bounds, Tests/test_network_guard). No allow_network marker added.
6. Record findings, tick ACs, write Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five reproduced exactly as described, plus a SIXTH from the same cause as #5. Every one was triaged against the commit that changed the behaviour before anything was rewritten. Two product defects were found and fixed; the other five were stale test doubles/pins, each repaired to assert the ORIGINAL claim as a property rather than a magic value.

**#1 `test_console_button_routing::test_browser_section_toggle_persists_its_collapse_preference`** — STALE POLARITY PIN. Cause: 7dbbc401b (TASK-2154.3 LY-04) made an EMPTY Starred section default-collapsed. The handler flips the section's current collapsed state taken from `_build_console_workspace_context_state()` — the screen's real (row-less) state, NOT the fabricated tray state `_sync_tray` painted — so the first press expands and persists False. Product correct. Test now reads the collapsed state from the handler's own source and asserts persisted == not-that, plus a round-trip. Two sub-findings: `is True` also PASSED a set-to-constant mutation (the new form fails it), and the round-trip only works if the toggle is re-queried — the sync rebuilds the Button, and the stale object still answers `is_mounted` True while pressing it is a silent no-op.

**#2 `test_console_composer_menu::test_impersonate_payload_obeys_the_provider_contract`** — STALE TEST DOUBLE, in THREE layers from TWO commits. c2038dfe3 (roleplay identity) added `message.metadata` (read by `_context_content_for`) and turned `_seeded_greeting_text` from a @staticmethod(session_messages) into a method (self, session_id, session_messages) — so the test's `controller._seeded_greeting_text = ConsoleChatController._seeded_greeting_text` re-bind dropped `self`. 5be9e6a04 (task-14803) then routed `impersonate_user_reply` through `resolve_turn_execution_context` -> `_turn_context_provider`, which the `__new__`-built stub lacked. Product correct. The stub now wires the real `_turn_context_provider` seam, uses the real `ConsoleChatMessage` instead of a 3-attribute stand-in, and drops the stale re-bind.

**#3 `test_console_auto_rag_on_send::test_happy_path_stages_then_send_consumes`** — HARNESS BLIND SPOT, not a live regression, PROVEN by measurement. 5be9e6a04 moved the auto-retrieve toggle read off live `get_cli_setting` onto the frozen `ConsoleTurnExecutionContext.rag_defaults`, built from `app.app_config`. `_build_test_app` hands TldwCli a synthetic 2-key app_config with no `[chat_defaults]`, and production correctly refuses to refresh a snapshot that never came from `load_settings()`. Instrumented: `get_cli_setting`=True but the app_config key MISSING, so the context captured auto_retrieve_on_send=False. The live app does `self.app_config = load_settings()` (app.py:4730), whose result carries the toggle AND both disk-load markers — verified — so the shipping path takes the fresh branch and the toggle still works. Test now sources the app config from disk exactly as app.py does.

**#3b (NEW, found in the same file)** — `test_send_proceeds_when_auto_retrieve_fails` was GREEN VACUOUSLY from the same cause: auto-retrieve never fired, so the exploding backend was never called and the test only asserted that an ordinary send works. It now asserts `exploding_search.await_count == 1`.

**#4 `test_capture_seam_calls_the_hook_before_consuming`** — STALE STUB SIGNATURE. 5be9e6a04 threads `turn_context` to `_maybe_auto_retrieve_for_send`; the one-argument stub made the seam's call raise TypeError, which the seam's deliberate `except Exception` swallowed. Product correct (the ordering contract holds). The stub takes the production signature and the test now also asserts THIS turn's context reaches the hook.

**#5 `test_console_command_popup::test_slash_opens_popup_and_typing_filters`** — STALE COUNT PIN, as suspected. d6c2e9756 (/generate-video, task-3401.5) and 72a2ff3c5 (/stream-video, task-3401.11) grew the registry 6 -> 8. The claim was never 'there are six'; it is 'the popup offers the registered commands, and typing filters'. Now asserts labels == the registry's own `available_names()` plus a subset check for the original six, so honest additions pass and a removal still reds (mutation-checked by deleting /rewind's registration).

**PRODUCT FIX A (from #5's blast radius).** `console_command_suggestions` documents `COMMAND_DESCRIPTION_FALLBACK` as reachable 'only [by] non-built-in registrations'. Nothing pinned it, so /generate-video and /stream-video shipped rendering 'Custom command' in the popup. Added both descriptions and a guard test that asserts the invariant (no built-in falls back) rather than a name list.

**PRODUCT FIX B (SIXTH failure, same cause as #5).** `Tests/Library/test_library_skills_state::test_shadow_name_set_stays_in_sync_with_real_sources` was also red: `_SHADOWED_BUILTIN_NAMES` in `Library/library_skills_state.py` — a PRODUCT set — was missing generate-video/stream-video, so `skill_name_shadows_builtin` would not warn a user installing a skill that shadows either slash command. That set already carries a task-580 comment saying this exact drift had been carried as an accepted baseline before. Fixed and mutation-checked.

**Evidence.** RED confirmed for each before touching anything, and each fail-reason verified by instrumentation rather than inference (three throwaway probe files, all deleted). Mutation checks: registry deletion (#5), greeting-fold disabled (#2), turn-context toggle key broken (#3/#3b both red), hook moved after the consume (#3/#4 both red), collapse flip set to a constant (#1 — which the old `is True` pin would have passed), shadow names removed (fix B). Every mutation restored via Edit and verified with an empty `git diff`. READ counts: touched + adjacent modules whole = 225 passed; remaining sweep modules + `test_console_native_chat_flow` = 383 passed, 1 xfailed (task-15120's strict xfail intact); keep-green (`test_console_moved_seam_guard`, `test_background_signal_bounds`, `Tests/test_network_guard`, `test_console_turn_execution_context`, `test_console_local_server_probe_isolation`) = 40 passed. No `allow_network` marker was needed anywhere.

**Modified:** tldw_chatbook/Chat/console_command_suggestions.py, tldw_chatbook/Library/library_skills_state.py, Tests/Chat/test_console_command_suggestions.py, Tests/UI/test_console_auto_rag_on_send.py, Tests/UI/test_console_button_routing.py, Tests/UI/test_console_command_popup.py, Tests/UI/test_console_composer_menu.py.

**Left open (reported, not fixed):** every mounted `_build_test_app` Console test sees an app_config with no `[console]`/`[chat_defaults]`, so ANY feature reading through the turn-context snapshot is silently defaulted in those tests — a systemic blind spot far wider than this task. `Tests/UI/test_console_auto_rag_on_send.py` also carries a pre-existing ruff F401 (`Static`, used only inside a query string), left untouched.
<!-- SECTION:NOTES:END -->
