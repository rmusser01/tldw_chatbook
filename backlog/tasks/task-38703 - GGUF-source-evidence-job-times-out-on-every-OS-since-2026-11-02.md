---
id: TASK-38703
title: GGUF source evidence job times out on every OS since 2026-11-02
status: In Progress
created_date: '2026-11-03 13:47'
updated_date: '2026-11-03 14:30'
labels:
- ci
- flaky-test
priority: medium
references:
- backlog/tasks/task-38206 - GGUF-source-mode-recompose-test-times-out-on-Windows-CI-intermittently.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The "TASK-2062.2 GGUF source evidence" workflow was green through 2026-10-31. From the first run on 2026-11-02 (17:41Z) it fails on PR heads based on current dev, on ubuntu, macOS and Windows alike. The keyboard cases `test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp]` and `[llamafile-llamafile]` hit pytest-timeout's 60 s limit (one run also raised `console_view_worker_owner_changed` after the timeout), and `test_claim_authority_survives_screen_recompose_and_not_window_selection` fails with "screen recompose did not reach deferred mount". The failure is not limited to Windows, so it is not the executor starvation TASK-38206 diagnosed. The job is the only evidence lane for the GGUF source-mode UI, and while it is red it hides real regressions in that UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The root cause is identified, with the dev commit (or PR) that introduced it and evidence from bisection.
- [x] #2 The job's full-app node set, run exactly as the workflow runs it (one process per node, private profile, --timeout=60), passes at least 5 consecutive times locally on macOS, with per-node timings back in the range of the last green CI runs.
- [x] #3 If the cause is a product bug, a test that fails without the fix and passes with it guards it.
- [x] #4 No assertion is loosened, no wait is dropped, and no timeout or settle budget is raised.
- [ ] #5 The job passes on ubuntu-latest and macos-latest in CI.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the failing CI logs (runs 41595165317, 41647054961, 41601553525, 41582510679) and a last-green log (41235218213) for per-node timings and timeout stack dumps.
2. Reproduce on macOS with the job's exact node list, one process per node with TLDW_TEST_PRIVATE_PROFILE_NODE set (the workflow's private-profile mechanism, which also avoids the local raw_source_selection_changed setup error), and --timeout=60.
3. Bisect dev first-parent merges between the last green base and the first red base in scratch worktrees, timing the llamacpp keyboard node; then bisect inside the introducing PR.
4. Find what keeps the process busy (stack sampling, call counters in an uncommitted plugin) and fix it at the shared function every caller routes through, test first.
5. Re-run the node set 5+ times locally; run the touched test files, ruff and preflight; record findings in TASK-38206.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Root cause: a Console refresh loop, a product bug.** It is not a test-harness problem and not Windows starvation. `ConsoleReadinessConfigProjection.tokenizer_ready` keeps a single prepared-tokenizer request and compared `(model, provider)` as raw strings. Five callers in `context_spend.py` route through `console_tokenizer_display_ready`. The four context-estimate callers pass the session's provider config key ("openai"). The cost-chip caller (`_build_console_cost_state_under_config`) passes `_active_console_provider_model_display()`'s display label ("OpenAI"). So each sync did the following, without end:
1. It recaptured the other caller's request.
2. That forced a new readiness config read: `read_current` on a worker thread, through config-participant storage admission.
3. The read published `changed`, which ran another `_sync_native_console_chat_ui`.

Counted in one llamacpp keyboard run before the fix: 3,450 refreshes and 6,896 recaptures in 225 s. After the fix: 4 refreshes and 1 capture. The Console under the pushed LLM screen kept the process busy, so every `pilot.pause()`/`press()` (`wait_for_idle`, up to 1 s while the process uses CPU) ran to its 1 s cap. It also floods the default executor. Each read is `run_preparation_read`'s `run_in_executor(None, context.run, ...)`. Counted across one llamacpp keyboard run:

| | Before the fix (246 s) | After the fix (31 s) |
|---|---|---|
| Default-executor submissions | 6,837 | 182 |
| `Context.run` submissions | 6,825 | 170 |
| `ConsoleRuntime._canvas_enabled` calls | 694 | 91 |
| `_local_pause_requested` calls | 211 | 26 |

This is the same signature the TASK-38462 stream reported as "high-frequency polling through thread workers": 4,700-7,800 jobs, 701 `_canvas_enabled` and 216 `_local_pause_requested`. It is this loop, not a separate poller. The visible idle Console loops too: a scratch probe counted 71 refreshes and 7.0 CPU-s in 10 s before the fix, and 3 refreshes and 4.2 CPU-s after.

**Introduced by e017592b44 ("Keep Windows cold display reads off the UI thread"),** the first commit of PR #3166 (codex/windows-interaction-performance), which reached dev in merge 6fbd29f9a3 on 2026-10-31 21:32Z. Bisection timed the llamacpp keyboard node with the CI invocation in scratch worktrees, mostly 2-3 probes in parallel:

| Commit | Role | Time | Result |
|---|---|---|---|
| 14de6845b3 | last green CI base | 19.9 s | green |
| 0d05d27fa1 | dev | 30.1 s | green |
| a0dbc1174c | PR #3162, parent of #3166 | 32.0 s | green |
| 6fbd29f9a3 | merge of #3166 | 225.8 s | red |
| e017592b44 | first commit inside #3166 | 243.0 s | red |
| 73b03cd060, b1e19a1dba, 88af9f9379 | later dev | 221-254 s | red |

No GGUF run happened between that merge and 11-02 17:41Z, because the workflow is path-filtered. The red Windows runs before 335f984bd2 also still had the 10 s settle budget: TASK-38206's 120 s Windows deadline reached dev only in that merge (#3352).

**Why codex/shared-console-mobile-pr went green.** Its `_console_sync_suspended` flag stops Console syncs while the Chat screen is covered, which is what the GGUF tests do. That hides the loop under the LLM screen, but not on a visible Console.

**Fix (test first).** `tokenizer_ready` now compares providers through `config.normalize_provider_config_key`, the canonical form both callers share. Lower-casing alone would leave labels such as "local-llm" against the key "local_llm" looping. Config-key normalization never merges two strings that the tokenizer's heuristic provider patterns treat differently: the patterns are plain single words, and the only engine aliases are ollamacloud and opencodezen. No assertion, wait, timeout or budget changed.

**Evidence.**
- Regression test `test_provider_label_and_config_key_share_one_preparation`, cases openai and local_llm. Both fail against the pre-fix module ("tokenizer_ready('model', 'OpenAI') is False") and pass with the fix. The file passes 10 of 10.
- The CI full-app node set, run as the workflow runs it: one process per node, with `TLDW_TEST_PRIVATE_PROFILE_NODE` set (the same private-profile path, so no `raw_source_selection_changed`) and `--timeout=60`. It passed 5 of 5 consecutive runs, 35 of 35 nodes. Per-node call times across the runs:

| Node | Range | Last green macOS CI | Before the fix |
|---|---|---|---|
| lazy-mount | 11.6-17.6 s | 19.2 s | 25.0 s |
| replay[False] | 11.6-14.4 s | 14.5 s | 12.4 s |
| replay[True] | 11.9-19.5 s | 14.8 s | 15.3 s |
| recompose | 10.7-17.0 s | 15.4 s | 16.3 s |
| external-copy | 11.4-22.8 s | 19.6 s | 37.5 s |
| keyboard llamacpp | 22.1-29.2 s | 32.8 s | 114.2 s |
| keyboard llamafile | 22.9-30.8 s | 28.3 s | 192.8 s |

- The workflow's 18 backend nodes passed 5 of 5 (31 passed each).
- Related suites: tokenizer display, checked display scope, compaction forecast, measured cost and readiness projection. 121 passed and 1 failed. The failure is `test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation` (AttributeError: SimpleNamespace has no `content`), which fails identically without the fix on dev 335f984bd2.
- `Tests/Architecture/test_console_fleet_ownership.py::test_fleet_policy_has_one_non_dom_owner` also fails on a dev 335f984bd2 scratch worktree. It is unrelated.
- ruff: no new findings. The 4 in console_spend_projection.py are pre-existing.
- preflight: all checks passed.

**AC#5 is open.** It needs the ubuntu and macOS jobs to pass in CI. Windows still has TASK-38206's executor starvation, which TASK-38462 addresses.

**Process note.** The fix edit was made just before this file was committed; the plan above was written before any commit.

**Files.** `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `Tests/UI/test_console_tokenizer_preparation.py`, `backlog/docs/lessons-testing-evidence.md`, and the TASK-38206 notes.

**Review round (PR review, 2026-11-03).** Commit 27aa494818.
- **The remaining loop.** The provider-key fix left the model leg on a single slot. The cost chip asks for the selection's model (`explicit_model or configured_model`); the context estimates ask for the session's `settings.model`. When the two differ, each ask recaptured the other's request, and every recapture forced another config read that published another sync.
- **The fix.** `ConsoleReadinessConfigProjection` now keeps a bounded dict of `_TokenizerSlot`s keyed by `(model, normalized provider key)`. Eviction is oldest-first past `TOKENIZER_SLOTS` (8).
- **Invalidation is unchanged per slot:**
  - a slot whose `preparation_current` fails is recaptured on its next ask;
  - a refresh marks only the slots it captured before its read, each still guarded by the owner key and by retirement;
  - `tokenizer_ready` still requires the published key and the current runtime;
  - a failed slot keeps its own `retry_at`, as the single slot did.
- **Test changes.** Tests that read the old private slot now read the dict. In `test_changed_owner_cannot_publish_tokenizer_readiness[model]`, a model asked mid-read is a second slot, not a new owner: the read prepares only the slot it captured.
- **RED/GREEN.** `test_differing_models_keep_their_own_preparations` returned `[False, False]` on every sync before the change, because each sync recaptured. After it, both are ready and further syncs add no capture, read or publication. `test_tokenizer_slots_stay_bounded` pins the eviction.
- **Evidence.**
  - `Tests/UI/test_console_tokenizer_preparation.py`: 12 passed.
  - Neighbouring projection suites, including `Tests/Performance/test_console_tokenizer_owned.py` with the real wiring: 206 passed, 4 failed. Three of the four fail identically with HEAD's and with dev 335f984bd2's `console_spend_projection.py`, so they predate this change: `test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation`, `test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` and `test_actual_settings_open_waits_for_shared_cold_checked_policy`. The fourth, `test_console_mount_refreshes_never_wait_for_a_config_rebuild[roleplay]`, passes in isolation and only fails under xdist load.
  - ruff: no new findings.

**Re-review round (2026-11-03).** Commit c13fd0ad13. (1) A slot whose screen moved to another model stayed in the dict, so every refresh and `warm()` re-ran its failed preparation (probe: A ran 5 times across 4 refreshes while only B was asked). `_TokenizerSlot.asked` is set by `tokenizer_ready`; `tokenizer_due` and the refresh snapshot consider only asked slots, and the refresh clears the mark, so an unasked slot is neither read nor waited on, and the alternating chip/session asks of one sync still stay prepared. (2) Eviction is now least recently asked: `tokenizer_ready` pops the key and re-inserts it on every hit or recapture. RED/GREEN: `test_slot_nobody_asks_for_is_not_prepared_again` (A ran 9 times before, 1 after) and `test_tokenizer_slots_evict_the_least_recently_asked` (wrong order before); `Tests/UI/test_console_tokenizer_preparation.py` 14 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
