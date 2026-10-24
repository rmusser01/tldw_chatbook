---
id: TASK-36390
title: Tests/Performance compact-setup, voice-latency, tab-sync and keystroke tests are red on dev and run in no PR lane
status: Done
assignee: ['@claude']
created_date: 2026-10-22 23:58
updated_date: 2026-10-24 00:57
labels:
- follow-up
- tests
- performance
priority: medium
references:
- Tests/Performance/_compact_original_config_gate.py
- Tests/Performance/test_compact_setup_ownership.py
- Tests/Performance/test_compact_model_original_checked_read.py
- Tests/Performance/test_speculative_voice_latency.py
- Tests/Performance/test_console_tab_sync_source_ownership.py
- Tests/Performance/test_console_keystroke_work_census.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while burning down the Settings UX follow-ups (2026-10-22, dev 165e31fab8). None of these tests is in the UI census, a `derived-artifacts.yml` lane or `perf-guard.yml`, so no PR run reports them:
- **The compact-setup gate (one defect, 14 tests).** All 12 parameters of `test_compact_setup_ownership.py::test_compact_setup_retains_owner_source_choices_and_native_callback` and both of `test_compact_model_original_checked_read.py` fail with `AttributeError: _build_console_settings_summary_state`.
  - It is raised by `inspect.getattr_static` at `Tests/Performance/_compact_original_config_gate.py:996`.
  - The name is listed at :934.
  - The method now lives on `_context_spend`.
- **`test_speculative_voice_latency.py` (2 tests).** The completed-pair locator inventory is stale (`assert 28 == 19`), and the history gate fails with "completed-pair baseline distribution drifted". The second is probably the same stale inventory; not yet confirmed.
- **`test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[cold_full_schedules_its_pending_replay]`** fails with `KeyError: "'__name__' not in globals"` at `chat_screen.py` around :17546.
- **`test_console_keystroke_work_census.py`: two behaviour nodes.** Both failed in a parallel run and in a serial run. perf-guard runs neither, and the tasks that last named them (33005.10, 33008, 33374) are Done.
  - `test_typing_never_snapshots_the_transcript`: "3 messages_for_session calls across 24 keystrokes … callers={character.py:368=2, message.py:995=1}", `assert 3 == 0`.
  - `test_keystroke_work_does_not_scale_with_transcript_length`: "per-keystroke work differs with transcript length".

Not in scope here:
- `test_console_three_turn_profile.py`'s 14 failures are TASK-33372 (the unreachable harness commit).
- `test_console_keystroke_work_census.py`'s storage-ratchet nodes fail only on macOS (`os_opens` 57 against a ceiling of 54 × 1.05). They pass in `perf-guard.yml` on Linux, where CI measured 39.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each named test passes on dev, or is retired with a recorded reason; the 14 compact-setup gate nodes are carried by TASK-36511 (owner ruling 2026-10-23)
- [x] #2 The compact-setup gate's attribute list matches the code it pins, so a future move fails with a message that names the moved attribute
- [x] #3 Each named test runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->

## Implementation Plan

1. Compact gate: point the two moved names at `context_spend.ConsoleContextSpendController`, and make every pinned-attribute lookup fail with an AssertionError naming owner and attribute. Show it with a temporary rename.
2. Voice latency: the 9 new locators come from new schema tables (38467b8fc6, d24c4c04b2), so they are legitimate. Re-pin the inventory count and re-split the 1,000-row baseline over 24 forbidden locators.
3. Tab sync: the harness execs the method without module context. The cold FULL case was red from birth (NameError `run_controls` at 3e618d7119). Give the exec namespace `__package__` and stand-ins for the two module globals that the cold path reads.
4. Keystroke reds: a timeline probe shows a cold readiness/context projection refresh, started by the fixture before typing, publishing a full sync into the burst. Replace the flat settle with a poll that waits until that publication chain is quiet. No product change.
5. Lanes: add the files and nodes to perf-guard.yml (plus a Packaging path trigger), measure runtimes, and run them repeatedly under load.

## Implementation Notes

All measured at base dev 52b3207ff1, serially (`-p no:xdist`) unless a parallel run is named. Review round 1 fixes are folded in.

**Compact-setup gate (AC#2).** The gate was stale from birth: in 3e618d7119, which created it, `_build_console_settings_summary_state` and `_active_console_settings_context_estimate` already lived on `context_spend.ConsoleContextSpendController`. The entry now names that class. Two kinds of failure now name what moved:
- A pinned product attribute that is missing raises `AssertionError: compact gate pins <module>.<Class>.<name>, which no longer exists there; ...` from `_pinned()`. That covers the external-caller lists, the config, private_paths, compact_model_bar and participants pins, `sync_from_sidebar`, `MessagePump._parent`, and the two unguarded `__wrapped__` reads.
- A missing owner class is looked up through `_pinned(module, class_name)`. Before, it was skipped silently.

Raw lookups remain only for the gate's own methods, the `hasattr`-guarded `__wrapped__` reads, and the observer's defaulted runtime checks. Evidence:
- Moved method: re-listing the moved names under `ChatScreen` gave the named message in both files.
- Moved class: renaming the listed class to `ConsoleContextSpendControllerMoved` under the old lookup let checked-read `[compose]` pass, with that owner's pins dropped. Under `_pinned`, both nodes fail naming `...context_spend.ConsoleContextSpendControllerMoved`.

**The 14 compact nodes stay off the lane; TASK-36511 carries them (owner ruling 2026-10-23: keep the 100 ms bound, fix the attach).** With install passing, the 100 ms loop probe (`progress.wait(0.1)` in `_control`) fails on 1-5 of 14 nodes per run at load 9-18. Every failure was at script line 211 (`action_task` never created) or line 104 ("read blocked its actual shared loop"). Probe delays were bimodal: 0 ms, or 179-271 ms. `TLDW_LOCAL_COMPACT_STACKS=1` stacks put Main in synchronous Console attach work: `_reconcile_console_after_attach`, `_prepare_allocation_reconcile`, and `_sync_native_console_chat_ui` -> `_build_console_provider_selection`. Last run: 16 passed, 1 failed at line 211.

**Voice latency.** Re-pinning the count is correct: the inventory rose from 19 to 28 because 38467b8fc6 and d24c4c04b2 classified nine new schema locators (hook-continuation receipts, fleet progress, machine follow-up receipts, response rules). The "drifted" error was the 1,000-row split: 71/66 over 15 forbidden locators became 49/41 over 24.

The history node then exposed a real regression, so it is now `xfail(strict=True, raises=AssertionError)` naming TASK-36510:
- Eight locator columns have no leading index, so the promotion's `SELECT 1 ... WHERE col IN (?, ?)` scans them. They are both message columns of `console_hook_continuation_receipts` and of `console_machine_followup_receipts`, plus `message_id` on `console_response_rule_drafts`, `_fixtures`, `_validations` and `_assessments`. The assessments index is `(conversation_id, message_id)`.
- The gate reports 14 scans (7 x 2) only because it runs ANALYZE after seeding with an all-NULL `conversation_id`, so the planner skip-scans the assessments index.
- The fresh-commit median paired delta is 5.1 ms against a 2.0 ms limit. Indexing seven of the columns in a probe gave scan_count 0 and a 0.06 ms delta.
- Result: XFAIL, and `--runxfail` shows `assert 14 == 0` at the scan-count assert. A strict XPASS fails the lane step, so whoever fixes TASK-36510 removes the marker.

**Tab sync.** This was a harness gap, not a product bug. The cold FULL control was red from birth: `NameError: run_controls` at 3e618d7119, then `KeyError: '__name__' not in globals` once 267413eb56 added `from ..Console_Modules import config_sync`. Both were shown by running each commit's test against its own chat_screen. The exec namespace now has `__package__`, a stand-in `ChatScreen`, and a stand-in `run_controls` whose poll source is None, which is what the real one returns for a non-ChatScreen. The real `poll_source` imports chat_screen, which the teardown check forbids. Red at dev: 1 failed, 23 passed. Now 24 passed.

**Rebase onto dev 5b5920b8e2.** Dev's a896d33fdd (#3191) moved chat_screen's `config_sync` import to module level and stubbed `begin_console_sync` in this harness. The resolution compiles config_sync's installed source into the namespace, so the cold FULL control runs its real `sync_console_core_state`, `project_console_tick` and `replay_whole_console_sync`. `__package__` stays: without it the cold path's function-local `character_context` import raises the `KeyError`. Evidence: real code 24/24; `replay_whole_console_sync` returning True is red ("ran past its deferred projection"); dropping its retry is red (`[] == [{'delayed': True}]`).

**Keystroke reds: two setup-cost modes, no product change.** The call sites are old (character.py:368 is 9aa12e5584, TASK-22204; message.py:995 is 391b7bf69a). Both modes end in one full sync inside the burst, which takes 3 snapshots: 2 from avatar, 1 from transcript.
- *In-flight cold read.* A probe on `run_worker`/`_record_ui_worker_started` caught 5 failures in 12 runs (6 parallel x 2). Each time, a `console-readiness-config` refresh started before typing was still running at COUNT-ON. `_settle_publications()` waits for those groups to be quiet; with it, 0 of 12 runs showed the mode.
- *Trailing refresh, found by the review (3 of 9 serial runs).* The composer's 0.2 s trailing draft-spend refresh fires on any gap over 0.2 s between presses. Its `ConsoleContextReadSnapshot.inputs` read was the first since the fixture's back-door `_ingest_linear_messages`, so it was cold. That work starts inside the burst, so no settle can see it. The census now calls `_sync_console_settings_summary()` and `_sync_console_cost_chip()`, the two calls the refresh makes, with the other cold reads, before the settle. The trailing refresh itself is untouched.

Evidence for the second mode:
- Forced probe, a 0.35 s gap after key 0, without the warm: all 3 behaviour nodes red (400 messages: 3 calls; empty: 2).
- The same probe with the warm: 3 green.
- Three 1.2 s gaps (TTL-expired steady-state refreshes) with the warm: green, so a settled pause does not publish.
- Unforced, with the warm: 24 of 24 serial runs of both nodes at load 7.6-12.3 (12 sequential, then 3 concurrent loops of 4). Without the warm, 9 of 9 unforced runs also passed here, so the forced probe, not the run count, is the red/green evidence.
- Whole file: 10/10 serially, storage ratchets included.

**Lanes (AC#3).** perf-guard.yml gets two steps:
- "Console tab-publication and voice gates" runs the tab-sync file and the whole voice file: 26 passed, 1 xfailed in 3 s. The latency gate runs on a synthetic clock.
- "Console keystroke work census" runs the two named nodes: 2 passed in 41.7 s. Run 39931853847 used about 6 of the job's 15 minutes.

The workflow also gets a `Packaging/speculative_voice_*.py` path trigger. The compact gates stay off, with TASK-36511 named in the comment. No Tests/CI contract pins perf-guard's targets. Tests/CI: 1025 passed, 1 skipped.

Files: Tests/Performance/_compact_original_config_gate.py, test_speculative_voice_latency.py, test_console_tab_sync_source_ownership.py, test_console_keystroke_work_census.py; Packaging/speculative_voice_history_gate.py; .github/workflows/perf-guard.yml; backlog/docs/lessons-testing-evidence.md.
