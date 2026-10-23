---
id: TASK-36390
title: Tests/Performance compact-setup, voice-latency, tab-sync and keystroke tests are red on dev and run in no PR lane
status: In Progress
assignee: ['@claude']
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 21:58
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
- [ ] #1 Each named test passes on dev, or is retired with a recorded reason
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

All measured at base dev 52b3207ff1, serially (`-p no:xdist`) unless a parallel run is named.

**Compact-setup gate (AC#2).** The gate was stale from birth: in 3e618d7119, which created it, `_build_console_settings_summary_state` and `_active_console_settings_context_estimate` already lived on `context_spend.ConsoleContextSpendController`. The entry now names that class. Every pinned lookup goes through `_pinned()`, which raises `AssertionError: compact gate pins <module>.<Class>.<name>, which no longer exists there; ...`. Red: re-listing the moved names under `ChatScreen` gave that message for both files. Green: the 14 nodes get past `install()`.

**But the 14 compact nodes are load-flaky, so they stay off the lane (AC#1 not met for them).** Once install passes, a 100 ms loop probe (`progress.wait(0.1)` in `_control`) fails on 3-5 of 14 nodes per run, at load average 9-18. Every failure was at script line 211 (`action_task` never created) or line 104 ("read blocked its actual shared loop"). The measured probe delays were bimodal: 0 ms, or 179-271 ms. `TLDW_LOCAL_COMPACT_STACKS=1` stacks put Main in synchronous Console work at the probe: `_reconcile_console_after_attach`, `_prepare_allocation_reconcile`, and `_sync_native_console_chat_ui` -> `_build_console_provider_selection`. Whether the bound should be "answers while held", and whether those segments are acceptable, is a decision left to the owner.

**Voice latency.** Re-pinning the count is correct: the inventory rose from 19 to 28 because 38467b8fc6 and d24c4c04b2 classified nine new schema locators (hook-continuation receipts, fleet progress, machine follow-up receipts, response rules). The "drifted" error was the 1,000-row split: 71/66 over 15 forbidden locators became 49/41 over 24. The inventory and warm-latency nodes pass and are on the lane. **The history node is still red (AC#1 not met), on a real regression.** `scan_count == 14`: 7 of the new locator columns have no index, so the promotion's `SELECT 1 ... WHERE col IN (?, ?)` scans them. Those columns are `console_hook_continuation_receipts`.(parent_)assistant_message_id, `console_machine_followup_receipts`.(parent_)assistant_message_id, and `console_response_rule_drafts`/`_fixtures`/`_validations`.message_id. The fresh-commit median paired delta is 5.1 ms against a 2.0 ms limit. A probe that indexed those 7 columns gave scan_count 0, a 0.06 ms delta and a pass. The fix needs a schema migration and is out of scope here.

**Tab sync.** This was a harness gap, not a product bug. The cold FULL control was red from birth: `NameError: run_controls` at 3e618d7119, then `KeyError: '__name__' not in globals` once 267413eb56 added `from ..Console_Modules import config_sync`. Both were shown by running each commit's test against its own chat_screen. The exec namespace now has `__package__`, a stand-in `ChatScreen`, and a stand-in `run_controls` whose poll source is None (what the real one returns for a non-ChatScreen). The real module imports Textual, which this file must not. Red at dev: 1 failed, 23 passed. Now 24 passed.

**Keystroke reds: a census settle gap, not a typing regression.** The call sites are old (character.py:368 is 9aa12e5584, TASK-22204; message.py:995 is 391b7bf69a). A probe wrapping `run_worker` and `_record_ui_worker_started` caught 5 failures in 12 runs (6 parallel x 2). In every one, a `console-readiness-config` refresh was still RUNNING when counting began. A `console-context-publication` full sync then ran at key 1 and took 3 snapshots: 2 from avatar, 1 from transcript. The empty run never showed this, and those reads were started before typing. `_settle_publications()` now waits for no worker in the publication groups, no sync in progress and no coalesced sync scheduled, 3 polls in a row, with a 60 s deadline. With the same probe, 0 of 12 runs had in-window snapshots, and every COUNT-ON showed only the app scheduler worker. The whole file passed 10/10 serially, storage ratchets included. No product change.

**Lanes (AC#3).** perf-guard.yml gets two steps. "Console tab-publication and voice gates" runs the tab-sync file, the voice warm-latency node and the inventory node: 26 passed in 0.7 s; the latency gate runs on a synthetic clock. "Console keystroke work census" runs the two named nodes: 2 passed in 41.7 s, and the job used about 6 of its 15 minutes in run 39931853847. The workflow also gets a `Packaging/speculative_voice_*.py` path trigger. Off the lane, with the reason in the workflow comment: the voice history node (red) and the compact gates (flaky). No Tests/CI contract pins perf-guard's targets; Tests/CI passed (1025 passed, 1 skipped).

Files: Tests/Performance/_compact_original_config_gate.py, test_speculative_voice_latency.py, test_console_tab_sync_source_ownership.py, test_console_keystroke_work_census.py; Packaging/speculative_voice_history_gate.py; .github/workflows/perf-guard.yml; backlog/docs/lessons-testing-evidence.md.

