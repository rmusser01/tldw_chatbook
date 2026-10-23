---
id: TASK-36390
title: Tests/Performance compact-setup, voice-latency, tab-sync and keystroke tests are red on dev and run in no PR lane
status: To Do
created_date: 2026-10-22 23:58
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
- [ ] #2 The compact-setup gate's attribute list matches the code it pins, so a future move fails with a message that names the moved attribute
- [ ] #3 Each named test runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->
