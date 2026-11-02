---
id: TASK-38644
title: "Console: the late-opened Inspector parity test can start its second app with the first app's rail still open"
status: To Do
created_date: 2026-11-02 21:52
labels:
- follow-up
- tests
- console
- ci-flake
priority: medium
references:
- Tests/UI/test_console_collapsed_inspector_send_layout.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). `Tests/UI/test_console_collapsed_inspector_send_layout.py::test_an_inspector_opened_after_a_send_matches_one_open_throughout` is a UI census entry, so a failure fails the required check.

**CI failures.** It failed twice in UI Fast Lane (4) on 2026-10-31, both times at `assert not rail.display` (:302), which means the second app started with the Inspector open:
- run 41189972193 attempt 1, on fu/h-settingsreds (#3315), which changed only `settings_screen.py`; green on rerun;
- run 41224394415 attempt 1, on fu/h-railstate @ edd6260383 (#3225's own branch), which changed Console rail-state code (`chat_screen.py`, `Chat/console_rail_state.py`), including d02046bf61 "retain first-use rail defaults across restarts".

**Candidate causes.** The two candidates below are not yet told apart.
- **Worker-shutdown race.** The test boots two apps in one private profile, so they share `config.toml`. The first app collapses the rail, then relies on a 0.3 s pause (:312-315, "Leave the stored rail preference collapsed for the next arm"). That save is a background-thread worker (`_save_console_rail_preferences`, chat_screen.py:11815), and Textual's shutdown cancels workers without waiting for them. So the second app can read the rail as still stored open. This fits the #3315 failure, which ran without any rail-state change.
- **Retained first-use seeds.** d02046bf61's change reached dev as 71ddbecdbb in #3225 (merged 2026-10-31 22:24Z). Since then the mount latches a first-use rail seed. `_console_rail_state_config()` (chat_screen.py:11808) re-injects that seed whenever the stored key is absent, and TASK-38643's cause C shows it overriding records staged after mount. If the second app gets a seed that says open, it starts with the rail open whatever the first app saved.

**A second symptom, locally.** In 12 runs under `-n 4` load at dev 88af9f9379, it failed once at `assert opened_late == open_throughout` (:352). The rail opened after the send painted no "▼ more sections — scroll" hint. That is TASK-37000.5's missing fold hint, here without a chat switch.

TASK-37000 (Done) built this test, which last changed in a9ea81a09f (2026-10-27).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the open-rail start is identified (worker-shutdown race, retained first-use seed, or another) and recorded; if it is a product rule, that rule is pinned rather than worked around
- [ ] #2 The second app starts with the Inspector collapsed in 10 of 10 runs under -n 4 load, and not because of a longer fixed pause
- [ ] #3 The witness-row, fold-hint and painted-frame comparisons stay as strict as they are now
- [ ] #4 Any remaining fold-hint mismatch is recorded on TASK-37000.5 with this test's evidence
<!-- AC:END -->
