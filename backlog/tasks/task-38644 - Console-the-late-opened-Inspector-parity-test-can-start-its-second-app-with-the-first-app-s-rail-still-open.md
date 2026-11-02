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
- run 41189972193 attempt 1 (green on rerun);
- run 41224394415 attempt 1.

Neither PR touched Console code.

**Likely cause.** The test boots two apps in one private profile, and they share `config.toml`. The first app collapses the rail and then relies on a 0.3 s pause (:312-315, "Leave the stored rail preference collapsed for the next arm"). That save is a background-thread worker (`_save_console_rail_preferences`, chat_screen.py:11815). Textual's shutdown cancels workers without waiting for them. So the second app can read the rail as still stored open.

**A second symptom, locally.** In 12 runs under `-n 4` load at dev 88af9f9379, it failed once at `assert opened_late == open_throughout` (:352). The rail opened after the send painted no "▼ more sections — scroll" hint. That is TASK-37000.5's missing fold hint, here without a chat switch.

TASK-37000 (Done) built this test (last changed a9ea81a09f, 2026-10-27). #3225 (merged 2026-10-31 22:24Z) changed how rail state is seeded at start-up, but not this toggle save.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The second app starts with the Inspector collapsed in 10 of 10 runs under -n 4 load, because the test waits for the stored collapse or sets it explicitly, not because of a longer fixed pause
- [ ] #2 The witness-row, fold-hint and painted-frame comparisons stay as strict as they are now
- [ ] #3 Any remaining fold-hint mismatch is recorded on TASK-37000.5 with this test's evidence
<!-- AC:END -->
