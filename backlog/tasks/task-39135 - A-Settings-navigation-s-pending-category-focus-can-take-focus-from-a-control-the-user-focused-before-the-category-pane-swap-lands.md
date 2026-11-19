---
id: TASK-39135
title: "A Settings navigation's pending category focus can take focus from a control the user focused before the category pane swap lands"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- settings
- ui
- focus
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_configuration_hub.py
dependencies:
- TASK-38953
- TASK-39046
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39046 fixed only its test and lists this product edge as a follow-up. Line numbers are dev's; round-11 branch fu/r11-settings leaves this path unchanged.

- A real category switch records a pending rail focus (`_select_category`, `settings_screen.py:25800`).
- The asynchronous pane swap then applies it after its recompose (`:4774-4777`). It prefers it over the `focus_id` captured before the swap (`:4762`, `:4778`), and nothing checks whether the user moved focus in between.
- TASK-39046's probe saw the swap take focus about 40 ms after a Review press, and the pressed control never got it back. The window is the swap's duration, so it grows with load.
- TASK-2831 (To Do) is the opposite race, a pending focus lost too early. Any change here must not bring back that loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the user moves focus to a control after a navigation but before the pane swap lands (held deterministically in a test), that control keeps focus after the swap; the test is red on dev
- [ ] #2 When the user has not moved focus, the navigation's category focus still lands; the existing category switch, deep-link and Theme-list focus pins stay green, named in the notes
- [ ] #3 The behaviour rides a deterministic state check, not a timer
<!-- AC:END -->
