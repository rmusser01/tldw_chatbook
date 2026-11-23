---
id: TASK-42464
title: "Find why the Console edge-focus restyle failures started on 2026-11-22 at 04:05Z, or close with what was ruled out"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- ci
- console
- investigation
priority: low
references:
- Tests/UI/test_console_edge_focus_restyle_scope.py
- Tests/UI/test_console_shell_regions.py
- tldw_chatbook/Widgets/Console/console_setup_modal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42416. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-42416 fixed the cause of `Tests/UI/test_console_edge_focus_restyle_scope.py`'s CI failures. The shell harness (`make_console_pilot` in `Tests/UI/test_console_shell_regions.py`) hid the first-run setup overlay once, after a fixed pause. Every later `ConsoleSetupModal.sync_card_state` (`Widgets/Console/console_setup_modal.py:243`) set it back to blocking, so a sync landing after the hide covered the Inspect handle and `pilot.click` returned False. TASK-42416's local spy counted 11-14 setup syncs before the hide, from the credential-readiness poll's first pass, `_refresh_console_hooks`, the identity-surface retry and the transcript ticks.

That race existed before 2026-11-22, yet in TASK-42416's scan of 140 Derived Artifacts PR runs (2026-11-20 00:00Z to about 21:50Z on 2026-11-22, every attempt) the PR fast-lane job that runs the file failed 9 times, all on this file, from 2026-11-22 04:05Z to 17:57Z, and never earlier. The first failure came about two hours after round 19's merges. Something raised the rate then: a change that added a later setup sync or moved one, or a runner change. If a product change made the overlay re-show later after startup, a real first run would see that too. Close the task if nothing is found.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes list the commits merged to dev from 2026-11-21 00:00Z to 2026-11-22 04:05Z that touch the setup overlay's sync sources or the shell harness, each marked with whether it changes when the overlay is re-shown after startup
- [ ] #2 If one does, the notes name it with a local before/after measure (the setup syncs after the harness's hide, by count or timing); a user-visible change to the first-run overlay is fixed with a test shown red first, or recorded as a proposed follow-up
- [ ] #3 If none does, the task closes with what was ruled out
<!-- AC:END -->
