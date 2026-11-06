---
id: TASK-38950
title: "Every return to the Console restyles the whole screen, and a vertical breakpoint flip restyles it again"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Navigation/base_app_screen.py
- Tests/Performance/run_console_mount_profile.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-33273 (To Do) says "Warm resume restyles all 533 nodes", but none of its acceptance criteria covers it. TASK-36511.5 (Done, no code change) measured it and concluded it needs a design change.

- In Textual 8.2.8, `Screen._on_screen_resume` restyles every widget on the screen (`update_node_styles(animate=False)`) when `refresh_styles` is set (`textual/screen.py:1465-1484`). `ScreenResume.refresh_styles` defaults to True (`textual/events.py:919`), and `switch_screen` and `push_screen` post it with that default (`textual/app.py:2616`, `:2622`, `:2956`).
- `Screen._on_resize` calls `update_classes(breakpoints)` (`textual/screen.py:1510-1545`). When a `BaseAppScreen.VERTICAL_BREAKPOINTS` class flips (`tldw_chatbook/UI/Navigation/base_app_screen.py:66`), the whole screen restyles again.
- TASK-36511.5 measured:
  - in the send harness, 538-554 of 555 nodes restyled per resume, for 55-58 ms of CPU at load 9;
  - with the sampler on the compact gates, a 285 ms CPU median per resume, with the breakpoint restyle in 27% of the resume samples;
  - with the warm-visit profiler, a 205 ms median to full ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A return to the Console with no style input changed while it was away restyles fewer nodes than today's 538-554 of 555 in the send harness, with per-resume counts recorded before and after; or, if that is impossible without re-implementing Textual's _on_screen_resume, the notes say why, backed by a measured prototype
- [ ] #2 Any reduction keeps everything else the resume handler does: removing the suspended class, bumping stack_updates, refreshing notifications, setting auto-focus, and re-laying out after a size change. A theme or CSS change made while the Console was away still shows on return, and a test for that is red under a variant that always skips the restyle
- [ ] #3 A shell-header-compact/shell-header-normal breakpoint flip restyles only widgets whose rules depend on those classes, counted, or the notes say why not
- [ ] #4 The warm-visit profiler's full-ready median and the first-mount and boot-budget tests do not regress
<!-- AC:END -->
