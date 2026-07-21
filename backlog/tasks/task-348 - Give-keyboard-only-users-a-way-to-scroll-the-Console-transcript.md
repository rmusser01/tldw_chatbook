---
id: TASK-348
title: Give keyboard-only users a way to scroll the Console transcript
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With an overflowing transcript loaded (idle): PageUp with composer focus does nothing; cycling panes with F6 four times and pressing PageUp after each cycle never scrolled the transcript; Home never worked. PageUp/ArrowUp scroll only after clicking directly on message text with the mouse - a mouse-only affordance in a footer-advertised keyboard-first UI ('F6 next pane'). In a terminal app this makes scrollback effectively mouse-gated.

**Repro:** Load a long conversation -> without touching the mouse press PageUp, then F6+PageUp repeatedly (4 cycles) -> view never moves; click any message text with the mouse -> PageUp now scrolls.

**Verifier note:** Live observation accepted, but the stated mechanism is wrong: the transcript IS in the F6 cycle (CONSOLE_FOCUS_TARGETS_BY_PANE maps console-transcript-surface → console-native-transcript, chat_screen.py:371-379; _focus_console_workbench_target force-sets can_focus and focuses) and Tests/UI/test_workbench_pane_focus.py::test_console_f6_cycles_between_workbench_panes_and_wraps_backward passes in this worktree (ran it: 1 passed). ConsoleTranscript(VerticalScroll) inherits PageUp/PageDown bindings. So the live failure is most plausibly focus being silently stolen back (sync/refocus between F6 and PageUp) and/or invisible focus state making the landed pane unknowable — a real keyboard-only gap not covered by any ledger item (keyboard-bindings covers c/e/r + selection arrows only). P2 appropriate; needs live-rig reproduction against the passing harness.

**Source:** Console UX expert review 2026-07-20 (finding j4-keyboard-cannot-reach-transcript-scroll; P2, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-30b-idle-pageup-after-click.png`, `j4-29-wal-loaded.png`, `j4-30a-idle-wheel.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 F6 pane cycling should be able to land focus on the transcript scroller (with a visible focus indicator), after which PageUp/PageDown/Home/End scroll it
<!-- AC:END -->
