---
id: TASK-3070
title: chat_screen size ratchet red on dev after console decomposition wave 3
status: To Do
assignee: []
created_date: '2026-08-07 18:20'
labels:
  - console
  - architecture-gate
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/chat_screen.py]`
fails on dev: 18,930 lines against a budget of 18,909. Introduced by PR #1408 (console
decomposition wave 3) itself — confirmed byte-identical on a clean dev-tip worktree at
`15407a641` during the TASK-3035/3045 architecture-gate refresh (PR #1416), so it is not
an artifact of any other branch. The decomposition stream is actively shrinking this
screen; the 21-line overage is presumably transitional. Filed so the red gate is owned
rather than becoming the next "pre-existing noise" that hides something real (see
lessons-testing-evidence.md's TASK-2610 entry for how that ends). Resolve by shrinking
the screen below budget in the next wave — not by raising the budget, unless the
decomposition stream's owner explicitly decides the budget is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The chat_screen size-ratchet test passes on dev
- [ ] #2 The resolution shrinks the screen (or an explicit, documented owner decision adjusts the budget)
<!-- AC:END -->

## Addendum 2026-08-08 — the overage is now 540 lines, and its cause changed

When filed, this was a 21-line transitional overage from wave 3 itself. It is now
**18,267 lines against a budget of 17,727 (+540)**, and the cause is no longer the
decomposition stream: waves 4 and 5 both LOWERED the screen (18,930 -> 17,727 ->
17,685). The growth arrived from feature work merging past a red gate — six commits,
chiefly the TASK-2154 UX remediation and the collapsed-rail work.

**The budget was never raised**, which matters: the mechanism refused correctly and
the refusal was merged past rather than defeated. So this is the first time the
ratchet has fired against real feature growth rather than a wave, which is exactly
what it was built for.

Measured inventory of the growth (`ast`, dev at the wave-5 rebase):

- **11 new `ChatScreen` methods, 221 lines.** Most have an obvious existing home:
  `_stack_collapsed_rail_labels` (10) and `_reveal_console_inspector_rail` (15) ->
  the rail regions; `_adapt_console_workspace_to_width` (23) and
  `_console_tab_region_selector` (28) -> `workspace.py`; `_console_run_chip_activated`
  (6), `_console_tools_chip_activated` (6), `_console_sources_chip_activated` (11),
  `_console_active_run_copy` (17), `_notify_console_run_failure` (16) -> the agent or
  session controllers. `_sync_console_compact_status_marker` (48) and
  `_console_library_provider_factory` (41) need a judgement call.
- **+251 lines in existing methods**, led by `compose_content` (314 -> 357),
  `_sync_console_rail_visibility` (42 -> 64), `action_focus_next` (14 -> 31),
  `_sync_console_mode_bar` (12 -> 28), `on_key` (309 -> 325).

Wave 5 deliberately did NOT touch the budget: lowering it to the 42 lines its
composer-keymap task earned would have been meaningless against a +540 overage, and
raising it to the measured number would have defeated the mechanism. The number is
left exactly as dev has it.
