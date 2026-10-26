---
id: TASK-35970
title: >-
  Console: the collapsed Character section recomposes twice per send
  (Refreshing..., then the result)
status: Done
assignee:
  - '@claude'
created_date: '2026-10-21 04:37'
updated_date: '2026-10-26 12:50'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while measuring TASK-33620.15.7 (2026-10-20, live, 160x45, a 36-turn chat, Anthropic claude-haiku-4-5 through the review proxy). After the run-tick fixes there, about 12-13 whole-screen reflows remain per send window; two of them come from ConsoleCharacterContext alone. Every send ends with a Character refresh that publishes a REFRESHING state (loading=True) and then the loaded result (data_revision and scope_fingerprint move), and ConsoleCharacterContext.sync_state recomposes the whole section, search Input included, for each, whether or not the Character section is expanded. Measured: 2 recomposes per send live (8.4-74 ms reflows each); in the mounted test harness the refresh errors and the section recomposed every few poll ticks while a run was held (Tests/UI/test_console_run_tick_layout_requests.py exempts it and names this task).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A send with the Character section collapsed causes no Character-section recompose or layout request
- [x] #2 Expanding the Character section after such a send shows the current chats with no stale Refreshing line
- [x] #3 The held-run layout-request test in Tests/UI/test_console_run_tick_layout_requests.py no longer exempts the Character section
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on dev: count Character publishes, recomposes and layout requests over a held send with the section collapsed (probe).
2. Tests first: drop the Character exemption from the held-run test (AC#3); add a real-send test counting Character-section recomposes and layout requests with the section collapsed, then expanding it (AC#1, AC#2); add a rail-level test that a collapsed section renders the newest chats on expand with no Refreshing line (AC#2).
3. Fix in ConsoleCharacterContext: while the browser or an ancestor is not displayed, keep the snapshot and skip the recompose and identity-line write; render the newest snapshot when the section opens (rail disclosure hook) or the browser is shown again (Show event). Controller semantics untouched.
4. Mutation-check the tests, run Character/rail test files against a dev worktree, rebase, preflight, live paired check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** The Character browser now renders only while it can be seen. `ConsoleCharacterContext.sync_state` checks whether the browser and every ancestor are displayed (`_presented`). When one is not (the section collapsed, or the whole rail hidden), it stores the newest snapshot and returns, skipping the recompose and the identity-line write. The 2 Hz progress-count sweep pauses the same way. `compose()` clears the deferred flag, since it always renders the newest `_state`. Two paths render the deferred snapshot:
- `ConsoleLeftRail.apply_section_open` calls `render_deferred()` as the Character body opens, before `display` flips, so the first painted frame already shows the newest chats.
- `on_show` handles any other ancestor that hid the browser.

The controller is untouched: its scope check, the one-handle merged check and idle memo (TASK-33620.15.6), the run recheck interval (TASK-33620.15.1) and every published state stay as they were, so `controller.state`, the collapsed header title and the rail-visibility input still change when the data changes.

**A regression found live and fixed.** The first version opened the section with the rows below the fold, where dev scrolls them into view. While the deferred render was still mounting, an allocation pass measured the Character body at 0 rows. `fallback_active_section` then moved the active section to Conversations and dropped the open's deliberate reveal. The rail now holds allocation passes (`_character_render_pending`) until the render has mounted (task done-callback → `_character_render_settled` → one reconcile), and the reveal then runs as on dev. The mounted test reproduces this only when the rail's activation and the screen's toggle run back to back, the order the live app uses. Through `pilot.click`, the harness runs a queued pass in between that drops the reveal on dev too. Lesson recorded in `lessons-testing-evidence.md`.

**The hidden caption.** Live, one Character-section layout request was left per app session, on its first send. The avatar painter's first write changed the never-displayed `#console-character-name` caption from the screen's "Roleplay character" to "No character in this chat". `sync_character_caption` now passes `layout=widget.display`, so a hidden caption takes new copy without a layout, and the shown Reaction line lays out as before.

**Tests (red on dev 0d7e418db1, or on the previous head for the caption, and green here).** `Tests/UI/test_console_character_section_collapsed_render.py` is added to the UI PR gate (~18 s; floor 906 -> 907). Its cases:
- A real held send with the section collapsed: dev had 14 recomposes, the branch has 0. Opening the section afterwards shows the current state.
- A send-end check over a controller that loads real groups, publishing Refreshing then the result: dev had 2 recomposes, the branch has 0. On opening, the rows are the new chats in every painted frame, and the rows are revealed.
- A browser hidden by another ancestor, then shown again.
- The hidden caption's new copy requests no layout, with a positive control for the shown Reaction line.

The held-run test in `test_console_run_tick_layout_requests.py` drops its Character exemption; dev made 8 Character layout requests over six poll ticks. Mutation checks:
- No rail hook: one painted frame shows the old chats.
- No `on_show`: the hidden-ancestor case stays stale.
- No allocation gate: the rows end below the fold, with Conversations active.
- No sweep gate: the new sweep-stub case fails.

Two existing tests were adjusted. `test_production_avatar_repaint_never_overwrites_identity_variants` opens the section first, because the identity line now renders only while it can be seen. The sweep stub gained a `_presented` seam. In `test_render_awaits_never_resume_a_stale_avatar_paint`, the one-argument caption `update` doubles now accept `**kwargs`: the painter passes `layout=`, and the old doubles raised inside it and recorded no write. Locally, most Console rail files fail at setup with `RecoveryRequired: raw_source_selection_changed`, so I also ran them on the bootstrap profile (a probe-only plugin, not committed) against a detached dev worktree, comparing by test id:
- 14 files show no branch-only failure: the left rail, rail reconciliation, context rail content, Character context, async ownership, progress owners, the switcher and its geometry, Character context geometry, resize reflow, workbench contract, presentation config, screen reuse, and radio portrait.
- The one difference was rail reconciliation's `[sources]` case, which failed once in the loaded batch and then passed 3 of 3 alone.

The timer-path inventory fails identically on both arms.

**Live (real app, Anthropic claude-haiku-4-5 via the review proxy, 160x45, isolated profiles, Character collapsed, plain chat, paired ABBA, 12 sends per arm per round, `uxrev/evidence/bd5/CH/`).**
- Round 1: dev had 26 Character recomposes and 106 Character-section layout requests; the branch had 0 and 2.
- Final head 2b6e5c5980: dev had 28 and 114; the branch had 0 and 0, on every send including the first.
- Median whole-screen reflows per send were within noise at load average 8-19: 17.0 vs 15.5 in round 1, then 18.0 vs 18.0 on the final head. A recomposing collapsed section's layout requests mostly coalesce into reflows the send makes anyway.
- On expanding the section after the sends, no capture on either arm shows "Refreshing local character chats…". The branch renders the newest state once, and the deliberate reveal runs as on dev (current=True; outer scroll 10 vs dev 10-11).

**Observed, not changed.** In most rounds on both arms (dev 5 of 7), the Chats tree's "Today" group did not list the renamed chat after its first send. It is unrelated to this change.

**Files.** `Widgets/Console/console_character_context.py`, `UI/Console_Modules/left_rail.py`, `Tests/UI/test_console_character_section_collapsed_render.py`, `Tests/UI/test_console_run_tick_layout_requests.py`, `Tests/UI/test_console_character_context_sweep_gate.py`, `Tests/UI/test_console_character_avatar.py`, `Tests/UI/test_console_character_context.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, the User Guide's Character section paragraph (`Docs/User_Guide/console/sessions-tabs-workspaces.md`) and `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
