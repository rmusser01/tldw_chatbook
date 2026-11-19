---
id: TASK-38945
title: "Console rebuilds the workspace context on every sync, because its build fingerprint cannot prove the inputs unchanged"
status: Won't Do
assignee:
  - '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 17:29
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Console_Modules/workspace.py
- tldw_chatbook/Workspaces/display_state.py
- Tests/UI/test_console_projection_reads_off_loop.py
- Tests/UI/test_console_workspace_display_reads_off_loop.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

TASK-36511.2 (Done) brought the Console down to one workspace-context build per sync. It narrowed its AC#2 to that, and its notes leave a cross-sync cache "as a follow-up for the coordinator to file". Nobody filed it, so here it is.

- The build cache (`ConsoleTickWorkspaceBuilds`) lives only for one scope: `tick_workspace_build_scope` (`tldw_chatbook/UI/Console_Modules/workspace.py:6973`) and `live_workspace_build_scope` (`:7003`).
- The cache cannot simply outlive the scope. `_console_workspace_build_fingerprint` (`workspace.py:7510`) says in its docstring (`:7523`) that it is "Deliberately NOT exhaustive": stars, unseen markers and conversation-browser collapse preferences are not in it. The build reads them at `:4652` (`_console_conversation_browser_collapse_preferences`, defined at `:1705`) and through `fleet_unseen_ids_accessor` (`:697`).
- TASK-36511.2 measured each build at 3.2-3.6 ms of CPU per sync at load 15. Whether that adds up to enough to justify a cache is not known. AC#1 measures it first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace-context builds per minute, their CPU, and how many had inputs identical to the previous build are measured for an idle Console and for one during a run; if the saving does not justify a cache, the notes record the numbers and the task closes as Won't Do
- [ ] #2 Consecutive syncs with unchanged inputs reuse one build, shown by a build_console_workspace_state call count over N syncs before and after the change
- [ ] #3 Changing any single build input between two syncs produces a fresh build: every input the fingerprint covers today, plus a star, an unseen marker and a group collapse preference; each of those last three is shown red with that input temporarily removed from the reuse key
- [ ] #4 test_a_control_bar_sync_builds_the_workspace_context_once and test_a_control_bar_sync_reads_a_moved_registry_live_during_a_run pass unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first (AC#1) with a scratch probe in the send harness at 160x45, through run.sh, on the base and the branch: count live workspace-context builds (a `_build_console_workspace_context_state` call not served from an open build scope) over 60 s for an idle Console and for one streaming a run, with each build's thread CPU, and how many had the same inputs as the previous build (the build fingerprint plus the unseen markers and collapse preferences it leaves out) and the same output.
2. Decide from the numbers. If an idle Console builds rarely and a run's builds mostly see changed inputs, the saving cannot pay for a cross-sync cache, whose reuse key would also have to cover the long-tail inputs and skip the build's side effects; record the numbers and close the task as Won't Do (AC#1).
3. Only if the numbers justify it: extend the reuse key with stars, unseen markers and collapse preferences, keep one validated build across syncs, prove reuse with a `build_console_workspace_state` count over N syncs (AC#2), prove each input forces a rebuild, with the last three red when removed from the key (AC#3), and keep the two named tests unchanged (AC#4).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Closed as Won't Do under AC#1: the measured saving does not justify a cross-sync cache.** No product code changed.

**Measured (AC#1)** on the base (dev 2684e5453f), in the mounted send harness at 160x45, through run.sh, at load 31-55. The probe was a scratch file, `test_r11c_ws_builds.py`, not committed. It counts each live workspace-context build: a `_build_console_workspace_context_state` call that is not served from an open build scope, including the one inside a scope's shared build. For each build it records the build's thread CPU, and compares it with the previous build twice:
- same inputs: the build fingerprint, the unseen markers and the collapse preferences;
- same output: the built state.

| 60 s window | builds | per minute | CPU | median per build | same inputs as the previous build | same output |
|---|---|---|---|---|---|---|
| Idle Console, after a 3 s settle | 0 | 0 | 0 ms | — | — | — |
| One streaming run (a chunk every 0.25 s; the run poll ticks every 0.2 s) | 571 | 571 | 290 ms | 0.42 ms | 570 | 569 |
| 10 explicit `_sync_console_control_bar()` calls, for scale | 14 | — | 17 ms | 0.60 ms | 13 | 13 |

**Why it does not pay:**
- An idle Console builds nothing, so the cache would save nothing there.
- During a run nearly every build repeats the one before it, but all of them together cost about 290 ms of CPU per minute of run: 0.5% of one core, about 2 builds per 0.2 s tick. That is the most a cross-sync cache could save in this harness.
- To be correct, such a cache would have to:
  - extend the reuse key to the stars (read through the marks service), the unseen markers and the collapse preferences;
  - keep each build's side effects: the workspace-files availability refresh requests and the canonical-owner pruning;
  - keep ADR-028's per-build recompute of filesystem binding status from disk, which skipping a build would drop.
- AC#4 also pins that every control-bar sync builds once, so only the run ticks could reuse a build.

**Caveat.** TASK-36511.2 recorded 3.2-3.6 ms of workspace-context CPU per control-bar sync in the same harness, against 0.4-0.6 ms per build here. This probe times the build alone with `time.thread_time()` and no profiler hook. A Console with a real workspace registry and many browser rows builds more per call. If a real-app profile ever shows multi-millisecond builds during a run, revisit a cache that reuses builds only across run ticks.

**Files:** this task file only.
<!-- SECTION:NOTES:END -->
