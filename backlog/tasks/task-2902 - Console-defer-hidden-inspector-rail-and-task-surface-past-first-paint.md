---
id: TASK-2902
title: Console — defer the hidden inspector rail and task surface past first paint
status: To Do
assignee: []
created_date: '2026-08-07 02:00'
labels:
  - console
  - performance
  - defer-past-first-paint
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Screen survey (task-2725 follow-up): the Console screen mounts 357 widgets and is the app's default tab, so its cost is also perceived cold-start cost. 124 widgets arrive hidden in three roots: `ConsoleInspectorRail#console-right-rail` (76), `ChatTaskCards#console-task-surface` (32), `CompactModelBar#console-compact-model-bar` (16) — ~35% deferrable.

OWNER RULING 2026-08-07: proceed now (supersedes the soak gate below — 2725/2900/2901 shipped; the owner directed Console next after Schedules re-measurement showed its 1.11s baseline evaporated on current dev, 0.47s, no work needed there). Originally deliberately LAST in the defer-past-first-paint series: `chat_screen.py` is the app's most complex screen and its sync pipeline (`_sync_native_console_chat_ui` and delegates) touches the rail, so the compose→load window audit is substantially harder than 2725/2900/2901. Do not start until both prior tasks have shipped and soaked. The audit must cover: every query of the three roots reachable from the sync path, `restore_state`, the control-bar build, and the session controllers introduced by console-decomposition wave 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] Console first paint excludes the three hidden roots; rail/task-surface/model-bar all function once revealed.
- [ ] Console switch latency improves measurably live; cold-start-to-interactive improves.
- [ ] The full Console test surface stays green (including the worker-lifecycle and generation-actions suites).
- [ ] The compose→load window audit is recorded in the task notes (which query sites can run early, and why each is safe).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Investigation Notes (2026-08-07 — implemented, destabilized a core interaction, PARKED with evidence)

<!-- SECTION:NOTES:BEGIN -->
The owner directed proceeding (soak gate superseded). The full implementation was built and its own tests passed (3/3: mechanism, order-integrity, pre-mount state replay): collapsed-rail deferral (~76 widgets, conditional so restored-open rails stay eager), task-surface deferral (~32) with a `TaskSurfaceMounted` replay, CompactModelBar deliberately excluded (resize-revealed primary control at narrow widths).

**Why it is parked:** the 31-file mount-path console surface exposed a probabilistic regression in `test_console_workspace_conversation_switch_restores_transcript_messages`. A/B under identical machine load: pristine dev 5/5 pass, branch 2-4/5 fail. Bisecting the change itself: EITHER half alone reproduces it (cards-only 5/5 fail, rail-only 2/5). Instrumentation chain, each hypothesis refuted in turn (rail state at deferred mount is right_open=False/left_open=True every run; the press handler NEVER fires in failing runs; every `pilot.click` returns False for 4s while the row reports `display=True disabled=False` at `y=33` — but passing runs show the row hit-tested at `y=30`). Net: any post-paint mount into the Console's nested fr-width Horizontal leaves the compositor hit-map and widget regions in **persistent multi-second disagreement** (~3-row offset) in ~40% of loaded runs — a framework-interaction failure none of the three simpler screens (VerticalScroll stack / ContentSwitcher / plain Container) triggered. Shipping a probabilistic click-eating window on the app's primary screen is the wrong trade for the measured ~0.9s.

**Live target confirmed on current dev:** Console switch 1.35–1.38s (2.7× the ~0.5s median) — the prize is real when the blocker falls.

**Paths forward:** (a) minimal Textual repro of the stale-region/hit-map divergence (upstream or workaround at the framework level); (b) restructure the reveal path first — give workspace-row activation a stable, index-free identity and re-run the A/B; (c) attack the 1.35s from the other side (profile what Console's compose spends beyond widget count). All code reverted; only this record and the tests' design (in this note) survive.
<!-- SECTION:NOTES:END -->

## Investigation Notes (2026-08-07, round 2 — blocker SOLVED, deferral measured ineffective, CLOSED as wrong-lever)

<!-- SECTION:NOTES:BEGIN -->
**The round-1 blocker is fully named.** The "compositor hit-map divergence" was `pilot.click(selector)` mechanics: it computes click coordinates from the target's `.region`, which is ZERO for a freshly (re)mounted widget until first layout — and the workspace rows are rebuilt by every Console sync pass, so synthetic clicks race rebuilds and land at (0,0) or stale coordinates (logged: pilot hit-tested `Static#None` at (0,0) while the row lived at y=6). A ~90-line standalone repro (rebuild rows → click) reproduces it in pure Textual; real users are unaffected (driver input resolves against the rendered cell map). Fixed for this suite by making `_click_console_workspace_conversation_for_session` drive `Button.press()` — the identical Pressed→handler chain without coordinate derivation. 8/8 stable where 2-5/5 failed.

**The deferral itself was then completed behavior-preservingly** (compose-time rail INSTANCE with pre-built live-work card; descendant-visibility completion scoped to the mounted rail; two fresh-state-sampling regressions found and fixed via the internals tests) and passed the 31-file surface (only the 6 known pre-existing parity failures + 1 solo-passing batch flake). **And then the measurement killed it**: interleaved A/B push probes, dev vs branch, same machine — first paint 2.50–2.54s vs 2.27–2.44s. Deferring ~108 hidden widgets bought 4–8%. Console is NOT widget-mount-bound (the Schedules lesson, one level deeper). Reverted rather than shipped: structural complexity in the app's most complex screen is not worth 5%.

**Where Console's ~2.5s actually goes (cProfile, one push):** `_sync_console_control_bar` — 11 calls × 102ms = 1.12s; `_build_console_inspector_state` — rebuilt 28× = 0.75s; `Workspace_DB` — **1,352 fresh private-SQLite connections opened during one push** = 0.64s; CSS apply 1.69s across ~830 composes. The real levers are filed as task-3010 (coalesce the mount-window sync storm) and task-3011 (Workspace_DB connection reuse). This task stays open only as the umbrella target (Console 1.35s live → ≤2× median) and should be re-measured after 3010/3011.
<!-- SECTION:NOTES:END -->

## Re-measurement attempt (2026-08-21) — partial, and NOT sufficient to close

Round 2 said this task "should be re-measured after 3010/3011". Both are now
**Done**, so I re-measured — and then found my own measurement cannot close
the AC. Recording both halves.

**What I measured.** `Tests/UI/app_factory._build_test_app`, 235x52, timing
app construction through first `pilot.pause()`, 5 runs each, interleaved on
one machine:

    chat    0.221, 0.188, 0.250, 0.242, 0.201   median 0.221s
    notes   0.218, 0.218, 0.181, 0.179, 0.225   median 0.218s

Console is **no longer distinguishable** from a structurally simpler screen
(1.4% apart, well inside the spread). Round 2 measured Console at 2.27-2.54s
and live switch at 1.35-1.38s against a ~0.5s median, i.e. a 2.7x outlier.

**Why this does NOT close the AC.** `_build_test_app` fakes every real I/O
seam, including the DB. Task-3011 -- one of the two levers whose landing
prompted this re-measurement -- was specifically about `Workspace_DB` opening
1,352 real SQLite connections per push. A harness that fakes the DB cannot
observe the cost 3011 removed, so it cannot confirm the live improvement; it
can only show that no *other* Console-specific outlier remains at
mount time. The AC asks for a live measurement ("Console switch latency
improves measurably live"), and this is not one.

**Recommended next step:** a live interleaved A/B push probe of the kind
round 2 used (real DB, real config, same machine, dev vs dev) to confirm the
1.35s live switch has fallen to within 2x the ~0.5s median. If it has, close
this task as achieved-by-3010/3011 rather than by widget deferral -- round 2
already measured the deferral itself at 4-8% and reverted it as the wrong
lever, so **do not re-attempt the deferral**; that ground is covered twice.
