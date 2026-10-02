---
id: TASK-33561
title: Stop Persona Visual from republishing at idle
status: Done
created_date: 2026-09-29 20:30
labels:
- performance
- persona-visual
- perf-audit-2026-09
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while measuring PERF-08 on 2026-09-29. On a scratch profile with Console open and no user activity, publish_persona_visual (Persona_Visual/publication.py, via persona_visual_participants.guarded) is the largest remaining source of idle open() calls: about 650 per second, once PERF-06 and PERF-08 part 1 remove the config and admission churn. Nothing changes at idle, so the publication should run only when its inputs change. Find what schedules the idle publishes and make them change-driven.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With Console open and idle for 10 s on the boot/idle probe, publish_persona_visual runs only when its inputs change (zero publishes after settle)
- [x] #2 A persona visual change still publishes promptly (existing publication tests stay green, plus one test that pins no-op idle ticks)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Not a bug: the premise was a measurement artifact.** Investigated by a subagent on 2026-09-29 and verified against its evidence.

**What actually happens.** `publish_persona_visual` runs **once per profile**: it is the first-boot install of the bundled `pixel-migu` Buddy. The call chain is:
1. `app._post_mount_setup`
2. `_schedule_deferred_startup_work`
3. the staggered boot worker `actor_pack_recovery` (`Utils/boot_worker_policy.py`, the first STAGGERED entry)
4. `app_service_wiring.ensure_actor_pack_recovery`
5. `Persona_Buddy/library.ensure_builtin`
6. `publish_review`
7. `publish_persona_visual`

`ensure_builtin` returns early once the builtin record exists. No timer, watcher or poll republishes it; the other callers are user saves. The PERF-08 probe used a fresh profile on every run and opened its idle window at `_ui_ready`, 0.1-0.7 s before this worker ran, so one install (58,781 opens on dev, 7,487 with PERF-08) was read as an idle rate.

**Evidence** (subagent probe, macOS scratch profiles)
- On dev: one publish in the first window, then 0 and 0.
- Second boot on the same profile: 0 publishes.
- With the idle window opened after the boot fleet drains: 0 publishes; dev at 3,108 opens/s, PERF-08 part 1 at 746.

**AC #1** already holds on dev once the probe settles.

**AC #2** is pinned by `Tests/Persona_Buddy/test_buddy_library.py::test_installed_builtin_is_never_republished` (subagent commit f2efa0e264, cherry-picked). It installs the builtin, then makes `publish_persona_visual` raise and calls `ensure_builtin()` repeatedly, plain and legacy-retired. Negative control: with `_source_record` forced to None it fails "installed builtin Buddy was republished".

**Regression.** Across 6 Persona Buddy/Visual/recovery test groups, the branch's 42 failures all fail identically on origin/dev (the TASK-33370 family).

**Not filed.** The one-shot install is costly on dev (64 assets, about 918 opens each), but it is a first-boot cost and PERF-08 already cuts it about 8x.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
