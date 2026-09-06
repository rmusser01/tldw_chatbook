---
id: TASK-31806
title: 13 'No handler found for worker' warnings on every fresh boot pollute the Logs Warnings+ filter
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 14:47'
labels:
  - bug
  - logs
  - boot
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Every fresh boot emits 13 'No handler found for worker ...' WARNING lines, drowning the Warnings+ filter's signal. Either register handlers, downgrade to debug, or fix the worker wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh boot emits zero spurious 'No handler found for worker' warnings.
<!-- AC:END -->

## Implementation Plan

1. Reproduce: temporarily instrument the `on_worker_state_changed` warning site, live-boot an isolated scratch profile, and capture the unhandled (name, group) pairs.
2. Analyse register-vs-downgrade: confirm each is a fire-and-forget boot/startup one-shot whose failures the diagnostics hook already persists (the ERROR branch runs before registry delegation). Register/acknowledge them, matching the task-2726 precedent.
3. Fix: derive the boot-fleet groups from `BOOT_WORKER_POLICY` and add the research startup groups to `MiscWorkerHandler.HANDLED_GROUPS`.
4. Verify: parametrized "does not warn unhandled" tests over the live-captured groups + a derivation guard; live re-boot shows zero warnings.

## Implementation Notes

Root cause: the app-wide `on_worker_state_changed` hook warns "No handler found for worker" for any worker whose group is not in `MiscWorkerHandler.HANDLED_GROUPS`. Seven fire-and-forget startup one-shots were unregistered, and each warns once per state transition (PENDING/RUNNING/SUCCESS), so a fresh boot emitted a burst of these (13 in the reporter's profile; a live capture on dip 5894f4755e showed 21 = 7 groups x 3 states, the count varying with which data-gated workers run). Live-captured groups: `ingest_restore`, `actor_pack_recovery`, `actor_pack_staging_sweep`, `chachanotes-fts-backfill`, `research_source_association_startup`, `research_paste_staging_startup`, `research-quick-notes-startup-reconciliation`.

These are all self-contained boot/startup sweeps that consume their own results and whose failures are ALREADY persisted by the diagnostics hook before registry delegation — so acknowledging them (registering, not silencing a real result) is correct, exactly as task-2726 did for `screen-navigation`/`scheduling`/`subscriptions-fts-backfill`.

Chosen fix is register-and-harden: the boot-fleet groups are now DERIVED from `BOOT_WORKER_POLICY` (`_BOOT_WORKER_GROUPS`) rather than hand-listed, so a future `BootWorkerSpec` can never silently reintroduce the warning (this closes a maintenance gap — four boot-policy groups had gone unregistered while two others were already present). The research startup sweeps spawned directly (not via the policy) are listed explicitly in `_RESEARCH_STARTUP_GROUPS`, including the data-gated `research_source_held_startup` sibling. Genuinely unknown workers still warn (guard test retained).

Live verification: re-booted the isolated profile on the fixed build and confirmed ZERO "No handler found" warnings across the full boot + staggered-worker settle window.

Modified files:
- `tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py`
- `Tests/App/test_worker_failure_event.py` (parametrized no-warn tests + boot-policy derivation guard)
