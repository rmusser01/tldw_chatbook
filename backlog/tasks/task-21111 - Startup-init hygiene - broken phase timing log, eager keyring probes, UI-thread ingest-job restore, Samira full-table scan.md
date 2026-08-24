---
id: TASK-21111
title: >-
  Startup-init hygiene - broken phase timing log, eager keyring probes, UI-thread ingest-job restore, Samira full-table scan
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - diagnostics
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21111).

Four small, verified startup defects: (a) the `__init__` parallel-task timing log measures
durations AFTER `future.result()` returns (app.py:5823-5825), so every task logs ~0 s and the
STARTUP TIMING SUMMARY cannot attribute the parallel phase - fix first, it makes all other
startup work measurable; (b) 2-3 `keyring.get_keyring()` backend discoveries run during
`__init__` (server credentials ~13 ms + Security.framework ctypes load; skills trust x2) for
features not in use; (c) `_restore_ingest_jobs` (app.py:2211-2239) does DB open + read +
reconcile writes synchronously on the UI thread in on_mount; (d) `ensure_builtin_samira` full-
scans `character_cards` parsing every `extensions` JSON per boot
(`Character_Chat/visual_identity.py:3044-3060`).

## Acceptance Criteria

- [x] The startup timing summary reports real per-task durations (measured around the task's own execution, not around `result()`)
- [x] **AMENDED** — no OS keyring call of any kind (backend discovery *or* credential read) happens during app import, construction, or mount; each of the four sites resolves on first real use
- [x] Ingest-job restore runs off the UI thread; behavior unchanged
- [x] The Samira preflight uses a targeted query (json_extract + LIMIT 1) instead of a full scan with per-row JSON parsing

### Why AC 2 was amended

The filing named three keyring sites (server credentials ~13 ms, skills trust
x2) and prescribed lazy properties for them. Measured on the merge base
(`f49956038`, macOS, isolated profile), that fix would have banked **0 ms**:

| site | order at boot | cost |
|---|---|---|
| `Video_Generation/config._keyring_get` via `VideoStore.enforce_retention` (**not in the filing**) | 1st | **18.2 ms** -- 11.3 ms backend discovery + a real `keyring.get_password` Keychain query |
| `build_default_server_credential_store` | 2nd | 0.33 ms |
| skill-trust marker store | 3rd | 0.41 ms |
| skill-trust key cache | 4th | 0.04 ms |

`keyring.get_keyring()` memoizes the discovered backend, so **whichever site
runs first pays for all of them**. Fixing the three named sites just promotes
the unnamed one to first place. The AC is therefore restated as a property of
boot ("zero keyring calls"), which is both honest and testable, and all four
sites are fixed.

A second correction of the same kind was found *inside* the fix: deferring the
whole skills stack to a lazy app property moved 16.45 ms from `__init__` into
Chat-screen mount, because `ChatScreen._ensure_console_agent_bridge` reads
`skills_scope_service` at mount. Only deferring the trust service itself,
behind a factory `LocalSkillsService` calls on the first trust decision,
actually removes it.

## Implementation Plan

1. Build the measurement harness FIRST (isolated-profile boot probe, mounted-app
   Pilot probe, keyring call-site tracer, per-N micro-benchmarks) and take a
   baseline against the merge base `f49956038` before writing any fix.
2. Fix (a) -- the timing wrapper -- first, because it makes the rest measurable.
3. Attribute every keyring touch by stack, then fix all sites the trace names,
   not just the ones the filing names.
4. Move the ingest restore to a thread worker, keeping the registry's
   UI-thread-only contract and closing the window the deferral opens.
5. Replace the Samira card scan with a targeted query, verified against the
   retained scan as an oracle on malformed/NaN/non-object rows.
6. A/B every claim against a pristine copy of `f49956038`; mutate every new
   test against a deliberately broken implementation before trusting it.

## Implementation Notes

Four independent startup defects, each measured before and after against a
pristine copy of the merge base `f49956038` sharing the same warm profile.

**(a) Parallel-init timing.** `_timed_init_task` wraps each phase-3
initializer and stamps its duration on the worker thread; the `as_completed`
loop reads the recorded value instead of timing an already-completed
`result()`. The summary now nests the four sub-phases under `parallel_init`.
Before: every task logged `0.000s` while the phase really took 0.499 s
(fresh profile). After: logged values match independently-measured truth to
the logged precision (`notes_service` 0.016 vs 0.0156, `media_db` 0.006 vs
0.0061, `prompts_service` 0.005 vs 0.005, `providers_models` 0.000 vs 0.0001).

**(b) Keyring at boot.** Four sites, not three -- see the amended AC above for
why fixing only the named three banks nothing. `VideoStore` now reads a
secrets-free `VideoStorePolicy` (the only three settings it ever touches);
`server_credential_store` became a lazy property with
`RuntimeServerContextProvider` taking a `credential_store_factory`; the skill
trust service is built by a factory `LocalSkillsService` calls on the first
trust decision. Measured on a mounted app: **4 keyring calls / 18.2 ms -> 0
calls**, and `TldwCli` construction **77.0 ms -> 60.9 ms** (median of 8 warm
runs per arm; `basic_init` 24.1 -> 10.0 ms). Removing the Keychain
`get_password` also removes a call that can block or raise a consent dialog on
a locked keychain.

**(c) Ingest-job restore.** `_restore_ingest_jobs` now starts a thread worker
(`exit_on_error=False`); the store open, read, plan and reconcile writes run
off the UI thread and only the registry seeding marshals back, preserving the
registry's documented UI-thread-only contract. UI-thread block at 500
persisted jobs (150 interrupted): **26.7 / 8.1 / 7.7 ms -> 0.076 / 0.067 /
0.074 ms**, same 500 restored jobs and same `next_id`. The deferral opens a
narrow window in which a job could be submitted before the seeding lands, so
the registry gained `merge_restored`, which is byte-identical to `restore` on
an empty registry and otherwise keeps the live job, drops the colliding
persisted id, and takes `max` of the two `next_id`s.

**(d) Samira preflight.** `_find_builtin_samira_card` asks SQLite
(`json_valid`-guarded `json_extract` + `ORDER BY id LIMIT 1`) instead of
reading every card and parsing its `extensions`. **2.6-2.9x faster**
(100 cards 0.19 -> 0.07 ms; 2,000 cards 4.1 -> 1.5 ms), byte-identical results
on NULL / empty / malformed / array / scalar / `NaN` / numeric-value
`extensions`. The old scan is retained as `_find_builtin_samira_card_by_scan`
for JSON1-less SQLite builds; the fallback is gated on `no such function`
rather than being a catch-all, because a catch-all would silently reinstate
the full scan on every boot of a profile with one corrupt row.

Modified: `tldw_chatbook/app.py`, `tldw_chatbook/Character_Chat/visual_identity.py`,
`tldw_chatbook/Library/library_ingest_jobs.py`,
`tldw_chatbook/Video_Generation/config.py`,
`tldw_chatbook/Video_Generation/video_store.py`,
`tldw_chatbook/runtime_policy/server_context.py`,
`tldw_chatbook/Skills_Interop/local_skills_service.py`.
Added: `Tests/App/test_startup_init_hygiene.py`,
`Tests/Character_Chat/test_samira_preflight_query.py`, plus two
`merge_restored` cases in `Tests/Library/test_library_ingest_jobs_restore.py`.
