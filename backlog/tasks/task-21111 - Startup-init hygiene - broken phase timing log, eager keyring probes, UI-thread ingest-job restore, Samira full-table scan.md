---
id: TASK-21111
title: >-
  Startup-init hygiene - broken phase timing log, eager keyring probes, UI-thread ingest-job restore, Samira full-table scan
status: To Do
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

- [ ] The startup timing summary reports real per-task durations (measured around result(), not after it)
- [ ] Keyring backend discovery happens on first server-mode/skills-trust use, not at boot
- [ ] Ingest-job restore runs off the UI thread; behavior unchanged
- [ ] The Samira preflight uses a targeted query (json_extract + LIMIT 1) or a cached id instead of a full scan with per-row JSON parsing
