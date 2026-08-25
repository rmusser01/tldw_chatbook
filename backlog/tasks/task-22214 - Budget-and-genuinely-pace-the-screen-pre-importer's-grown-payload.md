---
id: TASK-22214
title: >-
  Budget and genuinely pace the screen pre-importer's grown payload
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - startup
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22214).

The whole-registry pre-importer payload grew +99 modules / +74,524 LOC since the pin
(568k -> 552k LOC compiled on a daemon thread; library_screen route 92,758 -> 135,933 LOC,
settings 43,762 -> 72,963). Pacing (`app.py:12725-12731`) is
`min(previous_cost * SCREEN_PREIMPORT_YIELD_RATIO, SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS)`
with the cap at 0.10 s (`app.py:795-796`) — for a 1.2 s route compile that is ~92% GIL
duty exactly while the user first touches the UI. `_usable_cpu_count` falls back to
`os.cpu_count()` on macOS (no sched_getaffinity), so laptops take the unthrottled tier.
Honest history: TASK-21113 shipped as a wash because a sleep cannot subdivide one
`import_module` — the lever here is payload size, route order, and the gap CAP, not finer
sleeping.

## Acceptance Criteria

- [ ] Pre-importer GIL duty cycle over the first 5 s after mount is measured (tip) and reduced, or the payload is trimmed per-route with the top growers listed and justified
- [ ] The gap cap / yield ratio is retuned with measurements at both a high-core and a low-core tier, honestly reporting overlap if results wash (the 21113 precedent)
- [ ] First-navigation latency to Library and Settings measured before/after (the pre-import exists to protect it — do not trade it away silently)
- [ ] A payload budget (module count or LOC per route) is pinned so the next +30k LOC lands in review, not in users' laps
