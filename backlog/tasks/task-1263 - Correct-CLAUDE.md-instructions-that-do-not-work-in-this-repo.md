---
id: TASK-1263
title: Correct CLAUDE.md instructions that do not work in this repo
status: Done
assignee: []
created_date: '2026-07-28 19:13'
labels:
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two commands CLAUDE.md gives as canonical fail when run. The setup section says 'pip install -e ".[dev]"', but the project venv is uv-managed and ships no pip module, so both 'pip' and 'python3 -m pip' fail; the working form is 'VIRTUAL_ENV=.venv uv pip install'. Separately, the Backlog.md section documents '--ac "Must work,Must be tested"' as producing two criteria; CLI v1.44.0 does not split on commas and writes a single run-on criterion, which cannot be checked off independently and so breaks the Definition of Done's per-item completion. Both cost time in the 2026-07-27 aiohttp startup investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The setup section documents the uv-based install command that actually works in this venv
- [x] #2 The Backlog.md --ac example shows a form that produces separate criteria (repeated --ac flags, verified working on CLI v1.44.0)
- [x] #3 No other CLAUDE.md guidance is altered
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two documented commands failed when run; both are corrected in place rather than deleted, since the canonical form is still right for a normal pip venv.

**Setup.** `pip install -e ".[dev]"` fails against the checked-out `.venv`, which is uv-managed (`.venv/pyvenv.cfg`: `uv = 0.11.7`) and ships no `pip` module — both `pip` and `python3 -m pip` error. Added the working `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` form plus a note that dev deps are often absent entirely, so a missing `pytest` is not evidence the suite is broken. Also added `image_generation` to the example extras list (task-1262).

**Backlog CLI.** The `--ac "Must work,Must be tested"` example does not produce two criteria on CLI v1.44.0 — it writes one run-on criterion that cannot be ticked off individually, which quietly breaks the DoD's per-item completion. Verified the working form (repeat the flag) while filing task-1262: four `--ac` flags produced four separate criteria.

Both traps are already recorded with their incidents in `lessons-backlog-hygiene.md`; this fixes the source that produced them.

**Modified files.** `CLAUDE.md` (two passages; no other guidance touched).
<!-- SECTION:NOTES:END -->
