---
id: TASK-21248
title: >-
  The tldw module-count budget cannot catch a single-deferral regression -
  retire it or govern it
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - startup
  - test-health
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; finding from the
TASK-21108 review.

The TASK-21108 branch (`origin/fix/task-21108-wave5`, **not merged into dev** at close-out)
adds `MAX_TLDW_MODULE_COUNT = 660` in `Tests/Performance/test_app_import_weight.py` as a drift
signal over this repo's own modules. Two measured problems with it as a guard:

1. **Headroom.** It leaves roughly 30 modules of room against a **measured +61 in one
   four-day window** (604 → 665, 2026-08-12 to 2026-08-16). At the observed accretion rate the
   budget is consumed in days, and the routine response to a red budget test is to bump the
   number.
2. **Sensitivity.** It does not catch a single-deferral regression. Reverting only the panel
   deferral gives **649**; reverting only the notes deferral gives **645**. Both are under
   budget, so either win can be undone silently.

That second point is not hypothetical: TASK-21200 was filed this same week because
TASK-21103's import guard went red only once CI started enforcing it — another session's
Actor Packs feature had already put PIL and `Persona_Visual` back on the boot path, undoing a
−80-module / −1.28 s win. A shipped guard only protects while something runs it, and an
aggregate budget with 30 modules of slack does not run against any individual win.

The named-module closure guards (the shape TASK-21103 and TASK-21104 shipped, and the one
TASK-21200 restored) do catch a single deferral, and they fail with the offending import chain
rather than a number.

## Acceptance Criteria

- [ ] A revert of any single import deferral shipped by this burn-down fails a test, and the failure names the chain that came back
- [ ] The aggregate module-count budget is removed in favour of those named guards, or it stays with an explicit review process for raising it — a docstring bump is not that process
- [ ] No guard in `Tests/Performance/test_app_import_weight.py` can be satisfied by editing a constant without a recorded decision
