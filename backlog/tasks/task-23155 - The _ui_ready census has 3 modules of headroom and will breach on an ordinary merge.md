---
id: TASK-23155
title: The _ui_ready census has 3 modules of headroom and will breach on an ordinary merge
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - performance
  - startup
priority: high
dependencies:
  - task-23029
  - task-23112
---

## Description

The `_ui_ready` module census measures **967 against its 970 ratchet — headroom 3**, up from the
963 that ADR-097 recorded at `b5eaa9cf64` a few commits earlier. At this repo's merge rate that
breaches within a day or two, and under ADR-097 the limit does not rise: the modules have to defer
or shed.

This is the same shape as TASK-23112, which was filed for the import-weight ratchet and repaid it
666 → 646. That one is worth reading first — it establishes that the import-parent tracer's
attribution is an **upper bound** (it records only the first importer), so two of four edges its
filing named bought nothing, and the real deferrals were found by re-measuring after each step.

Filing this before it goes red is the point: a ratchet that only gets attention when it breaks
blocks whoever happens to merge next, which is rarely the person who spent the headroom.

## Acceptance Criteria

- [ ] The `_ui_ready` census passes with meaningful headroom and `MAX_UI_READY_MODULES` unchanged
  (no exception-ledger row)
- [ ] The modules that consumed the headroom since `b5eaa9cf64` are named, each with the commit that
  introduced its edge, and each deferral's yield is **measured** rather than inferred from the tracer
- [ ] Every deferred import is proven to still resolve on its real use path by a subprocess-isolated
  guard, mutation-tested (re-adding the eager import turns it red)
- [ ] ADR-097's tightening convention is assessed against the final number and either applied or
  explicitly declined with the arithmetic shown

## Evidence

Headroom line from a green run on `fix/task-23112`:
`ui-ready-census: 967/970 modules (headroom 3); snapshot drift +5/-1`. TASK-23112 verified against a
pristine base worktree that its own change does not move this census (968 both arms at the time),
so the consumption is dev's, not that task's.
