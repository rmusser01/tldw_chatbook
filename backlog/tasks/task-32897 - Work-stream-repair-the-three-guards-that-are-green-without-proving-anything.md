---
id: TASK-32897
title: "Work stream: repair the three guards that are green without proving anything"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-guards
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**This is the review's headline finding and it outranks every individual defect in the report.** Three
of this repo's wired, CI-required guards pass on `origin/dev d0face3ebe` while failing to check what they
claim. `./scripts/preflight.sh` is fully green at that SHA.

1. `check_timestamp_writers.py` -- its docstring states `datetime.now(timezone.utc).isoformat()`
   "does not match and is fine". **ADR-173 names that exact expression as one of the drifting shapes it
   exists to eliminate** and mandates a `Z` suffix. 101 live writers emit `+00:00`; the census file is
   empty and the check prints `0 site(s) ... OK`. Because `'+'` (0x2B) sorts before `'Z'` (0x5A), a row
   written at the same instant by a `+00:00` writer is **silently excluded** by `WHERE ts >= '<...>Z'` --
   which is exactly the failure TASK-32803.3 names, while TASK-32803.5 is marked Done.
2. `check_textual_worker_contract.py` W002 -- the ancestor walk accepts **any** `ast.Try` ancestor as
   protection, but a `query_one` in a `finally:`/`except:` body is a *child* of the `Try` and is not
   covered by that `Try`'s handlers. Measured like-for-like: 269 sites reported (matching preflight's
   printed number) vs **319** with the predicate corrected -- **50 hidden across 21 functions**.
3. `check_canvas_mermaid_assets.py` -- 2 of its 6 "reproduced" outputs are copied out of the directory
   it then compares against, so they self-certify.

Paired data point: the duplication census has not moved. Re-run with the committed tier-1 script,
`dup_by_name` went 1943 -> **1944** over the 25 commits since the review baseline; verbatim and shape are
dead flat. Consolidation tasks are closing without the census being consulted.

The newly-visible sites are **re-pinned as baseline, not fixed here** -- fixing them is the other
streams' work. Keep this PR to the guards so the re-pin stays reviewable.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `check_timestamp_writers.py` enforces the ADR-173 emitted format, not naivety, and its census is re-pinned with the live writers
- [ ] #2 W002 treats only `Try.body` as guarded and its census is re-pinned at the corrected count
- [ ] #3 No Canvas Mermaid output is compared against a copy of itself
- [ ] #4 Each repaired guard has a negative control: a deliberately bad sample that makes it fail
- [ ] #5 `./scripts/preflight.sh` is green after the re-pin, with the newly-visible sites recorded as baseline
<!-- AC:END -->
