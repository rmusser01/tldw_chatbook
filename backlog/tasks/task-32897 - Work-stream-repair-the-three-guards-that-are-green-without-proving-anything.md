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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on branch `fix/tier2-guards` as `f3bb728761` (8 files, +730/-74). Not pushed.

All three were real. **Guard 3 was understated, not overstated**: a naive tamper of either self-certifying
file was already caught as manifest drift, so they were not simply unprotected. The real hole is a
laundering path -- `vendor_canvas_mermaid.py --output-dir` defaults to `STATIC`, so re-running the
documented `reproducible_command` regenerates a self-consistent manifest **from the tampered bytes**.
Reproduced end to end: the pre-repair checker returned green on an 85 KB browser runtime with
`/*backdoor*/` appended. Fixed by pinning digests in `VENDORED_OUTPUTS` inside the checker -- deliberately
in `scripts/`, which the vendor script never writes to, so the pin is outside the laundering path.

Guard 1's predicate is now the emitted format, not naivety: new kinds `offset_now_iso` and `strftime_iso`.
Census **0 -> 117 occurrences across 109 rows** (99 + 18). The AST count is 99, not the 101 quoted in the
task description above -- that figure was grep-derived and counted lines, not call sites. Notably, the
suite's own `test_does_not_flag_aware_now_isoformat` **asserted the whitelisting**; it has been inverted.

Guard 2: only `Try.body` guards, and the ascent continues past a non-guarding `Try` so an outer one still
counts (also matches `ast.TryStar`). **269 -> 319 sites across 157 functions**, matching the pre-repair
measurement exactly.

Deliberately out of scope: `isoformat().replace("+00:00","Z")` at microsecond precision is treated as
conforming though it is variable-width; widening costs 40+ census rows and belongs with the writer
migration (TASK-32914). Recorded in the guard's docstring.

Verified independently of the implementing agent: preflight GREEN and now reporting real numbers; size
ratchet unchanged at exactly 5 failed / 58 passed; the timestamp ratchet bites when a writer is injected
into a scanned module (99 -> 100, exit 1); the Canvas pin rejects an appended `/*backdoor*/`.

One process note: the first negative-control attempt was invalid -- the probe writer was injected into
`Utils/timestamps.py`, which the guard exempts by design as ADR-173's one sanctioned producer. A negative
control aimed at an exempt path proves nothing. Re-run against a scanned module, it behaved correctly.

Follow-up filed as TASK-32897.1: `canvas_shell.js`/`.css`/`.html` (51 KB + 10 KB + 8 KB, served by
`gateway.py:126-130`) are under no integrity check at all, and `canvas_shell.js` is the code that reads the
gateway's bootstrap **token** from the URL fragment. Larger exposure than the hole just closed.
<!-- SECTION:NOTES:END -->
