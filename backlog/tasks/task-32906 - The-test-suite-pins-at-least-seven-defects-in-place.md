---
id: TASK-32906
title: The test suite pins at least seven defects in place
status: To Do
assignee: []
created_date: '2026-09-21 23:55'
labels:
  - tier2-review
  - review-guards
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The tier-2 review's verdict was that three CI-required guards are green without proving anything. Burning
down the first two streams showed that is the narrower half of the problem. **The test suite itself holds
several of these defects in place** -- not by omission, but by explicit assertion. A fixer who corrects the
code gets a red test and, reasonably, assumes they broke something.

This is why the defects survived contact with a green CI for as long as they did, and it is the single
highest-leverage thing in the whole review: every other stream is slower while these tests stand.

## The seven

Two were found by *fixing* the code and watching the wrong test go red:

1. **`test_does_not_flag_aware_now_isoformat`** -- existed to assert that `check_timestamp_writers.py`
   does **not** flag `datetime.now(timezone.utc).isoformat()`. ADR-173 names that exact expression as a
   shape to eliminate. The guard, its docstring and its test all agreed with each other and all disagreed
   with the ADR. Inverted in TASK-32897.
2. **`test_replacing_with_no_valid_chunks_still_clears_the_old_rows`** -- docstring: *"The DELETE is
   unconditional; only the INSERT is skipped when empty."* It asserts that a rechunk with zero valid
   replacement chunks **destroys the user's existing chunks**. Inverted in TASK-32893.

Five more were recorded by the review's per-finding `Pinning test` field and are not yet addressed:

3. **`test_confluence_make_request_gets_timeout_and_guard`** -- asserts the guard *is* used, but only on the
   branch that is reachable; the unreachable branch is what the finding is about. Green for the wrong
   reason. Owned by TASK-32894.
4. **`Tests/LLM_Management/test_mlx_lm.py`** -- **30 tests asserting the current behaviour of code that is
   dead.** A future fixer will "repair" unreachable code because these go red. Owned by TASK-32899.
5. **`Tests/CI/test_canvas_mermaid_asset_checker.py`** -- duplicated the comparator's own `EXPECTED_OUTPUTS`
   constants instead of importing them, so it structurally could not notice the comparator drifting.
   Fixed in TASK-32897 by replacing the duplicate with an import; the *pattern* is the finding.
6. **`test_legacy_state_exports_remain_serialization_compatible`** -- keeps dead exports importable, which
   is precisely what makes them look alive to a dead-code sweep.
7. **`test_slot_string_is_canonical_utc_iso`** (`Tests/Scheduling/test_schedule_compute.py:58`) -- asserts a
   canonical UTC ISO shape that does not match ADR-173's mandated `Z` form. Needs reconciling with the
   repaired `check_timestamp_writers.py`.

## The generalisable rule

**A green test is evidence about the test, not about the code, until someone reads what it asserts.**

Three distinct mechanisms produced these, and they are worth naming separately because they need different
remedies:
- **Assertion of the defect** (1, 2) -- the test says the wrong behaviour is correct. Invert it.
- **Green for the wrong reason** (3, 6, 7) -- the test passes without exercising the thing it names. Make it
  fail first, then make it pass.
- **Self-referential** (5) -- the test duplicates the constants of the thing under test, so both drift
  together. Import, never duplicate.

## Scope

This task is the **audit and the rule**, not the individual inversions -- those belong to the streams that
own each defect (noted per item above). Deliver a short written rule plus whatever mechanical check is
honest, and resist inventing a guard that cannot really detect this: "a test that asserts a defect" is not
mechanically decidable in general. A realistic check is narrow -- e.g. flag test modules that duplicate a
literal constant also defined in `scripts/` or `tldw_chatbook/` (mechanism 3), which is decidable and
covers the one case that already bit.

Source: tier-2 code review 2026-09-21. Items 1 and 2 discovered while implementing TASK-32897 and
TASK-32893; items 3-7 from the review's per-finding `Pinning test` fields in
`qa/tier2-code-review-2026-09-21/slices/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the seven is inverted, re-grounded, or recorded as deliberate with a reason
- [ ] #2 A written rule exists where contributors will meet it, covering the three mechanisms
- [ ] #3 Any check added is honest about what it can and cannot detect, with a negative control
- [ ] #4 No check is added for a mechanism it cannot actually decide
<!-- AC:END -->
