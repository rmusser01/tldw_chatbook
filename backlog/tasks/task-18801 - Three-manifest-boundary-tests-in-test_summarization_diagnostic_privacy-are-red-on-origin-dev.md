---
id: TASK-18801
title: >-
  Three manifest-boundary tests in test_summarization_diagnostic_privacy are red
  on origin/dev
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-08-18 23:52'
labels:
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py has three failing tests on a clean origin/dev checkout at 7d87686a3, unrelated to any feature branch:

  test_manifest_boundary_changes_only_summarization_owner_diagnostics
  test_manifest_boundary_rejects_owned_digest_schema_changes
  test_manifest_boundary_rejects_unreconciled_owned_digest

The first asserts that a normalized projection of the checked-in diagnostic inventory hashes to the SHA recorded in the review fixture, and it does not:

  AssertionError: checked inventory changed outside the two summarization owners
  assert 'b0187e7972ac...85edefc4ed7ee' == '8b0633e98e95...05bfd61aac9de'

The other two fail as a consequence -- both call the first test as their control before applying their mutant.

Reproduced on a dedicated detached worktree of origin/dev with nothing else applied: 3 failed, 254 passed. The same three, and only those three, appear when running the file from a feature branch, so branches touching tldw_chatbook/LLM_Calls/ currently inherit a red gate they did not cause. Either the checked-in inventory JSON or the fixture SHA needs regenerating and reconciling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The three manifest-boundary tests pass on a clean origin/dev checkout
- [x] #2 Whichever of the checked-in inventory or the recorded fixture SHA is stale is identified and the fix explains which drifted and why
- [x] #3 Tests/LLM_Calls/test_summarization_diagnostic_privacy.py runs green as a whole
<!-- AC:END -->

## Implementation Plan (added 2026-10-02, [rmusser01])

ADR required: no
ADR path: N/A
Reason: test-fixture re-pin only; no production code, schema, migration, or security-policy change (the inventory content itself is untouched — it already matches a fresh regeneration).

1. Verify the premise at the assigned base (origin/dev tip ecc0a531c8): run the file; confirm exactly the three named tests fail.
2. Determine which side drifted: compute the reconciliation test's own normalized SHA for (a) the checked-in inventory and (b) a fresh `build_inventory()`; compare each against the fixture's two recorded SHAs; diff checked vs generated non-owned content.
3. Review the inventory delta since the last fixture re-pin row-by-row (task-3750 obligation) and attribute every delta to landed dev commits; confirm no CI lane runs this file (explains persistence).
4. Re-pin the fixture's two `manifest_boundary` SHA fields to the verified current value, using the reconciliation test's own normalization.
5. Run the three tests and the whole file green; corroborate with the sibling inventory gate (Tests/Architecture/test_persistent_diagnostic_inventory.py).

## Implementation Notes (2026-10-02, [rmusser01])

**Classification: (b) live defect at the assigned base — fixed red to green by re-pinning the review ledger; the checked-in inventory needed no change.**

**Premise verified.** At origin/dev tip `ecc0a531c8`:
`python -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -q`
→ 3 failed (exactly the three named tests), 253 passed — same trio as filed, though the
hash values have moved on since the filing (drift is ongoing, see below).

**Which side drifted (AC #2).** The recorded fixture SHA is the stale side; the
checked-in inventory is current:

- Computing the reconciliation test's own normalized projection at HEAD:
  checked inventory and a fresh `diagnostic_inventory.build_inventory()` are
  **byte-identical after normalization** — both hash to
  `d520941d73ab00d874e4333d4c916fae4fca9fb2de36d8c2a758a67ecf295039`
  (the pytest failure output printed the same value as the actual side, confirming the
  computation). Zero non-owned owner-row diffs, zero owner-path set diffs, zero
  top-level section diffs between checked and generated.
- The fixture's two `manifest_boundary` pins still read `e1c0d9a0…`, recorded by
  `7a3d946161` (TASK-32856, 2026-09-22, "fixture manifest re-pinned with the
  reconciliation test's own normalization"). Forensically, that pin never matched its
  own landed tree — the inventory at `7a3d946161` normalizes to `7f94d5a5…` — so the pin
  was born stale (evidently computed against an uncommitted intermediate state), and
  every later inventory regeneration inherited the mismatch.
- Since then, 100+ commits regenerated `Docs/security/production-diagnostic-inventory.json`
  (tier2 waves, theme waves, TASK-33011 app decomposition, console fixes,
  TASK-33005 model-config) — several say "re-pin", but they re-pin the inventory's own
  derived totals/`reviewed_exclusions`, never this ledger fixture. **No CI lane runs
  this file** (no reference in `.github/workflows/` or `scripts/`), which is how it
  stayed red for ~10 days unnoticed — the task description's "branches touching
  tldw_chatbook/LLM_Calls/ inherit a red gate" is the local-run path, not CI.

**Review obligation (task-3750 discipline) before re-pinning.** Row-by-row diff of the
inventory between `7a3d946161` and HEAD, non-owned rows only: 51 added, 16 removed,
55 count/digest-changed, plus sink-topology and path-privacy-candidate movement — all
attributable to landed dev work: TASK-33011's app.py decomposition (the new `app_*`
modules, `app.py` 400→119), tier2 dead-module deletions (`MediaWindow_v2` −69,
`Chatbooks_Window` −20, `CodeRepoCopyPasteWindow` −13, …), TASK-32948 theme picker,
omnivoice TTS, SSH remote-workspace tools, Dreams phase, TASK-33621/.33622 console
fixes, PERF-03 logging. Owned summarization entries: call_counts 229/269 = 242−13 and
281−12, exactly the ledger's deleted-site arithmetic. Nothing unexplained.

**Fix.** `Tests/fixtures/summarization_diagnostic_review.json`: both
`manifest_boundary` SHA fields re-pinned `e1c0d9a0…` → `d520941d…` (the value the
reconciliation test itself computes at HEAD). No production or inventory content
changed.

**Evidence (red → green):**

- Before: `…::test_manifest_boundary_changes_only_summarization_owner_diagnostics` →
  FAILED ("checked inventory changed outside the two summarization owners",
  `d520941d… != e1c0d9a0…`); the two mutant tests failed behind it (they call the
  primary as control).
- After: the three named tests + `…tracks_reconciled_checked_and_generated_baselines`
  → 4 passed; whole file `python -m pytest
  Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -q` → **256 passed,
  0 failed**. The file's own mutants (`rejects_unknown_top_level_sections`,
  `rejects_new_generated_origin_dev_drift`) still reject drift, so the guard retains
  teeth after the re-pin.

**Adjacent observation for the owner (not this task's scope, pre-existing at base).**
`Tests/Architecture/test_persistent_diagnostic_inventory.py` is itself red on base:
`test_reviewed_diagnostic_changes_are_metadata_only` and
`test_task_15743_final_rebase_diagnostics_are_metadata_only` fail, and whole-file runs
add ~15 errors at setup (they pass individually). Verified pre-existing by baseline
A/B (`git checkout HEAD -- Tests/fixtures/summarization_diagnostic_review.json`,
rerun, restore): the failures are identical with and without this task's re-pin, and
that file never reads this ledger. It needs its own owner.

- **ADR required: no.** Reason: one-line test-fixture hash re-pin; no production code,
  schema, migration, or security-policy change; follows the established re-pin remedy
  (TASK-16207's, `7a3d946161`'s, and TASK-19191's precedents for this artifact class).
- **Modified files:** `Tests/fixtures/summarization_diagnostic_review.json` (2 lines),
  this task file.
