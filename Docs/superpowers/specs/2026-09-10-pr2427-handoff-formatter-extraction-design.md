# PR 2427: pure Handoff formatter extraction

Status: approach approved; written-spec review pending.
Task: TASK-31932, step 159. Baseline: `02597164ff` on dev `3afa68f1b9`.

## Scope and decision

Move only the existing Handoff summary formatter's unchanged body from
`LibraryScreen._workspace_handoff_summary_label` into the existing
`UI/Library_Modules/screen_helpers.py` support layer. Keep the Screen method
with its current signature as a thin delegate. All four production call sites
and existing test calls remain unchanged, including calls with `None` as self.
The formatter has no self references, performs no I/O, and creates no widgets.

This is the first small size reduction, not a resolution of the size gate:
LibraryScreen currently has 32,724 lines against a 31,689 ceiling. Expect about
60 net lines removed; measure the exact result. Keep all existing caps unchanged
while the file remains above its ceiling. No comment compression or code golf.

Alternatives considered: keeping the function in the Screen leaves this excess
unchanged; a new dedicated module/controller adds a boundary for one pure helper.
The existing support module already owns related Handoff text functions and is
the smallest established home. No other formatter or ownership move is included.

ADR required: no new ADR.
ADR path: N/A.
Reason: mechanical refactor within the existing support-layer boundary, governed
by `DESIGN.md` section 7 and
`Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md`.
No state owner, service contract, storage, security or UX policy changes.

## Compatibility and dependencies

- Preserve every branch, string, comment and docstring in the moved body, apart
  from indentation. Preserve the existing input state and string return type.
- Import the existing handoff-prefix constant and eligibility formatter/constant
  from their canonical owners. Do not duplicate their values or policy logic.
- Keep the Screen's existing constant re-export and eligibility imports used by
  other Screen methods. Do not move those other callers or change binding policy.
- Keep the helper independent of LibraryScreen; use a type-only import for
  `LibraryWorkspaceDepthState` from its defining display-state module where safe.
- Census free-name lookups and monkeypatch targets before the move. If a relevant
  Screen-global patch seam exists, stop and revise this design rather than bypass
  it. The initial named-call census found none.
- No new exception handling, validation, normalization, caching or mutation.
  Prefix stripping, eligible/blocked counts, mixed reasons/types, recovery text,
  singular/plural wording and missing-prefix cases retain their exact output.
- No widget construction, refresh, eligibility, focus, scroll or lifecycle changes.

## Verification and acceptance

1. Add a narrow support-layer location/delegation guard before moving code; see
   it fail on the baseline. Preserve all existing behavioral assertions.
2. Compare the moved body with the baseline byte-for-byte after dedenting, and
   compare ASTs. Verify the wrapper forwards the identical state object once.
3. Run the complete affected files: `test_library_crit9_rail.py`,
   `test_library_rail_workspace_action_sync.py`, `test_library_rail.py`,
   `test_post_release_workspaces_library_depth.py`, and
   `test_library_entry_compose_once.py`. Retain the existing mounted identity,
   focus and scroll coverage; use native resource observation for this cohort.
4. Run the complete support-layer/import-surface guard and existing Screen and
   Library-module size/private-owner guards. Report still-failing ceilings
   honestly; do not turn the expected remaining failures into skips or xfails.
5. Run scoped lint, whitespace and all derived preflight checks, and obtain an
   independent implementation review. Preserve unrelated worktree changes.

No full-repository sweep, SQLite fixture repair, additional owner extraction,
new dependency or merge bypass is authorized by this design. The 493 retained
Console SQLite handles and other size failures remain separately tracked.

## Workflow checklist

- [x] Inspect current function, callers, target module and governing doctrine.
- [x] Compare alternatives and obtain approval of the bounded approach.
- [x] Record this written design; visual companion is not applicable.
- [ ] Complete independent spec review.
- [ ] Obtain user review of the written spec before implementation planning.
- [ ] Write and execute the bounded implementation plan after that approval.
