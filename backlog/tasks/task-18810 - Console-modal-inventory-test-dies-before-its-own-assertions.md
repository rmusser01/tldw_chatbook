---
id: TASK-18810
title: Console modal inventory test dies before its own assertions
status: Done
assignee: ['@Robert']
created_date: '2026-08-18'
labels: [console, testing, modals]
dependencies: []
priority: medium
---

## Description (the why)

`Tests/UI/test_console_modal_dismissal.py::test_console_modal_inventory_matches_runtime_ast_and_transitive_launches` fails on dev at its FIRST assertion — an undeclared `WorkspaceCreateModal` launch (which itself launches an undeclared `SelectDirectory`, so the drift is two levels deep). Everything after that line is dead-lettered: the modal-count bump and the `reachable_modal_types == all_contract_types` set comparison never execute, so the inventory contract silently stops guarding new modals even while the file looks maintained. Found during task-18515's whole-branch review, which had to re-run the walk with assertions disabled to confirm its own edits were correct.

## Acceptance Criteria (the what)

- [x] `WorkspaceCreateModal` and its transitive `SelectDirectory` launch are declared (or explicitly excluded with a recorded reason), so the test reaches its later assertions
- [x] The whole test passes on dev, with the count and set-equality assertions actually executing
- [x] A guard exists against the same silent-skip class: the assertions that matter are ordered or structured so an early failure cannot mask them (e.g. collect all mismatches and assert once, or split into independent tests)

## Implementation Notes

Two undeclared launches, not one. `WorkspaceCreateModal` (opened by the
Console workspace browser's `_create_console_workspace`) and its vendored
`SelectDirectory` picker are now declared as launch edges with real
TASK4 dismissal contracts — both already subclass `SafeModalDismissMixin`
and carry `escape -> request_safe_cancel`, so they qualified as-is.
`SelectDirectory` is excluded from the ROOT's declared set the way
`ChangeRevertConfirmModal` already was: the root does not construct it, its
opener does.

Declaring those let the walk run further, where it immediately caught a
SECOND undeclared launch: `ConsoleReviewNotesModal -> ConfirmationDialog`
(the per-note delete confirmation), which had shipped in task-18515 without
this test ever checking it. That is the silent-skip this task was filed
about, demonstrated on live code rather than argued.

The structural fix: `_walk_modal_launch_graph` now COLLECTS mismatches and
raises once after the traversal instead of asserting per owner, so one
stale declaration can no longer abort the walk and mask both later owners
and the calling test's own count/set assertions. Pinned by
`test_launch_walk_reports_every_mismatch_not_just_the_first`, which fails
if only the first mismatch is reported.

Reachable-modal count 42 -> 44; TASK4 contracts 9 -> 11. Whole file green
(117 tests) — it had been red on dev at its first assertion.

## Review round (Qodo, PR #1821)

Two of three findings were real defects in the first fix, and both were
about the guarantee being weaker than claimed:

- Collecting mismatches but still RAISING at the end of the walk left the
  caller's count/set assertions unreached in exactly the failure case the
  task was about. The walk now returns `_LaunchWalkResult(reachable,
  mismatches)` and raises nothing; the mismatch check is its own test, so a
  stale declaration fails that test while the inventory test still runs and
  reports its own drift.
- The frontier only followed DECLARED launches, so an undeclared modal was
  reported but never traversed and could still hide every mismatch beneath
  it. Strays are now scanned too, without being promoted into `reachable`
  (which stays the declared set the contract table is compared against).
  Pinned by a test proven to fail with the traversal disabled.

Third finding was a docstring-style fix. File now 119 green.
