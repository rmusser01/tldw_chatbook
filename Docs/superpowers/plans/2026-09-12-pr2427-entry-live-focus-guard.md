# Library entry live-focus guard implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development with
> independent spec and quality reviews. Root is the sole Git writer.

**Goal:** Preserve newer live user focus when an idle entry callback runs before
the queued focus event disarms it.

**Architecture:** Add one receipt-free guard to the existing continuation,
using existing identities and disarm logic. Preserve every non-null Media
receipt path and current timer policy.

**Tech Stack:** Python, Textual Pilot, pytest, native macOS FD observation.

ADR required: no.
ADR path: N/A; TASK-2856 existing user-navigation contract.
Reason: routine event-ordering repair, no new ownership or stored policy.

Spec: `Docs/superpowers/specs/2026-09-12-pr2427-entry-live-focus-guard-design.md`,
independently reviewed and explicitly approved by the user for implementation.
Baseline: published `bf76bb1cd1` in `.worktrees/pr2427-review-recovery`.
Task: TASK-31932 AC3, still In Progress. Preserve the unrelated untracked
reader-paydown plan without reading or editing it.

## Execution evidence

Independent plan review approved without material gaps. The explicit-anchor
mounted regression failed on unchanged runtime exactly because the actual
retry callback replaced Retry with the type filter:
`/private/tmp/pr2427-entry-red-explicit.KJQmq9/pytest.log`.

After the receipt-free guard, the complete new module (15 cases) plus both
unchanged stale-callback parameters and the unchanged initial-error recovery
case pass: 18 passed, three dependency warnings, 8.13s. Native evidence:
`/private/tmp/pr2427-entry-native-final.3ANu6Y/{pytest.log,fd_identity.jsonl}`.
Final inventory: six descriptors, zero SQLite and zero instance-lock handles.
This successful run does not close the separately recorded older Shell
failure-unwind resource finding. Full new-test Ruff/format, fatal Screen Ruff
and whitespace checks pass. Independent implementation reviews and frozen
complete-file qualification follow; no whole-PR readiness claim.

Independent spec and quality reviews both approve the bounded implementation
without findings. Source is frozen for complete-file qualification at these
SHA256 values:
- New test: `7684de50852b9c2f7de212df56855e590f15d12ebbd910abe97137dac132a1d4`
- Screen: `44d4f30886d4f8153886340b30f3b418c566f34082e89c4a200a55c5531fed86`
- Unchanged Shell: `6cc1bb7ca31d7581917f60bb2daaf480d90c5f7c6973b9fee5257a06d87d4022`

The first complete four-file run is terminal: 97 passed, two failed, three
dependency warnings, 161.57s. Report root:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-entry-complete.zZmTNmj82J`.
Final native inventory: six descriptors, zero SQLite/instance locks. Both
failures are in the settlement module's `COMPACT_ROW_SCROLL_Y` pin: real
capture/restoration is `(0, 29)` versus pinned `(0, 28)`. Baseline-method
comparison and geometry provenance remain under investigation; no assertion
was weakened and source remains frozen while complete Shell runs finish.

A process-local baseline comparison restores only the AST-extracted
`bf76bb1cd1` continuation against current module globals, without editing any
repository source. Both exact cases fail identically (2 failed, two warnings,
4.88s): `/private/tmp/pr2427-settlement-baseline.GZdXI0/pytest2.log`.
The guard is not causal; accepted geometry provenance is being checked before
changing the test pin. The diagnostic is not unmodified native qualification.

Complete Shell partition independently approved: full 872 unique node IDs,
shards 293/298/281, exact-once Counter equality and each node's installed
pytest-shard SHA256 modulo-3 assignment verified. Collection evidence:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-entry-inventory.H0ckJ9Ji0J`.
All three native runs use separate report/basetemp roots and unchanged source.

Frozen complete Shell is terminal, with the runtime/Shell hashes unchanged:
871 passed and one failed across all 872 exact-once cases. Shard 0 passed 293
in 642.09s (three warnings, final six FDs); shard 1 passed 298 in 751.03s
(three warnings, final seven FDs); shard 2 passed 280 and failed one in 636.07s
(ten warnings, final three FDs). Every final inventory has zero SQLite and
instance-lock handles. Report roots under the per-user temporary directory:
`pr2427-entry-shell0.BZ3YpQKMsi`, `pr2427-entry-shell1.L0Kzsnap8R`,
`pr2427-entry-shell2.VP5vXaDbg4`.

The sole failure is
`test_library_media_durable_mutation_gates_and_refreshes_applied_scope[False]`:
the initial transition times out with "Media page 2 never applied" before
mutation assertions. An isolated current-source native rerun passes (1 passed,
three warnings, 3.94s; final six FDs, zero SQLite/locks) in
`pr2427-entry-mutation-repro.1ArgPtBW18`. The full-run failure is still open;
an isolated pass does not establish its cause or close it. All earlier six
complete-Shell failures, including both original Retry parameters, pass here.

## Follow-up: reconcile the proven compact scroll pin

ADR required: no.
ADR path: N/A; accepted TASK-32350 scope-line contract.
Reason: test-only geometry reconciliation, no product or ownership change.

TASK-32350 (`d8982e1f5a`) adds the measured one-row scope line under the Media
header after TASK-32064 removed the one-row Chunking strip. The later pin
adjustment accounted only for the removal. Current observation confirms the
scope row is one row high, the scroller is three rows high, and row 15 reveals
at exactly 29. Old and current continuation runs both fail identically.

After all frozen Shell batches finish, independently review this bounded plan,
then change only `COMPACT_ROW_SCROLL_Y` from 28 to 29 and its adjacent
provenance comment in `Tests/UI/test_library_media_return_settlement.py`.
Do not change either exact assertion, helpers, production, shared fixtures or
any other test body. Verify executable AST equivalence except this one module
constant, rerun both failed cases, then the complete four-file cohort natively.
Review the exact diff independently and record static checks without claiming
the old module's pre-existing formatter debt is new or fully clean.

Follow-up implemented after all Shell processes ended. Independent spec and
quality reviews approve; AST equality except the one constant is verified.
Exact two native cases pass (5.58s, three warnings, six final FDs and zero
SQLite/locks) in `/private/tmp/pr2427-scroll-pin-native.cPEYft`.
Final complete four-file qualification passes **99 tests**, three dependency
warnings, 153.55s, exit zero, with six final FDs and zero SQLite/instance locks:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-entry-complete-final.BbuL5DddDi`.
Settlement-file SHA256:
`3da6419d575997634d903f0aea8a3de07e39c40a4dd54dcce07798a414113bbe`;
all other frozen hashes remain unchanged. Full Ruff for both touched tests,
new-test formatter check, fatal Screen Ruff and whitespace checks pass.
The older settlement module retains identical pre-existing formatter debt.
This closes the approved focus repair and observed scroll-pin mismatch, not
the separate Shell readiness timeout or whole-PR gates.

## Task 1: Regression first, then the shared continuation guard

Files:
- Create `Tests/UI/test_library_entry_live_focus.py` for focused mounted and
  supporting branch controls; reuse existing Library harness/service fixtures.
- Modify only `_focus_library_list_entry_if_current` in
  `tldw_chatbook/UI/Screens/library_screen.py`.

- [x] Read TASK-2856, the approved spec, existing continuation/arm/disarm/focus
  handlers and their callers. Read testing-evidence entries on deferred focus
  and input-owned intent. Do not change unrelated focus or layout owners.
- [x] Add a mounted Media-failure regression using the real Retry and existing
  `FailingFirstLibraryMediaScopeService`. After real layout, set the original
  arm anchor and arm entry. Capture the actual callback passed to `set_timer`
  by `_retry_library_list_entry_focus_while_armed`, forwarding to the real
  timer API. Synchronously set Retry focus, assert it is attached/current, then
  deliver that captured callback without yielding to DescendantFocus.
  Assert exact Retry focus identity, old request disarmed, and timers cleared.
  Do not replace the focus continuation with a recording mock for this test.
- [x] Run the new regression on unchanged runtime using a fresh report root:
  `PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_library_entry_live_focus.py -q --tb=short`.
  Required RED: actual Retry focus is replaced by the type filter, not setup,
  attachment or a missing timer callback. Record evidence before runtime edits.
- [x] Add bounded controls for permitted anchor, armed-list row, no focus,
  detached focus, adaptive grip and exact current programmatic target; reject
  a foreign non-row, foreign-list row, stale programmatic target and same-ID
  replacement for an old identity. Disarmed/superseded generations must not
  move focus or disarm a newer arm. Supplement the real mounted regression
  with lightweight branch fixtures only where actual attachment is not the
  behavior being claimed. Use real existing methods wherever possible.
- [x] Implement the following predicate inside existing pending/generation
  admission, immediately before the final ordinary `_focus_library_list_entry`
  call. Preserve the complete current candidate-receipt branch above it:

  ```python
  if receipt is None:
      focused = self.focused
      row_class = _LIBRARY_LIST_ROW_CLASS_BY_ROW_ID.get(self._library_selected_row_id)
      if (
          focused is not None
          and focused.is_attached
          and focused is not self._library_pending_list_entry_focus_anchor
          and focused is not self._library_notes_programmatic_focus_target
          and not focused.has_class("library-adaptive-reader-pane-grip")
          and not (row_class is not None and focused.has_class(row_class))
      ):
          self._disarm_library_list_entry_focus()
          return
  ```

  Update the method's short docstring to cover ordinary entry as well as
  semantic returns. No new helper/state, timing change, cap change or exception
  suppression. Non-candidate non-null receipts bypass this new check too.
- [x] Verify non-null candidate and non-candidate receipt routing remains
  unchanged, with direct branch controls plus existing real Media-return tests.
- [x] The new module owns all apps it creates: reuse the established
  `close_owned_console_resources`, `close_owned_console_test_apps` and
  `close_owned_console_workers` fixtures with its local `_build_test_app`
  binding. Do not modify shared fixtures or apply cleanup to older modules.
- [x] Rerun the complete new file, both original stale-callback parameters and
  `test_library_media_initial_error_is_unknown_and_retry_is_unique` unchanged.
  Use the native observer and fresh report directories. Never infer failure
  unwind cleanup from a clean success-only run.
- [x] Run scoped full Ruff on the new test and fatal Ruff on the Screen;
  whitespace check, actual-diff self-review, independent spec review then
  independent quality review. Address findings before declaring source frozen.

## Task 2: Frozen integration qualification and checkpoint

- [x] Root qualifies complete `test_library_entry_live_focus.py`,
  `test_library_media_entry_focus_cleanup.py`,
  `test_library_media_return_settlement.py` and
  `test_library_media_side_by_side.py` with the unchanged native observer.
  Retain all exact scroll, replaced-origin, source/generation and newer-focus
  assertions. Stop to diagnose any failure; do not edit expectations blindly.
- [x] Run complete `Tests/UI/test_library_shell.py` on frozen source; this is
  the affected file, not a full repository sweep. A separately reviewed exact-
  once partition is allowed if needed. Preserve strict focus/geometry/cleanup
  observations and report exact terminal counts/warnings/resources.
- [ ] Update TASK-31932 and this plan with results, then checkpoint reviewed
  work. A scoped progress checkpoint may precede the long complete Shell run,
  but must explicitly leave its qualification open. Never claim whole-PR
  readiness from targeted runs. Other size/preload/CSS/resource/rebase/review
  gates remain separate; no unattended automation update is authorized here.
