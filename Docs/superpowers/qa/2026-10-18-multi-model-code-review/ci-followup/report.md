# PR3286 CI follow-up

Baseline: `98bb849eb6a92f429fe2104a39222427602a0e39`. The required normal CI
failed because UI shard9 raised StopIteration; its aggregate failed downstream.
Separate Windows eval and GGUF jobs timed out. This candidate repairs the
reproduced UI test race and preserves more precise evidence for the unexplained
Windows failures. Hosted validation is pending; AC24 remains open and PR3286 draft.
No qualification, provider request, product child launch or CI rerun occurred.
The approved single qualification allowance remains60requests/3children, wholly
unused; cumulative actual usage stays386requests/26children.

## Demonstrated UI defect

The notes-sync attention journey waited for any folder row, then used a strict
lookup for VSync inside its polling predicate. An unfinished/recomposed folder
list can lack VSync even after that generic wait. The lookup then raises instead
of waiting for the specific row. Both attention and healed-state predicates now
check VSync plus the required status without throwing. The existing30s deadline,
real Recovery action and every UI/database/file-content assertion remain.

A control holds the real root-folder page, observes VSync absent, evaluates the
actual predicate, then releases the page and completes the original journey.
The original code fails with the exact StopIteration; the final pair passes2tests.
Only test synchronization changed; no production notes behavior changed.

## Windows limits and diagnostic changes

The eval failure is a restore child exceeding45s after its first full-app child
completed. Its retained child log was empty. The same code previously passed
Windows on4526f771b7; a fresh macOS phase trace completed restore in9.3s. No
production cause or timeout increase is established. A10s repeating faulthandler
watchdog now writes stacks to the existing owned log before the unchanged45s
kill. The complete capture/restore/reopen case passes locally in36.81s.

The first GGUF failure is inventory replay[False] at180s, after startup completed
in2.796s. A later candidate keyboard diagnostic times out during close with all
eight default-executor threads in Textual Windows timer waits. Both pinned-dev
keyboard controls passed; the earlier shorthand that pinned-dev also failed was
incorrect. Portable controls using the actual installed timer implementation
reproduce cancellation/executor starvation, but do not establish the first
inventory timeout's cause. No product timer workaround is installed.

The existing finite-code observer now includes six original loaded timer code
objects and their suspension coordinates. It records bounded counts/locations,
no argument values, callback wrappers or executor inspection. The first Windows
inventory case uses this observer ONCE in place of its original invocation, with
no profiling and the same180s deadline. Existing keyboard diagnostics retain60s.
A failure still fails CI; these diagnostics cannot turn it into a success.

## Validation and review

- UI original journey plus deterministic control:2passed.
- Exact retained eval capture/restore/reopen journey:1passed. A bounded blocked
  child emitted a traceback and the successful-child control cancelled its timer.
- Observer and exact workflow-shell controls:49passed. Windows/Linux/macOS command
  selections execute each original node once and preserve nonzero exits.
- Changed Python ranges are formatted with AST equivalence verified. Ruff passes
  for diagnostic/new test files; two unchanged import warnings remain in existing
  UI/eval tests. Whitespace passes.
- Two scoped reviews found no remaining actionable issue. The shell path oracle
  uses Bash's own PWD, preserving Windows path spelling. No completed feature or
  V1 review was repeated.

ADR required:no. These are test synchronization and bounded observation changes;
existing notes-sync, native recovery/admission and runtime contracts are unchanged.
Raw logs, reproducers and scoped review summaries are retained separately in
[retention.json](retention.json). No full-suite run, timeout increase, skip,
permission/storage relaxation, automatic CI retry or merge was made. Successful
hosted CI is required before consuming the approved qualification.
