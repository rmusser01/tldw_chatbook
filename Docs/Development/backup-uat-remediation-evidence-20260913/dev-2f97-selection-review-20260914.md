# File Notes CI inclusion review

APPROVED. Read-only AST/literal-selection comparison against HEAD confirms the sole change is adding each of the two new four-case parameterized nodes once to `_PRODUCT_TESTS` and `_SUPPORT_DIAGNOSTIC_TESTS`. Removing those four literals yields an identical whole-module AST.

Full, support and support-diagnostic each gain exactly these eight cases. No prior node is dropped or reordered, no selected whole-file entry duplicates either new node, and every other selection is identical. Public selection names, indexed leading selections, installed-package qualification, deadlines, workflow, assertions and product calls are unchanged.

The new tests already independently passed all eight cases in 9.28s during the source review. This review did not rerun tests or claim the parent's in-progress collection result. Expected collection delta: +8. Root reports runner Ruff/Bandit zero findings.

Runner SHA-256: `2391fc83e35c1b909f910b3afa8f7a11c62cc30cd9f50e264e4aed32f75a83a1`.

The first disposable selection-evaluation attempt omitted the unchanged `_RESTORE_DIAGNOSTIC_TESTS` input and raised NameError; including that existing literal tuple completed the proof. No repository change was made.
