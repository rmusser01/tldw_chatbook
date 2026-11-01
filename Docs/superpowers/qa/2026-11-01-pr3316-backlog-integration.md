# PR3316 local integration evidence

Source: 1a536c706c42efbcc63340d603d66061fb19accc. Prepared on PR3315 integration head 5d7cad1c8598495e398cf25cd976352c8689b23a; final dev rebase and review remain pending.

The change retains checkout-free cleanup of obsolete same-repository PR runs and makes artifact checks precede test fan-out. Parsed comparison preserves all seven application jobs except their preflight dependency and two explicit test-selector expansions. All thirteen moved guard steps remain identical; all 32 UI shards, max-parallel 17, UAT steps, installs, timeouts and aggregate verdict conditions survive.

## Local execution

- Frozen focused CI run: 641 passed, 2 failed, 32 unchanged exhaustive shard-distribution cases deselected. Both failures reproduced on the exact provisional base.
- After the first selector repair: 642 passed, 1 failed, 32 deselected. The remaining fixture-derived compact-now selector required the same repair. Final affected selector/preservation contracts: 64 passed. All 306 static node selectors resolve.
- Replace edit/resend's two stale fixture-derived IDs with all six cases of its function, and compact-now's stale single ID with all three cases. The inherited-target assertion retains the already-present terminal-input file. Coverage expands; the checker and required verdict are unchanged.
- The thirteen real artifact guards initially passed 11/13. The Mermaid mismatch came from an incomplete local LF export; exporting the nine omitted tracked vendor inputs and verifying their Git blob hashes made the unchanged checker pass with Python 3.12.11.
- The markup guard found two unchanged Kanban error sinks. Two mounted notification regressions failed with real Textual MarkupError before the fix. Adding markup=False at the existing load/mutation sinks made both pass (r-3de4f7d8, 2 passed, 0 errors/skips, source hashes stable). The unchanged markup guard then passed; its census remains untouched.
- Scoped Ruff fatal/error checks passed for the modified tests and Kanban source. Final base-sensitive checks and independent immutable-head review must be recorded before publication.

Receipts are in the sibling pr3295-resync directory: ci-66f8deea,ci-56740ef1,ci-selector-final.xml,pr3316-all-selector-scan.json,pr3316-preservation-after-selectors.json,guards-829339ec,r-f44199da (RED),r-3de4f7d8 (GREEN). The exact-base CI comparison is in sibling pr3315-resync/pr3316-ci-base.xml. Earlier failures remain recorded; incremental reruns are not represented as one clean full run.

No full-suite, hosted cancellation, throughput, or live runner-scheduling qualification is claimed. Hosted CI and size limits are waived by the user for this backlog. Draft PRs and PR #3267/#3297 are excluded. No release or main promotion is part of this session. ADR-103/218 govern the source policy; the Kanban change enforces existing literal-notification convention under ADR-237/150 without a new boundary.
