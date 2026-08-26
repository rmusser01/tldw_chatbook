# Final whole-branch review correction report

Date: 2026-08-26

Status: implementation and controller-owned verification complete; `TASK-22507.1`,
`TASK-22507.2`, and `TASK-22507.4` remain **In Progress** with the affected
acceptance criteria open for independent re-review.

Implementation commit: `df67e53bb1` (`fix(console): close semantic capture review gaps`).

Fix round 2 implementation: `873972639f` (`fix(db): suppress raw exchange error chain`).

## Decision record

ADR required: no

ADR path: `backlog/decisions/089-console-full-semantic-capture-and-export.md`

Reason: these corrections enforce ADR-089's existing Safe-first,
capture-never-breaks-send, Safe-before-disk, content-free logging, and truthful
purge/UI contracts. They do not introduce a new storage, ownership, security,
or cross-module policy decision.

Controller ruling: when a persisted conversation policy is unavailable or
corrupt under Global Full, publish explicit session Safe, retain save-pending,
and keep hydration retryable; do not disable Safe capture entirely. ADR-089
defines Safe as the fail-closed diagnostic mode, while disabled capture is
reserved for admission bookkeeping failure that cannot safely resolve policy.
If wrong, Safe diagnostic metadata may still be retained during a repository
outage, but semantic bodies cannot escalate to Full.

## Corrections

1. Persisted conversation-policy reads now return typed
   `ABSENT`/`FOUND`/`UNAVAILABLE_OR_CORRUPT` outcomes. Only conclusive absence
   inherits; unavailable/corrupt/schema-error reads publish explicit Safe,
   remain save-pending, and retry on a later hydration attempt.
2. The provider gateway sanitizes final accumulated response and tool-call
   content after streaming joins. Split data URIs and plain base64 therefore
   cannot evade the per-chunk sanitizer or reach SQLite/Full export, and the
   shared capture budget is not consumed twice.
3. Full-to-Safe conversation mutation publishes a runtime Safe override while
   holding the mutation reservation and before durable I/O. Failure or
   cancellation retains truthful Safe/save-pending state; Full escalation
   remains durable-first.
4. The runtime capture-policy cache tracks canonical config generation and
   atomically rebuilds from the already-published config snapshot. Deferred
   `CaptureDetail` import remains intact without config-file I/O.
5. Purge bindings freeze controller-owned availability. The modal starts
   disabled, exposes the exact blocker, takes fresh immutable title/count/
   effective-policy/availability and policy/capture revisions before both
   confirmation and mutation, and gives complete destructive-action copy.
   Global scope disables Inherit.
6. Store, runtime, and resolver failures at capture admission are caught and
   logged with content-free phase/type categories. The accepted model run
   continues with capture disabled, and one-shot ownership is not consumed by
   failed policy resolution.
7. The lowest SQLite exchange-write boundary logs only a stable category,
   message id, and exception class, then raises a content-free
   `CharactersRAGDBError` chained from the original.
8. Inspector status includes the immutable conversation title and names an
   armed `Next eligible send: Full` override in the compact two-line layout.
9. Global policy editing offers only explicit Safe/Full detail and presents
   scope-accurate guidance.

## RED evidence

- The first executable focused correction run collected 19 regression items:
  **17 passed, 2 failed**. The failures isolated (a) the incorrect assumption
  that unavailable hydration disabled Safe capture rather than publishing
  explicit Safe, and (b) purge becoming enabled before its fresh availability
  snapshot settled. Both were corrected without weakening their assertions.
- A new immediate pre-mutation policy-revision fence test was then run alone:
  **1 failed**, because mutation still executed after confirmation changed
  policy state; after the fence was added it was **1 passed**.
- The first exact 16-file privacy/UI matrix after generation-aware runtime
  projection was **882 passed, 4 failed, 2 skipped**. All four failures proved
  the first implementation accidentally performed config-file I/O under a
  changed `TLDW_CONFIG_PATH`. Rebuilding from the already-published snapshot
  removed that side effect; the exact matrix then passed.
- Named reproductions cover schema-unavailable/corrupt/error hydration,
  blocked Full-to-Safe writes, all three admission-resolution seams, split
  streamed binary/base64 through real SQLite/export owners, canonical config
  generations/concurrency, DB semantic/path/binary canaries, fresh purge
  state/revision/blocker/80x24 behavior, Global Inherit, and compact Inspector
  truthfulness.
- Fix round 2 extended the real lowest-DB-seam canary and produced **1 failed**:
  the sanitized outer `CharactersRAGDBError.__cause__` was the raw
  `OperationalError`, so recursive exception inspection and
  `traceback.format_exception(...)` exposed semantic, path, and binary
  canaries.

## GREEN evidence

- Changed focused repository/store/controller/gateway/export/config/DB/UI:
  **423 passed, 2 skipped** in 36.65s after fix round 2.
- Prior cancellation/race compatibility focus: **8 passed**.
- Exact prior 16-file privacy/UI matrix: **886 passed, 2 skipped** in 381.03s.
- Complete changed DB area (`Tests/ChaChaNotesDB Tests/DB -q`):
  **1831 passed, 1 skipped** in 192.28s after fix round 2.
- Fix-round-2 lowest-seam focus: **1 passed**; complete message-exchange DB
  file: **12 passed**. The outer error now has neither cause nor context, the
  recursive exception graph and rendered traceback are content-free, and the
  stable log category/message-id/error-type contract is unchanged.
- Prior Task 2 policy/controller/provider gate: **570 passed, 2 skipped** in
  30.99s.
- Real 80x24 policy/Inspector gate: **114 passed** in 41.73s.
- Settings/config/layout gate: **381 passed** in 304.70s.
- Production-shaped gateway/controller/store/SQLite/export sentinel plus the
  explicit Task-4 base-delta node: **2 passed**.
- Ruff on every changed Python file: all checks passed. Changed production
  modules compile. CSS build and all five source-sync checks pass. Docs grep
  required no factual edit. `git diff --check` passes.
- `chat_screen.py` measures **20,093 lines / 633 methods**, within the reviewed
  Task 4 base **20,099 / 633**. The independently stale absolute ratchet was
  neither run nor raised.

Skips are environmental and unchanged: two loopback-listener tests cannot bind
inside the sandbox, and one DB posture test is Windows-only. Dependency and
pytest temporary-directory cleanup warnings are incumbent and did not fail a
gate.

## Residual risk

No known Critical or Important implementation defect remains in the corrected
surfaces after fix round 2. Independent re-review is still required before the
reopened child criteria can be checked or their statuses returned to Done. The
one-time Impeccable detector was not rerun, as directed.
