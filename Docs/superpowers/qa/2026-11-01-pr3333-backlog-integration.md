# PR3333 integration verification

Integrated source a16f79c7a2e34ab81db4d101d43da34cf3a8b650 by rebasing onto
dev 2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5. Preserved the current
attention-refresh fallback, publication/session ownership, diagnostic owners,
and UI census order. The MiniMax repair distinguishes cancelled, refused,
deleted and unconfirmed outcomes. Refusal no longer asserts continued
rendering; deletion reports a finished record without claiming success or a
certain charge. Existing ADR-044 applies; no new boundary or ADR is required.

## Qualified local evidence

The existing Python interpreter and native private-fixture runner executed only
selected affected cases. Bootstrap-profile isolation, native private ownership,
source-selection checks, network guards and shutdown grace remain intact. Raw
logs/configuration stay in the dedicated private receipt root. Content-free
receipts bind exact commands, selected case identities, base/head, applicable
conftests/local fixture dependencies and before/after source hashes.

- Two pre-repair adapter cases failed at their intended behavioral assertions:
  already-cancelled refusal must not claim continued running, and a succeeded
  task's deletion must have the distinct deleted outcome.
- `green-contract`: 27 distinct selected adapter, ComfyUI, worker, Generate
  helper and Regenerate cases passed, with no failures/errors/skips and
  unchanged enrolled sources. Accepted DELETE request/headers/empty body,
  strict refusal parsing, generic-400 and transport controls remain pinned.
- `green-mounted-preservation`: 19 distinct selected cases passed, with no
  failures/errors/skips and unchanged enrolled sources: the five-case Stop module
  (including its real failure control), both draft routes, both switched-session routes,
  nine publication/cancellation cases and the metadata-only diagnostic guard.
  The real Stop path traverses the controller, worker, adapter and recorded
  HTTP transport, asserting stored and compositor-painted transcript rows.
- Fresh isolated `scripts/check_persistent_diagnostic_inventory.py --diff`
  passed: 722 owners, 16 persistent sink files. The integration repair changes
  only the adapter pin; the source PR's two Console Stop logs remain reviewed.
- `scripts/check_ui_pr_gate_census.py` passed: 1324 entries, floor 1317.
  All 1323 base entries retain their exact order; the Stop file is appended.
- Changed-file Ruff comparison against the exact integration base: five
  diagnostics on each, no introduced diagnostics. No new formatter failure.
  Existing formatting debt remains in the diagnostic test, send-freeze
  harness and Console video controller. An introduced mixed-newline formatter
  failure in the census comment was repaired with AST-equality evidence.
  No CSS source or artifact changed, so CSS regeneration and unrelated gates
  were unnecessary.

There are 46 distinct qualified behavioral cases; repeated attempts are not
added to that total. The immutable review package records the final head,
whole-branch diff, hashes and source/evidence binding without relabeling earlier
tested heads/bytes.

## Retained unsuccessful attempts

The first mounted attempt failed at an incorrect paint-observation helper and
also errored during native-worker teardown. The amended observation passed its
body assertions, but teardown still failed. Content-free thread evidence
identified the actual app-owned ChaChaNotes FTS backfill worker importing
built-in content. The shared harness now retains and waits for that real
worker plus Console attachment readiness, using the existing startup settle
bound. Product shutdown guards/grace were unchanged. The final mounted batch
qualifies that fixture correction. This diagnosis applies to PR3333 only.

The intermediate mounted attempt also overlapped an inventory writer because
the orchestration lost its running-session metadata. That attempt and the
overlapping inventory check are disqualified even though their source hashes
matched. The writer's exit status is unavailable; its process retirement was
confirmed before the fresh isolated inventory check and final frozen test run.
No unavailable exit or teardown error is reported as a pass.

## Limits and open work

Recorded-response replay proves parsing, the emitted request and local control
flow. It does not qualify current live vendor behavior, billing or latency.
No live DELETE, new provider request, paid generation, optional installation,
full-suite sweep or hosted CI ran. Hosted CI and numeric-size enforcement were
explicitly waived; no limits or product/security/custody checks were weakened.
Independent whole-branch review is the publication gate owned by the coordinator.

TASK-33621.45 is In Progress with universal Stop AC#1 unchecked: in-flight
GET/download can still resolve as failure/success at inherited poll boundaries.
TASK-33621.50.1 (prevent terminal DELETE/recover a clip), .50.2 (paid Regenerate
consent/progress) and .50.3 (pre-submit Stop) remain open. TASK-33621.50's
qualified notes separate historical 2026-10-30 observations from current replay.
The Backlog CLI was unavailable locally; the existing task files were edited
directly, and no new task was marked Done.
