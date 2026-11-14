# Task 15 implementation report

## Result and scope

Implemented the bounded repairs from `task-15-brief.md` at clean BASE `100fa9d8191620645e003b2e9ec5e076b7763ad2`. Only the five owned source/test paths change. Task14 source `34cf170802` is retained in ancestry. No publication, merge, dependency install, full sweep, new skip/xfail, warning suppression, timeout increase, or budget change occurred.

ADR required: no new ADR.
ADR path: `backlog/decisions/219-console-chat-destinations-and-bounded-starts.md`.
Reason: restore existing physical custody, exact settlement and truthful launch outcomes without changing storage, authority, unknown-root policy, or public abort semantics.

## Implementation

- `Chat/console_chat_start.py`: both restriction-registration sites use one narrowly scoped cleanup helper. It logs only phase and exception type, preserves an installed restriction, and retries registration on unresolved settlement. Both original physical drain blocks remain exact and precede cleanup/release. Nested finalization keeps the outcome fallback, exact active-item removal and captured owner/token release reachable through publication, abandonment and run-state faults. Existing preparation/current-state/replacement fences remain.
- `DB/automatic_work.py`: `_confirm_chat_start_absent` reads the exact attempt only under the existing FULL/BEGIN IMMEDIATE transaction and a present matching durable runtime-owner row. It returns its positive result after successful context exit, including commit and policy restoration. It makes no attempt/reservation mutation. The public `abort_chat_start` signature and body are byte-identical to BASE.
- After ordinary settlement fails, the coordinator may call this private observation only for unaccepted, unreceipted, context-unavailable preparation and after physical draining. Confirmed absence clears only the attempt/captured-owner entry and returns existing `not_started`/`start_failed` (preserving withdrawal override). Prepared commit-then-raise retains the original refund and `preparation_unconfirmed` result. Accepted work stays charged and root-paused; false or failed observations retain uncertainty. No opening text retry or identity redesign was introduced.
- `Chat/console_chat_fork.py`: only the `console_fork_visible_selection` Google docstring changed. Whole-module executable AST, signature, returns and every other helper remain exact.
- The two test files append controls; both original files remain exact prefixes, preserving every original assertion, wait, marker and import.

## RED evidence before implementation

All commands use the shared Python3.12, worktree PYTHONPATH, canonical private pytest fixtures and existing 300s timeout. The exact argv, execution hashes, result, case IDs, full log and XML are stored by run prefix in `task-15-safe-evidence`.

`red`: the four initial new selectors expanded to 19 cases: 14 failed/5 passed. The native rig with real held commit/provider thread workers reproduced both registration escape sites; real SQLite trigger rollback reproduced a missing-row wildcard restriction, including combined registration failure and another-attempt controls. The new predicate was absent. Prepared/accepted commit-then-raise and missing/stale/failed-observation controls preserved their existing conservative outcomes.

One failure was test setup: updating immutable attempt owner_id was correctly rejected. Setup was corrected to prepare under a foreign runtime owner and then replace the durable runtime-owner row. `red-extra` reran only that case plus real writer-contention and post-drain cleanup controls: 4 failed/2 passed. The corrected foreign case failed because the private predicate did not exist; contention failed with false review/uncertainty; abandonment and run-state faults escaped cleanup. Publication fallback and replacement-token/current-state preservation already passed.

Provider-first registration is an explicitly adversarial local receipt-flag loss after real native dual-store acceptance and provider-thread entry. It retains the already published started outcome. Commit-thread cases cover unresolved pre-publication outcomes. These controls do not claim a normal provider can dispatch before both durable fences. The private preflight probes remain diagnostic only and are not source acceptance evidence.

## GREEN and existing controls

- `green`: 24/24 passed, including all initial new controls and corrected foreign-owner setup (24.66s pytest).
- `green-boundaries`: 6 added guards passed: failed BEGIN/read/commit/policy restoration and first-registration failure followed by durable-settlement failure/re-registration, for real commit and provider owners. One expanded replacement-preparation setup failed; evidence is retained.
- `replacement-green`: the first setup correction still failed because ordinary abandonment cannot cancel a COMMITTING preparation. `replacement-final`: 1/1 passed after using the existing COMMITTING rollback→PAUSED→exact abandonment path before installing a real replacement. Its active item, token, run state and preparation all survive old cleanup.
- `existing`: exactly the 16 preflight selectors expanded to 31/31 passing cases in 31.74s. No unrelated cohort was replayed. Exact IDs are recorded in `actual-case-outcomes.json` and `final-case-union.json`.
- Final union: 61 unique authorized passing cases (30 new, 31 existing). No pytest skips, xfails or warnings in successful receipts. Deliberate injected failures produce existing diagnostic logs; no warnings were suppressed.

`failure-attribution.json` separates the original defects, new test setup mistakes and two private verifier mistakes (a filename-suffix collision and accessor/runner name collision). Every affected receipt remains available. No production scope expansion followed those failures.

`execution-source-AST-map.json` reconstructs every RED/GREEN source version and verifies it against the hashes recorded before actual execution. The final coordinator/ledger executable AST equals the successful GREEN version; later production edits were formatting and the fork docstring. New boundary/decorator additions and replacement setup changes remain explicitly mapped, not relabeled as earlier executions.

## Static checks, self-review and preservation

Fatal Ruff (`E9,F63,F7,F82`), all-five-file Ruff format check and `git diff --check` pass. Changed-hunk formatting records preserve AST. These five owners have zero inherited formatter debt; no unrelated formatting was performed.

`source-AST-import-authority-custody-map.json` and `public-abort-body-exact.json` verify the exact public abort body, all other ledger methods, all coordinator authority/preparation/acceptance/receipt/currentness methods, imports, original main-run body/handlers and both distinct physical drains. Only `_run`, the new private registration helper, the private absence predicate and the one docstring differ. Self-review checked exact SQL, post-transaction positive return, absence guard, both cleanup sites, physical custody, final release, type-only logging and replacement fences.

All 29987 original tracked paths are accounted for; only the five owned paths differ. All 10409 QA paths at BASE are unchanged. All 1446 frozen safe evidence/report artifacts are exact. Task14's seven latest-dev source exceptions and exact reversible CI fixture repair remain unchanged. Root's live coordination metadata is intentionally excluded from immutable pins.

The qualification ZIP remains 63,166,118 bytes, SHA256 `acecbdfb6ddbafe6df679c39f9f137f2bfb49a1f87b62556e1345021f8384c84`. Exact whole-container preservation carries all 2118 original entry bytes, all 99 visible reports, five packing files and the recorded olderQA carry, without decompression/export/replay. Existing historical receipt maps are unchanged.

Historical 35 loading evidence remains historical at its original source. Controller 29301/store 22344/interrupt 6479/compaction 4185 and screen 25218/759 caps, the 50-line slack rule and all startup/import/storage budgets are unchanged. Existing concerns remain visible: app681/686, ready1033/1033, preimport557/557;415310/425347LOC;library127527/135111LOC;CSS607326/608090bytes; Task13 FD start14/end383,growth369 over200, unsuppressed and without new causal attribution. No new loading result is claimed.

## Limits and handoff

If both in-memory restriction registration and durable settlement are unavailable, this bounded fix cannot guarantee cross-handle denial. It records the failure and preserves physical custody/outcome/release; it does not invent a fallback database identity or expand authority. This is the brief's explicitly retained limit. No unresolved source correctness concern remains from self-review.

All owned pytest processes have exited; actual native workers are drained in the controls and all run receipts have exit codes. Safe export excludes private profiles/config bodies/databases/caches/probe data. Source/index/HEAD ownership passes to root after the scoped commit and final clean-state receipt. Root owns the independent scoped review and any later publication/current-head Qodo/CI/PerfGuard/latest-dev ancestry/merge gates.

Evidence: `task-15-safe-evidence/`; manifest: `task-15-safe-evidence-manifest.json`.

Commit: `9c0aa539712a2e2659a2b355d531228408932a83` — `fix(console): finish bounded chat-start cleanup after registration failures`. Final worktree/index clean; only the five owned source/test paths are committed.
