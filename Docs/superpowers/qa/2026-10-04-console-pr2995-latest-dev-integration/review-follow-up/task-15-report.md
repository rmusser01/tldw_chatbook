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


# Task 15 fix round 1 — Important I1

## Ruling, source and change

This appended section supersedes the original report's retained-limit choice. Root ruling 67 and the amended Task 15 brief require same-store/current-owner denial when ordinary registration and durable settlement both fail. BASE is `41155ce71ed84e6cb5d9c720a6d1a2f736c35e53` (metadata-only after previously reviewed source `9c0aa539712a2e2659a2b355d531228408932a83`). No original report text or evidence was rewritten.

ADR required: no new ADR.
ADR path: `backlog/decisions/219-console-chat-destinations-and-bounded-starts.md`.
Reason: restore its existing charged-and-paused uncertainty contract with the same canonical identity, registry and recovery snapshot tokens.

`AutomaticWorkLedger.__init__` now captures the existing resolved file-path identity before that ledger can authorize work. Memory stores retain their shared per-DB private UUID; separate memory DBs remain isolated. Initial resolution failure raises before a ledger is returned. The real `start()` control proves that this occurs before capacity claiming, active-item insertion, attempt/reservation writes or draft consumption.

The identity getter now returns the captured value. Existing clear, lookup and recovery bodies reuse it without filesystem observation under the restriction lock. Ordinary registration delegates to a small `_retain_chat_start_restriction` method that uses only the captured identity, existing registry/lock, owner, attempt, chain and a fresh original-shape snapshot token. If the ordinary registration seam raises, coordinator cleanup invokes that same pure insertion directly. It performs no SQL or filesystem operation. Type-only fallback logging retains the existing physical-cleanup protection.

Only four owned files change in this round: the coordinator, ledger and two append-only test files. The fork file remains byte-exact to the reviewed source. No second registry, lexical fallback key, wildcard narrowing, schema, authority floor, opening retry, dependency, timer, worker topology or public abort change was introduced.

## RED and setup attribution

The first `red` run expanded to 11 cases: 10 failures and one existing memory-isolation pass. The initial-identity case failed correctly because construction did not refuse. Nine other cases stopped at setup: the private-path guard rejects symlink directories. The tests were corrected to the repository's existing `directory/../database` alias pattern; the guard was not changed.

`red-corrected` retained nine failures. Its global Path.resolve injection also affected unrelated Backup_Recovery admission and prevented subsequent observations. This is preserved as an injection-scope issue, not final I1 evidence. The fault was narrowed to Path.resolve calls from DB.automatic_work, leaving native workers, real SQL and the healthy alias-handle admission path observable.

`red-focused` then produced the intended 10 failures before any production change: eight native cases reached physical thread exit, retained an accepted row and one charged generation, encountered a real SQLite trigger rejection of review settlement, and failed because the healthy alias handle did NOT refuse sibling admission. They cover commit/provider owners, persistent ordinary-registration failure versus a controlled OSError at the real Path.resolve boundary, and original versus replacement capacity/active ownership. File cleanup failed at canonical resolution, and the real start-boundary initial-identity case failed because start proceeded rather than refusing. The direct-constructor initial-identity RED and memory pass remain in the first receipt.

The Path.resolve failure is injected at that concrete operation; it is not claimed to reproduce a particular OS outage. Durable settlement rejection is a real SQLite trigger rollback. Provider cases retain the earlier Task 15 explicit adversarial loss of the local receipt flag after native dual-store acceptance/provider entry; they do not claim that ordinary dispatch can precede durable fences.

## Final verification

The only GREEN run after implementation passed **27/27** cases in **39.78s pytest**: 12 new controls and 15 parameter-expanded cases from exactly the four existing selectors in `task-15-fix1-brief.md`. No new skips, xfails or pytest warnings occurred. No full 61/31-case cohort or 35-loading cohort was replayed. The source hashes recorded before GREEN exactly match the final source; there were no post-GREEN source edits.

The new controls prove repeated cancellation retains physical ownership until commit/provider exit, cleanup resolves its outcome and releases only the captured token, replacement active/token state survives, the accepted charge remains, the healthy alias handle refuses uncertain work after exit, and unrelated roots remain admitted. Additional controls prove same-store alias equivalence, memory UUID sharing/isolation, same-owner retention, replacement-owner recovery and initial real-start refusal with zero attempts/reservations/claims and the saved draft intact. The four existing selectors retain recovery token fencing and all original assertions/waits/markers.

Fatal Ruff (`E9,F63,F7,F82`), all-five-file Ruff formatting and whitespace checks passed. These owners have zero baseline formatter debt; formatting preserved AST and both entire prior test files remain exact prefixes. No warning suppression, installs, timeout increase, budget change or user-profile change occurred. Shared Python 3.12, worktree PYTHONPATH, canonical private profiles and the existing 300s pytest timeout were retained.

## Exact source and evidence carry

`source-AST-import-authority-custody-map.json` explicitly names the authorized exceptions: ledger constructor, identity getter, ordinary registrar and added pure insertion helper; coordinator `_restrict_cleanup`. All other ledger methods remain AST-exact, including the private absence predicate and the public abort/transaction/recovery/read/clear/lookup/settlement bodies. Public abort and recovery also have byte-exact body checks. `_run`, both physical drains, outcome fallback, captured-token release, all admission/currentness/receipt/preparation methods, imports and fork remain exact. The fallback's sole call inside its insertion body is `object()` for the unchanged recovery snapshot-token shape.

All unowned tracked source and QA paths remain exact to BASE. This carries Task 14, eager-loading/import, worker, route, native/Close and unchanged cap/profile guards without asserting that the newly changed constructor is unchanged. The original qualification ZIP retains its exact 63,166,118 bytes and SHA256 `acecbdfb6ddbafe6df679c39f9f137f2bfb49a1f87b62556e1345021f8384c84`, preserving all 2118 original entry bytes without export/decompression/replay. Historical 35-loading evidence and concerns remain unchanged and historical; the original Task 15 61-case receipts are not relabeled as this source.

The original 69-file package is preserved as **68 unchanged child files plus the exact original report at `task-15-report-before-I1.md`**. The original manifest and brief are retained byte-for-byte at their before-I1 paths. The current report preserves that exact report prefix and appends this fix. The separate R1 manifest uses the same `{relative_path: {sha256, bytes}}` files mapping. The current publication manifest transparently maps the original snapshot, untouched child rows, R1 evidence and appended current report. No private profiles, config bodies, databases, caches or probe data are exported.

Self-review found no remaining I1 correctness concern. All owned pytest/native worker processes have exited; all command receipts carry exit codes and native exits are asserted in GREEN. Root owns the single scoped I1/new-breakage re-review and external publication/Qodo/CI/PerfGuard/latest-dev ancestry/merge gates.

## Exact covering commands and outputs

### red

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task15-fix1/red Tests/DB/test_automatic_chat_starts.py::test_restriction_identity_is_captured_for_aliases_and_memory_peers Tests/DB/test_automatic_chat_starts.py::test_initial_restriction_identity_failure_refuses_automatic_authority Tests/Chat/test_console_chat_start.py::test_persistent_registration_and_settlement_failure_retains_native_denial --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-15-fix1-safe-evidence/red.xml
```

Exit 1; 17.497 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/red.log`; exact source hashes: `red-argv.json`.

### red-corrected

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task15-fix1/red-corrected 'Tests/DB/test_automatic_chat_starts.py::test_restriction_identity_is_captured_for_aliases_and_memory_peers[False]' Tests/Chat/test_console_chat_start.py::test_persistent_registration_and_settlement_failure_retains_native_denial --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-15-fix1-safe-evidence/red-corrected.xml
```

Exit 1; 27.474 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/red-corrected.log`; exact source hashes: `red-corrected-argv.json`.

### red-focused

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task15-fix1/red-focused 'Tests/DB/test_automatic_chat_starts.py::test_restriction_identity_is_captured_for_aliases_and_memory_peers[False]' Tests/Chat/test_console_chat_start.py::test_persistent_registration_and_settlement_failure_retains_native_denial Tests/Chat/test_console_chat_start.py::test_initial_identity_failure_precedes_native_start_authority --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-15-fix1-safe-evidence/red-focused.xml
```

Exit 1; 21.352 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/red-focused.log`; exact source hashes: `red-focused-argv.json`.

### green

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task15-fix1/green Tests/DB/test_automatic_chat_starts.py::test_restriction_identity_is_captured_for_aliases_and_memory_peers Tests/DB/test_automatic_chat_starts.py::test_initial_restriction_identity_failure_refuses_automatic_authority Tests/Chat/test_console_chat_start.py::test_persistent_registration_and_settlement_failure_retains_native_denial Tests/Chat/test_console_chat_start.py::test_initial_identity_failure_precedes_native_start_authority Tests/Chat/test_console_chat_start.py::test_restriction_registration_failure_still_drains_exact_native_owners Tests/Chat/test_console_chat_start.py::test_failed_review_settlement_blocks_siblings_across_ledger_handles Tests/DB/test_automatic_chat_starts.py::test_transient_uncertainty_is_store_scoped_and_stale_owner_cannot_poison_replacement Tests/DB/test_automatic_chat_starts.py::test_late_recovery_cleanup_preserves_new_owner_uncertainty --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-15-fix1-safe-evidence/green.xml
```

Exit 0; 43.191 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/green.log`; exact source hashes: `green-argv.json`.

### fatal-ruff

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/Chat/console_chat_start.py tldw_chatbook/DB/automatic_work.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_start.py Tests/DB/test_automatic_chat_starts.py
```

Exit 0; 0.103 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/fatal-ruff.log`; exact source hashes: `fatal-ruff-argv.json`.

### format-check

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Chat/console_chat_start.py tldw_chatbook/DB/automatic_work.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_start.py Tests/DB/test_automatic_chat_starts.py
```

Exit 0; 0.091 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/format-check.log`; exact source hashes: `format-check-argv.json`.

### whitespace

```sh
PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook git diff --check 41155ce71ed84e6cb5d9c720a6d1a2f736c35e53
```

Exit 0; 0.072 seconds in the owning process. Full output: `task-15-fix1-safe-evidence/whitespace.log`; exact source hashes: `whitespace-argv.json`.

R1 source commit: `eb7cb871390616de22d28c7a6679bbc9202841d7` — `fix(console): retain uncertain-start denial with captured store identity`. Clean source/index/worktree handoff; only the four owned source/test paths above are committed.
