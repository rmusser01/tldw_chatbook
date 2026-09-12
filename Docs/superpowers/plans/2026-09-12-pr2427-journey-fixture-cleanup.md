# PR 2427 Notes journey fixture cleanup implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development with
> separate spec and quality review before root's complete-file qualification.

**Goal:** Close the journey fixtures' owned worker-thread SQLite connections
and preserve meaningful narrow-screen authority coverage.

**Architecture:** Reuse the existing same-file quiescence boundary only for
private test-owned databases after their enclosing UI/runtime lifetime exits.
Two local standard-library context managers express those lifetimes; no new
production helper, global fixture, connection policy or scheduling behavior.

**Tech Stack:** Python contextlib, real SQLite, pytest, Textual Pilot, native FD
observation.

ADR required: no.
ADR path: N/A; existing TASK-32360 presentation and database quiescence contract.
Reason: test-oracle reconciliation and exact fixture-owned resource cleanup,
without changing production storage, service, UI or ownership boundaries.

## Scope and evidence

- Worktree: `.worktrees/pr2427-review-recovery`, branch
  `codex/dev-test-review-20260904`, published checkpoint `550979d01c`.
- TASK-31932 remains In Progress. Root owns docs, Git and integrated runs.
- Implementation owner changes only
  `Tests/UI/test_library_notes_files_sync_journey.py`.
- Preserve the unrelated untracked reader-paydown plan without opening it.
- The complete Shell run is frozen at
  `/private/tmp/pr2427-resize-shell-complete.CAd9yu`. Do not edit its test,
  shared imports, runtime modules or fixtures. Tracked Python search found no
  imports of the journey module; its only references are self/helper strings.
- Fresh complete journey baseline:
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-journey-resource-current.wuEgry1q9X`:
  29 passed / 1 failed in 51.25s; final 82 SQLite handles, zero instance locks.
  Forty handles belong to the import-receipt test's private notes.sqlite;
  forty-two to the four conflict-choice tests' private notes.sqlite3 files.
- The one behavioral failure is the 60x20 authority-prefix oracle. TASK-32360
  explicitly removes that duplicate prefix below 64 columns because the
  selected source-strip button already names Library notes.
- Read `backlog/docs/lessons-testing-evidence.md`, TASK-32360, and the existing
  `_import_ready_host` plus its setup/body/cancel fault controls in
  `Tests/UI/test_library_note_import_flow.py` before implementation.

## Task 1: Reconcile the one authority oracle

- [x] In `test_database_notes_import_once_journey_is_painted_focused_and_retained`,
  retain the authority Static prefix assertion at the wide size. Below 64,
  instead pin `#library-notes-source-database` as a displayed, selected Button
  labelled exactly `Library notes`, with positive region fully contained in
  both its source strip's content region and the screen. Require the narrow
  Static not to start with the duplicate `Library notes · ` prefix.
- [x] Preserve every purpose, painted-content, retained-canvas/work-pane,
  focus, chooser containment, exactly-once selection and dialog assertion.
  No runtime or layout change. Run both original size parameters after the
  correction and retain the fresh baseline RED above.

## Task 2: Express the exact fixture lifetimes and prove closure

- [x] Add actual-helper tests with real worker-thread connections retained
  after joining a one-worker `ThreadPoolExecutor`. Keep one separate
  foreign-path database connection alive throughout each control. Assert the
  exact owned registration count becomes zero, the retained owned connection
  raises `sqlite3.ProgrammingError` when used after teardown, and foreign
  `SELECT 1` still succeeds. Close only those test-created authorities afterward.
- [x] Use one local sync context manager `_real_notes_authority` for creation
  of database, folders, optional interop and scope service. It yields the
  existing objects, not a new state owner. Test partial setup failure after
  database creation, body failure and delivered `CancelledError`, preserving
  the original raised object when cleanup succeeds.
- [x] Replace `_start_real_conflict_stack` and `_close_real_conflict_stack`
  with one `_real_conflict_stack` async context manager. Nest the authority
  context; construct the existing runtime/controller; protect `owner.start()`
  and the yielded body with `try/finally` calling existing `owner.shutdown()`.
  Both current conflict journeys use `async with`. Keep all journey assertions
  and the fresh-owner restart/history/Undo sequence unchanged.
- [x] Route the import-receipt test through the sync authority context before
  folder/service/app setup, then run its unchanged production-CSS harness and
  cancel/retry/durable-receipt body. Remove only its obsolete manual close.
- [x] If introducing the context helpers prevents directly applying controls
  to the old names, first extract the existing weak current-thread cleanup
  into the new helper shape. Run the real worker-handle controls RED there
  before adding the quiescence fix. A missing helper/name error is not the
  required RED evidence. Record the extraction as a mechanical plan step and
  compare original journey assertions/operations independently of indentation.
- [x] Final authority cleanup follows the established pattern:

  ```python
  finally:
      try:
          if interop is not None:
              interop.close_all_user_connections()
      finally:
          with database.quiesce_connections(timeout_seconds=2.0):
              pass
          assert database.registered_connection_count() == 0
  ```

  Protect setup from the database allocation onward. The barrier is only for
  that fixture's disposable private database after owned work is done; never
  enumerate all database instances, force GC, raise limits or close a foreign
  path. Leave the seed helper unchanged: its measured current-thread lifetime
  creates no worker handle.
- [x] Add actual async-helper start/body cancellation and post-shutdown failure
  controls; inject a shutdown error after the real shutdown has completed so
  the test does not pretend a still-running owner is quiescent. Inject an
  interop-close error and prove remaining exact database cleanup still runs.
  Do not swallow either cleanup failure or replace it with a success result.
  Preserve normal setup/body exceptions when cleanup succeeds; do not invent
  an exception aggregation policy for simultaneous unrelated failures.
- [x] Run the new controls RED then GREEN, all four original conflict choices,
  original import receipt and both authority viewport cases. Keep cleanup
  evidence distinct from the behavioral assertions. No production changes.

## Task 3: Review, qualify and save

- [x] Independent spec review then quality review of the actual diff. Fix
  demonstrated issues only; verify unchanged journey assertions and logical
  operation ordering despite context-manager indentation changes.
- [x] Freeze the file, then root runs the complete journey file with the
  unchanged native observer and a fresh per-user temporary directory:

  ```text
  PYTHONPATH=. .venv/bin/python
  /private/tmp/pr2427-fd-identity.OTL9up/native_fd_identity.py
  <fresh-existing-report-directory>
  Tests/UI/test_library_notes_files_sync_journey.py
  ```

  Require all cases passing and zero final SQLite/instance-lock handles.
  Compare process-lifetime log/event-loop descriptors separately; the observer
  adds no cleanup, forced collection or application patches.
- [x] Run fatal scoped Ruff and whitespace checks. Existing shared runtime
  and fixtures must remain byte-identical; do not rerun the already qualified
  four Notes files unless their dependency source changed.
- [ ] Root records terminal results in TASK-31932 and this plan, commits only
  reviewed explicit paths and publishes to the existing PR. The ongoing Shell
  result, size/preload/CSS/older Console resource gates, latest-dev integration
  and final-head review/CI remain separate. No whole-PR success or merge claim.

## Implementation notes and focused evidence

The approved single-file implementation uses nested standard-library authority
and runtime contexts. Setup, body and shutdown unwind through the existing
private-file quiescence boundary; no shared fixture or production source
changed. The wide authority prefix remains explicit and the narrow case pins
the selected source button's actual displayed geometry.

Evidence under `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/`:

- `pr2427-journey-oracle-red.XROebf/pytest.log`: one pass / one intended
  narrow-prefix failure; corrected original viewport pair passes 2/2.
- `pr2427-journey-mechanical-green.kauWKv/pytest.log`: original import and
  four conflict journeys pass 5/5 after weak-helper mechanical extraction.
- `pr2427-journey-helper-red.S0SJQZ/pytest.log`: seven actual-helper failures
  establish retained worker registrations or an owner still active after the
  injected start failure. Missing helper/name errors are not counted.
- `pr2427-journey-focused-native.QI7yWa/pytest.log`: final focused 14/14 pass
  in 20.33 seconds; terminal six descriptors, zero SQLite/instance locks.

Independent spec review confirms unchanged seed-helper AST, all 11 import
assertions and the full mounted body, plus all 30 conflict assertions and both
runtime bodies in their original order. Frozen file SHA256 is
`6a9f042c165e848810846f34510a6bafc8756f5fbca69e3e3acb083239190288`.
Full scoped Ruff and whitespace pass. The formatter identifies one untouched
pre-existing location, not a clean whole-file format claim. Independent final
quality review approves without findings.

Root's fresh complete-file native run is terminal at
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-journey-complete-final.2z5ipn0Rdx`:
**37 passed, three dependency warnings, 56.47 seconds**, exit zero. Final
inventory contains seven descriptors and zero SQLite/instance-lock handles.
Frozen SHA256 remains identical. This supersedes the journey's old 76/82/86
retained-handle findings, not the separate Console fixture leaks. The existing
testing lesson on exact-file worker cleanup already records this general trap;
no duplicate lesson or new ADR is needed.

The complete Shell run remains pending with six reported failures; its source
is unchanged by this isolated journey test-module repair. Current-head Console
diagnosis separately passes three cases but retains 25 SQLite handles. Those
findings, size/preload/CSS/integration and final PR gates remain open.
