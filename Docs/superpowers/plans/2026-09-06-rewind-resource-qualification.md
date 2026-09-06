# Rewind resource qualification implementation plan

> **For agentic workers:** Execute inline with verification and independent review checkpoints.

**Goal:** Close the reproduced 209-descriptor leak without changing rewind behavior.
**Architecture:** Explicitly imported test ownership fixture, real constructors and existing controller shutdown / same-file database quiescence. No production or global conftest changes.
**Tech Stack:** pytest-asyncio, SQLite, existing Console lifecycle APIs.
**Spec:** TASK-31923 and the user's remaining resource/process test-review request.

ADR required: no
ADR path: N/A
Reason: Test-only lifecycle repair preserves existing resource owners and interfaces.

## Constraints

- Keep real store/controller/database classes and all existing behavioral assertions.
- No new garbage collection, relaxed FD limits, or foreign database cleanup.
- TASK-31923 activates the fixture in the two importing rewind files. TASK-31814
  separately extends it to ten positively attributed recovery/boundary files.

## TASK-31923: exact test-owned resource cleanup

Files: `Tests/console_resource_fixtures.py`,
`Tests/Chat/test_console_rewind_summarize.py`,
`Tests/integration/test_console_rewind_e2e.py`.

- [x] Reproduce all 78 summary/rewind/parent cases and attribute surviving SQLite handles after existing teardown: 209 aggregate growth.
- [x] Add the explicitly imported fixture that retains real controllers and exact tmp_path ChaChaNotes instances, then assert `database.registered_connection_count() == 0` after `await controller.shutdown()`; demonstrate this fails without database quiescence.
- [x] After every controller has drained, close each owned database with `with database.quiesce_connections(timeout_seconds=2.0): pass`, then retain the zero-connection assertion. Release tracking lists before the existing GC finalizer via fixture dependency.
- [x] For the agent-only payload-ordering test, create `WorkspaceDB(tmp_path / "workspaces.db")` and `AgentRunsDB(tmp_path / "runs.db", client_id="test")`, inject a real `LocalWorkspaceRegistryService` through `workspace_file_roots._registry_factory`, and register their `close` finalizers. Preserve payload ordering assertions.
- [x] Run the three complete files with the native descriptor probe; require no retained test SQLite handles and no aggregate warning. The shutdown-only RED control omits quiescence and fails the zero-handle guard in both selected cases.
- [x] Review-driven failure containment: five focused cases cover successful cleanup and shutdown/quiescence/count faults, individually and together. All controller attempts precede database cleanup, foreign paths remain excluded, and ordinary exceptions are reported together after attempting remaining owners. RED: 1 passed / 4 failed; GREEN: 5 passed.
- [x] Run scoped Ruff/format/diff checks, obtain independent review and update TASK-31923 / checkpoint. Commit the verified bounded repair separately from ongoing inventory work.

## TASK-31814: separately attributed extension

Native post-finalizer probes reproduced 378 retained descriptors in 108 recovery
tests, and 234 in 54 boundary tests. Explicit imports in those ten complete files
reuse the same cleanup without modifying behavior or shared conftest. Combined
final verification includes these 162 cases, the 78 rewind/parent cases and five
cleanup failure-path controls: **245 passed**, no retained SQLite descriptor
lines and no aggregate FD-growth warning. Three dependency warnings remain.
Evidence: `/private/tmp/tldw-owned-console-resources-final.xml`.

Cancellation review refinement: two additional controls failed when CancelledError
escaped ordinary exception handling. Preserve it in BaseExceptionGroup after
attempting all owners; ordinary groups automatically remain ExceptionGroup.
Final full selection: **247 passed in 147.85s**, same three dependency warnings,
no retained SQLite descriptor lines and no aggregate FD-growth warning.
Evidence: `/private/tmp/tldw-owned-console-resources-cancellation-final.xml`.
