# UI26 response-rules lifecycle repair

Fixed the demonstrated retired-composer snapshot failure under TASK-38544 AC27. The optional old-visible input snapshot now tolerates its session already being closed, while reconciliation continues to the new active draft and binding. Current-active snapshots, newer surviving resident drafts, switch-window suffixes and command/source authority remain unchanged. No sleep, timeout, original assertion or admission gate was weakened.

The fault predates this PR: the entire pre-repair draft-sync method AST is identical at dev88af, b110 and5a8. A controlled original resume tree-read hold plus original close reproduced it without profiling. The permanent regression failed directly at that original close/read before repair (**1 failed in 19.00s**), then passed with the repair. It also checks resumed identity/transcript, exact composer binding, and actual Send→`/rules` modal with no model/helper calls or App exception.

To respect the unchanged 6,990-line cap, the unchanged `_consume_visible_agent_handoff` body moved to the existing, already resident `composer_run_controls` owner. The original SessionController method/signature/doc remains as a lazy facade. Independent precheck approved this boundary; exact executable AST equality preserves all incarnation, handoff revision, generation, edit-serial, segment and commit checks/order. Session is now **6,973** lines; receiving owner **1,567**. No existing pin was raised and no documentation/source was compressed to fit.

## Verification on frozen bytes

- **13 passed in 125.26s** under normal bootstrap: original four response-rule E2E cases, permanent close/resume/actual-command regression, three saved-projection newer-resident/suffix routes, two original exact handoff controls and three received-input authority controls. `repair-green.log` and the exact 13 node IDs in `commands-and-results.json` are authoritative. No in-memory observer, overlay, private marker or assertion change was used for this GREEN.
- The exact original session ceiling and anti-slack nodes passed. The same already-started static run then stopped at the unrelated historical Wave6 ChatScreen ≤ 22,172 assertion (**2 passed, 1 failed in 2.18s**). Its first predicate and historical test-function AST are unchanged at dev88af/b110/head5a8; those screen lengths are 24,652 / 24,675 / 24,590. `historical-wave6-baseline-proof.json` is read-only source/predicate evidence, not a rerun claim. No historical guard was edited/exempted and later selected tests were not reached.
- Whole new test-file formatting, changed source-range formatting, and owned diff whitespace pass. Ruff totals remain 54: session 51→50, receiving owner 3→4, test 0→0. One unchanged SIM102 warning moved with the exact copied function; per-file relocation is explicit in `lint-comparison.json`. No blanket clean-lint claim or suppression.
- `repair-ast-proof.json` proves moved executable-body equality, original facade signature/doc equality and original parametrized E2E function AST equality. All three owned hashes still match `frozen-source-proof.json` after GREEN; see `post-green-source-validation.json`.

## Diagnosis qualifications

Original hosted evidence is `/private/tmp/mixins-5a8-ui26.log`. Learning and repair succeeded before the final modal wait. Only two Send-handler entries appear in that log, so the final click did not reach command parsing; the hosted traceback alone does not timestamp the first App failure.

A private b110 source overlay plus whole-interpreter metadata profiling reproduced the same modal timeout and KeyError. Its timeline records first WorkerFailed handling at 37.6397, before the `/rules` click at 59.4627, establishing that the retired-owner error caused lost command delivery in that schedule. **The run took 105.45s and has substantial profiling overhead.** It establishes causal order, not normal-speed failure rate or the hosted exception timestamp. Later DuplicateIds reports followed the first fatal KeyError and are not separate findings. A subsequent no-profiler controlled original-owner probe confirmed the retired read directly in 7.70s. Both source/observer scripts and failed/successful observations are preserved.

The earlier exact normal node passed in 12.47s with possible overlap from a previously started UI31 run; the wrapper-observed node passed in 11.75s but patched callable identities. Those are qualified observations, not stronger source-authority or performance evidence. See `diagnosis.md` for details. No arbitrary readiness wait was added to the original E2E journey.

## Handoff

Owned source: `UI/Console_Modules/session.py`, `UI/Console_Modules/composer_run_controls.py`, and `Tests/UI/test_response_rules_end_to_end.py`. Source is frozen and all owned test processes have stopped. The native slot was released directly to the UI31 owner. No further tests/source edits are planned by this agent.

Exact private artifact paths/hashes are in `evidence-inventory.json`. Root owns task/census/size metadata, consolidated guards, startup/performance qualification, independent acceptance, publication and merge. No full suite, startup/performance run, live provider inference or PTY tour was performed here. This is ready for scoped independent re-review, not a current-CI or merge-completion claim.
