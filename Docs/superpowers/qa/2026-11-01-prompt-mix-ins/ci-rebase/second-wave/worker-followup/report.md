# Final worker-contract callback correction

Source settled. No guard rule or census modification; no full suite, commits or pushes. Three owned files are listed with current hashes in source-inventory.json.

## Cause and correction

W003 follows `self` according to the enclosing class. Moving `_run_agent_reply` to a module helper removed that receiver context: the field `_select_project_instruction_binding` then matched every same-named definition, including the unrelated UI selector that pushes a wait modal. Exact graph comparison against untouched ba1 is in graph-before-fix.json: baseline class-bound field reaches no push; the pre-fix extracted function resolves `defs` by name and creates the two reported rows.

Neither automatic chat-start nor mobile background tasks are claimed to be Textual workers. Mobile constructs the controller without a selector. ConsoleRuntime explicitly overwrites caller-supplied selection with its event-owned decision service, whose attached UI projection uses a nonblocking callback. This explains why the new static edges were not actual calls to that UI method.

The original decorated facade now supplies one explicit nested async dependency by keyword. It reads the current controller field only when invoked at the original recovery point. Missing field value returns `(False, None)`; a present callback returns `(True, await callback(...))`. The helper checks presence before unpacking the result, so malformed custom `None` still raises rather than becoming an absent-selector refusal. The original recovery exception is passed into the dependency; `str(error)` happens after the field presence check exactly as before. No new task is scheduled: the nested await runs in the same task, and absent callback returns without yielding. Generation/remote-session decorators, cancellation handling and source checks are untouched.

## Evidence

- `covering`: 8 passed, 6 failed. Passing controls: two worker-graph tests using the actual current facade/helper AST (absent selector safe; real waiting selector still reported), original agent five and real fork live-binding control.
- The six failures are `test_console_project_setup_refusal.py` native cases at the pre-entry IDLE-vs-BLOCKED assertion. `baseline-recovery` reproduces all six on untouched ba1 (6 failed). This test file was not edited and these failures are not labeled green. Their fixture lacks the bootstrap-profile marker used by existing native agent-owner tests; paired reproduction establishes they precede this change.
- `owned-recovery`: 5 passed in 3.75 seconds. Existing bootstrap-owned project-instruction controls cover disable, cancel, absent selector and durable/ephemeral retry; original assertions and operational owners remain unchanged.
- `worker-guard.log`: whole worker-contract guard passes, no new W001/W002/W003 findings and no census edits. Exact command: shared `.venv/bin/python scripts/check_textual_worker_contract.py` (exit 0).
- `lint.json`: zero added Ruff diagnostics relative to the settled pre-fix controller/helper and HEAD guard test. Scoped git diff --check passed. Controller is 29173 lines; extracted helper 1156. Parent owns final exact size enrollment and cold-import measurement.
- `independent-review.md`: read-only reviewer found no defects in callback lookup timing, tagged result handling, string-conversion timing, task/cancellation semantics or decorator order. No tests were repeated by the reviewer.

Exact targeted argv/cwd/status/logs/XML are retained for each run. `graph-before-fix.json` deliberately describes the pre-fix candidate, not the final source. The earlier core extraction report/source hashes and 40-declaration AST proof describe the candidate before this explicit callback amendment; source-inventory.json and this report qualify the current three-file amendment. No old evidence was silently relabeled as final.
