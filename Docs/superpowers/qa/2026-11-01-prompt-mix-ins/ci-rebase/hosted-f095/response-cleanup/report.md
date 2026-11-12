# UI28 response-rule fixture cleanup investigation

Source baseline: f09597cdd5, with other agents’ disjoint pending CI fixes. Test-only changes: `Tests/UI/response_rules_fixtures.py` and `Tests/UI/test_console_response_rules.py`. No production, timeout, assertion, authority, cancellation policy, census or size-pin changes. Parent owns task38544, packaging and publication. Existing plan covers confirmed CI fixture repairs; no new ADR required.

## Finding and limits

The actual zero-based shard27/32 selects `test_rule_phase_events_project_only_the_mounted_current_session`, parameter order none/rebuild/maintenance/no-controller. It does not select the first registry or Stop tests. The hosted single dot is consistent with none completing and rebuild hanging, but compact output does not identify an exact phase. The original hosted triggering assertion or timeout was **not reproduced**. Exact rebuild passes locally before the fix (1 passed in5.96s).

A separate cleanup defect is confirmed. The rebuild test starts and holds the actual response-rule HTTP helper, then performs readiness/source/Stop assertions before entering the inner try/finally that releases the helper. An early failure reaches mounted_rules_console’s original owner.dispose while that fixture-owned transport hold is still unreleased. Runtime disposal legitimately waits for physical helper retirement.

Private observational fault injection raises a sentinel AssertionError at the real credential-timer assertion boundary after helper entry. Async chain observations at +1.30,+4.31,+7.32s show response_rules.resources.worker → provider gateway → http_transport → helper_release.wait, alongside helper wait_for_cleanup and test context exit. A diagnostic-only callback releases the actual event at8s; cleanup then finishes and reports the exact original sentinel failure (1 expected failed in14.82s). This demonstrates the cleanup defect without claiming the same hosted assertion failed. Probe/run source and complete phase records are retained.

## Correction and verification

The fixture’s existing finally now sets whichever primary/helper release events it owns, then awaits the unchanged owner.dispose. It neither catches nor replaces the original exception. Production cancellation, source fences and retirement remain intact.

New parameterized native controls start actual primary/helper transport through the composer, observe the current task waiting on the real Event, prove it is pending and unreleased, and raise a sentinel assertion before any test release block. On context exit they require the exact same AssertionError, event released, actual task done and zero unsettled helpers. No scheduled rescue release is used. The controls complete under a10s test-local bound (the existing hosted180s timeout is unchanged).

Commands are recorded verbatim with cwd/return code/duration in each JSON artifact. All use the real bootstrap/private profile and an observational-only phase plugin; no authority bypass or synthetic product owner.

- `rebuild`: original exact rebuild, 1 passed in5.96s.
- `injected`: original exact rebuild with explicit private fault injection, 1 expected failed in14.82s; sentinel preserved after8s rescue release.
- `cleanup-controls`: both new primary/helper negative controls, 2 passed in9.55s.
- `gated-controls`: all four original gated parameters, 4 passed in14.64s.

The targeted invocations retain --timeout=180. All tests stopped. No full module or broader sweep. Local Python3.12.11/pytest8.4.2 differs from hosted Linux3.12.15/pytest9.2, explicitly limiting non-reproduction claims.

Both changed files Ruff-clean before/after. Fixture formatting clean. Existing response-rules module formatting debt reproduces at f095; new function is already formatter-identical. AST proof confirms every original test/function unchanged and exactly one added regression function. git diff --check passes. Static hashes identify final reviewed source. An initial Ruff invocation could not write its cache in the managed worktree; --no-cache completed normally. No source changes were made for inherited formatting.

Independent read-only reviewer reports no findings on the settled two-file patch and all evidence. Hosted CI must still retest; if the original early precondition recurs, normal cleanup now lets it surface its real failure instead of hiding it behind a held transport.
