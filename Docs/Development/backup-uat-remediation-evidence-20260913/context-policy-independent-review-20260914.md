# Context policy config lifetime: independent review

APPROVED. No actionable findings in the controller change and five-case native regression module. Hash receipt: /private/tmp/uat-context-policy-independent-hashes.json.

The only behavioral scope added is the existing synchronous config operation around the nine-read values comprehension. Lazy imports use the installed config source. AST comparison confirms the original key tuple, comprehension and parser-return expression are identical; the parser executes after this operation exits. No await, thread join, UI callback, cache, new policy handling or native guard change is introduced. Existing REBUILD→FILE timed locking, nested selected-source checks, native lease retirement and enclosing-operation restoration remain supplied by operation(config).

Independent author-suite run:5 passed in5.90s (/private/tmp/uat-context-policy-independent.log). Six additional disposable native cases:6 passed in10.87s (/private/tmp/uat-context-policy-probes.log). Exact reusable cases are /private/tmp/test_context_policy_review_probes.py:

- nested_failure: an actual installed config write encounters an injected native os.write error inside the first real getter; original file bytes remain intact, later getters finish, persistence failure remains and real config participant drain refuses.
- preexisting_failure: an earlier failed native write remains sticky after the read boundary and still refuses drain.
- raw_success/raw_parser: the enclosing real config operation remains the same through all nine getters and is restored before parsing; exact parser error survives when caught inside that enclosing scope; no false persistence failure or resource leak.
- core_success/core_parser: the real Notes repository operation is suspended only during the config body and restored before parsing/on error; native note readback still works, final raw/pending states are empty and config drain succeeds.

These probes use the exact controller method with narrow observation wrappers around real getters/parser. They make no backup authority or database-result mock. The native write-failure injection restores os.write in finally. No repository file was edited by reviewer.

This approves the lifetime correction and its tests, not a Windows startup improvement. The profiling artifact supplied by the parent aggregates acquisition counts by caller chain; it is not per-invocation timing, and its separate no-network assertion failed after readiness. Parent owns installed measurement, static/security checks and eventual platform acceptance.

Final promoted scope APPROVED: AST equality confirms all six disposable native child cases and their parameterization/call are identical after only _PROBE→_ENCLOSING and test-name renaming. The final module contains the original five plus these six cases. Runner adds this module only to existing full and support-diagnostic tuples; no new mode/deadline/filter. Product hash remains unchanged. Final test/runner hashes refreshed in the receipt. Parent final11 and installed validations are separate pending evidence; unchanged probes were not redundantly rerun. Parent reports a separate alternating native comparison of9→1 actual acquisitions per call and equal parsed result; that Mac measurement is not a Windows speed claim.
