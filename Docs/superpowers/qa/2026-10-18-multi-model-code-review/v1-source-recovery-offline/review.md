What this change does: Invalid captured-source ranges still fail, but now return bounded same-document retry arguments and a verifier checkpoint hint. The agent loop preserves this feedback, while guidance directs accepted dispositions toward the existing completion flow.

No actionable findings in the uncommitted V1 source-range/completion diff against `2b544776f91a6b972784e9f387785b9a32b34f8d`. Applied the requesting-code-review scope and Ponytail review criteria; assumed the existing single-session, bounded review workload.

Checked the changed source/tool/runtime paths and their callers, source permission and reference validation, `encode` result/input limits, cursor selection, body-free projections, unchanged failure counting, checkpoint construction, accepted-submission replay, completion validation, and report sealing. The new feedback derives only from admitted captured-document metadata; retries re-enter normal authority checks. Neither completion acceptance nor a successful loop alone seals the report.

Independent verification: `python -m pytest -q Tests/Agents/test_code_review_source_recovery.py` using the repository Python environment — **11 passed in 11.68s**. `git diff --check` passed. Pytest emitted cleanup warnings for old temporary `garbage-*` directories after the passing run; no test failed.

Verdict: Ship this scoped offline fix to the existing draft PR. This is not AC24 or live qualification evidence.

Not checked: full suite, live models, real provider-wire/Console continuation, or live budget settlement; the new native/fenced tests exercise the real loop with scripted model turns, and sealing uses injected offline host receipts. No live calls, product children, PR mutation, or merge was performed.
