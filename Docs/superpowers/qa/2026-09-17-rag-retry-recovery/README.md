# Search/RAG failure and retry — TASK-32712

2026-09-17 UTC, `feat/component-pattern-library`, based on `58ed80ffb2`.

After retrieval failed, the query retained focus and Run re-enabled while the
detailed failure remained below the compact viewport. The query region now shows
a brief failure/recovery notice beside Run. Its existing provider disclosure
stays separate, and Evidence retains the detailed recovery text. Retry clears
the notice synchronously with the Run gate; an older conditional status refresh
cannot restore it. Query text, source selection and focus remain intact.

The notice uses the existing token-backed blocked callout. No styles or tokens
changed. ADR required: no; this repairs existing feedback under
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and the Library ownership in
[ADR-003](../../../../backlog/decisions/003-settings-library-rag-defaults.md).

## Automated evidence

- [Red](regression-red.txt): 6 failed / 2 passed before the fix. Both compact
  failure cases lacked visible feedback. All four unavailable cases lacked the
  explicit unavailable notice; wide layouts already showed the detailed Evidence
  recovery. These are distinct assertions, not six instances of compact clipping.
- [Initial green](regression-green.txt): all eight original cases pass.
- [Final targeted gate](targeted-tests.txt): **132 passed in 135.86s**, without
  exclusions. Includes 13 new checks: eight failure/unavailable-to-retry journeys,
  four provider-disclosure paint cases, and a suspended old-refresh race. The
  broader gate covers result focus, query-gate races, gate16 and the local service.
  Counts overlap with the earlier runs and must not be summed.
- New tests and the native runner pass Ruff lint/format. Changed production
  ranges pass formatting. The [baseline comparison](lint-comparison.json) finds
  no added diagnostics: controller 2/2, exports 2/2, panel 5/5. Diff whitespace
  checks pass.
- [Independent review](review.json) found no actionable issue in recovery-text
  escaping, persistent notice updates, disclosure preservation or race coverage.

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_retry_recovery.py \
  Tests/UI/test_library_rag_result_focus.py \
  Tests/UI/test_library_rag_query_gate_race.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/Library/test_library_rag_service.py \
  -q --tb=short --show-capture=no
```

## Native evidence

[Runner](native_check.py), [result](result.json), [lifecycle](lifecycle.json),
[capture hashes](capture-hashes.json). Private profile:
`/private/tmp/tldw-32712-run-001`.

Real TldwCli runs with LinuxDriver, both output streams attached to an owned tmux
terminal, exclusive private storage and a primed terminal-capability probe. A real
Media record has real workspace membership. Initial navigation/query setup is
programmatic; Enter submits each failure and retry, then Tab reaches Evidence.
For each theme/size, the first retrieval deliberately raises a controlled error,
and the second makes the service unavailable. After each failure, the runner
restores real local keyword retrieval and checks the exact Media ID/title.
Source counts and retry results are not injected.

| Theme/size | Failure | Unavailable | Result after retry |
| --- | --- | --- | --- |
| Dark 170×48 | [View](textual-dark-170-failed.svg) | [View](textual-dark-170-unavailable.svg) | [View](textual-dark-170-retried.svg) |
| Dark 80×24 | [View](textual-dark-80-failed.svg) | [View](textual-dark-80-unavailable.svg) | [View](textual-dark-80-retried.svg) |
| Light 170×48 | [View](textual-light-170-failed.svg) | [View](textual-light-170-unavailable.svg) | [View](textual-light-170-retried.svg) |
| Light 80×24 | [View](textual-light-80-failed.svg) | [View](textual-light-80-unavailable.svg) | [View](textual-light-80-retried.svg) |

All eight failure-to-retry journeys pass, including eight real keyword searches.
All twelve captures were rendered and inspected: failure copy wraps readably,
query and Run remain painted, and Tab reaches the successful result. The runner
also checks failure clearance and retained query/scope before navigating away.
The injected exception detail does not appear in the rendered failure UI.

All ten private databases pass read-only SQLite quick_check. The source record is
unchanged and no conversation messages were created. Default config/UI/runtime
fingerprints match. The log has no error, critical or unhandled-exception entries;
normal app stopping is logged, both faulthandler files are empty, app.run returned
and the shell status is zero. The exact PID was absent before closing the owned
terminal. The stored runner hash matches the executed runner; SVG normalization
only removes trailing whitespace, with raw and stored hashes retained.

Limits: controlled retrieval failures and real local keyword retries are covered.
Provider disclosure is checked in the mounted harness; no provider generation or
semantic/remote retrieval is qualified. No full suite, push or merge was performed.
Next bounded review: Search/RAG answer-generation and recovery states, followed by
the remaining feature/component surfaces.
