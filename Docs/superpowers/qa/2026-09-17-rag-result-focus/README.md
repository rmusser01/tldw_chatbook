# Search result focus — TASK-32751

2026-09-17 UTC, `feat/component-pattern-library`, based on `bddafc724d`.

The run-start and result-arrival callbacks unconditionally scrolled Evidence to
the top of the Search panel. Focus stayed on the query input, which ended up at
y=-19 outside the compact viewport. A newer focus choice during retrieval could
also be scrolled away.

The existing reveal callback now reads live focus after refresh. If a panel
control owns focus, it keeps that control visible; otherwise it reveals Evidence
as before. It neither transfers focus nor stores a submit-time focus target.
Query text and the existing Tab route to Evidence remain intact. At compact
sizes, preserving the query means results remain below the fold until the user
navigates to them. At wide size, both query and result can fit in the viewport.

ADR required: no. This repairs existing focus behavior under
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and the unchanged Library ownership in
[ADR-003](../../../../backlog/decisions/003-settings-library-rag-defaults.md).
No tokens, styles, persistence, retrieval semantics or provider contracts changed.

## Automated evidence

- [Red](regression-red.txt): 7 failed / 1 passed before the fix. Failures show
  focused query/mode controls absent from the compositor at both sizes/themes.
  One wide mode-control case happened to remain visible; it is not counted as a
  reproduction. An earlier test arrangement used a nonexistent top-k widget;
  that selector was corrected before the retained red run.
- [Initial green](regression-green.txt): all eight original cases pass.
- [Expanded focus checks](focus-tests.txt): 20 pass, covering query and newer
  in-panel focus, ready/empty outcomes, and no-focus/foreign-focus fallback.
- [Final targeted gate](final-targeted-tests.txt): **118 passed in 138.71s**.
  Includes the expanded focus file, scope-recovery and query-gate regressions,
  the Search/RAG gate16 file, and the existing run-reveals-Evidence test. No
  exclusions. These runs overlap; counts must not be summed.
- The first [broader run](targeted-tests.txt) sampled one naturally scheduled
  focus animation before final paint (1 failed / 113 passed). The tests now wait
  for scheduled animations and a fresh frame after worker completion; they do
  not force scrolling. Production did not change for that adjustment.
- New tests and native runner pass Ruff lint/format. Changed production ranges
  pass formatting; [baseline comparison](lint-comparison.json) finds no added
  controller diagnostics (2 before and after). Diff whitespace checks pass.
- [Independent review](review.json) requested explicit no/foreign-focus fallback
  coverage. Those cases were added; final review has no remaining findings.

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_result_focus.py \
  Tests/UI/test_library_rag_scope_recovery.py \
  Tests/UI/test_library_rag_query_gate_race.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_library_honesty_accessibility.py::test_run_reveals_the_evidence_region_instead_of_leaving_the_fold_intact \
  -q --tb=short --show-capture=no
```

## Native evidence

[Runner](native_check.py), [result](result.json), [lifecycle](lifecycle.json),
[capture hashes](capture-hashes.json). Final profile:
`/private/tmp/tldw-32707-run-003`.

Real TldwCli uses LinuxDriver with both output streams attached to an owned tmux
terminal, an exclusive private profile and a primed terminal-capability probe.
A gate delays the real local keyword search without altering its inputs/results.
The source is a real persistent Media record with real workspace membership;
source snapshots are not injected. Initial navigation/focus/query setup is
programmatic; Enter submits, Shift+Tab makes the newer focus choice, and Tab
reaches Evidence. Each of four theme/size cells runs both focus cases and verifies
the exact Media ID/title, for eight actual keyword searches.

| Theme/size | Query retained | New keyboard choice | Evidence reached by Tab |
| --- | --- | --- | --- |
| Dark 170×48 | [View](textual-dark-170-query-focus.svg) | [View](textual-dark-170-newer-focus.svg) | [View](textual-dark-170-evidence-focus.svg) |
| Dark 80×24 | [View](textual-dark-80-query-focus.svg) | [View](textual-dark-80-newer-focus.svg) | [View](textual-dark-80-evidence-focus.svg) |
| Light 170×48 | [View](textual-light-170-query-focus.svg) | [View](textual-light-170-newer-focus.svg) | [View](textual-light-170-evidence-focus.svg) |
| Light 80×24 | [View](textual-light-80-query-focus.svg) | [View](textual-light-80-newer-focus.svg) | [View](textual-light-80-evidence-focus.svg) |

All twelve final captures were rendered and inspected. Focused query/mode controls
are fully painted, and Evidence titles/actions and their focus cue remain readable.
All ten private databases pass read-only SQLite quick_check using the app's Python;
the seeded source is unchanged and no conversation messages were created. Default
config/UI/runtime fingerprints match across all attempts. The final log has no
error, critical or unhandled-exception entries; the normal app-stopping event is
present, faulthandler is empty, app.run returned and shell status is zero. The exact
PID was absent before the owned terminal was closed.

[Run-001](run-001.json) retrieved a result but stalled on a runner-only blanket
screen-worker wait, which includes long-lived app jobs. It was quit normally,
returned status 1, and its PID was verified absent. This repeats the documented
[worker-wait lesson](../../../../backlog/docs/lessons-live-verification.md#screen-worker-waits-can-include-unrelated-app-jobs).
[Run-002](run-002.json) passed after waiting for retrieval state and panel refresh;
run-003 confirms the same matrix with the final lint-clean runner. Each attempt
used a separate profile and terminal; all were closed.

Limits: local keyword retrieval and mounted ready/empty timing are covered. This
is not qualification of semantic retrieval, provider generation, remote services,
or every error/retry path. No full suite, push or merge was performed. The next
bounded review is Search/RAG failure and retry behavior.
