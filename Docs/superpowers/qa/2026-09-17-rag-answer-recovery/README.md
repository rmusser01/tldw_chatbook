# RAG answer failure recovery — TASK-32713

2026-09-17 UTC, `feat/component-pattern-library`, based on `664eb24996`.

Answer generation correctly settled failures and re-enabled Run, but compact
layouts kept the focused query visible while its failure/retry text remained
below the viewport. The existing persistent notice beside Run now also reports
answer failure. Provider disclosure remains separate, and the Answer region
retains its detailed error and retry hint. The existing Run gate clears the
notice while busy; a successful retry removes the failure.

The change is one additional branch in the panel's notice helper. It introduces
no new state, widgets or styling. ADR required: no; existing
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-003](../../../../backlog/decisions/003-settings-library-rag-defaults.md)
apply. Retrieval and provider contracts are unchanged.

## Automated evidence

- [Red](regression-red.txt): 2 failed / 2 passed before the fix. Both 80×24
  themes lacked painted failure feedback; both wide layouts already showed the
  detailed Answer failure.
- [Expanded green](regression-green.txt): eight cases pass, covering exception
  and empty-answer failures in both themes at 170×48 and 80×24. Each case submits
  through Enter, retains query/scope/focus and retrieval evidence, checks painted
  notice/disclosure/Run, holds the retry provider to inspect busy clearance, then
  releases a cited reply through the real answer service and citation validator.
- [Final targeted gate](targeted-tests.txt): **326 passed in 146.31s**, without
  exclusions. Includes the eight new cases, retrieval-retry regressions, gate16,
  the answer service and RAG state tests. Counts overlap with earlier runs.
- New tests and native runner pass Ruff lint/format. The changed production
  range passes formatting; [baseline comparison](lint-comparison.json) shows the
  same five existing lint diagnostics and no additions. Whitespace checks pass.
- [Independent review](review.json) found no actionable issue in mode/busy
  gating, retrieval precedence, stale lifecycle, privacy or regression coverage.

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_answer_retry.py \
  Tests/UI/test_library_rag_retry_recovery.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/Library/test_library_rag_answer_service.py \
  Tests/Library/test_library_rag_state.py \
  -q --tb=short --show-capture=no
```

## Native evidence

[Runner](native_check.py), [result](result.json), [lifecycle](lifecycle.json),
[capture hashes](capture-hashes.json). Final profile:
`/private/tmp/tldw-32713-run-003`.

Real TldwCli uses LinuxDriver and an owned tmux terminal with both output streams
attached. Its private profile has exclusive ownership and a primed terminal
probe. A real Media record has actual workspace membership. The runner adapts
RAG requests to real local keyword retrieval and substitutes only the synchronous
provider chat seam, with dummy readiness credentials. Each cell exercises a
raised exception and an empty reply, each followed by a held then successful
cited reply. The real answer service builds prompts and validates citations.
The exact Media ID/title is checked; source counts and result rows are not
injected. Sixteen keyword searches and sixteen controlled chat calls complete.

| Theme/size | Exception | Empty answer | Answer after retry |
| --- | --- | --- | --- |
| Dark 170×48 | [View](textual-dark-170-exception.svg) | [View](textual-dark-170-empty.svg) | [View](textual-dark-170-retried.svg) |
| Dark 80×24 | [View](textual-dark-80-exception.svg) | [View](textual-dark-80-empty.svg) | [View](textual-dark-80-retried.svg) |
| Light 170×48 | [View](textual-light-170-exception.svg) | [View](textual-light-170-empty.svg) | [View](textual-light-170-retried.svg) |
| Light 80×24 | [View](textual-light-80-exception.svg) | [View](textual-light-80-empty.svg) | [View](textual-light-80-retried.svg) |

All eight failure-to-retry journeys pass. All twelve captures were rendered and
inspected: failure copy, provider disclosure and Run remain readable; the new
answer and neutral citation note render after success. Query/scope/focus retention
is checked before the runner explicitly scrolls the answer into view for capture.
That inspection scroll does not qualify keyboard navigation to the answer.

All ten private databases pass read-only quick_check, the source is unchanged,
and no conversation messages were created. Default config/UI/runtime fingerprints
match across all three attempts. The final log has no error, critical or
unhandled-exception entries; expected controlled failure warnings remain.
Normal stopping is logged, faulthandler is empty, app.run returned and shell status
is zero. The exact PID was absent before the owned terminal was closed. The
executed runner hash matches; SVG normalization only removes trailing whitespace.

[Attempt 001](run-001.json) failed a runner assertion that expected full Media
body text in keyword evidence. Keyword Media snippets intentionally contain a
source label; the corrected check requires the actual source title in the prompt.
[Attempt 002](run-002.json) passed four dark-theme journeys, then timed out because
the inspection scroll left an already-focused query off screen before the next
theme. The runner now restores that inspection scroll before advancing. Neither
attempt changed production code; both quit normally with status 1 and their exact
PIDs were verified absent before terminal cleanup.

Limits: provider behavior, semantic retrieval and answer factual grounding are
not qualified by controlled replies over keyword source labels. Citation
validation establishes reference resolution here. Initial navigation/setup and
answer-inspection scrolling are programmatic; Enter drives failure and retry.
No full suite, push or merge was performed. Next bounded review: keyboard access
to generated answers and citation warnings, then remaining feature surfaces.
