# Recent-search replay and clearing — TASK-32716

2026-09-17 UTC, `feat/component-pattern-library`, based on `cfa89372b5`.

Keyboard replay ran the chosen query with the current mode/sources, but the rail
input retained the previously typed query. The replay handler writes query state
before the canvas `Input.Changed` event; that event's equality guard consequently
skips sibling synchronization. The handler now uses the existing sibling-input
patch helper, which suppresses a redundant Changed event.

After that fix, both wide RAG cases exposed a second defect: answer growth pushed
the focused Recent searches heading below the viewport. Answer arrival now
schedules the panel's existing current-focus reveal after layout. It checks live
attachment and focus ancestry and never restores a captured focus choice. History
clearing already returned focus to its retained disclosure heading and needed no
production change.

ADR required: no. These are routine repairs under existing
[ADR-031](../../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md).
No mode/source persistence, bindings, tokens or service contracts change.

## Automated evidence

Trailing whitespace in pytest diagnostic lines is trimmed in stored text logs.

- [Replay red](replay-red.txt): eight cases fail on the stale rail input; four
  keyboard-clear cases pass.
- [Answer-focus red](answer-focus-red.txt): after synchronizing the rail, fourteen
  pass and both wide RAG cases fail on an offscreen focused history heading.
- [Callback red](callback-red.txt): both delayed-reveal cases fail before wiring
  the callback. The final cases hold only that callback, leaving framework focus
  scrolling active, then choose a newer query or rail focus.
- [Main targeted run](targeted-tests.txt): **135 passed in 255.94s**. The eighteen
  new cases cover both themes and
  170×48/80×24, producer-created history, literal bracketed query labels, current
  mode/scope, exact service call counts, visible focus, and clear preserving an
  existing answer/results. Related runs cover query return, keystroke retention,
  result focus, answer failure/retry and Search/RAG integration.
- [Existing history checks](existing-history-tests.txt): **13 passed in 13.11s**;
  persistence fallback/precedence, record/replay/clear, bracketed labels, manual
  disclosure retention and bounded/deduplicated history all pass. Combined
  targeted total: **148 passed**, without exclusions.
- [Lint comparison](lint-comparison.json): two existing production diagnostics,
  no additions. New test/runner lint and format checks and changed production
  range formatting pass.
- Independent production/test review found no actionable findings in either
  repair or the current-focus race coverage.

## Native evidence

The [result](result.json) passes all eight journeys, with 24 real keyword searches
and four controlled answers. The [runner](native_check.py) boots real TldwCli in a private tmux
terminal/profile and seeds a real Media record plus workspace membership. Every
cell seeds history with two keyword searches and then makes a third search by
replaying the older query under the intended current mode. Tab/Enter opens history,
selects the query and runs it; Shift+Tab returns to the query, followed by keyboard
history clearing. Query focus and mode selection are
explicit setup steps; replay and clear are measured keyboard journeys.

The eight-cell matrix crosses both themes, both sizes and Search/RAG modes. RAG
requests use an adapter to real keyword retrieval and a controlled synchronous
answer seam; the real answer service constructs prompts and resolves citations.
The native scope is Media; the mounted tests additionally change source scope
between original submission and replay. No result counts are injected.

All sixteen captures were rendered and inspected. Both query fields show the
replayed text; the keyboard-returned query is visible. Cleared history displays
the focused disclosure heading and empty notice while evidence remains present.
The answer remains visible in the wide cleared-RAG captures. Compact captures
show the keyboard viewport; state identity checks establish retained answers.
[Capture hashes](capture-hashes.json) record raw/stored SVG bytes; only trailing
source-line whitespace was normalized.

| Theme / size / mode | Query after replay and keyboard return | Cleared history |
| --- | --- | --- |
| Dark / 170×48 / Search | [Replay](textual-dark-170-search-replayed.svg) | [Clear](textual-dark-170-search-cleared.svg) |
| Dark / 170×48 / RAG | [Replay](textual-dark-170-rag-replayed.svg) | [Clear](textual-dark-170-rag-cleared.svg) |
| Dark / 80×24 / Search | [Replay](textual-dark-80-search-replayed.svg) | [Clear](textual-dark-80-search-cleared.svg) |
| Dark / 80×24 / RAG | [Replay](textual-dark-80-rag-replayed.svg) | [Clear](textual-dark-80-rag-cleared.svg) |
| Light / 170×48 / Search | [Replay](textual-light-170-search-replayed.svg) | [Clear](textual-light-170-search-cleared.svg) |
| Light / 170×48 / RAG | [Replay](textual-light-170-rag-replayed.svg) | [Clear](textual-light-170-rag-cleared.svg) |
| Light / 80×24 / Search | [Replay](textual-light-80-search-replayed.svg) | [Clear](textual-light-80-search-cleared.svg) |
| Light / 80×24 / RAG | [Replay](textual-light-80-rag-replayed.svg) | [Clear](textual-light-80-rag-cleared.svg) |

[Lifecycle checks](lifecycle.json) confirm exit 0, app-run return and PID absence
before closing the owned terminal. All ten private databases pass read-only
`quick_check`; no conversation messages were added. The Media record and three
default profile files are unchanged. Cleared history remains empty in the
private config after shutdown. Logs show a normal stop, no error/critical or
unhandled-exception lines, and empty faulthandler logs.

The [independent review](review.json) found no actionable code/test issue. The
[task-ID check](task-id-check.json) found no other owner across 310 refs and 27
worktrees. No additional lesson entry is needed: this uses the already recorded
current-focus and selective-callback verification patterns.

This does not qualify real-provider behavior, semantic/remote retrieval, factual
grounding or very narrow single-pane layouts. No full suite, push or merge was
performed. Next bounded review: Search/RAG mode and source-scope changes.

## Reproduce targeted checks

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_history_keyboard.py \
  Tests/UI/test_library_rag_query_return.py \
  Tests/UI/test_library_rag_keystroke.py \
  Tests/UI/test_library_rag_result_focus.py \
  Tests/UI/test_library_rag_answer_retry.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  -q --tb=short --show-capture=no
```

The separate 13-test run selected the eight history record/load/precedence/replay/
bracket/clear/manual-disclosure cases in `Tests/UI/test_library_shell.py` plus
`Tests/Library/test_library_rag_state.py::TestUpdateSearchHistory`.
