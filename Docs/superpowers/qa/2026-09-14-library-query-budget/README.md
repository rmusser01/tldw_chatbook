# Library focus and resize query budgets

TASK-32599; baseline `af5a26a2af`, branch `feat/component-pattern-library`.

Library now resolves active rail, focusability and resize chrome through its
existing live-reference cache. The ordinary width updater reuses the existing
adaptive-shell check. References still reject detached/pruning widgets; the
focus chain still decides whether a hidden or disabled control is reachable.
File Notes retains its explicit active-rail authority. No width policy, token,
stylesheet, focus binding, storage or service boundary changed.

## Measured outcome

| Measurement | Before | After |
| --- | ---: | ---: |
| Library-attributed lookups across 168/167/166-column resize frames | 23 | 0 |
| Library-attributed lookups per measured Tab | 5 | 0 |
| Warm full focusability helper, median | 348.89 µs | 121.09 µs |
| Warm active-rail helper, median | 0.18 µs | 0.26 µs |

The lookup budgets remain zero for the resize sequence and at most one per Tab.
The timed helper uses seven batches of 500 calls on a mounted production-styled
screen. Avoiding the selector scan reduced focusability-helper time by about
65%; this is not a measurement of end-to-end input latency. The already-cheap
active-rail lookup became slightly slower, so query count alone is not a speed
claim. Full samples and attributed call sites are retained in JSON.

The resize trace found one legitimate geometry change: at 166 columns the
resolved ordinary rail moves from 36 to 35 cells. The gate must preserve that
change, despite no compact/emergency breakpoint crossing. New assertions pin
the actual width and no whole-screen recompose while the unchanged zero-query
budget now passes using cached references.

## Verification and limits

- Full task-23025 file: **22 passed**. Both original query gates now run under
  the original and complete production stylesheets. Added checks cover hidden,
  disabled, removed and replaced search controls, plus an adaptive Media →
  ordinary Search/RAG return with 24/48-cell saved widths at 120 → 80 → 120.
- Neighboring selection: **10 passed, 5 failed**. Passing cases include visible
  rail focus in both themes, compact width transitions, File Notes active-rail
  ownership and visible search focus, and the prior Workspace transitions.
- The four older custom-width failures share a postlude that treats Collections
  as ordinary even though it now has an adaptive reader shell. Restoring all four
  modified methods from baseline `af5a26a2af` reproduces that exact failure for
  the representative 24-cell case. The new return check uses the actually
  ordinary Search/RAG surface; the legacy Collections pins remain unchanged.
- The fifth failure expects a visible File Notes task-return control. The same
  baseline-method control reproduces it. This is separate existing test debt;
  the related compact Notes authority audit remains tracked under TASK-32600.
- Native app with a copied audit note, all ten configured DB paths and data/user
  directories isolated: Tab from search into the rail; resize to 80×24; open
  Notes and the saved note; resize 120 → 80 with editor focus/content retained;
  Escape and restore 120×45 returns focus to the selected note row. ANSI captures
  retain styles. The app exited 0 and its temporary terminal was closed. No data
  edits or model calls were made.
- Ruff check/format pass for the changed test and archived probes. The Library
  monolith has the same **205 pre-existing Ruff diagnostics** before and after;
  all four changed methods pass formatting. No new diagnostic or whitespace
  error was introduced. No full-suite run was requested or performed.

The native run still logs the previously observed missing `#app-log-display`
startup error, optional-dependency warnings and project-skills worker warnings.
See `native-log-review.txt` and `startup-exit.ansi`; successful rendering is not
claimed to mean an entirely clean startup. Pytest warnings concern cleanup of
old temporary Kokoro directories. Neither class of warning is changed here.

## Reproduce

```sh
.venv/bin/python -m pytest -q Tests/UI/test_library_resize_focus_gates_t23025.py
```

`neighbor-selection.txt` contains the exact neighboring nodes. The disposable
probes must be copied to `Tests/UI/` before running with pytest, then removed:
`_query_budget_probe.py` records current query attribution;
`_query_budget_timing.py` times complete helpers;
`_query_budget_control.py` restores the four methods from the pinned local git
baseline and intentionally reproduces the two older failure types. The baseline
control does not modify repository files. Raw logs and private data remain under
`.superpowers/sdd/2026-09-14-library-query-budget/`.

ADR required: no. Existing ADR-086 governs exact width/reader ownership and
ADR-150/161 govern design tokens and component patterns. This is a routine
implementation optimization within those boundaries.
