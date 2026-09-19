# Search/RAG mode and source-scope keyboard review — TASK-32717

2026-09-17 UTC, `feat/component-pattern-library`, based on `ce3da505d2`.

Changing mode or source scope replaced the focused toggle and left focus on the
panel. All eight initial keyboard journeys reproduced the defect. The panel now
calls its inherited `preserve_same_id_focus_after_recompose()` before syncing
state, matching other Library canvases. The helper restores the replacement
control only when a newer attached focus has not taken precedence.

ADR required: no. This is a routine focus repair under existing
[ADR-031](../../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md).
No state, service, binding, token or persistence contracts change.

## Automated evidence

- [Red evidence](keyboard-red.txt): all eight keyboard mode/scope journeys fail
  because focus lands on the panel. Captured diagnostic noise is omitted and
  trailing whitespace is trimmed; assertions and the final summary are retained.
- [Targeted run](targeted-tests.txt): **144 passed in 260.64s**. This includes
  sixteen new production-styled cases: eight theme/size/toggle journeys, four
  delayed callbacks after newer panel/rail focus choices, and four changes while
  retrieval or answering is held at a controlled gate. Related coverage checks
  scope recovery, history, query return, result focus, canvas identity and the
  existing Search/RAG integration contracts.
- [Existing mode/scope checks](existing-mode-scope-tests.txt): **5 passed in
  8.72s**, covering mode cycling, stale retrieval, selected-source requests,
  filtering/selection alignment and manual history disclosure retention.
- Combined targeted total: **149 passed**, without exclusions or a full suite.
- [Lint comparison](lint-comparison.json): five existing production diagnostics,
  no additions. New test/runner lint and format checks and changed production
  range formatting pass.
- [Independent review](review.json): no actionable code, test or runner finding.

The gates retain the existing distinction: changing mode discards outcomes from
the mode left behind; changing scope filters rendered evidence without restarting
the original retrieval or answer. An answer still uses that run's evidence.
Queries/history survive both actions, and toggles make no new service calls.

## Native evidence

The [runner](native_check.py) boots real TldwCli in a fresh private tmux terminal
and profile, seeds a real Media record and workspace membership, and uses actual
keyword retrieval. RAG mode is adapted to that local retrieval and a controlled
synchronous provider seam; the real answer service builds prompts and resolves
citations. It does not exercise a remote provider or semantic retrieval.

All four theme/size journeys [pass](result.json), totaling eight real keyword
searches and four controlled answers. Each cell turns Media off and back on,
cycles Search → RAG → Search → RAG, submits the retained query, and finally
returns to Search. Focus setup selects the first control; subsequent Enter
presses reverse or cycle it without another focus assignment. Turning off the
only available source hides its evidence and disables Run; turning it back on
restores the same retrieved rows without a new search. Mode changes clear the
answer/evidence and preserve both query fields and history.

All twelve captures were rendered and inspected. Scope-off captures show the
focused Media toggle and the disabled Run remedy. RAG-mode captures show the
focused mode toggle and provider disclosure at both sizes. Wide answered captures
show the answer/citation feedback; compact answered captures establish visible
query focus, with answer completion established by the runner's state checks.
[Capture hashes](capture-hashes.json) record raw and stored bytes; only trailing
SVG source-line whitespace was normalized.

| Theme / size | Scope off | RAG mode | After answering |
| --- | --- | --- | --- |
| Dark / 170×48 | [Capture](textual-dark-170-scope-off.svg) | [Capture](textual-dark-170-rag-mode.svg) | [Capture](textual-dark-170-answered.svg) |
| Dark / 80×24 | [Capture](textual-dark-80-scope-off.svg) | [Capture](textual-dark-80-rag-mode.svg) | [Capture](textual-dark-80-answered.svg) |
| Light / 170×48 | [Capture](textual-light-170-scope-off.svg) | [Capture](textual-light-170-rag-mode.svg) | [Capture](textual-light-170-answered.svg) |
| Light / 80×24 | [Capture](textual-light-80-scope-off.svg) | [Capture](textual-light-80-rag-mode.svg) | [Capture](textual-light-80-answered.svg) |

[Lifecycle checks](lifecycle.json) verify normal exit 0, app-run return and PID
absence before closing the owned terminal. All ten private databases pass
read-only `quick_check`, no conversation messages were added, and source/default
profile files remain unchanged. The submitted query remains in private history
after shutdown. Logs contain no error/critical or unhandled-exception lines;
both faulthandler logs are empty. [Task-ID checks](task-id-check.json) find no
other owner across 307 refs and 27 worktrees.

The user guide documents keyboard toggling and existing in-flight behavior. This
uses the established canvas recompose/focus lesson; no new lesson is needed.
Real-provider behavior, semantic/remote retrieval, factual grounding and very
narrow single-pane layouts remain outside qualification. No full suite, push or
merge was performed. Next bounded review: older-engine chunk reporting and the
Re-chunk control.

## Reproduce the main targeted run

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_mode_scope_keyboard.py \
  Tests/UI/test_library_rag_scope_recovery.py \
  Tests/UI/test_library_rag_history_keyboard.py \
  Tests/UI/test_library_rag_query_return.py \
  Tests/UI/test_library_rag_result_focus.py \
  Tests/UI/test_library_canvas_scoped_sync.py::test_media_choice_and_rag_toggles_are_canvas_scoped \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  -q --tb=short --show-capture=no
```
