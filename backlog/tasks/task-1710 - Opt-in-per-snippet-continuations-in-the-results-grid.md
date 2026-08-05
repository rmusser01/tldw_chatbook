---
id: TASK-1710
title: Opt-in per-snippet continuations in the results grid
status: Done
assignee: []
created_date: '2026-08-01 07:00'
updated_date: '2026-08-01 08:08'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1691 shipped a continuation for the CANARY prompt — one per target, captured at preflight, rendered in Readiness. That answers "what does this target do with a known prompt", which is what diagnosing a degenerate column needs. It does NOT answer the other half of the original UAT request: "what would the model actually say after MY snippet".

The motivating case: a bench over "The sky is" against an instruction-tuned gemma renders top-1 `"<|channel>"` at 70.4%. The canary continuation now explains the target is incoherent in raw mode, but a user studying THIS snippet still cannot see that the model, given four more tokens, reaches `**blue` after its template scaffolding. The distribution and the continuation are different instruments and both are legitimately interesting per cell.

Why this is deliberately NOT what 1691 did: a per-snippet continuation costs one additional request PER CELL (snippets × targets), where the canary continuation costs one per target. On a 50-snippet × 3-target bench that is 150 extra requests against a local model — a real, user-visible cost that must be chosen, not inherited. Hence opt-in.

Design constraints (carried from 1691's implementation, verified in code there):
- Must NOT perturb the measured distribution. 1691 established the rule: never lengthen the request whose response reaches `normalize_logprobs`, because it takes the first NON-CONTROL token within `CONTENT_TOKEN_WINDOW` and a longer response can change WHICH token is measured. A per-cell continuation must therefore be a separate request (or be proven equivalent with a corpus test).
- Storage: `CellCapture` currently has no text field. Adding one is additive and must default empty so every historical run still renders; the snapshot writer/reader follow `PreflightResult.continuation`'s precedent from 1691.
- Rendering: raw model output, so `markup=False`, the `␣` whitespace convention via `render_snippet_cell`, the `⏎` single-line guard, and a bounded preview — the same rules 1691's readiness sub-line follows. The cell inspector (which already shows the full top-K for a focused cell) is the natural home; the grid cell itself is probably too narrow.
- The opt-in belongs on the bench (it changes what a run costs and what the snapshot contains), not on a view toggle — a run either captured continuations or it did not, and the UI must be honest about which.
- Cost must be visible BEFORE running: the inspector's Estimate section already renders call count and time; enabling this must update that estimate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A bench can opt into capturing a per-snippet continuation, off by default
- [x] #2 With it off, request count per cell is unchanged from today
- [x] #3 With it on, the Estimate reflects the added calls before the run starts
- [x] #4 A captured continuation is visible for a focused cell alongside its top-K
- [x] #5 Measured distributions are provably unaffected by the continuation capture
- [x] #6 Runs recorded without continuations still render
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine + storage: `BenchConfig.capture_continuations` (default False, additive); when on, each cell capture also fetches a short continuation through a SEPARATE request that never reaches `normalize_logprobs`; `CellCapture.continuation` additive, persisted per cell in the snapshot, defaulting empty for historical runs.
2. UI: an opt-in control in the bench editor (saved with the bench, part of the form/dirty contract); the Estimate reflects the doubled call count BEFORE running; the focused-cell inspector renders the continuation beside the top-K using the established markup=False/␣/⏎/bounded-preview rules.
3. E2E + live verification: a bench with the flag on, run against a real llama.cpp, focused cell shows both the distribution and what the model went on to say; a flag-off run is unchanged in request count.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
T1 (`40b3d1afd`): engine + storage only. `BenchConfig.capture_continuations` (default `False`) and `CellCapture.continuation` (default `""`), both additive. `WordBenchCaptureClient.capture_with_continuation` reuses task-1691's exact machinery, generalized to steer off the SNIPPET rather than the fixed canary prompt: chat mode salvages the continuation for free from the SAME response already fetched for measurement (zero extra requests); raw mode issues one genuinely separate, `logprobs`-free request per cell (`_capture_raw_continuation`, generalized to accept a `content` parameter) whose response never reaches `normalize_logprobs` -- structurally, not just empirically, incapable of perturbing the measured distribution. `WordBenchRunner._capture_cell` dispatches on the flag: off, calls `client.capture()` unchanged (the byte-identical single-request path AC #2 pins); on, calls `capture_with_continuation()` and folds the text into the returned `CellCapture` via `dataclasses.replace`. A failed cell is never charged a continuation request. Concurrency: the continuation leg runs inside the same per-target semaphore-guarded span the measurement request already used, so enabling continuations can never raise how many requests a single target has in flight, nor how many targets run concurrently -- it only lengthens one already-serial unit of work. T1's own review caught (but did not fix, out of scope) a save-path bug: `bench_editor.py`'s Save handler threaded `concurrency=` through its `BenchConfig(...)` reconstruction but had no equivalent for the new `capture_continuations` field -- saving ANY existing bench through the editor would have silently reset the flag back to `False`.

T2 (`239a19873`, `d0d1408eb`): UI. Fixed the T1-flagged save-path bug (`capture_continuations=capture_continuations` now threaded through `_on_save_pressed`'s `BenchConfig(...)` construction, audited against all nine non-`InitVar` fields). Added the opt-in `Checkbox` (`#evals-bench-capture-continuations`) to the bench editor form, mounted outside the targeted-rebuild `#evals-bench-targets-section` so it survives Add/Remove/prompt-mode-flip rebuilds; wired into `is_dirty()`. `inspector.py`'s `_continuation_call_count` makes the Estimate honest before a run starts: raw mode + flag on doubles the call count (one genuinely separate request per measured cell); chat mode returns 0 regardless (salvaged for free); flag off returns 0. `EvalsCellInspector` renders `CellCapture.continuation` as a permanent, always-composed, targeted-update (`show_cell()`) sub-line below the top-K/probe body -- never a recompose, so grid cursor position survives an arrow-key focus change -- under the same `markup=False`/"whitespace-marker"/"single-line newline guard"/bounded-preview rules task-1691's readiness pane established, but a distinct label ("Continuation: ", not "Canary prompt continuation: ") since this pane is already scoped to one specific (snippet, target) cell. In passing, T2 fixed a pre-existing, out-of-scope `#evals-cell-inspector` clipping bug (an unstyled `Vertical` defaulting to `height: 1fr`, claiming a 36-row region for ~9 rows of content and pushing `#evals-primary-action` out of the viewport at 235x52) with the same `height: auto` rule its sibling `#evals-inspector-bench` already carries -- kept in-commit rather than split out since the CSS neighborhood and the task's own brief both pointed at it; pinned with a dedicated 235x52 geometry test.

T3 (this task):
A. New `Tests/UI/test_evals_cell_continuation_e2e.py` drives the full opt-in loop through the real UI/worker seam (import a 2-snippet dataset -> "+ New bench" -> create one target -> flip the checkbox -> Save -> Run through the real `WordBenchRunner` -> focus each cell), using a fake capture client that returns a DIFFERENT continuation per snippet text -- the direct proof this is captured per CELL, not reused across every row the way task-1691's per-target canary continuation is (that distinction is this task's entire premise vs. 1691). One test also proves persistence across a fresh `storage.load_grid` read (select away to the bench, back to the run group, refocus, same continuation reappears). A second test is the flag-off control: the same snippet/target cell shape renders no continuation line at all, while the top-K still renders normally.
B. Two review corrections, no behaviour change: (1) `capture_client.py`'s `_capture_raw_continuation` docstring wrongly implied the measured raw-mode request could be `top_k`-sized for `max_tokens`; corrected -- `top_k` only ever sizes `logprobs`/`top_logprobs`, the measured request is always `max_tokens: 1`. (2) T1's own report overstated its test count ("+38 new", itemized to 28); verified against `git show 40b3d1afd`'s own diff (11+4+6+4+1) and corrected to the true 26 (capture_client +11, runner +6, models +4, storage +4, storage_authoring +1), with a correction note left in the report.
C. This closeout.

AC verification (re-checked against actual behavior, not the plan):
- A bench can opt into capturing a per-snippet continuation, off by default: `BenchConfig.capture_continuations: bool = False` (`models.py`); `#evals-bench-capture-continuations` checkbox (`bench_editor.py`), off by default, threaded through Save.
- With it off, request count per cell is unchanged from today: `WordBenchRunner._capture_cell` calls `client.capture()` unchanged on the flag-off path; proven by `Tests/Evals/word_bench/test_runner.py::test_capture_continuations_off_by_default_produces_the_same_request_shape_as_today` (its own docstring names this "AC #2"; its fake has no `capture_with_continuation` defined at all, so a dispatch regression would raise `AttributeError` rather than pass silently) and by this task's own E2E flag-off test.
- With it on, the Estimate reflects the added calls before the run starts: `inspector._continuation_call_count`; proven by `Tests/UI/test_evals_bench_editor.py::test_estimate_reflects_doubled_calls_when_capture_continuations_is_on_in_raw_mode` (20 -> 40 calls) and its off/chat-mode siblings.
- A captured continuation is visible for a focused cell alongside its top-K: `EvalsCellInspector.show_cell` renders both `#evals-cell-inspector-body` (top-K) and `#evals-cell-inspector-continuation` together; proven by `Tests/UI/test_evals_results_grid.py`'s dedicated cell-continuation test section and this task's own E2E test (a real Run through the screen's worker, not synthetic fixtures).
- Measured distributions are provably unaffected by the continuation capture: `Tests/Evals/word_bench/test_capture_client.py::test_measured_distribution_is_identical_whether_or_not_a_continuation_is_captured` asserts byte-identical `top_k`/`k_returned`/`k_requested`/`content_offset`/`canary`/`prompt_mode` between `capture()` and `capture_with_continuation()` against identical fake responses; the raw-mode continuation request structurally never reaches `normalize_logprobs` (it never requests `logprobs` at all).
- Runs recorded without continuations still render: `CellCapture.continuation` defaults to `""`; `Tests/Evals/word_bench/test_storage.py::test_load_grid_defaults_continuation_for_cells_recorded_before_this_change`; `Tests/UI/test_evals_results_grid.py::test_focused_cell_with_no_continuation_renders_nothing_extra_but_still_renders_the_cell`; this task's own E2E flag-off control test.

Modified/added files: `tldw_chatbook/Evals/word_bench/{models,capture_client,runner,storage}.py` (T1, capture_client.py docstring also T3-corrected); `tldw_chatbook/UI/Evals/{bench_editor,inspector}.py`, `tldw_chatbook/css/features/_evals.tcss` (+ regenerated `tldw_cli_modular.tcss`) (T2); `Tests/Evals/word_bench/{test_models,test_capture_client,test_runner,test_storage,test_storage_authoring}.py` (T1); `Tests/UI/{test_evals_bench_editor,test_evals_screen,test_evals_results_grid}.py` (T2); `Tests/UI/test_evals_cell_continuation_e2e.py` (new, T3).

Live verification against a real llama.cpp server was never performed across T1/T2/T3 -- every test in this chain, including this task's own new E2E, runs against a fake/mocked capture client. Flagged as outstanding, mirroring task-1691's own identical gap noted in its closeout.
<!-- SECTION:NOTES:END -->
