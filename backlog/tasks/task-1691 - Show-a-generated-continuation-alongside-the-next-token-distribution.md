---
id: TASK-1691
title: Show a generated continuation alongside the next-token distribution
status: Done
assignee: []
created_date: '2026-08-01 03:15'
updated_date: '2026-08-01 06:19'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A word bench measures ONE next token: `capture_client` sends `max_tokens: 1` and `CellCapture` stores only `top_k`, so nothing in the UI ever shows what the model would actually go on to say. That is the right instrument for the tool's purpose, but it leaves a real diagnostic gap that UAT hit immediately.

Found during UAT (2026-08-01) of a bench over "The sky is" against a local llama.cpp serving an instruction-tuned gemma. The grid reported the top token as `"<|channel>"` at 70.4%, with `"thought"` 15.2% and `"GB"` 11.9% — a distribution that reads as nonsense about the sky until you know the model is emitting its own chat-template scaffolding in raw mode. Probing the server directly outside the app made it obvious in one line: `"The sky is"` continues as `'<|channel><|channel>thought\n<channel|>The sky is **blue'` — the real answer is there, three template tokens later.

The app was not wrong: the degenerate-canary banner fired and the column carried `[warned]`, which is exactly the "this target's raw continuation is out-of-distribution" signal. But diagnosing WHY still required leaving the app for curl. A short generated continuation shown beside the distribution would have made the template scaffolding legible at a glance, and would help any user distinguish "the model has an odd distribution here" from "this model is not a raw-completion model at all".

Design notes and constraints:
- Sampling a continuation is a SECOND request per cell (or per target) — it must not silently multiply run cost or latency. Prefer one continuation per TARGET at preflight (the canary already generates one and throws it away) over one per cell; per-cell is a possible opt-in.
- CORRECTION (2026-08-01, verified in code before starting): an earlier draft of this task claimed the canary "already generates a continuation and throws it away". It does not. `preflight` calls `capture(CANARY_PROMPT, ...)`, which in raw mode sends `max_tokens: 1` and classifies pass/degenerate by testing whether `" Paris"`/`"Paris"` appears among the TOP-K TOKENS of that single token. No generated text exists anywhere in the system today. A continuation therefore needs a real request, not a salvage.
- Chat mode already sends `max_tokens: CHAT_TOKEN_WINDOW` and discards the generated text, so that half genuinely is a salvage; raw mode is not.
- Cost shape: one continuation per TARGET at preflight (preflight already runs once per target), never per cell. Raising the canary's own `max_tokens` in raw mode instead of issuing a second request is tempting but RISKY: `normalize_logprobs` scans up to `CONTENT_TOKEN_WINDOW` entries and takes the first non-control token's distribution, so lengthening the response can change WHICH token's top-K the canary judges, and therefore the pass/degenerate verdict itself. Prefer a separate, explicitly-scoped continuation request at preflight; if the cheaper single-request route is taken instead, it must ship with a test proving canary verdicts are unchanged across a corpus of shapes.
- Whatever is captured is user-facing model output: render `markup=False`, make whitespace visible with the existing ␣ convention, and cap the preview length.
- Historical runs must keep rendering — the field is additive and absent from every existing snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can see a short generated continuation for a target without leaving the app
- [x] #2 The continuation makes a degenerate-canary target's behaviour legible (e.g. template scaffolding is visible as text)
- [x] #3 Capturing it does not increase per-cell request count for a normal run, or is explicitly opt-in if it does
- [x] #4 Continuations render markup-safe with visible whitespace and a bounded preview length
- [x] #5 Runs recorded before this change still render
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine: `preflight` captures a short continuation for the target (one request per target, cell requests untouched); `PreflightResult` gains an additive `continuation` field; persistence carries it and old snapshots default to empty.
2. UI: the readiness surface renders the continuation for a target — markup-safe, whitespace visible via the ␣ convention, bounded preview length.
3. E2E + live verification against a real llama.cpp instruction-tuned model, where the continuation should make the chat-template scaffolding legible at a glance (the UAT case that motivated this task).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented across three tasks (T1 engine, T2 render, T3 E2E + hardening + closeout).

**Corrected premise**: an earlier draft claimed the canary "already generates a continuation and throws it away." Verified false before starting: raw mode's canary sends `max_tokens: 1` and classifies pass/degenerate purely from the single token's TOP-K, generating no text at all; only chat mode's canary (`max_tokens: CHAT_TOKEN_WINDOW`) genuinely discards generated text. The plan was corrected in the task file (2026-08-01) before implementation began.

**Why option (a) (separate request) over lengthening the canary**: `normalize_logprobs` scans up to `CONTENT_TOKEN_WINDOW` entries and judges the FIRST non-control token's distribution — lengthening the canary's own `max_tokens` in raw mode could shift which token's top-K gets judged, silently changing the pass/degenerate verdict itself. A wholly separate, `logprobs`-free continuation request is structurally incapable of perturbing the verdict (proven by `test_preflight_canary_verdict_is_unaffected_by_the_continuation_capture` and, now, the timeout variant added in T3), which is why it was chosen despite costing one extra HTTP call.

**Cost shape**: chat mode costs zero extra requests (the canary's own response is salvaged for its `message.content`, already paid for). Raw mode costs exactly one extra request PER TARGET at preflight (never per cell) — cell captures (`capture()`, still `max_tokens: 1`) are completely untouched, so a normal run's per-cell request count is unchanged regardless of dataset size; the added cost is bounded by target count, not sample count.

**T1** (`38136d661`): `PreflightResult.continuation: str = ""` (additive); `capture_client.py` splits `capture()` into a payload-returning `_capture_with_payload()` so `preflight` can read the raw response, adds `_resolve_continuation`/`_extract_chat_continuation`/`_capture_raw_continuation`, `CONTINUATION_MAX_TOKENS=24`, `CONTINUATION_CHAR_CAP=200`; `storage.py` persists/reads `continuation` in the run snapshot, defaulting missing keys to `""` for historical runs.

**T2** (`267e5492d`): `inspector.py` renders each target's continuation as a sub-line under its readiness badge — `_CONTINUATION_LABEL`, `_CONTINUATION_PREVIEW_MAX_LEN=100`, `_continuation_preview_text` (⏎ guard), `_continuation_static` (markup=False, reuses `snippet_editor.render_snippet_cell`'s ␣ convention, returns `None`/renders nothing for an absent continuation). New `.evals-target-continuation` CSS rule.

**T3** (this task): 
- New `Tests/UI/test_evals_continuation_e2e.py` drives the full loop through the real screen worker (dataset import → "+ New bench" → create target → Save → Run) with a fake capture client whose `preflight()` returns a UAT-shaped, degenerate-canary continuation (control-token scaffolding + a leading space, exercising both the "⏎" and "␣" markers). Asserts the readiness row renders it literally via `.visual.plain`, then re-selects the bench after a detour through the run group to prove it survives a fresh DB round-trip through the persisted snapshot, not just the in-memory `PreflightResult` the worker built. This is the joint none of T1/T2's own tests covered.
- Two review Minors: `test_preflight_continuation_degrades_to_empty_string_on_a_timeout` (T1 reviewer — `httpx.TimeoutException` on the continuation-only request, canary verdict untouched) in `Tests/Evals/word_bench/test_capture_client.py`; `test_readiness_rows_with_several_continuations_paint_inside_the_inspector_viewport` (T2 reviewer — a 235x52 painted-geometry test with a warned+continuation+callout target, an unreachable/no-continuation target, and a long truncated one, asserting the primary action and Estimate section stay reachable) in `Tests/UI/test_evals_screen.py`. Both are pure test additions; verified each is load-bearing via a temporary, reverted production mutation (narrowing the timeout except-clause; forcing `.evals-target-continuation { height: 0; }`) before committing — `git diff --quiet` confirmed byte-identical afterward.
- All 5 ACs verified against the actual rendered/persisted behavior (not just re-asserted from the plan) and checked.

**Live verification still owed**: the plan's step 3 called for "E2E + live verification against a real llama.cpp instruction-tuned model." This task delivers the E2E half only — every test in T1/T2/T3, including the new E2E file, runs against a fake/mocked HTTP layer (`httpx.MockTransport` or an in-process fake capture client), never a real server. The controller still owes a live run against an actual instruction-tuned model in raw mode (the original UAT's own repro shape) to confirm the captured continuation genuinely renders the chat-template scaffolding legibly end to end outside the test harness.
<!-- SECTION:NOTES:END -->
