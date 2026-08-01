---
id: TASK-1691
title: >-
  Show a generated continuation alongside the next-token distribution
status: In Progress
assignee: []
created_date: '2026-08-01 03:15'
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
- [ ] A user can see a short generated continuation for a target without leaving the app
- [ ] The continuation makes a degenerate-canary target's behaviour legible (e.g. template scaffolding is visible as text)
- [ ] Capturing it does not increase per-cell request count for a normal run, or is explicitly opt-in if it does
- [ ] Continuations render markup-safe with visible whitespace and a bounded preview length
- [ ] Runs recorded before this change still render
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine: `preflight` captures a short continuation for the target (one request per target, cell requests untouched); `PreflightResult` gains an additive `continuation` field; persistence carries it and old snapshots default to empty.
2. UI: the readiness surface renders the continuation for a target — markup-safe, whitespace visible via the ␣ convention, bounded preview length.
3. E2E + live verification against a real llama.cpp instruction-tuned model, where the continuation should make the chat-template scaffolding legible at a glance (the UAT case that motivated this task).
<!-- SECTION:PLAN:END -->
