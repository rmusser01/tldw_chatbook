---
id: TASK-1691
title: >-
  Show a generated continuation alongside the next-token distribution
status: To Do
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
- The canary path in `capture_client` already has this text in hand (`CANARY_PROMPT` continuation, used only to classify pass/degenerate and then logged). Surfacing what already exists is the cheapest first slice.
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
