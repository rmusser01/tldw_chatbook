---
id: TASK-1710
title: >-
  Opt-in per-snippet continuations in the results grid
status: To Do
assignee: []
created_date: '2026-08-01 07:00'
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
- [ ] A bench can opt into capturing a per-snippet continuation, off by default
- [ ] With it off, request count per cell is unchanged from today
- [ ] With it on, the Estimate reflects the added calls before the run starts
- [ ] A captured continuation is visible for a focused cell alongside its top-K
- [ ] Measured distributions are provably unaffected by the continuation capture
- [ ] Runs recorded without continuations still render
<!-- AC:END -->
