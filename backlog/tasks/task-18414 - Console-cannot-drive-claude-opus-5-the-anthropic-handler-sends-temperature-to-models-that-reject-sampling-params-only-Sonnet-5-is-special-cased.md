---
id: TASK-18414
title: >-
  Console cannot drive claude-opus-5: the anthropic handler sends temperature to
  models that reject sampling params (only Sonnet 5 is special-cased)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18 16:20'
updated_date: '2026-08-18 23:59'
labels:
  - llm
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console cannot complete a turn against several current Anthropic models because the request builder decides two model-gated questions with ad-hoc name checks that only know about Claude Sonnet 5.

Provenance: OBSERVED LIVE during the PR 3b steering live pass (2026-08-18, real Anthropic key, scratch profile), and separately established by code reading. The pane is preserved at `steer-t6-live/panes/A1-sent.txt`: a scratch-profile Console configured `provider="anthropic", model="claude-opus-5"` with the shipped default temperature failed its first send with

    Agent run failed: provider returned HTTP 400 (Provider error from anthropic: bad
    request. Status: 400. Selected model: claude-opus-5. The provider rejected this
    request. Confirm the model is still available, or choose another model from the
    model picker.)

and the same session succeeded immediately after switching only the model to `claude-sonnet-5` (the one family both gates know). The 400 body was not captured, so WHICH of the two rejected parameters the provider named is still unestablished — that is the part a fresh reproduction still owes.

Two independent gates in LLM_Calls/LLM_API_Calls.py both omit the Claude 5 Opus/Fable tier:

1. _anthropic_is_sonnet_5() matches ONLY claude-sonnet-5*. It is the sole suppressor of temperature/top_p/top_k. Sampling parameters are rejected with a 400 on Fable 5, Mythos 5, Opus 5, Opus 4.8 and Opus 4.7 -- not just Sonnet 5.

2. _ANTHROPIC_ADAPTIVE_THINKING_MODEL_MARKERS lists only opus-4-7/4-8 and sonnet-4-6. Models absent from it fall through to the legacy budget_tokens branch, and budget_tokens is likewise rejected with a 400 on the same tier.

The result is that claude-opus-5 fails BOTH ways depending on the configured thinking effort: with an effort set it sends budget_tokens; with effort off or unset it sends temperature. claude-fable-5 and claude-mythos-5 are in the same position. A second, narrower case exists for claude-opus-4-8 and claude-opus-4-7: they are in the adaptive marker list, but when no effort is configured the mapper returns no thinking config, which re-opens the temperature branch and sends a parameter those models also reject.

The underlying defect is that a per-model API capability is encoded as two hand-maintained name checks in the request builder, so every new model release silently breaks a provider the app claims to support. The fix should express 'this model rejects sampling parameters' and 'this model rejects a fixed thinking budget' as capability predicates covering the whole family, ideally sourced from model_capabilities.py rather than duplicated string markers.
<!-- SECTION:DESCRIPTION:END -->

Console always sends a temperature (session default 0.6; the handler falls back to `data["temperature"] = current_temp` even when the caller passes none), so none of the affected models is usable from Console at all — and the surfaced error blames the model name and suggests the model picker, which cannot fix it.

Repro: scratch profile, `[chat_defaults] provider="anthropic" model="claude-opus-5"`, send anything. Switching the same session to `claude-sonnet-5` works immediately (the carve-out).

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Console turn against claude-opus-5 completes successfully with thinking effort unset
- [x] #2 A Console turn against claude-opus-5 completes successfully with a thinking effort configured
- [x] #3 No request to a model that rejects sampling parameters includes temperature, top_p or top_k (covering the Fable 5, Mythos 5, Opus 5, Opus 4.8, Opus 4.7 and Sonnet 5 families)
- [x] #4 No request to a model that rejects a fixed thinking budget includes budget_tokens
- [x] #5 claude-opus-4-8 and claude-opus-4-7 omit sampling parameters even when no thinking effort is configured
- [x] #6 Models that still accept sampling parameters and budget_tokens (e.g. Opus 4.6 and earlier) are unchanged, pinned by a regression test
- [x] #7 The model-family decision is expressed once as a capability predicate rather than duplicated name checks in the request builder
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the real 400 bodies from api.anthropic.com for every shape the builder can emit (temperature/top_p/top_k, budget_tokens) plus controls that must still succeed (Opus 4.6, Sonnet 4.5, Haiku 4.5).
2. Write payload pins RED first: no-sampling model with effort unset / set, opus-4-8+4-7 with no effort, and a legacy model that must still receive temperature.
3. Add two capability predicates to tldw_chatbook/model_capabilities.py (rejects sampling params / rejects a fixed thinking budget) over one Anthropic family table with a boundary-safe matcher covering bare, dotted, dated, suffixed and provider-prefixed ids.
4. Rewire LLM_API_Calls.py: sampling suppression keys off the predicate (not thinking_config is None + a sonnet-5 name check); the adaptive-thinking branch fires for every model that rejects a fixed budget.
5. Mutation-test the core predicate, run the targeted suites, then live-verify a real Console turn on claude-opus-5 with effort unset and with an effort set.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both model-gated questions in the Anthropic request builder are now capability predicates in `tldw_chatbook/model_capabilities.py` -- `anthropic_model_rejects_sampling_params` and `anthropic_model_rejects_fixed_thinking_budget` -- over one family table matched by (tier, major, minor).

**Step 1 first: the 400 bodies were captured before any fix** (the task's open question). Direct curl against api.anthropic.com with the real key. Both shapes the code can emit for claude-opus-5 are rejected, and the provider names the parameter:
- temperature/top_p/top_k -> 400 ```X` is deprecated for this model.`
- thinking={type: enabled, budget_tokens: N} -> 400 `"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.`
Same rejection confirmed on opus-4-8, opus-4-7, fable-5, sonnet-5. Controls confirmed still 200: opus-4-6 (temperature AND budget_tokens), sonnet-4-5, haiku-4-5. claude-mythos-5 returns 404 on this key (Project Glasswing only), so it is the one row documented rather than live-observed.

**Approach.** The predicates deliberately sit outside the config-driven capability tables in that module, for two concrete reasons: (1) `get_model_capabilities` returns on a direct mapping before consulting patterns, and `claude-sonnet-5` already has one -- a pattern-based implementation would have missed exactly the one model that already worked; (2) those tables are wholly replaceable from config.toml, and the only edit a user can make to a request-validity fact is one that reintroduces the 400. Pinned by `test_predicates_survive_a_user_configured_capability_table`.

**Trade-off.** `_anthropic_is_sonnet_5` was kept, narrowed to Sonnet 5's thinking *shape* only, rather than generalised. Widening its 'effort = off' branch to the whole family would introduce a NEW 400 on Fable 5, which rejects an explicit thinking={type: disabled} outright. That third capability is filed as TASK-18800, not half-fixed here.

**Key fix detail.** AC #5 came from decoupling sampling suppression from `thinking_config is None`: Opus 4.8/4.7 are in the adaptive set but produce no thinking config when no effort is set, which reopened the temperature branch. The hand-maintained marker list shrank to `sonnet-4-6` alone -- the one model that merely prefers adaptive thinking rather than requiring it.

**Evidence.** Red-first 12 failed/56 passed -> green 68 passed. Targeted gate 1620 passed, 3 failed (the 3 are test_summarization_diagnostic_privacy manifest-boundary tests, reproduced identically on a clean origin/dev worktree at 3 failed/254 passed -- filed as TASK-18801). Collect-only sweep 50634 tests, 0 errors. Mutation-tested the predicate twice: narrowing the table to sonnet-5 killed 30 pins, shifting the 4-7 boundary to 4-6 killed 9 in both directions (including the AC #6 over-match pins). Live: scratch tmux profile with the shipped default sampling values, real key -- claude-opus-5 completes a Console turn with effort unset and with effort=high, opus-4-8 and opus-4-6 also complete, and the same scratch config driven by clean origin/dev reproduces the filed HTTP 400 verbatim in both directions.

**Files.** tldw_chatbook/model_capabilities.py (predicates), tldw_chatbook/LLM_Calls/LLM_API_Calls.py (both gates rewired), Tests/Chat/test_anthropic_model_capabilities.py (new, 68 pins), Tests/Chat/test_chat_functions.py (one dev test asserted the retired warning string and patched logger.warning with a single-arg list.append; message updated, payload assertions untouched), Docs/superpowers/plans/2026-08-18-task-18414-report.md.
<!-- SECTION:NOTES:END -->
