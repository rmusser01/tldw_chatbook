---
id: TASK-18414
title: >-
  Console cannot drive claude-opus-5: the anthropic handler sends temperature to
  models that reject sampling params (only Sonnet 5 is special-cased)
status: To Do
assignee: []
created_date: '2026-08-18 16:20'
updated_date: '2026-08-18 18:51'
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
- [ ] #1 A Console turn against claude-opus-5 completes successfully with thinking effort unset
- [ ] #2 A Console turn against claude-opus-5 completes successfully with a thinking effort configured
- [ ] #3 No request to a model that rejects sampling parameters includes temperature, top_p or top_k (covering the Fable 5, Mythos 5, Opus 5, Opus 4.8, Opus 4.7 and Sonnet 5 families)
- [ ] #4 No request to a model that rejects a fixed thinking budget includes budget_tokens
- [ ] #5 claude-opus-4-8 and claude-opus-4-7 omit sampling parameters even when no thinking effort is configured
- [ ] #6 Models that still accept sampling parameters and budget_tokens (e.g. Opus 4.6 and earlier) are unchanged, pinned by a regression test
- [ ] #7 The model-family decision is expressed once as a capability predicate rather than duplicated name checks in the request builder
<!-- AC:END -->
