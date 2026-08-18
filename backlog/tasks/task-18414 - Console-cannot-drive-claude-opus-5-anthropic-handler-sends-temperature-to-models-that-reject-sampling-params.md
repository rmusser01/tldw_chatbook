---
id: TASK-18414
title: >-
  Console cannot drive claude-opus-5: the anthropic handler sends temperature to
  models that reject sampling params (only Sonnet 5 is special-cased)
status: To Do
assignee: []
created_date: '2026-08-18 16:20'
labels:
  - llm
  - console
priority: high
dependencies: []
---

## Description (the why)

Found by PR 3b Task 6's live pass (2026-08-18, real Anthropic key, scratch
profile). A fresh Console session configured with `provider = "anthropic"`,
`model = "claude-opus-5"` and the shipped default temperature (0.6) fails
EVERY send:

    Agent run failed: provider returned HTTP 400 (Provider error from
    anthropic: bad request. Status: 400. Selected model: claude-opus-5.
    The provider rejected this request. Confirm the model is still
    available, or choose another model from the model picker.)

Root cause, verified in code: `chat_with_anthropic`
(`LLM_Calls/LLM_API_Calls.py`, the sampling block around lines 1419-1441)
omits `temperature`/`top_p`/`top_k` only when thinking is enabled or when
`_anthropic_is_sonnet_5(current_model)` — Sonnet 5 is the single
special-cased model. But the entire current Anthropic top tier rejects
non-default sampling parameters with a 400: claude-opus-5, opus-4-8,
opus-4-7, and fable-5 all removed `temperature`/`top_p`/`top_k`. Console
always sends a temperature (session default 0.6; the handler falls back
to `data["temperature"] = current_temp` even when the caller passes none),
so none of those models is usable from Console at all — the error blames
the model name and suggests the picker, which cannot fix it.

Repro: scratch profile, `[chat_defaults] provider="anthropic"
model="claude-opus-5"`, send anything. Switching the same session to
`claude-sonnet-5` works immediately (the carve-out).

## Acceptance Criteria (the what)

- [ ] A Console send to claude-opus-5 (and opus-4-8 / opus-4-7 / fable-5) with default session settings succeeds — sampling params are omitted for every model that rejects them, not just Sonnet 5
- [ ] The capability check lives in one place (e.g. `model_capabilities.py` or a shared predicate), not a per-model string check duplicated in the handler
- [ ] A test pins the request payload for one no-sampling model (no temperature/top_p/top_k on the wire) and one legacy model (temperature still sent)
