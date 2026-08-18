---
id: TASK-18802
title: >-
  Summarization path sends sampling params that modern Anthropic and OpenAI
  models reject
status: To Do
assignee: []
created_date: '2026-08-18 23:55'
labels:
  - llm
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Summarization_General_Lib.py builds its own provider payloads and applies no per-model capability gate at all, so it reproduces exactly the defect TASK-18414 fixed on the chat path -- on a different, still-live code path.

summarize_with_anthropic (around lines 1048-1061) unconditionally sends temperature, top_k and top_p alongside get_cli_setting('anthropic_api', 'model', ...). PROBE-VERIFIED: the exact payload that function builds, sent to api.anthropic.com with a real key, returns

  HTTP 400 {"type":"error","error":{"type":"invalid_request_error","message":"`temperature` is deprecated for this model."},"request_id":"req_011CeB7m2VQbn2JXsd3LcQQy"}

on claude-opus-5. The same holds for Fable 5, Mythos 5, Opus 4.8, Opus 4.7 and Sonnet 5 -- and the shipped [api_settings.anthropic] model default is now claude-sonnet-5, one of them. summarize_with_openai (around lines 870-880) has the same shape, unconditionally sending max_tokens and temperature with openai_api.model (CODE-READ, not probe-verified: expected to 400 on gpt-5 / o-series / gpt-5.6).

This is reachable: analyze() in this module is imported by Web_Scraping/WebSearch_APIs.py, Web_Scraping/Article_Extractor_Lib.py, Local_Ingestion/{Book,Document,Image}_Processing_Lib.py and Local_Ingestion/XML_Ingestion.py, so ingestion and web-search summarization fail against the default configured model.

Found by a deliberate sweep for the same pattern while fixing TASK-18414, which added anthropic_model_rejects_sampling_params to model_capabilities.py; this path simply never consults it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Summarizing with a modern Anthropic model (Opus 5, Fable 5, Opus 4.8, Opus 4.7, Sonnet 5) completes instead of returning HTTP 400
- [ ] #2 Summarizing with an OpenAI reasoning model completes, with the correct token-cap parameter name
- [ ] #3 The summarization path reuses the model_capabilities predicates rather than adding its own model-name checks
- [ ] #4 Models that still accept these parameters are unchanged, pinned by a regression test
<!-- AC:END -->
