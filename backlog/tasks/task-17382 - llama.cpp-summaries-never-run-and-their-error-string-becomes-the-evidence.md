---
id: TASK-17382
title: llama.cpp summaries never run and their error string becomes the evidence
status: Done
assignee: []
created_date: '2026-08-17 08:20'
labels:
  - websearch
  - research
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Per-result summarization is dead for llama.cpp, and the failure is silently promoted into the evidence the synthesis reads. The summarizer looks up a configuration section that has never existed, so it raises before it ever contacts the server; it reports failure by returning an error string rather than raising; and the deep-search caller's guard for that convention only recognizes strings beginning with "Error:", which this one does not. The error text is therefore stored as the result's content, while the real scraped text survives only alongside it under another key.

The consequence is that on a llama.cpp run the synthesis prompt is built from titles and gate reasoning with an error string where each source body should be. Citation verification is unaffected because it matches quotes against the scraped text first, which is why recorded runs could still show verified quotes and resolved markers — the reports were graded as sound while the model had never been shown the sources. Every live baseline recorded in this work stream used llama.cpp, so this bounds what those numbers can be read to mean. The relevance gate is not affected: it judges results before summarization, on the scraped content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A llama.cpp summarization request reaches the server instead of failing on a configuration lookup
- [x] #2 The deep-search caller treats a provider's error-string return as a failure regardless of which provider produced it, falling back to the source content
- [x] #3 No provider error text can be stored as a result's content
- [x] #4 The sibling local summarizers are checked for the same wrong-section-name defect and fixed or cleared
- [x] #5 Tests cover the configuration lookup and the caller's error-string detection for a non-"Error:" prefix
- [x] #6 The eval baseline doc records which recorded metrics this bounds, and which it does not
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in two halves. The summarizer resolves `llama_cpp_api` once,
defensively, and prefers `api_settings.llama_cpp.api_url` (what the chat
handler reads and what a run priming a local endpoint sets) over the legacy
`api_ip`, so summaries go where the run's model is. The deep-search caller
replaces its `startswith("Error:")` test with `_is_summary_failure`, matching
the markers the summarizers actually emit anchored to the leading clause
(optionally after a provider prefix), applied at BOTH guard sites -- the
per-result summary and the map-reduce chunk summary.

AC #4: checked and NOT clean -- probing the loaded config shows `api_keys`,
`local_api_ip` and `models` are absent too, so the Kobold and TabbyAPI
summarizers fail identically. Filed as task-17383 rather than fixed here; the
caller-side detector already protects the evidence for every provider.

AC #6: the eval baseline doc records the bound in the task-17370 arm section --
`gate_pass_rate` is unaffected (the gate judges scraped content before
summarization), while `cited_sentence_ratio`, `claim_support_rate` and
`quote_grounding` in every recorded table were graded on reports written
without source bodies.

Modified: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py`,
`tldw_chatbook/Web_Scraping/WebSearch_APIs.py`,
`Tests/LLM_Calls/test_llama_summarizer_config.py` (new),
`Tests/Web_Scraping/test_deep_search_pipeline.py`.

Evidence gathered when the defect surfaced during the task-17370 fan-out arm
(2026-08-17), before any fix:

- `Local_Summarization_Lib.py:261` reads `loaded_config_data["llama_api"]
  ["api_retries"]`. Probing the loaded config: `llama_api` is MISSING, while
  `llama_cpp_api`, `kobold_api`, `ooba_api` and `ollama_api` are all present
  with `api_retries`/`api_retry_delay`. Only the llama.cpp summarizer names a
  section that does not exist.
- The lookup sits above the HTTP call, so the request is never sent -- the
  observed failure took about a millisecond, not a timeout.
- Calling `analyze(api_name="llama_cpp", ...)` directly returns the string
  `"Llama: Error occurred while processing summary with Llama: 'llama_api'"`.
- `WebSearch_APIs.py:1393` guards with `summary.startswith("Error:")`, which
  is False for that string, so `WebSearch_APIs.py:1399` stores it as
  `content`.
- `WebSearch_APIs.py:1677` builds the synthesis payload from `content`;
  `deep_search_citations.py:138` verifies quotes against `original_content`
  first, then `content` -- which is why verification kept passing.
<!-- SECTION:NOTES:END -->
