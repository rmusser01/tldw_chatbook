---
id: TASK-17384
title: Chunk summarization fails on large inputs with an unparseable 200
status: To Do
assignee: []
created_date: '2026-08-17 11:20'
labels:
  - websearch
  - llm-calls
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Map-reduce chunk summarization still fails against a local llama.cpp endpoint after the configuration, endpoint and response-shape defects were fixed. Per-result summarization succeeds on the same code path in the same run, so this is input-dependent rather than a configuration problem: the failing calls are the ones carrying a whole chunk of concatenated evidence rather than a single page.

The server answers with a success status whose body carries neither of the shapes the parser accepts, so the summarizer reports its no-usable-content failure. The most likely explanation is an error body returned for an oversized prompt, but that has not been confirmed, and the payload was not captured at the time.

The consequence is bounded rather than silent: the caller recognizes the failure and falls back to the chunk's own source text, marking the chunk as not generated. So the synthesis still receives real evidence, and the failure costs a wasted LLM call plus the summarization quality on large evidence pools -- which is exactly the case multi-hop runs produce most of.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failing response body is captured and the cause identified rather than inferred
- [ ] #2 A chunk-sized summarization request against a local endpoint either succeeds or fails with a message naming the actual cause
- [ ] #3 If the cause is prompt size, the chunk packer's bound is derived from something the endpoint reports rather than a fixed constant
- [ ] #4 A test covers the identified failure mode without contacting a network
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Raised to high priority on task-17372's arm G evidence (2026-08-17): this now
blocks BOTH decomposition mechanisms, not just an edge case. That arm admitted
56% more evidence than its predecessor (32 -> 50 sources) and produced HALF the
resolved citation markers (70 -> 37), because it was the first fan-out arm large
enough to trigger map-reduce chunking -- 8 chunk operations, 6 of them failing
here. With multi-hop shipping on by default, every larger evidence pool runs
into this, so the defect converts retrieval improvements into worse reports.

Evidence from the task-17370 arm F run (2026-08-17), all three earlier defects
already fixed:

- Per-result summarization: 7 attempted, 7 succeeded, durations 42/94/131/64/
  81/115/93 seconds. Same function, same endpoint, same run.
- Chunk summarization: 2 attempted, 2 failed, both "Llama: No choices in
  response data".
- That message comes from the parser's fallthrough, so the response was a
  success status carrying neither `choices[].message.content`,
  `choices[].text`, nor a top-level `content`; a non-200 would have produced
  "Llama: API request failed" with the status instead.
- The llama-server instance was started with `-c 64000`, and the recorder
  primes `max_tokens=16384`.
<!-- SECTION:NOTES:END -->
