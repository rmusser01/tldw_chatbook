---
id: TASK-17384
title: Chunk summarization fails on large inputs with an unparseable 200
status: In Progress
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
- [x] #1 The failing response body is captured and the cause identified rather than inferred
- [x] #2 A chunk-sized summarization request against a local endpoint either succeeds or fails with a message naming the actual cause
- [x] #3 The bound that actually governs the failure is read from configuration rather than a fixed constant (AMENDED: the cause is not prompt size -- see notes)
- [x] #4 A test covers the identified failure mode without contacting a network
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

## Cause, captured rather than inferred (AC #1)

**The oversized-prompt guess recorded below when this was filed is DISPROVEN.**
The chunk packer caps chunks at 6000 characters (`_build_chunk_infos`,
`max_chars=6000`) against a server started with a 64k context, and a probe at
2,000 and 20,000 characters returned HTTP 200 with `choices` present. "No
choices in response data" was this function's parser fallthrough on EMPTY
content, not a missing `choices` array.

A faithful reproduction -- the real 6000-char packed chunk, the verbatim chunk
prompt, `max_tokens=4096` as the summarizer resolved it -- captured the actual
answer:

    finish_reason='stop'  content_len=465  reasoning_len=13209
    usage: completion_tokens=4028  (of max_tokens=4096)

The model spends its completion budget on `reasoning_content` and emits content
at the very edge of it. A chunk that reasons slightly longer returns empty
content, which is why 6 of 8 chunk summarizations failed in the task-17372 arm
while per-result calls in the same run succeeded: the margin is razor-thin and
input-dependent, not a size threshold.

An earlier probe appeared to refute this because it used a small input and spent
only 168 completion tokens -- 24x less than the real chunk. Recorded because the
mistake is instructive: a token-budget hypothesis cannot be tested on an input
that does not provoke the reasoning.

## Fix

`max_tokens` now prefers the modern `api_settings.llama_cpp` entry before the
legacy `llama_cpp_api` section -- the same split that had been sending these
requests to the wrong port (task-17382). A run priming a local endpoint sets
16384 there; reading only the legacy section pinned summarization to 4096 while
the chat path ran on four times that. That is AC #3 in its amended form: the
bound that governs this failure is the token budget, and it now comes from
configuration rather than a constant.

The failure message now names the real cause -- finish reason, tokens spent
against the budget, and whether the completion was reasoning-only -- while the
`logging.error` text stays verbatim, because that one is tracked in the reviewed
diagnostic inventory (task-492/3750) and this change is about the RETURNED
value a caller surfaces in a run's warnings. The "no choices in response" prefix
is preserved so the deep-search failure detector still recognizes it.

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
