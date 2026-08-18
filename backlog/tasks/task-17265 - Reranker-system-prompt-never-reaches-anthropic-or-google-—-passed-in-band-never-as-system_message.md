---
id: TASK-17265
title: >-
  Reranker system prompt never reaches anthropic or google — passed in-band,
  never as system_message
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-17'
updated_date: '2026-08-17 23:58'
labels:
  - rag
  - llm-calls
dependencies:
  - TASK-17065
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read reranker.py:299-350 and PROVIDER_PARAM_MAP's differing system_message targets (anthropic->system_prompt, google->system_message, llama_cpp->system_prompt).
2. RED: new Tests/RAG_Search/test_reranker_system_prompt.py driving the REAL rerank() with requests.Session.post faked, asserting the assembled wire payload for anthropic/google/openai/llama_cpp.
3. Implement: drop the in-band {'role':'system'} entry; pass system_message= to chat_api_call (omit when falsy).
4. GREEN + ruff + counts + gate; commit and push.
<!-- SECTION:PLAN:END -->

## Description (the why)

Found by TASK-17065's final whole-branch review (finding F3), on the branch
that made the reranker actually call providers for the first time.

`BaseReranker._call_llm_impl` (`RAG_Search/reranker.py`) puts its system
prompt INSIDE `messages_payload` as `{"role": "system", ...}` and never
passes `chat_api_call`'s own `system_message=` argument, even though all 29
provider maps carry it. Two mainstream providers drop an in-band system
turn on the floor:

* `chat_with_anthropic` assembles its payload from user/assistant turns only
  ("Skipping message with unsupported role: system") and fills `data["system"]`
  from the `system_message` argument alone. The reviewer's fake-transport probe
  produced `PAYLOAD messages roles: ['user']`, `PAYLOAD has top-level system:
  False` — the system turn was discarded outright.
* `chat_with_google` maps `user→user`, `assistant→model` and `continue`s past
  every other role, and sets `system_instruction` only from `system_message`.

The system prompt is what tells the model "return only a JSON object with a
`score` field". Without it the parse-failure rate rises, and every parse
failure is a call that was BILLED and produced no score (`scored=False`; for
listwise, a single failure fails the entire rerank). This is not a regression
— nothing called anything before TASK-17065 — but it is newly reachable, and
it weakens that task's "completes a scoring call" claim for two of the
providers its picker offers.

`chat_with_openai` keeps an in-band system message and de-duplicates against
`system_message`, so passing both is safe there; the fix shape the review
suggests is to pass `system_message=` in addition to the in-band copy.

## Acceptance Criteria (the what)

- [ ] #1 The reranker's system prompt reaches the model on anthropic and google — asserted at the provider-handler boundary (fake transport, no live call) by inspecting the assembled payload, not just the reranker's own call site
- [ ] #2 Providers that accept an in-band system turn (openai and the local family) still receive exactly one system instruction — no duplicated or conflicting system text on the wire
- [ ] #3 A test pins the reranker's dispatch as carrying the system prompt in a form each of the sampled providers actually forwards, alongside the existing seam guard in `Tests/RAG_Search/test_reranker_degraded_paths.py`
- [ ] #4 No live provider call is made to satisfy any of the above
