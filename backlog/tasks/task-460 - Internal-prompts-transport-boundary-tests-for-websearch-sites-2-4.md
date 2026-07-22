---
id: TASK-460
title: 'Internal prompts: transport-boundary tests for websearch sites 2-4'
status: To Do
assignee: []
created_date: '2026-07-21 20:06'
labels:
  - internal-prompts
  - test-coverage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The P1 websearch migration added a live transport-boundary integration test only for the sub-question-generation call site. The other three migrated sites — result_relevance_eval, result_summarization, answer_synthesis in `Web_Scraping/WebSearch_APIs.py` — are covered only by kwarg-match verification plus golden parity. A wrong kwarg name at those sites fails silently (the token survives unsubstituted, no crash). Add one transport-boundary test each (fake only `chat_api_call`, override via scratch_config, assert the override text reaches the payload) so a wiring regression is caught. Deferred minor m3 from the P1 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A transport-boundary test proves a config override reaches the LLM payload for result_relevance_eval
- [ ] #2 Same for result_summarization
- [ ] #3 Same for answer_synthesis
- [ ] #4 Each test fakes only the transport (chat_api_call); pipeline code runs real
<!-- AC:END -->
