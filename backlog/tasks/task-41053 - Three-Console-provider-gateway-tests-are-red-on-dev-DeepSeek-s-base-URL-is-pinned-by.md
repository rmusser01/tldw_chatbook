---
id: TASK-41053
title: "Three Console provider-gateway tests are red on dev: DeepSeek's base URL is pinned by default, and an OpenAI stop is not recorded"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- providers
priority: low
references:
- Tests/Chat/test_console_provider_gateway.py
- tldw_chatbook/Chat/console_provider_gateway.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev ecad404bd4 (2026-11-15) (serially, at `-n 4` and under the bootstrap profile), three nodes in `Tests/Chat/test_console_provider_gateway.py` fail:
- `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]` and `test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base`: the handler kwargs carry `api_base_url='https://api.deepseek.com'` where the tests expect no pinned base URL. The other six providers in the parametrize pass.
- `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`: the streamed content is never recorded within 5 s.

Either DeepSeek now pins a base URL on purpose (the hosted-provider engine, ADR-179) and the tests are stale, or a Console send pins a URL that the user's config should own. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say for each node whether the product or the test is wrong, and name the commit
- [ ] #2 The three nodes pass on dev; a fixed product defect is shown red first
<!-- AC:END -->
