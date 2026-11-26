---
id: TASK-41053
title: "Three Console provider-gateway tests are red on dev: DeepSeek's base URL is pinned by default, and an OpenAI stop is not recorded"
status: Done
created_date: 2026-11-15 02:03
updated_date: '2026-11-26 07:03'
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
**Note (2026-11-17, TASK-41136).** The file now runs on the `pr-fast-lane-arch` PR lane with these three nodes deselected by name (`--deselect=` lines in `.github/workflows/derived-artifacts.yml`, pinned by `ARCH_REVIEW_DESELECTED` in `Tests/CI/test_ci_queue_pressure_contract.py`). When a node passes, remove its deselect line and its `ARCH_REVIEW_DESELECTED` row in the same commit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task notes say for each node whether the product or the test is wrong, and name the commit
- [x] #2 The three nodes pass on dev; a fixed product defect is shown red first
<!-- AC:END -->

## Implementation Notes

Duplicate of TASK-40413, which fixed these three nodes (round 27, commit f9fe23947e). All three were test drift; no product code changed.
- **The two DeepSeek nodes.** The pin is intended. Since f5a63125ff (ADR-064's 2026-10-12 amendment), DeepSeek has shared QwenCloud's explicit-mode branch in all three kwargs builders, and the trace verifier reconstructs the same pin. Both tests now expect it.
- **The OpenAI Stop node.** Since c1cad552f4 (TASK-38589), the adapter reads bytes from `iter_lines()` and decodes UTF-8 itself. The fake's required `decode_unicode` keyword raised `TypeError` before any content arrived.

The deselect lines and `ARCH_REVIEW_DESELECTED` were removed in the same commit. See TASK-40413's notes for the evidence and the runs.
