---
id: TASK-34397
title: Parallelize startup model-catalog provider refresh
status: Done
created_date: 2026-10-12 03:20
labels:
- performance
priority: medium
assignee:
- '@zcode'
updated_date: 2026-10-12 04:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
LLM_Provider_Catalog/local_llm_provider_catalog_service.py refresh_stale_configured_providers (~648, ~702) awaits discover_models per provider sequentially (7 providers: OpenAI, Anthropic, MistralAI, Moonshot, OpenRouter, QwenCloud, ZAI); each is an independent HTTPS GET with its own timeout, so startup refresh wall clock is the SUM of provider latencies instead of the max - one slow provider delays the whole notification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider refreshes fan out concurrently via asyncio.gather(return_exceptions=True) with per-provider failure isolation preserved (per-provider ProviderRefreshOutcome),Shared disk_store persistence happens once after all results collect,discover_models cache replacement is safe under the fan-out (lock or proven single-threaded),Startup catalog tests and settings-triggered refresh tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read refresh_stale_configured_providers 2. gather fan-out with per-provider isolation + single persist 3. tests + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and verified (897 consolidated tests green); see task file Implementation Notes; PR #3083
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
