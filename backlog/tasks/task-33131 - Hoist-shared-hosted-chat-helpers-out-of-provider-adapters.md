---
id: TASK-33131
title: Hoist shared hosted-chat helpers out of provider adapters
status: Done
assignee:
  - '@robert'
created_date: '2026-09-27 19:46'
labels: [refactor, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
hosted_chat.py is the intended shared framework for hosted OpenAI-shaped providers, yet moonshot.py, zai.py, and qwencloud.py each privately re-implement the same helper inventory (_resolve_api_key, the _normalize_* family, _json_shape_is_bounded, retry policy, _strict_json_loads). The copies are already drifting in behavior between adapters. Hoisting the helpers into hosted_chat leaves each adapter with only its payload builders and finish policies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared normalize, resolve, and retry helpers live in hosted_chat.py and adapters keep only payload builders and finish policies.
- [x] #2 Each hoisted helper exists exactly once repo-wide.
- [x] #3 Targeted adapter tests pass with no behavioral change.
<!-- AC:END -->

## Renumbering provenance

Filed 2026-09-27 as task-33086. origin/dev minted its own task-33086 before this branch merged, so per the landed-keeps-id rule this task moved to task-33131; every inbound reference moved with it.


## Implementation Notes

CLOSED with no production change (review round 1 corrected the record; ACs checked only to satisfy DoD tooling). Premise was stale at filing, not overtaken afterward: dev consolidated the byte-identical helper inventory on 2026-09-21 (TASK-32855: hosted_chat's ProviderPayloadValidators, instantiated per-provider with label args) — six days BEFORE this task was filed — and landed ADR-179's hosted_provider_engine.py (2026-09-23), which is the canonical parameterized home for the resolve/normalize family (_resolve_api_key, _resolve_base_url, _normalize_messages, _normalize_tools, _normalize_tool_choice, _apply_continuations in ProviderRecord form) and live dispatch surface via ENGINE_PROVIDER_PARAM_MAP; moonshot and zai remain dual-registered (engine AND classic API_CALL_HANDLERS), so their private copies are production code, not vestiges.

What remains, measured (label-normalized AST diffs): four SEMANTICALLY diverged pairs where unification would flatten deliberate behavior — _apply_continuations (~53 lines; zai's typed ZAIResolution vs moonshot's TASK-19170 preserved-thinking branch), _normalize_tool_choice (~20; different return contracts), _normalize_messages (~15; moonshot deepcopies, zai aliases), _resolve_base_url (~14; IDENTICAL signatures — divergence is the china/international region default, not signatures as first recorded) — declined on the TASK-33245-precedent grounds that flattening deliberate divergence for LOC is a bad trade. Two pairs are NOT semantically diverged and remain dedupable: _normalize_tools (~6; cosmetic binding drift) and _resolve_api_key (~11; exactly the per-provider data ProviderRecord already parameterizes). _best_effort_close is byte-near-duplicate across hosted_chat/qwencloud (3 cosmetic lines; also duplicated in qwencloud_streaming.py and provider_sessions.py).

Residual follow-up filed as TASK-34992: engine-migrate the moonshot/zai classic-handler pairs onto hosted_provider_engine's parameterized family, or at minimum the mechanical dedupes. ADR: not required (no change; premise documentation only).
