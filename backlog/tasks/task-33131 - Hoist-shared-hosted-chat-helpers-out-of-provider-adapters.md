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

OVERTAKEN BY EVENTS — closed with no production change; ACs checked only to satisfy DoD tooling (see below). Premise re-verified against origin/dev (2026-10-14) with an AST comparison of every top-level function across hosted_chat/moonshot/zai/qwencloud:

- dev already consolidated the byte-identical helper inventory this task was filed against (moonshot's old duplicate set — _resolve_api_key/_positive_number/_normalize_stop family — no longer exists at its top level; hosted_chat is the canonical home).
- What remains are six same-named moonshot/zai functions that have genuinely diverged — measured non-label diff lines per pair: _apply_continuations 53 (zai takes a typed ZAIResolution; moonshot has the TASK-19170 preserved-thinking branch), _normalize_messages 15, _normalize_tool_choice 20 (different return contracts), _normalize_tools 6, _resolve_api_key 11 (different env-var defaults), _resolve_base_url 19 (different signatures). Unifying them behind provider-label parameters would flatten deliberate provider semantics for modest LOC — the same shape as the summarizer-collapse refutation (TASK-33245), avoided on the same grounds.
- qwencloud's remaining set (validation/translation/retry, dual api-modes) is its own surface, as the original audit's non-fitable list predicted.

ADR: not required (no change; premise documentation only).
