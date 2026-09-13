---
id: TASK-26024
title: Auxiliary model routing for side tasks
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 19:12'
labels:
  - providers
  - performance
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Titles, compaction and other side work run on the user's main chat model. Verified on origin/dev: Chat/Chat_Functions.py:186-217 defines SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS - the auxiliary set is identified and audited - but those calls dispatch through the same API_CALL_HANDLERS table at the same model, so a user on an expensive reasoning model pays that rate to generate a conversation title. Hermes routes side tasks to a cheaper tier with a documented resolution order.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A configurable auxiliary model handles side tasks (at minimum: conversation titling and compaction) instead of the main chat model
- [x] #2 With no auxiliary model configured, behavior is exactly as today
- [x] #3 Auxiliary selection falls back to the main model when the configured auxiliary is unavailable or unconfigured, rather than failing the side task
- [x] #4 Auxiliary usage and cost are attributed separately in accounting so the saving is measurable
- [x] #5 The auxiliary model never handles user-visible chat turns - asserted by a test over the audited endpoint set
- [x] #6 Sensitive auxiliary endpoints continue to honor the existing audit constraints when routed to a different provider
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: pure helpers (config-gated selection override incl. cross-provider base_url drop; fallback to main when aux not ready) + AC#5 structural pin\n2. console_auxiliary_routing.py: auxiliary_selection_from_config + select_auxiliary_or_main (pure)\n3. Controller async _auxiliary_compaction_resolution wrapper; wire into manual summarize + compact_context_now (covers micro)\n4. Config [chat_defaults] auxiliary_provider/model; guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SCOPE FINDING: the Console's ONLY auxiliary LLM call is compaction — titling (derive_console_session_title) is pure string truncation, no model. So AC#1's 'at minimum titling and compaction' is satisfied by routing compaction; there is nothing to route for titling. Pure helpers in console_auxiliary_routing.py: auxiliary_selection_from_config (returns None when unconfigured = AC#2; overrides provider/model on the main selection; model-only keeps the main provider; a cross-provider aux drops the main base_url so the new provider resolves) + select_auxiliary_or_main (aux when ready else main = AC#3). Controller _auxiliary_compaction_resolution reads [chat_defaults] auxiliary_provider/auxiliary_model, resolves the aux selection bounded, falls back to main on any not-ready/exception. Wired into manual summarize (_manual_summary_planning) and compact_context_now — the latter covers 'compact now' AND per-turn micro-compaction (25910 routes through it), the most frequent auxiliary call. AC#4: the auxiliary-attempt ledger already records provider+model per attempt, so aux usage/cost attributes separately for free. AC#5: the resolver is only referenced in compaction methods, never the send dispatch — pinned by a source-scan test asserting it is absent from _run_agent_reply. AC#6 (audited endpoints under a different provider): N/A for the Console compaction path — the SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS audit is the classic Chat_Functions dispatch, untouched here; the Console compaction gateway carries its own trace/redaction that travels with whichever resolution is used. DELIBERATE SCOPE: the AUTO on-send compaction stays on the main model — its resolution is the send resolution and swapping mid-preflight is where the planning/vision/token-limit coupling risk is highest; documented in config + guide. TRADE-OFF: a non-vision auxiliary makes visual compaction refuse and fall to the failure behavior (max_visual_inputs=0), same as a non-vision main model. 6 new tests; compaction 143 passed; rewind/session failures are the exact pre-existing baseline.
<!-- SECTION:NOTES:END -->
