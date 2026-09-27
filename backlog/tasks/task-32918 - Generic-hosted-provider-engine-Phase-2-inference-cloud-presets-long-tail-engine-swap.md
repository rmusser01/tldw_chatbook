---
id: TASK-32918
title: >-
  Generic hosted provider engine Phase 2: inference-cloud presets + long-tail
  engine swap
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-09-24 20:38'
updated_date: '2026-09-25 00:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md Phase 2 (as amended 2026-09-24)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real-server fixtures (plain+tools+streamed rounds; cloud /models) captured before swap/presets
- [ ] #2 Together/Fireworks/Cerebras selectable+usable with fixture-derived level-keyed allowances
- [x] #3 Preset-cost test proves no per-provider module; moonshot/zai allowances byte-identical
- [x] #4 custom-ep family executes via engine: gateway-site swap, shared CUSTOM_OPENAI_EXECUTION_KEYS, base-URL+credential forwarding, saved sessions unchanged
- [x] #5 Reasoning parity and api_settings.custom fallbacks carry over; kill switch ships
- [x] #6 Keyless entries work; curated presets hard-require keys
- [x] #7 Tolerant profile fixture-gated and scoped
- [x] #8 Docs updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-09-24-generic-hosted-provider-engine-phase2.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 7 (close-out) status: ACs #3-#8 checked -- preset-cost + moonshot/zai byte-identity pinned; gateway-site swap with shared CUSTOM_OPENAI_EXECUTION_KEYS, forwarding, and unchanged saved sessions; reasoning parity + full api_settings.custom fallbacks (incl. streaming=false, max_tokens=4096) + [console] custom_endpoints_use_engine kill switch; bearer_optional keyless + curated hard-require-keys; tolerant profile fixture-gated to the custom family; docs landed (Settings inference-clouds + strict custom-endpoints section, Console subsection, README env vars) plus double-gated live probes Tests/Chat/test_live_inference_cloud_api.py and the parity-test reconciliation in Tests/Chat/test_console_session_settings.py. OPEN (why not Done): AC #1's cloud /models fixtures and AC #2's fixture-derived allowances -- this environment holds no Together/Fireworks/Cerebras keys, so the three records ship EMPTY allowances behind the PROVISIONAL PENDING FIRST LIVE CAPTURE marker. First live runs of the Task 7 probes (TLDW_LIVE_<P>=1 + <P>_API_KEY) emit the evidence line (route, model, response key names) that reconciles each record's allowances; amend, never silent. Status stays In Progress.
<!-- SECTION:NOTES:END -->
