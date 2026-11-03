---
id: TASK-38541
title: Preserve eligible clean-history caches for mix-in sends
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38540
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
assignee:
- '@codex'
updated_date: 2026-11-01 23:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reuse stable provider prefixes where supported without replaying disabled instructions or changing existing caching policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cache boundaries identify stable history before the volatile mix-in tail; no-mix-in payloads and unsupported provider/endpoints retain existing behavior.
- [x] #2 Anthropic history breakpoints preserve its supported breakpoint budget, tool/system entries, disable switches, TTLs and degrade behavior; OpenAI stable cache keys and supported endpoint/model gating remain correct.
- [x] #3 Targeted payload/usage tests distinguish eligible reuse from a guaranteed hit, preserve instructions across cache degradation, and show toggles cannot reintroduce historical injections.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace semantic ownership, provider conversion and existing cache gates; confirm current official provider support.
2. Add failing canonical mocked-HTTP regressions for stable clean-history placement and mapping failures.
3. Carry a verified representation-hidden boundary into supported adapters, retaining switches, TTLs and degradation.
4. Run targeted cache/preparation/usage checks and scoped baseline lint; update provider architecture and report.
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: direct implementation of the approved cache-boundary contract; no new provider or storage boundary.
Review fix round 1 (BASE 817618048a): add RED real-HTTP cases for all-invalid and mixed historical assistant tool_calls; preserve legacy conversion while refusing an exact cache-boundary claim on discarded source tails; run the new caching file plus the existing native fallback and 5m/1h degrade cases, compare per-file Ruff diagnostics to FIX_BASE, and append exact evidence to the Task 6 report. ADR required: no; routine correction under backlog/decisions/286-console-prompt-mix-ins.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented an ephemeral semantic clean-history boundary and exact adapter-input witness for applied Console mix-ins. Anthropic maps it through message conversion/coalescing into a free fourth breakpoint, retaining existing opt-ins, kill-switch, TTL and degrade retry. OpenAI keeps its stable system/tool cache key and automatic caching; explicit breakpoint support is unavailable in the current client capability contract, with no speculative fields on custom endpoints.
Current architecture docs and canonical HTTP payload, empty/all-off, windowing, attachment, tool-loop, 1h degrade, native prompt-cache toggle and usage tests were added/extended. Final focused cache/usage31 and gateway5 pass; native conversion/prefix/streaming45 pass identically on BASE/current with exact bootstrap-profile qualification. Full-file Ruff differential introduces zero diagnostics. An unchanged checked-in diagnostic inventory/fixture mismatch is recorded separately in the task-6 report.
ADR required: no; direct implementation of backlog/decisions/286-console-prompt-mix-ins.md. Remains In Progress for independent review; no live cloud/cache-hit claim, full suite, push or merge.
Review fix I1: fully discarded historical assistant tool-call tails now make the new optional history-end mapping ineligible, while preserving the legacy plain-content fallback. Actual HTTP regressions cover all-invalid and mixed-valid/invalid tails, unchanged active body/marker and fixed-category unavailable status. Focused caching plus existing fallback/degrade checks: 29 passed. FIX_BASE 817618048a whole-file Ruff differential: zero introduced diagnostics; no new logging. ADR-286 unchanged; task remains In Progress for independent re-review.
Independent review cleared after2cc49ffc82 omitted optional history markers for all-invalid discarded tool-call tails, preserving legacy conversion.29 targeted fix checks pass; full-file lint/log inventories introduce no diagnostics. Actual composed→gateway→HTTP payload proof and source-qualified BASE fixture/manifest qualifications are recorded above. No live cache-hit/savings claim; OpenAI explicit capability remains undeclared, automatic caching/key preserved. ADR286 applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
