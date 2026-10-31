---
id: TASK-35591
title: Prepare Console token-estimation backends off the UI thread
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 04:35'
updated_date: '2026-10-19 19:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cold custom-tokenizer admission and encoding loads stall Console interaction on Windows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cold context and cost display preparation leaves the UI loop responsive while original storage checks remain active
- [x] #2 Model and session changes cannot publish stale tokenizer readiness; custom callbacks retain their existing path
- [x] #3 Targeted regression checks and native before/after profiles document the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-230/231 display-only finite-read ownership applies. Reason: routine scheduling fix preserving all storage and runtime boundaries. 1. Measure the existing cold tokenizer path. 2. Prepare the stock model/provider backend in the existing finite Console readiness worker and gate context/cost display until ready. 3. Verify lifecycle, model changes, storage safety and custom callback behavior with targeted tests. 4. Repeat native profiling and document measured limits.

Review follow-up: preserve measured-only Current cost totals when tokenizer preparation is unavailable; keep next-send/context estimation gated and prevent local token estimation. Add a failing measured-history regression, verify the fix and mixed-history refusal, and rerun targeted controls. ADR required: no; existing display-only boundary and ADR-230/231 apply.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved cold stock custom-manager admission and encoding initialization into the existing finite Console readiness worker; context and cost stay unavailable until their selected backend is prepared. Original storage checks and live counters are unchanged. Model/source/mapping/runtime guards and physical cancellation retirement apply; custom callbacks and heuristic discovery retain direct fallback. Added 21 backend/projection controls and an original-storage held mounted test. Native tour: tokenizer loading samples 7 to 0, worst idle loop overshoot 158.29 to 70.98 ms; typing ~3 ms unchanged. Related run: 149 passed, four separately proved baseline failures excluded, one pre-assertion startup timeout passed on unchanged isolated recheck; timeout remains documented. New lint/format and diff checks pass, existing lint debt unchanged. No full sweep. No new ADR: applies ADR126/230/231. See Docs/Development/evidence/windows-performance-2026-10-18/TOKENIZER_RESULTS.md and Docs/Architecture/console.md. Review cold-heuristic issue corrected and tested.

PR review follow-up: fixed an overly broad tokenizer gate that hid recorded Current spend. Measured-only ProviderUsage history remains visible without invoking token estimation; mixed history defers instead of publishing a partial total. Readiness is part of the settled-cache key, and historical payload/fingerprint/cache-break/stale context projections remain gated. New regression verified RED, then 68 focused cost/spend/tokenizer checks passed. Independent reviewer approved the correction with no remaining blocker. Ruff and diff checks pass. See Docs/Development/evidence/windows-performance-2026-10-18/PR_REVIEW.md. No new ADR: existing ADR-230/231 apply.
<!-- SECTION:NOTES:END -->
