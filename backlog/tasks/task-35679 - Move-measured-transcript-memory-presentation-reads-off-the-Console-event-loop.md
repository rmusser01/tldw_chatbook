---
id: TASK-35679
title: Move measured transcript memory presentation reads off the Console event loop
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20 02:27'
updated_date: '2026-10-20 02:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate slower Console attachment and input tails, remove the next measured synchronous native presentation read while preserving storage and branch-memory correctness, and retain all controlled measurements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The transcript memory banner uses the existing finite presentation reader without main-thread native config or memory reads for qualified stock owners.
- [ ] #2 Memory selection, custom-reader compatibility, source and owner changes, cancellation and physical retirement retain their original safety behavior.
- [ ] #3 Before and after native profiles retain all return/input/attachment timings, exact exits, source hashes and guard counts, including adverse results.
- [ ] #4 Targeted verification and independent review are recorded; PR3166 is updated with auto-merge disabled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: adopt the existing disposable context-presentation memo and finite native ownership under ADR126/230; no new cache, TTL, storage/schema, permission or service boundary. 1. Retain original tail samples and separate original callback/worker timing: transcript live-policy read568ms during leave, styling215ms before worst input, config/workspace waits on attach. 2. RED/GREEN actual transcript tests pin zero unused live-policy/native config entry and publication reopening its unchanged gate. 3. Reuse the already-wired finite context inputs for qualified original callbacks before the transcript gate, include effective-memory publication in that gate, and leave custom/partial callbacks synchronous. Do not await context completion or stall streaming. 4. Verify cold/held-worker streaming progress, memory banner parity, source/owner drift, original custom affinity and native cancellation/retirement. 5. Run serial preceding-head control/current ABBA with independent review, preserve adverse outcomes, update PR3166 with auto-merge off and remove exact owned fixtures.
<!-- SECTION:PLAN:END -->
