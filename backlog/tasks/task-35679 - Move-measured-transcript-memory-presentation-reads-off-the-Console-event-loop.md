---
id: TASK-35679
title: Move measured transcript memory presentation reads off the Console event loop
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-20 02:27'
updated_date: '2026-10-20 03:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate slower Console attachment and input tails, remove the next measured synchronous native presentation read while preserving storage and branch-memory correctness, and retain all controlled measurements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The transcript memory banner uses the existing finite presentation reader without main-thread native config or memory reads for qualified stock owners.
- [ ] #2 Memory selection, custom-reader compatibility, source and owner changes, cancellation and physical retirement retain their original safety behavior.
- [x] #3 Before and after native profiles retain all return/input/attachment timings, exact exits, source hashes and guard counts, including adverse results.
- [x] #4 Targeted verification and independent review are recorded; PR3166 is updated with auto-merge disabled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: adopt the existing disposable context-presentation memo and finite native ownership under ADR126/230; no new cache, TTL, storage/schema, permission or service boundary. 1. Retain original tail samples and separate original callback/worker timing: transcript live-policy read568ms during leave, styling215ms before worst input, config/workspace waits on attach. 2. RED/GREEN actual transcript tests pin zero unused live-policy/native config entry and publication reopening its unchanged gate. 3. Reuse the already-wired finite context inputs for qualified original callbacks before the transcript gate, include effective-memory publication in that gate, and leave custom/partial callbacks synchronous. Do not await context completion or stall streaming. 4. Verify cold/held-worker streaming progress, memory banner parity, source/owner drift, original custom affinity and native cancellation/retirement. 5. Run serial preceding-head control/current ABBA with independent review, preserve adverse outcomes, update PR3166 with auto-merge off and remove exact owned fixtures.

### 2026-10-31 current-dev integration

ADR required: no.
ADR path: N/A; existing ADR-126 and ADR-230 apply.
Reason: rebase the reviewed display scheduling work onto current native admission, source ownership, config-lock and kernel callers without changing those contracts.

1. Rebase original PR head 34f88d6322ee2819d3e189e97eda5bc0e3f419f1 onto dev b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4 in the isolated PR3166 worktree. Resolve overlapping implementation at current defining owners.
2. Preserve finite native retirement, custom-callback affinity, exact drafts and attachment authority, measured Current cost on unavailable tokenizers, conservative CSS matching and deferred hidden Home reads.
3. Run focused native Windows checks with genuine private config/temp roots and targeted static checks. Review any diagnostic inventory changes individually; do not weaken behavioral guards or timing bounds. Hosted CI and size limits are waived by the user's 2026-10-31 backlog instruction.
4. Record current-base results and unresolved historical performance/safety qualifications, then obtain fresh independent review before any root-agent publication or merge. AC2 and existing In Progress qualification remain open unless directly established.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented qualified transcript reuse of existing finite context inputs with source-identity invalidation; original custom-reader cadence and storage checks remain. Product commit9220460203. 104 distinct targeted checks passed; three connected tab-publication controls remain unqualified (also fail at different stages on exact original sources), so AC2 and task status remain open. Serial ABBA and independent review recorded in MEMORY_RESULTS.md: no consistent overall speedup; median input177.02 to186.68ms, attachment1023.92 to1038.45ms; worst input203.79 to247.49ms, worst attachment1973.95 to1590.37ms. Original startup policy call479ms to zero updated; neither settled return trace calls it. PR3166 update prepared with auto-merge off. ADR required:no under existing ADR126/230; no storage/schema or authority change.
<!-- SECTION:NOTES:END -->
