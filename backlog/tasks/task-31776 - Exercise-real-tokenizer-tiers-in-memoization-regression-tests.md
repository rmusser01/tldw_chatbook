---
id: TASK-31776
title: Exercise real tokenizer tiers in memoization regression tests
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 22:59'
updated_date: '2026-09-05 23:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The cache-clear regression spies only on the character fallback, so it fails with the current bundled tokenizer; the growing-history guard can falsely pass with zero observed work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cache-clear and append-only history checks observe positive real work on character and tiktoken tiers
- [x] #2 Cache bypass or broken invalidation fails the tests while complete token tests and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced cache-clear failure and trace tier selection against ADR-093.
2. Parameterize both recomputation guards over the real character and bundled-tiktoken functions with explicit tier selection and automatically restored spies. Tighten growing-history accounting to the independently counted unique inputs.
3. Verify complete token files and a cache-bypass mutation, then record results.
ADR required: no
ADR path: backlog/decisions/093-offline-tiktoken-runtime-assets.md
Reason: Test-only repair exercising the existing tier and memo contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both cache guards now parameterize real character and bundled-tiktoken work. The growing-history guard requires exactly 204 computations rather than accepting zero under a loose ceiling. A process-local cache-bypass mutation failed all four variants as intended (20400 versus 204 computations for growing history). Review caught that the tiktoken wrapper could silently fall back; this is now rejected explicitly and independently re-reviewed. Complete token files first passed 63 tests; final seven-file Chat selection passed 205 tests. Whole-file Ruff and formatter pass. No production token change; ADR-093 remains unchanged. Final XML: /private/tmp/tldw-current-chat-repair-final.xml; mutation XML: /private/tmp/tldw-31776-cache-bypass-mutation.xml. The combined summary selection retains a preexisting aggregate descriptor-growth warning of 209; no resource-closure claim.
<!-- SECTION:NOTES:END -->
