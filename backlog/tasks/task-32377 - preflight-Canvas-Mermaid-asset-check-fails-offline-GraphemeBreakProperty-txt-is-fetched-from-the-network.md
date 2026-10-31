---
id: TASK-32377
title: preflight Canvas Mermaid asset check fails offline (GraphemeBreakProperty.txt
  is fetched from the network)
status: Done
assignee:
- '@codex'
created_date: 2026-09-11 01:55
labels:
- tooling
- ci
dependencies: []
priority: low
updated_date: 2026-10-18 05:01
---

## Description

`./scripts/preflight.sh` reports "Canvas Mermaid assets do not reproduce: declared input is unavailable: GraphemeBreakProperty.txt" whenever `scripts/vendor_canvas_mermaid.py` cannot fetch that input from unicode.org; the failure is identical on untouched branches, so it masks real drift and forces every reader to re-diagnose it. Seen across all four approval-card wave worktrees on 2026-09-11.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The Canvas Mermaid check either vendors its Unicode inputs in-repo or reports "skipped: input unavailable" as a distinct non-failing status when offline
- [x] #2 preflight's summary distinguishes an unavailable input from a reproduction mismatch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/238-offline-authenticated-canvas-build-inputs.md
Reason: same repository-owned pinned input decision as the overlapping TASK-32380; no separate bypass or skip policy.
1. Verify that the TASK-32380 offline default satisfies both original outcomes: original Unicode inputs are available from the repository and missing-input diagnostics remain distinct from output drift.
2. Reuse the real offline reproduction, missing/corrupt-input controls and shared preflight evidence. Record this overlap and close both tasks without introducing a second implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed as the overlapping preflight manifestation of TASK-32380. The eight original manifest-authenticated build inputs now live in the repository; the default checker and preflight rebuild offline without a skip or network fallback. Missing/corrupt declared inputs retain input-specific diagnostics distinct from generated-output drift. The actual network-refused reproduction and missing/corrupt/override controls pass; final affected CI selection: 137 passed, exit 0, and all derived-artifact preflight checks passed. Independent review found no remaining findings. Scoped Ruff and changed checker/test format checks pass; unrelated queue-contract baseline formatting is unchanged. ADR-238 and the shared build/architecture docs record the decision. No second implementation, full-suite or fresh hosted-CI claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
