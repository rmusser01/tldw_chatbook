---
id: TASK-31937
title: Compile and execute declarative Canvas diagrams
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 22:12'
updated_date: '2026-09-07 03:34'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31936
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect inert Mermaid declarations to one transactional QuickJS startup before authored scripts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 V1 wire output remains unchanged and V2 plans preserve exact source identity with closed diagram records and verified profiles.
- [x] #2 All declarations render once before authored scripts with shared byte, time, memory, DOM and patch limits and no partial startup commit.
- [x] #3 Renderer and worker reject invalid output and stale failures while preserving zero-egress bootstrap acknowledgement and private controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing V2 source-preservation and closed-schema compiler tests; implement bounded single-parse profile admission.
2. Integrate verified inert library data and private QuickJS diagram handles into the existing single startup budget and transaction.
3. Exercise real worker/renderer startup, malformed wire, atomic failure, private handles and zero-egress regressions; regenerate assets and self-review.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of the approved profile wire and transactional runtime boundary; no new privileges.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented closed V2 plans and one-parse profile-aware preparation; exact verified source, manifest, declaration identities and evaluated-script admission are independently checked at compiler, renderer and worker boundaries. V2 owns separate worker/renderer files and immutable manifest closure; V1 worker/renderer/manifest match base deae88b57 byte-for-byte. Private QuickJS handles prepare all diagrams, allocate through the existing typed virtual DOM, then run authored scripts in one startup deadline/transaction. Unexpected engine exceptions remain bounded generic failures rather than elapsed-time quota claims. Added real-browser transactional, corruption, four-mixed-diagram and private-handle tests; existing zero-egress regressions passed. Main affected gate: 220 passed, 2 unavailable optional-browser skips; final changed-subset gate: 65 passed; final compiler/scheduling gate: 106 passed. Baseline RequestsDependencyWarning and 10 pre-existing Ruff findings in compiler/test_runtime_assets reproduced directly from base; all other changed Python files lint clean. Mermaid offline rebuild now carries both V2 runtime files and reproduces exact packaged outputs. Existing ADR-124 and V2 compatibility document describe separate closures and private parent data envelopes. Candidate remains disabled/default null; product parent delivery and complete qualification remain separate work. Status intentionally In Progress for independent controller review.

Review round1: closed the worker pre-VM admission gap for generic plan collections, text budgets, node/tag/namespace/text/attribute schemas, asset identities/base64/raster metadata and nested CSS rule quotas. Diagram installation now removes the original text children through existing typed mutations, preserving virtual/native textContent agreement without changing the global V1 facade. Added real-browser post-renderer corruption regressions, exact Tea target/descendant text comparison and valid CSS/four-raster-format admission. RED: 15 failed/1 passed; final focused browser GREEN: 18 passed; requested compiler/V2 compiler/assets/V2 browser gate: 149 passed, 1 existing warning, no skips. Regenerated V2 hashes; V1 worker/renderer/manifest remain byte-identical to dbff84e19. Renderer comments now describe inert source transport accurately. Status remains In Progress for follow-up independent review.

Independent review and scoped round1 re-review passed. Both Important findings addressed; no new Critical/Important breakage. Generic worker admission and virtual/native diagram text agreement verified. Existing ADR-124, per-profile closure and private-envelope contract retained. Production ownership/delivery and release qualification remain in their assigned follow-on tasks.
<!-- SECTION:NOTES:END -->
