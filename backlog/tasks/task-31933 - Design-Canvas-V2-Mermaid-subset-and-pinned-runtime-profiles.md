---
id: TASK-31933
title: Design Canvas V2 Mermaid subset and pinned runtime profiles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 18:16'
updated_date: '2026-09-06 18:22'
labels:
  - canvas
  - design
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/Canvas/V2_MERMAID_SPIKE.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the first offline diagram library experience without weakening Canvas isolation or losing revision reproducibility and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Written spec incorporates the seven approved review corrections and defines the supported Mermaid subset and exclusions.
- [x] #2 A canonical ADR and durable spike evidence explain runtime profile ownership, security refusal, and compatibility trade-offs.
- [x] #3 The design specifies measurable qualification gates for native and served delivery, resource limits, lifecycle, and archives.
- [ ] #4 The written design is self-reviewed and the user approves it before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
Reason: Extend ADR-121 with a pinned library/runtime boundary, exact-profile portability, and security refusal policy.

1. Verify the merged Canvas contracts and preserve the disposable spike findings as bounded evidence.
2. Draft the architectural spec and proposed ADR incorporating all seven approved review corrections.
3. Self-review scope, ambiguity, contradictions, links, task hygiene, and documentation diff.
4. Commit only design documentation and request written-spec approval before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted the Canvas V2 Mermaid subset spec, proposed ADR-124 extending ADR-121, retained spike findings, and ADR index entry. Incorporated security-first profile refusal, honest save/preview outcomes, aggregate quotas, explicit semantic admission, centralized profile selection, once-before-script lifecycle, and bounded layout/fidelity promises. Existing runtime_profile storage and archive format suffice; no runtime code, schema, dependencies, or security limits changed.

Self-review checked contradictions, scope, placeholders and error/recovery semantics. Local-link and fence validation passed for the three new documents (14 links); Backlog guard passed across 3383 task files. Product tests were not run for this documentation-only change; the spike results are explicitly historical, not V2 qualification. Final staged whitespace/scope verification precedes commit.

Awaiting user review of the written spec, including proposed numeric admission ceilings. AC4 remains open and this task remains In Progress; implementation planning has not begun.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

At creation on 2026-09-06 the CLI assigned 31901, already claimed by
`Preserve-Loguru-capture-sinks-across-app-mount` on a remote branch. Before
attaching design work, this new task moved to 31933, above the freshly swept
local/remote-ref maximum of 31932 and worktree maximum of 31861. Subsequent CLI
edits dropped this nonstandard section, so it was restored after the final edit.
