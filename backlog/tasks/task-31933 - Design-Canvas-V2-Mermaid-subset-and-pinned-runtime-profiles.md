---
id: TASK-31933
title: Design Canvas V2 Mermaid subset and pinned runtime profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 18:16'
updated_date: '2026-09-06 22:20'
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
- [x] #4 The written design is self-reviewed and the user approves it before implementation planning begins.
- [x] #5 The revised spec defines restart-required policy updates, complete profile compatibility, assistant subset guidance, and bounded geometry and typography.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
Reason: Extend ADR-121 with a pinned library/runtime boundary, exact-profile portability, and security refusal policy.

1. Verify merged Canvas contracts and retain bounded spike evidence.
2. Draft the spec and proposed ADR with the seven original corrections.
3. Incorporate the four additionally approved review findings: restart-required policy updates, engine/facade/Unicode profile identity, assistant guidance, and geometry/typography bounds.
4. Self-review consistency, scope, links, task hygiene and documentation diff; commit documentation only.
5. Request review of the revised written spec before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted the Canvas V2 Mermaid subset spec, proposed ADR-124 extending ADR-121, retained spike findings, and ADR index entry. Incorporated security-first profile refusal, honest save/preview outcomes, aggregate quotas, explicit semantic admission, centralized profile selection, once-before-script lifecycle, and bounded layout/fidelity promises. Existing runtime_profile storage and archive format suffice; no runtime code, schema, dependencies, or security limits changed.

Self-review checked contradictions, scope, placeholders and error/recovery semantics. Local-link and fence validation passed for the three new documents (14 links); Backlog guard passed across 3383 task files. Product tests were not run for this documentation-only change; the spike results are explicitly historical, not V2 qualification. Final staged whitespace/scope verification precedes commit.

Awaiting user review of the written spec, including proposed numeric admission ceilings. AC4 remains open and this task remains In Progress; implementation planning has not begun.

Follow-up design review (2026-09-06): incorporated all four approved findings into the spec and ADR-124. Packaged policy updates now require complete host/server restart with snapshot mismatch refusal; existing explicit disable remains immediate containment. Profiles pin engine/facade/plan and Unicode compatibility, with new identities for changed pinned inputs and separate nonsemantic security-validation identity. Added requirements for bounded profile-aware assistant guidance, executable example fixtures and source-free repair hints, plus geometry/area ceilings, explicit typography, scrolling and CSS-override limits.

Validation: documentation whitespace check passed; 14 local links, code fences, table columns and placeholder checks passed across the three design/evidence documents; Backlog guard passed across 3383 task files. Only spec, proposed ADR and design task changed; no product tests or implementation work. AC4 remains open for review of the revised written spec.

User approved the revised written design on 2026-09-06. ADR-124 is Accepted. The implementation handoff plan covers eight independently testable slices with exact integration owners, red/green examples, shared constraints and release gates; implementation tasks remain To Do. Documentation link/fence/table/placeholder checks and Backlog uniqueness/path validation passed. Product tests, performance and runtime security qualification are not applicable to this documentation-only design closeout and remain explicit implementation gates; no application code, dependencies, schema or runtime permissions changed. No implementation has started.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

At creation on 2026-09-06 the CLI assigned 31901, already claimed by
`Preserve-Loguru-capture-sinks-across-app-mount` on a remote branch. Before
attaching design work, this new task moved to 31933, above the freshly swept
local/remote-ref maximum of 31932 and worktree maximum of 31861. Subsequent CLI
edits dropped this nonstandard section, so it was restored after the final edit.
