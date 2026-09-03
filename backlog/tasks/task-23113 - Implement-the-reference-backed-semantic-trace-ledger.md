---
id: TASK-23113
title: Implement the reference-backed semantic trace ledger
status: Done
assignee: []
created_date: '2026-08-28 15:21'
labels:
  - console
  - storage
  - privacy
  - tracing
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-28-console-reference-backed-semantic-trace-ledger-design.md
  - >-
    Docs/superpowers/plans/2026-08-28-console-reference-backed-semantic-trace-ledger.md
  - backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
  - >-
    backlog/tasks/task-23026 -
    Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace repeated Console exchange-capture transcripts with the ADR-097 reference-backed semantic trace ledger so saved conversations remain the ordinary-content source of truth while provider-only semantics, edits, forks, privacy projections, and legacy traces remain coherently inspectable without quadratic storage growth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] Saved conversation revisions, not copied transcript rows, are the source of truth for ordinary provider-visible message content.
- [x] Provider-only context, request metadata, call boundaries, responses, retries, failures, and tool-loop events remain durably and coherently inspectable.
- [x] Edits, regenerations, forks, deletions, and temporary-chat promotion preserve explicit semantic lineage without mutating historical trace records.
- [x] Safe and Full viewer profiles operate on the same immutable trace, enforce credential filtering, support optional PII masking, and never redact the ordinary saved transcript.
- [x] Existing exchange captures remain readable through isolated legacy snapshot surfaces and become reclaimable through trace-graph garbage collection.
- [x] The 200-turn storage, latency, migration, and compaction-heavy acceptance gates in ADR-097 pass before normalized capture becomes the default.
- [x] Bounded custom PII regex execution and automatic physical SQLite compaction ship only after the core ledger gates pass.
- [x] Lossless chunk-row encoding remains independently tracked by TASK-24206 and is not required to complete this program.

## Implementation Plan

1. Deliver TASK-23113.1 through TASK-23113.9 in dependency order, keeping normalized writes gated until privacy and release checks pass.
2. Deliver the bounded custom-regex and physical-compaction children only after the core gates are green.
3. Use test-driven implementation plus independent specification and code-quality review for every plan task.
4. Complete the ADR-097 verification matrix, documentation, and Backlog hygiene before closing the umbrella.

ADR required: yes
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: ADR-097 defines the storage, privacy, provider-runtime, deletion, and rollout boundaries implemented by this umbrella.

## Implementation Notes

Completed the ADR-097 program through TASK-23113.1–TASK-23113.11. The delivered ledger keeps ordinary conversation content reference-backed, records immutable provider-call semantics and lineage, preserves coherent edits/forks/promotion, applies credential-safe Safe/Full projections with optional PII masking, normalizes legacy captures, reclaims unreachable trace graphs, and performs bounded custom-regex masking plus automatic SQLite compaction.

Each child task is Done with all acceptance criteria checked and its focused verification recorded. The release child proved the 200-turn linear-growth and latency gates before enabling normalized capture by default; the later privacy and maintenance children retained those gates while adding subprocess isolation and physical compaction. The final compaction PR passed its focused 206-test matrix, direct post-rebase regression checks, repository guardrails, and all required GitHub checks before merge.

ADR check: ADR-097 remains the canonical decision; no new ADR is required for this administrative closeout. Lossless token-delta chunk-row encoding remains deliberately outside this program and independently tracked by TASK-24206.
