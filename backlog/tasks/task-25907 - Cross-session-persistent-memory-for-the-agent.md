---
id: TASK-25907
title: Cross-session persistent memory for the agent
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 15:09'
updated_date: '2026-09-25 18:21'
labels:
  - agents
  - memory
dependencies: []
references:
  - backlog/decisions/182-personal-context-memory-evolution.md
documentation:
  - backlog/docs/personal-context-memory-roadmap.md
  - >-
    Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how Chatbook improves cross-session user memory through its existing Personal Context, Notes, and Agent Lessons owners. Preserve encrypted canonical records, user-reviewed learning, scoped authority, and the distinction between human facts, procedural lessons, and conversation summaries. The Muse diagram is comparative reference material, not an implementation specification. This task owns the architecture positioning and roadmap; application changes are independently tracked follow-ups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design is recorded as an ADR covering: what may be written, who writes it, when it is read, where it is stored, and the retention and deletion story
- [x] #2 The ADR states the privacy boundary explicitly, consistent with the local-private-data stance, and whether any of it is ever synced
- [x] #3 The ADR states how persisted memory interacts with the existing approval-gated lesson promotion rather than duplicating it
- [x] #4 A first implementation slice is scoped in the ADR as an independently shippable follow-up, not built under this task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/182-personal-context-memory-evolution.md (Accepted; extends ADR-102, with ADR-024 and ADR-052 as supporting contracts)
Reason: Position evidence inspection, derived-state retention, and future maintenance across existing memory owners without creating a fourth authoritative store.

1. Reconcile current code, the earlier owner deferral, and the image comparison.
2. Write a proposed positioning ADR and a reviewable first-release design, separating current behavior from proposed changes.
3. Create atomic Backlog follow-ups and a roadmap with dependencies, acceptance evidence, and explicit design gates.
4. Validate task IDs, criteria, links, and scope; self-review the draft.
5. Obtain review of the written design before preparing executable feature plans or changing application behavior.

Tracker: backlog/docs/personal-context-memory-roadmap.md
Design: Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DEFERRED by owner (2026-09-02) until Personal Context 02 (interviews + agent tools) lands. Re-scope recorded: chatbook already has three memory pillars — Notes (freeform), Agent-Lessons (human-gated learned facts -> AGENTS.md), and the in-flight Personal Context program (encrypted USER-profile core, plans 01-04 merged, append_personal_context live in agent_service). Any 25907 ADR should be a short POSITIONING decision over those pillars (no fourth store), mapping the four deferred rows onto them (curator -> later lessons/notes maintenance automation; external MemoryProviders + journey graph -> rejected or server-side). Do not design a parallel memory system.

2026-09-25 pre-implementation review: verified current code and corrected the design, proposed ADR-182, roadmap and six child task criteria. Unknown provenance history, request/expiry ownership, offline metric definitions, positive lexical matching and shipped-safeguard gates are explicit. Existing device-only disclosure and unscoped quarantine signals remain unresolved runtime gaps assigned to the disclosure design. An independent read-only re-review found the documentation contradictions resolved. Scoped validation passed for 10 task IDs, 59 child criteria and 31 local links across 14 owned files; global duplicate IDs are unrelated. No application code or runtime tests changed. Review report: Docs/superpowers/reviews/2026-09-25-personal-context-memory-preimplementation-review.md. Governing proposal: backlog/decisions/182-personal-context-memory-evolution.md. Ready for executable planning; task remains In Progress.

2026-09-25 architecture checkpoint complete: the user requested continuation after reviewing the corrected design. ADR-182 is accepted for the bounded first-release scope; future schema/disclosure/forgetting decisions remain separate. All four foundation criteria are satisfied by the accepted ownership, lifecycle/privacy limits, existing lesson boundaries and independently scoped follow-ups. Documentation verification passed across 15 owned files, including 10 unique task IDs and resolved local links. No application logic changed, so runtime tests and runtime lint are not applicable to this positioning task. Executable baseline plan: Docs/superpowers/plans/2026-09-25-personal-context-memory-baseline.md. Source review: Docs/superpowers/reviews/2026-09-25-personal-context-memory-preimplementation-review.md. Foundation completion does not complete any feature task.
<!-- SECTION:NOTES:END -->

## Planning status — 2026-09-25

The user resumed the memory comparison and requested a plan/tracker. The earlier
deferral above remains historical context; the no-fourth-store constraint is
preserved. The current description now reflects the existing Personal Context
implementation rather than the obsolete per-conversation-only assessment.

Accepted ADR: [ADR-182](../decisions/182-personal-context-memory-evolution.md).
Written design: [Memory evolution](../../Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md).
Program tracker: [Memory roadmap](../docs/personal-context-memory-roadmap.md).

The user endorsed the direction and requested a technical review before
continuing. The [review report](../../Docs/superpowers/reviews/2026-09-25-personal-context-memory-preimplementation-review.md)
records corrections to provenance, preview freshness, privacy claims,
evaluation, retrieval and consolidation gates. Six child tasks now have
tighter acceptance criteria. Existing model-disclosure and quarantine-signal
gaps remain explicitly open under the provider-disclosure design; no runtime
fix is claimed. The user requested continuation after review. This foundation
is now Done with all four criteria checked; the baseline task owns the next
execution plan and remains separate from overall feature completion.
