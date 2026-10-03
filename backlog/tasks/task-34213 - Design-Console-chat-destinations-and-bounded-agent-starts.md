---
id: TASK-34213
title: Design Console chat destinations and bounded agent starts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 17:54'
updated_date: '2026-10-02 23:37'
labels:
  - console
  - agents
  - design
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how a primary Console agent creates a chat in its own workspace or casual scope, prepares a draft or starts bounded background work, and preserves destination defaults, scoped approvals, and recoverable outcomes. Record the approved behavior in a repository spec and canonical ADR before implementation planning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The written spec covers both destinations and both modes, destination defaults, explicit instructions overrides, and scoped session approvals.
- [x] #2 A canonical ADR defines conversation ownership, shared automatic budgets, machine-origin start authority, persistence, and recovery boundaries.
- [x] #3 Spec self-review confirms complete requirements, consistent lifecycle rules, valid local links, and a focused verification strategy.
- [x] #4 The user reviews the written spec before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md
Reason: Extends the agent-facing chat-creation contract and adds shared automatic-budget lineage, durable launch authority, and chat-draft recovery semantics.

1. Record the approved destination, mode, assistant, instructions, approval, and recovery behavior in Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md.
2. Write the canonical ADR and link the existing conversation, scratch, defaults, and automatic-work decisions.
3. Review the spec for contradictions, missing lifecycle cases, placeholders, valid links, and focused verification.
4. Obtain written-spec review before invoking writing-plans; keep application implementation outside this design task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recorded the approved Console new_chat extension in Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md and accepted ADR-211 in backlog/decisions/211-console-chat-destinations-and-bounded-starts.md; updated the ADR index.

The design covers workspace/casual destinations, draft/start modes, destination defaults with explicit instructions overrides, destination-and-mode session grants, shared allowance roots with per-conversation run ownership, trusted machine-origin admission, and durable draft/restart recovery. Application implementation remains outside this design task.

The user approved the written spec and requested another review before proceeding. Source-backed review clarified manual-send priority before busy guards, the exact acceptance/cancellation cutoff, root-wide pause/clock/uncertainty mutations, explicit version-2 handoffs with legacy decoding, paused preflight refusal without a second dialog, and exact-attempt receipts across the two databases. These preserve the approved behavior.

Validation passed for 19 local links, required contract/lifecycle/recovery fields, placeholders, and git diff whitespace checks. No application test run was needed for documentation-only changes. All four acceptance criteria are complete; implementation planning follows the approved spec and ADR-211.

Renumbering provenance: Backlog CLI assigned TASK-33167. A scan of all locally available Git refs and 107 worktrees found task IDs through 33801, so this design task moved to TASK-33802 before any inbound references were written.

Planning identified the remaining conversation-storage dependency: dispatch checkpoints currently restrict origin to manual/queued, so the spec now explicitly requires a conversation-database migration alongside AgentRunsDB. The implementation plan is saved in Docs/superpowers/plans/2026-10-02-console-chat-destinations-and-starts.md, with the foundation implementation task and the feature implementation task created for planning; their implementation acceptance criteria remain unchecked. Plan review checked source owner boundaries, exact test/guide paths, signatures, 22 local links and seven syntactically valid Python snippets. Application code and runtime tests remain outside this completed design task.
<!-- SECTION:NOTES:END -->

Current-dev publication identity: 34213 replaces unmerged 33802. The landed TASK33802 census keeps its identity; the design/foundation/feature chain moved together to preserve dependency ordering. Historical QA retains original identifiers and bytes. Current evidence: Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration/README.md.
