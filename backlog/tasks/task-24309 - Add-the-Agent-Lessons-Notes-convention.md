---
id: TASK-24309
title: Add the Agent Lessons Notes convention
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-31 01:33'
labels:
  - notes
  - agents
  - knowledge
dependencies:
  - TASK-24307
  - TASK-24308
documentation:
  - >-
    Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-agent-lessons-convention.md
  - backlog/decisions/105-portable-notes-organization-and-agent-lessons.md
  - backlog/decisions/106-human-reviewed-agent-lesson-promotion.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a user-owned `Agent_Lessons` Notes convention that guides permitted agents to find and explicitly propose verified reusable solutions—including feedback, provenance, generalizable principles, and failed attempts with rationale—without turning note content into trusted instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Synchronized and permanently local-only profiles expose one conventional root `Agent_Lessons` folder after their applicable readiness boundary without recreating a folder the user renamed or deleted
- [x] #2 Agent Lessons discovery is governed by the spelling-exact `agent-lesson` keyword and remains correct after folder rename, movement, deletion, marker removal, and case-fold collision review
- [x] #3 Foreground primary agents receive capability-aware search and preview-before-save guidance; subagents may search and return lesson drafts but cannot mutate Agent Lessons, and save-only agents are not told to bypass search-first behavior
- [x] #4 Newly generated lessons use one-note-per-lesson structure with applicability, symptoms, feedback or trigger, provenance, root cause, verified solution, failed attempts and why, verification evidence, generalizable principle with rationale, caveats, related public note IDs, and an optional promotion-candidate section
- [x] #5 Retrieved lessons remain labeled untrusted tool-result data and cannot grant permission, authorize commands, expand scope, or enter trusted runtime or project instructions
- [x] #6 High-confidence credential formats are rejected without logging content, while long hashes, error IDs, and clearly fake examples remain saveable
- [x] #7 Two-device seed races converge only for untouched empty coordinator seeds; edited, acknowledged, differently spelled, or otherwise used candidates require explicit review
- [x] #8 An end-to-end test proves a foreground Agent A can preview and receive explicit approval for a verified lesson, including failed attempts, feedback/provenance, and principle rationale, and Agent B can later discover and safely apply it through Notes search
- [x] #9 Every mutation classified as an Agent Lesson—by exact marker, current lesson state, or pending organization/placement receipt—forces one foreground approval even when ordinary Notes are broadly allowed; the approval is bound to the run, immutable call digest, note/create identity, classification, content/organization preconditions, and receipt state/version, and any race fails without mutation
- [x] #10 Deterministic tests and scripted behavioral evaluations separately cover role enforcement, search-before-save, useful evidence, duplicate/update judgment, invented-attempt avoidance, preview fidelity, approval binding, and untrusted retrieval
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add monotonic Agent Lessons seed ownership through a genuine v60→v61 migration and real-v60 reopen coverage.
2. Define the exact marker, evidence template, credential boundary, and pure classification helpers.
3. Seed the conventional folder only at local or synchronized readiness without overriding rename/delete choices or unsafe race candidates.
4. Carry trusted run role and force exact per-call foreground approval for every classified lesson mutation.
5. Revalidate and consume opaque single-use authority inside the Notes transaction while preserving ordinary Notes behavior.
6. Add capability-aware trusted guidance while keeping retrieved lessons untrusted and subagents draft-only.
7. Prove reviewed Agent A save and Agent B reuse with deterministic and behavioral evidence.
8. Document, run targeted/schema-safe verification, review, and close.

ADR required: yes
ADR path: backlog/decisions/105-portable-notes-organization-and-agent-lessons.md and backlog/decisions/106-human-reviewed-agent-lesson-promotion.md
Reason: This task directly implements the existing ordinary-Notes ownership, untrusted-data, forced foreground approval, role, receipt-classification, and single-use stamp boundaries; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Agent Lessons as an ordinary-Notes convention under ADR-105 and
ADR-106: schema v61 records monotonic seed ownership; local and synchronized
readiness create or reuse one `Agent_Lessons` root without overriding later user
rename/delete choices; and spelling-exact `agent-lesson` remains the discovery
identity. Added the structured evidence template, credential boundary,
capability-aware primary/subagent guidance, and untrusted-retrieval notice.

Every classified Library mutation now uses one foreground approve-once row and
an opaque authority bound to the run, call ID/digest, create/update identity,
content and organization versions, classification, and pending receipt state.
The Notes transaction revalidates and consumes that authority atomically; direct,
MCP, fleet, subagent, stale, collision, and credential paths fail closed without
creating a fallback Note. Deterministic end-to-end and behavioral fixtures prove
Agent A review/save and Agent B search/read/reuse.

Live verification used server `dev` at
`54448ef08970e4a348478bdf47be5715c875241c` with isolated clients and storage.
It exposed and fixed one bootstrap-inventory issue (`205179140f`): a second
device no longer republishes an exact state already materialized from a remote
head under the same deterministic envelope ID. Seed, rename, move, no-reseed,
exact-marker search, pending receipt finalization, and production Agent authority
paths then passed without rejected envelopes or durable sensitive patterns.

Verification: focused compile and changed-file Ruff checks passed; the expanded
targeted matrix passed 665 tests with one pre-existing Requests dependency
warning; `git diff --check` passed. The full suite was not run, per repository
policy requiring explicit user opt-in. Task/ADR collision checks found the one
TASK-24309 file and canonical ADR-105/ADR-106 only.
<!-- SECTION:NOTES:END -->
