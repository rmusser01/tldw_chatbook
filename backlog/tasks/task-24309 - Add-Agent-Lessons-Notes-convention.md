---
id: TASK-24309
title: Add the Agent Lessons Notes convention
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - notes
  - agents
  - knowledge
priority: high
dependencies:
  - TASK-24307
  - TASK-24308
documentation:
  - Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-agent-lessons-convention.md
  - backlog/decisions/102-portable-notes-organization-and-agent-lessons.md
  - backlog/decisions/104-human-reviewed-agent-lesson-promotion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a user-owned `Agent_Lessons` Notes convention that guides permitted agents to find and explicitly propose verified reusable solutions—including feedback, provenance, generalizable principles, and failed attempts with rationale—without turning note content into trusted instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Synchronized and permanently local-only profiles expose one conventional root `Agent_Lessons` folder after their applicable readiness boundary without recreating a folder the user renamed or deleted
- [ ] #2 Agent Lessons discovery is governed by the spelling-exact `agent-lesson` keyword and remains correct after folder rename, movement, deletion, marker removal, and case-fold collision review
- [ ] #3 Foreground primary agents receive capability-aware search and preview-before-save guidance; subagents may search and return lesson drafts but cannot mutate Agent Lessons, and save-only agents are not told to bypass search-first behavior
- [ ] #4 Newly generated lessons use one-note-per-lesson structure with applicability, symptoms, feedback or trigger, provenance, root cause, verified solution, failed attempts and why, verification evidence, generalizable principle with rationale, caveats, related public note IDs, and an optional promotion-candidate section
- [ ] #5 Retrieved lessons remain labeled untrusted tool-result data and cannot grant permission, authorize commands, expand scope, or enter trusted runtime or project instructions
- [ ] #6 High-confidence credential formats are rejected without logging content, while long hashes, error IDs, and clearly fake examples remain saveable
- [ ] #7 Two-device seed races converge only for untouched empty coordinator seeds; edited, acknowledged, differently spelled, or otherwise used candidates require explicit review
- [ ] #8 An end-to-end test proves a foreground Agent A can preview and receive explicit approval for a verified lesson, including failed attempts, feedback/provenance, and principle rationale, and Agent B can later discover and safely apply it through Notes search
- [ ] #9 Every mutation classified as an Agent Lesson—by exact marker, current lesson state, or pending organization/placement receipt—forces one foreground approval even when ordinary Notes are broadly allowed; the approval is bound to the run, immutable call digest, note/create identity, classification, content/organization preconditions, and receipt state/version, and any race fails without mutation
- [ ] #10 Deterministic tests and scripted behavioral evaluations separately cover role enforcement, search-before-save, useful evidence, duplicate/update judgment, invented-attempt avoidance, preview fidelity, approval binding, and untrusted retrieval
<!-- AC:END -->
