---
id: TASK-33013
title: Design Tools and Skills pack composition with managed plugins
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-27 15:14'
labels:
  - packs
  - skills
  - plugins
  - design
dependencies: []
references:
  - backlog/decisions/107-portable-tool-use-packs.md
  - backlog/decisions/162-managed-agent-plugins.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define portable sharing of a Tool policy profile with the skills and tool capabilities needed to use it, composing with ADR-162 package ownership and ADR-107 policy authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The requester approves a first milestone and an explicit choice between installed-capability references, included immutable package content, and later runtime extensions.
- [ ] #2 An approved specification defines export contents, dependency identity, inspection, installation, permission review, activation and missing-capability behavior.
- [ ] #3 Update, rollback, removal, provenance and licensing contracts compose with managed plugin ownership and never transfer trust grants or credentials.
- [ ] #4 A canonical ADR and reviewed specification record the decisions, verification cases and a dependency-ordered atomic implementation breakdown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing policy packs, skill import and managed-plugin contracts. 2. Clarify the portable capability milestone with the requester. 3. Compare approaches and review each design section. 4. Write and review the approved specification and canonical ADR. 5. After written-spec approval, create the implementation plan and atomic tasks. ADR required: yes. ADR path: to be allocated after design decisions, linking ADR-107 and ADR-162. Reason: new portable composition and cross-service lifecycle contract.
<!-- SECTION:PLAN:END -->
