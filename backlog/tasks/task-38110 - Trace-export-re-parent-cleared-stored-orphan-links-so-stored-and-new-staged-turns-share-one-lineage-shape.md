---
id: TASK-38110
title: >-
  Trace export: re-parent cleared stored orphan links so stored and new staged
  turns share one lineage shape
status: To Do
assignee: []
created_date: '2026-10-30 05:55'
labels:
  - console
  - trace-export
  - follow-up
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the TASK-33621.8 review. When Trace export clears a stored link to an unrecorded event (reason reference_not_recorded), the event becomes a lineage root with no parent. A staged-Library turn sent after TASK-33621.8 has its context_attached event parented to the turn's message root (message:<id>) when the snapshot is derived. So one conversation can export with two lineage shapes for the same kind of turn, which makes shared bundles harder to read and compare.

TASK-33621.8 AC#3 allowed clearing; re-parenting was left out because the bundle's field-state vocabulary has no state for an inferred parent. This task decides how to express that (a new field state or reason, or clearing before snapshot derivation so the message-root fill applies) and makes the shapes uniform.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stored staged-Library turn and a new staged-Library turn export with the same parent shape for their context_attached events under all three profiles
- [ ] #2 Any parent the exporter infers is disclosed in the bundle (field state or provenance), never presented as observed
- [ ] #3 Bundles written before the change still import
- [ ] #4 A regression test covers a conversation holding both a stored orphan turn and a new staged turn
<!-- AC:END -->
