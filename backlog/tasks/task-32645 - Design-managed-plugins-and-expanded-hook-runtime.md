---
id: TASK-32645
title: Design managed plugins and expanded hook runtime
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-15 17:39'
labels:
  - design
  - plugins
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a reviewable plugin system and Git marketplace experience for Chatbook, including portable, Cursor and Codex interoperability and the shared hook runtime needed to support it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The plugin spec records all approved scope and review amendments with explicit ownership, compatibility, lifecycle, UI and verification contracts.
- [x] #2 The companion hook spec defines events, ordering, structured effects, authority, cancellation and bounded resource behavior.
- [x] #3 Canonical ADRs record the package and hook architecture and are linked from both specs and this task.
- [x] #4 Document checks and a self-review resolve placeholders, broken local links and contradictory requirements; the written specs are ready for user review.
<!-- AC:END -->

## Implementation Plan

1. Consolidate the six approved design sections and all review amendments.
2. Write the managed-plugin spec and companion expanded-hook-runtime spec with concrete contracts and resource limits.
3. Record package ownership and shared hook-runtime decisions in ADR-162 and ADR-163 and link all documents.
4. Self-review consistency, local links, examples and acceptance coverage; commit only this documentation set.
5. Present the written specs for user review before creating implementation plans.

ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: New package storage, trust, runtime and adapter boundaries plus shared lifecycle and UI contracts.

## ID allocation

Created through Backlog CLI, then renumbered from its uncommitted offer of TASK-32632 to TASK-32645. Fresh origin refs, reachable object paths and 46 worktrees showed an existing maximum of TASK-32644; ADR maximum was 161. Numbers remain subject to the normal pre-merge collision check.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote the reviewed architectural design as a main plugin spec and a companion
shared-hook spec. The documents preserve the approved partial interop,
workspace activation, immutable revisions, explicit trust, recovery,
revocation and UI ownership contracts. Added concrete native definitions,
limits and acceptance criteria. Direct MCP qualification explicitly covers
the current per-request protocol and legacy handshake profiles.

- [Plugin spec](../../Docs/superpowers/specs/2026-09-15-managed-plugins-design.md)
- [Hook-runtime spec](../../Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md)
- [ADR-162](../decisions/162-managed-agent-plugins.md) and
  [ADR-163](../decisions/163-expanded-console-hook-runtime.md), both Proposed
  pending written-spec review; added their index entries.

Self-review resolved ambiguous native variable declarations, total event
deadlines, protocol-era assumptions and copied foreign configuration boundaries.
Documentation checks validated four JSON examples, local links, fenced blocks,
unfinished-marker absence, whitespace and the exact two-row ADR index change.
The new task ID is unique and its filename is Windows-compatible. The global
Backlog ID guard still reports 38 pre-existing duplicate filename/frontmatter
IDs elsewhere in the working tree; none is TASK-32645.

Added a [Backlog hygiene lesson](../docs/lessons-backlog-hygiene.md) from the
final allocation scan: a Codex tree snapshot of this uncommitted task is the
same owner, not a conflicting ID claim. Fresh refs and 46 worktrees showed
no competing claim for TASK-32645, ADR-162 or ADR-163.

No product code changed and no runtime tests were run. Runtime acceptance,
cross-platform qualification and implementation plans remain future work.
Task stays In Progress for the brainstorming skill's written-spec review
checkpoint. Implementation planning starts after the user reviews these files.
<!-- SECTION:NOTES:END -->
