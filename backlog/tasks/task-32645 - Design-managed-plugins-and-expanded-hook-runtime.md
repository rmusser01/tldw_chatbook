---
id: TASK-32645
title: Design managed plugins and expanded hook runtime
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-15 17:39'
updated_date: '2026-09-16 00:55'
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
- [x] #5 The approved written-spec review gaps are resolved consistently across both specs and ADRs, with a concrete failure/control acceptance scenario for each.
- [x] #6 The MCP result, pending post-hook barrier, PreToolUse phase and credential renewal contracts are unambiguous and have failure/control acceptance scenarios in the specs and matching ADR decisions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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

### Approved written-spec amendments

1. Separate inspection from activation when a recognized extension loses dependency information.
2. Bind activation, selection, mappings and hook requirements into authenticated authority.
3. Specify complete recovery snapshots, marker identity and authenticated SQLite commit evidence.
4. Define required hook success/context, post-event failure checkpoints and master-switch behavior.
5. Define provisional SessionStart admission and MCP initialization prerequisites/cycle rejection.
6. Add aggregate v2 admission limits, fair scheduling and distinct notification/reaping deadlines.
7. Add failure/control acceptance scenarios, align both ADRs, verify the documentation and commit the amendment.

### Approved integration-contract amendments

1. Specify MCP result preservation, error-first validation and exact v2 normalization.
2. Establish pending required post-event barriers before the next model admission or normal settlement.
3. Classify all PreToolUse effect combinations and distinguish transformations from final-argument guards.
4. Authenticate stable credential bindings while allowing token renewal within unchanged authority.
5. Add targeted acceptance scenarios, align both ADRs, verify the documents and commit the amendment.
<!-- SECTION:PLAN:END -->

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

Written-spec review amendment: resolved all six approved findings in the two specs and ADR-162/ADR-163. Recognized malformed extensions preserve inspection while blocking activation with lost constraints. Authenticated authority now explicitly covers activation, selection, dependency/hook policy and execution/credential mappings. Recovery names a complete protected snapshot and marker tuple, with commit proof published only after durable SQLite commit; missing proof in that crash window requires reviewed recovery.

Hook definitions now distinguish required success from required nonempty context, retain requirements when hooks are disabled, and fence subsequent input after required post-event failure without replaying settled work. Provisional initialization uses ordinary authority and independently eligible connected MCP dependencies. Application-wide v2 execution/reservation/observation limits use fair admission and keep cleanup-pending children counted; notification and host-reaping deadlines are separate.

Added concrete failure and successful-control acceptance scenarios for each finding. Self-review aligned failure scopes, initialization timing, recovery evidence and budget units across the specs and ADRs. Documentation checks passed for four JSON examples, 34 local links, balanced fences, unresolved-marker absence and whitespace. No runtime tests were run: the scenarios describe required implementation evidence, not implemented behavior. Implementation planning remains the next phase after written-spec review.

Integration-contract amendment: resolved the four approved follow-up findings in both specs and ADR-162/ADR-163. MCP hooks preserve typed error/structured fields, reject errors before effects and accept only explicit bounded result forms. Required post-events establish pending checkpoints before next model admission or normal settlement; effect acceptance and checkpoint release occur together.

PreToolUse declarations now have exhaustive phase classification: mixed transform/deny handlers run once, final constraints require separate non-transforming guards, context-only handlers see final arguments and optional effect-free observers use the bounded queue. Credential authority uses stable account/issuer/endpoint/scope bindings and generations; ordinary verified token renewal preserves trust, while authority changes invalidate captured mappings.

Added failure and successful-control scenarios for MCP error payloads/normalization, pending checkpoint races, later transformers changing arguments, and credential renewal versus authority changes. Aligned both ADRs and identified the existing MCP content-only client projection as an implementation prerequisite. Document checks passed for four JSON examples, 35 local links, balanced fences, placeholder absence and whitespace. Runtime scenarios remain future implementation evidence; no runtime tests were run for this documentation amendment.
<!-- SECTION:NOTES:END -->
