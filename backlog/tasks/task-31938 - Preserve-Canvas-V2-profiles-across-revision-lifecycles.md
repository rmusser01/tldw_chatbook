---
id: TASK-31938
title: Preserve Canvas V2 profiles across revision lifecycles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:13'
updated_date: '2026-09-07 03:34'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31937
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep exact runtime semantics across updates, branches, temporary promotion and archive portability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All production creation, update, rename, historical and replay paths use one profile resolver and preserve optimistic parent checks.
- [x] #2 Temporary and durable histories retain exact profiles through atomic commit, rollback, cancellation and promotion without new storage or sync.
- [x] #3 Real conversation and Chatbook export-import preserve source and graphs; unknown and revoked profiles remain inert and archives cannot install runtime bytes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing real-controller profile transition tests and integrate one captured snapshot and prepared-plan identity admission.
2. Preserve exact profiles through durable/staged creation, update, rename, branch/replay and post-compilation ownership fencing.
3. Verify temporary atomic promotion/close and real conversation/Chatbook export-import with inert unavailable profiles and rollback; self-review and targeted static checks.
4. Carry independently computed profile selection in an internal preparation value through existing owners; compare legacy offered plans against fresh preparation. Keep public compiler/tool contracts unchanged. Repository and provider projection hardcodes are included where real-controller/provider regressions demonstrate they prevent exact-profile ownership.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of approved profile ownership and archive contracts; schema68/archive3.0 and sync exclusion remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented exact-profile preparation and metadata through Console, durable and
temporary ownership, provider result projection and native stored reads. Internal
preparation binds independently computed selection to source/parent/snapshot;
legacy offered plans cannot choose their own profile. Existing post-compile
ownership, replay and stale-parent checks remain. Metadata-only rename preserves
unavailable source/profile; HTML update and preview refuse unavailable profiles.

Repository validation now treats safe bounded profile IDs as storage data; the
existing owners enforce execution admission. This additional repository/provider
scope was required by observed real transition failures. Metadata annotations now
describe stored profile strings without changing public tool or wire schemas.

Real single- and multi-conversation ChatbookCreator/ChatbookImporter archives retain
V1, candidate V2, sibling, renamed and unavailable histories, tombstones and
remapped origins. Source/manifest/insert/commit failure tests retain atomicity.
Temporary promotion also tests a failure after Canvas writes followed by exact
retry, origin remapping, restart replay and session disposal. Archive bytes cannot
install a profile. Legacy plain text/JSON exports are not Canvas graph archives.

Verification: named ownership/archive/scheduling/provider suites passed 378 tests
with no skips. Added direct profile-ID storage tests passed 8; final affected
controller/native/scheduling/service run passed 178 with one source-exception
context regression, then the corrected service suite passed all 79 and both exact
sanitization regressions passed. Ruff has no new findings; two existing findings
and four whole-file formatting baselines were reproduced at BASE652834bfb.
Updated V2 compatibility docs and the evidence lesson with the actual provider
projection incident. Self-review covered snapshot/source admission, ownership
fences, archive inertness, transaction identity and source-free failures.

ADR: existing ADR-124 (extends ADR-121), direct implementation; schema68,
archive3.0, frozen runtime assets and sync exclusion are unchanged. Candidate
admission remains disabled. Status remains In Progress for independent review.
