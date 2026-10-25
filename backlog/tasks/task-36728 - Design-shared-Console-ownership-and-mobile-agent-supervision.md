---
id: TASK-36728
title: Design shared Console ownership and mobile agent supervision
status: Done
created_date: 2026-10-25 16:01
assignee:
- '@codex'
labels:
- console
- web-server
- mobile
- design
documentation:
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 17:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a mobile interface to existing served Console chats and agents, remove the separate mobile data and execution island, and preserve running work across browser and VPN disconnects. Include direct subagent steering and safe consolidation of existing mobile history; native Console attachment follows in a later implementation phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A source-anchored spec defines canonical storage, one live served owner per chat, reconnectable process lifetime, and explicit-target remote commands without introducing another execution engine.
- [x] #2 The spec defines the approved Chats-first phone workflow, live transcript and decisions, main-agent controls, and direct subagent steering and Stop.
- [x] #3 The spec defines restartable legacy-mobile consolidation, identity/reference conflicts, preserved permissions, uncertain-command reconciliation, and targeted automated/live verification.
- [x] #4 A linked ADR records the replacement ownership contract and identifies the superseded parts of ADR-234 and ADR-236.
- [x] #5 The written spec has passed self-review for scope, contradictions, placeholders and local links, and the user has reviewed it before implementation planning begins.
- [x] #6 All six accepted design-review findings are resolved in the spec and ADR, with matching verification scenarios for independent phone inputs, execution ownership, imported automatic work, migration attribution, receipt expiry and terminal interaction after reconnect.
- [x] #7 The spec and ADR require lifecycle-owned private event draining before publication with no phone subscribers, preserve execution/control ownership under projection overflow, and define bounded snapshot recovery plus a matching no-phone/overflow acceptance scenario.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Replaces frontend-specific storage and execution ownership, changes served process lifetime and authenticated control contracts, and requires a recoverable data consolidation. Existing ADR-094, ADR-126, ADR-150, ADR-161 and ADR-233 remain governing boundaries.

1. Trace the existing Console runtime, served child lifetime, mobile engine/databases, control broker, decision host, and subagent commands.
2. Write the user-approved ownership, migration, mobile workflow, reconnect and verification contracts in the design spec and replacement ADR.
3. Self-review scope and failure semantics, validate source/local links and task metadata, and commit only this design's artifacts.
4. Present the committed spec for user review; keep product code and implementation planning pending that review.
5. Apply the six source-backed design-review corrections approved by the user, verify their coverage and local links, and obtain a focused independent review before implementation planning.
6. Apply the user-approved projection-overflow correction: move private event-drain lifetime to the retained host, define bounded projection reset without execution-owner loss, add targeted acceptance scenarios, and validate the revised documents before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the user-reviewed design spec and ADR-244, linked the ADR index, and added metadata-only supersession/amendment pointers to ADR-234/236. The design replaces frontend-specific storage and engine construction with canonical Console ownership, retained served hosts, exact-target mobile controls and recoverable legacy-history consolidation.

Applied the six initial design-review corrections approved by the user: command-owned phone text that preserves desktop composition; an executing-child/native fence for the physical agent_runs.db resource; quarantine of all imported automatic/result-delivery authority, including completed runs; one reviewed claim for a physical legacy source across configurations; authenticated five-minute command expiry with safe receipt retention and available Stop; and terminal-mode restoration plus interaction checks after reconnect. Each has explicit acceptance scenarios. Existing runtime admission, permissions, recovery and decision owners remain authoritative.

A subsequent source review found that the broker closes its private control channel on event-queue overflow while event draining begins only on first mobile use. The user approved correcting that lifecycle gap. The spec and ADR now require draining before child publication and without phone subscribers, bounded projection coalescing/drop with snapshot reset, preservation of the host/control channel/execution fence and authoritative command/decision outcomes, and unchanged shutdown on genuine owner loss. Added no-phone, mobile-disabled, slow-subscriber, forced-overflow and actual-control-loss acceptance scenarios. A focused independent review found no remaining issues in this correction.

Documentation validation checked 21 local links, task frontmatter/owned sections, placeholders and scoped whitespace; all passed. Design-only acceptance criteria are complete. Implementation planning is the next stage.

ADR required: yes; backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md records the approved contract. No product implementation plan or product code was written. No real user databases, provider calls, runtime tests, full-suite run, push or merge were part of this documentation task. Runtime behavior and migration remain unverified until implementation and targeted/live qualification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
