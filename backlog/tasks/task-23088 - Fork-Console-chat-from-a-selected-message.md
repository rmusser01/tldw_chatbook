---
id: TASK-23088
title: Fork Console chat from a selected message
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-26 21:00'
updated_date: '2026-08-26 21:00'
labels:
  - console
  - chat
  - ui
  - persistence
references:
  - Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md
  - Docs/superpowers/plans/2026-08-26-console-chat-fork.md
  - backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user create and immediately open an independently owned Console chat copied through one selected stable message while leaving the source chat and all of its live and durable state unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eligible selected USER and ASSISTANT messages expose Fork immediately before Regenerate plus the `f` action, and a compact naming dialog clearly identifies the boundary, saved or temporary destination, exclusions, validation, progress, cancellation, and degraded-success recovery.
- [ ] #2 Confirming copies exactly the canonical active lineage through the selected boundary and its fenced visible text or generated-image choices with fresh mutable ownership, while off-path, later, display-only, unsettled, unsaved-durable, and unsupported state is rejected or excluded as designed and the source remains byte-for-byte and live-state unchanged.
- [ ] #3 Durable forks commit conversation ancestry, messages, supported sidecars, active leaf, policy, governed citation owner links, and sanitized project context atomically before publication; temporary forks remain detached and sanitized, and a non-ephemeral source without durable IDs produces a saved independent-root fork without persisting the source.
- [ ] #4 Forks preserve declarative Workspace, model, role, Library, RAG, and project-instruction selections without copying scratch, approvals, permissions, resolved instruction bodies, continuations, recovery, derived context, usage, tool activity, or ephemeral video authority; citation and media degradation remains truthful.
- [ ] #5 One preallocated conversation or session identity makes retries idempotent, precommit failure creates nothing, and postcommit publication or activation failure identifies and reopens the already-created fork without duplication.
- [ ] #6 The USER and ASSISTANT action row uses the approved stable direct actions and labelled `More…` menu with captured message targeting, safe teardown, and deterministic focus fallback at 80x24 and wider production-shaped layouts.
- [ ] #7 Targeted domain, real-SQLite persistence, authority, media, cancellation-race, action/menu, modal, reload, and live local TUI verification pass, and Console user documentation describes the boundary, temporary behavior, shortcut, exclusions, and video/citation caveats.
<!-- AC:END -->

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

Reason: ADR-092 already governs the durable copy, identity, authority, and publication boundaries; this task implements that accepted contract without a schema migration or new ADR.

1. Define the pure allowlisted fork projection, title rules, eligibility, and sanitized project-instruction contract.
2. Fence and revalidate the canonical active-lineage prefix plus selected generated-image state, then stage fresh independent ownership without source mutation.
3. Add one idempotent real-SQLite bundle for ancestry, messages, supported sidecars, policy, governed citations, project context, and active leaf.
4. Add the direct Fork action, captured-target More menu, media-card controls, compact six-state modal, and cancellable controller orchestration.
5. Verify atomic failure, cancellation races, reload, layout/focus, temporary promotion, source immutability, and the provider-free live journey; update user docs and task evidence.

Detailed TDD steps and commands: `Docs/superpowers/plans/2026-08-26-console-chat-fork.md`.
