---
id: TASK-19501
title: Make Change Review opt-in per workspace
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - console
  - performance
  - privacy
dependencies: []
priority: critical
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stop ordinary Console chats from silently creating and retaining shadow Git history. Change Review must require explicit workspace consent, expose unavailable state honestly, initialize enabled roots without blocking chat, and disclose retained file-content history in Settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing or unreadable per-workspace Change Review state is disabled; only an explicit stored true enables tracking
- [x] #2 The global Change Review capability distinguishes enabled, disabled, and unavailable without treating a read/coercion failure as enabled
- [x] #3 Workspace reads return an opaque durable revision and compare-and-set writes reject stale or ABA toggle attempts without changing runtime state
- [x] #4 Admission, toggle publication, and initializer completion share one app-owned consent lock and obey the documented linearization rules
- [x] #5 Enabling initializes each canonical root in the background with preparing, ready, failed, and bounded retry states; chat never waits for initialization
- [x] #6 Disabled workspaces create no initial-snapshot or per-turn snapshot work
- [x] #7 Settings disables unavailable toggles and discloses shadow Git file-content retention, including that disabling does not erase existing history
- [x] #8 Real-database, mounted Settings, and barrier-controlled concurrency tests cover missing, failure, stale revision, admission/toggle ordering, and disable-reenable initializer ABA
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED registry and capability tests for tri-state reads, missing-row disabled behavior, monotonic opaque revisions, and state-plus-revision compare-and-set writes.
2. Add RED service tests for consent-lock admission, background readiness, failed retry, and disable/re-enable ABA rejection.
3. Implement the smallest typed Change Review consent/readiness service over the existing workspace table and initializer seam; do not add a schema migration.
4. Add RED mounted Settings tests for unavailable state, opt-in copy, retention disclosure, and background preparation/retry presentation.
5. Wire Settings and folder registration through the app-owned service, preserving the global master switch and preventing disabled snapshot work.
6. Run focused registry, Change Review, and Settings suites; mutation-check the default, failure, CAS, and ABA guards.

ADR required: yes
ADR path: `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md`
Reason: This changes privacy-sensitive shadow-content ownership, workspace consent, cross-module state, and lifecycle policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented explicit per-workspace consent with typed unavailable state,
opaque revisions, and compare-and-set toggles under one app-owned lifecycle
lock. Replaced the registry's unowned snapshot thread with a fixed bounded
daemon initializer pool, immutable turn admission, alias-only skipped-root
warnings, revision/generation-safe completion, bounded retry, and bounded
shutdown. Settings now renders the same revision-consistent service snapshot,
discloses shadow Git file-content retention, and keeps chat/tools available
through preparation or failure. No schema migration was needed; the design
follows ADR-077.

Verification covered real SQLite registry behavior, deterministic concurrency
barriers, mounted Settings flows, Console turn-context capture, bridge warning
isolation from `change_snapshots`, shutdown ordering, focused Ruff checks, and
whitespace validation.
