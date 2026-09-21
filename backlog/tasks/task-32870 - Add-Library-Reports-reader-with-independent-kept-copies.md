---
id: TASK-32870
title: Add Library Reports reader with independent kept copies
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 21:57'
updated_date: '2026-09-20 08:18'
labels:
  - library
  - artifacts
dependencies:
  - TASK-32869
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make all local report copies discoverable through the existing Library reader layout, with an explicit Kept filter and durable access after Watchlist deletion. Keep generation, retention, and playback with their current owners. Governed by ADR-172 and Docs/superpowers/specs/2026-09-19-library-artifacts-design.md; execution is stage 1 of Docs/superpowers/plans/2026-09-19-library-artifacts.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reports default to All reports, Kept shows durable copies independently of Subscriptions, and imported source-ID collisions never hide or mislabel content.
- [x] #2 Every matching report is reachable through bounded pages; detail loads and actions are fenced to the current selected identity.
- [x] #3 Library geometry, field-first Escape, focus restoration, local scope, artifact-only onboarding, and truthful error recovery work with production CSS.
- [x] #4 Existing Keep, report export, Watchlists handoff, scripts, audio, and demo recovery capabilities remain accessible and targeted checks pass.
- [x] #5 A Keep conflict is refused inside the service before any parent or script mutation, including a concurrent-create conflict; compatible re-keeps preserve saved origin and still add complete scripts.
- [x] #6 Each page and exact-item locator reads coherent metadata and rows from an explicit read snapshot; coordinated WAL writes cannot split a response and borrowed transactions retain their owner.
- [x] #7 Enter focuses the reader, reader arrows preserve list selection, and Back or unconsumed Escape restores the selected list row before arrows navigate it again.
- [x] #8 Library suspend stops owned debounce timers and stale presentation effects while compatible data reads may settle without leaving Loading stranded on resume.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Reason: Implements the accepted Library browse and navigation contract. 1. Add coherent bounded report metadata reads, namespaced identity and exact target location. 2. Enforce service-owned Keep conflict checks before all writes. 3. Mount Reports and Kept in the shared Library reader with lifecycle and focus guards. 4. Verify targeted SQLite, Library, production CSS and live interaction evidence; record results. PR review correction: reproduce F6 pane cycling and adaptive-shell detection, wire the artifact shell into existing global focus/layout seams, route artifact storage calls through the existing finite worker boundary, and verify native handle retirement on success, error, cancellation and borrowed ownership before closing review.

Qodo review: reproduce and fix kept export metadata and validated audio probing; annotate the reader composition boundary and document the shared artifact contracts; verify real SQLite and mounted export behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Reports and Kept in Library’s shared adaptive reader with bounded snapshot pages, exact identities, durable export/scripts/audio/watchlist actions, and service-owned Keep compatibility guards. Real tests cover imported collisions, rollback, borrowed transactions, source deletion, delayed replies, query and focus, and stale-action Retry. Final reader regressions passed, including native Enter/Down/Escape. Behavior lives in focused catalog/controller/widget modules with thin screen wiring. ADR: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Evidence and explicit baseline limits: Docs/superpowers/plans/2026-09-19-library-artifacts-verification.md. User guide updated. No full suite requested; changed/new focused tests pass, added-line Ruff diagnostics are zero, generated CSS checks pass, and native private-profile TldwCli was verified. Existing recovery-initialization and repository-wide screen-size/workflows-style failures remain documented; their limits were not raised.

PR #2754 independent review: fixed finite artifact-worker cache retirement through the existing Library service boundary, adaptive-shell recognition, and F6/Shift+F6 pane/grip navigation. Narrow focus reveals the reader without a deferred callback that can override newer focus. Verification: 12 real SQLite lifecycle cases, seven production-CSS focus cases, 21 affected Reports/Chatbooks integration cases, six adjacent interaction cases, and five artifact-controller governance checks passed (overlapping focused runs). Final private native TldwCli F6/Shift+F6 at 160/64/50 columns exited 0. Independent reviewer accepted all findings as resolved; no full suite. Updated verification record and worker-lifetime lesson; ADR-172 remains governing.

Derived-artifact CI correction: reviewed every changed diagnostic statement against inventory commit 0e10b0b72b22 (four controller warnings with fixed action/type metadata; formatting-only share count and moved existing exception call). No new sink or path-privacy candidate. Regenerated Docs/security/production-diagnostic-inventory.json; the CI checker with --diff passed with no drift (610 owners, 14 sink files).

Qodo review: normalized kept export status/date before filename and document generation, used validated normalized audio paths for capability and playback, and documented/typed artifact contracts. Eight behavior regressions passed; affected artifact navigation/canvas cases passed (38), final shell navigation/Ctrl6 passed (49), source/sharing integration passed (83), and tour/controller checks passed (6). Fresh native Ctrl6 and F6 verification exited 0. Rebased onto dev de10a62e67124a2b21b78edf1a4887cea03ff139; diagnostic inventory has no drift. Ruff/format and added-line checks pass. ADR-172 and the updated verification record govern; no full suite.
<!-- SECTION:NOTES:END -->
