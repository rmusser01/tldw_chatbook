---
id: TASK-21281
title: Make Watchlists Reader permanent with independently collapsible panes
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23 22:11'
updated_date: '2026-08-24 17:15'
labels:
  - watchlists
  - ux
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md
  - backlog/decisions/042-watchlists-reader-first-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Watchlists Read screen's vertically stacked centre with a NetNewsWire-style permanent Reader and independently collapsible Navigation, Feed Items, and Inspector panes. Preserve the shipped reading, Smart Feed, search, refresh, pagination, selection, and item-action behavior while making manual layout preferences durable and responsive/Article Focus overrides transient.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read always mounts Reader as the primary centre and shows 'Select a feed item to display it here.' when no item is selected; management tabs keep their active centre canvas.
- [x] #2 Navigation, Feed Items, and Inspector expose focusable, clickable five-column full-height grips with the approved literal ASCII directions; Feed Items and its grip are absent from management tabs.
- [x] #3 Navigation, Feed Items, and Inspector preferences toggle independently, persist across restarts, and Inspector state is shared across all seven Watchlists tabs.
- [x] #4 Responsive layout derives an effective state without changing the saved preference, collapses Inspector then Navigation then Feed Items on Read, and Inspector then Navigation on management tabs.
- [x] #5 Article Focus temporarily collapses all side panes, restores the exact preferred layout, and a manual grip action exits focus before applying its toggle.
- [x] #6 Reader and management canvas are never collapse targets; z acts only on a focused collapsible pane/grip, Z controls Article Focus, and existing left/right rail shortcuts remain valid.
- [x] #7 Versioned config normalization preserves valid side-pane choices, removes unknown/Reader entries, writes values plus version atomically, and retries after failed migration or ordinary preference writes.
- [x] #8 Layout changes preserve selected item, active scope, search/filter/page state, list focus/visible offset, and Reader scroll; scope changes do not auto-select an item.
- [x] #9 Production CSS renders target/minimum pane widths and exact five-column grips without horizontal overflow at declared boundaries and the 60-column supported floor.
- [x] #10 Reader's always-visible action row contains only Star/Unstar, Mark read/unread, and Open in browser; advanced Ingest and Queue actions remain available through Inspector.
- [x] #11 Open in browser accepts only valid HTTP(S) URLs and performs the operating-system browser call off the Textual UI thread, with honest failure feedback.
- [x] #12 Server-backed Read shows the local-only recovery state and Switch to Local without issuing local Reader, Smart Feed, search, or refresh queries under a Server label.
- [x] #13 Focused Watchlists/UI tests, CSS bundle integrity, lint/static checks, isolated-profile live keyboard/pointer verification, and self-review pass; documentation, ADR link, task notes, and acceptance criteria are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute `Docs/superpowers/plans/2026-08-23-watchlists-collapsible-reader-layout.md`
task-by-task using TDD and the dedicated Watchlists worktree.

ADR required: no new ADR

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: ADR-042 already contains the accepted permanent-Reader/collapsible-side-pane
amendment; this task directly implements it without changing the architecture boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reworked Watchlists Read around a permanent Reader with independently collapsible
  Navigation, Feed Items, and Inspector panes. Manual preferences are versioned and
  durable; responsive collapse and Article Focus remain transient. Management tabs
  retain their existing centre canvases and share the Inspector preference.
- Added production five-column ASCII grips, responsive width resolution, focus and
  view-state restoration, serialized retryable preference persistence, Reader core
  actions/footer, validated off-thread browser launch, and honest server-backed Read
  recovery. Advanced item actions remain in Inspector.
- Modified areas: Watchlists screen/controller, Watchlists-local pane/layout/store
  modules, Watchlists CSS and generated CSS bundle, and focused Watchlists/UI tests.
  No database/schema/migration, Media, or shared split-pane framework changes were
  made. Implementation commit range: `e986e8c..2c77eac`, including the final
  test-only lint cleanup and review fixes.
- Final review hardening made Server-backed Read query-free both during mounted
  transitions and cold/deep-linked entry, parks the local navigation/snapshot model
  while recovery is visible, and restores it when moving to a management tab. It
  also tightened compact-width controls and migrated stale layout contracts.
- Verification on `753739f`: focused Watchlists/UI suite collected 888 tests and
  passed 887 with one pre-existing Personas tooltip-audit skip and two dependency
  warnings in 460.59s; the lint-cleanup files passed 143 focused tests; scoped Ruff,
  all five CSS bundle-sync checks, and `git diff --check` passed. Isolated-profile
  mounted production-screen verification covered keyboard and terminal-cell pointer
  interaction, all seven tabs, restart persistence, Reader copy/actions/footer, and
  widths 145, 144, 115, 114, 91, 90, 60, and 40 without overlap, horizontal overflow,
  or compositor exceptions. Self-review confirmed state survival, retry behavior,
  permanent-centre boundaries, production CSS use, and Watchlists-only ownership.
- Final verification on `2c77eac`: all 87 collections-screen tests passed in 81.02s;
  the focused five-test Server recovery/switch set passed; scoped Ruff and
  `git diff --check` passed. Independent final code review reported no Critical or
  Important findings. Pytest emitted unrelated dependency and temporary-directory
  cleanup warnings only.
- Verification deviation: at the user's explicit direction, the repository-wide
  58,117-test suite was skipped as a completion gate in favor of tests related to the
  modified functionality/code. A superseded broad run was terminated at 71% without
  a terminal summary; no pass, failure-set, or baseline-comparison claim is made from
  that run.
- ADR required: no. Existing
  [ADR-042](../decisions/042-watchlists-reader-first-ia.md) already records the
  permanent-Reader/collapsible-side-pane architecture, and this work does not change
  its boundary. The approved design spec and ADR remain factually current.
<!-- SECTION:NOTES:END -->
