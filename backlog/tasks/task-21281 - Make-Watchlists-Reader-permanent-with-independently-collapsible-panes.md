---
id: TASK-21281
title: Make Watchlists Reader permanent with independently collapsible panes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-23 22:11'
updated_date: '2026-08-23 22:17'
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
- [ ] #1 Read always mounts Reader as the primary centre and shows 'Select a feed item to display it here.' when no item is selected; management tabs keep their active centre canvas.
- [ ] #2 Navigation, Feed Items, and Inspector expose focusable, clickable five-column full-height grips with the approved literal ASCII directions; Feed Items and its grip are absent from management tabs.
- [ ] #3 Navigation, Feed Items, and Inspector preferences toggle independently, persist across restarts, and Inspector state is shared across all seven Watchlists tabs.
- [ ] #4 Responsive layout derives an effective state without changing the saved preference, collapses Inspector then Navigation then Feed Items on Read, and Inspector then Navigation on management tabs.
- [ ] #5 Article Focus temporarily collapses all side panes, restores the exact preferred layout, and a manual grip action exits focus before applying its toggle.
- [ ] #6 Reader and management canvas are never collapse targets; z acts only on a focused collapsible pane/grip, Z controls Article Focus, and existing left/right rail shortcuts remain valid.
- [ ] #7 Versioned config normalization preserves valid side-pane choices, removes unknown/Reader entries, writes values plus version atomically, and retries after failed migration or ordinary preference writes.
- [ ] #8 Layout changes preserve selected item, active scope, search/filter/page state, list focus/visible offset, and Reader scroll; scope changes do not auto-select an item.
- [ ] #9 Production CSS renders target/minimum pane widths and exact five-column grips without horizontal overflow at declared boundaries and the 60-column supported floor.
- [ ] #10 Reader's always-visible action row contains only Star/Unstar, Mark read/unread, and Open in browser; advanced Ingest and Queue actions remain available through Inspector.
- [ ] #11 Open in browser accepts only valid HTTP(S) URLs and performs the operating-system browser call off the Textual UI thread, with honest failure feedback.
- [ ] #12 Server-backed Read shows the local-only recovery state and Switch to Local without issuing local Reader, Smart Feed, search, or refresh queries under a Server label.
- [ ] #13 Focused Watchlists/UI tests, CSS bundle integrity, lint/static checks, isolated-profile live keyboard/pointer verification, and self-review pass; documentation, ADR link, task notes, and acceptance criteria are complete.
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
