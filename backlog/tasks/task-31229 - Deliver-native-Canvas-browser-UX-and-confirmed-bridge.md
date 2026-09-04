---
id: TASK-31229
title: Deliver native Canvas browser UX and confirmed bridge
status: To Do
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-03'
labels: [canvas, browser, console, ux]
dependencies: [TASK-31226, TASK-31228]
priority: high
---

## Description

Give terminal users a trusted loopback Canvas shell that opens in their system browser, renders the selected revision through the strict runtime, and supports version navigation, source actions, hot reload, and narrowly confirmed return actions.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first Canvas open lazily starts one loopback-only gateway on an OS-assigned port and uses Textual's URL-opening path without creating a second conversation authority
- [ ] #2 Creation opens automatically; matching completed updates hot-reload a following view and show `Updated · Undo / View previous` without redirecting a changed chat or branch
- [ ] #3 Multiple named Canvases have one session selection, editable revisioned titles, following and exact pinned URLs, and transcript cards that reopen the originating revision
- [ ] #4 Assistant HTML code blocks expose idempotent `Open in Canvas` and explicit `Open as new` actions with compatibility-repair drafts
- [ ] #5 The trusted toolbar supports source inspection/copy, inert source download, warned runnable-HTML download, reload, close, provenance, connection state, and scripts-disabled recovery
- [ ] #6 `canvas.submit()` shows the complete bounded text/JSON payload for confirmation and inserts it only as an unsent draft in the exact matching composer
- [ ] #7 `canvas.download()` accepts only bounded passive formats and performs a one-shot browser download only after trusted confirmation
- [ ] #8 Frame capabilities are short-lived, single-use, revision/session scoped, absent from logs/history, and unusable for top-level navigation or sibling content
- [ ] #9 Browser-open, runtime, confirmation, branch-race, accessibility, failure-recovery, and native live-flow tests pass
<!-- AC:END -->

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
