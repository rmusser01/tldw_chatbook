---
id: TASK-33130
title: >-
  Fix textual-serve webui resize feedback loop that pegs CPU and makes the web
  UI laggy
status: Done
assignee:
  - '@robert'
created_date: '2026-09-23 15:18'
updated_date: '2026-09-27 19:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live measurement showed the served browser JS sends ~54 resize websocket messages/second at idle: the first-byte patch hook runs the full resize handler on EVERY app output frame, sendSize() is unconditional, and each repaint forces an all-rows xterm refresh. The app relayouts and emits output for every resize, sustaining the loop. Both the browser tab and the app child sit at 60-100% CPU at idle, and the disabled WebGL/Canvas renderers make each forced refresh a full DOM rebuild. Kill the loop: once-only first-byte trigger, dimension-gated sendSize, trailing-debounced after-write repaint, restore upstream WebGL/Canvas renderers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Idle served session sends ~0 resize messages (no feedback loop),App child CPU at idle is near headless baseline (~0-5%),No per-write all-rows xterm refresh during output streams,Browser resize still repaints correctly (original patch intent preserved),Viewport patch tests updated and passing,Live verification with measurements recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rewrite patch_textual_serve_viewport_js: once-per-connection first-byte resize trigger, dimension-gated sendSize (only when fit() changed cols/rows), after-write repaint as 250ms trailing debounce, stop nulling the WebGL/Canvas renderers (restore upstream renderers while keeping the resize repaint machinery incl. clearTextureAtlas).\n2. Update Tests/Web_Server/test_textual_web_viewport.py to assert the new invariants (renderers preserved, first-byte once-guard, gated sendSize, debounced after-write, no per-write rAF full refresh).\n3. Run the Web_Server test module (targeted).\n4. Live verification: serve the app, connect a browser, measure idle resize messages, child CPU, DOM churn on click, and confirm resize repaint still works; record numbers in Implementation Notes.\n\nADR required: no\nADR path: N/A\nReason: bug fix within the existing textual-serve integration seam (Web_Server/serve.py patch), no new architectural boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_chatbook/pull/2856 (against dev). Branch fix/task-32905-webui-resize-feedback-loop also carries the served-shell fixes: same-origin terminal websocket URL (127.0.0.1/LAN IP pages now connect), canvas session poll stops after a 404, textual.js served with a 1h public cache policy, and the TASK-32376 YAML quoting fix.
<!-- SECTION:NOTES:END -->


## Renumbering provenance

This task previously held id TASK-32905, colliding with the older
"Product-decision-needed-the-Tamagotchi-widget-half" task (created
2026-09-21 23:05, already on `dev`), which arrived first. This task was
created 2026-09-23 15:18 in a local backlog checkout that had not yet
synced that file (PR #2856). Per the owner rule decided 2026-08-21 in
TASK-19601 (**the older arrival by `created_date` keeps the id regardless
of status; the younger task renumbers with a provenance note**), it
renumbered to TASK-33130. Citations to TASK-32905 in this PR's earlier
commit messages, in `tldw_chatbook/Web_Server/serve.py`, in
`Tests/Web_Server/test_textual_web_viewport.py`, and in
`backlog/docs/lessons-live-verification.md` refer to THIS task; the other
TASK-32905 holder is the Tamagotchi product-decision task.
