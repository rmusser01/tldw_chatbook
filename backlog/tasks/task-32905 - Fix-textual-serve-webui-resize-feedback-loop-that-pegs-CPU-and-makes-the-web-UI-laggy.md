---
id: TASK-32905
title: >-
  Fix textual-serve webui resize feedback loop that pegs CPU and makes the web
  UI laggy
status: Done
assignee:
  - '@robert'
created_date: '2026-09-23 15:18'
updated_date: '2026-09-23 15:25'
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
Investigation and fix implemented.

Root cause (measured live, pre-fix): the first-byte patch hook fired the full resize handler on EVERY app output frame, sendSize() was unconditional inside every repaint, and the after-write hook scheduled an all-rows xterm refresh per write; the app answers every resize with output, so the loop never settled. Measured: ~54 resize websocket messages/second at pure idle, app child 60-100% CPU at idle (unattached WebDriver child: 0.4%), one click churned ~44k DOM nodes (WebGL/Canvas had been nulled by the patch, forcing the DOM renderer), startup floods 2-8MB of splash output through the same path.

Fix (tldw_chatbook/Web_Server/serve.py, patch_textual_serve_viewport_js):
- first-byte trigger now goes through _chatbookViewportFirstByte, a once-per-connection guard;
- _chatbookViewportRepaint sends the resize only when fit() actually changed cols/rows (textual-serve's own onResize->sendSize remains the primary path; worst case one duplicate resize per real change, which the app treats idempotently);
- after-write repaint is now a 250ms trailing debounce (self-heal after traffic pauses) instead of a per-write rAF full-screen refresh;
- stopped nulling the WebGL/Canvas addons: upstream GPU renderers restored, while the resize repaint keeps clearTextureAtlas + full refresh (the remedy for the GPU-renderer resize staleness this patch family was built for).

Tests: Tests/Web_Server/test_textual_web_viewport.py updated -- renderers preserved, first-byte once-guard, dimension-gated sendSize, debounced after-write (no per-write rAF). 17/17 pass; full Tests/Web_Server/ package 74 passed.

Live verification (post-fix, same environment): idle websocket sends 0 messages in 8.2s (was 370 in 6.9s); app child 0.9-1.1% CPU at idle (was 60-100%); click DOM churn 0 nodes (was ~44k); a real viewport change sends a bounded burst of exactly 2 resizes per grid change (128x45 / 182x45 observed) with no re-ignition of the loop afterwards; rendering verified via canvas pixel sampling after resize (239 distinct colors, 25% lit) and websocket alive (-closed absent).

Lesson recorded in backlog/docs/lessons-live-verification.md (measure both ends of the served websocket; per-message vs once-per-connection hook sites when patching minified bundles).

Known remaining (separate tasks): served shell hardcodes the ws URL from public_url (default localhost), so accessing via 127.0.0.1 or an IP kills the websocket on the session cookie boundary; task-32904 tracks the broader app-side stall work (36 recorded event_loop_stall events, 250ms-2.4s).
<!-- SECTION:NOTES:END -->
