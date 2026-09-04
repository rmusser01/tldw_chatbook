---
id: TASK-31230
title: Add same-origin served Canvas and remote web authentication
status: In Progress
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-04 21:46'
labels:
  - canvas
  - web-server
  - authentication
  - security
dependencies:
  - TASK-31229
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend `--serve` with a Chatbook-owned split-pane Canvas shell on the existing origin, connecting textual-serve's parent process to the authoritative Chatbook child while protecting every remote authority-bearing route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A versioned private parent/child control protocol authenticates one AppService child, carries only bounded typed Canvas messages, and fails closed on unknown versions or message types
- [ ] #2 Canvas routes use the existing Chatbook server origin and never require a remote browser to reach another localhost port
- [ ] #3 The owned responsive shell presents the terminal and Canvas as sibling regions without adding new string patches to textual-serve's minified bundle
- [ ] #4 Control-channel loss disables Canvas for that browser session without terminating the Textual session or exposing another conversation
- [ ] #5 Binding beyond validated loopback refuses startup without a configured Chatbook web access token; provider and legacy server tokens are never reused
- [ ] #6 Non-loopback plaintext HTTP refuses by default, with HTTPS/trusted-proxy guidance and an explicit warned insecure override
- [ ] #7 Login nonces, HttpOnly/SameSite sessions, Host/Origin/CSRF checks, websocket checks, proxy trust, rate limits, expiry, and revocation cover terminal and Canvas routes origin-wide
- [ ] #8 Two simultaneous AppService children and browser profiles cannot list, open, submit to, or receive events for each other's conversations or Canvases
- [ ] #9 Focused protocol, authentication, dependency-compatibility, responsive-browser, isolation, and authenticated live-flow tests pass
<!-- AC:END -->

## Implementation Plan

ADR required: yes
ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this task implements ADR-115's accepted same-origin served topology, authenticated child-control boundary, origin-wide web authentication, and browser-session isolation; the ADR will be amended with the concrete textual-serve extension seam and protocol version rather than duplicated.

1. Inspect the pinned textual-serve parent/AppService spawn and routing APIs and lock the supported extension seam with compatibility tests.
2. Implement a versioned, bounded, authenticated loopback parent/child Canvas control protocol with exact request ownership, deadlines, backpressure, cancellation, rotation, shutdown, and two-child isolation tests.
3. Add dedicated Chatbook web authentication and remote-bind policy across the entire served origin, including login bootstrap, secure sessions, CSRF/origin/websocket validation, proxy trust, expiry, revocation, and bounded rate limits.
4. Mount a Chatbook-owned responsive sibling terminal/Canvas shell on the existing origin, reuse the trusted Canvas renderer/handlers, and fail Canvas closed without terminating the terminal when the child channel is unavailable.
5. Run targeted protocol, web-auth, textual-serve compatibility, responsive browser, and two-profile isolation tests, then perform real authenticated proxy and cross-browser outer-path verification.
6. Request independent security and UX review, update ADR-115 and this task with the concrete contracts and evidence, and mark the task Done only after every acceptance criterion is verified.
