---
id: TASK-31230
title: Add same-origin served Canvas and remote web authentication
status: To Do
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-03'
labels: [canvas, web-server, authentication, security]
dependencies: [TASK-31229]
priority: high
---

## Description

Extend `--serve` with a Chatbook-owned split-pane Canvas shell on the existing origin, connecting textual-serve's parent process to the authoritative Chatbook child while protecting every remote authority-bearing route.

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

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
