---
id: TASK-31230
title: Add same-origin served Canvas and remote web authentication
status: Done
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-05 01:04'
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
- [x] #1 A versioned private parent/child control protocol authenticates one AppService child, carries only bounded typed Canvas messages, and fails closed on unknown versions or message types
- [x] #2 Canvas routes use the existing Chatbook server origin and never require a remote browser to reach another localhost port
- [x] #3 The owned responsive shell presents the terminal and Canvas as sibling regions without adding new string patches to textual-serve's minified bundle
- [x] #4 Control-channel loss disables Canvas for that browser session without terminating the Textual session or exposing another conversation
- [x] #5 Binding beyond validated loopback refuses startup without a configured Chatbook web access token; provider and legacy server tokens are never reused
- [x] #6 Non-loopback plaintext HTTP refuses by default, with HTTPS/trusted-proxy guidance and an explicit warned insecure override
- [x] #7 Login nonces, HttpOnly/SameSite sessions, Host/Origin/CSRF checks, websocket checks, proxy trust, rate limits, expiry, and revocation cover terminal and Canvas routes origin-wide
- [x] #8 Two simultaneous AppService children and browser profiles cannot list, open, submit to, or receive events for each other's conversations or Canvases
- [x] #9 Focused protocol, authentication, dependency-compatibility, responsive-browser, isolation, and authenticated live-flow tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this task implements ADR-115's accepted same-origin served topology, authenticated child-control boundary, origin-wide web authentication, and browser-session isolation; the ADR will be amended with the concrete textual-serve extension seam and protocol version rather than duplicated.

1. Inspect the pinned textual-serve parent/AppService spawn and routing APIs and lock the supported extension seam with compatibility tests.
2. Implement a versioned, bounded, authenticated loopback parent/child Canvas control protocol with exact request ownership, deadlines, backpressure, cancellation, rotation, shutdown, and two-child isolation tests.
3. Add dedicated Chatbook web authentication and remote-bind policy across the entire served origin, including login bootstrap, secure sessions, CSRF/origin/websocket validation, proxy trust, expiry, revocation, and bounded rate limits.
4. Mount a Chatbook-owned responsive sibling terminal/Canvas shell on the existing origin, reuse the trusted Canvas renderer/handlers, and fail Canvas closed without terminating the terminal when the child channel is unavailable.
5. Run targeted protocol, web-auth, textual-serve compatibility, responsive browser, and two-profile isolation tests, then perform real authenticated proxy and cross-browser outer-path verification.
6. Request independent security and UX review, update ADR-115 and this task with the concrete contracts and evidence, and mark the task Done only after every acceptance criterion is verified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented private control protocol v1 in `bd9ad4e381`, using strict length-prefixed JSON, one-use per-AppService spawn secrets, typed bounded requests/events, deadlines, cancellation, replay protection, backpressure, revocation, and cross-child isolation through the supported textual-serve 1.1.3 environment seam.
- Implemented complete-origin browser authentication in `b89c365e20`: dedicated credential precedence, remote bind/TLS policy, trusted proxies, one-time bootstrap, opaque bounded sessions, secure cookies, Host/Origin/CSRF/WebSocket checks, expiry/revocation, and rate limits. Security review fixes covered stale local cookies, Unicode comparison failures, credential-safe representations/logs, strict authority parsing, and bounded state.
- Implemented the authenticated same-origin terminal/Canvas shell in `62909fd7c7`, sharing the trusted Canvas gateway and runtime while binding each browser to exactly one child authority. Cross-session shell, event, source, submit, and download access fails with indistinguishable not-found responses; control loss leaves the terminal usable. The production-path CORS and durable/runtime scope mismatch found by outer verification were fixed in `88c91794dd`.
- Independent code/security reviews and Impeccable finish reviews concluded SHIP with no remaining Critical or Important findings. Final targeted verification was 243 passed and two skips for unavailable optional Firefox/WebKit engines; mandatory Chromium, Ruff/format, Python/JavaScript syntax, wheel build, packaged asset inspection, and diff checks passed.
- The live release checkpoint used the production server behind a trusted ephemeral TLS reverse proxy, one-time authentication, two Chromium profiles, two real AppService children, and terminal-driven Canvas tools. It verified distinct rendering, copied/guessed capability denial, exact events/source/action isolation, confirmed unsent-draft insertion, passive download, zero egress, and terminal survival after one Canvas control channel was revoked. All temporary processes, credentials, profiles, certificates, databases, and downloads were removed.
- ADR check: ADR-115 was updated with the actual textual-serve seam, protocol and limits, credential/authentication flow, proxy policy, settlement contract, and live isolation evidence. No new ADR was needed.
<!-- SECTION:NOTES:END -->
