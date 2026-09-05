---
id: TASK-31229
title: Deliver native Canvas browser UX and confirmed bridge
status: Done
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-04 21:45'
labels:
  - canvas
  - browser
  - console
  - ux
dependencies:
  - TASK-31226
  - TASK-31228
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give terminal users a trusted loopback Canvas shell that opens in their system browser, renders the selected revision through the strict runtime, and supports version navigation, source actions, hot reload, and narrowly confirmed return actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first Canvas open lazily starts one loopback-only gateway on an OS-assigned port and uses Textual's URL-opening path without creating a second conversation authority
- [x] #2 Creation opens automatically; matching completed updates hot-reload a following view and show `Updated · Undo / View previous` without redirecting a changed chat or branch
- [x] #3 Multiple named Canvases have one session selection, editable revisioned titles, following and exact pinned URLs, and transcript cards that reopen the originating revision
- [x] #4 Assistant HTML code blocks expose idempotent `Open in Canvas` and explicit `Open as new` actions with compatibility-repair drafts
- [x] #5 The trusted toolbar supports source inspection/copy, inert source download, warned runnable-HTML download, reload, close, provenance, connection state, and scripts-disabled recovery
- [x] #6 `canvas.submit()` shows the complete bounded text/JSON payload for confirmation and inserts it only as an unsent draft in the exact matching composer
- [x] #7 `canvas.download()` accepts only bounded passive formats and performs a one-shot browser download only after trusted confirmation
- [x] #8 Frame capabilities are short-lived, single-use, revision/session scoped, absent from logs/history, and unusable for top-level navigation or sibling content
- [x] #9 Browser-open, runtime, confirmation, branch-race, accessibility, failure-recovery, and native live-flow tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this delivery implements ADR-121's trusted loopback gateway, capability-token, browser-shell, and confirmed bridge boundaries; no new ADR is needed unless implementation changes those accepted boundaries.

1. Add the core aiohttp dependency and implement a lazily started loopback-only gateway with one app-owned lifecycle, typed same-scope routes, hardened headers, and hashed short-lived single-use capabilities delivered outside URLs.
2. Build the packaged preview-first shell and native transcript/message actions for auto-open, following versus pinned revisions, branch-safe hot reload, title/source controls, accessibility, and HTML-code-block import.
3. Add bounded confirmed submit and passive-download requests that route only to the exact live Console session/composer and never auto-send or execute downloads without trusted confirmation.
4. Run focused gateway/capability/runtime, Console widget/controller, package-wheel, and Playwright native-flow verification plus static checks and a real terminal outer-path check.
5. Request independent security and UX review, then update TASK-31229 and the implementation plan with capability lifetime, browser-open, accessibility, screenshots, and end-to-end evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the native Canvas browser workflow described by ADR-121. One app-owned `aiohttp` gateway starts lazily on an OS-assigned loopback port and the Console retains conversation/branch authority. The trusted shell provides named Canvas selection, revisioned rename, follow/pin/previous navigation, provenance, source controls, compatibility and scripts-disabled recovery, transcript-card reopen, and parsed HTML-block import. Tool creation automatically uses Textual's browser-open seam; failure leaves a copyable loopback URL in the terminal.

The confirmed bridge keeps generated code inside the zero-egress runtime. `canvas.submit()` and allowlisted passive `canvas.download()` requests cross into trusted shell code only after an exact, complete, five-minute confirmation. Submit replaces only an unchanged, exact-session composer draft and never sends it. Download validates decoded raster signatures or bounded literal passive text; runnable source HTML is a separate warned action outside Canvas protections.

Capability lifetimes are intentionally short and bounded: shell boot 30 seconds, frame 20 seconds, action 30 seconds, browser session 30 minutes, and confirmation/idempotency settlement at most five minutes. Bearers are random, stored only as hashes, single-use where action-shaped, bound to the browser/session/Canvas/revision/action, excluded from query parameters after bootstrap, and revoked on reload, branch/session change, close, or shutdown. Browser sessions are capped at 64; settlement receipts are capped at 64 with at most 16 waiters per pending request.

The responsive Precision Workbench shell was reviewed at desktop and 390 px widths. Keyboard checks cover toolbar reachability and horizontal reveal at narrow width; dialogs trap focus, background content becomes inert, branch-unavailable and expiry states use bounded live-region announcements, and focus moves to a usable recovery control. Evidence is in `.impeccable/review/canvas-confirmation-desktop.png` and `.impeccable/review/canvas-confirmation-narrow.png`.

The final real-terminal proof used a deterministic local provider and a disposable isolated data/config root. An assistant `canvas_create` produced revision 1 and automatically opened the system Chrome session through the configured browser seam. Playwright attached to that exact already-open tab, confirmed the durable Canvas did not show `Temporary`, interacted with generated HTML, displayed the exact two-line payload and explicit target, and confirmed it. The terminal then showed `CANVAS_D4_OUTER_OK` plus `second line` as an unsent draft; the provider received no third request. SQLite held one conversation, two messages, one Canvas, one revision, a complete assistant message, and zero pending dispatch checkpoints. Scratch captures are under `output/canvas-d4-outer-20260904/captures/`.

Live review found one CSS cascade defect: `.state-badge { display: inline-flex }` overrode a correct semantic `hidden` attribute and made a durable Canvas look temporary. Commit `698af61a44` adds a computed-visibility Chromium regression and a narrowly scoped `.state-badge[hidden]` rule. Independent review approved the fix with no findings.

Targeted verification included the gateway/capability, runtime/zero-egress, Console authority/widget, packaging, and native browser groups. The final native browser file passed `10 passed`; the bridge hardening gate passed `47 passed` with three genuine environment skips; `git diff --check` passed. The full repository suite was not run, per the repository rule requiring explicit user opt-in.
<!-- SECTION:NOTES:END -->
