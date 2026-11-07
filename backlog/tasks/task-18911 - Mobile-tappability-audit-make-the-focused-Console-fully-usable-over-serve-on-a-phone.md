---
id: TASK-18911
title: >-
  Mobile tappability audit: make the focused Console fully usable over --serve
  on a phone
status: To Do
assignee: []
created_date: '2026-08-19 22:50'
labels:
  - serve
  - console
  - mobile
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Focus mode (task-18812) made the Console chrome-free for phone use over --serve, but it is 'necessary but not sufficient' per ADR-071. This task is the sufficiency half: audit and fix the three spec items — (1) send/tool-approvals must be real tap targets, not key-only actions; (2) no hover-only information (hover reveals need tap equivalents); (3) soft-keyboard escape flows — the Console binds several Escape actions (exit hands-free, expand composer, focus composer) and phones have no Escape key, so critical flows need visible affordances. Scope: the three spec items plus whatever the first live phone-sized session over --serve surfaces; NOT a full mobile retrofit. Narrow-terminal CSS already handles layout (single pane <84 cols, footer width tiers).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Composer send is achievable purely by tap (button or equivalent),Tool approval/rejection flows are fully tappable,No information is reachable only via hover — every hover reveal has a tap path,Flows that depend on Escape (hands-free exit, composer expand, composer focus) have visible tap/keyboard alternatives on a phone,Verified live over --serve at a phone-sized viewport (e.g. 390x844),Focus-mode desktop behavior unchanged (existing test suites stay green)
<!-- AC:END -->

## Deferral note (2026-10-11, dev `9f8533686c`)

DEFERRED from the t2-hermes porting slice. The AC requires live verification
over `--serve` at a phone-sized viewport (390x844), which needs an
interactive browser session against a served app — not executable within
this slice's budget alongside the code-port tasks. Nothing here was
implemented; no claim is made about the three spec items' current state.

Starting points for the next session (verified in code at dev `9f8533686c`):
the composer's Send/Queue button lives in the composer bar
(`Widgets/Console/console_composer_bar.py` — Tab reaches it per the
chat-basics Composer section, so a tap target exists; the audit question is
its tap hit-area at 390px, not its existence); approval flows render through
`ChatApprovalCard` (see `Tests/UI/test_approval_*` for the surface's test
seams); hover-reveals and Escape-dependent flows (hands-free exit, composer
expand/focus) are concentrated in `chat_screen.py`'s Escape routing and the
composer-bar hover styles in `css/features/_console*.tcss`. A browser-driven
session over `--serve` (browser automation against localhost) is the AC's
verification path.
