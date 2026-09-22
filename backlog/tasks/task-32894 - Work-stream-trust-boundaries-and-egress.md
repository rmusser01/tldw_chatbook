---
id: TASK-32894
title: "Work stream: trust boundaries and egress"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six boundary defects: an endpoint probe that reaches the network without the SSRF check its sibling
branch already applies four lines away, a watchlist that seeds `trusted_origins` from the discovered
`<loc>` rather than the subscription's own source, a Confluence auth branch whose predicate makes the
check unreachable (with a test that is green for the wrong reason), eight credentialed calls that follow
redirects with no byte cap, two TTS backends that bypass `resolve_provider_api_key`, and four XML
parsers still on the stdlib parser while `defusedxml` is already a base dependency.

ADR-012's 2026-09-19 amendment is cited, not re-decided.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The endpoint probe applies one SSRF check on every branch
- [ ] #2 No credentialed request follows a redirect without an explicit allow and a byte cap
- [ ] #3 The four XML parsers use `defusedxml` and their `_KNOWN_UNHARDENED` rows are dropped in the same commit
- [ ] #4 The Confluence false-green test asserts the behaviour it claims to
<!-- AC:END -->
