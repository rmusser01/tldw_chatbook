---
id: TASK-36763
title: Project Console chats and retained decisions from runtime owners
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36760
- TASK-36762
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 22:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose current shared Console state and answer supported decisions from the phone even while the relevant chat is in the background or its screen is unmounted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bounded chat, transcript, run and subagent projections use canonical live and saved identities without duplicate rows, unintended agent starts or unrestricted metadata reads.
- [x] #2 Current retained decisions are visible and supported answers resolve once through native owners; unsupported kinds remain visibly pending with a terminal path.
- [x] #3 Answerable-client accounting survives independent client disconnects and preserves native timeout, permission, hook and project-instruction policies.
- [x] #4 Snapshot/subscription races, overflow resets, stale targets and long inert content preserve currentness and thinking/capture privacy policies without widget scraping.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of approved runtime projection and retained-decision contracts.

1. Add failing runtime-owned background decision/independent-answerability/privacy tests using real native sessions and retained rounds without ChatScreen.
2. Extend existing runtime/round owners with exact-target bounded projections, native decision support and answerable-client accounting, preserving native policy/deadline behavior.
3. Publish allowlisted invalidations through the lifecycle-owned bounded handoff and implement snapshot/cursor/byte boundaries, registering transport cursor before native capture and preserving unseen changes/reset state.
4. Route supported answers through original retained kind-specific resolvers, leave unsupported confirmations visibly pending with terminal path, and preserve capture/thinking privacy.
5. Run the specified targeted projection/round/publisher/runtime/protocol checks, scoped static/format verification, update docs, commit explicit paths and obtain independent spec+quality review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented runtime-owned bounded canonical saved/live discovery, native/canonical transcript continuation, fleet and retained decision projections through ConsoleRemoteProjection; exact supported answers use original native resolvers and independent bounded answerable-client claims. Shared pure native approval summary preserves widget helpers, destination-first bounded redaction and strict phone failure; raw shell/unfaithful reviews remain unsupported/pending with terminal path. Privatev3 auth now requires broker-issued validated host identity. Child native capture/ack sequence is explicit and separate from parent transport cursor. Existing ADR-244 applies; controller-recorded scope/timing/privacy rulings govern shared helper and handshake integration. Files: new console_remote_projection.py/approval_review.py and focused tests; existing runtime/round/command/publisher/protocol/App wiring/widget helper/docs. Independent spec+quality review found one Important rename witness gap; dbb45f8dbc fixes it after4cbcb1fb01 and scoped re-review is clean. Evidence249 targeted passes, final affected106 passes+2probes, fix2RED→29covering passes; exactBASE28UI fixture errors reproduced before documented marker. Default Ruff84→82/0introduced perfile, syntax/newformat/existingratchet/diff pass. Task9 remains owner of authenticatedWS/revocation/parent-generatedclient wiring, parent-before-capture cursor reconciliation, saved-only history/create-open-save and cutover; these are not claimed complete here. Full suite/livephone/provider skipped. Detailed report in plan-specific .superpowers task-5-report.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
