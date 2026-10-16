---
id: TASK-35033
title: Desktop-to-phone approval relay (Phase 2c)
status: Done
created_date: 2026-10-16 17:26
updated_date: 2026-10-16 19:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Relay message types + served-child publisher + parent merge per spec 2026-10-15 §4.3/§5
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The five ADR-236 relay message types round-trip through the control protocol registry (relay.decide request/reply pair registered; unknown-type and malformed-payload rejection still holds)
- [x] #2 The served-child publisher forwards mounted approval rounds and run-status deltas through the child control client without altering TUI approval behavior (call-through-first, bounded, never raising; native TUI runs never publish)
- [x] #3 Desktop rounds merge into the mobile approvals surface: list_approvals and the approval_requested event carry surface desktop + child_id while mobile rounds stay surface mobile and untouched
- [x] #4 Phone decisions on desktop rounds route back over the loopback broker (relay.decide.request -> response) and apply on the owning child's Console controller, clearing the round; the child control handler answers relay.decide.request from its console resolver (unknown round or absent controller -> delivered False, never an error frame)
- [x] #5 Failure paths fail closed (disconnect/timeout/dead child -> round dropped, False returned, deny-side semantics govern) and the canvas-disabled case leaves the relay inert; the phone UI tags each approval card desktop/phone, run_status events resync the open chat and refresh the inbox, and the README documents what works, the mounted-active-session lifecycle limit, fail-closed semantics, and the inert case
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Four-task implementation of ADR-236 (spec 2026-10-15 sections 4.3/5, v1-limit removal), one commit per task on feat/served-mobile-phase2c: 11a36d6ef4 protocol registry (5 relay message types, request/reply pair, payload field specs; 2 new round-trip tests), 820179c1fc served-child publisher (ServedRelayPublisher wrap-at-binding-site of set_pending_approval, call-through-first, bounded queue drainer, run-status polling; 15 tests), e126608a56 parent-side MobileChatService relay (attach_relay supervisor + per-child drainer on the broker event queue, merged list_approvals with surface/child_id, desktop decision routing via broker.request with 2s budget, disconnect fail-closed cleanup; 9 tests), plus the Task-4 commit (child decide handler + phone UI + docs + this close-out).

Task-4 additions: (a) controller ruling folded in from Task 3's plan-gap review -- the child side of relay.decide.request now exists: ServedCanvasControlHandler gained a relay.decide.request branch placed BEFORE the scope lookup (no canvas authority involved; unbound handler still answers) that dispatches to an injected relay_decide resolver and ALWAYS answers relay.decide.response {round_id, delivered} -- unknown round, absent resolver, or a raising resolver answers delivered False, never a control.error frame (fail-closed; deny-side governs). app.py wires the resolver to the app-owned Console controller via _served_relay_decide: armed-round pre-check on the controller's _pending_approval_rounds registry (the same registry resolve_pending_approval reads; the method itself returns None by contract so the pre-check supplies the truthiness) then resolve_pending_approval -- the identical entrypoint the desktop approval card uses. (b) phone UI: each approval card carries a desktop/phone surface tag (approval-surface pill, reusing the folder-badge styling) and a run_status case resyncs the open chat + refreshes the approvals inbox. (c) Web_Server/README relay paragraph documenting what works, the Task-2 review carry limitation verbatim (relay covers ACTIVE-session mounted-card rounds only; unmounted and parked-background rounds are not relayed), fail-closed semantics, and the canvas-disabled inert case; the two stale v1-limit bullets rewritten.

Deviations: decide_approval became ASYNC in Task 3 (desktop routing awaits broker.request; mobile_api handle_decide awaits it; all call sites updated); run status is read by polling the controller run-state seam rather than wrapping a callback (app-wide, no screen coupling); the E2E test closes the child client BEFORE broker.aclose() -- the broker's accept task blocks in _read_message until the client socket goes away (same teardown order as the scripted-child harness).

Verification (worktree venv, targeted runs only): protocol 63/63; mobile service 21/21, api 27/27, page 21/21; served_relay_publisher 15/15; web_auth 83/83; session_store 7/7; cli_pair 3/3; approval_constants 2/2; webauthn 7/7. gateway (82) + canvas-spawn (4): 3 failures name-identical on the pre-change tree (config RecoveryRequired, environment-dependent, verified via git stash) -- zero new failures. No ruff in this worktree's venv; diff follows file-local style.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
