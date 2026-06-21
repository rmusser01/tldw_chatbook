---
id: TASK-129
title: Console chat lifecycle UAT completion
status: Done
assignee: []
created_date: 2026-06-21 00:36
labels:
- console
- chat
- uat
dependencies:
- TASK-128
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden the core Console chat lifecycle so users can create chats, send messages, recover from blocked sends, close tabs, and return to saved conversations without losing context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can create a new chat tab from Console and clearly identify the active tab.
- [x] #2 User can close a chat tab with the compact close control without accidentally activating the wrong tab.
- [x] #3 User can type, paste, and send a normal message while seeing all composer text during focus.
- [x] #4 Blocked-send states produce visible, actionable recovery feedback without silently failing.
- [x] #5 Sent user messages and completed assistant responses render in the transcript with readable separation.
- [x] #6 Chat lifecycle state survives leaving and returning to Console within the same app session.
- [x] #7 Focused regression coverage verifies new chat, close tab, send, blocked-send feedback, transcript rendering, and session return behavior.
- [x] #8 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read existing Console chat lifecycle tests and UI seams to identify the smallest regression surface for tabs, composer, send feedback, transcript rows, and in-session return.
2. Add focused failing regression/UAT coverage before product code changes for new chat tab activation/identity, compact close behavior, composer focused text visibility, normal send transcript rendering, blocked-send recovery feedback, and session return.
3. Implement only the minimal Console shell/composer/tab/transcript/controller changes needed to pass the new coverage, avoiding provider settings internals, workspace ownership policy, and per-message action semantics.
4. Run the focused test command, then a relevant broader Console/UI test subset if the focused suite passes.
5. Capture actual Textual-web/CDP screenshot evidence for visible Console states under Docs/superpowers/qa/console-uat-parallelization/ using the task naming protocol, or document the concrete blocker if the harness is not yet runnable.
6. Add concise implementation notes and update acceptance checklist only for criteria verified by tests/evidence; do not mark Done until main-thread visual approval is obtained.

ADR required: no
ADR path: N/A
Reason: The task hardens existing Console lifecycle UI/controller behavior and in-session state. It does not introduce storage/schema migrations, sync/conflict policy, provider/runtime boundary changes, service contracts, cross-module ownership contracts, security/privacy/encryption/authentication changes, or dependency/tooling decisions.
<!-- SECTION:PLAN:END -->

## Parallel Ownership

Owns chat tabs, composer lifecycle, send/blocked-send feedback, transcript baseline rendering, and basic in-session persistence. Avoid provider configuration internals, workspace ownership rules, and per-message action semantics beyond selection hooks required for transcript stability.

## Implementation Notes

Completed the Chat Lifecycle stream for the Console UAT split. The Console now serializes native chat lifecycle state through `ChatScreen.save_state()` and `ChatScreen.restore_state()` so chat tab identity, active tab selection, transcript rows, and composer text survive leaving and returning to Console within the same app session. Blocked setup sends now append durable, actionable recovery feedback into the transcript instead of failing silently.

Added focused regression coverage in `Tests/UI/test_console_native_chat_flow.py` for new tab creation, compact close controls, focused composer text visibility, normal send transcript rendering, blocked-send recovery feedback, and screen recreation return behavior. Captured approved rendered CDP/Textual-web evidence:

- `Docs/superpowers/qa/console-uat-parallelization/task-129-chat-lifecycle-cdp-2026-06-21.png`
- `Docs/superpowers/qa/console-uat-parallelization/task-129-composer-text-visible-cdp-2026-06-21.png`

ADR required: no
ADR path: N/A
Reason: This PR hardens existing Console UI/controller behavior and in-session state only; it does not alter storage/schema, sync/conflict policy, provider/runtime boundaries, service contracts, security/privacy controls, or long-lived workspace ownership contracts.
