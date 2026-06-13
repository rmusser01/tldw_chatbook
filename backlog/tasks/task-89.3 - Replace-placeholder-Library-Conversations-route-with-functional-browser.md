---
id: TASK-89.3
title: Replace placeholder Library Conversations route with functional browser
status: Done
dependencies:
- TASK-89.1
labels:
- library
- conversations
- ux
priority: high
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the placeholder Library Conversations shell with a functional conversation browser aligned with the Library content-hub model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library Conversations lists available saved conversations or an honest empty/recovery state.
- [x] #2 Users can select a conversation and inspect title, message count or available metadata, source authority, and handoff eligibility.
- [x] #3 Open in Console and Use as source actions are visible only when supported, with recovery copy when blocked.
- [x] #4 The route preserves Library ownership language and does not look like an accidental legacy exit.
- [x] #5 The route follows the Library contract from TASK-89.1 for content ownership, workspace eligibility, handoff payloads, and visual layout expectations.
- [x] #6 Focused regressions cover list loading, selection, empty/error states, route activation, and action availability.
- [x] #7 Actual CDP/Textual-web screenshot QA is approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This slice uses the existing conversation scope service and Library UI shell. It does not change storage/schema, sync policy, provider/runtime boundaries, service contracts, or security posture.

1. Add failing mounted regressions proving the Library Conversations action stays in Library, renders a conversation browser, supports selection metadata, and shows honest empty/recovery states.
2. Reuse the existing Library conversation snapshot records already loaded by chat_conversation_scope_service.
3. Add a Library-native Conversations mode with list/detail/inspector rows and stable selectors.
4. Preserve owner language and workspace-gated handoff behavior; expose Open in Console / Use as source only when supported or show disabled recovery copy.
5. Run focused Library tests and visual-parity checks, then capture an actual CDP/Textual-web screenshot for approval.
<!-- SECTION:PLAN:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not rebuild the full Conversation/Character management destination in this route.
- Do not add conversation editing beyond the minimum needed to inspect and hand off saved conversations as Library sources.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a Library-native Conversations mode behind the existing `Open Conversations` action so the route stays inside Library rather than exiting to a legacy destination.
- Reused the Library source snapshot records from the existing conversation scope service to render saved conversation rows, metadata, selection state, source authority, and workspace-gated Console handoff eligibility.
- Added blocked/empty-state recovery copy, a direct empty-state `Open Console` action, inline disabled-action explanation, and a mode-specific Library status row so users can understand location, state, and recovery without inspecting tooltips.
- Wired selected conversation handoff actions through the existing `ChatHandoffPayload` path, preserving local ownership and workspace policy checks.
- Added focused mounted regressions for route activation, list/empty state, selection metadata, action availability, and handoff payload behavior.
- QA evidence: `Docs/superpowers/qa/product-maturity/screen-qa/library/conversations-browser-polish-cdp-2026-06-09.png` approved in Codex chat on 2026-06-09.
- ADR required: no. No storage/schema, sync policy, provider/runtime boundary, service contract, or security boundary changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Conversations is now a functional Library-owned browser with honest empty-state recovery, selectable conversation metadata, explicit handoff eligibility, and guarded Console/source handoff actions. The route remains visually aligned with the Library content hub shell and no longer behaves like an accidental legacy exit.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria checked.
- [x] Implementation plan followed; no ADR required.
- [x] Focused mounted regressions added and verified.
- [x] Actual CDP/Textual-web screenshot approved.
- [x] QA notes updated with screenshot and verification evidence.
<!-- DOD:END -->
