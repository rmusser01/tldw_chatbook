---
id: TASK-401
title: Show armed response prefill in Console context snapshot preview
status: To Do
assignee: []
created_date: '2026-07-21 03:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console context snapshot modal claims to show the assembled next-send payload, but an armed /prefill (one-shot or pinned) adds a trailing assistant turn and bypasses the agent loop for that send, neither of which the preview reflects. Surfaced by the response-prefill final review (spec Docs/superpowers/specs/2026-07-20-console-response-prefill-design.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Context snapshot preview includes the trailing assistant prefill turn when one is armed,Preview indicates the agent loop is bypassed for that send,No change when no prefill is armed
<!-- AC:END -->
