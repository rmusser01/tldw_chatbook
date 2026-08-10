---
id: TASK-14803
title: Capture immutable owning-session Console turn context
status: To Do
created_date: 2026-08-10 06:04
labels:
- console
- agents
- architecture
priority: high
references:
- backlog/decisions/046-visible-bounded-console-prompt-queue.md
- backlog/decisions/033-application-session-state-ownership.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 06:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every Console provider turn use one immutable execution context resolved for its owning session so background work cannot mix the viewed tab or mid-validation settings into a different session run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A frozen Console turn-execution context captures the target session provider selection, capabilities, system prompt, workspace context and roots, generation parameters, RAG defaults, tool configuration, and other provider-payload settings.
- [ ] #2 Manual send, retry, continue, regenerate, edit-resend, and summarize paths resolve the context from the owning session and thread the same instance through validation, payload construction, capability checks, fingerprinting, caching, and execution.
- [ ] #3 Switching the viewed session or changing settings after capture cannot produce a mixed turn; completed changes apply to the following turn.
- [ ] #4 Credentials, approval grants, skill trust, and other authority remain live runtime checks and are not retained in the execution context.
- [ ] #5 Existing manual-send behavior and the no-argument submission-accepted compatibility hook remain intact, with no prompt-queue UI introduced by this task.
- [ ] #6 Joined async tests use real production seams to prove cross-session provider, model, system-prompt, workspace, and settings isolation, including a settings change during validation.
- [ ] #7 The frozen context is detached from mutable sources: later mutation or replacement of settings, mappings, sequences, roots, or workspace context cannot change captured turn inputs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
