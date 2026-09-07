---
id: TASK-14803
title: Capture immutable owning-session Console turn context
status: Done
created_date: 2026-08-10 06:04
labels:
- console
- agents
- architecture
priority: high
references:
- backlog/decisions/098-visible-bounded-console-prompt-queue.md
- backlog/decisions/033-application-session-state-ownership.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 08:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every Console provider turn use one immutable execution context resolved for its owning session so background work cannot mix the viewed tab or mid-validation settings into a different session run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A frozen Console turn-execution context captures the target session provider selection, capabilities, system prompt, workspace context and roots, generation parameters, RAG defaults, tool configuration, and other provider-payload settings.
- [x] #2 Manual send, retry, continue, regenerate, edit-resend, and summarize paths resolve the context from the owning session and thread the same instance through validation, payload construction, capability checks, fingerprinting, caching, and execution.
- [x] #3 Switching the viewed session or changing settings after capture cannot produce a mixed turn; completed changes apply to the following turn.
- [x] #4 Credentials, approval grants, skill trust, and other authority remain live runtime checks and are not retained in the execution context.
- [x] #5 Existing manual-send behavior and the no-argument submission-accepted compatibility hook remain intact, with no prompt-queue UI introduced by this task.
- [x] #6 Joined async tests use real production seams to prove cross-session provider, model, system-prompt, workspace, and settings isolation, including a settings change during validation.
- [x] #7 The frozen context is detached from mutable sources: later mutation or replacement of settings, mappings, sequences, roots, or workspace context cannot change captured turn inputs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map every Console turn-entry and provider-payload seam that currently reads viewed-session or mutable settings, then add joined red tests for owning-session and mid-validation isolation.\n2. Add a focused frozen ConsoleTurnExecutionContext model that defensively detaches mappings, sequences, roots, workspace values, provider selection/capabilities, generation, RAG, and tool configuration while excluding credentials and live authority.\n3. Add one owning-session context resolver and thread the exact captured instance through manual send, retry, continue, regenerate, edit-resend, summarize, validation, payload construction, capability/fingerprint/cache logic, and provider or agent execution.\n4. Preserve one-shot and pinned prefill semantics, the no-argument submission-accepted compatibility hook, and all live credential, approval, and skill-trust checks; introduce no queue UI in this slice.\n5. Run focused red-to-green and mutation checks, then every reached provider-selection, payload, stream, workspace, RAG, retry/regenerate, run-state, ratchet, and import suite.\n6. Ruff the changed Python files, self-review for mutable-source leakage and cross-session reads, record any ambient failures, and complete TASK-14803 notes and acceptance criteria.\n\nADR required: yes\nADR path: backlog/decisions/098-visible-bounded-console-prompt-queue.md\nReason: This task directly implements the immutable owning-session execution-context boundary accepted by ADR-098 and follows ADR-033 ownership; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a detached frozen ConsoleTurnExecutionContext and an owning-session builder in ConsoleSessionController, then threaded one captured instance through send, retry, continue, regenerate, edit-resend, summarize, capability gates, payload construction, fingerprint/windowing, RAG capture, and direct/agent execution. Provider/model/system/workspace roots, generation values, RAG defaults, and tool-mode configuration are stable per turn; credentials, kill switches, approvals, trust, and cancellation remain live. Preserved the no-argument accepted-send callback and no queue UI was introduced. Verification: 14 focused context tests, 319 controller/action tests, 90 provider/model/generation tests, 148 of 149 agent/local/library tests, 7 mounted provider/settings checks, and 18 controller-wiring checks passed; targeted Ruff/compile checks passed. Both owning-selection and post-await payload guards were mutation-checked. Ambient failures reproduced independently: the Windows HOME-only expanduser test resolves USERPROFILE, the warning-sink test captures config first-use posture output, and the architecture ratchet is already exceeded by concurrent Console growth; the new builder was moved out of ChatScreen in response. ADR required: yes; existing ADR-098 implements the accepted boundary, with ADR-033 ownership, so no new ADR was created. Modified: tldw_chatbook/Chat/console_turn_context.py, console_chat_controller.py, console_agent_bridge.py, UI/Console_Modules/session.py, wiring.py, the smallest ChatScreen compatibility seams, and focused tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console turns now execute from one immutable owning-session configuration snapshot, preventing tab switches or mid-validation settings changes from mixing provider inputs while keeping runtime authority live.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
