---
id: TASK-42346
title: "Replace the open-coded Console runtime currentness checks with one shared helper"
status: In Progress
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:56
labels:
- follow-up
- console
- refactor
priority: low
references:
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_received_dispatch.py
- tldw_chatbook/Chat/console_automatic_submission.py
- tldw_chatbook/Chat/console_context_repository.py
- tldw_chatbook/Chat/console_configuration_preparation.py
- tldw_chatbook/Chat/console_reference_preparation.py
- tldw_chatbook/Chat/console_raw_cli_custody.py
- tldw_chatbook/Chat/console_hook_preparation.py
- tldw_chatbook/Chat/console_workflow_runs.py
- tldw_chatbook/Agents/hooks_v2/native_commands.py
dependencies:
- TASK-34975
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Proposed by the review of the PR #3493 port (TASK-34975, round 20) as optional cleanup. Read at 59cc99e61b (dev 8a55555798 plus the round-20 stack). The `console_context_repository.py` copy cited below was added by the TASK-34975 port (9470ee5528) and is on dev only once that port lands.

**Why.** Several Console modules decide whether a runtime still owns a session by reading its private fields directly, in the same shape each time: `runtime._disposed`, `runtime._chat_controller is not controller`, `runtime._chat_store is not store`, `session_id in runtime._admission_fenced_sessions`. Examples: `console_received_dispatch.py:137-145`, `console_automatic_submission.py:573-579` and `:644`, `console_context_repository.py:1386-1394`, `console_configuration_preparation.py:1703`, `:2421-2459` and `:2749`. Other files read `_admission_fenced_sessions` with a different mix of fields: `console_reference_preparation.py:295`, `console_raw_cli_custody.py:110`, `console_hook_preparation.py:222`, `console_workflow_runs.py:221` and `Agents/hooks_v2/native_commands.py:310`. When the meaning of "current" changes, every copy has to change with it. TASK-34975's port had to add one more copy, and a missed copy fails open.

`console_configuration_preparation.py` is in another session's open work (tldw-chatbook-30). Leave it out, or coordinate before editing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One runtime-owned check answers whether a runtime is current for a given controller, store and session, and the copies of the common shape call it, except in files held by other open work, which the implementation notes list
- [ ] #2 Sites that check a different set of fields either use the shared check plus their own extra conditions, or the implementation notes say why they keep their own
- [ ] #3 Behaviour is unchanged: the covering tests of every edited module pass through the scratch wrapper at `-n 4`, and the module-size ratchets pass
<!-- AC:END -->

## Implementation Plan

1. Compare runtime ownership checks with their surrounding guards. Move the common disposed, controller, store and session-admission check behind a runtime method.
2. Preserve checks with a different contract and leave configuration preparation to its current owner.
3. Run the edited modules' ownership regressions and the module-size ratchets.

ADR required: no
ADR path: N/A
Reason: Mechanical consolidation of the current runtime authority boundary.

## Implementation Notes

The runtime predicate is shared by received sends, automatic submissions, context presentation, reference preparation and hook preparation. Each caller retains its checks of the original app, source or receipt where applicable. Configuration preparation remains under the other active work identified above. Raw CLI capture owns a session and store without a controller binding. Workflow submission has a separate native receipt and closing check. Terminal command hooks may settle after admission closes. Those paths retain their existing tests.
