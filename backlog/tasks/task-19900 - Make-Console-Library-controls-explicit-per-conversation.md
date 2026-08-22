---
id: TASK-19900
title: Make Console Library controls explicit per conversation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-22'
labels:
  - console
  - rag
  - agents
  - privacy
  - ux
dependencies:
  - task-3170
priority: high
references:
  - https://github.com/rmusser01/tldw_chatbook/pull/1933
documentation:
  - Docs/superpowers/specs/2026-08-22-console-library-controls-design.md
  - backlog/decisions/079-console-library-conversation-authority.md
---

## Description

Console currently combines three different ways of consulting the Library:
user-initiated search, automatic pre-send retrieval, and model-initiated
Library tools. Their global or implicit controls make it difficult for a user
to predict which actor may read local Library data in a conversation. Replace
that ambiguity with locally stored per-conversation policy, truthful runtime
disclosure, fail-closed assistant access, and separate review surfaces for
staged/cited evidence and assistant Library activity.

This task also subsumes the still-open zero-result disclosure defect recorded
by TASK-3504.

## Acceptance Criteria

- [ ] Each Console conversation exposes two independent, locally persisted
      controls: automatic pre-send Library retrieval is Never or Automatic,
      and assistant-initiated Library access is Blocked or Allowed. Manual
      Search Library remains available in every combination.
- [ ] Shipped defaults for newly created local sessions are Never and Blocked;
      global defaults seed only newly created local sessions, existing
      conversations preserve their upgrade-time effective behavior, and a
      synced/imported conversation with no device-local policy fails closed to
      Never and Blocked.
- [ ] When assistant access is Blocked, no built-in Library provider is
      available to the primary agent or subagents. All 18 direct Library names
      plus `search_library_rag` remain reserved against Skill and MCP
      collisions in every policy/mode combination.
- [ ] When assistant access is Allowed, the existing global
      `direct_library_tools` setting selects Direct or RAG for the turn; it is
      not interpreted as an enable/disable control. Direct retains all six
      Library categories and RAG retains Notes, Media, and Conversations.
- [ ] Automatic retrieval uses the draft at actual turn execution, searches
      the fixed Notes/Media/Conversations categories, honors the current item
      scope, and skips when explicit evidence is already staged. It never
      inherits a prior manual search's source-type filters.
- [ ] Automatic retrieval visibly prepares before provider dispatch. Failure
      or timeout pauses with Retry, Send once without Library, and Cancel;
      zero matches proceeds only with a persistent disclosure that the turn
      was sent without Library evidence. Draft and conversation policy remain
      unchanged by all one-shot recovery actions.
- [ ] One immutable policy/tool-mode snapshot governs each executed turn,
      including queued turns and every subagent spawned by that turn. Policy
      changes during a run affect only later executed turns.
- [ ] Assistant-initiated Library reads produce bounded, content-minimized,
      local-only `library_activity` records attributed to the durable turn,
      run, and actor. Activity never enters staged evidence, Sources, prompts,
      model context, sync, or ordinary logs, and default trajectory export
      redacts its query/source details.
- [ ] The Console presents one fixed-order two-axis status chip, separate
      Library Access and Search Library modals, one-row-per-source evidence,
      and a Selected turn Inspector group that distinguishes Cited sources
      from Library activity. Save, conflict, unavailable, narrow-viewport,
      keyboard, focus, and dirty-dismiss states are explicit and truthful.
- [ ] Migration, policy lifecycle, CAS conflicts, first persistence,
      ephemeral promotion rollback, four policy combinations, Direct/RAG
      selection, name reservation, queued/subagent snapshots, automatic-send
      recovery, activity minimization/attribution/projection, and the real
      Textual composition are covered by targeted automated tests and a live
      Console walkthrough.

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/079-console-library-conversation-authority.md`

Reason: this task changes local storage and migration, conversation policy
ownership, assistant permission/runtime composition, privacy disclosure, and
the long-lived Console review model.

1. Replace the inherited PR design with the approved architecture, record
   ADR-079, self-review both documents, and obtain written-spec approval.
2. Write the detailed Superpowers implementation plan and re-check the task
   and ADR identifiers against all refs before implementation.
3. Deliver the approved design as dependency-ordered atomic subtasks:
   TASK-19900.1 policy storage/lifecycle; TASK-19900.2 runtime enforcement;
   TASK-19900.3 automatic send gate; TASK-19900.4 policy/search/source UI;
   TASK-19900.5 activity capture/review; TASK-19900.6 documentation and
   production-path qualification.
4. Start each subtask separately, add its concrete plan only after it enters
   In Progress, implement RED-first, and close it only after its targeted
   verification and notes satisfy the repository Definition of Done.
5. Re-check identifiers and cross-task integration at closeout. Do not run the
   full suite unless the owner separately opts in.

## Implementation Notes

<!-- Added after implementation. -->
