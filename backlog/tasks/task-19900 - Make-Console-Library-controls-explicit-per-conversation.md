---
id: TASK-19900
title: Make Console Library controls explicit per conversation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-22'
updated_date: '2026-08-28 15:23'
labels:
  - console
  - rag
  - agents
  - privacy
  - ux
dependencies:
  - task-3170
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/1933'
documentation:
  - Docs/superpowers/specs/2026-08-22-console-library-controls-design.md
  - Docs/superpowers/plans/2026-08-22-console-library-controls.md
  - backlog/decisions/079-console-library-conversation-authority.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console currently combines three different ways of consulting the Library:
user-initiated search, automatic pre-send retrieval, and model-initiated
Library tools. Their global or implicit controls make it difficult for a user
to predict which actor may read local Library data in a conversation. Replace
that ambiguity with locally stored per-conversation policy, truthful runtime
disclosure, fail-closed assistant access, and separate review surfaces for
staged/cited evidence and assistant Library activity.

This task supersedes the narrower zero-result notice proposal in TASK-3504;
TASK-19900.3 owns the replacement persistent sent-turn disclosure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each Console conversation exposes two independent, locally persisted
      controls: automatic pre-send Library retrieval is Never or Automatic,
      and assistant-initiated Library access is Blocked or Allowed. Manual
      Search Library remains available in every combination.
- [x] #2 Shipped defaults for newly created local sessions are Never and Blocked;
      global defaults seed only newly created local sessions, existing
      conversations receive their prior effective behavior inside one
      sanitized-seed migration transaction, and a synced/imported conversation
      with no device-local policy fails closed to Never and Blocked.
- [x] #3 When assistant access is Blocked, no built-in Library provider is
      available to the primary agent or subagents. All 18 direct Library names
      plus `search_library_rag` remain reserved against Skill and MCP
      collisions in every policy/mode combination.
- [x] #4 When assistant access is Allowed, the existing global
      `direct_library_tools` setting selects Direct or RAG for the turn; it is
      not interpreted as an enable/disable control. Direct retains all six
      Library categories and RAG retains Notes, Media, and Conversations.
- [x] #5 Automatic retrieval uses the draft at actual turn execution, searches
      the fixed Notes/Media/Conversations categories, honors the current item
      scope, and skips when explicit evidence is already staged. It never
      inherits a prior manual search's source-type filters.
- [x] #6 Automatic retrieval visibly prepares before provider dispatch. Failure
      or timeout pauses with Retry, Send once without Library, and Cancel;
      zero matches proceeds only with a persistent disclosure that the turn
      was sent without Library evidence. A store-owned exactly-once state
      machine preserves draft, attachments, staged evidence, queue ownership,
      conversation policy, ordinary-session persistence identity, and title
      through every recovery action and injected commit failure. One bounded
      row in a dedicated device-local dispatch-checkpoint table and its
      assistant owner make post-commit Retry/Discard explicit without retaining
      or auto-replaying a provider request; terminal/Discard settlement is
      atomic, while ephemeral recovery stays in memory and blocks promotion
      until settled. A synced closed assistant-generation state makes unresolved
      remote/imported owners inert and truthful, and an atomic handoff leaves
      ADR-063 as the sole owner before any durable tool continuation executes.
- [x] #7 Durable policy is re-read at actual execution, then combined with a
      gateway-resolved conservative destination record into one immutable turn
      context governing queued turns and every subagent. Policy/destination
      changes after capture affect only later executed turns.
- [x] #8 Assistant-initiated Library reads produce bounded, content-minimized,
      device-local `library_activity` records attributed to the durable turn,
      run, and actor. Activity never enters staged evidence, Sources, prompts,
      model context, sync, or ordinary logs, and default trajectory export
      redacts its query/source details.
- [x] #9 The Console presents one fixed-order two-axis status chip, separate
      Library Access and Search Library modals, one-row-per-source evidence,
      and a Selected turn Inspector group that distinguishes Cited sources
      from Library activity. Canonical Settings owns future-session defaults;
      missing-row, Save, conflict, unavailable, narrow-viewport, keyboard,
      focus, and dirty-dismiss states are explicit and truthful.
- [x] #10 Migration, policy lifecycle, CAS conflicts, first persistence,
      repository-level hard-purge cascade, ephemeral execution/promotion
      rollback, four policy combinations, Direct/RAG selection, name
      reservation, queued/subagent snapshots, destination classification,
      preparation races, activity minimization/attribution/projection, and the
      real Textual composition are covered by targeted automated tests; the
      live Console walkthrough covers only actions the Console exposes,
      including soft delete/restore rather than hard purge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: `backlog/decisions/079-console-library-conversation-authority.md`

Reason: this task changes local storage and migration, conversation policy
ownership, assistant permission/runtime composition, privacy disclosure, and
the long-lived Console review model.

1. Replace the inherited PR design with the approved architecture, record
   ADR-079, self-review both documents, and obtain written-spec approval.
2. Follow the approved detailed plan at
   `Docs/superpowers/plans/2026-08-22-console-library-controls.md` and re-check
   the task and ADR identifiers against all refs before implementation.
3. Deliver the approved design as dependency-ordered atomic subtasks:
   TASK-19900.1 policy/checkpoint storage, Sync compatibility, and lifecycle; TASK-19900.2 runtime enforcement;
   TASK-19900.3 automatic send gate; TASK-19900.4 policy/search/source UI;
   TASK-19900.5 activity capture/review; TASK-19900.6 documentation and
   production-path qualification.
4. Start each subtask separately, add its concrete plan only after it enters
   In Progress, implement RED-first, and close it only after its targeted
   verification and notes satisfy the repository Definition of Done.
5. Re-check identifiers and cross-task integration at closeout. Do not run the
   full suite unless the owner separately opts in.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered the complete ADR-079 work stream through six independently reviewed
children. Device-local policy and recovery storage preserve upgrades and fail
closed; execution-time authority gates Direct/RAG providers and reserved names;
automatic retrieval is an explicit pre-dispatch state machine with durable,
content-minimized recovery; the Console separates policy, manual search,
sources, and assistant activity into truthful responsive surfaces; and bounded
Library activity remains outside evidence, model context, sync, and ordinary
logs.

The final qualification reconciled all governed documentation and joined the
migration, policy, runtime, send, recovery, queue, continuation, activity,
sync/export, Settings, and Textual paths. Exact post-rebase Delivery 6
verification passed 1,760 tests in 339.21 seconds, the focused new contract
passed 41 tests in 2.61 seconds, and 23 exact counterfactual node IDs expanded
to 29 passing cases across nine security/correctness guard families;
TASK-19900.6 records the full list. Ruff, compileall, CSS bundle sync,
screen-size ratchet, production
diagnostic inventory (541 owners, 1,260 TASK-492 calls, 7,396 TASK-494 calls,
and 8 sink files), backlog-ID, and diff checks passed. An explicitly
isolated real-app walkthrough proved Direct/RAG disclosure, all four policy
states, exact manual-search draft preservation, atomic first persistence,
restart hydration, and soft-delete/restore; repository coverage proves hard
purge cascades all device-local sidecars.

PR review also closed Qodo's three findings by moving purge reads under the
transaction boundary, routing scripted tool/continuation turns through the
production streaming adapter, and restoring exact privacy-safe direct-tool
argument assertions. TASK-19900.6 records the red/green and focused evidence.
The final `dev` rebase also inherited two Virtual CLI callback warnings that
could persist raw exception text; canary-based regression coverage now proves
they emit exception types only, and the reviewed diagnostic inventory includes
that new two-call owner.

TASK-19900.1 through TASK-19900.6 are Done with checked acceptance criteria.
The owner approved the integrated targeted scope and chose not to run the full
repository suite. ADR-079 remains authoritative, ADR-063 exclusively owns
durable continuation after handoff, and no new ADR or lesson entry was needed
at closeout.
<!-- SECTION:NOTES:END -->
