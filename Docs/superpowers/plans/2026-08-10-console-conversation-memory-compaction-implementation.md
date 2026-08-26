# Console Conversation Memory and Auto-Compaction Implementation Plan

Date: 2026-08-10

Parent task: [TASK-14811](../../../backlog/tasks/task-14811%20-%20Console-conversation-memory-and-auto-compaction.md)

Design: [Console conversation memory and auto-compaction](../specs/2026-08-10-console-conversation-memory-compaction-design.md)

ADR required: yes

ADR path: [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md)

Reason: This feature introduces durable per-conversation policy and summary
provenance, a cost-bearing model-call service boundary, cross-module ownership
between Console and Settings, and a long-lived context-injection contract.

## Delivery rule

The parent feature is delivered through dependency-ordered, single-PR
subtasks. Before implementation of each subtask, put that task In Progress,
add its file-level plan through the Backlog CLI, confirm the next available
database schema version where applicable, and rebase its branch on current
`dev`. Do not combine all six slices into one PR.

## Phase 1: Persist and resolve policy — TASK-14811.1

1. Inventory every current reader/writer of Console session settings,
   `context_summary`, summary boundaries, model context-window values, and
   global Console defaults.
2. Define typed policy/default/override/effective-value models outside widgets.
3. Add repository migration and access methods for conversation overrides and
   branch-valid memory provenance plus the content-free auxiliary-attempt
   ledger, using the next free schema version at implementation time.
4. Add explicit legacy-summary migration/compatibility and rollback handling.
5. Restore durable policy through Console close/resume/restart without
   expanding application-root state ownership; stage new-empty-tab overrides
   until first conversation persistence without creating empty rows.
6. Add validation, serialization, precedence, migration, corruption, and model
   switch tests.
7. Update task notes and evidence; do not mark Done until focused suites,
   linting, documentation, and self-review pass.

Primary seams to inspect:

- `tldw_chatbook/Chat/console_session_settings.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `tldw_chatbook/UI/Console_Modules/session.py`
- `tldw_chatbook/config.py`

## Phase 2: Prepare and account exact provider requests — TASK-14811.2

1. Define an immutable `PreparedConsoleRequest` with separately owned system,
   memory, mandatory, compactable, tool, source, and active-request segments.
2. Add one sensitive `PreparedProviderRequest` serialization boundary in the
   gateway. Estimates and dispatch consume the same instance; no second
   provider payload builder remains.
3. Specify deterministic provider mappings for distinct-role and single-
   preamble adapters. Keep stored original system content unchanged while
   preserving tagged memory delimiters and attribution in merged wire forms.
4. Resolve total context, separate input/output caps, requested and effective
   response reservation, safety margin, and configured/effective conversation
   budget without the historical hidden half-window clamp.
5. Classify compactable and mandatory token categories and build atomic direct
   chat and agent tool-call/result units from the exact prepared payload.
6. Retain deterministic whole-unit safety windowing and stop known-window
   overflow; label unknown limits and user-supplied caps without claiming
   provider safety.
7. Test provider serialization, estimator/dispatch artifact identity,
   response limits, direct/agent paths, multimodal payloads, unknown windows,
   and deterministic trimming.

Primary seams to inspect:

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_history_budget.py`
- `tldw_chatbook/Chat/console_provider_gateway.py`
- provider gateway/adapters used by the Console

## Phase 3: Add bounded branch-safe compaction — TASK-14811.2.1

1. Resolve high-water/target decisions from the effective policy and exact
   prepared request.
2. Select the largest oldest contiguous complete span whose wrapper, editable
   prompt, prior memory, selected units, and adaptive output reserve fit the
   exact active model window.
3. Refactor manual rewind summarization behind the existing sensitive
   auxiliary completion boundary with ordinary chat augmentations disabled.
4. Extend the auxiliary result/telemetry contract with normalized provider
   usage when available and record every admitted attempt in the content-free
   ledger, including failure, cancellation, and stale outcomes.
5. Add per-conversation admission locking, cancellation, and revision-based
   stale-result rejection.
6. Persist branch-valid memory with a summarized-prefix digest covering
   message versions, selected variants, and relevant attachments. Revalidate
   it before every injection and include prior active memory in later passes.
7. Inject memory as a separate semantic segment and use the Phase 2 provider
   serializer without mutating stored original system content.
8. Test adaptive input/output bounds, all policy modes, hysteresis, prefix
   validity, branch/edit/model races, iterative memory, usage accounting,
   privacy, one-call-insufficient, and non-compactable-material behavior.

Primary seams to inspect:

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Internal_Prompts/console_prompts.py`
- `tldw_chatbook/Chat/console_provider_gateway.py`
- `tldw_chatbook/Chat/provider_usage.py`
- `tldw_chatbook/LLM_Calls/pricing_catalog.py`

## Phase 4: Current-conversation Console UX — TASK-14811.3

1. Replace ambiguous `Max tokens` copy with `Response max tokens` in the
   affected Console settings surface.
2. Keep the fixed quick model popover compact: show separate Request and
   Conversation values, expose compaction policy, and deep-link to the full
   Context & memory view for custom numeric budget editing.
3. Give the existing modal stable Model & generation and Context & memory
   in-modal views with one action bar, capacity math, inherited overrides,
   breakdown, policy, and inline plain-text memory review.
4. Add Ask-before-compacting, Compact now, undoable current-branch reset,
   confirmed reset-all, busy, stale, failure, and overhead-recovery flows.
5. Keep Save scoped to the current conversation. Preserve the existing global
   provider workflow as `Save provider defaults`; it never writes memory
   defaults or the internal prompt.
6. Add mounted UI, focus, keyboard, accessibility, geometry, and snapshot tests
   before any visual-only polishing.

Primary seams to inspect:

- `tldw_chatbook/Widgets/Console/console_model_popover.py`
- `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- existing shared form/status/modal primitives

## Phase 5: Canonical Settings UX — TASK-14811.4

1. Add the Conversation memory group to Console Behavior using staged-save
   semantics and the typed global policy model.
2. Add a filtered deep link to `console.rewind_summarize` in Internal Prompts;
   do not add a duplicate editor.
3. Show and repair context-window capability data in Providers and Models,
   with detected/override provenance and reset behavior, reusing TASK-320's
   existing capability/config authority rather than adding another store.
4. Add policy preview and honest cost/transcript/safety copy.
5. Verify that global changes do not overwrite explicit conversation
   overrides.
6. Add persistence, validation, navigation, mounted UI, focus, and narrow-width
   tests.

Primary seam:

- `tldw_chatbook/UI/Screens/settings_screen.py`

## Phase 6: Hardening and live evidence — TASK-14811.5

1. Re-run the full cross-surface race matrix: send, edit, branch, model,
   policy, reset, close, and restart during compaction.
2. Verify auxiliary-call usage/cost metadata and content-free diagnostics at
   the final provider adapter boundary.
3. Run focused unit/integration/UI suites and the relevant broader regression
   suites; run lint/format checks specified by the repository at that time.
4. Exercise a real configured provider for every live scenario in the design,
   including failure and overhead recovery.
5. Run a narrow-terminal pass and record geometry/focus observations.
6. Document evidence in the task, capture any generalizable incident in the
   appropriate lessons file, and complete the parent AC only after all child
   tasks are Done.

## Cross-phase invariants

- Transcript rows are never deleted or rewritten by compaction.
- Stored original user/character system content is byte-identical before and
  after memory projection; provider serialization may deterministically place
  separately owned tagged segments into one wire preamble.
- Off means no summary call, not no safety boundary.
- One send attempt makes at most one automatic summary call.
- No stale async result can commit across a changed lineage, model, prompt,
  policy, request, or reset revision.
- No widget becomes a persistence or provider-service owner.
- Core controls remain visible and keyboard reachable.
- Logs, errors, and auxiliary usage records contain no transcript or summary
  bodies, including failed, cancelled, and stale attempts.
- New settings land only in canonical `settings_screen.py`, not deprecated
  settings parallels.

## Required review gates

Each implementation PR must include:

- an updated Backlog implementation plan and notes;
- the applicable checked acceptance criteria;
- focused automated evidence and exact commands/results;
- migration/rollback review when storage changes;
- provider-payload review when request semantics change;
- mounted/narrow geometry evidence when UI changes;
- a self-review against ADR-052 and the design failure matrix;
- an explicit note if live-provider verification was unavailable, with the
  task left open until the missing evidence is supplied.
