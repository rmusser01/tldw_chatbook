# Chat-First Shell And Label Cleanup Design

Date: 2026-04-21
Status: Approved for spec review
Scope: Shell UX, navigation labeling, chat context visibility, and handoff continuity

## Summary

tldw_chatbook has already shifted to a screen-based shell, but the product still reads like a collection of legacy modules rather than one coherent workspace. The next UX slice should make Chat the obvious home for agentic work, reduce label and grouping ambiguity in global navigation, and make preserved handoff context visible at the shell level.

This slice does not attempt a full information architecture migration. It focuses on the smallest set of shell changes that materially improve orientation, recognition, consistency, and system-status visibility without breaking stable route contracts.

## Problem Statement

The current shell has three usability problems that cut across the whole product:

1. Agentic work is conceptually chat-first, but the shell still presents `Coding` as a peer primary destination, which muddies the product model.
2. Navigation labels still reflect implementation history rather than user mental models, most visibly with `CCP` / `Conv/Char`.
3. Cross-shell handoffs preserve meaningful state in code, but the Chat shell does not surface enough of that context. Users cannot reliably see which workspace, assistant identity, or backend the current chat session belongs to.

These issues violate core Nielsen heuristics:

- Visibility of system status: the current chat shell does not clearly show scope, backend, or task context.
- Match between system and the real world: labels like `CCP` are internal shorthand rather than user-facing language.
- Recognition rather than recall: users should not have to remember which hidden module state the current chat session is carrying.
- Consistency and standards: handoffs should preserve the same context fields across notes, study, search, media, and persona flows.

## Goals

- Make Chat the clear shell-level home for agentic programming and control workflows.
- Improve global navigation labels and grouping without changing route IDs.
- Surface preserved context directly in the Chat shell.
- Standardize the session state contract used by Chat handoffs.
- Preserve existing compatibility routes, especially `coding`, while visually demoting them.

## Non-Goals

- Remove the `coding` route in this slice.
- Rewrite all cross-shell handoff entry points.
- Rebuild the entire Chat UI hierarchy.
- Perform cosmetic restyling unrelated to shell clarity.
- Expand feature depth inside secondary modules such as Study, Notes, or Media.

## User And Persona Fit

The primary persona for this slice is a knowledge worker or developer using Chat as the orchestration surface for:

- conversation
- artifact-aware assistance
- programming and operational control
- inline approvals for privileged actions
- task continuity and resume workflows

This persona benefits more from reduced orientation cost than from additional visual novelty. The shell should answer, at a glance:

- what am I doing
- where am I scoped
- which assistant or persona is active
- what backend am I using
- what the agent needs from me next

## Current Constraints

- Stable route IDs already exist and must remain stable.
- The app already uses a screen shell built around `BaseAppScreen`.
- Chat already has inline task-surface primitives:
  - `ChatApprovalCard`
  - `ChatResumePanel`
- Chat state already stores important continuity data in `TabState`:
  - `runtime_backend`
  - `assistant_kind`
  - `assistant_id`
  - `scope_type`
  - `workspace_id`
- The product direction documented in existing internal docs is already chat-first for agentic workflows.

## Proposed Approach

The recommended approach is a shell-level refinement rather than a structural rewrite:

1. Keep route IDs stable and avoid deep routing churn.
2. Update user-facing global navigation labels and visual clustering to reflect the intended product model.
3. Fold shell-context visibility into the existing compact Chat top bar rather than adding a second always-visible chrome layer.
4. Bind that shell bar to the active chat session contract so preserved handoffs become visible and durable across restore and tab switches.
5. Leave deeper handoff expansion and routing deprecation for later slices.

This achieves the highest UX payoff with controlled risk.

## Information Architecture Changes

### Route Contract

No route IDs change in this slice. Existing route IDs remain the source of truth:

- `chat`
- `coding`
- `chatbooks`
- `notes`
- `media`
- `ingest`
- `search`
- `subscriptions`
- `ccp`
- `study`
- `llm`
- `stts`
- `evals`
- `tools_settings`
- `customize`
- `logs`
- `stats`

### User-Facing Navigation Changes

Global navigation should change in presentation only.

#### Before

- Workspace: `Chat`, `Coding`, `Chatbooks`
- Content: `Notes`, `Media`, `Ingest`, `Search`, `Subscriptions`
- Characters: `Conv/Char`, `Study`
- AI Config: `LLM`, `S/TT/S`, `Evals`
- System: `Settings`, `Customize`, `Logs`, `Stats`

#### After

- Work: `Chat`, `Chatbooks`
- Content: `Notes`, `Media`, `Ingest`, `Search`, `Subscriptions`
- Library: `Library`, `Study`
- AI: `LLM`, `S/TT/S`, `Evals`
- System: `Settings`, `Customize`, `Logs`, `Stats`, `Coding`

### Grouping Semantics

The current top navigation is a compact single-row control. In this slice, grouping should be expressed through:

- ordered clustering of destinations
- existing separators between clusters
- user-facing item labels

Do not add a second navigation row or persistent textual group headers in this slice. Group names such as `Work` and `Library` are design taxonomy and test vocabulary, not a requirement to render extra section-title chrome.

### Label Rules

- `Chat` remains unchanged and stays first.
- `ccp` becomes `Library` in user-facing global navigation.
- `Coding` remains available but is moved into the system/utility group to communicate its compatibility role rather than a primary-work role.
- `Chatbooks` stays visible as a peer work artifact shell.

### Rationale

- `Library` is more legible than `CCP` or `Conv/Char` and better matches the combined mental model of conversations, characters, prompts, and persona-like reusable assets.
- Demoting `Coding` reduces the false impression that programming control is a separate primary mode from Chat.
- Grouping by user intent rather than implementation lineage improves recognition and lowers navigation ambiguity.

## Chat Shell Changes

### Shell Stack

The Chat shell should read top-to-bottom as:

1. shared app chrome
2. chat inline task surface
3. combined chat shell bar
4. active chat content
5. chat composer

The task surface remains the primary place for approvals and resume status. The shell bar does not replace it. It provides orientation and compact session controls without adding another stacked bar.

### Combined Chat Shell Bar

Use a single compact shell bar inside the Chat screen shell. It should be positioned below the existing task surface and above the active chat content.

This slice should extend or repurpose the existing compact model bar instead of introducing a second always-visible strip. The shell bar therefore combines:

- session context
- quick model/runtime controls already exposed by the compact bar

The combined shell bar must preserve existing quick-control behavior. Folding context into the compact bar is a consolidation, not a feature reset.

The context portion should show:

- backend: `Local` or `Server`
- scope: `Global` or `Workspace: <name or id>`
- assistant identity:
  - `General`
  - `Character: <name>`
  - `Persona: <id or resolved label>`
- session title: current conversation or tab title

Layout rules:

- context chips should be visually first
- quick controls may remain on the same row when space allows
- if width is constrained, truncate secondary detail before dropping primary context
- the shell bar must remain a compact one-row-first surface rather than a verbose settings region

### Behavior Rules

- The shell bar must update immediately when Chat restores a saved tab/session state.
- When chat tabs are enabled, the shell bar must always reflect the active session rather than stale restore-time state.
- The shell bar must update on tab create, tab reuse, tab switch, and tab close.
- If the session is workspace-scoped, the workspace label must be explicit.
- If workspace scope is unavailable in the current backend, the strip should still show the scope and backend combination rather than hiding it.
- The shell bar should not depend on the sidebars being open.
- The shell bar should remain visible for both tabbed and single-session chat modes.

### Why This Matters

This addresses the most important shell-level question: what context is this conversation operating in right now. Users should not have to infer that from prior screens or hidden state.

## Chat Task Surface Rules

The existing inline task surface remains the primary system-status region for agentic work.

### Approval Card

`ChatApprovalCard` continues to represent privileged actions that need user confirmation.

Rules:

- approval requests must remain inline in Chat
- the approval card must stay above the shell bar
- approvals should not be duplicated in another shell surface

### Resume Panel

`ChatResumePanel` continues to summarize the current task:

- summary
- last step
- diff summary
- next action

Rules:

- resume state should remain visible when present
- the shell bar should complement resume state, not restate it

## Handoff State Contract

This slice standardizes the fields that must survive Chat handoffs and become visible in the Chat shell.

### Required Persisted Fields

For every reusable chat session or tab state, preserve:

- `runtime_backend`
- `assistant_kind`
- `assistant_id`
- `scope_type`
- `workspace_id`
- `character_id`
- `character_name`
- `title`

### Display Mapping

These state fields map to shell UI as follows:

- `runtime_backend` -> backend chip
- `scope_type` + `workspace_id` -> scope chip
- `assistant_kind` + related name/id fields -> assistant chip
- `title` -> session title chip or summary text

### Display Resolution Order

To reduce raw-ID leakage and keep the shell understandable, resolve labels in this order:

- workspace label: resolved workspace name if available at runtime, otherwise `workspace_id`
- character label: `character_name`, otherwise `character_id` if present, otherwise `General`
- persona label: resolved persona label if available from current runtime context, otherwise `assistant_id`
- session title: current session title, otherwise `New chat`

This slice does not require new persisted display-only fields. Resolved names may come from currently available runtime context, with explicit ID fallback when no live label is available.

### Contract Rule

New or modified handoffs in this slice should populate the contract fields above.

Untouched legacy handoffs should not hard-fail solely because they do not yet provide a complete contract. Instead, the shell must fall back explicitly and visibly using the default rules below rather than silently presenting misleading empty metadata.

## Implementation Boundaries

This slice should stay tightly bounded.

### In Scope

- update global navigation labels and cluster ordering
- extend or repurpose the compact Chat shell bar to include active-session context
- bind the shell bar to `ChatScreenState` / `TabState` and active `ChatSessionData`
- keep inline approvals and resume state in place
- update focused UI tests

### Out of Scope

- rewrite all `Use in Chat` entry points
- remove `coding`
- redesign Chat sidebars
- add new agent features beyond status and context visibility

## Error And Empty-State Rules

- If session identity is unknown, show `Assistant: General` rather than blank.
- If workspace scope is present but the workspace label cannot be resolved, show `Workspace: <workspace_id>`.
- If there is no meaningful session title, show `New chat`.
- The shell bar should never disappear entirely just because some state is missing.
- The task surface should remain hidden when there is no resume state or pending approval, but the shell bar should remain visible.
- If a legacy handoff opens a generic chat session, the shell bar must show fallback metadata rather than implying a scoped or persona-bound session.

## Accessibility And Usability Notes

- The shell bar context must be readable without requiring hover.
- Keep labels short and scannable.
- Avoid abbreviations that require insider knowledge.
- Do not rely on color alone to communicate backend or scope.
- Ensure the shell bar works with keyboard-only navigation and does not trap focus.
- Preserve useful context at narrow widths by truncating session title before backend, scope, or assistant identity.

## Test Strategy

Add or extend focused tests in these areas:

### Navigation

- verify updated user-facing labels and clustered ordering
- verify route IDs remain stable
- verify `Coding` is still reachable after regrouping

### Chat Shell

- verify the task surface still mounts above the chat log
- verify the combined shell bar renders backend, scope, assistant identity, and session title from active chat state
- verify workspace-scoped restored sessions display workspace context immediately
- verify tab create, reuse, switch, and close all update shell context immediately when tabs are enabled
- verify legacy partial handoffs render explicit fallback labels rather than blank state
- verify compact model/runtime controls still behave correctly after being folded into the shell bar
- verify narrow-width behavior truncates session title before backend, scope, or assistant identity
- verify keyboard traversal can reach shell controls without trapping focus

### Regression Coverage

Keep current coverage for:

- screen routing
- inline approvals and resume state
- tab navigation
- study shell scope behavior

## Acceptance Criteria

- Global navigation uses user-facing labels that match the new shell model, including `Library` in place of `CCP`-style copy.
- `Coding` remains routable but is visually demoted from the primary work cluster.
- Chat shows a compact combined shell bar above the active chat content.
- The shell bar reflects backend, scope, assistant identity, and session title from the active chat state.
- The combined shell bar retains current quick model/runtime controls and their sync behavior.
- When chat tabs are enabled, create, reuse, switch, and close all update the shell bar immediately.
- Legacy partial handoffs render explicit fallback metadata rather than blank shell state.
- Inline approvals and resume state remain in Chat and continue to render above the chat log.
- Focused UI tests cover the new labels/clustering and Chat shell bar behavior.

## Risks

### Risk: Label confusion from `Library`

`Library` is intentionally broader than `Characters`, but that breadth may be unfamiliar.

Mitigation:

- keep route IDs stable
- scope the change to user-facing copy only
- verify discoverability in tests and follow-up UX review

### Risk: Too much shell chrome above the chat log

The task surface and shell bar could create vertical clutter.

Mitigation:

- consolidate context into the existing compact shell bar
- avoid duplicating task or approval text in the shell bar
- keep the shell bar one-row-first and truncate secondary detail early

### Risk: State mismatch between actual session state and displayed shell state

Mitigation:

- derive the shell bar from the same saved session/tab state used for restoration
- add focused state-sync tests

## Rollout Notes

This slice is intentionally the first stage of a larger chat-first migration.

Later slices can build on it by:

- redirecting more `Use in Chat` flows into reused sessions
- further demoting or eventually removing `coding` from primary navigation
- tightening artifact-to-chat handoffs across notes, media, search, study, and chatbooks

## Recommendation

Implement this slice as a contained shell refinement:

- navigation label/group cleanup
- combined Chat shell bar
- preserved handoff state visibility

Do not expand scope into full route migration yet.
