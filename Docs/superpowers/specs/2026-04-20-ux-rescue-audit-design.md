# tldw_chatbook UX Rescue Audit Design

**Date:** 2026-04-20  
**Primary Repo:** `tldw_chatbook`  
**Design Lens:** Nielsen Norman Group usability heuristics, adapted to a Textual/TUI product  
**Primary Persona:** Power user  
**Primary Interaction Bias:** Mouse-friendly first, efficient for keyboard users  
**Platform Constraint:** Stay within the current Textual/TUI product model

## Goal

Define a top-down UX rescue plan for `tldw_chatbook` that makes the product usable again without abandoning its Textual/TUI architecture. The redesign should reduce orientation cost, remove structural inconsistency, simplify screen behavior, and make high-value workflows understandable and recoverable under a mouse-friendly interaction model.

This is not a visual refresh. It is a product-structure and interaction-model redesign.

The redesign should also reposition `tldw_chatbook` as a chat-first agentic workspace for programming and control, with a product model closer to tools such as Claude Code, Codex, or Gemini CLI than to a collection of unrelated utility tabs.

## Problem Statement

The current UI should be treated as structurally broken rather than locally rough. Based on repo reconnaissance, existing docs, and the current screen/navigation architecture, the main issue is not lack of features but lack of a coherent operating model.

The product currently appears to suffer from:

- Mixed navigation paradigms across app shell and screens
- Inconsistent labels, destinations, and module boundaries
- Too many competing controls and panes within individual screens
- Status and error feedback that is too often detached from the active task
- Workflow fragmentation across modules that should feel related

For a power user, this does not read as efficient density. It reads as instability and re-orientation overhead.

## Heuristic Basis

The redesign should be evaluated primarily through these NN/g heuristics:

- Visibility of system status
- Match between system and the real world
- User control and freedom
- Consistency and standards
- Recognition rather than recall
- Flexibility and efficiency of use
- Aesthetic and minimalist design
- Help users recognize, diagnose, and recover from errors

Reference:

- [10 Usability Heuristics for User Interface Design](https://www.nngroup.com/articles/ten-usability-heuristics/)

## Product Constraints

- The redesign must remain inside `Textual` and TUI constraints.
- The app should become more mouse-friendly, not more keyboard-exclusive.
- Power-user needs still matter, but expert acceleration should sit on top of a clean base rather than replace clarity.
- Existing domains such as workspaces, personas, flashcards, and quizzes should be integrated into the architecture rather than treated as afterthoughts.
- `Chat` should become the main interface for agentic programming and control, not just one module among many.

## Product Positioning

The product should be reframed as a `chat-first agentic console`.

That means:

- `Chat` is the default home and primary operating surface
- programming, repo control, tool execution, approvals, and agent progress should happen primarily through Chat
- other destinations support the chat-led workflow by organizing assets, context, study outputs, reusable behavior, and system configuration

The product should therefore feel closer to a Textual-native agentic coding console than to a tabbed multipurpose workstation.

## Recommended Approach

### Option A: Screen-by-screen cleanup

Improve the worst screens individually without changing the shell.

Why not recommended:

- Leaves the highest-order usability failure in place
- Preserves product-wide inconsistency
- Makes local improvements feel disconnected

### Option B: Expert overlay on top of the current app

Add stronger shortcuts, command palette behavior, and quick switching while leaving the product structure mostly intact.

Why not recommended:

- Helps insiders more than ordinary users
- Masks architecture problems instead of fixing them
- Is the wrong sequencing for a product described as unusable

### Option C: Shell-first rescue, then workflow repair

Unify the shell, define one screen contract, then repair major workflows inside that system.

Why recommended:

- Fixes the largest structural failures first
- Aligns with NN/g heuristics around consistency, recognition, status, and minimalist design
- Supports a mouse-friendly but still efficient product model

**Chosen approach:** Option C, shell-first rescue.

## Design Principles

- One product shell, not a collection of tools
- Chat is the main operating surface for agentic work
- One naming system for destinations and sections
- One screen contract per destination
- One dominant task per screen
- Secondary controls should be progressively disclosed
- Status belongs near the task, not only in logs or global chrome
- Power-user efficiency comes from predictability and stable structure, not from exposing everything at once

## Information Architecture

The app should be reorganized around user intent rather than implementation history.

### Global Shell Structure

The persistent shell should contain only three layers:

1. `Global header`
2. `Primary navigation`
3. `Screen workspace`

### Global Header

The header should contain:

- Product title
- Current workspace selector
- Current contextual scope when relevant
- Universal search or quick-switch entry point
- Status indicators that are global rather than task-specific
- Access to settings

### Workspaces

`Workspaces` should be treated as a global context selector, not as a secondary module. The selected workspace should scope Chat, Library, and Study and should remain visible in the shell at all times.

### Workspace Scope Rules

Workspace behavior must be explicit and consistent across destinations.

- `Chat`: mixed scope, but always visibly labeled. The active workspace is the default context, and any repo, branch, or execution-scope override must be shown inline.
- `Library`: workspace-scoped by default. Any global search or cross-workspace mode must be an explicit toggle with a visible label.
- `Study`: workspace-scoped by default.
- `Characters`: mixed scope. Characters, personas, prompts, and dictionaries may be workspace-bound or global, but every object must visibly indicate its scope.
- `Models & Tools`: global by default, with workspace-specific overrides shown only when relevant.
- `Automation / Feeds`: mixed scope. Automations and feeds must clearly show whether they are global, workspace-specific, or attached to the current chat session.
- `Settings`: global only.

Any object or view that is not workspace-scoped should display a visible `Global` marker or equivalent scope label.

### Top-Level Destinations

The recommended top-level destinations are:

- `Chat`
- `Library`
- `Study`
- `Characters`
- `Models & Tools`
- `Automation / Feeds`
- `Settings`

These labels should remain stable across the application. No legacy aliases or internal tab IDs should leak into the user-facing model.

`Chat` should also be the default landing destination and the place users return to after most major actions.

### Migration Decisions

The redesign must explicitly resolve the current split between `Chat` and the separate coding surface.

- `Chat` becomes the only primary runtime surface for agentic programming and control.
- The existing standalone `Coding` destination should be deprecated as a top-level destination during the shell rescue.
- Current coding-specific tools such as code map, repo inspection, step-by-step helpers, and repo copy or paste workflows should be migrated into Chat as contextual tools, drawers, inspectors, or launchable panels.
- `Models & Tools` may still contain coding-related configuration or utility access points, but it must not become a second primary place where users conduct agentic work.
- If temporary migration overlap is required, the standalone coding surface should be marked as transitional and removed from primary navigation once Chat reaches functional parity for core workflows.

### Destination Roles

- `Chat`: primary command surface for agentic programming, control, and interaction
- `Library`: source material management, search, ingest, packaging
- `Study`: review, reinforcement, flashcards, quizzes, guides, learning structures
- `Characters`: reusable behavior shaping
- `Models & Tools`: engine configuration, speech, evals, tooling setup, and observability
- `Automation / Feeds`: recurring inputs and monitored sources
- `Settings`: app-level preferences only

## Screen Contract

Every top-level destination should follow one common screen grammar. This is the core mechanism for restoring consistency and reducing relearning costs.

### Required Screen Layers

Every screen should contain:

1. `Screen header`
2. `Local section switcher` when applicable
3. `Primary work area`
4. `Optional inspector`
5. `Inline system status`

### Screen Header

The screen header should always contain:

- Screen title
- Current workspace scope
- Optional persona or character scope where relevant
- One primary action
- Short screen summary or state line

### Local Section Switcher

If a destination contains subareas, they must appear as one consistent section switcher directly under the screen header.

Examples:

- `Library`: Notes, Media, Search, Ingest, Chatbooks
- `Study`: Flashcards, Quizzes, Guides, Mindmaps, Paths
- `Characters`: Characters, Personas, Prompts, Dictionaries

### Primary Work Area

The center of the screen is reserved for the main task. No screen should have multiple equally dominant work regions.

Rules:

- One dominant task per screen
- One dominant content region
- Secondary controls should not compete visually with the primary task

### Optional Inspector

Metadata, advanced options, properties, and detail actions should move into a consistent right-side inspector or modal detail panel.

Rules:

- Closed by default on browse-heavy screens
- Open by default only where detailed editing is the task
- Same meaning across modules: details, advanced options, linked actions

### Inline System Status

Task status must appear near the task itself.

Examples:

- Import progress in Ingest
- Review counts and generation state in Study
- Sync status in Notes
- Tool and model execution state in Chat

## Reusable Screen Archetypes

To reduce layout drift, the product should standardize on three archetypes.

### Browser

Used for Library and parts of Characters.

Structure:

- Header
- Section switcher
- List, table, or grid
- Optional inspector

### Studio

Used for Chat and focused parts of Study.

Structure:

- Header
- Main canvas or thread
- Optional context rail
- Optional inspector

### Workflow

Used for Ingest, packaging flows, setup flows, quiz generation, and export/import tasks.

Structure:

- Header
- Stepper or workflow sections
- Focused form or staged body
- Review, result, or progress panel

## Module-Level UX Patterns

### Chat

`Chat` should use the `Studio` pattern.

Primary job:

- Talk to the agent, run programming and control tasks, inspect context, and iterate quickly

Recommended structure:

- Center: conversation thread, task plan state, tool calls, approvals, diffs, and results
- Optional left rail: sessions, recent tasks, workspaces, repositories, and reusable contexts
- Optional right inspector: model, tools, retrieval context, persona, repo context, prompt settings, and task details
- Fixed bottom composer with attachments, quick actions, and send state

Chat at-rest layout:

- Always visible: screen header, current chat thread, composer, current workspace, and current repo or execution scope when relevant
- Visible on demand: left rail for sessions or recent tasks, right inspector for settings and task details
- Collapsed by default: detailed tool traces, diffs, deep model settings, and advanced repo context
- Blocking inline cards: approvals, destructive confirmations, and active task failures
- At most one secondary side panel should be open at a time during normal use
- The default state should read as `conversation-first`, not `control panel first`

Rules:

- Workspace and persona must be visible as context chips near the screen header
- Current repository, branch, working directory, or execution scope should be visible whenever agentic programming is active
- Model and tool configuration are secondary controls and should be collapsed by default
- Streaming, token, retrieval, tool state, command execution, file changes, and approval requests should be visible inline in the active conversation area
- Users should not need to leave Chat to perform core agentic programming work
- Existing coding-specific utilities should be reframed as contextual panels, inspectors, or sub-tools launched from Chat rather than as a competing primary workflow

### Library

`Library` should use the `Browser` pattern.

Primary job:

- Find, collect, inspect, import, and package content

Sections:

- Notes
- Media
- Search
- Ingest
- Chatbooks

Behavior rules:

- Ingest should feel like a content acquisition workflow inside Library, not an unrelated mode
- Chatbooks should be reframed as packaging and export of knowledge assets
- Search scope must always be explicit
- Bulk action bars only appear when a selection exists

### Study

`Study` should be a first-class top-level destination rather than a buried utility.

Primary job:

- Practice and reinforce knowledge

Sections:

- Flashcards
- Quizzes
- Guides
- Mindmaps
- Paths

Recommended behavior:

- Default entry is a dashboard with due work, recents, and resume actions
- Flashcards and quizzes should switch into focused single-task layouts once entered
- Flashcards and quizzes should be presented as sibling sections even if implementation maturity differs

Quiz scope rule:

- `Quizzes` must be visible in the Study IA from the start of the shell redesign.
- Phase 3 does not require a fully mature quiz platform. The minimum acceptable quiz deliverable is a visible section, a coherent entry flow, a focused quiz session layout, and a clear path to generate or launch quizzes from selected content.
- Advanced quiz authoring, adaptation, or analytics can remain post-Phase-3 work.

### Characters

`Characters` should use a `Browser + Editor` model.

Primary job:

- Create and manage reusable AI behaviors

Sections:

- Characters
- Personas
- Prompts
- Dictionaries

Rules:

- Persona should not become a separate top-level destination
- Persona should be visible and swappable from Chat
- Deep editing should live in Characters

### Models & Tools

`Models & Tools` should be operational rather than exploratory.

Primary job:

- Configure the engine safely and predictably

Potential sections:

- Models
- Speech
- Evals
- Tools

Rules:

- Separate run-time controls from deep configuration
- Frequent, low-risk controls should live close to the workflow or in light inspectors
- Rare, risky, or advanced controls should live deeper in this area
- This area should configure the engine, not compete with Chat as the place where agentic work happens

### Automation / Feeds

`Automation / Feeds` should behave like a clean collection-management screen.

Primary job:

- Manage recurring inputs and monitored sources

Required visible fields:

- Source name
- Status
- Last run
- Next run
- Errors or failures

This area should read like an operational list + detail surface, not like a settings dump.

## Cross-Surface Handoffs

Because the product is chat-first, supporting destinations must have explicit handoffs into Chat and back out again.

Required handoff patterns:

- `Library -> Chat`: `Use in Chat` sends the selected note, media item, search result, or package context into the active chat session or a new session with visible confirmation.
- `Study -> Chat`: `Ask agent`, `Explain this`, or `Generate from selected material` uses current study context without forcing the user to rebuild scope manually.
- `Characters -> Chat`: `Use persona in current chat` applies the selected character or persona to the active session with a visible scope change.
- `Chat -> Library`: users can save outputs, attach generated artifacts, or package selected content into Library destinations without losing conversation state.
- `Chat -> Models & Tools`: deep configuration opens as a temporary panel or modal and returns users to the active chat context after completion.

Rules:

- Cross-surface actions should preserve the current session whenever possible.
- Handoff actions must show what context was transferred.
- Returning to Chat should preserve the active conversation, task state, and execution context.
- Supporting destinations should feel like structured context providers for Chat, not detached products.

## Interaction Model

### Mouse-First Behavior

- Every primary action must be visible and clickable
- Hover and focus states should clarify affordance rather than add decoration
- Click targets should be larger and calmer than the current dense-control pattern
- Inspectors and panes are acceptable, but basic operation should not depend on precision targeting

### Selection Model

- Browser screens must support clear single-select and multi-select states
- Selection count, active scope, and active filters must remain visible
- Bulk actions should appear only on selection
- Contextual actions should live in action bars or inspectors rather than hidden interaction models

### Progressive Disclosure

- Default views should show only routine controls
- Advanced controls go in `Advanced`, `Inspector`, or `More actions`
- Rare, dangerous, or system-level controls should be visually separated from everyday actions

### Context Visibility

Users should always be able to answer:

- Where am I?
- What workspace am I in?
- What object am I acting on?
- What mode am I in?

The UI should make these answers visible through labels, chips, and stable screen framing rather than requiring recall.

### Approvals And Safety UX

Approvals are a core part of the chat-first agentic model and should be designed as first-class interactions rather than treated as generic errors or logs.

Required approval card content:

- Requested action summary
- Why approval is needed
- Affected scope such as files, commands, repo, or system resource
- Risk level or destructive-action indicator
- Clear action choices

Default approval actions:

- `Allow once`
- `Allow for this session` when appropriate
- `Deny`
- `Review details`

Rules:

- Approval requests appear inline in Chat when Chat is the active working surface.
- Destructive or high-risk actions require stronger copy and, when appropriate, a second confirmation step.
- Users must be able to understand what will happen before granting approval.
- After approval or denial, the result should be recorded inline with the task state so the conversation remains comprehensible.

### Task Continuity And Resume

Agentic work is often long-running or interruptible. The UI should preserve enough state that users can leave and return without reconstructing context manually.

Required resume state:

- Current task summary
- Last completed step or tool action
- Pending approval state, if any
- Recent file changes or diff summary, if any
- Current workspace, repo, branch, and execution scope
- Recommended next action

Rules:

- Recent task history should be accessible from Chat.
- Resuming a task should restore visible execution context.
- Suspended or failed tasks should remain understandable without requiring users to read raw logs.

## Error Handling And Recovery

The app should stop treating logs as the main user explanation layer.

### Rules

- Errors appear at the point of failure
- Messages explain what happened, what the user can do next, and whether data is safe
- Long-running operations show progress, current phase, and cancel or retry affordances when possible
- Partial success must be explicit

### Examples

- `Ingest`: show imported count, failed count, and `Review failed items`
- `Chat`: show model or tool failure inline with `Retry`, `Edit settings`, or `Switch model`
- `Notes sync`: show conflict source and a concrete resolution path
- `Study`: show generation scope, source, and failure state in the active study surface

## Non-Negotiable UX Rules

- No duplicate navigation systems on the same screen
- No persistent left sidebar plus local sidebar plus tab strip on the same screen
- No logs as the primary explanation of task state
- No more than one visually dominant CTA per major region
- Advanced or system options must not visually compete with routine controls
- Bulk actions appear only after selection
- Empty states must explain the next recommended action
- Destructive actions must be clearly separated from routine actions

## Heuristic Audit Of Current Failures

The product should be treated as failing in this order:

1. `Shell incoherence`
2. `Screen pattern drift`
3. `Poor status and feedback placement`
4. `Workflow fragmentation`
5. `Uncontrolled information density`

These are foundational failures. If the redesign jumps straight to local screen polish, the product will still feel broken because users must repeatedly re-orient.

## Rescue Roadmap

### Phase 1: Stabilize the shell

Goal:

- Make the product understandable in roughly 30 seconds

Changes:

- Introduce the new top-level IA
- Make workspace a persistent global context
- Promote Study to top-level
- Remove overlapping navigation patterns
- Normalize labels and destinations
- Remove or demote the standalone Coding destination from primary navigation in favor of chat-first agentic flow

Outcome:

- Users can predict where things live

### Phase 2: Standardize screen contracts

Goal:

- Make every destination feel like the same product

Changes:

- Apply the common header, section switcher, work area, inspector model
- Convert destinations into Browser, Studio, or Workflow archetypes
- Remove unnecessary persistent sidebars
- Standardize empty states, bulk actions, and inspector behavior

Outcome:

- Users can transfer learning across modules

### Phase 3: Repair the three highest-value workflows

Goal:

- Make the product practically usable, not just structurally cleaner

Priority order:

- Chat
- Library
- Study

Why:

- They represent the highest-frequency and highest-value user jobs
- They demonstrate the new shell and screen model most clearly

Minimum acceptable outputs:

- `Chat`: can start and manage agentic programming tasks, approvals, diffs, and task resume without forcing a destination change to a separate coding screen
- `Library`: can ingest, find, inspect, and send content into Chat with visible context transfer
- `Study`: can expose dashboard, flashcards, and quiz entry flow, with quizzes at least available as a coherent focused flow even if advanced features remain deferred

### Phase 4: Add expert acceleration

Goal:

- Support power users without reintroducing structural noise

Changes:

- Command palette and quick switcher
- Recent items and pinned views
- Batch actions
- Saved scopes and filters
- Keyboard shortcuts that mirror visible structure

Outcome:

- Experts move faster without relying on insider knowledge

### Phase 5: Clean up secondary surfaces

Goal:

- Eliminate the last incoherent pockets

Targets:

- Models and tools separation
- Automation and feed clarity
- Character and persona cross-links
- Settings rationalization

## Success Criteria

The redesign should be considered successful when the product meets these conditions:

- Chat is the default home and primary agentic control surface
- One shell navigation model only
- One screen contract per destination
- Study is top-level and visibly contains flashcards and quizzes
- Workspace context is visible globally
- Task feedback is inline on the active screen
- Advanced controls are progressively disclosed
- No duplicate navigation chrome remains
- No blank or silent empty states remain
- No separate top-level coding destination remains in the primary navigation after Chat reaches core parity
- Every workspace-scoped or global object shows visible scope labeling
- Every long-running action in Chat, Library, and Study exposes inline progress and terminal state
- A first-time evaluator can start a core agentic task from Chat using only visible on-screen cues in 60 seconds or less
- Core workflows should require no more than one top-level destination change after landing in Chat

## Validation Model

### Heuristic Pass

For each top-level destination, confirm:

- A user can identify the purpose of the screen within 5 seconds
- The primary action is obvious
- The current context is visible
- Advanced controls are separated from routine ones
- Mistakes can be recovered from without reading docs or logs

### Task-Based Validation

Test at least these workflows:

- Start an agentic programming task from Chat in the correct workspace or repository and understand the current execution context
- Start a chat in the correct workspace with the intended persona
- Ingest new content and verify where it landed
- Find an existing note or media item from Library
- Generate or review flashcards or quizzes from Study
- Package or export a chatbook from Library

Expected outcomes:

- No more than one top-level navigation change for the listed core workflows after arriving in Chat
- No unlabeled workspace or global scope transitions during the listed workflows
- Less visual competition on each screen, measured by one clearly dominant task region and no duplicate navigation chrome
- Faster completion on common tasks, including first-run start of an agentic programming task from Chat in 60 seconds or less
- Clearer recovery when something fails, including inline error or approval state within the active region for all long-running actions

## What Not To Do

- Do not start with visual polish
- Do not add more shortcuts before fixing architecture
- Do not optimize every module equally at the start
- Do not preserve top-level destinations only because implementation history created them
- Do not preserve a separate primary coding destination if it competes with Chat as the main agentic workspace

## Immediate Planning Implications

The implementation plan should begin with shell consolidation and screen-contract standardization rather than local screen redesign in isolation. The first implementation wave should target architecture and high-frequency workflows, not cosmetic cleanup.
