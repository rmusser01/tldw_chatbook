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
- Coding

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

## Validation Model

### Heuristic Pass

For each top-level destination, confirm:

- A user can identify the purpose of the screen quickly
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

- Fewer wrong turns between destinations
- Fewer "where is this?" moments
- Less visual competition on each screen
- Faster completion on common tasks
- Clearer recovery when something fails

## What Not To Do

- Do not start with visual polish
- Do not add more shortcuts before fixing architecture
- Do not optimize every module equally at the start
- Do not preserve top-level destinations only because implementation history created them
- Do not preserve a separate primary coding destination if it competes with Chat as the main agentic workspace

## Immediate Planning Implications

The implementation plan should begin with shell consolidation and screen-contract standardization rather than local screen redesign in isolation. The first implementation wave should target architecture and high-frequency workflows, not cosmetic cleanup.
