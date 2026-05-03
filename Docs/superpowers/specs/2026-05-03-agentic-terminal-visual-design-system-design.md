# Agentic Terminal Visual Design System

Date: 2026-05-03
Status: Phase 0 visual foundation for implementation planning
Primary Repo: `tldw_chatbook`
Scope: Visual language, reusable screen grammar, semantic tokens, component anatomy, state language, and Textual implementation constraints for the Chat-first agentic shell

## Purpose

Define the reusable visual language for the Chat-first agentic shell before runtime implementation begins.

The goal is not to copy concept screenshots 1:1. The goal is to extract a durable visual system that can survive terminal width changes, keyboard-first workflows, and incremental migration from legacy screens to the master shell.

## Product Model

- Home is the first-run and returning-user orientation surface for readiness, active work, attention, and next-best actions.
- Console is the primary agentic control surface for live conversations, runs, tools, approvals, RAG, coding/control flows, and recovery.
- Top bar is primary destination navigation, not local workflow state.
- Workspaces are global context, not a buried note feature.
- Personas shape behavior, identity, prompts, dictionaries, characters, and context style.
- Flashcards and quizzes remain visible Study modules.
- Library, Search, Media, Notes, Artifacts, Handoffs, MCP, ACP, Skills, Schedules, and Workflows remain visible in the product model.
- Destination screens prepare, browse, stage, configure, or inspect work. Live work opens or follows in Console.

## Visual Principles

- Command-line credible, not decorative terminal cosplay.
- Dense enough for power users, legible enough for first-time users.
- Status and authority are visible without reading logs.
- Recovery is part of the interface, not hidden in errors.
- Beginner orientation is skippable and local to Home or destination headers.
- Power-user speed is preserved through command palette, keyboard shortcuts, dense lists, and predictable focus.
- Important states are text-labeled; color never carries meaning alone.
- Concept screenshots are references, not literal screen mandates.

## Nielsen Norman Alignment

- Visibility of system status: use readable badges, live work rows, source authority chips, and footer shortcut context.
- Match between system and real world: use product language such as Console, Workspace, Source, Artifact, Persona, Flashcards, Quizzes, Approval, and Recovery.
- User control and freedom: keep command palette, keyboard navigation, cancel/retry/recover controls, and reversible staging paths.
- Consistency and standards: use shared `.ds-*` components and state classes across destinations.
- Error prevention: surface readiness, source authority, missing auth, dry-run, and approval requirements before launch.
- Recognition rather than recall: headers explain destination purpose; tooltips and command-palette help expose compact labels.
- Flexibility and efficiency: compact density, shortcuts, search, command palette, and direct legacy routes remain available.
- Aesthetic and minimalist design: dense terminal UI should prioritize clear hierarchy over decorative effects.
- Help users diagnose and recover: recovery callouts name owner, impact, and next action.
- Help and documentation: Home and destination headers provide lightweight orientation without blocking work.

## Screen Grammar

Every full-page shell surface uses the same high-level grammar.

```text
+--------------------------------------------------------------------------------+
| Top Navigation: Home Console Library Artifacts Personas W+C ... More: Ctrl+P   |
+--------------------------------------------------------------------------------+
| Destination Header: Title | purpose | readiness | authority | primary action   |
+------------------------------+---------------------------------+---------------+
| Main Work Area               | Main Work Area                   | Inspector     |
| lists, panels, transcript,   | selected table, cards, forms,    | context,      |
| events, builder, source      | preview, composer, approvals     | provenance,   |
| staging, dashboard           |                                 | recovery      |
+------------------------------+---------------------------------+---------------+
| Footer: active shortcut context | global status | selected/running/blocked      |
+--------------------------------------------------------------------------------+
```

### Top Navigation

- Primary destination navigation only.
- Home and Console stay visible at supported widths.
- Compact labels are allowed only when full labels are exposed through tooltip, command palette, and destination headers.
- Overflow must be discoverable. A `More: Ctrl+P` hint is acceptable as the minimal V1 affordance.
- Local tabs, filters, backend state, source authority, and workflow readiness do not belong in the top nav.

### Destination Header

Every destination page begins with:

- destination title
- one-line purpose
- local readiness
- source/workspace/backend authority where relevant
- one primary action where relevant
- local recovery callout when blocked

### Main Work Area

The main area is task-specific:

- Home: dashboard, active work, attention, readiness, next-best actions.
- Console: transcript/events, staged context, composer, approvals, run inspector.
- Library: workspaces, sources, notes, media, imports, search/RAG, source staging.
- Artifacts: generated outputs, reports, datasets, Chatbooks, reuse/send-to-Console.
- Personas: characters, personas, prompts, dictionaries, behavior profiles.
- Study: flashcards, quizzes, decks, review progress, generation handoffs.
- W+C: watchlists, monitored sources, collections, follow-in-Console.
- Schedules and Workflows: timing, dry-run, retry, recipe, and launch surfaces.
- MCP, ACP, Skills, Settings: capability, auth, policy, runtime, and configuration surfaces.

### Inspector

The inspector is optional but follows one role: selected item detail, provenance, readiness, permissions, recovery, or related artifacts. It must not hide blockers that affect the main action.

### Footer

The footer shows active shortcut context and compact status. It does not infer shortcuts by scraping widgets. If no context is registered, it falls back to global commands such as command palette, help, and quit.

## Token System

Tokens are semantic first. Design discussion can use conceptual names; implementation must use Textual-safe hyphenated names.

| Role | Implementation token examples |
| --- | --- |
| Canvas and surfaces | `ds-surface-canvas`, `ds-surface-panel`, `ds-surface-panel-raised`, `ds-surface-field`, `ds-surface-overlay`, `ds-surface-divider` |
| Text | `ds-text-primary`, `ds-text-secondary`, `ds-text-muted`, `ds-text-disabled`, `ds-text-inverse`, `ds-text-code` |
| Actions | `ds-action-primary`, `ds-action-secondary`, `ds-action-hover`, `ds-action-focus`, `ds-action-disabled`, `ds-action-destructive` |
| Status | `ds-status-ready`, `ds-status-running`, `ds-status-info`, `ds-status-warning`, `ds-status-approval-required`, `ds-status-blocked`, `ds-status-error`, `ds-status-paused`, `ds-status-unsaved`, `ds-status-recovered` |
| Authority | `ds-authority-local`, `ds-authority-server`, `ds-authority-workspace`, `ds-authority-remote-only`, `ds-authority-dry-run`, `ds-authority-syncing`, `ds-authority-synced`, `ds-authority-conflict` |
| Source roles | `ds-source-role-context`, `ds-source-role-evidence`, `ds-source-role-editable-target`, `ds-source-role-output-seed` |
| Structure | `ds-structure-border`, `ds-structure-border-strong`, `ds-structure-grid-line`, `ds-structure-focus-ring`, `ds-structure-active-row`, `ds-structure-inactive-row` |
| Density | `ds-density-panel-padding`, `ds-density-row-height`, `ds-density-field-height`, `ds-density-panel-gap`, `ds-density-footer-height`, `ds-density-header-height`, `ds-density-table-cell-padding`, `ds-density-inspector-width` |

### Color Intent

- Cyan: focus, active structure, section titles, selected control outlines.
- Green: ready, success, allowed, online, synced, healthy.
- Amber: warning, approval required, unsaved, review needed, stale.
- Red: blocked, failed, denied, destructive, missing auth, unrecoverable.
- Blue: active navigation and strong active row selection.
- Gray: inactive controls, dividers, metadata, disabled copy, unselected borders.
- Near-black: canvas and panel backgrounds.

## Component Anatomy

### Top Nav

Purpose: global destination switching.

Anatomy:

- destination buttons
- active destination state
- compact/full label contract
- overflow or command-palette affordance
- tooltip for compact labels

States: active, focus, hover, unavailable only when a destination truly cannot open.

### Destination Header

Purpose: orient first-time users and expose local readiness for power users.

Anatomy:

- title
- purpose
- readiness badge
- authority badge
- primary action
- recovery callout when blocked

### Panel

Purpose: titled region with one clear job.

Anatomy:

- title
- optional subtitle
- status badge
- toolbar
- content
- empty or recovery state

### Toolbar

Purpose: local actions only.

Anatomy:

- primary action
- secondary actions
- filters or sort when local
- disabled action recovery tooltip

### Status Badge

Purpose: compact readable state.

Anatomy:

- label
- semantic class
- optional icon or ASCII marker
- tooltip or inspector detail for complex states

### Recovery Callout

Purpose: owner, problem, impact, and next action.

Structure:

```text
Owner: problem
Impact or explanation
Primary recovery action
```

### Source Role Chip

Purpose: show how a source/artifact will be used by Console or a workflow.

Roles:

- context
- evidence
- editable target
- output seed

### Approval Card

Purpose: decision surface before writes, external actions, destructive actions, or policy-bound operations.

Anatomy:

- requested action
- target
- consequence
- requesting agent/persona/run
- diff or preview
- approve/reject controls
- shortcut labels

### Event Row

Purpose: scan live work, tool calls, audits, schedules, and workflow progress.

Fields:

- timestamp
- actor
- event type
- target
- result
- duration
- recovery state if failed

### Field Row

Purpose: consistent settings/builders/forms.

Anatomy:

- label
- control
- help text
- validation state
- source of value
- restart/apply impact

### Inspector

Purpose: selected item context, provenance, permissions, readiness, recovery, and related artifacts.

### Shortcut Bar

Purpose: active shortcut context and compact status.

Rules:

- renders explicit `ShortcutContext`
- falls back to global commands
- never keeps stale screen shortcuts
- does not duplicate global navigation

## State Language

| State | Meaning | Required text |
| --- | --- | --- |
| Ready | action can proceed | `Ready` |
| Running | live work is active | `Running` |
| Paused | job/schedule/workflow is paused | `Paused` |
| Blocked | action cannot proceed | `Blocked` |
| Unavailable | capability not installed/configured | `Unavailable` |
| Approval required | user must decide | `Approval required` |
| Unsaved | local edits are pending | `Unsaved` |
| Stale | data may be outdated | `Stale` |
| Conflict | state conflicts need resolution | `Conflict` |
| Recovered | retry/recovery succeeded | `Recovered` |
| Local | local source is authoritative | `Local` |
| Server | server is authoritative | `Server` |
| Workspace | workspace scope is authoritative | `Workspace` |
| Remote-only | local CRUD is unavailable | `Remote-only` |
| Dry-run | preview/read-only mode | `Dry-run` |

## Reference Screens

Reference screens clarify reusable layout grammar only. They are not pixel targets.

### Home Dashboard Shell

```text
+--------------------------------------------------------------------------------+
| Home | Console | Library | Artifacts | Personas | W+C | ... | More: Ctrl+P      |
+--------------------------------------------------------------------------------+
| Home  Local workspace: Research Lab  Ready  Server: local                       |
| Purpose: resume active work, review blockers, and choose the next action        |
+-----------------------------+-----------------------------+--------------------+
| Attention                   | Active Work                 | Readiness          |
| [Approval required] 2       | Console run: indexing       | Local DB Ready     |
| [Blocked] MCP auth missing  | Schedule: RSS daily paused  | RAG Stale          |
| [Unsaved] Persona draft     | Workflow dry-run complete   | MCP Needs auth     |
+-----------------------------+-----------------------------+--------------------+
| Next Best Actions           | Recent Sources                                   |
| > Open Console approvals    | notes/meeting.md  media/paper.pdf  quiz deck A  |
| > Fix MCP auth              | workspace: Research Lab                          |
+--------------------------------------------------------------------------------+
| Ctrl+P palette | Enter open | R refresh | global status                         |
+--------------------------------------------------------------------------------+
```

### Console Agentic Control Surface

```text
+--------------------------------------------------------------------------------+
| Home | Console | Library | Artifacts | Personas | W+C | ... | More: Ctrl+P      |
+--------------------------------------------------------------------------------+
| Console  Workspace: Research Lab  Ready  Source stack: 4 items                  |
| Purpose: run agentic work, inspect tool use, approve writes, recover failures   |
+----------------------+---------------------------------------+-----------------+
| Staged Context       | Transcript / Event Stream             | Run Inspector   |
| [evidence] paper.pdf | user: summarize this workspace        | Provider Ready  |
| [context] notes.md   | tool: rag.search Running              | RAG Workspace   |
| [editable] draft.md  | agent: proposed edits need approval   | Approval needed |
| [output seed] quiz A | [Approval Card] View Diff Approve     | Artifacts: 2    |
+----------------------+---------------------------------------+-----------------+
| Composer: /run, /diff, /approve, attach source, ask persona                    |
+--------------------------------------------------------------------------------+
| Ctrl+Enter send | Ctrl+K commands | Esc cancel | Approval required              |
+--------------------------------------------------------------------------------+
```

### Library / Workspace Source Context

```text
+--------------------------------------------------------------------------------+
| Home | Console | Library | Artifacts | Personas | W+C | ... | More: Ctrl+P      |
+--------------------------------------------------------------------------------+
| Library  Workspace: Research Lab  Source authority: Local  Search index: Stale  |
| Purpose: browse, import, search, and stage sources for Console                  |
+-------------------------+-------------------------------+----------------------+
| Workspaces              | Sources                       | Source Inspector     |
| > Research Lab          | notes/meeting.md  Local Ready | Role: evidence       |
|   Personal              | media/paper.pdf   Indexed     | Used by: Console     |
|   Course Prep           | web/rss item      Remote-only | Recovery: reindex    |
+-------------------------+-------------------------------+----------------------+
| Toolbar: Import | Search | Reindex | Stage for Console | Use in Chat          |
+--------------------------------------------------------------------------------+
| Ctrl+F search | Space select | Ctrl+Enter stage | 3 selected             |
+--------------------------------------------------------------------------------+
```

### Study / Personas Visibility Screen

```text
+--------------------------------------------------------------------------------+
| Home | Console | Library | Artifacts | Personas | W+C | ... | More: Ctrl+P      |
+--------------------------------------------------------------------------------+
| Study  Workspace: Course Prep  Ready  Persona: Tutor                            |
| Purpose: review flashcards, take quizzes, generate study material via Console   |
+----------------------+-----------------------------+-------------------------+
| Study Modules        | Current Decks               | Persona / Generation   |
| > Flashcards         | Biology Deck  Ready         | Tutor persona active   |
| > Quizzes            | Exam Quiz     Unsaved       | Generate from sources  |
| > Review Schedule    | Missed cards  Warning       | Send to Console        |
+----------------------+-----------------------------+-------------------------+
| Toolbar: Start Review | Create Quiz | Generate Cards | Open in Console        |
+--------------------------------------------------------------------------------+
| Enter open | G generate | Ctrl+P palette | Study direct route retained        |
+--------------------------------------------------------------------------------+
```

## Textual Implementation Constraints

- Use hyphenated TCSS variables and `Theme.variables` keys.
- Avoid dotted token names in TCSS.
- Use shared classes and state classes instead of raw per-widget colors.
- Require fallback glyphs and readable text labels.
- Design for compact terminal widths.
- Preserve keyboard-first workflows and command-palette access.
- Do not rely on screenshots being present in the branch.
- Tests should assert readable labels, state classes, IDs, and route behavior, not raw color values.

## Do Not Implement Yet

- Do not redesign every destination screen in one PR.
- Do not add a second navigation system.
- Do not convert concept images into literal layouts.
- Do not hide Study, Workspaces, Personas, or handoff flows until a replacement information architecture is approved.
- Do not make aesthetic-only changes without workflow or usability value.
