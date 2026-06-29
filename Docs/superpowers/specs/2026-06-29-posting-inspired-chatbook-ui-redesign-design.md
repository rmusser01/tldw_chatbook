# Posting-Inspired Chatbook UI Redesign Design

Date: 2026-06-29
Status: User-approved design; spec review approved; pending implementation plan
Primary Repo: `tldw_chatbook`
Scope: All major Chatbook screens, shared Textual UI primitives, responsiveness contract, discoverability rules, and migration policy.

## Summary

Redesign Chatbook as a coherent Textual workbench system inspired by the engineering discipline of Darren Burns' Posting project. The goal is not to copy Posting's exact visual style. The goal is to apply the same quality bar: stable composition, dense but legible workflow surfaces, predictable focus, explicit state, visible recovery, command acceleration, and strong UI regression tests.

This is a near-complete redo of the current screen internals while keeping the approved master shell framing: destination navigation, destination header, local mode strip, workbench panes, and footer status. The redesign begins with Console as the reference implementation, then migrates every destination through the same shared widget and responsiveness contract.

The redesign also treats Chatbook's reported random freezes and lockups as a first-class engineering risk. Before major screen replacement, implementation must add instrumentation for event-loop stalls, worker backlog, timer ownership, mount/remove churn, and repeated route-switch soak behavior.

## Inputs

- `PRODUCT.md`
- `DESIGN.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- `Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md`
- `Docs/Development/textual-best-practices-analysis.md`
- `Docs/Development/textual-refactoring-plan.md`
- `backlog/tasks/task-65 - Redesign-master-shell-navigation-tabs-as-terminal-tabs.md`
- `backlog/tasks/task-103 - Define-a-shell-wide-pane-focus-keyboard-convention-for-workbench-screens.md`
- `backlog/decisions/005-console-workspace-server-readiness.md`
- `backlog/decisions/010-console-conversation-local-marks.md`
- Posting source studied from `darrenburns/posting`, including app, commands, TCSS/SCSS, widgets, file watchers, jump overlay, themes, docs, and snapshot tests.

## Problem Statement

Chatbook already has a strong product direction: local-first agentic knowledge work, terminal-native density, source authority, and recoverable workflows. The current UI implementation is inconsistent across screens and still contains legacy surfaces, direct widget manipulation, broad recomposition paths, dynamic mount/remove regions, ad hoc controls, and unclear command/focus ownership.

The user also reports random freezes or lockups after differing amounts of time. That symptom should not be treated as a cosmetic issue. A UI redesign that increases remount churn, polling, hidden actions, or blocking work would make the product worse even if it looks cleaner.

This design establishes a new shared UI system and a responsiveness contract before implementation begins.

## Goals

- Redesign all major screens using a shared Chatbook Workbench UI System.
- Keep the existing master shell and route framing.
- Replace internal widgets and per-screen visual grammar with shared primitives plus workflow-specific widgets.
- Use Console as the reference implementation.
- Preserve feature parity before retiring any legacy screen/widget path.
- Default to normal density, with compact density available and tested.
- Make core workflows discoverable without requiring the command palette.
- Apply Posting's Textual engineering practices: stable compose trees, reactive class toggles, lazy panes, typed messages, async workers, timeouts, and snapshot tests.
- Add a freeze-investigation and responsiveness gate before major replacement work.
- Make local, server, workspace, provider, sync, dry-run, blocked, and unavailable states explicit.
- Use ASCII wireframes as the binding visual reference.

## Non-Goals

- Do not clone Posting's exact theme, side accents, command layout, or jump overlay.
- Do not require users to use `Ctrl+P` to discover or complete core workflows.
- Do not turn every destination into a live agent console.
- Do not collapse Personas, Skills, MCP, ACP, Schedules, Workflows, Library, Study, and Workspaces into one generic agents area.
- Do not remove legacy widgets before feature parity and responsiveness checks pass.
- Do not make a single giant implementation PR for all screens.
- Do not treat visual polish as a substitute for workflow completion.

## Posting Reference Findings

Posting's useful lessons are mostly engineering and interaction discipline:

- Top-level layout is compact and stable: header, URL/control strip, collection browser, request editor, response area, footer.
- Main layout changes use reactive class toggles such as compact mode, horizontal/vertical layout, hidden sections, and ready states.
- Heavy tab panes use `textual.lazy.Lazy`.
- Widgets emit typed messages upward rather than sharing one giant controller path.
- Request execution uses `@work(exclusive=True)`, `httpx.AsyncClient`, explicit timeout handling, and user-readable notifications.
- File and theme watching runs as exclusive async workers rather than UI polling loops.
- Tables and editors update rows, labels, classes, and selections in place.
- The command provider is contextual, but commands largely duplicate or accelerate reachable actions.
- Snapshot tests cover density, command palette, focus switching, dialogs, request loading, variables, and stateful widgets.

Chatbook should copy those practices, not Posting's exact look.

## Architecture Scope

The redesign is a new Chatbook Workbench UI System inside the existing app framing.

```text
+------------------------------------------------------------------------------+
| TldwCli shell                                                                |
|   Title / destination tabs / global command search / global status            |
+------------------------------------------------------------------------------+
| Destination screen                                                           |
|                                                                              |
|  +---------------- DestinationHeader ----------------+                       |
|  | title | purpose | readiness | authority | action   |                      |
|  +----------------------------------------------------+                       |
|  | ModeStrip / filters / local actions                |                      |
|  +----------------+----------------------+------------+                      |
|  | SourceRail     | WorkSurface          | Inspector  |                      |
|  | lists/queues   | editor/transcript    | detail     |                      |
|  | selections     | preview/builder      | recovery   |                      |
|  +----------------+----------------------+------------+                      |
+------------------------------------------------------------------------------+
| FooterStatus: focused shortcuts | route | selected/running/blocked state     |
+------------------------------------------------------------------------------+
```

Shared widget kit:

```text
Frame
  DestinationHeader
  WorkbenchFrame
  WorkbenchPane
  ModeStrip
  CommandStrip

Lists
  SourceRail
  DenseList
  ReconciledList
  ResultRow
  SelectableRow
  StatusBadge

Work
  TranscriptSurface
  EditorPanel
  FormGrid
  OutputViewer
  PreviewPane

State
  EmptyState
  LoadingState
  BlockedState
  ErrorState
  RecoveryCallout

Input
  FieldRow
  SelectRow
  ToggleRow
  ActionButton
  TextEditor
  DataEditor
```

Workflow widgets:

```text
Console
  ConsoleContextRail
  ConsoleTranscript
  ConsoleComposerBar
  ConsoleInspectorPanel
  ConsoleSessionTabs
  ConsoleRunStatusStrip
  ConsoleSettingsSummary

Library
  LibrarySourceBrowser
  LibrarySearchSurface
  LibraryRagResultList
  LibrarySourceInspector

Notes
  NotesNavigator
  NotesEditorSurface
  NotesSyncInspector
  NotesWorkspacePanel

Media and Ingest
  MediaSourceQueue
  MediaPreviewSurface
  IngestBuilder
  ProcessingInspector

Personas, Settings, Workflows, Schedules, MCP, ACP, Skills
  Use the same frame/list/work/inspector primitives with domain-specific panels.
```

Architecture rule: domain screens own domain services and state snapshots. Shared widgets receive explicit state objects and emit Textual messages. No shared primitive should directly query DBs, mutate global app state, or know provider-specific behavior.

## Canonical Route Inventory And Migration Ownership

Implementation planning must start from the current route inventory, not only from the simplified destination list in this spec. The binding sources are:

- `tldw_chatbook/Constants.py`
- `tldw_chatbook/UI/Navigation/shell_destinations.py`
- `tldw_chatbook/UI/Navigation/screen_registry.py`

Migration ownership means "which destination workbench owns the redesign task." It does not require deleting a legacy route or changing its route ID unless a later implementation plan proves that migration is safe.

| Current route or tab | User-facing label | Migration owner | Notes |
| --- | --- | --- | --- |
| `home` | Home | Home | Command center, notifications, status, next actions. |
| `chat` | Console | Console | Live work reference implementation. |
| `coding` | Coding | Console | Coding-focused live assistant surface; keep distinct workflow if still needed. |
| `library` | Library | Library | Source and workspace hub. |
| `notes` | Notes | Library / Notes | Library-owned source workflow with dedicated Notes workbench. |
| `media` | Media | Library / Media | Library-owned source workflow with dedicated Media workbench. |
| `ingest` | Ingest | Library / Media + Ingest | Import, normalization, indexing, recovery. |
| `search` | Search | Library / Search/RAG | Retrieval and evidence workflow. |
| `conversation` | Conversations | Library / Conversations | Conversation source browsing and staging. |
| `study` | Study | Library / Study | Flashcards, quizzes, and study artifacts. |
| `artifacts` | Artifacts | Artifacts | Generated outputs, reports, datasets, bundles. |
| `chatbooks` | Chatbooks | Artifacts | Portable Chatbook context packs and exports. |
| `writing` | Writing | Artifacts / Writing | Output authoring surface; may hand off to Console and Library. |
| `research` | Research | Library / Research | Research workflow over local/server/workspace sources. |
| `personas`, `ccp`, `conversations_characters_prompts`, `characters`, `prompts` | Personas | Personas | Characters, personas, prompts, dictionaries, behavior profiles. |
| `watchlists_collections` | Watchlists | Watchlists + Collections | Monitored sources, runs, alerts, recovery. |
| `subscriptions`, `subscription` | Subscriptions | Watchlists + Collections | Legacy subscription/watchlist route. |
| `schedules` | Schedules | Schedules | Timing, triggers, retries, pauses, recovery. |
| `workflows` | Workflows | Workflows | Reusable procedures, recipes, dry-runs, outputs. |
| `mcp`, `tools_settings` | MCP | MCP | MCP servers, tools, permissions, auth, audit; `tools_settings` is legacy MCP ownership. |
| `acp` | ACP | ACP | ACP agents, sessions, runtimes, diffs, terminals. |
| `skills` | Skills | Skills | Agent Skills discovery, validation, attachments. |
| `settings` | Settings | Settings | Global preferences, accounts, storage, app behavior. |
| `customize` | Customize | Settings / Appearance | Legacy appearance customization route. |
| `llm`, `llm_management` | Models | Settings / Models | Provider and model management. |
| `stts` | Speech | Settings / Speech | Speech-to-text and text-to-speech tools/configuration. |
| `evals` | Evals | Diagnostics / Evals | Evaluation tools and benchmarking surfaces. |
| `stats` | Stats | Diagnostics / Stats | Application and usage statistics. |
| `logs` | Logs | Diagnostics / Logs | Application logs and operational debugging. |

Every route in this table receives either a destination-specific replacement or an explicit reviewed decision to consolidate it under its migration owner.

## Textual Engineering And Responsiveness Contract

Every redesigned screen and shared widget follows this contract:

```text
+------------------------------+
| compose once                 |
| keep stable widget IDs       |
| lazy-load heavy panes        |
+--------------+---------------+
               |
               v
+------------------------------+
| state changes update:        |
| classes / labels / rows      |
| not whole screen rebuilds    |
+--------------+---------------+
               |
               v
+------------------------------+
| slow work runs outside UI:   |
| worker / to_thread / async   |
| explicit timeout / cancel    |
+--------------+---------------+
               |
               v
+------------------------------+
| UI applies result snapshots  |
| with stale-result guards     |
+------------------------------+
```

Rules:

- Use small typed widgets that emit messages upward.
- Use reactive values for coarse state, but avoid broad `recompose=True` except for tiny isolated widgets.
- Prefer `set_class`, `update`, `disabled`, `display`, row reconciliation, and stable IDs over remove/remount.
- Use `Lazy(...)` or equivalent deferred construction for rarely opened heavy panes.
- Use `@work(exclusive=True)` for async work.
- Use `thread=True` or `asyncio.to_thread` for blocking DB, file, network-wrapper, or CPU paths.
- Give request, search, load, indexing, and discovery operations explicit timeout, cancellation, and stale-result checks.
- Use async watchers or event signals instead of frequent polling timers where possible.
- Timers must have a clear owner, start path, stop path, and suspend/unmount behavior.

Freeze investigation gate:

```text
+--------------------------+
| event-loop heartbeat     | logs callbacks that block the loop
| worker backlog snapshot  | records stuck or repeated workers
| timer registry audit     | tracks owner, interval, cancel path
| mount churn counter      | detects repeated rebuild loops
| screen soak test         | idle + navigation + streaming
+--------------------------+
```

Required freeze investigation artifacts:

```text
ui_heartbeat.log
worker_snapshot.log
timer_registry.log
mount_churn_summary.log
route_switch_soak_result.txt
```

No redesigned screen may retire its legacy path if it increases event-loop stalls, worker buildup, timer leaks, or mount/remove churn.

## Visual Language

The visual reset should feel like Chatbook's own "neon workbench": terminal-native, cyberpunk-cozy, restrained, local-first, and authority-aware.

```text
+------------------------------------------------------------------------------+
| Global shell: quiet, stable, always navigable                                 |
+------------------------------------------------------------------------------+
| Destination header: title | purpose | readiness | authority | primary action  |
+------------------------------------------------------------------------------+
| Local mode strip: compact tabs / filters / scope / visible commands           |
+------------------------------------------------------------------------------+
| Workbench panes: source rail | live work surface | inspector / recovery       |
+------------------------------------------------------------------------------+
| Footer: focused shortcuts | active route | running / blocked / selected state |
+------------------------------------------------------------------------------+
```

Style rules:

- Normal density is default. Compact mode is global and tested.
- Accent color is for focus, selected state, primary actions, and meaningful status only.
- Every important state has text, not color alone.
- No side-stripe accents, gradient text, glass effects, decorative motion, or nested cards.
- Buttons are compact command controls, not oversized web calls to action.
- Inputs, tables, list rows, transcript messages, and tabs share one focus language.
- Hover and focus never change dimensions.
- Empty, loading, blocked, disabled, and error states are first-class surfaces.

Example:

```text
+-- Context --------------------------------------------------+
| [workspace] Local research     [ready] 12 sources staged    |
| > note: API migration plan                      synced      |
|   media: transcript-042                         indexed     |
|   convo: router notes                           local       |
+------------------------------------------------------------+

+-- Console --------------------------------------------------+
| User     Compare the router migration notes                 |
| Assistant Streaming... [provider: local] [sources: 3]       |
|                                                            |
| [Send] [Attach] [Use staged context] [Stop]                 |
+------------------------------------------------------------+

+-- Inspector -----------------------------------------------+
| Selected: router notes                                      |
| Authority: workspace-local                                  |
| Recovery: none                                              |
| Next: cite source or remove from context                    |
+------------------------------------------------------------+
```

## Command Palette Discoverability Rule

The command palette is not where core UX goes to hide. It is for speed, search, and secondary access to actions that are discoverable somewhere else.

```text
Visible UI owns
  primary actions
  recovery actions
  destructive action confirmation
  current mode switches
  source/workspace/provider readiness
  actions needed to complete the current workflow

Command palette owns
  keyboard-speed duplicates
  global navigation
  fuzzy search across commands
  rarely used utilities
  advanced diagnostics
  view/layout/density toggles
  power-user shortcuts
```

For every screen, apply the new-user test:

```text
Can a new user complete this workflow without Ctrl+P?

If no:
  the missing action belongs visibly in the workbench, header, mode strip,
  inspector, or recovery callout.
```

## Console Reference Implementation

Console is the reference screen because it has live work, provider state, staged context, transcript streaming, message actions, settings, and recovery.

```text
+------------------------------------------------------------------------------+
| Console | Live work surface | workspace: local-research | provider: ready     |
+------------------------------------------------------------------------------+
| Mode: Chat | Staged: 3 sources | Tools: enabled | Run: idle                  |
+----------------------+-----------------------------------+-------------------+
| Context              | Transcript / Event Stream          | Inspector         |
|                      |                                   |                   |
| [workspace] local    | User                              | Selected message  |
| [sources] 3 staged   |   Summarize these notes...        | Actions           |
| [convos] recent      |                                   | Provenance        |
|                      | Assistant                         | Recovery          |
| Saved prompts        |   Streaming response...           | Settings summary  |
| Attachments          |                                   |                   |
|                      | [composer......................................]      |
|                      | [Send] [Attach] [Use context] [Stop]                 |
+----------------------+-----------------------------------+-------------------+
| Footer: F6 pane | F1 help | run status | selected source/message            |
+------------------------------------------------------------------------------+
```

Console parity checklist:

- Multi-session tabs.
- Provider and model selection.
- Streaming and non-streaming fallback.
- Stop, retry, regenerate, continue, edit, delete, copy, feedback.
- Staged Library, Notes, Media, and Conversation context.
- Attachments and image handling.
- Tool-call visibility and result messages.
- Workspace context and readiness.
- Recovery for missing provider, blocked source, unavailable RAG, failed run.
- Existing keyboard flows and command palette entries.
- Existing persistence and handoff behavior.

Console engineering requirements:

- Keep and expand the current `ConsoleTranscript` row reconciliation pattern.
- Do not fully remount the transcript during streaming unless message structure changes.
- Composer, context rail, inspector, and status strip update from explicit state snapshots.
- Provider calls and long-running work are cancellable and never block the UI loop.
- Streaming sync timers must have owner, start, stop, and stale-screen guards.
- Console soak tests cover streaming, stop/retry/regenerate, tab switching, and route switching.

## Cross-Screen Migration Map

Migration order:

```text
1. Responsiveness instrumentation and current-state baseline
2. Shared workbench primitives and shell-wide focus/help conventions
3. Console reference implementation, including Coding ownership decision
4. Home command center
5. Library research workbench
6. Notes writing/research workbench
7. Media + Ingest source workbench
8. Artifacts, Chatbooks, Writing, and Research ownership decisions
9. Personas administrative workbench
10. Settings, Models, Speech, and Customize administrative workbenches
11. Watchlists + Collections, Schedules, and Workflows operational workbenches
12. MCP / ACP / Skills capability workbenches
13. Evals / Stats / Logs diagnostics workbenches
```

Destination patterns:

```text
Console
  Context rail | Transcript/live run | Inspector
  Purpose: live work, streaming, tool calls, recovery

Library
  Source rail | Search/RAG/source preview | Inspector
  Purpose: browse, stage, retrieve, hand off to Console

Notes
  Navigator | Editor/preview | Inspector/sync/provenance
  Purpose: write, organize, sync, stage as source

Media + Ingest
  Source queue | Ingest/review/preview | Processing inspector
  Purpose: import, normalize, chunk, index, recover failed jobs

Personas
  Persona list | Profile/character editor | Usage/test inspector
  Purpose: configure behavior, preview, validate handoff

Settings
  Category rail | Settings form surface | Readiness/security inspector
  Purpose: configure providers, privacy, storage, sync, runtime boundaries

Watchlists + Collections / Schedules / Workflows
  Object list | Builder/editor | Run/readiness inspector
  Purpose: monitored sources, operational setup, dry-run, handoff, scheduling state

MCP / ACP / Skills
  Capability list | Configuration/test surface | Authority inspector
  Purpose: expose availability, permissions, blocked states, test results
```

Common per-screen requirements:

- Each destination has a real header: purpose, readiness, authority, primary action.
- Each destination has visible empty, loading, error, and blocked states.
- Each destination has a focused visible command strip for frequent actions.
- Advanced actions may appear in the command palette only when the workflow is still completable from visible UI.
- Every list becomes a stable reconciled list or table.
- Heavy detail panes lazy-load and update from explicit state snapshots.
- Timers and workers are owned by the screen or service and cancelled on suspend or unmount.
- Existing old UI remains until the replacement reaches feature parity.

Highest freeze-audit priority:

```text
Console streaming
Notes autosave/search/sync
Library Search/RAG and dynamic panes
Media ingestion/background processing
Settings provider/model discovery
```

## Interaction, Focus, And Help

The UI is keyboard-first but not keyboard-secret.

```text
Discoverability ladder
+--------------------------------------------------+
| 1. Visible controls for current workflow          |
| 2. Inspector/recovery actions for selected state  |
| 3. Footer hints for focused region                |
| 4. F1 contextual help                             |
| 5. Ctrl+P command palette for search/speed        |
+--------------------------------------------------+
```

Focus model:

```text
Global focus zones
  shell nav
  destination header / mode strip
  left pane
  main work surface
  inspector
  footer is informational only

Pane movement
  Tab / Shift+Tab: normal widget traversal
  F6 / Shift+F6: next or previous pane
  Esc: leave local edit/action mode
  F1: contextual help for focused widget or pane
```

Rules:

- No Posting-style jump overlay.
- All controls have visible focus states.
- Focus never disappears after route switches, modal close, worker completion, or screen refresh.
- Text inputs and text areas keep normal editing behavior.
- Global pane keys must not steal common editing shortcuts.
- Disabled controls explain why and point to recovery.
- Footer hints update based on the focused region.
- F1 help is contextual to the focused widget or pane and names available actions, state meanings, and blockers.

Footer examples:

```text
Console transcript focused:
  j/k select message | Enter actions | Esc clear | F6 next pane | F1 help

Notes editor focused:
  Ctrl+S save | F6 inspector | F1 help | autosave: ready

Library source row focused:
  Enter preview | Space stage | F6 inspector | source: local indexed
```

## State And Recovery

Every screen should operate from explicit state snapshots.

```text
Screen state snapshot
+--------------------------------------------------+
| data: visible rows, selected item, active mode    |
| readiness: local/server/workspace/provider state  |
| authority: local, server, synced, dry-run, remote |
| work: idle, loading, streaming, saving, syncing   |
| recovery: blocked reason + next action            |
| focus: current pane/widget intent                 |
+--------------------------------------------------+
```

Recovery callout structure:

```text
Owner: Console / Library / Notes / Settings / runtime service
Problem: provider missing, source unavailable, sync conflict, job failed
Impact: what cannot be done right now
Next action: visible button or guided path
```

No silent failure states:

- Disabled controls explain why.
- Loading states show what is loading.
- Empty states say what action makes sense next.
- Worker failures land in the owning pane or inspector, not only logs.
- Local, server, workspace, synced, dry-run, and remote-only are text-labeled.
- Stale async results are ignored if the user changed mode, route, query, or selected item.

## Verification Gates

Before a destination replacement retires legacy widgets:

```text
[ ] Feature parity checklist complete
[ ] Event-loop heartbeat shows no new stalls
[ ] Worker backlog does not grow during soak
[ ] Timer registry shows no leaked screen timers
[ ] Mount/remove churn stays bounded during repeated actions
[ ] Normal + compact snapshots pass
[ ] Focus traversal and F1 help tests pass
[ ] Route switching preserves or restores sensible focus
[ ] Core workflow works without command palette
[ ] Command palette duplicates visible workflows and exposes utilities
```

Required UI checks:

```text
snapshot: normal + compact
snapshot: command palette
snapshot: focus states
interaction: repeated route switching
interaction: streaming / long-running worker
interaction: F1 help and footer hints
soak: idle for differing durations with heartbeat logging
```

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/NNN-chatbook-workbench-ui-system.md`

Reason: This redesign creates a long-lived UX and application structure, shared widget boundaries, keyboard/focus conventions, command palette discoverability policy, instrumentation expectations, and migration policy. The ADR must be created before implementation begins and linked from the implementation plan and related Backlog task.

## Implementation Policy

Implementation should proceed in these gates:

```text
1. Create ADR and Backlog task.
2. Add responsiveness instrumentation and capture current baseline.
3. Build shared workbench primitives.
4. Migrate Console as the reference implementation.
5. Verify Console parity, focus, discoverability, and responsiveness.
6. Migrate destinations one at a time in the approved order.
7. Remove each destination's legacy widgets only after its gates pass.
```

Do not implement all screens in one change. Each destination or migration owner should have a task with acceptance criteria for parity, discoverability, instrumentation, snapshots, interaction tests, route inventory coverage, and legacy removal.

## Open Questions For Implementation Planning

- Whether instrumentation should be always available behind config or test-only.
- Exact timer registry shape and whether it should wrap Textual timers globally or be screen-owned.
- Exact snapshot test harness to use for Textual UI comparisons in this repo.
- Whether F1 contextual help extends the current footer/status components or introduces a new help panel primitive.
- How to split Backlog tasks so each task is atomic but still preserves the approved migration order.
