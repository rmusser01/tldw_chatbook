# Destination Layout And IA Contracts Design

Date: 2026-05-06
Status: User-approved design; spec review approved; implementation verified
Primary Repo: `tldw_chatbook`
Scope: Phase 3.0 design gate for top-level destination layouts, major subflows, Textual-native ASCII wireframes, and non-binding image-generation reference briefs.

## Summary

The product-maturity roadmap already verifies the shell and the core agentic loop. It does not yet define binding screen-level layout contracts for each destination and major subflow.

Add **Phase 3.0: Destination Layout And IA Contracts** before additional Phase 3 Knowledge/Study feature work continues. Any Phase 3 PR already in flight can merge if it satisfies its existing workflow contract. After that, new Phase 3+ screen work should either follow the approved destination contract or document a reviewed deviation.

This is a contract phase, not a visual polish phase. The goal is to define what every screen must make clear so later work does not produce attractive but unusable surfaces.

## Inputs

- `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- `Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md`
- `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- `Docs/Design/master-shell-design-system-contract.md`
- `Docs/Design/master-shell-route-inventory.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`

## Problem Statement

The current plan is workflow-first. That is correct, but incomplete.

The app now has top-level destinations for Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, and Settings. Many destinations still bridge legacy screens and routes. Without explicit layout contracts:

- future PRs can improve a workflow while leaving the screen hard to understand.
- legacy direct routes can keep their own visual grammar indefinitely.
- Library, Study, Workspaces, Search/RAG, and Import/Export can become fragmented.
- generated reference images can be mistaken for literal implementation requirements.
- Phase 6 can become the place where structural layout problems are discovered too late.

Phase 3.0 fixes this by defining destination-owned layout contracts before deeper Knowledge/Study visual work continues.

## Goals

- Define one binding layout and IA contract per top-level destination.
- Define concise layout contracts for major subflows under the destination that owns them.
- Preserve the verified master shell: Home first, Console second, Console as the live agentic surface.
- Keep screens Textual-native, keyboard-first, mouse-safe, and dense enough for power users.
- Add enough beginner orientation to reduce recall burden without slowing repeated use.
- Make empty, loading, blocked, error, and recovery states first-class layout requirements.
- Make local/server/workspace/source authority visible where relevant.
- Include ASCII wireframes as the binding visual reference.
- Include one non-binding image-generation reference brief per top-level destination.
- Prevent Phase 3+ work from closing when affected screens remain visually broken, structurally confusing, or unusable.

## Non-Goals

- Do not rewrite every screen in Phase 3.0.
- Do not implement the image references as 1:1 screenshots.
- Do not treat visual style as a substitute for workflow completion.
- Do not turn every destination into a live agent surface.
- Do not collapse ACP into MCP or Skills into Personas.
- Do not move legacy route IDs unless a later implementation plan proves that migration is safe.
- Do not make subflow contracts as detailed as feature specifications.

## Roadmap Placement

Phase 3.0 is inserted before further Phase 3 Knowledge/Study implementation after any already-open Phase 3 PR predating this gate is merged or closed.

Phase 3.0 deliverables:

- parent destination layout contract.
- appendix for every top-level destination.
- appendix for major subflows under their owning destination.
- ASCII wireframe for each top-level destination and major subflow.
- image-generation reference brief for each top-level destination.
- roadmap/backlog update that marks Phase 3.0 as a prerequisite for later Phase 3+ visual work.

Phase 3.0 does not require runtime screen rewrites by itself. Later PR-sized gates implement the relevant contracts as they touch real workflows.

## Authority Model

| Artifact | Authority | Rule |
| --- | --- | --- |
| Text contract | Binding | Required regions, states, ownership, focus path, and source/recovery placement must be followed. |
| ASCII wireframe | Binding | Defines structural layout and relative information hierarchy, not exact pixel or column sizes. |
| Image-generation brief | Non-binding | Guides mood, density, hierarchy, and visual language only. |
| Generated image | Non-binding | Inspiration reference only. If it conflicts with text or ASCII, text and ASCII win. |
| Existing implementation | Informational | Current screens are evidence, not automatically the desired future layout. |
| Future implementation deviation | Review required | Must document why the contract should change or why this PR needs a scoped exception. |

## Minimum Viable Contract Rule

Every destination appendix must answer only these questions:

1. What does this destination own?
2. What does the user need to do here?
3. What regions must be visible?
4. What is the primary keyboard/focus path?
5. What states must be recoverable?
6. How does it hand off to Console?
7. What source/runtime authority must be visible?

If an appendix starts defining full service behavior, DB shape, or detailed feature implementation, it is too broad and belongs in a later implementation plan.

## Global Screen Grammar

Every full-page destination follows this grammar unless its appendix documents a narrower layout.

```text
+--------------------------------------------------------------------------------+
| Global Nav: Home Console Library Artifacts Personas W+C Schedules ... Ctrl+P   |
+--------------------------------------------------------------------------------+
| Destination Header: title | purpose | readiness | authority | primary action   |
+--------------------------------------------------------------------------------+
| Local Nav / Mode Bar: tabs, filters, scope, workspace, source set              |
+------------------------+--------------------------------------+----------------+
| Primary List / Queue   | Main Workspace / Preview / Builder   | Inspector      |
| browse, select, run    | compose, review, edit, compare       | detail, state, |
| status, recents        | staged context, output, form         | provenance,    |
|                        |                                      | recovery       |
+------------------------+--------------------------------------+----------------+
| Footer: focused shortcuts | active route | selected/running/blocked status     |
+--------------------------------------------------------------------------------+
```

Not every screen needs every region. The binding requirement is that source selection, generated outputs, runtime readiness, and recoverable blockers must have an obvious place.

## Region Rules

### Global Nav

- Shows top-level destinations only.
- Keeps Home and Console reachable at every supported width.
- Uses compact labels only when full names remain discoverable through header, tooltip, command palette, or help text.
- Does not contain source scope, runtime readiness, active filters, or workflow state.

### Destination Header

Every destination header includes:

- title.
- one-line purpose.
- readiness/status label.
- authority/scope label when relevant.
- one primary action when the screen has an obvious next action.
- recovery callout when the primary action is blocked.

### Local Nav Or Mode Bar

The local mode bar owns destination-specific modes, tabs, filters, workspace selectors, and source sets. It must not duplicate global nav.

### Primary List Or Queue

The left region normally contains browseable objects, queues, recent items, sources, runs, tools, skills, personas, or schedules.

### Main Workspace

The center region is where the user reads, edits, builds, reviews, asks, previews, or runs the selected work.

### Inspector

The inspector shows selected item detail, provenance, readiness, permissions, source authority, recovery, or related artifacts. It must not hide blockers that affect primary actions.

### Footer

The footer shows active shortcut context and compact status. It must not retain stale shortcuts after navigation.

## Terminal Size Rules

Phase 3.0 contracts must specify behavior for:

- **Compact**: narrow terminal where the inspector can collapse below or behind an explicit command. Home and Console remain reachable. Primary actions remain keyboard reachable.
- **Default/Laptop**: the standard three-region layout should fit when the destination needs list, workspace, and inspector.
- **Large**: additional width may expose inspectors, histories, or secondary panels, but must not create a separate interaction model.

No destination may require a large terminal to complete its primary workflow.

## Focus And Shortcut Rules

- First focus lands on the primary safe action or the first meaningful local control.
- Tab order follows header action -> local modes -> primary list -> main workspace -> inspector -> footer/help.
- `Ctrl+P` or the command palette remains the fallback for destination switching and hidden actions.
- `Esc` or equivalent recovery returns from local detail/subflow to the owning destination overview when safe.
- Disabled actions must expose why they are disabled and what to do next.
- Repeated power-user workflows must be possible without mouse-only steps.

## State Rules

Every affected screen or subflow must define these states when applicable:

- **Empty**: no local data yet; show what the screen is for and the next safe action.
- **Loading**: show what is loading and whether the user can navigate away.
- **Blocked**: show owner, cause, impact, and recovery action.
- **Error**: show user-readable failure, retry path, and where detail/logs live.
- **Missing provider/model/runtime**: show readiness owner and route to setup.
- **Missing optional dependency**: show capability impact and install/configure path.
- **Server unavailable**: show local/server authority and whether local mode still works.
- **Permission/approval required**: show requested action, risk, and approve/reject path.

## Source And Authority Rules

Use explicit readable labels for:

- local.
- server.
- workspace.
- remote-only.
- dry-run.
- syncing/synced/conflict.
- evidence/context/editable-target/output-seed source roles.

Authority labels belong in destination headers, local mode bars, inspectors, or source chips. They do not belong in global nav.

## Console Handoff Rules

Console remains the only live agent conversation/run surface.

Destinations can:

- stage context.
- prepare configuration.
- preview inputs.
- launch live work.
- follow running work.
- inspect historical output.
- reopen saved work.

Destinations should not become alternate live-run consoles. When live work starts or is followed, the user should enter Console with visible staged context, launch provenance, source authority, and recovery status.

## Route Owner Map

| Owner destination | Routes and subflows owned |
| --- | --- |
| Home | `home`, notifications, active work, next-best actions, lightweight run controls |
| Console | `chat`, live agent/chat/RAG/tool/approval/run surface |
| Library | `notes`, `media`, `ingest`, `search`, `conversation`, `study`, Search/RAG, Import/Export, Workspaces, Study Dashboard, Flashcards, Quizzes, source detail |
| Artifacts | `artifacts`, `chatbooks`, generated outputs, exports, bundles, reports, datasets |
| Personas | `personas`, `ccp`, characters, prompts, dictionaries, lore/world books, behavior profiles |
| W+C | `watchlists_collections`, `subscription`, `subscriptions`, watchlists, collections, feeds, alerts, run history |
| Schedules | `schedules`, schedule detail, run history, pause/resume/retry |
| Workflows | `workflows`, workflow builder, workflow run detail, approvals |
| MCP | `mcp`, `tools_settings`, MCP tools/resources/servers/permissions/audit |
| ACP | `acp`, ACP agents/sessions/runtime readiness |
| Skills | `skills`, Agent Skills discovery/import/detail/validation/edit/attach |
| Settings | `settings`, `customize`, global preferences, providers, storage, appearance, diagnostics |

Legacy routes remain searchable and routable during migration. The owner destination defines the future layout contract.

## Destination Contracts

### Home

User goal: understand system status and take immediate action.

Screen role: cross-module dashboard, notification center, status page, and lightweight control surface.

Binding regions:

- readiness summary.
- notifications/attention queue.
- active work with approve/pause/resume/retry/open controls.
- next-best actions.
- recent/resumable work.
- compact inspector for selected alert or active item.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Home | System status, notifications, active work | Ready | Local | New Console  |
+--------------------------------------------------------------------------------+
| Scope: All modules | Filter: Needs attention / Running / Recent                |
+----------------------+-------------------------------+-------------------------+
| Attention Queue      | Active Work                   | Inspector               |
| ! Provider missing   | RUN Watchlist: daily papers   | Selected item           |
| ! 2 approvals        | PAUSED Workflow: digest       | Owner / impact          |
| i New artifact       | WAIT ACP runtime              | Approve Pause Retry     |
+----------------------+-------------------------------+-------------------------+
| Next Best Actions: Configure provider | Review approvals | Resume Chatbook       |
+--------------------------------------------------------------------------------+
```

Primary actions:

- start or resume Console.
- review notification.
- approve/pause/resume/retry active work.
- open details.

Focus path: attention queue -> selected item controls -> active work -> next-best actions -> recent work.

Console handoff: active work and recent Chatbooks can open or resume in Console with provenance.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Home destination in a local-first agentic knowledge console. Show a compact dashboard with global readiness, notifications, active agent/schedule/workflow runs, next-best actions, recent Chatbooks, and inline approve/pause/retry/open controls. Use dense bordered panes, readable status labels, source/runtime badges, and command-palette affordance. Avoid glossy SaaS cards, browser chrome, and decorative widgets that do not expose action or status.
```

QA checks:

- Home shows status from more than one module without hiding the selected item's action target.
- A mixed active-work scenario preserves access to both watchlist/run controls and Chatbook artifact resume controls.
- Compact terminal still exposes Home, Console, and active-work recovery actions.

### Console

User goal: ask, control, approve, inspect, and recover live agentic work.

Screen role: primary live chat, RAG, tool, approval, ACP, MCP, workflow, and run-control surface.

Binding regions:

- transcript/event stream.
- staged context/source tray.
- composer.
- run/tool/approval inspector.
- artifact links and save/reopen controls.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Console | Live agent control, chat, RAG, tools, approvals | Ready | Workspace  |
+--------------------------------------------------------------------------------+
| Mode: Chat / RAG / Run Follow | Persona: Default | Sources: 3 staged          |
+----------------------+--------------------------------------+------------------+
| Staged Context       | Transcript / Event Stream             | Run Inspector    |
| [evidence] note.md   | User: summarize these sources         | Tools ready: 4   |
| [context] chatbook   | Assistant: grounded answer...         | Approval needed  |
| [output seed] draft  | Tool: search complete                 | Artifacts: 1     |
+----------------------+--------------------------------------+------------------+
| Composer: Ask or command...                          Send | Save Chatbook      |
+--------------------------------------------------------------------------------+
```

Primary actions:

- send message/run command.
- review staged context.
- approve/reject tool or run action.
- save/update artifact or Chatbook.

Focus path: staged context -> transcript -> composer -> inspector approvals -> artifact controls.

Console handoff: Console consumes all live-work launches and staged context from other destinations.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Console destination in a local-first agentic programming and knowledge control surface. Show a transcript/event stream, staged source context tray, composer, tool calls, approvals, RAG evidence, MCP/ACP run status, and artifact save controls. Use dark terminal panels, high-density text, semantic badges, visible source roles, and a clear keyboard-first footer. Avoid a simple chat app look; this should feel closer to Codex, Claude Code, or Gemini CLI adapted to local knowledge work.
```

QA checks:

- Staged sources remain visible before send.
- User can remove or change staged source role before send.
- Blocked generation shows cause and setup/retry path.

### Library

User goal: ingest, browse, organize, search, retrieve, and reuse source material.

Screen role: source material and knowledge access hub.

Binding regions:

- source/mode browser.
- Search/RAG and Import/Export modes.
- Workspaces and Collections scope selectors.
- source preview/detail.
- inspector with metadata, authority, and Console/study actions.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Library | Sources, imports, Search/RAG, Workspaces, Study | Ready | Local      |
+--------------------------------------------------------------------------------+
| Modes: Sources Search/RAG Import/Export Workspaces Study Flashcards Quizzes    |
+----------------------+--------------------------------------+------------------+
| Source Browser       | Source Detail / Search Results        | Source Inspector |
| Notes                | Title, content preview, chunks        | Authority: local |
| Media                | RAG query and evidence list           | Use in Console   |
| Conversations        | Study generation entry                | Generate Study   |
+----------------------+--------------------------------------+------------------+
| Footer: / search | Enter open | C stage in Console | I import | E export       |
+--------------------------------------------------------------------------------+
```

Primary actions:

- search/query selected sources.
- import/export sources.
- stage source/evidence into Console.
- generate flashcards or quizzes from selected material.
- open workspace or source detail.

Focus path: mode bar -> source list -> source/search detail -> inspector actions.

Console handoff: selected sources, RAG results, notes, media, conversations, and study outputs stage into Console with source roles.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Library destination in a local-first agentic knowledge console. Show global nav with Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, Settings. The Library screen should include local modes for Sources, Search/RAG, Import/Export, Workspaces, Flashcards, Quizzes, Notes, and Media. Use dense bordered panes, visible local/server/workspace authority badges, a selected source detail inspector, and a primary action to ask in Console. Avoid web cards, browser chrome, floating modals, or glossy SaaS dashboard styling.
```

QA checks:

- Search/RAG is reachable from Library without knowing legacy routes.
- Import/Export appears under Library and does not blur with Artifact export.
- Study entry points make Flashcards and Quizzes visible without creating a top-level Study destination.
- The existing `study` route and Study Dashboard remain Library-owned and preserve section routing from Library.

### Artifacts

User goal: find, inspect, reopen, export, and reuse generated outputs.

Screen role: generated and portable output hub, including Chatbooks.

Binding regions:

- artifact type filters.
- artifact/output list.
- artifact preview/detail.
- provenance/reuse/export inspector.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Artifacts | Outputs, Chatbooks, reports, datasets, exports | Ready | Local     |
+--------------------------------------------------------------------------------+
| Types: All Chatbooks Reports Datasets Drafts Exports | Sort: Recent            |
+----------------------+--------------------------------------+------------------+
| Artifact List        | Artifact Preview                     | Provenance       |
| Chatbook: RAG answer | Saved response and source summary    | Created in Console|
| Report: run output   | Export/package status                | Reopen Console   |
| Dataset: extracted   |                                      | Export / Bundle  |
+----------------------+--------------------------------------+------------------+
| Footer: Enter preview | C reopen in Console | X export | B bundle             |
+--------------------------------------------------------------------------------+
```

Primary actions:

- reopen in Console.
- export.
- bundle/import.
- attach to workflow.

Focus path: type filters -> artifact list -> preview -> provenance/actions.

Console handoff: Chatbooks and artifacts reopen into Console with saved-response provenance and source authority.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Artifacts destination in a local-first agentic console. Show generated outputs, Chatbooks, reports, datasets, drafts, and export bundles. Include an artifact list, selected artifact preview, provenance inspector, reopen-in-Console action, export/bundle actions, and clear distinction from raw Library sources. Use dense terminal panels, semantic status badges, and source provenance chips. Avoid making it look like a file manager or web gallery.
```

QA checks:

- Chatbooks are visible as first-class artifacts.
- Artifact source provenance is visible before reopen/export.
- Artifacts are not presented as raw source material.

### Personas

User goal: configure behavior, identity, prompts, dictionaries, characters, and lore for agent work.

Screen role: behavior and identity management.

Binding regions:

- persona/character list.
- behavior profile detail.
- edit/import/export controls.
- attachment/readiness inspector.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Personas | Behavior, characters, prompts, lore | Ready | Local/Server          |
+--------------------------------------------------------------------------------+
| Modes: Personas Characters Prompts Dictionaries Lore Import/Export             |
+----------------------+--------------------------------------+------------------+
| Persona List         | Behavior Profile Detail              | Attachments      |
| Research Analyst     | Goals, tone, constraints, exemplars  | Console: ready   |
| Fiction Character    | Dictionaries and lore links          | Skills: 2        |
| Coding Assistant     |                                      | Export / Import  |
+----------------------+--------------------------------------+------------------+
| Footer: N new | Enter edit | C attach to Console | X export                 |
+--------------------------------------------------------------------------------+
```

Primary actions:

- create/edit persona.
- import/export persona or character.
- attach to Console, Workflow, ACP, or Skill recommendation.

Focus path: mode bar -> persona list -> profile detail -> attachment inspector.

Console handoff: selected persona can attach to a Console session with visible behavior profile summary.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Personas destination in a local-first agentic console. Show personas, characters, prompts, dictionaries, lore/world books, archetypes, exemplars, and behavior profiles. Include a list, detailed behavior editor/preview, import/export controls, and an attachment inspector for Console, Workflows, ACP agents, and Skills. Use dense terminal panes and readable labels. Avoid avatar-heavy social UI or generic chatbot settings.
```

QA checks:

- User can tell whether they are editing behavior, character data, prompt text, or lore.
- Attachment target and readiness are visible before launch.
- Import/export failures are recoverable.

### W+C

User goal: monitor sources and manage curated collections.

Screen role: Watchlists and Collections as local/server parity destination.

Binding regions:

- top-level internal tabs: Watchlists and Collections.
- list of watchlists/collections.
- run/feed/item detail.
- status/history/retry inspector.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| W+C | Watchlists and Collections | Mixed readiness | Local/Server              |
+--------------------------------------------------------------------------------+
| Tabs: Watchlists Collections | Filter: Running Failed Recent Alerts            |
+----------------------+--------------------------------------+------------------+
| Watch/Collection List| Detail / Items / Runs                 | Status Inspector |
| Watch: arxiv papers  | Latest run, scraped items, outputs   | Retry/backoff    |
| Collection: reading  | Highlights, saved searches, feeds   | Follow Console   |
+----------------------+--------------------------------------+------------------+
| Footer: N new | R run | C follow in Console | A alerts | X export           |
+--------------------------------------------------------------------------------+
```

Primary actions:

- create/edit watchlist or collection.
- run/retry watchlist.
- inspect items/outputs.
- follow live work in Console.
- import/export.

Focus path: tabs -> list -> detail -> status/history inspector.

Console handoff: active runs, outputs, and selected items can follow or stage into Console.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the W+C destination, short for Watchlists+Collections. Show two internal tabs: Watchlists for monitored sources, jobs, runs, alerts, retry/backoff, scraped items, and outputs; Collections for curated reading/content items, highlights, saved searches, feeds, archives, and templates. Include local/server authority badges, run status, alert state, and follow-in-Console action. Avoid generic bookmark manager visuals or web dashboard cards.
```

QA checks:

- Watchlists and Collections are visibly distinct.
- Run history and retry/backoff are visible for watchlists.
- Collection items can feed Library/RAG or Console without becoming Artifacts by default.

### Schedules

User goal: control when work runs.

Screen role: timing, triggers, run health, pause/resume/retry, and history.

Binding regions:

- schedule list.
- schedule detail/calendar/trigger view.
- run history.
- control/recovery inspector.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Schedules | When work runs | Ready | Local/Server                              |
+--------------------------------------------------------------------------------+
| Filter: Active Paused Failed Upcoming | Scope: All Workflows/Watchlists        |
+----------------------+--------------------------------------+------------------+
| Schedule List        | Schedule Detail / Upcoming Runs       | Controls         |
| Daily digest         | Cron, next run, last run, target      | Pause Resume     |
| Weekly review        | History and missed-run recovery       | Retry Open Run   |
+----------------------+--------------------------------------+------------------+
| Footer: N new | Space pause/resume | R retry | C open run in Console          |
+--------------------------------------------------------------------------------+
```

Primary actions:

- create/edit schedule.
- pause/resume.
- retry missed/failed run.
- open run in Console.

Focus path: filters -> schedule list -> detail -> controls/history.

Console handoff: live or failed run opens in Console with schedule provenance.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Schedules destination in a local-first agentic console. Show timing-focused schedules, triggers, next run, last run, active/paused/failed states, pause/resume/retry controls, and run history. Include follow/open-in-Console for live runs and clear recovery for missed or failed schedules. Use dense terminal tables and status badges. Avoid calendar-app ornamentation or a web cron dashboard look.
```

QA checks:

- User can distinguish schedule timing from workflow procedure.
- Pause/resume/retry are reachable and target the selected schedule.
- Failed schedule states explain cause and recovery.

### Workflows

User goal: define what procedure runs and inspect its outputs.

Screen role: procedure builder, dry-run, approvals, launch, and run detail.

Binding regions:

- workflow list.
- builder/step detail.
- inputs/outputs/approval points.
- run/recovery inspector.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Workflows | Procedures, steps, inputs, outputs, approvals | Ready | Workspace  |
+--------------------------------------------------------------------------------+
| Modes: Browse Build Runs Templates | Filter: Draft Active Failed               |
+----------------------+--------------------------------------+------------------+
| Workflow List        | Builder / Run Detail                  | Run Inspector    |
| Research digest      | Step 1 sources -> Step 2 summarize   | Inputs ready     |
| Code audit           | Approvals and output targets         | Tools ready      |
+----------------------+--------------------------------------+------------------+
| Footer: N new | D dry run | C launch/follow in Console | A approvals       |
+--------------------------------------------------------------------------------+
```

Primary actions:

- create/edit workflow.
- dry-run.
- attach sources/persona/skills/tools.
- launch/follow in Console.

Focus path: mode bar -> workflow list -> builder/run detail -> run inspector.

Console handoff: launch and follow actions open Console with workflow step/run context.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Workflows destination. Show reusable procedures with steps, inputs, outputs, approvals, dry-run status, tool/persona/skill attachments, and launch/follow-in-Console. Include a workflow list, builder canvas expressed as terminal rows, selected step inspector, and run recovery. Avoid node-graph web UI, canvas drag-and-drop assumptions, or decorative process diagrams that cannot translate to Textual.
```

QA checks:

- User can tell "what runs" here and "when it runs" belongs in Schedules.
- Dry-run/readiness is visible before launch.
- Approval points and output targets are visible.

### MCP

User goal: manage external tools/resources and their readiness.

Screen role: MCP servers, tools, resources, auth, permissions, and audit.

Binding regions:

- server/resource/tool list.
- selected tool/server detail.
- permissions/auth/readiness inspector.
- test/recover controls.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| MCP | Tool/resource protocol management | Ready | Local/Server                 |
+--------------------------------------------------------------------------------+
| Modes: Servers Tools Resources Permissions Audit | Filter: Blocked Ready       |
+----------------------+--------------------------------------+------------------+
| Server/Tool List     | Tool Or Server Detail                 | Readiness        |
| filesystem           | Schema, auth, risk, last test         | Auth: ok         |
| browser              | Resource inventory                    | Permission: ask  |
+----------------------+--------------------------------------+------------------+
| Footer: T test | P permissions | C use/follow in Console | L audit           |
+--------------------------------------------------------------------------------+
```

Primary actions:

- test tool/server.
- manage permission.
- inspect audit.
- use/follow in Console when relevant.

Focus path: modes -> server/tool list -> detail -> readiness/permission controls.

Console handoff: tool use happens from Console or is followed in Console with visible approval state.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the MCP destination. Show MCP servers, tools, resources, permissions, auth status, risk/approval rules, test actions, and audit log. Include local/server authority and readiness badges. Make it clear MCP is protocol/tool management, not global app settings. Avoid generic settings-page visuals or hiding risk behind icons.
```

QA checks:

- `tools_settings` resolves as MCP, not global Settings.
- Tool readiness and permission status are visible before use.
- Blocked tools show owner, impact, and recovery.

### ACP

User goal: manage agent protocol runtimes, agents, sessions, and follow live collaboration.

Screen role: ACP runtime/session readiness separate from MCP.

Binding regions:

- runtime/agent/session list.
- selected session/detail.
- compatibility/readiness inspector.
- launch/resume/follow controls.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| ACP | Agent protocol sessions and runtimes | Runtime needed | Local/Remote     |
+--------------------------------------------------------------------------------+
| Modes: Agents Sessions Runtimes Compatibility | Filter: Ready Blocked         |
+----------------------+--------------------------------------+------------------+
| Agent/Session List   | Session Detail / Runtime Setup        | Compatibility    |
| No runtime configured| Diffs, terminal, files when ready     | Missing runtime  |
|                      | Recovery setup steps                  | Configure        |
+----------------------+--------------------------------------+------------------+
| Footer: N agent | R resume | C follow in Console | S setup runtime           |
+--------------------------------------------------------------------------------+
```

Primary actions:

- configure runtime.
- discover/install agent.
- launch/resume/follow session.

Focus path: modes -> agent/session list -> detail/setup -> compatibility inspector.

Console handoff: ACP live session opens or follows in Console; ACP does not become a second Console.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the ACP destination, separate from MCP. Show agents, sessions, runtimes, compatibility, diffs, terminals, launch/resume/follow-in-Console, and an honest runtime-unconfigured recovery state. Use terminal-native panes and explicit readiness labels. Avoid making ACP look like MCP tools or a second chat surface.
```

QA checks:

- Runtime-unconfigured state is honest and recoverable.
- ACP and MCP purposes remain visibly distinct.
- Launch/follow enters Console for live work.

### Skills

User goal: discover, validate, inspect, edit, and attach Agent Skills-compatible capability packs.

Screen role: Agent Skills library and validation surface.

Binding regions:

- installed/discovered skill list.
- `SKILL.md` and directory detail.
- validation/compatibility inspector.
- attach/import/export/edit controls.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Skills | Agent Skills packs, validation, attachments | Ready | Local/Server    |
+--------------------------------------------------------------------------------+
| Modes: Installed Discover Import Validate Attach | Filter: Valid Broken        |
+----------------------+--------------------------------------+------------------+
| Skill List           | SKILL.md / Files / Instructions       | Validation       |
| pdf-processing       | name, description, allowed-tools      | Frontmatter ok   |
| code-review          | scripts, references, assets           | Attach targets   |
+----------------------+--------------------------------------+------------------+
| Footer: I import | V validate | E edit | C attach to Console | X export       |
+--------------------------------------------------------------------------------+
```

Primary actions:

- import/discover skill.
- validate skill.
- inspect/edit `SKILL.md`.
- attach to Console, Personas, Workflows, or ACP.

Focus path: modes -> skill list -> skill detail -> validation/attachment inspector.

Console handoff: selected skill can be attached to Console as a capability with visible compatibility and allowed-tools state.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Skills destination. Show Agent Skills-compatible packs with SKILL.md metadata, scripts, references, assets, validation status, compatibility, allowed tools, import/export/edit actions, and attach-to-Console/Personas/Workflows/ACP controls. Use dense terminal panels, readable validation badges, and file-tree detail. Avoid marketplace-store visuals or generic plugin cards.
```

QA checks:

- User can identify valid vs invalid skills before attachment.
- `SKILL.md` frontmatter and bundled directories are visible.
- Attachment targets are explicit and recoverable when incompatible.

### Settings

User goal: configure global app preferences, providers, privacy, storage, appearance, and diagnostics.

Screen role: global configuration only.

Binding regions:

- settings category list.
- selected setting form/detail.
- diagnostics/impact inspector.
- save/revert/test controls.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ACP Skills |
+--------------------------------------------------------------------------------+
| Settings | Global preferences, providers, privacy, diagnostics | Ready | Local  |
+--------------------------------------------------------------------------------+
| Categories: Providers Models Storage Privacy Appearance Diagnostics            |
+----------------------+--------------------------------------+------------------+
| Category List        | Setting Form / Diagnostic Detail      | Impact           |
| Providers            | API keys, model defaults, app config  | Affects Console  |
| Storage              | Paths, backups, encryption            | Test / Revert    |
+----------------------+--------------------------------------+------------------+
| Footer: S save | R revert | T test | D diagnostics                         |
+--------------------------------------------------------------------------------+
```

Primary actions:

- save/revert setting.
- test provider/runtime.
- open diagnostics.

Focus path: categories -> setting form -> validation/impact inspector -> save/revert.

Console handoff: Settings does not stage work into Console except provider/runtime diagnostics that explain Console readiness.

Image reference brief:

```text
Create a Textual-native terminal UI concept for the Settings destination in a local-first agentic console. Show global preferences only: providers, models, storage, privacy, appearance, diagnostics, config reload, and test controls. Include a category list, setting form, impact/diagnostics inspector, save/revert/test actions, and readable validation. Avoid absorbing MCP, ACP, Skills, Personas, Schedules, or Workflows setup into Settings.
```

QA checks:

- Settings does not duplicate task-specific configuration owned by MCP, ACP, Skills, Personas, Schedules, or Workflows.
- Validation errors are local to the setting and include recovery.
- Provider readiness explains downstream Console impact.

## Major Subflow Contracts

### Library: Search/RAG

```text
+--------------------------------------------------------------------------------+
| Library > Search/RAG | Query sources deliberately | Index: ready | Local       |
+--------------------------------------------------------------------------------+
| Source Set: Workspace / Collection / All | Mode: Search / RAG Answer           |
+----------------------+--------------------------------------+------------------+
| Source Scope         | Results / Evidence / Draft Answer     | Retrieval Detail |
| Workspaces           | ranked chunks, citations, confidence  | Index status     |
| Collections          |                                      | Ask in Console   |
+----------------------+--------------------------------------+------------------+
```

Required behavior: supports deliberate Library retrieval and can stage results into Console. It must show index/model/source blockers before query execution.

### Library: Import/Export

```text
+--------------------------------------------------------------------------------+
| Library > Import/Export | Source movement | Ready | Local/Server               |
+--------------------------------------------------------------------------------+
| Modes: Import Export Jobs History                                             |
+----------------------+--------------------------------------+------------------+
| Source Types         | Import/Export Form Or Job Progress    | Validation       |
| Notes Media Web      | selected files, mapping, progress     | Conflicts        |
| Conversations        |                                      | Recovery        |
+----------------------+--------------------------------------+------------------+
```

Required behavior: Library import/export is for source material. Artifact export is separate.

### Library: Workspaces

```text
+--------------------------------------------------------------------------------+
| Library > Workspaces | Cross-source user context | Ready | Workspace           |
+--------------------------------------------------------------------------------+
| Workspace List       | Workspace Detail: sources, chats, notes, artifacts       |
+----------------------+--------------------------------------+------------------+
| Workspaces           | Scope contents and recent activity    | Scope Inspector  |
| Current workspace    | Study entry, RAG source set, outputs  | Authority/conflict|
+----------------------+--------------------------------------+------------------+
```

Required behavior: Workspaces define broad user context and scope. They do not replace Collections.

### Library: Study Dashboard

```text
+--------------------------------------------------------------------------------+
| Library > Study Dashboard | Study from source material | Ready | Scope         |
+--------------------------------------------------------------------------------+
| Study Sections      | Dashboard: decks, quizzes, progress, source seeds         |
+----------------------+--------------------------------------+------------------+
| Dashboard           | Recent study activity and generation  | Scope Inspector  |
| Flashcards          | Start review or generate from source  | Source authority |
| Quizzes             | Continue attempt or generate quiz     | Back to Library  |
+----------------------+--------------------------------------+------------------+
```

Required behavior: the existing `study`, Study Dashboard, Flashcards, and Quizzes surfaces are Library-owned. Library can route to Study Dashboard, Flashcards, or Quizzes with the requested section preserved, and Study must expose the active source/workspace scope instead of becoming an unrelated top-level destination.

### Library: Flashcards

```text
+--------------------------------------------------------------------------------+
| Library > Flashcards | Study cards from sources or outputs | Ready | Scope     |
+--------------------------------------------------------------------------------+
| Decks / Source Seeds | Card Review / Generate / Edit         | Study Inspector  |
+----------------------+--------------------------------------+------------------+
```

Required behavior: flashcards are reachable from Library and can be generated from selected source material or Console outputs. Scope must be visible.

### Library: Quizzes

```text
+--------------------------------------------------------------------------------+
| Library > Quizzes | Study questions from sources or outputs | Ready | Scope    |
+--------------------------------------------------------------------------------+
| Quiz Sets / Seeds    | Quiz Attempt / Generate / Review      | Score/Source     |
+----------------------+--------------------------------------+------------------+
```

Required behavior: quizzes are reachable from Library and can be generated from selected sources or Console outputs. Source provenance must remain visible.

### Library: Notes, Media, Conversations, Source Detail

```text
+--------------------------------------------------------------------------------+
| Library > Source Detail | Inspect source material | Ready | Local/Server       |
+--------------------------------------------------------------------------------+
| Source List          | Detail / Preview / Metadata           | Actions          |
| Notes Media Chats    | chunks, transcript, note body         | RAG Console Study |
+----------------------+--------------------------------------+------------------+
```

Required behavior: source detail exposes use-in-Console, Search/RAG, metadata, and import/export recovery where relevant.

### Artifacts: Chatbooks

```text
+--------------------------------------------------------------------------------+
| Artifacts > Chatbooks | Saved agent sessions and packages | Ready | Local      |
+--------------------------------------------------------------------------------+
| Chatbook List        | Chatbook Preview / Transcript         | Provenance       |
| Recent Saved Runs    | source summary, response, metadata    | Reopen Console   |
+----------------------+--------------------------------------+------------------+
```

Required behavior: Chatbooks are artifact outputs and can reopen/resume into Console.

### Artifacts: Exports And Reuse

```text
+--------------------------------------------------------------------------------+
| Artifacts > Export | Package reusable output | Ready | Local/Server           |
+--------------------------------------------------------------------------------+
| Artifact Selection  | Export Format / Bundle Options         | Validation       |
+----------------------+--------------------------------------+------------------+
```

Required behavior: export blockers identify missing files, incompatible format, permission, or server/local authority issues.

### Personas: Detail/Edit/Import/Export

```text
+--------------------------------------------------------------------------------+
| Personas > Detail | Edit behavior and identity | Ready | Local/Server          |
+--------------------------------------------------------------------------------+
| Persona List       | Profile Editor / Preview                 | Attach/Validate  |
+----------------------+--------------------------------------+------------------+
```

Required behavior: edits expose behavior impact and attachment compatibility.

### W+C: Watchlists

```text
+--------------------------------------------------------------------------------+
| W+C > Watchlists | Monitored sources and runs | Ready | Local/Server          |
+--------------------------------------------------------------------------------+
| Watchlist List    | Sources Jobs Runs Items Outputs          | Retry/Alerts     |
+----------------------+--------------------------------------+------------------+
```

Required behavior: run status, retry/backoff, alerts, and follow-in-Console are visible.

### W+C: Collections

```text
+--------------------------------------------------------------------------------+
| W+C > Collections | Curated content sets | Ready | Local/Server             |
+--------------------------------------------------------------------------------+
| Collection List   | Items Highlights Saved Searches Feeds    | RAG/Export       |
+----------------------+--------------------------------------+------------------+
```

Required behavior: Collections feed RAG, Library, schedules, workflows, and monitoring without duplicating Workspaces.

### Schedules: Detail And History

```text
+--------------------------------------------------------------------------------+
| Schedules > Detail | Timing and run health | Ready | Local/Server            |
+--------------------------------------------------------------------------------+
| Schedule List      | Trigger Detail / History / Missed Runs  | Pause Retry      |
+----------------------+--------------------------------------+------------------+
```

Required behavior: schedule screens explain when work runs and expose failed/missed-run recovery.

### Workflows: Builder And Run Detail

```text
+--------------------------------------------------------------------------------+
| Workflows > Builder | Procedure definition | Ready | Workspace               |
+--------------------------------------------------------------------------------+
| Workflow List       | Steps Inputs Outputs Approvals Dry Run | Launch/Recover   |
+----------------------+--------------------------------------+------------------+
```

Required behavior: builder uses terminal rows/lists, not a web-only node canvas assumption.

### MCP: Tools/Resources/Readiness

```text
+--------------------------------------------------------------------------------+
| MCP > Tools | Tool readiness and permissions | Ready | Local/Server           |
+--------------------------------------------------------------------------------+
| Servers Tools       | Schema Auth Risk Test Result            | Permissions      |
+----------------------+--------------------------------------+------------------+
```

Required behavior: test, auth, permission, risk, and audit are visible before tool use.

### ACP: Agents/Sessions/Runtime

```text
+--------------------------------------------------------------------------------+
| ACP > Sessions | Agent protocol runtime | Blocked | Runtime                 |
+--------------------------------------------------------------------------------+
| Agents Sessions     | Runtime Setup / Session Detail          | Compatibility    |
+----------------------+--------------------------------------+------------------+
```

Required behavior: runtime missing state is honest and recoverable; sessions follow into Console.

### Skills: Validation/Edit/Attach

```text
+--------------------------------------------------------------------------------+
| Skills > Validation | Agent Skills contract | Ready | Local/Server           |
+--------------------------------------------------------------------------------+
| Skill List          | SKILL.md Files Validation               | Attach Targets   |
+----------------------+--------------------------------------+------------------+
```

Required behavior: validation exposes `name`, `description`, compatibility, allowed tools, scripts, references, assets, and attach targets.

## Image Generation Governance

When generating destination inspiration images:

- Generate one image per top-level destination unless the user requests fewer.
- Include the destination's correct vocabulary in the prompt.
- Include explicit "Textual-native terminal UI" wording.
- Include "avoid web dashboard/card/browser chrome" wording.
- Preserve Home and Console in the global nav.
- Do not include impossible gestures or controls as requirements.
- Store generated references separately from the binding spec if committed.
- Captions must state: "Non-binding inspiration; text and ASCII contract are authoritative."

## Phase Integration Rules

After Phase 3.0 is approved:

- Phase 3 Knowledge/Study tasks must reference the Library contract and relevant subflow contract.
- Phase 4 Agent Configuration tasks must reference Personas, Skills, MCP, ACP, Schedules, and Workflows contracts.
- Phase 5 parity tasks must update the affected destination contract if server parity changes visible ownership.
- Phase 6 release hardening must verify that implemented screens still match approved contracts or documented deviations.

## QA Gate

Phase 3.0 is done when:

- all top-level destinations have approved layout contracts.
- all major subflows listed in this spec have approved owner placement.
- ASCII wireframes exist for every top-level destination and major subflow.
- one image-generation brief exists for every top-level destination.
- route-owner map resolves legacy routes to owning destinations.
- terminal-size, focus, state, source-authority, and Console-handoff rules are defined.
- product-maturity roadmap/backlog records Phase 3.0 as a prerequisite before further Phase 3 visual rewrites.

Later implementation gates are done only when:

- affected screens match the relevant contract or document a reviewed deviation.
- compact/default/large terminal checks pass.
- keyboard and command-palette paths reach primary actions.
- empty/loading/error/blocked states are understandable and recoverable.
- Console handoffs preserve provenance, source authority, and recovery status.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Spec becomes too large to implement. | Keep Phase 3.0 as contracts only; implement by later PR-sized slices. |
| Images become pseudo-requirements. | Text and ASCII are authoritative; captions and prompts must say images are non-binding. |
| Layout work becomes visual-only polish. | Every appendix includes user goal, states, focus path, and QA checks. |
| Legacy routes duplicate destination layouts. | Route-owner map assigns each legacy route/subflow to one owner destination. |
| Study placement stays confusing. | Study remains Library-owned; Flashcards and Quizzes are visible Library subflows. |
| Workspaces and Collections blur. | Workspaces are global context; Collections are reusable source/content sets. |
| Phase 6 discovers structural problems too late. | Phase 3.0 becomes the prerequisite before deeper Phase 3+ visual work. |

## Open Implementation Questions

These are intentionally deferred to implementation planning:

- Whether destination contracts should be stored in one Markdown file or split into per-destination docs after approval.
- Whether generated image references should be committed as assets or kept as disposable design artifacts.
- Whether the first implementation slice should update Library first or create shared contract-checking tests first.
- Whether existing legacy direct screens should be wrapped first or redesigned opportunistically by workflow priority.
