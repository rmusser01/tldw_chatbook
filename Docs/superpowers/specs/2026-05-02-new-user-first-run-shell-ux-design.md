# Master Shell UX Design

Date: 2026-05-02
Last Updated: 2026-05-02
Status: User-approved design, pending spec review and implementation planning
Primary Repo: `tldw_chatbook`
Scope: Master shell UX, first-run dashboard, top-level information architecture, Console-centered live work, local parity destinations, and migration constraints
Primary Persona: Power user with beginner-safe orientation
Primary Interaction Bias: Keyboard-first, mouse-safe, compact TUI
Platform Constraint: Stay within the current Textual/TUI product model

## Summary

`tldw_chatbook` should present itself as a local-first agentic knowledge console, not as a flat collection of unrelated utility tabs.

The app already contains substantial capability: chat, notes, media, RAG/search, personas, chatbooks, study tools, MCP, server parity, watchlists, collections, schedules, workflows, and skill-like extension points. The UX problem is that these capabilities are not organized into a clear product model. New users lack a strong front door, while power users still pay too much re-orientation cost when moving between modules.

The approved direction is a master shell organized around:

- `Home` as the default first-run page and always-available dashboard.
- `Console` as the only live agent conversation/run surface.
- clear top-level destinations for source material, outputs, personas, monitoring, scheduling, workflows, protocols, skills, and settings.
- local parity with the corresponding `tldw_server2` domains where applicable.
- capability-aware status, recovery, and next-best actions before users hit dead ends.

This is not a cosmetic refresh. It is a product-structure and workflow-completion redesign.

## Problem Statement

The existing UI has been progressively remediated through many small PRs, but the larger product model remains under-specified.

Current user-facing risks:

- The app still exposes too many implementation-era surfaces at equal weight.
- `Chat` undersells the intended role of the main agentic control surface.
- Home/onboarding work is useful but too narrow if treated as only a first-run feature.
- RAG/search, imports, notes, media, conversations, and artifacts need clearer boundaries.
- MCP, ACP, and Skills are distinct concepts but can easily blur without explicit ownership rules.
- Watchlists and Collections should mirror server-side local parity domains rather than become a generic bucket.
- Schedules and Workflows can blur unless the app clearly separates "when it runs" from "what procedure runs."
- Live agent work can fragment unless every launch/follow path converges on Console.

For a new user, this creates uncertainty about where to start. For a power user, it creates avoidable friction and state ambiguity.

## Goals

- Make `Home` useful both as first-run orientation and as an ongoing dashboard.
- Rename the user-facing `Chat` concept to `Console`.
- Make `Console` the single live agent conversation/run surface.
- Preserve feature breadth while reducing first-run and cross-module cognitive load.
- Define a stable top-level IA that can absorb the concept-screen direction without copying it 1:1.
- Keep local/server/workspace source authority visible.
- Preserve power-user speed, keyboard shortcuts, command palette access, and dense TUI workflows.
- Keep every destination's role clear enough to guide implementation and test coverage.
- Align Watchlists and Collections with the local versions of their `tldw_server2` functionality.
- Treat Skills as Agent Skills spec-compatible capability packs with discovery/import and attachment points.

## Non-Goals

- Do not implement the concept images as literal 1:1 screen requirements.
- Do not rebuild every screen in one slice.
- Do not remove advanced features.
- Do not force users to complete onboarding before using the app.
- Do not convert the TUI into a web-style card dashboard.
- Do not collapse all configuration into Console.
- Do not duplicate live run surfaces across destination screens.
- Do not change stable internal route IDs unless a later implementation plan proves it is necessary.

## Design Principles

- One product shell, not a bundle of unrelated tabs.
- Home explains and controls the system at a glance.
- Console is where live agent work happens.
- Configuration surfaces launch or prepare work; they do not become alternate live-run surfaces.
- Recognition before recall.
- Core workflows first, expert depth one step away.
- Status before failure.
- Source authority must remain visible: local, server, workspace, remote-only, and dry-run states cannot be blended.
- Short labels are acceptable only when the destination purpose is taught by copy, tooltip, status line, or command palette.
- Keyboard speed and mouse safety are both required.

## Approved Top-Level Information Architecture

The top-level shell should expose these destinations:

- `Home`
- `Console`
- `Library`
- `Artifacts`
- `Personas`
- `Watchlists+Collections`
- `Schedules`
- `Workflows`
- `MCP`
- `ACP`
- `Skills`
- `Settings`

### Home

Home is the default landing page for new users and an always-available dashboard for experienced users.

Home owns:

- global readiness and status
- notifications across modules
- active work overview
- schedule and workflow health
- next-best actions
- pending approvals
- lightweight approve, pause, resume, retry, and open-details controls
- recent work and re-entry

Home should answer:

1. What is ready?
2. What needs attention?
3. What is currently running?
4. What should I do next?
5. Where do I resume?

Home must stay compact and actionable. It is not a decorative dashboard.

### Console

Console replaces `Chat` as the user-facing label. The current internal `chat` route and code paths may remain during migration.

Console is the primary live surface for:

- direct chat
- agentic programming and control
- RAG-backed answers
- staged source context
- plans and task progress
- tool calls and observations
- approvals
- MCP tool use
- ACP agent sessions
- diffs and shell/terminal interaction
- run logs and recovery
- artifact creation

Any destination can prepare or launch work, but live execution opens or follows in Console.

### Library

Library owns source material and knowledge access.

Library includes:

- Notes
- Media
- Conversations
- Import/Export
- Search/RAG
- source browsing and metadata

Search/RAG must exist in both Library and Console:

- Library Search/RAG supports deliberate retrieval and explicit RAG query workflows.
- Console RAG supports conversational answers that retrieve from Library/RAG as part of the agent response.

Both paths must expose sources, citations or evidence where available, and source authority.

### Artifacts

Artifacts owns generated, portable, or reusable outputs.

Artifacts includes:

- reports
- datasets
- workflow outputs
- exported/imported bundles
- drafts
- saved deliverables
- Chatbooks

Artifacts should not become a second Library. Its focus is outputs and reusable packages, not raw source browsing.

### Personas

Personas owns behavior and identity.

Personas includes:

- characters
- personas
- assistant profiles
- prompts
- dictionaries
- lore/world books
- persona-scoped defaults

Personas can be attached to Console sessions, Workflows, ACP agents, and Skills.

### Watchlists+Collections

Watchlists+Collections is one top-level destination with two primary internal tabs: `Watchlists` and `Collections`.

This destination represents local versions of the corresponding `tldw_server2` functionality.

Watchlists owns monitored-source workflows:

- sources
- groups
- tags
- filters
- jobs
- runs
- scraped items
- outputs
- templates
- alerts
- telemetry/status
- retry and backoff state

Collections owns curated reading/content workflows:

- reading/content items
- highlights
- saved searches
- archive state
- lightweight annotations
- note links
- output templates
- feeds/subscriptions where available
- import/export

Watchlists+Collections can feed Library, Schedules, Workflows, Artifacts, and Console, but it should not be reduced to any one of them.

### Schedules

Schedules owns when things run.

Schedules includes:

- time-based schedules
- event-based triggers
- next and last run state
- missed-run recovery
- paused state
- retry policy
- schedule health
- links to affected workflows, watchlists, sources, or Console runs

Schedules must not duplicate workflow-builder complexity.

### Workflows

Workflows owns what procedure runs.

Workflows includes:

- recipes
- builders
- inputs
- steps
- required tools
- required skills
- required personas
- dry-run preview
- outputs
- approvals
- versions

Starting or following a workflow run opens Console.

### MCP

MCP owns tool and server capability plumbing.

MCP includes:

- MCP servers
- connectors
- tools
- tool permissions
- auth
- availability
- risk and approval rules
- audit logs
- test actions
- recovery for blocked tools

MCP should answer: what can agents use, what is blocked, and how do I fix it?

### ACP

ACP owns Agent Client Protocol interoperability.

ACP includes:

- available and installed ACP agents
- local and remote agent runtimes
- coding-agent client sessions
- session resume
- diff review
- terminal/shell integration
- agent runtime status
- compatibility and installation state
- discovery and launch

ACP launches or resumes live work in Console. It does not become a second Console.

### Skills

Skills owns Agent Skills spec-compatible capability packs.

Skills includes:

- installed local skills
- discovery/import from external skill registries
- `SKILL.md` inspection
- metadata validation
- `scripts/`, `references/`, and `assets/` management
- compatibility and allowed-tool review
- attachment to Console sessions, Personas, ACP agents, and Workflows

The Skill model follows the Agent Skills structure:

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
├── assets/
└── ...
```

Skills are user-facing reusable instructions and capability packs, not low-level MCP tool configuration.

### Settings

Settings owns app-level preferences and global configuration.

Settings includes:

- appearance
- storage
- accounts
- global preferences
- default behavior
- non-task-specific configuration

Settings should not be the place a new user must discover to make AI work. Runtime readiness belongs in Home, Console, MCP, ACP, and Skills as appropriate.

## Route Compatibility And Migration Rules

The user-facing IA may change before internal route IDs do.

### Route Mapping

| New destination | Current or likely route mapping |
| --- | --- |
| `Home` | new screen |
| `Console` | current `chat` route |
| `Library` | wraps or groups `notes`, `media`, `search`, `ingest`, conversation browsing |
| `Artifacts` | wraps Chatbooks and output/bundle surfaces |
| `Personas` | character/persona/prompt/dictionary/lore portions of current `ccp` |
| `Watchlists+Collections` | local watchlists and collections parity surfaces |
| `Schedules` | schedule and trigger surfaces |
| `Workflows` | workflow definitions, builders, templates, dry-runs |
| `MCP` | MCP/tool configuration and audit surfaces |
| `ACP` | ACP agent/session/runtimes surface |
| `Skills` | Agent Skills management and discovery surface |
| `Settings` | current settings/customize/global preferences |

### Migration Constraints

- Keep stable internal route IDs where practical.
- Rename user-facing `Chat` to `Console` without breaking existing chat internals.
- Split or wrap legacy composite screens only when needed for the new destination role.
- Do not expose `ccp` as a user-facing label.
- Do not mount competing global navigation systems.
- Each top-level screen must have one primary shell nav.
- Destination wrappers are acceptable during migration if they provide clear labels, state, and subsection entry.
- Implementation must separate presentation relabeling from deeper screen decomposition.

## Home Design

Home is both first-run entry and ongoing control center.

### Required Sections

#### Status

Show compact readiness across:

- model/provider readiness
- server/local state
- MCP availability
- ACP agent availability
- RAG/index readiness
- sync/dry-run status
- storage health
- optional dependency gaps

#### Attention

Show items requiring action:

- notifications
- failed jobs
- blocked tools
- auth gaps
- stale schedules
- pending approvals
- missing dependencies
- import failures
- watchlist/collection alerts

#### Active Work

Show running and resumable work:

- Console sessions
- ACP agent sessions
- workflow runs
- watchlist runs
- scheduled jobs
- resumable tasks

Each item should provide `Open in Console` when live inspection is needed.

#### Next Best Action

Show exactly one dominant recommendation at a time.

Priority order:

1. Fix critical blockers.
2. Approve pending work.
3. Resume active work.
4. Set up required model, agent, or tools.
5. Import or search Library content.
6. Start a Console task.
7. Explore Personas, Workflows, Skills, or Watchlists+Collections.

The recommendation must be derived from deterministic state, not ad hoc UI checks.

#### Quick Controls

Allow lightweight action without leaving Home:

- approve
- reject
- pause
- resume
- retry
- open details
- open in Console

Complex inspection should move to Console or the owning destination.

#### Recent Work

Show compact re-entry:

- recent Console sessions
- recent Library items
- recent Artifacts
- recent Workflows
- recent Watchlists/Collections activity
- recently used Personas
- recently used Skills

### First-Run And Returning-User Behavior

New users should land on Home.

Existing users should not be forcibly interrupted if strong prior-use evidence exists:

- saved conversations
- notes or media
- personas
- artifacts or chatbooks
- configured providers or tools
- prior successful Console session
- stored last-active-screen state

For returning users, default reopen behavior should prefer:

1. explicit user preference
2. last active screen
3. Home

Home remains visible and useful even when it is not the reopen target.

## Console Design

Console is the only live agent conversation/run surface.

### Console Responsibilities

Console must support:

- normal chat
- RAG-backed questions
- staged source context
- source roles: context, evidence, editable target, output seed
- ACP session launch and resume
- MCP tool calls
- plans
- observations
- approvals
- diffs
- terminal/shell interactions
- run logs
- recovery guidance
- artifact creation and links

### Handoff Rules

- Handoffs from Library, Artifacts, Personas, Watchlists+Collections, Schedules, Workflows, MCP, ACP, and Skills should open or follow in Console when live work begins.
- Staged source/context must be visible before send/run.
- Nothing should auto-send without explicit user action.
- The user must be able to edit, remove, or change role before sending.
- Source authority and scope must remain visible.

### RAG Rules

RAG appears in two places:

- Library for explicit search and RAG query workflows.
- Console for conversational RAG-backed answering.

Both flows must show:

- source set
- retrieval status
- citations or evidence when available
- local/server/workspace authority
- recovery when the index, model, or dependency is unavailable

## Destination Screen Contract

Every top-level destination should use the same compact page grammar:

1. title
2. one-line purpose statement
3. scope/status line
4. one primary action
5. main work area
6. optional inspector or status panel

This contract reduces relearning cost and makes UI audits repeatable.

### Library Contract

Library must make source type and action clear.

Required sections or subsections:

- Notes
- Media
- Conversations
- Import/Export
- Search/RAG

Empty states should route users forward:

- no notes -> create/import note
- no media -> import media
- no searchable content -> import or add source
- no conversations -> start Console

### Artifacts Contract

Artifacts must distinguish outputs from sources.

Required capabilities:

- browse generated outputs
- browse Chatbooks
- inspect reports/datasets/bundles
- export/import bundles
- reuse artifact in Console
- attach artifact to Workflow

### Personas Contract

Personas must explain behavior shaping in plain language.

Required capabilities:

- create/import persona or character
- inspect behavior profile
- attach persona to Console
- attach persona to Workflow
- attach persona to ACP agent
- recommend compatible Skills

### Watchlists+Collections Contract

Watchlists+Collections must expose two internal tabs:

- `Watchlists`
- `Collections`

Watchlists sections:

- Overview
- Sources
- Groups/Tags
- Jobs
- Runs
- Items
- Outputs
- Templates
- Settings/Status

Collections sections:

- Reading/content items
- Highlights
- Saved searches
- Feeds/subscriptions
- Archive state
- Note links
- Templates
- Import/Export

The screen must preserve the distinction between monitored streams and curated content.

### Schedules Contract

Schedules owns temporal control.

Required capabilities:

- create/edit schedule
- show next run
- show last run
- pause/resume
- retry missed or failed run
- inspect schedule health
- open live run in Console

### Workflows Contract

Workflows owns procedure definition.

Required capabilities:

- create/edit workflow
- define inputs
- define steps
- attach Persona
- attach Skills
- attach MCP tools
- preview with dry run
- define outputs
- define approval points
- launch/follow in Console

### MCP Contract

MCP owns tool/server readiness and permissions.

Required capabilities:

- list servers
- list tools
- show connection/auth status
- manage permissions
- show risk and approval rules
- test tool
- inspect audit log
- recover blocked tools

### ACP Contract

ACP owns agent protocol interoperability.

Required capabilities:

- discover/install ACP agents
- inspect compatibility
- configure local/remote runtime
- launch session
- resume session
- inspect diffs
- inspect terminal/shell integration
- open or follow live session in Console

### Skills Contract

Skills owns Agent Skills.

Required capabilities:

- discover/import skills
- list installed skills
- inspect `SKILL.md`
- validate frontmatter and directory structure
- inspect scripts/references/assets
- review compatibility and allowed tools
- attach skill to Console, Personas, ACP agents, and Workflows

### Settings Contract

Settings owns global preferences only.

Settings should not absorb task-specific setup that belongs in MCP, ACP, Skills, Console, or Home readiness.

## Cross-Surface Flow Rules

### Live Work

Live work happens in Console.

Destinations can:

- configure
- select
- prepare
- preview
- schedule
- launch
- inspect historical output

But the active run/conversation surface is Console.

### Source Roles

Source and artifact handoffs should use explicit roles:

- `context`: background information
- `evidence`: facts and citations
- `editable target`: content the agent may modify after approval
- `output seed`: input for generated output

### Approvals

Approvals should be visible in:

- Home for lightweight action
- Console for live context and full decision
- owning destination for configuration/history

### Error Recovery

Errors must identify owner and recovery path:

- model/provider
- Library/RAG
- MCP
- ACP
- Skills
- schedule
- workflow
- permission
- dependency
- server/auth
- storage

Avoid generic errors when a concrete destination can fix the issue.

### Source Authority

Local, server, workspace, remote-only, and dry-run state must remain visible.

Do not imply:

- automatic write sync
- queued mutation replay
- completed local mirroring
- local CRUD for remote-only domains

unless the backend/local contract explicitly supports it.

## Nielsen Norman Heuristic Alignment

### Visibility Of System Status

Home, Console, and each destination must show readiness, active state, blockers, and source authority close to the task.

### Match Between System And Real World

Labels should match user intent:

- `Console` for active agent work
- `Library` for source material
- `Artifacts` for outputs
- `Schedules` for when runs happen
- `Workflows` for what procedure runs
- `MCP` for tool/server plumbing
- `ACP` for agent protocol interoperability
- `Skills` for reusable capability packs

### User Control And Freedom

Users can skip Home, edit staged context, cancel/pause runs, reject approvals, and recover from failed schedules or agents.

### Consistency And Standards

Every destination uses the same page frame, status model, and Console handoff pattern.

### Error Prevention

Unavailable capabilities should be visible before action: missing model, missing index, blocked MCP tool, unavailable ACP runtime, invalid schedule, missing skill dependency.

### Recognition Rather Than Recall

Destinations expose purpose lines, primary actions, status lines, tooltips, and command palette labels.

### Flexibility And Efficiency Of Use

Keyboard shortcuts, command palette, and dense workflows remain available, but critical actions also remain discoverable.

### Aesthetic And Minimalist Design

Compact TUI density is acceptable; competing dominant regions and duplicated navigation are not.

### Help Users Recognize, Diagnose, And Recover From Errors

Recovery copy should name the failing owner and provide the next action.

### Help And Documentation

Home and destination empty states should teach enough to complete the next action without opening external docs.

## Implementation Phases

### Phase 1: Spec And Route Contract

- Treat this file as the authoritative master shell UX spec.
- Define final user-facing labels.
- Define route compatibility.
- Add tests for legacy route IDs if not already covered.
- Do not change behavior beyond documentation alignment.

### Phase 2: Navigation Model

- Introduce the agreed top-level labels.
- Preserve internal route IDs where practical.
- Prevent duplicate global nav.
- Keep command palette labels aligned with shell labels.

### Phase 3: Home Dashboard

- Build Home as first-run and always-available dashboard.
- Add readiness, notifications, next-best action, active work, controls, and recent work.
- Preserve returning-user last-screen behavior unless Home is explicitly selected or no meaningful prior use exists.

### Phase 4: Console Reframing

- Rename user-facing Chat to Console.
- Preserve Chat internals during migration.
- Make Console the only live-run surface.
- Add staged sources, RAG mode, approvals, ACP session follow, MCP tool calls, diffs, run status, and artifact links incrementally.

### Phase 5: Destination Containers

- Create or refactor top-level shells for Library, Artifacts, Personas, Watchlists+Collections, Schedules, Workflows, MCP, ACP, Skills, and Settings.
- Wrapping existing screens is acceptable if the new destination contract is visible.

### Phase 6: Cross-Surface Flows

- Wire source roles.
- Wire RAG handoffs.
- Wire artifact reuse.
- Wire workflow launch/follow.
- Wire schedule follow/open-in-Console.
- Wire ACP launch/resume.
- Wire skill attachment.
- Wire Home approval controls.

### Phase 7: Audit Replay

- Run first-time user and power-user workflows against the new shell.
- Validate against Nielsen Norman heuristics.
- Capture screenshots or ASCII mockups only where they clarify a decision.

## Risks And Mitigations

### Risk: Home becomes decorative

Mitigation:

- one dominant next-best action
- compact status
- real notifications
- active work controls
- direct open-in-Console paths

### Risk: Console becomes overloaded

Mitigation:

- Console owns live work only
- configuration stays in MCP, ACP, Skills, Personas, Workflows, Schedules, and Settings
- inspector/status panes should be optional and scoped

### Risk: Library becomes too broad

Mitigation:

- explicit subsections
- clear distinction between source material and outputs
- Search/RAG has deliberate Library mode and conversational Console mode

### Risk: Artifacts overlaps Library

Mitigation:

- Library owns sources
- Artifacts owns generated/portable outputs and Chatbooks

### Risk: Schedules and Workflows blur

Mitigation:

- Schedules owns when
- Workflows owns what
- Console owns live run

### Risk: MCP, ACP, and Skills confuse users

Mitigation:

- MCP: tools and servers
- ACP: agent protocol/session interoperability
- Skills: reusable instruction/capability packs

### Risk: Local parity drifts from server domains

Mitigation:

- Watchlists+Collections spec references server domain concepts
- implementation should inspect `tldw_server2` schemas before building local parity

### Risk: Existing users lose muscle memory

Mitigation:

- preserve route IDs
- preserve command palette compatibility
- introduce labels with transitional copy
- keep shortcuts stable
- avoid forcing Home for existing users with prior activity

## Success Metrics

The redesign should improve:

- time to first useful action
- time to first successful Console task
- time to first successful import or Library search
- time to recover from missing model/tool/agent setup
- number of navigation hops before first successful task
- percent of live runs followed in Console rather than scattered surfaces
- blank-search rate on first run
- successful schedule/workflow recovery rate
- qualitative ability of a new user to explain the product model within the first minute

## Measurement Plan

Use lightweight local counters or structured logs where full telemetry is not available.

Suggested events:

- app launch and resolved landing destination
- Home next-best action shown
- Home next-best action completed
- first successful Console send/run
- first successful import
- first successful Library RAG query
- Console RAG answer with sources
- MCP blocked tool recovery
- ACP session launched/resumed
- Skill imported/validated/attached
- schedule paused/resumed/retried
- workflow dry-run and launch
- watchlist run recovery
- artifact created/reused

Measurement must not block the shell migration unless no other verification is possible.

## Acceptance Tests For The Design

- A new user lands on Home and can identify one sensible next action without opening Settings.
- Home shows global status, notifications, active work, and lightweight controls.
- A returning user with prior work can resume from last active screen or Home without forced onboarding.
- User-facing `Chat` is reframed as `Console` while preserving internal route compatibility.
- Live agent work from Workflows, Schedules, ACP, Personas, Library, Skills, and Watchlists+Collections opens or follows in Console.
- Library includes Search/RAG and Import/Export, while Console can also answer using RAG.
- Artifacts is separate from Library and contains Chatbooks plus generated/portable outputs.
- Watchlists+Collections is one destination with `Watchlists` and `Collections` internal tabs.
- Schedules and Workflows remain separate top-level destinations with clear ownership.
- MCP, ACP, and Skills have distinct, testable boundaries.
- Source authority and runtime capability states remain visible.
- Power-user shortcuts remain available.

## Final Recommendation

Proceed with the master shell redesign using this IA:

- `Home`
- `Console`
- `Library`
- `Artifacts`
- `Personas`
- `Watchlists+Collections`
- `Schedules`
- `Workflows`
- `MCP`
- `ACP`
- `Skills`
- `Settings`

This gives `tldw_chatbook` a coherent front door, a single live agent surface, and a scalable product model for local parity with the broader `tldw_server2` ecosystem.
