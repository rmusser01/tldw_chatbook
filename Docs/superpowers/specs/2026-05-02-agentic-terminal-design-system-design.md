# Agentic Terminal Design System

Date: 2026-05-02
Status: User-approved design, pending implementation planning
Primary Repo: `tldw_chatbook`
Scope: Design system for the master shell UX, default visual identity, reusable UI grammar, Textual/TCSS mapping, and validation rules
Product Authority: `Docs/Design/master-shell-route-inventory.md`
Visual References: `Docs/Design/New_UI/*.png`

## Summary

`tldw_chatbook` should adopt a canonical design system for a local-first agentic knowledge console. The New_UI concept screenshots define the default visual direction, but they are not literal screen requirements. The master shell UX spec defines the product behavior, destination model, and cross-surface flow rules that this design system must support.

The system is two-layered:

1. Product/UI grammar: principles, layout rules, status language, density, component behavior, and validation expectations.
2. Textual/TCSS mapping: semantic tokens, reusable classes, state classes, component hooks, and migration constraints that fit the existing Textual app.

This is not a theme-only refresh. It is the visual and interaction grammar for the approved master shell: `Home`, `Console`, `Library`, `Artifacts`, `Personas`, `Watchlists+Collections`, `Schedules`, `Workflows`, `MCP`, `ACP`, `Skills`, and `Settings`.

## Source Material

### Product Source Of Truth

The design system follows the master shell UX design:

- Home is the first-run and returning-user dashboard.
- Console is the only live agent conversation/run surface.
- Destination screens configure, prepare, browse, inspect, schedule, or launch work; live work opens or follows in Console.
- Source authority must remain visible: local, server, workspace, remote-only, and dry-run states cannot be blended.
- Route IDs should remain stable where practical while user-facing labels evolve.

### Visual Direction

The concept images in `Docs/Design/New_UI/` provide the default identity:

- dark terminal canvas
- thin bordered panels
- cyan structure, focus, and section labeling
- green ready/success states
- amber warning, review, and approval states
- red blocked, failed, denied, or destructive states
- blue active navigation/selection emphasis
- dense mono typography
- persistent shortcut/status bar
- table-heavy, inspector-heavy expert workflows

They are reference compositions. They should not be implemented 1:1 unless a later screen-specific plan proves that a composition is the right answer.

## Goals

- Make the app feel like one coherent agentic terminal workbench.
- Provide a default visual identity for the approved master shell.
- Preserve compact power-user workflows while supporting comfortable beginner-readable density.
- Standardize status, source authority, recovery, approvals, staged context, and shortcuts.
- Make UI work implementable through Textual and modular TCSS rather than ad hoc inline styles.
- Reduce local screen-by-screen invention of panels, forms, badges, tables, inspectors, and recovery states.
- Provide validation criteria that map to Nielsen Norman usability heuristics and focused Textual tests.

## Non-Goals

- Do not redefine the approved top-level IA from the master shell UX spec.
- Do not implement the New_UI screenshots as literal 1:1 targets.
- Do not create a separate optional theme as the primary design direction.
- Do not remove advanced features or reduce power-user density.
- Do not force onboarding before use.
- Do not introduce a second global navigation system.
- Do not hide critical status through color-only styling.
- Do not start implementation from this document without a separate implementation plan.

## Foundation

The design system should make `tldw_chatbook` feel like a local-first agentic knowledge console: compact, terminal-native, keyboard-first, mouse-safe, and status-forward.

### Binding Principles

- Home and Console drive the system. Home needs dashboard/readiness components; Console needs live-agent, staged-source, tool-call, approval, recovery, and artifact-link components.
- Status appears before failure. Components expose ready, blocked, pending approval, unavailable, unsaved, running, paused, failed, and recovered states before users hit dead ends.
- Source authority is a first-class visual language. Local, server, workspace, remote-only, and dry-run states need consistent badges, labels, and recovery treatments.
- Dense does not mean obscure. Compact mode mirrors the New_UI density; comfortable mode preserves semantics with more spacing and clearer labels.
- Components are binding; templates are examples. Tokens and component behavior are enforceable. Screen templates demonstrate composition without making screenshots literal targets.
- The design system supports the approved shell labels and flows. IA decisions remain governed by the master shell UX spec.

## Semantic Tokens

The token system should be semantic first. Components should request intent tokens such as `status.warning` or `source-role.evidence`, not raw colors such as yellow or cyan.

### Surface Tokens

- `surface.canvas`: app background.
- `surface.panel`: standard panel background.
- `surface.panel-raised`: emphasized or selected panel.
- `surface.field`: input and editable field background.
- `surface.selection`: active row or selected object.
- `surface.overlay`: modal, popover, or floating surface.
- `surface.divider`: grid lines, panel borders, and separators.

### Text Tokens

- `text.primary`: normal readable text.
- `text.secondary`: supporting text.
- `text.muted`: metadata, helper text, low-emphasis labels.
- `text.disabled`: disabled labels and unavailable values.
- `text.inverse`: text on strong fills.
- `text.code`: commands, paths, IDs, tool names, and logs.

### Action Tokens

- `action.primary`: dominant action.
- `action.secondary`: secondary action.
- `action.hover`: mouse-hover state.
- `action.focus`: keyboard focus and active structure.
- `action.disabled`: disabled control.
- `action.destructive`: delete, cancel run, revoke, deny, archive.

### Status Tokens

- `status.ready`: ready, online, healthy, synced.
- `status.running`: running, streaming, indexing, generating.
- `status.info`: neutral informational state.
- `status.warning`: caution, stale, missing optional dependency.
- `status.approval-required`: user review required before write or external action.
- `status.blocked`: blocked capability or denied permission.
- `status.error`: failed operation.
- `status.paused`: paused schedule, workflow, or run.
- `status.unsaved`: unsaved draft or pending config changes.
- `status.recovered`: retry or recovery succeeded.

### Source Authority Tokens

- `authority.local`: local storage/service is authoritative.
- `authority.server`: active server is authoritative.
- `authority.workspace`: active server plus workspace/resource scope is authoritative.
- `authority.remote-only`: server-owned capability has no local CRUD.
- `authority.dry-run`: read-only readiness or sync preview only.
- `authority.syncing`: sync or mirror check in progress.
- `authority.synced`: source is up to date.
- `authority.conflict`: conflicting state requires resolution.

### Source Role Tokens

- `source-role.context`: background information.
- `source-role.evidence`: facts and citations.
- `source-role.editable-target`: content the agent may modify after approval.
- `source-role.output-seed`: seed for generated output.

### Structure Tokens

- `structure.border`: standard panel border.
- `structure.border-strong`: active or emphasized panel border.
- `structure.grid-line`: table and matrix separators.
- `structure.focus-ring`: keyboard focus outline.
- `structure.active-row`: selected row.
- `structure.inactive-row`: non-selected row.

### Density Tokens

The system supports two density modes:

- `density.compact`: the default expert mode; tight rows, dense tables, short labels, visible shortcuts.
- `density.comfortable`: beginner-readable mode; larger padding, clearer labels, more whitespace, same component semantics.

Density tokens cover:

- panel padding
- row height
- field height
- panel gap
- footer height
- header height
- table cell padding
- inspector width
- shortcut bar spacing

### Default Visual Mapping

- Cyan maps to focus, active structure, primary links, section titles, and selected control outlines.
- Green maps to ready, success, allowed, online, synced, and healthy states.
- Amber maps to warning, approval required, unsaved, review needed, and stale states.
- Red maps to blocked, failed, denied, destructive, missing auth, missing API key, and unrecoverable error states.
- Blue maps to active navigation and strong active row selection.
- Gray maps to inactive controls, dividers, metadata, disabled copy, and unselected borders.
- Black and near-black map to canvas and panel backgrounds.

### Accessibility Constraints

- Do not encode status by color alone.
- Pair all important status color with labels, glyphs, or explicit text.
- Red/green distinctions must include words such as `Allowed`, `Denied`, `Ready`, `Failed`, or `Blocked`.
- Every token must have a high-contrast-compatible fallback.
- Comfortable density must preserve status, authority, recovery, and shortcut information.

## Component Rules

Components are defined by purpose, not by destination. New screens should compose these primitives instead of inventing local variants.

### App Frame

The app frame owns:

- global top navigation
- active destination indication
- content viewport
- bottom shortcut/status bar

The frame must not include destination-specific context fields that consume page space. Destination context belongs inside the destination page.

### Top Navigation

The top bar is global primary destination navigation.

Rules:

- It exposes the approved shell destinations.
- It is always available unless a modal or focused full-screen workflow has a documented reason to suppress it.
- It does not duplicate local section tabs.
- It does not display workspace/backend/readiness fields as its main job.
- It preserves keyboard and command-palette parity.

### Destination Header

Every destination page should begin with a local header that provides:

- destination title
- one-line purpose statement
- local scope/status line
- one primary action when applicable
- workspace/backend/source authority when relevant

This header replaces ad hoc page-level orientation copy.

### Panel

Panels are titled bordered regions with a clear purpose.

Supported panel slots:

- title
- subtitle or purpose
- status badge
- toolbar
- main content
- empty state
- recovery state

Panel rules:

- Every panel must have one understandable role.
- A panel cannot be a dumping ground for unrelated controls.
- Empty panels must provide a next action or recovery path.

### Inspector

Inspectors are local detail/recovery panels for the selected item or current operation.

Use inspectors for:

- selected item metadata
- permissions
- readiness
- source authority
- linked sources
- recovery actions
- related artifacts
- run status

Inspectors must not hide blockers that affect the primary action.

### Data Table And List

Tables and lists support dense expert scanning.

Required states:

- active row
- hover row where applicable
- keyboard focus
- empty state
- loading state
- selected count
- row-level status
- row-level actions where applicable

Rules:

- Keyboard selection and mouse click must both work.
- Status columns should use labels plus semantic color.
- Row actions should remain discoverable without forcing users into hidden context menus.

### Status Badge

Badges are compact semantic labels for:

- readiness
- source authority
- sync
- approval
- running
- blocked
- stale
- unsaved
- conflict

Rules:

- Badges must use words, not color alone.
- Badges should be short enough for dense tables.
- Tooltips or inspector detail should explain complex status.

### Recovery Callout

Recovery callouts identify owner, problem, and next action.

Required structure:

```text
Owner: problem
Impact or explanation
Primary recovery action
```

Examples:

- `MCP: web_search blocked -> Open MCP permissions`
- `Provider: API key missing -> Open Settings > Providers`
- `RAG: index unavailable -> Open Library indexing`
- `ACP: runtime missing -> Open ACP setup`

Rules:

- Avoid generic errors when a concrete destination owns recovery.
- Recovery copy should be actionable and specific.
- Critical blockers should appear near the affected action.

### Source Role Chip

Source role chips label how a source or artifact will be used:

- `context`
- `evidence`
- `editable target`
- `output seed`

Rules:

- Role chips must appear in Console staged context, Library staging, Workflows, Artifacts reuse, and Watchlists+Collections send-to-Console flows.
- Role changes must be visible before launch/send.
- Editable-target role must expose approval implications.

### Approval Card

Approval cards are used when the user must decide before a write, publish, external action, destructive action, or policy-restricted operation.

Required content:

- action requested
- target
- consequence
- requesting agent/persona/run where relevant
- `View Diff` or equivalent when content changes
- `Approve`
- `Reject`
- shortcut labels

Approval cards must be visible in Console for full decision context and summarized on Home for lightweight action.

### Tool And Run Event Row

Tool/run event rows support Console, Home active work, Workflows, ACP, MCP audit, and Schedules.

Required fields when available:

- timestamp
- actor
- event type
- target
- result
- duration
- recovery state when failed

### Form Field Row

Form field rows standardize settings, builder, workflow, persona, MCP, ACP, and skill configuration.

Required structure:

- label
- control
- help text
- validation state
- source of value when relevant
- restart/apply impact when relevant

### Shortcut Bar

The bottom bar shows active shortcuts and compact status for the current context.

Rules:

- It must not duplicate primary navigation.
- It must update when the active destination/context changes.
- It must not show stale shortcuts.
- It should preserve power-user speed without hiding mouse-safe actions.

## Composition Patterns

Screen templates are examples, not binding pixel targets. They show how components compose across the master shell.

### Home Dashboard

Recommended composition:

- destination header
- readiness summary
- attention queue
- active work
- one next-best action
- recent work
- quick controls

Home should be compact and actionable, not decorative.

### Console Live Run

Recommended composition:

- destination header with session/provider/source status
- staged sources or source stack
- transcript/event stream
- composer
- run inspector
- approval and recovery cards
- artifact links

Console is the only live run surface.

### Destination Workspace

Recommended composition:

- destination header
- local section tabs
- main list/table/work area
- optional local inspector
- one primary action

Use for Library, Artifacts, Personas, Watchlists+Collections, MCP, ACP, Skills, and Settings where appropriate.

### Builder Or Wizard

Recommended composition:

- step outline
- current step form
- readiness panel
- dry-run preview
- summary
- save, dry-run, schedule, or launch actions

Use for Workflows, Schedules, complex imports, and policy setup.

### Management Matrix

Recommended composition:

- tree/list navigation
- matrix/table
- policy preview
- inspector
- audit/recovery controls

Use for MCP permissions, ACP compatibility, Skills permissions, Settings validation, and source authority matrices.

### Review Or Approval Surface

Recommended composition:

- item diff or preview
- consequence summary
- approval controls
- recovery path
- related run context

Use in Console and Home summaries.

### Artifact Or Source Staging

Recommended composition:

- selected items table
- role chips
- authority badges
- clear/edit/send-to-Console controls

Use in Library, Artifacts, Workflows, Watchlists+Collections, and Console staged context.

### Layout Rules

- Top bar is global primary navigation only.
- Destination context, workspace, backend/source authority, and readiness belong inside destination page headers or local status regions.
- Each page gets the full content area between the top nav and bottom shortcut/status bar.
- Left rails are optional local navigation inside a destination, not global nav.
- Right inspectors are optional local detail/recovery panels.
- Bottom bar shows current-context shortcuts/status and must not duplicate primary navigation.
- Comfortable density may stack or collapse local panels, but it must not remove status, authority, recovery, or shortcut information.

## Textual And TCSS Mapping

The design system should map onto the existing Textual app without requiring a rewrite.

### Theme Identity

Add a canonical default theme, tentatively named `agentic_terminal`.

The theme should provide:

- dark terminal canvas
- semantic cyan focus/action tokens
- green success/ready tokens
- amber warning/approval tokens
- red blocked/error/destructive tokens
- neutral gray panel/grid tokens
- high-contrast-safe variants

### Semantic Variables

Extend the TCSS variable layer around:

- surfaces
- text
- action states
- status states
- source authority
- source roles
- borders
- focus
- density
- shortcuts

Avoid hard-coded color values in widgets and feature-specific CSS.

### Component Classes

Introduce shared classes such as:

- `.ds-panel`
- `.ds-inspector`
- `.ds-destination-header`
- `.ds-status-badge`
- `.ds-recovery-callout`
- `.ds-source-role`
- `.ds-approval-card`
- `.ds-event-row`
- `.ds-field-row`
- `.ds-toolbar`
- `.ds-shortcut-bar`

These classes should live in modular TCSS component files, not Python `DEFAULT_CSS` strings.

### Base Screen Contract

Keep `BaseAppScreen` as the current migration seam, but evolve the shell contract:

- shell owns global top navigation
- shell owns bottom shortcut/status bar
- content screens own the full page body between top nav and bottom bar
- local rails and inspectors are screen-owned
- no competing global navigation systems are mounted

### Density Classes

Use class-level density selectors:

- `.density-compact`
- `.density-comfortable`

Components should not need separate implementations for density changes.

### State Classes

Standardize state class names:

- `.is-active`
- `.is-disabled`
- `.is-blocked`
- `.is-running`
- `.is-paused`
- `.is-unsaved`
- `.is-stale`
- `.is-conflict`
- `.needs-approval`
- `.source-local`
- `.source-server`
- `.source-workspace`
- `.source-remote-only`
- `.source-dry-run`

### Testing Hooks

Components need stable IDs or classes for Textual tests.

Priority hooks:

- primary action
- disabled recovery
- status badge
- source authority
- source role
- approval card
- shortcut bar
- next-best action
- open/follow in Console

## Governance

The design system is a product-quality contract, not just a style guide.

Rules:

- New UI work should use semantic tokens/classes before adding screen-specific styling.
- New components must document status, disabled, empty, loading, error, and recovery states.
- Screen reviews should check Nielsen Norman heuristics explicitly.
- Accessibility checks must verify non-color status labels, contrast, keyboard reachability, and mouse-safe affordances.
- Each migrated surface should include focused Textual tests for primary action, disabled recovery, status badge, source authority, and shortcut visibility.
- Migration should proceed in layers: tokens first, shared components second, shell/navigation third, destination-by-destination adoption after that.
- Existing screenshots are reference examples only; a screen can differ if it follows tokens, component rules, density, and shell contracts.

## Nielsen Norman Validation

### Visibility Of System Status

Components must show readiness, active state, blockers, approvals, and source authority near the affected task.

### Match Between System And Real World

Labels must align with the approved master shell vocabulary: Home, Console, Library, Artifacts, Personas, Watchlists+Collections, Schedules, Workflows, MCP, ACP, Skills, and Settings.

### User Control And Freedom

Users can skip Home, edit staged context, remove source roles, cancel/pause runs, reject approvals, and recover from failed schedules or agents.

### Consistency And Standards

Every destination uses the same component grammar for page headers, panels, status, recovery, tables, inspectors, and shortcuts.

### Error Prevention

Unavailable capabilities are visible before action: missing model, missing index, blocked MCP tool, unavailable ACP runtime, invalid schedule, missing skill dependency, server/auth failure, and unsupported local/server action.

### Recognition Rather Than Recall

Components expose purpose lines, primary actions, status lines, tooltips, command palette labels, and shortcuts.

### Flexibility And Efficiency Of Use

Compact density, keyboard shortcuts, command palette, and dense tables preserve power-user speed. Comfortable density and explicit labels support new users.

### Aesthetic And Minimalist Design

Dense TUI structure is acceptable. Competing dominant regions, decorative dashboards, duplicated global navigation, and unrelated controls in the same panel are not.

### Help Users Recognize, Diagnose, And Recover From Errors

Recovery callouts name the owner, the problem, and the next action.

### Help And Documentation

Destination empty states and Home next-best actions should teach enough to complete the next useful action without external docs.

## Migration Strategy

### Phase 1: Token And Theme Foundation

- Define `agentic_terminal` semantic tokens.
- Map default colors to New_UI visual references.
- Add high-contrast-safe fallback expectations.
- Avoid broad screen rewrites.

### Phase 2: Shared Component Classes

- Add modular TCSS component classes.
- Move new UI styling toward shared classes instead of inline CSS.
- Document component states and testing hooks.

### Phase 3: Shell Contract

- Align global top nav with primary destination navigation.
- Preserve page ownership of the area between top nav and bottom bar.
- Define bottom shortcut/status bar behavior.
- Preserve route IDs where practical.

### Phase 4: Destination Adoption

- Adopt shared headers, panels, inspectors, tables, status badges, recovery callouts, and shortcut bars destination by destination.
- Start with Home and Console because the master shell spec depends on them.
- Continue through Library, Artifacts, Personas, Watchlists+Collections, Schedules, Workflows, MCP, ACP, Skills, and Settings.

### Phase 5: Audit Replay

- Re-run first-time user and power-user workflows.
- Verify source authority, status, recovery, and shortcut consistency.
- Capture screen mockups only where they clarify a remaining design decision.

## Open Questions For Implementation Planning

- Which Textual theme variables should be extended versus encoded as TCSS classes?
- Should the top navigation be a single Textual widget shared across all primary screens or app-shell-owned outside screen content?
- Where should global density mode be stored and exposed?
- Which existing screens should be wrapped first after Home and Console?
- How should legacy inline `DEFAULT_CSS` be migrated without blocking feature work?
- Which automated tests should become design-system contract tests?

## Acceptance Criteria For The Design System

- The default design identity is documented as the New_UI-inspired agentic terminal system.
- The system supports both compact and comfortable density.
- Component rules cover panels, inspectors, status, recovery, source roles, approvals, tables/lists, forms, and shortcuts.
- Top bar is defined as primary destination navigation.
- Destination page context is kept inside the page body, not the global top nav.
- Textual/TCSS mapping is explicit enough for implementation planning.
- Governance and Nielsen validation are explicit.
- The design system defers implementation to a separate plan.
