# Agentic Terminal Visual System Contract

Date: 2026-05-03
Status: Phase 0 implementation-facing visual contract
Source Spec: `Docs/superpowers/specs/2026-05-03-agentic-terminal-visual-design-system-design.md`

## Purpose

This document translates the visual design-system spec into implementation-facing rules for Textual, TCSS, tests, and future shell slices.

Runtime shell implementation must not introduce new visual patterns, token names, or state treatments that are not represented here. If a runtime slice needs a new pattern, update this contract first.

## Product Frame

- Home is the orientation, readiness, active work, and next-action surface.
- Console is the primary agentic control surface.
- Top navigation is global primary destination navigation.
- Destination pages own local context, readiness, source authority, recovery, and primary actions.
- Footer owns explicit shortcut context and compact global/active-work status.
- Workspaces, Personas, Flashcards, Quizzes, Library, Search, Media, Notes, Artifacts, Handoffs, MCP, ACP, Skills, Schedules, and Workflows remain visible in labels, help, command palette, headers, or route inventory.

## Verified Runtime Notes

- The visible top navigation may use compact labels, but runtime metadata must expose full labels through `ShellDestination.full_label`/`accessible_label`, tooltips, and command-palette help/search.
- `W+C` is the verified compact-label case; `Watchlists+Collections` remains searchable and visible in help text.
- The footer shortcut display is backed by `ShortcutContext` and `ShortcutAction`; changing context replaces stale shortcuts instead of appending to them.
- `BaseAppScreen` is still the current screen wrapper seam. Tests verify it mounts one `MainNavigationBar`; a full app-owned chrome migration is deferred.
- Runtime TCSS uses hyphenated `$ds-*` variables only. Dotted concept names are allowed in design discussion, not TCSS variable references.
- The token table below is the design-system vocabulary. The current hardened runtime contract verifies the semantic tokens already consumed by shared agentic terminal classes and should expand only as new classes need additional aliases.
- These notes do not claim screen-level redesign completion. Home and Console shell contracts are hardened first; other destinations keep their product-model commitments until separately implemented and verified.

## Semantic Tokens

TCSS variables and `Theme.variables` keys use hyphenated names only.

| Token | Purpose |
| --- | --- |
| `ds-surface-canvas` | app background |
| `ds-surface-panel` | standard panel background |
| `ds-surface-panel-raised` | selected or emphasized panel background |
| `ds-surface-field` | input/editable field background |
| `ds-surface-overlay` | modal/popover/floating surface |
| `ds-surface-divider` | separators, grid lines, passive borders |
| `ds-text-primary` | normal readable text |
| `ds-text-secondary` | supporting text |
| `ds-text-muted` | metadata/helper text |
| `ds-text-disabled` | unavailable values |
| `ds-text-inverse` | text on strong fills |
| `ds-text-code` | commands, paths, IDs, logs |
| `ds-action-primary` | dominant action |
| `ds-action-secondary` | secondary action |
| `ds-action-hover` | hover state |
| `ds-action-focus` | keyboard focus |
| `ds-action-disabled` | disabled control |
| `ds-action-destructive` | delete, revoke, deny, cancel run |
| `ds-status-ready` | ready/healthy/synced |
| `ds-status-running` | active live work |
| `ds-status-info` | neutral information |
| `ds-status-warning` | caution/stale/review needed |
| `ds-status-approval-required` | user decision required |
| `ds-status-blocked` | blocked capability or permission |
| `ds-status-error` | failed operation |
| `ds-status-paused` | paused run/schedule/workflow |
| `ds-status-unsaved` | draft or pending edits |
| `ds-status-recovered` | recovery succeeded |
| `ds-authority-local` | local authority |
| `ds-authority-server` | server authority |
| `ds-authority-workspace` | workspace authority |
| `ds-authority-remote-only` | remote-only capability |
| `ds-authority-dry-run` | read-only preview |
| `ds-authority-syncing` | sync in progress |
| `ds-authority-synced` | up to date |
| `ds-authority-conflict` | conflicting state |
| `ds-source-role-context` | background source role |
| `ds-source-role-evidence` | evidence/citation role |
| `ds-source-role-editable-target` | editable target role |
| `ds-source-role-output-seed` | output seed role |
| `ds-structure-border` | standard panel border |
| `ds-structure-border-strong` | active/emphasized border |
| `ds-structure-grid-line` | table/list separators |
| `ds-structure-focus-ring` | keyboard focus outline |
| `ds-structure-active-row` | selected row |
| `ds-structure-inactive-row` | inactive row |

## Component Classes

| Class | Required use |
| --- | --- |
| `.ds-destination-header` | page title, purpose, readiness, authority, primary action |
| `.ds-panel` | bordered region with one clear job |
| `.ds-inspector` | selected detail, provenance, permissions, recovery |
| `.ds-status-badge` | readable semantic status label |
| `.ds-recovery-callout` | owner/problem/impact/next action |
| `.ds-source-role` | context/evidence/editable-target/output-seed chip |
| `.ds-approval-card` | decision surface for writes/external/destructive actions |
| `.ds-event-row` | tool/run/audit/schedule/workflow event |
| `.ds-field-row` | label/control/help/validation row |
| `.ds-toolbar` | local action group |
| `.ds-shortcut-bar` | active shortcut context and compact status |

## State Classes

| Class | Meaning |
| --- | --- |
| `.is-active` | active destination, selected row, current item |
| `.is-disabled` | unavailable control |
| `.is-blocked` | blocked by missing requirement |
| `.is-running` | live work in progress |
| `.is-paused` | paused work |
| `.is-unsaved` | pending local edits |
| `.is-stale` | stale source/index/config |
| `.is-conflict` | conflicting state |
| `.needs-approval` | user decision required |
| `.source-local` | local authority |
| `.source-server` | server authority |
| `.source-workspace` | workspace authority |
| `.source-remote-only` | no local CRUD |
| `.source-dry-run` | read-only preview |

## Density Rules

Use shared density classes instead of separate widgets.

| Density | Use |
| --- | --- |
| `.density-compact` | default expert mode, dense lists/tables, short labels, visible shortcuts |
| `.density-comfortable` | first-time-readable mode, more padding, clearer labels, same semantics |

Density affects panel padding, row height, field height, panel gap, footer height, header height, table cell padding, and inspector width. Density must not remove status, authority, recovery, or shortcuts.

## Layout Grammar

```text
+--------------------------------------------------------------------------------+
| Top nav: primary destinations only, Home and Console always visible             |
+--------------------------------------------------------------------------------+
| Destination header: title, purpose, readiness, authority, primary action        |
+--------------------------+-----------------------------------+-----------------+
| Main work area           | Main work area                     | Inspector       |
| dashboard/list/table     | transcript/forms/staging/preview   | context/recover |
+--------------------------+-----------------------------------+-----------------+
| Footer: explicit ShortcutContext, compact global/active-work status             |
+--------------------------------------------------------------------------------+
```

## Reference Mockups

These mockups are layout contracts, not pixel targets.

### Home

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ... More   |
+--------------------------------------------------------------------------------+
| Home | Workspace: Research Lab | Ready | Server: local                         |
+-------------------------+--------------------------+---------------------------+
| Attention               | Active Work              | Readiness                 |
| Approval required: 2    | Console run: indexing    | Local DB Ready            |
| MCP auth blocked        | RSS schedule paused      | RAG Stale                 |
+-------------------------+--------------------------+---------------------------+
| Next Best Action: Open Console approvals | Recent: notes, media, decks         |
+--------------------------------------------------------------------------------+
| Ctrl+P palette | Enter open | R refresh | global status                       |
+--------------------------------------------------------------------------------+
```

### Console

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ... More   |
+--------------------------------------------------------------------------------+
| Console | Workspace: Research Lab | Ready | Sources: 4                         |
+-------------------+------------------------------------------+-----------------+
| Staged Context    | Transcript / Event Stream                | Inspector       |
| evidence paper    | user: summarize workspace                | Provider Ready  |
| context notes     | tool: rag.search Running                 | RAG Workspace   |
| editable draft    | approval: View Diff Approve Reject       | Approval needed |
+-------------------+------------------------------------------+-----------------+
| Composer: ask, run, diff, approve, attach source, select persona               |
+--------------------------------------------------------------------------------+
| Ctrl+Enter send | Ctrl+K commands | Esc cancel | Approval required            |
+--------------------------------------------------------------------------------+
```

### Library / Workspace

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ... More   |
+--------------------------------------------------------------------------------+
| Library | Workspace: Research Lab | Source authority: Local | Index: Stale     |
+--------------------+--------------------------------------+--------------------+
| Workspaces         | Sources                              | Source Inspector   |
| > Research Lab     | notes/meeting.md Local Ready         | Role: evidence     |
|   Course Prep      | media/paper.pdf Indexed              | Recovery: reindex  |
+--------------------+--------------------------------------+--------------------+
| Import | Search | Reindex | Stage for Console | Use in Chat                     |
+--------------------------------------------------------------------------------+
| Ctrl+F search | Space select | Ctrl+Enter stage | 3 selected                   |
+--------------------------------------------------------------------------------+
```

### Study / Personas

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas W+C Schedules Workflows MCP ... More   |
+--------------------------------------------------------------------------------+
| Study | Workspace: Course Prep | Ready | Persona: Tutor                       |
+-------------------+------------------------------+-------------------------------+
| Study Modules     | Decks / Quizzes              | Persona / Generation         |
| > Flashcards      | Biology Deck Ready           | Tutor persona active         |
| > Quizzes         | Exam Quiz Unsaved            | Generate from Library source |
| > Review Schedule | Missed cards Warning         | Open in Console              |
+-------------------+------------------------------+-------------------------------+
| Start Review | Create Quiz | Generate Cards | Open in Console                  |
+--------------------------------------------------------------------------------+
| Enter open | G generate | Ctrl+P palette | Study direct route retained       |
+--------------------------------------------------------------------------------+
```

## Behavioral Testing Guidance

Tests should assert:

- stable IDs or classes for behavior hooks
- readable status text
- state classes and source authority classes
- command-palette discoverability for compact labels and hidden/direct routes
- footer context clears stale shortcuts
- destination context remains inside page headers/panels, not top nav

Tests should not assert:

- raw color values
- exact pixel/character placement unless layout behavior depends on it
- concept screenshot parity

## Do Not Implement Yet

- Do not redesign every destination screen in this hardening slice.
- Do not create a second global navigation system.
- Do not remove legacy direct command-palette routes.
- Do not make screenshots mandatory review assets.
- Do not hide Workspaces, Personas, Flashcards, Quizzes, Search, Media, Notes, Artifacts, or Handoffs.
- Do not add new visual patterns without updating this contract first.
