# Destination Visual Parity Correction Design

Date: 2026-05-08
Status: Draft for review
Target branch: `dev` after PR #282 / `75c6f76f`

## Purpose

The current destination screens often render the right labels and stable selectors, but the actual terminal layouts do not match the approved ASCII destination contracts. Several screens are visually unusable at normal terminal sizes because mode bars, control bars, or explanatory stacks consume the viewport before the primary workbench appears.

This correction pass brings the visible Textual layouts into parity with the ASCII contracts before other agents continue deeper feature work. The goal is full visual parity across the major destinations, not full backend feature completion.

## Source Of Truth

- Binding layout contract: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Merged Collections split: `Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md`
- Design system direction: `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- Current visual evidence from mounted `140x42` harness:
  - Home dashboard primary grid is visible, but next-best and recent work fall below the viewport.
  - Console renders a two-column stacked layout; the ASCII contract requires staged context, transcript, and inspector as three horizontal workbench panes with composer at the bottom.
  - Library mode bar renders as a tall block and pushes the source workbench down.
  - Artifacts, Personas, Watchlists, Schedules, Workflows, ACP, Skills, and Settings render mostly as vertical explanatory stacks.
  - MCP embeds a real panel, but internal rows overflow far below the viewport and do not match the server/tool tree contract.

## Reference Patterns

Use the referenced Textual apps as pattern sources, not literal visual targets:

- Frogmouth: compact navigation, document/workbench focus, keyboard-first controls.
- Dooit: dense list/detail task surfaces, quick actions, help affordance.
- Bagels: tables, filters, jump navigation, fast repeated workflows.
- hledger-textual: report/list drill-down, high-density rows, stable inspector context.
- Toad: agentic terminal shell, session resume, significant-key footer, status-heavy workbench.
- Prosaic: focused writing mode and outline/metrics as optional density references.

The shared principle is a terminal-native workspace that keeps the active object, detail area, inspector, and key actions visible without web-style card stacks.

## Non-Goals

- Do not implement full backend feature depth for skeleton modules.
- Do not add node-graph workflow editing, browser-style cards, or modal-heavy flows.
- Do not move source Import/Export into Artifacts.
- Do not reintroduce Collections under Watchlists.
- Do not absorb MCP, ACP, Skills, Personas, Schedules, or Workflows setup into Settings.
- Do not rewrite Chat provider execution, RAG retrieval semantics, or artifact persistence unless required by layout wiring.

## Shared Visual Grammar

Every top-level destination should fit this grammar at `140x42`:

```text
+--------------------------------------------------------------------------------+
| Global nav, one compact row. More/Ctrl+P must not overlap visible nav items.    |
+--------------------------------------------------------------------------------+
| Destination | Role/status/authority badges | Primary action or readiness        |
+--------------------------------------------------------------------------------+
| Mode/filter/category strip, one row preferred, two rows maximum.                |
+----------------------+--------------------------------------+------------------+
| List/tree/scope pane  | Detail/workspace/transcript pane      | Inspector/actions|
+----------------------+--------------------------------------+------------------+
| Footer: significant keys and state.                                             |
+--------------------------------------------------------------------------------+
```

Exceptions:

- Home uses a dashboard grid instead of a strict list/detail/inspector model.
- Console uses staged context, transcript/event stream, and run inspector, with composer pinned below the transcript or spanning the bottom.
- Settings uses category/form/impact panes instead of mode/list/detail.

### Layout Constraints

- Top navigation must consume no more than 3 rows.
- Destination header/status/purpose must consume no more than 3 rows total.
- Mode/filter/category strips must consume no more than 2 rows.
- Primary workbench panes must begin by row 12 at `140x42`.
- Primary workbench panes must have at least 20 visible rows at `140x42`, unless the screen is intentionally in an empty/recovery state.
- No primary action footer or selected-item inspector may render below the visible viewport at `140x42`.
- At `100x32`, each destination must still expose nav, destination identity, primary object list, selected detail, and one recovery/action path. Less important panes may collapse under the detail pane only with explicit labels.

## Destination Corrections

### Home

Current issue: the dashboard grid exists, but next-best action and recent work render below the visible viewport at `140x42`.

Target layout:

```text
| Home | Status / notifications / active work | Ready/Blocked | Local          |
| Model Ready | RAG Missing | Watchlists Ready | 2 active | 1 approval       |
+----------------------+--------------------------+---------------------------+
| Attention Queue      | Active Work              | Selected Item             |
| > Approval needed    | > Daily papers running   | Daily papers              |
|   Failed schedule    |   RAG Summary Chatbook   | Status: running           |
|                      | [Approve] [Pause] [Retry]| [Open details] [Console]  |
+----------------------+--------------------------+---------------------------+
| Next: Review pending approval | Recent: RAG Summary Chatbook | Last Console |
| Footer: / search | Enter open | A approve | P pause | R retry           |
```

Required correction:

- Compress title/purpose/status into compact rows.
- Keep attention, active work, selected item, next-best action, and recent work visible at `140x42`.
- Preserve dedicated Chatbook controls from earlier Home fixes.

### Console

Current issue: Console visually renders as a control-heavy two-column stack. The control bar consumes too much height, the transcript is only half-width, the inspector is stacked under staged context, and the composer is not a dominant bottom command surface.

Target layout:

```text
| Console | Agent workbench | Provider/model | Persona | RAG | Tools | Approvals |
+----------------------+--------------------------------------+------------------+
| Staged Context       | Transcript / Event Stream             | Run Inspector    |
| [evidence] note.md   | User: summarize these sources         | Provider: ready  |
| [context] chatbook   | Assistant: grounded answer...         | Tools: 4 ready   |
|                      | Tool: search complete                 | Approval needed  |
+----------------------+--------------------------------------+------------------+
| Composer: ask or command...                         [Send] [Stop] [Save CB]   |
| Footer: C context | I inspector | A approval | Z zen | Esc cancel         |
```

Required correction:

- Convert the workspace to three horizontal panes: staged context, dominant transcript/event stream, run inspector.
- Compress provider/model/persona/RAG/source/tool status into a compact strip.
- Pin the composer/action row to the bottom of the Console workspace.
- Keep blocked provider/RAG/setup reasons visible in the inspector without displacing the transcript.

### Library

Current issue: the mode bar consumes far too much height, leaving too little visible workbench. Collections now belong in Library after PR #282, so the workbench must support that without breaking source/search modes.

Target layout:

```text
| Library | Sources/Search/RAG/Workspaces/Collections/Study | Ready | Local      |
| Modes: Sources Search/RAG Import/Export Workspaces Collections Study Cards Quiz |
+----------------------+--------------------------------------+------------------+
| Source Browser       | Source Detail / Mode Workspace        | Source Inspector |
| Notes                | Preview / chunks / transcript         | Authority: local |
| Media                | Search/RAG or Collections panel       | Ask in Console   |
| Collections          |                                      | Generate Cards   |
+----------------------+--------------------------------------+------------------+
| Footer: / search | Enter open | C stage | I import | E export           |
```

Required correction:

- Make the Library mode bar a compact one-row strip or two-row wrap, never a tall button stack.
- Keep source browser, detail/workspace, and inspector visible at `140x42`.
- In Collections mode, mount the Collections list/detail/form inside the center/detail workspace while keeping a collection inspector visible.
- Keep Search/RAG evidence/citations/snippets visible without pushing the workbench below the viewport.

### Artifacts

Current issue: Artifacts is a vertical Chatbooks launcher/recovery page.

Target layout:

```text
| Artifacts | Outputs, Chatbooks, reports, datasets, exports | Ready | Local     |
| Types: All Chatbooks Reports Datasets Drafts Exports | Sort: Recent            |
+----------------------+--------------------------------------+------------------+
| Artifact List        | Artifact Preview / Detail             | Provenance       |
| > Chatbook: latest   | Title, summary, transcript preview    | Created: Console |
|   Report             |                                      | Reopen Console   |
|   Dataset            |                                      | Export / Bundle  |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Add visible type/filter strip, artifact list pane, preview/detail pane, and provenance/action inspector.
- In empty state, keep Open Console, Open Library, and Import Artifact visible inside the same layout.
- Keep Chatbooks first-class but not the only visible concept.

### Personas

Current issue: Personas is a vertical snapshot page with a legacy open button.

Target layout:

```text
| Personas | Behavior, characters, prompts, lore | Ready | Local/Server        |
| Modes: Personas Characters Prompts Dictionaries Lore Import/Export             |
+----------------------+--------------------------------------+------------------+
| Persona List         | Behavior Profile Detail               | Attachments      |
| > Research Analyst   | Goals, tone, constraints, exemplars   | Console: ready   |
|   Fiction Character  | Dictionaries and lore links           | Workflows: ready |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Add mode strip and three-pane workbench.
- Preserve current local snapshot data as list rows.
- Put legacy `Open Personas` and `Attach to Console` into the inspector/action pane, not the top of a vertical stack.

### Watchlists

Current issue after PR #282: label is corrected to Watchlists, but the screen still needs the ASCII watchlist control-plane layout.

Target layout:

```text
| Watchlists | Monitored sources, runs, alerts, recovery | Mixed | Local/Server |
| Filters: Running Failed Recent Alerts Sources Feeds                            |
+----------------------+--------------------------------------+------------------+
| Watchlist List       | Detail / Items / Runs                 | Status Inspector |
| > Daily papers       | Source/query/schedule/latest items    | State: running   |
|   Security feeds     |                                      | Retry/backoff    |
| Alerts               |                                      | Follow Console   |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Remove any visual implication that Collections are managed here.
- Use a list/detail/inspector structure.
- Keep stage-to-Console and follow-in-Console actions in the inspector.

### Schedules

Current issue: Schedules is a placeholder vertical list.

Target layout:

```text
| Schedules | When work runs | Ready | Local/Server                         |
| Filter: Active Paused Failed Upcoming | Scope: Workflows Watchlists Library   |
+----------------------+--------------------------------------+------------------+
| Schedule List        | Schedule Detail / Upcoming Runs       | Controls         |
| > Morning digest     | Trigger, timezone, next/last run      | Pause / Run Now  |
|   Broken workflow    | Failure detail when selected          | Retry / Console  |
+----------------------+--------------------------------------+------------------+
| Run History: latest success / failed cause / retry readiness                    |
```

Required correction:

- Convert placeholder labels into list/detail/control/history panes.
- Empty and failed states must explain cause/recovery inside the same workbench.

### Workflows

Current issue: Workflows is a placeholder vertical list.

Target layout:

```text
| Workflows | Procedures, steps, inputs, outputs, approvals | Ready | Workspace |
| Modes: Browse Build Runs Templates | Filter: Draft Active Failed               |
+----------------------+--------------------------------------+------------------+
| Workflow List        | Builder / Run Detail                  | Readiness        |
| > Research digest    | Step 1: Select Library sources        | Inputs: ready    |
|   Code audit         | Step 2: Search/RAG                    | Tools: 3 ready   |
| Templates            | Approval points and outputs           | Dry Run / Launch |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Convert placeholder labels into a terminal-row builder/workbench.
- Keep Launch/Follow Console action in the readiness inspector.
- Clearly separate "what runs" from Schedules' "when it runs."

### MCP

Current issue: MCP has real controls, but it does not visually match the server/tool tree contract and overflows below the viewport.

Target layout:

```text
| MCP | Tool/resource protocol management | Ready | Local/Server             |
| Modes: Servers Tools Resources Permissions Audit | Filter: Blocked Ready       |
+----------------------+--------------------------------------+------------------+
| Server / Tool Tree   | Tool Or Server Detail                 | Readiness        |
| v filesystem         | Schema / payload / result preview     | Auth: ok         |
|   > read_file        |                                      | Permission: ask  |
| v browser            |                                      | Test / Audit     |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Wrap or refactor `UnifiedMCPPanel` into the shared three-pane grammar.
- Make Source/Server/Scope/Section compact controls, not vertical blocks.
- Ensure no MCP controls render outside the viewport at `140x42`.
- Use server-first grouping with visible tool readiness/permission.

### ACP

Current issue: ACP is an honest but vertical runtime-unconfigured shell.

Target layout:

```text
| ACP | Agent protocol sessions and runtimes | Runtime needed | Local/Remote |
| Modes: Agents Sessions Runtimes Compatibility | Filter: Ready Blocked        |
+----------------------+--------------------------------------+------------------+
| Agent/Session List   | Session Detail / Runtime Setup        | Compatibility    |
| > Codex local        | Runtime setup steps                   | ACP version: n/a |
|   No sessions        |                                      | Launch disabled  |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Keep the runtime-missing state, but place it in detail/setup plus compatibility inspector panes.
- Keep launch/follow disabled with target-specific reasons.

### Skills

Current issue: Skills is a vertical local-directory page with disabled import.

Target layout:

```text
| Skills | Agent Skills packs, validation, attachments | Ready | Local/Server |
| Modes: Installed Discover Import Validate Attach | Filter: Valid Broken        |
+----------------------+--------------------------------------+------------------+
| Skill List / Tree    | SKILL.md / Files / Instructions       | Validation       |
| > pdf-processing     | frontmatter and instructions preview  | Frontmatter: ok  |
|   scripts/           |                                      | Attach targets   |
+----------------------+--------------------------------------+------------------+
```

Required correction:

- Show installed local skills as a left list/tree.
- Show `SKILL.md`/directory placeholder detail in the center.
- Move validation, import, and attach actions into the inspector.

### Settings

Current issue: Settings is a vertical category list with one Appearance button.

Target layout:

```text
| Settings | Global preferences, providers, privacy, diagnostics | Ready | Local |
| Categories: Providers Models Storage Privacy Appearance Diagnostics            |
+----------------------+--------------------------------------+------------------+
| Category List        | Setting Form / Diagnostic Detail      | Impact / Status  |
| > Providers          | Default provider/model/test controls  | Affects Console  |
|   Appearance         |                                      | Saved / reload   |
+----------------------+--------------------------------------+------------------+
| Boundary: destination-specific config stays in owning destinations.             |
```

Required correction:

- Convert static category labels into category list/detail/impact panes.
- Keep `Open Appearance` as an action in the detail or inspector pane.
- Preserve explicit boundary copy.

## Test And QA Requirements

Add mounted visual-geometry tests that verify:

- Global nav does not overlap `More: Ctrl+P` with a visible destination label at `140x42`.
- Each destination's primary workbench starts by row 12 at `140x42`.
- Mode/filter strips are no more than 2 rows high.
- Home's next-best and recent work regions are visible at `140x42`.
- Console has three horizontal primary panes and a visible composer.
- Library has a compact mode strip and visible source/detail/inspector panes.
- Skeleton destinations expose list/detail/inspector geometry rather than a single vertical stack.
- MCP content and actions do not render outside the viewport at `140x42`.
- Compact `100x32` screens still show identity, primary object list, detail, and at least one recovery/action path.

Manual QA must include a mounted or live Textual walkthrough of all top-level destinations at `140x42` and `100x32`, with saved geometry or screenshot evidence.

## Implementation Boundaries

The implementation plan should split work into reviewable slices:

1. Shared shell/nav/density and reusable layout classes.
2. Home, Console, and Library visual corrections.
3. Output/source-prep destinations: Artifacts, Personas, Watchlists, Skills.
4. Operational destinations: Schedules, Workflows, MCP, ACP, Settings.
5. QA closeout and roadmap/backlog tracking.

Each slice should use TDD:

- Add failing geometry/layout tests first.
- Implement the smallest safe layout changes.
- Run focused UI tests and `git diff --check`.

## Acceptance Criteria

- The current visual layouts match the ASCII destination contracts closely enough that a user can identify the same regions without reading code.
- The app no longer relies on vertical explanatory stacks for top-level destination screens.
- At `140x42`, all top-level destinations show their primary workbench and primary action/recovery path without scrolling.
- At `100x32`, all top-level destinations remain usable through explicit compact behavior.
- Collections remain Library-owned, and Watchlists remains top-level without Collections management.
- Console remains the primary live agentic control surface.
- No destination falsely implies unavailable backend functionality is implemented; blocked or placeholder states remain honest and recoverable.
