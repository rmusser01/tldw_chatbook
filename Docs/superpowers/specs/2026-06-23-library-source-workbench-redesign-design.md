# Library Source Workbench Redesign

Date: 2026-06-23
Status: draft for review
Source baseline: `Docs/superpowers/specs/2026-06-09-library-content-hub-contract-design.md`
Actual-use evidence: `Docs/superpowers/qa/library-content-hub-closeout/2026-06-22-library-content-hub-actual-use-closeout.md`

## Purpose

The accepted Library content hub is functional, visually approved, and honest about blocked or deferred states. The next redesign should turn it from a mostly explanatory hub into a first-class source workbench.

The redesign should help users answer these questions quickly:

- What sources do I have?
- What workspace am I operating in?
- What can I search, stage, organize, import, export, or study?
- What saved collection items can I read, review, search, or hand off?
- What object is selected?
- What actions are allowed, blocked, or WIP?
- How do I recover from an empty or blocked state?

## Collections Reference From tldw_server

This spec treats `Collections` as a destination-native content page, not as a simple folder/group manager. The reference model comes from the local server project:

- `../tldw_server2/Docs/Product/Completed/Content_Collections_PRD.md`
- `../tldw_server2/Docs/Product/Completed/Content_Collections_UX_Backlog_PRD.md`
- `../tldw_server2/tldw_Server_API/app/core/Collections/README.md`
- `../tldw_server2/apps/tldw-frontend/pages/collections.tsx`
- `../tldw_server2/apps/packages/ui/src/components/Option/Collections/index.tsx`
- `../tldw_server2/apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemsList.tsx`
- `../tldw_server2/apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemDetail.tsx`

Reference implications:

- Collections unify reading-list items, watchlist-derived items, output artifacts, tags, saved searches, and import/export workflows.
- A collection item is readable content first: title, domain, status, favorite, tags, summary, clean text, notes, highlights, origin metadata, and updated time.
- The primary page behavior is list/review/view/consume stored content. Grouping, membership, and sync metadata are secondary facets.
- The server WebUI exposes Reading List, Highlights, Templates, Digest Schedules, and Import/Export. Chatbook does not need full parity in the first visual pass, but it must reserve the correct information architecture so later parity work does not require another redesign.
- Search/RAG handoff should understand collection items as eligible evidence sources when local data and workspace policy allow it.

## Data and Integration Boundaries

The server Collections module is the product reference, not a runtime dependency for this visual redesign.

- Do not call tldw_server APIs from Library as part of this redesign unless a later task explicitly owns server integration and completes the ADR check.
- Chatbook should render collection item rows from local Chatbook data adapters only. If a local adapter is missing or incomplete, show an honest empty/WIP state rather than activating fake controls.
- Collection item UI should use capability flags before enabling actions: `can_read`, `can_update_status`, `can_favorite`, `can_tag`, `can_note`, `can_highlight`, `can_export`, `can_generate_output`, `can_stage_for_rag`, and `can_use_in_console`.
- Unsupported capabilities remain visible as disabled actions with visible reasons, so the IA is future-compatible without implying functionality exists.
- Search/RAG and Console handoff must respect workspace policy: global browse/search is allowed, but staging and active context use are limited to the current workspace and eligible item types.

## ADR Check

ADR required: no
ADR path: N/A
Reason: this spec changes screen layout, interaction hierarchy, and QA expectations only. It does not change storage/schema, sync conflict policy, provider/runtime boundaries, security policy, or service ownership. If a later implementation changes collection item storage, workspace eligibility rules, RAG persistence, note/highlight ownership, or sync promotion behavior, that implementation should perform a new ADR check.

## UX/HCI Findings From Current Rendered Library

### P1: Center pane reads like implementation notes

The current center pane exposes correct ownership and blocked-state copy, but it is mostly prose. Users must read dense paragraphs to understand what to do next. This reduces recognition and increases cognitive load.

Required improvement: convert mode detail into structured rows, object lists, evidence/result regions, and clear next-best actions.

### P1: Search/RAG primary action is visually underpowered

Search/RAG is one of Library's core workflows, but the query field, retrieval status, evidence list, and Console handoff do not dominate the mode. The empty recovery checklist is text-only and the inspector does not add enough specific guidance.

Required improvement: make query, scope, evidence, selected evidence, and Console handoff the obvious flow.

### P1: Inspector is not reactive enough

The inspector often repeats generic explanation. It should be the selected-object command surface: status, allowed actions, blocked actions, recovery, handoff target, shortcuts, and authority.

Required improvement: inspector content must depend on mode and selected item. If no item is selected, show concise mode help and the next recoverable action.

### P2: Left rail mixes navigation, status, and actions

The current left rail includes owner actions, module shortcuts, and workspace state in one visual stack. Users can operate it, but the grouping is unclear.

Required improvement: split left rail into `Workspace Context`, `Source Map`, and `Quick Actions`.

### P2: Workspaces policy is correct but too prose-heavy

The current Workspaces mode correctly says browse/search remain global while staging/manipulation are workspace-scoped. The presentation should be a matrix, not paragraphs.

Required improvement: use a visibility and eligibility grid.

### P1: Collections is mis-modeled as grouping instead of content consumption

The current draft frames Collections as local grouping, selected collection metadata, and membership placeholders. That is not aligned with the server Collections model or intended Chatbook purpose. Collections should behave like Notes or Conversations: a user-facing page for listing, reviewing, opening, reading, and acting on stored collection items.

Required improvement: list/filter saved collection items first, selected item reader/review second, item metadata/actions third, grouping/sync/template parity later.

### P1: Backing capability state is under-specified

The spec reserves future Collections actions, but without an explicit capability model an implementation could render buttons as active even when local Chatbook cannot read, tag, highlight, export, or stage a collection item.

Required improvement: every selected-object action must be enabled by a local capability flag or disabled with a visible reason. Server-only features must be labeled `WIP` or `server parity pending`.

## Design Principles

- Library is a source workbench, not a second Console.
- Console remains the live agentic control surface.
- Workspaces never hide global browse/search results. They only gate staging, manipulation, and active-context use.
- Collections are stored content pages, not just folders. Users should be able to list, read, review, tag, favorite, search, and later export or generate outputs from collection items.
- Empty, blocked, WIP, local-only, server-backed, and sync-unavailable states must be text-labeled.
- The interface should support repeated power use without removing first-time orientation.
- Keep the terminal-native three-column grammar, but make the columns earn their space.
- Prefer structured rows, short labels, and visible actions over explanatory paragraphs.
- Every screen change must be verified with actual CDP/Textual-web screenshots before approval.

## Target Shell

The Library shell keeps global nav, destination header, compact mode strip, and three visible columns. The left and right columns should be narrower than the center because the center is the active workbench.

Recommended wide layout:

```text
┌ Library | Global browse/search | Workspace: Default | Local ───────────────────────────────┐
│ Modes: Hub  Search/RAG  Import/Export  Collections  Workspaces  Study                       │
├────────────── Source Map ─────────────┬──────────── Active Workbench ────────────┬──────── Inspector ────────┤
│ Workspace Context                     │ Mode-specific primary task surface       │ Selected object          │
│ > Default                             │                                           │ Status                   │
│ Browse: all workspaces                │ Lists, results, forms, preview, matrix   │ Allowed actions          │
│ Use: current workspace only           │                                           │ Blocked actions          │
│                                       │                                           │ Recovery                 │
│ Source Map                            │                                           │ Handoff target           │
│ Notes           0                     │                                           │ Shortcuts                │
│ Media           0                     │                                           │                         │
│ Conversations   0                     │                                           │                         │
│ Collection items 1                    │                                           │                         │
│                                       │                                           │                         │
│ Quick Actions                         │                                           │                         │
│ Import sources                        │                                           │                         │
│ Run Search/RAG                        │                                           │                         │
└───────────────────────────────────────┴───────────────────────────────────────────┴─────────────────────────┘
│ Footer: / filter | Tab panes | Enter open/select | u use in Console | i import | e export                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Recommended proportions:

| Region | Role | Wide target | Compact behavior |
| --- | --- | --- | --- |
| Left column | workspace context, source map, quick actions | 24-30 cells | stay visible; collapse help text first |
| Center column | active mode workbench | 1fr dominant | never narrower than primary input/results need |
| Right column | selected-object inspector | 28-36 cells | hide lower-priority guidance first, not status |

## Mode Layouts

### Hub

Hub should show the user's source inventory and next-best actions, not a description of every subsystem.

```text
┌ Source Map ────────────────┬ Hub: Library Inventory ─────────────────────────────┬ Inspector ───────────────┐
│ Workspace                  │ Ready to build context                              │ No source selected       │
│ > Default                  │                                                      │ Browse/search: all       │
│ Browse: all                │ Sources                                             │ Use: workspace-scoped    │
│ Use: Default only          │ ┌ Notes          0  Empty      Open Notes    New ┐  │                         │
│                            │ ├ Media          0  Empty      Open Media Import │  │ Next best action        │
│ Sources                    │ ├ Conversations  0  Empty      Open Console      │  │ Import sources          │
│ Notes          0           │ ├ Collections    1  Local      Open Collections  │  │                         │
│ Media          0           │ └ Search/RAG     blocked      Add sources first  │  │ Blocked                 │
│ Conversations  0           │                                                      │ Use in Console: no src   │
│ Collections    1           │ Next                                                │ Recovery                │
│                            │ [Import sources] [Create note] [Run Search/RAG]     │ Import or create note    │
└────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘
```

Hub requirements:

- Replace long owner prose with rows containing count, readiness, owner, and primary action.
- Keep owner boundaries visible in short labels: `Owner: Notes`, `Owner: Media`, `Owner: Library`.
- Show next-best action based on state: no sources means import/create; sources means search/stage/organize.
- Do not show source preview controls until a source exists or is selected.

### Search/RAG

Search/RAG should read as a retrieval workbench.

```text
┌ Source Map ────────────────┬ Search/RAG ─────────────────────────────────────────┬ Evidence Inspector ──────┐
│ Scope                      │ Query                                                │ Status: blocked          │
│ All local                  │ [Ask Library sources                               ] │ Reason: no sources       │
│ Workspace: Default         │ [Run Search/RAG] [Use selected in Console]          │ Recovery: import source  │
│                            │                                                      │                         │
│ Sources                    │ Scope                                                │ Selected evidence        │
│ Notes          0           │ [All Library] [Current Workspace] [Collection items] │ none                    │
│ Media          0           │                                                      │                         │
│ Collections    1           │ Evidence                                             │ Handoff                 │
│                            │ No evidence yet. Add/import sources, then query.     │ Use in Console disabled │
└────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘
```

Search/RAG requirements:

- Query and run action are first-class, visually dominant, and keyboard reachable.
- Scope is explicit before the query runs.
- Evidence list supports future snippets and citations without another layout rewrite.
- `Use in Console` only enables after selected usable evidence exists.
- Inspector explains selected evidence, source authority, citation/snippet availability, and handoff state.

### Collections

Collections should become a stored-content reading and review workbench. It should let users browse saved collection items across origins, open an item, read or review its content, inspect status/tags/notes/highlights, and choose eligible handoffs. Grouping and sync controls are later facets, not the primary layout.

```text
┌ Source Map ────────────────┬ Collections ────────────────────────────────────────┬ Item Inspector ──────────┐
│ Collections                │ Filters                                              │ Selected item            │
│ > All items          1     │ [All] [Saved] [Reading] [Read] [Archived] [★]        │ Deep dive: local         │
│   Reading List       1     │ Query/tag/domain filter                              │ Status: saved           │
│   Watchlist Items    0     │ [filter collection items                            ] │ Origin: reading         │
│   Outputs            0     │                                                      │ Tags: research          │
│   Highlights         0     │ Items                                                │                         │
│                            │ ┌ Deep dive into RAG                     saved ★ ┐   │ Allowed                 │
│ Quick Actions              │ │ example.com | 8 min | tags: research          │   │ Open/read, favorite    │
│ Save URL                  │ │ Summary preview...                            │   │ Mark read/archive     │
│ Import/export             │ └────────────────────────────────────────────────┘   │ Tag, add note/highlight │
│ Run Search/RAG             │                                                      │                         │
│                            │ Reader / Review                                      │ Handoff                 │
│                            │ Title: Deep dive into RAG                            │ Use in Search/RAG       │
│                            │ Readable content preview or empty recovery.          │ Use in Console if gated │
│                            │ Notes and highlights summary.                        │                         │
│                            │                                                      │ Deferred                │
│                            │ Outputs/templates/digests: WIP unless local service. │ Server sync/parity WIP  │
└────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘
```

Collections requirements:

- Collection item list stays visible and scannable: title, domain/origin, status, favorite, tags, reading time or word count, and updated time.
- Selected item detail uses a reader/review region with readable content or a clear recovery state when content is missing.
- Inspector shows item identity, status, origin, workspace eligibility, allowed actions, blocked actions, and handoff targets.
- Data-backed item actions, when supported by local capability flags: open/read, mark saved/reading/read/archived, favorite/unfavorite, tag, add/edit notes, show highlights, export, generate output, and delete/archive with confirmation.
- Unsupported item actions stay visible but disabled with reasons such as `No local collection item store`, `Highlights service unavailable`, `Server parity pending`, or `Workspace blocks Console use`.
- Bulk action affordances are reserved: multi-select, set status, favorite, add/remove tags, delete/archive, generate output from selected items.
- Saved searches, highlights, templates, digest schedules, and import/export are visible as future-compatible lanes, but disabled or WIP-labeled unless local backing exists.
- Search/RAG and Console handoff target collection items, not abstract collection folders. If workspace policy blocks use, the reason must be visible.
- Do not imply server sync or tldw_server parity exists locally until the service is wired.

### Workspaces

Workspaces should show scope policy as a matrix.

```text
┌ Workspace Map ─────────────┬ Workspace Eligibility ──────────────────────────────┬ Handoff Inspector ───────┐
│ Workspaces                 │ Active workspace: Default                           │ Selected workspace       │
│ > Default                  │                                                      │ Default                  │
│   Project A                │ Visibility / Use                                     │                         │
│                            │ ┌ Area          Browse/Search   Stage/Use   Why ┐   │ Allowed                 │
│ Quick Actions              │ │ Notes         all             Default     safe│   │ Browse all sources      │
│ Create workspace           │ │ Media         all             Default     safe│   │ Switch workspace        │
│ Import sources             │ │ Collection items all          Default     safe│   │                         │
│                            │ │ File tools    no              no          default workspace locked │
│                            │ └──────────────────────────────────────────────┘   │ Blocked                 │
│                            │                                                      │ No source handoff yet    │
└────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘
```

Workspaces requirements:

- State clearly that switching workspace does not hide Library items.
- Show browse/search and stage/use as separate columns.
- Default workspace should be safe by default and should not imply filesystem tool permissions.
- If no workspace sources exist, recovery is import/assign sources, not "create workspace" alone.

### Import/Export

Import/Export should be a workflow launcher with explicit ownership boundaries.

```text
┌ Source Map ────────────────┬ Import / Export ────────────────────────────────────┬ Inspector ───────────────┐
│ Sources                    │ Bring content into Library                           │ Selected action          │
│ Notes          0           │ [Import files/media] [Open ingest] [Open media]      │ Import files/media       │
│ Media          0           │                                                      │ Owner: Ingest/Media     │
│ Conversations  0           │ Export Library content                               │                         │
│                            │ [Export selected] disabled, no selected source       │ Recovery                │
│                            │                                                      │ Import first            │
└────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘
```

Import/Export requirements:

- Explain where import/export routes go and how content returns to Library.
- Disable export until a source, collection item, or evidence set is selected.
- Avoid making route handoff look like accidental navigation.

### Conversations

Conversations should be a real browser when data exists and an explicit recovery state when it does not.

```text
┌ Source Map ────────────────┬ Conversations ──────────────────────────────────────┬ Conversation Inspector ─┐
│ Conversations              │ Saved conversations                                  │ Selected conversation    │
│ All              0         │ No saved conversations yet.                          │ none                    │
│ Workspace        0         │ Start in Console, then saved chats appear here.      │                         │
│                            │ [Open Console]                                       │ Allowed                 │
│                            │                                                      │ Open Console            │
└────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────┘
```

Conversations requirements:

- When empty, explain how conversations get here.
- When populated, list title, workspace, provider/model, updated time, and resume eligibility.
- Resume should be explicit and should not look like opening a new chat.

### Study, Flashcards, Quizzes

Study-related modes are handoff lanes unless Library gains deeper native study behavior.

Recommended simplification:

- Keep one `Study` mode with three action rows: `Study Dashboard`, `Flashcards`, `Quizzes`.
- Show source eligibility and selected snapshot.
- Keep separate mode chips only if each mode gets distinct Library-native state.

## Interaction Model

### Keyboard

- `Tab`: move between Source Map, Active Workbench, Inspector, footer/status.
- Arrow keys: move within lists and evidence results.
- `Enter`: activate selected row/action.
- `/`: focus Library filter or Search/RAG query depending on mode.
- `u`: use selected eligible source/evidence in Console.
- `i`: import sources.
- `e`: export selected source, collection item, or eligible filter set when supported.
- `Esc`: clear filter/query or return focus to mode root.

### Selection

Selection should update the inspector immediately.

No selected object:

- Show mode purpose.
- Show next-best action.
- Show why handoff is blocked or available.

Selected source/evidence/collection item/conversation:

- Show identity.
- Show authority and workspace eligibility.
- Show allowed actions.
- Show blocked actions and recovery.
- Show handoff targets.

### Disabled Actions

Disabled controls must have one visible reason near the control or in the inspector. Tooltips are useful but not sufficient because terminal users and screenshot QA need visible text.

## Implementation Stages

### Stage A: Shared Source Workbench Shell

Goal: make the shell hierarchy clearer without changing services.

Deliverables:

- Rename left column conceptually to Source Map.
- Split left content into Workspace Context, Source Map, and Quick Actions.
- Convert inspector to a selected-object contract with sections: Status, Allowed, Blocked, Recovery, Handoff, Shortcuts.
- Do not add new service calls in Stage A. Use existing mode data and honest placeholder copy only.
- Add mounted tests for region headings, focus visibility, and no horizontal overflow.
- Capture CDP screenshots for Hub, Search/RAG, Workspaces, Collections.

### Stage B: Hub Inventory Rows

Goal: replace prose hub copy with structured source inventory rows.

Deliverables:

- Render Notes, Media, Conversations, Collections, Search/RAG, Import/Export, and Study as rows.
- Each row shows count/readiness, owner, and primary action.
- Empty state chooses one next-best action.
- Add tests for counts, labels, owner copy, and disabled Console handoff.

### Stage C: Search/RAG Retrieval Workbench

Goal: make retrieval the primary task surface.

Deliverables:

- Promote query input and run action to the top of the center pane.
- Add visible scope controls and evidence list region.
- Make selected evidence update inspector.
- Preserve non-source blocked recovery.
- Add tests for query focus, blocked state, selected evidence handoff, and future citation/snippet placeholders.

### Stage D: Collections Item Reading Workbench

Goal: make collection item list/detail/reader behavior usable without hiding deferred server-parity features.

Deliverables:

- Prioritize collection item filters, item list, and selected item reader/review detail.
- Introduce or use a local collection-item view model that exposes item data plus capability flags before enabling controls.
- Render item rows with status, favorite, origin/domain, tags, and updated/reading metadata.
- Add selected item inspector sections for status, allowed actions, blocked actions, recovery, Search/RAG handoff, and Console handoff.
- Reserve disabled or WIP-labeled lanes for highlights, templates, digest schedules, import/export, and bulk output generation when local backing is unavailable.
- Add tests for selected item inspector, status/favorite/tag affordances, blocked workspace handoff, readable content empty state, and deferred server-parity copy.

### Stage E: Workspace Eligibility Matrix

Goal: make workspace policy legible and hard to misinterpret.

Deliverables:

- Add active workspace summary.
- Render browse/search versus stage/use as separate columns.
- Explicitly label Default workspace safety restrictions.
- Add tests for global browse visibility, workspace-gated staging, and blocked filesystem/tool implications.

### Stage F: Study Lane Simplification

Goal: reduce mode-strip complexity if distinct Study/Flashcards/Quizzes modes remain shallow.

Deliverables:

- Either consolidate under one Study mode with action rows or deepen each mode's unique Library-native state.
- Preserve existing route actions.
- Add tests for labels, disabled/eligible state, and handoff copy.

## QA Gates

Every stage must include:

- Focused mounted regression tests for stable selectors and copy.
- `git diff --check`.
- Actual CDP/Textual-web screenshots at wide and supported compact sizes.
- User approval of rendered screenshots before claiming visual acceptance.
- A short QA note that records what works, what is blocked, and what remains deferred.

Suggested CDP evidence set:

- Hub inventory, empty and with seeded source counts.
- Search/RAG empty/blocked and seeded result/evidence selected.
- Collections empty, selected collection item with reader/review detail, and selected item with unsupported capability disabled reasons.
- Workspaces Default and non-default workspace eligibility matrix.
- Import/Export empty and source-selected export-eligible state.
- Conversations empty and saved conversation selected.
- Study lane with no source and source-selected handoff.

## Non-Goals

- Do not turn Library into a second Console.
- Do not implement sync writes as part of the visual redesign.
- Do not hide global content when workspace changes.
- Do not implement full citation/snippet persistence unless a stage explicitly owns it.
- Do not implement full tldw_server Collections parity in the first visual redesign. The spec reserves the IA for Reading List, Highlights, Templates, Digest Schedules, Import/Export, bulk actions, and outputs, but implementation should only wire what local Chatbook services actually support.
- Do not rebuild Notes, Media, Study, Flashcards, or Quizzes inside Library.
- Do not accept ASCII layouts or generated mockups as final visual approval; actual screenshots remain required.

## Planning Decisions

These defaults remove ambiguity for implementation planning:

1. Keep the existing Study, Flashcards, and Quizzes mode chips through Stages A-E. Stage F may consolidate them into one Study lane, but only after a dedicated screenshot review because it changes the user's mode model.
2. Do not add a global Source Map filter in Stage A. Keep filtering mode-specific until there is a unified source list with real source rows across Notes, Media, Conversations, Collections, and evidence.
3. Collections starts as a collection-item reading/review surface in Stage D. Do not add new server sync, membership mutation, template, digest, or highlight services as part of the visual hierarchy redesign unless a local service already exists and the task explicitly owns it.
4. Capability flags decide action availability. Do not enable controls from labels, row presence, seeded test data, or visual placeholders alone.
5. Compact behavior should preserve the three regions at the currently supported visual QA sizes. At narrower widths, hide lower-priority inspector help first while preserving Status, Allowed, Blocked, and Recovery. Exact breakpoint values should be set by mounted tests and CDP screenshots, not guessed in the spec.

## Recommended First PR

Start with Stage A only. It is the highest-leverage change because it improves every Library mode without changing backend service behavior.

Acceptance criteria for the first PR:

- Library still renders the accepted TASK-89 modes.
- Left column visibly groups Workspace Context, Source Map, and Quick Actions.
- Center pane remains the dominant work area.
- Inspector uses the selected-object contract even when no object is selected.
- Collections copy and source-map labels describe stored collection items, not abstract folder membership.
- No new Library service calls or server API calls are introduced in Stage A.
- Existing mode switching and fixed chip hit targets remain intact.
- CDP screenshots are captured and approved before merge.
