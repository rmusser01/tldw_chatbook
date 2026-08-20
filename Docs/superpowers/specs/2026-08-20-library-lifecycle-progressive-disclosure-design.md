# Library Lifecycle-Aware UX Improvements

Date: 2026-08-20
Status: approved design
Source: Library UX/HCI review and UAT, 2026-08-20

## Purpose

Improve Library for four personas without weakening its existing keyboard,
authority, paging, mutation, and recovery contracts:

- first-time non-technical users;
- first-time technical users;
- regular non-technical power users; and
- regular technical power users.

The work proceeds in three ordered waves:

1. first-run and lifecycle-aware empty states;
2. Prompt and Skill editor simplification; and
3. power-user layout and truthful Study handoffs.

The central design rule is progressive composition from real lifecycle state.
The product does not gain a global Beginner/Expert mode, a disconnected tour,
or a second set of simplified data models.

## Product Outcomes

- A new, empty profile reaches Import or New note without first choosing among
  the full Library information architecture.
- Existing and technical users can expand or deep-link immediately, and that
  decision is remembered.
- Empty screens prioritize value and recovery instead of inactive list
  mechanics.
- Basic Prompt and Skill editing preserves the exact stored representation,
  trust state, and round-trip behavior.
- Returning users regain Library navigation while browsing Notes and can scan
  several Media rows at 100x30.
- Study handoffs state exactly which concrete sources are included, what is
  merely summarized, what was omitted, and why.

## Existing Contracts Preserved

- Library remains the content hub and source-preparation owner.
- Console remains the live agentic work surface.
- Study owns sessions, generation, review, Flashcards, and Quizzes.
- Top-level browse sources retain their source-owned 20-row exact paging,
  stable identity, requested/applied scope, stale, mutation, and Retry
  contracts from ADR-067.
- Workspaces remain global operating context rather than a Library-only
  visibility filter.
- Important states use text and structure; color is never the only carrier.
- Existing guarded Escape, dirty-state, conflict, trust, and durable mutation
  behavior remains authoritative.

## Non-Goals

- A global Beginner/Expert preference.
- A first-run modal wizard or separate tutorial dataset.
- A generic Library lifecycle controller, generic canvas, or editor base class.
- New Library data services solely to populate the landing page.
- A recommendation engine, activity database, telemetry pipeline, or pinned
  shortcut system.
- Merging the three Study/Flashcard/Quiz Library routes in this programme.
- Making Conversations concrete Study generation inputs before Study supports
  that source type.
- Full application-wide bidirectional-text layout support.
- Changing page size, source authority, storage schemas, or mutation policy.

## Design Principles

1. **Real work is onboarding.** Starter actions use production Import and Note
   creation paths rather than tutorial substitutes.
2. **Positive and negative evidence differ.** One authoritative user-owned item
   can graduate onboarding; declaring the Library empty requires all relevant
   source owners to settle successfully.
3. **Lifecycle state controls density.** First-use, cleared, filtered-zero,
   loading, failed, stale, and populated states do not share one composition.
4. **Disclosure never changes data.** Basic/Advanced is a view preference over
   the exact Prompt or Skill representation.
5. **Safety overrides preference.** Conflict, compatibility, conversion,
   quarantine, trust, and script-access states remain visible when required.
6. **Truth outranks optimistic copy.** Displayed Study counts and the payload
   opened in Study come from the same immutable manifest.
7. **Power-user layout favors the primary task.** At compact sizes, scanning
   records outranks preview and repeated explanatory chrome.
8. **Reuse owners, not abstractions.** Similar UX grammar is implemented in the
   existing source canvas or screen owner.

## Wave 1: First-Run And Empty States

### Onboarding Rail State

Library owns one validated profile-local rail state:

```text
UNKNOWN
   | authoritative user content
   v
GRADUATED

UNKNOWN -- all owners settled empty and profile is new --> STARTER
UNKNOWN -- legacy/existing profile without preference --> EXPANDED

STARTER  -- Explore all tools --> EXPANDED
STARTER  -- user content ------> GRADUATED
EXPANDED -- Back to Get started --> STARTER
EXPANDED -- user content ---------> GRADUATED
```

Rules:

- No state moves backward automatically.
- Deleting all content does not undo graduation.
- `UNKNOWN` never renders a false empty claim or steals focus during startup.
- One settled positive source is enough to graduate.
- A no-content conclusion requires every relevant owner to settle without an
  unresolved failure.
- A new profile with partial source failure still receives Import, New note,
  and Explore all actions beside an explicit "Some Library sources are
  unavailable" recovery. Its persisted rail state remains Unknown until the
  evidence is authoritative; uncertainty never blocks first value.
- A corrupt preference fails to Expanded rather than trapping the user in
  Starter.
- Existing profiles without this preference preserve the full Library unless
  existing first-run evidence unambiguously identifies a new empty profile.
- The state is profile-local, not workspace-local.
- Deep links and command-palette navigation bypass rail filtering.

User-owned usable content includes active Notes, Media, Conversations, Prompts,
Skills, and Collections created or imported by the user. A saved Console
conversation counts. Bundled/system/sample records, Trash-only records, failed
or incomplete imports, and inaccessible records do not graduate onboarding.

### Starter Composition

Wide:

```text
+----------------------+----------------------------------------+
| LIBRARY              | GET STARTED                            |
|                      |                                        |
| > Import content     |  1. Add a file, URL, or note           |
|   New note           |  2. Find and organize it               |
|                      |  3. Use it in Console or Study          |
|   Explore all tools  |                                        |
|                      | [ Import content ]  [ New note ]        |
+----------------------+----------------------------------------+
```

Compact:

```text
GET STARTED
Add something useful, then use it in Console.

> Import content
  New note
  Explore all tools

1 Add  2 Find  3 Use
```

The onboarding surface is concise and terminal-native. It does not add
decorative ASCII art, a tour overlay, badges, or a separate sample-data mode.

After successful creation or Import, the new item receives focus first. The
rail expands only during the settled transition, with a text announcement that
Library tools are available. It never recomposes while the user is traversing
an unrelated control.

### Lifecycle Composition Grammar

```text
loading            -> named progress; no empty claim
first use          -> value statement + primary creation action
user cleared all   -> quiet recreate/import action
filtered to zero   -> preserve filter + Clear filter
content in Trash   -> View Trash / Restore recovery
failed import      -> Retry / Review failure
service error      -> retained state or Retry
stale              -> retained read-only state + authoritative recovery
populated          -> toolbar + list + pager + contextual actions
```

The shared grammar is a design standard, not a generic widget. Irrelevant
controls are omitted. A visible disabled control uses the existing non-color
marker and a readable reason only when its unavailable state is meaningful.

### Source-Specific First-Use States

- **Notes:** Create note primary; Import note secondary. Export, bulk, and
  database/file implementation mechanics appear only when relevant.
- **Media:** Import media primary. Type, selection, Export, Trash, and zero
  pager are omitted for true first use.
- **Conversations:** Start a conversation in Console primary. Empty list and
  pager mechanics are omitted.
- **Prompts and Skills lists:** Create primary; Import secondary. Paging and
  bulk mechanics are omitted for true first use.
- **Collections:** Create collection is the empty action. Rename and Delete
  appear only after selection.
- **Search:** Plain Search framing leads. Answer with sources is an explained
  optional mode. Recovery reflects actual local/server/workspace source
  authority and stays directly below the query.
- **Import:** One progressive canvas: source, verified summary, optional
  details, Import. Drafts survive validation failures and the source remains
  editable. Advanced behavior stays collapsed.
- **Export:** When nothing is exportable, replace the dead form with exact
  recovery for the active scope. Populated quality choices describe user
  outcomes rather than implementation vocabulary.
- **Rail:** Keep the primary Import entry and remove the duplicate lower Import
  row.

## Wave 2: Prompt And Skill Simplification

Prompt and Skill each persist a separate profile-local Basic/Advanced display
preference. Invalid values use Basic. A forced safety or compatibility view does
not overwrite the remembered preference.

Mode switching captures the current draft before any recompose, preserves undo,
scroll, and semantic focus, and does not itself mark the item dirty.

### Prompt Basic Eligibility

Basic is available only when the current Prompt contains:

- at most one editable System content block;
- at most one editable User content block;
- no unsupported block type or recipe-only structure; and
- no compatibility/conversion state requiring user judgment.

Basic renders a simplified view over those existing block identities. It does
not create parallel text fields or a second draft representation. Saving from
Basic preserves block IDs, lane ordering, structured metadata, and normal
version-history behavior.

An incompatible Prompt opens Advanced temporarily and explains why. Advanced
to Basic is disabled rather than flattening content:

```text
○ Basic view — this prompt uses multiple structured blocks
```

### Prompt Composition

Basic:

```text
PROMPT · BASIC

Name
Description

Instructions
What the model should always follow.

Message template
What the user supplies or asks.

[ Advanced ]                         Draft · unsaved
----------------------------------------------------
[ Save prompt ]  [ Cancel ]
```

Advanced reveals the existing structured block editor, recipes, starter
content, compiled previews, keywords, author, memberships, and history.
Compiled previews remain read-only and are clearly distinguished from editable
content.

Lifecycle actions:

```text
new                Save prompt | Cancel
saved, clean       Use in Console | More actions
saved, dirty       Save changes | Discard changes
conflict           Save as new | Reload
mutation running   progress + readable disabled reason
```

`More actions` is an existing-style inline disclosure, not a new popover
framework. It contains Export, Copy Markdown, Duplicate, Collections, History,
and Delete; Escape closes it and restores its opener. Cancel and Escape retain
existing guarded-discard behavior.

### Skill Composition

Basic:

```text
SKILL · BASIC

Name
Description
Instructions

Who can invoke?
[x] You
[x] The agent

Trust: Approved                       [View details]
[ Advanced ]                         Draft · unsaved
----------------------------------------------------
[ Save skill ]  [ Cancel ]
```

User and agent invocation remain independent choices, including an intentional
neither/reference-only state. When neither is selected, the editor states that
the Skill cannot currently be invoked. Effective agent use distinguishes the
configured choice from trust/runtime availability.

Argument hints appear only when relevant to user invocation or under Advanced.
Advanced adds:

- a bounded searchable tool multiselect using Textual's native selection-list
  behavior rather than one widget per tool;
- execution context in plain language: This conversation or Isolated
  sub-agent;
- supporting-file inventory and technical warnings; and
- imported model metadata only when a value exists, labeled read-only and not
  currently applied by the runtime.

The tool allowlist is explained as a restriction, not a permission grant.
Unknown imported tool names remain visible, unavailable, and losslessly
round-tripped. Normal runtime approval and workspace policy remain separate.

Healthy trust is a compact one-line state. Pending, changed, quarantined,
script-access, manifest-error, or other actionable safety states expand the
existing trust workbench automatically. Users may open healthy trust details
manually. Security state is never hidden merely because Basic is selected.

After first save, status names the next safety step rather than implying the
Skill is immediately agent-ready.

Skill lifecycle actions:

```text
new                Save skill | Cancel
saved, clean       Back to list | More actions
saved, dirty       Save changes | Discard changes
conflict           Reload
delete armed       Delete | Cancel
mutation running   progress + readable disabled reason
```

Trust and script-access actions remain inside the trust region they own. `More
actions` contains only lifecycle actions that are valid for the saved clean
record, such as Delete; it does not duplicate trust recovery.

## Wave 3: Power-User Layout And Study Context

### Notes Navigation

- At the existing wide breakpoint, Notes browse retains the Library rail.
- The note editor and folder-backed Files workspace use a focused full-width
  task surface with a persistent Library/Notes return cue.
- Existing dirty, sync, conflict, and Escape guards remain authoritative.
- Back restores browse source, scope, selection, scroll, and rail position.
- Compact Notes remains navigation-first rather than squeezing a two-pane
  layout.

### Compact Media

At the existing compact breakpoint, Media browse uses one-line rows and omits
the preview. Activation opens the existing detail viewer; Back restores selected
stable identity, page, type, scroll, and selection mode.

```text
MEDIA (93) · type: All
+--------------------------------------------------------------+
| > Research interview       audio      Aug 20                 |
|   Project brief            document   Aug 19                 |
|   Lecture notes            video      Aug 18                 |
|   Saved article            article    Aug 17                 |
|   Demo recording           audio      Aug 16                 |
|                                                              |
| Showing 21–40 of 93                                        |
| [ Previous ]          Page 2 of 5                 [ Next ]   |
+--------------------------------------------------------------+
```

Stable IDs remain internal. Existing short date/update metadata is reused; no
relative-time timer or new formatting subsystem is introduced. Text truncates
by terminal cell width and remains safe for CJK, emoji, combining characters,
markup characters, and mixed-direction text.

Geometry requirements at exact 100x30:

- fresh page: at least five complete one-line rows;
- retained loading/page failure: at least four complete rows;
- initial empty/error: recovery replaces the list;
- a mutation warning may use one status row but cannot cover the pager;
- row viewport and pager remain independently contained and visible.

Wide mode keeps the list/preview workbench. Resizing does not reset source
state or selection.

### Returning-User Landing

The populated wide landing uses only existing truthful state:

```text
CONTINUE
> Last successfully applied Library route and scope

NEEDS ATTENTION
! Failed import or current recoverable stale state       [Review/Retry]

RECENT
Existing recent-item summaries

QUICK ACTIONS
[Import] [New note] [Search]
```

Sections with no trustworthy data are omitted. Stale state is shown only while
the current screen owns it; it is not implied to survive restart. A deleted or
invalid Continue target falls back to the source's valid scope and explains the
adjustment. Composition performs no synchronous scan or ranking work.

Compact landing remains rail-first.

### Library-To-Study Handoff Manifest

Library constructs one detached, immutable manifest that the handoff canvas
displays and Study validates and consumes. It contains:

- destination;
- concrete included source identities;
- summary-only context;
- loaded candidate counts;
- known Library totals;
- eligibility exclusions and readable reasons;
- selection policy: automatic or user-reviewed;
- source-snapshot generation/fingerprint;
- active workspace/authority context; and
- supported context limit.

The manifest does not log titles, IDs, excerpts, queries, or private workspace
labels. Diagnostics may log only destination, bounded counts, policy,
generation, and fixed error classifications.

Study validates required fields, uniqueness, limit, source type, destination
compatibility, workspace eligibility, and generation authority. Invalid
manifests fail closed in Library with recovery; Study does not open with silently
partial context.

### Study Disclosure

The handoff distinguishes three quantities:

```text
known Library totals
        |
        v
loaded concrete eligible candidates
        |
        v
source identities actually included
```

Conversation records are labeled summary-only until Study supports them as
concrete generation inputs. Unknown totals remain unknown rather than producing
omitted-count arithmetic.

Example:

```text
STUDY · PREPARE SOURCES

Carrying 25 generation sources
  Notes   13
  Media   12

Selected automatically from 50 loaded candidates
Library totals
  Notes          25
  Media          25
  Conversations  25 — summarized only

Limit: 25 of 25 source slots used
Policy: most recent eligible Notes and Media

[ Review 50 candidates ]              [ Open in Study ]
```

Use `Open in Study` unless a real resumable session identity exists. Merely
having visited Study does not justify `Continue`. Flashcard and Quiz actions
name their actual destination section.

### Optional Source Review

Review uses one bounded native `SelectionList`-style widget, not one widget per
candidate. It shows concrete candidates only, selected/excluded state, source
type, and eligibility reason. The context limit is visible. A 26th selection is
refused with a request to deselect another item; nothing is silently dropped.

Opening Review freezes its candidate generation. If Library changes, the user
may refresh candidates or keep the currently available subset. Missing or
newly ineligible identities are removed, the change is disclosed, and the user
must confirm again. Cancel preserves the original automatic sample and restores
opener focus.

Automatic and user-reviewed manifests carry distinct policy labels.

## Failure And Rollback Behavior

- Invalid onboarding preference: Expanded.
- Unknown source authority: neutral/loading, never false Starter.
- Partial source failure: visible recovery; no negative graduation decision.
- Basic Prompt/Skill presentation failure: preserve draft and reopen Advanced.
- Forced compatibility/trust view: do not change remembered preference.
- Compact layout regression: preserve reachable rows/pager; never hide records.
- Invalid or stale Study manifest: remain in Library and offer rebuild/review.
- Disappeared reviewed source: remove explicitly, disclose, reconfirm.
- Failed preference persistence: keep current session behavior and report that
  the choice may not be remembered; do not block the task.

## Accessibility And Keyboard Contract

- All new actions are keyboard reachable in logical order.
- Disclosure controls carry explicit active-state text or a non-color marker.
- Dynamic graduation, source removal, and recovery changes are announced in
  readable status copy.
- Focus returns to semantic owners across disclosure, review, detail, and
  full-width transitions.
- Disabled labels remain readable against their own background and include a
  reason when the disabled control remains visible.
- Long text cannot push actions or pager controls outside their containing pane.
- Compact and wide layouts preserve global help, command palette, and quit
  conventions.
- No new screen binding shadows terminal-convention or global keys.

## Delivery Boundaries

The programme remains three ordered waves, decomposed into atomic tasks:

1. Starter rail and landing graduation.
2. Paged browse empty-state grammar.
3. Notes and Collections empty-state grammar.
4. Import and Export staged/recovery states.
5. Search first-use and source-readiness states.
6. Prompt Basic eligibility and editor composition.
7. Prompt lifecycle actions.
8. Skill disclosure and compact trust state.
9. Skill tool multiselect and unknown-tool round trip.
10. Notes wide browse continuity.
11. Media compact scan layout.
12. Returning-user landing.
13. Study manifest contract and truthful disclosure.
14. Study optional review and explicit selection.

Study disclosure ships before optional review so misleading copy is not blocked
on the larger selection surface.

## ADR Check

ADR required: yes

Planned ADRs:

1. `backlog/decisions/NNN-library-lifecycle-progressive-disclosure.md`
   - profile-local onboarding graduation;
   - lifecycle-aware composition; and
   - separate Prompt and Skill disclosure preferences.
2. `backlog/decisions/NNN-library-study-handoff-manifest.md`
   - immutable cross-screen manifest;
   - eligibility, sampling, limits, review, privacy, and validation ownership.

Reason: the first decision establishes long-lived Library application
structure and persisted UX state. The second changes the Library-to-Study
cross-module interface and truth/privacy boundary. Responsive Notes/Media
presentation and the returning landing remain implementation details under
existing ownership decisions.

ADRs are created and linked before implementation begins. Task IDs and ADR
numbers are resolved only when their files are created against current remote
and worktree state.

## Verification Strategy

Per explicit user direction, repository-wide pytest is not run. Each task runs
only modified/touched components and direct owners.

Every task includes the smallest meaningful RED/GREEN and inverse check for its
owned behavior. Each wave adds:

- pure state/composition tests;
- mounted tests using the production screen hierarchy;
- keyboard-only task completion and focus restoration;
- loading, first-use, cleared, filtered-zero, error, stale, conflict, and
  mutation states as applicable;
- long labels, markup characters, CJK, emoji, combining characters, and
  mixed-direction text;
- exact 100x30 and 170x48 compositor geometry;
- readable disabled-state and non-color state assertions;
- Ruff on exact touched Python files;
- CSS source generation/parity only when CSS changes; and
- `git diff --check`.

Compact Media and wide Notes geometry tests use the exact `TldwCli.CSS_PATH`
stack and production ancestor hierarchy. They assert compositor visibility and
containment, not only declared widget sizes.

One bounded isolated-profile live UAT is run after each completed wave, not
after every task. It covers all four personas at 100x30 and 170x48. Real profile
paths remain fingerprinted and untouched.

## Persona Acceptance

- **First-time non-technical:** can add first content without confronting
  advanced Library terminology or inactive mechanics.
- **First-time technical:** can Explore all or deep-link immediately and retain
  that choice.
- **Regular non-technical power user:** sees remembered per-editor density,
  lifecycle-correct actions, and clear recovery language.
- **Regular technical power user:** retains keyboard/focus throughput, scans at
  least five fresh Media rows at 100x30, and can audit the exact Study manifest.

## Programme Acceptance

- Onboarding never flickers, regresses, or graduates from bundled content.
- Empty screens do not fabricate zero, hide filtered recovery, or prioritize
  irrelevant mechanics.
- Basic mode cannot flatten, reorder, or silently drop Prompt/Skill data.
- Trust remains visible and enforceable in every disclosure mode.
- Compact Media preserves exact paging and shows at least five fresh rows at
  100x30.
- Notes browse retains wide Library navigation while focused tasks preserve
  their guards.
- Returning landing sections appear only from trustworthy existing state.
- Study disclosure and the opened payload use the same validated manifest.
- User-reviewed sources remain stable until explicitly refreshed or
  reconfirmed.
- No full-suite claim is made; only touched-component and direct-owner evidence
  is reported.
