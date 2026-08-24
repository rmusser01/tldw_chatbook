# Console Rail Interaction and Authority Design

**Date:** 2026-08-24
**Status:** Approved for planning
**Surface mode:** Operate
**Target:** Native Textual Console Context and Inspect rails
**Backlog task:** `TASK-20937.6`
**Amends:** `Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md`

## Goal

Make Context navigation deterministic and forgiving, preserve one stable rail
layout across workspace switches by default, and let every user answer what a
send will do without scanning the entire Inspect rail.

The design serves first-time and regular users, both technical and
non-technical, without removing the compact keyboard-first paths used by power
users.

## Evidence

UAT reproduced a production-shaped pointer failure inside the mounted Context
rail. Pressing a workspace label activated its section and moved the outer rail
four rows before Textual resolved the final click at the original screen
coordinate. A press begun on a workspace therefore dispatched the conversation
that moved under that coordinate. The same reflow could leave an opaque Tree
tooltip showing a stale label over neighboring rows.

The review also found two information-architecture problems:

1. Section disclosure is stored per workspace, so switching workspaces can
   silently rearrange Context and break spatial continuity.
2. Inspect places the active workspace and conversation below repeated run,
   source, and zero-value telemetry, making the next send's authority difficult
   to verify.

## Approved decisions

1. A single click selects a Tree row. A collapsed workspace also expands. A
   rapid double-click or Enter activates the selected workspace or conversation.
2. Rail disclosure layout is global by default. Per-workspace layouts remain an
   explicit Console Behavior preference.
3. Inspect pins a `What happens if I send now?` summary above its outer scroll
   owner. Empty Tools, Approvals, and Artifacts groups move under `More` until
   they contain actionable or nonzero state.

## Interaction model

### Stable selection before activation

Workspace and conversation rows have two states with distinct responsibilities:

- **Selected** identifies the row being inspected or prepared for activation.
- **Active** identifies the workspace or conversation currently owning the
  Console session.

The Tree must communicate both states textually or with a glyph in addition to
color. Moving selection never changes the active Console context.

Pointer behavior is:

- A single click on a collapsed workspace selects and expands it.
- A single click on an expanded workspace selects it without collapsing it.
- Collapse remains available through the disclosure glyph and the existing
  Left/Space keyboard paths.
- A single click on a conversation selects it without resuming it.
- A rapid double-click on a selected workspace or conversation activates it.
- Textual's native multi-click chain is authoritative; the Console does not own
  a timer or define a second double-click interval.

Keyboard behavior is:

- Enter activates the selected workspace or conversation.
- Space toggles workspace disclosure without activation.
- Left and Right retain the approved branch-navigation behavior.
- Load-more and Retry nodes retain their existing action behavior and are not
  converted into two-stage activation rows.

### Pointer target preservation

The node resolved at press time owns the gesture through release and click.
Outer-rail activation, focus painting, allocation reconciliation, disclosure,
or tooltip updates may not replace that stable node identity with a second
coordinate lookup.

The implementation may defer deliberate section reveal until the pointer
gesture finishes or carry the pressed node's stable key through the gesture.
The observable contract is the same: layout movement cannot retarget a click.

### Full-label help

A Tree row receives a full-label tooltip only when its rendered label is
actually truncated. Short labels such as `Research Lab` do not produce a
tooltip. Any tooltip is cleared or recomputed after Tree or outer-rail reflow,
and it must not remain associated with a stale hover line.

## Rail layout scope

### Default global layout

Console Behavior exposes one layout-scope preference with two values:

- `Global` — the default. All workspaces read and write one disclosure layout.
- `Per workspace` — each workspace reads and writes its existing layout key.

The scope applies to rail-open and direct-section disclosure preferences. It
does not persist transient local or outer scroll offsets, selected Tree rows,
search disclosure, focus history, or tooltip state.

Switching workspaces in Global mode must not change section disclosure merely
because the active workspace changed. Existing resize and compact-collapse
overrides remain rendering rules and do not rewrite either preference scope.

### Compatibility and migration

Existing per-workspace layout records remain intact. Opting into Per workspace
therefore restores the user's prior layouts rather than starting over.

When Global mode has no saved global layout, the currently active workspace's
effective saved layout seeds the global record once. If neither exists, product
defaults apply. Switching scope never deletes the inactive scope's records.

No database or schema migration is required. The existing Console preference
store remains authoritative; only key selection and the additive scope setting
change.

## Inspect authority summary

### Placement and content

`What happens if I send now?` is a pinned, compact summary between the Inspect
rail heading/project control and the outer scrolling detail body. It remains
visible while the user scrolls Inspect.

It answers five questions from one atomic display snapshot:

1. **Where?** `Workspace › Conversation`, including Default or temporary state.
2. **Scope?** What the next send will use, including any one-shot prefill or
   narrowed retrieval scope.
3. **Run?** Ready, running, waiting for approval, blocked, or recovery required.
4. **Sources?** Count of staged sources, including an explicit none state.
5. **Approvals?** Pending count, promoted visually when action is required.

The summary reuses existing Console display state. It adds no database read,
provider call, secondary cache, or independent reactive owner. A workspace or
conversation switch updates all five facts atomically; mixed old/new authority
must never paint.

### Detail deduplication and More

Facts promoted into the pinned summary are removed from lower detail groups or
rewritten to provide additional detail rather than repeat the same sentence.

Tools, Approvals, and Artifacts follow conditional disclosure:

- When empty, zero, or unavailable without an action, their groups live under
  one `More` disclosure boundary.
- When nonzero, pending, blocked, available, or otherwise actionable, the
  affected group promotes to the ordinary visible Inspect sequence.
- Promotion does not move keyboard focus to the group automatically.
- Collapsing More does not hide an actionable group.

Source Readiness remains visible because a zero-source state changes the
meaning of the next send. Detailed Run and Selected Conversation rows remain
only where they add information beyond the pinned summary.

## Error and recovery behavior

- A removed selected Tree node follows the existing keyed focus-recovery order
  and does not activate a neighboring row.
- A double-click whose first selected node disappears is ignored with the
  existing recovery notification; it never activates the row that replaces it.
- A malformed layout-scope value falls back to Global without deleting stored
  layouts.
- An incomplete Inspector snapshot retains the existing resilient ownership
  policy and reports `Inspector data incomplete`; the summary never invents
  missing authority.

## Verification

### Automated interaction coverage

Use a production-shaped mounted Console/Context host with the complete
stylesheet stack. Tests must cover:

- one real pointer click on a workspace while Workspaces is initially inactive;
- no workspace or conversation activation after single-click;
- collapsed workspace selection and expansion;
- double-click and Enter activation for workspace and conversation rows;
- exact pressed-node identity across outer-rail reflow;
- disclosure-glyph, Space, Left, and Right behavior;
- overflow and non-overflow Tree geometry;
- tooltips absent for complete labels and correct for truncated labels;
- tooltip clearing/recalculation after reflow.

### Layout-scope coverage

Tests must prove:

- Global is the default and preserves one disclosure layout across workspace
  switches.
- Per-workspace is opt-in and restores distinct existing records.
- first-use global seeding is deterministic and one-time;
- scope switches do not delete inactive records;
- compact responsive overrides do not mutate saved layouts;
- transient scroll, focus, search, and selection state remain unpersisted.

### Inspect coverage

Tests must prove:

- the summary is outside and above the outer scroll owner;
- the five facts update atomically on workspace/conversation changes;
- repeated lower rows are absent;
- empty Tools/Approvals/Artifacts render under More;
- each group promotes when actionable and demotes when cleared;
- keyboard order remains deterministic after promotion/demotion;
- narrow and wide production geometry contains the pinned summary and both
  rail edges without clipping.

Final UAT repeats the interaction paths in iTerm2 and Windows Terminal using
equivalent reported rows and columns. Physical pixels are not the layout oracle.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Keep immediate single-click activation and only freeze the pressed ID | It fixes retargeting but retains accidental navigation and leaves selection unable to support preview or future contextual actions. |
| Add a separate Activate button | It is discoverable but consumes scarce rail space and slows the power-user path without adding capability beyond double-click/Enter. |
| Keep per-workspace layouts as the default | It preserves customization but silently rearranges the rail during routine navigation and weakens learned spatial memory. |
| Delete old per-workspace layouts during migration | It simplifies storage but irreversibly discards user preferences and prevents a safe opt-in round trip. |
| Put the authority summary inside the Inspect scroll body | It would disappear exactly when users need to compare lower details with the active send authority. |
| Permanently show zero-value detail groups | It preserves today's structure but keeps the repeated telemetry wall and delays access to consequential state. |

## ADR check

ADR required: yes

ADR path: `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`

Reason: the design changes the long-lived Tree activation grammar and the
persisted rail-layout scope that ADR-083 explicitly preserves. ADR-083 is
amended rather than duplicated because it already owns both contracts.
