---
target: TASK-21351 Ctrl+K activity views design
total_score: 23
max_score: 40
na_heuristics: []
p0_count: 0
p1_count: 4
timestamp: 2026-08-24T04-08-30Z
slug: console-session-switcher-activity-views-design-md
---
# TASK-21351 Ctrl+K Activity Views Critique

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 3 | Sync, stale, unavailable, gap, and result-change states are named; accepted live-update/focus behavior is incomplete. |
| 2 | Match with the real world | 2 | `Switch Session` can open standalone workflow runs, and `Finished unseen` is system vocabulary. |
| 3 | User control and freedom | 2 | Esc and explicit modes help, but terminal acknowledgement has no undo and History is reset every invocation. |
| 4 | Consistency and standards | 2 | Strong keyboard conventions, but a session picker also acts as a cross-destination work inbox. |
| 5 | Error prevention | 3 | Explicit targets and version guards are strong; stale async-search activation is not explicitly prevented. |
| 6 | Recognition rather than recall | 2 | Controls are visible, but destination, authority, state, and metadata compete in one subtitle. |
| 7 | Flexibility and efficiency | 3 | Keyboard, pointer, search, and paging are covered; historical search adds a recurring F3 step. |
| 8 | Aesthetic and minimalist design | 2 | Geometry is bounded, but five groups and many subtitle tokens exceed the two-row budget. |
| 9 | Error recognition and recovery | 2 | Failures are non-destructive, but `Waiting for you` can lead to a read-only destination. |
| 10 | Help and documentation | 2 | Key hints exist; cross-destination behavior and unavailable recovery are not taught. |
| **Total** | | **23/40** | **Acceptable; significant UX correction remains.** |

## Design Specificity Verdict

The systems design is strongly Chatbook-specific: lifecycle/activity separation,
local/server authority, FLEET outcomes, exact Workflows handoff, and fail-closed
degradation match the local-first control-room product. The visible modal is
under-authored by comparison. Its name, row grammar, destination cues, focus
behavior under live updates, and recovery experience are not yet as precise as
the distributed persistence design beneath it.

The deterministic detector scanned the Markdown target once and returned zero
rules/findings. That is not evidence that the interaction design is clean; the
detector has no implemented markup/widget tree to inspect. No browser overlay
was created because the target is a future-design document, not a rendered UI.
Line-numbered incumbent source and tests were used as the mechanical fallback.

## Overall Impression

The spec is unusually rigorous about data loss and authority, but its biggest
opportunity is to choose a narrower product identity and ship local value before
building a distributed activity subsystem. The highest-risk UX moment is an
urgent `Waiting for you` row that leaves Console for a Workflows card where the
user cannot perform the action being requested.

## What's Working

1. Activity membership comes from user consequence rather than overloading
   lifecycle, recency, or open state.
2. Explicit activation targets and authority/version checks prevent server IDs
   from falling through local conversation-resume paths.
3. Keyboard, degraded-state, migration, replay, and acknowledgement race
   contracts are substantially more concrete than the incumbent switcher.

## Priority Issues

### P1 — The session switcher silently becomes a universal work inbox

The `Switch Session` heading and existing Ctrl+K muscle memory promise a
conversation/session destination, while standalone workflow rows navigate to
Workflows. Choose one identity. Preferred v1: keep Ctrl+K conversation-scoped,
hydrate correlated activity onto conversation rows, and defer standalone runs.
If universal navigation is intentional, rename it `Jump to Work`, change the
placeholder, and make `Opens Console` / `Opens Workflows` non-truncatable.

### P1 — `Waiting for you` can lead somewhere the user cannot act

The highest-priority group includes approval, human input, paused, failed, and
stuck runs, while the minimal Workflows card explicitly excludes approval,
retry, pause, inputs, outputs, and details. Only label a row `Waiting for you`
when its destination provides a concrete next action. Otherwise use honest
read-only attention copy with owner/problem/impact/recovery instructions. Do not
acknowledge failures merely because a skeletal card painted; use explicit
`Mark seen` or ensure meaningful diagnostic detail is visible first.

### P1 — V1 is a distributed activity subsystem disguised as a modal change

The spec adds multiple local receipt/mark states, polling, a server inbox,
global event cursors, reset/gap recovery, reconciliation, metadata hydration,
capability bundling, and cross-repository migrations. Phase it:

1. Local Active/History from open sessions, controller state, and AgentRunsDB.
2. Correlated server status after Workflows has useful exact-run detail.
3. Standalone server workflows and lossless server activity feed as a separate
   server-activity task/ADR.

Use one versioned local activity-receipt seam for ordinary and survivor outcomes.
Keep the coarse FLEET mark as compatibility state instead of turning generic
conversation marks into an epoch/cause ledger.

### P1 — Async search and live regrouping can activate the wrong row

The spec discards late search responses, but Enter from search activates the top
matching rendered row without requiring it to match the current query and result
generation. The incumbent modal rebuilds positional buttons, so reordering can
also move focus onto another subject. Require immutable entry payloads keyed by
stable subject ID; Enter may activate only a result committed for the current
modal/query/mode/authority/result generation. While the current query is pending,
Enter waits or reports `Searching…`. Keyed reconciliation preserves the focused
subject, with an announced deterministic fallback if it disappears.

### P2 — History and row presentation add recurring friction

Every invocation resetting to Active makes existing historical lookup require
F3 after an empty search. Let typed search widen to History when Active has zero
matches, while keeping Active as the default browse mode. Specify the exact
two-row grammar and truncation order. Attention, source/authority, destination,
and stale/error state must never truncate; lifecycle, workspace, and recency are
the first omissions. Add width behavior, literal selected-mode text, and page
location such as `51–100 of 243`.

## Cognitive Load

Five of eight checks fail: single focus, chunking, hierarchy, minimal choices,
and progressive disclosure. Five Active groups plus a metadata-rich two-row row
shape ask the user to process too many concepts at once. Consider three urgency
groups (`Needs attention`, `Working`, `New results`) and express Current/Open as
badges rather than separate sections.

## Persona Red Flags

- **Alex, power user:** History gains a repeated F3 tax; page resets and five
  groups weaken the type-and-Enter switcher idiom.
- **Sam, keyboard/accessibility user:** selected mode is not defined as literal
  announced text; dynamic regrouping lacks a focus/announcement contract; the
  subtitle can truncate critical authority or destination information.
- **Riley, stress tester:** background mutation can race activation; fallback
  workflow labels can be ambiguous across servers; corrupt-inbox recovery does
  not name an exact user action.
- **Solo builder/operator:** local value is held behind a cross-repository event
  feed, and an urgent read-only destination violates inspectable control.

## Minor Observations

- `Finished unseen` is internal language; `New results` is clearer.
- The placeholder becomes inaccurate if workflow runs are searchable.
- F3 conflicts with ADR-031's blanket single-letter screen-action wording; the
  ADR or modal-scoped exception must be clarified before implementation.
- Current History persistence is offset-based and has no result generation;
  keyset/generation behavior is a real new service contract, not widget work.
- The spec should name how a stable ordinary outcome ID reaches the existing
  in-memory `_unvisited_outcomes` transition seam.
- The design defines a height ceiling but no narrow-width degradation contract.

## Questions to Consider

1. Is Ctrl+K fundamentally a conversation switcher or a universal work inbox?
2. Can a row honestly say `Waiting for you` when Chatbook cannot provide the
   action at its destination?
3. What user value requires the signed global cursor before local Active/History
   ships?
4. If only three row tokens survive at 72 columns, which three must never
   disappear?
